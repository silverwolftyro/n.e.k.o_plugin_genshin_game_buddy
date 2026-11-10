"""事件流 -> YUI 说话：B 轨事件到宿主对话通道的适配器（2026-10-08）

★ 它解决什么
-----------
感知层（B 轨 autopilot）把事件写成 ``run/<run_id>.stream.jsonl``
（``StateEvent``），overlay 和 Console 在读；**YUI 那边读不到** ——
不是格式不对，是**没人告诉它**。

本模块补上这一段：追尾读事件流 -> 按 ``code`` 翻译成一句话 ->
``push_message`` 推给宿主 -> YUI 开口。

★ 为什么轮询文件而不订阅 EventBus
-------------------------------
EventBus 在**引擎进程内**，而本适配器在**插件进程**里。两者之间的边界
是文件（这就是 ``autopilot/ui/protocol.py`` 的设计：``schema`` 只管数据
长什么样、``protocol`` 只管怎么落盘，将来换 socket 只重写后者）。
所以**只能轮询文件**。好消息是轮询成本极低：``stat`` 短路，没新事件
时**零 IO**。

★ 红线
----
模块顶层**不做任何 IO**（同 ``autopilot_bridge.py`` 的 R8）：不建线程、
不读文件、不连宿主。全部动作都发生在 :meth:`EventToYui.start` 之后。

★ 节流
------
同一个 ``code`` 在 ``throttle_s`` 内**只说一次**。原因：``CHANNEL_DEGRADED``
之类的 code 在一次故障里会连着写好几条；不节流的话 YUI 会像卡带一样
把同一句话重复十几遍。
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 场景状态机
# ★ 为什么是**双路** import
#   生产路径：本模块是插件包的一部分
#   （``plugin.plugins.genshin_game_buddy.event_to_yui``），相对 import 成立。
#   独立路径：单测与 ``autopilot/tools/demo_preview.py`` 把它当**顶层模块**导入，
#   此时相对 import 会抛 ImportError（"no known parent package"），
#   必须退回绝对 import（插件根在 ``sys.path`` 上）。
#
# ★ 为什么可以依赖 ``autopilot.ui.chat_state``
#   本模块一直刻意「不依赖 ``autopilot.*``，免得把 ctypes/Win32 拖进来」
#   （见 :data:`THROTTLE_REASON_PREFIXES` 那段说明）。但
#   ``autopilot.ui.chat_state`` 是**纯逻辑 + 纯标准库**：
#   ``autopilot/__init__.py`` 与 ``autopilot/ui/__init__.py`` 都是空壳
#   （只有 ``__version__`` / ``__all__``），import 它拖不进任何重依赖。
#   这条依赖是**单向**的 —— 状态机不 import 本模块（否则成环）。
try:  # 作为插件包的一部分被导入（生产路径）
    from .autopilot.ui.chat_state import (
        SCENE_GROUP,
        STATE_COMBAT,
        STATE_IDLE,
        STATE_UNKNOWN,
        STATE_WORLD,
        ChatState,
    )
except ImportError:  # pragma: no cover - 独立模块路径（单测 / demo 工具）
    from autopilot.ui.chat_state import (
        SCENE_GROUP,
        STATE_COMBAT,
        STATE_IDLE,
        STATE_UNKNOWN,
        STATE_WORLD,
        ChatState,
    )

#: 事件流文件名（与 ``autopilot/ui/protocol.py`` 的 ``RUN_ID_NAME`` 一致）
RUN_ID_NAME = "run_current.txt"

#: 适配器默认轮询间隔（秒）。2s 足够快 —— 事件都是「已经发生的事」，
#: 晚 1~2 秒不影响语义；再快只是白耗 stat。
DEFAULT_POLL_SECONDS = 2.0

#: 默认节流窗口（秒）
DEFAULT_THROTTLE_SECONDS = 5.0


#: ★ 同一条台词在一次 run 里最多说几次（2026-10-16 补）
#:
#: 节流解决的是「**太快**」，解决不了「**太久**」。
#: 真机两个 run 的实测（离线按 SKIP_CODES + 5s 节流模拟）：
#:
#:     20261010_225316  207s  「画面有点看不清，先停一下」 x 39
#:     20261011_180732  155s  「这个动作我没敢做」       x 31
#:
#: 也就是**每 5.1 秒重复同一句，连讲 3.5 分钟**。5 秒节流把它从
#: 「像卡带」变成了「慢速卡带」—— 观感没救回来。
#: 同一句话第 3 次起不再开口：观众听到前两次就知道出事了，
#: 第 39 次只是在消耗注意力。
MAX_LINES_PER_RUN = 2

#: ★ 同槽位互斥：同一个 slot 在一次 run 里只让**第一条**开口（2026-10-16 补）
#:
#: 槽位表解决的是「同一件事被拆成两个 code 上报」的情况
#: （``RUN_STOPPING`` 与 ``RUN_STOPPED``）。
#:
#: 开场的 ``CORE_STARTED`` / ``RUN_STARTED`` **不走这里** —— 走
#: :data:`SKIP_CODES` 直接不播。原因是槽位是「先到先得」，而真机
#: ``CORE_STARTED`` 永远排 seq=1，它会抢到槽位，于是观众听到的是
#: 「引擎起来了」而不是更口语的「启动了」。「引擎」正是审计里
#: 「台词全是工程黑话」的典型来源，不能靠时序碰运气。
SPEECH_SLOTS: dict[str, str] = {
    "RUN_STOPPING": "lifecycle.stop",
    "RUN_STOPPED": "lifecycle.stop",
    "CORE_STOPPED": "lifecycle.stop",
}

#: 这些台词里不该出现的枚举值 —— 原样念出来就是工程黑话。
#: ``UNKNOWN`` 是内部场景枚举，观众听到「场景换成了UNKNOWN」只会觉得
#: 「是不是没做完」。真机 2026-10-14 run 结尾正是这条。
_SUPPRESS_CTX_VALUES = frozenset({"UNKNOWN", "NONE", "NULL", ""})
#: 其余 code（RUN_STARTED / ENEMY_DETECTED 等）本来就低频或有语义变化，
#: 不受上限约束。
REPEAT_PRONE_CODES = frozenset({
    "PERCEPTION_DOWN",     # 暂停态下感知继续跑，一个 run 能发几百条
    "INTENT_REJECTED",     # 意图被拒时每 tick 一条
    "BLACK_FRAME",         # 已在 SKIP_CODES，这里兜底
    "SCREENSHOT_FAIL",
    "CAPTURE_FAILED",
    "OCR_FAILED",
    "YOLO_UNAVAILABLE",
    "ENTITY_DETECTOR_UNAVAILABLE",
})

#: 无模板时用的兜底前缀（``EventCode`` 是开放集合，读侧不许崩）
FALLBACK_PREFIX = "游戏那边"

#: 台词要不要**逐字**出现在对话框里。
#:
#: 这是踩过坑才有的开关。真机日志（2026-10-09 23:23:39）证明事件确实一路
#: 走到了宿主：message plane 转发 -> proactive bridge PUSH -> EventBus 交给
#: delivery manager -> YUI 开口说了两句。但用户**看不到**「走，去那边看看」。
#:
#: 原因不是参数写错，而是 ``visibility`` 的语义：``push_message_schema.py``
#: 写明「visibility —— 用户能看到插件 parts 的频道，**原样渲染**；空列表表示
#: 用户看不到 parts 本身，只看 AI 的回复」。也就是说：
#:
#:   visibility=[]            -> 台词只当 LLM 上下文，YUI **用自己的话复述**
#:   visibility=["chat"]      -> 台词以**带来源标签的系统消息**原样上屏
#:
#: host 侧对应分支在 ``app/main_server/character_runtime.py``：``"chat" in
#: visibility`` 时走 ``mgr.render_chat_blocks(...)``，注释里明说「保留插件自己的
#: 措辞 —— 插件完全可以用人设口吻写 —— 同时让来源可见」。适配器的台词本来就是
#: 按 YUI 口吻写的（「又卡住了……」「到了到了」），让模型重新组织一遍只会把
#: 状态说丢，所以这里选 True。
#:
#: ⚠️ 若嫌「系统消息 + YUI 接话」两句话太吵，把它改成 False 即可退回原行为；
#: 若想更彻底（只要上屏、不要 YUI 再接一句），把 ``push_one`` 里的
#: ``ai_behavior`` 同时改成 ``"blind"``。
VERBATIM_CHAT = True

#: ★ 陪伴台词塞给 LLM 时的**硬指令**（2026-10-22）
#:
#: ★ 为什么需要它 —— 用户实测症状
#:   独立触发器（``_demo_standalone``）推的 ``COMPANION_LINE`` 走的是
#:   普通 ``respond`` 流程，宿主把它当成「插件发来的通知」，
#:   于是 YUI 像复读机一样说「连插件都在夸你啦」，
#:   而不是自己说出「主人你随便莽」。
#:
#: ★ 宿主有没有给插件的「直发 TTS」接口？—— **没有**（2026-10-22 查证）
#:   · ``push_message`` 的 ``ai_behavior`` 只有 ``respond|read|blind``
#:     （``plugin/sdk/shared/core/push_message_schema.py`` 的
#:     ``AI_BEHAVIOR_VALUES``）—— 没有「原样念出来」这个选项；
#:   · 宿主**内部**确实有绕过 LLM 的直发通道
#:     （``app/main_server/character_runtime.py`` 的 ``direct_reply``
#:     + ``mgr.send_lanlan_response(detail_text, True)``），
#:     但那条路只给宿主自己的 task_result 发射器用：
#:     ``plugin/server/messaging/proactive_bridge.py`` 构造的
#:     proactive_message 事件里**根本没有** ``direct_reply`` 这个键，
#:     插件的 ``push_message`` 参数表里也**没有**它 ——
#:     插件够不着那条路。
#:   ⇒ 只能走 LLM，那就用**硬指令**把模型钉死在「第一人称原样说」。
#:
#: ★ 台词用【】包住
#:   给模型一个明确的「要复述的边界」。少了括号，模型经常把指令本身
#:   也复述出来 —— 观众会看到「你现在就是 YUI 本人…」整句。
COMPANION_LINE_PREFIX = (
    "你现在就是 YUI 本人。请完全以第一人称直接、自然地对主人说出下面这句话，"
    "不要做任何补充，也不要说这是插件说的："
)

#: ★ 陪伴台词的**显示策略**（``visibility``）。
#:
#: ★ 为什么是空列表（= 不直接上屏）
#:   硬指令 + 台词整条塞进 ``visibility=["chat"]`` 的话，
#:   聊天框里会出现「你现在就是 YUI 本人。请完全以第一人称…」这种工程腔 ——
#:   观众看到的应该是 YUI 说的话，不是我们的内部指令。
#:   空列表 = 台词**只当 LLM 上下文**，观众只看 YUI 说的那一句。
#:
#: ★ 为什么不用 ``blind``
#:   声音挂在 LLM 回复上（``send_lanlan_response`` -> TTS 会话）。
#:   ``blind`` = 台词完全不进 LLM = **视频里没有 YUI 的声音**。
#:   ⇒ 保留 ``respond``（有声音），用硬指令约束模型**原样复述**。
COMPANION_LINE_VISIBILITY: list[str] = []


def _compose_push_text(code: str, text: str) -> tuple[list[str], str, str]:
    """``(code, 干净的台词)`` -> ``(visibility, ai_behavior, 真正推出去的文本)``。

    ★ 为什么台词要在这里**再包一层**
      ``render()`` 渲染出来的是**干净的台词**（``{text}`` 模板，无前缀）。
      硬指令是在**推送前**才加的 —— 这样日志里记的仍然是那句干净的台词，
      排查时一眼能看出「我们到底说了什么」。

    ★ 为什么只有 ``COMPANION_LINE`` 走特殊分支
      其它 code 是**状态事件**（「发现了 3 个敌人」），本来就该让模型
      用自己的话接一句；只有陪伴台词是**写好的台词**，必须原样说出来。
    """
    if code == "COMPANION_LINE":
        # 有声音（respond）+ 硬指令钉死「第一人称原样说」
        return (list(COMPANION_LINE_VISIBILITY), "respond",
                f"{COMPANION_LINE_PREFIX}【{text}】")
    return (["chat"] if VERBATIM_CHAT else [], "respond", text)

#: 动作台词：``INTENT_EXECUTED`` 的 ``context["action"]`` -> 台词。
#:
#: ★ 为什么要这张表，而不是给 INTENT_EXECUTED 配一条模板
#:   ---------------------------------------------------
#:   ``INTENT_EXECUTED`` 真机一个 run 触发 **754 次**（全是 USE_PICKUP，
#:   约 4.9 次/秒），所以它整类在 :data:`SKIP_CODES` 里。但「捡到东西」
#:   恰恰是用户最想听的反馈之一。这里按**动作**开口子，而不是整类放行。
#:
#: 只放值得说的动作。战斗类（普攻/技能/大招）不加 —— 那些每分钟几十次，
#: 说出来比不说更吵。
#:
#: ★ 2026-10-29（Companion Mode）：台词按 ``_PROMPT_GUIDE.md`` §6.4 改写
#:   「这个我收下了」是**系统口吻**（报告动作），不是陪玩口吻。
#:   新的三句都走「人 + 情绪」，并且**不出现**「摩拉 / 拾取 / 物品」这类系统名词 ——
#:   用户想听的是「又捡到小钱钱啦」，不是「获得摩拉 100」。
ACTION_SPEECH: dict[str, str] = {
    "USE_PICKUP": "哇，又捡到小钱钱啦！",
}

#: 动作台词的**随机池**（每条至少 2 句）。
#:
#: ★ 为什么要池子
#:   「不能像复读机」是这一轮改造的第一诉求。同一个动作连着触发几十次，
#:   就算有节流，第 2 次和第 20 次说出来的还是同一句 ——
#:   池子把「同一件事」变成「同一件事有几种说法」，复读感立刻消失。
#:
#: ★ 不变量
#:   池子的第一句**必须**等于 :data:`ACTION_SPEECH` 里那一句
#:   （由 ``tests/unit/test_yui_persona_lines.py`` 钉住）——
#:   这样「只读字典」的读侧永远拿得到一句能播的话，
#:   不会因为池子抽空了而失声。
ACTION_SPEECH_POOLS: dict[str, tuple[str, ...]] = {
    "USE_PICKUP": (
        "哇，又捡到小钱钱啦！",
        "这个我收下咯~ 谢谢主人。",
        "又又又捡到了，主人背包是不是要满啦~",
        # ★ 陪玩版：还是「小钱钱」那套词，不换成系统黑话。
        "小助手又捡到小钱钱啦，主人快夸夸它~",
    ),
}

#: 动作维度的节流窗口（秒）。比 code 维度的 5s 长，因为动作触发频率高得多
#: （实测 ~4.9 次/秒），按 5 秒算仍是一次运行 30 句。
ACTION_THROTTLE_S: float = 30.0

# ---------------------------------------------------------------- 拒绝语义
#: 「被**限流**」的拒绝原因前缀（2026-10-17）
#:
#: ★ 为什么限流和安全拦截必须分开说（2026-10-17 真机联调）
#:   ``safety.py::check()`` 返回的 reason 有两类，**性质完全不同**：
#:
#:   * **限流**（``MIN_GAP`` / ``RATE_LIMITED``）—— 同一个动作间隔太密，
#:     属于**输出节流**。动作本身是合法的，只是「现在太快了」。
#:   * **安全拦截**（``SCENE_NOT_ALLOWED`` / ``WINDOW_NOT_FOREGROUND`` /
#:     ``ACTION_FORBIDDEN`` / ``BLACKLISTED`` / ``PANIC_KEY_PRESSED`` /
#:     ``BREAKER_OPEN`` / ``MAX_ACTIONS_REACHED``）—— 动作**不该做**。
#:
#:   之前两者共用一句「这个动作我没敢做」。于是引擎连点太快被 ``MIN_GAP``
#:   拦下时，YUI 一脸紧张地说「我没敢做」—— 其实它**敢做**，只是被节流了。
#:   这是**误导**：用户会以为急停/熔断/黑名单在拦它，去查错方向；
#:   而真正的问题是「动作间隔太密」（真机同一 run 754 次 USE_PICKUP）。
#:
#: ⚠ 判定用的前缀来自 ``safety.py`` 的 reason 格式（``MIN_GAP:{action}(..)``），
#:   是**字符串前缀**，不是枚举。改 safety 的 reason 格式时这里要跟着改。
THROTTLE_REASON_PREFIXES: tuple[str, ...] = ("MIN_GAP", "RATE_LIMITED")

#: 被限流时的台词。刻意**不说**「没敢做」—— 那是不存在的问题。
THROTTLED_SPEECH: str = "手速太快啦，慢一点"

#: 「权限/焦点不足」时的台词（2026-10-17）。
#:
#: ★ 为什么不能复用「这个动作我没敢做」
#:   ``WINDOW_NOT_FOREGROUND`` 的真实成因是**权限**：主控台/引擎不是
#:   管理员时，Windows UIPI 会把注入拦掉（游戏窗口在前台时也一样）。
#:   说「没敢做」会让人以为规则写错了、动作不合法；实际上动作完全合法，
#:   只是**没权限**。用户该做的是「用管理员身份重跑」，
#:   而不是去查规则。
PERMISSION_SPEECH: str = "权限不足，请用管理员身份运行"

#: 「用户接管，引擎主动让」时的台词（2026-10-17）。
#:
#: ★ 措辞要点：这是**礼让**，不是拒绝
#:   「我没敢做」会让用户以为引擎坏了或规则不认；
#:   「你先来」明确表达「我让给你了」。
USER_TAKEOVER_SPEECH: str = "你先来，我不抢"

#: 拒绝原因的**前缀分类**（2026-10-17）。
#:
#: ★ 这里抄了一份 ``autopilot/actuator/safety.py`` 里的常量 ——
#:   **故意**不 import：本模块是**独立适配器**（不依赖 ``autopilot.*``，
#:   免得把 ctypes/Win32 拖进来）。抄一份的代价是可能漂移，
#:   代价由 ``tests/unit/test_yui_speech_debounce.py`` 里的
#:   ``test_reason_prefixes_match_safety_module`` 钉住 ——
#:   抄可以，**不一致不行**。
PERMISSION_REASON_PREFIXES: tuple[str, ...] = (
    "WINDOW_NOT_FOREGROUND",
    "UIPI",
    "ACCESS_DENIED",
)

#: 「用户接管」的原因码前缀（与 safety.py 的 ``REASON_USER_TAKEOVER`` 对齐）
REASON_USER_TAKEOVER: str = "USER_TAKEOVER"

# ---------------------------------------------------------------- 刷怪防抖
#: ENEMY_DETECTED 的播报防抖窗口（秒，2026-10-17）
#:
#: ★ 为什么要再加一层（引擎侧已经去重过了）
#:   ``core/loop.py:996`` 只在「敌人数**变化**」时才发事件。但真机
#:   ``count`` 会在 1↔2↔1↔2 之间抖 —— **每次变化都发一条**，
#:   5 秒节流只能压到「每 5 秒一句」，仍是复读机。
#:
#: ★ 这纯粹是**播报层**的防抖
#:   决策侧 ``world_attack_only_with_enemy`` 读的是
#:   ``has_nearby_enemy()``（走 policy context），与本文件**零关系** ——
#:   ``event_to_yui`` 只被 ``autopilot/tools/demo_preview.py`` 和测试引用，
#:   policy / rules / actuator 都不 import 它。改这里动不了决策。
#:
#: ★ 为什么按「上次**播报**」计时而不是「上次**看见**」
#:   引擎每 tick 都在看见敌人。若按「上次看见」计时，敌人持续在场时
#:   窗口会被无限刷新，于是**数量真的翻倍了也不会播报** ——
#:   而那恰恰是最该说的一次。按「上次播报」计时才对。
ENEMY_DEBOUNCE_SECONDS: float = 30.0

#: 敌人数量变化不超过这个数 -> 视为「没变」，压制播报。
#:
#: 取 2 的理由：真机 count 在个位数区间来回抖，阈值 1 的话
#: 1↔3↔1↔3 这种摆动每跳都超阈值，等于没防抖；阈值 2 能吃掉
#: 「2 变 3」这种（用户点名要压的情况），同时 2 变 5 仍然会播。
ENEMY_COUNT_DELTA: int = 2

# ---------------------------------------------------------------- 让权防抖
#: 物理抢占播报的专用窗口（秒）。**比通用节流长一倍**（2026-10-19）
#:
#: ★ 为什么要单独一张窗口
#:   真机复盘 run 20261019_000606（273 秒干跑）：
#:
#:       USER_INTERRUPT        42 次   ← 42 次物理按键事件
#:       其中真的播报          11 次   ← 被 5s 通用节流压过一遍
#:
#:   11 次「你先来，我不抢」就是用户看到的刷屏。通用节流是「同一个 code
#:   多久能再说一次」，它对**所有**事件一视同仁 —— 但让权的语义完全不同：
#:
#:   · 用户在 3 秒让权窗口里**连按十个键**是**同一次抢占**，
#:     不是十次抢占。而通用节流按 5 秒切窗，于是每 5 秒就报一次，
#:     一个「让权 3 秒」的过程被播成 11 句。
#:
#:   · 「同一个抢占」的正确判据是**时间间隔**：10 秒内发生的让权视作
#:     同一轮（用户在连续操作），只说一次。
#:
#:   ⚠️ 不能把它塞进 :data:`REPEAT_PRONE_CODES` 的次数上限当唯一手段 ——
#:     次数上限管「一共说几句」，这里要管「隔多久才允许再说一句」。
TAKEOVER_DEBOUNCE_SECONDS: float = 10.0

#: 干跑（``--dry-run``）下**不播报**的 code（2026-10-19）
#:
#: ★ 为什么要这张表 —— 它是第二道防线，不是第一道
#:   第一道是 ``autopilot/core/loop.py`` 的 DRY_RUN 决策闸门：干跑时决策层
#:   压根不跑，INTENT_* / COMBO_* / REFLEX_* 一条都不会产生。理论上这里
#:   一条都遇不到。
#:
#:   留这张表是因为**旧发行包不带那个闸门**（本次是同版本重打包，
#:   但用户手上可能还跑着上一版）。适配器读不到引擎有没有闸门 ——
#:   它只看得见事件流。所以认一下干跑状态、把战斗 chatter 咽掉，
#:   是唯一能覆盖「引擎版本比适配器老」这条路的办法。
#:
#:   ⚠️ 只压**战斗 chatter**，不压故障：
#:     PERCEPTION_DOWN / CAPTURE_FAILED / PANIC_TRIGGERED 照播 ——
#:     干跑时引擎也可能在抓不到窗口，引擎该让用户知道它没在工作。
DRY_RUN_MUTED_CODES: frozenset[str] = frozenset({
    "ENEMY_DETECTED",
    "USER_INTERRUPT",
    "INTENT_REJECTED",
    "COMBO_STARTED",
    "COMBO_FINISHED",
    "COMBO_STEP_UNKNOWN",
    "REFLEX_FIRED",
    "POSITION_CHANGED",
    "SCENE_CHANGED",
})

# ---------------------------------------------------------------- code -> 台词
#: ``code`` -> 模板。占位符从 ``StateEvent.context`` 里取，缺失就不替换
#: （``str.format_map`` 配一个返回空串的 dict）。
#:
#: ★ 只收**已经确证存在**的 code（``EventCode`` 里的 16 个）。
#:   表外 code 走 :data:`FALLBACK_PREFIX`，不会报错也不会沉默。
EVENT_TO_SPEECH: dict[str, str] = {
    # ---- 生命周期 ----
    "RUN_STARTED": "开始咯~ 主人今天想干嘛？",
    "RUN_STOPPING": "准备收工了",
    "RUN_STOPPED": "歇下了~ 主人辛苦啦。",
    # ---- 通道 ----
    "CHANNEL_DEGRADED": "截图通道不太顺啊，切了个备用的",
    "CHANNEL_RECOVERED": "截图恢复了",    # ---- 急停 ----
    "PANIC_TRIGGERED": "主人按停啦，那我也先歇着~",
    "PANIC_CLEARED": "没事了，继续",
    # ---- 熔断 ----
    "BREAKER_OPENED": "唔……有点不对劲，我先躲一下下~",
    "BREAKER_CLOSED": "恢复了",
    # ---- 意图 ----
    "INTENT_EXECUTED": "按计划走了",
    "INTENT_REJECTED": "这个动作我没敢做",
    # ---- 用户接管（2026-10-17）----------------------------------
    "USER_INTERRUPT": "你先来，我不抢",
    # ---- 故障（2026-10-11 补）--------------------------------
    # 这几个都是**一次故障里会连着写好几条**的码，全靠 5 秒节流压住。
    # 措辞一律避开工程词（阈值 / provider / fps / **检测 / 感知 / 激活 / 意图**）——
    # YUI 说的是「发生了什么」，不是「哪个模块报了错」。
    # ★ 「检测」这两个字尤其不能用（tests/test_event_to_yui.py 明令）：
    #   它是工程腔。第一版这里写的是「实体检测暂时用不上」，
    #   结果把那条守护测试直接打红了 —— 那条测试是对的。
    #   用户真正需要知道的是「我判断不了周围有没有怪」（WORLD 那三条规则
    #   全靠它），所以照这个说，而不是照模块名说。
    "ENTITY_DETECTOR_UNAVAILABLE": "我暂时看不清周围有没有怪",
    "SCREENSHOT_FAIL": "画面抓不到了，我等一下再试",
    "YOLO_UNAVAILABLE": "眼睛有点花，先歇会儿",
    "OCR_FAILED": "刚才的字看不清了",
    "POSITION_LOST": "我有点找不着路了",
    "CAPTURE_FAILED": "截图不太顺，我再试试",
    "PERCEPTION_DOWN": "我眼睛花了……主人稍等，我眯一会儿~",
    # ---- 场景 / 故障 ----
    # ★ 2026-10-29（Companion Mode）：**不再念场景 id**
    #   「场景换成了{to}」有两个毛病：
    #     1. ``to`` 是**工程枚举**（``COMBAT`` / ``BAG_INVENTORY`` …），
    #        念出来等于把内部实现暴露给观众
    #        （``_PROMPT_GUIDE.md`` §3 硬边界第 1 条）。
    #     2. 它是**报告**（「场景换成了X」），不是陪玩。
    #   现在只说「换个地方啦~」。真要点名时用「这边 / 那边 / 洞里 / 城里」。
    #
    #   ⚠️ 原来的 ``{to}`` 占位符**必须一起去掉**
    #     ``_SafeDict`` 会把缺失的键换成空串，留着它等于在句尾挂一个空位。
    "SCENE_CHANGED": "换个地方啦~",
    "TICK_CRASH": "刚才卡了一下",
    # ---- 核心循环（2026-10-11 补）----
    # 原来表里只有 RUN_STARTED，漏了 CORE_STARTED，于是它走兜底播成
    # 「游戏那边主循环已启动（目标 5 Hz ≈ 200ms/帧）」—— 把工程参数
    # 直接念给用户听，正是「YUI 说的话全是工程黑话」的观感来源。
    "CORE_STARTED": "引擎起来了",
    "CORE_PAUSED": "我先停一下，等会儿再继续",
    "CORE_RESUMED": "好了，继续",
    "CORE_STOPPED": "引擎歇了",
    # ---- 命令 ----
    "COMMAND_RECEIVED": "收到指令了",
    # ---- 业务事件（2026-10-08）：这才是「游戏里发生了什么」-----------
    # ★ 语气参照 i18n/zh-CN.json 里已有的 perception.* 系列 ——
    #   「（你注意到主人 …，可以自然地提起这个话题。）」
    #   那是**给人设看的旁白**，而这里是要 YUI **直接开口说的话**，
    #   所以去掉括号、不解释推理过程、不出现「关键词/激活/感知」这类工程词。
    #   原则：短句、有情绪、像在跟人说话，不像在报日志。
    "POSITION_CHANGED": "走了{dist_m}米啦，主人腿不酸吗~",
    "ENEMY_DETECTED": "有{count}个坏东西来了，主人小心~",
    "PATHFIND_START": "走啦走啦，那边有东西~",
    "PATHFIND_ARRIVED": "到啦~ 就在这儿哦。",
    "PATHFIND_STUCK": "唔……卡住了，主人推一下下~",
    "ESCAPE_SUCCESS": "逃出来啦~ 主人命真大。",
    # ---- 陪伴台词（2026-10-20）------------------------------------
    #   由 ``companion_push.py`` 写进事件流 —— **只是让 YUI 按原话说一句**，
    #   没有任何游戏控制能力（不按键、不截图、不连招）。
    #   ★ 为什么要单独一个 code，而不是复用现成的
    #     表外的 code 会走 ``FALLBACK_PREFIX``（「游戏那边…」）——
    #     陪伴台词带这个前缀就不是「陪着说话」了，是「转述播报」。
    "COMPANION_LINE": "{text}",
}

#: 台词的**随机池**（2026-10-29，Companion Mode）。
#:
#: ★ 为什么要有池子
#:   「不能像复读机」是这一轮改造的第一诉求。节流 / 重复上限 / 场景组冷却
#:   全都是「**多久**能再说一句」，而池子解决的是「**说什么**」——
#:   同一件事换个说法说，复读感立刻消失。
#:
#: ★ 覆盖范围
#:   只收 ``_PROMPT_GUIDE.md`` §6.2 / §6.3 / §6.4 明确列出的那些 code。
#:   其它 code（故障、核心循环、意图…）保持**单句** ——
#:   那些是「一次性事件」，本来就只说一次，给它编一堆变体是浪费。
#:
#: ★ 不变量（由 ``tests/unit/test_yui_persona_lines.py`` 钉住）
#:   池子的第一句**必须**等于 :data:`EVENT_TO_SPEECH` 里那一句 ——
#:   这样「只读字典」的读侧永远拿得到一句能播的话，
#:   不会因为池子抽空了而失声。
EVENT_TO_SPEECH_POOLS: dict[str, tuple[str, ...]] = {
    # ---- §6.2 关键异常（tier 1，场景无关）----
    "PANIC_TRIGGERED": (
        "主人按停啦，那我也先歇着~",
        "主人喊停啦，我这就收手~",
    ),
    "BREAKER_OPENED": (
        "唔……有点不对劲，我先躲一下下~",
        "有点不对劲呀……我先避一避，不硬撑~",
        # ★ 陪玩版：熔断 = 连续执行失败 = 打不动 = 刮痧。
        #   ⚠ 2 档 → 战斗态会被**场景门控**压掉（战斗态上限 1 档），
        #     只有大世界 / 挂机态听得到。这是现有档位设计的直接后果：
        #     要在战斗中也能说这句，得先有一个「战斗超时」事件码
        #     —— 那属于引擎事件契约改动，不在本次文案范围内。
        "搁这儿刮痧呢？主人你配的队行不行啊。",
    ),
    "PERCEPTION_DOWN": (
        "我眼睛花了……主人稍等，我眯一会儿~",
        "看不清了……主人稍等一下下哦~",
        # ★ 陪玩版：感知停摆 = 打工搭子暴毙。
        #   1 档（tier1）→ 战斗态（上限 1 档）**照播**，
        #   且 tier1 不受场景组冷却约束（见 render() 里的场景组那段）。
        "哈哈，它暴毙了！笨蛋主人，你看它笨手笨脚的。",
    ),
    # ⚠ ``USER_INTERRUPT`` **故意不在这个池子表里**。
    #   它是「主人按下键、引擎让权」这条**协议信号**：用户必须能靠
    #   「你先来，我不抢」这一句认出"YUI 放手了"，随机化会毁掉可识别性。
    #   · 池子表只收 §6.2 / §6.3 / §6.4 明确列出的 code；
    #     表外的 code 走 :data:`EVENT_TO_SPEECH` 里那**唯一一句**
    #     （见 render() 里 ``pool = self._pools.get(code)`` 的回退分支）。
    #   · 「每个池子至少 2 句」是随机池的规矩（一句谈不上随机池），
    #     而让权台词不属于随机池 —— 它只有一句，且必须是这一句。
    #   守卫：``tests/unit/test_dry_run_and_speech_noise.py::
    #   TestTakeoverDebounce::test_one_takeover_burst_speaks_once``
    #   （「让权台词只有一句，不该有第二种说法」）+
    #   ``tests/unit/test_yui_companion_mode.py::
    #   TestPersonaPools::test_every_pool_has_at_least_two_lines``。
    #   ★ 战斗向台词因此**没有**挂在这里。
    # ---- §6.3 大世界闲聊（tier 2）----
    "RUN_STARTED": (
        "开始咯~ 主人今天想干嘛？",
        "好咯，那我陪主人走一趟~",
    ),
    "RUN_STOPPED": (
        "歇下了~ 主人辛苦啦。",
        "今天到这儿吧，主人休息一下~",
    ),
    "PATHFIND_START": (
        "走啦走啦，那边有东西~",
        "出发咯，主人跟紧我~",
        # ★ 陪玩版：探路的是它，我们跟在后面捡垃圾。
        "打工搭子在前面探路，主人我们跟在后面捡垃圾就行~",
    ),
    "PATHFIND_ARRIVED": (
        "到啦~ 就在这儿哦。",
        "到了到了，主人歇口气吧。",
    ),
    "PATHFIND_STUCK": (
        "唔……卡住了，主人推一下下~",
        "又卡了！是不是主人在捣乱呀。",
    ),
    "ENEMY_DETECTED": (
        "有{count}个坏东西来了，主人小心~",
        "来了来了！{count}个，主人别怕呀。",
    ),
    "ESCAPE_SUCCESS": (
        "逃出来啦~ 主人命真大。",
        "出来了出来了，主人没事就好~",
    ),
    # ---- §6.4 收集类（tier 3，受场景组冷却）----
    "POSITION_CHANGED": (
        "走了{dist_m}米啦，主人腿不酸吗~",
        "又走了{dist_m}米，主人加油~",
    ),
    "SCENE_CHANGED": (
        "换个地方啦~",
        "唔，这边风景不一样了嘛。",
    ),
}


#: 业务事件的 code（:data:`autopilot.ui.schema.EventCode` 的扩展集）。
#: 单独列出来是为了让「有没有覆盖全」一眼可查 —— 引擎加了新业务事件而
#: 这里漏了映射，YUI 就只会走兜底说出「游戏那边…」这种废话。
BUSINESS_CODES = (
    "POSITION_CHANGED", "ENEMY_DETECTED", "PATHFIND_START",
    "PATHFIND_ARRIVED", "PATHFIND_STUCK", "ESCAPE_SUCCESS",
)

#: **不受通用节流限制**的 code。
#:
#: ★ 为什么需要它
#:   通用节流是「同一个 code 多久能再说一次」（默认 5 秒），
#:   那是给**状态类**事件设计的 —— 5Hz 下反复出现的码要压住。
#:   但 ``COMPANION_LINE`` 是**用户主动**推的陪伴台词：推的节奏由
#:   ``companion_push.py`` 的 ``--every`` 决定（用户自己看着办），
#:   5 秒的窗口会**悄悄吞掉**其中一半 —— 症状是「我明明推了，它没说」。
#:   凡是「由人按节奏触发」的 code 都该进这张表。
THROTTLE_EXEMPT_CODES: frozenset[str] = frozenset({
    "COMPANION_LINE",
})

#: 明确**不播报**的 code，哪怕它在 :data:`EVENT_TO_SPEECH` 里有模板。
#:
#: ★ 为什么需要这张表（2026-10-11 真机实测）
#:   -----------------------------------------
#:   节流是**按 code** 记的（``self._last_spoke[code]``），所以 5Hz 下反复
#:   出现的码会被压到「每 5 秒一句」—— 听起来不多，但真机一个 run 实测：
#:
#:       INTENT_EXECUTED   533 次   ← 单次运行
#:       ENEMY_DETECTED     15 次
#:       SCENE_CHANGED      11 次
#:       CORE_STARTED        1 次
#:       RUN_STARTED         1 次
#:
#:   也就是说**九成以上的播报都会是 INTENT_EXECUTED**，而且内容全是
#:   「按计划走了」—— 同一个句子重复一百遍，等于噪音而不是信息。
#:   节流挡不住这件事（5 秒一句仍然很吵），只能整类不播。
#:
#:   ⚠️ 别把 ``NORMAL_ATTACK`` 加进来：它不是 code，而是
#:      ``INTENT_EXECUTED`` 事件里 ``context["action"]`` 的值。加错层级
#:      会让这张表看着在拦、实际一点用没有。
SKIP_CODES: frozenset[str] = frozenset({
    "INTENT_EXECUTED",
    # ★★ CORE_STARTED / CORE_STOPPED（2026-10-16 补，A4）
    #   这两条与 RUN_STARTED / RUN_STOPPED 是**同一件事的两次上报**，
    #   真机在 t=0.00s 同时触发（两个不同 code，各走各的节流窗口，
    #   压不住互相），于是开场连着两句「引擎起来了 / 启动了」——
    #   观众只会听到一句，且「引擎」是工程词。
    #
    #   为什么进 SKIP_CODES 而不是靠 SPEECH_SLOTS 抢槽：
    #   槽位是**先到先得**，而 CORE_STARTED 恒为 seq=1，
    #   抢到的必然是「引擎起来了」。这里必须显式跳过。
    "CORE_STARTED",
    "CORE_STOPPED",
    # ★★ BLACK_FRAME（2026-10-11）：**绝不能播**。
    #   真机一个 run 实测发了 2655 条 —— 因为引擎 tick 3 就 PERCEPTION_DOWN
    #   进入 CORE_PAUSED，而**暂停态下感知继续跑**，于是每 tick 都发一条
    #   BLACK_FRAME，一直发到停机。节流把它压成「每 5 秒一句」，那仍然是
    #   **连续五分钟的同一句话**，比不播还糟。
    "BLACK_FRAME",
    # ---- 工程内部指标（2026-10-11 拍板：跳过）----
    #   这四个是给开发者看的：模板匹配得分低是**常态**（阈值 0.5 不是为
    #   庆祝设的）；OCR 恢复 / YOLO 超时 / 配置错误都属于「自己知道就行、
    #   说给用户只会让人紧张」。真出现故障时前面的截图失败 / 识别不可用
    #   已经说过一次了，重复说只会变成噪音。
    "TEMPLATE_SCORE_LOW",
    "OCR_RECOVERED",
    "YOLO_TIMEOUT",
    "CONFIG_ERROR",
    # ★★ COMBO_STEP_DONE（2026-10-20 补）
    #   引擎新增的「这一步真的按完了」事件。它是给**内部消费者**用的
    #   （演出的可爱嘴替靠它起手），不是给用户听的。
    #   不登记的话，走的是兜底模板 -> YUI 会念出
    #   「游戏那边 连招第 1 步完成：USE_SKILL」——
    #   那是**工程腔**，而且一条连招五步就是五句，吵到没法用。
    "COMBO_STEP_DONE",
})


# ============================================================================
# ★★ 播报档位（verbosity，2026-10-19）
# ============================================================================
#: 三档。数字越大越啰嗦。
#:
#: :data:`SKIP_CODES` 管的是「**这一类永远不该播**」（复读机级别的）。
#: 本表管的是「这一类**值不值得在当前档位播**」。
#: 两者**不重叠、也不能互相放行** —— 档位调到最啰嗦也播不出
#: ``BLACK_FRAME``：那是节流和重复上限都救不回来的噪音。
TIER_CRITICAL = 1
TIER_NORMAL = 2
TIER_VERBOSE = 3

#: 每个 code 属于哪一档。**表里没有的一律算 :data:`TIER_VERBOSE`**。
#:
#: ★ 为什么「没登记 = 最啰嗦档」而不是「= 标准档」
#:   新加一个 code 时，它默认会**出现在静音用户的屏幕上**。
#:   而「静音」这个需求存在的全部理由就是不被吵到 ——
#:   一个没人想过的新事件在观众耳边响起来，比漏播一次严重得多。
#:   所以规矩反过来：想让某条台词进标准档，必须有人**显式**把它
#:   登记进来；不登记就是最啰嗦档，不吵人。
#:   （守这条规矩的是 ``test_every_speakable_code_is_tiered``。）
#:
#: 1 档 · 不说就等于出事：急停、用户接管、感知停摆
CODE_TIER: dict[str, int] = {
    # ---- 1 档 · 关键 ----
    #   静音档仍然播这一档。用户选「静音」是嫌吵，不是瞎了。
    "PANIC_TRIGGERED": TIER_CRITICAL,
    "USER_INTERRUPT": TIER_CRITICAL,
    #   引擎自己停了 —— 这是「现在没在工作」，玩家有权知道。
    "PERCEPTION_DOWN": TIER_CRITICAL,
    "CAPTURE_FAILED": TIER_CRITICAL,
    "SCREENSHOT_FAIL": TIER_CRITICAL,
    # 陪伴台词（companion_push.py 推的）—— 用户**主动**要它说，
    # 所以按「关键」档走：滑块拉到静音也照样能听见。
    # ★ 不登记的话就是「最啰嗦档」（``tier_of`` 的默认值），
    #   用户把滑块调到 0.5 就全被压掉了。
    "COMPANION_LINE": TIER_CRITICAL,

    # ---- 2 档 · 标准：战斗与流程状态 ----
    #   用户 0.5 档要的就是这一组：「打怪、切人、吃药、到没到」。
    "RUN_STARTED": TIER_NORMAL,
    "RUN_STOPPING": TIER_NORMAL,
    "RUN_STOPPED": TIER_NORMAL,
    "CORE_PAUSED": TIER_NORMAL,
    "CORE_RESUMED": TIER_NORMAL,
    "COMMAND_RECEIVED": TIER_NORMAL,
    "ENEMY_DETECTED": TIER_NORMAL,
    "PANIC_CLEARED": TIER_NORMAL,
    "BREAKER_OPENED": TIER_NORMAL,
    "BREAKER_CLOSED": TIER_NORMAL,
    "INTENT_REJECTED": TIER_NORMAL,
    "POSITION_LOST": TIER_NORMAL,
    "PATHFIND_START": TIER_NORMAL,
    "PATHFIND_ARRIVED": TIER_NORMAL,
    "PATHFIND_STUCK": TIER_NORMAL,
    "ESCAPE_SUCCESS": TIER_NORMAL,
    "CHANNEL_DEGRADED": TIER_NORMAL,
    "CHANNEL_RECOVERED": TIER_NORMAL,

    # ---- 3 档 · 详尽：每一个发现与状态变更 ----
    #   用户拉到 1.0 就是想看这些（调试 / 录演示）。
    "USE_PICKUP": TIER_VERBOSE,       # 走 ACTION_SPEECH 口子，见 ACTION_SPEECH
    "ENTITY_DETECTOR_UNAVAILABLE": TIER_VERBOSE,
    "YOLO_UNAVAILABLE": TIER_VERBOSE,
    "OCR_FAILED": TIER_VERBOSE,
    "SCENE_CHANGED": TIER_VERBOSE,
    "POSITION_CHANGED": TIER_VERBOSE,
    "TICK_CRASH": TIER_VERBOSE,
}

#: 默认档位。**1.0 = 详尽 = 与加这个特性之前逐字一致**。
#:
#: ★ 为什么默认不是「安静」
#:   把默认值定在 0.0 的话，这次改动一上线就让所有人闭嘴 ——
#:   那是**静默的行为变更**，而用户想要的是「能调」，不是「默认更吵」。
#:   定在 1.0 则默认行为与今天**完全相同**，只有主动动滑块才变。
DEFAULT_VERBOSITY = 1.0


def verbosity_tier(verbosity: float) -> int:
    """0.0~1.0 -> 该放行的最高档位。

    ``1 + round(v * 2)``：0.0→1 / 0.5→2 / 1.0→3，**没有第四档**。
    档位表本身就是三级的，四档会让人以为存在一个「比详尽更啰嗦」的
    档，而它只会和详尽表现完全一样 —— 那是个假选项。
    """
    try:
        v = float(verbosity)
    except (TypeError, ValueError):
        return TIER_VERBOSE              # 读不懂 -> 当最啰嗦，宁吵不漏
    v = min(1.0, max(0.0, v))            # 越界夹回，不抛
    return TIER_CRITICAL + round(v * (TIER_VERBOSE - TIER_CRITICAL))


def tier_of(code: str) -> int:
    """这个 code 属于哪一档。**没登记 = 最啰嗦档**（见 :data:`CODE_TIER`）。"""
    return CODE_TIER.get(code, TIER_VERBOSE)


# ============================================================================
# ★★ 场景门控（Companion Mode，2026-10-29）
# ============================================================================
#: 各状态下**允许播报的最高档位**。
#:
#: ★ 判据来自 ``_PROMPT_GUIDE.md`` §3「说话边界（场景 × 档位）」
#:   tier1 关键异常    -> 任何状态都播（用户选静音是嫌吵，不是瞎了）
#:   tier2 世界活动    -> WORLD / IDLE 播；COMBAT / UNKNOWN 闭嘴
#:   tier3 收集 chatter -> 只有 WORLD 播
#:
#: ★ 为什么战斗态只放 tier1
#:   用户的第一诉求原文：「战斗状态 YUI 闭嘴，仅保留关键异常播报」。
#:   所以「发现敌人」「路径开始」在打怪时一律不说 ——
#:   那正是「指导教练」式的干扰。
SCENE_TIER_LIMIT: dict[str, int] = {
    STATE_COMBAT: TIER_CRITICAL,
    STATE_WORLD: TIER_VERBOSE,
    STATE_IDLE: TIER_NORMAL,
    STATE_UNKNOWN: TIER_CRITICAL,
}

#: 同场景组的最小间隔（秒）。**非 tier1** 台词在同组内 90s 只准一句。
#:
#: ★ 这是「同场景不刷屏」的落点
#:   旧机制全是**按事件码**的（5s 节流 / 槽位 / 重复上限），
#:   于是「20 条 POSITION_CHANGED」在 5s 窗口里仍能说出 4 句。
#:   场景组把粒度提到「同一段场景里」，20 条只说 1 句。
#:
#: ★ tier1 不受此限
#:   关键异常不该被「上一句还没凉」压住 ——
#:   漏报一次「感知停摆」比多说一句严重得多。
SCENE_GROUP_COOLDOWN_S: float = 90.0

#: 挂机主动开口的台词池（``_PROMPT_GUIDE.md`` §6.5）。
#:
#: ★ 硬约束：每句都要能被 YUI **原样复述**
#:   所以不能含占位符（``{x}``）、不能含指令、不能含「（……）」这类旁白括号
#:   —— 旁白是给 LLM 看的（见 ``i18n/zh-CN.json`` 的 ``perception.proactive_switch`` 族）。
#:   台词走 ``COMPANION_LINE`` 通路（硬指令钉死第一人称原样说）。
IDLE_PROACTIVE_LINES: tuple[str, ...] = (
    "主人是不是去喝水啦？",
    "……主人不在吗？我数到一百哦。",
    "喂～再不理我我要闹了哦。",
    "主人是睡着了还是偷偷溜了呀？",
    "我一个人待着好无聊的啦……",
    # ★ 陪玩版：它在打工，主人在摸鱼（挂机 = 主人不在）。
    "小助手还在辛苦打工，主人是不是去摸鱼啦？",
)

#: 挂机主动开口的**重试间隔**（秒）。
#:
#: ★ 为什么需要它
#:   :meth:`ChatState.can_speak_proactively` 的两个条件里，「距上次开口
#:   ≥600s」是**开口之后**才推进的 —— 也就是说，**推送失败**（宿主不可用、
#:   队列满）的那一次不会推进这个计时，于是每一轮 poll（默认 2s）都会
#:   重试一次，把日志刷成一片。
#:
#:   这里加一道**本地**重试间隔：失败了至少等这么久再试。
#:   它只影响「多久再试一次」，不影响「什么时候真的开口」
#:   （那 600s 的判据在 :class:`ChatState` 里，没动）。
IDLE_PROACTIVE_RETRY_S: float = 10.0


class _SafeDict(dict):
    """占位符缺失时返回空串，而不是抛 KeyError。"""

    def __missing__(self, key: str) -> str:  # noqa: D105
        return ""


def _as_int(value: Any) -> int | None:
    """把 context 里的值转成 int。**转不了就返回 None**（不当成 0）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_throttled_rejection(ctx: Mapping[str, Any], message: str) -> bool:
    """这次拒绝是「被限流」而不是「被安全拦下」吗？

    ★ reason 的来源有**两条**，都要认（2026-10-17）
    -------------------------------------------
    1. 新版引擎把 reason 放进 ``context["reason"]``
       （``core/loop.py`` 的 ``INTENT_{outcome}`` 事件已补）。
    2. **在跑的旧发行包**只有 message，形如::

           USE_PICKUP REJECTED: MIN_GAP:USE_PICKUP(205<800ms)

       那里 reason 藏在 message 尾巴上（``core/loop.py`` 的
       ``message=f"{action} {outcome}: {res.reason}"``）。

    两条都扫，才能对**新旧引擎**都判对。只认 context 的话，
    旧包上会继续误报「没敢做」—— 而那正是要修的。

    ⚠ 扫描式判定要求这些 token 不会误伤：场景 id / 动作名里
      都不含 ``MIN_GAP`` / ``RATE_LIMITED``（见 ``safety.py`` 的 reason 格式）。
    """
    hay = f"{ctx.get('reason') or ''}\n{message or ''}"
    return any(prefix in hay for prefix in THROTTLE_REASON_PREFIXES)


def _is_permission_rejection(ctx: Mapping[str, Any], message: str) -> bool:
    """这次拒绝是「权限/焦点不足」吗（2026-10-17）。

    ``WINDOW_NOT_FOREGROUND`` / UIPI 注入被拦 —— 用户该做的
    是「用管理员身份重跑」，不是去查规则。播报「没敢做」会把人
    引向错误的方向。
    """
    hay = f"{ctx.get('reason') or ''}\n{message or ''}"
    return any(prefix in hay for prefix in PERMISSION_REASON_PREFIXES)


def _is_takeover_rejection(ctx: Mapping[str, Any], message: str) -> bool:
    """这次拒绝是「用户接管，引擎主动让」吗（2026-10-17）。

    ★ 这类**不播报**
      让权是**引擎主动做的**，不是「有个动作被否了」。
      让权本身已经有一条 ``USER_INTERRUPT`` 事件会说话；
      这里再补一句「我没敢做」纯属噪音，还会让人以为引擎坏了。
    """
    hay = f"{ctx.get('reason') or ''}\n{message or ''}"
    return REASON_USER_TAKEOVER in hay


class EventToYui:
    """追尾读事件流 -> 渲染台词 -> ``push_message``。

    :param push_fn: ``push_message`` 的薄封装。测试时注入假函数，
        生产时传 ``plugin.push_message``。**不直接依赖宿主**，
        这样本模块能被单独驱动。
    :param runtime_dir: 事件流所在目录（含 ``run_current.txt``）。
    :param source: ``push_message`` 的 ``source`` 字段。
    :param state: 场景状态机（Companion Mode）。**不传 = 自建一个**，
        单测可注入带假时钟的实例来精确推进 180s / 600s。
    :param pools: 台词随机池。**不传 = 用模块默认的**
        :data:`EVENT_TO_SPEECH_POOLS`；传了就**完全替换**默认池。
    :param picker: 抽词器 ``(key, pool) -> str``。**不传 = 内置规则**
        （第一次说某句时用池子第一句，之后随机 —— 见 :meth:`_pick_line`）。
    """

    def __init__(
        self,
        push_fn: Callable[..., Any],
        runtime_dir: Path | str,
        *,
        source: str = "genshin_game_buddy.autopilot",
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        throttle_seconds: float = DEFAULT_THROTTLE_SECONDS,
        templates: Mapping[str, str] | None = None,
        logger_: logging.Logger | None = None,
        verbosity: float = DEFAULT_VERBOSITY,
        verbosity_fn: Callable[[], float] | None = None,
        run_mode_fn: Callable[[], str] | None = None,
        state: ChatState | None = None,
        pools: Mapping[str, Sequence[str]] | None = None,
        picker: Callable[[str, Sequence[str]], str] | None = None,
        scene_group_cooldown_s: float = SCENE_GROUP_COOLDOWN_S,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._push = push_fn
        self._dir = Path(runtime_dir)
        self._source = source
        self._poll = max(0.2, float(poll_seconds))
        self._throttle = max(0.0, float(throttle_seconds))
        self._tpl = dict(EVENT_TO_SPEECH if templates is None else templates)
        # ★ 台词随机池（Companion Mode）
        #   三种来源，优先级从高到低：
        #     1. 显式传 ``pools``      -> 完全用它
        #     2. 用了默认模板           -> 用模块默认池
        #     3. 注入了自定义 ``templates`` -> **池子作废**
        #   第 3 条是刻意的：自定义模板的调用方要的是「逐字可控」，
        #   而池子会让同一句话随机变脸 —— 那正是它们在躲的事。
        if pools is not None:
            self._pools: dict[str, tuple[str, ...]] = {
                str(k): tuple(v) for k, v in pools.items()}
        elif templates is None:
            self._pools = {str(k): tuple(v)
                           for k, v in EVENT_TO_SPEECH_POOLS.items()}
        else:
            self._pools = {}
        #: 抽词器。``None`` = 用内置的「首次确定 + 之后随机」。
        self._picker = picker
        #: 已经说过一次的池子键。用来实现「第一次固定、之后随机」。
        self._pool_seen: set[str] = set()
        #: 每个池子键上一次抽中的台词。用来避免**紧接着说同一句**。
        self._pool_last: dict[str, str] = {}
        # ★ 场景状态机（Companion Mode）
        #   它管「现在能不能开口」；档位管「这一类值不值得听」。
        #   两者正交，必须都生效（``_PROMPT_GUIDE.md`` §2.3 的判定顺序）。
        self._state = state if state is not None else ChatState()
        #: ★ 本适配器用的时钟。
        #:
        #:   默认**延迟绑定** ``time.monotonic``（调用时才去取）——
        #:   直接 ``clock or time.monotonic`` 会把函数对象冻在构造那一刻，
        #:   之后单测再 monkeypatch ``time.monotonic`` 就对本实例无效了。
        #:
        #:   ★ 注入状态机时**跟它对齐**
        #:     场景组冷却窗口与状态判定必须同源 ——
        #:     两边各用各的时钟，单测注入假时钟时就会出现
        #:     「状态机已经推进了 90 秒，冷却窗口却一动没动」。
        self._clock: Callable[[], float] = (
            clock
            or (self._state.clock if state is not None
                else (lambda: time.monotonic())))
        #: ★ 场景组冷却窗口（秒）。0 = 关掉这一层。
        #: 调小它 = YUI 更健谈；调成 0 = 退回「只按事件码节流」的老行为。
        self._scene_group_cooldown_s = max(0.0, float(scene_group_cooldown_s))
        self.log = logger_ or logger
        #: ★ 播报档位。**可变**：Console 的滑块经 bridge 实时改它，
        #:   不需要重启这个适配器（它是 daemon 线程，重启=关插件）。
        #:   放在 ``_lock`` 之外读（float 赋值在 CPython 下是原子的），
        #:   档位本身也不是需要严格互斥的状态。
        self._verbosity = self._clamp(verbosity)
        self._tier_limit = verbosity_tier(self._verbosity)
        #: ★ 档位供应函数。宿主传一个「当前该多少」的读取器进来，
        #:   每轮 poll 调一次；返回 None 就当这一轮没读到，保持原值。
        #:   **不传 = 档位固定在构造时的值**（本模块可以被单独驱动）。
        self._verbosity_fn = verbosity_fn
        #: ★ 执行模式供应函数（2026-10-19）。返回 ``"LIVE"`` / ``"DRY_RUN"``。
        #:   干跑时把 :data:`DRY_RUN_MUTED_CODES` 整类咽掉 ——
        #:   引擎没在打，YUI 不该喊「发现 3 个敌人」。
        #:   **默认 LIVE**：不传这个钩子 = 按老行为播（不传 -> 不改变现状），
        #:   而宿主 ``__init__.py`` 会传一个读快照的。
        self._run_mode_fn = run_mode_fn
        #: ★ 本 run 内**是否确认过 LIVE**。
        #:
        #:   ⚠️ **没有探针时初值是 True**，不是 False。
        #:     「拿不到执行模式」和「知道引擎在干跑」是两件事：
        #:       · 没有 ``run_mode_fn`` = 本模块被单独驱动（demo_preview、
        #:         大部分单测），调用方压根没法告诉你它在哪种模式 ——
        #:         此时按干跑处理会让这些驱动方**一个字都说不出来**，
        #:         属于把「加个功能」做成了「删掉原有功能」；
        #:       · 探针读了快照、**明确读到** DRY_RUN = 引擎真的在干跑，
        #:         这才静默。
        #:
        #:   有探针时初值反而是 False：引擎刚起、快照还没落盘的那一两秒
        #:   探针读不到东西，那段时间按干跑算，第一批战斗事件不会播出声。
        self._saw_live = run_mode_fn is None
        # 追尾状态
        self._offset = 0
        self._stream: Path | None = None
        self._run_id = ""
        self._last_spoke: dict[str, float] = {}
        #: ★ 场景组冷却窗口（Companion Mode）。键是场景组（见
        #: :data:`SCENE_GROUP`），值是「上次**真的播报**的时刻」。
        #: 与 ``_last_spoke`` 分开是刻意的：一个按 code 记、一个按组记，
        #: 粒度差一个数量级，共用会让「刷怪」和「捡东西」互相偷窗口。
        self._last_group_spoke: dict[str, float] = {}
        #: ★ 挂机主动开口的**上次尝试**时刻（无论成功与否）。
        #: 用来做 :data:`IDLE_PROACTIVE_RETRY_S` 的本地重试间隔，
        #: 防止推送失败时每一轮 poll 都重试、把日志刷成一片。
        self._last_idle_attempt: float = 0.0
        #: 动作维度的节流时间戳（见 :data:`ACTION_SPEECH`）。**与
        #: ``_last_spoke`` 分开**是刻意的：一个是 code 粒度、一个是动作
        #: 粒度，共用一张表会让「开宝箱」和「刷怪」互相偷走窗口。
        self._last_action_spoke: dict[str, float] = {}
        #: ★ 本 run 内每个槽位是否已被占用（见 :data:`SPEECH_SLOTS`）。
        self._slot_used: set[str] = set()
        #: ★ 本 run 内每个 code 已说过几次（见 :data:`MAX_LINES_PER_RUN`）。
        self._repeat_count: dict[str, int] = {}
        #: ★ ENEMY_DETECTED 的播报防抖状态（2026-10-17）
        #:
        #: ``_last_enemy_time`` = **上次真的播报**的时刻（不是上次看见）。
        #: ``_last_enemy_count`` = 那次播报时的敌人数。
        #:
        #: 刻意**不跟** ``_last_spoke`` 共用：``_last_spoke`` 是 code 粒度
        #: 的 5 秒窗口，粒度差一个数量级，共用会让「有怪」被别的事件偷走窗口。
        self._last_enemy_time: float = 0.0
        self._last_enemy_count: int | None = None
        #: ★ 让权播报窗口（见 :data:`TAKEOVER_DEBOUNCE_SECONDS`）。
        #: ``_last_spoke["USER_INTERRUPT"]`` 记的是**通用 5s 窗口**，
        #: 这里记的是**让权专用 10s 窗口**，两者独立 —— 否则改通用节流
        #: 会连带改掉让权的行为，反过来也一样。
        self._last_takeover_time: float = 0.0
        #: 让权这一轮**打算**开口的时刻。压住时用它不上账，
        #: 真的说出口了才搬到 ``_last_takeover_time``。
        self._pending_takeover_time: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handled = 0

    # ---------------------------------------------------------------- 台词池
    def _pick_line(self, key: str, pool: Sequence[str]) -> str:
        """从台词池里挑一句。**池子空了就返回空串**（调用方自己兜底）。

        ★ 默认规则：「**第一次固定，之后随机，且不连着说同一句**」
          第一次说某个 key 时用池子的第一句 —— 那一句等于
          :data:`EVENT_TO_SPEECH` / :data:`ACTION_SPEECH` 里的值，
          所以「只读字典」的读侧拿到的仍是它期望的那句。
          第二次往后才随机，于是同一件事不会永远是同一句话。

        ★ 为什么第一次要固定
          开场第一句是最容易被听见的 —— 随机的话，调试时每次看到的都不一样，
          而且「第一次随机」意味着**没法用断言钉住**。
          「首次固定」把可测性和多样性两头都顾上了。

        ★ 为什么排除掉上一句
          池子小的时候（2 句），纯随机会有一半概率抽回上一句 ——
          那就是**紧接着复读**。池子存在的意义正是消灭这个现象，
          抽词器不该自己制造它。

        :param key: 池子的键（事件 code 或 ``"action:xxx"``）。
        :param pool: 候选台词。
        """
        if not pool:
            return ""
        picker = self._picker
        if picker is not None:
            return str(picker(key, pool))
        if key not in self._pool_seen:
            self._pool_seen.add(key)
            first = str(pool[0])
            self._pool_last[key] = first
            return first
        previous = self._pool_last.get(key)
        candidates = [p for p in pool if str(p) != previous] or list(pool)
        choice = str(random.choice(candidates))
        self._pool_last[key] = choice
        return choice

    # ---------------------------------------------------------------- 场景组
    def _mark_group_spoke(self, group: str, now: float) -> None:
        """记一次「这个场景组刚播过非 tier1 台词」。

        ★ 只查不占
          调用点都在**确认要开口之后** —— 被档位 / 节流 / 组冷却压住的
          事件不会推进窗口。推进了等于宣称「刚播过」，
          而用户其实什么都没听到，下一条真该说的话又要多等一轮。
        """
        if group:
            self._last_group_spoke[group] = now

    # ---------------------------------------------------------------- 档位
    @staticmethod
    def _clamp(v: Any) -> float:
        """把档位夹进 0.0~1.0。**读不懂就当详尽** —— 宁吵不漏。"""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return DEFAULT_VERBOSITY
        return min(1.0, max(0.0, f))

    @property
    def verbosity(self) -> float:
        """当前档位（0.0~1.0）。给诊断面板读。"""
        return self._verbosity

    def set_verbosity(self, value: float) -> int:
        """实时改档位。返回改完之后的**放行档位**（便于调用方打日志）。

        ★ 为什么允许**热改**而不只在构造时给
          这个适配器活在 NEKO 宿主进程里，重启它 = 停插件 + 起插件，
          而用户在 Console 里拖一下滑块就想要立刻生效。
          Console 5Hz 写 bridge、这里每轮 poll 读一次 —— 最多 2 秒延迟。

        :returns: 新阈值。**降到 0 不会清空已说的话** ——
            已经推给 YUI 的消息收不回来，也不需要清。
        """
        self._verbosity = self._clamp(value)
        old = self._tier_limit
        self._tier_limit = verbosity_tier(self._verbosity)
        if old != self._tier_limit:
            self.log.info("播报档位 -> %.2f（放行到 %d 档）",
                          self._verbosity, self._tier_limit)
        return self._tier_limit

    def refresh_verbosity(self) -> float:
        """问一次供应函数，更新档位。**没有供应函数就是空操作。**

        ★ 为什么包 try
          供应函数是宿主给的，读的是**另一个进程写的文件**。
          抛异常不能拖垮播报线程 —— 它只是让这一轮保持原档位。

        :returns: 刷新之后的档位。
        """
        fn = self._verbosity_fn
        if fn is None:
            return self._verbosity
        try:
            v = fn()
        except Exception:  # noqa: BLE001
            self.log.debug("读播报档位失败，保持 %.2f", self._verbosity,
                           exc_info=True)
            return self._verbosity
        if v is None:
            return self._verbosity        # 读不到 = 这一轮不算数
        self.set_verbosity(v)
        return self._verbosity

    # ---------------------------------------------------------------- 读
    def is_live(self) -> bool:
        """引擎现在是不是真发按键。

        ★ 为什么要问供应函数而不是读事件流
          事件流是**追加**的，引擎在干跑 / 完全接管之间切换时
          不会补发任何一条标记事件。适配器只看流的话，
          从头到尾都不知道引擎在哪种模式 —— 而这正是 2026-10-19
          要治的病（干跑时 YUI 照着喊战斗）。

        ★ 为什么「本 run 没见过 LIVE 就按干跑算」
          引擎刚起、快照还没落盘的那一两秒里供应函数读不到任何东西。
          那段时间若按 LIVE 算，第一批事件会被原样播出去 ——
          而用户看到的会是「一启动就喊有怪」。宁可先安静。

        ★ 为什么读不到时**不**当干跑（见 :func:`refresh_run_mode`）
          供应函数本身抛异常属于「读不到」，与「明确读到 DRY_RUN」
          是两回事，不能混。
        """
        return self._saw_live

    def refresh_run_mode(self) -> bool:
        """问一次供应函数。返回「是不是 LIVE」。

        ★ 为什么包 try / 读不到时保持原值
          供应函数读的是**另一个进程写的文件**，抛异常不能拖垮播报线程。
          更重要的是：**读不到 ≠ 干跑**。
          引擎没跑的时候快照文件根本不存在，那不是「引擎在干跑」，
          而是「没有引擎」—— 此时保持原值（本 run 见过 LIVE 就是 LIVE）
          才对。而 :func:`autopilot_bridge.read_engine_run_mode` 已经把
          「读到但不认识的取值」收敛成 DRY_RUN 了，那才是真干跑。

        :returns: 刷新之后是否判定为 LIVE。
        """
        fn = self._run_mode_fn
        if fn is None:
            return self._saw_live
        try:
            mode = fn()
        except Exception:  # noqa: BLE001
            self.log.debug("读引擎执行模式失败，保持 %s",
                           "LIVE" if self._saw_live else "DRY_RUN",
                           exc_info=True)
            return self._saw_live
        if mode is None:
            return self._saw_live          # 这一轮没读到，不算数
        live = str(mode).strip().upper() == "LIVE"
        if live:
            self._saw_live = True
        return self._saw_live

    def run_id(self) -> str:
        """当前 run_id。按 mtime 短路 —— 没变时零 IO。"""
        p = self._dir / RUN_ID_NAME
        try:
            rid = p.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return self._run_id
        if rid and rid != self._run_id:
            self.log.info("事件流 run 切换：%s -> %s",
                          self._run_id or "(空)", rid)
        self._run_id = rid
        return rid

    def stream_path(self) -> Path | None:
        rid = self.run_id()
        return self._dir / f"{rid}.stream.jsonl" if rid else None

    def read_new(self) -> list[dict[str, Any]]:
        """读新增事件行。**没有新增时只做一次 stat，不 read。**"""
        path = self.stream_path()
        if path is None:
            return []
        if path != self._stream:
            self._stream = path
            self._offset = 0            # 换 run = 换文件，从头读
            # ★ 换 run 必须把**播报计数**一起清掉。
            #   槽位/重复上限都是「一次 run 内」的语义；不清的话第二个 run
            #   开场的「启动了」会被上一个 run 占掉的槽位直接压制掉，
            #   而故障台词的上限也会被上一个 run 用光 —— 表现为
            #   「第一次跑会说话，第二次跑一声不吭」。
            self._slot_used.clear()
            self._repeat_count.clear()
            # ★ 刷怪防抖状态也跟着清（2026-10-17）
            #   新 run = 新一场遭遇，「上次播报过有怪」不该影响这一场。
            #   不清的话第二个 run 的第一只怪会被上一场的窗口吃掉，
            #   表现是「第二次跑就一声不吭」—— 和槽位/重复上限同款毛病。
            self._last_enemy_time = 0.0
            self._last_enemy_count = None
            # ★ 场景组窗口也得清（Companion Mode）
            #   新 run = 新一段场景，上一场刚说过「换个地方啦~」不该压住
            #   这一场的第一句。
            self._last_group_spoke.clear()
            # ★ 场景状态机同样重置（Companion Mode）
            #   不重置的话，新 run 的第一件事仍按**上一场**的场景判定 ——
            #   「新 run 开场是战斗态」会让开场那句话被门控吃掉。
            self._state.reset()
            # ★ 让权窗口也得跟着清：新 run = 新的用户会话，
            #   上一场刚说过「你先来」不该压住这一场的第一次让权。
            self._last_takeover_time = 0.0
            # ★★ 执行模式判定也必须重置（2026-10-19）。
            #   Console 切模式会**重启引擎** -> 新 run_id。这里不清的话，
            #   用户「Live 跑一场 -> 切回干跑跑一场」，第二个 run 继承上一场
            #   的 `_saw_live=True`，于是干跑静默**整个失效** ——
            #   而症状是「刚切回干跑，YUI 又开始喊有怪」，极难自查。
            #   无探针的驱动方（demo / 单测）重置成 True，见 __init__ 的说明。
            self._saw_live = self._run_mode_fn is None
        try:
            size = os.stat(path).st_size
        except OSError:
            return []                    # 文件还没建
        if size <= self._offset:
            return []
        out: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._offset)
                # ★★ offset 必须**逐行**推进，不能在循环后统一 `fh.tell()`。
                #   引擎正在写一行时，本轮读到的是半行 JSON —— 直接跳过是对的，
                #   但若最后用 fh.tell() 结算，offset 会越过那半行，下一轮
                #   从完整行之后开始读，**这半行永久丢失**（实测：offset 从
                #   0 跳到 16 == 文件大小，事件没了）。所以这里记「最后一条
                #   **成功**解析的行」的起始位置，交给循环结束后回写。
                _good_end = self._offset
                for raw in fh:
                    line_start = _good_end
                    _good_end += len(raw.encode("utf-8", "replace"))
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        # ★ 半行：引擎正在写。**不推进**到这行之后 ——
                        #   下一轮会把完整的这行重新读到。
                        _good_end = line_start
                        continue
                    if isinstance(item, dict):
                        out.append(item)
                self._offset = _good_end
        except OSError as exc:
            self.log.debug("事件流读取失败：%s", exc)
        return out

    # ---------------------------------------------------------------- 翻译
    def render(self, ev: Mapping[str, Any]) -> str | None:
        """``StateEvent`` -> 一句话。没模板 / 被节流 / 空文本都返回 None。"""
        code = str(ev.get("code") or "").strip()
        if not code:
            return None
        ctx = ev.get("context")
        ctx = ctx if isinstance(ctx, Mapping) else {}

        # ★★ 全局时刻只取一次
        #   下面每一层（让权窗口 / 组冷却 / 节流 / 记账）都拿它当基准 ——
        #   各取各的会让「这一轮到底跨没跨过窗口」取决于机器快慢。
        now = self._clock()

        # ★★ 场景状态机：**必须最先观察**（Companion Mode）
        #   在所有抑制之前喂进去，否则被黑名单 / 档位压掉的事件
        #   不会更新状态，状态机会停在过时的场景上
        #   （“在战斗里却按大世界的话说话”）。
        #
        #   它只认三个输入源：SCENE_CHANGED / POSITION_CHANGED /
        #   PERCEPTION_DOWN（含 CAPTURE_FAILED），其它 code 进来等于没进。
        self._state.observe(code, ctx, now=now)

        # ★★ INTENT_EXECUTED 的例外口子（2026-10-11）
        #   INTENT_EXECUTED 整类在 SKIP_CODES 里 —— 真机一个 run 触发 **754 次**
        #   （全是 USE_PICKUP，约 4.9 次/秒），整类播报等于复读机。
        #   但「捡到东西了」是**值得说一句**的事，所以按 context["action"]
        #   开一个口子，而不是整类放行。
        #   ⚠️ 节流必须**另算**：动作维度比 code 维度细，共用 ``_last_spoke``
        #   会让「刷怪」和「捡东西」互相抢窗口。
        action = str(ctx.get("action") or "")
        override = ACTION_SPEECH.get(action)
        if code in SKIP_CODES and override is None:
            self.log.debug("SKIP_CODES 命中，不播报：%s", code)
            return None

        # ★★ 播报档位（2026-10-19）
        #   放在 SKIP_CODES 之后：黑名单是绝对的，档位再高也放行不了。
        #   放在其余所有抑制之前：档位是「这一档要不要听」，应当第一个
        #   决定 —— 否则会先把模板渲染一遍再丢掉，白干活。
        tier = tier_of(code)
        if tier > self._tier_limit:
            self.log.debug("档位压制：%s（%d 档 > %d 档，verbosity=%.2f）",
                           code, tier, self._tier_limit, self._verbosity)
            return None

        # ★★ 干跑静默（2026-10-19）
        #   引擎没在打，YUI 就不该喊「发现 3 个敌人」「你先来，我不抢」。
        #
        #   位置在档位之后、其余抑制之前：这层比档位更绝对 ——
        #   用户把滑块拉到 1.0（详尽）也不该在干跑模式下听战斗 chatter，
        #   因为那不是「详细」，那是「假的」。
        #
        #   ⚠️ 故障类**不压**：PERCEPTION_DOWN / CAPTURE_FAILED /
        #     PANIC_TRIGGERED 不在表里 —— 干跑时引擎也可能在抓不到窗口，
        #     用户有权知道它没在工作。
        #
        #   ⚠️ 调用 ``refresh_run_mode()`` 而不是读缓存：这一层是
        #     「宁可少播」的兜底，引擎刚起来时快照还没落盘，
        #     而播报线程可能已经追到第一批事件了。
        self.refresh_run_mode()
        if not self._saw_live and code in DRY_RUN_MUTED_CODES:
            self.log.debug("干跑静默：%s（引擎没在真发按键）", code)
            return None

        # ★★ 场景门控（Companion Mode，``_PROMPT_GUIDE.md`` §3）
        #   档位管的是「这一类值不值得听」，场景管的是「**此刻**能不能开口」。
        #   两者正交：战斗时 tier2 的「发现敌人」「路径开始」一律闭嘴，
        #   只留 tier1 关键异常（阵亡 / 急停 / 感知停摆）。
        #
        #   ★ 位置在档位之后、既有节流之前
        #     §2.3 的判定顺序就是这么排的：先问「值不值得」，再问「现在行不行」。
        state = self._state.state
        limit = SCENE_TIER_LIMIT.get(state, TIER_CRITICAL)
        if tier > limit:
            self.log.debug("场景门控：%s（状态 %s，%d 档 > %d 档）",
                           code, state, tier, limit)
            return None

        # ★★ 同场景组冷却（Companion Mode，``_PROMPT_GUIDE.md`` §5）
        #   旧机制全是**按事件码**的（5s 节流 / 槽位 / 重复上限），
        #   于是「20 条 POSITION_CHANGED」在 5s 窗口里仍能说出 4 句。
        #   场景组把粒度提到「同一段场景里」—— 20 条只说 1 句。
        #
        #   ★ tier1 不受此限
        #     关键异常不该被「上一句还没凉」压住 ——
        #     漏报一次「感知停摆」比多说一句严重得多。
        #
        #   ★ 只查不占
        #     窗口在**真的开口之后**才推进（见下面两处 ``_mark_group_spoke``），
        #     与让权窗口 / 刷怪防抖同一套规矩：
        #     被压住的事件不能宣称「刚播过」。
        group = self._state.group
        if tier > TIER_CRITICAL and self._scene_group_cooldown_s > 0.0:
            last_g = self._last_group_spoke.get(group)
            if (last_g is not None
                    and (now - last_g) < self._scene_group_cooldown_s):
                self.log.debug(
                    "场景组冷却：%s（%s 组 %.1fs 内已播过，窗口 %.0fs）",
                    code, group, now - last_g, self._scene_group_cooldown_s)
                return None

        # ★★ 物理抢占防抖（2026-10-19）
        #   一个「让权 3 秒」的过程 = 用户连按十个键 = **42 条
        #   USER_INTERRUPT**，而通用 5s 节流会把它播成 11 句。
        #   语义上它们是**同一次抢占**，所以这里要一张独立的 10s 窗口
        #   （见 :data:`TAKEOVER_DEBOUNCE_SECONDS`）。
        #
        #   位置在通用节流**之前**：通用窗口只有 5s，它会先放行第 11 秒
        #   那一条 —— 顺序反了就等于这张表白写。
        #
        #   ⚠️ 只**真的开口**之后才推进窗口（下面成功分支里更新），
        #     与 ENEMY_DETECTED 防抖同一个道理：被压住的事件不能
        #     宣称「刚播过」。
        if code == "USER_INTERRUPT":
            now_tk = now
            if (self._last_takeover_time > 0.0
                    and (now_tk - self._last_takeover_time)
                    < TAKEOVER_DEBOUNCE_SECONDS):
                self.log.debug(
                    "让权防抖：%.1fs 内已说过（窗口 %.0fs）",
                    now_tk - self._last_takeover_time,
                    TAKEOVER_DEBOUNCE_SECONDS)
                return None
            self._pending_takeover_time = now_tk

        # ★★ A2：ENEMY_DETECTED 的 count<=0 不播（2026-10-16）
        #   真机 2026-10-14 run 的 count 序列是 0→2→1→1→1→0。
        #   count=0 时模板照样渲染成「发现0个敌人，没看清是哪种」——
        #   这句话是**全片唯一的亮点台词**，重复播两次伤害翻倍。
        #
        #   ★ 只压「**能解析成数字且 <=0**」，**不碰缺失**。
        #     缺 count 时仍然走原来的 _SafeDict 退化路径（渲染成
        #     「发现个敌人，没看清是哪种」），那是被 test_yui_speech_templates
        #     刻意钉住的**已知退化**：只有手工造的事件才会走到，
        #     真实写方恒带 count。它要修的是数据源，不是这个模板。
        if code == "ENEMY_DETECTED" and ctx.get("count") is not None:
            try:
                if int(ctx["count"]) <= 0:
                    self.log.debug("敌人数为 %r，不播报（没有敌人）",
                                   ctx["count"])
                    return None
            except (TypeError, ValueError):
                pass            # 解析不了就当缺失，走退化路径

        # ★★ A3：SCENE_CHANGED 的 to=UNKNOWN 不播（2026-10-16）
        #   UNKNOWN 是内部场景枚举，原样念出来等于「暴露内部实现」。
        if code == "SCENE_CHANGED":
            to = str(ctx.get("to") or "").strip().upper()
            if to in _SUPPRESS_CTX_VALUES:
                self.log.debug("场景目标为 %r，不播报（工程枚举）", to)
                return None

        # ★★ A4：槽位互斥（2026-10-16）
        #   开场/收尾各只让一句话出口，见 :data:`SPEECH_SLOTS`。
        #   **只查不占** —— 占位推迟到确认要开口之后，否则一次渲染失败
        #   就会白白烧掉这个槽位（开场那句话可能就再也不会说了）。
        slot = SPEECH_SLOTS.get(code)
        if slot is not None and slot in self._slot_used:
            self.log.debug("槽位 %s 已被占用，压制 %s", slot, code)
            return None

        # ★★ A1：同一句在一次 run 里最多说 N 次（2026-10-16）
        #   只约束 REPEAT_PRONE_CODES —— 那些是「持续态」code，
        #   第 3 次起纯噪音。RUN_STARTED / ENEMY_DETECTED 不受限。
        #   同理：只查不增，增量放在成功之后。
        if code in REPEAT_PRONE_CODES:
            if self._repeat_count.get(code, 0) >= MAX_LINES_PER_RUN:
                self.log.debug("同句已达上限 %d 次，压制 %s",
                               MAX_LINES_PER_RUN, code)
                return None

        if override is not None:
            last_a = self._last_action_spoke.get(action)
            if last_a is not None and (now - last_a) < ACTION_THROTTLE_S:
                self.log.debug("动作节流：%s（%.1fs 内已说过）", action, now - last_a)
                return None
            self._last_action_spoke[action] = now
            # ★ 动作台词也走池子（Companion Mode）
            #   键前缀 ``action:`` 是为了不和事件 code 撞车
            #   （``USE_PICKUP`` 既是动作名，也可能是未来的 code）。
            line = self._pick_line(
                f"action:{action}", ACTION_SPEECH_POOLS.get(action, ()))
            if not line:
                line = override
            # ★ 场景组窗口推进（真的开口了）
            if tier > TIER_CRITICAL:
                self._mark_group_spoke(group, now)
            self.log.info("YUI 播报：%s <- %s/%s", line, code, action)
            return line

        # ★ 再问节流。
        #   顺序很要紧：节流是**按 code** 记时间的，而跳过时不能写
        #   ``_last_spoke`` —— 否则一个被跳过的 code 会把节流窗口"点亮"，
        #   逻辑上等于"说过一次"，将来真要播它时反而要多等一个窗口。
        last = self._last_spoke.get(code)
        if (last is not None and (now - last) < self._throttle
                and code not in THROTTLE_EXEMPT_CODES):
            self.log.debug("节流：%s（%.1fs 内已说过）", code, now - last)
            return None

        # ★★ ENEMY_DETECTED 的播报防抖（2026-10-17）
        #   引擎侧只在「数量变化」时发事件，但真机 count 会在 1↔2↔1↔2 之间抖，
        #   每次变化都发一条。这里再加一层：15 秒窗口内，数量变化不大就不播。
        #
        #   ⚠ 放在**通用节流之后**：通用节流是「同一个 code 多久能再说一次」，
        #     它的优先级更高，行为不能被这一层改掉（既有测试钉的就是它）。
        #   ⚠ ``self._throttle > 0`` 是「节流全关时本层一并关」——
        #     ``throttle_seconds=0`` 的语义是「别再压我的播报」，
        #     留着这一层会让「关掉节流」名不副实。
        if code == "ENEMY_DETECTED" and self._throttle > 0.0:
            n = _as_int(ctx.get("count"))
            if (n is not None and self._last_enemy_time > 0.0
                    and (now - self._last_enemy_time) < ENEMY_DEBOUNCE_SECONDS
                    and self._last_enemy_count is not None
                    and abs(n - self._last_enemy_count) <= ENEMY_COUNT_DELTA):
                self.log.debug(
                    "刷怪防抖：%d->%d（%.1fs 内，变化 <=%d），不播报",
                    self._last_enemy_count, n,
                    now - self._last_enemy_time, ENEMY_COUNT_DELTA)
                return None

        tpl = self._tpl.get(code)
        # ★★ INTENT_REJECTED 的四种语义（2026-10-17，2026-10-19 调整）
        #   「被限流」「权限不足」「用户让权」「真的被安全拦下」——
        #   对用户是四件不同的事，不能共用一句话。
        #
        #   · 限流   -> **不播报**（2026-10-19）
        #   · 权限   -> :data:`PERMISSION_SPEECH`（该用管理员重跑）
        #   · 让权   -> **不播报**（USER_INTERRUPT 事件已经说了）
        #   · 其余   -> 保留「没敢做」（场景不对 / 急停 / 熔断…）
        #
        # ★★ 为什么限流改成**不播报**（2026-10-19 实机反馈）
        #   限流是引擎**自己**在节流（动作间隔太密），动作本身完全合法。
        #   播报它有两个害处：
        #     1. 连招跑一轮能触发十几次限流 -> 每 5 秒顶一句
        #        「手速太快啦，慢一点」，用户听着像引擎在抱怨/出错；
        #     2. 更要紧的是它会**强化「引擎不敢动」这个错觉** ——
        #        用户本来就在怀疑引擎是不是坏了，再听它自己说
        #        「太快了」，只会更不敢放手。
        #   真正的信息量在别处：连招进度、抢权提示、真正的故障。
        #   所以限流**彻底闭嘴**（只留在 debug 日志里）。
        if code == "INTENT_REJECTED":
            msg = str(ev.get("message") or "")
            if _is_takeover_rejection(ctx, msg):
                self.log.debug("让权导致的拒绝，不播报（USER_INTERRUPT 已说过）")
                return None
            if _is_throttled_rejection(ctx, msg):
                self.log.debug("限流导致的拒绝，不播报（引擎自己节流，正常现象）")
                return None
            if _is_permission_rejection(ctx, msg):
                tpl = PERMISSION_SPEECH
        # ★ 台词随机池（Companion Mode）
        #   有池子就从池子里挑一句当模板；没有就还是 :data:`EVENT_TO_SPEECH`
        #   里那一句。池子的第一句恒等于字典里的值（不变量），
        #   所以「只读字典」的读侧拿到的仍是它期望的那句。
        pool = self._pools.get(code)
        if pool:
            picked = self._pick_line(code, pool)
            if picked:
                tpl = picked
        if tpl is None:
            # ★ EventCode 是开放集合，表外 code **不许崩**，也不许沉默。
            self.log.debug("无模板的 code：%s", code)
            tpl = FALLBACK_PREFIX + "{message}"
        try:
            text = tpl.format_map(_SafeDict({**ctx, "message": ev.get("message", "")}))
        except (IndexError, ValueError) as exc:
            self.log.warning("模板渲染失败 code=%s: %s", code, exc)
            return None
        text = " ".join(str(text).split())    # 折叠多余空白
        # ★ 占位符全缺时渲染出来的是光秃秃的前缀（「游戏那边」）——
        #   那是一句**没有任何信息**的话，说出来只会让 YUI 像个复读机。
        #   这里要求至少有实质内容才开口，同时记节流免得每帧重试。
        _bare = text.strip(FALLBACK_PREFIX).strip("。，,！!？? ")
        if not _bare:
            self.log.debug("渲染后无实质内容，不播报（code=%s）", code)
            self._last_spoke[code] = now
            return None
        self._last_spoke[code] = now
        # ★ 场景组窗口**只在真的开口之后**推进（Companion Mode）
        #   与让权窗口 / 刷怪防抖同一套规矩：被压住的事件不能宣称「刚播过」。
        if tier > TIER_CRITICAL:
            self._mark_group_spoke(group, now)
        # ★ 让权窗口**只在真的开口之后**推进（2026-10-19）
        #   被档位/节流/干跑静默压住的 USER_INTERRUPT 不能算「刚播过」——
        #   推进了等于宣称说过，而用户其实什么都没听到，下一条真该说的
        #   就会被白白多压 10 秒。与 ENEMY_DETECTED 防抖同一套规矩。
        if code == "USER_INTERRUPT":
            self._last_takeover_time = self._pending_takeover_time
        # ★ 刷怪防抖状态只在**真的开口**之后更新（2026-10-17）
        #   被节流/被防抖压住的事件**不**推进窗口 ——
        #   推进了等于宣称「刚播过」，下一条真该播的又要多等一轮。
        #
        #   ⚠ count 缺失时**不**记状态：记了 ``_last_enemy_count=None``，
        #     下一条就无从比较，防抖会静默失效。宁可少压一次。
        if code == "ENEMY_DETECTED":
            n = _as_int(ctx.get("count"))
            if n is not None:
                self._last_enemy_time = now
                self._last_enemy_count = n
        # ★ 到这里才**确认要开口**，此刻才占槽位 / 计重复次数。
        #   放早一步会让一次渲染失败白白烧掉配额（开场那句话可能
        #   就此永远说不出口）。
        if slot is not None:
            self._slot_used.add(slot)
        if code in REPEAT_PRONE_CODES:
            self._repeat_count[code] = self._repeat_count.get(code, 0) + 1
        return text

    # ---------------------------------------------------------------- 推送
    def push_one(self, ev: Mapping[str, Any]) -> bool:
        text = self.render(ev)
        if not text:
            return False
        code = str(ev.get("code") or "")
        # ★ 陪伴台词在这里才裹上硬指令
        #   ``render()`` 给的是干净的台词，日志里记的仍然是干净的那句；
        #   真正推给 LLM 的文本才带「你就是 YUI 本人」的约束。
        visibility, ai_behavior, push_text = _compose_push_text(code, text)
        try:
            receipt = self._push(
                source=self._source,
                # ★ visibility / ai_behavior 由 _compose_push_text 决定
                #   普通事件：visibility=["chat"] 台词**原样**上屏；
                #   陪伴台词：visibility=[] —— 硬指令一个字都不能漏到聊天框里。
                visibility=visibility,
                # ★ 仍然惊动 YUI 开口：普通事件用她自己的话接一句，
                #   陪伴台词由硬指令逼她**第一人称原样说**。
                ai_behavior=ai_behavior,
                parts=[{"type": "text", "text": push_text}],
                priority=0,
                metadata={"kind": "autopilot_event", "code": code},
            )
        except Exception as exc:                          # noqa: BLE001
            # ★ 宿主不可用**不能拖死线程** —— 事件流是诊断通道，
            #   它挂了不该影响感知。
            self.log.warning("push_message 失败（code=%s）：%s", code, exc)
            return False
        ok = not (isinstance(receipt, Mapping)
                  and receipt.get("submitted") is False)
        if ok:
            self._handled += 1
            # ★ 记一次「开口」（Companion Mode）
            #   挂机主动话术的 600s 间隔按这个算 ——
            #   「距**上次开口**」而不是「距上次**主动**开口」。
            #   主人不在的时候 YUI 说什么都没用，但只要 YUI 刚说过话，
            #   就不该立刻再冒一句。
            self._state.note_spoke()
            self.log.info("YUI 播报：%s <- %s", text, code)
        else:
            self.log.warning("push_message 未被提交（code=%s）：%s", code, receipt)
        return ok

    # ---------------------------------------------------------------- 挂机话术
    def check_idle_proactive(self) -> bool:
        """挂机态主动找话题。返回**是否真的开口了**。

        ★ 判据在状态机里（``_PROMPT_GUIDE.md`` §6.5）
          · 挂机态（连续 180s 无坐标变化）
          · 距上次开口 ≥ 600s
        两条都由 :meth:`ChatState.can_speak_proactively` 判，
        这里只负责「说一句」。

        ★ 走 ``COMPANION_LINE`` 通路
          不新增 code —— 这条通路本来就是「让 YUI **原样复述**一句写好的话」
          （``COMPANION_LINE_VISIBILITY = []`` + ``COMPANION_LINE_PREFIX`` 硬指令），
          而挂机话术要的正是「原样说一句」，不需要她改写。

        ★ tier1，所以不受场景门控 / 组冷却影响
          挂机时世界组的那把 90s 锁多半还锁着，
          而主动找话题恰恰要等主人回来 —— 它必须能插队。
        """
        if not self._state.can_speak_proactively():
            return False
        # ★ 用适配器的时钟（本层唯一的时钟源）
        #   它与状态机**同源**（注入状态机时会自动对齐），
        #   否则本地重试间隔会拿另一只钟去比，
        #   「推进了 600s」也会被它挡下来。
        now = self._clock()
        # ★ 本地重试间隔
        #   推送失败时 ``note_spoke()`` 不会执行，600s 的判据就不推进 ——
        #   没有这道间隔的话，每一轮 poll（2s）都会重试一次，日志刷成一片。
        if (self._last_idle_attempt > 0.0
                and (now - self._last_idle_attempt) < IDLE_PROACTIVE_RETRY_S):
            return False
        self._last_idle_attempt = now
        line = str(random.choice(list(IDLE_PROACTIVE_LINES)))
        ev = {"code": "COMPANION_LINE", "message": line,
              "context": {"text": line}}
        ok = self.push_one(ev)
        if not ok:
            self.log.debug("挂机话术没送出去：%s", line)
        return ok

    def drain_once(self) -> int:
        """读一轮并推送。返回**成功推送**的条数。"""
        n = 0
        for ev in self.read_new():
            if self.push_one(ev):
                n += 1
        return n

    # ---------------------------------------------------------------- 线程
    def _loop(self) -> None:
        self.log.info("事件流适配器启动：%s", self._dir)
        while not self._stop.is_set():
            try:
                # ★ 档位**先**刷新再播 —— 反过来的话，这一轮会用旧档位
                #   播完，下一轮才生效，听起来像「延迟了一拍」。
                self.refresh_verbosity()
                self.drain_once()
                # ★ 挂机主动找话题（Companion Mode）
                #   放在 drain 之后：先把手上的事件处理完，
                #   再决定要不要**主动**冒一句。
                self.check_idle_proactive()
            except Exception as exc:                      # noqa: BLE001
                self.log.warning("适配器循环异常（继续）：%s", exc)
            self._stop.wait(self._poll)
        self.log.info("事件流适配器停止（共播报 %d 条）", self._handled)

    def start(self) -> bool:
        """启动后台线程。已在跑时返回 False（**不重复起线程**）。"""
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="autopilot-event-to-yui", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """优雅停止：置 stop 事件 + join（超时不等，避免卡住插件关停）。"""
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


__all__ = [
    "RUN_ID_NAME", "DEFAULT_POLL_SECONDS", "DEFAULT_THROTTLE_SECONDS",
    "ACTION_SPEECH", "ACTION_THROTTLE_S",
    "BUSINESS_CODES", "EVENT_TO_SPEECH", "SKIP_CODES",
    # 2026-10-16 新增（demo 出厂前粗糙度修复 A1/A3/A4）
    "MAX_LINES_PER_RUN", "REPEAT_PRONE_CODES", "SPEECH_SLOTS",
    # 2026-10-17 新增（限流语义分流 + 刷怪防抖）
    "THROTTLE_REASON_PREFIXES", "THROTTLED_SPEECH",
    "ENEMY_DEBOUNCE_SECONDS", "ENEMY_COUNT_DELTA",
    # 2026-10-19 新增（无级播报调节）
    "CODE_TIER", "TIER_CRITICAL", "TIER_NORMAL", "TIER_VERBOSE",
    "DEFAULT_VERBOSITY", "verbosity_tier", "tier_of",
    # 2026-10-20 新增（陪伴台词通道：用户按节奏推，通用节流无权吞）
    "THROTTLE_EXEMPT_CODES",
    # 2026-10-29 新增（Companion Mode：场景状态机 + 同场景组冷却 + 台词池）
    "SCENE_GROUP", "SCENE_TIER_LIMIT", "SCENE_GROUP_COOLDOWN_S",
    "IDLE_PROACTIVE_LINES", "IDLE_PROACTIVE_RETRY_S",
    "EVENT_TO_SPEECH_POOLS", "ACTION_SPEECH_POOLS",
    "EventToYui",
]
