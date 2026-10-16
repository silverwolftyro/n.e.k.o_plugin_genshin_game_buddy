"""事件流 -> YUI 说话：B 轨事件到宿主对话通道的适配器（2026-10-08）

★ 它解决什么
-----------
感知层（B 轨 autopilot）把��件写成 ``run/<run_id>.stream.jsonl``
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
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

#: 事件流文件名（与 ``autopilot/ui/protocol.py`` 的 ``RUN_ID_NAME`` 一致）
RUN_ID_NAME = "run_current.txt"

#: 适配器默认轮询间隔（秒）。2s 足够快 —— 事件都是「已经发生���的事」，
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

#: 动作台词：``INTENT_EXECUTED`` 的 ``context["action"]`` -> 台词。
#:
#: ★ 为什么要这张表，而不是给 INTENT_EXECUTED 配一条模板
#:   ---------------------------------------------------
#:   ``INTENT_EXECUTED`` 真机一个 run 触发 **754 次**（全��� USE_PICKUP，
#:   约 4.9 次/秒），所以它整类在 :data:`SKIP_CODES` 里。但「捡到东西」
#:   恰恰是用户最想听的反馈之一。这里按**动作**开口子，而不是整类放行。
#:
#: 只放值得说的动作。战斗类（普攻/技能/大招）不加 —— 那些每分钟几十次，
#: 说出来比不说更吵。
ACTION_SPEECH: dict[str, str] = {
    "USE_PICKUP": "这个我收下了",
}

#: 动作维度的节流窗口（秒）。比 code 维度的 5s 长，因为动作触发频率高得多
#: （实测 ~4.9 次/秒），按 5 秒算仍是一次运行 30 句。
ACTION_THROTTLE_S: float = 30.0

# ---------------------------------------------------------------- code -> 台词
#: ``code`` -> 模板。占位符从 ``StateEvent.context`` 里取，缺失就不替换
#: （``str.format_map`` 配一个返回空串的 dict）。
#:
#: ★ 只收**已经确证存在**的 code（``EventCode`` 里的 16 个）。
#:   表外 code 走 :data:`FALLBACK_PREFIX`，不会报错也不会沉默。
EVENT_TO_SPEECH: dict[str, str] = {
    # ---- 生命周期 ----
    "RUN_STARTED": "启动了",
    "RUN_STOPPING": "准备收工了",
    "RUN_STOPPED": "歇下了",
    # ---- 通道 ----
    "CHANNEL_DEGRADED": "截图通道不太顺啊，切了个备用的",
    "CHANNEL_RECOVERED": "截图恢复了",    # ---- 急停 ----
    "PANIC_TRIGGERED": "先停一下！",
    "PANIC_CLEARED": "没事了，继续",
    # ---- 熔断 ----
    "BREAKER_OPENED": "有点不对劲，我先避一避",
    "BREAKER_CLOSED": "恢复了",
    # ---- 意图 ----
    "INTENT_EXECUTED": "按计划走了",
    "INTENT_REJECTED": "这个动作我没敢做",
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
    "PERCEPTION_DOWN": "画面有点看不清，先停一下",
    # ---- 场景 / 故障 ----
    # ⚠️ 占位符是 ``{to}`` 不是 ``{scene}``（2026-10-11 真机核对）。
    #   实测那条 SCENE_CHANGED 的 context 是
    #   ``{"from": "WORLD", "to": "UNKNOWN"}`` —— 没有 ``scene`` 键，
    #   原来写 ``{scene}`` 会被 _SafeDict 换成空串，播出来是光秃秃的
    #   「场景换成了」，等于什么都没说。用结构化的 context，别去解析
    #   message —— message 是给人看的自由文本，随时会改。
    "SCENE_CHANGED": "场景换成了{to}",
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
    "POSITION_CHANGED": "跑了{dist_m}米了",
    "ENEMY_DETECTED": "有怪！{count}个，小心点",
    "PATHFIND_START": "走，去那边看看",
    "PATHFIND_ARRIVED": "到了到了",
    "PATHFIND_STUCK": "又卡住了……",
    "ESCAPE_SUCCESS": "出来了，继续",
}


#: 业务事件的 code（:data:`autopilot.ui.schema.EventCode` 的扩展集）。
#: 单独列出来是为了让「有没有覆盖全」一眼可查 —— 引擎加了新业务事件而
#: 这里漏了映射，YUI 就只会走兜底说出「游戏那边…」这种废话。
BUSINESS_CODES = (
    "POSITION_CHANGED", "ENEMY_DETECTED", "PATHFIND_START",
    "PATHFIND_ARRIVED", "PATHFIND_STUCK", "ESCAPE_SUCCESS",
)

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
})


class _SafeDict(dict):
    """占位符缺失时返回空串，而不是抛 KeyError。"""

    def __missing__(self, key: str) -> str:  # noqa: D105
        return ""


class EventToYui:
    """追尾读事件流 -> 渲染台词 -> ``push_message``。

    :param push_fn: ``push_message`` 的薄封装。测试时注入假函数，
        生产时传 ``plugin.push_message``。**不直接依赖宿主**，
        这样本模块能被单独驱动。
    :param runtime_dir: 事件流所在目录（含 ``run_current.txt``）。
    :param source: ``push_message`` 的 ``source`` 字段。
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
    ) -> None:
        self._push = push_fn
        self._dir = Path(runtime_dir)
        self._source = source
        self._poll = max(0.2, float(poll_seconds))
        self._throttle = max(0.0, float(throttle_seconds))
        self._tpl = dict(EVENT_TO_SPEECH if templates is None else templates)
        self.log = logger_ or logger
        # 追尾状态
        self._offset = 0
        self._stream: Path | None = None
        self._run_id = ""
        self._last_spoke: dict[str, float] = {}
        #: 动作维度的节流时间戳（见 :data:`ACTION_SPEECH`）。**与
        #: ``_last_spoke`` 分开**是刻意的：一个是 code 粒度、一个是动作
        #: 粒度，共用一张表会让「开宝箱」和「刷怪」互相偷走窗口。
        self._last_action_spoke: dict[str, float] = {}
        #: ★ 本 run 内每个槽位是否已被占用（见 :data:`SPEECH_SLOTS`）。
        self._slot_used: set[str] = set()
        #: ★ 本 run 内每个 code 已说过几次（见 :data:`MAX_LINES_PER_RUN`）。
        self._repeat_count: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handled = 0

    # ---------------------------------------------------------------- 读
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

        # ★★ A2：ENEMY_DETECTED 的 count<=0 不播（2026-10-16）
        #   真机 2026-10-14 run 的 count 序列是 0→2→1→1→1→0。
        #   count=0 时模板照样渲染成「有怪！0个，小心点」——
        #   这句话是**全片唯一的亮点台词**，重复播两次伤害翻倍。
        #
        #   ★ 只压「**能解析成数字且 <=0**」，**不碰缺失**。
        #     缺 count 时仍然走原来的 _SafeDict 退化路径（渲染成
        #     「有怪！个，小心点」），那是被 test_yui_speech_templates
        #     刻意钉住的**已知退化**：只有手工造的事件才会走到，
        #     真实写方恒带 count。它要修的是数据源，不是这个模板。
        if code == "ENEMY_DETECTED" and ctx.get("count") is not None:
            try:
                if int(ctx["count"]) <= 0:
                    self.log.debug("敌人数为 %r，不播报（有怪！0个）",
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
            now = time.monotonic()
            last_a = self._last_action_spoke.get(action)
            if last_a is not None and (now - last_a) < ACTION_THROTTLE_S:
                self.log.debug("动作节流：%s（%.1fs 内已说过）", action, now - last_a)
                return None
            self._last_action_spoke[action] = now
            self.log.info("YUI 播报：%s <- %s/%s", override, code, action)
            return override

        # ★ 再问节流。
        #   顺序很要紧：节流是**按 code** 记时间的，而跳过时不能写
        #   ``_last_spoke`` —— 否则一个被跳过的 code 会把节流窗口"点亮"，
        #   逻辑上等于"说过一次"，将来真要播它时反而要多等一个窗口。
        now = time.monotonic()
        last = self._last_spoke.get(code)
        if last is not None and (now - last) < self._throttle:
            self.log.debug("节流：%s（%.1fs 内已说过）", code, now - last)
            return None
        tpl = self._tpl.get(code)
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
        try:
            receipt = self._push(
                source=self._source,
                # ★ visibility=["chat"] 台词**原样**上屏；留空则用户只看到
                #   YUI 的复述，看不到我们写的台词（见 VERBATIM_CHAT 的坑记）。
                visibility=["chat"] if VERBATIM_CHAT else [],
                ai_behavior="respond",  # ★ 仍然惊动 YUI 让她用自己的话接一句
                parts=[{"type": "text", "text": text}],
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
            self.log.info("YUI 播报：%s <- %s", text, code)
        else:
            self.log.warning("push_message 未被提交（code=%s）：%s", code, receipt)
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
                self.drain_once()
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
    "EventToYui",
]
