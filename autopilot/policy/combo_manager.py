"""连招序列（Combo Sequence）：**配置 -> 状态机 -> IntentPacket**。

为什么要它
----------
原来的做法是把一条四人连招**摊成 7 条规则**，靠 priority 62~68 和
``onfield()`` 守卫串起来。这在「固定一队」时能跑，但玩家可以自由配队
（四人四种顺序，共 24 种排列），写死角色名的梯子对绝大多数队伍都不成立。

这里把「角色行为」（开技能 / 切人 / 普攻）和「连招序列」（把行为按顺序
串起来）拆成两层：

* **规则**（``rules/*.toml``）仍然只回答「这一帧该做什么」，
  **单步、命中即停**的契约一个字没改；
* **连招**（``combos/combos.toml``）只回答「接下来这一串该做什么」，
  每次仍然只产出**一个** :class:`IntentPacket`。

⚠️ **连招不新增意图类型。** 序列里的每一步产出的都是普通 ``IntentPacket``，
  和规则产出的一模一样 —— 于是 ``validator`` / ``safety`` / ``traits``
  那一整套安全审查**零改动即生效**。连招没有后门。

三类数据
--------
:class:`ComboStep`     一步（做什么、对谁、重复几次、什么时候等）
:class:`ComboSequence` 一条连招（什么时候开始、按什么顺序、被打断的条件）
:class:`ComboBook`     一份配置（若干条连招，加载期全部校验过）
:class:`ComboRunner`   **状态机**（把上面三者变成一串意图）

★ 隐式切人
----------
序列写 ``{ char = "夜兰", action = "USE_BURST" }`` 时，执行器并不知道
「夜兰」对应哪个数字键，也不该在这里硬编码。所以：

* 名册（:class:`~autopilot.actuator.active_slot.CharSlotIndex`）说
  夜兰在 2 号位，在场记账（:class:`~autopilot.actuator.active_slot.ActiveSlotTracker`）
  说现在站着的是 1 号位 -> **自动先发一个 ``SWITCH_CHARACTER(夜兰)``**，
  等记账变成 2 再发大招；
* 名册里查不到 / 在场是谁不知道 -> **跳过这一步**并记一笔，**不猜**。
  理由与 ``PolicyContext.onfield`` 相同：猜一个「大概是钟离吧」会让
  夜兰的大招打在钟离身上（实际上是在钟离身上按 Q），比不做更糟。

失败方向
--------
每个「等」都有超时兜底，且**失败一律朝「不做」倒**：

============  ==============  ==================================
等待           超时             超时后
============  ==============  ==================================
切人完成       switch_timeout   中断整条连招（发了 WARN 事件）
整条连招        timeout          中断整条连招
============  ==============  ==================================

外加三个**立刻**中断的入口：应急规则命中、物理输入抢占、感知不健康
（见 :meth:`ComboRunner.abort` 与 ``local_tree.LocalPolicyTree.poll``）。

🔴 启动后约 20 秒内连招大概率不出手
----------------------------------
编队名册靠大世界 OCR 轮转建立（单次 1.0~1.5 秒，每 25 tick 转 1 格）。
在那之前「谁在场」「某人几号位」都是 UNKNOWN，于是上面「查不到就跳过」
会**逐条跳过整个序列**。这不是 bug 也不静默 —— 会发一条 WARN 事件说清楚
缺的是哪一步。想立刻可用就手填 ``keymap.toml`` 的 ``[char_slots]``
（见 :mod:`autopilot.config.keymap`）。

配置格式见 ``autopilot/combos/combos.toml`` 与
``docs/combat_architecture.md``。
"""

from __future__ import annotations

import logging
import threading
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts.events import Event, EventCategory, Severity
from ..contracts.intent_packet import ActionType, IntentPacket
from ..contracts.state_packet import VALID_SCENE_IDS
# ★ 组合键的键名校验用 KeyId（public API，认得 W / SHIFT / LBUTTON…）。
#   policy 依赖 actuator 不是新事 —— ``local_tree.py`` 早就 import 了
#   ``actuator.humanizer``。这里只借一个**纯查表**的解析器。
from ..actuator.simulator import KeyId, KeyIdError
from .context import PolicyContext
from .dsl import CompiledExpr, DslEvaluator, RuleEvalError

logger = logging.getLogger(__name__)

#: 内置连招配置目录。**缺这个文件不是错误** —— 那表示玩家没写连招，
#: 引擎退回纯反射（只有规则）的模式，见 :func:`load_combos` 的说明。
DEFAULT_COMBOS_PATH = Path(__file__).resolve().parents[1] / "combos" / "combos.toml"

#: 命中 priority **小于**它的普通规则会**中断正在进行的连招**。
#:
#: 20 的来历（不是随便挑的）：
#:
#: * 比它小 —— 5/8/12 恢复类（感知不健康 / OCR 失败 / 场景没把握）、
#:   10 被围后退、15 开锁宝箱按 ESC。这些都是「不打会出事」，
#:   连招必须让路（用户要求的「没血了立刻中断」就落在这里）；
#: * 比它大 —— 20 开宝箱、25 躲宝箱风暴、30~45 采集拾取、50 收手、
#:   60 集火、80 通用技能、85 普攻兜底。战斗中捡矿石不该打断连招。
#:
#: 设成 ``0`` = 任何规则都打不断连招（连招绝对优先）。
DEFAULT_BREAK_BELOW_PRIORITY = 20

#: 连招产出的意图 priority。只是**给日志和事件看**的 —— 执行层不按它
#: 排优先级（唯一读它的地方是 ``validator`` 的 0~100 范围检查）。
#: 70 = 落在旧连招梯子的 62~68 与通用兜底 80/85 之间，读起来不误导。
COMBO_INTENT_PRIORITY = 70

#: 危机反射的意图优先级。**必须高于一切** —— 包括连招自己。
#:
#: ★ 为什么不是「比连招高一点」
#:   执行层/策略树是按 priority 从高到低挑一个意图的。反射要在**任何**
#:   情况下第一个被选中，哪怕此刻连招正打着、甚至另一条反射也在冷却中。
#:   90 是留给「真正生死攸关」的那一层（急停走的是另一条通道：
#:   物理拦截，不经这里）。
REFLEX_INTENT_PRIORITY = 90

EmitFn = Callable[[Event], None]


# --------------------------------------------------------------------------
# 错误
# --------------------------------------------------------------------------


#: 执行层「限流」拒绝的原因前缀（2026-10-19）。
#:
#: ★ 抄了一份 ``actuator/safety.py`` 的 reason 格式（``MIN_GAP:{action}(..)``）。
#:   故意不 import：``policy`` 不该依赖 ``actuator``（层次反了）。
#:   代价是可能漂移 —— 改 safety 的 reason 格式时这里要跟着改。
THROTTLE_REASON_PREFIXES: tuple[str, ...] = ("MIN_GAP", "RATE_LIMITED")


def _is_throttled_reason(*texts: str) -> bool:
    """这个拒绝是「**被限流**」吗（动作合法，只是现在太快了）。

    ★ 为什么要单独认出来（2026-10-19）
      ``MIN_GAP`` / ``RATE_LIMITED`` 的含义是「这个动作**可以**做，
      只是离上一次太密」。它**不是**失败：

        · 切人被限流 -> 重发一次就过去了；
        · 技能被限流 -> 重发一次就过去了。

      而「配置错了」「窗口没焦点」「用户抢了」是**真失败**，
      重试没有意义，必须中止或跳过。

      把两者混为一谈（以前一律「响亮跳过」）的后果是：
      技能**根本没放出来**，连招却当它做过了接着往下打。
    """
    hay = " ".join(t for t in texts if t)
    if not hay:
        return False
    return any(p in hay for p in THROTTLE_REASON_PREFIXES)


class ComboLoadError(Exception):
    """连招配置非法。**一定带 文件 / 行号 / 连招名 / 字段名**。

    为什么不复用 :class:`~autopilot.policy.rule_loader.RuleLoadError`：
    那条错误里写死了「规则」这个词，而连招不是规则。把两者混成一个类
    会让报错文案说「规则 xxx」而实际是连招，排查时第一眼就拐错弯。
    """

    def __init__(
        self,
        message: str,
        *,
        source_file: str = "",
        line_no: int = 0,
        combo_name: str = "",
        field_name: str = "",
        cause: BaseException | None = None,
    ) -> None:
        self.message = message
        self.source_file = source_file
        self.line_no = line_no
        self.combo_name = combo_name
        self.field_name = field_name
        self.cause = cause
        super().__init__(self._render())

    def _render(self) -> str:
        loc = self.source_file or "<未指定文件>"
        if self.line_no:
            loc += f" 第 {self.line_no} 行"
        head = loc
        if self.combo_name:
            head += f"  连招 {self.combo_name!r}"
        if self.field_name:
            head += f"  字段 {self.field_name}"
        return head + f"\n  {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "source_file": self.source_file,
            "line_no": self.line_no,
            "combo_name": self.combo_name,
            "field_name": self.field_name,
        }


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ComboStep:
    """连招里的**一步**。加载后只读。

    :param action: 做什么。产出的意图类型就是它。
    :param char: 对谁做。``SWITCH_CHARACTER`` 时是**必需**的
        （转成 ``params.char_id``）；其他动作时用来决定**要不要先切人**。
    :param repeat: 这一步连做几次。每 tick 一次，所以节奏 = tick 频率。
    :param when: 这一步的额外守卫（同一套 DSL）。**为假或 UNKNOWN 就跳过
        这一步**，与「条件为假不触发」一致 —— 例如 ``e_ready()``：
        E 没好就直接切走，而不是卡在原地。
    :param gap_ms: 透传给执行层的 ``params.gap_ms`` —— 按完键后在执行层里
        **多睡一会儿**（代价：这一 tick 会超出 200ms 预算，故意的）。
    :param settle_ms: 执行器返回后，执行器**自己**再等这么久才出下一步。
        与 ``gap_ms`` 的区别是它不占用执行线程。
    :param hold_ms: 透传给执行层的 ``params.hold_ms``（按键按住时长）。
    """

    action: ActionType
    char: str = ""
    #: ★ 位置。``1``~``4``，或 0 = 没写。
    #:
    #: ★★ 为什么要有它，而不只是 ``char``
    #:   队伍组合是 ``A_120^4 ≈ 2 亿`` 种，写死角色名的配置对**任何**
    #:   具体队伍都只对一次。这里直接说「第 1 号位」，配置对**所有**
    #:   队伍都成立 —— 谁在 1 号位由 OCR 名册决定，不由配置决定。
    slot: int = 0
    #: 对**当前在场**的那一个做，不切人。``slot`` 与它互斥。
    #:
    #: ★ 这是「通用战斗」的全部秘密
    #:   ``{ onfield = true, action = "USE_SKILL", when = "e_ready()" }``
    #:   读作「 whoever 站着谁就按 E」。于是连招**不需要知道队伍是谁**
    #:   就仍然连贯：谁上来谁按，谁的 E 亮了就按。
    onfield: bool = False
    repeat: int = 1
    when: str = ""
    gap_ms: int | None = None
    hold_ms: int | None = None
    settle_ms: int = 0

    #: ★ 动作执行期间**额外按住**的键（2026-10-19，演示用）。
    #:
    #: 写 ``hold_keys = ["W"]`` 就是「这一步执行时同时按住 W」——
    #: 用来做「按住 W 冲刺 + 同时按 E 拉线」这种**组合按键**。
    #:
    #: ★ 为什么需要它（而不是用 ``MOVE_FORWARD`` 顶着）
    #:   ``MOVE_FORWARD`` 的 W 是**跨 tick 结转**的，而下一条意图开头的
    #:   「先松后按」会先把它放掉 —— W 和 E 永远不可能同时按着。
    #:   而 ``hold_keys`` 是**动作期内**的：动作结束就抬起。
    hold_keys: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # 流程控制三件套（2026-10-19）
    # ------------------------------------------------------------------
    #: ★ ``"ready"`` = **这一步真的做完了才推进**（唯一合法值）。
    #:
    #: ★ 为什么必须要有它，以及为什么它不叫「等后摇」
    #:   BetterGI 的反射表里每条动作串尾部几乎都挂着一个 ``ready``。
    #:   那是它的**同步点**：串跑完 → 等这个角色的动作落地 → 才隐式切下一条。
    #:   少了它，「钟离套盾 → 切人」就会在套盾后摇里把人切走，
    #:   症状是盾根本没上出来（而不是报错）。
    #:
    #:   我们**观测不到**角色的动画状态 —— 感知层没有这个信号。
    #:   所以这里不假装能算后摇帧，而是等一个**真的能观测到**的东西：
    #:   执行层 ``IntentExecutor.execute()`` 的**返回**。它是同步的，
    #:   返回即「按键按下 + hold/gap 都走完」，由
    #:   :meth:`ComboRunner.notify_done` 从主循环接进来。
    #:
    #:   必须配一个**上限**（``settle_ms`` 或整条连招的 ``timeout_ms``），
    #:   否则执行层一挂就永久卡在那一步 —— 所以加载期强制校验。
    wait: str = ""

    #: ★ 执行完成后的**后置条件**：「这一下真的生效了吗」。
    #:
    #:   ``wait = "ready"`` 只证明「键按出去了」。游戏里这一下可能被
    #:   **动画取消**吞掉 —— 技能没放出来，而键确实按了。
    #:   典型的可观测痕迹：按完 E 之后那个角色的 ``e_ready()`` 会翻成
    #:   false（技能进冷却）。所以写成::
    #:
    #:       { onfield = true, action = "USE_SKILL", wait = "ready",
    #:         until = "not e_ready()" }
    #:
    #:   不成立 = **响亮跳过**这一步（发 ``COMBO_STEP_NO_EFFECT``），
    #:   与 ``when`` 为假同一个方向：这一轮不产出，但看得见。
    until: str = ""

    #: ★ 执行完成后**跳到第几步**（1 起），而不是 N+1。``0`` = 不跳。
    #:
    #:   这是本引擎**唯一的控制流**。刻意不做 if/else 子句：
    #:   ``when`` 已经能当条件用，再加上 ``jump_to`` 就够写出
    #:   「能量不够就去普攻攒球，攒够了回来放大招」这种真实需求，
    #:   而整套 DSL 仍然只有「表达式 + 一张有序表」两样东西。
    #:
    #:   ⚠️ 死循环不靠配置自觉：连续跳超过 :data:`MAX_JUMPS` 次就中止
    #:     整条连招并记 ``COMBO_JUMP_LOOP``。
    jump_to: int = 0

    #: 加载时编译好的守卫（避免每帧重解析）
    compiled: CompiledExpr | None = field(default=None, compare=False, repr=False)
    #: ``until`` 的编译结果。与 :attr:`compiled` 分开 —— 两者求值时机不同
    #: （一个是出队前，一个是执行完成后）。
    until_compiled: CompiledExpr | None = field(
        default=None, compare=False, repr=False)
    line_no: int = 0

    @property
    def is_switch(self) -> bool:
        return self.action is ActionType.SWITCH_CHARACTER

    @property
    def needs_done(self) -> bool:
        """这一步是否需要**等执行完成**才推进。"""
        return bool(self.wait) or bool(self.until) or self.jump_to > 0

    @property
    def is_positional(self) -> bool:
        """这一步用的是位置而不是角色名。"""
        return bool(self.onfield) or self.slot > 0

    @property
    def expected_exec_ms(self) -> int:
        """这一步**预计**要占执行层多久（毫秒）。

        ★ 它只算「按键按住」和「按完的空档」—— 也就是配置里
          **明写**出来的部分。执行层真实耗时还包含开销，但它有
          自己的超时闸兜底，这里不需要重复算。

        ★ 为什么需要它（2026-10-19 加）
          ``wait = "ready"`` 的等待期限默认是 2000ms。长按的步骤
          （钟离长按 E 1500ms）光按住就 1500ms，期限只剩 380ms 余量 ——
          慢一点就判成 ``COMBO_DONE_TIMEOUT``，连招以为技能没放出去。
          症状是「盾没上出来」，日志里只有一条超时。
        """
        return max(0, int(self.hold_ms or 0)) + max(0, int(self.gap_ms or 0))

    def params(self) -> dict[str, Any]:
        """这一步要带进 ``IntentPacket.params`` 的东西。"""
        p: dict[str, Any] = {}
        if self.is_switch:
            p["char_id"] = self.char
        if self.gap_ms is not None:
            p["gap_ms"] = int(self.gap_ms)
        if self.hold_ms is not None:
            p["hold_ms"] = int(self.hold_ms)
        if self.hold_keys:
            p["hold_keys"] = list(self.hold_keys)
        return p

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "char": self.char,
            "slot": self.slot,
            "onfield": self.onfield,
            "repeat": self.repeat,
            "when": self.when,
            "gap_ms": self.gap_ms,
            "hold_ms": self.hold_ms,
            "settle_ms": self.settle_ms,
            "hold_keys": list(self.hold_keys),
            "wait": self.wait,
            "until": self.until,
            "jump_to": self.jump_to,
            "line_no": self.line_no,
        }

    def describe(self) -> str:
        if self.onfield:
            who = " -> 在场"
        elif self.slot:
            who = f" -> {self.slot} 号位"
        else:
            who = f"-> {self.char}" if self.char else ""
        rep = f" x{self.repeat}" if self.repeat > 1 else ""
        guard = f" when {self.when}" if self.when else ""
        wait = f" settle {self.settle_ms}ms" if self.settle_ms else ""
        if self.hold_keys:
            wait += f" hold_keys={list(self.hold_keys)}"
        if self.wait:
            wait += f" wait={self.wait}"
        if self.until:
            wait += f" until({self.until})"
        if self.jump_to:
            wait += f" jump->{self.jump_to}"
        return f"{self.action.value}{who}{rep}{guard}{wait}"


@dataclass(frozen=True)
class ComboSequence:
    """一条连招。加载后只读。"""

    name: str
    #: 什么时候开始跑。同一套 DSL，白名单与规则**完全一致**。
    trigger: str
    steps: tuple[ComboStep, ...]
    #: 默认**开着**（与 ``rule_loader`` 一致：「你写了就是开了」）。
    #: 发布的那份 ``combos.toml`` 里每条都显式写了 ``enabled = false`` ——
    #: 关不关是**配置的事**，不是解析器猜的事。
    enabled: bool = True
    note: str = ""
    #: 场景过滤。跑到一半场景变了 -> 中断（不是继续盲打）。
    scenes: tuple[str, ...] = ("WORLD",)
    #: 跑完 / 被打断后多久不许再启动。
    cooldown_ms: int = 0
    #: 命中 priority 小于它的普通规则会打断它（见模块常量说明）。
    break_below_priority: int = DEFAULT_BREAK_BELOW_PRIORITY
    #: 等一次切人完成的上限。超了 = 中断。
    switch_timeout_ms: int = 3000
    #: 整条连招的上限。超了 = 中断。防「永远走不完」。
    timeout_ms: int = 20000
    #: 产出的 IntentPacket 的 ttl。序列每步都是当帧决策当帧执行，
    #: 所以很短 —— 过了就该作废，而不是让一个 3 秒前的意图落地。
    ttl_seconds: int = 2

    source_file: str = ""
    line_no: int = 0
    compiled: CompiledExpr | None = field(default=None, compare=False, repr=False)

    @property
    def total_actions(self) -> int:
        return sum(s.repeat for s in self.steps)

    def matches_scene(self, scene_id: str) -> bool:
        return not self.scenes or scene_id in self.scenes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trigger": self.trigger,
            "enabled": self.enabled,
            "note": self.note,
            "scenes": list(self.scenes),
            "cooldown_ms": self.cooldown_ms,
            "break_below_priority": self.break_below_priority,
            "switch_timeout_ms": self.switch_timeout_ms,
            "timeout_ms": self.timeout_ms,
            "ttl_seconds": self.ttl_seconds,
            "steps": [s.to_dict() for s in self.steps],
            "source_file": self.source_file,
            "line_no": self.line_no,
        }

    def describe(self) -> str:
        flag = "" if self.enabled else "  (已禁用)"
        body = " -> ".join(s.describe() for s in self.steps)
        return (f"{self.name}{flag}\n"
                f"    when: {self.trigger}\n"
                f"    序列: {body}")


@dataclass(frozen=True)
class ReflexRule:
    """一条危机反射：**无论连招走到哪一步，先救眼前这一下**。

    ★ 为什么它不是「一条连招」
      连招的 ``trigger`` 只在**空闲**时生效，而危机恰恰发生在连招
      **正在打**的时候 —— 等连招走完再闪避，人已经被打死了。
      它也不能靠 ``break_below_priority`` 让路解决：那只是让规则
      插队，不保证「立刻、不管什么情况」。

      所以它是一个**独立于连招的、更高的层**：命中即
      **中止正在进行的连招**，发自己的意图，之后有一段**静默锁**
      让这一次闪避的无敌帧真的过去。

    ★ ``cooldown_ms`` 与 ``lock_ms`` 缺一不可
      没有 ``cooldown_ms``：条件一旦成立，每 tick 都会触发一次闪避 ——
      表现为「贴着怪疯狂翻滚」，比不动还糟。
      没有 ``lock_ms``：闪避按完的那一帧条件**照样成立**，于是下一 tick
      又闪一次。冷却挡不住它，因为冷却是从**上一次触发**算起，
      而 ``gap_ms`` 可能比冷却短。

    ★ 为什么允许 ``onfield`` / ``slot``
      「用大招躲」要指人。而和连招一样的两条规矩在这里也成立：
      **不写角色名**（写死名字对 A_120^4 种队伍只对一种成立），
      用 ``slot`` 或 ``onfield``。
    """

    name: str
    action: ActionType
    when: str
    note: str = ""
    #: 重复触发的冷却。**必须有**（见类注释）。
    cooldown_ms: int = 3000
    #: 触发后这段时间内**什么都不做**，让无敌帧过去。
    lock_ms: int = 300
    hold_ms: int | None = None
    gap_ms: int | None = None
    slot: int = 0
    onfield: bool = False
    scenes: tuple[str, ...] = ()
    enabled: bool = True
    compiled: CompiledExpr | None = field(default=None, compare=False, repr=False)
    line_no: int = 0

    def params(self) -> dict[str, Any]:
        p: dict[str, Any] = {}
        if self.gap_ms is not None:
            p["gap_ms"] = int(self.gap_ms)
        if self.hold_ms is not None:
            p["hold_ms"] = int(self.hold_ms)
        return p

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "action": self.action.value, "when": self.when,
            "note": self.note, "cooldown_ms": self.cooldown_ms,
            "lock_ms": self.lock_ms, "slot": self.slot, "onfield": self.onfield,
            "scenes": list(self.scenes), "enabled": self.enabled,
            "line_no": self.line_no,
        }

    def describe(self) -> str:
        who = " -> 在场" if self.onfield else (
            f" -> {self.slot} 号位" if self.slot else "")
        return f"{self.action.value}{who} when {self.when}"


@dataclass(frozen=True)
class ComboBook:
    """一份连招配置。**加载期就已经全部校验过**。"""

    sequences: tuple[ComboSequence, ...] = ()
    source_file: str = ""
    #: 危机反射（2026-10-19 新增）。同一份文件里的 ``[[reflex]]``。
    reflexes: tuple[ReflexRule, ...] = ()

    def __iter__(self) -> Iterator[ComboSequence]:
        return iter(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def enabled(self) -> tuple[ComboSequence, ...]:
        return tuple(s for s in self.sequences if s.enabled)

    @property
    def enabled_reflexes(self) -> tuple[ReflexRule, ...]:
        return tuple(r for r in self.reflexes if r.enabled)

    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sequences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "count": len(self.sequences),
            "enabled": [s.name for s in self.enabled],
            "sequences": [s.to_dict() for s in self.sequences],
            "reflexes": [r.to_dict() for r in self.reflexes],
        }


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------

#: [[combo]] 的合法字段。表以外的字段 = 大概率拼错了。
COMBO_FIELDS: frozenset[str] = frozenset({
    "name", "trigger", "sequence", "enabled", "note", "scenes",
    "cooldown_ms", "break_below_priority", "switch_timeout_ms",
    "timeout_ms", "ttl_seconds",
})

#: [[reflex]] 的合法字段。
REFLEX_FIELDS: frozenset[str] = frozenset({
    "name", "action", "when", "note", "cooldown_ms", "lock_ms",
    "hold_ms", "gap_ms", "slot", "onfield", "scenes", "enabled",
})

#: sequence 里每一步的合法字段。
STEP_FIELDS: frozenset[str] = frozenset({
    "action", "char", "slot", "onfield", "repeat", "when",
    "gap_ms", "hold_ms", "settle_ms", "hold_keys",
    "wait", "until", "jump_to",
})

#: 顶层允许的表。
#: ``reflex`` 是危机反射（2026-10-19 新增）—— 见 :class:`ReflexRule`。
TOP_LEVEL_TABLES: frozenset[str] = frozenset({"meta", "combo", "reflex"})

#: ``wait`` 的唯一合法值。
#:
#: ★ 为什么只有一个值还留这个字段
#:   将来真能观测到动画状态了，加第二个值（``"anim"``）是**加一个枚举**，
#:   而不是在改语义。现在留着一个单值字段，是为了让配置里能写出来、
#:   能被读出来、能进单测 —— 也让「这里有个同步点」在文件里**看得见**。
WAIT_VALUES: frozenset[str] = frozenset({"ready"})

#: 一次执行里最多允许连续跳多少次。超了中止整条连招。
#:
#: ★ 为什么要有上限，而不能只靠 ``timeout_ms``
#:   ``jump_to`` 能写出死循环（「跳回自己」），那种死循环**每一 tick 都在
#:   正常推进**，于是 ``timeout_ms`` 要等整整几十秒才兜得住 ——
#:   而那段时间里连招会反复发同一个动作。
#:   跳计数是**结构性**的兜底：64 次跳转在任何配置下都远超过正常轮转。
MAX_JUMPS = 64

#: ``until`` 的观察窗上限（毫秒）。
#:
#: ★ 为什么是常量而不是连招配置项
#:   「按完键多久之内能在画面上看出来生效」是**游戏机制**决定的，
#:   不是口味。把它开放成配置项只会让人调到 300ms 然后得到一堆
#:   ``COMBO_STEP_NO_EFFECT``，而症状看起来像「引擎坏了」。
#:   2000ms 覆盖了技能进冷却的感知延迟（引擎 200ms 一帧）。
DEFAULT_UNTIL_TIMEOUT_MS = 2000

REPEAT_RANGE = (1, 64)
GAP_RANGE = (0, 10_000)
SETTLE_RANGE = (0, 10_000)
COOLDOWN_RANGE = (0, 3_600_000)
SWITCH_TIMEOUT_RANGE = (100, 60_000)
COMBO_TIMEOUT_RANGE = (500, 300_000)
BREAK_PRIORITY_RANGE = (0, 101)
TTL_RANGE = (1, 300)


def _parse_reflex(p: Path, raw: Any, idx: int, ev: DslEvaluator,
                  text: str) -> ReflexRule:
    """解析一条 ``[[reflex]]``。

    ★ 校验为什么这么啰嗦
      反射是**最高优先级**的一层：它错了不是「少闪一次」，而是
      「在你最需要它的时候它按了别的东西」。所以每一条错误信息
      都得让人一眼知道是哪条、哪个字段、怎么改。
    """
    tag = f"第 {idx + 1} 条反射"
    if not isinstance(raw, dict):
        raise ComboLoadError(
            f"{tag}不是表（每条反射应写成一张表）",
            source_file=str(p), field_name="reflex",
        )
    _check_fields(p, raw, REFLEX_FIELDS, "reflex", tag)

    name = str(raw.get("name") or f"反射{idx + 1}").strip()
    line_no = _find_line(text, "name", 0)

    action = _parse_action(p, raw.get("action"), f"{tag}", line_no,
                           allowed=_REFLEX_ACTIONS)
    when = str(raw.get("when") or "").strip()
    if not when:
        raise ComboLoadError(
            f"{tag}（{name}）必须写 when —— 反射就是「满足条件才插手」。"
            f"\n  写法与连招的 when 完全一致，例如："
            f'\n    when = "hp() < 0.35"',
            source_file=str(p), line_no=line_no, combo_name=name,
            field_name="when",
        )
    try:
        compiled = ev.compile(when, rule_name=f"reflex[{name}]",
                             source_file=str(p), line_no=line_no)
    except RuleEvalError as exc:
        raise ComboLoadError(
            f"{tag}（{name}）的 when 不合法：{exc.message}",
            source_file=str(p), line_no=line_no, combo_name=name,
            field_name="when", cause=exc,
        ) from exc

    def num(key: str, default: int, lo: int, hi: int) -> int:
        v = raw.get(key, default)
        if isinstance(v, bool) or not isinstance(v, int):
            raise ComboLoadError(
                f"{tag}的 {key} 必须是整数，当前是 {v!r}",
                source_file=str(p), line_no=line_no, combo_name=name,
                field_name=key,
            )
        if not lo <= v <= hi:
            raise ComboLoadError(
                f"{tag}的 {key}={v} 超出 {lo}~{hi}",
                source_file=str(p), line_no=line_no, combo_name=name,
                field_name=key,
            )
        return v

    cooldown = num("cooldown_ms", DEFAULT_REFLEX_COOLDOWN_MS,
                   1, 60_000)
    lock = num("lock_ms", DEFAULT_REFLEX_LOCK_MS, 0, 10_000)

    slot = int(raw.get("slot") or 0)
    onfield = bool(raw.get("onfield", False))
    if slot and onfield:
        raise ComboLoadError(
            f"{tag}同时写了 slot={slot} 和 onfield=true —— 二选一",
            source_file=str(p), line_no=line_no, combo_name=name,
            field_name="onfield",
        )
    if slot and not 1 <= slot <= 4:
        raise ComboLoadError(
            f"{tag}的 slot={slot} 越界，队伍只有 4 个位置（1~4）",
            source_file=str(p), line_no=line_no, combo_name=name,
            field_name="slot",
        )

    scenes = _parse_scenes(p, raw.get("scenes"), line_no, name)
    enabled = bool(raw.get("enabled", True))

    if not enabled:
        note = str(raw.get("note") or "")
        if not any("一" <= c <= "鿿" for c in note):
            # 与连招同一条规矩：关着的必须说明为什么关。
            # 反射表比连招更需要 —— 一条「关掉了但没说为什么」的闪避
            # 是最难在真机上发现的东西。
            raise ComboLoadError(
                f"{tag}（{name}）enabled = false 但没写中文 note。"
                f"\n  关着的反射必须说明「为什么关」—— 用户看不到它，"
                f"也就永远不知道它曾经存在。",
                source_file=str(p), line_no=line_no, combo_name=name,
                field_name="note",
            )

    return ReflexRule(
        name=name, action=action, when=when,
        note=str(raw.get("note") or ""),
        cooldown_ms=cooldown, lock_ms=lock,
        hold_ms=raw.get("hold_ms"), gap_ms=raw.get("gap_ms"),
        slot=slot, onfield=onfield, scenes=scenes, enabled=enabled,
        compiled=compiled, line_no=line_no,
    )


def _parse_action(p: Path, raw: Any, tag: str, line_no: int,
                  *, allowed: Iterable[ActionType] | None = None
                  ) -> ActionType:
    """把 ``action = "USE_SKILL"`` 变成枚举，非法就**列出合法值**。"""
    if not isinstance(raw, str) or not raw.strip():
        raise ComboLoadError(
            f"{tag}缺 action",
            source_file=str(p), line_no=line_no, field_name="action",
        )
    try:
        action = ActionType(raw.strip())
    except ValueError as exc:
        allowed_set = tuple(allowed) if allowed else tuple(ActionType)
        raise ComboLoadError(
            f"{tag}的 action={raw!r} 不在白名单里。"
            f"\n  合法动作：{', '.join(a.value for a in allowed_set)}",
            source_file=str(p), line_no=line_no, field_name="action",
            cause=exc,
        ) from exc
    if allowed is not None and action not in tuple(allowed):
        raise ComboLoadError(
            f"{tag}的 action={action.value} 不能用在反射上。"
            f"\n  反射只允许：{', '.join(a.value for a in allowed)}",
            source_file=str(p), line_no=line_no, field_name="action",
        )
    return action


#: 反射允许的动作。
#:
#: ★ 为什么把 ``SWITCH_CHARACTER`` 挡在外面
#:   「闪避」的定义是**立刻解决眼前这一下**。反射里切人会让这一下
#:   变成「切过去，站定，再等切换完成」—— 那是**延迟**的相反面。
#:   想用反射救人 -> 那不是反射，那是一条带高优先级的连招。
_REFLEX_ACTIONS: tuple[ActionType, ...] = (
    ActionType.DODGE,
    ActionType.USE_SKILL,
    ActionType.USE_BURST,
    ActionType.NORMAL_ATTACK,
    ActionType.SPRINT,
    ActionType.MOVE_BACK,
    ActionType.MOVE_LEFT,
    ActionType.MOVE_RIGHT,
    ActionType.USE_PICKUP,
)

#: 反射的默认冷却。
#:
#: 3000ms 是「闪避 + 无敌帧 + 落地」的实际时长量级。它必须存在，
#: 否则条件一旦成立就会每 tick 触发一次 —— 表现是贴着怪无限翻滚。
DEFAULT_REFLEX_COOLDOWN_MS = 3000

#: 反射触发后的静默锁（让无敌帧真的过去）。
#:
#: 与 ``cooldown_ms`` 分开的原因见 :class:`ReflexRule`：冷却防的是
#: 「条件持续成立时反复触发」，锁防的是「刚闪完那一帧又闪」。
DEFAULT_REFLEX_LOCK_MS = 300


def _check_fields(p: Path, raw: dict[str, Any], allowed: frozenset[str],
                  field: str, tag: str) -> None:
    """表里有表以外的字段 = 拼错了。**响亮**，别当没看见。

    静默忽略的代价：用户写了 ``jump_to = 5`` 拼成 ``jumpto = 5``，
    看到「配了没生效」会先去怀疑引擎，而不是自己的手滑。
    """
    extra = sorted(set(raw) - set(allowed))
    if extra:
        raise ComboLoadError(
            f"{tag}里有不认识的字段：{', '.join(extra)}。"
            f"\n  合法字段：{', '.join(sorted(allowed))}",
            source_file=str(p), field_name=field,
        )


def _parse_scenes(p: Path, raw: Any, line_no: int,
                  name: str) -> tuple[str, ...]:
    """``scenes`` -> 合法场景 id 元组。空 = 不限。"""
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ComboLoadError(
            f"scenes 必须是数组（场景不写 = 不限）",
            source_file=str(p), line_no=line_no, combo_name=name,
            field_name="scenes",
        )
    out: list[str] = []
    for s in raw:
        sid = str(s).strip()
        if sid not in VALID_SCENE_IDS:
            raise ComboLoadError(
                f"场景 {sid!r} 不认识。"
                f"\n  合法场景：{', '.join(sorted(VALID_SCENE_IDS))}",
                source_file=str(p), line_no=line_no, combo_name=name,
                field_name="scenes",
            )
        out.append(sid)
    return tuple(out)


def load_combos(
    path: Path | str,
    *,
    evaluator: DslEvaluator | None = None,
) -> ComboBook:
    """加载一份连招配置。

    :raises ComboLoadError: 文件不存在 / 语法错 / 字段非法 / DSL 不合法
    """
    return load_combos_from_files([Path(path)], evaluator=evaluator)


def load_combos_from_files(
    files: Iterable[Path | str],
    *,
    evaluator: DslEvaluator | None = None,
) -> ComboBook:
    """加载多个文件。连招**保持声明顺序**（不用 priority 排）——
    它们不参与同一条决策链的竞争，顺序就是玩家写的顺序。"""
    ev = evaluator or _DEFAULT_EVALUATOR
    # ★ 先物化成 list：下面要在遍历完之后再回头拿 ``files[0]`` 当
    #   source_file，传生成器的话那时已经被耗空了（StopIteration）。
    paths = [Path(f) for f in files]
    if not paths:
        raise ComboLoadError("没有给任何连招文件")
    seqs: list[ComboSequence] = []
    reflexes: list[ReflexRule] = []
    seen: dict[str, str] = {}          # 连招名 -> 首次出现的文件
    seen_files: dict[str, Path] = {}

    for p in paths:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen_files:
            raise ComboLoadError(
                f"同一个文件被重复加载：{seen_files[key]} 与 {p}",
                source_file=str(p),
            )
        seen_files[key] = p

        data, text = _read_toml(p)
        _check_top_level(p, data, text)
        # ---- 危机反射（可选表）-----------------------------------
        #   反射表可以一条都没有（那时纯反射模式照常跑），
        #   但**写了就一定要能加载** —— 半条反射比没有更危险。
        rtable = data.get("reflex")
        if rtable is not None:
            if not isinstance(rtable, list) or not rtable:
                raise ComboLoadError(
                    "[[reflex]] 必须是非空数组",
                    source_file=str(p), field_name="reflex",
                )
            for idx, raw_reflex in enumerate(rtable):
                reflexes.append(_parse_reflex(p, raw_reflex, idx, ev, text))
        table = data.get("combo")
        if table is not None and not isinstance(table, list):
            raise ComboLoadError(
                "[[combo]] 必须是数组",
                source_file=str(p), field_name="combo",
            )
        # ★ 只有 ``[[combo]]`` 不再是硬要求（2026-10-19）
        #   有人可能只要「低血闪避」这类危机反射、不要任何连招 ——
        #   那时写一份只有 ``[[reflex]]`` 的文件是完全合理的，
        #   要求他再补一条没用的连招才能过加载，是纯粹的门槛。
        #   「至少有一张表」的检查在下面统一做。
        if table is None:
            if rtable is None:
                raise ComboLoadError(
                    "缺少 [[combo]] 与 [[reflex]] 数组。连招应写成：\n"
                    "  [[combo]]\n"
                    '  name = "通用战斗"\n'
                    '  trigger = "has_nearby_enemy()"\n'
                    "  sequence = [\n"
                    # ★ 示例里**不写角色名**
                    #   写死名字对 A_120^4 种队伍只对一种成立；
                    #   而且照抄示例的人会把别人的队伍抄进来。
                    '    { onfield = true, action = "USE_SKILL", when = "e_ready()" },\n'
                    "  ]\n"
                    "  危机反射（可选，另起一张表）：\n"
                    "  [[reflex]]\n"
                    '  name = "低血闪避"\n'
                    '  action = "DODGE"\n'
                    '  when = "hp() < 0.35"',
                    source_file=str(p), field_name="combo",
                )
            table = []
        for idx, raw_combo in enumerate(table):
            seqs.append(_parse_combo(p, raw_combo, idx, ev, seen, text))

    if not seqs and not reflexes:
        raise ComboLoadError(
            "文件里 [[combo]] 与 [[reflex]] 都是空的 —— 至少有一样要写。",
            source_file=",".join(str(f) for f in paths))
    return ComboBook(sequences=tuple(seqs), reflexes=tuple(reflexes),
                     source_file=str(paths[0]))


def load_optional_combos(
    path: Path | str | None = None,
    *,
    evaluator: DslEvaluator | None = None,
) -> ComboBook | None:
    """**文件不存在就返回 None**，存在但坏了就抛。

    这是「没配 ``combos.toml`` 就退回纯反射模式」的那条兼容通道。

    ⚠️ 两种情况必须分开：
      * **没有文件** = 玩家没写连招 = 正常配置，返回 None；
      * **文件坏了** = 写了但写错 = 故障，响亮失败。
      把后者也吞成 None 的话，玩家会看到「引擎不认我的连招」，
      而屏幕和日志上什么都没有 —— 最难查的一类静默失效。
    """
    p = Path(path) if path else DEFAULT_COMBOS_PATH
    if not p.exists():
        logger.info("没有连招配置（%s），按纯反射模式运行", p)
        return None
    book = load_combos(p, evaluator=evaluator)
    logger.info("连招配置已加载：%s（%d 条，开着 %d 条）",
                p, len(book), len(book.enabled))
    return book


def _read_toml(p: Path) -> tuple[dict[str, Any], str]:
    if not p.exists():
        raise ComboLoadError(f"连招文件不存在：{p}", source_file=str(p))
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ComboLoadError(f"读取失败：{exc}", source_file=str(p),
                             cause=exc) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ComboLoadError(f"文件不是 UTF-8：{exc}", source_file=str(p),
                             cause=exc) from exc
    try:
        return tomllib.loads(text), text
    except tomllib.TOMLDecodeError as exc:
        raise ComboLoadError(f"TOML 语法错误：{exc}", source_file=str(p),
                             cause=exc) from exc


def _check_top_level(p: Path, data: dict[str, Any], text: str) -> None:
    """顶层表白名单。``[[combos]]`` 写成复数会被整个静默忽略。"""
    unknown = set(data) - TOP_LEVEL_TABLES
    if not unknown:
        return
    names = sorted(unknown)
    line = next(
        (i for i, ln in enumerate(text.splitlines(), 1)
         if ln.strip().strip("[]").strip() in unknown), 0)
    raise ComboLoadError(
        f"顶层有未知表 {names}。只允许 {sorted(TOP_LEVEL_TABLES)}。"
        f"（拼错的表会被整个忽略掉 —— 比如 [[combos]] 不会被当成 [[combo]]）",
        source_file=str(p), line_no=line, field_name=names[0],
    )


def _find_line(text: str, needle: str, start: int = 0) -> int:
    """尽力定位一个片段所在行。

    纯文本扫描，不是真解析器 —— ``sequence`` 是内联表数组，逐元素算行号
    要真 TOML 解析器的位置跟踪，投入产出不划算。**差一两行不影响排障**，
    找不到就返回 0（调用方据此省略行号）。
    """
    if not needle:
        return 0
    for i, ln in enumerate(text.splitlines()[start:], start + 1):
        if needle in ln:
            return i
    return 0


def _parse_combo(
    p: Path, raw: Any, idx: int, ev: DslEvaluator,
    seen: dict[str, str], text: str,
) -> ComboSequence:
    where = str(idx)
    if not isinstance(raw, dict):
        raise ComboLoadError(
            f"第 {where} 段不是表（每条连招应写成 [[combo]]）",
            source_file=str(p),
        )

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ComboLoadError(
            f"第 {where} 段缺少 name，或不是非空字符串（当前 {name!r}）",
            source_file=str(p),
        )
    name = name.strip()
    line_no = _find_line(text, f'name = "{name}"')
    if name in seen:
        raise ComboLoadError(
            f"连招名 {name!r} 重复（已出现在 {seen[name]}）。"
            f"连招名必须全局唯一，否则日志里分不清是哪条在跑。",
            source_file=str(p), line_no=line_no, combo_name=name,
        )
    seen[name] = str(p)

    unknown_fields = set(raw) - COMBO_FIELDS
    if unknown_fields:
        near = _suggest_fields(unknown_fields, COMBO_FIELDS)
        raise ComboLoadError(
            f"有未知字段 {sorted(unknown_fields)}。"
            + (f" 是否想写 {near}？" if near else "")
            + f"\n  合法字段：{', '.join(sorted(COMBO_FIELDS))}",
            source_file=str(p), line_no=line_no, combo_name=name,
        )

    raw_trigger = raw.get("trigger")
    if not isinstance(raw_trigger, str) or not raw_trigger.strip():
        raise ComboLoadError(
            f"缺少 trigger（开始条件），当前值 {raw_trigger!r}",
            source_file=str(p), line_no=line_no, combo_name=name,
            field_name="trigger",
        )
    try:
        compiled = ev.compile(raw_trigger.strip(), rule_name=name,
                              source_file=str(p),
                              line_no=_find_line(text, "trigger", line_no - 1))
    except RuleEvalError as exc:
        raise ComboLoadError(
            f"trigger 表达式不合法：{exc.message}",
            source_file=str(p),
            line_no=_find_line(text, "trigger", line_no - 1),
            combo_name=name, field_name="trigger", cause=exc,
        ) from exc

    steps = _parse_sequence(p, raw.get("sequence"), name, ev, text, line_no)

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ComboLoadError(
            f"enabled 必须是布尔，当前是 {type(enabled).__name__}",
            source_file=str(p), line_no=line_no, combo_name=name,
            field_name="enabled",
        )
    note = raw.get("note", "")
    if not isinstance(note, str):
        raise ComboLoadError(
            f"note 必须是字符串，当前是 {type(note).__name__}",
            source_file=str(p), line_no=line_no, combo_name=name,
            field_name="note",
        )
    if not enabled and not note.strip():
        # 与规则同一条规矩：关着的东西必须解释为什么关，
        # 否则「临时关一下」会变成垃圾桶，没人再敢删。
        raise ComboLoadError(
            "enabled = false 但没有 note。关着的连招必须写一句中文原因。",
            source_file=str(p), line_no=line_no, combo_name=name,
            field_name="note",
        )

    scenes = _scenes_field(raw.get("scenes"), p, name, line_no)
    return ComboSequence(
        name=name, trigger=raw_trigger.strip(), steps=steps,
        enabled=enabled, note=note, scenes=scenes,
        cooldown_ms=_int_field(raw, "cooldown_ms", COOLDOWN_RANGE, p, name,
                               line_no, default=0),
        break_below_priority=_int_field(
            raw, "break_below_priority", BREAK_PRIORITY_RANGE, p, name,
            line_no, default=DEFAULT_BREAK_BELOW_PRIORITY),
        switch_timeout_ms=_int_field(
            raw, "switch_timeout_ms", SWITCH_TIMEOUT_RANGE, p, name,
            line_no, default=3000),
        timeout_ms=_int_field(raw, "timeout_ms", COMBO_TIMEOUT_RANGE, p, name,
                              line_no, default=20000),
        ttl_seconds=_int_field(raw, "ttl_seconds", TTL_RANGE, p, name,
                               line_no, default=2),
        source_file=str(p), line_no=line_no, compiled=compiled,
    )


def _parse_sequence(
    p: Path, raw: Any, name: str, ev: DslEvaluator,
    text: str, combo_line: int,
) -> tuple[ComboStep, ...]:
    seq_line = _find_line(text, "sequence", max(combo_line - 1, 0))
    if raw is None:
        raise ComboLoadError(
            "缺少 sequence（要按什么顺序打）。"
            "没有序列的连招就是普通规则，别写成连招。",
            source_file=str(p), line_no=combo_line, combo_name=name,
            field_name="sequence",
        )
    if not isinstance(raw, list) or not raw:
        raise ComboLoadError(
            f"sequence 必须是非空数组，当前是 "
            f"{'空数组' if isinstance(raw, list) else type(raw).__name__}",
            source_file=str(p), line_no=seq_line, combo_name=name,
            field_name="sequence",
        )

    steps: list[ComboStep] = []
    for i, raw_step in enumerate(raw):
        steps.append(_parse_step(p, raw_step, i, name, ev, text, seq_line))

    # ---- jump_to 的范围校验 ------------------------------------------
    # ★ 为什么放在这里而不是 :func:`_parse_step`
    #   「第 N 步跳到第几步合法」只有**知道总步数**才答得出来。
    #   在 ``_parse_step`` 里只能验「是正整数」，越界要等到运行时才炸 ——
    #   那时候用户已经跑起来了，而症状是「连招跑到某一步突然停了」。
    total = len(steps)
    for i, st in enumerate(steps):
        if st.jump_to and not 1 <= st.jump_to <= total:
            raise ComboLoadError(
                f"第 {i + 1} 步的 jump_to = {st.jump_to} 越界 —— "
                f"这条连招一共 {total} 步。\n"
                f"  jump_to 从 **1** 起数（就是上面写的「第 N 步」那个 N）。",
                source_file=str(p), line_no=st.line_no or seq_line,
                combo_name=name, field_name="jump_to",
            )
    return tuple(steps)


def _parse_step(
    p: Path, raw: Any, idx: int, combo: str, ev: DslEvaluator,
    text: str, seq_line: int,
) -> ComboStep:
    tag = f"第 {idx + 1} 步"
    if not isinstance(raw, dict):
        raise ComboLoadError(
            f"{tag}不是表（每步应写成 "
            f'{{ slot = 1, action = "USE_SKILL" }}）',
            source_file=str(p), line_no=seq_line, combo_name=combo,
            field_name="sequence",
        )

    unknown_fields = set(raw) - STEP_FIELDS
    if unknown_fields:
        near = _suggest_fields(unknown_fields, STEP_FIELDS)
        raise ComboLoadError(
            f"{tag}有未知字段 {sorted(unknown_fields)}。"
            + (f" 是否想写 {near}？" if near else "")
            + f"\n  合法字段：{', '.join(sorted(STEP_FIELDS))}",
            source_file=str(p), line_no=seq_line, combo_name=combo,
            field_name="sequence",
        )

    char = raw.get("char", "")
    if not isinstance(char, str):
        raise ComboLoadError(
            f"{tag}的 char 必须是字符串，当前是 {type(char).__name__}",
            source_file=str(p), line_no=seq_line, combo_name=combo,
            field_name="char",
        )
    char = char.strip()

    # ---- slot（位置）与 onfield（在场）----------------------------
    # ★ 为什么两个都要，而不是把 slot 写成字符串 "onfield"
    #   slot 的值域是 1..4。混进一个字符串之后，每次用都得写
    #   ``if isinstance(slot, str)``，而那种分支迟早会被漏掉 ——
    #   漏掉的表现是「onfield 被当成 0，于是这一步对谁都能按」。
    # ★ 缺省 ≠ 0。``raw.get("slot", 0)`` 会把「没写 slot」变成
    #   「slot=0」，而 0 是 int，于是掉进下面的越界校验里炸掉 ——
    #   表现是「只写 onfield 的一步永远加载不了」。不写就是没指定。
    raw_slot = raw.get("slot")
    if raw_slot is None or raw_slot == "":
        slot = 0                      # 未指定（不是「0 号位」）
    elif isinstance(raw_slot, bool) or not isinstance(raw_slot, int):
        raise ComboLoadError(
            f"{tag}的 slot 必须是 1~4 的整数（谁在几号位由 OCR 决定），"
            f"当前是 {raw_slot!r}。想表达「当前在场的那个人」请写 "
            f'onfield = true。',
            source_file=str(p), line_no=seq_line, combo_name=combo,
            field_name="slot",
        )
    elif 1 <= raw_slot <= 4:
        slot = int(raw_slot)
    else:
        raise ComboLoadError(
            f"{tag}的 slot={raw_slot} 越界，队伍只有 4 个位置（1~4）。"
            f"\n  「1~4 之外」没有对应的人；想表达「当前在场的那个人」"
            f"请写 onfield = true。",
            source_file=str(p), line_no=seq_line, combo_name=combo,
            field_name="slot",
        )

    onfield = bool(raw.get("onfield", False))
    if slot and onfield:
        # ★ 两个都写不是「以其中一个为准」，是**配错了**。
        #   静默取一个的话，用户改配置时完全不知道自己改了半条。
        raise ComboLoadError(
            f"{tag}同时写了 slot={slot} 和 onfield=true —— "
            f"「固定几号位」和「当前在场」是两种互斥的意思。"
            f"\n  只要位置就去掉 onfield；只要在场就去掉 slot。",
            source_file=str(p), line_no=seq_line, combo_name=combo,
            field_name="onfield",
        )
    if char and (slot or onfield):
        raise ComboLoadError(
            f"{tag}同时写了 char={char!r} 和位置 —— 同上，二选一。"
            f"\n  ★ 写死角色名对 A_120^4 种队伍只对一种成立；"
            f"新配置请用 slot / onfield。",
            source_file=str(p), line_no=seq_line, combo_name=combo,
            field_name="char",
        )

    if slot:
        line_no = _find_line(text, f"slot = {slot}", max(seq_line - 1, 0)) \
            or seq_line
    elif onfield:
        line_no = _find_line(text, "onfield = true", max(seq_line - 1, 0)) \
            or seq_line
    else:
        line_no = _find_line(text, f'char = "{char}"', max(seq_line - 1, 0)) \
            or seq_line

    raw_action = raw.get("action")
    action = _parse_action(p, raw_action, tag, line_no)
    if action is ActionType.SWITCH_CHARACTER and not char:
        raise ComboLoadError(
            f"{tag}是 SWITCH_CHARACTER，必须写 char（切谁）。"
            f"切人靠角色名换数字键，没有名字就切不动。",
            source_file=str(p), line_no=line_no, combo_name=combo,
            field_name="char",
        )

    compiled = None
    raw_when = raw.get("when", "")
    if not isinstance(raw_when, str):
        raise ComboLoadError(
            f"{tag}的 when 必须是字符串，当前是 {type(raw_when).__name__}",
            source_file=str(p), line_no=line_no, combo_name=combo,
            field_name="when",
        )
    if raw_when.strip():
        try:
            compiled = ev.compile(raw_when.strip(), rule_name=f"{combo}#{idx}",
                                  source_file=str(p),
                                  line_no=_find_line(text, "when", line_no - 1))
        except RuleEvalError as exc:
            raise ComboLoadError(
                f"{tag}的 when 不合法：{exc.message}",
                source_file=str(p),
                line_no=_find_line(text, "when", line_no - 1),
                combo_name=combo, field_name="when", cause=exc,
            ) from exc

    def opt(key: str) -> int | None:
        if key not in raw:
            return None
        v = raw[key]
        if isinstance(v, bool) or not isinstance(v, int):
            raise ComboLoadError(
                f"{tag}的 {key} 必须是整数，当前是 {type(v).__name__}（{v!r}）",
                source_file=str(p), line_no=line_no, combo_name=combo,
                field_name=key,
            )
        return v

    # ---- hold_keys（组合按键，2026-10-19）---------------------------
    # 写 ``hold_keys = ["W"]`` = 「这一步执行时同时按住 W」。
    # 用它做「按住 W 冲刺 + 同时按 E 拉线」这种组合动作。
    hold_keys: tuple[str, ...] = ()
    raw_hold_keys = raw.get("hold_keys")
    if raw_hold_keys is not None:
        if isinstance(raw_hold_keys, str):
            raw_hold_keys = [raw_hold_keys]
        if not isinstance(raw_hold_keys, (list, tuple)):
            raise ComboLoadError(
                f"{tag}的 hold_keys 必须是键名数组，当前是 "
                f"{type(raw_hold_keys).__name__}（{raw_hold_keys!r}）",
                source_file=str(p), line_no=line_no, combo_name=combo,
                field_name="hold_keys",
            )
        names: list[str] = []
        for item in raw_hold_keys:
            raw_name = str(item).strip()
            if not raw_name:
                continue
            try:
                kid = KeyId.from_name(raw_name)
            except KeyIdError as exc:
                raise ComboLoadError(
                    f"{tag}的 hold_keys 里有认不出来的键名 {raw_name!r}。\n"
                    f"  这里写的是**键名**（W / SHIFT / LBUTTON），不是动作名。\n"
                    f"  {exc}",
                    source_file=str(p), line_no=line_no, combo_name=combo,
                    field_name="hold_keys", cause=exc,
                ) from exc
            if kid.is_sentinel:
                raise ComboLoadError(
                    f"{tag}的 hold_keys 里有 {raw_name!r} —— "
                    f"那是「不发按键」的哨兵，不能当组合键。",
                    source_file=str(p), line_no=line_no, combo_name=combo,
                    field_name="hold_keys",
                )
            names.append(kid.display_name)
        hold_keys = tuple(names)

    # ---- 流程控制三件套 --------------------------------------------
    # ★ 解析放在 ``action`` 之后、构造之前，顺序无所谓，但**错误信息
    #   一律带 tag / 行号 / 字段名** —— 配错一定要知道是哪一步的哪个字段。
    wait = raw.get("wait", "")
    if not isinstance(wait, str):
        raise ComboLoadError(
            f"{tag}的 wait 必须是字符串，当前是 {type(wait).__name__}",
            source_file=str(p), line_no=line_no, combo_name=combo,
            field_name="wait",
        )
    wait = wait.strip()
    if wait and wait not in WAIT_VALUES:
        raise ComboLoadError(
            f"{tag}的 wait = {wait!r} 不认识。合法值："
            f"{', '.join(sorted(WAIT_VALUES))}\n"
            f"  目前只有 ready = 「等执行层真的做完这一步再推进」"
            f"（BetterGI 反射表里每条动作串尾部那个同步点）。",
            source_file=str(p), line_no=line_no, combo_name=combo,
            field_name="wait",
        )

    until_compiled = None
    raw_until = raw.get("until", "")
    if not isinstance(raw_until, str):
        raise ComboLoadError(
            f"{tag}的 until 必须是字符串，当前是 {type(raw_until).__name__}",
            source_file=str(p), line_no=line_no, combo_name=combo,
            field_name="until",
        )
    if raw_until.strip():
        try:
            until_compiled = ev.compile(
                raw_until.strip(), rule_name=f"{combo}#{idx}.until",
                source_file=str(p),
                line_no=_find_line(text, "until", line_no - 1))
        except RuleEvalError as exc:
            raise ComboLoadError(
                f"{tag}的 until 不合法：{exc.message}",
                source_file=str(p),
                line_no=_find_line(text, "until", line_no - 1),
                combo_name=combo, field_name="until", cause=exc,
            ) from exc

    jump = opt("jump_to")
    if jump is not None and jump < 0:
        raise ComboLoadError(
            f"{tag}的 jump_to = {jump} 不能为负（0 = 不跳）",
            source_file=str(p), line_no=line_no, combo_name=combo,
            field_name="jump_to",
        )
    jump = jump or 0

    gap = opt("gap_ms")
    if gap is not None and not (GAP_RANGE[0] <= gap <= GAP_RANGE[1]):
        raise ComboLoadError(
            f"{tag}的 gap_ms = {gap} 超出范围 {list(GAP_RANGE)}",
            source_file=str(p), line_no=line_no, combo_name=combo,
            field_name="gap_ms",
        )
    hold = opt("hold_ms")
    if hold is not None and hold < 0:
        raise ComboLoadError(
            f"{tag}的 hold_ms = {hold} 不能为负",
            source_file=str(p), line_no=line_no, combo_name=combo,
            field_name="hold_ms",
        )
    settle = opt("settle_ms") or 0
    if not (SETTLE_RANGE[0] <= settle <= SETTLE_RANGE[1]):
        raise ComboLoadError(
            f"{tag}的 settle_ms = {settle} 超出范围 {list(SETTLE_RANGE)}",
            source_file=str(p), line_no=line_no, combo_name=combo,
            field_name="settle_ms",
        )
    repeat = opt("repeat")
    if repeat is None:
        repeat = 1
    if not (REPEAT_RANGE[0] <= repeat <= REPEAT_RANGE[1]):
        raise ComboLoadError(
            f"{tag}的 repeat = {repeat} 超出范围 {list(REPEAT_RANGE)}",
            source_file=str(p), line_no=line_no, combo_name=combo,
            field_name="repeat",
        )

    return ComboStep(
        action=action, char=char, slot=slot, onfield=onfield,
        repeat=repeat, when=raw_when.strip(),
        gap_ms=gap, hold_ms=hold, settle_ms=settle,
        hold_keys=hold_keys,
        wait=wait, until=raw_until.strip(), jump_to=jump,
        compiled=compiled, until_compiled=until_compiled,
        line_no=line_no,
    )


def _int_field(
    raw: dict[str, Any], key: str, rng: tuple[int, int],
    p: Path, combo: str, line: int, *, default: int | None = None,
) -> int:
    if key not in raw:
        if default is None:
            raise ComboLoadError(
                f"缺少 {key}", source_file=str(p), line_no=line,
                combo_name=combo, field_name=key,
            )
        return default
    v = raw[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise ComboLoadError(
            f"{key} 必须是整数，当前是 {type(v).__name__}（{v!r}）",
            source_file=str(p), line_no=line, combo_name=combo, field_name=key,
        )
    lo, hi = rng
    if not (lo <= v <= hi):
        raise ComboLoadError(
            f"{key} = {v} 超出范围 [{lo}, {hi}]",
            source_file=str(p), line_no=line, combo_name=combo, field_name=key,
        )
    return v


def _scenes_field(v: Any, p: Path, combo: str, line: int) -> tuple[str, ...]:
    if v is None:
        return ("WORLD",)
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        raise ComboLoadError(
            f"scenes 必须是字符串数组，当前是 {type(v).__name__}",
            source_file=str(p), line_no=line, combo_name=combo,
            field_name="scenes",
        )
    out: list[str] = []
    for s in v:
        if not isinstance(s, str):
            raise ComboLoadError(
                f"scenes 里必须是字符串，当前有 {type(s).__name__}（{s!r}）",
                source_file=str(p), line_no=line, combo_name=combo,
                field_name="scenes",
            )
        sid = s.strip()
        if sid not in VALID_SCENE_IDS:
            import difflib  # noqa: PLC0415

            near = difflib.get_close_matches(sid, sorted(VALID_SCENE_IDS), 1, 0.6)
            raise ComboLoadError(
                f"场景 {sid!r} 不存在。"
                + (f" 是否想写 {near[0]!r}？" if near else "")
                + f"\n  合法场景：{', '.join(sorted(VALID_SCENE_IDS))}",
                source_file=str(p), line_no=line, combo_name=combo,
                field_name="scenes",
            )
        if sid not in out:
            out.append(sid)
    return tuple(out)


def _suggest_fields(unknown: set[str], known: frozenset[str]) -> str:
    import difflib  # noqa: PLC0415

    hits: list[str] = []
    for u in sorted(unknown):
        m = difflib.get_close_matches(u, sorted(known), n=1, cutoff=0.6)
        if m:
            hits.append(f"{u} -> {m[0]}")
    return "; ".join(hits)


# --------------------------------------------------------------------------
# 状态机
# --------------------------------------------------------------------------

#: 状态机的三个相。**没有「READY」相** —— 没在跑的时候
#: :attr:`ComboRunner._seq is None`，少一个状态就少一处可能漏清的分支。
PHASE_STEP = "STEP"              # 准备出队下一步
PHASE_WAIT_SWITCH = "WAIT_SWITCH"  # 已发切人，等记账器更新
PHASE_SETTLE = "SETTLE"          # 等 settle_ms
#: 键按完了，等一个**感知可观测**的后置条件成立（``until``）。
#:
#: ★ 为什么它必须占一个相位而不是就地求值
#:   按完键那一刻，上下文里的 ``e_ready()`` 还是**按之前**的值 ——
#:   技能进冷却要等下一帧的感知才看得出来。就地问等于「刚按完 E 就问
#:   E 亮不亮」，永远答「亮」，后置条件形同虚设。
PHASE_UNTIL = "UNTIL"

#: 同一条原因的最短重复间隔（毫秒）。启动后 20 秒名册没填满时
#: 每 tick 都会撞同一个坑，不节流的话事件总线会被刷爆。
WARN_THROTTLE_MS = 5000


class ComboRunner:
    """把 :class:`ComboBook` 变成一串 :class:`IntentPacket`。

    :param book: 连招配置。``None`` = 没配连招（纯反射模式），
        整个 runner 不该被创建；调用方要自己处理。
    :param now_fn: 取当前时刻（毫秒）。测试注入虚拟时钟。
    :param active_slots: ``actuator.active_slot.ActiveSlotTracker``。
        用来**可观测地**确认切人完成 —— 不用固定 sleep 去猜。
    :param slot_index: ``actuator.active_slot.CharSlotIndex``。
        角色名 -> 槽位。查不到就跳过那一步（不猜）。
    :param static_slots: ``keymap.toml`` 的 ``[char_slots]``（名字 -> 槽位）。
        **优先级高于** ``slot_index``，与
        :meth:`autopilot.actuator.executor.IntentExecutor._slot_of` 同口径。

        ⚠️ 为什么必须同口径而不是「只用 OCR 名册」：
          玩家填 ``[char_slots]`` 的**唯一动机**是绕开启动后 ~20 秒的
          名册空窗（见模块文档）。连招只认 OCR 名册的话，那份配置对连招
          一点用没有 —— 而文档却让玩家填它。那就是**文档在骗人**。

    ★ 线程安全
      写入方有三个：tick 线程（:meth:`next_intent`）、
      物理输入监听线程（:meth:`abort`）、急停监视线程（:meth:`abort`）。
      临界区只做几次字典/字段读写。
    """

    def __init__(
        self,
        book: ComboBook,
        *,
        now_fn: Callable[[], int] | None = None,
        active_slots: Any = None,
        slot_index: Any = None,
        emit_event: EmitFn | None = None,
        evaluator: DslEvaluator | None = None,
        static_slots: Mapping[str, int] | None = None,
    ) -> None:
        self._book = book
        self._now = now_fn or (lambda: int(time.time() * 1000))
        self._active = active_slots
        self._index = slot_index
        self._static = dict(static_slots or {})
        self._emit_fn = emit_event
        self._evaluator = evaluator or _DEFAULT_EVALUATOR

        self._lock = threading.RLock()
        self._seq: ComboSequence | None = None
        self._step_idx = 0
        self._repeat_left = 0
        self._phase = PHASE_STEP
        self._phase_started_ms = 0
        self._settle_until_ms = 0
        self._switch_target: int | None = None
        self._switch_target_name = ""
        self._started_ms = 0
        self._cooldown_until_ms = 0
        self._last_warn_ms = -10_000_000

        # ---- 执行完成门（wait="ready" / until / jump_to）--------------
        #: 正在等 :meth:`notify_done` 的那一步。``None`` = 没在等。
        #:
        #: ★ 为什么用「等通知」而不是「算时间」
        #:   ``settle_ms`` 是拍一个数：按下去到能切人之间最少等这么久。
        #:   它永远有**两个错法** —— 拍短了切在后摇里（技能被吞），
        #:   拍长了白等。而 :meth:`notify_done` 是执行层**真的返回**的时刻：
        #:   按键按完、``hold_ms`` 按够、``gap_ms`` 等完。
        #:   那是唯一不靠猜的信号。
        #:
        #: ★ 为什么必须同时保留 ``_await_deadline_ms``
        #:   执行层万一卡死/抛异常，通知就永远不会来 ——
        #:   那时候连招会**永久停在这一步**，比不打更糟。
        #:   到点就放行并记 ``COMBO_DONE_TIMEOUT``，让它带着伤继续跑。
        self._await_step: ComboStep | None = None
        self._await_started_ms = 0
        self._await_deadline_ms = 0
        #: ``until`` 观察窗里盯着的那一步。
        self._until_step: ComboStep | None = None
        self._until_deadline_ms = 0
        #: 连续跳转计数，防 ``jump_to = 本步`` 这种死循环。
        self._jumps = 0
        #: 统计：等到过几次执行完成 / 抖掉过几次。
        self._done_waits = 0
        self._done_timeouts = 0
        # ---- 危机反射 ------------------------------------------------
        #: 每条反射下次可再触发的时间。
        self._reflex_cooldown: dict[str, int] = {}
        #: 触发后的静默锁到点时刻。**锁期间连招也让路**。
        self._reflex_lock_until_ms = 0
        self._reflex_fired = 0
        #: 累计统计。诊断面板看这个，不看日志。
        self._started_count = 0
        self._finished_count = 0
        self._aborted_count = 0
        self._last_reason = ""

    # -- 只读 ------------------------------------------------------------
    @property
    def book(self) -> ComboBook:
        return self._book

    @property
    def active(self) -> bool:
        """当前有正在进行的连招吗。"""
        with self._lock:
            return self._seq is not None

    def active_sequence(self) -> ComboSequence | None:
        with self._lock:
            return self._seq

    # -- 执行完成门 ---------------------------------------------------
    @property
    def awaiting_done(self) -> bool:
        """正在等执行层回报「上一步做完了」。

        主循环**可以**问这个来决定要不要接 :meth:`notify_done` 的回报，
        但不强制 —— 多回报一次只是多一次没人在等的通知。
        """
        with self._lock:
            return self._await_step is not None

    def notify_done(self, *, ok: bool = True, at_ms: int | None = None,
                    error: str = "", reason: str = "") -> bool:
        """执行层报「上一个意图做完了」。

        由 :class:`~autopilot.core.loop.CoreLoop` 在
        ``IntentExecutor.execute()`` **返回之后**调用。

        :param ok: 这一下**真按出去了吗**。``False`` 时与 ``when`` 为假
            同一个方向处理（响亮跳过该步），而不是假装完成。
        :param at_ms: 完成时刻。不传用当前时间。
        :param error: 失败原因，进事件上下文。**不要抛给调用方** ——
            主循环不该因为「一个意图失败了」而崩，它要继续跑下一 tick。
        :param reason: 执行层的**详细**拒绝原因（``MIN_GAP:...`` /
            ``KEYMAP: ...`` / ``USER_TAKEOVER:E`` 之类）。
            ★ 与 ``error`` 的区别：``error`` 是粗粒度结果（"REJECTED"），
              ``reason`` 才说清**为什么**。连招靠它分「该重试」还是
              「该中止」—— 只看 ``error`` 的话，这两种长得一模一样。
        :returns: 确实是在等某一步（也就是这次回报被收下了）吗。
            ``False`` = 没人等 —— 多半是那一步没有 ``wait``。
        """
        now = self._now() if at_ms is None else int(at_ms)
        throttled = False
        with self._lock:
            step = self._await_step
            if step is None:
                # ★ 没人等「完成门」—— 但**切人**也算一次执行（2026-10-19）。
                #   切人步没有 ``wait``，完成门是空的，所以切人的成败
                #   以前**根本没人收** —— 切人被 MIN_GAP 拒掉时，
                #   连招就那么卡在 PHASE_WAIT_SWITCH 里等到 3 秒超时，
                #   整条连招被中断（真机症状）。
                return self._on_switch_report(ok=ok, error=error,
                                              reason=reason, at_ms=now)
            seq = self._seq
            self._await_step = None
            waited = max(0, now - self._await_started_ms)
            if not ok and _is_throttled_reason(reason, error):
                # ★ 限流**不是失败**（2026-10-19）。
                #   MIN_GAP / RATE_LIMITED 的意思是「这个动作合法，
                #   只是现在离上一次太密了」—— 动作本身完全没问题。
                #   以前这里一律「响亮跳过该步」，于是连招把
                #   「被节流了一下」当成「这一步做过了」接着往下走，
                #   技能**根本没放出来**。
                #   正确处理是**重试**：把游标退回去，下一 tick 重发。
                throttled = True
                self._rewind_step_locked(step)
        self._done_waits += 1

        ctx: dict[str, Any] = {
            "combo": seq.name if seq else "",
            # ★ 用 ``_step_idx`` 而不是 ``_step_idx + 1``
            #   出队时游标已经 +1 了（``post`` 分支），所以此刻
            #   ``_step_idx`` **就是**刚做完那一步的 1 起编号。
            #   多加一个 1 的话，事件里写的步号会系统性地大 1 ——
            #   而用户是拿这条事件去对照 ``combos.toml`` 的。
            "step": self._step_idx,
            "action": step.action.value,
            "ok": bool(ok),
            "waited_ms": waited,
        }
        if error:
            ctx["error"] = str(error)

        if not ok:
            if throttled:
                # ★ 被限流 —— **重试**，不跳过（游标已经在上面退回去了）。
                #   用 INFO 而不是 WARN：这是引擎自己节流，**正常现象**，
                #   报成故障会让人以为引擎出问题。
                self._event(Severity.INFO, "COMBO_STEP_THROTTLED",
                            f"连招 {seq.name if seq else '?'} 的第 "
                            f"{ctx['step']} 步被限流（{step.action.value}）："
                            f"{reason or error or '太快了'}——下一拍重试",
                            ctx)
                return True
            # ★ 执行层自己都说了没按出去 -> 这一步等于没发生。
            #   当成「条件不满足」跳过，但**响亮**：症状是「连招在打，
            #   但某一步从来没生效过」，日志里没这行就完全查不出来。
            self._event(Severity.WARN, "COMBO_STEP_NOT_EXECUTED",
                        f"连招 {seq.name if seq else '?'} 的第 "
                        f"{ctx['step']} 步没按出去（{step.action.value}）："
                        f"{reason or error or '执行层拒绝'}——跳过这一步",
                        ctx)
            self._apply_jump_locked(step)
            return True

        if step.until_compiled is not None and seq is not None:
            self._enter_until_locked(seq, step, now)
        else:
            # ★ 这里**必须**也走推进。
            #   出队时 ``_step_idx`` 已经指向下一步了（``post`` 分支），
            #   所以本方法只负责「jump_to 与否」。漏掉这一行的话，
            #   带 wait 的最后一步会**永远挂在那里**等一个已经收到
            #   的回报 —— 症状是「配了 wait 之后连招走到一半就停了」。
            self._apply_jump_locked(step)
        return True

    def _rewind_step_locked(self, step: ComboStep) -> None:
        """把游标退回到 ``step``，让下一 tick **重发**这一步。

        ★ 为什么能退（2026-10-19）
          出队时（``post`` 分支）游标已经 ``+1`` 了，所以「退回去」
          就是 ``-1``。

          ★ 为什么 ``-1`` 一定对得上
            完成门只挂在**最后一次**重复上（``post`` 才挂），而只有
            挂了门的步骤才会走到本方法 —— 所以被报告的那一次**一定**
            是 ``post``，一定发生过 ``+1``。中间那几次重复没挂门，
            它们失败时根本不会走到这里。

        ★ 退回去之后要重置 ``_repeat_left``
          否则 ``repeat`` 计数器已经被扣掉了，重发时会少做几次。
          设成 ``step.repeat`` 就是「从头再做一遍」。

        ★ 不碰 ``_phase`` 以外的相位
          ``until`` 观察窗还没进（那要等成功之后才进），
          ``settle`` 也不该在这里重置 —— 那些是「做成了之后」的事。
        """
        if self._step_idx > 0:
            self._step_idx -= 1
        self._repeat_left = max(1, int(step.repeat))
        self._phase = PHASE_STEP

    def _on_switch_report(self, *, ok: bool, error: str, reason: str,
                          at_ms: int) -> bool:
        """处理「切人意图的执行结果」。返回是否收下了这份报告。

        ★ 为什么需要它（2026-10-19）
          切人步**没有** ``wait``，所以完成门是空的 —— 切人的成败
          以前根本没人收。于是切人被 ``MIN_GAP`` 拒掉时：

            切人没按出去 -> 执行层没记账 -> 在场槽位没变
              -> 连招卡在 ``PHASE_WAIT_SWITCH`` 等「切到 X 生效」
              -> 3 秒后 ``switch_timeout_ms`` 到点 -> **整条连招被中断**

          真机症状就是这个：连招「卡死中断」，日志里只有一句
          「切到 X 等了 3000ms 还没生效」，完全看不出是被限流了。

        ★ 三种结局
          · 成功      -> 什么都不做（记账器会更新，相位自己会看到）
          · 被限流    -> **重试**：相位退回去，下一 tick 重新发切人
          · 其它失败  -> 立刻中断并说清原因（别让用户等 3 秒超时）
        """
        if self._phase != PHASE_WAIT_SWITCH:
            # 不是切人的回报 —— 没人等就是没人等。
            return False
        if ok:
            # 切下去了。记账器会更新，PHASE_WAIT_SWITCH 自己会看到。
            return True
        if _is_throttled_reason(reason, error):
            # ★ 限流 -> 重试。相位退回去，**不动 _step_idx**：
            #   隐式切人的那一步游标本来就没动过，显式切人步
            #   （已经 +1）则会由下一步的隐式切人重新接手。
            target_name = self._switch_target_name
            self._phase = PHASE_STEP
            self._switch_target = None
            self._switch_target_name = ""
            logger.info("切人被限流（%s），下一拍重试", reason or error)
            # ★ 也要**响亮**（但用 INFO）：连招被限流是「引擎自己踩了刹车」，
            #   不是故障 —— 但它得**看得见**，否则排障时只看到「切人慢了」。
            self._event(Severity.INFO, "COMBO_STEP_THROTTLED",
                        f"连招切到 {target_name or '?'} 被限流："
                        f"{reason or error or '太快了'}——下一拍重试",
                        {"char": target_name, "action":
                         ActionType.SWITCH_CHARACTER.value,
                         "ok": False, "reason": reason or error})
            return True
        # 其它失败：重试也没用（配置错 / 焦点丢了 / 用户抢了）。
        # 立刻中断并说清原因 —— 比干等 3 秒超时再报一句
        # 「切到 X 等了 3000ms 还没生效」有用得多。
        seq = self._seq
        self._abort_locked(
            f"切到 {self._switch_target_name or '?'} 失败："
            f"{reason or error or '未知原因'}")
        return True

    def _enter_until_locked(self, seq: ComboSequence, step: ComboStep,
                            now: int) -> None:
        """进 ``until`` 观察窗。

        ★ 为什么 ``until`` **不是**在 :meth:`notify_done` 里就地求值
          那一刻上一步刚按完，上下文里的 ``e_ready()`` 还是**按之前**的
          值（技能要等下一帧的感知才看得出来进冷却）。就地求值等于
          「刚按完 E 就问 e_ready()」—— 永远是 True，后置条件形同虚设。

          所以它必须**占住一个相位**，由之后每一 tick 用**新鲜上下文**
          重估。这就是感知类后置条件唯一诚实的实现方式。
        """
        self._phase = PHASE_UNTIL
        self._phase_started_ms = now
        self._until_deadline_ms = now + DEFAULT_UNTIL_TIMEOUT_MS
        self._until_step = step

    def _check_until_locked(self, seq: ComboSequence, ctx: Any) -> None:
        """每 tick 重估 ``until``。真 -> 推进；到点还不真 -> 响亮跳过。"""
        step = self._until_step
        if step is None or step.until_compiled is None:
            self._apply_jump_locked(step) if step else None
            return
        try:
            ok = self._eval_step(step, ctx, expr=step.until_compiled,
                                 label="until")
        except RuleEvalError as exc:
            self._eval_failed(seq, exc, step=step)
            self._apply_jump_locked(step)
            return
        if ok:
            self._apply_jump_locked(step)
            return
        now = self._now()
        if now < self._until_deadline_ms:
            return                       # 还没到观察窗，继续等
        self._event(Severity.WARN, "COMBO_STEP_NO_EFFECT",
                    f"连招 {seq.name} 的第 {self._step_idx} 步"
                    f"（{step.action.value}）按了但 {DEFAULT_UNTIL_TIMEOUT_MS}ms 内"
                    f"没生效（until: {step.until} 一直不成立）——多半被动画取消了，"
                    f"跳过这一步",
                    {"combo": seq.name, "step": self._step_idx,
                     "action": step.action.value, "until": step.until})
        self._apply_jump_locked(step)

    @property
    def break_priority(self) -> int:
        """命中 priority 小于它的普通规则会中断连招。

        空闲时返回默认值 —— **空闲时这个数没有意义**：连招没在跑，
        「要不要打断」根本不存在。调用方（策略树）会先问
        :attr:`active`，这里的默认值只是为了让「先判规则再问连招」
        这种写法不必先分支。
        """
        with self._lock:
            if self._seq is None:
                return DEFAULT_BREAK_BELOW_PRIORITY
            return self._seq.break_below_priority

    def progress(self) -> dict[str, Any]:
        """当前进度快照。给诊断/Console 看。"""
        with self._lock:
            if self._seq is None:
                return {"active": False, "phase": "-",
                        "last_reason": self._last_reason}
            return {
                "active": True,
                "combo": self._seq.name,
                "step": self._step_idx + 1,
                "steps": len(self._seq.steps),
                "phase": self._phase,
                "repeat_left": self._repeat_left,
                "switch_target": self._switch_target_name or None,
                "elapsed_ms": self._now() - self._started_ms,
            }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "configured": len(self._book),
                "enabled": len(self._book.enabled),
                "source_file": self._book.source_file,
                "started": self._started_count,
                "finished": self._finished_count,
                "aborted": self._aborted_count,
                # 流程控制的观测量。「done_waits 高而 finished 也高」
                # 是正常的；「done_timeouts > 0」说明执行层有问题，
                # 而它**不会**让连招停下来 —— 只是那一步要等满兜底时间。
                "done_waits": self._done_waits,
                "done_timeouts": self._done_timeouts,
                "jumps": self._jumps,
                "reflex_fired": self._reflex_fired,
                "reflex_locking": self._now() < self._reflex_lock_until_ms,
                "last_reason": self._last_reason,
                "progress": self.progress(),
            }

    # -- 中断 ------------------------------------------------------------
    def abort(self, reason: str) -> bool:
        """**立刻**中断正在进行的连招。返回是否真的中断了。

        从哪个线程调都行（物理输入监听 / 急停监视都在别的线程）。
        没在跑时返回 ``False`` —— 幂等，调用方不用先问一遍。

        调用点（少一个就是一个「用户抢了但连招还在打」）：

        * ``core/loop.py::_on_physical_input`` —— 用户动手了；
        * ``core/loop.py::_on_panic`` —— F12 急停；
        * ``local_tree.LocalPolicyTree.poll`` —— 感知不健康 / 应急规则命中。
        """
        with self._lock:
            if self._seq is None:
                return False
            self._abort_locked(reason)
            return True

    def _reset_flow_locked(self) -> None:
        """把流程控制的状态**全部**清回「没在等」。

        ★ 为什么单独一个函数
          ``_await_step`` / ``_until_step`` / ``_phase`` / ``_jumps`` 现在
          有四个字段，任何一个漏清都会留下一个「以为自己还在等」的 runner。
          症状是**下一次执行**莫名其妙停在某一步 —— 而那一步可能
          根本不是它自己触发的进来的。收成一处，漏清就只剩「漏了一处」
          一种可能。
        """
        self._await_step = None
        self._await_started_ms = 0
        self._await_deadline_ms = 0
        self._until_step = None
        self._until_deadline_ms = 0
        self._jumps = 0
        self._phase = PHASE_STEP
        self._settle_until_ms = 0
        self._phase_started_ms = 0

    def _abort_locked(self, reason: str) -> None:
        seq = self._seq
        self._seq = None
        self._step_idx = 0
        self._repeat_left = 0
        self._reset_flow_locked()
        self._switch_target = None
        self._switch_target_name = ""
        self._aborted_count += 1
        self._last_reason = reason
        if seq is not None:
            # 冷却从**中断那一刻**起算，不是从启动起 —— 否则一条
            # 「刚开始就被应急规则打断」的连招会在几帧后立刻重开，
            # 变成「吃药 - 连招 - 吃药 - 连招」的抖动。
            self._cooldown_until_ms = max(
                self._cooldown_until_ms, self._now() + seq.cooldown_ms)
        logger.info("连招中断：%s", reason)
        self._event(Severity.WARN, "COMBO_ABORTED",
                    f"连招已中断：{reason}",
                    {"combo": seq.name if seq else "-", "reason": reason})

    def _finish_locked(self) -> None:
        seq = self._seq
        # ★ 先把名字留下来 —— 下面 self._seq = None 就拿不到了。
        name = seq.name if seq else "-"
        n_steps = len(seq.steps) if seq else 0
        self._seq = None
        self._step_idx = 0
        self._repeat_left = 0
        self._reset_flow_locked()
        self._switch_target = None
        self._switch_target_name = ""
        self._finished_count += 1
        self._last_reason = "走完"
        if seq is not None:
            self._cooldown_until_ms = max(
                self._cooldown_until_ms, self._now() + seq.cooldown_ms)
        logger.info("连招走完：%s", name)
        # ★★ COMBO_FINISHED（2026-10-19 加）
        #   走完一条连招是**值得被外部知道**的时刻：插件侧的
        #   ``DRY_RUN_MUTED_CODES`` 里早就列着这个 code，适配器也
        #   按它准备过 —— 但引擎从来不发它，于是那一格永远空着。
        #
        #   现在补上：演示模式的「可爱嘴替」靠它判断
        #   「连招打完了，该说第一句台词了」。
        #   级别用 INFO —— 走完是**正常**结局，不是问题。
        self._event(Severity.INFO, "COMBO_FINISHED",
                    f"连招走完：{name}",
                    {"combo": name, "steps": n_steps})

    # -- 主循环 ----------------------------------------------------------
    def next_intent(self, ctx: PolicyContext) -> IntentPacket | None:
        """走一步。返回要发的那一个意图，没有就 ``None``。

        **一 tick 最多一个** —— 和规则树同一个契约。连招没有「一次发一串」
        的特权：那样执行层的限频、安全闸、按键释放全都绕过去了。
        """
        with self._lock:
            if self._seq is None:
                seq = self._pick(ctx)
                if seq is None:
                    return None
                self._seq = seq
                self._step_idx = 0
                self._repeat_left = 0
                self._reset_flow_locked()
                self._started_ms = self._now()
                self._started_count += 1
                logger.info("连招启动：%s（%d 步 / %d 个动作）",
                            seq.name, len(seq.steps), seq.total_actions)
                self._event(Severity.INFO, "COMBO_STARTED",
                            f"连招启动：{seq.name}",
                            {"combo": seq.name, "steps": len(seq.steps)})
            return self._advance(ctx)

    def _pick(self, ctx: PolicyContext) -> ComboSequence | None:
        """空闲时挑一条该启动的连招。**声明顺序即优先级**。"""
        now = self._now()
        for seq in self._book:
            if not seq.enabled:
                continue
            if now < self._cooldown_until_ms:
                continue
            if not seq.matches_scene(ctx.state.scene.id):
                continue
            try:
                out = self._eval(seq, ctx)
            except RuleEvalError as exc:
                self._eval_failed(seq, exc)
                continue
            if out.value:
                return seq
        return None

    def _advance(self, ctx: PolicyContext) -> IntentPacket | None:
        seq = self._seq
        if seq is None:                     # 防御：不该发生，但别崩
            return None
        now = self._now()

        # ── 场景掉出去了 ───────────────────────────────────────────
        # 传送 / 开菜单 / 进战斗场景都会改 scene。看不见画面还在按技能，
        # 是最坏的一类错 —— 中断。
        if not seq.matches_scene(ctx.state.scene.id):
            self._abort_locked(
                f"场景变成 {ctx.state.scene.id}，连招限定 {list(seq.scenes)}")
            return None

        # ── 整条连招超 ───────────────────────────────────────────
        if now - self._started_ms > seq.timeout_ms:
            self._abort_locked(
                f"整条连招跑了 {now - self._started_ms}ms 还没走完"
                f"（卡在第 {self._step_idx + 1}/{len(seq.steps)} 步 {self._phase}）")
            return None

        # ── 等切人 ─────────────────────────────────────────────────
        if self._phase == PHASE_WAIT_SWITCH:
            if self._switch_target is not None \
                    and self._current_slot() == self._switch_target:
                self._phase = PHASE_STEP
            else:
                waited = now - self._phase_started_ms
                if waited >= seq.switch_timeout_ms:
                    self._abort_locked(
                        f"切到 {self._switch_target_name or '?'} 等了 {waited}ms "
                        f"还没生效（名册里查不到这个人，或切人记账器没更新）")
                    return None
                return None          # 还没好，下一 tick 再问

        # ── 等 settle ──────────────────────────────────────────────
        if self._phase == PHASE_SETTLE:
            if now < self._settle_until_ms:
                return None
            self._phase = PHASE_STEP

        # ── 等执行完成（wait = "ready"）────────────────────────────
        #   ★ 必须排在出队**之前**，且在 WAIT_SWITCH 之后 ——
        #     切人和按键是两种等待，各有各的信号，混在一起就会出现
        #     「切人时等按键完成、按键时等切人记账」的交叉等待。
        if self._phase == PHASE_STEP and self._await_step is not None:
            waited = now - self._await_started_ms
            if waited < self._await_deadline_ms:
                return None                  # 还没回报，下一 tick 再问
            # 执行层没回报 = 它卡了/抛了/忘了。到点必须放行，
            # 否则连招会**永久停在这一步** —— 那比打得不好更糟。
            self._done_timeouts += 1
            step = self._await_step
            self._event(Severity.WARN, "COMBO_DONE_TIMEOUT",
                        f"连招 {seq.name} 的第 {self._step_idx} 步"
                        f"（{step.action.value}）等执行完成等了 {waited}ms "
                        f"还没回报 —— 执行层可能卡住了，放行继续",
                        {"combo": seq.name, "step": self._step_idx,
                         "action": step.action.value, "waited_ms": waited})
            self._await_step = None
            if step.until_compiled is not None:
                self._enter_until_locked(seq, step, now)
            else:
                self._apply_jump_locked(step)

        # ── until 观察窗 ────────────────────────────────────────────
        if self._phase == PHASE_UNTIL:
            self._check_until_locked(seq, ctx)
            if self._phase == PHASE_UNTIL:
                return None                  # 还在观察窗里

        # ── 出队 ───────────────────────────────────────────────────
        # 循环上限 = 步数 + 2：每轮要么发一个意图（返回），要么跳过一步
        # （步数有限）。上限只是防「代码写错导致死循环」的最后一道，
        # 正常路径永远到不了。
        for _ in range(len(seq.steps) + 2):
            if self._step_idx >= len(seq.steps):
                self._finish_locked()
                return None

            step = seq.steps[self._step_idx]

            # ── ① 显式切人步 ─────────────────────────────────────
            #   守卫仍然**先**判：它门控的是「要不要切」，
            #   跟动作步的守卫（门控「要不要做这个动作」）不是一回事。
            if step.is_switch:
                if step.compiled is not None:
                    try:
                        ok = self._eval_step(step, ctx)
                    except RuleEvalError as exc:
                        self._eval_failed(seq, exc, step=step)
                        self._step_idx += 1
                        continue
                    if not ok:
                        self._step_idx += 1
                        continue
                # ★ 先查得到槽位再发。查不到就跳过这一步，**不发**那个
                #   执行层一定会跳过的意图 —— 否则会白占一个 tick 的决策，
                #   并且在日志里留下一条「已执行」的假记录。
                target = self._slot_of(step.char)
                if target is None:
                    self._warn_unknown(seq, step, None, self._current_slot())
                    self._step_idx += 1
                    continue
                intent = self._make_intent(step.action, step.params(),
                                           seq, ctx)
                self._enter_wait_switch(step.char, target)
                self._step_idx += 1
                self._repeat_left = 0
                return intent

            # ── ② 隐式切人：这一步指定了谁，但那个人现在不在场 ──
            # ★★★ 三种写法在这里汇合：
            #     slot = 2      -> 固定第 2 号位（名字运行时从 OCR 名册反查）
            #     onfield=true  -> 谁在场上就动谁，**永不切人**
            #     char = "..."  -> 老写法，靠 _slot_of 查名字
            #    统一的目标是「算出 target / cur 两个槽位」。
            target: int | None = None
            need_switch = False
            if step.onfield:
                # ★ onfield **不需要切人**，所以直接落到 ③。
                #   这正是「通用战斗」成立的原因：它对队伍是谁零假设。
                pass
            elif step.slot:
                target = step.slot
                name = self._name_of(target)
                if name is None:
                    # 名册还没填满（~20 秒）或这个槽位真的没人。
                    self._warn_unknown(seq, step, None, self._current_slot())
                    self._step_idx += 1
                    continue
                cur = self._current_slot()
                if cur != target:
                    need_switch = True
            elif step.char:
                target = self._slot_of(step.char)
                cur = self._current_slot()
                if target is None:
                    # ★ 不知道**目标**是谁 -> 跳过，**不猜**。
                    #   猜错的后果是「夜兰的大招打在钟离身上」，
                    #   比不做更糟（不做只是这一轮没输出）。
                    self._warn_unknown(seq, step, target, cur)
                    self._step_idx += 1
                    continue
                # ★★ ``cur is None``（不知道现在是谁在场）**也要切**（2026-10-19）。
                #   以前这里把「不知道在场是谁」和「不知道目标是谁」一起
                #   当成「跳过」—— 于是**删掉显式切人步之后**，连招第一步
                #   会因为「不知道现在是谁在场」被整个跳过：
                #   钟离的盾根本放不出来。
                #
                #   ★ 目标已知时切人**不靠猜**：按目标号的数字键，
                #     切完在场的一定是目标角色（显式切人步也是这么做的，
                #     它压根不需要知道 cur）。
                if cur != target:
                    need_switch = True

            if need_switch:
                # ★ 切人意图：名字写法给 ``char_id``，位置写法给 ``slot``。
                #   执行层两个都认（``executor.py`` 的 SWITCH_CHARACTER 分支
                #   先看 slot 再看 char_id），所以位置式步骤不必等
                #   OCR 名册填满名字。
                switch_name = step.char or (self._name_of(target or 0) or "")
                sw = ComboStep(action=ActionType.SWITCH_CHARACTER,
                               char=switch_name)
                params = sw.params()
                if not step.char and target is not None:
                    params["slot"] = int(target)
                intent = self._make_intent(
                    ActionType.SWITCH_CHARACTER, params, seq, ctx)
                self._enter_wait_switch(switch_name, target)
                return intent

            # ── ③ 守卫 ───────────────────────────────────────────
            # ★★★ **顺序是「先切人、后判守卫」**（2026-10-19 改）
            #
            #   为什么不能先判守卫
            #     ``when = "e_ready()"`` 问的是**在场角色**的 E 好了没
            #     （``context._active_cd`` 读 ``state.player.active_member``）。
            #     而 ``cd_ready`` **只挂在在场角色身上** —— 它读的是
            #     **右下角 E/Q 图标**，那显示的是当前在场角色的技能栏。
            #     目标角色**还没切上场**时，他的 CD **根本读不到**
            #     （UNKNOWN -> 判为假）。
            #
            #     所以先判守卫的后果是：钟离的 E 刚进冷却 ->
            #     「纳西妲那一步」的守卫读的是**钟离**的 CD -> 为假 ->
            #     纳西妲那一步被**整个跳过** —— 技能**根本没放出来**，
            #     而日志里一条线索都没有。
            #
            #   ★ 切过去之后再判，守卫读到的就是**对的人**的 CD。
            #     这也正是人类的操作顺序：先切过去，再看 E 亮不亮。
            #
            #   ★ 代价（诚实说）：守卫为假时我们**已经切过去了**。
            #     但那时我们**不可能**提前知道 —— 目标角色的 CD
            #     在切过去之前读不到。真人也是这么做的：
            #     切过去发现 E 是灰的，再切走。
            if step.compiled is not None:
                try:
                    ok = self._eval_step(step, ctx)
                except RuleEvalError as exc:
                    self._eval_failed(seq, exc, step=step)
                    self._step_idx += 1
                    continue
                if not ok:
                    self._step_idx += 1
                    continue

            # ④ 正常出队
            if self._repeat_left == 0:
                self._repeat_left = step.repeat
            intent = self._make_intent(step.action, step.params(), seq, ctx)
            # ★ 出队时就先把游标挪到「下一步」—— 完成门只是决定
            #   **能不能**往下走。所以挂门之后**绝不能再动 _step_idx**：
            #   动了的话这一步就永远等不到 notify_done，
            #   症状是「配了 wait=ready 之后连招彻底不动了」。
            post = (self._repeat_left <= 1)
            self._repeat_left -= 1
            if post:
                self._step_idx += 1
                self._repeat_left = 0
            if step.settle_ms:
                self._phase = PHASE_SETTLE
                self._phase_started_ms = now
                self._settle_until_ms = now + step.settle_ms
            if step.needs_done:
                # 只有「最后那一次重复」才挂门 —— 中间的重复不需要
                # 等完再打下一发（那会把 repeat=5 拖成 5 倍时长）。
                if post:
                    self._await_step = step
                    self._await_started_ms = now
                    # ★ 期限必须算上这一步**自己**要花的时间。
                    #   长按的步骤（钟离长按 E 1.5 秒）执行层要按满
                    #   hold_ms 才返回，而默认期限是
                    #   DEFAULT_UNTIL_TIMEOUT_MS = 2000ms ——
                    #   1500 按住 + 120 空档 = 1620ms，只剩 380ms 余量。
                    #   慢一点就判成 COMBO_DONE_TIMEOUT：连招以为
                    #   「技能没放出去」，接着往下打 —— 而技能其实按满了。
                    self._await_deadline_ms = (
                        now + min(seq.timeout_ms,
                                  DEFAULT_UNTIL_TIMEOUT_MS + step.settle_ms
                                  + step.expected_exec_ms))
            elif post:
                self._apply_jump_locked(step)
            # ⚠️ 这里**刻意不记** ``CadenceGate``（它在策略树那边）。
            #   连招里的 repeat 是玩家显式写的次数，压它会让序列走不完；
            #   而节奏门是「拟人化分布」，不是安全闸。防连点的活儿由
            #   ``actuator/traits.py`` 的 ``min_gap_ms`` 负责 —— 那是
            #   执行层该管的事，且它对规则和连招一视同仁。
            return intent

        # 到这里说明循环被上限截断了 —— 真的写错了才会发生。
        self._abort_locked("内部错误：出队循环没有收敛")
        return None

    # -- 内部 ------------------------------------------------------------
    # -- 危机反射 -------------------------------------------------------
    def reflex_step(self, ctx: PolicyContext) -> IntentPacket | None:
        """危机反射这一轮要不要出手。**在连招之前问**。

        ★ 为什么它住在 :class:`ComboRunner` 而不是策略树
          反射要做的第一件事是**中止正在进行的连招** —— 而连招的
          状态就在 runner 里。放策略树就得把 abort 传下去，
          多一层就多一个「忘了 abort」的路径。

        ★ 为什么 ``wait`` 期间什么都返回 ``None``
          锁的意义就是「这一下之后谁也别动」。连招也要让路 ——
          闪避刚按下去就接着放技能，等于把无敌帧浪费掉。

        :returns: 要发的意图，或者 ``None``（这一轮不管）。
        """
        with self._lock:
            now = self._now()
            if now < self._reflex_lock_until_ms:
                return None                  # 静默锁：无敌帧还没过去
            for rule in self._book.enabled_reflexes:
                ready_at = self._reflex_cooldown.get(rule.name, 0)
                if now < ready_at:
                    continue
                if rule.scenes and ctx.state.scene.id not in rule.scenes:
                    continue
                if rule.compiled is None:
                    continue
                try:
                    hit = bool(self._evaluator.evaluate(
                        rule.compiled, ctx,
                        rule_name=f"reflex[{rule.name}]",
                        source_file="", line_no=rule.line_no))
                except RuleEvalError as exc:
                    self._event(Severity.ERROR, "REFLEX_EVAL_ERROR",
                                f"反射 {rule.name} 的 when 求值失败："
                                f"{exc.message}",
                                {"reflex": rule.name,
                                 "message": exc.message})
                    # ★ 出错的反射要**冷却**，不能每 tick 炸一次。
                    self._reflex_cooldown[rule.name] = now + rule.cooldown_ms
                    continue
                if not hit:
                    continue

                # ---- 命中 ----
                params = rule.params()
                target = 0
                if rule.onfield:
                    target = self._current_slot() or 0
                elif rule.slot:
                    target = rule.slot
                if target and not self._reflex_ready(target):
                    self._reflex_cooldown[rule.name] = now + rule.cooldown_ms
                    self._event(Severity.INFO, "REFLEX_NO_TARGET",
                                f"反射 {rule.name} 要用 {target} 号位，"
                                f"但现在站在 {self._current_slot()} —— "
                                f"反射不切人（切过去的那几帧就够挨打了），"
                                f"这次跳过",
                                {"reflex": rule.name, "slot": target,
                                 "onfield": self._current_slot()})
                    return None

                self._reflex_cooldown[rule.name] = now + rule.cooldown_ms
                self._reflex_lock_until_ms = now + rule.lock_ms
                self._reflex_fired += 1
                interrupted = self.active
                if interrupted:
                    # ★ 中止连招是**顺带**的：连招没在跑时也要能反射。
                    #   只在真的要中止时才调，避免把「空闲」也记成
                    #   COMBO_ABORTED —— 那会让「连招崩了」的告警
                    #   在纯反射模式下一直响。
                    self._abort_locked(f"危机反射 {rule.name} 抢在前面")
                intent = self._make_reflex_intent(rule, ctx)
                self._event(Severity.WARN, "REFLEX_FIRED",
                            f"危机反射 {rule.name}：{rule.describe()}"
                            + ("（打断了连招）" if interrupted else ""),
                            {"reflex": rule.name,
                             "action": rule.action.value,
                             "interrupted": interrupted})
                return intent
        return None

    def _reflex_ready(self, slot: int) -> bool:
        """反射要用 ``slot`` 时，**此刻就站在那儿吗**。

        ★ 为什么反射**不做隐式切人** —— 这是个真实的取舍
          反射的定义是「立刻解决眼前这一下」。切过去再闪：
          等切人记账的那几帧里，人已经被打了；而且一 tick 只能发
          一个意图，那一帧就浪费在切人上了。

          ★ 曾经的实现是「先切一步、下一帧再发技能」，那是**错的**：
            切人那一帧之后，这条反射已经进了冷却，而**真正要发的
            闪避会跟着一起丢掉** —— 于是配置写了「用 3 号位大招躲」，
            实际表现是「永远只切人，不放技能」，而且**一条错误都没有**。
            （症状像「反射不生效」，真因是一次机会被换成了两步。）

          所以这里只认「已经在那一格」。不在就**放弃这次反射**：
          一次没触发的反射只浪费一次机会，而一次「切过去再闪」
          把机会换成了伤害。
        """
        return self._current_slot() == slot

    def _make_reflex_intent(self, rule: ReflexRule,
                            ctx: PolicyContext) -> IntentPacket:
        return IntentPacket.make(
            rule.action,
            params=rule.params(),
            state_version=ctx.state.state_version,
            ttl_seconds=1,
            priority=REFLEX_INTENT_PRIORITY,
            source="LOCAL",
            scene_required=list(rule.scenes) if rule.scenes else None,
            constraints={"max_attempts": 1},
        )

    def _apply_jump_locked(self, step: ComboStep) -> None:
        """一步做完之后：清掉完成门，并按 ``jump_to`` 决定去哪。

        ★ 为什么它是**唯一**决定「下一步去哪」的地方
          ``jump_to`` 要和「出队时已经把游标 +1 了」抢同一个决定权。
          分开放就会出现「跳转被 +1 顶掉了」这种交叉，而症状是
          「配了循环但只打一次」—— 极难从动作序列里看出来。

        ★ 为什么这里**不负责 +1**
          出队时（``post`` 分支）游标就已经指向下一步了。本方法只
          回答一个问题：「要跳回去吗」。所以多调一次不会有副作用 ——
          而这条性质是「超时兜底也能安全地调它」的前提。
        """
        self._await_step = None
        self._until_step = None
        self._until_deadline_ms = 0
        if self._phase in (PHASE_UNTIL, PHASE_WAIT_SWITCH):
            self._phase = PHASE_STEP
        if not step.jump_to:
            return
        seq = self._seq
        if seq is None:
            return
        self._jumps += 1
        if self._jumps > MAX_JUMPS:
            self._abort_locked(
                f"一次执行里跳了 {self._jumps} 次（上限 {MAX_JUMPS}）——"
                f"jump_to 写成了环")
            return
        nxt = step.jump_to - 1
        if not 0 <= nxt < len(seq.steps):
            self._abort_locked(
                f"第 {self._step_idx} 步的 jump_to={step.jump_to} 越界"
                f"（共 {len(seq.steps)} 步）")
            return
        self._step_idx = nxt
        self._repeat_left = 0

    def _enter_wait_switch(self, char: str, slot: int | None = None) -> None:
        self._phase = PHASE_WAIT_SWITCH
        self._phase_started_ms = self._now()
        self._switch_target = slot if slot is not None else self._slot_of(char)
        self._switch_target_name = char

    def _name_of(self, slot: int) -> str | None:
        """槽位 -> 角色名。**``slot`` 式配置能落地全靠它。**

        ★ 为什么需要它
          执行层的 ``SWITCH_CHARACTER`` 收的是**名字**
          （``params.char_id``），它自己再用 ``_slot_of`` 翻成数字键。
          所以「按第 2 号位」这件事，链路上必须有一个时刻是拿着
          **名字**的 —— 而那个名字只能来自 OCR 名册。

        :returns: 查不到返回 ``None`` —— 名册启动后约 20 秒才填满，
            那之前 slot 式连招会跳过并发 ``COMBO_STEP_UNKNOWN``
            （**响亮跳过，不是静默不动**）。
        """
        key = int(slot or 0)
        if not 1 <= key <= 4:
            return None
        # ★ 静态配置优先，与 _slot_of 的顺序**完全一致** ——
        #   两边反过来会出现「规则切对了人、连招切错人」。
        #   （keymap 的 [char_slots] 是「名字 -> 槽位」，这里要反着查。）
        for nm, sl in (self._static or {}).items():
            try:
                if int(sl) == key:
                    return str(nm)
            except (TypeError, ValueError):
                continue
        # 再问 OCR 名册。``name_of`` 是**方法**不是 property —— 写成
        # ``getattr(...)`` 再当属性用会拿到 bound method，
        # 那一步的 TypeError 会被下面的 except 吃掉，变成「永远查不到」。
        fn = getattr(self._index, "name_of", None)
        if fn is None:
            return None
        try:
            v = fn(key)
        except Exception:  # noqa: BLE001
            logger.debug("name_of(%d) 失败", key, exc_info=True)
            return None
        return str(v) if v else None

    def _slot_of(self, char: str) -> int | None:
        """角色名 -> 槽位。两级查找，**顺序与执行层完全一致**：

        1. ``keymap.toml`` 的 ``[char_slots]``（用户显式配的，以它为准）；
        2. 大世界 OCR 建出来的名册（免配置那条路）。

        ⚠️ 静态优先这条不能反过来。反过来的话，玩家手填
          ``[char_slots]`` 纠正 OCR 的错名字时，连招会继续用 OCR 的旧值 ——
          于是**规则切对了人、连招切错了人**，两套机制各走各的。
          同一个错名字在两边表现不一致，是最难查的一类问题。
        都没接 / 都没有 -> ``None``（那一步会被跳过，不猜）。
        """
        key = (char or "").strip()
        if not key:
            return None
        if self._static:
            v = self._static.get(key)
            if v is not None:
                return int(v)
        fn = getattr(self._index, "slot_of", None)
        if fn is None:
            return None
        try:
            v = fn(key)
        except Exception:  # noqa: BLE001 - 名册坏了不该崩连招
            logger.debug("查槽位失败（%r）", char, exc_info=True)
            return None
        return None if v is None else int(v)

    def _current_slot(self) -> int | None:
        fn = getattr(self._active, "current", None)
        if fn is None:
            return None
        try:
            v = fn()
        except Exception:  # noqa: BLE001 - 同上
            logger.debug("读在场槽位失败", exc_info=True)
            return None
        return None if v is None else int(v)

    def _eval(self, seq: ComboSequence, ctx: PolicyContext):
        """求 trigger。**用的是这一帧的 ctx** —— 触发器要看的是
        「这一帧世界什么样」，不是「某一次启动时什么样」。"""
        return self._evaluator.evaluate(
            seq.compiled, ctx,
            rule_name=seq.name, source_file=seq.source_file,
            line_no=seq.line_no)

    def _eval_step(self, step: ComboStep, ctx: PolicyContext,
                   *, expr: CompiledExpr | None = None,
                   label: str = "when") -> bool:
        """求值一步的守卫。

        :param expr: 求哪个表达式。默认 ``when``；``until`` 传
            :attr:`ComboStep.until_compiled`。**必须传而不能就地取** ——
            两者语义不同（一个是出队前的准入，一个是执行后的验收），
            混用会让「E 亮了才按 E」变成「按完 E 再判断 E 亮不亮」。
        :param label: 报错时指明是哪个条件，别让用户以为 ``when`` 坏了。
        """
        target = step.compiled if expr is None else expr
        assert target is not None
        return self._evaluator.evaluate(
            target, ctx,
            rule_name=f"{step.action.value}({step.char or '-'}).{label}",
            source_file="", line_no=step.line_no)

    def _make_intent(
        self, action: ActionType, params: dict[str, Any],
        seq: ComboSequence, ctx: PolicyContext,
    ) -> IntentPacket:
        # ★ 和规则用同一条构造路径（``IntentPacket.make``）——
        #   于是 scene_required / ttl / constraints 全部一致，
        #   执行层的 validator + safety 闸门**不需要认识连招**。
        return IntentPacket.make(
            action,
            params=params,
            state_version=ctx.state.state_version,
            ttl_seconds=seq.ttl_seconds,
            priority=COMBO_INTENT_PRIORITY,
            scene_required=seq.scenes,
            source="LOCAL",
            constraints={"max_attempts": 1},
        )

    def _warn_unknown(
        self, seq: ComboSequence, step: ComboStep,
        target: int | None, cur: int | None,
    ) -> None:
        now = self._now()
        if now - self._last_warn_ms < WARN_THROTTLE_MS:
            return
        self._last_warn_ms = now
        if target is None:
            # ★ 位置式步骤的 step.char 是空的。直接套用老文案会打印
            #   「名册里还没有 ''」—— 那等于没告诉玩家发生了什么。
            if step.slot:
                why = (f"名册里还没有 {step.slot} 号位是谁 —— "
                       f"大世界 OCR 一圈要 ~20 秒，或这个格子确实没人。"
                       f"\n  手填 keymap.toml 的 [char_slots] 可以立刻解决；"
                       f"不想等的话改用 onfield = true（谁在场上就动谁）")
            else:
                why = (f"名册里还没有 {step.char!r} —— "
                       f"大世界 OCR 一圈要 ~20 秒，"
                       f"或该角色的名字根本 OCR 不出来。手填 keymap.toml 的 "
                       f"[char_slots] 可以立刻解决")
        else:
            why = ("在场记账器说不知道现在站着谁（还没切过人，或刚启动）—— "
                   "切人记账靠的是「引擎/用户按了哪个数字键」")
        self._event(
            Severity.WARN, "COMBO_STEP_UNKNOWN",
            f"连招 {seq.name} 跳过第 {self._step_idx + 1} 步"
            f"（{step.describe().split(' -> ')[0]} / {step.action.value}）：{why}",
            {"combo": seq.name, "step": self._step_idx + 1,
             "char": step.char, "action": step.action.value,
             "slot": target, "onfield": cur})

    def _eval_failed(self, seq: ComboSequence, exc: RuleEvalError,
                     step: ComboStep | None = None) -> None:
        self._last_warn_ms = self._now()
        where = (f"第 {self._step_idx + 1} 步的 when" if step is not None
                 else "trigger")
        self._event(Severity.ERROR, "COMBO_EVAL_ERROR",
                    f"连招 {seq.name} 的 {where} 求值失败：{exc.message}",
                    {"combo": seq.name, "message": exc.message})
        logger.error("连招 %s 的 %s 求值失败：%s", seq.name, where, exc)

    def _event(self, sev: Severity, code: str, msg: str,
               ctx: dict[str, Any]) -> None:
        ev = Event(severity=sev, category=EventCategory.DECISION,
                   code=code, message=msg, context=ctx)
        if self._emit_fn is not None:
            try:
                self._emit_fn(ev)
            except Exception:  # noqa: BLE001 - 事件总线坏了不该拖垮连招
                logger.debug("连招事件投递失败", exc_info=True)
        elif sev is Severity.ERROR:
            logger.error("[%s] %s", code, msg)


#: 模块级求值器。:class:`ComboRunner` 默认用它 ——
#: DSL 编译器无状态，而它的编译缓存放在全局能让
#: 「同一表达式只编一次」在规则和连招之间也成立。
_DEFAULT_EVALUATOR = DslEvaluator()


__all__ = [
    "COMBO_INTENT_PRIORITY",
    "DEFAULT_BREAK_BELOW_PRIORITY",
    "DEFAULT_COMBOS_PATH",
    "DEFAULT_UNTIL_TIMEOUT_MS",
    "MAX_JUMPS",
    "PHASE_UNTIL",
    "REFLEX_INTENT_PRIORITY",
    "REFLEX_FIELDS",
    "ComboBook",
    "ComboLoadError",
    "ComboRunner",
    "ComboSequence",
    "ComboStep",
    "ReflexRule",
    "load_combos",
    "load_combos_from_files",
    "load_optional_combos",
]
