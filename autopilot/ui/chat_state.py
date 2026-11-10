"""场景状态机（Companion Mode，2026-10-29）

★ 它解决什么
-----------
播报层此前**只有**「这一类值不值得播」（verbosity 档位，
见 ``event_to_yui.CODE_TIER``），**没有**「现在是不是能说话」。
档位管的是「这类话要不要听」，状态机管的是「**此刻**能不能开口」。

两者的区别就是用户说的「不能像复读机，也不能像指导教练」：

* 档位 1.0 时 ``ENEMY_DETECTED`` 照播 —— 但打怪的时候它不该播。
* 战斗时 YUI 闭嘴（只留 tier1 关键异常），大世界才闲聊，挂机才主动找话题。

★ 输入源（**只有三个**，不新增任何事件）
------------------------------------
1. ``SCENE_CHANGED``    —— 场景 id 变了（``context["to"]``）
2. ``POSITION_CHANGED`` —— 坐标变了（= 玩家在动）
3. ``PERCEPTION_DOWN``  —— 看不见了

★ 为什么 ``CAPTURE_FAILED`` 也算第 3 类
-------------------------------------
``_PROMPT_GUIDE.md`` §2.1 把 UNKNOWN 的判定来源写成
「``PERCEPTION_DOWN`` / ``CAPTURE_FAILED`` / 场景读不到」。
两者是**同一族**（感知/抓屏失败），把它们分成两个输入源只会让
状态机多一个分支、少一份一致性，所以这里按同义词处理。

★ 红线
----
* **纯逻辑**：零 IO、零第三方依赖（只用标准库），可单测。
* **不 import** ``event_to_yui`` —— 反过来依赖会成环
  （播报层要 import 本模块）。
* 模块顶层**不做任何 IO**（对齐插件 R8 红线）。

★ 初值为什么是 ``WORLD`` 而不是 ``UNKNOWN``
--------------------------------------------
「**还没开始看**」不等于「**看不见**」。

初值取 ``UNKNOWN`` 的话，插件刚起来、还没收到第一个 ``SCENE_CHANGED``
的那段时间里，tier2 / tier3 会被整体静音 —— 而真机上前若干秒**恰好**
是引擎刚跑起来、事件最密的时候。那等于「加了陪玩功能」变成了
「开场先哑三十秒」。

而且 ``_PROMPT_GUIDE.md`` §9 第 9 条钉着「默认 1.0 时行为与改造前
**逐字一致**」：初值取 ``UNKNOWN`` 会让「没收到过场景信息」这件事
变成一次静默的行为变更。

所以初值取 ``WORLD``（乐观），而 ``UNKNOWN`` **只在确知看不见时**进入：

* 收到 ``PERCEPTION_DOWN`` / ``CAPTURE_FAILED``
* 场景是 ``LOADING`` / ``CUTSCENE`` / ``UNKNOWN``（或不认识�� id）

★ 挂机判定为什么「安全侧优先」
------------------------------
挂机 = 「连续 180s 没有坐标变化」。但坐标事件本身是**引擎至今没发过**的
（见 ``autopilot/core/loop.py`` 里 ``POSITION_CHANGED`` 那段注释：
quality 判据写成了永远不成立的 ``"TRACKING"``）。

所以判据分成两档：

* **见过坐标事件** -> 基线是「最后一次坐标变化」，标准的 180s 无位移。
* **一个坐标事件都没见过** -> 基线退化成「进入大世界场景的时刻」。
  这是**唯一**能让挂机话术在真机上生效的办法，否则那个功能是死代码。

两档都要求「**确实在大世界**」（``_base == WORLD``），
战斗中和看不见时永远不会误判成挂机。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

# ---------------------------------------------------------------- 状态

#: 战斗 —— YUI 闭嘴，只留 tier1 关键异常
STATE_COMBAT = "COMBAT"

#: 大世界 —— 可闲聊、可吐槽
STATE_WORLD = "WORLD"

#: 挂机 —— 主动找话题
STATE_IDLE = "IDLE"

#: 未知（看不见 / 加载 / 过场）—— 只播 tier1
STATE_UNKNOWN = "UNKNOWN"

#: 全部状态。顺序即「从吵到静」。
ALL_STATES: tuple[str, ...] = (
    STATE_COMBAT, STATE_WORLD, STATE_IDLE, STATE_UNKNOWN,
)

# ---------------------------------------------------------------- 阈值

#: 挂机判定：连续这么久没有坐标变化（秒）
IDLE_SECONDS: float = 180.0

#: 挂机后主动开口的最小间隔（秒）—— 距**上次开口**。
#:
#: ★ 为什么是「距上次开口」而不是「距上次主动开口」
#:   主人不在的时候 YUI 说什么都没用，但只要 YUI 刚说过话，
#:   就不该立刻再冒一句 —— 那是「找存在感」过头了。
IDLE_PROACTIVE_MIN_GAP_S: float = 600.0

# ---------------------------------------------------------------- 场景

#: 战斗场景。``COMBAT_RESULT``（战斗结算）也算战斗 ——
#: 结算画面里开口还是在打怪。
COMBAT_SCENES: frozenset[str] = frozenset({
    "COMBAT", "COMBAT_RESULT",
})

#: 大世界场景。菜单类（背包 / 地图 / 通知…）也在内 ——
#: 玩家在菜单里闲逛时，YUI 陪着聊两句是合适的。
WORLD_SCENES: frozenset[str] = frozenset({
    "WORLD", "MAIN_MENU", "DIALOG", "BAG_INVENTORY", "MAP_BIG",
    "GAME_MENU", "NOTICE", "TELEPORT_MAP",
})

#: 「看不见」的输入源。两者是同一族（感知 / 抓屏失败）。
PERCEPTION_DOWN_CODES: frozenset[str] = frozenset({
    "PERCEPTION_DOWN", "CAPTURE_FAILED",
})

# ---------------------------------------------------------------- 场景组

#: 场景组。四组，覆盖全部 13 个真实场景 id。
#:
#: ★ 场景组 ≠ 状态
#:   状态决定「现在能不能说话」（四态），场景组决定「**多久**能再说一句」
#:   （同场景防刷屏）。两套划分故意不一样：
#:     · ``LOADING`` / ``CUTSCENE`` 在状态上是 UNKNOWN（闭嘴），
#:       在组上却要跟 ``UNKNOWN`` 归到一起，否则加载画面切换一次
#:       就会开一个新窗口，等于没有冷却。
#:     · ``MENU`` 组里的 7 个场景在状态上都是 WORLD（能说话），
#:       但它们之间**共享**一把 90s 锁 —— 玩家在背包和地图之间来回点，
#:       不该被播成「换场景了」「又换场景了」。
GROUP_WORLD = "WORLD_GROUP"
GROUP_COMBAT = "COMBAT_GROUP"
GROUP_MENU = "MENU_GROUP"
GROUP_TRANSITION = "TRANSITION_GROUP"

#: 13 个真实场景 id -> 场景组。
#:
#: 键来自 ``autopilot/contracts/state_packet.py::SCENE_IDS``（去掉通配 ``*``）。
#: **13 个全都在这张表里**，一个不漏 —— 漏一个会让那个场景的台词
#: 永远不受组冷却约束（退化成老行为，且没人会发现）。
SCENE_GROUP: dict[str, str] = {
    "WORLD": GROUP_WORLD,
    "COMBAT": GROUP_COMBAT,
    "COMBAT_RESULT": GROUP_COMBAT,
    "MAIN_MENU": GROUP_MENU,
    "DIALOG": GROUP_MENU,
    "BAG_INVENTORY": GROUP_MENU,
    "MAP_BIG": GROUP_MENU,
    "GAME_MENU": GROUP_MENU,
    "NOTICE": GROUP_MENU,
    "TELEPORT_MAP": GROUP_MENU,
    "LOADING": GROUP_TRANSITION,
    "CUTSCENE": GROUP_TRANSITION,
    "UNKNOWN": GROUP_TRANSITION,
}


def state_for_scene(scene_id: str) -> str:
    """场景 id -> 状态。

    * ``COMBAT`` / ``COMBAT_RESULT`` -> :data:`STATE_COMBAT`
    * ``WORLD`` 与 7 个菜单类 -> :data:`STATE_WORLD`
    * 其它（``LOADING`` / ``CUTSCENE`` / ``UNKNOWN`` / 不认识的）
      -> :data:`STATE_UNKNOWN`

    ★ 不认识的场景 id 一律按 UNKNOWN
      「没见过的场景」等于「不知道自己在哪」，那就不该乱说话。
      这与 :func:`group_for_scene` 的兜底方向**相反**（组是防刷屏的，
      兜底到 TRANSITION 最保守；状态是管开口的，兜底到 UNKNOWN 最保守）。
    """
    scene = (scene_id or "").strip().upper()
    if scene in COMBAT_SCENES:
        return STATE_COMBAT
    if scene in WORLD_SCENES:
        return STATE_WORLD
    return STATE_UNKNOWN


def group_for_scene(scene_id: str) -> str:
    """场景 id -> 场景组。**不认识的一律落到 TRANSITION 组**。"""
    scene = (scene_id or "").strip().upper()
    return SCENE_GROUP.get(scene, GROUP_TRANSITION)


# ---------------------------------------------------------------- 状态机


class ChatState:
    """四态场景状态机（``COMBAT`` / ``WORLD`` / ``IDLE`` / ``UNKNOWN``）。

    :param idle_seconds: 挂机判定阈值（秒）。默认 :data:`IDLE_SECONDS`。
    :param clock: 时钟。默认 :func:`time.monotonic`；
        单测注入假时钟即可精确地推进 180s / 600s，不用真的 sleep。

    ★ 用法
      >>> st = ChatState()
      >>> st.observe("SCENE_CHANGED", {"to": "COMBAT"})
      'COMBAT'
      >>> st.observe("SCENE_CHANGED", {"to": "WORLD"})
      'WORLD'
    """

    def __init__(
        self,
        *,
        idle_seconds: float = IDLE_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._idle_seconds = max(0.0, float(idle_seconds))
        # ★ 默认时钟**延迟绑定**（调用时才去取 ``time.monotonic``）
        #   直接写 ``clock or time.monotonic`` 会把函数对象**冻在构造那一刻**，
        #   之后单测再 monkeypatch ``time.monotonic`` 就对本实例无效了
        #   （``test_dry_run_and_speech_noise.py`` 正是这么控时间的）。
        self._clock: Callable[[], float] = clock or (lambda: time.monotonic())
        #: 基础状态。**初值 WORLD** —— 见模块文档「初值为什么是 WORLD」。
        self._base: str = STATE_WORLD
        #: 最近一次看到的场景 id。初值与 ``_base`` 一致（乐观）。
        self._scene: str = "WORLD"
        #: 进入大世界的时刻。
        #:
        #: ★ 初值就是「现在」
        #:   不是 ``None``。引擎的**第一条** ``SCENE_CHANGED`` 会被吞掉
        #:   （``autopilot/core/loop.py`` 里「首次赋值被吞」那段），
        #:   而 ``POSITION_CHANGED`` 至今没被引擎真正发出来 ——
        #:   两个基线都没建立的话，挂机话术永远等不到 180s 那一刻，
        #:   整个挂机功能就是死代码。
        #:   「从现在起 180s 没动过」正是挂机的本意。
        self._world_since: float | None = self._now()
        #: 最后一次坐标变化的时刻（``None`` = 从来没见过坐标事件）
        self._last_move_at: float | None = None
        #: 最后一次**开口**的时刻（``None`` = 从来没开口过）
        self._last_spoke_at: float | None = None

    # ---------------------------------------------------------------- 读
    @property
    def base_state(self) -> str:
        """不含挂机叠加的基础状态（``COMBAT`` / ``WORLD`` / ``UNKNOWN``）。"""
        return self._base

    @property
    def scene(self) -> str:
        """最近一次看到的场景 id。"""
        return self._scene

    @property
    def group(self) -> str:
        """当前场景组（见 :data:`SCENE_GROUP`）。"""
        return group_for_scene(self._scene)

    @property
    def state(self) -> str:
        """当前状态（四态之一）。

        ★ 这是**算出来的**，不是一个存起来的字段
          挂机是一个**时间条件**，时间在走，状态就跟着变 ——
          哪怕一条新事件都没来。把它存成字段就得靠某个事件来刷新，
          而「180 秒没人动」这件事本身**就是**「没有事件」。
        """
        if self._base == STATE_COMBAT:
            return STATE_COMBAT
        if self._base == STATE_UNKNOWN:
            return STATE_UNKNOWN
        if self._idle_elapsed(self._now()) >= self._idle_seconds:
            return STATE_IDLE
        return STATE_WORLD

    @property
    def idle_seconds(self) -> float:
        """挂机判定阈值（秒）。"""
        return self._idle_seconds

    def idle_elapsed(self) -> float:
        """已经「没有坐标变化」多久了（秒）。"""
        return self._idle_elapsed(self._now())

    def now(self) -> float:
        """当前时钟读数。

        ★ 为什么要暴露它
          播报层的「挂机话术重试间隔」必须和状态判定**用同一个时钟** ——
          否则单测注入假时钟时，重试间隔会拿真实时间去比，
          结果是「明明推进了 600s 却仍然被本地间隔挡住」。
        """
        return self._now()

    @property
    def clock(self) -> Callable[[], float]:
        """本状态机用的时钟（给播报层对齐用）。

        ★ 为什么要暴露它
          播报层的场景组冷却窗口要和状态机**同源** ——
          两边各用各的时钟，单测注入假时钟时就会出现
          「状态机已经推进了 90 秒，冷却窗口却一动没动」。
        """
        return self._clock

    # ---------------------------------------------------------------- 写
    def observe(
        self,
        code: str,
        context: Mapping[str, Any] | None = None,
        *,
        now: float | None = None,
    ) -> str:
        """喂一条事件进来，返回**观察之后**的状态。

        只认三个输入源（见模块文档）：``SCENE_CHANGED`` /
        ``POSITION_CHANGED`` / ``PERCEPTION_DOWN``（含 ``CAPTURE_FAILED``）。
        其它 code 一律忽略 —— 状态机不是事件总线，它只对「场景/位置/感知」有反应。

        ⚠ 调用方**必须在所有抑制之前**调用它
          否则被档位/黑名单压掉的事件就不会更新状态，
          状态机会停在过时的场景上（模块文档里「状态机接入播报」那一条）。
        """
        moment = self._now(now)
        ctx = context if isinstance(context, Mapping) else {}
        name = str(code or "").strip().upper()

        if name == "SCENE_CHANGED":
            scene = str(ctx.get("to") or "").strip().upper()
            self._scene = scene or "UNKNOWN"
            self._set_base(state_for_scene(self._scene), moment)
        elif name == "POSITION_CHANGED":
            # ★ 坐标变了 = 玩家在动 = 挂机计时归零
            self._last_move_at = moment
        elif name in PERCEPTION_DOWN_CODES:
            self._set_base(STATE_UNKNOWN, moment)
        return self.state

    def note_spoke(self, *, now: float | None = None) -> None:
        """记一次「开口」。挂机主动话术的 600s 间隔按这个算。"""
        self._last_spoke_at = self._now(now)

    def can_speak_proactively(self, *, now: float | None = None) -> bool:
        """现在可以**主动**开口吗（挂机找话题）。

        判据（``_PROMPT_GUIDE.md`` §6.5）：
        挂机态 + 距上次开口 ≥ 600s。

        ★ 只有 ``IDLE`` 才行
          大世界里主动找话题 = 没话找话，那是打扰；
          战斗中主动开口 = 喧宾夺主，直接违反用户的第一条诉求。
        """
        moment = self._now(now)
        if self.state != STATE_IDLE:
            return False
        if self._last_spoke_at is None:
            return True
        return (moment - self._last_spoke_at) >= IDLE_PROACTIVE_MIN_GAP_S

    def reset(self) -> None:
        """换 run 时重置场景相关的状态。

        ★ 「上次开口」**不清**
          那是对话维度的状态（主人还在不在），跨 run 依然成立。
          清掉它会让每换一个 run 就重新触发一次挂机话术。
        """
        self._base = STATE_WORLD
        self._scene = "WORLD"
        self._world_since = self._now()
        self._last_move_at = None

    # ---------------------------------------------------------------- 内部
    def _now(self, now: float | None = None) -> float:
        return self._clock() if now is None else float(now)

    def _set_base(self, base: str, now: float) -> None:
        """切换基础状态。

        ★ 只有**状态真的变了**才重置挂机基线
          ``WORLD -> MAIN_MENU`` 状态没变（都是 WORLD），
          不该把挂机计时清零 —— 玩家打开背包不代表他「动过了」。
        """
        if base == self._base:
            # ★ 兜底：已经在大世界里，但基线还没建立（刚构造 / 刚 reset）
            #   这时补上「现在」，否则挂机永远等不满 180s。
            if base == STATE_WORLD and self._world_since is None:
                self._world_since = now
            return
        self._base = base
        if base == STATE_WORLD:
            self._world_since = now
        else:
            self._world_since = None
        # 状态变了，旧的坐标时间戳不再作数（可能是很久以前在别的场景里的）
        self._last_move_at = None

    def _idle_elapsed(self, now: float) -> float:
        """「没有坐标变化」已经多久了。

        基线优先取**最后一次坐标变化**；一个坐标事件都没见过时
        退化成「进入大世界的时刻」（见模块文档「安全侧优先」）。
        """
        baseline = self._last_move_at
        if baseline is None:
            baseline = self._world_since
        if baseline is None:
            return 0.0
        return max(0.0, now - baseline)


__all__ = [
    "STATE_COMBAT", "STATE_WORLD", "STATE_IDLE", "STATE_UNKNOWN",
    "ALL_STATES",
    "IDLE_SECONDS", "IDLE_PROACTIVE_MIN_GAP_S",
    "COMBAT_SCENES", "WORLD_SCENES", "PERCEPTION_DOWN_CODES",
    "GROUP_WORLD", "GROUP_COMBAT", "GROUP_MENU", "GROUP_TRANSITION",
    "SCENE_GROUP",
    "state_for_scene", "group_for_scene",
    "ChatState",
]
