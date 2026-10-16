"""意图执行器：把 IntentPacket 变成真实的键鼠事件。

对应架构文档 6.7。

三条硬规则（写在这个文件最显眼的位置，因为它们最容易被违反）：

1. **展开后不重新决策**（原则 3）
   一个意图展开成多步（"使用大招" = 按 Q + 等待 + 按 W）之后，
   这几步**不再回头问决策层**。否则第二步的状态判断已经过期。

2. **finally 里无条件 release_all()**
   任何异常/中断都可能留下卡住的按键 —— W 被按住不放 = 角色一直往前跑。

3. **所有 SendInput 必须校验返回值**
   由 InputSimulator 负责；这里只处理抛出的 InputInjectionError。

**测试友好**：`InputSimulator(dry_run=True)` 可以在不发任何真实输入的
情况下完整跑通本文件。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..contracts.events import Event
from ..contracts.intent_packet import IntentPacket
from ..contracts.state_packet import StatePacket
from ..nav import steer as nav_steer
from ..nav.steer import DEFAULT_CONFIG as _STEER_CFG
from .humanizer import Humanizer, HumanizerConfig
from .safety import SafetyGuard, is_hard_stop
from .simulator import (
    DryRunBackend,
    InputInjectionError,
    InputSimulator,
    KeyId,
    VK,
    Win32Backend,
)
from .traits import ActionTraits, traits_of
from .validator import IntentValidator, ValidationResult

logger = logging.getLogger(__name__)

#: MOVE_TO 转向分段之间的间隔（秒）。
#:
#: ★ 必须**大于游戏的一帧**（33.3ms @30fps），否则多段会被并进同一渲染帧，
#:   重新触发游戏的单次位移上限 —— 拆分等于没拆（真机踩过：20ms < 33.3ms）。
#:   40ms 覆盖到 25fps，比观测到的最差帧率还安全一档。
#:   ⚠️ 代价：站定的 8 段最多阻塞 320ms，主循环帧间隔会从 200ms 涨到
#:   ~520ms。这会加速位移窗口见底 —— 所以站定期间有安全阀（见 nav.steer）。
NAV_TURN_SEG_SLEEP_S = 0.040

#: MOVE_TO 位移窗口的宽度（秒）。与 e2e_pathfind.py 的 WINDOW_S 同量级。
NAV_WINDOW_S = 2.0


class _NavWindow:
    """MOVE_TO 的**位移窗口** —— 朝向估计的输入。

    ★ 为什么必须自己维护
    ------------------
    ``PositionInfo`` 里**确实有** ``heading_deg`` 字段，但它
    **恒为 None**：朝向提取已被证伪（规划文档 §7，朝向 σ 中位 17.3°，
    两段失控 R=0.388/0.402），``perception/position_mapping.py`` 明确
    「宁可不填，也不填一个假的」。

    所以唯一的可用来源就是**自己攒位移历史**：滑动窗口内各段位移方向的
    圆周均值。这正是 ``autopilot.nav.steer.heading_deg`` 的算法。

    ⚠️ 平面轴名：``nav.steer`` 的形参叫 ``pos_z`` / ``target_z``，那是抽库
    时沿用寻路脚本的叫法。**值就是平面的第二个分量**，也就是
    ``PositionInfo.y``（它存的是世界 z）。这里统一按 ``y`` 称呼。
    """

    __slots__ = ("_clock", "_hist", "stand_active", "stand_budget_deg",
                 "target_key")

    #: 窗口最多留多少个样本（防止长时间无 tick 时无限增长）。
    MAX_SAMPLES = 64

    def __init__(self, clock=None) -> None:
        self._hist: list[tuple[float, float, float]] = []
        #: 时钟源。**可注入**是为了让闭环仿真能推进虚拟时间 ——
        #: 窗口老化完全依赖它，用真实 monotonic 的话，仿真里不 sleep
        #: 就等于「窗口永不老化」，旧样本会一直把朝向均值拖偏。
        #: 那种失真会让调参调到一个根本不存在的问题上。
        self._clock = clock if clock is not None else time.monotonic
        #: 站定状态是**跨帧**的：进站定后要一直站到夹角降到恢复阈值。
        #: 由 :func:`nav.steer.steer` 产出，调用方负责存回来。
        self.stand_active: bool = False
        #: 站定阶段的**剩余转向预算**（度）。2026-10-14 加。
        #:
        #: 站定时角色不动 -> 朝向窗口冻结 -> ``heading`` 陈旧。若每帧按
        #: 陈旧 Δ 重算转向，角色会冲过头、反向、再冲 —— 闭环振荡。
        #: 所以站定必须**开环**：进站定时算一次总预算，之后按帧扣减。
        #: 不传回（每帧都当 0）就会退化成老行为，见 nav.steer 的说明。
        self.stand_budget_deg: float = 0.0
        #: 当前目标（用于检测「换目标 -> 窗口作废」）。
        self.target_key: tuple[float, float] | None = None

    def reset(self) -> None:
        self._hist.clear()
        self.stand_active = False
        self.stand_budget_deg = 0.0
        self.target_key = None

    def push(self, x: float, y: float) -> None:
        """记一个样本。**同一位置重复采样直接顶掉上一条** ——
        定位没动的时候，重复的同点只会稀释窗口。"""
        now = self._clock()
        if self._hist:
            lx, ly = self._hist[-1][1], self._hist[-1][2]
            if lx == x and ly == y:
                self._hist[-1] = (now, x, y)
                return
        self._hist.append((now, x, y))
        if len(self._hist) > self.MAX_SAMPLES:
            del self._hist[:-self.MAX_SAMPLES]

    def _prune(self, now: float) -> None:
        """丢掉窗口外的样本。**永远至少留 2 个**。

        少于 2 个样本就凑不出一段位移，``heading_deg`` 会直接返回 None ——
        而 None 会让 MOVE_TO 退化成盲走（朝错方向冲），比慢一拍糟得多。
        """
        floor = now - NAV_WINDOW_S
        i = 0
        while i < len(self._hist) - 2 and self._hist[i][0] < floor:
            i += 1
        if i:
            del self._hist[:i]

    def heading(self) -> float | None:
        """窗口朝向（度）。有效段不足时返回 ``None``。"""
        now = self._clock()
        self._prune(now)
        return nav_steer.heading_deg(self._hist, now, NAV_WINDOW_S, _STEER_CFG)

    def window_move_m(self) -> float | None:
        """窗口内首尾的净位移（米）。样本不足返回 ``None``。

        这是站定期的**安全阀**输入：站定时不走路，位移窗口会见底，
        而朝向的唯一来源就是位移窗口 —— 窗口一死转向就静默失效。
        """
        now = self._clock()
        self._prune(now)
        if len(self._hist) < 2:
            return None
        (_, x0, y0), (_, x1, y1) = self._hist[0], self._hist[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


class Outcome(str, Enum):
    OK = "OK"
    REJECTED = "REJECTED"       # 校验没过
    FAILED = "FAILED"           # 执行中出错
    INTERRUPTED = "INTERRUPTED"  # 被急停/熔断打断
    TIMEOUT = "TIMEOUT"         # 超过 traits.max_duration_ms


@dataclass
class ExecutionResult:
    """一次意图执行的结果。**永远带原因**（原则 1）。"""

    outcome: Outcome
    reason: str
    intent_id: str
    action: str
    started_ms: int
    finished_ms: int = 0
    steps: list[str] = field(default_factory=list)
    released_keys: list[int] = field(default_factory=list)
    event: Event | None = None

    #: 进入 execute() 时的 **monotonic** 时刻。
    #:
    #: ★ 为什么不用墙钟算耗时：`started_ms` 可以是调用方注入的（回放/测试），
    #: 而 `time.time()` 是真实墙钟，两者相减会得到「耗时 56 年」这种垃圾值；
    #: 墙钟还会被 NTP 校正跳变。时长只认 monotonic。
    start_mono: float = field(default=0.0, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.OK

    @property
    def elapsed_ms(self) -> int:
        if self.start_mono > 0:
            return int((time.monotonic() - self.start_mono) * 1000)
        # 未经过 execute()（测试直接构造）时的退化路径
        return (self.finished_ms or int(time.time() * 1000)) - self.started_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "reason": self.reason,
            "intent_id": self.intent_id,
            "action": self.action,
            "elapsed_ms": self.elapsed_ms,
            "steps": self.steps,
            "released_keys": self.released_keys,
        }


class IntentExecutor:
    """意图执行器。

    组装：Validator（合法性）→ SafetyGuard（允不允许）→ 展开 → Simulator。
    """

    def __init__(
        self,
        *,
        simulator: InputSimulator | None = None,
        validator: IntentValidator | None = None,
        guard: SafetyGuard | None = None,
        humanizer: Humanizer | None = None,
        keymap: dict[str, KeyId] | None = None,
        char_slots: dict[str, int] | None = None,
        dry_run: bool = False,
    ) -> None:
        # ★ 注意 `InputSimulator()` / `InputSimulator(backend=DryRunBackend())`
        #   而不是 `backend=DryRunBackend() if dry_run else None` ——
        #   显式传 None 会覆盖 dataclass 的 default_factory，
        #   于是 IntentExecutor() 不带参数（最自然的用法）拿到 backend=None，
        #   第一次执行时炸成 "'NoneType' object has no attribute 'send'"。
        if simulator is not None:
            self.sim = simulator
        else:
            self.sim = InputSimulator(
                backend=DryRunBackend() if dry_run else Win32Backend()
            )
        self.validator = validator or IntentValidator()
        self.guard = guard or SafetyGuard()
        self.humanizer = humanizer or Humanizer(HumanizerConfig())
        #: 动作名 -> KeyId（可配置，keymap_editor 写入 config/keymap.toml）
        self.keymap: dict[str, KeyId] = dict(keymap or DEFAULT_KEYMAP)
        #: 角色 id -> 位置编号 1..4（``SWITCH_CHARACTER`` 靠它把 char_id
        #: 翻译成「按哪个数字键」）。
        #:
        #: ★★ 这里曾经是一个**从未被赋值**的属性
        #:   旧代码是 ``getattr(self, "_char_slots", {}) or {}``，
        #:   而全插件**没有任何一处**给它赋值 —— 于是它恒为 ``{}``、
        #:   ``_slot_of()`` 恒返回 None、``SWITCH_CHARACTER`` 被静默跳过
        #:   （只在 steps 里留一句「跳过（无可切的位）」，不抛异常）。
        #:
        #: ★★ 为什么不能从 ``self.keymap`` 里取
        #:   旧代码写了 ``self.keymap.get("__char_slots__")``，但
        #:   :func:`actuator.keymap.load_keymap` **只接受** ``DEFAULT_KEYMAP``
        #:   里的动作名，未知名字一律 ``KeymapError``（拼错要报错是故意的）。
        #:   所以 ``__char_slots__`` 永远进不了 keymap —— 那行是死代码，
        #:   赋了值从没读过。
        #:
        #: 所以它必须是**独立的注入点**：构造参数或 :meth:`set_char_slots`，
        #: 配置来源见 :func:`actuator.keymap.load_char_slots`。
        self._char_slots: dict[str, int] = {
            str(k): int(v) for k, v in (char_slots or {}).items()}
        # ★ 执行前再确认一次 backend 可用（构造期已查，这里防的是
        #   手工改写 sim.backend 之后绕过检查的情况）
        self.sim.require_backend()
        self._history: list[ExecutionResult] = []
        #: 上一次 execute 结束时**仍然按住**的 VK 码（长按语义，§37）。
        #: 空 tuple = 什么都没按住。用来判断下一条意图要不要先松手。
        self._held: tuple[int, ...] = ()
        #: 本次 execute 按下但**不打算松**的键。由 :meth:`_run_sequence`
        #: 在真正 key_down/mouse_down 的地方登记 —— 不另建映射表，
        #: 避免"哪个动作按哪个键"这份知识在两处漂移。
        self._pending_hold: list[int] = []
        #: 上一次**成功**执行的意图动作名。用来判断意图有没有换。
        self._last_action: str = ""
        #: MOVE_TO 的位移窗口（朝向估计）。**跨 tick 存活** —— 朝向只能
        #: 从一段时间的位移方向里估，单帧位置给不出朝向。
        self._nav = _NavWindow()

    # -- 对外 ------------------------------------------------------------

    @property
    def history(self) -> list[ExecutionResult]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    @property
    def held_keys(self) -> tuple[int, ...]:
        """当前**仍然按住**的键（长按语义，§37）。空 = 什么都没按住。

        给调试工具看「左键是不是还按着」用 —— 这正是本次修的那个 bug
        最直接的证据。
        """
        return self._held

    def release_held(self) -> list[int]:
        """松开所有长按中的键，并清空持有状态。

        ★ **loop 在「本轮无意图」时必须调它**（§37，尚未接线）::

            if intent is None:
                self.ctx.executor.release_held()   # ← 就这一行
                ...
                return

        为什么需要：``execute()`` 只在**有意图**时被调用，而
        ``NORMAL_ATTACK`` 是长按。敌人死亡 -> 规则不再命中 -> 没有任何
        execute -> 左键一直按住，直到用户暂停/停止/按 F12。

        ⚠️ 这次没接是因为本轮约束是「不改 core/loop.py」。
           方法先留在这里，接线时不必再动执行层。
        """
        released = self.sim.release_all()
        self._held = ()
        self._pending_hold = []
        return released

    # -- MOVE_TO 的按键管理 --------------------------------------------
    #
    # 为什么不能直接复用 :meth:`_run_sequence` 内部的 ``hold_key``：
    #   ``hold_key`` 只会「按下」，**从不主动松**。而 MOVE_TO 是按 plan
    #   逐帧决定走不走的 —— 上一帧按了 W，这一帧判「该站着转」，
    #   就必须真的把 W 松开。只「不按下新键」会让 ``_held`` 把它无限
    #   结转下去，角色一直往前冲。症状极难自查：规则没报错、事件正常、
    #   只是「明明该停下却没停」。

    def _nav_hold(self, vk: VK) -> bool:
        """MOVE_TO 用：按住一个键（已按着就什么都不做）。返回是否真按了。"""
        code = int(vk)
        if code in self._held:
            return False
        self.sim.key_down(vk)
        self._pending_hold.append(code)
        return True

    def _nav_release(self, vk: VK) -> bool:
        """MOVE_TO 用：**主动**松开一个键并从 ``_held`` 里摘掉。

        返回是否真的松了（本来就没按着则返回 False，不发多余输入）。
        """
        code = int(vk)
        if code not in self._held:
            return False
        self.sim.key_up(vk)
        self._held = tuple(c for c in self._held if c != code)
        return True

    def set_keymap(self, keymap: dict[str, KeyId]) -> None:        self.keymap = dict(keymap)

    def set_char_slots(self, char_slots: dict[str, int]) -> None:
        """注入「角色 id -> 位置编号 1..4」。

        ★ 与 :meth:`set_keymap` 分开，而不是混在 keymap 里
          keymap 是 ``动作名 -> 按键``，语义是「按哪个键」；
          char_slots 是 ``角色 id -> 槽位``，语义是「第几个人」。
          塞进同一张表只会让 :func:`load_keymap` 的严格校验没法用
          （它会为未知动作名报错，那是对的）。
        """
        self._char_slots = {str(k): int(v) for k, v in char_slots.items()}

    def resolve_key(self, action: str) -> KeyId:
        """动作 -> KeyId。**未配置直接抛**（响亮失败，不猜默认键）。"""
        kid = self.keymap.get(action)
        if kid is None:
            raise KeyIdError(
                f"动作 {action} 没有配置按键。\n"
                f"运行 python -m autopilot.tools.keymap_editor 编辑，"
                f"或检查 config/keymap.toml"
            )
        if kid is KeyId.NONE:
            raise KeyIdError(f"动作 {action} 的按键被显式设为 NONE（不执行）")
        return kid

    # -- 主流程 ----------------------------------------------------------

    def execute(
        self,
        intent: IntentPacket,
        state: StatePacket,
        now_ms: int | None = None,
    ) -> ExecutionResult:
        """执行一个意图。**这个方法不抛异常** —— 失败都变成 Outcome。"""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        t = traits_of(intent.action.value)
        res = ExecutionResult(
            outcome=Outcome.FAILED,
            reason="",
            intent_id=intent.intent_id,
            action=intent.action.value,
            started_ms=now,
        )
        # ★ 时长/超时一律走 monotonic，不受注入时钟或 NTP 跳变影响
        res.start_mono = time.monotonic()

        try:
            # ① 合法性（含"这动作在这个场景/这个状态下合不合法"）
            v: ValidationResult = self.validator.validate(intent, state, now, commit=True)
            if not v:
                return self._finish(res, Outcome.REJECTED, str(v))

            # ② 允不允许执行（急停 / 熔断 / 频率 / 焦点）
            ok, reason = self.guard.check(intent, state, now)
            if not ok:
                outcome = (
                    Outcome.INTERRUPTED if is_hard_stop(reason)
                    else Outcome.REJECTED
                )
                return self._finish(res, outcome, reason)

            # ③ 解析按键（配置错在这一步就炸，不会带着错键去按）
            #    只有**真的需要按键**的动作才查 —— WAIT / SCREENSHOT / NOOP
            #    不发任何输入，DEFAULT_KEYMAP 里它们的键位是 NONE（哨兵），
            #    对哨兵查按键会误报「未配置」。
            try:
                key_id = None
                if intent.action.value not in _KEYLESS_ACTIONS:
                    key_id = self.resolve_key(intent.action.value)
            except KeyIdError as exc:
                return self._finish(res, Outcome.REJECTED, f"KEYMAP: {exc}")

            # ── ④ 展开并逐动作执行 ──────────────────────────────────
            #
            # ★★ **意图切换必须「先松后按」**（§37）。
            #   NORMAL_ATTACK 走 ``mouse_down(LBUTTON)``（长按语义，
            #   故意不松），而 ``release_all()`` 只在 outcome 非 OK 时调 ——
            #   于是「普攻成功 -> 切闪避」这条最常见的路径上，左键会一直
            #   按住直到暂停/停止。
            #
            #   ⚠️ 为什么不能放到 finally 里松：那样会变成「先按 SHIFT、
            #      再 release_all」—— 连刚按下的 SHIFT 一起松开，
            #      闪避等于没按。顺序反了比不修更糟。
            action_name = intent.action.value
            prev_action = self._last_action
            pre_released = 0
            if self._held and action_name != prev_action:
                pre_released = len(self.sim.release_all())
                self._held = ()
            self._last_action = action_name
            self._pending_hold = []
            try:
                res.steps = self._run_sequence(
                    intent, t, key_id, res, state
                )
            except InputInjectionError as exc:
                ev = self.guard.mark_failure(str(exc), now)
                return self._finish(res, Outcome.FAILED, f"INPUT_INJECT: {exc}", ev)
            except _PanicStop:
                return self._finish(res, Outcome.INTERRUPTED, "PANIC_KEY_PRESSED")
            except _Timeout as exc:
                return self._finish(res, Outcome.TIMEOUT, str(exc))
            if pre_released:
                res.steps.insert(
                    0, f"release(切换自 {prev_action}): 松开 {pre_released} 个键")

            if res.elapsed_ms > t.max_duration_ms:
                return self._finish(
                    res, Outcome.TIMEOUT,
                    f"耗时 {res.elapsed_ms}ms 超过 max_duration_ms={t.max_duration_ms}",
                )

            # ⑤ 记账。**必须传 now**，不能传 time.time()*1000 ——
            #    guard.check 读的是 now，记账写墙钟会让频率限制读到另一条时间线。
            self.guard.mark_executed(intent.action.value, now)
            self.guard.mark_success()
            return self._finish(res, Outcome.OK, "OK")

        except Exception as exc:  # noqa: BLE001 - 执行器不外抛
            logger.exception("执行意图时发生未预期异常")
            ev = self.guard.mark_failure(f"{type(exc).__name__}: {exc}", now)
            return self._finish(res, Outcome.FAILED, f"UNEXPECTED: {exc}", ev)
        finally:
            # ── 松键策略（§37）──────────────────────────────────────
            # ⚠️ **绝不在这里写 return** —— finally 里的 return 会吞掉
            #    异常，KeyboardInterrupt（Ctrl+C）会被静默吃掉。
            #
            # ① 非 OK（校验拒绝 / 熔断 / 异常 / 超时 / 急停）
            #    —— 无条件全松，这是原有的异常安全底线，一字未改。
            # ② OK —— 「意图切换时的先松后按」已经在 ④ 之前做完了，
            #    这里只把按住状态结转给下一条意图。**不做任何松键**：
            #    同一条 NORMAL_ATTACK 连着来时左键保持按住（真·长按）。
            if res.outcome is not Outcome.OK:
                res.released_keys = self.sim.release_all()
                self._held = ()
            else:
                # ★ 只有**真的新按下**了键才更新 _held。
                #   hold_key 跳过的那些（同一条 NORMAL_ATTACK 连着来）
                #   **仍然按着** —— 无条件写 ``_held = tuple(_pending_hold)``
                #   会把它们误清空，于是第 3 帧又按一次，长按退化成
                #   「按-按-按」（这正是 R1 抓到的那条）。
                if self._pending_hold:
                    self._held = tuple(self._pending_hold)
            self._pending_hold = []

    def _finish(
        self,
        res: ExecutionResult,
        outcome: Outcome,
        reason: str,
        event: Event | None = None,
    ) -> ExecutionResult:
        res.outcome = outcome
        res.reason = reason
        # ★ finished_ms 由 elapsed_ms 推出，保证 (started, finished) 自洽。
        #   直接写 time.time()*1000 会在 now_ms 注入时算出「56 年」。
        res.finished_ms = res.started_ms + res.elapsed_ms
        self._history.append(res)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        if outcome is not Outcome.OK:
            logger.warning("意图 %s (%s) -> %s: %s",
                           res.intent_id, res.action, outcome.value, reason)
        return res

    # -- 动作序列 --------------------------------------------------------

    def _run_sequence(
        self,
        intent: IntentPacket,
        t: ActionTraits,
        key_id: KeyId,
        res: ExecutionResult,
        state: StatePacket | None = None,
    ) -> list[str]:
        """把一个意图展开成动作序列并执行。

        ★ 这里的每一步都**不回头问决策层**（原则 3）。

        :param state: 2026-10-14 新增。**只有 MOVE_TO 需要它**（要读
            当前位置与朝向窗口）。做成可选是为了不破坏既有的直接调用方
            —— ``tests/unit/test_switch_character_slots.py`` 就绕开
            ``execute()`` 直接调这里来验槽位解析。``None`` 时 MOVE_TO
            会响亮失败而不是崩。
        """
        action = intent.action.value
        p = intent.params or {}
        steps: list[str] = []
        # ★ deadline 走 monotonic，不走 res.started_ms（那可能是注入的假时间）
        deadline_mono = res.start_mono + t.max_duration_ms / 1000.0

        def check() -> None:
            """每个动作之间都问一次急停 —— 急停必须**立刻**生效。"""
            if self.guard.panic_active:
                raise _PanicStop()
            if time.monotonic() > deadline_mono:
                raise _Timeout(f"超过 max_duration_ms={t.max_duration_ms}")

        def hold_key(vk) -> bool:
            """长按一个**键盘**键（按下 + 登记为「本次不松」）。返回是否真按了。

            ★ **已经按着就什么都不做**（§37）。重复 down 不是无害的：
              同一条 MOVE_FORWARD 每 tick 命中一次，不做这个判断就会
              每帧发一次 ``KEYDOWN``，Windows 对「已按下的键再报 down」
              有自动重复语义。

            ⚠️ §39 起这里**只管键盘键**了。原实现有 ``mouse=True``
              分支给 NORMAL_ATTACK 用，但普攻已改成点按（tap），走
              :meth:`InputSimulator.mouse_tap`，不再进这个函数。留着
              死参数会让人以为「鼠标也能长按」—— 而那恰恰是这次修掉的
              错误语义。

            ⚠️ 名字别叫 ``hold`` —— JUMP / USE_SKILL 分支里已经有
              ``hold = self.humanizer.hold_ms(...)`` 这个局部变量，
              同名会把本函数**整个覆盖掉**（同名局部变量的遮蔽是
              编译期决定的，不会报错，只会静默失效）。
            """
            code = int(vk)
            if code in self._held:        # 上一条意图已经按着了
                return False
            self.sim.key_down(vk)
            # int() 对 VK(IntEnum) 与 KeyId(int 子类) 都取到真实键码
            self._pending_hold.append(code)
            return True

        if action == "WAIT":
            ms = int(p.get("ms", 1000))
            d = self.humanizer.delay_ms(ms)
            steps.append(f"wait {ms}->{d}ms")
            time.sleep(d / 1000.0)
            return steps

        if action == "SCREENSHOT":
            steps.append("screenshot")
            return steps

        if action == "NOOP":
            steps.append("noop")
            return steps

        if action == "NORMAL_ATTACK":
            # ★ 普攻 = **点按**（短 tap），不是长按（§39）。
            #   真机实测：长按左键在原神里是**蓄力重击**，不是「攻击得更久」。
            #   6 帧意图只打出 1 次重击就是这么来的。
            #   每次调用都是完整的一对 down/up，不进 _pending_hold ——
            #   点按结束即抬起，没有「还按着」的状态需要结转。
            hold_ms = int(p.get("hold_ms", t.tap_hold_ms or 120))
            steps.append(f"mouse tap {hold_ms}ms")
            self.sim.mouse_tap(VK.LBUTTON, hold_ms=hold_ms)
            check()
            return steps

        # ★★ MOVE_TO：走到世界坐标 (x, y) —— **每 tick 只推进一步**
        # --------------------------------------------------------
        # 它是**目标**不是动作。走到目标可能要几十秒，而 execute() 是
        # 单次调用即返回的；在这里 while 循环跑到达会把主循环堵死几十秒。
        # 所以：每次只做「转一段 + 决定走不走」，下一 tick 同一意图继续
        # 推进 —— 这与 MOVE_FORWARD 靠 ``_held`` 跨 tick 按住 W 是同一个
        # 模型。
        #
        # ⚠️ 这个分支**必须在下面那个 `key_id is None` 兜底之前**：
        #   MOVE_TO 在 ``_KEYLESS_ACTIONS`` 里（它不走「一个动作 = 一个
        #   键」的映射），所以 key_id 恒为 None —— 放到兜底之后就会被
        #   截成「MOVE_TO: 无需按键」，一步都走不出去。
        #
        # 坐标读 ``target.world_xy()``（x/y 规范，兼容遗留 z），
        # **不要**直接 ``world_position["x"]``。
        if action == "MOVE_TO":
            wp = intent.target.world_xy()
            if wp is None:
                # 响亮失败，不猜一个"大概是那边"的点走下去。
                return [f"MOVE_TO: target.world_position 缺少可用的 x/y"
                        f"（实得 {intent.target.world_position!r}）"]
            tx, ty = wp
            radius = intent.target.arrive_radius(
                default=_STEER_CFG.arrive_radius_m)

            pos = state.position if state is not None else None
            if pos is None or pos.x is None or pos.y is None:
                # traits.abort_on_target_lost=True：定位没了就停，
                # **禁止**用运动预测填补（PositionInfo 的明文契约）。
                self._nav.reset()
                return ["MOVE_TO: 定位不可用，已放弃本段行走"]

            # 换目标 -> 旧目标的位移方向对新目标毫无意义，窗口作废。
            key = (round(tx, 1), round(ty, 1))
            if self._nav.target_key != key:
                self._nav.reset()
                self._nav.target_key = key

            # ⚠️ 平面第二分量在 contracts 里叫 ``y``（存的是世界 z），
            #    而 nav.steer 的形参叫 ``pos_z``/``target_z``。值一样，
            #    只是叫法不同 —— 这里按 contracts 的说法传。
            self._nav.push(float(pos.x), float(pos.y))

            plan = nav_steer.steer(
                pos_x=float(pos.x), pos_z=float(pos.y),
                target_x=tx, target_z=ty,
                heading=self._nav.heading(),
                arrive_radius_m=radius,
                stand_active=self._nav.stand_active,
                want_sprint=bool(p.get("sprint", False)),
                window_move_m=self._nav.window_move_m(),
                stand_budget_deg=self._nav.stand_budget_deg,
                cfg=_STEER_CFG,
            )
            self._nav.stand_active = plan.stand_active
            # ★ 转向预算必须**跨帧传回**。不传就等于每帧重新开预算，
            #   站定期间就会一直发满转向 -> 过冲 -> 振荡。
            self._nav.stand_budget_deg = plan.stand_budget_deg

            if plan.action == "ARRIVED":
                steps.append(
                    f"move_to: 到达 (方位 {plan.bearing:.1f}°) "
                    f"dist={plan.dist_m:.2f}m <= r{radius:g}")
                self._nav.reset()
                check()
                return steps

            # ── 转向：分段注入，每段之间留够一个游戏帧 ──────────────
            for seg in plan.segments:
                self.sim.mouse_move_relative(int(seg), 0)
                if NAV_TURN_SEG_SLEEP_S:
                    time.sleep(NAV_TURN_SEG_SLEEP_S)
            if plan.segments:
                steps.append(
                    f"move_to: turn {sum(plan.segments):+d}px "
                    f"in {len(plan.segments)} seg(s)"
                    + (f", 欠 {plan.owed_px:+d}px" if plan.owed_px else ""))
            check()

            # ── 行走 / 冲刺：按 plan 决定，并**显式松掉不该按的键** ──
            # 上一 tick 按着 W、这一 tick plan 说别走 —— 必须真的松。
            # 只「不按下新键」是不够的：``_held`` 会把 W 一直结转下去，
            # 症状是「明明该停下却一直往前冲」，最难自查。
            if plan.walk:
                steps.append(f"key {VK.W.name} down (move_to walk)"
                             if self._nav_hold(VK.W)
                             else f"key {VK.W.name} 保持按住")
                if plan.sprint:
                    steps.append(f"key {VK.SHIFT.name} down (move_to sprint)"
                                 if self._nav_hold(VK.SHIFT)
                                 else f"key {VK.SHIFT.name} 保持按住")
            else:
                if self._nav_release(VK.W):
                    steps.append(f"key {VK.W.name} up (move_to 不走了)")
                if self._nav_release(VK.SHIFT):
                    steps.append(f"key {VK.SHIFT.name} up (move_to 停冲刺)")
            return steps

        if key_id is None:  # pragma: no cover - 上面已过滤
            return [f"{action}: 无需按键"]

        if action == "DODGE":
            # ★ 闪避 = **点按** Shift，不是长按（§40）。
            #   原神里 Shift 是同一个键：点按=闪避，长按=冲刺。
            #   原来这里用 hold_key() 长按，于是 world_emergency_dodge
            #   触发时角色**冲刺**而不是闪避 —— 而且冲刺移速更快，
            #   看起来更像「闪避成功了」，是极难自查的一类 bug。
            #
            #   直接用 ``key_press``：它本来就是「按下 -> 真实保持 ->
            #   抬起」（down/up 分两次 SendInput，中间真 sleep）。
            #   键盘侧早就有这个原语，**不需要**为 DODGE 新增什么 ——
            #   与 NORMAL_ATTACK 的 mouse_tap 不同，鼠标侧只有原子的
            #   mouse_click，所以那边才要补一个 mouse_tap。
            #
            #   ⚠️ 时长不过 humanizer：点按时长是**游戏机制阈值**
            #     （超过就变成冲刺），加抖动等于拿机制边界赌运气。
            hold_ms = int(p.get("hold_ms", t.tap_hold_ms or 120))
            steps.append(f"key {key_id.name} tap {hold_ms}ms")
            self.sim.key_press(key_id, hold_ms=hold_ms)
            check()
            return steps

        if action == "SPRINT":
            steps.append(f"key {key_id.name} down (sprint hold)"
                         if hold_key(key_id)
                         else f"key {key_id.name} 保持按住（跳过重复 down）")
            check()
            return steps

        if action in ("MOVE_FORWARD", "MOVE_BACK", "MOVE_LEFT", "MOVE_RIGHT"):
            # 移动类：按方向键，保持住（原则 5：移动中才转得动视角）
            vk = _move_key(action)
            steps.append(f"key {vk.name} down (move hold)"
                         if hold_key(vk)
                         else f"key {vk.name} 保持按住（跳过重复 down）")
            check()
            return steps

        if action == "JUMP":
            hold = self.humanizer.hold_ms(int(p.get("hold_ms", 50)))
            steps.append(f"key {key_id.name} press {hold}ms")
            self.sim.key_press(key_id, hold_ms=hold)
            check()
            return steps

        if action in ("USE_SKILL", "USE_BURST", "CONFIRM", "CANCEL"):
            hold = self.humanizer.hold_ms(int(p.get("hold_ms", 30)))
            steps.append(f"key {key_id.name} press {hold}ms")
            self.sim.key_press(key_id, hold_ms=hold)
            check()
            gap = self.humanizer.delay_ms(int(p.get("gap_ms", 120)))
            if gap:
                time.sleep(gap / 1000.0)
                steps.append(f"gap {gap}ms")
            return steps

        if action == "SWITCH_CHARACTER":
            want = p.get("char_id")
            slot = p.get("slot")
            if slot is None and want is not None:
                slot = self._slot_of(want)
            if slot is None:
                steps.append("switch: 无 slot/char_id，跳过（无可切的位）")
                return steps
            k = _slot_key(int(slot))
            steps.append(f"key {k.display_name} press (switch slot {slot})")
            self.sim.key_press(k, hold_ms=self.humanizer.hold_ms(30))
            check()
            return steps

        if action in ("OPEN_INVENTORY", "OPEN_MAP", "OPEN_PAIMON_MENU", "CLOSE"):
            steps.append(f"key {key_id.name} press (menu)")
            self.sim.key_press(key_id, hold_ms=self.humanizer.hold_ms(30))
            check()
            return steps

        if action == "CLICK_ELEMENT":
            if p.get("x") is not None and p.get("y") is not None:
                x, y = int(p["x"]), int(p["y"])
            elif p.get("element"):
                # 按元素名点击需要 perception 提供 ROI 能力，第 1 期还没有 ——
                # 响亮失败，不猜一个"大概的位置"点下去。
                return [f"CLICK_ELEMENT: 只给了 element={p['element']!r}，"
                        f"没有屏幕坐标。当前版本需要 x + y。"]
            else:
                return ["CLICK_ELEMENT: 缺少 x/y"]
            jx, jy = self.humanizer.jitter_click(x, y)
            steps.append(f"mouse click ({x},{y})->({jx},{jy})")
            self.sim.mouse_move_to(jx, jy)
            time.sleep(self.humanizer.delay_ms(40) / 1000.0)
            self.sim.mouse_click(VK.LBUTTON, hold_ms=self.humanizer.hold_ms(25))
            check()
            return steps

        # 兜底：traits 表里没有对应实现的动作，用单键按压
        hold = self.humanizer.hold_ms(int(p.get("hold_ms", 30)))
        steps.append(f"key {key_id.name} press {hold}ms (fallback)")
        self.sim.key_press(key_id, hold_ms=hold)
        return steps

    def _slot_of(self, char_id: str) -> int | None:
        """角色 id -> 位置编号 1..4。

        ★ 数据源是 :attr:`_char_slots`，由构造参数或
          :meth:`set_char_slots` 注入 —— 见那里的长注释说明「为什么不能
          从 ``self.keymap`` 里取」。

        :returns: 查不到返回 ``None``（调用方据此跳过并**记进 steps**，
            不是静默无事发生）。
        """
        v = self._char_slots.get(char_id)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # -- 诊断 ------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "guard": self.guard.health(),
            "humanizer": self.humanizer.describe(),
            "keymap_entries": len(self.keymap),
            "history": len(self._history),
        }


# --------------------------------------------------------------------------
# 内部异常
# --------------------------------------------------------------------------


class _PanicStop(Exception):
    """F12 被按下。立即中止当前意图。"""


class _Timeout(Exception):
    """超过 traits.max_duration_ms。"""


# --------------------------------------------------------------------------
# 默认键位
# --------------------------------------------------------------------------

#: 动作 -> 键位 的**默认**映射（原神 PC 版常见默认键）。
#: 玩家键位不同，跑 keymap_editor 改，不要在这里猜。
#: 值用 ``VK`` 成员（IntEnum，与 KeyId 兼容）；NONE 表示该动作不发按键。
DEFAULT_KEYMAP: dict[str, KeyId] = {
    # COMBAT
    "NORMAL_ATTACK": VK.LBUTTON,
    "USE_SKILL": VK.E,
    "USE_BURST": VK.Q,
    "DODGE": VK.SHIFT,
    "SWITCH_CHARACTER": VK.ONE,
    # NAV
    "MOVE_FORWARD": VK.W,
    "MOVE_BACK": VK.S,
    "MOVE_LEFT": VK.A,
    "MOVE_RIGHT": VK.D,
    "SPRINT": VK.SHIFT,
    "JUMP": VK.SPACE,
    # ★ MOVE_TO 用 NONE 哨兵，不是「漏配了」。
    #   它确实要按键（W / Shift），但**不走「一个动作 = 一个键」的映射**
    #   —— 按哪个键、什么时候松，由 nav.steer 的规划逐帧决定。
    #   写成 W 会让「换目标要不要继续按住 W」这个问题失去表达力，
    #   而且 keymap.toml 里那一行会变成一个**骗人的可配置项**
    #   （玩家改了它也不会生效）。
    "MOVE_TO": KeyId.NONE,
    # UI
    "OPEN_INVENTORY": VK.B,
    "OPEN_MAP": VK.M,
    "OPEN_PAIMON_MENU": VK.ESCAPE,
    "CLICK_ELEMENT": VK.LBUTTON,
    "CONFIRM": VK.ENTER,
    "CANCEL": VK.ESCAPE,
    "CLOSE": VK.ESCAPE,
    # INTERACT：原神 PC 默认键位里拾取/采集/开箱/对话/激活锚点共用 F。
    # （键位表：techbloat 2024 PC 布局 "Interact (pick up / talk / use) F"，
    #   以及多个中文键位指南一致确认）
    "USE_PICKUP": VK.F,
    # SYSTEM
    "WAIT": KeyId.NONE,
    "SCREENSHOT": KeyId.NONE,
    "NOOP": KeyId.NONE,
}

#: 方向键是硬编码的（不是玩家可配项）—— 它们不参与 keymap 编辑
_MOVE_KEYS: dict[str, int] = {
    "MOVE_FORWARD": int(VK.W),
    "MOVE_BACK": int(VK.S),
    "MOVE_LEFT": int(VK.A),
    "MOVE_RIGHT": int(VK.D),
}

#: 队伍槽位 1..4 -> 主键盘 1..4
_SLOT_KEYS: dict[int, int] = {
    1: int(VK.ONE), 2: int(VK.TWO), 3: int(VK.THREE), 4: int(VK.FOUR),
}

#: 不发任何按键的动作（它们的 keymap 项是 NONE 哨兵，不要去查）
#:
#: ``MOVE_TO`` 也在里面，但**含义不同**：它确实要按键（W / Shift），
#: 只是不走「一个动作 = 一个键」的映射 —— 它自己按 :mod:`autopilot.nav.steer`
#: 的规划去驱动 W 与 Shift。若让它去 ``resolve_key``，会因为 keymap 里
#: 没有 ``MOVE_TO`` 而抛 ``KeyIdError``，表现为「意图命中但永远被拒」。
_KEYLESS_ACTIONS: frozenset[str] = frozenset(
    {"WAIT", "SCREENSHOT", "NOOP", "MOVE_TO"})


def _move_key(action: str) -> VK:
    """移动方向键。返回 **VK 枚举成员**，不是 KeyId。

    ⚠️ §39 顺带修掉的既有 bug：这里原来返回 ``KeyId(...)``，而调用方
      按 ``vk.name`` 取键名 —— ``KeyId`` 是 int 子类，只有
      ``display_name``，**没有** ``name``。结果 MOVE_FORWARD / BACK /
      LEFT / RIGHT 四个动作一执行就
      ``AttributeError: 'KeyId' object has no attribute 'name'`` -> FAILED。

      之所以一直没被发现：nav 的 11 条规则全部 ``enabled=false``（§30），
      **没有任何路径会走到这个分支**。是本轮把 MOVE_FORWARD 拿来当
      长按测试替身才暴露出来的 —— 测不到的东西不等于不存在。

      改成返回 VK 是为了和本文件其余分支保持一致（``resolve_key``
      给出的是 DEFAULT_KEYMAP 里的 VK 成员）。
    """
    return VK(_MOVE_KEYS[action])


def _slot_key(slot: int) -> KeyId:
    """槽位号 -> 主键盘数字键。**不是 VK 码本身** —— 槽位 1 是 VK.ONE(0x31)，
    不是 VK.LBUTTON(0x01)。这是很容易写错的地方。"""
    k = _SLOT_KEYS.get(int(slot))
    if k is None:
        raise InputInjectionError(
            f"队伍槽位必须在 1-4 之间，收到 {slot}。"
            f"（注意：槽位号不是 VK 码，别把 1 当成鼠标左键）"
        )
    return KeyId(k)


__all__ = [
    "DEFAULT_KEYMAP",
    "ExecutionResult",
    "IntentExecutor",
    "KeyIdError",
    "Outcome",
]

from .simulator import KeyIdError  # noqa: E402 - 供上层从本模块 import
