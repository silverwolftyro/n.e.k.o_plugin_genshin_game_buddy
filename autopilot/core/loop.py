"""主循环：把 perception / policy / actuator 三层串起来。

    ┌──────────┐   StatePacket   ┌────────┐   IntentPacket   ┌──────────┐
    │ 截图+识别 │ ──────────────> │  决策  │ ──────────────> │ 执行按键 │
    │perception│                 │ policy │                 │ actuator │
    └──────────┘                 └────────┘                 └──────────┘
           │                                                   │
           └──────────── WorldState（快照 + 事件 + 历史）────────┘

职责边界（硬约束）
--------------------
主循环只做四件事：**调三个方法、按状态机流转、汇事件、聚统计**。

它**不**做：
* 识别（那是 perception.build 的事，主循环不碰 ROI / 模板 / YOLO）
* 规则求值（policy.decide 的事，主循环不解析 when 表达式）
* 按键组装（executor 的事，主循环不碰 VK / traits / 人类化）

越界的代价是「改主循环才能改识别」，而主循环是唯一三层都要碰的文件 ——
一旦它长出业务逻辑，每次调整都会牵动全部三层。

★ 关于 validator
----------------
一个 tick 的标准链条是「感知 → 决策 → 合并 → **校验** → 执行」，
但本文件**不单独调 validator**。原因：`IntentExecutor.execute()` 内部
已经调了 ``validator.validate(intent, state, now, commit=True)``，
而 ``commit=True`` 会把 intent_id 写进去重表。

主循环再调一次 = 同一条意图提交两次去重记录，于是**下一个**意图会被
误判成"重放"而拒绝。这类 bug 只在"恰好连续两条意图"时出现，
排查起来极其痛苦（看起来像随机的间歇性失败）。

所以校验归执行层所有 —— 它才是真正发按键的地方，也是唯一需要
保证"发出去的意图一定合法"的地方。

线程模型
--------
``start()`` 起一个 daemon 线程跑 tick；控制方法在**调用方线程**执行。
急停/暂停/停止一律用 :class:`threading.Event` 通信，不用裸 flag 轮询 ——
flag 轮询在两个线程同时读写时既慢又容易漏（Python 里读一个 bool
不原子，理论上有窗口）。

F12 那一侧仍然需要轮询（``GetAsyncKeyState`` 没有事件接口），
但轮询只负责**置位 Event**，后续全是 Event 语义。20ms 轮询 +
Event 传递，实测停机延迟 < 20ms。
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

from ..contracts.events import Event, EventCategory, Severity
from ..contracts.intent_packet import IntentPacket
from ..contracts.state_packet import StatePacket
# 敌人判据**只用这一份**。2026-10-15 之前，本文件曾用 `kind` + `.y`
# 自己抄了一份，而 WorldEntity 根本没有 `.y` 字段，`getattr` 兜底成 1.0
# 让条件恒假 —— `health_bar` 永远计不进 ENEMY_DETECTED，规则却照常开火。
# 详见 policy/context.py::is_enemy_evidence 的长注释。
from ..policy.context import count_enemy_entities
from .runtime import (
    IllegalTransition,
    RuntimeContext,
    RuntimeState,
    check_transition,
)

logger = logging.getLogger(__name__)

#: F12 监视线程的轮询间隔。20ms 足够快（人察觉不到延迟），
#: 又不至于把 CPU 吃满。真正的停机延迟实测 < 20ms。
PANIC_POLL_S = 0.02

#: 连续多少次 tick 崩溃后主动暂停。
DEFAULT_MAX_CONSECUTIVE_CRASHES = 5


class CoreLoop:
    """主循环。

    典型用法::

        ctx = RuntimeContext(capturer=cap, builder=b, policy=pol,
                             executor=ex, tick_rate_hz=5)
        loop = CoreLoop(ctx)
        loop.start()
        ...
        loop.stop()          # 优雅退出：松键 -> 等 tick -> 关资源
    """

    def __init__(
        self,
        ctx: RuntimeContext,
        *,
        panic_check: Callable[[], bool] | None = None,
        on_event: Callable[[Event], None] | None = None,
        max_consecutive_crashes: int = DEFAULT_MAX_CONSECUTIVE_CRASHES,
        stop_timeout_s: float = 10.0,
    ) -> None:
        self.ctx = ctx
        self._state = RuntimeState.IDLE
        self._state_lock = threading.RLock()

        #: 每 tick 结束后的回调（调试叠加层 / 监控画画面用）
        self.on_tick: Callable[[StatePacket | None, IntentPacket | None], None] \
            | None = None

        # ── 线程与信号 ──────────────────────────────────────────
        #: 急停信号。**用 Event 不用 bool flag** —— 唤醒与判定都在
        #: 这里，读写不跨线程竞争。
        self._panic_event = threading.Event()
        #: 停止信号
        self._stop_event = threading.Event()
        #: 恢复信号。**独立于 _wake** —— _wake 在 pause() 里被置位且
        #: 只有 resume() 才清，如果 PAUSED 分支 wait(_wake) 会立刻返回，
        #: 变成 42kHz 的 CPU 空转（实测 0.4 秒跑了 16975 帧）。
        #: 专用的 _resume_event 由 pause() 清、resume()/stop() 置，语义干净。
        self._resume_event = threading.Event()
        #: 补齐 tick 间隔用的可打断等待（stop/急停都能立刻唤醒）
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._panic_thread: threading.Thread | None = None

        self._panic_check = panic_check
        self._on_event = on_event
        self._max_consecutive_crashes = max(1, max_consecutive_crashes)
        self._stop_timeout_s = stop_timeout_s

        self._consecutive_capture_failures = 0
        self._last_scene: str | None = None
        #: 收尾时用过哪些资源的记录，供诊断
        self.shutdown_report: dict[str, bool] = {}
        #: 当前这一 tick 有没有出过问题。内层 handler 处理完异常后
        #: _tick 是正常返回的，靠它区分「成功的 tick」与「出过错但没让
        #: 循环崩的 tick」—— 否则连续崩溃计数永远攒不满。
        self._tick_failed = False
        # ---- 业务事件状态（2026-10-08，给「事件流 -> YUI 说话」用）----
        #: 上一次报过的位置。None = 还没报过（第一帧只记不发）。
        self._biz_last_pos: tuple[float, float] | None = None
        #: 上一次报过的敌人数。<0 = 还没跑过实体检测（所以第一次不发）。
        self._biz_last_enemy_count: int = -1
        #: 是否已经报过到达（离开目标 6m 才解除，允许重新进入）。
        self._biz_arrived = False
        #: 上一次的目标坐标（换目标才报 PATHFIND_START）。
        self._biz_target: tuple[float, float] | None = None
        #: 上一次报过的卡住次数（次数**变化**才报 PATHFIND_STUCK）。
        self._biz_last_stuck: int = 0

    #: 位置变化超过这么多米才报一次 POSITION_CHANGED（米）。
    #: 15 的来由：5Hz 主循环下 15m ≈ 5 秒走一段，是「用户能感知到
    #: 位置变了」的下限；再小就变成逐帧刷屏。
    BIZ_POS_DELTA_M = 15.0

    #: 卡住检测（``self._consecutive_capture_failures``）达到这么多次就
    #: 报一次 PATHFIND_STUCK，并把它记住；之后**涨**到下一档再报，
    #: 免得同一个卡住被说十几遍。
    #:
    #: ★ 2026-10-12 修复：原来这里读的是
    #:   ``getattr(self.ctx.stats, "consecutive_capture_failures", 0)``，
    #:   而**那个字段在 ``LoopStats`` 上根本不存在** —— 真实计数器是本实例的
    #:   ``self._consecutive_capture_failures``（在 ``_perceive()`` 里维护，
    #:   截图失败 +1、成功归零，loop.py:576/598）。
    #:   ``getattr`` 的默认值让 ``n`` 恒为 0，于是
    #:     · ``n >= 3`` 永不成立 -> ``PATHFIND_STUCK`` 从未发出
    #:     · ``n == 0`` 恒成立，但 ``_biz_last_stuck`` 只在第一个分支里被抬高，
    #:       所以 ``ESCAPE_SUCCESS`` 也从未发出
    #:   即「卡住了」和「走出来了」两条 YUI 语音一直是死的。
    #:   已由 ``tests/unit/test_core_loop.py::test_check_stuck__*`` 覆盖。
    def _check_stuck(self, state_version: int = 0) -> None:
        # ⚠️ 读实例计数器，不要读 stats —— 见上方注释。
        n = int(self._consecutive_capture_failures or 0)
        if n >= 3 and n > self._biz_last_stuck:
            self._biz_last_stuck = n
            self._emit(Event.pathfind_stuck(n, 0.0, state_version),
                       dedupe_key="biz_stuck")
        elif n == 0:
            # ★ 恢复即播报「走出来了」—— 用户最想听的就是这句，
            #   而之前从来没有「ESCAPE_SUCCESS」这条路。
            if self._biz_last_stuck >= 3:
                self._emit(Event.escape_success(
                    self._biz_last_stuck, 0.0, state_version),
                    dedupe_key="biz_escape")
            self._biz_last_stuck = 0

    # ==================================================================
    # 状态
    # ==================================================================

    @property
    def state(self) -> RuntimeState:
        with self._state_lock:
            return self._state

    def _transition(self, dst: RuntimeState) -> None:
        with self._state_lock:
            check_transition(self._state, dst)
            src, self._state = self._state, dst
        logger.info("状态 %s -> %s", src.value, dst.value)

    # ==================================================================
    # 控制
    # ==================================================================

    def start(self) -> None:
        """启动。IDLE -> RUNNING，起后台线程。"""
        self._transition(RuntimeState.RUNNING)
        self._stop_event.clear()
        self._panic_event.clear()
        self.ctx.stats.started_ms = int(time.time() * 1000)
        self.ctx.stats.stopped_ms = 0
        self._consecutive_capture_failures = 0
        self.ctx.bus.start()
        self._start_panic_watcher()
        self._thread = threading.Thread(
            target=self._run, name="autopilot-core", daemon=True)
        self._thread.start()
        self._emit(
            Event(
                severity=Severity.INFO, category=EventCategory.SYSTEM,
                code="CORE_STARTED",
                message=(f"主循环已启动（目标 "
                         f"{self.ctx.tick_rate_hz:g} Hz ≈ "
                         f"{self.ctx.target_interval_s * 1000:.0f}ms/帧）"),
                context={"tick_rate_hz": self.ctx.tick_rate_hz},
                state_version=self.ctx.store.get().state_version,
            )
        )

    def pause(self, reason: str = "用户暂停") -> None:
        """暂停。RUNNING -> PAUSED。

        ★ 暂停**只停决策与执行，感知继续跑**。用户暂停时可能要切回桌面
          干别的，几分钟后恢复 —— 那时状态包得是新鲜的，否则会基于
          旧状态做决策。

        但**立刻松键**：正在执行的 tick 会在下一个检查点返回，
        而 executor 的 finally 里有 release_all。
        """
        self._transition(RuntimeState.PAUSED)
        self._wake.set()
        self._resume_event.clear()          # ★ 进入暂停：清掉恢复信号
        self._release_all("暂停")
        self._emit(
            Event(
                severity=Severity.INFO, category=EventCategory.SYSTEM,
                code="CORE_PAUSED",
                message=f"主循环已暂停（{reason}）—— 感知继续，执行已停",
                context={"reason": reason},
                state_version=self.ctx.store.get().state_version,
            )
        )

    def resume(self) -> None:
        """恢复。PAUSED -> RUNNING。"""
        self._transition(RuntimeState.RUNNING)
        self._wake.clear()
        self._resume_event.set()           # ★ 唤醒暂停中的工作线程
        self._emit(
            Event(
                severity=Severity.INFO, category=EventCategory.SYSTEM,
                code="CORE_RESUMED", message="主循环已恢复",
                context={}, state_version=self.ctx.store.get().state_version,
            )
        )

    def stop(self, reason: str = "用户停止") -> None:
        """优雅停止。-> STOPPING -> STOPPED。

        顺序（**刻意如此**）::

            1. 置停止信号 + 转 STOPPING（in-flight tick 会在下个检查点返回）
            2. release_all          —— 先松键，不等 tick
            3. join 线程             —— 等 in-flight tick 真正结束
            4. close_resources      —— 最后关总线/截图器

        第 2 步在第 3 步之前是刻意的：一个 tick 里可能正卡在 60 秒的
        MOVE_FORWARD 里，用户按了停止还干等 60 秒是不能接受的。
        先松键保证游戏那边立刻静止，再慢慢等线程收工。
        """
        if self.state in (RuntimeState.STOPPING, RuntimeState.STOPPED):
            return
        self._transition(RuntimeState.STOPPING)
        self._stop_event.set()
        self._wake.set()
        self._resume_event.set()           # 唤醒暂停中的工作线程
        self._release_all("停止")                       # ② 先松键

        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=self._stop_timeout_s)        # ③ 等 tick 结束
            if t.is_alive():
                # 线程还在 = 某个 tick 卡住了。**继续转 STOPPED 是错的**
                # （线程可能还在发按键），所以明确报告，不假装停了。
                self._emit(
                    Event(
                        severity=Severity.ERROR,
                        category=EventCategory.SYSTEM,
                        code="CORE_STOP_TIMEOUT",
                        message=(f"主循环线程 {self._stop_timeout_s:g}s 内"
                                 f"未退出（某个 tick 卡住了）。"
                                 f"按键已强制释放，但线程可能仍在运行。"),
                        context={"state": self._state.value},
                        state_version=self.ctx.store.get().state_version,
                    )
                )
                self._transition(RuntimeState.STOPPED)
                raise IllegalTransition(
                    f"停止超时：主循环线程 {self._stop_timeout_s:g}s 内未退出。"
                    f"通常是某个 tick 卡在 SendInput 或长 sleep 里。"
                )

        self._stop_panic_watcher()
        self.shutdown_report = self.ctx.close_resources()   # ④ 最后关
        self.ctx.stats.stopped_ms = int(time.time() * 1000)
        self._transition(RuntimeState.STOPPED)
        self._emit(
            Event(
                severity=Severity.INFO, category=EventCategory.SYSTEM,
                code="CORE_STOPPED",
                message=f"主循环已停止（{reason}）",
                context={"shutdown": self.shutdown_report},
                state_version=self.ctx.store.get().state_version,
            )
        )
        # 停止事件是最后一条，必须确保它也被投出去
        self.ctx.bus.drain_now()

    def reset(self) -> None:
        """STOPPED -> IDLE，允许重新 start()。"""
        self._transition(RuntimeState.IDLE)
        from .stats import LoopStats  # noqa: PLC0415

        self.ctx.stats = LoopStats()
        self._last_scene = None
        self._consecutive_capture_failures = 0
        self._panic_event.clear()
        self._stop_event.clear()

    def __enter__(self) -> "CoreLoop":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.state.alive:
            self.stop("上下文退出")

    # ==================================================================
    # 主循环
    # ==================================================================

    def _run(self) -> None:
        """工作线程主体。

        ★ 循环条件是 ``state is not STOPPED`` 而不是 ``is RUNNING`` ——
          暂停时**线程必须活着**（只是不执行）。早先写成
          ``while state is RUNNING`` 时，一暂停线程就退出了，
          resume() 之后没有任何东西在跑。

        ★ 外层 finally 里的 release_all 是**最后一道保险**：
          线程无论怎么退出（异常、正常、STOPPING），都保证不留下
          卡住的按键。W 被按住不放 = 角色一直往前跑。
        """
        try:
            while self.state is not RuntimeState.STOPPED:
                # ★ 这里**不能**对 PAUSED 提前 continue。
                #   早先写成 `if state is PAUSED: wait(0.1); continue`，
                #   结果整条路径跳过了 _tick()，感知根本不跑 ——
                #   与「暂停只停执行，不停感知」的约定正好相反。
                #   RUNNING / PAUSED 都交给 _tick()，由它在感知之后
                #   判断要不要进入决策。
                if self.state not in (RuntimeState.RUNNING,
                                      RuntimeState.PAUSED):
                    break
                t0 = time.perf_counter()
                self._tick_guarded()
                # ★ 传 float，不要 int() —— 整 tick 快过 1ms 时截断成 0，
                #   统计里就会出现 avg_tick_ms=0.0 这种骗人的数字。
                self.ctx.stats.record_tick((time.perf_counter() - t0) * 1000)
                if self.state is RuntimeState.RUNNING:
                    self._sleep_to_target(t0)
                else:
                    # 暂停态：等 resume / stop。★ 等的是 _resume_event，
                    # 不是 _wake（后者在 pause() 里被置位，会立刻返回）。
                    self._resume_event.wait(0.05)
        finally:
            # ★ 不管怎么退出都要松键
            self._release_all("工作线程退出")
            logger.debug("工作线程退出，最终状态 %s", self.state.value)

    def _tick_guarded(self) -> None:
        """_tick 的异常护栏。

        单个 tick 崩溃不该毁掉整个循环 —— 但也**不能静默**：
        记事件 + 计数 + 连续 N 次后主动暂停。连续崩说明是系统性问题
        （配置错、驱动挂），再崩下去只是刷屏。
        """
        try:
            self._tick_failed = False
            self._tick()
            # ★ 只有「这一 tick 完全没出过问题」才清连续崩溃计数。
            #   内层 handler 处理完异常后 _tick 是**正常返回**的 ——
            #   早先无条件 clear_crashes() 会把刚记上的崩溃立刻抹掉，
            #   连续崩溃阈值永远攒不满，那条保护形同虚设。
            if not self._tick_failed:
                self.ctx.stats.clear_crashes()
        except Exception as exc:  # noqa: BLE001 - 循环绝不能因单帧崩掉
            self._note_crash(
                f"tick 抛出未预期异常：{type(exc).__name__}: {exc}",
                "TICK_CRASH")

    def _note_crash(self, message: str, code: str) -> None:
        """记一次崩溃。

        ★ 内层的 handler（builder / policy / executor）**也必须走这里**，
          不能只发事件了事。早先它们就地 try/except 掉，
          `_tick_guarded` 永远看不到异常，于是 tick_crashes 恒为 0、
          「连续 N 次崩溃后暂停」这条保护形同虚设 —— 感知层一直崩
          循环就一直跑下去。
        """
        self._tick_failed = True
        self.ctx.stats.record_crash()
        self._emit(
            Event(
                severity=Severity.ERROR, category=EventCategory.SYSTEM,
                code=code, message=message,
                context={"consecutive": self.ctx.stats.consecutive_crashes,
                         "max": self._max_consecutive_crashes},
                state_version=self.ctx.store.get().state_version,
            ),
            dedupe_key=f"{code}:{message[:60]}",
        )
        logger.error("%s（第 %d 次连续）", message,
                     self.ctx.stats.consecutive_crashes)
        # 崩溃可能发生在"已经按下去了"之后
        self._release_all(f"{code} 崩溃")
        if self.ctx.stats.consecutive_crashes >= \
                self._max_consecutive_crashes:
            self._auto_pause(
                f"连续 {self.ctx.stats.consecutive_crashes} 次异常（{code}）")

    def _sleep_to_target(self, tick_start: float) -> None:
        """补齐到目标间隔。**用 Event.wait 而非 sleep** ——
        pause/stop/急停能立刻唤醒，不用干等一个睡满的周期。
        """
        interval = self.ctx.target_interval_s
        if interval <= 0:
            # 不限速也要给急停一个检查点，否则满速跑会饿死监视线程
            self._wake.wait(PANIC_POLL_S)
            return
        remain = interval - (time.perf_counter() - tick_start)
        if remain > 0:
            self._wake.wait(min(remain, 0.5))
        else:
            # 本 tick 已超时，仍要让出时间片并给急停检查点
            self._wake.wait(PANIC_POLL_S)

    def _tick(self) -> None:
        """一次完整循环。

        PAUSED 与 RUNNING 的差别**只在这一层**：感知都跑，
        PAUSED 时跳过决策与执行。
        """
        if not self._can_act("tick 开头"):
            return
        paused = self.state is RuntimeState.PAUSED

        # ── ① 感知 ──────────────────────────────────────────────
        state = self._perceive()
        if state is None:
            return
        self.ctx.stats.states_built += 1

        if not self._can_act("感知后"):
            return

        # ── PAUSED 分支：到此为止 ────────────────────────────────
        if paused or self.state is not RuntimeState.RUNNING:
            self.ctx.stats.record_paused_perception()
            logger.debug(
                "tick#%d [PAUSED] sv=%d scene=%s —— 只感知，不决策",
                self.ctx.stats.ticks, state.state_version, state.scene.id)
            self._notify_tick(state, None)
            return

        # 场景切换 -> info 级（关键事件，不是每帧都打）
        if state.scene.id != self._last_scene:
            if self._last_scene is not None:
                self._emit(
                    Event(
                        severity=Severity.INFO,
                        category=EventCategory.DECISION,
                        code="SCENE_CHANGED",
                        message=f"场景切换 {self._last_scene} -> {state.scene.id}",
                        context={"from": self._last_scene,
                                 "to": state.scene.id},
                        state_version=state.state_version,
                    )
                )
            self._last_scene = state.scene.id

        # ══════════════════════════════════════════════════════════════
        # ★ 业务事件补发（2026-10-08）—— 给「事件流 -> YUI 说话」用
        # ────────────────────────────────────────────────────────────
        #   放在**感知之后、决策之前**：这几个事件描述的是「世界现在怎样」
        #   （我在哪、周围有什么），与「决定做什么」无关。
        #   ★ 阈值节流是刚需：5Hz 主循环下每帧都发 POSITION_CHANGED，
        #     5 秒就是 25 条同义事件，YUI 会被淹掉。
        self._emit_business_events(state)
        self._check_stuck(state.state_version)

        # ── ② 决策 ──────────────────────────────────────────────
        # ★ 传的是**世界状态**（含历史），不是裸 StatePacket ——
        #   决策层要用 count() / last_exec() 判断"待了多久""上次何时执行"。
        try:
            # ★ 世界实体（F2 方案：结果在 builder 上，不进 StatePacket）。
            #   这里把 builder 的侧信道搬进 policy，poll 签名不变。
            #   ⚠️ 必须**保留 None**：None = 没跑（→ UNKNOWN，规则不触发），
            #      空列表 = 跑了但没有（→ False）。压平会丢掉三态语义。
            _ents = getattr(self.ctx.builder, "last_world_entities", None)
            self.ctx.policy.last_world_entities = (
                None if _ents is None else tuple(_ents))
            local = self.ctx.policy.poll(self.ctx.world().current_state)
        except Exception as exc:  # noqa: BLE001
            self.ctx.stats.policy_errors += 1
            self._note_crash(
                f"决策层异常：{type(exc).__name__}: {exc}",
                "POLICY_POLL_CRASH")
            return None

        # ── ③ 合并（本期云端恒为空，接口先立住）─────────────────
        intent = self._merge(local, state)
        if intent is None:
            # ★★ 本轮无意图 -> **必须松掉长按中的键**（§38）。
            #   ``executor.execute()`` 只在**有意图**时被调用，而
            #   ``NORMAL_ATTACK`` 走的是 ``mouse_down``（长按语义，
            #   故意不在本次执行内松手）。于是：敌人死亡 -> 规则不再命中
            #   -> 没有任何 execute -> 左键一直按住，直到用户暂停/停止/
            #   按 F12。游戏里的表现是「角色一直平A」，看起来还像规则
            #   在生效 —— 是最难自己发现的一类 bug。
            #
            #   放在**合并之后、record_idle 之前**：此刻已经确定本轮
            #   不会有任何按键动作，长按的键就是无主的状态。
            #   ``release_held()`` 幂等 —— 空转时没有键按着就是空操作，
            #   所以每帧调它没有代价。
            #
            #   ⚠️ 这里**故意不用 getattr 兜底**（同文件 ``_release_all``
            #   那种写法）：执行器缺这个方法是**接口缺失**，应该立刻炸
            #   （经 ``_tick_guarded`` -> ``_note_crash`` -> 自动暂停，
            #   并发出 TICK_CRASH 事件），而不是静默留下一个卡住的键。
            #   静默走错比响亮失败危险得多（原则 1）。
            self.ctx.executor.release_held()
            self.ctx.stats.record_idle()
            logger.debug("tick#%d sv=%d scene=%s action=-（空转）",
                         self.ctx.stats.ticks, state.state_version,
                         state.scene.id)
            self._notify_tick(state, None)
            return

        self.ctx.stats.record_hit(state.scene.id)

        if not self._can_act("决策后"):
            return

        # ★ PATHFIND_START（2026-10-08 业务事件）：只在**换了目标**时报。
        #   每 tick 都报的话 5 秒 25 条，YUI 会被淹。
        # ★ 平面轴名统一为 x / y（2026-10-14）。``PositionInfo.y`` 存的是
        #   **世界 z**（见 perception/position_mapping.py 的 ``y=wz``）。
        _pz = getattr(state, "position", None)
        _ty_ = getattr(state, "target", None)
        if _pz is not None and _ty_ is not None:
            _k = (round(float(_ty_.x), 1), round(float(_ty_.y), 1)) \
                if getattr(_ty_, "x", None) is not None else None
            if _k is not None and _k != self._biz_target:
                _first = self._biz_target is None
                self._biz_target = _k
                self._biz_arrived = False
                if not _first:
                    self._emit(
                        Event.pathfind_start(_k[0], _k[1], math.hypot(
                            float(_pz.x) - _k[0], float(_pz.y) - _k[1]),
                            state.state_version),
                        dedupe_key="biz_pfstart")

        # ── ④ 执行（校验在 executor 内部，见模块 docstring）──────
        self._execute(intent, state)

    # ------------------------------------------------------------------
    # ① 感知
    # ------------------------------------------------------------------
    def _perceive(self) -> StatePacket | None:
        try:
            frame = self.ctx.capturer.grab()
        except Exception as exc:  # noqa: BLE001 - 截图器抛什么都算失败
            self.ctx.stats.capture_failures += 1
            self._consecutive_capture_failures += 1
            self._emit(
                Event(
                    severity=Severity.WARN, category=EventCategory.PERCEPTION,
                    code="CAPTURE_FAILED",
                    message=f"截图失败：{exc}",
                    context={
                        "consecutive": self._consecutive_capture_failures,
                        "max": self.ctx.max_consecutive_capture_failures,
                    },
                ),
                dedupe_key="CAPTURE_FAILED",
            )
            # 连续失败到上限 -> 自动暂停。**不是继续无限重试**：
            # 游戏窗口被关掉时那只会刷屏，而且用户以为程序还活着。
            if self._consecutive_capture_failures >= \
                    self.ctx.max_consecutive_capture_failures:
                self._auto_pause(
                    f"连续 {self._consecutive_capture_failures} 次截图失败"
                    f"（游戏窗口可能已关闭）")
            return None

        self._consecutive_capture_failures = 0

        try:
            state = self.ctx.builder.build(frame)
        except Exception as exc:  # noqa: BLE001
            self.ctx.stats.perception_failures += 1
            self._note_crash(
                f"感知构建崩溃：{type(exc).__name__}: {exc}",
                "PERCEPTION_BUILD_CRASH")
            return None

        # 感知层自己的事件汇入总线（**不吞掉**）
        for ev in self._drain_builder_events():
            self.ctx.bus.publish(ev)

        # 感知层明确说"我这帧不可信" -> 立即暂停，不带着脏状态往下走
        if not state.resource.perception_ok:
            self.ctx.stats.perception_failures += 1
            self._emit(
                Event(
                    severity=Severity.ERROR,
                    category=EventCategory.PERCEPTION,
                    code="PERCEPTION_DOWN",
                    message="感知层判定本帧不可信（PERCEPTION_DOWN），已自动暂停",
                    context={"scene": state.scene.id},
                    state_version=state.state_version,
                ),
                dedupe_key="PERCEPTION_DOWN",
            )
            self._auto_pause("感知层不健康")
            return None

        # ★ 必须入 store：决策层的 count() / last_exec() 依赖历史帧，
        #   不入 store 的话它们永远拿到空历史（UNKNOWN）。
        self.ctx.store.update(state)
        return state

    def _drain_builder_events(self) -> list[Event]:
        drain = getattr(self.ctx.builder, "drain_events", None)
        if drain is None:
            return []
        try:
            return list(drain())
        except Exception:  # noqa: BLE001 - 事件出口坏了不该拖垮感知
            logger.exception("drain_events 失败")
            return []

    # ------------------------------------------------------------------
    # ③ 合并
    # ------------------------------------------------------------------
    def _merge(self, local: IntentPacket | None,
               state: StatePacket) -> IntentPacket | None:
        """本地 + 云端 -> 一个意图。

        本期云端恒为 None（**不实现云端是有意的**：一个连不上的云端
        留在代码里只会让人以为有回退路径）。合并逻辑仍走 merger，
        接口先立住，将来接上不用改主循环。
        """
        try:
            from ..policy.merger import merge_intents  # noqa: PLC0415
        except ImportError:
            return local
        result = merge_intents(local, None, state)
        if result.cloud_rejected:
            self._emit(
                Event(
                    severity=Severity.WARN, category=EventCategory.DECISION,
                    code="CLOUD_INTENT_REJECTED",
                    message=result.cloud_rejected,
                    context={"reason": result.reason},
                    state_version=state.state_version,
                )
            )
        return result.intent

    # ------------------------------------------------------------------
    # ④ 执行
    # ------------------------------------------------------------------
    def _execute(self, intent: IntentPacket, state: StatePacket) -> None:
        # ★ 校验不在这里做 —— executor 内部已经调了
        #   validator.validate(..., commit=True)。重复调会二次提交
        #   intent_id 去重记录，导致下一条意图被误判为"重放"。
        self._emit(
            Event(
                severity=Severity.INFO, category=EventCategory.EXECUTION,
                code="INTENT_EXECUTED",
                message=(f"{intent.action.value} <- {intent.source} "
                         f"(p{intent.priority}, sv{intent.state_version})"),
                context={"action": intent.action.value,
                         "source": intent.source,
                         "intent_id": intent.intent_id},
                state_version=state.state_version,
            )
        )
        logger.debug("tick#%d sv=%d scene=%s action=%s",
                     self.ctx.stats.ticks, state.state_version,
                     state.scene.id, intent.action.value)

        try:
            res = self.ctx.executor.execute(intent, state)
        except Exception as exc:  # noqa: BLE001 - executor 理论上不外抛
            self.ctx.stats.record_intent(intent.action.value, "FAILED")
            self._note_crash(
                f"执行器异常（{intent.action.value}）："
                f"{type(exc).__name__}: {exc}",
                "EXECUTE_CRASH")
            return

        outcome = res.outcome.value
        self.ctx.stats.record_intent(intent.action.value, outcome)

        if res.event is not None:
            self.ctx.bus.publish(res.event)

        if res.ok:
            logger.info("✓ %s (%.0fms) %s", intent.action.value,
                        res.elapsed_ms, "; ".join(res.steps))
        else:
            # 非 OK 是**需要知道**的事，但不算故障（REJECTED 很常见）
            logger.info("✗ %s -> %s: %s", intent.action.value, outcome,
                        res.reason)
            self._emit(
                Event(
                    severity=Severity.WARN if outcome == "REJECTED"
                    else Severity.ERROR,
                    category=EventCategory.EXECUTION,
                    code=f"INTENT_{outcome}",
                    message=f"{intent.action.value} {outcome}: {res.reason}",
                    context={"steps": res.steps,
                             "elapsed_ms": res.elapsed_ms},
                    state_version=state.state_version,
                ),
                dedupe_key=f"INTENT_{outcome}:{intent.action.value}",
            )

        self._notify_tick(state, intent)

    # ==================================================================
    # 急停
    # ==================================================================

    def _start_panic_watcher(self) -> None:
        if self._panic_check is None:
            return
        self._panic_thread = threading.Thread(
            target=self._watch_panic, name="autopilot-panic", daemon=True)
        self._panic_thread.start()

    def _stop_panic_watcher(self) -> None:
        # 监视线程靠 sleep 自然退出（最多多跑 20ms），它不持任何锁，
        # 所以这里不需要 join —— join 反而可能在状态已变时卡住。
        self._panic_thread = None

    def _watch_panic(self) -> None:
        """独立线程轮询急停键，命中后**置位 Event**。

        ★ 必须独立于 tick —— 一个 tick 可能长达 60s
          （MOVE_FORWARD 的 traits 上限），把急停检查放在 tick 里
          就意味着"按了急停但最多等 60 秒"。

        这里用 GetAsyncKeyState 轮询（Win32 没有按键事件接口），
        但轮询只负责**置位 Event**：后续全是 Event 语义，
        跨线程读写不靠裸 flag。
        """
        while self.state in (RuntimeState.RUNNING, RuntimeState.PAUSED) \
                and not self._stop_event.is_set():
            try:
                if self._panic_check():  # type: ignore[misc]
                    self._panic_event.set()
                    self._on_panic()
                    return
            except Exception:  # noqa: BLE001
                logger.exception("急停检查异常")
            time.sleep(PANIC_POLL_S)

    def _on_panic(self) -> None:
        """急停：**先松键，再改状态**。

        顺序不能反：改状态要走状态机校验，万一抛异常，玩家那边
        角色还按着 W 往前跑。

        ★ 两条路径都会调它（tick 线程 + 监视线程），所以必须**幂等** ——
          并发调用时第二次会看到状态已不是 RUNNING/PAUSED，直接返回。
        """
        if self.state not in (RuntimeState.RUNNING, RuntimeState.PAUSED):
            return
        logger.warning("F12 急停触发")
        self._panic_event.set()
        self._stop_event.set()
        self._wake.set()
        self._release_all("F12 急停")
        self._emit(
            Event(
                severity=Severity.ERROR, category=EventCategory.SYSTEM,
                code="PANIC_KEY",
                message="检测到急停键（F12），已释放全部按键并停止循环",
                context={},
                state_version=self.ctx.store.get().state_version,
            )
        )
        try:
            self._transition(RuntimeState.STOPPING)
            self._transition(RuntimeState.STOPPED)
        except IllegalTransition as exc:
            # 并发下另一线程可能已经改过状态；已停机就不是故障
            logger.debug("急停时状态转移被跳过：%s", exc)

    # ==================================================================
    # 辅助
    # ==================================================================

    def _can_act(self, where: str) -> bool:
        """每一步之间都要问：还能动吗？

        ★ 这里发现急停时**必须走 `_on_panic()`**，不能只置一个标志位。
          早先只置 `_panic_pressed = True`，结果监视线程的守卫
          ``if pressed and not self._panic_pressed`` 判定为「已处理」，
          于是**急停永远不触发停机** —— tick 线程把监视线程的活干了，
          却没干完。

        ★★ 急停检查抛异常时**不放行**（fail-closed）
          急停检查失败 = 「我们不知道用户有没有按 F12」。选择放行的话，
          一旦它开始抛异常，引擎就会在用户可能已经按下急停的情况下
          继续发按键 —— 而那正是急停这个功能存在的理由被推翻的时刻。

          代价是「`_panic_check` 一直坏 -> 引擎什么都不做」。这个代价
          是可以接受的：**停着总比乱动好**，而且 `logger.exception`
          会把病因完整留在日志里。

          （姊妹代码 :785 的监视线程是 fail-open 的：那里异常后只是
          「本轮不检查、下一轮继续」，不影响是否执行动作 —— 两处
          语义不同，所以行为不同是对的。）
        """
        if self.state not in (RuntimeState.RUNNING, RuntimeState.PAUSED):
            logger.debug("tick 在「%s」处因状态 %s 中止", where, self.state.value)
            return False
        if self._panic_event.is_set():
            return False
        if self._stop_event.is_set():
            return False
        check = self._panic_check
        if check is not None and not self._panic_event.is_set():
            try:
                if check():
                    logger.info("急停在「%s」处被 tick 线程发现", where)
                    self._on_panic()
                    return False
            except Exception:  # noqa: BLE001
                # ★ fail-closed + 留痕（2026-10-13 修）。
                #   原来是 ``pass`` + ``return True`` —— 零日志地放行本轮动作。
                #   同函数姊妹代码 :785 早就用了 logger.exception，说明这是
                #   遗留而不是设计。
                logger.exception(
                    "急停检查在「%s」处异常 —— 本轮**不放行**"
                    "（不知道用户有没有按 F12，就当按了）", where)
                return False
        return True

    def _release_all(self, why: str) -> None:
        """无条件松键。**和 executor 用的是同一个 release_all**，
        不另写一套 —— 两套松键逻辑迟早会不一致。"""
        sim = getattr(self.ctx.executor, "sim", None)
        release = getattr(sim, "release_all", None)
        if release is None:
            return
        try:
            released = release()
        except Exception:  # noqa: BLE001
            logger.exception("release_all 失败（%s）", why)
            return
        if released:
            logger.info("已释放 %d 个按键（%s）", len(released), why)

    def _auto_pause(self, reason: str) -> None:
        """关键失败自动暂停。**不静默继续**。"""
        if self.state is RuntimeState.RUNNING:
            try:
                self.pause(reason)
            except IllegalTransition as exc:  # pragma: no cover
                logger.error("自动暂停失败：%s", exc)

    def _emit(self, ev: Event, *, dedupe_key: str = "") -> None:
        """投递事件到总线（异步，绝不阻塞主循环）。"""
        self.ctx.bus.publish(ev, dedupe_key=dedupe_key)
        self._forward(ev)

    # ==================================================================
    # 业务事件（2026-10-08）
    # ==================================================================
    # ★ 为什么单独一个方法而不是就地 publish
    #   ----------------------------------
    #   补发点散落在 tick 各处（位置每帧、敌人每帧、卡住时、到达时），
    #   而「发不发」全靠阈值节流。一处一个 if 会让阈值散落各处、迟早
    #   有人忘 —— 收在一处，节流规则才看得见、改得动。

    def _emit_business_events(self, state: StatePacket) -> None:
        """把「游戏里正在发生什么」翻译成结构化事件发上总线。"""
        sv = state.state_version

        # ---- 位置：只报**有意义的位移**，不是每帧都报 ----------------
        p = getattr(state, "position", None)
        pos = None
        quality = ""
        if p is not None:
            q = getattr(p, "quality", "") or ""
            px = getattr(p, "x", None)
            # ★ 2026-10-14：``.z`` -> ``.y``。``PositionInfo`` 从来没有 ``z``
            #   字段，所以这行以前**恒为 None**，下面的分支**从未进入过**。
            #   坐标轴现在统一是 x / y（y 存的是世界 z）。
            py = getattr(p, "y", None)
            # ⚠️ 但这一段**仍然不发射**，原因在下一行的 quality 判据：
            #   ``PositionInfo.quality`` 的取值是 GOOD / DEGRADED / LOST
            #   （见 perception/position_mapping.py::_quality_table），
            #   **从来不是 "TRACKING"**（那是 LocalizationStateMachine
            #   的状态名，不是 quality）。所以 POSITION_CHANGED 与
            #   PATHFIND_ARRIVED 至今没有从引擎发出过。
            #   ⚠️ 本轮**只**统一轴名，不改这个判据 —— 改它等于让一批
            #   十年没发过的业务事件突然开始发，那是独立的一次决策。
            if (px is not None and py is not None
                    and (not q or q.upper() == "TRACKING")):
                pos = (float(px), float(py))
                quality = q
        if pos is not None:
            if self._biz_last_pos is None:
                self._biz_last_pos = pos
            else:
                d = math.hypot(pos[0] - self._biz_last_pos[0],
                               pos[1] - self._biz_last_pos[1])
                if d >= self.BIZ_POS_DELTA_M:
                    self._biz_last_pos = pos
                    _tgt = getattr(state, "target", None)
                    _dist = 0.0
                    if _tgt is not None:
                        _tx = getattr(_tgt, "x", None)
                        _ty = getattr(_tgt, "y", None)
                        if _tx is not None and _ty is not None:
                            _dist = math.hypot(pos[0] - float(_tx),
                                               pos[1] - float(_ty))
                    self._emit(
                        Event.position_changed(pos[0], pos[1], _dist, sv),
                        dedupe_key="biz_pos")

        # ---- 敌人：数量**变化**时才报（0->3 值得说，3->3 不用）-------
        _ents = getattr(self.ctx.builder, "last_world_entities", None)
        if _ents is not None:               # None = 没跑（不算 False）
            _c, _bad_bbox = count_enemy_entities(_ents)
            # ★ 缺失不是「当作不成立」，是「不知道」——
            #   静默按 1.0 处理会让这批实体整批消失在计数里，
            #   而现象只是 YUI 少说了一句话，排查时毫无线索。
            if _bad_bbox:
                logger.warning(
                    "敌人数统计：%d 个实体缺 bbox/坐标，按「不可判读」处理",
                    _bad_bbox)
            if _c != self._biz_last_enemy_count:
                _first = self._biz_last_enemy_count < 0
                self._biz_last_enemy_count = _c
                if not _first:
                    self._emit(Event.enemy_detected(_c, -1.0, sv),
                               dedupe_key="biz_enemy")

        # ---- 到达：位置进入到达半径（这里不读 arrive 参数，交给调用方）---
        #   到达判定属于「决策结果」，由 policy 产生意图；
        #   这里只在**位置不再变化**且贴着目标时补一条，走路走完也听得见。
        if pos is not None and quality.upper() == "TRACKING":
            _t = getattr(state, "target", None)
            if _t is not None:
                _tx = getattr(_t, "x", None)
                _ty = getattr(_t, "y", None)
                if _tx is not None and _ty is not None:
                    _dd = math.hypot(pos[0] - float(_tx), pos[1] - float(_ty))
                    if _dd <= 3.0 and not self._biz_arrived:
                        self._biz_arrived = True
                        self._emit(Event.pathfind_arrived(
                            pos[0], pos[1], 0.0, sv), dedupe_key="biz_arrive")
                    elif _dd > 6.0:
                        self._biz_arrived = False

    def _notify_tick(self, state: StatePacket | None,
                     intent: IntentPacket | None) -> None:
        """每 tick 回调。调试叠加层 / 监控用它画画面。

        失败**不能影响主循环** —— 一个画图异常不该让机器人停下。
        """
        if self.on_tick is None:
            return
        try:
            self.on_tick(state, intent)
        except Exception:  # noqa: BLE001
            logger.exception("on_tick 回调异常")

    def _forward(self, ev: Event) -> None:
        """把事件交给上层（插件总线 / 通知）。失败不影响主循环。"""
        if self._on_event is None:
            return
        try:
            self._on_event(ev)
        except Exception:  # noqa: BLE001
            logger.exception("事件回调异常")

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return self.ctx.stats.snapshot()


__all__ = ["DEFAULT_MAX_CONSECUTIVE_CRASHES", "CoreLoop", "PANIC_POLL_S"]
