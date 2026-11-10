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
# 物理输入让权（2026-10-17）：独立后台线程轮询 GetAsyncKeyState，
# 检测到用户按键就松键 + 让权。2026-10-19 起让权改为**动态**：
# 用户还按着就一直让权，松手后 1 秒内也让权（见 TakeoverState）。
from ..actuator.physical_input import (TAKEOVER_IDLE_S, TakeoverState,
                                       PhysicalInputWatcher)
# 松键（2026-10-19）：让权时**必须**跳过用户正按着的键 ——
# 旧代码无条件 release_all()，把用户刚按下的 E/W 一起弹起来，
# 用户的长按/走位被引擎当场掐断。release_all_compat 兼容旧替身。
from ..actuator.simulator import _vk_by_name, release_all_compat
# 敌人判据**只用这一份**。2026-10-15 之前，本文件曾用 `kind` + `.y`
# 自己抄了一份，而 WorldEntity 根本没有 `.y` 字段，`getattr` 兜底成 1.0
# 让条件恒假 —— `health_bar` 永远计不进 ENEMY_DETECTED，规则却照常开火。
# 详见 policy/context.py::is_enemy_evidence 的长注释。
from ..policy.combo_manager import COMBO_INTENT_PRIORITY as _COMBO_PRIORITY
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

        # ── 物理输入让权（2026-10-17）─────────────────────────────
        #: 监听物理按键的后台线程。**独立线程** —— 主循环 5Hz，
        #:   一个 tick 要做感知+决策，塞进按键轮询只会拖慢它。
        self._input_watcher: PhysicalInputWatcher | None = None
        #: 让权状态。**动态**的 —— 用户还按着就一直让权，松手后
        #: 再宽限 :attr:`_takeover_idle_s` 秒。
        #: ★ 用 monotonic 而不是 time.time()：系统时间被改（对时、
        #:   夏令时）会让窗口算错。
        self._takeover: TakeoverState = TakeoverState(idle_s=TAKEOVER_IDLE_S)
        #: 让权的**空闲宽限**（秒）：用户最后一次动作之后还让多久。
        self._takeover_idle_s: float = TAKEOVER_IDLE_S
        #: 上一次「让权提示」的时刻（monotonic）。用来节流终端提示 ——
        #: 用户连点鼠标时逐条刷屏会盖掉真正的故障信息。
        self._last_takeover_notice_mono: float = 0.0

        # ★ 把「用户此刻正按着的键」接到模拟器上（2026-10-19）。
        #   有了它，让权时的 release_all 才会**跳过用户正按着的键** ——
        #   旧代码无条件把用户刚按下的 E/W 一起弹起来，用户的长按/走位
        #   被引擎**当场掐断**。
        #   ★ 这里接的是**方法**而不是快照：模拟器每次要用时才问，
        #     拿到的是「此刻」的状态，不是启动时的旧值。
        self._wire_physical_hold_provider()

        # ★ 默认「不接管」（2026-10-18 补）
        #   直接构造 CoreLoop 而不走 run_loop 的人（测试、插件内嵌）
        #   **不会**调 set_run_mode()。默认值必须朝着安全的方向：
        #   「没声明过的一律当作不动用户的键」。反过来默认 LIVE
        #   会让任何漏调用的调用方直接接管键鼠。
        #
        # ⚠️ 常量在 set_run_mode() 里**惰性** import，不在模块级 ——
        #   ui.schema 是协议层、core 是运行层，core 依赖 ui 等于把
        #   层次反过来（ui/schema.py 自己也写了同一条规矩）。
        self._run_mode: str = "DRY_RUN"

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

    @property
    def last_scene(self) -> str | None:
        """最近一次 tick 判出的场景 ID。``None`` = 还没跑出第一个 tick。

        ★ 为什么需要它（而不是只靠 ``SCENE_CHANGED`` 事件）
          ------------------------------------------------------
          事件只在**变化**时发，且**第一次赋值被刻意吞掉**（见 ``_tick``
          里那个 ``if self._last_scene is not None`` —— 不然开场就会
          多一句「场景切换 None -> WORLD」）。于是只订阅事件的消费者
          （``run_loop`` 的 UI 线程）在整个 run 都停在初始的 UNKNOWN：

              快照 scene="UNKNOWN"
              快照 raw_stats.by_scene={"WORLD": 355}    ← 自相矛盾

          真机 2026-10-14 整场都是这么显示的。事件流有它的用处
          （「什么时候变的」），但读「**现在**是什么」只能问这里。

        只读、无副作用；UI 线程每 200ms 读一次，
        ``_last_scene`` 是普通字符串属性赋值，读取本身是原子的。
        """
        return self._last_scene

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
        self._start_input_watcher()
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
            # ★ 监听线程也要停 —— 它是 daemon，但显式停掉更干净，
            #   免得引擎重启时旧线程还在监听（两个线程同时回调，
            #   让权窗口会被旧线程续期）。
            self._stop_input_watcher()
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

        # ══════════════════════════════════════════════════════════════
        # ★★ DRY_RUN 决策闸门（2026-10-19）
        # ────────────────────────────────────────────────────────────
        #   ``--dry-run`` 以前**只换了执行层的注入后端**（DryRunBackend），
        #   决策层照跑、连招照走、组合照发 —— 于是引擎明明一个键都没发，
        #   却像真的在打：203 条 INTENT_EXECUTED、10 条 INTENT_REJECTED、
        #   YUI 照着念战斗播报。用户在物理按键时看到的是「引擎在我打字的时候
        #   还在喊『我不打了』」。
        #
        #   闸门放在**业务事件之后、决策之前**，两个原因：
        #     · 业务事件（ENEMY_DETECTED / COMBAT_ENTERED / POSITION_CHANGED）
        #       描述的是「世界现在怎样」，卡片要显示、用户要看着 —— 必须继续发；
        #     · 决策层往后一步都不走，于是 INTENT_EXECUTED / INTENT_REJECTED /
        #       COMBO_STARTED / REFLEX_FIRED **一条都不会产生**，不是靠下游
        #       过滤，是压根没跑。
        #
        #   ⚠️ 统计口径用 ``record_paused_perception`` 而不是 ``record_idle``：
        #     干跑是「**永远**只感知不决策」，与 PAUSED 同类；用 record_idle
        #     会让 idle_streak 每 tick 无限增长，卡片上看着像引擎卡住了。
        if not self.is_live():
            self.ctx.executor.release_held()
            self.ctx.stats.record_paused_perception()
            logger.debug(
                "tick#%d [DRY_RUN] sv=%d scene=%s —— 只感知，不决策",
                self.ctx.stats.ticks, state.state_version, state.scene.id)
            self._notify_tick(state, None)
            return

        # ── ② 决策 ──────────────────────────────────────────────
        # ★ 用户物理输入让权（2026-10-17）
        #   位置很讲究：在**感知之后、决策之前**。
        #     · 感知照跑 —— 停了眼睛，恢复时看到的是陈旧画面
        #     · 决策不做 —— 用户在操作的这段时间里引擎**一个键都不发**
        #   和 PAUSED 分支的区别：PAUSED 是用户按的暂停按钮，
        #   让权是**用户按了游戏里的键** —— 引擎不该抢，但也不该
        #   把整个循环停掉（停了就没法自动恢复）。
        if self._user_takeover_active():
            self.ctx.stats.record_paused_perception()
            logger.debug(
                "tick#%d [让权] 用户在操作，引擎不动作", self.ctx.stats.ticks)
            self._notify_tick(state, None)
            return

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
            # ★ 异常也必须回报 —— 否则连招上挂着「等执行完成」的那一步
            #   会一直等到兜底超时才放行，白等 2 秒。
            self._notify_combo_done(intent, ok=False,
                                    error=f"{type(exc).__name__}: {exc}")
            self._note_crash(
                f"执行器异常（{intent.action.value}）："
                f"{type(exc).__name__}: {exc}",
                "EXECUTE_CRASH")
            return

        outcome = res.outcome.value
        self.ctx.stats.record_intent(intent.action.value, outcome)
        # ★ 执行完成回报（``wait = "ready"`` 的信号源）。
        #   ``IntentExecutor.execute`` 是**同步**的，返回即「按键按完、
        #   hold_ms 按够、gap_ms 等完」—— 这是唯一一个不靠估算的
        #   「这一步真的做完了」信号。
        #   没有人等它时它是空操作（返回 False），所以对规则意图
        #   和没有 wait 的步骤调用它毫无成本。
        #
        # ★★ ``reason`` 一定要一起传（2026-10-19）
        #   只传 outcome（"REJECTED"）的话，连招**分不出**为什么被拒：
        #   「被限流（MIN_GAP）」和「配置错了（KEYMAP）」长得一模一样。
        #   而这两种该做的处理完全相反 —— 前者该**重试**，后者该**中止**。
        self._notify_combo_done(intent, ok=bool(res.ok),
                                error="" if res.ok else outcome,
                                reason="" if res.ok else res.reason)

        if res.event is not None:
            self.ctx.bus.publish(res.event)

        if res.ok:
            logger.info("✓ %s (%.0fms) %s", intent.action.value,
                        res.elapsed_ms, "; ".join(res.steps))
        else:
            # 非 OK 是**需要知道**的事，但不算故障（REJECTED 很常见）
            logger.info("✗ %s -> %s: %s", intent.action.value, outcome,
                        res.reason)
            # ★ 记拒绝原因分类（2026-10-18）
            #   「为什么被拒」是用户排障的第一问。intents_rejected 只给
            #   总数不说原因 —— Console 的权限红牌就靠这个计数器。
            self.ctx.stats.record_rejection(res.reason)
            self._emit(
                Event(
                    severity=Severity.WARN if outcome == "REJECTED"
                    else Severity.ERROR,
                    category=EventCategory.EXECUTION,
                    code=f"INTENT_{outcome}",
                    message=f"{intent.action.value} {outcome}: {res.reason}",
                    # ★★ reason 进 context（2026-10-17）
                    #   拒绝原因是**结构化数据**，不是给人看的自由文本。
                    #   只塞在 message 里，读侧就得去解析字符串才能分清
                    #   「被限流」（MIN_GAP）和「被安全拦下」（场景不对/急停）——
                    #   而 YUI 播报层正需要这个区分（见 event_to_yui.py）。
                    #   message 保持原样，兼容在跑的旧读侧。
                    context={"steps": res.steps,
                             "elapsed_ms": res.elapsed_ms,
                             "reason": res.reason},
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
        # ★ 立刻中断连招（2026-10-18）。
        #   放在状态守卫**之后**：循环已经在停机的话没有「正在进行的连招」
        #   要清（abort 返回 False，幂等）。
        #   为什么还要显式清：连招状态挂在 policy 上，进程还会继续跑到
        #   reset。如果用户急停后没 reset 就又 start()，一条走了一半的
        #   连招会在新一场战斗里接着往下打 —— 听起来像是「AI 记得刚才」。
        self._abort_combo("F12 急停")
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

    def _abort_combo(self, reason: str) -> bool:
        """通知策略层中断正在进行的连招。

        **刻意用 getattr 而不是直接调 ``self.ctx.policy.abort_combo``**：
        ``RuntimeContext.policy`` 是可注入容器（测试塞假策略、
        ``DISABLED`` 模式塞 ``DisabledSource``），那些实现没有这个方法。
        没有 getattr 的话，每换一个 PolicySource 实现都要回来改这里，
        可替换性就漏了。**但也不能静默** —— 缺方法只说明「没接连招」，
        不是出错，所以只打 debug。

        :returns: 是否真的中断了一条连招（没连招 / 没在跑时 False）
        """
        fn = getattr(self.ctx.policy, "abort_combo", None)
        if fn is None:
            logger.debug("当前策略源没有连招（abort_combo 缺失），忽略")
            return False
        try:
            hit = bool(fn(reason))
        except Exception:  # noqa: BLE001 - 中断失败不该打断让权 / 急停
            logger.exception("中断连招失败：%s", reason)
            return False
        if hit:
            logger.info("连招已被中断：%s", reason)
        return hit

    def _notify_combo_done(self, intent, *, ok: bool, error: str,
                           reason: str = "") -> bool:
        """回报「上一个连招动作真的做完了」。

        与 :meth:`_abort_combo` 同一个理由用 ``getattr``：策略源是可注入
        的容器，``DisabledSource`` 和测试假策略都没有连招。缺方法只说明
        「没接连招」，不是出错。

        :param intent: 刚执行完的那个意图。用来确认它**确实来自连招** ——
            规则意图也会走到这里，回报给连招会放行一个它没在等的门。
        :returns: 连招确实在等这个回报。
        """
        if getattr(intent, "priority", None) != _COMBO_PRIORITY:
            return False
        fn = getattr(self.ctx.policy, "notify_combo_done", None)
        if fn is None:
            logger.debug("当前策略源没有连招（notify_combo_done 缺失），忽略")
            return False
        try:
            return bool(fn(ok=ok, error=error, reason=reason))
        except Exception:  # noqa: BLE001 - 回报失败不该打断这一 tick
            logger.exception("回报连招执行完成失败")
            return False

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

    # ==================================================================
    # 物理输入让权（2026-10-17）
    # ==================================================================

    def _wire_physical_hold_provider(self) -> None:
        """把「用户此刻正按着的键」接到模拟器上。

        ★ 为什么要接（2026-10-19）
          ``release_all()`` 遍历所有按下的键无条件发 KeyUp，
          而 ``GetAsyncKeyState`` **分不出**「引擎按的」和「用户按的」。
          于是用户按住 E 长按技能时，一次让权就把 E 弹起来 ——
          合成 KeyUp 到了游戏里，用户的长按**当场断掉**，
          表现是「长按出不来」「按 A/W 走不动」。

          模拟器拿到「用户按着哪些键」之后，就能：
          * 让权时**跳过**这些键（人的优先级最高）；
          * 但**引擎自己按着的键照旧全松**（异常安全底线不变）。

        ★ 失败不致命：接不上就退化成旧行为（整体松键），
          引擎照样能跑 —— 少一次保护不该换来一次崩溃。
        """
        sim = getattr(getattr(self.ctx, "executor", None), "sim", None)
        setter = getattr(sim, "set_physical_hold_provider", None)
        if setter is None:
            return
        try:
            setter(self._user_held_vks)
        except Exception:  # noqa: BLE001 - 接不上就退化成旧行为
            logger.debug("接「用户按着的键」失败", exc_info=True)

    def _user_held_vks(self) -> frozenset[int]:
        """用户**此刻正按着**的键（VK 码）。

        两个来源取并集：

        ① :meth:`PhysicalInputWatcher.held_physical_keys`
           —— 监听器轮询到的**原始**按键状态。
        ② :attr:`TakeoverState.held`
           —— 让权状态里**回声过滤后**确认是用户按的键。

        ★ 为什么要 ②
          没有真实监听器的环境（单测、离线仿真、非 Windows）里 ① 是空的，
          但「用户按了 E」这件事**确实发生过**（仿真里是直接调
          ``_on_physical_input("E")`` 模拟的）。② 记的就是它 ——
          于是仿真里也能验证「让权不碰用户的键」。

        ★ 两个来源都**分不出**「用户按的」和「引擎注入的」
          （``GetAsyncKeyState`` 两边都看得见）。调用方
          （:class:`autopilot.actuator.simulator.InputSimulator`）
          会把「引擎自己按着的键」减掉。
        """
        out: set[int] = set()
        watcher = self._input_watcher
        if watcher is not None:
            try:
                out |= {int(vk) for vk in watcher.held_physical_keys()}
            except Exception:  # noqa: BLE001 - 读不到就少一个来源
                logger.debug("读监听器的按着键失败", exc_info=True)
        for name in self._takeover.held:
            vk = _vk_by_name(name)
            if vk is not None:
                out.add(int(vk))
        return frozenset(out)

    def _start_input_watcher(self) -> None:
        """起物理输入监听线程。

        ★ 幂等，重复 start() 不会起第二个线程。
        ★ 拿不到 user32（非 Windows）时**静默跳过** ——
          监听是**增强**，不是必需功能，缺了不该让引擎起不来。
        """
        if self._input_watcher is not None and self._input_watcher.running:
            return
        try:
            watcher = PhysicalInputWatcher(
                self._on_physical_input,
                on_release=self._on_physical_release,
            )
            watcher.start()
        except Exception:  # noqa: BLE001
            logger.warning("物理输入监听起不来（不影响引擎运行）",
                           exc_info=True)
            return
        self._input_watcher = watcher
        # ★ 监听器起来后重接一次「用户按着的键」——
        #   来源 ①（监听器的原始状态）这时才可用。
        self._wire_physical_hold_provider()

    def _stop_input_watcher(self) -> None:
        watcher = self._input_watcher
        if watcher is None:
            return
        self._input_watcher = None
        try:
            watcher.stop()
        except Exception:  # noqa: BLE001
            logger.debug("停物理输入监听失败", exc_info=True)

    def _on_physical_input(self, key_name: str) -> None:
        """检测到用户物理按键 —— 立刻让权。

        顺序是刻意的：

        1. **先松键** —— 用户按的键和引擎按的键可能**是同一个**。
           不先松，用户的 E 会被引擎按着的 E 顶掉。
        2. **再进让权窗口** —— 光松一次不够：引擎下一帧（200ms 后）
           又会按下去，用户还是按不动。必须让它**停手一段时间**。
        3. **最后发事件** —— 让 YUI / Console 知道「刚才是你抢的」。

        ★ 这里跑在**监听线程**里，所以不能做重活。
          松键是一次 SendInput 批量，很快。

        🔴 2026-10-18：先问一句「这是不是引擎自己刚按的」
        ------------------------------------------------------------
        ``GetAsyncKeyState`` **看得到 SendInput 注入的按键**（实测：
        down 后 30 ms 读高位 = 1）。所以不加过滤时，引擎每按一次键，
        监听器就在 20 ms 内判定「用户在按」-> ``release_all()`` 把
        引擎**自己刚按下的键**释放掉 + 冻结 3 秒 + 执行层抛
        ``_UserTakeover``。引擎会把自己的输出掐掉。

        这不是推断：``sim.key_press(VK.HELP, hold_ms=30)``（USE_SKILL
        走的就是这条路径）在本机 100% 触发了监听器回调。
        ``InputSimulator.is_own_echo`` 负责判别，窗口
        ``simulator.ECHO_SUPPRESS_S`` = 120 ms。

        """
        # ★ 先过滤自己的回声。放在最前面是有意的：后面每一步都会
        #   松键 / 冻结 / 抛异常，一步都不该对「自己」做。
        sim = getattr(self.ctx.executor, "sim", None)
        own_echo = getattr(sim, "is_own_echo", None)
        if own_echo is not None:
            try:
                if own_echo(key_name):
                    logger.debug("忽略引擎自己的注入回声：%s", key_name)
                    return
            except Exception:  # noqa: BLE001 - 判别失败宁可放过，不拦用户
                logger.debug("回声判别异常，按「用户在按」处理", exc_info=True)

        # ★ 立刻中断连招（2026-10-18）。
        #   排在回声过滤**之后**、切人记账**之前**：走到这里已经确定是
        #   真人按键了，而「用户在操作」和「连招还在按」不能共存。
        #
        #   ⚠️ 这条是**必须**的而不是锦上添花：让权窗口内主循环**不再决策**
        #     （见 ``_tick`` 里 ``_user_takeover_active()`` 那个 early return），
        #     所以连招不会在这 3 秒里推进 —— 但它**也没被清掉**。
        #     用户让权结束时连招会从中断处继续，接着往下打他的队友。
        #     「用户抢了一把之后机器人接着自己的连招打」= 抢不动。
        #   放在这里而不是让权窗口里轮询，是因为**用户的第一次按键**
        #   就是最该让权的那一次 —— 晚一帧都是他在跟机器人抢。
        self._abort_combo(f"用户按了 {key_name}")

        # ★ 先记账「用户按了数字键 = 切人了」（2026-10-18）。
        #   必须排在回声过滤**之后**：引擎自己按的 1-4 已经在上面 return 了，
        #   走到这里的数字键才是真人按的。
        #   为什么需要：``PlayerInfo.active_member`` 恒 None 时
        #   ``onfield()`` 恒 UNKNOWN，按名字判的连招规则一条都触发不了。
        #   视觉判据不可用（见 perception/detectors/character_detector.py
        #   模块文档第 4 条），而**键盘是唯一无需标定就 100% 准的切人信号**。
        tracker = getattr(self.ctx, "active_slots", None)
        if tracker is not None:
            try:
                if tracker.note_physical_key(key_name):
                    logger.info("用户在切人：按了 %s（槽位 %s）",
                                key_name, tracker.current())
            except Exception:  # noqa: BLE001 - 记账失败不该打断让权
                logger.debug("切人记账异常", exc_info=True)

        self._takeover.note_press(key_name)
        # ★ 这次按下**还没松**（边沿检测只在「刚按下」时触发）——
        #   告诉让权状态「这个键正被按着」，它会一直让权到收到松开为止。
        self._takeover.note_hold(key_name)
        released = 0
        sim = getattr(self.ctx.executor, "sim", None)
        try:
            # ★ exclude_physical=True：**跳过用户正按着的键**（2026-10-19）。
            #   旧代码在这里无条件 release_all()，把用户**刚按下**的 E/W
            #   也一起弹起来 —— 游戏收到合成的 KeyUp，用户的长按/走位
            #   **当场被掐断**（「长按出不来」「走不动」）。
            #   引擎自己按着的键**照旧全松**（异常安全底线不变）。
            released = len(release_all_compat(sim, exclude_physical=True))
        except Exception:  # noqa: BLE001
            logger.exception("让权时 release_all 失败")
        # 安全闸也要知道 —— 它是发按键前的最后一道，
        # 知道「用户在操作」就不会在让权窗口里再放行任何意图。
        guard = getattr(self.ctx.executor, "guard", None)
        note = getattr(guard, "note_user_input", None)
        if note is not None:
            try:
                note(key_name, self._takeover_idle_s, still_held=True)
            except Exception:  # noqa: BLE001
                logger.exception("让权通知安全闸失败")
        logger.info("用户按下 %s：已松 %d 个键，让权（空闲宽限 %.1fs）",
                    key_name, released, self._takeover_idle_s)
        # ★★ 终端提示（2026-10-19）
        #   真机反馈：「用户频繁按鼠标触发让权，导致觉得引擎不敢动」。
        #   根因是**信息不对称**：引擎让了权，但用户只看到「它不动了」，
        #   于是以为引擎坏了/不敢动。
        #
        #   ★ 为什么必须写清楚「松开所有按键，等待 N 秒」
        #     新版让权是**动态**的：用户**还按着**键时引擎一直让权。
        #     只说「已让权」的话，用户会以为按一下就行 —— 于是他一直
        #     按着鼠标，引擎就一直不动，误会更深。
        #
        #   ★ 为什么要节流
        #     用户连点鼠标时每次都是一次抢占，逐条刷屏会盖掉真正的
        #     故障信息。3 秒内只提示一次：第一次**一定**会出现，
        #     之后同样的提示短时间内不重复。
        now_mono = time.monotonic()
        if now_mono - self._last_takeover_notice_mono >= 3.0:
            self._last_takeover_notice_mono = now_mono
            # ★ 措辞（2026-10-19 定稿）
            #   「松开按键 1 秒后恢复」比「请松开所有按键，等待 1 秒」更准：
            #   动态让权下**用户还按着**时引擎一直让权，必须让用户知道
            #   「松手 + 等 1 秒」就能拿回控制权，而不是「引擎坏了不动了」。
            logger.warning(
                "[提示] 你按下了 %s，引擎已让权。松开按键 %.0f 秒后恢复。",
                key_name, self._takeover_idle_s)
        self._emit(Event.user_interrupt(key_name, released,
                                        self._takeover_idle_s))

    def _on_physical_release(self, key_name: str) -> None:
        """检测到用户**松手** —— 让权状态少一条「还按着」的理由。

        ★ 没有这一条，引擎就不知道用户还在按着 W ——
          而「还在按」正是动态让权的第一条规则：用户按住走位时，
          引擎必须一直不抢，而不是等 1 秒后抢方向。

        ★ 回声过滤同样要做：引擎自己的长按（例如钟离长按 E 1.5 秒）
          也会产生一次「松开」边沿。不滤掉的话，引擎会把自己的
          长按当成「用户松手了」，凭空多出 1 秒让权 —— 自己掐自己。
          （长按的「松开」落在注入后 120ms 回声窗口内，因为
          ``_note_echo`` 在**每一次**注入上都重新登记窗口。）

        """
        sim = getattr(self.ctx.executor, "sim", None)
        own_echo = getattr(sim, "is_own_echo", None)
        if own_echo is not None:
            try:
                if own_echo(key_name):
                    logger.debug("忽略引擎自己的注入回声（松开）：%s", key_name)
                    return
            except Exception:  # noqa: BLE001 - 判别失败宁可放过，不拦用户
                logger.debug("回声判别异常，按「用户在松手」处理", exc_info=True)
        self._takeover.note_release(key_name)
        guard = getattr(self.ctx.executor, "guard", None)
        release_note = getattr(guard, "note_user_release", None)
        if release_note is not None:
            try:
                release_note(key_name)
            except Exception:  # noqa: BLE001
                logger.exception("通知安全闸「用户松手」失败")

    def _user_takeover_active(self) -> bool:
        """现在是不是在让权。**动态**判据。"""
        return self._takeover.active

    @property
    def user_takeover_active(self) -> bool:
        """让权窗口是否生效（给快照/Console 用）。"""
        return self._user_takeover_active()

    # ------------------------------------------------------------------
    # 接管状态（2026-10-18 补）
    # ------------------------------------------------------------------
    def run_mode(self) -> str:
        """执行模式（``"LIVE"`` / ``"DRY_RUN"``）。"""
        return self._run_mode

    def is_live(self) -> bool:
        """是不是真的会动用户的键。

        ★ 为什么不用 ``self._run_mode == "LIVE"`` 直接比字符串
          ``LIVE`` 是 ``ui.schema`` 的常量，而 ui 是协议层、core 是运行层，
          模块级 import 会把层次反过来（见 :meth:`set_run_mode` 的注释）。
          这里惰性 import 一次再比，层次不反转，也不用在模块级留一个
          硬编码的裸字符串（裸字符串迟早和常量分叉）。

        读侧遇到不认识的值一律当干跑 —— 宁可不动，也不动错。
        """
        from ..ui.schema import LIVE  # noqa: PLC0415

        return self._run_mode == LIVE

    def set_run_mode(self, live: bool) -> None:
        """声明本进程的执行模式。**由 run_loop 在启动时调用一次**。

        ⚠️ 为什么不从 ``sys.argv`` 反查
          engine exe 经 Nuitka 编译，argv 会被 self-execution 护栏改写；
          而且「我被怎么启动的」这件事在进程内已经不可靠 —— 只有
          argparse 解析出来的 ``args`` 才是真值，由 run_loop 传进来。
        """
        from ..ui.schema import DRY_RUN, LIVE  # noqa: PLC0415

        self._run_mode = LIVE if live else DRY_RUN
        logger.info("执行模式：%s", self._run_mode)


    def runtime_info(self) -> tuple[str, bool]:
        """``(run_mode, user_takeover)`` —— 写快照用的一次性读取。

        ★ 为什么一次返回两个而不是让调用方分两次读
          两次读之间状态可能已经变了（让权窗口刚好到期），于是快照会
          记下「不在让权中」而 UI 刚读到 True —— 那种不一致极难排查。
          一次取值保证两个字段来自同一时刻。
        """
        return self._run_mode, self._user_takeover_active()

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
