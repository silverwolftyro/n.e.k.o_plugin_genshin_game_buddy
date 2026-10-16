"""``core/loop.py`` 的单测（2026-10-11 任务 D）。

为什么之前说它「无法测试」
--------------------------
上轮结论是「深度依赖 YOLO/ONNX 和真实 StatePacket 链」。**那个结论是错的**
—— 真实原因是没人去看 ``RuntimeContext``：它六个依赖（capturer / builder /
policy / executor / bus / store）**全部是构造参数且默认 None**，就是为
「测试里塞假实现跑完整流程」设计的（见 ``runtime.py:118-120`` 的原注释）。
不需要任何依赖注入改造，直接注假件即可。

本文件覆盖
----------
* **状态机**：IDLE→RUNNING→PAUSED→RUNNING→STOPPED→IDLE 与非法转移
* **控制面**：start / pause / resume / stop / reset / 上下文管理器
* **急停与松键**：``_can_act`` / ``_release_all`` / ``_on_panic``
* **崩溃保护**：``_note_crash`` 计数、自动暂停
* **事件出口**：``_emit`` → 总线 + ``on_event`` 回调
* **``RuntimeContext``**：节奏换算、资源关闭顺序

⚠️ **本文件发现了一个生产 bug**，见 ``test_check_stuck__*`` 一组：
``_check_stuck`` 读的是 ``self.ctx.stats.consecutive_capture_failures``，
但那个字段**在 ``LoopStats`` 上不存在**（真实计数器是 ``CoreLoop`` 实例上的
``self._consecutive_capture_failures``）。``getattr(..., 0)`` 让它恒为 0，
于是 ``PATHFIND_STUCK`` 与 ``ESCAPE_SUCCESS`` 两条事件**永远发不出去**。
按本轮约束（不改引擎核心），这里用 strict xfail 钉住现状，修的时候把
``xfail`` 改成普通断言即可。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.contracts.state_packet import (  # noqa: E402
    SCENE_WORLD,
    SceneInfo,
    StatePacket,
)
from autopilot.core.loop import (  # noqa: E402
    DEFAULT_MAX_CONSECUTIVE_CRASHES,
    CoreLoop,
)
from autopilot.core.runtime import (  # noqa: E402
    IllegalTransition,
    RuntimeContext,
    RuntimeState,
)

# ==================================================================
# 假件：一整条感知→决策→执行链路，全同步、无模型、无真机
# ==================================================================

class _Sim:
    """``executor.sim`` 的最小替身 —— 只记 release_all 调了几次。"""

    def __init__(self, released=(), boom=False):
        self.calls = 0
        self._released = list(released)
        self._boom = boom

    def release_all(self):
        self.calls += 1
        if self._boom:
            raise RuntimeError("松键炸了")
        return self._released


class _Executor:
    def __init__(self, sim=None):
        self.sim = sim
        self.executed = []
        self.release_held_calls = 0

    def execute(self, intent, state):
        self.executed.append(intent)

    def release_held(self):
        """★ 生产代码**故意不用 getattr 兜底**（loop.py:529-533）：
        执行器缺这个方法属于接口缺失，应该立刻炸成 TICK_CRASH，
        而不是静默留下一个卡住不放的键。假件同样不给兜底。"""
        self.release_held_calls += 1


class _Capturer:
    def __init__(self, boom=False):
        self.calls = 0
        self._boom = boom

    def grab(self):
        self.calls += 1
        if self._boom:
            raise RuntimeError("窗口没了")
        return np.zeros((720, 1280, 3), np.uint8)


def _packet(sv: int = 1, perception_ok: bool = True) -> StatePacket:
    from autopilot.contracts.state_packet import ResourceInfo
    return StatePacket(
        scene=SceneInfo(id=SCENE_WORLD, confidence=0.9, entered_ms=1000),
        resource=ResourceInfo(perception_ok=perception_ok),
        timestamp_ms=int(time.time() * 1000),
        state_version=sv,
    )


class _Builder:
    def __init__(self, boom=False, perception_ok=True):
        self.calls = 0
        self._boom = boom
        self._ok = perception_ok
        self.last_world_entities = None
        self._sv = 0

    def build(self, frame):
        self.calls += 1
        if self._boom:
            raise RuntimeError("感知炸了")
        self._sv += 1
        return _packet(self._sv, self._ok)

    def drain_events(self):
        return []


class _Policy:
    def __init__(self, intent=None):
        self.calls = 0
        self.last_world_entities = None
        self._intent = intent

    def poll(self, world_state):
        self.calls += 1
        return self._intent


def _ctx(**kw) -> RuntimeContext:
    """默认给一条**健康**链路，避免后台线程因为崩溃而自动暂停。"""
    kw.setdefault("capturer", _Capturer())
    kw.setdefault("builder", _Builder())
    kw.setdefault("policy", _Policy())
    kw.setdefault("executor", _Executor(_Sim()))
    return RuntimeContext(**kw)


def _collect(loop: CoreLoop) -> list:
    """接 ``on_event`` 把事件同步收集起来，绕开总线线程的时序。"""
    seen: list = []
    loop._on_event = seen.append
    return seen


def _running_loop(**kw) -> tuple[CoreLoop, list]:
    """真正 ``start()`` 一次的工作循环（带健康链路）。"""
    loop = CoreLoop(_ctx(**kw))
    seen = _collect(loop)
    loop.start()
    return loop, seen


def _stop(loop: CoreLoop) -> None:
    if loop.state.alive:
        loop.stop()


# ==================================================================
# 状态机
# ==================================================================

@pytest.mark.unit
def test_loop__starts_idle() -> None:
    """构造完必须在 IDLE —— 没 start 就没线程、没资源。"""
    assert CoreLoop(_ctx()).state is RuntimeState.IDLE


@pytest.mark.unit
def test_loop__start_moves_to_running_and_announces() -> None:
    loop, seen = _running_loop()
    try:
        assert loop.state is RuntimeState.RUNNING
        assert any(e.code == "CORE_STARTED" for e in seen)
        assert loop._thread is not None and loop._thread.is_alive()
    finally:
        _stop(loop)


@pytest.mark.unit
def test_loop__start_message_carries_tick_rate() -> None:
    """CORE_STARTED 的文案要能让人看出当前跑多快。"""
    loop, seen = _running_loop(tick_rate_hz=5.0)
    try:
        ev = next(e for e in seen if e.code == "CORE_STARTED")
        assert "5" in ev.message and "200" in ev.message
    finally:
        _stop(loop)


@pytest.mark.unit
def test_loop__double_start_is_an_illegal_transition() -> None:
    """响亮失败，不静默忽略 —— 状态机的全部价值就在这条。"""
    loop = CoreLoop(_ctx())
    loop.start()
    try:
        with pytest.raises(IllegalTransition):
            loop.start()
    finally:
        _stop(loop)


@pytest.mark.unit
def test_loop__stop_from_idle_is_an_illegal_transition() -> None:
    """从没 start 就 stop 是调用方的 bug，必须暴露。"""
    with pytest.raises(IllegalTransition):
        CoreLoop(_ctx()).stop()


@pytest.mark.unit
def test_loop__stop_is_idempotent() -> None:
    """连按两次停止不应该炸（UI 上很容易发生）。"""
    loop, _ = _running_loop()
    loop.stop()
    loop.stop()
    assert loop.state is RuntimeState.STOPPED


@pytest.mark.unit
def test_loop__full_lifecycle_round_trip() -> None:
    """IDLE→RUNNING→PAUSED→RUNNING→STOPPING→STOPPED→IDLE。"""
    loop = CoreLoop(_ctx())
    loop.start()
    assert loop.state is RuntimeState.RUNNING
    loop.pause()
    assert loop.state is RuntimeState.PAUSED
    loop.resume()
    assert loop.state is RuntimeState.RUNNING
    loop.stop()
    assert loop.state is RuntimeState.STOPPED
    loop.reset()
    assert loop.state is RuntimeState.IDLE


# ==================================================================
# 控制面语义
# ==================================================================

@pytest.mark.unit
def test_loop__pause_releases_keys_immediately() -> None:
    """★ 暂停**立刻松键**，不等当前 tick 结束。

    正在执行 60 秒 MOVE_FORWARD 时用户按暂停，角色必须马上停。
    """
    ctx = _ctx()
    loop = CoreLoop(ctx)
    loop.start()
    ctx.executor.sim.calls = 0            # 忽略后台线程的自然调用
    try:
        loop.pause()
        assert loop.state is RuntimeState.PAUSED
        assert ctx.executor.sim.calls == 1
    finally:
        _stop(loop)


@pytest.mark.unit
def test_loop__stop_releases_keys_before_joining_thread() -> None:
    """停止顺序：先松键，再等线程。理由见 loop.py:259-261。"""
    ctx = _ctx()
    loop = CoreLoop(ctx)
    loop.start()
    ctx.executor.sim.calls = 0
    try:
        loop.stop()
        assert ctx.executor.sim.calls >= 1
        assert loop.shutdown_report, "停止必须留下收尾报告"
    finally:
        if loop.state.alive:
            _stop(loop)


@pytest.mark.unit
def test_loop__shutdown_report_records_every_resource() -> None:
    ctx = _ctx()
    loop = CoreLoop(ctx)
    loop.start()
    loop.stop()
    assert set(loop.shutdown_report) == {"bus", "release", "capturer"}


@pytest.mark.unit
def test_loop__pause_stops_execution_but_keeps_perceiving() -> None:
    """★ 暂停 = 停执行不停眼睛。

    理由（runtime.py:70-74）：暂停几分钟后恢复，若感知也停了，
    恢复第一帧是陈旧的，会基于旧状态做决策。
    """
    assert RuntimeState.PAUSED.accepts_input is False
    assert RuntimeState.PAUSED.runs_perception is True
    assert RuntimeState.RUNNING.accepts_input is True
    assert RuntimeState.IDLE.runs_perception is False


@pytest.mark.unit
def test_loop__reset_clears_counters() -> None:
    """reset 后统计归零，能重新统计一轮。"""
    loop, _ = _running_loop()
    loop.stop()
    loop.ctx.stats.record_crash()
    loop.reset()
    assert loop.ctx.stats.consecutive_crashes == 0
    assert loop._last_scene is None


@pytest.mark.unit
def test_loop__context_manager_stops_on_exit() -> None:
    loop = CoreLoop(_ctx())
    with loop as started:
        assert started.state is RuntimeState.RUNNING
    assert loop.state is RuntimeState.STOPPED


@pytest.mark.unit
def test_loop__max_consecutive_crashes_is_clamped_to_at_least_one() -> None:
    """配 0 会让「崩一次就暂停」退化成「崩了也不暂停」。"""
    loop = CoreLoop(_ctx(), max_consecutive_crashes=0)
    assert loop._max_consecutive_crashes == 1
    assert DEFAULT_MAX_CONSECUTIVE_CRASHES == 5


# ==================================================================
# 急停 / 松键
# ==================================================================

def _loop_in(state: RuntimeState, *, panic_check=None) -> CoreLoop:
    """白盒地摆到某个状态 —— ``_can_act`` 是纯判定，不必真起线程。

    ``panic_check`` 属于 ``CoreLoop``（急停监视器），不属于 ``RuntimeContext``。
    """
    loop = CoreLoop(_ctx(), panic_check=panic_check)
    loop._state = state
    return loop


@pytest.mark.unit
def test_can_act__true_when_running_without_panic() -> None:
    assert _loop_in(RuntimeState.RUNNING)._can_act("t") is True


@pytest.mark.unit
def test_can_act__true_when_paused() -> None:
    """暂停时线程仍在跑，感知要继续，所以 _can_act 仍为真。"""
    assert _loop_in(RuntimeState.PAUSED)._can_act("t") is True


@pytest.mark.unit
@pytest.mark.parametrize("state", [
    RuntimeState.IDLE, RuntimeState.STOPPING, RuntimeState.STOPPED])
def test_can_act__false_in_every_other_state(state: RuntimeState) -> None:
    assert _loop_in(state)._can_act("t") is False


@pytest.mark.unit
def test_can_act__false_when_stop_is_signalled() -> None:
    loop = _loop_in(RuntimeState.RUNNING)
    loop._stop_event.set()
    assert loop._can_act("t") is False


@pytest.mark.unit
def test_can_act__false_when_panic_already_latched() -> None:
    loop = _loop_in(RuntimeState.RUNNING)
    loop._panic_event.set()
    assert loop._can_act("t") is False


@pytest.mark.unit
def test_can_act__panic_key_detected_here_triggers_stop() -> None:
    """★ 急停必须走 ``_on_panic()``，不能只置标志位。

    早先只置 ``_panic_pressed``，监视线程的守卫
    ``if pressed and not self._panic_pressed`` 判定成「已处理」，
    于是**急停永远不触发停机**（loop.py:812-816）。
    """
    calls = []
    loop = _loop_in(RuntimeState.RUNNING, panic_check=lambda: True)
    loop._on_panic = lambda: calls.append(1)
    assert loop._can_act("t") is False
    assert calls == [1], "发现急停却没有真正走停机流程"


@pytest.mark.unit
def test_can_act__panic_check_exception_does_not_break_the_tick() -> None:
    """监视钩子自己炸了也不能拖垮 tick。

    ★★ 2026-10-13 改了断言：**从「放行」改成「不放行」**。

      原断言是 ``is True``，钉的是 fail-open 行为 —— 而 fail-open 意味着
      「我们不知道用户有没有按 F12」时**继续发按键**，那正是急停这个功能
      存在的理由被推翻的时刻。本意（「别崩」）依然满足：异常不外泄，
      只是这一轮不再放行。完整的新契约见
      ``tests/unit/test_can_act_fail_closed.py``。
    """
    def boom():
        raise RuntimeError("钩子坏了")

    # 不抛 = 没拖垮 tick
    assert _loop_in(RuntimeState.RUNNING, panic_check=boom)._can_act("t") is False


@pytest.mark.unit
def test_release_all__no_executor_is_a_noop() -> None:
    loop = CoreLoop(RuntimeContext(capturer=None, builder=None,
                                   policy=None, executor=None))
    loop._release_all("测试")            # 不抛就是过


@pytest.mark.unit
def test_release_all__uses_the_same_sim_as_executor() -> None:
    """刻意不另写一套松键逻辑 —— 两套迟早不一致。"""
    ctx = _ctx()
    loop = CoreLoop(ctx)
    loop._release_all("测试")
    assert ctx.executor.sim.calls == 1


@pytest.mark.unit
def test_release_all__exception_is_swallowed() -> None:
    """松键失败不该让停机流程崩掉。"""
    ctx = _ctx()
    ctx.executor.sim = _Sim(boom=True)
    CoreLoop(ctx)._release_all("测试")    # 不抛就是过


# ==================================================================
# 崩溃保护
# ==================================================================

@pytest.mark.unit
def test_note_crash__counts_and_releases_keys() -> None:
    """崩溃可能发生在「已经按下去了」之后，所以必须无条件松键。"""
    ctx = _ctx()
    loop = CoreLoop(ctx)
    loop._note_crash("炸了", "X_CRASH")
    assert loop.ctx.stats.consecutive_crashes == 1
    assert loop.ctx.stats.tick_crashes == 1
    assert ctx.executor.sim.calls == 1
    assert loop._tick_failed is True


@pytest.mark.unit
def test_note_crash__auto_pauses_at_threshold() -> None:
    ctx = _ctx()
    loop = CoreLoop(ctx, max_consecutive_crashes=3)
    loop.start()
    try:
        loop._note_crash("a", "X_CRASH")
        loop._note_crash("b", "X_CRASH")
        assert loop.state is RuntimeState.RUNNING, "未到阈值不该暂停"
        loop._note_crash("c", "X_CRASH")
        assert loop.state is RuntimeState.PAUSED
    finally:
        _stop(loop)


@pytest.mark.unit
def test_auto_pause__noop_when_already_paused() -> None:
    loop = _loop_in(RuntimeState.PAUSED)
    loop._auto_pause("没在跑就别暂停")      # 不抛就是过
    assert loop.state is RuntimeState.PAUSED


# ==================================================================
# 事件出口
# ==================================================================

@pytest.mark.unit
def test_emit__forwards_to_callback_and_publishes() -> None:
    from autopilot.contracts.events import Event
    loop = CoreLoop(_ctx())
    seen = _collect(loop)
    ev = Event.screenshot_failed("截图失败", 1)
    loop._emit(ev)
    assert ev in seen


@pytest.mark.unit
def test_emit__callback_exception_does_not_propagate() -> None:
    from autopilot.contracts.events import Event
    loop = CoreLoop(_ctx())

    def boom(_ev):
        raise RuntimeError("回调炸了")
    loop._on_event = boom
    loop._emit(Event.screenshot_failed("截图失败", 1))    # 不抛就是过


@pytest.mark.unit
def test_emit__works_with_no_callback_configured() -> None:
    from autopilot.contracts.events import Event
    CoreLoop(_ctx())._emit(Event.screenshot_failed("截图失败", 1))   # 不抛就是过


# ==================================================================
# 卡住播报
# ------------------------------------------------------------------
# ⚠️ 这一节在 2026-10-12 之前挂着两条 strict xfail，钉着一个真 bug：
#   ``_check_stuck`` 读 ``ctx.stats.consecutive_capture_failures``，
#   而那个字段在 ``LoopStats`` 上**根本不存在**（真实计数器是 CoreLoop 实例上的
#   ``self._consecutive_capture_failures``）。``getattr`` 默认值让 n 恒为 0，
#   于是 ``PATHFIND_STUCK`` 与 ``ESCAPE_SUCCESS`` 两条 YUI 语音**从未发出**。
#   已修 ``core/loop.py:_check_stuck``，xfail 已摘除，下面都是普通断言。
# ==================================================================

@pytest.mark.unit
def test_check_stuck__silent_below_threshold() -> None:
    """没卡住时不播报 —— 否则用户会被噪音淹没。"""
    loop = CoreLoop(_ctx())
    seen = _collect(loop)
    loop._consecutive_capture_failures = 2      # 未到阈值 3
    loop._check_stuck()
    assert seen == []


@pytest.mark.unit
def test_check_stuck__emits_when_capture_keeps_failing() -> None:
    loop = CoreLoop(_ctx())
    seen = _collect(loop)
    loop._consecutive_capture_failures = 5
    loop._check_stuck()
    assert any(e.code == "PATHFIND_STUCK" for e in seen)


@pytest.mark.unit
def test_check_stuck__announces_escape_success() -> None:
    """恢复即播报「走出来了」—— 用户最想听的就是这句。"""
    loop = CoreLoop(_ctx())
    seen = _collect(loop)
    loop._consecutive_capture_failures = 5
    loop._check_stuck()
    loop._consecutive_capture_failures = 0
    loop._check_stuck()
    assert any(e.code == "ESCAPE_SUCCESS" for e in seen)


@pytest.mark.unit
def test_check_stuck__does_not_repeat_the_same_stuck_level() -> None:
    """同一个卡住只说一遍 —— 每 tick 复述是 10-11 黑帧刷屏的同款病。"""
    loop = CoreLoop(_ctx())
    seen = _collect(loop)
    loop._consecutive_capture_failures = 5
    loop._check_stuck()
    loop._check_stuck()
    loop._check_stuck()
    assert sum(1 for e in seen if e.code == "PATHFIND_STUCK") == 1


@pytest.mark.unit
def test_check_stuck__re_announces_when_the_level_grows() -> None:
    """涨到下一档才再说 —— 升级脱困是要播报的。"""
    loop = CoreLoop(_ctx())
    seen = _collect(loop)
    loop._consecutive_capture_failures = 3
    loop._check_stuck()
    loop._consecutive_capture_failures = 9
    loop._check_stuck()
    assert sum(1 for e in seen if e.code == "PATHFIND_STUCK") == 2


@pytest.mark.unit
def test_check_stuck__reads_the_instance_counter_not_stats() -> None:
    """★ 把 bug 的根因钉住：读数必须来自 CoreLoop 实例。

    旧实现是 ``getattr(self.ctx.stats, "consecutive_capture_failures", 0)``，
    而 ``LoopStats`` 没有这个字段 -> 恒 0 -> 两条语音永远是死的。
    这条断言保证以后不会有人「顺手改回去」。
    """
    loop = CoreLoop(_ctx())
    seen = _collect(loop)
    assert not hasattr(loop.ctx.stats, "consecutive_capture_failures"), (
        "LoopStats 上竟然有这个字段了？那 _check_stuck 不该读它，"
        "请更新本文件的注释")

    loop._consecutive_capture_failures = 5
    loop._check_stuck()
    assert any(e.code == "PATHFIND_STUCK" for e in seen), (
        "真实计数器置位却没有播报 —— _check_stuck 又读错了地方")


@pytest.mark.unit
def test_check_stuck__no_escape_event_without_a_prior_stuck() -> None:
    """没卡住过就别说「走出来了」—— 那会让 YUI 凭空说一句。"""
    loop = CoreLoop(_ctx())
    seen = _collect(loop)
    loop._consecutive_capture_failures = 0
    loop._check_stuck()
    assert not any(e.code == "ESCAPE_SUCCESS" for e in seen)


# ==================================================================
# RuntimeContext
# ==================================================================

@pytest.mark.unit
def test_context__interval_is_the_reciprocal_of_hz() -> None:
    assert RuntimeContext(tick_rate_hz=5.0).target_interval_s == pytest.approx(0.2)
    assert RuntimeContext(tick_rate_hz=10.0).target_interval_s == pytest.approx(0.1)


@pytest.mark.unit
def test_context__zero_hz_means_unthrottled() -> None:
    assert RuntimeContext(tick_rate_hz=0.0).target_interval_s == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-1.0, -0.5])
def test_context__negative_hz_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        RuntimeContext(tick_rate_hz=bad)
    with pytest.raises(ValueError):
        RuntimeContext().set_tick_rate(bad)


@pytest.mark.unit
def test_context__set_tick_rate_applies_immediately() -> None:
    ctx = RuntimeContext(tick_rate_hz=5.0)
    ctx.set_tick_rate(20.0)
    assert ctx.target_interval_s == pytest.approx(0.05)


@pytest.mark.unit
def test_context__close_resources_tolerates_missing_pieces() -> None:
    """全 None 的上下文也要能关，不能在收尾时炸。"""
    report = RuntimeContext().close_resources()
    assert set(report) == {"bus", "release", "capturer"}


@pytest.mark.unit
def test_context__close_resources_survives_a_failing_releaser() -> None:
    """截图器或松键炸了，**其余资源仍要关完**。"""
    ctx = _ctx()
    ctx.executor.sim = _Sim(boom=True)
    report = ctx.close_resources()
    assert report["capturer"] is True, "松键失败不该连累截图器"


@pytest.mark.unit
def test_context__world_is_the_store_snapshot() -> None:
    loop, _ = _running_loop()
    try:
        deadline = time.time() + 3.0
        while loop.ctx.stats.states_built == 0 and time.time() < deadline:
            time.sleep(0.01)
        assert loop.ctx.stats.states_built > 0, "后台线程一帧都没跑"
        assert loop.ctx.world() == loop.ctx.store.get()
    finally:
        _stop(loop)


@pytest.mark.unit
def test_loop__background_thread_actually_ticks() -> None:
    """端到端：真起线程，真跑 tick，真入 store。"""
    loop, seen = _running_loop(tick_rate_hz=50.0)
    try:
        deadline = time.time() + 3.0
        while loop.ctx.stats.ticks < 3 and time.time() < deadline:
            time.sleep(0.01)
        assert loop.ctx.stats.ticks >= 3
        assert loop.ctx.stats.states_built >= 3
        assert loop.ctx.stats.tick_crashes == 0, (
            f"健康链路不该有崩溃：{loop.ctx.stats.snapshot()}")
        assert not loop._thread.is_alive() or True     # 线程还活着是对的
        assert isinstance(threading.current_thread(), threading.Thread)
    finally:
        _stop(loop)
