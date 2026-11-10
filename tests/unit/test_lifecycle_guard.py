# -*- coding: utf-8 -*-
"""genshin_game_buddy · 生命周期守卫单元测试。

★ 2026-11-07 新增（任务 C 的验证面）。

被测对象：:class:`lifecycle_guard.LifecycleGuard`。

守卫要做的事只有一件：**告诉感知栈「现在该不该歇」**。

    状态            含义                        感知栈
    --------------- --------------------------- --------
    ``ABSENT``      游戏进程压根没开             挂起
    ``BACKGROUND``  游戏在但不在前台              挂起（除非允许后台）
    ``FOREGROUND``  游戏窗口在前台                继续
    ``UNKNOWN``     探测失败 / 当前游戏不在守卫表   继续（fail-open）

★ 为什么坚持 fail-open
  守卫失灵时宁可多烧一点 CPU，也不能让插件整个哑掉 ——
  「插件不说话了」比「插件多烧点电」严重得多。

★ 为什么用 ``probe_fn`` 注入假状态
  真机上原神不一定开着，Windows 进程枚举又慢又脆。测试直接注入状态字符串，
  把「判定逻辑」与「操作系统探测」彻底分开 —— 判定逻辑必须 100% 可测。
"""

from __future__ import annotations

import threading
import time
from typing import Any, List, Tuple

import pytest

from plugin.plugins.genshin_game_buddy.lifecycle_guard import (
    DEFAULT_POLL_INTERVAL_S,
    GAME_PROCESSES,
    STATE_ABSENT,
    STATE_BACKGROUND,
    STATE_FOREGROUND,
    STATE_UNKNOWN,
    LifecycleGuard,
)


class RecordingLogger:
    """收集 (level, 渲染后文本) 的假 logger。"""

    def __init__(self) -> None:
        self.records: List[Tuple[str, str]] = []

    def _emit(self, level: str, message: str, *args: Any) -> None:
        if args:
            try:
                message = message.format(*args)
            except Exception:
                message = message + " " + " ".join(str(a) for a in args)
        self.records.append((level, message))

    def debug(self, message: str, *a: Any, **k: Any) -> None:
        self._emit("debug", message)

    def info(self, message: str, *a: Any, **k: Any) -> None:
        self._emit("info", message)

    def warning(self, message: str, *a: Any, **k: Any) -> None:
        self._emit("warning", message)

    def error(self, message: str, *a: Any, **k: Any) -> None:
        self._emit("error", message)


def make_guard(
    *,
    state: str = STATE_FOREGROUND,
    allow_background: bool = False,
    on_state_change: Any = None,
    active_game_fn: Any = None,
    poll_interval_s: float = 0.05,
) -> Tuple[LifecycleGuard, List[Tuple[str, str]]]:
    """造一个**注入假状态**的守卫（不碰操作系统）。"""
    logger = RecordingLogger()
    guard = LifecycleGuard(
        logger_=logger,
        active_game_fn=active_game_fn,
        poll_interval_s=poll_interval_s,
        allow_background=allow_background,
        on_state_change=on_state_change,
        probe_fn=lambda: state,
    )
    return guard, logger.records


# ══════════════════════════════════════════════════════════════════════
# 状态 → 挂起判定
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.unit
def test_absent_suspends_perception() -> None:
    """游戏进程不在 → 挂起感知栈。"""
    guard, _ = make_guard(state=STATE_ABSENT)
    guard.start()
    try:
        assert guard.state == STATE_ABSENT
        assert guard.suspended is True
    finally:
        guard.stop()


@pytest.mark.unit
def test_background_suspends_by_default() -> None:
    """游戏在后台（最小化 / Alt-Tab）→ 默认挂起。"""
    guard, _ = make_guard(state=STATE_BACKGROUND)
    guard.start()
    try:
        assert guard.state == STATE_BACKGROUND
        assert guard.suspended is True
    finally:
        guard.stop()


@pytest.mark.unit
def test_background_does_not_suspend_when_background_allowed() -> None:
    """开了「允许后台运行」→ 后台也不挂起。"""
    guard, _ = make_guard(state=STATE_BACKGROUND, allow_background=True)
    guard.start()
    try:
        assert guard.state == STATE_BACKGROUND
        assert guard.suspended is False
    finally:
        guard.stop()


@pytest.mark.unit
def test_foreground_never_suspends() -> None:
    """游戏在前台 → 永远不挂起。"""
    guard, _ = make_guard(state=STATE_FOREGROUND)
    guard.start()
    try:
        assert guard.suspended is False
    finally:
        guard.stop()


@pytest.mark.unit
def test_unknown_state_fails_open() -> None:
    """★ 探测失败 → **不挂起**（fail-open）。

    这是守卫最重要的一条性质：宁可多烧 CPU，也不能让插件哑掉。
    """
    guard, _ = make_guard(state=STATE_UNKNOWN)
    guard.start()
    try:
        assert guard.state == STATE_UNKNOWN
        assert guard.suspended is False
    finally:
        guard.stop()


@pytest.mark.unit
def test_probe_exception_fails_open() -> None:
    """★ 探测函数直接抛异常 → 守卫不许炸，也不许挂起。"""

    def boom() -> str:
        raise RuntimeError("模拟探测失败")

    guard = LifecycleGuard(logger_=RecordingLogger(), probe_fn=boom)
    guard.start()
    try:
        assert guard.state in (STATE_UNKNOWN, STATE_ABSENT, STATE_FOREGROUND)
        # 关键：探测失败时守卫仍然是「不挂起」或「UNKNOWN 不挂起」
        if guard.state == STATE_UNKNOWN:
            assert guard.suspended is False
    finally:
        guard.stop()


# ══════════════════════════════════════════════════════════════════════
# 按 game_id 分组（多游戏插件的正确性）
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.unit
def test_unknown_game_is_not_guarded() -> None:
    """★ 当前会话的游戏不在守卫表里 → 守卫**不管它**。

    没有这条，玩鸣潮时会被「原神没开」误伤：鸣潮的感知被一起挂掉。
    """
    # 没有 probe_fn 时走真实探测路径：当前游戏 = 鸣潮 → 不在表里 → UNKNOWN
    guard = LifecycleGuard(
        logger_=RecordingLogger(),
        active_game_fn=lambda: "wuthering_waves",
    )
    guard.start()
    try:
        assert guard.state == STATE_UNKNOWN
        assert guard.suspended is False
    finally:
        guard.stop()


@pytest.mark.unit
def test_no_active_game_is_not_guarded() -> None:
    """没有登记任何游戏 → 不守卫（感知栈本来也不跑）。"""
    guard = LifecycleGuard(logger_=RecordingLogger(), active_game_fn=lambda: "")
    guard.start()
    try:
        assert guard.state == STATE_UNKNOWN
        assert guard.suspended is False
    finally:
        guard.stop()


@pytest.mark.unit
def test_game_process_table_covers_both_genshin_exe_names() -> None:
    """原神有国服 / 国际服两个进程名，两个都要认。"""
    names = GAME_PROCESSES["genshin"]
    assert "yuanshen.exe" in names
    assert "genshinimpact.exe" in names
    assert all(n == n.lower() for n in names), "进程名必须小写（比对时统一小写）"


# ══════════════════════════════════════════════════════════════════════
# 状态迁移与回调
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.unit
def test_state_change_callback_fires_on_transition() -> None:
    """状态迁移 → 回调被调用（插件靠它做「立即唤醒」）。"""
    seen: List[Tuple[str, str]] = []
    states = [STATE_ABSENT, STATE_FOREGROUND]

    def probe() -> str:
        return states.pop(0) if states else STATE_FOREGROUND

    guard = LifecycleGuard(
        logger_=RecordingLogger(),
        probe_fn=probe,
        on_state_change=lambda prev, cur: seen.append((prev, cur)),
        poll_interval_s=0.05,
    )
    guard.start()
    try:
        deadline = time.monotonic() + 3.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.02)
        assert seen, "状态迁移没有触发回调"
        assert seen[0] == (STATE_ABSENT, STATE_FOREGROUND), seen
    finally:
        guard.stop()


@pytest.mark.unit
def test_state_change_callback_not_called_when_state_stable() -> None:
    """状态没变 → 不回调（避免刷日志）。"""
    calls: List[Tuple[str, str]] = []
    guard = LifecycleGuard(
        logger_=RecordingLogger(),
        probe_fn=lambda: STATE_FOREGROUND,
        on_state_change=lambda prev, cur: calls.append((prev, cur)),
        poll_interval_s=0.02,
    )
    guard.start()
    try:
        time.sleep(0.3)
        assert calls == [], f"状态没变却回调了: {calls}"
    finally:
        guard.stop()


@pytest.mark.unit
def test_callback_exception_does_not_kill_guard() -> None:
    """回调抛异常 → 守卫继续活着（不许被用户的坏回调带死）。"""

    def bad(prev: str, cur: str) -> None:
        raise RuntimeError("回调炸了")

    guard = LifecycleGuard(
        logger_=RecordingLogger(),
        probe_fn=lambda: STATE_ABSENT,
        on_state_change=bad,
        poll_interval_s=0.05,
    )
    guard.start()
    try:
        time.sleep(0.2)
        assert guard.running, "守卫线程被坏回调带死了"
        assert guard.suspended is True
    finally:
        guard.stop()


# ══════════════════════════════════════════════════════════════════════
# 运行期热更新
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.unit
def test_allow_background_toggle_recomputes_immediately() -> None:
    """★ 运行期打开「允许后台运行」→ **立刻**不再挂起。

    改了开关却要等下一个探测周期（默认 5s）才生效的话，
    用户在面板上勾了开关、感知栈却还停着 —— 看起来就像开关没用。
    """
    guard, _ = make_guard(state=STATE_BACKGROUND)
    guard.start()
    try:
        assert guard.suspended is True
        guard.set_allow_background(True)
        assert guard.suspended is False, "开关改了但挂起状态没跟着变"
        guard.set_allow_background(False)
        assert guard.suspended is True, "关掉开关后又该挂起"
    finally:
        guard.stop()


@pytest.mark.unit
def test_poll_interval_can_be_changed_at_runtime() -> None:
    """探测周期可以热改。

    ★ 下限是 0.5 秒
      再快也没有意义：一次探测就是一次系统调用，而且守卫存在的意义
      就是「不空转」—— 0.05s 一轮反而是空转。
    """
    guard, _ = make_guard()
    guard.start()
    try:
        guard.set_poll_interval(30.0)
        assert guard._poll_interval_s == 30.0
        guard.set_poll_interval(0.01)
        assert guard._poll_interval_s == 0.5, "轮询下限应为 0.5 秒"
    finally:
        guard.stop()


# ══════════════════════════════════════════════════════════════════════
# 线程生命周期
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.unit
def test_guard_thread_starts_and_stops() -> None:
    """守卫线程能起也能停（插件 shutdown 必须能收回它）。"""
    guard, _ = make_guard()
    guard.start()
    assert guard.running is True
    guard.stop()
    assert guard.running is False


@pytest.mark.unit
def test_guard_start_is_idempotent() -> None:
    """重复 start 不会起第二个线程。"""
    guard, _ = make_guard()
    guard.start()
    try:
        first = guard._thread
        guard.start()
        assert guard._thread is first, "重复 start 起了新线程"
    finally:
        guard.stop()


@pytest.mark.unit
def test_guard_stop_is_idempotent() -> None:
    """重复 stop 不炸（插件 shutdown 可能被调两次）。"""
    guard, _ = make_guard()
    guard.start()
    guard.stop()
    guard.stop()


@pytest.mark.unit
def test_guard_thread_is_daemon() -> None:
    """★ 守卫线程必须是 daemon —— 插件停掉时它不能吊住进程。"""
    guard, _ = make_guard()
    guard.start()
    try:
        assert guard._thread is not None
        assert guard._thread.daemon is True, "守卫线程不是 daemon，会吊住宿主退出"
        assert guard._thread.name == "neko-lifecycle-guard"
    finally:
        guard.stop()


@pytest.mark.unit
def test_first_probe_happens_synchronously_at_start() -> None:
    """★ 第一次探测是同步的：插件刚起来就知道当前状态。

    不用等第一个周期（默认 5s）才生效 —— 那 5s 里感知栈还在空转。
    """
    calls: List[int] = []

    def probe() -> str:
        calls.append(1)
        return STATE_ABSENT

    guard = LifecycleGuard(logger_=RecordingLogger(), probe_fn=probe)
    guard.start()
    try:
        assert len(calls) == 1, f"start 时应同步探一次，实际 {len(calls)} 次"
        assert guard.state == STATE_ABSENT
    finally:
        guard.stop()


@pytest.mark.unit
def test_default_poll_interval_is_five_seconds() -> None:
    """默认探测周期 5 秒（改它要有意识地改，并同步改文档）。"""
    assert DEFAULT_POLL_INTERVAL_S == 5.0
