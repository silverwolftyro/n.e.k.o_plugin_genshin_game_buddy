"""``CoreLoop._can_act()`` 的急停闸门（2026-10-13 修 fail-open）。

★★ 这次修的是什么
  ``_can_act()`` 里急停检查抛异常时，原来是 ``except: pass`` + ``return True``
  —— **零日志地放行本轮动作**。而同文件姊妹代码（监视线程 :785）早就
  用了 ``logger.exception``，说明这是遗留而不是设计。

  急停检查失败 = 「我们不知道用户有没有按 F12」。放行的话，一旦它开始
  抛异常，引擎就会在用户可能已经按下急停的情况下继续发按键 ——
  而那正是急停这个功能存在的理由被推翻的时刻。

  现在改成 **fail-closed**：异常时不放行，且 ``logger.exception`` 留痕。
"""

from __future__ import annotations

import logging

import pytest

from autopilot.core.loop import CoreLoop
from autopilot.core.runtime import RuntimeState

pytestmark = pytest.mark.unit


def _loop(running: bool = True, panic_check=None) -> CoreLoop:
    """造一个只够测 ``_can_act`` 的 CoreLoop（不碰截图/设备）。

    ★ 刻意不跑 ``__init__``：那会去建截图器和实体检测器（要 DirectML）。
      这里只摆 ``_can_act`` 真正读到的三个字段。
    """
    loop = CoreLoop.__new__(CoreLoop)
    import threading

    # ★ ``state`` 是只读 property，底下读三个字段：``_state_lock`` /
    #   ``_state`` / 还要有 panic / stop 两个 Event。
    loop._state_lock = threading.Lock()
    loop._state = RuntimeState.RUNNING if running else RuntimeState.STOPPED
    loop._panic_event = threading.Event()
    loop._stop_event = threading.Event()
    loop._panic_check = panic_check
    loop._on_panic = lambda: None          # 记下调用即可
    return loop


# ==================================================================
# 正常路径
# ==================================================================

def test_can_act__allows_when_no_panic_check_configured() -> None:
    """没配检查器 = 没有急停这回事，放行。"""
    assert _loop()._can_act("before") is True


def test_can_act__allows_when_check_returns_false() -> None:
    """检查器说「没按」-> 放行。"""
    assert _loop(panic_check=lambda: False)._can_act("before") is True


def test_can_act__blocks_and_calls_panic_when_check_is_true() -> None:
    """★ 发现急停必须走 ``_on_panic()``，不能只置标志位。

    早先只置 ``_panic_pressed = True``，结果监视线程判定「已处理」，
    急停永远不触发停机。
    """
    loop = _loop(panic_check=lambda: True)
    called: list[bool] = []
    loop._on_panic = lambda: called.append(True)
    assert loop._can_act("before") is False
    assert called == [True]


def test_can_act__blocks_when_already_panicking() -> None:
    loop = _loop()
    loop._panic_event.set()
    assert loop._can_act("before") is False


def test_can_act__blocks_when_stop_requested() -> None:
    loop = _loop()
    loop._stop_event.set()
    assert loop._can_act("before") is False


def test_can_act__blocks_when_state_is_not_running() -> None:
    assert _loop(running=False)._can_act("before") is False


# ==================================================================
# ★ 本次修的核心：异常时 fail-closed
# ==================================================================

def test_can_act__blocking_error_refuses_the_action() -> None:
    """★★ 急停检查抛异常 -> **不放行**。

    修之前这里是 ``return True``：引擎会在用户可能已经按下急停的情况下
    继续发按键。
    """

    def _boom() -> bool:
        raise RuntimeError("Win32 查询失败")

    assert _loop(panic_check=_boom)._can_act("before") is False


def test_can_act__blocking_error_is_logged(caplog) -> None:
    """★ 同一个异常必须留下 ``logger.exception`` 的完整 traceback。

    修之前是裸 ``pass``：默认 INFO 级别下**一个字都不会输出**。
    """
    with caplog.at_level(logging.ERROR, logger="autopilot.core.loop"):

        def _boom() -> bool:
            raise ValueError("读取键鼠状态失败")

        _loop(panic_check=_boom)._can_act("before")

    assert caplog.records, "异常没有留下任何日志"
    rec = caplog.records[-1]
    assert rec.levelno >= logging.ERROR
    assert rec.exc_info is not None, "没有附带 traceback，只有一行说明"
    assert "ValueError" in caplog.text


def test_can_act__log_message_says_the_action_was_refused(caplog) -> None:
    """★ 日志要说清「本轮不放行」—— 只说「异常了」不足以让人采取行动。"""
    with caplog.at_level(logging.ERROR, logger="autopilot.core.loop"):

        def _boom() -> bool:
            raise RuntimeError("x")

        _loop(panic_check=_boom)._can_act("发送按键前")

    text = caplog.text
    assert "不放行" in text
    assert "发送按键前" in text, "日志里没有说明是在哪一步失败的"


def test_can_act__repeated_errors_keep_refusing(caplog) -> None:
    """★ 持续出错就一直拒绝 —— 不能第一次拒绝、第二次就放行。"""
    calls = {"n": 0}

    def _boom() -> bool:
        calls["n"] += 1
        raise RuntimeError("一直坏")

    loop = _loop(panic_check=_boom)
    with caplog.at_level(logging.ERROR, logger="autopilot.core.loop"):
        results = [loop._can_act(f"step{i}") for i in range(5)]
    assert results == [False] * 5
    assert calls["n"] == 5


def test_can_act__recovers_once_the_check_works_again() -> None:
    """★ 检查恢复后要能重新放行 —— fail-closed 不能变成永久熔断。"""
    state = {"broken": True}

    def _maybe() -> bool:
        if state["broken"]:
            raise RuntimeError("临时故障")
        return False

    loop = _loop(panic_check=_maybe)
    assert loop._can_act("a") is False
    state["broken"] = False
    assert loop._can_act("b") is True


def test_can_act__does_not_panic_on_check_failure() -> None:
    """★ 异常绝不外泄 —— 那会把整个 tick 线程带走。"""

    def _boom() -> bool:
        raise KeyboardInterrupt()

    # KeyboardInterrupt 是 BaseException 的子类，不该被 except Exception 吞；
    # 用一个普通 RuntimeError 验证「不外泄」这条契约
    def _boom2() -> bool:
        raise RuntimeError("x")

    assert _loop(panic_check=_boom2)._can_act("a") is False
    with pytest.raises(KeyboardInterrupt):
        _loop(panic_check=_boom)._can_act("a")
