"""注入回声抑制测试 —— ``InputSimulator.is_own_echo``。

★ 这条链路防的是什么（2026-10-18 实机复现）
-------------------------------------------
``autopilot/actuator/physical_input.py`` 用 ``GetAsyncKeyState`` 轮询来
发现「用户在按键」。实测：**``SendInput`` 注入进去的按键同样会让
``GetAsyncKeyState`` 的高位变 1**（down 后 30ms 读到 1）。

于是不加过滤时，引擎按一次 E -> 监听器在 20ms 内看到 E 被按下 ->
判定「用户在按 E」-> ``release_all()`` 把引擎**自己刚按下的键**释放掉
+ 冻结 3 秒 + 执行层抛 ``_UserTakeover``。引擎会把自己的输出掐掉。

复现（``sim.key_press(VK.HELP, hold_ms=30)``，USE_SKILL 走的就是这条）：
过滤前 1/1 触发，过滤后 0/1 触发，窗口外的真人按键仍然 1/1 捕获。

⚠️ 这些用例用 ``DryRunBackend`` 而不是真注入 ——
   ``_note_echo`` 挂在 :meth:`InputSimulator._record` 上，那是
   **所有**注入的必经之路（真实后端与 DryRun 共用），所以干跑测的
   就是生产那条路径。真注入会往用户桌面敲键，CI 里不能干那件事。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.simulator import (  # noqa: E402
    ECHO_SUPPRESS_S, DryRunBackend, InputSimulator, KeyId, VK,
)


def _sim() -> InputSimulator:
    return InputSimulator(backend=DryRunBackend())


# ==================================================================
# 1 · 记账
# ==================================================================
class TestEchoRecording:
    def test_tap_registers_the_key(self) -> None:
        sim = _sim()
        sim.key_press(VK.E, hold_ms=1)
        assert sim.is_own_echo("E") is True

    def test_held_key_registers_too(self) -> None:
        """平A 那类「按下不松」也必须记账 —— 它才是触发窗口最长的那种。"""
        sim = _sim()
        sim.mouse_down(VK.LBUTTON)
        assert sim.is_own_echo("LBUTTON") is True
        sim.mouse_up(VK.LBUTTON)

    def test_mouse_button_maps_to_its_vk(self) -> None:
        """鼠标事件要折算成 VK 才有 ``is_key_down`` / ``is_own_echo`` 能问。"""
        sim = _sim()
        sim.mouse_tap(VK.RBUTTON, hold_ms=1)
        assert sim.is_own_echo("RBUTTON") is True

    def test_unpressed_key_is_not_an_echo(self) -> None:
        sim = _sim()
        assert sim.is_own_echo("E") is False

    def test_key_events_only(self) -> None:
        """鼠标移动不该把哪个键标成「我在按」。"""
        sim = _sim()
        sim.mouse_move_relative(5, 5)
        for name in ("E", "W", "LBUTTON"):
            assert sim.is_own_echo(name) is False, name


# ==================================================================
# 2 · 判别
# ==================================================================
class TestEchoQuery:
    def test_accepts_name_vk_and_keyid(self) -> None:
        sim = _sim()
        sim.key_press(VK.TWO, hold_ms=1)
        assert sim.is_own_echo("TWO") is True
        assert sim.is_own_echo(VK.TWO) is True
        assert sim.is_own_echo(int(VK.TWO)) is True
        assert sim.is_own_echo(KeyId(int(VK.TWO))) is True

    def test_unknown_name_does_not_suppress(self) -> None:
        """★ 认不出来就**不抑制**。

        宁可漏一次让权，也不要因为一个拼写问题把用户的按键吞掉。
        """
        sim = _sim()
        assert sim.is_own_echo("TOTALLY_NOT_A_KEY") is False
        assert sim.is_own_echo("") is False

    def test_vk_prefixed_name_is_accepted(self) -> None:
        sim = _sim()
        sim.key_press(VK.Q, hold_ms=1)
        assert sim.is_own_echo("VK_Q") is True

    def test_window_expires(self) -> None:
        """★ 窗口过期后必须返回 False —— 不能把让权永久关掉。"""
        sim = _sim()
        sim.key_press(VK.E, hold_ms=1)
        assert sim.is_own_echo("E") is True
        later = time.monotonic() + ECHO_SUPPRESS_S + 0.05
        assert sim.is_own_echo("E", now=later) is False

    def test_window_only_covers_the_press_it_belongs_to(self) -> None:
        """按过 E 之后按 W，W 是真人 —— 抑制必须**按键**而不是全局。"""
        sim = _sim()
        sim.key_press(VK.E, hold_ms=1)
        assert sim.is_own_echo("E") is True
        assert sim.is_own_echo("W") is False

    def test_window_covers_the_next_poll(self) -> None:
        """20ms 的轮询必须在窗口内 —— 否则等于没修。

        这条是**回归钉子**：有人把 ``ECHO_SUPPRESS_S`` 调到 0.001
        的话，这里会红。
        """
        assert ECHO_SUPPRESS_S >= 0.04, \
            "必须大于 physical_input.POLL_INTERVAL_S (0.02)，否则漏掉"


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
