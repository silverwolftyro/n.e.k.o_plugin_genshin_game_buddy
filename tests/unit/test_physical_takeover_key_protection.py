"""物理抢占时**不许**松开用户按键（2026-10-19）。

★ 这条链路防的是什么
--------------------
用户手动长按 E 放技能、或按 A/W 走位时，引擎的物理输入监听器一检测到
按键就调 ``release_all()`` 让权。而旧的 ``release_all()`` 是这样的::

    for vk in list(VK):
        if backend.is_key_down(vk):   # ← 分不出是引擎按的还是用户按的
            发 KeyUp

``GetAsyncKeyState`` 对**注入**的按键同样返回「按下」，所以引擎**分不出**
这个键是自己的还是**用户手指底下的**。于是用户按住 E 的长按，
一次让权就被合成 KeyUp **当场掐断** —— 表现是「长按出不来」「走不动」。

★ 修法
------
1. ``physical_input.PhysicalInputWatcher`` 维护「用户此刻正按着的键」
   （``_held_physical_keys``：按下时加入、松开时移除）。
2. ``simulator.InputSimulator`` 维护「引擎自己按着、还没抬起的键」
   （``_engine_down``），并把两者相减 —— 得到「**用户**正按着的键」。
3. ``release_all(exclude_physical=True)`` **跳过**这些键；
   引擎自己按着的键**照旧全松**（异常安全底线不变）。

★ 为什么相减这一步不能省
------------------------
监听器看到的「按着的键」里**混着引擎自己注入的键**（实测 SendInput
注入后 30ms 读高位 = 1）。不减去的话：引擎长按 E（钟离护盾 1.5 秒）时
E 会被当成「用户按着」→ 不松 → **引擎的 E 永远卡在游戏里**。
减掉之后：引擎的键一定被松，用户的键一定不被碰。
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import pytest

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.executor import IntentExecutor  # noqa: E402
from autopilot.actuator.physical_input import (  # noqa: E402
    WATCHED_KEYS, PhysicalInputWatcher)
from autopilot.actuator.safety import SafetyConfig, SafetyGuard  # noqa: E402
from autopilot.actuator.simulator import (  # noqa: E402
    DryRunBackend, InputSimulator, VK, _mk_keys)
from autopilot.contracts.intent_packet import ActionType, IntentPacket  # noqa: E402
from autopilot.contracts.state_packet import (  # noqa: E402
    SCENE_WORLD, PlayerInfo, ResourceInfo, SceneInfo, StatePacket)
from autopilot.core.loop import CoreLoop  # noqa: E402
from autopilot.core.runtime import RuntimeContext  # noqa: E402

pytestmark = pytest.mark.unit


# ==================================================================
# 夹具与工具
# ==================================================================

def _physical_press(backend: DryRunBackend, vk: VK) -> None:
    """模拟「用户的手指按住了这个键」。

    ★ 必须**绕过** :class:`InputSimulator` 直接打后端。
      物理按键不是引擎注入的 —— 走模拟器的话它会被记进
      ``_engine_down``，保护逻辑就会把它当成「引擎自己的键」，
      测试就成了自证其说。
    """
    backend.send(_mk_keys(int(vk), up=False))


def _key_ups(backend: DryRunBackend, vk: VK) -> list:
    """这个键被抬起过几次。"""
    return [e for e in backend.events
            if e.kind == "key" and not e.down and e.vk == int(vk)]


def _mouse_ups(backend: DryRunBackend, vk: VK) -> list:
    """这个鼠标键被抬起过几次。"""
    return [e for e in backend.events
            if e.kind == "mouse" and not e.down and e.vk == int(vk)]


def _state() -> StatePacket:
    return StatePacket(
        scene=SceneInfo(id=SCENE_WORLD, confidence=0.9),
        player=PlayerInfo(party=(), hp_ratio=1.0, hp_source="unit"),
        resource=ResourceInfo(perception_ok=True),
        state_version=1,
    )


def _sim(held: set[int] | None = None) -> tuple[InputSimulator, DryRunBackend]:
    backend = DryRunBackend()
    sim = InputSimulator(backend=backend)
    if held is not None:
        sim.set_physical_hold_provider(lambda: set(held))
    return sim, backend


# ==================================================================
# 1 · release_all 的新语义
# ==================================================================

class TestReleaseAllProtectsUserKeys:
    def test_the_users_key_is_not_released(self) -> None:
        """★★ 核心断言：用户按着的 E **绝对不能**收到 KeyUp。

        这是「长按出不来」的根因：合成 KeyUp 到了游戏里，
        用户按住 E 的长按当场断掉。
        """
        sim, backend = _sim(held={int(VK.E)})
        _physical_press(backend, VK.E)
        assert backend.is_key_down(int(VK.E)), "前提没成立：E 本来就没按着"

        released = sim.release_all(exclude_physical=True)

        assert _key_ups(backend, VK.E) == [], "用户的 E 被松了！"
        assert int(VK.E) not in released
        # ★ 键**仍然是按着的** —— 用户的输入没被打断。
        assert backend.is_key_down(int(VK.E))

    def test_the_engines_own_mouse_button_is_released(self) -> None:
        """★★ 对称的另一半：引擎自己按的**平A 左键**必须被正确抬起。

        保护用户的键 ≠ 不松引擎的键。两件事必须同时成立。
        """
        sim, backend = _sim(held={int(VK.E)})
        _physical_press(backend, VK.E)
        sim.mouse_down(VK.LBUTTON)          # 引擎的平A：按下不松

        released = sim.release_all(exclude_physical=True)

        assert int(VK.LBUTTON) in released
        ups = _mouse_ups(backend, VK.LBUTTON)
        assert len(ups) == 1, f"左键没被抬起：{ups}"
        # ★ 必须是**鼠标抬起**，不是键盘抬起（§37 的老坑）
        assert ups[0].name == "Lup", ups[0]
        assert not backend.is_key_down(int(VK.LBUTTON))

    def test_old_behaviour_would_have_broken_the_user(self) -> None:
        """★ 回归钉子：不带参数（旧行为）时 E **确实**会被松。

        证明这条 Bug 是真的存在过 —— 也证明新参数是有意义的，
        而不是「反正也没坏」。
        """
        sim, backend = _sim(held={int(VK.E)})
        _physical_press(backend, VK.E)

        sim.release_all()

        assert _key_ups(backend, VK.E), "旧行为就该松开 E（这条红了说明 Bug 不存在）"

    def test_without_the_flag_nothing_is_protected(self) -> None:
        """默认行为**一字未改** —— 异常安全那条底线不动。"""
        sim, backend = _sim(held={int(VK.E)})
        _physical_press(backend, VK.E)

        released = sim.release_all()

        assert int(VK.E) in released
        assert _key_ups(backend, VK.E)

    def test_the_engines_own_key_is_never_protected(self) -> None:
        """★★ 引擎长按的键**不能**被保护 —— 否则它永远卡在游戏里。

        监听器看得到引擎注入的键（``GetAsyncKeyState`` 一样返回按下），
        所以「用户按着的键」必须减去「引擎自己按着的键」。
        """
        sim, backend = _sim(held={int(VK.E)})
        sim.key_down(VK.E)                   # 引擎自己长按 E

        released = sim.release_all(exclude_physical=True)

        assert int(VK.E) in released, "引擎自己的 E 竟然没被松 —— 会卡在游戏里"
        assert _key_ups(backend, VK.E)

    def test_no_provider_means_no_protection(self) -> None:
        """没接 provider（单测/仿真/非 Windows）时退化成旧行为，不炸。"""
        sim, backend = _sim(held=None)
        _physical_press(backend, VK.E)

        released = sim.release_all(exclude_physical=True)

        assert int(VK.E) in released

    def test_a_broken_provider_does_not_break_the_release(self) -> None:
        """★ provider 抛异常时**不保护任何键**，但松键照做。

        判别失败宁可放过，也不能让让权路径整条炸掉。
        """
        sim, backend = _sim()

        def boom():
            raise RuntimeError("读不到")

        sim.set_physical_hold_provider(boom)
        _physical_press(backend, VK.E)

        released = sim.release_all(exclude_physical=True)

        assert int(VK.E) in released

    def test_engine_keys_are_released_even_when_user_holds_another(self) -> None:
        """★ 保护是**按键**的，不是「一保护就什么都不松」。"""
        sim, backend = _sim(held={int(VK.E)})
        _physical_press(backend, VK.E)
        sim.key_down(VK.W)                    # 引擎自己按着 W

        released = sim.release_all(exclude_physical=True)

        assert int(VK.W) in released
        assert int(VK.E) not in released


# ==================================================================
# 2 · 物理输入层：用户按着的键集合
# ==================================================================

class _FakeKeyState:
    """可编程的 GetAsyncKeyState 替身。"""

    def __init__(self) -> None:
        self.down: set[int] = set()

    def __call__(self, vk: int) -> int:
        return 0x8000 if vk in self.down else 0

    def press(self, vk: VK) -> None:
        self.down.add(int(vk))

    def release(self, vk: VK) -> None:
        self.down.discard(int(vk))


class TestHeldPhysicalKeys:
    def test_press_adds_and_release_removes(self) -> None:
        """★ 「用户正在按下的物理按键集合」：按下加入、松开移除。"""
        fake = _FakeKeyState()
        watcher = PhysicalInputWatcher(lambda name: None)
        watcher._poll_once(fake)             # 基线

        fake.press(VK.E)
        watcher._poll_once(fake)
        assert watcher.held_physical_keys() == frozenset({VK.E})

        fake.release(VK.E)
        watcher._poll_once(fake)
        assert watcher.held_physical_keys() == frozenset()

    def test_keeps_the_key_while_held(self) -> None:
        """★ 按住不放时集合里一直有它 —— 松键保护靠的就是这个。"""
        fake = _FakeKeyState()
        watcher = PhysicalInputWatcher(lambda name: None)
        watcher._poll_once(fake)

        fake.press(VK.W)
        for _ in range(5):
            watcher._poll_once(fake)

        assert watcher.held_physical_keys() == frozenset({VK.W})

    def test_startup_state_counts_as_held(self) -> None:
        """★ 引擎刚起、用户已经按着 W —— W **现在**就正被按着。"""
        fake = _FakeKeyState()
        fake.press(VK.W)
        watcher = PhysicalInputWatcher(lambda name: None)
        watcher._down = {vk for vk in WATCHED_KEYS if fake(int(vk)) & 0x8000}
        watcher._held_physical_keys = set(watcher._down)

        assert watcher.held_physical_keys() == frozenset({VK.W})

    def test_the_provider_is_read_only(self) -> None:
        """★ 返回的是**快照**，外部改不动内部状态。"""
        fake = _FakeKeyState()
        watcher = PhysicalInputWatcher(lambda name: None)
        watcher._poll_once(fake)
        fake.press(VK.E)
        watcher._poll_once(fake)

        got = watcher.held_physical_keys()
        assert isinstance(got, frozenset)
        got = got | {VK.A}                 # frozenset 运算得到新对象
        assert watcher.held_physical_keys() == frozenset({VK.E})


# ==================================================================
# 3 · 全链路：让权时松键
# ==================================================================

def _loop_with_sim(sim: InputSimulator) -> CoreLoop:
    """造一个**只够让权用**的主循环（不 start，不截图）。"""
    ex = IntentExecutor(simulator=sim, dry_run=True)
    ctx = RuntimeContext(executor=ex)
    return CoreLoop(ctx)


class TestTakeoverDoesNotStealTheUsersKey:
    def test_takeover_keeps_users_key_and_releases_engines(self) -> None:
        """★★ 全链路：用户按下 E 触发物理抢占。

        · E（用户的）**不能**被松 —— 否则长按出不来；
        · 左键（引擎的平A）**必须**被松 —— 否则角色一直平A。
        """
        sim, backend = _sim()
        loop = _loop_with_sim(sim)

        _physical_press(backend, VK.E)       # 用户按住 E
        sim.mouse_down(VK.LBUTTON)           # 引擎的平A 按着

        loop._on_physical_input("E")

        assert _key_ups(backend, VK.E) == [], "用户的 E 被松了！"
        assert backend.is_key_down(int(VK.E)), "用户的输入被打断了"
        assert _mouse_ups(backend, VK.LBUTTON), "引擎的左键没被松"
        assert not backend.is_key_down(int(VK.LBUTTON))

    def test_takeover_state_is_entered(self) -> None:
        """★ 让权本身照旧生效（这次修的是「松键」，不是「让权」）。"""
        sim, _ = _sim()
        loop = _loop_with_sim(sim)

        loop._on_physical_input("E")

        assert loop.user_takeover_active is True
        assert loop._takeover.held == frozenset({"E"})

    def test_executor_finally_does_not_release_the_users_key(self) -> None:
        """★★ 执行层的 finally 也一样。

        让权时执行层抛 ``_UserTakeover`` -> outcome=REJECTED ->
        finally 里松键。那次松键**同样**不能碰用户按着的键 ——
        否则用户的长按还是被掐断（只是换了个地方掐）。
        """
        backend = DryRunBackend()
        sim = InputSimulator(backend=backend)
        sim.set_physical_hold_provider(lambda: {int(VK.E)})
        guard = SafetyGuard(SafetyConfig())
        ex = IntentExecutor(simulator=sim, guard=guard, dry_run=True)

        _physical_press(backend, VK.E)
        guard.note_user_input("E", 1.0, still_held=True)

        res = ex.execute(
            IntentPacket.make(ActionType.USE_SKILL, state_version=1),
            _state(),
            now_ms=int(time.time() * 1000),
        )

        assert res.outcome.value == "REJECTED", res.reason
        assert "USER_TAKEOVER" in res.reason, res.reason
        assert _key_ups(backend, VK.E) == [], "执行层把用户的 E 松了！"
        assert backend.is_key_down(int(VK.E))

    def test_executor_finally_still_releases_the_engines_own_keys(self) -> None:
        """★ 底线不变：让权时引擎自己的键**照旧全松**（不能卡住）。"""
        backend = DryRunBackend()
        sim = InputSimulator(backend=backend)
        sim.set_physical_hold_provider(lambda: {int(VK.E)})
        guard = SafetyGuard(SafetyConfig())
        ex = IntentExecutor(simulator=sim, guard=guard, dry_run=True)

        _physical_press(backend, VK.E)
        sim.key_down(VK.W)                    # 引擎自己按着 W
        guard.note_user_input("E", 1.0, still_held=True)

        ex.execute(
            IntentPacket.make(ActionType.USE_SKILL, state_version=1),
            _state(),
            now_ms=int(time.time() * 1000),
        )

        assert _key_ups(backend, VK.W), "引擎自己的 W 没被松 —— 会卡住"
        assert not backend.is_key_down(int(VK.W))


# ==================================================================
# 4 · 终端提示
# ==================================================================

def test_takeover_notice_tells_the_user_how_to_get_control_back(
    caplog) -> None:
    """★ 提示要写清「按了哪个键」和「多久恢复」。

    动态让权下用户**还按着**时引擎一直让权 —— 只说「已让权」的话，
    用户以为按一下就行，于是他一直按着鼠标，引擎就一直不动。
    """
    sim, _ = _sim()
    loop = _loop_with_sim(sim)

    with caplog.at_level(logging.WARNING, logger="autopilot.core.loop"):
        loop._on_physical_input("E")

    texts = [r.getMessage() for r in caplog.records
             if "[提示]" in r.getMessage()]
    assert texts, "让权时必须有一条终端提示"
    msg = texts[0]
    assert "你按下了 E" in msg, msg
    assert "引擎已让权" in msg, msg
    assert "松开按键 1 秒后恢复" in msg, msg


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
