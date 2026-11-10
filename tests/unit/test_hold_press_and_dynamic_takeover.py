"""长按技能（钟离护盾）+ 动态物理抢占（2026-10-19）。

★ 为什么要有这个文件

两件事是**配套**的，拆开看都像小事，合起来就是实机事故：

1. **长按 E**：钟离的护盾必须长按约 1.5 秒才出得来，点按只会放柱子。
   所以连招里需要 ``hold_ms = 1500``。

2. **抢占**：用户在战斗中随时要接管。旧实现是「一按就让权 3 秒」——
   用户点一下走位要干等 3 秒（割裂），而按住 W 走 5 秒时 3 秒窗口一过
   引擎就来抢方向（抢不动）。

★ 长按把风险放大了 50 倍

点按 E 是 30ms 的按住；长按 E 是 1500ms。这 1500ms 里执行层是**阻塞**的：
· 用户在这期间按的键，会被引擎按着的 E 顶掉 —— **物理抢占失效**；
· 急停 F12 也要等 1.5 秒才生效。

所以长按**必须可中断**：``key_press(abort_check=...)`` 让用户抢占和急停
在 20ms 内就能把键松开。这是本文件最重要的一条。

★ 抖动会毁掉长按

``humanizer`` 给按键时长加 ±15% 抖动。点按抖 ±15% 无所谓，
但 1500ms 抖成 1275ms —— **差一点就放不出护盾**，
而症状是「按了没反应」，日志里一条错都没有。
所以显式写的 ``hold_ms`` **不抖**。
"""

from __future__ import annotations

import pathlib
import sys
import time

import pytest

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.executor import IntentExecutor  # noqa: E402
from autopilot.actuator.safety import SafetyConfig, SafetyGuard  # noqa: E402
from autopilot.actuator.simulator import (  # noqa: E402
    DryRunBackend,
    InputSimulator,
    VK,
)
from autopilot.contracts.intent_packet import ActionType, IntentPacket  # noqa: E402
from autopilot.contracts.state_packet import (  # noqa: E402
    SCENE_WORLD,
    PlayerInfo,
    ResourceInfo,
    SceneInfo,
    StatePacket,
)

pytestmark = pytest.mark.unit

COMBOS_TOML = PLUGIN_ROOT / "autopilot" / "combos" / "combos.toml"


def _state() -> StatePacket:
    return StatePacket(
        scene=SceneInfo(id=SCENE_WORLD, confidence=0.9),
        player=PlayerInfo(party=(), hp_ratio=1.0, hp_source="unit"),
        resource=ResourceInfo(perception_ok=True),
        state_version=1,
    )


def _executor(guard: SafetyGuard | None = None) -> IntentExecutor:
    return IntentExecutor(
        dry_run=True,
        guard=guard or SafetyGuard(SafetyConfig()),
    )


def _skill_intent(hold_ms: int | None = None) -> IntentPacket:
    params = {} if hold_ms is None else {"hold_ms": hold_ms}
    return IntentPacket.make(ActionType.USE_SKILL, params=params, state_version=1)


# ==================================================================
# 1 · 长按：时长必须**分毫不差**
# ==================================================================

def test_explicit_hold_ms_is_not_jittered() -> None:
    """★ 显式写的 hold_ms 是**游戏机制阈值**，不抖。

    humanizer 的 ±15% 会把 1500ms 抖成 1275ms —— 差一点就放不出护盾。
    """
    ex = _executor()
    seen: list[int] = []
    real = ex.sim.key_press

    def spy(vk, hold_ms=30, abort_check=None):
        seen.append(hold_ms)
        return real(vk, hold_ms=hold_ms, abort_check=abort_check)

    ex.sim.key_press = spy          # type: ignore[method-assign]
    ex.execute(_skill_intent(1500), _state(), now_ms=int(time.time() * 1000))

    assert seen == [1500], f"长按时长被改了：{seen}"


def test_explicit_hold_ms_is_stable_across_repeats() -> None:
    """★ 跑 5 次都必须是 1500 —— 抖动是随机的，单次断言会漏。"""
    ex = _executor()
    seen: list[int] = []
    real = ex.sim.key_press

    def spy(vk, hold_ms=30, abort_check=None):
        seen.append(hold_ms)
        return real(vk, hold_ms=hold_ms, abort_check=abort_check)

    ex.sim.key_press = spy          # type: ignore[method-assign]
    for _ in range(5):
        ex.execute(_skill_intent(1500), _state(), now_ms=int(time.time() * 1000))

    assert seen == [1500] * 5, f"长按时长不稳定：{seen}"


def test_default_tap_still_uses_the_humanizer() -> None:
    """没写 hold_ms 时**照旧**走 humanizer —— 点按的抖动是有意的拟人化。"""
    ex = _executor()
    seen: list[int] = []
    real = ex.sim.key_press

    def spy(vk, hold_ms=30, abort_check=None):
        seen.append(hold_ms)
        return real(vk, hold_ms=hold_ms, abort_check=abort_check)

    ex.sim.key_press = spy          # type: ignore[method-assign]
    ex.execute(_skill_intent(), _state(), now_ms=int(time.time() * 1000))

    assert len(seen) == 1
    assert 20 <= seen[0] <= 45, f"点按时长不该被固定：{seen[0]}"


def test_the_key_really_stays_down_for_the_whole_hold() -> None:
    """★ 「按住」是真的按住 —— down 和 up 之间必须有真实间隔。

    只看「调用了 key_press(hold_ms=1500)」是不够的：
    真正的证据是**两次注入之间的时间差**。
    """
    sim = InputSimulator(backend=DryRunBackend())
    t0 = time.monotonic()
    sim.key_press(VK.E, hold_ms=300)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms >= 300 * 0.9, f"没按住那么久：{elapsed_ms:.0f}ms"


# ==================================================================
# 2 · 长按不能被超时闸误杀
# ==================================================================

def test_long_press_is_not_reported_as_timeout() -> None:
    """★ 明明按满了却报 TIMEOUT —— 症状是「盾没上出来」，日志只有一条超时。

    USE_SKILL 的 max_duration_ms 是按「点一下」标的 2000ms，
    长按 1500 + 空档 120 = 1620ms，只剩 380ms 余量。
    """
    ex = _executor()
    res = ex.execute(_skill_intent(400), _state(), now_ms=int(time.time() * 1000))
    assert res.outcome.value == "OK", f"长按被判超时了：{res.reason}"


def test_effective_max_duration_covers_hold_plus_gap() -> None:
    """★ 显式长按要求的时间必须算进超时预算。"""
    from autopilot.actuator.traits import traits_of

    t = traits_of("USE_SKILL")
    intent = _skill_intent(1500)
    eff = IntentExecutor._effective_max_duration_ms(intent, t)
    assert eff >= 1500, f"有效上限没算进长按：{eff}"


def test_default_actions_keep_their_own_budget() -> None:
    """没写 hold_ms 时，**不**该放宽上限（否则超时就形同虚设）。"""
    from autopilot.actuator.traits import traits_of

    t = traits_of("USE_SKILL")
    eff = IntentExecutor._effective_max_duration_ms(_skill_intent(), t)
    assert eff == t.max_duration_ms


# ==================================================================
# 3 · ★ 长按必须可中断（这是长按能安全存在的前提）
# ==================================================================

def test_long_press_is_aborted_by_user_takeover() -> None:
    """★ 用户在长按期间抢占 -> 键立刻松开，这一步判为让权拒绝。

    没有中断闸的话：引擎按住 E 1.5 秒，用户按的 E 被顶掉，
    **物理抢占直接失效**。
    """
    ex = _executor()
    sim = ex.sim
    raised = {"n": 0}

    def abort_check():
        raised["n"] += 1
        if raised["n"] >= 2:            # 第 2 次轮询（约 40ms）就打断
            raise RuntimeError("用户抢占了")

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        sim.key_press(VK.E, hold_ms=3000, abort_check=abort_check)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms < 1000, f"中断太慢了：{elapsed_ms:.0f}ms"
    # 键一定被抬起了（DryRunBackend 记 down/up 成对）
    downs = [e for e in sim.backend.events if e.down]
    ups = [e for e in sim.backend.events if not e.down]
    assert downs and ups, "中断后键必须抬起"


def test_long_press_does_not_leave_the_key_down_on_abort() -> None:
    """★ 中断后键**一定**是抬起的 —— 不抬起的话游戏会以为还按着。"""
    sim = InputSimulator(backend=DryRunBackend())

    def abort_check():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        sim.key_press(VK.E, hold_ms=2000, abort_check=abort_check)

    assert sim.backend.pressed_keys == set(), f"键还按着：{sim.backend.pressed_keys}"


def test_takeover_during_long_press_rejects_the_action() -> None:
    """★ 从执行层这一侧看：长按被抢占打断 -> REJECTED(USER_TAKEOVER)。

    ★ 而且**不**计失败 —— 让权不是故障，不该推进熔断计数。
    """
    guard = SafetyGuard(SafetyConfig())
    ex = _executor(guard)
    real = ex.sim.key_press

    def interrupting(vk, hold_ms=30, abort_check=None):
        """模拟「按下去之后用户抢了」—— 让权在长按途中生效。"""
        guard.note_user_input("E", 1.0, still_held=True)
        if abort_check is not None:
            abort_check()          # 立刻抛 _UserTakeover
        return real(vk, hold_ms=hold_ms)

    ex.sim.key_press = interrupting      # type: ignore[method-assign]
    res = ex.execute(_skill_intent(1500), _state(), now_ms=int(time.time() * 1000))

    assert res.outcome.value == "REJECTED", res.reason
    assert "USER_TAKEOVER" in res.reason, res.reason
    assert guard.consecutive_failures == 0, "让权不该计入失败"


# ==================================================================
# 4 · 连招配置
# ==================================================================

def test_zhongli_step_uses_a_1500ms_hold() -> None:
    """★ 演示连招里钟离那一步必须真的写着 hold_ms = 1500。"""
    from autopilot.policy.combo_manager import load_combos

    book = load_combos(COMBOS_TOML)
    seq = next(s for s in book if s.name == "钟纳夜丝_全自动演示")
    step = next(s for s in seq.steps
                if s.action is ActionType.USE_SKILL and s.char == "钟离")

    assert step.hold_ms == 1500, f"钟离的长按时长不对：{step.hold_ms}"
    assert step.params()["hold_ms"] == 1500
    assert step.expected_exec_ms == 1500


def test_combo_step_expected_exec_ms_counts_hold_and_gap() -> None:
    """★ wait="ready" 的期限要算上这一步自己花的时间。"""
    from autopilot.policy.combo_manager import ComboStep

    step = ComboStep(action=ActionType.USE_SKILL, hold_ms=1500, gap_ms=120)
    assert step.expected_exec_ms == 1620


def test_combo_step_expected_exec_ms_is_zero_for_taps() -> None:
    from autopilot.policy.combo_manager import ComboStep

    assert ComboStep(action=ActionType.USE_SKILL).expected_exec_ms == 0
