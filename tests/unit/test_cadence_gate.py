"""连击节奏门单测（2026-10-16，B1）。

★ 为什么这个文件存在
-------------------
真机证据：``20261014_215401`` 那次跑出 **355 次普攻 / 76.8 秒**，
均值 0.217s、**标准差仅 0.089s**。那个标准差就是问题本身 ——
真人打连击不可能节拍器一样准。均值 0.217s 是**正常的**（5Hz tick 决定），
要改的是方差。

所以本文件的核心断言不是"平均变慢了"，而是 **间隔分布的离散度显著变大**。
一个只把均值抬上去的实现也能过"变慢了"这种弱断言，但观感不会变好。

★ 为什么门放在 LocalPolicyTree 而不是 PolicyContext
--------------------------------------------------
``PolicyContext`` 每个 tick 由 ``from_state()`` 重建，实例变量活不过一轮
（这条已作为 F22 路线被证伪）。跨 tick 的记忆必须住在
``LocalPolicyTree`` 上 —— 和 ``_pick_marks`` / ``_last_fire_ms`` 同一个理由。
"""

from __future__ import annotations

import random
import statistics

from autopilot.actuator.humanizer import (CadenceConfig, CadenceGate,
                                           DEFAULT_GATED_ACTIONS)

TICK_MS = 200


# ==================================================================
# 纯门逻辑
# ==================================================================

def test_first_shot_is_never_delayed() -> None:
    """从没打过 -> 第一下立刻放行。

    暖机如果也要等，下一次启动的第一刀会慢半拍，观感上像"卡了一下"。
    """
    g = CadenceGate(rng=random.Random(1))
    assert g.allows("NORMAL_ATTACK", 0) is True


def test_allows_is_pure__does_not_advance_state() -> None:
    """★ 钉住两段式 API：判断本身不推进状态。

    如果 ``allows`` 自己记账，那么"条件为假、没真发出"的那几轮也会
    白白消耗掉一次机会 —— 敌人一消失，下次打出来就莫名多等一个间隔。
    """
    g = CadenceGate(rng=random.Random(1))
    for _ in range(50):
        g.allows("NORMAL_ATTACK", 0)
    assert g.next_due_ms("NORMAL_ATTACK") is None, (
        "只调 allows 不该留下任何记账")


def test_note_fired_then_blocks_until_due() -> None:
    g = CadenceGate(rng=random.Random(1))
    g.note_fired("NORMAL_ATTACK", 1_000)
    due = g.next_due_ms("NORMAL_ATTACK")
    assert due is not None and due > 1_000
    assert g.allows("NORMAL_ATTACK", due - 1) is False
    assert g.allows("NORMAL_ATTACK", due) is True


def test_gap_lands_inside_the_configured_band() -> None:
    """抽出来的间隔必须落在配置区间内（含喘口气那一档）。"""
    cfg = CadenceConfig(fatigue_p=0.0, lo_ms=0, hi_ms=300)
    g = CadenceGate(cfg, rng=random.Random(7))
    for i in range(300):
        t = i * 10_000
        g.note_fired("NORMAL_ATTACK", t)
        gap = g.next_due_ms("NORMAL_ATTACK") - t
        assert 0 <= gap <= 300, f"间隔 {gap} 越界"


def test_fatigue_draws_land_in_the_long_band() -> None:
    """fatigue_p=1.0 时**每次**都该抽到长间隔那一档。"""
    cfg = CadenceConfig(fatigue_p=1.0,
                        fatigue_lo_ms=400, fatigue_hi_ms=900)
    g = CadenceGate(cfg, rng=random.Random(3))
    for i in range(50):
        t = i * 10_000
        g.note_fired("NORMAL_ATTACK", t)
        gap = g.next_due_ms("NORMAL_ATTACK") - t
        assert 400 <= gap <= 900, f"间隔 {gap} 不在喘气档"


# ==================================================================
# 作用域：只管普攻，绝不碰冷却类动作
# ==================================================================

def test_only_normal_attack_is_gated_by_default() -> None:
    """★ 「不要影响功能」的第一道闸：默认只管普攻。"""
    assert DEFAULT_GATED_ACTIONS == frozenset({"NORMAL_ATTACK"})


def test_ungated_actions_are_never_blocked() -> None:
    """闪避/技能/移动不受节奏门约束。

    它们自带冷却或本来就是持续按住，再加一道门只会平白增加延迟。
    """
    g = CadenceGate(rng=random.Random(1))
    for action in ("DODGE", "USE_SKILL", "USE_BURST", "MOVE_FORWARD",
                   "SWITCH_CHARACTER", "JUMP", "SPRINT"):
        assert g.governs(action) is False, f"{action} 不该被节奏门管辖"
        assert g.allows(action, 0) is True


def test_gating_one_action_does_not_affect_another() -> None:
    """普攻被挡时，闪避照发。

    用 ``lo_ms == hi_ms`` 的确定性区间，避免抽到 0 让断言变得随机。
    """
    cfg = CadenceConfig(fatigue_p=0.0, lo_ms=250, hi_ms=250)
    g = CadenceGate(cfg, rng=random.Random(1))
    g.note_fired("NORMAL_ATTACK", 1_000)
    assert g.allows("NORMAL_ATTACK", 1_100) is False
    assert g.allows("DODGE", 1_100) is True


def test_disabled_config_lets_everything_through() -> None:
    """kill switch：enabled=False 时门完全不存在（急停用）。"""
    g = CadenceGate(CadenceConfig(enabled=False), rng=random.Random(1))
    g.note_fired("NORMAL_ATTACK", 1_000)
    assert g.allows("NORMAL_ATTACK", 1_001) is True


# ==================================================================
# 分布形状 —— B1 的真正判据
# ==================================================================

def _simulate_intervals(gate: CadenceGate, *, ticks: int,
                        start_ms: int = 0) -> list[int]:
    """按 200ms tick 跑一遍门，返回实际发招间隔序列。

    这是真实节奏的忠实模拟：主循环每 tick 问一次门，问到就发。
    """
    fired: list[int] = []
    now = start_ms
    for _ in range(ticks):
        now += TICK_MS
        if gate.allows("NORMAL_ATTACK", now):
            fired.append(now)
            gate.note_fired("NORMAL_ATTACK", now)
    return [b - a for a, b in zip(fired, fired[1:])]


def test_uniform_gate_produces_a_materially_wider_distribution() -> None:
    """★ B1 的核心断言：标准差必须显著变大。

    没有门时，5Hz tick + 每 tick 都命中 = 间隔恒等于 200ms，标准差 0。
    加门之后间隔在 200/400/600… 之间跳，标准差必然 > 0。

    这条比"平均变慢了"强得多：一个只会抬高均值的实现照样过不了它。
    """
    plain = _simulate_intervals(
        CadenceGate(CadenceConfig(enabled=False), rng=random.Random(0)),
        ticks=400)
    human = _simulate_intervals(
        CadenceGate(rng=random.Random(0)), ticks=400)

    assert len(human) >= 100, "样本太少，统计没意义"
    plain_sd = statistics.pstdev(plain)
    human_sd = statistics.pstdev(human)

    assert plain_sd == 0.0, "对照组本该是恒定 200ms（节拍器）"
    assert human_sd > 30, (
        f"标准差只有 {human_sd:.1f}ms —— 间隔分布还是太规整，"
        "B1 没真的解决")


def test_human_cadence_does_not_collapse_the_dps() -> None:
    """★ 反向守卫：不能把攻击频率砍成一半。

    这是「不要影响功能」的第二道闸。门只该改变**形状**，
    不该让普攻变得明显更稀 —— 真人打本时也不至于一分钟才补一刀。
    """
    plain = _simulate_intervals(
        CadenceGate(CadenceConfig(enabled=False), rng=random.Random(0)),
        ticks=400)
    human = _simulate_intervals(
        CadenceGate(rng=random.Random(0)), ticks=400)

    ratio = statistics.mean(human) / statistics.mean(plain)
    assert 1.0 <= ratio <= 1.6, (
        f"平均间隔是原来的 {ratio:.2f} 倍 —— 超出 [1.0, 1.6] 说明"
        "节奏门把攻击频率压得太狠了")


def test_cadence_never_produces_a_burst() -> None:
    """★ 不许出现「连着两下之间没有间隔」。

    门只推迟、不提前：`note_fired(now)` 之后最早也是 `now + lo_ms`。
    真出现 0 间隔说明记账逻辑写反了。
    """
    intervals = _simulate_intervals(
        CadenceGate(rng=random.Random(11)), ticks=600)
    assert min(intervals) >= TICK_MS, f"出现了 {min(intervals)}ms 的连击"


def test_cadence_survives_a_long_idle_gap() -> None:
    """停了十分钟再打：第一下必须立刻放行，不能补一刀冷却。"""
    g = CadenceGate(rng=random.Random(1))
    g.note_fired("NORMAL_ATTACK", 1_000)
    later = 1_000 + 10 * 60 * 1000
    assert g.allows("NORMAL_ATTACK", later) is True


# ==================================================================
# 参数校验
# ==================================================================

def test_inverted_band_raises_instead_of_silently_swapping() -> None:
    """区间写反时宁可抛，也不要静默对调。

    静默对调出来的分布看起来完全正常，只是方向错了 —— 那种 bug
    要跑到真机才看得出来。
    """
    import pytest
    with pytest.raises(ValueError):
        CadenceConfig(lo_ms=350, hi_ms=200)
    with pytest.raises(ValueError):
        CadenceConfig(fatigue_p=1.5)


def test_describe_mentions_the_governed_actions() -> None:
    """诊断串必须说清管辖了谁 —— 否则出问题没法一眼定位。"""
    assert "NORMAL_ATTACK" in CadenceGate().describe()
    assert "关闭" in CadenceGate(CadenceConfig(enabled=False)).describe()
