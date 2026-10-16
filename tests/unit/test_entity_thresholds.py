"""实体检测的分档降级（2026-10-14）。

★★ 这组测试守护的是一条**真机教训**
  同一台机器，DirectML 实测 p50 在 **93 ~ 128 ms** 之间漂。而阈值写死成
  120 时，128 会把它**整个关掉** —— 关掉的代价是 WORLD 场景下唯三能触发的
  规则全都依赖 ``has_nearby_enemy()``，于是整条 WORLD 决策链哑掉，
  现场现象只是「卡片没动作」，排查时根本联想不到耗时阈值。
  **demo 就卡在这里两次。**

  核心原则：**宁可慢，也不要没有。**
"""

from __future__ import annotations

import pytest

from autopilot.perception import world_entities as W

pytestmark = pytest.mark.unit


# ==================================================================
# 分档
# ==================================================================

def test_tiers__fast_gets_every_tick() -> None:
    assert W.recommended_interval_ticks(10.0) == 1
    assert W.recommended_interval_ticks(29.9) == 1


def test_tiers__middle_gets_five_ticks() -> None:
    assert W.recommended_interval_ticks(30.0) == W.DEFAULT_ENTITY_INTERVAL_TICKS
    assert W.recommended_interval_ticks(150.0) == W.DEFAULT_ENTITY_INTERVAL_TICKS


def test_tiers__over_threshold_degrades_to_slow_not_off() -> None:
    """★★ 本次的核心改动。

    超阈值 **≠ 关闭**。降频到每 10 tick 但**继续跑**，
    因为关掉的代价（WORLD 决策链全哑）远大于慢的代价。
    """
    over = W.ENTITY_ENABLE_THRESHOLD_MS
    assert W.recommended_interval_ticks(over + 1.0) == \
        W.ENTITY_SLOW_INTERVAL_TICKS
    assert W.ENTITY_SLOW_INTERVAL_TICKS > W.DEFAULT_ENTITY_INTERVAL_TICKS


def test_tiers__only_way_over_shuts_it_down() -> None:
    """> 1.5 倍才认为「确实慢到不能用」。"""
    off_at = W.ENTITY_ENABLE_THRESHOLD_MS * W.ENTITY_SLOWDOWN_FACTOR
    assert W.recommended_interval_ticks(off_at - 0.1) == \
        W.ENTITY_SLOW_INTERVAL_TICKS
    assert W.recommended_interval_ticks(off_at + 0.1) == 0


def test_tiers__monotonic_non_decreasing_interval() -> None:
    """★ 耗时越高，间隔只能**更长** —— 不会出现「更慢反而跑得更勤」。"""
    seq = [W.recommended_interval_ticks(float(m))
           for m in (5, 25, 50, 199, 250, 400)]
    non_zero = [v for v in seq if v]
    assert non_zero == sorted(non_zero), seq
    assert seq[-1] == 0, "只有明显超预算才该彻底关掉"


# ==================================================================
# 阈值取值
# ==================================================================

def test_threshold__default_covers_the_observed_volatility() -> None:
    """★ 默认值必须**盖住真实观测到的全部波动**。

    已知本机实测过 93ms 和 128ms。默认 200 就是为这个 ——
    写死的阈值被实测值骑过去，是这轮两次 demo 失败的共同原因。
    """
    observed = (93.0, 128.0)
    assert max(observed) < W.ENTITY_ENABLE_THRESHOLD_MS
    assert W.ENTITY_ENABLE_THRESHOLD_MS <= 200.0


def test_threshold__real_observations_land_in_the_slow_tier_not_off() -> None:
    """★★ 把两次真实 demo 的实测值代进去，都该是「降频跑」而不是「关掉」。"""
    for ms in (93.0, 128.0):
        got = W.recommended_interval_ticks(ms)
        assert got == W.DEFAULT_ENTITY_INTERVAL_TICKS, (ms, got)


def test_threshold__env_override_still_wins(monkeypatch) -> None:
    """★ 环境变量优先 —— 用户可以按自己机器压回来。"""
    monkeypatch.setenv("NEKO_ENTITY_THRESHOLD_MS", "50")
    got = W._env_positive_float("NEKO_ENTITY_THRESHOLD_MS", 200.0)
    assert got == 50.0


@pytest.mark.parametrize("bad", ["", "  ", "abc", "0", "-10"])
def test_threshold__bad_env_falls_back_to_default(monkeypatch, bad) -> None:
    """★ 非法环境变量退回默认，而不是让整个实体检测带病运行。"""
    monkeypatch.setenv("NEKO_ENTITY_THRESHOLD_MS", bad)
    assert W._env_positive_float("NEKO_ENTITY_THRESHOLD_MS", 200.0) == 200.0


def test_threshold__slowdown_factor_is_sane() -> None:
    """★ 容忍倍数不能太宽（否则 1 秒的东西能拖到 3 秒还没关）。
    也不能 <= 1（那等于没有降级档）。"""
    assert 1.0 < W.ENTITY_SLOWDOWN_FACTOR <= 2.0


def test_threshold__slow_interval_is_slower_than_normal() -> None:
    assert W.ENTITY_SLOW_INTERVAL_TICKS > W.DEFAULT_ENTITY_INTERVAL_TICKS >= 1
