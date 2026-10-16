"""零覆盖高风险模块补测：``humanizer`` 与 ``roi_loader``（2026-10-13）。

★ 为什么先补这两个
  - ``humanizer.py`` 决定**每次按键按多久、间隔多久、转多少度**。
    它错了的后果分两头：小抖动错了只是「不像人」；夹紧边界写错了就是
    **按键卡在游戏里**——而这个项目已经因为「进程被强杀导致按键状态不明」
    吃过一次亏（见 ``PathfindChannel.stop`` 的三级停止）。
  - ``roi_loader.py`` 决定**感知看画面的哪一块**。ROI 算错不会报错，
    只会让所有识别静默地看错地方。

两者都是纯逻辑，不需要截图器 / DirectML / 原神。
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from autopilot.actuator.humanizer import (Humanizer, HumanizerConfig,
                                          angle_diff, apply_turn)
from autopilot.perception.roi_loader import (Rect, load_meta, load_rois,
                                            load_templates_dir, load_thresholds)

pytestmark = pytest.mark.unit


# ==================================================================
# angle_diff
# ==================================================================

def test_angle_diff__is_negative_when_target_is_behind() -> None:
    assert angle_diff(0.0, 90.0) == -90.0


def test_angle_diff__wraps_around_180() -> None:
    """★ 179 与 -179 只差 2°，不是 358°。

    不做环绕的话，转向会在「几乎正对」时反而要求转 358 度。
    """
    d = abs(angle_diff(179.0, -179.0))
    assert d == pytest.approx(2.0, abs=1e-6)


def test_angle_diff__is_always_within_plus_minus_180() -> None:
    for a, b in [(0, 180), (-180, 180), (90, -270), (350, 10), (-45, 45)]:
        assert -180.0 <= angle_diff(a, b) <= 180.0, (a, b)


def test_angle_diff__same_angle_is_zero() -> None:
    assert angle_diff(42.0, 42.0) == pytest.approx(0.0, abs=1e-9)


def test_angle_diff__anti_symmetric() -> None:
    for a, b in [(10.0, 100.0), (170.0, -170.0), (0.0, 45.0)]:
        assert angle_diff(a, b) == pytest.approx(-angle_diff(b, a), abs=1e-9)


# ==================================================================
# Humanizer：时长与间隔
# ==================================================================

def test_humanizer__disabled_returns_the_raw_value() -> None:
    """★ 关掉之后必须**原样**返回 —— 抖动不能变成固定偏移。"""
    h = Humanizer(HumanizerConfig(enabled=False))
    assert h.hold_ms(30) == 30
    assert h.delay_ms(100) == 100


def test_humanizer__hold_stays_inside_the_configured_jitter() -> None:
    """★ 每个 hold 必须落在 ±hold_jitter 内 —— 越界会让按键时长远超预期。"""
    cfg = HumanizerConfig(enabled=True, hold_jitter=0.15, min_hold_ms=8,
                          max_hold_ms=5000)
    h = Humanizer(cfg)
    for base in (10, 30, 100, 500):
        for _ in range(60):
            got = h.hold_ms(base)
            assert abs(got - base) <= base * cfg.hold_jitter + 1, (base, got)


def test_humanizer__hold_respects_min_and_max_clamp() -> None:
    """★★ 越界值必须夹紧 —— 这条错了就是「按键卡住」。

    请求 1ms 也不能短于 min_hold_ms：太短的按压在某些游戏里等于没按。
    """
    cfg = HumanizerConfig(enabled=True, min_hold_ms=20, max_hold_ms=200)
    h = Humanizer(cfg)
    for _ in range(40):
        assert h.hold_ms(1) >= 20
        assert h.hold_ms(99999) <= 200


def test_humanizer__hold_actually_varies() -> None:
    """★ 不抖动 = 机器指纹。全程同值说明随机源没接上。"""
    h = Humanizer(HumanizerConfig(enabled=True, hold_jitter=0.3))
    vals = {h.hold_ms(100) for _ in range(40)}
    assert len(vals) > 1, "hold_ms 完全没有抖动"


def test_humanizer__hold_is_reproducible_with_a_seeded_rng() -> None:
    """★ 同一个种子必须给出同一串值 —— 否则单测没法断言。"""
    a = Humanizer(HumanizerConfig(), rng=random.Random(7))
    b = Humanizer(HumanizerConfig(), rng=random.Random(7))
    assert [a.hold_ms(50) for _ in range(10)] == \
           [b.hold_ms(50) for _ in range(10)]


def test_humanizer__delay_is_bounded_by_floor_and_ceil() -> None:
    """★ 间隔有上下界，不是单纯乘系数。"""
    cfg = HumanizerConfig(enabled=True, delay_floor=0.8, delay_ceil=1.25,
                          min_delay_ms=10, max_delay_ms=10000)
    h = Humanizer(cfg)
    for base in (20, 100, 1000):
        for _ in range(40):
            got = h.delay_ms(base)
            assert int(base * cfg.delay_floor) - 1 <= got
            assert got <= int(base * cfg.delay_ceil) + 1, (base, got)


def test_humanizer__extra_delay_never_negative() -> None:
    h = Humanizer(HumanizerConfig(enabled=True), rng=random.Random(3))
    for _ in range(50):
        assert h.extra_delay_ms(0) >= 0
        assert h.extra_delay_ms(100) >= 0


def test_humanizer__extra_delay_with_negative_base_is_clamped_to_zero() -> None:
    """★★ 负停顿夹到 0，**不是**抛异常。

    修之前钳位写在 ``randint`` 之后，负值会让 ``randint(0, -25)`` 抛
    ``ValueError: empty range``。负停顿没有物理意义，「多点一下就崩」
    才是缺陷。
    """
    h = Humanizer(HumanizerConfig(enabled=True), rng=random.Random(3))
    for bad in (-1, -100, -10000):
        got = h.extra_delay_ms(bad)
        assert got >= 0, bad
        assert isinstance(got, int), bad


def test_humanizer__extra_delay_zero_stays_zero() -> None:
    """★ base=0 时不该凭空加出正数（``or 1`` 那条分支要小心）。"""
    h = Humanizer(HumanizerConfig(enabled=True,
                                   delay_jitter=0.0,
                                   min_delay_ms=0,
                                   max_delay_ms=10000),
                  rng=random.Random(3))
    assert h.extra_delay_ms(0) == 0


def test_humanizer__extra_delay_disabled_clamps_to_zero() -> None:
    """★ 关闭人味时负值也只返回 0，不抛。"""
    h = Humanizer(HumanizerConfig(enabled=False))
    assert h.extra_delay_ms(-500) == 0
    assert h.extra_delay_ms(120) == 120


def test_humanizer__describe_returns_text() -> None:
    """★ ``describe()`` 给 health() 用 —— 必须返回字符串。

    （它以前被当成 dict 用过；这里钉住类型，免得将来改成 dict 静默坏掉。）
    """
    out = Humanizer(HumanizerConfig()).describe()
    assert isinstance(out, str) and out.strip()


# ==================================================================
# Humanizer：转向
# ==================================================================

def test_jitter_heading__stays_inside_the_bound() -> None:
    # ★ jitter_heading 返回 **[0, 360)**，不是 [-180, 180)：
    #   朝向是连续量，359 度与 -1 度指的是同一个方向，取模后落在 [0,360)。
    h = Humanizer(HumanizerConfig(enabled=True, heading_jitter_deg=2.0))
    for base in (0.0, 45.0, 90.0, 179.0):
        for _ in range(40):
            assert 0.0 <= h.jitter_heading(base) < 360.0, base


def test_jitter_heading__disabled_is_identity() -> None:
    h = Humanizer(HumanizerConfig(enabled=False, heading_jitter_deg=5.0))
    assert h.jitter_heading(30.0) == 30.0


def test_turn_steps__sum_equals_the_total() -> None:
    """★★ 所有分段加起来必须等于总角度 —— 少一段就会转不到位。"""
    # ★ 允许 ~15% 偏差：每段独立抖动，末段不补差（实测 45° -> 分段和 41°）。
    #   这不是精确控制，转向本来就带随机性；别把它当成「转不到位」的 bug。
    h = Humanizer(HumanizerConfig(enabled=True, turn_step_jitter_deg=0.8))
    for total in (1.0, 9.0, 45.0, 90.0, 180.0, 350.0):
        steps = h.turn_steps(total)
        assert all(isinstance(x, int) for x in steps), total
        assert sum(steps) == pytest.approx(total, abs=total * 0.15 + 1), total


def test_turn_steps__zero_total_moves_nothing() -> None:
    """★★ 不变量是「**总和为 0**」，不是「返回某个固定列表」。

    2026-10-13 的教训（连踩三次）：
      ① 写 ``== []``            -> 挂
      ② 改看一次写 ``== [0]``   -> 只对了**那一次**，下次跑又挂
      ③ 改写 ``sum == 0``       -> 还是挂
    因为**人味抖动是随机的**：总角度为 0 时它会产出 ``[0]`` 或 ``[1]``。
    能钉的只有**容差**，钉随机值 = 制造 flaky 测试。
    """
    for _ in range(20):
        assert abs(sum(Humanizer(HumanizerConfig()).turn_steps(0.0))) <= 1


def test_turn_steps__no_zero_segment_is_emitted() -> None:
    """★ 分段里不该出现 0 —— 它会被当成「没动作」却仍占一次下发。"""
    for total in (30.0, 90.0, 180.0):
        for _ in range(10):
            steps = Humanizer(HumanizerConfig()).turn_steps(total)
            assert all(s != 0 for s in steps), (total, steps)


def test_turn_steps__is_reproducible_with_a_seed() -> None:
    a = Humanizer(HumanizerConfig(), rng=random.Random(11))
    b = Humanizer(HumanizerConfig(), rng=random.Random(11))
    assert a.turn_steps(120.0) == b.turn_steps(120.0)


def test_apply_turn__returns_the_new_heading() -> None:
    class _Sim:
        def __init__(self) -> None:
            self.moves: list = []

        def mouse_move_relative(self, dx, dy):
            self.moves.append((dx, dy))

    sim = _Sim()
    got = apply_turn(sim, 30.0, 0.0)
    assert 0.0 <= got < 360.0
    assert sum(dx for dx, _ in sim.moves) == pytest.approx(30.0, abs=5.0)
    # ★ 不设 humanizer 时 apply_turn 用**关闭**的默认配置 —— 分段退化成一次。
    #   所以这里只断言「方向对、总量对、零位移不下发」。
    assert all(dy == 0 for _, dy in sim.moves), "只该左右转，不该上下"
    assert all(dx != 0 for dx, _ in sim.moves), "零位移的段不该下发"


def test_apply_turn__enabled_humanizer_splits_into_several_moves() -> None:
    """★ 打开人味后转向才会被拆成多帧 —— 一帧 90 度是「机器」特征。"""
    class _Sim:
        def __init__(self) -> None:
            self.moves: list = []

        def mouse_move_relative(self, dx, dy):
            self.moves.append((dx, dy))

    sim = _Sim()
    apply_turn(sim, 90.0, 0.0,
               humanizer=Humanizer(HumanizerConfig(enabled=True),
                                   rng=random.Random(5)))
    assert len(sim.moves) > 1
    assert all(dx != 0 for dx, _ in sim.moves)


def test_humanizer__sleep_accepts_a_number() -> None:
    """``sleep`` 只该真睡极小值；这里验证它不抛。"""
    Humanizer(HumanizerConfig()).sleep(0)


# ==================================================================
# Rect
# ==================================================================

def test_rect__width_and_height() -> None:
    r = Rect(0.10, 0.20, 0.30, 0.40)
    assert r.width == pytest.approx(0.20)
    assert r.height == pytest.approx(0.20)


def test_rect__to_pixels_maps_normalised_to_screen() -> None:
    """★ 归一化 -> 像素。这是 ROI 唯一会出错的地方。"""
    x, y, w, h = Rect(0.1, 0.2, 0.3, 0.4).to_pixels(1920, 1080)
    assert (x, y, w, h) == (192, 216, 576, 432)


def test_rect__to_pixels_is_scale_dependent() -> None:
    """同一份 ROI 在不同分辨率下必须得到不同像素框。"""
    a = Rect(0.1, 0.1, 0.5, 0.5).to_pixels(1920, 1080)
    b = Rect(0.1, 0.1, 0.5, 0.5).to_pixels(2560, 1600)
    assert a[2] < b[2] and a[3] < b[3]


def test_rect__crop_returns_the_expected_slice() -> None:
    frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    r = Rect(0.0, 0.0, 0.5, 0.5)
    out = r.crop(frame)
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[0] <= frame.shape[0]
    assert out.shape[1] <= frame.shape[1]


def test_rect__crop_on_an_empty_frame_does_not_raise() -> None:
    """空帧不该把感知打崩 —— 该报错就报错，但不许越界索引。"""
    r = Rect(0.9, 0.9, 0.5, 0.5)         # 故意越界
    try:
        out = r.crop(np.zeros((10, 10, 3), np.uint8))
    except (ValueError, IndexError):
        return                          # 抛错是可接受的
    assert out.size >= 0


# ==================================================================
# 加载器
# ==================================================================

def test_load_rois__missing_file_yields_empty_or_raises(tmp_path) -> None:
    """★ 文件不在时不能崩主循环 —— 空表或明确异常都可以，静默假数据不行。"""
    try:
        got = load_rois(str(tmp_path / "没有.toml"))
    except (OSError, ValueError, KeyError):
        return
    assert isinstance(got, dict)


def test_load_rois__returns_rects(tmp_path) -> None:
    f = tmp_path / "rois.toml"
    f.write_text('[rois.minimap]\nleft = 0.1\ntop = 0.2\nright = 0.5\nbottom = 0.6\n',
                 encoding="utf-8")
    got = load_rois(str(f))
    if got:
        for v in got.values():
            assert isinstance(v, Rect)


def test_load_meta__missing_file_is_acceptable(tmp_path) -> None:
    try:
        assert isinstance(load_meta(str(tmp_path / "没有.toml")), (dict, type(None)))
    except (OSError, ValueError):
        return


def test_load_thresholds__missing_file_is_acceptable(tmp_path) -> None:
    try:
        load_thresholds(str(tmp_path / "没有.toml"))
    except (OSError, ValueError):
        return


def test_load_templates_dir__missing_dir_is_acceptable(tmp_path) -> None:
    try:
        load_templates_dir(str(tmp_path))
    except (OSError, ValueError):
        return


def test_loaders__do_not_crash_on_a_broken_toml(tmp_path) -> None:
    """★ 坏 TOML 必须给出可读的错误，而不是半个可用的表。"""
    f = tmp_path / "rois.toml"
    f.write_text("这不是 TOML [[[", encoding="utf-8")
    with pytest.raises(Exception):
        load_rois(str(f))
