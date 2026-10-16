"""导航转向律的单测（2026-10-14）。

★★ 这组测试守护的是**合并寻路的前提**
----------------------------------------
``autopilot/nav/steer.py`` 是把 ``e2e_pathfind.py`` 的转向算法抽出来的
纯函数库。它存在的意义是：**在动架构之前，先让算法可测**。

因为这一段的每一个常数都是真机标定出来的，而且抽出来的过程里最容易
出的错不是「算错」，是**「符号搞反」** —— 而符号搞反的现场表现是
「寻路朝反方向画圈」，肉眼很难第一时间联想到转向律的符号约定。

所以下面这几类用例是**故意挑的**：

* **符号方向**：Δ > 0（目标在左）必须下发**负**像素。
* **滞回不可抖**：进站定 60°、出站定 20°，中间那一大段绝不退出。
* **「不拆」≠「不限幅」**：段数为 1 时饱和阈值**依然**生效。
* **比例控制**：缩小 ``cfg`` 不会改变收敛比例（这是它不是绝对增益的原因）。
* **纯函数红线**：本模块不得引入任何 IO / 时间 / 线程模块。

另外，本组用例**刻意不做**「和 e2e_pathfind.py 同输入同输出对拍」——
那需要把那个 200KB 的脚本当模块导入，它在 import 期就会拉起整条感知栈。
等真正接线时再补对拍单测更合适（那时两边都在可控状态）。
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from autopilot.nav import steer as S
from autopilot.nav.steer import DEFAULT_CONFIG as C

pytestmark = pytest.mark.unit


# ==================================================================
# 纯函数红线
# ==================================================================

_FORBIDDEN_IMPORTS = {
    "os", "sys", "time", "threading", "subprocess", "socket",
    "pathlib", "json", "asyncio", "logging", "random",
}


def test_redline__steer_module_has_no_io_or_time_imports() -> None:
    """★ 本模块的价值全在「纯函数」上，破了这条它就没意义了。

    一旦有人往里塞 ``time.monotonic()`` 或 ``threading``，单测就会开始
    依赖真实时钟/线程调度 —— 而那正是当初非要把它抽出来的原因。
    """
    src = Path(S.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    leaked = imported & _FORBIDDEN_IMPORTS
    assert not leaked, (
        f"steer.py 必须保持纯函数，却引入了 {sorted(leaked)}。"
        f"IO / 时间源 / 线程应该留在调用方（core/loop.py），"
        f"否则这段算法就再也测不稳了。")


# ==================================================================
# 角度工具
# ==================================================================

class TestWrapDeg:
    def test_in_range_passes_through(self) -> None:
        assert S.wrap_deg(0.0) == 0.0
        assert S.wrap_deg(90.0) == 90.0
        assert S.wrap_deg(-90.0) == -90.0

    def test_wraps_across_plus_180(self) -> None:
        assert S.wrap_deg(190.0) == pytest.approx(-170.0)
        assert S.wrap_deg(-190.0) == pytest.approx(170.0)

    def test_180_folds_to_negative_half_open(self) -> None:
        """区间是 ``[-180, 180)`` —— 180 归到 -180，不重复表示。"""
        assert S.wrap_deg(180.0) == pytest.approx(-180.0)

    def test_large_multiples(self) -> None:
        """540° 与 180° 是同一个方向，都归到 -180（区间左闭右开）。"""
        assert S.wrap_deg(540.0) == pytest.approx(-180.0)
        assert S.wrap_deg(-540.0) == pytest.approx(-180.0)
        assert S.wrap_deg(900.0) == pytest.approx(-180.0)


class TestCircMeanDeg:
    def test_needs_at_least_two_samples(self) -> None:
        assert S.circ_mean_deg([]) is None
        assert S.circ_mean_deg([30.0]) is None

    def test_midpoint(self) -> None:
        assert S.circ_mean_deg([0.0, 90.0]) == pytest.approx(45.0)

    def test_across_the_seam_is_not_zero(self) -> None:
        """★ 这就是「不能用算术平均」的全部理由。

        +179 和 -179 几乎是同向，圆周均值应贴近 ±180；
        算术平均会给出 0°，把「几乎没差」误判成「差了一整圈」。
        """
        got = S.circ_mean_deg([179.0, -179.0])
        assert got is not None
        assert abs(got) == pytest.approx(180.0, abs=0.5)

    def test_degenerate_opposite_samples_return_none(self) -> None:
        assert S.circ_mean_deg([0.0, 180.0]) is None


class TestBearingDeg:
    @pytest.mark.parametrize("tx,tz,expected", [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 90.0),
        (-1.0, 0.0, 180.0),
        (0.0, -1.0, -90.0),
        (1.0, 1.0, 45.0),
    ])
    def test_from_origin(self, tx: float, tz: float, expected: float) -> None:
        assert S.bearing_deg(0.0, 0.0, tx, tz) == pytest.approx(expected)

    def test_is_translation_invariant(self) -> None:
        """位置只该通过「相对量」影响方位 —— 平移后角度不变。"""
        a = S.bearing_deg(0.0, 0.0, 3.0, 4.0)
        b = S.bearing_deg(100.0, -7.0, 103.0, -3.0)
        assert a == pytest.approx(b)


class TestHeadingDeg:
    def test_single_segment_is_not_enough(self) -> None:
        """只有 1 个有效段 -> 判不出朝向（必须返回 None，交给调用方）。"""
        assert S.heading_deg([(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)], 1.0, 5.0) \
            is None

    def test_moving_straight_x(self) -> None:
        hist = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]
        assert S.heading_deg(hist, 2.0, 10.0) == pytest.approx(0.0)

    def test_moving_straight_z(self) -> None:
        hist = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 2.0)]
        assert S.heading_deg(hist, 2.0, 10.0) == pytest.approx(90.0)

    def test_window_excludes_old_segments(self) -> None:
        """窗口只留 1 段时 -> None。窗口放宽到覆盖全部 -> 有值。"""
        hist = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]
        assert S.heading_deg(hist, 2.0, 1.0) is None
        assert S.heading_deg(hist, 2.0, 2.0) == pytest.approx(0.0)

    def test_standing_still_yields_none_not_noise(self) -> None:
        """★ 停在原地时位移是噪声，绝不能把它当成方向。

        全部段都小于 climb_eps -> 0 个有效段 -> None（而不是随机角度）。
        """
        hist = [(0.0, 0.0, 0.0), (1.0, 0.001, 0.001), (2.0, 0.002, 0.002)]
        assert S.heading_deg(hist, 2.0, 10.0) is None

    def test_uses_window_average_not_last_frame(self) -> None:
        """先用 +x 走，再单独用 +z 走一步。

        取最后一帧会得到 90°；取窗口均值应被前面的 +x 段拉回中间值。
        """
        hist = [(0.0, 0.0, 0.0), (1.0, 5.0, 0.0), (2.0, 5.0, 5.0)]
        got = S.heading_deg(hist, 2.0, 10.0)
        assert got is not None
        assert got == pytest.approx(45.0, abs=5.0)
        assert got != pytest.approx(90.0)


# ==================================================================
# 转向律
# ==================================================================

class TestTurnPlan:
    def test_small_error_is_proportional(self) -> None:
        want, cap, px = S.turn_plan(10.0)
        assert want == pytest.approx(-4.5)          # -0.45 * 10
        assert cap == pytest.approx(45.0)
        assert px == pytest.approx(-4.5 / 0.07)

    def test_large_error_hits_the_big_cap(self) -> None:
        want, cap, _px = S.turn_plan(179.0)
        assert cap == pytest.approx(60.0)
        assert want == pytest.approx(-60.0)        # -80.55 被限到 -60

    def test_cap_switches_exactly_at_threshold(self) -> None:
        """``>=`` 90.0 才放宽；89.99 仍是 45。边界写错会差一整档。"""
        assert S.turn_plan(89.99)[1] == pytest.approx(45.0)
        assert S.turn_plan(90.0)[1] == pytest.approx(60.0)

    def test_cap_is_always_bounded(self) -> None:
        """任意输入都不得超过上限 —— 旧代码这里曾经是 500px(=500°)。"""
        for d in (-179.0, -90.0, -12.0, 0.0, 12.0, 90.0, 179.0):
            want, cap, _px = S.turn_plan(d)
            assert abs(want) <= cap + 1e-9

    def test_positive_delta_yields_negative_pixels(self) -> None:
        """★★ 符号方向：Δ>0 = 目标在左 = 鼠标往右推 = **负**像素。

        搞反了寻路会朝反方向画圈。这是本模块最贵的一条不变量。
        """
        assert S.turn_plan(45.0)[2] < 0
        assert S.turn_plan(-45.0)[2] > 0

    def test_zero_delta_gives_zero_pixels(self) -> None:
        assert S.turn_plan(0.0)[2] == 0.0

    def test_sub_pixel_is_rounded_to_min_turn_px(self) -> None:
        """0.64px 会被 int 取整成 0 白发一次 -> 抬到 1px（保留符号）。"""
        assert S.turn_plan(0.1)[2] == pytest.approx(-1.0)
        assert S.turn_plan(-0.1)[2] == pytest.approx(1.0)

    def test_is_dimensionless_proportional_control(self) -> None:
        """★ ``want`` 必须**严格正比于 Δ**（没有任何绝对项）。

        这就是「无量纲比例控制」的定义。若哪天改成 ``want = -k*Δ + 常数``、
        或让 k 带 px 量纲，下面这条会立刻失败 —— 而那种改动正是当年
        四档绝对增益震荡的根因。
        """
        import dataclasses
        for k in (0.1, 0.45, 0.9):
            cfg = dataclasses.replace(C, turn_fraction=k)
            for d in (20.0, 45.0, 85.0, -30.0):
                want, cap, _px = S.turn_plan(d, cfg)
                if abs(want) < cap - 1e-9:      # 没触顶才是线性区
                    assert want == pytest.approx(-k * d), (k, d, want)


class TestSplitTurn:
    def test_zero_returns_nothing(self) -> None:
        assert S.split_turn(0.0) == ([], 0)

    def test_small_turn_goes_out_whole(self) -> None:
        segs, owed = S.split_turn(30.0)
        assert segs == [30]
        assert owed == 0

    def test_saturation_cap_applies_even_with_one_segment(self) -> None:
        """★★ 「不拆」≠「不限幅」。

        旧代码在 SPLIT_MAX_SEGMENTS<=1 时直接返回原值，于是 857px
        原样打出去落进饱和区（达成率 11%）。段数少只是「发得少」。
        """
        segs, owed = S.split_turn(100.0)
        assert segs == [40]              # 截到单段上限
        assert owed == 60                # 余量退回队列，不丢不硬塞
        assert sum(segs) + owed == 100

    def test_negative_is_symmetric(self) -> None:
        segs, owed = S.split_turn(-100.0)
        assert segs == [-40]
        assert owed == -60

    def test_stand_mode_unlocks_eight_segments(self) -> None:
        segs, owed = S.split_turn(320.0, max_segments=8)
        assert len(segs) == 8
        assert all(p == 40 for p in segs)
        assert owed == 0

    def test_segment_count_respects_its_own_limit(self) -> None:
        """预算按 8 段算，实际拆出来的段数也不能超过 8。"""
        segs, _owed = S.split_turn(10_000.0, max_segments=8)
        assert len(segs) <= 8

    def test_segments_never_exceed_saturation_each(self) -> None:
        for px in (39.0, 40.0, 41.0, 857.0, -857.0):
            segs, _ = S.split_turn(px, max_segments=8)
            assert all(abs(p) <= C.mouse_saturation_px for p in segs), px

    def test_remainder_is_absorbed_not_lost(self) -> None:
        """★ 取整余数必须由尾段吸收，否则会凭空少几个像素。

        拿 7px 试：3 段的话 ``7 // 3 = 2``，尾段必须补上第 3 个像素。
        """
        segs, owed = S.split_turn(7.0, max_segments=3)
        assert sum(segs) == 7
        assert owed == 0
        assert segs[-1] == max(segs)          # 尾段吸收余数，不比别的段小

    def test_total_px_always_fully_accounted(self) -> None:
        for px in (1.0, 7.0, 40.0, 41.0, 399.0, 400.0, 857.0, -1234.0):
            segs, owed = S.split_turn(px, max_segments=8)
            assert sum(segs) + owed == int(round(px)), px


class TestStandTurnStep:
    def test_enters_above_threshold(self) -> None:
        assert S.stand_turn_step(False, 61.0) == (True, "start")

    def test_does_not_enter_at_exactly_threshold(self) -> None:
        """60.0 是**严格大于**才进。边界写 ``>=`` 会提前站定。"""
        assert S.stand_turn_step(False, 60.0) == (False, "")

    def test_negative_enters_symmetrically(self) -> None:
        assert S.stand_turn_step(False, -61.0) == (True, "start")

    def test_does_not_enter_inside_deadzone(self) -> None:
        assert S.stand_turn_step(False, 8.0) == (False, "")

    def test_hysteresis_band_never_exits(self) -> None:
        """★ 站定后，20°~60° 这一整段**绝不退出**。

        单阈值必然在阈值附近震荡：转出 59° 就走，走歪回 61° 又站定。
        """
        for d in (20.1, 30.0, 45.0, 59.9, -30.0, -59.9):
            assert S.stand_turn_step(True, d) == (True, ""), d

    def test_exits_at_resume_threshold(self) -> None:
        assert S.stand_turn_step(True, 20.0) == (False, "done")
        assert S.stand_turn_step(True, -20.0) == (False, "done")

    def test_events_are_exclusive(self) -> None:
        assert S.stand_turn_step(False, 100.0)[1] == "start"
        assert S.stand_turn_step(True, 100.0)[1] == ""
        assert S.stand_turn_step(True, 5.0)[1] == "done"
        assert S.stand_turn_step(False, 5.0)[1] == ""


class TestDynamicMaxStepPx:
    def test_bad_ratio_takes_the_conservative_floor(self) -> None:
        """★ 比例读不出来时取**下限**，不是上限。

        反过来（放行到上限）会让一个坏配置把跳变检测整个放开。
        """
        assert S.dynamic_max_step_px(0.0, 0.2) == C.step_limit_min_px
        assert S.dynamic_max_step_px(-1.0, 0.2) == C.step_limit_min_px
        assert S.dynamic_max_step_px(None, 0.2) == C.step_limit_min_px

    def test_scales_with_frame_interval(self) -> None:
        """掉帧时每帧位移变大，上限必须跟着放宽。"""
        a = S.dynamic_max_step_px(0.02, 0.2)
        b = S.dynamic_max_step_px(0.02, 0.4)
        assert b > a
        assert a == pytest.approx(11.0 / 0.02 * 0.2 * 3.0)

    def test_clamped_to_max(self) -> None:
        assert S.dynamic_max_step_px(0.001, 1.0) == C.step_limit_max_px

    def test_clamped_to_min(self) -> None:
        assert S.dynamic_max_step_px(10.0, 0.001) == C.step_limit_min_px

    def test_never_outside_band(self) -> None:
        for p2w in (0.0005, 0.01, 0.5, 10.0, 500.0):
            for dt in (0.001, 0.05, 0.2, 1.0, 5.0):
                v = S.dynamic_max_step_px(p2w, dt)
                assert C.step_limit_min_px <= v <= C.step_limit_max_px


# ==================================================================
# 顶层决策
# ==================================================================

class TestSteer:
    def test_arrives_within_radius(self) -> None:
        p = S.steer(0.0, 0.0, 0.0, 0.5, heading=0.0)
        assert p.action == "ARRIVED"
        assert p.walk is False
        assert p.sprint is False

    def test_arrival_clears_stand_state(self) -> None:
        """到达后站定必须复位，否则下一段路开头会莫名站定。"""
        p = S.steer(0.0, 0.0, 0.0, 0.5, heading=0.0, stand_active=True)
        assert p.stand_active is False

    def test_exactly_on_radius_arrives(self) -> None:
        p = S.steer(0.0, 0.0, 1.5, 0.0, heading=0.0, arrive_radius_m=1.5)
        assert p.action == "ARRIVED"

    def test_unknown_heading_blind_walks(self) -> None:
        """★ 判不出朝向时**照走**，但 action 必须如实标成 BLIND_WALK。

        不能默默当成「已对准」—— 那是把「没有输入」伪装成「输入是 0」。
        """
        p = S.steer(0.0, 0.0, 50.0, 0.0, heading=None)
        assert p.action == "BLIND_WALK"
        assert p.delta_deg is None
        assert p.walk is True
        assert p.turn_px == 0.0

    def test_aligned_walks(self) -> None:
        p = S.steer(0.0, 0.0, 50.0, 0.0, heading=0.0)
        assert p.action == "WALK"
        assert p.delta_deg == pytest.approx(0.0)
        assert p.walk is True

    def test_inside_deadzone_walks(self) -> None:
        p = S.steer(0.0, 0.0, 50.0, 0.0, heading=5.0)
        assert p.action == "WALK"

    def test_outside_deadzone_turns(self) -> None:
        """目标在正前方、但当前朝向偏了 45° -> 必须往**右**转（正像素）。"""
        p = S.steer(0.0, 0.0, 50.0, 0.0, heading=45.0)
        assert p.action == "TURN"
        assert p.delta_deg == pytest.approx(-45.0)
        assert p.turn_px > 0

    def test_target_on_the_left_turns_left(self) -> None:
        """★ 端到端符号：目标在 +z（左边）-> Δ 为正 -> 下发负像素。"""
        p = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0)
        assert p.bearing == pytest.approx(90.0)
        assert p.delta_deg == pytest.approx(90.0)
        assert p.turn_px < 0
        assert all(seg < 0 for seg in p.segments)

    def test_target_on_the_right_turns_right(self) -> None:
        p = S.steer(0.0, 0.0, 0.0, -50.0, heading=0.0)
        assert p.delta_deg == pytest.approx(-90.0)
        assert p.turn_px > 0
        assert all(seg > 0 for seg in p.segments)

    def test_big_error_enters_stand_mode_and_splits_eight(self) -> None:
        p = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0)
        assert p.stand_active is True
        assert p.event == "start"
        assert len(p.segments) == 8

    def test_stand_state_must_be_fed_back(self) -> None:
        """站定是**跨帧**状态：站定标志与转向预算都得不传回去就等于没有。

        不传预算的退化行为是「预算已花完」-> 不发转向 -> 立刻结束站定
        并走一帧。那是**安全**的（不会原地打转），但代价是白进一次站定。
        """
        first = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0)
        assert first.stand_active is True
        assert first.stand_budget_deg > 0.0
        again = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0,
                        stand_active=first.stand_active,
                        stand_budget_deg=first.stand_budget_deg)
        assert again.event == ""            # 不再重复报 start
        assert again.stand_active is True
        assert again.stand_budget_deg < first.stand_budget_deg

    def test_missing_budget_fails_safe(self) -> None:
        """★ 不传预算时**不发任何转向**并立刻退出站定。

        这是有意的 fail-safe：宁可少转一点走一帧重新观测，也不要
        每帧按陈旧 Δ 发满转向（那正是 2026-10-14 的原地振荡）。
        """
        p = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0,
                    stand_active=True, stand_budget_deg=0.0)
        assert p.segments == ()
        assert p.turn_px == 0.0
        assert p.stand_active is False
        assert p.walk is True               # 走一帧让朝向重新可观测

    def test_budget_never_allows_overshoot(self) -> None:
        """★ 预算花完就停：站定期间发出的总角度**不超过**预定的 95%。"""
        heading, budget, sent, standing = 0.0, 0.0, 0.0, False
        for _ in range(20):
            p = S.steer(0.0, 0.0, 0.0, 50.0, heading=heading,
                        stand_active=standing,
                        stand_budget_deg=budget)
            budget = p.stand_budget_deg
            sent += abs(p.turn_px) * C.mouse_deg_per_px
            heading += p.turn_px * C.mouse_deg_per_px   # px -> 度
            standing = p.stand_active
            if not standing:
                break
        assert sent <= 90.0 * C.stand_budget_ratio + 1e-6, \
            f"发了 {sent:.1f}°，超过预算 90°×0.95"

    def test_hysteresis_survives_the_round_trip(self) -> None:
        """进入站定后转到一个「单阈值会退出」的夹角，必须**保持**站定。"""
        enter = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0)
        assert enter.stand_active is True
        mid = S.steer(0.0, 0.0, 0.0, 50.0, heading=35.0,
                      stand_active=enter.stand_active,
                      stand_budget_deg=enter.stand_budget_deg)
        assert mid.stand_active is True
        done = S.steer(0.0, 0.0, 0.0, 50.0, heading=85.0,
                       stand_active=mid.stand_active,
                       stand_budget_deg=mid.stand_budget_deg)
        assert done.stand_active is False
        assert done.event == "done"

    def test_window_blind_walks_one_frame_during_stand(self) -> None:
        """★ 安全阀：站定期间窗口快见底 -> 走一帧把朝向数据养回来。

        这是与「按角度收」**独立**的第二个出口 —— 转向判据的唯一输入
        是位移窗口，窗口一死整套站定逻辑会静默失效。
        """
        p = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0,
                    stand_active=True, window_move_m=0.0)
        assert p.window_blind is True
        assert p.walk is True

    def test_healthy_window_does_not_walk_during_stand(self) -> None:
        """窗口健康 + 预算充足 -> **站着**转，不走路。"""
        p = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0,
                    stand_active=True, stand_budget_deg=50.0,
                    window_move_m=5.0)
        assert p.window_blind is False
        assert p.stand_active is True
        assert p.walk is False

    def test_window_blind_not_reported_when_window_unknown(self) -> None:
        """没传窗口数据时**不能**谎称「窗口健康」。"""
        p = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0,
                    stand_active=True, stand_budget_deg=50.0)
        assert p.window_blind is False
        assert p.walk is False

    def test_sprint_flag_flows_through_when_walking(self) -> None:
        p = S.steer(0.0, 0.0, 50.0, 0.0, heading=0.0, want_sprint=True)
        assert p.sprint is True

    def test_sprint_is_off_while_turning(self) -> None:
        """转弯时不冲刺 —— 冲刺会把弧长拉长（真机绕路系数 1.82）。"""
        p = S.steer(0.0, 0.0, 50.0, 0.0, heading=45.0, want_sprint=True)
        assert p.action == "TURN"
        assert p.sprint is False

    def test_walks_while_turning_outside_stand_mode(self) -> None:
        """★★ 边走边转（原则 5）：小角度转向时**必须继续走**。

        2026-10-14 修的 bug：原先写 ``walk = blind and now_stand``，
        等于「平时转向一步都不走」。角色不前进 -> 朝向窗口攒不起来 ->
        ``heading`` 恒为 ``None`` -> 退化成盲走 -> **原地无限打转**。
        """
        p = S.steer(0.0, 0.0, 50.0, 0.0, heading=45.0,
                    window_move_m=5.0)          # 窗口健康
        assert p.action == "TURN"
        assert p.stand_active is False
        assert p.walk is True, "非站定的转向不该停步"

    def test_does_not_walk_while_standing(self) -> None:
        """站定（大角度）才停步 —— 这正是站定存在的意义。"""
        p = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0,
                    window_move_m=5.0)
        assert p.stand_active is True
        assert p.walk is False

    def test_segment_total_matches_turn_px_bucket(self) -> None:
        p = S.steer(0.0, 0.0, 0.0, 50.0, heading=0.0)
        assert p.segment_total + p.owed_px == int(round(p.turn_px))

    def test_is_deterministic(self) -> None:
        a = S.steer(3.0, -4.0, 10.0, 20.0, heading=33.0)
        b = S.steer(3.0, -4.0, 10.0, 20.0, heading=33.0)
        assert a == b

    def test_plan_is_frozen(self) -> None:
        """返回的是不可变数据，调用方不能就地改坏下一帧的判据。"""
        p = S.steer(0.0, 0.0, 10.0, 0.0, heading=0.0)
        with pytest.raises(Exception):
            p.walk = True          # type: ignore[misc]

    def test_distance_and_bearing_are_reported(self) -> None:
        p = S.steer(0.0, 0.0, 3.0, 4.0, heading=0.0)
        assert p.dist_m == pytest.approx(5.0)
        assert p.bearing == pytest.approx(math.degrees(math.atan2(4, 3)))

    def test_heading_across_the_seam_does_not_flip(self) -> None:
        """朝向 179°、目标方位 179.4° -> Δ 只有 +0.4°，不是 -358°。

        不做环绕归一的话会判成「差了一整圈」，于是狂转 358°。
        """
        p = S.steer(0.0, 0.0, -10.0, 0.1, heading=179.0)
        assert p.bearing == pytest.approx(179.427, abs=0.01)
        assert p.delta_deg == pytest.approx(0.427, abs=0.01)
        assert p.action == "WALK"           # 在死区内，不必转
