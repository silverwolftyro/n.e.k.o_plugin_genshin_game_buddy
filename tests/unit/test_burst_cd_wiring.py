"""``key_q``（大招就绪）接线测试。

覆盖链路：rois.toml[key_q] -> detectors.key_q.read_burst_cd()
          -> StateBuilder._read_burst_cd -> PartyMember.cd_ready["burst"]
          -> policy/context.q_ready()

★ 这份测试最要紧的一条是 :meth:`TestCalibration.test_threshold_sits_in_the_gap`
  —— 它把 296 帧实测出来的「就绪 0.822~0.824 / 不就绪 ≤0.091」钉成断言。
  将来谁改了环形半径或颜色阈值，这条会先炸，而不是等到真机才发现
  「大招永远放不出来」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.config.loader import BurstCdConfig, SkillCdConfig  # noqa: E402
from autopilot.perception import builder as B  # noqa: E402
from autopilot.perception.builder import StateBuilder  # noqa: E402
from autopilot.perception.detectors import key_q  # noqa: E402
from autopilot.perception.detectors.key_q import (  # noqa: E402
    CD_KEY, CYAN_B_MIN, CYAN_RATIO_READY, CYAN_V_MIN, R_IN, R_OUT,
    KeyQState, annulus_mask, detect_key_q, read_burst_cd, ring_stats,
)

#: 296 帧实测（probe_key_q_ring.py，见 key_q.py 模块文档）
MEASURED_READY = (0.822, 0.824)
MEASURED_NOT_READY_MAX = 0.091

ROI_NAME = "key_q"


def _frame(h: int = 720, w: int = 1280) -> np.ndarray:
    return np.zeros((h, w, 3), np.uint8)


def _sb(*, skill_cd=None, burst_cd=None, **kw) -> StateBuilder:
    return StateBuilder(skill_cd=skill_cd, burst_cd=burst_cd, **kw)


# ==================================================================
# 1 · 标定：阈值必须落在实测的空档里
# ==================================================================
class TestCalibration:
    def test_threshold_sits_in_the_gap(self) -> None:
        """★ 296 帧实测的就绪区间与不就绪上限之间有一条 0.73 宽的空档。

        阈值必须**严格落在中间**：调到 0.91 就贴上就绪样本的下沿，
        调到 0.05 就会把亮沙地判成就绪。
        """
        lo, hi = MEASURED_READY
        assert hi > lo
        assert MEASURED_NOT_READY_MAX < CYAN_RATIO_READY < lo, \
            (MEASURED_NOT_READY_MAX, CYAN_RATIO_READY, lo)

    def test_annulus_avoids_the_character_art(self) -> None:
        """内半径不能小于 0.30 —— 再往里就吃进角色图标（每角色不同）。

        吃进图标 = 特征重新变成角色相关 = 换角色就废，
        那正是 ``key_e`` 踩过的坑。
        """
        assert R_IN >= 0.30, R_IN
        assert R_OUT < 0.50, R_OUT

    def test_colour_gate_needs_both_conditions(self) -> None:
        """只卡蓝度会选中夜空，只卡亮度会选中亮沙地。两条都要。"""
        assert CYAN_B_MIN > 0
        assert CYAN_V_MIN > 0

    def test_annulus_mask_is_scale_invariant(self) -> None:
        """相对半径 -> 换分辨率不用重标定。"""
        small = annulus_mask(64, 64)
        big = annulus_mask(640, 640)
        assert small.shape == (64, 64) and big.shape == (640, 640)
        assert abs(small.mean() - big.mean()) < 0.005, \
            (small.mean(), big.mean())

    def test_roi_is_configured(self) -> None:
        from autopilot.perception.roi_loader import load_rois

        assert load_rois().get(ROI_NAME) is not None, \
            f"rois.toml 里没有 {ROI_NAME}"


# ==================================================================
# 2 · 三态映射（刻意只有两态）
# ==================================================================
class TestStates:
    def test_there_is_no_cooldown_state(self) -> None:
        """★ 296 帧里零个冷却样本，所以这个检测器**不能**报 COOLDOWN。

        枚举里没有这个成员，就没人能不小心把它写出来 ——
        比在 docstring 里写「注意别用 COOLDOWN」可靠。
        """
        names = {s.name for s in KeyQState}
        assert names == {"READY", "UNKNOWN"}, names

    def test_cyan_ratio_at_or_above_threshold_is_ready(self, monkeypatch) -> None:
        monkeypatch.setattr(key_q, "ring_stats",
                            lambda frame: {"cyan_ratio": 0.90, "ok": True})
        assert detect_key_q(_frame()) == (KeyQState.READY, 0.90)

    def test_cyan_ratio_below_threshold_is_unknown_not_cooldown(
            self, monkeypatch) -> None:
        monkeypatch.setattr(key_q, "ring_stats",
                            lambda frame: {"cyan_ratio": 0.20, "ok": True})
        state, conf = detect_key_q(_frame())
        assert state is KeyQState.UNKNOWN
        assert conf == 0.20, "UNKNOWN 也要报出实测值，排查时用得上"

    def test_unreadable_frame_is_unknown(self, monkeypatch) -> None:
        monkeypatch.setattr(key_q, "ring_stats",
                            lambda frame: {"cyan_ratio": 0.0, "ok": False})
        assert detect_key_q(_frame())[0] is KeyQState.UNKNOWN

    def test_read_burst_cd_maps_ready_to_true(self, monkeypatch) -> None:
        monkeypatch.setattr(key_q, "detect_key_q",
                            lambda f: (KeyQState.READY, 0.85))
        assert read_burst_cd(_frame()) == (True, 0.85, "key_q")

    def test_read_burst_cd_never_returns_false(self, monkeypatch) -> None:
        """★ ``False`` 意味着「我看见了冷却读条」—— 那是在编。

        不就绪一律 None。对规则等价（都不触发），但 ``unknowns``
        里会出现 ``cd``，让人一眼看出「大招读不出来」。
        """
        monkeypatch.setattr(key_q, "detect_key_q",
                            lambda f: (KeyQState.UNKNOWN, 0.1))
        got, conf, src = read_burst_cd(_frame())
        assert got is None
        assert src == "none"

    def test_exception_becomes_unknown(self, monkeypatch) -> None:
        def _raise(frame):
            raise ValueError("roi 不在帧内")

        monkeypatch.setattr(key_q, "detect_key_q", _raise)
        assert read_burst_cd(_frame()) == (None, 0.0, "none")

    def test_empty_frame_is_unknown_not_crash(self) -> None:
        for bad in (np.zeros((0, 0, 3), np.uint8), np.zeros((5, 5, 3), np.uint8)):
            got, _, _ = read_burst_cd(bad)
            assert got is None, bad.shape

    def test_cd_key_matches_what_context_reads(self) -> None:
        """★ 契约一致性：用源码把 ``"burst"`` 这个字面量钉死。

        ``policy/context.py::q_ready()`` 里写的是 ``_active_cd("burst")``。
        改一边不改另一边不会报错，只会静默失效（规则永远 UNKNOWN）。
        """
        import inspect

        from autopilot.policy.context import PolicyContext

        src = inspect.getsource(PolicyContext.q_ready)
        assert f'_active_cd("{CD_KEY}")' in src, src
        assert CD_KEY == "burst"


# ==================================================================
# 3 · StateBuilder 接线
# ==================================================================
class TestBuilderWiring:
    def test_disabled_means_no_burst_key(self, monkeypatch) -> None:
        calls: list[int] = []

        def _boom(frame):
            calls.append(1)
            return True, 0.9, "key_q"

        monkeypatch.setattr(B, "read_burst_cd", _boom)
        p = _sb(skill_cd=SkillCdConfig(enabled=False),
                burst_cd=BurstCdConfig(enabled=False))._build_player(
            "WORLD", None, [], [], 1, frame=_frame())
        assert p.party == ()
        assert calls == [], "关着的时候一次都不该碰 ROI"

    def test_enabled_fills_burst(self, monkeypatch) -> None:
        monkeypatch.setattr(B, "read_burst_cd",
                            lambda frame: (True, 0.85, "key_q"))
        p = _sb(burst_cd=BurstCdConfig(enabled=True))._build_player(
            "WORLD", None, [], [], 1, frame=_frame())
        assert p.party[0].cd_ready == {"burst": True}

    def test_unknown_still_creates_the_member(self, monkeypatch) -> None:
        monkeypatch.setattr(B, "read_burst_cd",
                            lambda frame: (None, 0.0, "none"))
        p = _sb(burst_cd=BurstCdConfig(enabled=True))._build_player(
            "WORLD", None, [], [], 1, frame=_frame())
        assert len(p.party) == 1
        assert p.party[0].cd_ready == {"burst": None}

    def test_burst_and_skill_toggles_are_independent(self, monkeypatch) -> None:
        """★ 两个信号可信度不同（key_q 标定过、key_e 没有），
        所以不能并成一个开关。"""
        monkeypatch.setattr(B, "read_skill_cd",
                            lambda frame: (True, 30.0, "key_e"))
        monkeypatch.setattr(B, "read_burst_cd",
                            lambda frame: (True, 0.85, "key_q"))

        only_skill = _sb(skill_cd=SkillCdConfig(enabled=True),
                         burst_cd=BurstCdConfig(enabled=False))._build_player(
            "WORLD", None, [], [], 1, frame=_frame())
        assert only_skill.party[0].cd_ready == {"skill": True}

        only_burst = _sb(skill_cd=SkillCdConfig(enabled=False),
                         burst_cd=BurstCdConfig(enabled=True))._build_player(
            "WORLD", None, [], [], 1, frame=_frame())
        assert only_burst.party[0].cd_ready == {"burst": True}

    def test_detector_exception_degrades_to_unknown(self, monkeypatch) -> None:
        def _raise(frame):
            raise RuntimeError("cv2 抽风")

        monkeypatch.setattr(B, "read_burst_cd", _raise)
        p = _sb(burst_cd=BurstCdConfig(enabled=True))._build_player(
            "WORLD", None, [], [], 1, frame=_frame())
        assert p.party[0].cd_ready == {"burst": None}

    def test_no_frame_means_no_read(self, monkeypatch) -> None:
        monkeypatch.setattr(B, "read_burst_cd",
                            lambda frame: (True, 0.9, "key_q"))
        p = _sb(burst_cd=BurstCdConfig(enabled=True))._build_player(
            "WORLD", None, [], [], 1, frame=None)
        assert p.party == ()

    def test_defaults_config_enables_it(self) -> None:
        """默认开着 —— 这是标定过的信号，没理由默认关。"""
        from autopilot.config.loader import load_config

        assert load_config().burst_cd.enabled is True


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
