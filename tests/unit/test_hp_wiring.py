"""血量几何投影**接线**的单测（2026-10-12）。

覆盖「配置 -> ``StateBuilder`` -> ``PlayerInfo.hp_ratio`` -> ``hp()`` 信号」
这条链，以及 :mod:`perception.hp_estimator` 在真实截图与极端输入下的边界。

★ 最重要的一条：不猜
  ``hp() < 0.35`` 会据此决定要不要吃药。算不出来时必须回到 UNKNOWN，
  **绝不能**返回 0 或 1 —— 那两种猜错的代价都远大于「这一帧不知道」。

真机截图相关用例一律 ``skipif``：``_runtime/`` 被 git 忽略、也不随包分发，
CI 上不该因为缺一张本机截图就红。

★ 为什么测试里要造「整帧 + roi_frac」而不是直接把条喂进去
  ``_estimate_hp(frame)`` 收的是**整帧**，它自己按 ROI 比例去裁。
  直接把一根 60x12 的条当帧传进去，裁出来的区域是错的 —— 那测的是
  调用方的错误用法，不是接线本身。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autopilot.config.loader import HpEstConfig, load_config
from autopilot.contracts.state_packet import PlayerInfo, StatePacket
from autopilot.perception import builder as B
from autopilot.perception.hp_estimator import BarParams, estimate_fill_ratio
from autopilot.policy.context import PolicyContext, UnknownValue

#: 本机真机截图（2560x1600，WGC 捕获）。gitignore 覆盖，CI 上不存在。
_REAL_FRAME = (Path(__file__).resolve().parents[2]
               / "autopilot" / "_runtime" / "capdiag_wgc.png")
need_real = pytest.mark.skipif(not _REAL_FRAME.is_file(),
                               reason=f"本机没有真机截图 {_REAL_FRAME}")

#: 实测自真机截图
REAL_FILL_BGR = (34, 215, 150)
REAL_FRAME_BGR = (145, 88, 36)
#: 合成色（**不是实测**），与 extract_hp_template.SYNTH_EMPTY_BG 一致
SYNTH_BG = (25, 25, 25)

# -- 合成场景的尺寸 ---------------------------------------------------
FRAME_W, FRAME_H = 600, 400
BAR_X, BAR_Y, BAR_W, BAR_H = 60, 300, 120, 12
BAR_ROI_FRAC = (BAR_X / FRAME_W, BAR_Y / FRAME_H,
                BAR_W / FRAME_W, BAR_H / FRAME_H)


def _sb(**kw):
    """造一个不碰检测器的 StateBuilder（auto 会去建 DirectML，太慢）。"""
    return B.StateBuilder(entity_detector=None, **kw)


def _cfg(**kw) -> HpEstConfig:
    """接线配置，默认带上合成场景的 ROI 比例。"""
    kw.setdefault("enabled", True)
    kw.setdefault("roi_frac", BAR_ROI_FRAC)
    return HpEstConfig(**kw)


def _frame_with_bar(frac: float) -> np.ndarray:
    """整帧，底部放一根填充比例 ``frac`` 的条。

    ★ 边框固定 **1px**，与 ``BarParams.border_inset`` 默认值对齐 ——
      画成 2px 框而 inset 仍是 1，内部会剩一圈框，把每个截面都污染。
    """
    a = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    a[:] = (40, 30, 25)                       # 场景底色
    bar = np.empty((BAR_H, BAR_W, 3), np.uint8)
    bar[:, :] = REAL_FRAME_BGR                # 外框
    inner = BAR_W - 2
    n = int(round(inner * frac))
    if n:
        bar[1:BAR_H - 1, 1:1 + n] = REAL_FILL_BGR
    if n < inner:
        bar[1:BAR_H - 1, 1 + n:BAR_W - 1] = SYNTH_BG
    a[BAR_Y:BAR_Y + BAR_H, BAR_X:BAR_X + BAR_W] = bar
    return a


# ==================================================================
# 配置：默认必须是关的
# ==================================================================

def test_config__defaults_are_off_and_safe() -> None:
    c = HpEstConfig()
    assert c.enabled is False
    assert c.roi_frac is None
    assert c.background_bgr is None
    assert c.border_inset == 1


def test_config__defaults_toml_keeps_it_off() -> None:
    """★ 用户不改任何配置时，血量估算必须是关的。"""
    assert load_config().hp_estimation.enabled is False


def test_config__empty_toml_list_becomes_none() -> None:
    """TOML 里留空读出来是 ``[]``，要归一成 None —— 否则加载直接报错。"""
    c = HpEstConfig.model_validate({"roi_frac": [], "background_bgr": []})
    assert c.roi_frac is None and c.background_bgr is None


def test_config__roi_frac_accepts_four_numbers() -> None:
    assert HpEstConfig(roi_frac=(0.42, 0.94, 0.16, 0.01)).roi_frac == \
        (0.42, 0.94, 0.16, 0.01)


def test_config__unknown_field_is_rejected() -> None:
    """拼错参数名要立刻炸，而不是被静默忽略。"""
    with pytest.raises(Exception):
        HpEstConfig(background_bgrg=(1, 2, 3))


# ==================================================================
# 未启用 / 未接入时：与接入前完全一致
# ==================================================================

def test_builder__no_config_means_no_hp() -> None:
    assert _sb()._estimate_hp(_frame_with_bar(0.5)) == (None, "none")


def test_builder__disabled_means_no_hp() -> None:
    """给了配置但 ``enabled=False``，也不能算出东西来。"""
    cfg = HpEstConfig(enabled=False, roi_frac=BAR_ROI_FRAC)
    assert _sb(hp_estimation=cfg)._estimate_hp(_frame_with_bar(0.5)) \
        == (None, "none")


def test_builder__default_player_has_no_hp() -> None:
    """★ v1 的保证：``hp_ratio`` 恒 None = UNKNOWN。"""
    p = _sb()._build_player("WORLD", None, [], [], 1)
    assert isinstance(p, PlayerInfo)
    assert p.hp_ratio is None
    assert p.hp_source == "none"


def test_signal__hp_is_unknown_without_wiring() -> None:
    """从 StatePacket 一路走到 DSL 信号（沿用仓库既有的 frozen setattr 手法）。"""
    st = StatePacket.unknown(game_id="g", state_version=1, reason="t")
    ctx = PolicyContext.from_state(st, now_ms=1000)
    assert isinstance(ctx.hp(), UnknownValue)


# ==================================================================
# 启用后
# ==================================================================

def test_builder__partial_bar_needs_no_background_color() -> None:
    """★★ 部分血量的条内部有填充/空白分界，不靠底色也能测准。

    这是本功能真正可用的理由 —— 满血/空血那两档有歧义，中间的档没有。
    """
    for frac, want in ((0.25, 0.24), (0.5, 0.5), (0.75, 0.76)):
        got = _sb(hp_estimation=_cfg())._estimate_hp(_frame_with_bar(frac))
        assert got[1] == "geometry", (frac, got)
        assert got[0] == pytest.approx(want, abs=0.03), (frac, got)


def test_builder__full_bar_without_bg_is_unknown_not_guessed() -> None:
    """★★ 满血 + 没有底色 -> UNKNOWN，不是 1.0。

    满血条内部是一整片纯色，与「一块纯色补丁」像素上无法区分。
    猜 1.0 会让规则以为「满血」，而真实情况可能根本不是血条。
    """
    assert _sb(hp_estimation=_cfg())._estimate_hp(_frame_with_bar(1.0)) \
        == (None, "none")


def test_builder__full_bar_with_bg_reads_one() -> None:
    got = _sb(hp_estimation=_cfg(background_bgr=SYNTH_BG))._estimate_hp(
        _frame_with_bar(1.0))
    assert got[1] == "geometry"
    assert got[0] == pytest.approx(1.0)


def test_builder__empty_bar_with_bg_reads_zero() -> None:
    got = _sb(hp_estimation=_cfg(background_bgr=SYNTH_BG))._estimate_hp(
        _frame_with_bar(0.0))
    assert got[0] == pytest.approx(0.0)


def test_build_player__passes_the_ratio_through() -> None:
    p = _sb()._build_player("WORLD", None, [], [], 1, hp=(0.42, "geometry"))
    assert p.hp_ratio == 0.42 and p.hp_source == "geometry"


def test_build_player__unknown_hp_keeps_source_none() -> None:
    p = _sb()._build_player("WORLD", None, [], [], 1, hp=(None, "none"))
    assert p.hp_ratio is None and p.hp_source == "none"


def test_build_player__energy_is_still_empty() -> None:
    """能量条没有可用参考帧，**仍然留空** —— 不能顺手编一个。"""
    p = _sb()._build_player("WORLD", None, [], [], 1, hp=(0.5, "geometry"))
    assert p.energy is None and p.energy_state == "UNKNOWN"
    assert p.energy_ratio is None


# ==================================================================
# 极端输入：降级而不是崩
# ==================================================================

@pytest.mark.parametrize("bad", [
    np.zeros((0, 0, 3), np.uint8),         # 空帧
    np.zeros((5, 5, 3), np.uint8),         # 比 ROI 还小
])
def test_builder__degenerate_frames_never_crash(bad) -> None:
    assert _sb(hp_estimation=_cfg())._estimate_hp(bad)[0] is None


def test_builder__uniform_frame_does_not_crash() -> None:
    """纯色画面不该崩。给不给底色都可能出数（那是调用方的责任）。"""
    uniform = np.full((FRAME_H, FRAME_W, 3), 128, np.uint8)
    for cfg in (_cfg(), _cfg(background_bgr=SYNTH_BG)):
        ratio, source = _sb(hp_estimation=cfg)._estimate_hp(uniform)
        assert ratio is None or 0.0 <= ratio <= 1.0
        assert source in ("none", "geometry")


def test_builder__night_scene_dark_bar_is_not_invented() -> None:
    """★ 夜晚整体压暗时不该编一个「满血」出来。"""
    bar = _frame_with_bar(0.3)
    bar = (bar.astype(np.int16) * 0.35).astype(np.uint8)      # 整体压暗
    got = _sb(hp_estimation=_cfg(background_bgr=SYNTH_BG))._estimate_hp(bar)
    assert got[0] is None or got[0] < 0.95, got


def test_builder__noise_is_not_a_bar() -> None:
    """整屏随机噪声不该被读成某根条的某个比例。"""
    rng = np.random.default_rng(20261012)
    noise = rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    assert _sb(hp_estimation=_cfg())._estimate_hp(noise)[0] is None


@pytest.mark.parametrize("frac", [-0.5, 0.0, 2.0, 99.0])
def test_builder__nonsense_roi_does_not_crash(frac) -> None:
    cfg = HpEstConfig(enabled=True, roi_frac=(frac, frac, frac, frac))
    assert _sb(hp_estimation=cfg)._estimate_hp(_frame_with_bar(0.5))[0] is None


# ==================================================================
# 参考资产
# ==================================================================

def test_reference__is_shipped_and_readable() -> None:
    ref = B.load_hp_reference()
    assert ref is not None
    assert len(ref["roi_frac"]) == 4
    assert ref["fill_bgr"] == list(REAL_FILL_BGR)


def test_reference__marks_the_empty_template_as_synthetic() -> None:
    """★ 空条是**合成**的，文件里必须自曝，否则会被当成真机证据引用。"""
    ref = B.load_hp_reference()
    assert ref["empty_template_is_synthetic"] is True
    assert ref["empty_background_bgr"] == list(SYNTH_BG)


def test_reference__missing_file_degrades_to_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(B, "HP_TEMPLATE_DIR", tmp_path)
    assert B.load_hp_reference() is None


def test_reference__corrupt_file_degrades_to_none(monkeypatch, tmp_path) -> None:
    (tmp_path / "hp_bar_reference.json").write_text("{不是 JSON", encoding="utf-8")
    monkeypatch.setattr(B, "HP_TEMPLATE_DIR", tmp_path)
    assert B.load_hp_reference() is None


def test_roi_rect__scales_with_resolution() -> None:
    """★ 换分辨率必须跟着换算，否则框到别的 UI 上，读出的是编的值。"""
    ref = B.load_hp_reference()
    assert B._hp_roi_rect(ref, 2560, 1600) == (1076, 1500, 408, 15)
    x2, y2, w2, h2 = B._hp_roi_rect(ref, 1920, 1080)
    assert w2 < 408 and h2 < 15
    assert 0 <= x2 < 1920 and 0 <= y2 < 1080


def test_roi_rect__clamps_into_the_frame() -> None:
    x, y, w, h = B._hp_roi_rect({"roi_frac": (0.9, 0.95, 0.5, 0.5)}, 100, 100)
    assert x + w <= 100 and y + h <= 100


# ==================================================================
# 真机截图（CI 上跳过）
# ==================================================================

@need_real
def test_real__full_hp_frame_reads_one_with_bg() -> None:
    import cv2

    got = _sb(hp_estimation=HpEstConfig(
        enabled=True, background_bgr=SYNTH_BG))._estimate_hp(
        cv2.imread(str(_REAL_FRAME)))
    assert got == (1.0, "geometry"), got


@need_real
def test_real__full_hp_frame_is_unknown_without_bg() -> None:
    """真机满血帧 + 不给底色 -> UNKNOWN。这不是缺陷，是「不猜」。"""
    import cv2

    got = _sb(hp_estimation=HpEstConfig(enabled=True))._estimate_hp(
        cv2.imread(str(_REAL_FRAME)))
    assert got == (None, "none")


@need_real
def test_real__extracted_templates_round_trip() -> None:
    """提取脚本产出的两个模板必须能直接喂回估算器。"""
    import cv2

    tdir = B.HP_TEMPLATE_DIR
    full = cv2.imread(str(tdir / "full_hp_template.png"))
    empty = cv2.imread(str(tdir / "empty_hp_template.png"))
    if full is None or empty is None:
        pytest.skip("模板资产缺失（跑一下 autopilot.tools.extract_hp_template）")
    bg = tuple(json.loads((tdir / "hp_bar_reference.json")
                          .read_text(encoding="utf-8"))["empty_background_bgr"])
    assert estimate_fill_ratio(full, BarParams(background_color=bg)) == 1.0
    assert estimate_fill_ratio(empty, BarParams(background_color=bg)) == 0.0


@need_real
@need_real
def test_real__frame_is_a_readable_screenshot() -> None:
    """★ 契约测试：真截图**能加载、分辨率与标定一致**。

    2026-10-14 改。原来这里断言「这帧的天空有多少蓝像素、铺得多宽」
    —— 那是**过拟合到某一帧的构图**：用户重新采一次图（同一分辨率、
    不同画面）像素数就变，测试红，而代码什么都没做错。

    这里只守「资产可用」：能解码、分辨率与标定一致。
    画面的**内容**不该由一条测试钉死。
    """
    import cv2

    img = cv2.imread(str(_REAL_FRAME))
    assert img is not None, "真机截图解码失败"
    assert img.shape[:2] == (1600, 2560), (
        f"分辨率变成 {img.shape[:2]}，与 hp_bar_reference.json 标定的 "
        "2560x1600 不符 —— ROI 会框错位置")


@pytest.mark.unit
def test_synthetic__diffuse_colour_is_not_guessed_without_a_bg() -> None:
    """★ 大片同色区域，**不给**背景色时必须返回 None（不猜）。

    ⚠️ 2026-10-14 更正一处我自己的误解：
      给定 ``background_color`` 时，整片天空「每个像素都不是背景色」
      = 逻辑上就是一根满条 -> 返回 **1.0**，那是**正确**行为。
      所以「防天空」靠的不是颜色，而是**不给 bg 就不猜**，
      以及真实部署时先定 ROI（靠形状，不靠颜色）。
    """
    import numpy as np

    sky = np.full((180, 400, 3), (200, 150, 70), np.uint8)   # BGR 偏蓝
    assert estimate_fill_ratio(sky, border_inset=1) is None
    # 用它自己的颜色当背景色 -> 整片都是「空」 -> 0.0
    assert estimate_fill_ratio(
        sky, border_inset=1, background_color=(200, 150, 70)) == 0.0


@pytest.mark.unit
def test_synthetic__narrow_bar_inside_a_diffuse_field_is_found() -> None:
    """★ 反过来也要成立：真实存在的条**嵌在**大片同色背景里时仍能找出来。

    这才是这条测试线真正要守的东西 —— 与截图无关。
    """
    import numpy as np

    field = np.full((200, 600, 3), (200, 150, 70), np.uint8)     # 天空/地面
    field[100:115, 200:560] = (34, 215, 150)                      # 一根满条
    got = estimate_fill_ratio(field, border_inset=1,
                             background_color=(30, 30, 30))
    assert got is not None and got > 0.9, got


def test_estimator__uniform_image_is_unknown() -> None:
    """纯色图看起来像「满条」，也像「一块补丁」—— 不猜。"""
    assert estimate_fill_ratio(np.full((12, 60, 3), 77, np.uint8),
                               BarParams()) is None


def test_estimator__too_small_is_unknown() -> None:
    assert estimate_fill_ratio(np.zeros((2, 2, 3), np.uint8),
                               BarParams()) is None


def test_estimator__vertical_bar() -> None:
    """竖条（沿行方向填充）。

    ★ 内部空白区必须**就是**声明的底色。用描边色去填空白会让每一格
      都「不同于底色」-> 整条被判成满条（实测返回 1.0）。
    """
    h, w = 60, 12
    a = np.empty((h, w, 3), np.uint8)
    a[:, :] = REAL_FRAME_BGR              # 外框
    a[1:h - 1, 1:w - 1] = SYNTH_BG       # 内部 = 声明的底色
    a[1:31, 1:w - 1] = REAL_FILL_BGR     # 顶部往下填 30/58 ≈ 0.517
    got = estimate_fill_ratio(a, BarParams(orientation="vertical",
                                           background_color=SYNTH_BG))
    assert got is not None and 0.47 < got < 0.56, got


def test_estimator__align_end() -> None:
    """从另一端起算的条也要能读。

    ★ 要**先裁出条**再翻转 —— 直接把整帧喂进去会被判成「不是一根条」。
    """
    full = _frame_with_bar(0.5)
    bar = full[BAR_Y:BAR_Y + BAR_H, BAR_X:BAR_X + BAR_W]
    flipped = bar[:, ::-1].copy()
    got = estimate_fill_ratio(flipped, BarParams(align="end",
                                                 background_color=SYNTH_BG))
    assert got is not None and 0.45 < got < 0.55, got