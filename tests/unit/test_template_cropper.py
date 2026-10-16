"""``autopilot/tools/template_cropper.py`` 的单测（2026-10-12 任务 A）。

为什么这个工具好测
------------------
核心检测函数 ``find_bars`` 是**纯函数**：只吃图和 ``BarSpec``，不读配置、
不写盘、不开窗。所以能用**合成图**精确构造场景，不需要真机截图。

真机验证另说：``autopilot/_runtime/capdiag_wgc.png``（WGC 实拍 2560x1600）
上跑出来玩家血条 box=(1076,1500,408,15)、覆盖率 98%，与人工逐行扫描的结果
一致（见 ``test_module_docstring_records_real_calibration``）。

本文件覆盖的三条**设计约束**（都不是风格问题，是会直接导致误报/漏报的）
--------------------------------------------------------------------------
1. 必须按**色相**门控 —— 高饱和度不等于条（夜石地面 S=183，比血条只低一点）
2. 必须做**水平闭运算** —— 条上叠字会把色段切碎
3. 必须靠 **ROI** 隔离同名同色的不同条（玩家血条 vs 队友血条颜色完全一样）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

import cv2  # noqa: E402

from autopilot.tools.template_cropper import (  # noqa: E402
    ENERGY_BAR,
    HP_BAR,
    PARTY_HP_BAR,
    SPECS,
    TEMPLATE_DIR,
    BarCandidate,
    BarSpec,
    find_bars,
    pad_box,
    render_preview,
    save_crop,
    suggest_config_toml,
)

# ==================================================================
# 合成图工厂
# ==================================================================

def _blank(h=800, w=1280, bgr=(0, 0, 0)) -> np.ndarray:
    return np.full((h, w, 3), bgr, np.uint8)


def _hsv_canvas(h=800, w=1280, hsv=(0, 0, 0)) -> np.ndarray:
    """直接按 HSV 造图 —— 比按 BGR 猜颜色可靠。"""
    out = np.zeros((h, w, 3), np.uint8)
    out[:, :] = hsv
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)


def _draw_bar(img, x, y, bw, bh, hsv=(41, 215, 215)) -> None:
    """在图上直接涂一块指定 HSV 的横条。"""
    color = cv2.cvtColor(np.uint8([[list(hsv)]]), cv2.COLOR_HSV2BGR)[0, 0]
    img[y:y + bh, x:x + bw] = color


def _stone_background(h=800, w=1280) -> np.ndarray:
    """夜石地面：饱和度极高（H≈105 S≈183），真机实测值。

    这张图是「通用饱和度检测会翻车」的证据。
    """
    return _hsv_canvas(h, w, hsv=(105, 183, 164))


def _world_with_bars(h=800, w=1280, hp=True, energy=False):
    """一张有背景 + 可选血条/能量条的图。

    ★ 条必须画在 ``HP_BAR`` / ``ENERGY_BAR`` 的 ROI **之内** ——
      ROI 之外的部分会被裁掉，画了等于没画（踩过一次）。
    """
    img = _stone_background(h, w)
    # HP_BAR.roi = (0.35, 0.85, 0.65, 1.00) -> x 448..832, y 680..800
    if hp:
        _draw_bar(img, 460, 740, 350, 15)
    # ENERGY_BAR.roi = (0.35, 0.88, 0.65, 1.00) -> y 704..800
    if energy:
        _draw_bar(img, 460, 770, 350, 10, hsv=(25, 200, 220))
    return img


# ==================================================================
# 规格常量
# ==================================================================

@pytest.mark.unit
def test_specs__covers_the_three_bars() -> None:
    assert set(SPECS) == {"hp_bar", "energy_bar", "party_hp_bar"}


@pytest.mark.unit
def test_hp_spec__is_marked_as_calibrated_on_a_real_screenshot() -> None:
    """血条的 HSV 是在真机截图上实测的，必须如实标注。"""
    assert HP_BAR.calibrated is True
    assert "2560x1600" in HP_BAR.note


@pytest.mark.unit
def test_energy_spec__is_honestly_marked_uncalibrated() -> None:
    """⚠️ 手头那张真机截图里能量为 0，能量条是空的 —— 参数没实测过。

    假装它是标定过的，比不提供这个 spec 更危险。
    """
    assert ENERGY_BAR.calibrated is False
    assert "未实测" in ENERGY_BAR.note


@pytest.mark.unit
def test_hp_and_party_share_the_same_colour() -> None:
    """实测：玩家血条和队友血条 H/S/V 完全一样。

    所以只能靠 ROI 区分，这条断言把这个事实钉住 ——
    哪天有人给它们编了不同色相，反而说明结论变了，该更新实现。
    """
    assert (HP_BAR.h_lo, HP_BAR.h_hi) == (PARTY_HP_BAR.h_lo, PARTY_HP_BAR.h_hi)
    assert (HP_BAR.s_min, HP_BAR.v_min) == \
           (PARTY_HP_BAR.s_min, PARTY_HP_BAR.v_min)


@pytest.mark.unit
def test_hp_and_party_specs__have_disjoint_rois() -> None:
    """颜色一样，位置不同 —— ROI 是唯一的区分手段。"""
    hp_x = (HP_BAR.roi[0], HP_BAR.roi[2])
    party_x = (PARTY_HP_BAR.roi[0], PARTY_HP_BAR.roi[2])
    assert hp_x[1] <= party_x[0], "两个 ROI 的 x 区间不该重叠"


@pytest.mark.unit
def test_template_dir__points_at_the_configured_location() -> None:
    """存盘目录必须和 defaults.toml 里 [autopilot.templates.*] 的 path 对得上。"""
    assert TEMPLATE_DIR.name == "templates"
    assert TEMPLATE_DIR.parent.name == "perception"


# ==================================================================
# 约束 1：必须按色相门控
# ==================================================================

@pytest.mark.unit
def test_find_bars__ignores_highly_saturated_ground() -> None:
    """★ 夜石地面 S=183，比血条 S=215 只低一点。

    用「饱和且亮」当判据会把整块地砖框进来；按色相门控就不会。
    """
    img = _world_with_bars(hp=False)          # 只有地面，没有条
    assert find_bars(img, HP_BAR) == [], (
        "纯背景被判成了条 —— 说明判据没用色相门控")


@pytest.mark.unit
def test_find_bars__blue_ui_of_the_same_brightness_is_not_a_hp_bar() -> None:
    """同样亮、同样饱和，但色相不对 —— 必须不认。"""
    img = _hsv_canvas(800, 1280, hsv=(105, 215, 215))
    assert find_bars(img, HP_BAR) == []


@pytest.mark.unit
def test_find_bars__finds_the_hp_bar_on_a_realistic_background() -> None:
    img = _world_with_bars(hp=True)
    found = find_bars(img, HP_BAR)
    assert len(found) == 1
    best = found[0]
    assert best.box == (460, 740, 350, 15), "框必须贴紧条，不得被闭运算撑大"
    assert best.coverage > 0.9, "满条应该是高覆盖率"


# ==================================================================
# 约束 2：必须桥接条上的文字断裂
# ==================================================================

@pytest.mark.unit
def test_find_bars__bridges_text_that_splits_the_colour_run() -> None:
    """★ 实测：血条上的 "17640 / 17640" 会把连续绿色切碎。

    不做水平闭运算，最宽连续段只有 126px（真机实测值），
    框出来的模板只有半条，血量比例全错。
    """
    img = _stone_background()
    x0, y0, bw, bh = 500, 700, 408, 15
    _draw_bar(img, x0, y0, bw, bh)
    # 抹掉中间一段，模拟文字遮挡
    img[y0:y0 + bh, x0 + 150:x0 + 180] = (0, 0, 0)

    spec = BarSpec(name="t", label="t", h_lo=30, h_hi=80, s_min=120,
                   v_min=120, bridge=40, min_aspect=3.0)
    found = find_bars(img, spec)
    assert len(found) == 1
    assert found[0].box[2] == bw, (
        f"桥接后应该仍是整条 {bw}px，实际 {found[0].box[2]}px")


@pytest.mark.unit
def test_find_bars__bridge_of_zero_keeps_fragments_separate() -> None:
    """对照组：不桥接就确实会碎 —— 证明上面那条不是白测的。"""
    img = _stone_background()
    x0, y0, bw, bh = 500, 700, 408, 15
    _draw_bar(img, x0, y0, bw, bh)
    img[y0:y0 + bh, x0 + 150:x0 + 180] = (0, 0, 0)

    spec = BarSpec(name="t", label="t", h_lo=30, h_hi=80, s_min=120,
                   v_min=120, bridge=1, min_aspect=3.0)
    found = find_bars(img, spec)
    widest = max((c.box[2] for c in found), default=0)
    assert widest < bw, "不桥接时不该还能拿到整条"


# ==================================================================
# 约束 3：靠 ROI 隔离同名同色的条
# ==================================================================

@pytest.mark.unit
def test_find_bars__party_bars_are_found_in_their_own_roi() -> None:
    img = _blank(800, 1280)
    for i, y in enumerate((200, 300, 400)):
        _draw_bar(img, 1030, y, 95, 7)        # 右上角，四条队友血条
    found = find_bars(img, PARTY_HP_BAR)
    assert len(found) == 3


@pytest.mark.unit
def test_find_bars__party_bar_does_not_leak_into_the_hp_roi() -> None:
    """★ 同色同亮，只因位置不同而分属两个 ROI。

    若这条挂了，说明 ROI 计算坏了，队友血条会被当成玩家血条裁出来 ——
    那样 ``hp()`` 拿到的就是队友的血量。
    """
    img = _blank(800, 1280)
    _draw_bar(img, 1030, 200, 95, 7)          # 只放队友血条
    assert find_bars(img, HP_BAR) == []


@pytest.mark.unit
def test_find_bars__hp_bar_does_not_leak_into_the_party_roi() -> None:
    img = _blank(800, 1280)
    _draw_bar(img, 500, 750, 408, 15)         # 只放玩家血条
    assert find_bars(img, PARTY_HP_BAR) == []


# ==================================================================
# 形状过滤
# ==================================================================

@pytest.mark.unit
def test_find_bars__rejects_square_blobs() -> None:
    """竖条/方块不是横条 —— 宽高比门槛必须真的生效。"""
    img = _blank()
    _draw_bar(img, 600, 400, 20, 20)          # 正方形
    spec = BarSpec(name="t", label="t", h_lo=30, h_hi=80, s_min=120,
                   v_min=120, min_aspect=3.0)
    assert find_bars(img, spec) == []


@pytest.mark.unit
def test_find_bars__rejects_sparse_noise() -> None:
    """零散噪点覆盖率低，必须被挡掉。"""
    img = _blank()
    for i in range(0, 400, 7):
        img[400:415, 600 + i:606 + i] = (0, 255, 0)
    spec = BarSpec(name="t", label="t", h_lo=30, h_hi=80, s_min=120,
                   v_min=120, min_aspect=3.0, bridge=1)
    assert find_bars(img, spec) == []


@pytest.mark.unit
def test_find_bars__energy_bar_detected_when_present() -> None:
    """能量条在真机上没标定，但**检测逻辑**本身要能工作。"""
    img = _world_with_bars(hp=True, energy=True)
    spec = BarSpec(name="energy_bar", label="e", h_lo=15, h_hi=34,
                   s_min=120, v_min=150, roi=(0.35, 0.88, 0.65, 1.00),
                   bridge=25, min_aspect=3.0)
    assert len(find_bars(img, spec)) >= 1


# ==================================================================
# 边界与健壮性
# ==================================================================

@pytest.mark.unit
def test_find_bars__empty_image_is_rejected_not_crashed() -> None:
    assert find_bars(None, HP_BAR) == []
    assert find_bars(np.zeros((0, 0, 3), np.uint8), HP_BAR) == []


@pytest.mark.unit
def test_find_bars__roi_covering_nothing_returns_empty() -> None:
    img = _world_with_bars(hp=True)
    off = BarSpec(name="off", label="off", h_lo=30, h_hi=80, s_min=120,
                  v_min=120, roi=(0.0, 0.0, 0.01, 0.01))
    assert find_bars(img, off) == []


@pytest.mark.unit
def test_find_bars__results_are_sorted_best_first() -> None:
    img = _blank()
    # PARTY_HP_BAR.roi = (0.80, 0.25, 1.00, 0.75) -> x 1024..1280
    _draw_bar(img, 1050, 200, 40, 8)          # 窄的
    _draw_bar(img, 1050, 300, 180, 10)        # 宽的
    found = find_bars(img, PARTY_HP_BAR)
    assert len(found) >= 2, "两条都应该被找到"
    assert found[0].box[2] >= found[-1].box[2], "得分应与宽度正相关"


@pytest.mark.unit
def test_find_bars__box_coordinates_are_in_full_frame_space() -> None:
    """ROI 偏移必须被加回去 —— 否则裁出来的是 ROI 内的相对坐标。"""
    img = _blank(800, 1280)
    _draw_bar(img, 1030, 200, 95, 7)
    found = find_bars(img, PARTY_HP_BAR)
    assert found
    assert found[0].box[0] == 1030 and found[0].box[1] == 200


# ==================================================================
# 存盘 / 预览 / 配置片段
# ==================================================================

@pytest.mark.unit
def test_pad_box__grows_and_clamps_inside_the_frame() -> None:
    img = _blank(100, 100)
    assert pad_box((10, 10, 20, 20), 4, img.shape) == (6, 6, 28, 28)
    assert pad_box((0, 0, 5, 5), 10, img.shape) == (0, 0, 15, 15)
    assert pad_box((95, 95, 5, 5), 10, img.shape) == (85, 85, 15, 15)


@pytest.mark.unit
def test_pad_box__never_returns_a_degenerate_box() -> None:
    """全图边角的框也要有正面积，否则切片会出空数组。"""
    img = _blank(100, 100)
    for box in ((0, 0, 1, 1), (99, 99, 1, 1), (0, 0, 100, 100)):
        x, y, w, h = pad_box(box, 8, img.shape)
        assert w >= 1 and h >= 1


@pytest.mark.unit
def test_save_crop__writes_a_png_with_the_standard_name(tmp_path) -> None:
    img = _blank(100, 100)
    _draw_bar(img, 10, 10, 60, 12)
    cand = find_bars(img, BarSpec(name="hp_bar", label="HP", h_lo=30, h_hi=80,
                                  s_min=120, v_min=120, bridge=10))[0]
    p = save_crop(img, cand, tmp_path, pad=3)
    assert p.name == "hp_bar.png"
    assert p.exists()
    out = cv2.imread(str(p))
    assert out is not None and out.shape[0] == 12 + 6


@pytest.mark.unit
def test_save_crop__creates_the_output_dir(tmp_path) -> None:
    img = _blank(100, 100)
    _draw_bar(img, 10, 10, 60, 12)
    cand = find_bars(img, BarSpec(name="hp_bar", label="HP", h_lo=30, h_hi=80,
                                  s_min=120, v_min=120, bridge=10))[0]
    target = tmp_path / "deep" / "templates"
    save_crop(img, cand, target)
    assert (target / "hp_bar.png").exists()


@pytest.mark.unit
def test_render_preview__does_not_mutate_the_input() -> None:
    """预览是在副本上画的 —— 画完还要拿原图去裁，不能被画花了。"""
    img = _world_with_bars(hp=True)
    before = img.copy()
    cands = find_bars(img, HP_BAR)
    render_preview(img, cands, scale=0.5)
    assert np.array_equal(img, before)


@pytest.mark.unit
def test_render_preview__scales_the_canvas() -> None:
    img = _world_with_bars(hp=True)
    cands = find_bars(img, HP_BAR)
    small = render_preview(img, cands, scale=0.5)
    assert small.shape[0] == img.shape[0] // 2


@pytest.mark.unit
def test_suggest_config_toml__emits_a_pasteable_block() -> None:
    out = suggest_config_toml("hp_bar", 408, 15)
    assert "[autopilot.templates.hp_bar]" in out
    assert 'path = "perception/templates/hp_bar.png"' in out
    assert "threshold" in out and "roi" in out


@pytest.mark.unit
def test_suggest_config_toml__roi_is_relative_not_absolute() -> None:
    """ROI 表达式按相对位置生成，换分辨率时依然成立。"""
    a = suggest_config_toml("hp_bar", 408, 15)
    b = suggest_config_toml("hp_bar", 816, 30)   # 两倍分辨率
    assert a != b


# ==================================================================
# 候选对象本身
# ==================================================================

@pytest.mark.unit
def test_candidate__score_grows_with_coverage_and_width() -> None:
    weak = BarCandidate(spec=HP_BAR, box=(0, 0, 100, 10), coverage=0.5)
    strong = BarCandidate(spec=HP_BAR, box=(0, 0, 400, 10), coverage=0.95)
    assert strong.score > weak.score


@pytest.mark.unit
def test_candidate__width_term_is_capped_at_one() -> None:
    """很宽的条不该让得分无限涨，否则排序被超宽噪点支配。"""
    a = BarCandidate(spec=HP_BAR, box=(0, 0, 400, 10), coverage=0.9)
    b = BarCandidate(spec=HP_BAR, box=(0, 0, 4000, 10), coverage=0.9)
    assert a.score == b.score


@pytest.mark.unit
def test_candidate__as_dict_is_json_friendly() -> None:
    import json
    cand = find_bars(_world_with_bars(hp=True), HP_BAR)[0]
    json.dumps(cand.as_dict())          # 不抛就是过


@pytest.mark.unit
def test_module_docstring_records_the_real_calibration() -> None:
    """★ 实测标定的数字要留在代码里。

    这三条（H=41 / 126px 断裂 / S=183 地面）不是装饰 ——
    哪天有人「优化」掉色相门控或闭运算，这个测试会提醒他去看真机数据。
    """
    src = Path(TEMPLATE_DIR).parents[1] / "tools" / "template_cropper.py"
    doc = src.read_text(encoding="utf-8")
    assert "H=41 S=215 V=215" in doc
    assert "126px" in doc
    assert "S=183" in doc
