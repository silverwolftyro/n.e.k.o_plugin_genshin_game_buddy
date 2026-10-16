"""``perception/teleport_ocr.py`` 的单测（2026-10-12 任务 D）。

★ 先说清这个模块的处境
----------------------
它**从未被 import** —— 整条链路里没有任何地方用它。但它不是死代码：

* ``LocalizationService.on_teleport_ocr(target)`` 明确期待一个
  ``teleport_ocr.TeleportTarget``（localization_service.py:218）
* 契约写在 ``recognition/notes/teleport_contract.md``

所以它的状态是「**契约已定、实现待接线**」，不是「无人要的残骸」。
删掉等于扔掉一份已经写好的契约实现。

那为什么以前没有测试？因为它「不在链路上」，没人被它咬过。
本文件补的就是这一层 —— 而且是**最该验**的一层：模糊匹配。

    set_backend() / get_backend()  一对读写，NEKO_OCR_BACKEND 环境变量驱动
    set_ocr()                       注入自定义 OCR（与 StateBuilder(locator=) 同类）

它们「零引用」是因为整个模块没接线，不是这两个函数多余。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.perception import teleport_ocr as tp  # noqa: E402


#: 合成传送点表。**不读真数据** —— 936 个点的 JSON 在仓库外，
#: 而且真数据的字段名一变，测试就该跟着变，那不是它该守的东西。
#:
#: 字段名照抄生产代码的用法（``_score_pair`` / ``match_teleport``）：
#: ``id`` / ``name`` / ``areas`` / ``position`` / ``tran_position``。
def _pt(pid, name, areas, x, z):
    return {
        "id": pid,
        "name": name,
        "areas": list(areas),
        "position": [x, 50.0, z],
        # 传送后角色实际站的位置（传送契约 §4：落点取 tran_position）
        "tran_position": [x + 2.0, 50.0, z + 2.0],
    }


POINTS = [
    _pt("t1", "七天神像", ["蒙德", "蒙德城"], 100.0, 200.0),
    _pt("t2", "风起地", ["蒙德", "蒙德城"], 300.0, 400.0),
    _pt("t3", "望风山地", ["蒙德", "千风神殿"], 500.0, 600.0),
]


# ==================================================================
# 归一化
# ==================================================================

@pytest.mark.unit
def test_norm__strips_whitespace_and_normalizes_width() -> None:
    """OCR 会把全角括号、连接符搞乱 —— 归一化是匹配能不能成立的前提。"""
    assert tp._norm("  风 起　地 ") == "风起地"
    assert tp._norm("蒙德－城") == tp._norm("蒙德-城")


@pytest.mark.unit
def test_norm__handles_none_without_blowing_up() -> None:
    """副标题经常是空的 —— 不该在这一层炸。"""
    assert tp._norm(None) == ""
    assert tp._norm("") == ""


# ==================================================================
# 编辑距离 / 相似度
# ==================================================================

@pytest.mark.unit
def test_edit_distance__identical_strings_cost_nothing() -> None:
    assert tp._edit_distance("风起地", "风起地", 5) == 0


@pytest.mark.unit
def test_edit_distance__one_substitution_costs_one() -> None:
    assert tp._edit_distance("风起地", "风起池", 5) == 1


@pytest.mark.unit
def test_edit_distance__respects_the_cap_and_early_exits() -> None:
    """超 cap 直接返回 cap+1 —— 这是性能优化，也是个上限。"""
    assert tp._edit_distance("abc", "xyzxyzxyz", 2) == 3


@pytest.mark.unit
def test_edit_distance__length_gap_larger_than_cap_early_exits() -> None:
    assert tp._edit_distance("a", "a" * 50, 2) == 3


@pytest.mark.unit
def test_sim__identical_strings_are_fully_similar() -> None:
    assert tp._sim("风起地", "风起地") == pytest.approx(1.0)


@pytest.mark.unit
def test_sim__unrelated_strings_score_low() -> None:
    """⚠️ 别用「风起地 vs 望风山地」当反例 —— 共享「风」「地」，
    ``SequenceMatcher`` 给出 0.57，比看起来高得多。"""
    assert tp._sim("风起地", "枫丹廷") < 0.4


@pytest.mark.unit
def test_sim__empty_input_scores_zero_not_nan() -> None:
    """空串是副标题的常态 —— 不能在这一层炸。"""
    assert tp._sim("", "风起地") == 0.0
    assert tp._sim("", "") == 0.0


# ==================================================================
# 成对打分 —— 「任一不过直接 None」是契约的核心
# ==================================================================

#: 真正的接受门限。``_score_pair`` 返回的是**加权原始分**
#: （主 0.4 + 副 0.6），而 ``match_teleport`` 用的是
#: ``MATCH_SIMILARITY * 0.75`` 作为门槛。所以「能不能匹配上」要跟后者比，
#: 不是跟 ``MATCH_SIMILARITY`` 比 —— 拿错基准会写出永远红的断言。
ACCEPT = tp.MATCH_SIMILARITY * 0.75


@pytest.mark.unit
def test_score_pair__perfect_match_clears_the_gate() -> None:
    assert tp._score_pair("风起地", "蒙德", POINTS[1]) >= ACCEPT


@pytest.mark.unit
def test_score_pair__accepts_the_composite_name_form() -> None:
    """主标题在游戏里是「七天神像-水」这种复合形式。"""
    p = _pt("x", "七天神像", ["水"], 0.0, 0.0)
    assert tp._score_pair("七天神像-水", "水", p) is not None


@pytest.mark.unit
def test_score_pair__ocr_typo_still_clears_the_gate() -> None:
    """★ 这条是整个模块存在的理由：OCR 会认错字。

    差一个字必须仍然匹配上 —— 否则一次识别抖动就等于一次传送失败，
    而失败会静默跳过整个位置刷新（传送契约 §②）。

    ⚠️ 断言要跟 ``ACCEPT``（真门限）比，不是跟 ``MATCH_SIMILARITY`` 比：
    单字替换的原始分约 0.75，本就低于 0.8，但高于真门限 —— 匹配成立。
    """
    s = tp._score_pair("风起沲", "蒙德", POINTS[1])
    assert s is not None and s >= ACCEPT


@pytest.mark.unit
def test_match__a_single_typo_still_resolves_end_to_end() -> None:
    """端到端再验一次 —— 单元分够了不代表 ``match_teleport`` 会返回它。"""
    got = tp.match_teleport("风起沲", "蒙德", POINTS)
    assert got is not None and got.matched_id == "t2"


@pytest.mark.unit
def test_score_pair__wrong_name_returns_none() -> None:
    """名字对不上就该是 None —— 宁可判不出来，不可判错地方。"""
    assert tp._score_pair("望风山地", "蒙德", POINTS[1]) is None


@pytest.mark.unit
def test_score_pair__wrong_area_returns_none() -> None:
    """主标题对但地区不对 —— 跨地区同名点会误配，必须拦。"""
    assert tp._score_pair("风起地", "枫丹廷", POINTS[1]) is None


# ==================================================================
# 匹配
# ==================================================================

@pytest.mark.unit
def test_match__finds_the_right_point() -> None:
    got = tp.match_teleport("风起地", "蒙德", POINTS)
    assert got is not None
    assert got.matched_id == "t2"
    assert got.name == "风起地"


@pytest.mark.unit
def test_match__carries_tran_position_for_the_contract() -> None:
    """★ 传送契约 §4：落点取 ``tran_position``，不是 ``position``。

    角色传送后站在哪，和神像图标在哪，是两个坐标。
    """
    got = tp.match_teleport("风起地", "蒙德", POINTS)
    assert got is not None
    assert got.position == (300.0, 50.0, 400.0)
    assert got.tran_position == (302.0, 50.0, 402.0)


@pytest.mark.unit
def test_match__echoes_back_what_it_matched() -> None:
    """回传 OCR 原串 —— 排障时要能对得上屏幕上看到了什么。"""
    got = tp.match_teleport("风起地", "蒙德", POINTS)
    assert got.main_title == "风起地"
    assert got.sub_title == "蒙德"


@pytest.mark.unit
def test_match__returns_none_when_nothing_matches() -> None:
    assert tp.match_teleport("完全不相关", "火星", POINTS) is None


@pytest.mark.unit
def test_match__empty_catalogue_returns_none() -> None:
    assert tp.match_teleport("风起地", "蒙德", []) is None


# ==================================================================
# 图像处理
# ==================================================================

@pytest.mark.unit
def test_crop_info_box__returns_a_region_inside_the_frame() -> None:
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = tp.crop_info_box(frame)
    assert out is not None and out.size > 0
    h, w = out.shape[:2]
    assert 0 < h <= 720 and 0 < w <= 1280


@pytest.mark.unit
def test_crop_info_box__honours_a_custom_roi() -> None:
    frame = np.zeros((720, 1280, 3), np.uint8)
    a = tp.crop_info_box(frame, roi=(0.0, 0.0, 0.5, 0.5), margin=0.0)
    b = tp.crop_info_box(frame, roi=(0.0, 0.0, 0.25, 0.5), margin=0.0)
    assert a.shape[1] > b.shape[1], "更窄的 ROI 必须裁出更窄的图"
    assert a.shape[0] == b.shape[0]


@pytest.mark.unit
def test_crop_info_box__rejects_a_region_that_is_too_small() -> None:
    """★ 响亮失败：ROI 裁出来只剩几像素时抛，而不是返回一张废图。

    返回废图的后果是下游 OCR 在上面跑半天、什么都没认出来，
    最后报「传送点匹配失败」—— 症状离病因很远。
    """
    frame = np.zeros((720, 1280, 3), np.uint8)
    with pytest.raises(ValueError):
        tp.crop_info_box(frame, roi=(0.5, 0.5, 0.5005, 0.5005), margin=0.0)


@pytest.mark.unit
def test_crop_info_box__an_entirely_out_of_frame_roi_raises() -> None:
    """ROI 完全在画面外 —— 同样是响亮失败，不返回空切片。"""
    frame = np.zeros((720, 1280, 3), np.uint8)
    with pytest.raises(ValueError):
        tp.crop_info_box(frame, roi=(2.0, 2.0, 3.0, 3.0), margin=0.0)


@pytest.mark.unit
def test_preprocess__returns_both_polarities() -> None:
    """深底白字 / 浅底黑字两种 UI 都要覆盖 —— 契约要求同时返回。"""
    dark = np.full((40, 200, 3), 20, np.uint8)
    light = np.full((40, 200, 3), 230, np.uint8)
    for region in (dark, light):
        out = tp.preprocess(region)
        assert out is not None
        assert len(out) == 2, "应返回 (深底白字, 浅底黑字) 两种二值图"
        for img in out:
            if img is not None:
                assert img.ndim == 2, "二值图应是单通道"


# ==================================================================
# 行清洗
# ==================================================================

@pytest.mark.unit
def test_clean_lines__drops_non_chinese_noise() -> None:
    """OCR 会吐一堆符号碎片 —— 只留含中文的。"""
    got = tp._clean_lines(["风起地", "12345", "  ", "!!!", "蒙德"])
    assert got == ["风起地", "蒙德"]


@pytest.mark.unit
def test_clean_lines__drops_single_chinese_characters() -> None:
    """单个汉字多半是噪声 —— 契约要求至少 2 个字。"""
    assert tp._clean_lines(["风", "风起地"]) == ["风起地"]


@pytest.mark.unit
def test_clean_lines__handles_an_empty_list() -> None:
    assert tp._clean_lines([]) == []


@pytest.mark.unit
def test_clean_lines__None_is_not_supported_and_fails_loudly() -> None:
    """记下现状：``_clean_lines(None)`` 抛 ``TypeError``。

    上游 ``ocr_region`` 总是返回 list，所以这不是活 bug；
    但「以为它能吃 None」的人会踩到，所以把它写成事实而不是修掉。
    """
    with pytest.raises(TypeError):
        tp._clean_lines(None)


# ==================================================================
# 注入钩子 —— 「零引用」的那两个
# ==================================================================

@pytest.mark.unit
def test_set_ocr__injecting_a_fn_redirects_ocr_region() -> None:
    """★ 让 ``set_ocr`` 从「零引用」变成「被验证」。

    这是注入点（与 ``StateBuilder(locator=)`` 同类）—— 测试里注入假 OCR，
    正好验证契约本身，也能证明它确实接得上。
    """
    original = tp._OCR_FN
    try:
        tp.set_ocr(lambda region: ["风起地", "蒙德"])
        assert tp.ocr_region(np.zeros((40, 200, 3), np.uint8)) == \
            ["风起地", "蒙德"]
    finally:
        tp.set_ocr(original)


@pytest.mark.unit
def test_set_ocr__none_restores_the_default_backend() -> None:
    """``None`` = 恢复默认 —— 注入和撤销必须是成对的。"""
    original = tp._OCR_FN
    try:
        tp.set_ocr(lambda region: ["x"])
        assert tp._OCR_FN is not None
        tp.set_ocr(None)
        assert tp._OCR_FN is None
    finally:
        tp.set_ocr(original)


@pytest.mark.unit
def test_get_backend__reads_back_what_set_backend_wrote() -> None:
    """``set_backend`` / ``get_backend`` 是一对 —— 单留 getter 没有意义。"""
    original = tp.get_backend()
    try:
        tp.set_backend("winrt")
        assert tp.get_backend() == "winrt"
        tp.set_backend("rapidocr")
        assert tp.get_backend() == "rapidocr"
    finally:
        tp._OCR_BACKEND = original


@pytest.mark.unit
def test_backend_default_comes_from_the_environment() -> None:
    """``NEKO_OCR_BACKEND`` 是唯一的外部入口 —— 记下来免得找不到。"""
    assert tp._OCR_BACKEND in ("rapidocr", "winrt", "auto")


@pytest.mark.unit
def test_teleport_target__carries_everything_the_contract_needs() -> None:
    """记录现状：**它不是 frozen**。

    这条不是「应该冻结」的断言，而是「现在是可变的」这个事实 ——
    ``LocalizationService.on_teleport_ocr`` 拿到它之后会存进
    ``_pending_pos``（只取坐标），所以目前没人改它。
    真要改成 frozen 属于契约变更，那时这条测试会提醒你同步更新调用方。
    """
    t = tp.match_teleport("风起地", "蒙德", POINTS)
    assert t is not None
    assert t.matched_id and t.name
    assert isinstance(t.position, tuple) and len(t.position) == 3
    assert isinstance(t.tran_position, tuple) and len(t.tran_position) == 3
    assert 0.0 < t.confidence <= 1.0
    assert isinstance(t.areas, tuple)


@pytest.mark.unit
def test_module__exports_its_public_surface() -> None:
    """``__all__`` 是给 ``from ... import *`` 看的契约。"""
    for name in ("TeleportTarget", "match_teleport", "recognize_teleport",
                 "set_backend", "get_backend", "set_ocr"):
        assert name in tp.__all__, f"{name} 没进 __all__"

# ==================================================================
# 命中唯一性（2026-10-15）
#
# ★ 这组测试来自一次真机翻车
# ---------------------------
# OCR 只读到区域名（而非具体地名）时，``match_teleport`` 会在**几十个
# 同分点**里取第一个，当成一个确定结果交出去。实测匹配到 id=81，
# 落点 (548.8, 206.6, 341.5) —— 而角色实际在几千米外。定位原点一错，
# 之后每一条 MOVE_TO 都在错误的坐标系里走，角色撞墙不动，且**全程
# 没有任何报错**。
#
# 契约 ``teleport_contract.md`` §3① 明写「唯一命中；模糊/多候选都算
# 失败」—— 那是**要求**，而代码实现的是「取最高分的第一个」。
# ==================================================================


def _anchor(pid: str, areas: list[str], x: float, z: float) -> dict:
    return {"id": pid, "name": "传送锚点", "areas": areas,
            "position": (x, 0.0, z), "tran_position": (x, 0.0, z)}


class TestMatchUniqueness:
    @pytest.mark.unit
    def test_tied_top_scores_are_rejected(self) -> None:
        """★ 同分即失败 —— 这是真机那次翻车的根因。

        三个点的 ``name`` 与 ``areas`` 完全一致，分数必然相同。
        取第一个 = 随便挑一个 = 一个「看起来很确定」的错误原点。
        """
        pts = [_anchor("1", ["枫丹"], 100.0, 200.0),
               _anchor("2", ["枫丹"], 300.0, 400.0),
               _anchor("3", ["枫丹"], 500.0, 600.0)]
        assert tp.match_teleport("传送锚点", "枫丹", pts) is None

    @pytest.mark.unit
    def test_unique_match_still_works(self) -> None:
        """唯一命中必须照常通过 —— 不能为了安全把功能整个关掉。"""
        pts = [_anchor("1", ["枫丹廷·灰河", "枫丹"], 100.0, 200.0),
               _anchor("2", ["枫丹廷·绿河", "枫丹"], 300.0, 400.0)]
        got = tp.match_teleport("传送锚点", "枫丹廷·灰河", pts)
        assert got is not None
        assert got.matched_id == "1"

    @pytest.mark.unit
    def test_named_anchors_are_unambiguous(self) -> None:
        """具名传送（七天神像-水 等）名字本身唯一，不该被牵连。"""
        pts = [{"id": "a", "name": "七天神像-水", "areas": ["枫丹廷·灰河"],
                "position": (1.0, 2.0, 3.0), "tran_position": (4.0, 5.0, 6.0)},
               {"id": "b", "name": "七天神像-火", "areas": ["枫丹廷·灰河"],
                "position": (1.0, 2.0, 3.0), "tran_position": (4.0, 5.0, 6.0)}]
        got = tp.match_teleport("七天神像-水", "枫丹廷·灰河", pts)
        assert got is not None and got.matched_id == "a"

    @pytest.mark.unit
    def test_lower_score_runner_up_does_not_count_as_a_tie(self) -> None:
        """唯一性只看**并列最高**。低分的不算歧义。"""
        pts = [_anchor("1", ["枫丹廷·灰河"], 100.0, 200.0),
               _anchor("2", ["枫丹廷·绿河"], 300.0, 400.0)]
        got = tp.match_teleport("传送锚点", "枫丹廷·灰河", pts)
        assert got is not None and got.matched_id == "1"

    @pytest.mark.unit
    def test_same_landing_point_tie_is_still_allowed(self) -> None:
        """★ 落点**相同**的并列不算歧义 —— 取哪个都一样。

        数据表里有这种情况（同一处拆成两条记录）。因为「你要去哪」
        关心的是坐标，不是 id。
        """
        pts = [{"id": "1", "name": "传送锚点", "areas": ["枫丹"],
                "position": (100.0, 0.0, 200.0),
                "tran_position": (100.0, 0.0, 200.0)},
               {"id": "2", "name": "传送锚点", "areas": ["枫丹"],
                "position": (100.0, 0.0, 200.0),
                "tran_position": (100.0, 0.0, 200.0)}]
        got = tp.match_teleport("传送锚点", "枫丹", pts)
        assert got is not None, "落点相同却判歧义 -> 白白丢掉一次定位"
        assert got.tran_position == (100.0, 0.0, 200.0)

    @pytest.mark.unit
    def test_tie_with_different_landing_points_is_rejected(self) -> None:
        """落点不同却并列 -> 必须拒绝。那两个「看起来一样」的地方差 737m。"""
        pts = [_anchor("1", ["枫丹"], 100.0, 200.0),
               _anchor("2", ["枫丹"], 837.4, 200.0)]
        assert tp.match_teleport("传送锚点", "枫丹", pts) is None
