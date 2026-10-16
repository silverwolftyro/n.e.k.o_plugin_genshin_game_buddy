"""``perception/yolo.py`` 的纯函数部分（2026-10-12 任务 C）。

为什么只测这三个
----------------
``YoloDetector`` 本体要真 ONNX + DirectML/CPU 会话，离线跑不动。但模块里
**真正容易出错**的是三个纯函数，它们决定「模型吐出来的数字」怎么变成
「屏幕上的框」，而这一段曾经出过两个真 bug：

* ``_nms`` 在 Python list 上用 ``boxes[rest, 0]`` 取列 -> ``TypeError`` ->
  **任何 ≥2 个检测框的帧整个 detect() 静默返回空**（异常被 detect 吞掉），
  而 0/1 个框时提前 return 所以看不出来。
* ``_postprocess`` 要兼容两种导出形态，转置判断写错就会整片读成错的类别。

所以本文件全部用**合成张量**驱动，不需要模型、不需要显卡。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.perception.yolo import (  # noqa: E402
    DEFAULT_TIMEOUT_MS,
    PROVIDER_PREFERENCE,
    Detection,
    YoloUnavailableError,
    _letterbox,
    _nms,
    _postprocess,
    _postprocess_end2end,
)

CLASSES = ["enemy", "ore", "chest"]


# ==================================================================
# Detection
# ==================================================================

@pytest.mark.unit
def test_detection__center_is_the_middle_of_the_box() -> None:
    d = Detection("enemy", (10, 20, 100, 40), 0.9)
    assert d.center == (60, 40)


@pytest.mark.unit
def test_detection__is_json_serialisable() -> None:
    import json
    json.dumps(Detection("ore", (1, 2, 3, 4), 0.5).to_dict())


# ==================================================================
# provider 偏好
# ==================================================================

@pytest.mark.unit
def test_provider_preference__puts_gpu_backends_before_cpu() -> None:
    """DirectML / CUDA 是这个项目的命门 —— 排最后就等于没有。"""
    names = [p.upper() for p in PROVIDER_PREFERENCE]
    assert names.index("DMLEXECUTIONPROVIDER") < names.index("CPUEXECUTIONPROVIDER"), (
        f"顺序错了：{PROVIDER_PREFERENCE}")


@pytest.mark.unit
def test_provider_preference__ends_with_the_cpu_fallback() -> None:
    """必须有 CPU 兜底，否则没显卡的机器直接起不来。"""
    assert PROVIDER_PREFERENCE[-1] == "CPUExecutionProvider"


@pytest.mark.unit
def test_default_timeout_ms__is_a_sane_tick_budget() -> None:
    """500ms 已超出一个 200ms tick —— 超太久只会拖垮整条链路。"""
    assert 0 < DEFAULT_TIMEOUT_MS <= 1000


@pytest.mark.unit
def test_yolo_unavailable_error__is_a_runtime_error() -> None:
    """调用方普遍按 RuntimeError 兜底，降级路径才有意义。"""
    assert issubclass(YoloUnavailableError, RuntimeError)


# ==================================================================
# _letterbox
# ==================================================================

@pytest.mark.unit
def test_letterbox__always_produces_a_square_nchw_blob() -> None:
    img = np.zeros((480, 640, 3), np.uint8)
    blob, _r, _pad = _letterbox(img, 640)
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32


@pytest.mark.unit
def test_letterbox__normalises_to_0_1() -> None:
    """模型要的是归一化输入；忘了除 255 会让整帧变成纯白。"""
    img = np.full((100, 100, 3), 255, np.uint8)
    blob, _r, _pad = _letterbox(img, 128)
    assert blob.max() <= 1.0


@pytest.mark.unit
def test_letterbox__preserves_aspect_ratio() -> None:
    """等比缩放 —— 变形会让检测框全歪。"""
    img = np.zeros((480, 640, 3), np.uint8)
    _blob, ratio, _pad = _letterbox(img, 640)
    assert ratio == pytest.approx(1.0)


@pytest.mark.unit
def test_letterbox__ratio_is_the_binding_dimension_for_a_wide_image() -> None:
    img = np.zeros((100, 400, 3), np.uint8)
    _blob, ratio, _pad = _letterbox(img, 200)
    assert ratio == pytest.approx(0.5)      # 200/400 才是约束


@pytest.mark.unit
def test_letterbox__pad_is_roughly_centred() -> None:
    """padding 偏一边会把内容整体挪走，框还原时偏移就错。"""
    img = np.zeros((100, 400, 3), np.uint8)
    _blob, _ratio, (dx, dy) = _letterbox(img, 200)
    assert dy == pytest.approx(75, abs=2)    # (200 - 50) // 2
    assert dx == 0                          # 宽度刚好，不补


@pytest.mark.unit
def test_letterbox__is_deterministic() -> None:
    """同一张图两次必须完全一致，否则检测框会随机抖。"""
    img = np.arange(90 * 160 * 3, dtype=np.uint8).reshape(90, 160, 3)
    a, _r1, _p1 = _letterbox(img, 128)
    b, _r2, _p2 = _letterbox(img, 128)
    assert np.array_equal(a, b)


# ==================================================================
# _postprocess —— 坐标还原
# ==================================================================

def _row(cx, cy, w, h, scores):
    """一条 ultralytics 格式的预测：4 个框属性 + 各类分数。"""
    return [cx, cy, w, h, *scores]


@pytest.mark.unit
def test_postprocess__maps_a_centre_box_back_to_the_original_frame() -> None:
    """没有 pad、ratio=1 时，cx/cy/w/h 应该原样还原。"""
    out = np.array([[_row(50, 60, 20, 10, [0.9, 0.0, 0.0])]], dtype=np.float32)
    got = _postprocess(out, 1.0, (0, 0), (200, 200), CLASSES, 0.3, 0.5)
    assert len(got) == 1
    x, y, w, h = got[0].box
    assert (x, w) == (40, 20) and (y, h) == (55, 10)


@pytest.mark.unit
def test_postprocess__undoes_the_ratio() -> None:
    """letterbox 缩小过的图，还原时必须除回 ratio，否则框小一圈。"""
    out = np.array([[_row(50, 50, 20, 20, [0.9, 0, 0])]], dtype=np.float32)
    got = _postprocess(out, 0.5, (0, 0), (400, 400), CLASSES, 0.3, 0.5)
    x, y, w, h = got[0].box
    assert w == pytest.approx(40, abs=2)


@pytest.mark.unit
def test_postprocess__subtracts_the_letterbox_padding() -> None:
    out = np.array([[_row(60, 50, 20, 20, [0.9, 0, 0])]], dtype=np.float32)
    got = _postprocess(out, 1.0, (10, 0), (400, 400), CLASSES, 0.3, 0.5)
    x, _y, _w, _h = got[0].box
    assert x == pytest.approx(40, abs=2)


@pytest.mark.unit
def test_postprocess__picks_the_highest_scoring_class() -> None:
    out = np.array([[_row(50, 50, 20, 20, [0.1, 0.8, 0.2])]], dtype=np.float32)
    got = _postprocess(out, 1.0, (0, 0), (200, 200), CLASSES, 0.3, 0.5)
    assert got[0].label == "ore"


# ==================================================================
# _postprocess —— 阈值 / 退化
# ==================================================================

@pytest.mark.unit
def test_postprocess__drops_rows_below_the_confidence_threshold() -> None:
    out = np.array([[_row(50, 50, 20, 20, [0.1, 0, 0])]], dtype=np.float32)
    assert _postprocess(out, 1.0, (0, 0), (200, 200), CLASSES, 0.3, 0.5) == []


@pytest.mark.unit
def test_postprocess__drops_degenerate_boxes() -> None:
    """1×1 的框要么是噪声要么是填充，放行会让点击器往那儿戳。"""
    out = np.array([[_row(50, 50, 1, 1, [0.9, 0, 0])]], dtype=np.float32)
    assert _postprocess(out, 1.0, (0, 0), (200, 200), CLASSES, 0.3, 0.5) == []


@pytest.mark.unit
def test_postprocess__clamps_boxes_inside_the_frame() -> None:
    """负坐标会让下游 ROI 切片行为诡异。"""
    out = np.array([[_row(2, 2, 40, 40, [0.9, 0, 0])]], dtype=np.float32)
    got = _postprocess(out, 1.0, (0, 0), (200, 200), CLASSES, 0.3, 0.5)
    x, y, w, h = got[0].box
    assert x >= 0 and y >= 0
    assert x + w <= 200 and y + h <= 200


@pytest.mark.unit
def test_postprocess__rejects_a_bare_one_dimensional_array() -> None:
    assert _postprocess(np.zeros(10, np.float32), 1.0, (0, 0),
                        (100, 100), CLASSES, 0.3, 0.5) == []


@pytest.mark.unit
def test_postprocess__rejects_an_array_with_too_few_columns() -> None:
    """类别数对不上 -> 解析不了，宁可给空也不能给错的。"""
    out = np.zeros((1, 5), np.float32)
    assert _postprocess(out, 1.0, (0, 0), (100, 100), CLASSES, 0.3, 0.5) == []


@pytest.mark.unit
def test_postprocess__handles_both_export_layouts_identically() -> None:
    """ultralytics 默认 [1, 4+nc, N]，部分导出器给 [1, N, 4+nc]。

    两种都要认 —— 转置判断写错就会整片读成错的类别。
    """
    cols = _row(50, 50, 20, 20, [0.1, 0.9, 0.0])
    a = _postprocess(np.array([cols], np.float32), 1.0, (0, 0),
                     (200, 200), CLASSES, 0.3, 0.5)
    b = _postprocess(np.array([cols], np.float32).T, 1.0, (0, 0),
                     (200, 200), CLASSES, 0.3, 0.5)
    assert [d.label for d in a] == [d.label for d in b] == ["ore"]


# ==================================================================
# _postprocess_end2end
# ==================================================================

@pytest.mark.unit
def test_end2end__reads_xyxy_score_class_layout() -> None:
    pred = np.array([[10, 20, 60, 50, 0.9, 1]], np.float32)
    got = _postprocess_end2end(pred, (200, 200), CLASSES, 0.3)
    assert len(got) == 1
    assert got[0].label == "ore"
    assert got[0].box == (10, 20, 50, 30)


@pytest.mark.unit
def test_end2end__applies_the_confidence_threshold() -> None:
    pred = np.array([[10, 20, 60, 50, 0.1, 1]], np.float32)
    assert _postprocess_end2end(pred, (200, 200), CLASSES, 0.5) == []


@pytest.mark.unit
def test_end2end__names_unknown_classes_instead_of_crashing() -> None:
    """模型类目数与配置不一致时不能 IndexError —— 那会让整帧 detect 崩。"""
    pred = np.array([[10, 20, 60, 50, 0.9, 99]], np.float32)
    got = _postprocess_end2end(pred, (200, 200), CLASSES, 0.3)
    assert got[0].label == "class_99"


# ==================================================================
# _nms
# ==================================================================

def _det(label, box, score):
    return Detection(label, box, score)


@pytest.mark.unit
def test_nms__single_detection_passes_through() -> None:
    d = [_det("enemy", (0, 0, 10, 10), 0.9)]
    assert _nms(d, [d[0].box], [0.9], 0.5) == d


@pytest.mark.unit
def test_nms__suppresses_a_near_duplicate_of_the_same_class() -> None:
    """重叠的同类框必须合并 —— 不然同一个敌人被算两次。"""
    dets = [_det("enemy", (0, 0, 100, 100), 0.9),
            _det("enemy", (5, 5, 100, 100), 0.8)]
    got = _nms(dets, [d.box for d in dets], [d.conf for d in dets], 0.5)
    assert len(got) == 1
    assert got[0].conf == 0.9, "应保留分数更高的那个"


@pytest.mark.unit
def test_nms__keeps_overlapping_boxes_of_different_classes() -> None:
    """class-aware：不同类不互相抑制。"""
    dets = [_det("enemy", (0, 0, 100, 100), 0.9),
            _det("ore", (5, 5, 100, 100), 0.8)]
    got = _nms(dets, [d.box for d in dets], [d.conf for d in dets], 0.5)
    assert len(got) == 2


@pytest.mark.unit
def test_nms__keeps_distant_boxes() -> None:
    dets = [_det("enemy", (0, 0, 20, 20), 0.9),
            _det("enemy", (500, 500, 20, 20), 0.8)]
    got = _nms(dets, [d.box for d in dets], [d.conf for d in dets], 0.5)
    assert len(got) == 2


@pytest.mark.unit
def test_nms__two_or_more_boxes_does_not_raise() -> None:
    """★ 这条钉住一个真实回归。

    ``_nms`` 曾在 Python list 上写 ``boxes[rest, 0]`` 取列 ->
    ``TypeError`` -> 被 ``detect()`` 吞掉 -> **任何 ≥2 个框的帧静默返回空**。
    而 0/1 个框提前 return，所以这个 bug 能活很久。
    """
    dets = [_det("enemy", (0, 0, 10, 10), 0.9), _det("enemy", (50, 50, 10, 10), 0.8)]
    assert len(_nms(dets, [d.box for d in dets], [d.conf for d in dets], 0.5)) == 2


@pytest.mark.unit
def test_nms__results_come_back_sorted_by_score() -> None:
    dets = [_det("enemy", (0, 0, 10, 10), 0.5),
            _det("ore", (200, 200, 10, 10), 0.9)]
    got = _nms(dets, [d.box for d in dets], [d.conf for d in dets], 0.5)
    assert [d.conf for d in got] == [0.9, 0.5]


@pytest.mark.unit
def test_nms__does_not_mutate_the_input_list() -> None:
    """返回的是新列表；就地改会破坏调用方的顺序假设。"""
    dets = [_det("enemy", (0, 0, 10, 10), 0.5),
            _det("enemy", (100, 100, 10, 10), 0.9)]
    before = list(dets)
    _nms(dets, [d.box for d in dets], [d.conf for d in dets], 0.5)
    assert dets == before


@pytest.mark.unit
def test_nms__empty_input_returns_empty() -> None:
    assert _nms([], [], [], 0.5) == []


@pytest.mark.unit
def test_postprocess__runs_nms_and_returns_fewer_boxes() -> None:
    """端到端串一次：两个高度重叠的同类框进，两个框出。"""
    out = np.array([
        _row(50, 50, 100, 100, [0.9, 0, 0]),
        _row(52, 52, 100, 100, [0.85, 0, 0]),
    ], np.float32)
    got = _postprocess(out, 1.0, (0, 0), (400, 400), CLASSES, 0.3, 0.5)
    assert len(got) == 1
