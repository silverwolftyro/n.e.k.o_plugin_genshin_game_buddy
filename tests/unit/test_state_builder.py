"""``perception/builder.py`` 的单测（2026-10-11 任务 D）。

为什么之前说它「无法测试」
--------------------------
同上轮结论：**那个结论是错的**。``StateBuilder.__init__`` 的九个依赖
（capturer / yolo / template / ocr / roi / locator / entity_detector /
scene_debouncer / scene_detector）**全部是关键字参数且默认可空**。真正需要
避开的只有两个：

* ``entity_detector="auto"``（**默认值**）会探测并加载 10 MB 的 GPL ONNX
  —— 单次约 **1243 ms**。测试一律显式传 ``None`` 关闭。
* ``scene_detector=None`` 时会默认 new 一个真 ``SceneDetector``，
  依赖 ``coord_scene_tree.py``。测试一律注入假件。

所以「需要先做依赖注入改造」这个前提不存在。

本文件覆盖
----------
* 节流间隔 / 盒缩放 / 亮度等**纯函数**
* 构造期的规格校验（未知 entity_detector、yolo 弃用告警）
* ``build()`` 的三条早退：空帧、黑帧、时间戳倒退
* 正常路径：场景判定、``forced_scene`` 覆盖、``entered_ms``、``state_version``
* OCR 健康度与「OCR 挂了不影响 perception_ok」
* ``_track_scene_enter``（``since()`` 的数据源）
* ROI 表达式求值
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.perception.builder import (  # noqa: E402
    BLACK_FRAME_LUMA,
    DEFAULT_ENTITY_INTERVAL_TICKS,
    ENTITY_SPECS,
    LOCATOR_FAIL_THRESHOLD,
    LOCATOR_RECOVER_THRESHOLD,
    OCR_FAIL_THRESHOLD,
    OCR_RECOVER_THRESHOLD,
    YOLO_FAIL_THRESHOLD,
    YOLO_RECOVER_THRESHOLD,
    StateBuilder,
    _luma,
    _pick_entity_interval,
    _scale_box,
    clear_auto_cache,
)
from autopilot.perception.capturer import CaptureChannel, CapturedFrame  # noqa: E402
from autopilot.perception.screen_region import region_of_frame  # noqa: E402

# ==================================================================
# 假件
# ==================================================================

class _SceneState:
    def __init__(self, is_world=True, changed=False):
        self.is_world = is_world
        self.changed = changed


class _SceneDetector:
    """假场景判定器。``script`` 是一次一帧的返回值队列，用完就停在最后一个。"""

    def __init__(self, script=None, default=("world", True)):
        self.script = list(script or [])
        self._default = default
        self.calls = []

    def update(self, image, ts_sec):
        self.calls.append(ts_sec)
        if self.script:
            is_world, changed = self.script.pop(0)
            return _SceneState(is_world, changed)
        is_world, changed = self._default
        return _SceneState(is_world, changed)


class _Ocr:
    def __init__(self, available=True, last_error=""):
        self.available = available
        self.last_error = last_error


def _builder(**kw) -> StateBuilder:
    """默认关掉实体检测、注入假场景判定器 —— 纯离线、零模型加载。"""
    kw.setdefault("entity_detector", None)
    kw.setdefault("scene_detector", _SceneDetector())
    return StateBuilder(**kw)


def _image(value: int = 120, h: int = 720, w: int = 1280) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def _frame(image=None, ts_ms=None, seq: int = 1) -> CapturedFrame:
    return CapturedFrame(
        image=_image() if image is None else image,
        rect=(0, 0, 1280, 720),
        timestamp_ms=1_700_000_000_000 if ts_ms is None else ts_ms,
        frame_seq=seq,
        channel=CaptureChannel.BITBLT,
    )


def _codes(b: StateBuilder) -> list[str]:
    return [e.code for e in b.drain_events()]


# ==================================================================
# 纯函数
# ==================================================================

@pytest.mark.unit
def test_pick_entity_interval__explicit_beats_detector() -> None:
    """显式参数 > 检测器自带的建议。"""
    assert _pick_entity_interval(3, 9) == 3


@pytest.mark.unit
def test_pick_entity_interval__falls_back_to_detector_suggestion() -> None:
    assert _pick_entity_interval(None, 9) == 9


@pytest.mark.unit
def test_pick_entity_interval__default_when_nothing_says_anything() -> None:
    assert _pick_entity_interval(None, None) == DEFAULT_ENTITY_INTERVAL_TICKS


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, -10])
def test_pick_entity_interval__rejects_non_positive(bad: int) -> None:
    """间隔必须 >= 1：传 0 会让 ``% n`` 直接 ZeroDivisionError。

    响亮失败，不静默纠正成默认值。
    """
    with pytest.raises(ValueError):
        _pick_entity_interval(bad, None)


@pytest.mark.unit
def test_scale_box__scales_origin_and_size() -> None:
    assert _scale_box((10, 20, 100, 50), 0.5) == (5, 10, 50, 25)


@pytest.mark.unit
def test_scale_box__is_used_for_thumbnail_overlay_math() -> None:
    """缩放后必须是四元组，不能退化成别的形状。"""
    out = _scale_box((3, 7, 11, 13), 2.0)
    assert isinstance(out, tuple) and len(out) == 4
    assert out == (6, 14, 22, 26)


@pytest.mark.unit
def test_luma__black_frame_is_near_zero() -> None:
    assert _luma(_image(0)) == pytest.approx(0.0, abs=1.0)


@pytest.mark.unit
def test_luma__white_frame_is_near_255() -> None:
    assert _luma(_image(255)) == pytest.approx(255.0, abs=1.0)


@pytest.mark.unit
def test_black_frame_threshold__is_a_real_cut_between_the_two() -> None:
    assert _luma(_image(0)) < BLACK_FRAME_LUMA < _luma(_image(255))


# ==================================================================
# 规格常量
# ==================================================================

@pytest.mark.unit
def test_entity_specs__covers_exactly_the_three_keywords() -> None:
    assert ENTITY_SPECS == {"auto", "force_on", "force_off"}


@pytest.mark.unit
@pytest.mark.parametrize("pair", [
    (YOLO_FAIL_THRESHOLD, YOLO_RECOVER_THRESHOLD),
    (OCR_FAIL_THRESHOLD, OCR_RECOVER_THRESHOLD),
    (LOCATOR_FAIL_THRESHOLD, LOCATOR_RECOVER_THRESHOLD),
])
def test_guard_thresholds__all_positive_and_recoverable(pair) -> None:
    """每条熔断都必须有阈值且能复活 —— 只有阈值没有复活的会永久卡死。"""
    fail_t, recover_t = pair
    assert fail_t >= 1
    assert recover_t >= 1


# ==================================================================
# 构造期
# ==================================================================

@pytest.mark.unit
def test_builder__entity_detector_none_means_disabled() -> None:
    """默认是 "auto"（要加载 GPL 模型），测试必须显式关掉。"""
    b = _builder(entity_detector=None)
    assert b._entity_detector is None


@pytest.mark.unit
def test_builder__force_off_skips_model_loading_entirely() -> None:
    b = _builder(entity_detector="force_off")
    assert b._entity_detector is None


@pytest.mark.unit
def test_builder__unknown_entity_spec_is_rejected_loudly() -> None:
    with pytest.raises(ValueError) as ei:
        _builder(entity_detector="on")
    assert "auto" in str(ei.value)


@pytest.mark.unit
def test_builder__yolo_argument_warns_that_it_is_inert() -> None:
    """§28 起 YOLO 退出场景判定，但保留签名 —— 传它必须有弃用告警，
    否则调用方以为它还在生效。"""
    with pytest.warns(DeprecationWarning):
        StateBuilder(yolo=object(), entity_detector=None,
                     scene_detector=_SceneDetector())


@pytest.mark.unit
def test_builder__locator_defaults_to_none() -> None:
    assert _builder()._locator is None


@pytest.mark.unit
def test_clear_auto_cache__reports_whether_it_had_anything() -> None:
    first = clear_auto_cache()
    assert isinstance(first, bool)
    assert clear_auto_cache() is False, "刚清过就该是 False"


# ==================================================================
# build() 的早退分支
# ==================================================================

@pytest.mark.unit
def test_build__none_frame_yields_unknown_package() -> None:
    b = _builder()
    pkt = b.build(None)
    assert pkt.resource.perception_ok is False
    assert "SCREENSHOT_FAIL" in _codes(b)


@pytest.mark.unit
def test_build__empty_image_yields_unknown_package() -> None:
    b = _builder()
    pkt = b.build(_frame(image=np.zeros((0, 0, 3), np.uint8)))
    assert pkt.resource.perception_ok is False


@pytest.mark.unit
def test_build__black_frame_is_caught_before_any_heavy_work() -> None:
    """黑帧要**廉价地**救掉「游戏没开/切屏」，所以在场景判定之前。"""
    det = _SceneDetector()
    b = _builder(scene_detector=det)
    pkt = b.build(_frame(image=_image(0)))
    assert pkt.resource.perception_ok is False
    assert "BLACK_FRAME" in _codes(b)
    assert det.calls == [], "黑帧不该再去问场景判定器"


@pytest.mark.unit
def test_build__timestamp_going_backwards_raises() -> None:
    """★ 不钳零。倒退会让去抖计时错乱并一路污染 ``entered_ms``。"""
    b = _builder()
    b.build(_frame(ts_ms=1_700_000_000_000))
    with pytest.raises(ValueError) as ei:
        b.build(_frame(ts_ms=1_699_999_999_000))
    assert "时间戳倒退" in str(ei.value)


@pytest.mark.unit
def test_build__equal_timestamp_is_allowed() -> None:
    """只有**倒退**才抛；相同毫秒不算。"""
    b = _builder()
    b.build(_frame(ts_ms=1_700_000_000_000))
    b.build(_frame(ts_ms=1_700_000_000_000, seq=2))   # 不抛就是过


# ==================================================================
# 正常路径
# ==================================================================

@pytest.mark.unit
def test_build__healthy_frame_produces_a_usable_packet() -> None:
    b = _builder()
    pkt = b.build(_frame())
    assert pkt.resource.perception_ok is True
    assert pkt.scene.id == "WORLD"
    assert pkt.scene.confidence == 1.0


@pytest.mark.unit
def test_build__not_world_scene_is_unknown_not_a_fake_world() -> None:
    """宁可判不出来，也不要假装在大世界。"""
    b = _builder(scene_detector=_SceneDetector(default=(False, False)))
    pkt = b.build(_frame())
    assert pkt.scene.id == "UNKNOWN"
    assert pkt.scene.confidence == 0.0


@pytest.mark.unit
def test_build__forced_scene_overrides_the_detector() -> None:
    """显式覆盖优先，并且**不沿用**二值结果的 confidence。"""
    b = _builder(scene_detector=_SceneDetector(default=(False, False)))
    pkt = b.build(_frame(), forced_scene="DIALOG")
    assert pkt.scene.id == "DIALOG"
    assert pkt.scene.confidence == 1.0
    assert "forced" in pkt.scene.evidence


@pytest.mark.unit
def test_build__state_version_increases_every_frame() -> None:
    b = _builder()
    vs = [b.build(_frame(seq=i)).state_version for i in range(1, 4)]
    assert vs == [1, 2, 3], "state_version 必须每帧递增且从 1 起"


@pytest.mark.unit
def test_build__position_is_none_without_a_locator() -> None:
    """未注入 locator 时 position 恒为 None —— **绝不用预测值填补**。"""
    assert _builder().build(_frame()).position is None


@pytest.mark.unit
def test_build__records_last_detections_for_the_overlay() -> None:
    """调试叠加层画的必须是本帧真实产出，不是另算一份。"""
    b = _builder()
    b.build(_frame())
    assert isinstance(b.last_detections, list)
    assert isinstance(b.last_matches, list)


@pytest.mark.unit
def test_build__entered_ms_is_always_populated() -> None:
    """★ ``since()`` 的数据源。恒为整数毫秒，不许是 None。"""
    pkt = _builder().build(_frame())
    assert isinstance(pkt.scene.entered_ms, int)
    assert pkt.scene.entered_ms > 0


# ==================================================================
# 场景进入时刻（since() 的上游）
# ==================================================================

@pytest.mark.unit
def test_track_scene_enter__first_sighting_records_the_moment() -> None:
    b = _builder()
    t0 = b._track_scene_enter("WORLD", changed=True)
    assert b._scene_entered_ms["WORLD"] == t0


@pytest.mark.unit
def test_track_scene_enter__same_scene_keeps_the_original_moment() -> None:
    """每帧比对原始 id 会被单帧抖动带跑，所以靠 changed 沿触发。"""
    b = _builder()
    first = b._track_scene_enter("WORLD", changed=True)
    time.sleep(0.01)
    assert b._track_scene_enter("WORLD", changed=False) == first


@pytest.mark.unit
def test_track_scene_enter__change_flag_refreshes_the_moment() -> None:
    b = _builder()
    first = b._track_scene_enter("WORLD", changed=True)
    time.sleep(0.01)
    assert b._track_scene_enter("WORLD", changed=True) > first


@pytest.mark.unit
def test_track_scene_enter__different_scene_gets_its_own_entry() -> None:
    b = _builder()
    b._track_scene_enter("WORLD", changed=True)
    b._track_scene_enter("UNKNOWN", changed=True)
    assert set(b._scene_entered_ms) == {"WORLD", "UNKNOWN"}


@pytest.mark.unit
def test_track_scene_enter__unseen_scene_is_entered_implicitly() -> None:
    """即使 changed=False，没见过的场景也必须有进入时刻。"""
    b = _builder()
    assert isinstance(b._track_scene_enter("DIALOG", changed=False), int)


# ==================================================================
# OCR 健康度
# ==================================================================

@pytest.mark.unit
def test_ocr_health__no_engine_means_healthy() -> None:
    assert _builder()._ocr_health(1) == (True, 0)


@pytest.mark.unit
def test_ocr_health__available_engine_resets_the_streak() -> None:
    b = _builder(ocr=_Ocr(available=True))
    assert b._ocr_health(1) == (True, 0)
    assert b._ocr_health(2) == (True, 0)


@pytest.mark.unit
def test_ocr_health__unavailable_engine_grows_the_streak() -> None:
    b = _builder(ocr=_Ocr(available=False, last_error="超时"))
    assert b._ocr_health(1) == (True, 1), "未判死前仍可用，只是有 streak"
    assert b._ocr_health(2) == (True, 2)


@pytest.mark.unit
def test_ocr_health__dies_after_the_threshold_and_announces_it() -> None:
    b = _builder(ocr=_Ocr(available=False))
    for _ in range(OCR_FAIL_THRESHOLD):
        ok, _streak = b._ocr_health(1)
    assert ok is False, "到阈值应判死"
    codes = _codes(b)
    assert "OCR_DEAD" in codes
    assert "OCR_FAILED" in codes


@pytest.mark.unit
def test_ocr_health__failure_never_touches_perception_ok() -> None:
    """★ 用户 2026-10-02 拍板：OCR 走自己的专属通道，不翻总开关。"""
    b = _builder(ocr=_Ocr(available=False))
    for _ in range(OCR_FAIL_THRESHOLD + 3):
        pkt = b.build(_frame())
    assert pkt.resource.ocr_available is False
    assert pkt.resource.perception_ok is True


@pytest.mark.unit
def test_ocr_health__recovers_after_sustained_availability() -> None:
    b = _builder(ocr=_Ocr(available=False))
    for _ in range(OCR_FAIL_THRESHOLD + 1):
        b._ocr_health(1)
    _codes(b)
    b._ocr = _Ocr(available=True)
    for _ in range(OCR_RECOVER_THRESHOLD):
        b._ocr_health(2)
    assert "OCR_RECOVERED" in _codes(b)


@pytest.mark.unit
def test_compute_perception_ok__is_true_by_default() -> None:
    """§28 之后这个信号实际恒为 True（YOLO 已退出场景判定）。

    这里钉住现状：真要恢复可观测性，得先给场景判定器补健康信号。
    """
    assert _builder()._compute_perception_ok() is True


# ==================================================================
# 事件队列
# ==================================================================

@pytest.mark.unit
def test_drain_events__empties_the_queue() -> None:
    b = _builder()
    b.build(_frame(image=_image(0)))       # 产生 BLACK_FRAME
    assert len(b.drain_events()) >= 1
    assert b.drain_events() == [], "drain 之后必须清空，不能重复投递"


# ==================================================================
# ROI 表达式
# ==================================================================

@pytest.mark.unit
def test_roi__no_expression_means_search_whole_region() -> None:
    b = _builder()
    region = region_of_frame(_image())
    assert b._roi_box_for("hp_bar", region) is None


@pytest.mark.unit
def test_roi__valid_expression_resolves_against_the_frame() -> None:
    b = _builder()
    b.set_roi({"hp_bar": "rect(cw*0.72, ch*0.55, cw*0.28, ch*0.45)"})
    box = b._roi_box_for("hp_bar", region_of_frame(_image()))
    assert box is not None and len(box) == 4
    x, y, w, h = box
    assert 0 < x < 1280 and 0 < y < 720 and w > 0 and h > 0


@pytest.mark.unit
def test_roi__expression_scales_with_the_frame_size() -> None:
    """ROI 是**相对**画布的，换分辨率不该拿到同一个像素框。"""
    b = _builder()
    b.set_roi({"m": "rect(cw*0.5, ch*0.5, cw*0.25, ch*0.25)"})
    small = b._roi_box_for("m", region_of_frame(_image(h=720, w=1280)))
    big = b._roi_box_for("m", region_of_frame(_image(h=1440, w=2560)))
    assert small != big
    assert big[0] == pytest.approx(small[0] * 2, abs=2)


@pytest.mark.unit
def test_roi__invalid_expression_fails_loudly() -> None:
    """表达式本该加载期校验；运行期炸说明配置有问题，不能静默忽略。"""
    b = _builder()
    b.set_roi({"bad": "this is not python ((("})
    with pytest.raises(ValueError):
        b._roi_box_for("bad", region_of_frame(_image()))


@pytest.mark.unit
def test_roi__set_roi_copies_its_input() -> None:
    """配置对象被改不应回灌到 builder。"""
    exprs = {"m": "rect(0, 0, cw, ch)"}
    b = _builder()
    b.set_roi(exprs)
    exprs["m"] = "rect(0, 0, 1, 1)"
    assert b._roi_exprs["m"] == "rect(0, 0, cw, ch)"


@pytest.mark.unit
def test_roi__set_templates_copies_its_input() -> None:
    specs = {"hp_bar": object()}
    b = _builder()
    b.set_templates(specs)
    specs.clear()
    assert b._template_spec("hp_bar") is not None
    assert b._template_spec("never_configured") is None


# ==================================================================
# 构造期不产生噪声
# ==================================================================

@pytest.mark.unit
def test_builder__constructing_with_no_models_emits_nothing() -> None:
    """离线构造不该吐降级事件 —— 否则真机启动会被误判成故障。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = _builder()
    assert b.drain_events() == []
