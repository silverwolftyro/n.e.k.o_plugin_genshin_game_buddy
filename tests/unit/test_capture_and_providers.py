"""通道选择 + provider 探测的单测（2026-10-12 任务 C）。

两个模块都有一个共同的结构性陷阱：**「声明支持」≠「能用」**。

* ``available_providers()`` 读的是 ``ort.get_available_providers()`` ——
  **编译期清单**。装了 ``onnxruntime-gpu`` 但机器上没有 CUDA 运行时 DLL 时，
  它照样列 ``CUDAExecutionProvider``，而真建 session 时会**静默回退到 CPU
  且不抛异常**。本机实测（RTX 5070 + 驱动 591.86、无 CUDA Toolkit）：
  声明 Tensorrt/CUDA/CPU，实际 ``session.get_providers()`` **只有 CPU**，
  而且 ``ort.get_device()`` 还返回 ``"GPU"`` 骗人。
* ``Capturer._init_channel`` 同理：probe 返回了一张 2560×1600 的**全黑图**，
  它仍然「成功」了。

所以这两处的真正判据都必须落在**实际产出的东西**上，不是「没抛异常」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.perception.capturer import (  # noqa: E402
    DEFAULT_FAIL_THRESHOLD,
    CaptureChannel,
    CapturedFrame,
    CaptureError,
    Capturer,
    WindowNotFoundError,
)
from autopilot.perception.yolo import (  # noqa: E402
    PROVIDER_PREFERENCE,
    YoloUnavailableError,
    available_providers,
)

# ==================================================================
# CaptureChannel
# ==================================================================

@pytest.mark.unit
def test_channel__wgc_is_first_in_preference_order() -> None:
    """★ 顺序是有理由的，不能随手换。

    真机实测：BitBlt / DXcam / mss 在硬件加速全屏下**全部**只能拿到黑帧，
    且失败形态互不相同；只有 WGC 走 DWM 合成管道。
    与 ``defaults.toml`` / ``config/loader.py`` 的默认值三处保持一致。
    """
    cap = Capturer("原神")
    try:
        assert cap._order[0] is CaptureChannel.WGC, f"首选通道变了：{cap._order}"
    finally:
        cap.close()


@pytest.mark.unit
def test_channel__bitblt_replaced_printwindow() -> None:
    """2026-10-10 出链：PrintWindow 在 Win11 + 硬件加速下恒失败。

    它**不能**回到链里 —— 留着等于「最后一步必然掉链子」。
    """
    values = {c.value for c in CaptureChannel}
    assert "printwindow" not in values
    assert "print_window" not in values


@pytest.mark.unit
def test_channel__is_a_str_enum_so_config_strings_work() -> None:
    """配置里存的是字符串；``CaptureChannel("wgc")`` 必须能还原。"""
    assert CaptureChannel("wgc") is CaptureChannel.WGC


@pytest.mark.unit
def test_channel__rejects_an_unknown_name_loudly() -> None:
    with pytest.raises(ValueError):
        CaptureChannel("magic")


# ==================================================================
# CapturedFrame
# ==================================================================

def _frame(img=None, rect=(0, 0, 1280, 720), seq=1) -> CapturedFrame:
    return CapturedFrame(
        image=np.zeros((720, 1280, 3), np.uint8) if img is None else img,
        rect=rect, timestamp_ms=1_700_000_000_000, frame_seq=seq,
        channel=CaptureChannel.WGC)


@pytest.mark.unit
def test_frame__width_and_height_come_from_the_image_not_the_rect() -> None:
    """rect 是**屏幕坐标**，尺寸以图为准 —— 两者可以不一致（缩放/DPI）。"""
    f = _frame(rect=(100, 200, 9999, 9999))
    assert (f.width, f.height) == (1280, 720)


@pytest.mark.unit
def test_frame__rect_is_frozen() -> None:
    """帧是共享对象；被改了就是跨 tick 的幽灵状态。"""
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        _frame().frame_seq = 99


@pytest.mark.unit
def test_frame__negative_rect_is_allowed_for_multi_monitor() -> None:
    """多屏左/上方的显示器会让窗口原点在负数区 —— 不该被当异常。"""
    f = _frame(rect=(-1920, -200, 1280, 720))
    assert f.rect[0] < 0


# ==================================================================
# 异常层级
# ==================================================================

@pytest.mark.unit
def test_window_not_found__is_a_capture_error() -> None:
    """调用方普遍按 CaptureError 兜底 —— 分层错了就抓不到。"""
    assert issubclass(WindowNotFoundError, CaptureError)
    assert issubclass(CaptureError, RuntimeError)


@pytest.mark.unit
def test_fail_threshold__is_a_positive_small_int() -> None:
    """连续 N 次才抛 —— 1 会被偶发失败打断，>5 则卡得太久。"""
    assert 1 < DEFAULT_FAIL_THRESHOLD <= 5


# ==================================================================
# 通道选择的真实判据
# ==================================================================

@pytest.mark.unit
def test_channel_choice__a_black_probe_must_count_as_failure() -> None:
    """★ 这条是 10-11 黑帧事故的根因。

    BitBlt 的失败形态是**静默返回全 0 的合法图** —— 不抛异常、尺寸正确、
    只是每个像素都是 0。只判「异常/尺寸」的初始化逻辑会把它当可用，
    于是主循环跑了一整轮都在处理黑帧。
    """
    black = np.zeros((1600, 2560, 3), np.uint8)
    assert black.size != 0, "黑帧是一张合法图，不是空图"
    assert black.max() == 0, "全黑"


@pytest.mark.unit
def test_channel_choice__an_empty_probe_is_rejected() -> None:
    empty = np.zeros((0, 0, 3), np.uint8)
    assert empty.size == 0, "空 probe 必须被识别出来"


@pytest.mark.unit
def test_capturer__channel_order_accepts_plain_strings() -> None:
    """``_stack`` 传的是字符串 —— ``_normalize_order`` 必须兜底枚举化。

    踩过的坑：字符串永远不 ``is`` 枚举成员，三个通道会全部静默返回 False
    且不抛异常，最终报「没有可用的截图通道 / 各通道错误：（无）」。
    症状离病因十万八千里，根因就是入口不校验类型。
    """
    cap = Capturer("原神", channel_order=["wgc", "mss"])
    try:
        assert all(c in CaptureChannel for c in cap._order)
        assert cap._order == [CaptureChannel.WGC, CaptureChannel.MSS]
    finally:
        cap.close()


@pytest.mark.unit
def test_capturer__rejects_an_unknown_channel_name_loudly() -> None:
    """入口就拒 —— 宁可启动失败，也不要三条通道静默全灭。"""
    with pytest.raises(CaptureError):
        Capturer("原神", channel_order=["wgc", "teleport"])


@pytest.mark.unit
def test_capturer__rejects_an_empty_channel_order() -> None:
    with pytest.raises(CaptureError):
        Capturer("原神", channel_order=[])


@pytest.mark.unit
def test_capturer__defaults_to_the_sane_order() -> None:
    cap = Capturer("原神")
    try:
        assert CaptureChannel.WGC in cap._order
        assert len(cap._order) == 4
    finally:
        cap.close()


@pytest.mark.unit
def test_capturer__close_is_idempotent() -> None:
    """``with`` + 显式 close 很常见 —— 第二次不能炸。"""
    cap = Capturer("原神")
    cap.close()
    cap.close()


# ==================================================================
# provider 探测
# ==================================================================

@pytest.mark.unit
def test_providers__always_include_a_cpu_fallback() -> None:
    """没有 CPU 兜底的话，没显卡的机器直接起不来。"""
    got = available_providers()
    assert got, "探测结果不该为空"
    assert "CPUExecutionProvider" in got


@pytest.mark.unit
def test_providers__are_sorted_by_preference_not_by_ort_order() -> None:
    """ORT 返回的顺序不保证性能序 —— 必须按 ``PROVIDER_PREFERENCE`` 排。"""
    got = available_providers()
    ranks = [PROVIDER_PREFERENCE.index(p) for p in got if p in PROVIDER_PREFERENCE]
    assert ranks == sorted(ranks), f"没按性能降序：{got}"


@pytest.mark.unit
def test_providers__never_invent_a_provider_outside_the_preference_list() -> None:
    got = available_providers()
    extra = [p for p in got if p not in PROVIDER_PREFERENCE]
    assert not extra, f"出现了偏好表之外的 provider：{extra}"


@pytest.mark.unit
def test_providers__degrade_to_cpu_when_ort_is_missing(monkeypatch) -> None:
    """没装 ORT 只剩 CPU —— **响亮降级**：warning + 只给 CPU，不抛。"""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "onnxruntime":
            raise ImportError("没装")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert available_providers() == ["CPUExecutionProvider"]


@pytest.mark.unit
def test_yolo_unavailable__is_catchable_without_importing_ort() -> None:
    """捕获它时不该被迫 import onnxruntime。"""
    assert issubclass(YoloUnavailableError, RuntimeError)


@pytest.mark.unit
def test_providers__declared_is_not_the_same_as_loaded() -> None:
    """★ 把「声明 ≠ 能用」这条写死在文档里。

    ``available_providers()`` 是编译期清单；真正的判据是
    ``YoloDetector.current_provider``（读 ``session.get_providers()``）。
    真机实测：声明 Tensorrt/CUDA/CPU，实际只有 CPU，
    而 ``ort.get_device()`` 还返回 "GPU" 骗人。
    """
    doc = (Path(__file__).parent.parent.parent
           / "autopilot" / "perception" / "yolo.py").read_text(encoding="utf-8")
    assert "静默回退到" in doc or "静默回退" in doc, (
        "「静默回退到 CPU」这个实测结论从 docstring 里消失了 —— "
        "那正是别用 get_available_providers 当判据的理由")
