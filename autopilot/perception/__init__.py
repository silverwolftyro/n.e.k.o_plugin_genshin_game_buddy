"""感知层：屏幕帧 → 结构化状态。

对应架构文档第 4.1 / 6.1-6.5。

各模块职责：
    capturer        三通道截图 + 可重入锁
    screen_region   链式裁剪，坐标可回推
    roi_expr        ROI 表达式求值（ast 白名单，不手写 tokenizer）
    template        模板匹配 + NMS + 抗抖稳定计数
    yolo            ONNX 推理（**禁止 import ultralytics**，AGPL 避雷）
    ocr             文字识别（复用 2.0 的 WinRT 引擎）
    builder         融合 → StatePacket
    debug_overlay   把真实识别结果画在屏幕上
"""

from __future__ import annotations

from .builder import StateBuilder
from .capturer import (
    CaptureChannel,
    CaptureError,
    CapturedFrame,
    Capturer,
    WindowNotFoundError,
)
from .ocr import OcrEngine, OcrResult, TriState
from .roi_expr import CompiledRoi, RoiExprError, RoiExprEvaluator, eval_roi_expression
from .screen_region import ImageRegion, region_of_frame
from .template import (
    StableValueAccumulator,
    TemplateMatch,
    TemplateMatcher,
    TemplateSpec,
)
from .yolo import Detection, YoloDetector, YoloUnavailableError

__all__ = [
    "CaptureChannel",
    "CaptureError",
    "CapturedFrame",
    "Capturer",
    "CompiledRoi",
    "Detection",
    "ImageRegion",
    "OcrEngine",
    "OcrResult",
    "RoiExprError",
    "RoiExprEvaluator",
    "StableValueAccumulator",
    "StateBuilder",
    "TemplateMatch",
    "TemplateMatcher",
    "TemplateSpec",
    "TriState",
    "WindowNotFoundError",
    "YoloDetector",
    "YoloUnavailableError",
    "eval_roi_expression",
    "region_of_frame",
]
