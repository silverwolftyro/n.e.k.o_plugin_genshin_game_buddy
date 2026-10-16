"""YOLO 推理（ONNX + onnxruntime）。

🔴 **许可证硬约束**：
    ultralytics 是 **AGPL-3.0**，以库形式链接会传染 N.E.K.O.（Apache-2.0）。
    **本模块禁止 import ultralytics。** 只用 onnxruntime（MIT）加载导出的
    ONNX 模型。训练可以用 ultralytics（tools/train_yolo.py，工具不分发），
    但产出必须是 ONNX。

对应架构文档 6.2。前后处理与 ultralytics 保持一致，便于用训练脚本导出的
模型直接跑通。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: 推理超时预算（毫秒）。超过则丢弃本帧结果 + 事件，**不阻塞主循环**。
DEFAULT_TIMEOUT_MS = 500.0

#: 算「慢」的 provider。请求了它们之外的却没加载上，就是发生了静默回退。
_CPU_FALLBACKS: tuple[str, ...] = ("CPUExecutionProvider",
                                    "AzureExecutionProvider")


class YoloUnavailableError(RuntimeError):
    """YOLO 不可用。**启动期就该炸** —— 静默失败的识别器会让上层拿到空画面。"""


#: 候选 provider，**按性能降序**。第一个真正能建起 session 的就用它。
PROVIDER_PREFERENCE: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "AzureExecutionProvider",
    "CPUExecutionProvider",
)


def available_providers() -> list[str]:
    """探测**被 ORT 声明支持**的 provider，按性能降序。

    ⚠️ **这个结果只是「候选」，不代表能用。**
       ``ort.get_available_providers()`` 是静态编译期的清单：装了
       ``onnxruntime-gpu`` 但机器上没有 CUDA 13.x 运行时 DLL 时，它照样
       列 ``CUDAExecutionProvider``，而真正建 session 时会**静默回退到
       CPU**（不抛异常）。

       实测（本机 RTX 5070 + 驱动 591.86，但无 CUDA Toolkit）：
       ``get_available_providers()`` -> Tensorrt/CUDA/CPU，
       而 ``session.get_providers()`` -> **只有 CPU**，
       且 ``ort.get_device()`` 还返回 ``"GPU"`` 骗人。

    所以**真正的判据是 :meth:`YoloDetector.current_provider`** ——
    它读的是 ``session.get_providers()``（实际加载了什么），不是声明。
    """
    try:
        import onnxruntime as ort  # noqa: PLC0415
        have = set(ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001 - 没装 ORT 就只剩 CPU
        logger.warning("无法探测 onnxruntime provider: %s", exc)
        return ["CPUExecutionProvider"]
    ordered = [p for p in PROVIDER_PREFERENCE if p in have]
    return ordered or ["CPUExecutionProvider"]


@dataclass(frozen=True)
class Detection:
    """一个检测框。``box`` 是 (x, y, w, h)，在传入帧的坐标系里。"""

    label: str
    box: tuple[int, int, int, int]
    conf: float

    @property
    def center(self) -> tuple[int, int]:
        return (self.box[0] + self.box[2] // 2, self.box[1] + self.box[3] // 2)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "box": list(self.box), "conf": round(self.conf, 4)}


class YoloDetector:
    """YOLO11n ONNX 推理器。

    用法::

        det = YoloDetector("models/yolo11n.onnx", ["main_menu", "combat", ...])
        det.warmup()
        hits = det.detect(frame, conf=0.35, classes=["combat"])
    """

    def __init__(
        self,
        onnx_path: str | Path,
        class_names: list[str],
        *,
        providers: list[str] | None = None,
        default_imgsz: int = 640,
    ) -> None:
        self._path = Path(onnx_path)
        self._class_names = list(class_names)
        self._imgsz = int(default_imgsz)
        self._closed = False
        self._input_name: str = ""
        self._output_names: list[str] = []
        self._session = None
        self._is_dynamic = False
        self._last_ms = 0.0
        self._providers: list[str] = []
        self._requested_providers: list[str] = []
        self._fallback_reason = ""

        if not self._path.exists():
            raise YoloUnavailableError(
                f"ONNX 模型不存在: {self._path}\n"
                f"请先运行 tools/train_yolo.py 训练并导出，或 tools/convert_to_onnx.py 导出。"
            )
        if not self._class_names:
            raise YoloUnavailableError("class_names 不能为空")

        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as exc:
            raise YoloUnavailableError(
                "未安装 onnxruntime。运行: pip install onnxruntime-gpu"
            ) from exc

        avail = ort.get_available_providers()
        want = list(providers) if providers else available_providers()
        use = [p for p in want if p in avail]
        if not use:
            use = ["CPUExecutionProvider"]

        self._requested_providers = list(use)
        session = self._make_session(use)

        # ★ 关键校验：ORT **不保证**请求的 provider 真的加载成功。
        #   缺 CUDA 运行时 DLL 时它会静默回退 CPU 且不抛异常。
        #   所以必须读 session.get_providers()（实际加载了什么）而不是
        #   相信请求列表。
        actual = list(session.get_providers())
        fast_wanted = [p for p in use if p not in _CPU_FALLBACKS]
        missing = [p for p in fast_wanted if p not in actual]
        if missing:
            logger.warning(
                "provider %s 声明可用但实际未加载（缺运行时依赖？），"
                "已回退到 %s", missing, actual)
            self._fallback_reason = f"{missing} 不可用，已回退 {actual}"
            session = self._make_session([_CPU_FALLBACKS[0]])
            actual = list(session.get_providers())
        else:
            self._fallback_reason = ""
        self._session = session
        self._providers = actual
        if actual and actual[0] in _CPU_FALLBACKS:
            logger.warning(
                "YOLO 走 CPU 推理（会明显变慢）。要上 GPU 请装匹配的 "
                "CUDA/cuDNN 运行时 —— 装了 onnxruntime-gpu 本身**不够**，"
                "ORT 仍会静默回退。见 perception/yolo.py.available_providers 文档。")

        inputs = self._session.get_inputs()
        if not inputs:
            raise YoloUnavailableError("ONNX 模型没有输入节点")
        self._input_name = inputs[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        shape = inputs[0].shape
        # shape 可能是 [1,3,640,640] 或动态的 [1,3,'height','width']
        self._is_dynamic = any(
            (not isinstance(d, int)) or d <= 0 for d in (shape[2:] if len(shape) >= 4 else [])
        )
        if not self._is_dynamic and len(shape) == 4:
            if int(shape[2]) != int(shape[3]):
                raise YoloUnavailableError(
                    f"模型输入不是正方形: {shape}。本实现按 imgsz×imgsz 处理。"
                )
            self._imgsz = int(shape[2])

        logger.info(
            "YOLO loaded: %s providers=%s imgsz=%d dynamic=%s classes=%d",
            self._path.name, self._providers, self._imgsz, self._is_dynamic,
            len(self._class_names),
        )

    def _make_session(self, providers: list[str]):
        """建 session。失败抛 :class:`YoloUnavailableError`。"""
        import onnxruntime as ort  # noqa: PLC0415
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0  # 0 = 自动
        try:
            return ort.InferenceSession(
                str(self._path), sess_options=so, providers=providers
            )
        except Exception as exc:  # noqa: BLE001
            raise YoloUnavailableError(
                f"ONNX 会话创建失败 (providers={providers}): {exc}") from exc

    # -- 属性 ------------------------------------------------------------

    @property
    def providers(self) -> list[str]:
        """**实际加载**的 provider 列表。"""
        return list(self._providers)

    def current_provider(self) -> str:
        """当前真正在用的 provider（``session.get_providers()[0]``）。

        ★ 这是**唯一可信**的判据。``ort.get_available_providers()`` 只反映
          编译期清单，缺运行时 DLL 时它会撒谎（见
          :func:`available_providers` 的文档）。
        """
        return self._providers[0] if self._providers else "unknown"

    def fallback_reason(self) -> str:
        """为什么没用上更快的 provider。空串表示没有回退。"""
        return self._fallback_reason

    @property
    def is_gpu(self) -> bool:
        """当前是否真的在 GPU 上跑。读的是**实际加载**的 provider。"""
        return not (not self._providers
                    or self._providers[0] in _CPU_FALLBACKS)

    @property
    def last_inference_ms(self) -> float:
        return self._last_ms

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def close(self) -> None:
        self._session = None
        self._closed = True

    def __enter__(self) -> "YoloDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- 预热 ------------------------------------------------------------

    def warmup(self) -> None:
        """跑一张假图。避免第一帧把 CUDA 初始化 + 内存分配的耗时算进 tick。"""
        try:
            fake = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            self.detect(fake, conf=0.99)
            logger.info("YOLO warmup ok (%.0f ms)", self._last_ms)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO warmup 失败: %s", exc)

    # -- 推理 ------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int | None = None,
        classes: list[str] | None = None,
        timeout_ms: float = DEFAULT_TIMEOUT_MS,
    ) -> list[Detection]:
        """检测。

        ``classes`` 传白名单时只跑这些类别 —— 这是**场景路由的主要优化点**
        （已知在背包场景就只跑 enemy + pickup，不跑全部 13 类）。

        超时或异常时返回**空列表**并记日志，**不抛异常打断循环**。
        """
        if self._closed or self._session is None:
            raise YoloUnavailableError("YoloDetector 已关闭")
        if frame is None or frame.size == 0:
            return []

        size = int(imgsz or self._imgsz)
        t0 = time.perf_counter()

        try:
            blob, ratio, (dx, dy) = _letterbox(frame, size)
            outs = self._session.run(
                self._output_names, {self._input_name: blob}
            )
            dets = _postprocess(
                outs[0], ratio, (dx, dy), frame.shape[:2], self._class_names, conf, iou
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - t0) * 1000
            self._last_ms = elapsed
            logger.error("YOLO 推理失败: %s (%.0f ms)", exc, elapsed)
            return []

        elapsed = (time.perf_counter() - t0) * 1000
        self._last_ms = elapsed
        if elapsed > timeout_ms:
            logger.warning(
                "YOLO 推理 %.0f ms 超过预算 %.0f ms，本帧结果丢弃", elapsed, timeout_ms
            )
            return []

        if classes:
            allowed = set(classes)
            dets = [d for d in dets if d.label in allowed]
        return dets


# --------------------------------------------------------------------------
# 前后处理
# --------------------------------------------------------------------------


def _letterbox(
    img: np.ndarray, new_shape: int, color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比缩放 + 灰边填充到 new_shape×new_shape。

    返回 (blob NCHW float32, ratio, (dx, dy))。dx/dy 用于把检测框还原回原图。
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = new_shape - nh, new_shape - nw
    top, left = dh // 2, dw // 2
    img = cv2.copyMakeBorder(img, top, dh - top, left, dw - left,
                             cv2.BORDER_CONSTANT, value=color)
    blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    blob = np.ascontiguousarray(blob[None], dtype=np.float32) / 255.0
    return blob, r, (left, top)


def _postprocess(
    out: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    class_names: list[str],
    conf: float,
    iou: float,
) -> list[Detection]:
    """解析 YOLO 输出。

    兼容两种导出形态：
      A. [1, 4+nc, N]        —— ultralytics 默认（end2end 未开）
      B. [1, N, 4+nc]        —— 部分导出器转置过
    另外也支持 [1, N, 6] 的 end2end 形态（x1,y1,x2,y2,score,cls）。
    """
    pred = np.asarray(out)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        return []

    n_attrs = 4 + len(class_names)
    if pred.shape[0] == n_attrs and pred.shape[1] != n_attrs:
        pred = pred.T  # [N, 4+nc]
    elif pred.shape[1] == 6 and pred.shape[0] != 6:
        return _postprocess_end2end(pred, orig_shape, class_names, conf)

    if pred.shape[1] < n_attrs:
        return []

    oh, ow = orig_shape
    dx, dy = pad
    results: list[Detection] = []
    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []

    for row in pred:
        cls_scores = row[4:4 + len(class_names)]
        cid = int(np.argmax(cls_scores))
        score = float(cls_scores[cid])
        if score < conf:
            continue
        cx, cy, bw, bh = (float(v) for v in row[:4])
        # 还原到原图
        x = (cx - bw / 2 - dx) / ratio
        y = (cy - bh / 2 - dy) / ratio
        w = bw / ratio
        h = bh / ratio
        x1 = int(max(0, min(ow - 1, round(x))))
        y1 = int(max(0, min(oh - 1, round(y))))
        x2 = int(max(0, min(ow - 1, round(x + w))))
        y2 = int(max(0, min(oh - 1, round(y + h))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        label = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        results.append(Detection(label, (x1, y1, x2 - x1, y2 - y1), score))
        boxes.append((x1, y1, x2 - x1, y2 - y1))
        scores.append(score)

    return _nms(results, boxes, scores, iou)


def _postprocess_end2end(
    pred: np.ndarray, orig_shape: tuple[int, int], class_names: list[str], conf: float
) -> list[Detection]:
    """end2end 导出（[x1,y1,x2,y2,score,cls]）—— 模型内已做 NMS。"""
    oh, ow = orig_shape
    out: list[Detection] = []
    for row in pred:
        score = float(row[4])
        if score < conf:
            continue
        cid = int(row[5])
        x1 = int(max(0, min(ow - 1, round(float(row[0])))))
        y1 = int(max(0, min(oh - 1, round(float(row[1])))))
        x2 = int(max(0, min(ow - 1, round(float(row[2])))))
        y2 = int(max(0, min(oh - 1, round(float(row[3])))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        label = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        out.append(Detection(label, (x1, y1, x2 - x1, y2 - y1), score))
    return out


def _nms(
    dets: list[Detection], boxes: list[tuple], scores: list[float], iou_thr: float
) -> list[Detection]:
    """标准 NMS。ultralytics 的 class-aware 版本：同类之间才抑制。

    ⚠️ ``boxes`` 进来是 ``list[tuple]``，但下面要用 ``boxes[rest, 0]`` 这种
       numpy 风格批量取列。**在 Python list 上这么索引会抛**
       ``TypeError: list indices must be integers or slices, not tuple``。
       入口必须先转 ndarray —— 早先忘了转，导致**任何 ≥2 个检测框的帧
       整个 detect() 静默返回空**（异常被 :meth:`YoloDetector.detect`
       吞掉），而 0/1 个框时 :func:`_nms` 提前 return 所以看不出来。
    """
    if len(dets) <= 1:
        return dets
    # list -> ndarray，(N, 4)；空输入时 reshape 保证二维
    arr = np.asarray(boxes, dtype=np.float32).reshape(len(dets), 4)
    order = np.argsort(np.asarray(scores))[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(arr[i][0], arr[rest, 0])
        yy1 = np.maximum(arr[i][1], arr[rest, 1])
        xx2 = np.minimum(arr[i][0] + arr[i][2], arr[rest, 0] + arr[rest, 2])
        yy2 = np.minimum(arr[i][1] + arr[i][3], arr[rest, 1] + arr[rest, 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = arr[i][2] * arr[i][3]
        area_r = arr[rest, 2] * arr[rest, 3]
        iou = inter / np.maximum(1e-9, area_i + area_r - inter)
        same_cls = np.array([dets[j].label == dets[i].label for j in rest])
        order = rest[~((iou > iou_thr) & same_cls)]
    return [dets[i] for i in sorted(keep, key=lambda k: -scores[k])]


__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "PROVIDER_PREFERENCE",
    "Detection",
    "YoloDetector",
    "YoloUnavailableError",
    "available_providers",
]
