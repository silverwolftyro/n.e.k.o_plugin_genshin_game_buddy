"""模板匹配 + NMS + 抗抖稳定计数。

对应架构文档 6.3。

NMS 的实现抄自 BetterGI 的思路（实测得到的两个细节）：
  1. IoU 阈值 0.5，两个同尺寸矩形：
         intersection / (2 * area - intersection) >= 0.5
  2. 抑制窗口只遍历**局部邻域**，不全图扫描：
         maxDelta = floor(tplW * (1-0.5)/(1+0.5)) + 1 ≈ tplW/3 + 1

抗抖的 ``StableValueAccumulator`` 抄自 BetterGI 的同名原语，但**必须带诊断**：
不只回答"稳了没"，还要能回答"最高连续命中几次、阈值要几次"（原则 9）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .screen_region import ImageRegion

logger = logging.getLogger(__name__)

#: 模板里这个颜色不参与匹配（绿色遮罩，与 BetterGI 一致）
MASK_COLOR_BGR = (0, 255, 0)

DEFAULT_THRESHOLD = 0.80
DEFAULT_NMS_IOU = 0.5


@dataclass(frozen=True)
class TemplateSpec:
    """一个模板的完整定义。"""

    name: str
    path: Path
    threshold: float = DEFAULT_THRESHOLD
    use_3_channels: bool = False
    use_green_mask: bool = False
    use_binary: bool = False
    binary_threshold: int = 128
    # 匹配模式。默认 TM_CCOEFF —— 配合"响应图统一归一到 [0,1]"使用。
    #
    # 为什么不用 TM_CCOEFF_NORMED（BetterGI 的默认）：
    #   Normed 版对**每个位置单独**按局部窗口方差归一。游戏截图有大片平坦区域，
    #   那些位置的分母趋零、噪声被放大，实测会把峰值挤到错误位置
    #   （真值处 0.38，噪声处反而更高）。CCOEFF 用全局均值，
    #   再对整张响应图做一次 MinMax 归一，既定位准又让分数落在 [0,1]，
    #   阈值语义对所有模式统一（越大越严格）。
    #   纯色模板仍匹配不到（方差为零）—— 那是裁图问题，load 时会告警。
    match_mode: int = 3  # cv2.TM_CCOEFF
    # 抗抖：连续多少帧同一位置才算命中。0 = 不做稳定判定
    stable_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "threshold": self.threshold,
            "use_3_channels": self.use_3_channels,
            "use_green_mask": self.use_green_mask,
            "use_binary": self.use_binary,
            "binary_threshold": self.binary_threshold,
            "stable_count": self.stable_count,
        }


@dataclass(frozen=True)
class TemplateMatch:
    """匹配结果。``box`` 在传入的 region 坐标系里。"""

    name: str
    box: tuple[int, int, int, int]
    score: float
    point: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "box": list(self.box), "score": round(self.score, 4)}


class StableValueAccumulator:
    """抗抖原语：同一值连续出现 N 次才返回 True。

    解决的不是"怎么抖"，是"**抖了之后怎么告诉用户**"：识别失败时能报
    "最高连续命中 2 次，阈值要 3 次"，而不是干巴巴一句"识别失败"。

    用法::

        acc = StableValueAccumulator(required_count=3)
        for frame in frames:
            if acc.update(frame.detected_here):
                ...   # 稳了
        if not acc.settled:
            log(acc.max_consecutive_count)   # 差多少
    """

    def __init__(self, required_count: int = 3) -> None:
        self._required = max(1, int(required_count))
        self._current: Any = None
        self._consecutive = 0
        self._max_consecutive = 0
        self._last_score = 0.0
        self._total_updates = 0

    @property
    def required_count(self) -> int:
        return self._required

    @property
    def consecutive(self) -> int:
        return self._consecutive

    @property
    def max_consecutive_count(self) -> int:
        """历史最长连续命中次数 —— 失败诊断的核心数据。"""
        return self._max_consecutive

    @property
    def last_score(self) -> float:
        return self._last_score

    @property
    def settled(self) -> bool:
        """当前是否已达阈值。"""
        return self._consecutive >= self._required

    @property
    def ever_settled(self) -> bool:
        return self._max_consecutive >= self._required

    def update(self, value: Any, score: float = 0.0) -> bool:
        """喂一个观测值。相等则累加，不等则重置。返回是否达标。"""
        self._total_updates += 1
        self._last_score = score
        if value == self._current:
            self._consecutive += 1
        else:
            self._current = value
            self._consecutive = 1
        self._max_consecutive = max(self._max_consecutive, self._consecutive)
        return self.settled

    def reset(self) -> None:
        self._current = None
        self._consecutive = 0

    def clear_stats(self) -> None:
        self.reset()
        self._max_consecutive = 0
        self._total_updates = 0


class TemplateMatcher:
    """模板匹配器。模板加载后缓存，匹配时复用。"""

    def __init__(
        self,
        *,
        nms_iou: float = DEFAULT_NMS_IOU,
        min_search_margin: int = 2,
    ) -> None:
        self._nms_iou = float(nms_iou)
        self._min_margin = int(min_search_margin)
        self._cache: dict[tuple, np.ndarray] = {}
        self._mask_cache: dict[tuple, np.ndarray] = {}
        #: name -> StableValueAccumulator
        self._accumulators: dict[str, StableValueAccumulator] = {}

    # -- 模板加载 --------------------------------------------------------

    def load_template(self, spec: TemplateSpec) -> np.ndarray:
        """加载并缓存模板（含遮罩/二值化预处理）。"""
        key = (str(spec.path), spec.use_3_channels, spec.use_binary,
               spec.binary_threshold, spec.use_green_mask)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if not spec.path.exists():
            # 响亮失败：模板缺失是配置错误，不该静默
            raise FileNotFoundError(f"模板文件不存在: {spec.name} -> {spec.path}")

        mode = cv2.IMREAD_COLOR if spec.use_3_channels else cv2.IMREAD_GRAYSCALE
        # ⚠️ 不用 cv2.imread：它在 Windows 上按 ANSI 码页解路径，
        #    而模板图是**用户命名**的 —— 起个中文名就静默读不到，
        #    症状是「模板加载失败」却指向一个看起来完全正常的配置。
        from .image_io import imread_unicode

        img = imread_unicode(spec.path, mode)
        if img is None:
            raise FileNotFoundError(f"模板无法解码: {spec.name} -> {spec.path}")

        if spec.use_binary:
            _, img = cv2.threshold(
                img, spec.binary_threshold, 255,
                cv2.THRESH_BINARY if not spec.use_3_channels else cv2.THRESH_BINARY,
            )
        if not spec.use_3_channels:
            img = _to_gray3(img)  # 统一成 3 通道以便复用 mask 逻辑

        self._cache[key] = img
        if spec.use_green_mask:
            self._mask_cache[key] = _green_mask(img)

        # 退化检测：纯色模板的方差趋零，CCoeffNormed 会算出 0/0 而永不命中。
        # 这里提前警告，而不是让用户对着"明明在那儿却不认"的框调试半天。
        if spec.match_mode == cv2.TM_CCOEFF_NORMED:
            var = float(np.asarray(img, dtype=np.float64).var())
            if var < 1.0:
                logger.warning(
                    "模板 %s 几乎是纯色（方差 %.3f），TM_CCOEFF_NORMED 匹配不到。"
                    "请重新裁一块有结构的区域，或在配置里把 match_mode 设为 "
                    "cv2.TM_CCORR_NORMED(4)",
                    spec.name, var,
                )
        return img

    # -- 单点匹配 --------------------------------------------------------

    def find(
        self,
        region: ImageRegion | np.ndarray,
        spec: TemplateSpec,
        *,
        search_box: tuple[int, int, int, int] | None = None,
        use_stability: bool = True,
    ) -> TemplateMatch | None:
        """在 region（或其 search_box 子范围）内找最佳匹配。

        返回 None 是**合法结果**（没找到），不是错误。
        搜索区小于模板时**返回 None 而不是抛异常** —— OpenCV 会直接崩。
        """
        image = region.image if isinstance(region, ImageRegion) else region
        templ = self.load_template(spec)
        th, tw = templ.shape[:2]
        ih, iw = image.shape[:2]

        sx, sy, sw, sh = (0, 0, iw, ih)
        if search_box is not None:
            x, y, w, h = search_box
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
            if x2 - x1 < tw + self._min_margin or y2 - y1 < th + self._min_margin:
                self._accumulators.get(spec.name, StableValueAccumulator(0)).reset()
                return None
            sx, sy, sw, sh = x1, y1, x2 - x1, y2 - y1

        if sw < tw or sh < th:
            return None

        haystack = image[sy:sy + sh, sx:sx + sw]
        mask = self._mask_cache.get(
            (str(spec.path), spec.use_3_channels, spec.use_binary,
             spec.binary_threshold, spec.use_green_mask)
        )

        try:
            result = cv2.matchTemplate(haystack, templ, spec.match_mode, mask=mask)
        except cv2.error as exc:
            logger.error("matchTemplate 失败 (%s): %s", spec.name, exc)
            return None

        # 响应图统一归一到 [0,1]（1 = 最像），让阈值语义对所有模式统一。
        # 平方差类先取反成"越大越像"，再一起归一。
        if not _is_bounded_mode(spec.match_mode):
            cv2.normalize(result, result, 0.0, 1.0, cv2.NORM_MINMAX)
        if spec.match_mode in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            result = 1.0 - result

        _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(result)
        raw = float(max_v)
        # 护栏：非有限值直接判未命中；归一化后仍越界说明模板退化
        # （方差趋零），宁可漏检也不要给出离谱的置信度污染状态包。
        if raw != raw or raw in (float("inf"), float("-inf")):
            self._on_low_score(spec, 0.0)
            return None
        score = max(0.0, min(1.0, raw))
        if score < spec.threshold:
            self._on_low_score(spec, score)
            return None

        lx, ly = max_l  # 相对 search_box 的左上角
        box = (sx + int(lx), sy + int(ly), int(tw), int(th))
        point = (box[0] + tw // 2, box[1] + th // 2)

        if not use_stability or spec.stable_count <= 0:
            return TemplateMatch(spec.name, box, score, point)
        return self._apply_stability(spec, box, score)

    # -- 多点匹配 --------------------------------------------------------

    def find_multi(
        self,
        region: ImageRegion | np.ndarray,
        spec: TemplateSpec,
        max_count: int = 8,
    ) -> list[TemplateMatch]:
        """找多个匹配，做 IoU NMS 抑制。"""
        image = region.image if isinstance(region, ImageRegion) else region
        templ = self.load_template(spec)
        th, tw = templ.shape[:2]
        ih, iw = image.shape[:2]
        if iw < tw or ih < th:
            return []

        mask = self._mask_cache.get(
            (str(spec.path), spec.use_3_channels, spec.use_binary,
             spec.binary_threshold, spec.use_green_mask)
        )
        try:
            resp = cv2.matchTemplate(image, templ, spec.match_mode, mask=mask)
        except cv2.error as exc:
            logger.error("matchTemplate(multi) 失败 (%s): %s", spec.name, exc)
            return []

        if spec.match_mode in (cv2.TM_SQDIFF, cv2.TM_CCORR, cv2.TM_CCOEFF):
            cv2.normalize(resp, resp, 0, 1, cv2.NORM_MINMAX)
        if spec.match_mode in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            resp = 1.0 - resp
        # （原先这里有个 lower_better = False，写死且从未被读 ——
        #   上面两行已经把 SQDIFF 取反成「越大越像」，方向已经统一了。）
        thr = spec.threshold

        cand = (resp >= thr).astype(np.uint8)
        out: list[TemplateMatch] = []

        while len(out) < max_count and int(cand.sum()) > 0:
            _mn, mx, _mnl, mxl = cv2.minMaxLoc(resp, mask=cand)
            lx, ly = mxl
            raw = mx
            if raw != raw:  # NaN
                cand[ly, lx] = 0
                continue
            score = max(0.0, min(1.0, float(raw)))
            out.append(
                TemplateMatch(
                    spec.name,
                    (int(lx), int(ly), int(tw), int(th)),
                    float(score),
                    (int(lx) + tw // 2, int(ly) + th // 2),
                )
            )
            cand = _suppress_overlap(cand, lx, ly, tw, th, self._nms_iou)
        return out

    # -- 抗抖 ------------------------------------------------------------

    def _apply_stability(
        self, spec: TemplateSpec, box: tuple[int, ...], score: float
    ) -> TemplateMatch | None:
        acc = self._accumulators.get(spec.name)
        if acc is None or acc.required_count != spec.stable_count:
            acc = StableValueAccumulator(spec.stable_count)
            self._accumulators[spec.name] = acc
        if acc.update(tuple(box), score):
            return TemplateMatch(spec.name, box, score,
                                 (box[0] + box[2] // 2, box[1] + box[3] // 2))
        return None

    def _on_low_score(self, spec: TemplateSpec, score: float) -> None:
        acc = self._accumulators.get(spec.name)
        if acc is not None:
            acc.update(None, score)  # None 视为不匹配，触发连续计数

    def accumulator(self, name: str) -> StableValueAccumulator | None:
        return self._accumulators.get(name)

    def reset_all(self) -> None:
        for acc in self._accumulators.values():
            acc.reset()

    def clear_stats(self) -> None:
        self._accumulators.clear()

    # -- 计时 ------------------------------------------------------------

    @staticmethod
    def timed(fn, *args, **kwargs):
        """跑一次并返回 (结果, 耗时ms)。给 builder 填 resource.template_ms 用。"""
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs), (time.perf_counter() - t0) * 1000.0
        except Exception:
            raise


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _to_gray3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _is_bounded_mode(mode: int) -> bool:
    """该模式的响应图是否天然落在 [0,1]（True = 不需要再归一）。

    · *_NORMED 内部已归一（但 CCOEFF_NORMED 的逐位置归一在平坦画面上不可靠，
      所以默认模式不选它）
    · SQDIFF / CCORR / CCOEFF 无界 —— 需要整图 MinMax 归一
    """
    return mode in (
        cv2.TM_SQDIFF_NORMED,
        cv2.TM_CCORR_NORMED,
        cv2.TM_CCOEFF_NORMED,
    )


def _green_mask(img: np.ndarray) -> np.ndarray:
    """纯绿像素不参与匹配。"""
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return ((b != MASK_COLOR_BGR[0]) | (g != MASK_COLOR_BGR[1]) | (r != MASK_COLOR_BGR[2])).astype(
        np.uint8
    ) * 255


def _suppress_overlap(
    cand: np.ndarray, sx: int, sy: int, tw: int, th: int, iou_thr: float
) -> np.ndarray:
    """抑制与 (sx,sy) 处模板 IoU 达标的邻近候选点。"""
    max_dx = int(tw * (1.0 - iou_thr) / (1.0 + iou_thr)) + 1
    max_dy = int(th * (1.0 - iou_thr) / (1.0 + iou_thr)) + 1
    h, w = cand.shape
    x0, x1 = max(0, sx - max_dx), min(w - 1, sx + max_dx)
    y0, y1 = max(0, sy - max_dy), min(h - 1, sy + max_dy)
    area = float(tw * th)

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if cand[y, x] == 0:
                continue
            ow = tw - abs(sx - x)
            oh = th - abs(sy - y)
            if ow <= 0 or oh <= 0:
                continue
            inter = float(ow * oh)
            union = area * 2.0 - inter
            if inter / union >= iou_thr:
                cand[y, x] = 0
    return cand


__all__ = [
    "DEFAULT_NMS_IOU",
    "DEFAULT_THRESHOLD",
    "MASK_COLOR_BGR",
    "StableValueAccumulator",
    "TemplateMatch",
    "TemplateMatcher",
    "TemplateSpec",
]
