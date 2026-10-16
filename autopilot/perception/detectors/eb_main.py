"""``eb_main``（大招骰子）特征提取。

★ 只出数字，不判断
-----------------
用户观察到的规律是"f00005 蓝色占 94.74% -> 大招似乎满"，
但同一份数据里"不同角色元素不同，不能只看蓝色占比"。
这正是**本轮只做特征、不做分类**的原因：先把可分性量出来，
再决定阈值长什么样。现在写死任何判断都会把标定埋进代码里。

★ 口径说明
----------
所有通道都取自 OpenCV 的 HSV（BGR 图 ``cvtColor(..., COLOR_BGR2HSV)``）：
    H  0~179  （OpenCV 把 0~255 压成 0~179，色相跨度 2 度）
    S  0~255
    V  0~255
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ..roi_loader import Rect, load_rois

logger = logging.getLogger(__name__)

#: eb_main 在 rois.toml 里的键名
ROI_NAME = "eb_main"

#: H 直方图 bin 数。OpenCV 的 H 上限 180（0~179），一一对应。
HUE_BINS = 180

#: 「实色」的饱和度下限。低于这个值基本是灰/白，不是元素本身。
SAT_SOLID = 100
#: 「亮」的明度下限。
VAL_BRIGHT = 200

#: 内芯半径占比（相对 ROI 短边的一半）。取 0.5 表示内芯是外接圆的内切圆。
CORE_RADIUS_RATIO = 0.5


@dataclass(frozen=True)
class EbMainFeatures:
    """``eb_main`` 的一帧特征。全部是**可直接比较的标量**。

    :param dominant_hue: H 直方图峰值所在的 bin 中心（0~179）
    :param hue_concentration: 峰值 bin 的像素占比（0~1）。越大说明
        颜色越"纯"；元素和背景混色时这个值会明显掉下来
    :param mean_sat: 平均饱和度（0~255）
    :param mean_val: 平均明度（0~255）
    :param saturated_ratio: S > 100 的像素占比（0~1）—— 区分"实色元素"
        与"灰色/暗色背景"
    :param bright_ratio: V > 200 的像素占比（0~1）
    :param ring_sat_delta: 外圈平均饱和度 − 内芯平均饱和度。
        正值 = 中心比边缘"更实心"，符合骰子已充满的直觉；
        接近 0 = 整体是均匀色块，看不出充能程度
    :param pixel_count: ROI 有效像素数。用来识别"ROI 根本没截到东西"
    """

    dominant_hue: float
    hue_concentration: float
    mean_sat: float
    mean_val: float
    saturated_ratio: float
    bright_ratio: float
    ring_sat_delta: float
    pixel_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _roi(name: str = ROI_NAME) -> Rect:
    """取 ROI 定义。**没配就响亮失败** ——
    静默返回空 Rect 会让上层拿着 0 像素继续跑，症状是"元素永远不存在"。
    """
    rois = load_rois()
    r = rois.get(name)
    if r is None:
        raise KeyError(
            f"rois.toml 里没有 {name!r}。现有 ROI：{', '.join(sorted(rois))}")
    return r


def _ring_core_mask(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """把 ROI 切成（外圈, 内芯）两个布尔掩码。

    ★ 为什么用**圆形**而不是矩形内切
      骰子是圆的。用矩形切出来的"内芯"会包含四个角的背景，
      饱和度被背景拉低，ring/core 差值就测不出"实心程度"。
    """
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - cx, yy - cy)
    r_max = min(cx, cy)                     # 切到边界为止
    if r_max <= 0:
        full = np.ones((h, w), dtype=bool)
        return full, np.zeros((h, w), dtype=bool)
    r_core = r_max * CORE_RADIUS_RATIO
    core = dist <= r_core
    ring = ~core
    return ring, core


def extract_features(frame: np.ndarray, roi_name: str = ROI_NAME,
                     ) -> EbMainFeatures:
    """从整帧里切出 ``eb_main`` 并算特征。

    :param frame: BGR 图像，``numpy.ndarray``，形状 ``(H, W, 3)``
    :param roi_name: rois.toml 里的键，默认 ``eb_main``
    :raises TypeError: frame 不是图像数组
    :raises ValueError: ROI 在这帧上切不出像素（画面比 ROI 还小）

    ★ 为什么失败要抛而不是返回零值
      全零特征和"真的没有骰子"在数值上**完全一样**。批量跑 300 帧时，
      返回零值会安静地混进 CSV，标定的人根本发现不了。
      抛出来，调用方能逐帧记下是哪一帧出的问题。
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 \
            or frame.shape[2] not in (3, 4):
        raise TypeError(
            f"frame 应为 (H, W, 3) 的 BGR 图像，实际 shape={getattr(frame, 'shape', None)}")

    rect = _roi(roi_name)
    crop = rect.crop(frame[:, :, :3])
    if not isinstance(crop, np.ndarray) or crop.size == 0:
        raise ValueError(
            f"ROI {roi_name!r} 在 {frame.shape[1]}x{frame.shape[0]} 的画面上"
            f"切不出像素（{rect}）")

    h, w = crop.shape[0], crop.shape[1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # H 直方图。★ 注意色相是**环形**的：179 和 0 在 HSV 里是相邻的色，
    #   纯 argmax 会在跨 0 的颜色上把它们判成两种色调。
    #   本轮先出朴素的 argmax（标定时能看出这个坑），不做环绕平滑 ——
    #   那属于"怎么判"的一部分，和分类同批做。
    hist = cv2.calcHist([hsv], [0], None, [HUE_BINS], [0, HUE_BINS]).ravel()
    peak_bin = int(np.argmax(hist))
    total = float(hist.sum()) or 1.0

    ring_mask, core_mask = _ring_core_mask(h, w)
    ring_sat = float(sat[ring_mask].mean()) if ring_mask.any() else 0.0
    core_sat = float(sat[core_mask].mean()) if core_mask.any() else 0.0

    return EbMainFeatures(
        dominant_hue=float(peak_bin),
        hue_concentration=float(hist[peak_bin] / total),
        mean_sat=float(sat.mean()),
        mean_val=float(val.mean()),
        saturated_ratio=float((sat > SAT_SOLID).mean()),
        bright_ratio=float((val > VAL_BRIGHT).mean()),
        ring_sat_delta=ring_sat - core_sat,
        pixel_count=int(h * w),
    )


#: 队伍圆环的合法槽位。rois.toml 里是 ``[rois.eb]`` 的 "1"/"2"/"3"，
#: 展开后是 ``eb_1`` / ``eb_2`` / ``eb_3``（见 roi_loader 的拍平规则）。
SLOT_IDS = (1, 2, 3)


def extract_slot_features(frame: np.ndarray, slot: int) -> EbMainFeatures:
    """队伍 N 号位旁的**元素圆环**特征。

    :param frame: BGR 整帧
    :param slot: 队伍槽位，只能是 1 / 2 / 3
    :raises ValueError: slot 越界（比让 rois.toml 报 KeyError 好定位 ——
        传进来的 4 到底是笔误还是版本变更，这里能直接看出来）

    特征口径和 :func:`extract_features` **完全一致**（同一段 H/S/V 统计、
    同一个圆形 ring/core 掩码），所以 eb_1/2/3 和 eb_main 的数值可以直接
    横向比较。圆环是圆形，用圆形掩码切内芯/外圈比矩形贴切。
    """
    if slot not in SLOT_IDS:
        raise ValueError(
            f"slot 只能是 {SLOT_IDS} 之一，收到 {slot!r}"
            f"（rois.toml 的 [rois.eb] 只配了 1/2/3）")
    return extract_features(frame, roi_name=f"eb_{slot}")


__all__ = [
    "CORE_RADIUS_RATIO",
    "HUE_BINS",
    "ROI_NAME",
    "SAT_SOLID",
    "SLOT_IDS",
    "VAL_BRIGHT",
    "EbMainFeatures",
    "extract_features",
    "extract_slot_features",
]
