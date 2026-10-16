"""``hp_party_1/2/3``（后台角色血条）填充比例 —— **已冻结**。

★ 为什么要分槽阈值
------------------
血条越窄，边缘抗锯齿像素越多，它们和暗背景混合后明度被拉低。
后台血条只有 **87x8**（驻场那条是 370x15），所以驻场用的 ``V=150``
直接搬过来会**系统性偏低**：

    槽位   条尺寸      用 V=150 时的中位 fill_span
    hp_1   87x8                  0.744      <- 明显偏低
    hp_2   87x8                  0.988      <- 恰好还行
    hp_3   87x8                  0.831      <- 偏低

条本身的亮度就不同，三条**不能共用一个阈值**。

★ 阈值怎么来的
--------------
在 ``batch_20260929b`` 的 223 帧上扫描 V∈[60,200]，
选「中位 fill_span 达到 1.0 的前提下**取最大** V」。

★ 为什么取最大而不是最小 ★
  取最小 V 会把「暗空槽」也判成填充：实测 V=65 时满血帧冲到 82%，
  但 p25 也变成 1.000 —— 中间区间被**抹平，分辨率归零**。
  取最大 V 才能在「满血读 1.0」的同时保住中间的分辨力。

★ 换分辨率 / 换 UI 缩放必须重新标定
------------------------------------
这些是**绝对明度**阈值，依赖游戏内的美术设定。分辨率不是问题
（坐标是归一化的），但**画面整体亮度**（阴天 vs 晴天、室内 vs 野外）
会移动这个阈值。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ..roi_loader import load_rois

logger = logging.getLogger(__name__)

#: 分槽明度阈值。出处见模块 docstring。
V_BY_SLOT: dict[int, int] = {1: 90, 2: 130, 3: 120}

#: 血条判据（不管明暗，只要是绿就算"这是条血条"）
GREEN_H_LO, GREEN_HI, BAR_S = 35, 85, 80

#: 目前只配置了 1/2/3 号位。4 号位在 rois.toml 里没有 —— 之前人工确认
#: 「位置无效」，所以不猜、不硬塞。
SUPPORTED_SLOTS = (1, 2, 3)


@dataclass(frozen=True)
class HpPartySlot:
    """一个后台槽位的填充情况。"""

    slot: int
    fill_span: float        # 亮绿跨度 / 整条血条跨度，0~1
    bar_width: int

    @property
    def full(self) -> bool:
        return self.fill_span >= 0.97

    @property
    def empty(self) -> bool:
        return self.fill_span <= 0.03


def read_slot(frame: np.ndarray, slot: int) -> HpPartySlot:
    """读一个后台槽位。

    :param frame: BGR 整帧
    :param slot: 1 / 2 / 3
    :raises ValueError: slot 不在 :data:`SUPPORTED_SLOTS`
    :raises TypeError: frame 不是三通道图像
    """
    if slot not in SUPPORTED_SLOTS:
        raise ValueError(
            f"slot 只能是 {SUPPORTED_SLOTS} 之一，收到 {slot!r}"
            f"（rois.toml 没配 4 号位）")
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise TypeError(f"frame 应为 (H, W, 3) 的 BGR 图像")

    rect = load_rois().get(f"hp_party_{slot}")
    if rect is None:
        raise KeyError(f"rois.toml 里没有 hp_party_{slot}")
    crop = rect.crop(frame[:, :, :3])
    if crop is None or getattr(crop, "size", 0) == 0:
        return HpPartySlot(slot, 0.0, 0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bar = (h >= GREEN_H_LO) & (h <= GREEN_HI) & (s > BAR_S)
    W = crop.shape[1]
    if not bar.any():
        return HpPartySlot(slot, 0.0, W)

    bright_col = (bar & (v > V_BY_SLOT[slot])).any(axis=0)
    bar_col = bar.any(axis=0)
    if not bright_col.any() or not bar_col.any():
        return HpPartySlot(slot, 0.0, W)

    bl = int(np.argmax(bright_col))
    br = W - 1 - int(np.argmax(bright_col[::-1]))
    tl = int(np.argmax(bar_col))
    tr = W - 1 - int(np.argmax(bar_col[::-1]))
    bar_w = max(1, tr - tl + 1)
    return HpPartySlot(slot, round((br - bl + 1) / bar_w, 6), bar_w)


def read_all(frame: np.ndarray) -> list[HpPartySlot]:
    """读全部已配置的后台槽位。"""
    return [read_slot(frame, s) for s in SUPPORTED_SLOTS]


__all__ = [
    "BAR_S",
    "GREEN_HI",
    "GREEN_H_LO",
    "SUPPORTED_SLOTS",
    "V_BY_SLOT",
    "HpPartySlot",
    "read_all",
    "read_slot",
]
