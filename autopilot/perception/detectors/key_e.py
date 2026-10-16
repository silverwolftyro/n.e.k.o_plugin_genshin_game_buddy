"""``key_e``（元素战技）检测器 —— 中心方块饱和度 + 明度双闸门。

★ 为什么不用模板匹配
--------------------
E 是**角色技能**图标，每个角色图案都不同。实测模板分数随角色在
0.914 / 0.218 之间跳，换角色就废。只能靠"外观状态"而非"图案内容"。

★ 状态信号是饱和度，不是明度
----------------------------
就绪 = 彩色（金/角色元素色）；CD 中 = 去饱和的灰白。
两者的**明度几乎一样**（实测都在 190~205），动的是饱和度。
所以判据建在饱和度上，明度只用来识别"这里压根不是 E 图标"。

★ 三条分支的来源（都实测过）
----------------------------
1. 大地图/面板覆盖 —— ``center_sat_ratio >= 0.30``
   阈值 0.3 精确命中 9 帧，且这 9 帧的 sat/val/hue 几乎相同
   （sat 114~161、val 56~121、hue 74~101），是**同一类画面**。
   其中 2 帧是人工确认的大地图，另外 7 帧特征完全同源。
   阈值下面一刀切得很干净：ratio >= 0.2 时是 10 帧，>= 0.3 时是 9 帧，
   0.29 附近**没有**样本，落点有余量。

2. 活动页/非战斗 UI —— ``center_val >= 218 且 center_sat < 27``
   ⚠ 这条是**照着 3 帧（f00210/214/218）拟的**，而那 3 帧是连续帧、
      内容几乎一样。**样本量不足，泛化性未验证**。宁可误判也不能
      让机器人对着活动页按战技。

3. 其余按 ``center_sat`` 分 READY / COOLDOWN
   ⚠ 30 / 40 / 50 / 60 / 80 / 100 六个阈值都能"用"，
      分布是**连续的一团**（[10,20) 93 帧、[40,50) 35 帧、[110,120) 7 帧），
      没有天然谷底 —— 30 只是"就绪/未就绪大致对半"的位置，
      **需要下一轮用人工标注重新标定**。

★ 模块顶层不读文件、不建 Tk
--------------------------
纯 numpy / cv2 计算，import 这个包没有任何副作用。
"""

from __future__ import annotations

import logging
from enum import Enum

import cv2
import numpy as np

from ..roi_loader import load_rois

logger = logging.getLogger(__name__)

ROI_NAME = "key_e"

#: 中心方块边长占 ROI 短边的比例。
#: 整块 ROI 会被边缘透入污染（实测 f00086/87/89 圆圈边缘透进橙色地面），
#: 取中心 50% 只剩 44x44，但那是图标本体所在，够用。
CENTER_FRAC = 0.5

#: ① 大地图/面板：中心里 S>100 的像素占比下限
SATRATIO_MAP = 0.30
#: ② 活动页：明度很高但饱和度不高
VAL_UI_MIN = 218.0
SAT_UI_MAX = 27.0
#: ③ 就绪/未就绪的饱和度分界（**未标定，见模块 docstring**）
SAT_READY = 30.0


class KeyEState(Enum):
    """元素战技的三态。"""

    READY = "ready"
    COOLDOWN = "cooldown"
    UNKNOWN = "unknown"


def _center(crop: np.ndarray, frac: float = CENTER_FRAC):
    """取中心正方形并转 HSV，返回 ``(sat, val)`` 两个通道。"""
    h, w = crop.shape[:2]
    side = int(round(min(h, w) * frac))
    if side < 4:
        return None, None
    y0, x0 = (h - side) // 2, (w - side) // 2
    hsv = cv2.cvtColor(crop[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1], hsv[:, :, 2]


def center_stats(frame: np.ndarray) -> dict[str, float]:
    """中心方块的三个统计量。供批量分析与调参用。"""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise TypeError(f"frame 应为 (H, W, 3) 的 BGR 图像")
    rect = load_rois().get(ROI_NAME)
    if rect is None:
        raise KeyError(f"rois.toml 里没有 {ROI_NAME!r}")
    crop = rect.crop(frame[:, :, :3])
    if crop is None or getattr(crop, "size", 0) == 0:
        return {"center_sat": 0.0, "center_sat_ratio": 0.0, "center_val": 0.0}
    sat, val = _center(crop)
    if sat is None:
        return {"center_sat": 0.0, "center_sat_ratio": 0.0, "center_val": 0.0}
    return {
        "center_sat": float(sat.mean()),
        "center_sat_ratio": float((sat > 100).mean()),
        "center_val": float(val.mean()),
    }


def detect_key_e(frame: np.ndarray) -> tuple[KeyEState, float]:
    """检测元素战技状态。

    :param frame: BGR 整帧
    :returns: ``(状态, 置信度)``。置信度 = 中心方块平均饱和度
        （READY/COOLDOWN 都在这条轴上，UNKNOWN 时参考意义有限）。
    """
    st = center_stats(frame)
    sat, ratio, val = st["center_sat"], st["center_sat_ratio"], st["center_val"]

    # ① 大地图 / 面板盖上来：中心几乎全是高饱和地形色
    if ratio >= SATRATIO_MAP:
        return KeyEState.UNKNOWN, sat
    # ② 活动页之类的亮底 UI
    if val >= VAL_UI_MIN and sat < SAT_UI_MAX:
        return KeyEState.UNKNOWN, sat
    # ③ 正常图标：彩色 = 就绪，灰白 = CD
    return (KeyEState.READY if sat >= SAT_READY else KeyEState.COOLDOWN), sat


__all__ = [
    "CENTER_FRAC",
    "ROI_NAME",
    "SATRATIO_MAP",
    "SAT_READY",
    "SAT_UI_MAX",
    "VAL_UI_MIN",
    "KeyEState",
    "center_stats",
    "detect_key_e",
]
