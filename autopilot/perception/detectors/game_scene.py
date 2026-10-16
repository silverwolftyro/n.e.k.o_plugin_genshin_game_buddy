"""判断当前帧是不是「大世界游戏画面」。

★ 为什么需要它
--------------
录制过程中打开过大地图、传送界面、角色属性面板，这些画面**同样有小地图
UI**，但 eb_main（骰子）那个位置被 UI 盖住了。不滤掉的话，骰子特征聚类
会把"地图画面"当成一个视觉状态 —— 实测就是这样：KMeans 把 47 帧地图/面板
单独聚成了一簇，混进真正的骰子标定数据里。

★ ★ 重要：minimap 圆度**不是**有效判据 ★ ★
--------------------------------------------
最初的设计是"minimap ROI 有清晰圆边界 = 大世界"。**实测证伪**：

    帧      画面            Hough 找到的圆数
    f00280  蒙德城（世界）      33
    f00160  全屏世界地图        33     ← 一样多
    f00064  全屏世界地图        45
    f00155  全屏世界地图         0

因为**打开大地图时左上角的小地图仍然在**（原神自己的设计），圆边界一直都
在。照那个思路写出来的检测器会"看起来能跑"，但对地图帧一样给高分。

★ 实际用的两个判据（都是 2026-09-30 在 9 张人工标注帧上量化选出来的）
----------------------------------------------------------------
  ① 顶部 20% 区域的高亮像素占比 ``top_bright``
     大世界该区域是**天空或明亮建筑**；全屏地图该区域是**深色 UI 栏**。

        世界帧 0.38 ~ 0.89   |   地图帧 0.02 ~ 0.11     ← 完全分离

  ② 整帧高饱和绿色占比 ``green_frac``
     用来**单独排除属性面板** —— 面板顶部也很亮（0.66），但画面主体是
     角色立绘，几乎没有绿色（0.0009，比最低的世界帧还低 25 倍）。

        面板帧 0.0009   |   世界帧 0.023 ~ 0.248

★ ⚠ 阈值是 9 个样本调出来的，有过拟合风险 ★
------------------------------------------
所以这里给的是**软分数**而不是硬二分：``confidence`` 连续取值，
阈值放在 0.5 只是给 UI 一个默认切分点。拿到全量 296 帧后应当重新标定。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np


logger = logging.getLogger(__name__)

ROI_NAME = "minimap"

#: 顶部多大比例算"天/建筑区"
TOP_BAND = 0.20
#: 明度下限（0~255）。低于它算"暗"，即 UI 栏 / 深色地图底
BRIGHT_V = 150
#: 饱和度下限 —— 排除灰色 UI
BRIGHT_S = 40

#: 绿色判定：H 35~85（OpenCV 的 0~179 色相空间里的绿）
GREEN_H_LO, GREEN_HI = 35, 85
GREEN_S = 60

# ── 判别阈值（9 帧标定，见模块 docstring）──────────────────────────
#: top_bright 超过它算"有天空/建筑"。0.20 落在地图(<=0.11)与世界(>=0.38)中间。
BRIGHT_ON = 0.20
#: 过渡带宽度。地图 0.11 -> 世界 0.38，中间 0.27 宽，用它把分数摊开。
BRIGHT_SPAN = 0.09
#: green_frac 低于它算"没有地形色"，即属性面板之类。
GREEN_OFF = 0.01
#: green 过渡带
GREEN_SPAN = 0.02

#: confidence 低于它判为非大世界
IS_WORLD_CONF = 0.5


@dataclass(frozen=True)
class SceneCheck:
    """一次判定的结果。"""

    is_world: bool
    confidence: float
    reason: str
    #: 判据原值，调试/标定时要看
    top_bright: float
    green_frac: float


def _band_metrics(frame: np.ndarray) -> tuple[float, float]:
    """返回 ``(top_bright, green_frac)``。"""
    h = frame.shape[0]
    top = frame[: max(1, int(h * TOP_BAND)), :, :3]
    thsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    top_bright = float(((thsv[:, :, 2] > BRIGHT_V)
                        & (thsv[:, :, 1] > BRIGHT_S)).mean())

    fhsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat = fhsv[:, :, 0], fhsv[:, :, 1]
    green_frac = float(((sat > GREEN_S)
                       & (hue >= GREEN_H_LO) & (hue <= GREEN_HI)).mean())
    return top_bright, green_frac


def check_world_scene(frame: np.ndarray) -> SceneCheck:
    """判断整帧是否为大世界游戏画面。

    :param frame: BGR 整帧 ``numpy.ndarray``，``(H, W, 3)``
    :raises TypeError: 输入不是三通道图像

    ★ 注意：minimap ROI **只用于将来扩展**（比如加圆形/文本判据），
      当前的 top_bright / green_frac 是**整帧**统计，不依赖 ROI。
      保留 ROI 引用是为了让调用方看到"这个检测器和 ROI 配置是同一套资产"。
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 \
            or frame.shape[2] not in (3, 4):
        raise TypeError(
            f"frame 应为 (H, W, 3) 的 BGR 图像，实际 shape="
            f"{getattr(frame, 'shape', None)}")

    top_bright, green_frac = _band_metrics(frame)

    # ① 亮带分：地图帧此项很低（<=0.11），世界帧高（>=0.38）
    bright_score = _ramp(top_bright, BRIGHT_ON, BRIGHT_SPAN)
    # ② 绿量分：面板帧此项为 0
    green_score = 1.0 - _ramp(green_frac, GREEN_OFF, GREEN_SPAN)

    confidence = float(min(bright_score, green_score))

    if confidence >= IS_WORLD_CONF:
        reason = "world_sky_and_terrain"
    elif bright_score < green_score:
        reason = "map_fullscreen"      # 顶部是深色 UI 栏
    else:
        reason = "panel_or_flat_ui"    # 顶部落选但没有地形色
    return SceneCheck(
        is_world=confidence >= IS_WORLD_CONF,
        confidence=confidence,
        reason=reason,
        top_bright=top_bright,
        green_frac=green_frac,
    )


def _ramp(value: float, lo: float, span: float) -> float:
    """把 ``value`` 映射到 0~1：低于 ``lo`` 得 0，高出 ``lo+span`` 得 1。"""
    if span <= 0:
        return 1.0 if value >= lo else 0.0
    return max(0.0, min(1.0, (value - lo) / span))


__all__ = [
    "BRIGHT_ON",
    "GREEN_OFF",
    "IS_WORLD_CONF",
    "ROI_NAME",
    "SceneCheck",
    "check_world_scene",
]
