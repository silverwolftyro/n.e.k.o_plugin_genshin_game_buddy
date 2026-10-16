"""``eb_main``（大招骰子）三态分类：READY / NOT_READY / UNKNOWN。

★ 全部参数来自 ``batch_20260929b`` 的 249 张大世界帧
-------------------------------------------------
    训练集  : cluster_id != 1 的 249 帧
    来源    : eb_main_cluster.py 跑出的 KMeans(k=2) 簇心
    标准化  : 对上面 249 帧逐列求 mean / std（population std，ddof=0）

三个簇心的来源（**不是**拍脑袋的，是 KMeans 的输出）：

    EMPTY   (  2.6992, 0.0010,  32.81, 205.63)  n=234  骰子不亮
    READY   ( 86.9648, 0.6605, 121.92, 237.88)  n= 15  骰子满（蓝色发光）
    ANOMALY (-15.2727, 0.6203, 117.55, 104.69)  n= 47  **非大世界**（地图/面板）

ANOMALY 不是"第三种大招状态"，而是**画面本身不对**的那些帧 ——
它们的骰子位置被大地图或属性面板盖住了。把它们单列成 UNKNOWN 而不是
硬塞进 READY/NOT_READY，是因为"非大世界"下讨论大招状态没有意义，
而强行二分会让机器人对着地图按出大招。

★ 为什么手写标准化 + 欧氏距离，不用 sklearn
-----------------------------------------
引擎 venv 的 numpy 1.26.4 被 cv2 / onnxruntime 钉住，为一个分类器装
sklearn 不划算。最近簇心分类（标准化后欧氏距离）就是 k-means 的
"分类阶段"，无需迭代求解。

★ ★ 换分辨率/换批次必须重新标定 ★ ★
------------------------------------
这些常数是**针对这批数据**的。换一批录制、换一个游戏版本、
或者采集时的画面风格变了（比如全是室内），分布都会漂，
届时必须重跑 ``eb_main_cluster.py`` 更新这三个簇心和 scaler。
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np

from .eb_main import extract_features

logger = logging.getLogger(__name__)


class EbMainState(Enum):
    """大招骰子的三态。"""

    READY = "ready"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"


#: 参与分类的 4 个特征，顺序必须和下面三个簇心**严格一致**
FEATURES = ("ring_sat_delta", "saturated_ratio", "mean_sat", "mean_val")

#: 标准化参数：来自 249 张大世界帧的逐列 mean / population std
SCALER_MEAN = (7.775426, 0.040719, 38.179923, 207.569520)
SCALER_STD = (20.740059, 0.157102, 29.606812, 19.406132)

#: 三个簇心（KMeans k=2 的输出 + 非大世界簇）
READY_CENTER = (86.9648, 0.6605, 121.9197, 237.8830)     # n=15
EMPTY_CENTER = (2.6992, 0.0010, 32.8120, 205.6263)       # n=234
ANOMALY_CENTER = (-15.2727, 0.6203, 117.5487, 104.6897)  # n=47，非大世界

#: 低于这个置信度建议人工复核（两簇差不多远 = 分界样本）
LOW_CONFIDENCE = 0.3


def _standardize(v: np.ndarray) -> np.ndarray:
    m = np.asarray(SCALER_MEAN, dtype=np.float64)
    s = np.asarray(SCALER_STD, dtype=np.float64)
    s = np.where(s == 0, 1.0, s)          # 常数列不能除 0
    return (v - m) / s


def classify_features(feats) -> tuple[EbMainState, float, float]:
    """对**已提取的特征**分类。返回 ``(状态, 置信度, ring_sat_delta)``。

    拆出来是为了批量验证时不必反复裁图 —— 特征算一次、分类多次。

    :param feats: :class:`~.eb_main.EbMainFeatures` 或 4 元素序列
    """
    if hasattr(feats, "to_dict"):
        d = feats.to_dict()
        v = np.array([float(d[k]) for k in FEATURES], dtype=np.float64)
        ring = float(d["ring_sat_delta"])
    else:
        v = np.asarray(feats, dtype=np.float64)
        ring = float(v[0])

    z = _standardize(v)
    centers = {
        EbMainState.READY: _standardize(np.array(READY_CENTER, dtype=np.float64)),
        EbMainState.NOT_READY: _standardize(
            np.array(EMPTY_CENTER, dtype=np.float64)),
        EbMainState.UNKNOWN: _standardize(
            np.array(ANOMALY_CENTER, dtype=np.float64)),
    }
    dists = {k: float(np.linalg.norm(z - c)) for k, c in centers.items()}
    ordered = sorted(dists.items(), key=lambda kv: kv[1])
    (state, d1), (_, d2) = ordered[0], ordered[1]

    # 置信度：最近距离相对次近距离越小越自信。
    # d1==d2（正好在分界）-> 0；d1==0 -> 接近 1。
    conf = 0.0 if d2 <= 1e-9 else max(0.0, 1.0 - d1 / d2)
    return state, conf, ring


def classify_eb_main(frame: np.ndarray) -> tuple[EbMainState, float]:
    """输入**整帧**，输出 ``(状态, 置信度)``。

    :param frame: BGR 整帧 ``numpy.ndarray``，``(H, W, 3)``
    :raises TypeError: 输入不是三通道图像
    """
    feats = extract_features(frame)
    state, conf, _ring = classify_features(feats)
    if conf < LOW_CONFIDENCE:
        logger.debug("eb_main 低置信度：%s conf=%.3f ring=%.1f",
                     state.value, conf, _ring)
    return state, conf


__all__ = [
    "ANOMALY_CENTER",
    "EMPTY_CENTER",
    "FEATURES",
    "LOW_CONFIDENCE",
    "READY_CENTER",
    "SCALER_MEAN",
    "SCALER_STD",
    "EbMainState",
    "classify_eb_main",
    "classify_features",
]
