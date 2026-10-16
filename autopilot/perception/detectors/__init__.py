"""单 ROI 特征提取器。

每个子模块负责**一个** UI 元素的特征提取，共用 :mod:`autopilot.perception.roi_loader`
拿到的归一化 ROI、��用同一套 HSV 统计口径。

为什么单独成包
----------------
``perception`` 根下那些是**管线**（capturer -> builder -> StatePacket），
一个 ROI 一个特征是**叶子工具**，两者生命周期和依赖都不同。
混在一起会让 builder 的 import 链越滚越长。

★ 硬边界：本层**只出数字，不下结论**
------------------------------------
「蓝到什么程度算满」是**下一轮**的事（要按角色分别标定）。
这里绝不能写 ``if blue_ratio > 0.5: return "READY"`` ——
那会让阈值散落在十几个文件里，改一次标定要翻半个仓库。
"""

from __future__ import annotations

from .eb_main import EbMainFeatures, extract_features

__all__ = ["EbMainFeatures", "extract_features"]
