"""感知层资产路径 —— **全仓唯一出处**。

★ 铁律：所有资产路径统一从这里取，不允许在别处硬编码
--------------------------------------------------------
ROI 配置、模板图、模型权重、抽帧样本……**任何地方都不许再写
``D:/NEKO/...`` 或 ``templates/...`` 这类字面量**。要改路径就改这里，
改一处生效全局。

硬编码路径的代价不是"多打几个字"，而是：
    1. 换机器要全仓搜索替换，漏一处就静默用错文件；
    2. 拼错一个反斜杠，报错是"文件不存在"，而不是"路径写错了"；
    3. 没人说得清某��资源到底从哪来。

★ 根目录可以用环境变量覆盖
--------------------------
``NEKO_RECOGNITION_ROOT`` > 默认 ``D:\\NEKO\\recognition``。
换机器 / 做对照实验时用环境变量即可，不必改代码。
"""

from __future__ import annotations

import os
from pathlib import Path

#: 资产根目录。环境变量优先，其次默认路径。
RECOGNITION_ROOT: Path = Path(
    os.environ.get("NEKO_RECOGNITION_ROOT", "").strip()
    or r"D:\NEKO\recognition"
)

# ── 配置 ────────────────────────────────────────────────────────────────
#: ROI 定义（归一化坐标 + 阈值 + 模板目录）
ROIS_TOML: Path = RECOGNITION_ROOT / "configs" / "rois.toml"
#: 独立的阈值文件。**当前不存在** —— 阈值暂时和 ROI 写在同一个
#: rois.toml 的 ``[thresholds]`` 段里。等它被拆出来之前，
#: 这个常量只是先把位置占住，别让代码去拼 "configs/thresholds.toml"。
THRESHOLDS_TOML: Path = RECOGNITION_ROOT / "configs" / "thresholds.toml"

#: 小地图比例（小地图像素 <-> 世界单位）。任务驱动定位的**唯一比例出处**，
#: 由 TileTracker 读取。缺了它 TileTracker 会直接抛，不猜默认值。
MINIMAP_SCALE_TOML: Path = RECOGNITION_ROOT / "configs" / "minimap_scale.toml"

#: 底图像素 <-> 世界坐标的换算（23 个人工锚点拟合，RMS 12.65 px）。
#: 不在本包内（是离线脚本），由 TileTracker 按需单独加载。
COORD_TRANSFORM_PY: Path = (
    RECOGNITION_ROOT / "exports" / "benchmark" / "scripts" / "misc"
    / "coord_transform.py")

# ── 模型 ────────────────────────────────────────────────────────────────
MODELS_PRETRAINED: Path = RECOGNITION_ROOT / "models" / "pretrained"
MODELS_TRAINED: Path = RECOGNITION_ROOT / "models" / "trained"
MODELS_ARCHIVED: Path = RECOGNITION_ROOT / "models" / "archived"

#: BetterGI 预训练的世界实体检测模型（7 类）+ 配套类名表。
#: ⚠️ **BetterGI 是 GPL-3.0，这两份文件不进 Apache-2.0 的分发包**，
#:    本地保留。缺了就抛 ``YoloUnavailableError``（响亮失败），不静默降级。
#:    详见 ``recognition/notes/LICENSING.md``。
BGI_WORLD_ONNX: Path = MODELS_PRETRAINED / "bgi_world.onnx"
BGI_WORLD_CLASSES: Path = MODELS_PRETRAINED / "classes_bgi_world.txt"

# ── 模板图 ──────────────────────────────────────────────────────────────
TEMPLATES_UI: Path = RECOGNITION_ROOT / "templates" / "ui"
TEMPLATES_WORLD: Path = RECOGNITION_ROOT / "templates" / "world"

# ── 抽帧样本（调试 / 基准用）────────────────────────────────────────────
FRAMES_BATCH_A: Path = RECOGNITION_ROOT / "raw" / "frames" / "batch_20260929a"
FRAMES_BATCH_B: Path = RECOGNITION_ROOT / "raw" / "frames" / "batch_20260929b"

# ── 标注与导出 ──────────────────────────────────────────────────────────
LABELS_PRETRAINED: Path = RECOGNITION_ROOT / "labels" / "yolo" / "prelabeled"
EXPORTS_BENCHMARK: Path = RECOGNITION_ROOT / "exports" / "benchmark"

#: ``rois.toml`` 里声明的基准分辨率。归一化坐标换算像素时按它。
#: **不硬编码在这里** —— ROI 表自己带这个值（``[meta].base_resolution``），
#: 见 :func:`autopilot.perception.roi_loader.load_meta`。
DEFAULT_BASE_RESOLUTION: tuple[int, int] = (2560, 1600)


__all__ = [
    "BGI_WORLD_CLASSES",
    "BGI_WORLD_ONNX",
    "COORD_TRANSFORM_PY",
    "DEFAULT_BASE_RESOLUTION",
    "EXPORTS_BENCHMARK",
    "FRAMES_BATCH_A",
    "FRAMES_BATCH_B",
    "LABELS_PRETRAINED",
    "MINIMAP_SCALE_TOML",
    "MODELS_ARCHIVED",
    "MODELS_PRETRAINED",
    "MODELS_TRAINED",
    "RECOGNITION_ROOT",
    "ROIS_TOML",
    "TEMPLATES_UI",
    "TEMPLATES_WORLD",
    "THRESHOLDS_TOML",
]
