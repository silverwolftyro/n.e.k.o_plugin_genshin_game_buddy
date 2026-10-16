"""N.E.K.O. 3.0 Autopilot —— 读屏 + 决策 + 模拟输入的感知/决策/执行骨架。

分层（对应《N.E.K.O. 3.0 架构设计文档 v1.0》第 4 章）：
    contracts/   数据契约（frozen dataclass + JSON Schema）
    perception/  感知层：截图 → 识别 → StatePacket
    policy/     决策层：StatePacket → IntentPacket（第 2 期）
    actuator/   执行层：IntentPacket → 输入事件（第 2 期）
    config/     配置加载与校验
    tools/      独立可执行工具

许可证约束（重要）：
    本包运行时**禁止 import ultralytics**（AGPL-3.0，会传染 Apache-2.0 的 N.E.K.O.）。
    YOLO 推理只走 ONNX + onnxruntime（MIT）。训练脚本 tools/train_yolo.py 可用
    ultralytics（工具性质、不分发），但产出必须是 ONNX。

模块顶层无任何副作用（对齐插件既有 R8 红线）。
"""

from __future__ import annotations

__version__ = "3.0.0.dev0"
__all__ = ["__version__"]
