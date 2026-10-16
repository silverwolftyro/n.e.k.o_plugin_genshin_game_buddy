"""autopilot 独立工具集。

每个工具都可以直接跑（不依赖 N.E.K.O. 宿主）：

    python -m autopilot.tools.check_env
    python -m autopilot.tools.capture_demo --title 原神
    python -m autopilot.tools.template_demo --title 原神
    python -m autopilot.tools.autopilot_demo --title 原神
    python -m autopilot.tools.check_dataset --data D:/NEKO/yolo_data
    python -m autopilot.tools.train_yolo --data D:/NEKO/yolo_data
    python -m autopilot.tools.eval_yolo --data D:/NEKO/yolo_data
    python -m autopilot.tools.convert_to_onnx --weights runs/detect/train/weights/best.pt

也可以从插件目录直接跑单个文件（会自动 bootstrap sys.path）：
    python autopilot/tools/check_env.py
"""

from __future__ import annotations

__all__ = []
