"""autopilot 配置：TOML 加载 + pydantic 校验。

**加载期校验不可省**：模板不存在、ROI 表达式语法错、阈值超范围 ——
任何一项都应该在启动时炸掉并指出具体位置，而不是在感知循环里每帧失败一次。
"""

from __future__ import annotations

from .loader import (
    AUTOPILOT_ROOT,
    PLUGIN_ROOT,
    AutopilotConfig,
    CaptureConfig,
    OcrConfig,
    PerceptionConfig,
    TemplateEntry,
    YoloConfig,
    load_config,
    validate_config,
)

__all__ = [
    "AUTOPILOT_ROOT",
    "PLUGIN_ROOT",
    "AutopilotConfig",
    "CaptureConfig",
    "OcrConfig",
    "PerceptionConfig",
    "TemplateEntry",
    "YoloConfig",
    "load_config",
    "validate_config",
]
