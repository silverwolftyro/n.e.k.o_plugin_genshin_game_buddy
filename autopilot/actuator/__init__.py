"""执行层：意图 -> 键鼠事件。

对应架构文档 4.3 / 6.7。各模块职责：

    validator.py   合法性：这个意图在当前状态下该不该执行（带具体原因）
    safety.py      许可：现在允不允许执行（急停 / 熔断 / 频率 / 焦点 / 黑名单）
    traits.py      行为：每个动作的差异化参数（**表驱动，不散落 if-else**）
    humanizer.py   时序：给固定时序加抖动
    simulator.py   注入：ctypes SendInput（**返回值校验**）
    executor.py    编排：展开意图 -> 逐动作 -> finally 释放

**许可证**��本包**不 import ultralytics**（AGPL）。它连 ONNX 都不碰
（识别是 perception 的事），只有 Win32 ctypes。

**测试**：`IntentExecutor(dry_run=True)` 可以在不发任何真实输入的情况下
完整跑通本包。
"""

from __future__ import annotations

from .executor import DEFAULT_KEYMAP, ExecutionResult, IntentExecutor, Outcome
from .humanizer import Humanizer, HumanizerConfig
from .keymap import (
    KeymapError,
    dump_keymap,
    keymap_path,
    load_keymap,
    save_keymap,
)
from .safety import (
    DEFAULT_MAX_ACTIONS,
    PANIC_KEY,
    REASON_BREAKER,
    REASON_PANIC,
    BreakerState,
    BreakerTrip,
    SafetyConfig,
    SafetyGuard,
    is_hard_stop,
)
from .simulator import (
    IS_WINDOWS,
    BackendReport,
    DryRunBackend,
    InputBackend,
    InputInjectionError,
    InputSimulator,
    InvalidStateError,
    KeyId,
    KeyIdError,
    SentEvent,
    VK,
    Win32Backend,
    decode_events,
    probe_backend,
    process_integrity,
)
from .traits import (
    DEFAULT_TRAITS,
    TRAITS,
    ActionTraits,
    extra_traits,
    missing_traits,
    traits_of,
    validate_table,
)
from .validator import IntentValidator, SeenIntentStore, ValidationResult

__all__ = [
    "DEFAULT_KEYMAP",
    "DEFAULT_MAX_ACTIONS",
    "DEFAULT_TRAITS",
    "IS_WINDOWS",
    "PANIC_KEY",
    "REASON_BREAKER",
    "REASON_PANIC",
    "TRAITS",
    "ActionTraits",
    "BackendReport",
    "BreakerState",
    "BreakerTrip",
    "DryRunBackend",
    "ExecutionResult",
    "Humanizer",
    "HumanizerConfig",
    "InputBackend",
    "InputInjectionError",
    "InputSimulator",
    "IntentExecutor",
    "IntentValidator",
    "InvalidStateError",
    "KeyId",
    "KeyIdError",
    "KeymapError",
    "Outcome",
    "SafetyConfig",
    "SafetyGuard",
    "SeenIntentStore",
    "SentEvent",
    "ValidationResult",
    "VK",
    "Win32Backend",
    "decode_events",
    "dump_keymap",
    "extra_traits",
    "is_hard_stop",
    "keymap_path",
    "load_keymap",
    "missing_traits",
    "probe_backend",
    "process_integrity",
    "save_keymap",
    "traits_of",
    "validate_table",
]
