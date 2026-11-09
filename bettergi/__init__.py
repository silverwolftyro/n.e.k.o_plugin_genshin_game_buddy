"""genshin_game_buddy · BetterGI 桥接子包。

★ 2026-11-07：从独立插件 ``bettergi_companion`` **物理并入** ``genshin_game_buddy``。

单插件架构下，这个子包承担原独立插件的全部职责：

* :mod:`bettergi.webhook_server` —— 监听 BetterGI 的 Webhook（正向：事件进来）
* :mod:`bettergi.control` —— 指令队列 + 回执（反向：YUI 指挥 BetterGI）
* :mod:`bettergi.event_mapper` —— 事件码 → 台词（含限流 / 防复读 / 场景组冷却）
* :mod:`bettergi.bridge` —— 把上面三者接到 NEKO 宿主上（SDK 无关）
* ``config.toml`` —— 桥接自身的配置（用标准库 tomllib 读，不走宿主配置体系）

四个模块**都不 import NEKO 插件 SDK**，因此可以脱离宿主单独跑离线测试；
宿主相关的东西全部在 :class:`bettergi.bridge.BettergiBridge` 上通过构造参数注入。

顶层导出只是为了方便 ``from .bettergi import BettergiBridge`` 这种写法；
每个子模块仍可单独 import，不依赖本文件先被加载。
"""

from __future__ import annotations

from .bridge import (
    COMPANION_LINE_PREFIX,
    COMPANION_LINE_VISIBILITY,
    CONTROL_SELF_CHECK_REMINDER,
    CONTROL_TOOL_DESCRIPTION,
    PUSH_SOURCE,
    BettergiBridge,
)
from .control import (
    ACTION_CATALOG,
    ACTION_NAMES,
    ACTION_SUMMARY,
    COMMAND_TOOL_PARAMETERS,
    Command,
    CommandQueue,
    CommandResult,
    ControlAction,
)
from .event_mapper import DEFAULT_CONFIG, EventMapper
from .webhook_server import WebhookConfig, WebhookServer

__all__ = [
    "ACTION_CATALOG",
    "ACTION_NAMES",
    "ACTION_SUMMARY",
    "COMMAND_TOOL_PARAMETERS",
    "COMPANION_LINE_PREFIX",
    "COMPANION_LINE_VISIBILITY",
    "CONTROL_SELF_CHECK_REMINDER",
    "CONTROL_TOOL_DESCRIPTION",
    "Command",
    "CommandQueue",
    "CommandResult",
    "ControlAction",
    "BettergiBridge",
    "DEFAULT_CONFIG",
    "EventMapper",
    "PUSH_SOURCE",
    "WebhookConfig",
    "WebhookServer",
]
