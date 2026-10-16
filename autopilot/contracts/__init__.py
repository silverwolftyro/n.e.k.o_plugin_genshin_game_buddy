"""数据契约：状态包（上行）与意图包（下行）。

设计要点（对应架构文档第 5 章）：
  · 全部 frozen dataclass —— 状态一旦产生不可变，避免下游看到"半更新"的数据
  · 三态约定：bool | None，None = UNKNOWN（**既不就绪也不未就绪，决策时保守跳过**）
  · 位置丢失时 position=None，**禁止用运动预测填补**（原则 1 / 2）
  · 每个包都带 schema_version + state_version，前者管契约兼容，后者管状态新鲜度
"""

from __future__ import annotations

from .intent_packet import (
    ActionKind,
    ActionType,
    IntentConstraints,
    IntentPacket,
    IntentTarget,
)
from .state_packet import (
    PlayerInfo,
    PartyMember,
    PositionInfo,
    ResourceInfo,
    SceneInfo,
    StatePacket,
)
from .events import Event, EventCategory, Severity

__all__ = [
    "ActionKind",
    "ActionType",
    "Event",
    "EventCategory",
    "IntentConstraints",
    "IntentPacket",
    "IntentTarget",
    "PlayerInfo",
    "PartyMember",
    "PositionInfo",
    "ResourceInfo",
    "SceneInfo",
    "Severity",
    "StatePacket",
]

SCHEMA_VERSION = "1.0"
