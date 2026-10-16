"""WorldState：不可变的全局状态快照。

对应架构文档 4.5。

**为什么用不可变快照而不是全局可变单例**：
    BetterGI 的 RunnerContext / TaskContext / CancellationContext 三个单例，
    RunnerContext 里还有 14 个可变字段和惰性识别的 _combatScenes；
    MaaEnd 的 AutoFightTask.FightStatusFlag / ESkillCdTracker / AvatarMacro
    全是包级 static，配置靠 AsyncLocal 传递。
    两者都因为这个吃了亏 —— 跨任务耦合、无法单测。

3.0 的做法：每帧产生**新实例**，下游读到的一定是一致快照。
这样意图可以绑定 ``state_version``，执行层校验「我执行的时候状态变了吗」。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .contracts.intent_packet import IntentPacket
from .contracts.state_packet import StatePacket


class ConnectionLevel(str, Enum):
    """连接等级。**本地永远能跑**（原则：L0 也不依赖云端）。"""

    L0_STARTING = "L0_STARTING"   # 尚未连接，本地已可用
    L1_ONLINE = "L1_ONLINE"       # 双向正常
    L2_DEGRADED = "L2_DEGRADED"   # 断连/心跳超时，仅本地
    L3_OFFLINE = "L3_OFFLINE"     # 服务器明确不可用（Token 失效等）


@dataclass(frozen=True)
class ConnectionState:
    level: ConnectionLevel = ConnectionLevel.L0_STARTING
    rtt_ms: int = -1
    last_heartbeat_ms: int = 0
    reason: str = ""


@dataclass(frozen=True)
class IntentRecord:
    """一次意图的完整生命周期记录。"""

    intent: IntentPacket
    issued_ms: int
    decided_ms: int = 0
    outcome: str = "PENDING"   # PENDING / OK / FAILED / REJECTED / INTERRUPTED
    detail: str = ""
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent.intent_id,
            "action": self.intent.action.value,
            "issued_ms": self.issued_ms,
            "decided_ms": self.decided_ms,
            "outcome": self.outcome,
            "detail": self.detail,
            "attempts": self.attempts,
        }


@dataclass(frozen=True)
class FrameCounters:
    total_frames: int = 0
    failed_frames: int = 0
    events_emitted: int = 0
    started_ms: int = 0

    @property
    def uptime_s(self) -> float:
        if not self.started_ms:
            return 0.0
        return (time.time() * 1000 - self.started_ms) / 1000.0


@dataclass(frozen=True)
class WorldState:
    """全局状态的一次性快照。**不可变**。"""

    current_state: StatePacket
    last_states: tuple[StatePacket, ...] = ()
    intent_history: tuple[IntentRecord, ...] = ()
    connection: ConnectionState = field(default_factory=ConnectionState)
    counters: FrameCounters = field(default_factory=FrameCounters)
    state_version: int = 0

    @property
    def recent(self) -> tuple[StatePacket, ...]:
        return self.last_states

    def scene_history_ms(self) -> list[tuple[str, int]]:
        """[(场景, 停留毫秒)]，从最早到最新。用于"待了多久"的判断。"""
        if not self.last_states:
            return []
        out: list[tuple[str, int]] = []
        prev = self.last_states[0]
        for cur in self.last_states[1:]:
            if cur.scene.id != prev.scene.id:
                out.append((prev.scene.id, cur.timestamp_ms - prev.timestamp_ms))
            prev = cur
        out.append((prev.scene.id, self.current_state.timestamp_ms - prev.timestamp_ms))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "current": self.current_state.to_dict(),
            "connection": {
                "level": self.connection.level.value,
                "rtt_ms": self.connection.rtt_ms,
                "reason": self.connection.reason,
            },
            "counters": {
                "total_frames": self.counters.total_frames,
                "failed_frames": self.counters.failed_frames,
                "events_emitted": self.counters.events_emitted,
                "uptime_s": round(self.counters.uptime_s, 1),
            },
            "intents": [r.to_dict() for r in self.intent_history[-10:]],
        }


class WorldStateStore:
    """管理 WorldState 的更新。**每次 update 产生新实例**，不原地改。"""

    #: 保留最近 N 帧历史（约 3-6 秒，够做趋势判断）
    HISTORY_SIZE = 30
    #: 保留最近 N 条意图历史
    INTENT_HISTORY_SIZE = 50

    def __init__(self, *, history_size: int | None = None) -> None:
        self._states: deque[StatePacket] = deque(
            maxlen=history_size or self.HISTORY_SIZE
        )
        self._intents: deque[IntentRecord] = deque(maxlen=self.INTENT_HISTORY_SIZE)
        self._connection = ConnectionState()
        self._counters = FrameCounters(started_ms=int(time.time() * 1000))
        self._current: StatePacket | None = None

    # -- 读 --------------------------------------------------------------

    def get(self) -> WorldState:
        """取当前快照。没有帧时返回一个空壳（而不是崩）。"""
        if self._current is None:
            empty = StatePacket.unknown(reason="no_frame_yet")
            return WorldState(current_state=empty, state_version=0)
        return WorldState(
            current_state=self._current,
            last_states=tuple(self._states),
            intent_history=tuple(self._intents),
            connection=self._connection,
            counters=self._counters,
            state_version=self._current.state_version,
        )

    @property
    def has_state(self) -> bool:
        return self._current is not None

    # -- 写 --------------------------------------------------------------

    def update(self, state: StatePacket) -> WorldState:
        """记入一帧并返回新的 WorldState。"""
        self._current = state
        self._states.append(state)
        ok = state.resource.perception_ok
        self._counters = FrameCounters(
            total_frames=self._counters.total_frames + 1,
            failed_frames=self._counters.failed_frames + (0 if ok else 1),
            events_emitted=self._counters.events_emitted,
            started_ms=self._counters.started_ms,
        )
        return self.get()

    def record_intent(self, record: IntentRecord) -> WorldState:
        self._intents.append(record)
        return self.get()

    def set_connection(
        self,
        level: ConnectionLevel,
        *,
        rtt_ms: int = -1,
        reason: str = "",
    ) -> WorldState:
        self._connection = ConnectionState(
            level=level,
            rtt_ms=rtt_ms,
            last_heartbeat_ms=int(time.time() * 1000),
            reason=reason,
        )
        return self.get()

    def bump_events(self, n: int = 1) -> None:
        self._counters = FrameCounters(
            total_frames=self._counters.total_frames,
            failed_frames=self._counters.failed_frames,
            events_emitted=self._counters.events_emitted + n,
            started_ms=self._counters.started_ms,
        )

    def reset(self) -> None:
        self._states.clear()
        self._intents.clear()
        self._current = None
        self._connection = ConnectionState()
        self._counters = FrameCounters(started_ms=int(time.time() * 1000))


__all__ = [
    "ConnectionLevel",
    "ConnectionState",
    "FrameCounters",
    "IntentRecord",
    "WorldState",
    "WorldStateStore",
]
