"""意图包：决策层的结构化输出（执行层的唯一输入）。

对应架构文档 5.2。核心不变式：
  · action 必须在白名单内 —— 非法动作**拒绝执行**，不降级、不猜测
  · state_version 必须与执行时的状态匹配 —— 防止"在战斗里点背包按钮"
  · ttl_seconds 过期即丢 —— 云端往返期间世界可能已经变了
  · finally 里 release_all() —— 无论成败都松开所有按下的键
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Literal

from .state_packet import StatePacket


class ActionType(str, Enum):
    """动作白名单。

    **这是安全边界**：执行层只认这个枚举。LLM 或云端给出的任何其他字符串
    都会被拒绝（原则 1：响亮失败优于静默执行未知动作）。
    """

    # COMBAT
    NORMAL_ATTACK = "NORMAL_ATTACK"
    USE_SKILL = "USE_SKILL"
    USE_BURST = "USE_BURST"
    DODGE = "DODGE"
    SWITCH_CHARACTER = "SWITCH_CHARACTER"
    # NAV
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACK = "MOVE_BACK"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    SPRINT = "SPRINT"
    JUMP = "JUMP"
    #: ★ 走到**世界坐标 (x, y)** —— 全枚举里唯一带目标的导航意图。
    #: 坐标放在 ``IntentTarget.world_position``（``{"x":..,"y":..,"radius":..}``），
    #: **不要**塞进 ``params``：那个槽位绕过 ``POSITION_UNAVAILABLE``
    #: 校验（``IntentPacket.validate`` 第 5 步与
    #: ``actuator/validator.py::_check_position`` 都只看 world_position）。
    MOVE_TO = "MOVE_TO"
    # UI
    OPEN_INVENTORY = "OPEN_INVENTORY"
    OPEN_MAP = "OPEN_MAP"
    OPEN_PAIMON_MENU = "OPEN_PAIMON_MENU"
    CLICK_ELEMENT = "CLICK_ELEMENT"
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"
    CLOSE = "CLOSE"
    # ---------------- INTERACT（2026-10-11）----------------
    #: 交互键：**F**。原神 PC 默认键位里「拾取 / 采集 / 开宝箱 / 对话 /
    #: 激活锚点」**共用这一个键**（多个键位表一致），所以它是可靠的。
    #:
    #: ⚠️ **USE_FOOD / REST 为什么不在白名单里**（查证后判定为「不可靠」，
    #:   按规矩不做，而不是做个半成品）：
    #:
    #:   * ``USE_FOOD``（吃药）：原神 PC **没有食物快捷键**。使用食物必须
    #:     「B 打开背包 → 切到食物页签 → 点击具体食物 → Esc 关闭」，
    #:     是一串**依赖背包 UI 布局的复合操作**，而且要选「背包里的哪一个
    #:     食物」—— 那是物品选择问题，不是键位问题。当前 ``ActionType``
    #:     的整个模型是「一个动作 = 一个键」（见 ``DEFAULT_KEYMAP`` 的
    #:     ``KeyId`` 单键映射），塞不下这个。要做必须先扩动作层成
    #:     「多步序列 + UI 元素定位」，那是另一个量级的工程。
    #:
    #:   * ``REST``（休息回体力）：**原神没有这个机制**。原粹树脂只能每
    #:     8 分钟自然恢复 1 点，或用脆弱树脂/须臾树脂（+60）、原石兑换补充。
    #:     「食物、休息对恢复原粹树脂毫无作用」。原粹树脂也**不受是否在线
    #:     影响**，关机也在后台累积 —— 所以「等体力」的唯一正确实现是
    #:     :attr:`WAIT`，不是一个需要注入的游戏内动作。
    USE_PICKUP = "USE_PICKUP"
    # SYSTEM
    WAIT = "WAIT"
    SCREENSHOT = "SCREENSHOT"
    NOOP = "NOOP"

    @property
    def kind(self) -> "ActionKind":
        return _ACTION_KIND_MAP[self]


ActionKind = Literal["COMBAT", "NAV", "UI", "SYSTEM"]

_ACTION_KIND_MAP: dict[ActionType, ActionKind] = {
    **{a: "COMBAT" for a in (
        ActionType.NORMAL_ATTACK, ActionType.USE_SKILL, ActionType.USE_BURST,
        ActionType.DODGE, ActionType.SWITCH_CHARACTER,
    )},
    **{a: "NAV" for a in (
        ActionType.MOVE_FORWARD, ActionType.MOVE_BACK, ActionType.MOVE_LEFT,
        ActionType.MOVE_RIGHT, ActionType.SPRINT, ActionType.JUMP,
        ActionType.MOVE_TO,
    )},
    **{a: "UI" for a in (
        ActionType.OPEN_INVENTORY, ActionType.OPEN_MAP, ActionType.OPEN_PAIMON_MENU,
        ActionType.CLICK_ELEMENT, ActionType.CONFIRM, ActionType.CANCEL, ActionType.CLOSE,
        ActionType.USE_PICKUP,
    )},
    **{a: "SYSTEM" for a in (
        ActionType.WAIT, ActionType.SCREENSHOT, ActionType.NOOP,
    )},
}

#: 允许 state_version 漂移的帧数。状态每帧 +1，严格相等会让所有意图失效，
#: 但漂移过大又意味着世界已经变了。
MAX_STATE_VERSION_DRIFT = 10


@dataclass(frozen=True)
class IntentTarget:
    """执行前置条件。"""

    scene_required: tuple[str, ...] = ()
    ui_element: str | None = None
    world_position: dict | None = None
    precondition: str | None = None

    # -- 坐标读取（轴名兼容层，2026-10-14）--------------------------
    #
    # ★ 平面坐标的**规范键名是 ``x`` / ``y``**。
    #   ``PositionInfo`` 本来就是 ``x`` / ``y``（其中 ``y`` 存的是**世界 z**
    #   —— 见 ``perception/position_mapping.py`` 的 ``y=wz``），
    #   JSON schema 也是 ``x`` / ``y``。历史上有一批代码按 ``x`` / ``z``
    #   写（寻路脚本、事件 context、部分测试），本仓库内部已统一到 ``y``。
    #
    # ⚠️ **外部文件格式不变**：Console 的路点文件
    #   ``{"x":..,"z":..}`` 与 ``e2e_pathfind.py --start-world X,Z`` 是
    #   与外部脚本共享的约定，**仍然是 z**。转换发生在边界，不在契约里。
    #
    #: 仍接受 ``z`` 的原因：历史事件流与手写 JSON 里都是 ``z``，
    #: 读到 None 会让「坐标缺失」和「键名写错」变成同一种静默失败。

    @staticmethod
    def _coord(wp: dict, name: str) -> float | None:
        """取一个坐标分量。``y`` 优先，缺失时回退到遗留的 ``z``。"""
        for key in (name, "z" if name == "y" else name):
            v = wp.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
        return None

    def world_xy(self) -> tuple[float, float] | None:
        """``(x, y)``。缺任一分量或不是有限数则返回 ``None``。

        读侧**永远走这里**，不要直接 ``wp["x"]`` —— 那会把
        「键名写错」变成裸 KeyError。
        """
        wp = self.world_position
        if not isinstance(wp, dict) or not wp:
            return None
        x = self._coord(wp, "x")
        y = self._coord(wp, "y")
        if x is None or y is None:
            return None
        if not (isfinite(x) and isfinite(y)):
            return None
        return x, y

    def arrive_radius(self, default: float) -> float:
        """到达半径（米）。缺省或非法时返回 ``default``。"""
        wp = self.world_position or {}
        r = wp.get("radius")
        try:
            r = float(r)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return r if isfinite(r) and r >= 0.0 else default

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_required": list(self.scene_required),
            "ui_element": self.ui_element,
            "world_position": self.world_position,
            "precondition": self.precondition,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "IntentTarget":
        d = d or {}
        return cls(
            scene_required=tuple(d.get("scene_required") or ()),
            ui_element=d.get("ui_element"),
            world_position=d.get("world_position"),
            precondition=d.get("precondition"),
        )


@dataclass(frozen=True)
class IntentConstraints:
    """安全约束。"""

    max_attempts: int = 1
    requires_window_focus: bool = True
    forbid_during: tuple[str, ...] = ()
    cooldown_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "requires_window_focus": self.requires_window_focus,
            "forbid_during": list(self.forbid_during),
            "cooldown_ms": self.cooldown_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "IntentConstraints":
        d = d or {}
        return cls(
            max_attempts=int(d.get("max_attempts", 1)),
            requires_window_focus=bool(d.get("requires_window_focus", True)),
            forbid_during=tuple(d.get("forbid_during") or ()),
            cooldown_ms=int(d.get("cooldown_ms", 0)),
        )


@dataclass(frozen=True)
class IntentPacket:
    """决策层输出：要执行的一个动作。"""

    action: ActionType
    action_kind: ActionKind | None = None
    params: dict[str, Any] = field(default_factory=dict)
    target: IntentTarget = field(default_factory=IntentTarget)
    constraints: IntentConstraints = field(default_factory=IntentConstraints)

    intent_id: str = ""
    issued_at_ms: int = 0
    state_version: int = 0
    ttl_seconds: int = 10
    priority: int = 50
    source: Literal["LOCAL", "CLOUD"] = "LOCAL"
    signature: str | None = None
    schema_version: str = "1.0"

    # -- 构造 ------------------------------------------------------------

    @classmethod
    def make(
        cls,
        action: ActionType | str,
        *,
        params: dict[str, Any] | None = None,
        state_version: int = 0,
        ttl_seconds: int = 10,
        priority: int = 50,
        scene_required: tuple[str, ...] = (),
        source: Literal["LOCAL", "CLOUD"] = "LOCAL",
        constraints: dict[str, Any] | None = None,
        target: IntentTarget | None = None,
    ) -> "IntentPacket":
        """本地构造。``action`` 传非法字符串会**抛异常**（响亮失败）。"""
        act = ActionType(action)  # ValueError 即为响亮失败
        return cls(
            action=act,
            action_kind=act.kind,
            params=dict(params or {}),
            target=target or IntentTarget(scene_required=scene_required),
            constraints=IntentConstraints.from_dict(constraints),
            intent_id=f"int_{uuid.uuid4().hex[:12]}",
            issued_at_ms=int(time.time() * 1000),
            state_version=state_version,
            ttl_seconds=ttl_seconds,
            priority=priority,
            source=source,
        )

    # -- 校验 ------------------------------------------------------------

    def is_valid_for(self, state: StatePacket, now_ms: int) -> tuple[bool, str]:
        """检查意图是否仍适用于当前状态。

        返回 ``(是否有效, 原因)``。**任何一项不通过都拒绝执行**，
        不降级、不猜测。执行层在真正发按键前必须调这个。
        """
        # 1. TTL
        age_ms = now_ms - self.issued_at_ms
        if age_ms > self.ttl_seconds * 1000:
            return False, "EXPIRED"

        # 2. 感知层健康
        if not state.resource.perception_ok:
            return False, "PERCEPTION_DOWN"

        # 3. 状态版本（容忍轻微漂移，状态每帧都变）
        drift = abs(self.state_version - state.state_version)
        if drift > MAX_STATE_VERSION_DRIFT:
            return False, "STATE_STALE"

        # 4. 场景匹配
        if self.target.scene_required and state.scene.id not in self.target.scene_required:
            return False, "SCENE_MISMATCH"
        if state.scene.id in self.constraints.forbid_during:
            return False, "FORBIDDEN_SCENE"

        # 5. 位置可用性（依赖世界坐标的意图）
        if self.target.world_position and (
            state.position is None or not state.position.usable_for_navigation
        ):
            return False, "POSITION_UNAVAILABLE"

        # 6. 签名（云端意图必须有）
        if self.source == "CLOUD" and not self.signature:
            return False, "SIGNATURE_MISSING"

        return True, "OK"

    # -- 序列化 ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intent_id": self.intent_id,
            "issued_at_ms": self.issued_at_ms,
            "state_version": self.state_version,
            "ttl_seconds": self.ttl_seconds,
            "priority": self.priority,
            "source": self.source,
            "signature": self.signature,
            "intent": {
                "action": self.action.value,
                "action_kind": self.action_kind or self.action.kind,
                "params": self.params,
            },
            "target": self.target.to_dict(),
            "constraints": self.constraints.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IntentPacket":
        """反序列化。非法 action 在此**抛异常**（响亮失败，不静默丢弃）。"""
        inner = d.get("intent") or d
        act = ActionType(inner.get("action", ""))  # 非法值 -> ValueError
        return cls(
            action=act,
            action_kind=inner.get("action_kind") or act.kind,
            params=dict(inner.get("params") or {}),
            target=IntentTarget.from_dict(d.get("target")),
            constraints=IntentConstraints.from_dict(d.get("constraints")),
            intent_id=d.get("intent_id", ""),
            issued_at_ms=int(d.get("issued_at_ms", 0)),
            state_version=int(d.get("state_version", 0)),
            ttl_seconds=int(d.get("ttl_seconds", 10)),
            priority=int(d.get("priority", 50)),
            source=d.get("source", "LOCAL"),
            signature=d.get("signature"),
            schema_version=d.get("schema_version", "1.0"),
        )

    @classmethod
    def from_json(cls, s: str) -> "IntentPacket":
        return cls.from_dict(json.loads(s))


__all__ = [
    "MAX_STATE_VERSION_DRIFT",
    "ActionKind",
    "ActionType",
    "IntentConstraints",
    "IntentPacket",
    "IntentTarget",
]
