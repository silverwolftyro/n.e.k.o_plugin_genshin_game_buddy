"""状态包：感知层的结构化输出（上行到决策层 / 云端）。

对应架构文档 5.1。所有结构都是 frozen —— 感知层每帧产生新实例，
下游拿到的一定是**一致快照**，不会看到"场景变了但位置还是旧的"。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------
# 场景枚举（架构文档 5.5）
# --------------------------------------------------------------------------

#: 场景 ID 的**权威定义**。JSON Schema 只写了"见架构文档 5.5"没落枚举，
#: 导致此前唯一的列表散在 actuator/traits.py 里 —— 而 policy 不能反向依赖
#: actuator（动作白名单同理，那个在 ActionType）。所以权威源放在 contracts。
#:
#: actuator/traits.py 的 SCENE_* 常量现在**从这张表派生**，不再各写一份。
#:
#: ``SCENE_WORLD``（§29）：7 类场景分类器在本工作区没有模型，场景判定降级
#: 为 :class:`~autopilot.perception.scene_detector.SceneDetector` 的**二值**
#: 结果 —— ``is_world=True -> WORLD`` / ``is_world=False -> UNKNOWN``。
#: 之前 WORLD 只是 builder 内部的自造字符串，``policy/rule_loader`` 会拒绝
#: 任何引用它的规则；现在它是合法场景 id 了。
SCENE_IDS: dict[str, str] = {
    "SCENE_ANY": "*",               # 通配，仅用于 traits.allowed_scenes
    "SCENE_UNKNOWN": "UNKNOWN",
    "SCENE_WORLD": "WORLD",         # §29：二值场景判定的大世界态（见下）
    "SCENE_MAIN_MENU": "MAIN_MENU",
    "SCENE_COMBAT": "COMBAT",
    "SCENE_LOADING": "LOADING",
    "SCENE_CUTSCENE": "CUTSCENE",
    "SCENE_DIALOG": "DIALOG",
    "SCENE_BAG": "BAG_INVENTORY",
    "SCENE_MAP": "MAP_BIG",
    "SCENE_BATTLE_RESULT": "COMBAT_RESULT",
    "SCENE_GAME_MENU": "GAME_MENU",
    "SCENE_NOTICE": "NOTICE",
    "SCENE_TELEPORT": "TELEPORT_MAP",
}

#: 真实场景 ID 集合（去掉通配 "*"）。**不含 UNKNOWN** 的特殊含义，
#: UNKNOWN 本身就是合法场景，只是决策层对它只允许安全意图。
VALID_SCENE_IDS: frozenset[str] = frozenset(
    v for v in SCENE_IDS.values() if v != "*"
)


def is_valid_scene(scene_id: str) -> bool:
    return scene_id in VALID_SCENE_IDS


# 常用场景的具名常量（与 SCENE_IDS 同源，避免各处散落字面量）
SCENE_UNKNOWN = SCENE_IDS["SCENE_UNKNOWN"]
SCENE_WORLD = SCENE_IDS["SCENE_WORLD"]
SCENE_MAIN_MENU = SCENE_IDS["SCENE_MAIN_MENU"]
SCENE_COMBAT = SCENE_IDS["SCENE_COMBAT"]
SCENE_LOADING = SCENE_IDS["SCENE_LOADING"]
SCENE_CUTSCENE = SCENE_IDS["SCENE_CUTSCENE"]
SCENE_DIALOG = SCENE_IDS["SCENE_DIALOG"]
SCENE_BAG = SCENE_IDS["SCENE_BAG"]
SCENE_MAP = SCENE_IDS["SCENE_MAP"]
SCENE_BATTLE_RESULT = SCENE_IDS["SCENE_BATTLE_RESULT"]
SCENE_GAME_MENU = SCENE_IDS["SCENE_GAME_MENU"]
SCENE_NOTICE = SCENE_IDS["SCENE_NOTICE"]
SCENE_TELEPORT = SCENE_IDS["SCENE_TELEPORT"]


# --------------------------------------------------------------------------
# 子结构
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneInfo:
    """当前所在界面。

    ``id`` 取值见 :data:`SCENE_IDS`；UNKNOWN 是**合法的空状态**，
    不是错误（可能只是没识别出来）。决策层对 UNKNOWN 只允许安全意图。
    """

    id: str
    confidence: float
    evidence: tuple[str, ...] = ()
    entered_ms: int | None = None
    raw_label: str = ""

    @property
    def is_unknown(self) -> bool:
        return self.id == "UNKNOWN"


@dataclass(frozen=True)
class PositionInfo:
    """角色位置。

    **定位丢失时整个 position 为 None，禁止用运动预测填补** —— 那会让上层
    在没有实际观测的情况下继续动作，误差累积且无法自我修正（原则 1）。

    quality 三态：
        GOOD      局部精匹配成功且分数达标
        DEGRADED  全局匹配成功 / 分数偏低，依赖位置的意图需额外确认
        LOST      连续 N 帧无匹配，依赖位置的意图直接拒绝（原则 2）
    """

    quality: Literal["GOOD", "DEGRADED", "LOST"]
    zone: str | None = None
    x: float | None = None
    y: float | None = None
    heading_deg: float | None = None
    confidence: float = 0.0
    source: str = "none"

    @classmethod
    def lost(cls) -> "PositionInfo":
        return cls(quality="LOST", source="none")

    @property
    def usable_for_navigation(self) -> bool:
        """是否可用于导航类意图。"""
        return self.quality in ("GOOD", "DEGRADED") and self.x is not None


@dataclass(frozen=True)
class PartyMember:
    """队伍成员。

    ``cd_ready`` 的值是三态：True=确认就绪 / False=确认未就绪 / None=UNKNOWN。
    **None 不得压缩成 bool** —— 压成 False 会"永久卡住"，压成 True 会"白放技能"。
    """

    slot: int
    char_id: str
    name: str = ""
    active: bool = False
    cd_ready: dict[str, bool | None] = field(default_factory=dict)

    def cd(self, key: str) -> bool | None:
        """取某技能就绪态；未记录返回 None（UNKNOWN）。"""
        return self.cd_ready.get(key)

    @property
    def cd_known(self) -> bool:
        """是否有任何技能的 CD 是**已知**的。全 None 时为 False。"""
        return any(v is not None for v in self.cd_ready.values())


@dataclass(frozen=True)
class PlayerInfo:
    """玩家状态。

    hp_ratio / energy 计划用**几何投影**（血条长度比例）而非 OCR，
    见 perception 模块设计。这样无文字识别错误、无延迟、对光照免疫。
    """

    level: int | None = None
    hp_ratio: float | None = None
    hp_source: str = "none"
    energy: int | None = None
    energy_max: int | None = None
    energy_state: Literal["CHARGED", "CHARGING", "UNKNOWN"] = "UNKNOWN"
    party: tuple[PartyMember, ...] = ()
    resin: int | None = None
    resin_cap: int | None = None

    @property
    def active_member(self) -> PartyMember | None:
        for m in self.party:
            if m.active:
                return m
        return None

    @property
    def energy_ratio(self) -> float | None:
        if self.energy is None or not self.energy_max:
            return None
        return min(1.0, max(0.0, self.energy / self.energy_max))


@dataclass(frozen=True)
class ResourceInfo:
    """资源与健康度。

    全部字段必填 —— 这些数据既用于本地告警，也用于云端监控与计费。
    ``perception_ok=False`` 时决策层**不产出任何意图**（原则 1）。
    """

    fps: float = 0.0
    tick_ms: int = 0
    perception_ok: bool = True
    ocr_available: bool = True
    ocr_fail_streak: int = 0
    yolo_ms: float = 0.0
    template_ms: float = 0.0
    locator_ms: float = 0.0


# --------------------------------------------------------------------------
# 状态包
# --------------------------------------------------------------------------

#: 上行到云端时省略的字段（体积/隐私，见架构文档 9.2 与 11.4）
_COMPACT_OMIT_PLAYER_FIELDS = ("resin", "resin_cap")


@dataclass(frozen=True)
class StatePacket:
    """感知层输出：一帧的结构化游戏状态。"""

    scene: SceneInfo
    player: PlayerInfo = field(default_factory=PlayerInfo)
    position: PositionInfo | None = None
    resource: ResourceInfo = field(default_factory=ResourceInfo)

    schema_version: str = "1.0"
    game_id: str = "genshin_impact"
    timestamp_ms: int = 0
    state_version: int = 0

    # -- 构造 ------------------------------------------------------------

    @classmethod
    def unknown(
        cls,
        *,
        game_id: str = "genshin_impact",
        state_version: int = 0,
        reason: str = "perception_failed",
    ) -> "StatePacket":
        """感知失败时的状态包。

        **响亮失败原则的落地**：不返回 None、不给假的空状态，
        而是给一个显式标记 perception_ok=False 的包，让上层知道发生了什么。
        """
        now = _now_ms()
        return cls(
            scene=SceneInfo(id="UNKNOWN", confidence=0.0, evidence=(f"reason={reason}",)),
            player=PlayerInfo(),
            position=None,
            resource=ResourceInfo(perception_ok=False),
            timestamp_ms=now,
            state_version=state_version,
            game_id=game_id,
        )

    # -- 序列化 ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StatePacket":
        scene_d = data.get("scene") or {}
        scene = SceneInfo(
            id=scene_d.get("id", "UNKNOWN"),
            confidence=float(scene_d.get("confidence", 0.0)),
            evidence=tuple(scene_d.get("evidence") or ()),
            entered_ms=scene_d.get("entered_ms"),
            raw_label=scene_d.get("raw_label", ""),
        )

        pos_d = data.get("position")
        position = None
        if isinstance(pos_d, dict):
            position = PositionInfo(
                quality=pos_d.get("quality", "LOST"),
                zone=pos_d.get("zone"),
                x=pos_d.get("x"),
                y=pos_d.get("y"),
                heading_deg=pos_d.get("heading_deg"),
                confidence=float(pos_d.get("confidence", 0.0)),
                source=pos_d.get("source", "none"),
            )

        pl_d = data.get("player") or {}
        party: list[PartyMember] = []
        for m in pl_d.get("party") or ():
            party.append(
                PartyMember(
                    slot=int(m.get("slot", 0)),
                    char_id=m.get("char_id", ""),
                    name=m.get("name", ""),
                    active=bool(m.get("active", False)),
                    cd_ready=dict(m.get("cd_ready") or {}),
                )
            )
        player = PlayerInfo(
            level=pl_d.get("level"),
            hp_ratio=pl_d.get("hp_ratio"),
            hp_source=pl_d.get("hp_source", "none"),
            energy=pl_d.get("energy"),
            energy_max=pl_d.get("energy_max"),
            energy_state=pl_d.get("energy_state", "UNKNOWN"),
            party=tuple(party),
            resin=pl_d.get("resin"),
            resin_cap=pl_d.get("resin_cap"),
        )

        r_d = data.get("resource") or {}
        resource = ResourceInfo(
            fps=float(r_d.get("fps", 0.0)),
            tick_ms=int(r_d.get("tick_ms", 0)),
            perception_ok=bool(r_d.get("perception_ok", True)),
            ocr_available=bool(r_d.get("ocr_available", True)),
            ocr_fail_streak=int(r_d.get("ocr_fail_streak", 0)),
            yolo_ms=float(r_d.get("yolo_ms", 0.0)),
            template_ms=float(r_d.get("template_ms", 0.0)),
            locator_ms=float(r_d.get("locator_ms", 0.0)),
        )

        return cls(
            scene=scene,
            player=player,
            position=position,
            resource=resource,
            schema_version=data.get("schema_version", "1.0"),
            game_id=data.get("game_id", "genshin_impact"),
            timestamp_ms=int(data.get("timestamp_ms", 0)),
            state_version=int(data.get("state_version", 0)),
        )

    @classmethod
    def from_json(cls, s: str) -> "StatePacket":
        return cls.from_dict(json.loads(s))

    # -- 上行精简版 ------------------------------------------------------

    def to_compact_dict(self) -> dict[str, Any]:
        """上行到云端的精简版。

        裁剪策略（架构文档 9.2）：
          · 省略 resin / resin_cap（与决策无关，且属账号资源信息）
          · party 只保留 active 的那一个（云端不需要全队，除非用户主动请求）
          · resource 只保留健康度，不保留各阶段耗时
        **上传前必须确认不含 UID / 账号名**（合规，见架构文档 11.4）。
        """
        d = self.to_dict()
        player = d.get("player") or {}
        for k in _COMPACT_OMIT_PLAYER_FIELDS:
            player.pop(k, None)
        party = player.get("party") or []
        active = [m for m in party if m.get("active")]
        if active and len(party) > 1:
            player["party"] = active
        d["player"] = player
        d["resource"] = {
            "fps": self.resource.fps,
            "perception_ok": self.resource.perception_ok,
            "ocr_available": self.resource.ocr_available,
        }
        return d


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_jsonable(obj: Any) -> Any:
    """dataclasses.asdict 之后的类型归一（tuple→list，Enum→value）。"""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "value") and hasattr(obj, "name"):  # Enum
        return obj.value
    return obj


__all__ = [
    "PlayerInfo",
    "PartyMember",
    "PositionInfo",
    "ResourceInfo",
    "SCENE_WORLD",
    "SceneInfo",
    "StatePacket",
    "VALID_SCENE_IDS",
    "is_valid_scene",
]
