"""DSL 求值上下文：StatePacket -> 规则里能调用的函数集。

**为什么要有这一层**
------------------
规则（TOML 里的 `when = "hp() < 0.35"`）不应该直接读 StatePacket：

* StatePacket 是**协议**，字段会随识别方案演进（今天 `player.hp_ratio`，
  明天可能换成 `vitals.hp_ratio_from_bar`）。规则跟着改 = 全部规则重写。
* StatePacket 有嵌套（`player.party[0].cd_ready["burst"]`），DSL 里没法表达。
* 三态语义（True / False / **UNKNOWN**）需要在边界上统一处理，
  让每个字段各自决定「拿不到时怎么办」。

所以规则只认**函数**：`hp()` / `q_ready()` / `has_enemy()`。
StatePacket 怎么变，改这一个文件即可（原则 8：中间层吸收变化）。

**三态与 Unknown 不乐观**
------------------------
观测不到的量一律返回 :class:`UnknownValue`，**不返回 None 也不返回 0**：

* 返回 0 → 规则会认为「血量是 0」→ 触发急救 → 可能乱按
* 返回 None → 表达式里 `None < 0.35` 直接 TypeError → 规则崩掉
* 返回 Unknown → 所有比较恒为 False → 规则**不触发**，但会记进
  :attr:`PolicyContext.unknown_hits`，`rule_test.py` 能解释「为什么没命中」

「不触发」是安全的默认动作；「不解释」才是不安全的。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..contracts.state_packet import StatePacket
from ..state import IntentRecord

# --------------------------------------------------------------------------
# 三态
# --------------------------------------------------------------------------


class UnknownValue:
    """「观测不到」的哨兵。**所有比较恒为 False**，真值恒为 False。

    用 Null Object 而不是 None，因为规则里写的是普通 Python 表达式
    （``hp() < 0.35 and q_ready()``），None 会在比较处抛 TypeError，
    把「数据缺失」变成「规则崩溃」—— 那是把可诊断的问题变成难诊断的问题。

    比较返回 False（而非 True）是有意的方向性选择：数据缺失时**不动作**
    比**乱动作**安全得多。
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str = "未知") -> None:
        self.reason = reason

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"UNKNOWN({self.reason})"

    # 与 __eq__ 配套必须给 __hash__，否则对象不可哈希
    __hash__ = object.__hash__

    def _always_false(self, other: object) -> bool:
        return False

    __lt__ = __le__ = __gt__ = __ge__ = __eq__ = __ne__ = _always_false


UNKNOWN = UnknownValue()


@dataclass(frozen=True)
class PolicySignals:
    """StatePacket **尚未承载**的识别信号。

    ⚠ 存在的原因（一个真实的契约缺口，不是偷懒）：
    ``StatePacket`` 里**没有** ``has_enemy`` 字段 —— 第 1 期落地的
    契约只包含 scene / position / player / resource 四组。
    而战斗规则最需要的就是「视野里有没有敌人」。

    处理方式有两条路：
      A. 现在就往 StatePacket 加字段（改 Phase 1 契约 + JSON Schema + builder）
      B. 先留这个显式扩展点，等感知层真的能可靠输出 has_enemy 再接

    这里选 B，并**全部默认 None（UNKNOWN）**：宁可规则不触发，
    也不要用「有血量下降就算有敌人」这种猜测去填 —— 猜错一次
    就是战斗机器人冲进人群里乱按（原则 1）。

    接通方式：感知层 build 出 StatePacket 后，
    ``PolicyContext.from_state(state, signals=PolicySignals(has_enemy=True))``。
    """

    has_enemy: bool | None = None
    enemy_count: int | None = None
    #: 当前是否锁定/瞄准了某个敌人（决定普攻是否值得长按）
    target_locked: bool | None = None


# --------------------------------------------------------------------------
# 上下文
# --------------------------------------------------------------------------

#: 拾取标记的冷却（秒）。超过它就认为「那个东西已经拿完了 / 不是它了」。
#:
#: ★ 为什么需要它（2026-10-11 真机事故，run 20261011_180732）
#:   ---------------------------------------------------------------
#:   三条 ``USE_PICKUP`` 规则打开后：154.8 秒里 **151 次真执行** +
#:   603 次被 min_gap 挡掉。根因不是 min_gap 不灵，而是
#:   ``has_nearby_chest()`` 在宝箱被打开后**不消失**（检测框没刷新，
#:   或者箱子的视觉特征本来就跟「已开过」的一样），规则于是每 tick
#:   重新瞄上同一个箱子。
#:   min_gap 只能把它从 4.9 次/秒压到 1 次/秒，**压不掉重复**。
PICK_MARK_COOLDOWN_S: float = 8.0

#: 拾取标记的作用半径（像素，720 基准坐标系 —— 与 ``screen_pos`` 同系）。
#: 取得比判定框大一点，好让「同一堆掉落物」整堆被压住，而不是只压住
#: 正中间那个框。
PICK_MARK_RADIUS_PX: float = 120.0


@dataclass(frozen=True)
class PickMark:
    """「这个位置刚刚已经按过 F 了」的标记。"""

    group: str
    x: float
    y: float
    ts_ms: int


#: 实体 ``screen_pos`` 所在的基准画幅（720p）。
#:
#: ★ 刻意**不从 world_entities 导入** ``BASE_SIZE``：那个模块会连带拉进
#:   ``yolo`` / ONNX 依赖，而 policy 层不该为此背上感知层的依赖。
#:   两处都是 720 基准，改动时需要同步 —— 这里显式写出数值而不是
#:   默默假设一致。
_BASE_SIZE: tuple[int, int] = (1280, 720)


def _entity_screen_pos(entity) -> tuple[float, float] | None:
    """实体的屏幕坐标（720 基准）。读不到返回 None。"""
    for attr in ("screen_pos", "position"):
        v = getattr(entity, attr, None)
        if isinstance(v, (tuple, list)) and len(v) >= 2:
            try:
                return float(v[0]), float(v[1])
            except (TypeError, ValueError):
                return None
    return None


def _dist_px(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


@dataclass(frozen=True)
class PolicyContext:
    """一帧的 DSL 求值上下文。

    只依赖 ``contracts`` + ``state``。构造完成后**只读**。
    """

    state: StatePacket
    now_ms: int
    signals: PolicySignals = field(default_factory=PolicySignals)
    recent_states: tuple[StatePacket, ...] = ()
    intent_records: tuple[IntentRecord, ...] = ()
    #: 本帧检出的世界实体（F2 方案：**不进 StatePacket**，由 builder
    #: 侧信道传入）。**``None`` 表示「本帧没有实体信息」**，与
    #: ``[]``（跑了检测但确实没有）**语义不同** —— 前者返回 UNKNOWN，
    #: 后者返回 False。给空 tuple 也可以，等价于「没跑检测」。
    world_entities: tuple | None = None

    #: 「刚刚已经按过 F 的位置」标记。由 :class:`LocalPolicyTree` 持有并在
    #: 每轮决策时传进来 —— **不能存在 PolicyContext 自己的实例变量里**，
    #: 因为本类每个 tick 都会被 :meth:`from_state` 重建一次，实例变量活不过
    #: 一轮。跨 tick 的记忆必须由长生命周期的 PolicyTree 保管。
    pick_marks: tuple[PickMark, ...] = ()

    #: 本次求值里出现过的「未知」字段，形如 ``"hp(): hp_ratio 未观测"``。
    #: 供 rule_test / replay 解释「为什么没命中」。不是静默吞掉。
    unknown_hits: list[str] = field(default_factory=list)

    # -- 构造 ------------------------------------------------------------
    @classmethod
    def from_state(
        cls,
        state: StatePacket,
        *,
        now_ms: int | None = None,
        signals: PolicySignals | None = None,
        recent_states: Iterable[StatePacket] = (),
        intent_records: Iterable[IntentRecord] = (),
        world_entities: tuple | None = None,
        pick_marks: tuple[PickMark, ...] = (),
    ) -> "PolicyContext":
        """从一帧状态构造。

        ``recent_states`` / ``intent_records`` 是**可选**的历史输入。
        不给时依赖它们的函数（``count`` / ``last_exec``）返回 UNKNOWN
        而不是 0 / 0 —— 因为「没有历史」和「历史为 0」是两回事。

        ``world_entities`` 同理：**None = 没跑实体检测**（→ UNKNOWN），
        与空 tuple（跑了但确实没有 → False）语义不同。
        """
        return cls(
            state=state,
            now_ms=now_ms if now_ms is not None else int(time.time() * 1000),
            signals=signals or PolicySignals(),
            recent_states=tuple(recent_states),
            intent_records=tuple(intent_records),
            world_entities=world_entities,
            pick_marks=tuple(pick_marks),
        )

    # -- 内部 ------------------------------------------------------------
    def _unk(self, func: str, reason: str) -> UnknownValue:
        self.unknown_hits.append(f"{func}(): {reason}")
        return UnknownValue(reason)

    # ==================================================================
    # DSL 函数集。名字必须与 rules/*.toml 里写的一致。
    # ==================================================================

    # -- 时间 ------------------------------------------------------------
    def t(self) -> int:
        """当前时间戳（毫秒）。"""
        return self.now_ms

    def since(self) -> int | UnknownValue:
        """当前场景已停留毫秒数。"""
        entered = self.state.scene.entered_ms
        if entered is None:
            return self._unk("since", "scene.entered_ms 未设置")
        return max(0, self.now_ms - entered)

    def count(self, scene_id: str) -> int | UnknownValue:
        """历史帧里处于该场景的数量。**没有历史时是 UNKNOWN，不是 0**。"""
        if not self.recent_states:
            return self._unk("count", "未提供历史帧")
        return sum(1 for s in self.recent_states if s.scene.id == scene_id)

    def last_exec(self, action: str) -> int | UnknownValue:
        """距上次执行该动作过了多少毫秒。**从未执行过是 UNKNOWN，不是 0**。

        ⚠ 函数名用下划线：TOML 里写 ``last_exec()``。Python 标识符不允许
        连字符，写 ``last-exec()`` 会被 ast 解析成减法（last - exec()）。
        """
        if not self.intent_records:
            return self._unk("last_exec", "未提供意图历史")
        last = None
        for rec in self.intent_records:
            if rec.intent.action.value != action:
                continue
            t = rec.decided_ms or rec.issued_ms
            if last is None or t > last:
                last = t
        if last is None:
            return self._unk("last_exec", f"动作 {action} 从未执行过")
        return max(0, self.now_ms - last)

    # -- 场景 ------------------------------------------------------------
    def scene(self) -> str:
        """当前场景 ID。永远可观测（识别不出时是 "UNKNOWN"，不是空）。"""
        return self.state.scene.id

    def scene_confidence(self) -> float:
        """场景识别置信度。"""
        return self.state.scene.confidence

    # -- 角色状态 --------------------------------------------------------
    def hp(self) -> float | UnknownValue:
        """血量比例 0..1。"""
        v = self.state.player.hp_ratio
        if v is None:
            return self._unk("hp", f"hp_ratio 未观测 (source={self.state.player.hp_source})")
        return v

    def energy(self) -> float | UnknownValue:
        """元素充能比例 0..1。"""
        v = self.state.player.energy_ratio
        if v is None:
            return self._unk("energy", "energy / energy_max 未观测")
        return v

    def resin(self) -> int | UnknownValue:
        """原粹树脂。"""
        v = self.state.player.resin
        if v is None:
            return self._unk("resin", "resin 未观测")
        return v

    # -- 敌人 ------------------------------------------------------------
    def has_enemy(self) -> bool | UnknownValue:
        """视野内是否有敌人。

        ⚠ 当前依赖 :class:`PolicySignals`（StatePacket 尚无该字段）。
        未提供时是 UNKNOWN —— 规则因此**不会触发**。
        """
        v = self.signals.has_enemy
        if v is None:
            return self._unk("has_enemy", "感知层尚未提供该信号（见 PolicySignals）")
        return v

    def enemy_count(self) -> int | UnknownValue:
        """视野内敌人数。

        ★ 2026-10-11：加了一条**兜底**
          -------------------------------------------
          原来只读 :attr:`PolicySignals.enemy_count`，而感知层从没填过它，
          所以它恒 UNKNOWN —— 于是「附近有 3 个以上敌人」这类规则**一条都
          写不出来**（这是战斗细分的第一条）。

          现在：``PolicySignals`` 有值就用它（那是上层的权威口径）；
          没有就退回 :meth:`enemy_bar_count` —— 世界实体检测**已经在数敌方
          血条**了，只是没人把它接到这个信号上。零新增感知能力。

          ⚠️ 两条仍都拿不到时保持 UNKNOWN，**不退化成 0**：
          「不知道有没有敌人」和「确实没有敌人」必须分开，否则规则会在
          信息缺失时按「周围安全」行动。
        """
        v = self.signals.enemy_count
        if v is not None:
            return v
        return self.enemy_bar_count()

    def target_locked(self) -> bool | UnknownValue:
        """是否已锁定目标。"""
        v = self.signals.target_locked
        if v is None:
            return self._unk("target_locked", "感知层尚未提供该信号")
        return v

    # -- 队伍 / CD -------------------------------------------------------
    def onfield(self, name: str) -> bool | UnknownValue:
        """指定角色是否在场上。"""
        m = self.state.player.active_member
        if m is None:
            return self._unk("onfield", "队伍没有 active 成员（party 为空？）")
        if name and name not in (m.name, m.char_id):
            return False
        return bool(m.active)

    def _active_cd(self, key: str) -> bool | UnknownValue:
        m = self.state.player.active_member
        if m is None:
            return self._unk("cd", "队伍没有 active 成员")
        if not m.cd_ready:
            return self._unk("cd", f"未记录任何技能 CD（{m.name}）")
        v = m.cd_ready.get(key)
        if v is None:
            return self._unk("cd", f"{m.name} 的 {key} 状态未知")
        return bool(v)

    def q_ready(self) -> bool | UnknownValue:
        """当前角色的**大招**是否就绪。"""
        return self._active_cd("burst")

    def e_ready(self) -> bool | UnknownValue:
        """当前角色的**元素战技**是否就绪。"""
        return self._active_cd("skill")

    def slot(self, n: int) -> bool | UnknownValue:
        """第 n 个槽位是否有角色。"""
        for m in self.state.player.party:
            if m.slot == n:
                return True
        return False

    # -- 位置 ------------------------------------------------------------
    def pos_quality(self) -> str | UnknownValue:
        """定位质量："GOOD" / "DEGRADED" / "LOST"。

        ⚠ 与 traits 的命名不同：这里是**大写**，因为它直接对应
        `PositionInfo.quality` 的字面量，便于规则直接比字符串。
        """
        p = self.state.position
        if p is None:
            return self._unk("pos_quality", "无位置信息（position=None）")
        return p.quality

    def pos_usable(self) -> bool | UnknownValue:
        """定位是否可用于导航（GOOD / DEGRADED 且有坐标）。"""
        p = self.state.position
        if p is None:
            return self._unk("pos_usable", "无位置信息")
        return p.usable_for_navigation

    # -- 世界实体（F2：不进契约，走侧信道）---------------------------
    def _entities_in_group(self, group: str, func: str):
        """按业务分组筛实体。没跑检测时返回 UNKNOWN 而不是 False。

        ★ 会滤掉「最近已经对它用过 USE_PICKUP」的实体（见 :class:`PickMark`）。
          真机事故（run 20261011_180732）：宝箱被打开后
          ``has_nearby_chest()`` **不消失**，规则每 tick 重发同一个箱子，
          154.8 秒里按了 151 次 F。min_gap 只是把它从 4.9 次/秒压到
          1 次/秒，**治不了「同一个东西被反复瞄上」**。
        """
        if self.world_entities is None:
            return self._unk(func, "本帧无世界实体信息（未注入检测器或场景不允许）")
        out = []
        for e in self.world_entities:
            kind = getattr(e, "kind", None)
            if kind is None:
                continue
            if getattr(e, "group", kind) != group:
                continue
            if self._is_marked_picked(e, group, func):
                continue
            out.append(e)
        return out

    def _is_marked_picked(self, entity, group: str, func: str) -> bool:
        """这个实体是不是「刚刚已经对它按过 F 了」。"""
        if not self.pick_marks:
            return False
        pos = _entity_screen_pos(entity)
        if pos is None:
            return False          # 位置读不到就**不过滤** —— 宁可多按，
        return any(                # 也不要因为坐标缺失而永久压制这一类目标
            m.group == group
            and (self.now_ms - m.ts_ms) < int(PICK_MARK_COOLDOWN_S * 1000)
            and _dist_px(pos, (m.x, m.y)) <= PICK_MARK_RADIUS_PX
            for m in self.pick_marks
        )

    def has_nearby_enemy(self) -> bool | UnknownValue:
        """附近是否有敌人。

        ★ **两条判据**（§36）——
        ------------------------------------------------------------------
        **A. ``enemy_identify``（group=enemy）** —— 玩家主动锁定目标后才
        出现的青色标识。

        **B. 敌人血条（group=combat / kind=health_bar）且 y < 0.5**。

        为什么必须有 B：三次真机 smoke 里 ``enemy_identify`` **一次都没检出**
        （health_bar 分别检出 6 / 5 / 10 次）。用户确认 ——
        ``enemy_identify`` 是"锁定目标"标记，不按 Tab 就不出现。
        **只认 A 的话，三条战斗规则永远不会触发。**

        为什么 y < 0.5 能区分敌我：bgi_world 会把**玩家自己的血条**也标成
        ``health_bar``，但它在屏幕最底部。实测敌人血条的 bbox 左上角 y：

        =============  ==========================================
        smoke 次     y 值
        =============  ==========================================
        1            0.319
        2            0.161
        3            0.225 / 0.211
        =============  ==========================================

        全部 < 0.5；而自己的血条在 y > 0.9。**0.5 是两者之间的空档带**，
        不是实测出来的分界 —— 只有 4 个敌人数据 + 若干自己的血条样本。
        ⚠️ 相机角度变化大（角色在屏幕下方 vs 居中）时可能失效。
        """
        if self.world_entities is None:
            return self._unk(
                "has_nearby_enemy",
                "本帧无世界实体信息（未注入检测器或场景不允许）")

        # 判据本体在 :func:`is_enemy_evidence`（全仓唯一出处）。
        # 这里只负责「有没有」——三态里的 UNKNOWN 由上面判掉。
        return any(is_enemy_evidence(e) for e in self.world_entities)

    def has_nearby_chest(self) -> bool | UnknownValue:
        """附近是否有宝箱（``chest`` + ``locked_chest``）。"""
        ents = self._entities_in_group("chest", "has_nearby_chest")
        if isinstance(ents, UnknownValue):
            return ents
        return bool(ents)

    def has_nearby_ore(self) -> bool | UnknownValue:
        """附近是否有矿（``ore`` + ``unmined_ore``）。"""
        ents = self._entities_in_group("ore", "has_nearby_ore")
        if isinstance(ents, UnknownValue):
            return ents
        return bool(ents)

    def has_nearby_drops(self) -> bool | UnknownValue:
        """附近是否有掉落物（``drops``）。

        2026-10-11 补：``KIND_GROUPS`` 里``drops`` 分组一直登记着
        （``world_entities.KIND_GROUPS``），但 DSL 侧少一个函数，导致规则
        写不出「捡地上的掉落」——而这是 WORLD 场景里最该做的自动动作之一。

        语义与 :meth:`has_nearby_chest` / :meth:`has_nearby_ore` 完全一致：
        无实体信息时返回 :class:`UnknownValue`（让规则**不触发**），
        而不是返回 False —— 后者会让「检测器没注入」和「附近确实没有」
        变成同一种信号，规则就会在信息缺失时贸然行动。
        """
        ents = self._entities_in_group("drops", "has_nearby_drops")
        if isinstance(ents, UnknownValue):
            return ents
        return bool(ents)

    def nearby_entity_count(self) -> int | UnknownValue:
        """本帧实体总数。用于调试与规则调试器。"""
        if self.world_entities is None:
            return self._unk("nearby_entity_count", "本帧无世界实体信息")
        return len(self.world_entities)

    # ------------------------------------------------------------------
    # 2026-10-11 新增：可从现有实体检测**直接推导**的信号
    # ------------------------------------------------------------------
    # ★ 为什么专门加这一组
    #   全量探测（见 recognition/notes/session_20261011…）显示今天真正能出值
    #   的 DSL 函数只剩 11 个，其中 4 个是 has_nearby_*，**没有计数、没有距离、
    #   没有上锁区分**。于是「单体 vs 群体」「够不够近」「这个箱子锁了没」
    #   这三类判断全都写不出来 —— 而它们恰好是规则细分里最值钱的几个。
    #
    #   这几个函数**不需要任何新的感知能力**：世界实体检测已经在给出
    #   kind/group/conf/bbox/screen_pos，只是没人把它们用起来。

    def nearby_count(self, group: str) -> int | UnknownValue:
        """某分组的实体个数。``nearby_count('chest')``。

        ★ 这补上了一个关键空缺：``enemy_count()`` 返回 UNKNOWN
          （PolicySignals 里「感知层尚未提供该信号」），所以现在**写不出**
          「附近有 3 个以上敌人」这种规则 —— 而它是战斗细分的第一条。
        """
        ents = self._entities_in_group(group, "nearby_count")
        if isinstance(ents, UnknownValue):
            return ents
        return len(ents)

    def enemy_bar_count(self) -> int | UnknownValue:
        """画面里**敌方血条**的数量 —— 当作「附近有几个敌人」。

        ★ 为什么不用 ``enemy_identify``
          那是玩家主动锁定目标后才出现的青色标识，数量通常是 1；
          血条（``health_bar``）才是**成群出现**的那个。真机扫描里
          health_bar 检出 249 次 vs enemy_identify 344 次，但前者天然
          与敌人数成正比。
        """
        if self.world_entities is None:
            return self._unk("enemy_bar_count", "本帧无世界实体信息")
        n = 0
        for e in self.world_entities:
            group = getattr(e, "group", getattr(e, "kind", ""))
            if group == "combat":
                bb = getattr(e, "bbox", None)
                # 与 has_nearby_enemy 同口径：只看上半屏，且跳过读不到位置的
                if not bb or len(bb) < 2:
                    continue
                if float(bb[1]) < ENEMY_HEALTH_BAR_Y_MAX:
                    n += 1
        return n

    def has_locked_chest(self) -> bool | UnknownValue:
        """附近有没有**上锁**的宝箱。

        ``chest`` 与 ``locked_chest`` 在 :data:`KIND_GROUPS` 里同属
        ``chest`` 组，所以 ``has_nearby_chest()`` 分不出它们 —— 而
        「上锁的箱子需要钥匙」是完全不同的决策。
        """
        if self.world_entities is None:
            return self._unk("has_locked_chest", "本帧无世界实体信息")
        return any(getattr(e, "kind", None) == "locked_chest"
                   for e in self.world_entities)

    def nearest_group_px(self, group: str) -> float | UnknownValue:
        """某分组里**最近**的那个实体离画面中心多少像素（720 基准）。

        读不到任何坐标时返回 ``inf`` —— 「不知道多远」和「非常近」必须
        分得开，否则规则会在信息缺失时贸然行动。
        """
        if self.world_entities is None:
            return self._unk("nearest_group_px", "本帧无世界实体信息")
        cx, cy = _BASE_SIZE[0] / 2.0, _BASE_SIZE[1] / 2.0
        best = float("inf")
        for e in self.world_entities:
            g = getattr(e, "group", getattr(e, "kind", ""))
            if g != group:
                continue
            pos = _entity_screen_pos(e)
            if pos is None:
                continue
            best = min(best, _dist_px(pos, (cx, cy)))
        return best
        if self.world_entities is None:
            return self._unk("nearby_entity_count", "本帧无世界实体信息")
        return len(self.world_entities)

    # -- 资源健康 --------------------------------------------------------
    def ocr_ok(self) -> bool | UnknownValue:
        """OCR 是否健康：可用且当前无连续失败。"""
        r = self.state.resource
        if not r.ocr_available:
            return False
        if r.ocr_fail_streak > 0:
            return False
        return True

    def perception_ok(self) -> bool:
        """感知层关键路径是否全通。**False 时决策层不应产出任何意图**。"""
        return bool(self.state.resource.perception_ok)

    def fps(self) -> float:
        return float(self.state.resource.fps)

    # -- 工具函数（min/max 由 DSL 命名空间提供）------------------------
    def namespace(self) -> dict[str, Callable[..., Any]]:
        """给 DSL 求值器的名字表（绑定到本实例）。

        刻意**不含** ``min`` / ``max`` —— 它们由 dsl.py 注入，因为需要
        对 UnknownValue 做特殊处理（min(UNKNOWN, 0.5) 不该抛 TypeError）。

        函数清单的唯一来源是模块级 ``_FUNCTION_TABLE``，dsl.py 校验
        白名单时读的是同一份 —— 两边不会各写一份然后漂移。
        """
        return {name: getattr(self, name) for name in _FUNCTION_TABLE}


def _build_function_table() -> dict[str, Callable[..., Any]]:
    """DSL 可调用的函数清单（未绑定）。

    用**未绑定方法**，``namespace()`` 再用 ``getattr(self, ...)`` 绑定。
    这样「有哪些函数」这件事在模块级就确定，dsl.py 不用造一个假 context
    去反射。
    """
    return {
        "t": PolicyContext.t,
        "since": PolicyContext.since,
        "count": PolicyContext.count,
        "last_exec": PolicyContext.last_exec,
        "scene": PolicyContext.scene,
        "scene_confidence": PolicyContext.scene_confidence,
        "hp": PolicyContext.hp,
        "energy": PolicyContext.energy,
        "resin": PolicyContext.resin,
        "has_enemy": PolicyContext.has_enemy,
        "enemy_count": PolicyContext.enemy_count,
        "target_locked": PolicyContext.target_locked,
        "onfield": PolicyContext.onfield,
        "q_ready": PolicyContext.q_ready,
        "e_ready": PolicyContext.e_ready,
        "slot": PolicyContext.slot,
        "pos_quality": PolicyContext.pos_quality,
        "pos_usable": PolicyContext.pos_usable,
        "ocr_ok": PolicyContext.ocr_ok,
        "perception_ok": PolicyContext.perception_ok,
        "fps": PolicyContext.fps,
        # -- 世界实体（F2 侧信道；§29 注册进 DSL）----------------------
        # ⚠️ **别和上面的 ``has_enemy()`` 搞混**：
        #    ``has_enemy()`` 读的是 :class:`PolicySignals` 里**外部手动声明**的
        #    信号，默认 None -> UNKNOWN —— 「没人告诉我有没敌人」。
        #    ``has_nearby_enemy()`` 读的是 **bgi_world 检测器**的实际检出 ——
        #    「我看见了敌人标识」。两个来源不同，规则里别混用。
        "has_nearby_enemy": PolicyContext.has_nearby_enemy,
        "has_nearby_chest": PolicyContext.has_nearby_chest,
        "has_nearby_ore": PolicyContext.has_nearby_ore,
        "has_nearby_drops": PolicyContext.has_nearby_drops,
        "nearby_entity_count": PolicyContext.nearby_entity_count,
        # -- 2026-10-11：从实体检测直接推导的信号（见方法 docstring）------
        "nearby_count": PolicyContext.nearby_count,
        "enemy_bar_count": PolicyContext.enemy_bar_count,
        "has_locked_chest": PolicyContext.has_locked_chest,
        "nearest_group_px": PolicyContext.nearest_group_px,
    }


_FUNCTION_TABLE = _build_function_table()

#: DSL 允许调用的函数名（权威清单）。
FUNCTION_NAMES: frozenset[str] = frozenset(_FUNCTION_TABLE)


#: 敌人血条的 y 坐标上限（``WorldEntity.bbox`` 的**左上角** y，归一化）。
#:
#: ``y < ENEMY_HEALTH_BAR_Y_MAX`` 的 ``health_bar`` 视为**敌人**血条；
#: 玩家自己的血条在屏幕最底部（y > 0.9）被排除。
#:
#: ⚠️ **依据是 4 个实测点**，不是统计出来的分界：
#:    三次真机 smoke 的敌人血条 y = 0.319 / 0.161 / 0.225 / 0.211，
#:    自己的血条在 y > 0.9。0.5 取的是两者之间的空档带。
#:    样本太少，**相机角度变化大时可能失效**。
#:    边界按 ``<`` 判定：y 恰为 0.5 时**不算**敌人（保守）。
ENEMY_HEALTH_BAR_Y_MAX: float = 0.5


#: 累计有多少个「像敌人但坐标不可判读」的实体（见 :func:`is_enemy_evidence`）。
#: 单元素 list 纯粹为了可写；计数器本身不是状态契约的一部分。
BAD_BBOX_COUNTER: list[int] = [0]


def is_enemy_evidence(entity: object) -> bool:
    """单个世界实体是否构成「有敌人」的证据。

    ★ **全仓唯一的敌人判据**（2026-10-16 抽出）
      判据此前被抄成了两处：

        * :meth:`PolicyContext.has_nearby_enemy` —— 用 ``group`` + ``bbox[1]``
        * ``core/loop.py`` 的 ``ENEMY_DETECTED`` 计数 —— 用 ``kind`` + ``.y``

      而 :class:`~autopilot.perception.world_entities.WorldEntity`
      **压根没有 ``.y`` 字段**（只有 ``kind/bbox/confidence/screen_pos``），
      于是 loop 那边的 ``getattr(e, "y", 1.0)`` 恒得 1.0、条件恒假，
      ``health_bar`` 永远计不进事件 —— 规则照常开火，事件却报 0 个敌人。
      真机 run ``20261014_215401`` 的 355 次普攻就是这么打出来的。

      抄两遍判据是这次出事的**结构原因**，所以抽成单一出处，
      两处都改为调用它，之后不可能再漂。

    判据内容见 :meth:`PolicyContext.has_nearby_enemy` 的长注释
    （A 敌人标识 / B 上半屏血条，以及 y<0.5 分界的实测依据）。

    :returns: True = 算一个敌人。坐标不可判读时返回 False 并累加
        :data:`BAD_BBOX_COUNTER` —— **跳过但不静默**。
    """
    kind = getattr(entity, "kind", None)
    if kind is None:
        return False
    group = getattr(entity, "group", kind)
    if group == "enemy":
        return True
    if group != "combat":
        return False
    bb = getattr(entity, "bbox", None)
    if not bb or len(bb) < 2:
        BAD_BBOX_COUNTER[0] += 1
        return False
    return float(bb[1]) < ENEMY_HEALTH_BAR_Y_MAX


def count_enemy_entities(entities) -> tuple[int, int]:
    """:func:`is_enemy_evidence` 的批量版。返回 ``(敌人数, 不可判读数)``。"""
    before = BAD_BBOX_COUNTER[0]
    n = sum(1 for e in entities if is_enemy_evidence(e))
    return n, BAD_BBOX_COUNTER[0] - before


__all__ = [
    "BAD_BBOX_COUNTER",
    "ENEMY_HEALTH_BAR_Y_MAX",
    "FUNCTION_NAMES",
    "UNKNOWN",
    "PolicyContext",
    "PolicySignals",
    "UnknownValue",
    "count_enemy_entities",
    "is_enemy_evidence",
]
