"""切人记账链：**数字键 -> 在场槽位 -> PartyMember.active -> onfield()**。

★ 这份文件为什么改名不改内容
----------------------------
它原本叫 ``test_combo_ladder.py``，测的是 ``rules/genshin_combat.toml`` 里
那 7 条 ``combo_*`` 规则（priority 62~68）。那 7 条规则**已在 2026-10-18
删除** —— 它们把「钟离→夜兰→纳西妲→绫华」这一个玩家的队伍写进了代码，
换成 ``autopilot/combos/combos.toml`` 里的一条 :class:`ComboSequence`
（顺序写在配置里、切人由引擎隐式插入、不再抢 priority 位置）。

连招本身的顺序 / 抢占 / 超时 / 回退测试全部搬到了
``tests/unit/test_combo_sequence.py``。**只有下面这一组还留在这里** ——
它测的既不是规则也不是连招，而是**连招依赖的那条输入链**：

    切人按键 -> ActiveSlotTracker -> StateBuilder._build_player
             -> PartyMember.active=True -> PlayerInfo.active_member
             -> onfield() / e_ready() / q_ready()

这一组**既不属于连招也不属于规则**，它属于感知层。换个策略实现
（连招、云端、纯反射）这条链都得在，所以它单独成文件是合理的。

覆盖链路：数字键 -> ActiveSlotTracker -> builder -> onfield()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.active_slot import ActiveSlotTracker  # noqa: E402
from autopilot.contracts.state_packet import (  # noqa: E402
    PlayerInfo, StatePacket,
)
from autopilot.perception.builder import StateBuilder  # noqa: E402


# ==================================================================
# 1 · 记账器 -> builder -> onfield()（连招与规则的共同输入）
# ==================================================================
class TestSwitchBookkeepingFeedsOnfield:
    def test_tracker_marks_the_party_member_active(self) -> None:
        """引擎切完人，builder 下一帧就该给对应槽位一个 active 成员。

        ★ 名字此时**还是空的** —— 角色名靠 OCR 轮转，一圈要 4×25 tick。
          但「3 号位现在站着个人」是**记账**来的（引擎刚按了 3），
          那不是编造。所以成员照样要给：`active=True`，
          `e_ready()`/`q_ready()` 才有地方读 cd_ready。
        """
        tracker = ActiveSlotTracker(stale_after_ms=0)
        sb = StateBuilder(active_slots=tracker, character_id=None)
        tracker.note_switch(3, source="engine")

        party = sb._build_player("WORLD", None, [], [], 1, frame=None).party
        assert len(party) == 1
        assert party[0].slot == 3
        assert party[0].active is True
        assert party[0].name == "", "还没 OCR，别编名字"

    def test_active_member_resolves_through_player_info(self) -> None:
        """``PlayerInfo.active_member`` 取第一个 active=True 的成员。"""
        tracker = ActiveSlotTracker(stale_after_ms=0)
        sb = StateBuilder(active_slots=tracker, character_id=None)
        info = sb._build_player("WORLD", None, [], [], 1, frame=None)
        assert info.active_member is None      # 还没切过人

        tracker.note_switch(2, source="engine")
        info = sb._build_player("WORLD", None, [], [], 1, frame=None)
        assert info.active_member is not None
        assert info.active_member.slot == 2

    def test_no_tracker_means_unknown(self) -> None:
        """没接记账器 -> 谁在场 UNKNOWN（不是「默认 1 号位」）。"""
        sb = StateBuilder(character_id=None)
        assert sb._current_active_slot() is None

    def test_physical_digits_update_the_slot(self) -> None:
        """用户自己按数字键也要更新（监听器那一半）。"""
        t = ActiveSlotTracker(stale_after_ms=0)
        assert t.note_physical_key("TWO") is True
        assert t.current() == 2
        assert t.note_physical_key("W") is False, "W 不是切人键"
        assert t.current() == 2, "非数字键不该改动在场槽位"

    def test_illegal_slot_is_rejected(self) -> None:
        t = ActiveSlotTracker(stale_after_ms=0)
        t.note_switch(2)
        assert t.note_switch(9) is False
        assert t.current() == 2, "非法槽位不该覆盖已有状态"

    def test_stale_expiry_falls_back_to_unknown(self) -> None:
        clock = {"ms": 0}
        t = ActiveSlotTracker(stale_after_ms=1000,
                              now_fn=lambda: clock["ms"])
        t.note_switch(2)
        assert t.current() == 2
        clock["ms"] = 999
        assert t.current() == 2
        clock["ms"] = 1000
        assert t.current() is None, "过期必须回到 UNKNOWN 而不是留在旧值"

    def test_tracker_exception_degrades_to_unknown(self) -> None:
        """记账器挂了不能让感知层整条 tick 崩。"""

        class _Boom:
            def current(self):
                raise RuntimeError("tracker 挂了")

        sb = StateBuilder(active_slots=_Boom(), character_id=None)
        assert sb._current_active_slot() is None


# ==================================================================
# 2 · 这条链在 StatePacket 上真的读得出来
# ==================================================================
@dataclass
class _Tracker:
    """只实现 ``current()`` 的最小记账器。"""

    slot: int | None

    def current(self) -> int | None:
        return self.slot


class TestOnfieldReadsTheChain:
    def _ctx(self, slot: int | None, name: str):
        from autopilot.policy.context import PolicyContext

        tracker = _Tracker(slot)
        sb = StateBuilder(active_slots=tracker, character_id=None)
        player = sb._build_player("WORLD", None, [], [], 1, frame=None)
        st = StatePacket(player=player, state_version=0)
        return PolicyContext.from_state(st, now_ms=0)

    def test_onfield_true_for_the_remembered_name(self) -> None:
        """有人切过人 + OCR 读到名字 -> onfield('X') 为 True。

        这里用「名字已知」的 PartyMember 造一帧，验证 DSL 那一端
        真的能读出来 —— 前面几条只验证了 builder 产出，链路到
        :func:`PolicyContext.onfield` 为止还没接上。
        """
        from autopilot.contracts.state_packet import (
            PartyMember, PlayerInfo, SceneInfo, StatePacket, SCENE_WORLD,
        )
        from autopilot.policy.context import PolicyContext

        player = PlayerInfo(party=(
            PartyMember(slot=2, char_id="夜兰", name="夜兰", active=True),
            PartyMember(slot=1, char_id="钟离", name="钟离", active=False),
        ))
        st = StatePacket(scene=SceneInfo(id=SCENE_WORLD, confidence=0.9),
                         player=player, state_version=0)
        ctx = PolicyContext.from_state(st, now_ms=0)
        assert ctx.onfield("夜兰") is True
        assert ctx.onfield("钟离") is False

    def test_onfield_is_unknown_without_any_active_member(self) -> None:
        """★ 没人切过人的时候 onfield 恒 UNKNOWN（不是 False）。

        连招靠这个「不知道就不做」：猜一个「大概是钟离吧」会让
        夜兰的大招打在钟离身上。返回值是 ``UnknownValue`` 时
        ``unknowns`` 里会出现 ``onfield``，排查时能看出是感知缺口。
        """
        from autopilot.contracts.state_packet import (
            PartyMember, PlayerInfo, SceneInfo, StatePacket, SCENE_WORLD,
        )
        from autopilot.policy.context import PolicyContext, UnknownValue

        player = PlayerInfo(party=(
            PartyMember(slot=2, char_id="夜兰", name="夜兰", active=False),
        ))
        st = StatePacket(scene=SceneInfo(id=SCENE_WORLD, confidence=0.9),
                         player=player, state_version=0)
        ctx = PolicyContext.from_state(st, now_ms=0)
        out = ctx.onfield("夜兰")
        assert isinstance(out, UnknownValue), out
        assert not out, "UNKNOWN 的 __bool__ 恒 False —— 规则不触发"


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
