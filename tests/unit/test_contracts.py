"""``autopilot/contracts/`` 的单测（2026-10-12 任务 C）。

契约层是**跨进程、跨云端**的稳定边界，改动成本最高，所以测试的重点是
「它的语义不能被悄悄改掉」。三条最容易出事的语义：

1. **三态不能塌缩成两态** —— ``cd_ready`` 的 ``None``（UNKNOWN）不得压成
   False（永久卡住）或 True（白放技能）；``position=None`` 与
   ``quality="LOST"`` 语义不同。
2. **序列化必须往返无损** —— 云端 JSONL 链路靠它；往返丢字段就是静默的数据损坏。
3. **frozen 不可变** —— 上游共享同一个对象，改了就是跨 tick 的幽灵状态。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.contracts.intent_packet import (  # noqa: E402
    MAX_STATE_VERSION_DRIFT,
    ActionType,
    IntentConstraints,
    IntentPacket,
    IntentTarget,
)
from autopilot.contracts.state_packet import (  # noqa: E402
    SCENE_IDS,
    SCENE_WORLD,
    PartyMember,
    PlayerInfo,
    PositionInfo,
    ResourceInfo,
    SceneInfo,
    StatePacket,
    is_valid_scene,
)

# ==================================================================
# SceneInfo
# ==================================================================

@pytest.mark.unit
def test_scene_info__is_unknown_only_for_the_unknown_id() -> None:
    assert SceneInfo(id="UNKNOWN", confidence=0.0).is_unknown is True
    assert SceneInfo(id=SCENE_WORLD, confidence=0.9).is_unknown is False


@pytest.mark.unit
def test_scene_info__is_frozen() -> None:
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        SceneInfo(id="UNKNOWN", confidence=0.0).confidence = 1.0


@pytest.mark.unit
def test_is_valid_scene__accepts_every_declared_id() -> None:
    # ``SCENE_IDS`` 里含 "*" 通配键（模板 ``scenes = ["*"]`` 的含义），
    # 它不是场景 ID，要排除掉。
    for sid in SCENE_IDS.values():
        if sid == "*":
            continue
        assert is_valid_scene(sid), f"{sid} 是声明过的场景却被判非法"


@pytest.mark.unit
def test_is_valid_scene__the_wildcard_is_not_a_scene_id() -> None:
    """``SCENE_ANY`` 的值是 ``"*"``（模板 ``scenes = ["*"]`` 的含义），
    它是通配键而不是一个场景名 —— 别把它当合法场景写进规则。"""
    assert SCENE_IDS["SCENE_ANY"] == "*"
    assert is_valid_scene("*") is False


@pytest.mark.unit
def test_is_valid_scene__rejects_invented_ids() -> None:
    """凭空造的 id 必须拒 —— 否则规则会写出一个永不触发的场景。"""
    assert is_valid_scene("MADE_UP_SCENE") is False


# ==================================================================
# PositionInfo —— 三态
# ==================================================================

@pytest.mark.unit
def test_position__good_is_usable_for_navigation() -> None:
    assert PositionInfo(quality="GOOD", x=1.0, y=2.0).usable_for_navigation is True


@pytest.mark.unit
def test_position__lost_is_never_usable() -> None:
    """★ LOST = 确定不能用。放行它就是拿着空气走路。"""
    assert PositionInfo(quality="LOST", x=1.0, y=2.0).usable_for_navigation is False


@pytest.mark.unit
def test_position__degraded_is_usable_when_it_has_coordinates() -> None:
    """DEGRADED = 「能用但要额外确认」，所以**是**可导航的。

    与 LOST 的区别正是这一点：全局匹配成功但分数偏低，位置是真的，
    只是精度打折 —— 拒绝它等于「有点不确定就干脆别走」。
    """
    p = PositionInfo(quality="DEGRADED", x=1.0, y=2.0)
    assert p.usable_for_navigation is True


@pytest.mark.unit
def test_position__degraded_without_coordinates_is_not_usable() -> None:
    """打折归打折，没坐标还是没法走。"""
    assert PositionInfo(quality="DEGRADED", x=None, y=None
                        ).usable_for_navigation is False


@pytest.mark.unit
def test_position__good_without_coordinates_is_not_usable() -> None:
    """quality 好但没坐标 —— 有质量没事实，不算能用。"""
    assert PositionInfo(quality="GOOD", x=None, y=None).usable_for_navigation is False


@pytest.mark.unit
def test_position__lost_factory_produces_none_coordinates() -> None:
    p = PositionInfo.lost()
    assert p.quality == "LOST"
    assert p.x is None and p.y is None
    assert p.usable_for_navigation is False


# ==================================================================
# PartyMember —— CD 三态
# ==================================================================

@pytest.mark.unit
def test_party_member__unknown_cd_is_none_not_false() -> None:
    """★ 三态的核心。压成 False 就是「永久卡住」。"""
    m = PartyMember(slot=1, char_id="nahida")
    assert m.cd("E") is None


@pytest.mark.unit
def test_party_member__known_cd_is_returned_verbatim() -> None:
    m = PartyMember(slot=1, char_id="nahida", cd_ready={"E": False})
    assert m.cd("E") is False, "确认未就绪必须是 False 而不是 None"


@pytest.mark.unit
def test_party_member__cd_known_is_false_when_all_unknown() -> None:
    m = PartyMember(slot=1, char_id="x", cd_ready={"E": None, "Q": None})
    assert m.cd_known is False


@pytest.mark.unit
def test_party_member__cd_known_is_true_when_any_is_known() -> None:
    m = PartyMember(slot=1, char_id="x", cd_ready={"E": None, "Q": True})
    assert m.cd_known is True


# ==================================================================
# PlayerInfo
# ==================================================================

@pytest.mark.unit
def test_player__empty_party_has_no_active_member() -> None:
    assert PlayerInfo().active_member is None


@pytest.mark.unit
def test_player__active_member_finds_the_flagged_one() -> None:
    p = PlayerInfo(party=(PartyMember(slot=1, char_id="a"),
                          PartyMember(slot=2, char_id="b", active=True)))
    assert p.active_member.char_id == "b"


@pytest.mark.unit
def test_player__no_active_flag_means_none_not_the_first_one() -> None:
    """没标 active 就返回 None —— 「不知道谁在场」≠「第一个就是」。"""
    p = PlayerInfo(party=(PartyMember(slot=1, char_id="a"),))
    assert p.active_member is None


@pytest.mark.unit
def test_player__energy_ratio_is_none_when_unknown() -> None:
    assert PlayerInfo().energy_ratio is None
    assert PlayerInfo(energy=40).energy_ratio is None      # 有量没上限
    assert PlayerInfo(energy_max=40).energy_ratio is None  # 有上限没量


@pytest.mark.unit
def test_player__energy_ratio_computes_when_both_known() -> None:
    assert PlayerInfo(energy=20, energy_max=40).energy_ratio == pytest.approx(0.5)


@pytest.mark.unit
def test_player__energy_ratio_handles_a_full_bar() -> None:
    """满能量 -> 1.0，而不是被 max=0 之类的边界吞掉。"""
    assert PlayerInfo(energy=40, energy_max=40).energy_ratio == pytest.approx(1.0)


# ==================================================================
# StatePacket
# ==================================================================

@pytest.mark.unit
def test_state__unknown_factory_marks_perception_down() -> None:
    """感知失败要给**显式**的坏包，不返回 None、不给假空状态。"""
    p = StatePacket.unknown(reason="black_frame", state_version=7)
    assert p.resource.perception_ok is False
    assert p.scene.id == "UNKNOWN"
    assert p.position is None
    assert p.state_version == 7
    assert any("black_frame" in e for e in p.scene.evidence)


@pytest.mark.unit
def test_state__defaults_are_healthy() -> None:
    """裸构造的包应当是「健康」的 —— 否则测试夹具会到处假红。"""
    p = StatePacket(scene=SceneInfo(id=SCENE_WORLD, confidence=0.9))
    assert p.resource.perception_ok is True


@pytest.mark.unit
def test_state__json_round_trip_is_lossless() -> None:
    """云端 JSONL 靠它。往返丢字段就是静默的数据损坏。"""
    original = StatePacket(
        scene=SceneInfo(id=SCENE_WORLD, confidence=0.87,
                        evidence=("a", "b"), entered_ms=1234,
                        raw_label="world"),
        player=PlayerInfo(level=80, hp_ratio=0.5, hp_source="geom",
                          energy=30, energy_max=60,
                          party=(PartyMember(slot=1, char_id="nahida",
                                             active=True,
                                             cd_ready={"E": True, "Q": None}),)),
        position=PositionInfo(quality="GOOD", zone="z1", x=1.5, y=2.5,
                              confidence=0.9, source="localization"),
        resource=ResourceInfo(fps=5.0, tick_ms=42, perception_ok=True,
                              ocr_available=False, ocr_fail_streak=3),
        state_version=99, game_id="genshin_impact",
    )
    back = StatePacket.from_dict(json.loads(original.to_json()))
    assert back == original


@pytest.mark.unit
def test_state__round_trip_preserves_the_none_position() -> None:
    """position=None 与 position=LOST 必须都能原样往返 —— 两者语义不同。"""
    p = StatePacket(scene=SceneInfo(id=SCENE_WORLD, confidence=0.9),
                    position=None)
    assert StatePacket.from_dict(json.loads(p.to_json())).position is None


@pytest.mark.unit
def test_state__round_trip_preserves_a_lost_position() -> None:
    p = StatePacket(scene=SceneInfo(id=SCENE_WORLD, confidence=0.9),
                    position=PositionInfo.lost())
    back = StatePacket.from_dict(json.loads(p.to_json()))
    assert back.position is not None
    assert back.position.quality == "LOST"


@pytest.mark.unit
def test_state__from_dict_tolerates_a_minimal_payload() -> None:
    """宿主/云端可能只发最小字段 —— 缺字段不该让接收端崩。"""
    p = StatePacket.from_dict({})
    assert p.scene.id == "UNKNOWN"
    assert p.position is None
    assert p.resource.perception_ok is True


@pytest.mark.unit
def test_state__to_json_keeps_chinese_readable() -> None:
    """ensure_ascii=False —— 日志/排障时要能直接读中文。"""
    p = StatePacket(scene=SceneInfo(id="UNKNOWN", confidence=0.0,
                                    evidence=("原因=黑帧",)))
    assert "黑帧" in p.to_json()


# ==================================================================
# ActionType / IntentTarget
# ==================================================================

@pytest.mark.unit
def test_action_kinds_partition_the_actions() -> None:
    """每个动作都要有 kind —— 分派靠它，缺了就 AttributeError。

    ``ActionKind`` 是泛型别名，不能拿来做 ``isinstance``，
    所以断言「非空且在已声明的集合里」。
    """
    seen = set()
    for a in ActionType:
        kind = a.kind
        assert kind, f"{a} 没有 kind"
        seen.add(str(kind))
    assert len(seen) > 1, "所有动作挤在一个 kind 里等于没分"


@pytest.mark.unit
def test_action_type_rejects_an_invented_action() -> None:
    """响亮失败：本地构造传错名字必须抛，不静默兜底。"""
    with pytest.raises(ValueError):
        ActionType("DO_A_BARREL_ROLL")


@pytest.mark.unit
def test_intent_constraints__from_dict_applies_defaults() -> None:
    c = IntentConstraints.from_dict(None)
    assert c.max_attempts == 1
    assert c.requires_window_focus is True
    assert c.forbid_during == ()
    assert c.cooldown_ms == 0


@pytest.mark.unit
def test_intent_target__scene_requirement_defaults_to_empty() -> None:
    """空 = 不挑场景，不是「挑 UNKNOWN」。"""
    assert IntentTarget().scene_required == ()


@pytest.mark.unit
def test_intent_make__stamps_an_id_and_a_kind() -> None:
    p = IntentPacket.make(ActionType.USE_SKILL)
    assert p.intent_id.startswith("int_")
    assert p.action_kind is ActionType.USE_SKILL.kind


@pytest.mark.unit
def test_intent_make__gives_unique_ids() -> None:
    """id 重复会被去重表挡住 —— 那等于动作发不出去。"""
    ids = {IntentPacket.make(ActionType.JUMP).intent_id for _ in range(50)}
    assert len(ids) == 50


@pytest.mark.unit
def test_state_version_drift_allowance_is_bounded() -> None:
    """漂移容忍必须是小正数 —— 放太大等于「基于旧画面也能执行」。"""
    assert 0 < MAX_STATE_VERSION_DRIFT <= 30
