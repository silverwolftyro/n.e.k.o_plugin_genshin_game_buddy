"""``actuator/validator.py`` 的单测（2026-10-12 任务 C）。

这个模块是**发按键前的最后一道闸**。它拒绝的每一条都对应一类真实事故：

* TTL 过期 -> 拿着 30 秒前的状态去按键（人早走了）
* state_version 不匹配 -> 基于旧画面执行
* perception_ok=False -> 在黑屏/空帧上乱按
* 场景不符 -> 在野外按 B 开背包
* 没位置 -> 拿空气去走路
* 重复 intent_id -> 网络重投导致同一个动作发两遍

所以本文件重点是**每条拒绝路径都要报出对得上的 reason**，而不是笼统失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.validator import (  # noqa: E402
    DEFAULT_SEEN_TTL_MS,
    IntentValidator,
    SeenIntentStore,
    ValidationResult,
)
from autopilot.contracts.intent_packet import (  # noqa: E402
    ActionType,
    IntentConstraints,
    IntentPacket,
    IntentTarget,
)
from autopilot.contracts.state_packet import (  # noqa: E402
    SCENE_DIALOG,
    SCENE_WORLD,
    PositionInfo,
    ResourceInfo,
    SceneInfo,
    StatePacket,
)

NOW = 1_700_000_000_000


def _state(*, sv: int = 1, scene: str = SCENE_WORLD, ok: bool = True,
           position: PositionInfo | None = None) -> StatePacket:
    return StatePacket(
        scene=SceneInfo(id=scene, confidence=0.9, entered_ms=NOW - 5000),
        position=position,
        resource=ResourceInfo(perception_ok=ok),
        timestamp_ms=NOW,
        state_version=sv,
    )


def _intent(*, action=ActionType.USE_SKILL, intent_id: str = "int_1",
            issued_at_ms: int = NOW, sv: int = 1, ttl: int = 10,
            scene_required: tuple = (), source: str = "LOCAL",
            signature: str | None = "sig", params: dict | None = None,
            target=None, constraints=None) -> IntentPacket:
    return IntentPacket(
        action=action,
        action_kind=action.kind,
        params=dict(params or {}),
        target=target if target is not None
        else IntentTarget(scene_required=scene_required),
        constraints=constraints if constraints is not None
        else IntentConstraints(),
        intent_id=intent_id,
        issued_at_ms=issued_at_ms,
        state_version=sv,
        ttl_seconds=ttl,
        source=source,
        signature=signature,
    )


# ==================================================================
# ValidationResult
# ==================================================================

@pytest.mark.unit
def test_result__bool_reflects_ok() -> None:
    assert bool(ValidationResult(True, "OK")) is True
    assert bool(ValidationResult(False, "WHY", "detail")) is False


@pytest.mark.unit
def test_result__str_includes_the_detail_only_on_failure() -> None:
    assert str(ValidationResult(True, "OK")) == "OK"
    assert str(ValidationResult(False, "TTL_EXPIRED", "过期 3 秒")) == \
        "TTL_EXPIRED: 过期 3 秒"


# ==================================================================
# SeenIntentStore
# ==================================================================

@pytest.mark.unit
def test_seen_store__first_time_is_not_seen() -> None:
    s = SeenIntentStore()
    assert s.seen("a", NOW) is False


@pytest.mark.unit
def test_seen_store__remembered_id_is_seen_again() -> None:
    """网络重投 —— 同一个 intent_id 绝不能执行两次。"""
    s = SeenIntentStore()
    s.remember("a", NOW)
    assert s.seen("a", NOW) is True


@pytest.mark.unit
def test_seen_store__forgets_ids_after_the_ttl() -> None:
    """TTL 之后重新出现是**新意图**，不该被当成重投挡住。"""
    s = SeenIntentStore(ttl_ms=1000)
    s.remember("a", NOW)
    assert s.seen("a", NOW + 2000) is False


@pytest.mark.unit
def test_seen_store__is_bounded_by_capacity() -> None:
    """长期运行不能无限涨 —— 这是个每 tick 都要查的表。"""
    s = SeenIntentStore(capacity=4, ttl_ms=10 ** 9)
    for i in range(50):
        s.remember(f"id{i}", NOW)
    assert len(s) <= 4


@pytest.mark.unit
def test_seen_store__default_ttl_is_five_minutes() -> None:
    assert DEFAULT_SEEN_TTL_MS == 300_000


# ==================================================================
# 正常通过
# ==================================================================

@pytest.mark.unit
def test_validate__a_fresh_well_formed_intent_passes() -> None:
    v = IntentValidator()
    assert v.validate(_intent(), _state(), NOW).ok is True


@pytest.mark.unit
def test_validate__commit_false_does_not_register_the_id() -> None:
    """「先看一眼能不能做」不该占位 —— 否则真正执行时反而被自己挡住。"""
    v = IntentValidator()
    v.validate(_intent(intent_id="x"), _state(), NOW, commit=False)
    assert v.seen_count() == 0


@pytest.mark.unit
def test_validate__commit_true_registers_the_id() -> None:
    v = IntentValidator()
    v.validate(_intent(intent_id="x"), _state(), NOW, commit=True)
    assert v.seen_count() == 1


# ==================================================================
# 各条拒绝路径
# ==================================================================

@pytest.mark.unit
def test_validate__rejects_a_duplicate_after_commit() -> None:
    v = IntentValidator()
    intent = _intent(intent_id="dup")
    assert v.validate(intent, _state(), NOW, commit=True).ok is True
    again = v.validate(intent, _state(), NOW)
    assert again.ok is False
    assert again.reason == "INTENT_ID_DUPLICATE"


@pytest.mark.unit
def test_validate__duplicate_check_uses_the_caller_supplied_clock() -> None:
    """★ 查重必须和整条校验链认同**同一个** now。

    踩过的坑：``_check_id`` 曾经自己取 ``time.time()``，于是调用方传合成
    ``now_ms``（回放 / 确定性测试 / 合并层用状态时间戳）时，去重的 TTL
    比较对不上，重复意图直接漏过。
    """
    store = SeenIntentStore(ttl_ms=10_000)
    v = IntentValidator(seen_store=store)
    intent = _intent(intent_id="clock")
    assert v.validate(intent, _state(), NOW, commit=True).ok is True
    # 同一 now 再来一次 -> 必须判重（若内部用 wall clock，NOW 是 2023 年，
    # 与真实时间差了几十天，TTL 会把这条当成过期而放行）
    assert v.validate(intent, _state(), NOW).ok is False


@pytest.mark.unit
def test_validate__rejects_an_expired_intent() -> None:
    """拿着 30 秒前的状态去按键 —— 人早就走了。"""
    v = IntentValidator()
    r = v.validate(_intent(issued_at_ms=NOW - 60_000, ttl=1),
                   _state(), NOW)
    assert r.ok is False
    assert "TTL" in r.reason or "EXPIRE" in r.reason


@pytest.mark.unit
def test_validate__accepts_an_intent_inside_its_ttl() -> None:
    v = IntentValidator()
    assert v.validate(_intent(issued_at_ms=NOW - 1000, ttl=10),
                      _state(), NOW).ok is True


@pytest.mark.unit
def test_validate__rejects_a_stale_state_version() -> None:
    """意图基于第 1 版状态，但手上已经是第 50 版 —— 画面早变了。

    注意有 ``MAX_STATE_VERSION_DRIFT``（10 帧）的容忍：状态每帧 +1，
    严格相等会让所有意图失效。所以要漂移**超过**阈值才该拒。
    """
    from autopilot.contracts.intent_packet import MAX_STATE_VERSION_DRIFT
    v = IntentValidator()
    far = 1 + MAX_STATE_VERSION_DRIFT + 1
    r = v.validate(_intent(sv=1), _state(sv=far), NOW)
    assert r.ok is False
    assert "STATE_VERSION" in r.reason


@pytest.mark.unit
def test_validate__tolerates_a_small_state_version_drift() -> None:
    """几帧之内的漂移是正常的 —— 严格相等会把所有意图都毙掉。"""
    from autopilot.contracts.intent_packet import MAX_STATE_VERSION_DRIFT
    v = IntentValidator()
    assert v.validate(_intent(sv=1),
                      _state(sv=1 + MAX_STATE_VERSION_DRIFT - 1),
                      NOW).ok is True


@pytest.mark.unit
def test_validate__rejects_when_perception_is_down() -> None:
    """★ 黑屏 / 空帧上绝不执行 —— 这是最不该出事的一类。"""
    v = IntentValidator()
    r = v.validate(_intent(), _state(ok=False), NOW)
    assert r.ok is False
    assert "PERCEPTION" in r.reason


@pytest.mark.unit
def test_validate__rejects_a_scene_the_intent_did_not_ask_for() -> None:
    """在野外按 B 开背包 —— 误判场景的直接后果。

    注意场景检查是**两层**：一层的 traits 白名单（OPEN_INVENTORY 只允许
    MAIN_MENU/COMBAT/BAG_INVENTORY/MAP_BIG/UNKNOWN），一层是意图自己声明的
    ``scene_required``。这里测的是后者。
    """
    v = IntentValidator()
    r = v.validate(
        _intent(action=ActionType.OPEN_INVENTORY,
                scene_required=(SCENE_DIALOG,)),
        _state(scene="BAG_INVENTORY"), NOW)
    assert r.ok is False
    assert r.reason == "SCENE_MISMATCH"


@pytest.mark.unit
def test_validate__accepts_a_scene_that_matches() -> None:
    v = IntentValidator()
    r = v.validate(
        _intent(action=ActionType.OPEN_INVENTORY,
                scene_required=("BAG_INVENTORY",)),
        _state(scene="BAG_INVENTORY"), NOW)
    assert r.ok is True


@pytest.mark.unit
def test_validate__scene_requirement_of_none_means_any_allowed_scene() -> None:
    """没写 scene_required 就不该被意图层拦 —— 移动类动作如此。"""
    v = IntentValidator()
    r = v.validate(_intent(action=ActionType.MOVE_FORWARD),
                   _state(scene=SCENE_WORLD), NOW)
    assert r.ok is True


@pytest.mark.unit
def test_validate__rejects_an_intent_that_needs_a_position_it_does_not_have() -> None:
    """拿着空气去走路。

    触发条件是 ``target.world_position`` 非空 —— 也就是「这个意图指名了
    要去的地方」，而不是「所有导航动作都要定位」。
    """
    v = IntentValidator()
    r = v.validate(
        _intent(action=ActionType.MOVE_FORWARD,
                target=IntentTarget(world_position={"x": 4634.0, "y": 4687.0})),
        _state(position=None), NOW)
    assert r.ok is False
    assert r.reason == "POSITION_UNAVAILABLE"


@pytest.mark.unit
def test_validate__legacy_z_key_still_triggers_the_position_gate() -> None:
    """★ 轴名兼容层（2026-10-14）：``z`` 仍然算「指名了要去哪」。

    内部已统一 ``x`` / ``y``，但历史事件流与手写 JSON 里是 ``z``。
    如果兼容层失效，这里会变成 ``world_position`` 非空却**不再**触发
    定位校验 —— 那就是「拿着空气去走路」，而且不报错。
    """
    v = IntentValidator()
    r = v.validate(
        _intent(action=ActionType.MOVE_FORWARD,
                target=IntentTarget(world_position={"x": 4634.0, "z": 4687.0})),
        _state(position=None), NOW)
    assert r.ok is False
    assert r.reason == "POSITION_UNAVAILABLE"


@pytest.mark.unit
def test_validate__rejects_a_lost_position_explicitly() -> None:
    """LOST 位置不能拿来导航 —— 宁可停下也不猜。"""
    v = IntentValidator()
    r = v.validate(
        _intent(action=ActionType.MOVE_FORWARD,
                target=IntentTarget(world_position={"x": 1.0, "y": 2.0})),
        _state(position=PositionInfo.lost()), NOW)
    assert r.ok is False
    assert r.reason == "POSITION_UNAVAILABLE"


@pytest.mark.unit
def test_validate__accepts_a_navigation_intent_with_a_position() -> None:
    v = IntentValidator()
    pos = PositionInfo(quality="GOOD", x=4634.0, y=4687.0)
    r = v.validate(
        _intent(action=ActionType.MOVE_FORWARD,
                target=IntentTarget(world_position={"x": 1.0, "y": 2.0})),
        _state(position=pos), NOW)
    assert r.ok is True


@pytest.mark.unit
def test_validate__a_movement_without_a_target_needs_no_position() -> None:
    """没指名目的地就不该被位置卡住 —— 否则没定位时连原地转身都做不了。"""
    v = IntentValidator()
    assert v.validate(_intent(action=ActionType.MOVE_FORWARD),
                      _state(position=None), NOW).ok is True


@pytest.mark.unit
def test_validate__rejects_an_unsigned_cloud_intent_by_default() -> None:
    """云端来的意图默认要签名 —— 不然任何人都能让你按键。"""
    v = IntentValidator()
    r = v.validate(_intent(source="CLOUD", signature=None), _state(), NOW)
    assert r.ok is False
    assert "SIGN" in r.reason


@pytest.mark.unit
def test_validate__unsigned_cloud_can_be_allowed_explicitly() -> None:
    """本地开发/自建云端可以显式放行 —— 但必须显式。"""
    v = IntentValidator(allow_unsigned_cloud=True)
    assert v.validate(_intent(source="CLOUD", signature=None),
                      _state(), NOW).ok is True


@pytest.mark.unit
def test_validate__a_signed_cloud_intent_is_accepted() -> None:
    v = IntentValidator()
    assert v.validate(_intent(source="CLOUD", signature="sig"),
                      _state(), NOW).ok is True


@pytest.mark.unit
def test_validate__local_intent_needs_no_signature() -> None:
    """本地意图天生可信 —— 强制签名只会让人没法用。"""
    v = IntentValidator()
    assert v.validate(_intent(source="LOCAL", signature=None),
                      _state(), NOW).ok is True


# ==================================================================
# 杂项
# ==================================================================

@pytest.mark.unit
def test_validate__default_now_is_wall_clock() -> None:
    """不传 now_ms 也不能崩 —— 生产路径就是这么调的。"""
    v = IntentValidator()
    r = v.validate(_intent(issued_at_ms=0), _state())
    assert r.ok is False, "1970 年签发的意图早该过期"


@pytest.mark.unit
def test_forget__removes_a_single_id() -> None:
    """「允许重发」按钮的实现基础。"""
    v = IntentValidator()
    v.validate(_intent(intent_id="x"), _state(), NOW, commit=True)
    v.forget("x")
    assert v.validate(_intent(intent_id="x"), _state(), NOW).ok is True


@pytest.mark.unit
def test_seen_store_can_be_shared_across_validators() -> None:
    """合并层和执行层各有一个 validator 时，去重必须共享。"""
    store = SeenIntentStore()
    a = IntentValidator(seen_store=store)
    b = IntentValidator(seen_store=store)
    a.validate(_intent(intent_id="shared"), _state(), NOW, commit=True)
    assert b.validate(_intent(intent_id="shared"), _state(), NOW).ok is False


@pytest.mark.unit
def test_validate__every_rejection_carries_a_human_readable_detail() -> None:
    """reason 只有代码是不够的 —— 日志要能一眼看懂为什么没按键。"""
    v = IntentValidator()
    r = v.validate(_intent(issued_at_ms=NOW - 60_000, ttl=1),
                   _state(ok=False), NOW)
    assert not r.ok
    assert r.reason and r.detail, "被拒必须同时有 reason 和 detail"
