"""敌人数的两处判据**必须一致**（2026-10-16）。

背景
----
真机 run ``20261014_215401``：唯一启用的规则是 ``world_attack_only_with_enemy``
（``when = "has_nearby_enemy()"``），却打出 355 次普攻，而同一段时间
``ENEMY_DETECTED`` 的 count **多次为 0**。排查结论：

``WorldEntity`` 只有 ``kind / bbox / confidence / screen_pos``，
**没有 ``.y`` 字段**。于是

* ``core/loop.py``   ``getattr(e, "y", 1.0) < 0.5``  -> 恒 False（health_bar 不可见）
* ``policy/context.py``  ``bbox[1] < 0.5``           -> 正常

判据被抄成两处，其中一处读不存在的字段且被默认值静默兜住。

修法
----
把判据抽成 :func:`policy.context.is_enemy_evidence` 单一出处，两处都调它。
**本文件的作用是防它再漂回去** —— 断言「同一组实体，两边结论相同」，
而不是各自钉一个期望值。
"""

from __future__ import annotations

import pytest

from autopilot.perception.world_entities import WorldEntity
from autopilot.policy.context import (
    BAD_BBOX_COUNTER,
    PolicyContext,
    count_enemy_entities,
    is_enemy_evidence,
)

try:                                   # SceneInfo 的位置随 contracts 布局而动
    from autopilot.contracts.state_packet import SceneInfo
except ImportError:                    # pragma: no cover
    from autopilot.contracts.scene import SceneInfo


# ----------------------------------------------------------------------
# 夹具：按真实检测结果造的实体
# ----------------------------------------------------------------------
def _hb(y: float) -> WorldEntity:
    """一个 ``health_bar``（模型类名），bbox 左上角在 y 处。"""
    return WorldEntity(kind="health_bar", bbox=(0.30, y, 0.05, 0.02),
                       confidence=0.9, screen_pos=(216, 100))


def _enemy_identify() -> WorldEntity:
    return WorldEntity(kind="enemy_identify", bbox=(0.5, 0.4, 0.1, 0.1),
                       confidence=0.9, screen_pos=(360, 144))


def _chest() -> WorldEntity:
    return WorldEntity(kind="chest", bbox=(0.4, 0.6, 0.2, 0.2),
                       confidence=0.8, screen_pos=(288, 216))


def _ctx_with(entities) -> PolicyContext:
    """造一个只带 ``world_entities`` 的 PolicyContext。

    PolicyContext 是 **frozen dataclass**，不能直接赋值字段；
    ``world_entities`` 是 ``from_state`` 的可选关键字参数，走那里最干净。

    ``entities=None`` 必须原样透传 —— 那是 ``builder`` 的三态「没跑过」，
    转成 ``tuple()`` 就变成了「跑了但没看见」，UNKNOWN 会退化成 False。
    """
    ents = None if entities is None else tuple(entities)
    return PolicyContext.from_state(_minimal_state(), world_entities=ents)


def _minimal_state():
    """一个够用的 StatePacket —— 这里只用到 world_entities，其余留空。

    ``scene`` 是唯一必填字段，照 ``test_core_loop._packet`` 的写法给。
    """
    from autopilot.contracts.state_packet import ResourceInfo, StatePacket

    return StatePacket(
        scene=SceneInfo(id="WORLD", confidence=0.9, entered_ms=1000),
        resource=ResourceInfo(perception_ok=True),
        state_version=1,
    )


# ----------------------------------------------------------------------
# 1 · 根因本身：.y 不存在这件事必须被钉住
# ----------------------------------------------------------------------
def test_world_entity_really_has_no_y_attribute() -> None:
    """★★ 根因：``WorldEntity`` 没有 ``.y``。

    ``bbox`` 是 ``(x, y, w, h)``，归一化 y 是 ``bbox[1]``。
    谁再写 ``e.y``，这个测试会直接告诉他正确写法。
    """
    assert set(WorldEntity.__dataclass_fields__) == {
        "kind", "bbox", "confidence", "screen_pos"}
    assert not hasattr(WorldEntity, "y")
    assert not hasattr(_hb(0.2), "y")


# ----------------------------------------------------------------------
# 2 · 判据本身
# ----------------------------------------------------------------------
def test_enemy_health_bar_in_upper_half_counts() -> None:
    """★ 上半屏的血条 = 敌人（真机实测 y = 0.319/0.161/0.225/0.211）。"""
    assert is_enemy_evidence(_hb(0.22)) is True


def test_own_health_bar_at_bottom_does_not_count() -> None:
    """★ 自己的血条在 y > 0.9，必须排除。"""
    assert is_enemy_evidence(_hb(0.93)) is False


def test_enemy_identify_always_counts() -> None:
    """★ 敌人标识不受 y 限制。"""
    assert is_enemy_evidence(_enemy_identify()) is True


def test_chest_never_counts() -> None:
    assert is_enemy_evidence(_chest()) is False


def test_bbox_boundary_is_exclusive() -> None:
    """★ y 恰为 0.5 **不算**敌人（保守，边界按 ``<``）。"""
    assert is_enemy_evidence(_hb(0.5)) is False
    assert is_enemy_evidence(_hb(0.499)) is True


# ----------------------------------------------------------------------
# 3 · bbox 缺失：跳过但**不静默**
# ----------------------------------------------------------------------
def test_missing_bbox_is_skipped_but_counted() -> None:
    """★ bbox 读不到 -> 不算敌人（不能瞎猜开火），但**必须留下计数**。"""
    bad = WorldEntity(kind="health_bar", bbox=None, confidence=0.5,
                      screen_pos=(0, 0))
    before = BAD_BBOX_COUNTER[0]
    assert is_enemy_evidence(bad) is False
    assert BAD_BBOX_COUNTER[0] == before + 1, "缺 bbox 必须可观测"


def test_count_enemy_entities_reports_unreadable() -> None:
    good = [_hb(0.2), _hb(0.25)]
    bad = WorldEntity(kind="health_bar", bbox=None, confidence=0.5,
                      screen_pos=(0, 0))
    n, unreadable = count_enemy_entities([*good, bad])
    assert (n, unreadable) == (2, 1)


def test_short_bbox_is_also_unreadable() -> None:
    bad = WorldEntity(kind="health_bar", bbox=(0.3,), confidence=0.5,
                      screen_pos=(0, 0))
    before = BAD_BBOX_COUNTER[0]
    assert is_enemy_evidence(bad) is False
    assert BAD_BBOX_COUNTER[0] == before + 1


# ----------------------------------------------------------------------
# 4 · ★★ 一致性：本文件的核心
# ----------------------------------------------------------------------
@pytest.mark.parametrize("entities, expect", [
    # 真机实测形态：3 条上半屏血条 + 1 条自己的底��血条
    ([_hb(0.22), _hb(0.25), _hb(0.30), _hb(0.93)], 3),
    # 空场景
    ([], 0),
    # 只有自己的血条
    ([_hb(0.93)], 0),
    # 敌人标识 + 血条混合
    ([_enemy_identify(), _hb(0.22)], 2),
    # 全是宝箱
    ([_chest(), _chest()], 0),
    # 一半越过分界线
    ([_hb(0.10), _hb(0.10), _hb(0.80), _hb(0.80)], 2),
])
def test_event_count_and_policy_signal_agree(entities, expect) -> None:
    """★★ **两条消费路径对同一组实体必须给出相同结论**。

    左边 = ``ENEMY_DETECTED`` 的 count（``core/loop.py`` 的计数）
    右边 = ``has_nearby_enemy()``（规则 ``when`` 的判据）

    2026-10-15 之前左边恒为 0、右边按真值行动 ——
    差出来的就是 355 次无差别普攻。
    """
    ctx = _ctx_with(entities)

    n, _bad = count_enemy_entities(entities)
    policy_says = ctx.has_nearby_enemy()

    assert n == expect, f"事件计数应为 {expect}"
    assert bool(policy_says) is (expect > 0), "决策层结论与计数不一致"
    assert (n > 0) is bool(policy_says), "★★ 两条路径对同一组实体给出了相反结论"


def test_has_nearby_enemy_still_returns_unknown_without_entities() -> None:
    """★ 三态语义不许因为抽函数被破坏：``None``（没跑）仍是 UNKNOWN。"""
    r = _ctx_with(None).has_nearby_enemy()
    assert not isinstance(r, bool)
    assert str(r).startswith("UNKNOWN")
