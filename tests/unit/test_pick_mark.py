"""拾取标记（B2）单元测试 —— 2026-10-11。

复现的真机故障：run ``20261011_180732``，154.8 秒里**151 次真执行**
``USE_PICKUP`` + 603 次被 min_gap 挡掉。根因是宝箱被打开后
``has_nearby_chest()`` **不消失**，规则每 tick 重新瞄上同一个箱子。

这里用一个**临时规则目录**而不是真实的 ``autopilot/rules`` ——
因为真实文件里那三条规则现在是 ``enabled = false``（等真机再开），
拿它当夹具会让本测试跟着那个开关上下跳。用独立夹具测的是**机制**，
不是某个开关的当前状态。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.contracts.state_packet import (  # noqa: E402
    ResourceInfo,
    SceneInfo,
    StatePacket,
)
from autopilot.policy.context import (  # noqa: E402
    PICK_MARK_COOLDOWN_S,
    PICK_MARK_RADIUS_PX,
    PickMark,
)
from autopilot.policy.local_tree import LocalPolicyTree  # noqa: E402

RULE = """
[[rules]]
name = "t_open_chest"
action = "USE_PICKUP"
when = "has_nearby_chest()"
priority = 20
ttl_seconds = 3
scenes = ["WORLD"]
enabled = true
"""


class _Chest:
    """一个永远不会被「打开」的宝箱 —— 这正是故障现场。"""

    def __init__(self, x=640.0, y=360.0):
        self.kind = "chest"
        self.group = "chest"
        self.bbox = (0.5, 0.4, 0.1, 0.1)
        self.screen_pos = (x, y)
        self.conf = 0.9


def _state(version: int) -> StatePacket:
    """一个**健康**的 WORLD 场景状态包。

    ⚠️ 两个坑叠在一起，测试会**假绿**（一条都不命中，看不出是机制坏了
      还是夹具不对）：

    1. ``StatePacket.unknown()`` 的场景是 UNKNOWN，而
       ``decide()`` 在 UNKNOWN 下只放行 ``SAFE_ACTIONS``（全是
       「停下来/看一眼」），USE_PICKUP 直接被跳过。
    2. 它的 ``perception_ok=False``，而 ``decide()`` 在感知不健康时
       **连规则循环都不进**，直接发 ``PERCEPTION_DOWN_NO_DECISION``
       返回 —— 连 trace.entries 都是空的。

    所以两处都得改：场景设成 WORLD，resource 换成 perception_ok=True。
    """
    st = StatePacket.unknown(game_id="genshin", state_version=version,
                             reason="test")
    object.__setattr__(
        st, "scene",
        SceneInfo(id="WORLD", confidence=0.9, evidence=("test",)),
    )
    object.__setattr__(
        st, "resource",
        ResourceInfo(perception_ok=True),
    )
    return st


@pytest.fixture
def tree():
    d = Path(tempfile.mkdtemp(prefix="pick_mark_"))
    (d / "t.toml").write_text(RULE, encoding="utf-8")
    # ⚠️ 时钟必须从 ``now_fn`` 注入：``decide()`` 里的 ``now`` 取自它，
    #   直接改 ``tree._now_ms`` 之类是**没用**的（实测踩过：标记永远不过期）。
    #   本模块所有时间推进都写 ``tree._test_clock["t"]``。
    clock = {"t": 1_000_000}
    t = LocalPolicyTree(rules_dir=d, now_fn=lambda: clock["t"])
    t._test_clock = clock
    try:
        yield t
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _poll(tree, chest, t0_ms=1_000_000, n=10, step_ms=200):
    """连poll n 次，返回每次是否命中。"""
    hits = []
    for i in range(n):
        tree.last_world_entities = (chest,)
        tree._test_clock["t"] = t0_ms + i * step_ms
        trace = tree.decide(_state(i + 1))
        hits.append(trace.hit)
    return hits


@pytest.mark.unit
def test_pickmark__only_first_poll_matches(tree) -> None:
    """核心断言：同一个箱子连续在附近 10 帧，**只有第 1 帧触发**。"""
    hits = _poll(tree, _Chest())
    print(f"\n  10 帧命中情况 = {hits}")
    assert hits[0] is True, "第一帧应该命中"
    assert sum(1 for h in hits if h) == 1, (
        f"标记失效：{sum(1 for h in hits if h)} 帧命中，应该只有 1 帧")


@pytest.mark.unit
def test_pickmark__mark_expires_after_cooldown(tree) -> None:
    """超过冷却后，同一个箱子应该能**再次**触发 —— 否则就是永久失明。"""
    t0 = 1_000_000
    tree.last_world_entities = (_Chest(),)
    tree._test_clock["t"] = t0
    first = tree.decide(_state(1)).hit
    # 冷却内再试一次
    tree._test_clock["t"] = t0 + int(PICK_MARK_COOLDOWN_S * 1000 / 2)
    during = tree.decide(_state(2)).hit
    # 冷却外
    tree._test_clock["t"] = t0 + int(PICK_MARK_COOLDOWN_S * 1000) + 500
    after = tree.decide(_state(3)).hit
    print(f"\n  首帧={first}  冷却内={during}  冷却后={after}")
    assert first is True
    assert during is False, "冷却期内不该重复触发"
    assert after is True, "冷却期后必须能再触发（否则永久失明）"


@pytest.mark.unit
def test_pickmark__different_spot_not_suppressed(tree) -> None:
    """标记是**按位置**的：走到别处的箱子不该被远处的标记压住。"""
    tree.last_world_entities = (_Chest(x=640.0, y=360.0),)
    tree._test_clock["t"] = 1_000_000
    assert tree.decide(_state(1)).hit is True

    far = _Chest(x=640.0 + PICK_MARK_RADIUS_PX * 3, y=360.0)
    tree.last_world_entities = (far,)
    tree._test_clock["t"] = 1_000_000 + 200
    print("\n  远处箱子应该仍然命中")
    assert tree.decide(_state(2)).hit is True, (
        "标记作用半径内才该压制；换一个位置的箱子必须照常触发")


@pytest.mark.unit
def test_pickmark__other_group_not_suppressed(tree) -> None:
    """按 group 分别记忆：压住宝箱不该顺带压掉落物。"""

    class _Drops(_Chest):
        def __init__(self):
            super().__init__()
            self.kind = self.group = "drops"

    tree.last_world_entities = (_Chest(),)
    tree._test_clock["t"] = 1_000_000
    assert tree.decide(_state(1)).hit is True

    tree.last_world_entities = (_Drops(),)
    tree._test_clock["t"] = 1_000_000 + 200
    # 这条规则只查 chest，所以 drops 不该影响它 —— 换成没有 chest 的帧
    tree.last_world_entities = ()
    tree._test_clock["t"] = 1_000_000 + 400
    assert tree.decide(_state(3)).hit is False, "没有 chest 时不该命中"


@pytest.mark.unit
def test_pickmark__no_position_does_not_suppress(tree) -> None:
    """实体读不到坐标时**不过滤** —— 宁可多按一次，也不要永久失明。"""

    class _NoPos:
        kind = group = "chest"
        bbox = (0.5, 0.4, 0.1, 0.1)

    tree.last_world_entities = (_Chest(),)
    tree._test_clock["t"] = 1_000_000
    assert tree.decide(_state(1)).hit is True

    tree.last_world_entities = (_NoPos(),)
    tree._test_clock["t"] = 1_000_000 + 200
    print("\n  无坐标实体不应被标记压住")
    assert tree.decide(_state(2)).hit is True


@pytest.mark.unit
def test_pickmark__is_frozen_dataclass_value() -> None:
    """标记必须是不可变的值对象 —— 它被多个 PolicyContext 共享。"""
    m = PickMark(group="chest", x=1.0, y=2.0, ts_ms=3)
    with pytest.raises(Exception):
        m.x = 5.0  # type: ignore[misc]