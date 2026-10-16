"""反射层补齐的单元测试（2026-10-11）。

覆盖三件新东西：
1. ``PolicyContext.has_nearby_drops()`` —— DSL 侧新注册的掉落物谓词
2. ``ActionType.USE_PICKUP`` —— 动作白名单 / kind / 默认键位 / traits 四处接线
3. 「USEFOOD / REST 故意不做」这条决定本身

第 3 条看起来奇怪，但它是这份文件里**最该被守住**的一条：将来有人
「顺手把吃药也加上」时，这里会红，而且红得莫名其妙。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.executor import DEFAULT_KEYMAP  # noqa: E402
from autopilot.actuator.simulator import VK  # noqa: E402
from autopilot.actuator.traits import TRAITS  # noqa: E402
from autopilot.contracts.intent_packet import (  # noqa: E402
    _ACTION_KIND_MAP,
    ActionType,
)
from autopilot.policy.context import PolicyContext  # noqa: E402

# --------------------------------------------------------------- 夹具


class _FakeEntity:
    def __init__(self, group: str, bbox=None) -> None:
        self.kind = group
        self.group = group
        self.bbox = bbox


def _ctx(entities):
    """PolicyContext 是 frozen dataclass —— 只能构造，不能事后赋值。

    ``state`` / ``now_ms`` 是必填；这里造一个最小的 StatePacket 就够，
    本组用例只碰世界实体那条侧信道。
    """
    from autopilot.contracts.state_packet import StatePacket

    state = StatePacket.unknown(game_id="genshin", state_version=1,
                                reason="test")
    return PolicyContext(state=state, now_ms=0, world_entities=entities)


# --------------------------------------------------------------- 1. has_nearby_drops


@pytest.mark.unit
def test_drops__registered_in_dsl() -> None:
    """不注册就等于没有 —— 规则里写 has_nearby_drops() 会直接加载失败。"""
    from autopilot.policy.context import FUNCTION_NAMES

    assert "has_nearby_drops" in FUNCTION_NAMES, (
        "has_nearby_drops 没进 DSL 白名单，规则用不了它")


@pytest.mark.unit
def test_drops__true_when_present() -> None:
    c = _ctx([_FakeEntity("drops")])
    assert c.has_nearby_drops() is True


@pytest.mark.unit
def test_drops__false_when_none() -> None:
    c = _ctx([_FakeEntity("enemy")])
    assert c.has_nearby_drops() is False


@pytest.mark.unit
def test_drops__unknown_when_detector_absent() -> None:
    """没有实体信息时必须返回 Unknown 而不是 False。

    返回 False 会让「检测器没注入」和「附近确实没有掉落」变成同一种信号，
    规则就会在信息缺失时贸然按 F。
    """
    c = _ctx(None)
    got = c.has_nearby_drops()
    assert not isinstance(got, bool), (
        f"无实体信息时应返回 Unknown，实际返回 {got!r}")


@pytest.mark.unit
def test_drops__does_not_mix_with_other_groups() -> None:
    """矿/宝箱/掉落是三个独立分组 —— 别把 chest 当成 drops。"""
    c = _ctx([_FakeEntity("chest"), _FakeEntity("ore")])
    assert c.has_nearby_drops() is False
    assert c.has_nearby_chest() is True
    assert c.has_nearby_ore() is True


# --------------------------------------------------------------- 2. USE_PICKUP 接线


@pytest.mark.unit
def test_use_pickup__maps_to_f_key() -> None:
    """原神 PC：拾取/采集/开箱/对话共用 F。键错了整套采集逻辑全废。"""
    assert DEFAULT_KEYMAP["USE_PICKUP"] == VK.F
    assert int(VK.F) == 0x46


@pytest.mark.unit
def test_use_pickup__kind_and_traits_registered() -> None:
    assert ActionType.USE_PICKUP in _ACTION_KIND_MAP
    assert _ACTION_KIND_MAP[ActionType.USE_PICKUP] == "UI"
    t = TRAITS["USE_PICKUP"]
    assert t.allowed_scenes == ("WORLD", "UNKNOWN")
    assert t.holds_keys is False, "F 是点按；长按采集是另一件事"
    assert t.min_gap_ms >= 500, "别对同一个箱子连按"


@pytest.mark.unit
def test_use_pickup__every_action_type_has_keymap_and_traits() -> None:
    """白名单里的每个动作都必须有键位和 traits，否则运行期才炸。"""
    missing_key = [a.value for a in ActionType if a.value not in DEFAULT_KEYMAP]
    missing_traits = [a.value for a in ActionType if a.value not in TRAITS]
    assert not missing_key, f"这些动作没有默认键位：{missing_key}"
    assert not missing_traits, f"这些动作没有 traits：{missing_traits}"


# --------------------------------------------------------------- 3. 故意不做的两个


@pytest.mark.unit
def test_use_food_and_rest_are_absent_by_design() -> None:
    """USE_FOOD / REST 是**查证后判定不可靠**才不做的，不是漏了。

    * USE_FOOD：原神 PC 没有食物快捷键，必须开背包点 UI，还要先决定
      「背包里哪个食物」—— 当前动作模型是「一个动作 = 一个键」，塞不下。
    * REST：原神**根本没有休息回体力这个机制**，原粹树脂只按 8 分钟/点
      自然恢复，且不受在线状态影响。等体力 = WAIT。
    """
    names = {a.name for a in ActionType}
    assert "USE_FOOD" not in names
    assert "REST" not in names
    assert "USE_PICKUP" in names


@pytest.mark.unit
def test_world_rules_are_loadable_and_wired() -> None:
    """genshin_world.toml 三条规则能加载，且动作都在白名单里。

    ★ 2026-10-11：这条用例原先断言「三条全部 enabled = false」，守的是
      「真机没验过之前不许打开」。用户决定**现在开始实测**，所以守卫的
      前提没了 —— 断言随之改成「能加载 + 动作合法」。

      仍然保留的那部分守卫是**动作合法性**：动作不在 ActionType 白名单里，
      规则会在加载期直接失败（不是运行期才炸），所以必须静态守住。
    """
    import tomllib

    from autopilot.tools._stack import build_policy

    p = PLUGIN_ROOT / "autopilot" / "rules" / "genshin_world.toml"
    assert p.is_file(), "genshin_world.toml 不见了"
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    rules = data["rules"]
    assert rules, "规则表是空的"
    for r in rules:
        assert r["action"] in {a.value for a in ActionType}, (
            f"{r['name']} 的动作 {r['action']} 不在 ActionType 白名单里，"
            f"规则加载期就会失败")

    # 真加载一遍，别只信 TOML 文本
    _src, rules_dir, count = build_policy(None)
    assert count >= len(rules), f"只加载到 {count} 条，少于文件里的 {len(rules)} 条"
    names = {getattr(r, "name", None) for r in getattr(_src, "rules", ())}
    for r in rules:
        assert r["name"] in names, f"{r['name']} 没被策略树加载到（目录 {rules_dir}）"