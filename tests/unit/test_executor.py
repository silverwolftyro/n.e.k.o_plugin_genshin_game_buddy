"""执行层的键位标签：``KeyId`` 没有 ``.name``（2026-10-19 实机崩在这）。

★ 这个 bug 长什么样
  实机跑 ``--live`` 时，``USE_BURST`` / ``USE_SKILL`` 连续报

      AttributeError: 'KeyId' object has no attribute 'name'

  报满 5 次触发熔断器，整条连招被掐断，引擎从此空转。

★ 根因
  ``.name`` 是**枚举成员**才有的属性。键位值有两种来源：

    · 内置 ``DEFAULT_KEYMAP``  -> ``VK`` 成员（IntEnum，**有** .name）
    · ``load_keymap()`` 读文件 -> ``KeyId``（int 子类，**没有** .name）

  之前只有前者被走到过 —— ``run_loop`` 从不把 ``keymap_path`` 传给
  ``build_stack``，执行器一直用内置默认表。2026-10-19 那个修复让
  keymap 真的被读进来，于是第一次拿到 ``KeyId``，``executor.py`` 里
  那一排 ``key_id.name`` 立刻炸。

  也就是说：**这不是新代码写错了，是一条从未被走到的分支**。
  写死的角色名、读文件加载的键位表，这两个特性此前从未同时生效过。

★ 为什么单测要专门盯这条
  报错的现场是「按键发不出去」，而按键发不出去有**太多**别的原因
  （权限、焦点、窗口、熔断）。这个 AttributeError 藏在日志里，
  很容易被当成「引擎没打中」。所以单测直接断言**不抛**，
  并且断言**动作真的走到了注入那一步**（steps 非空）。
"""

from __future__ import annotations

import time

import pytest

from autopilot.actuator.executor import (
    DEFAULT_KEYMAP,
    IntentExecutor,
    _key_label,
)
from autopilot.actuator.keymap import load_keymap
from autopilot.actuator.safety import SafetyConfig, SafetyGuard
from autopilot.actuator.simulator import KeyId, VK
from autopilot.contracts.intent_packet import ActionType, IntentPacket
from autopilot.contracts.state_packet import (
    SCENE_WORLD,
    PlayerInfo,
    ResourceInfo,
    SceneInfo,
    StatePacket,
)

pytestmark = pytest.mark.unit

#: 执行器会为这些动作读键位表。每一个都曾经走过 ``key_id.name``。
KEYED_ACTIONS = (
    ActionType.USE_BURST,
    ActionType.USE_SKILL,
    ActionType.NORMAL_ATTACK,
    ActionType.SPRINT,
    ActionType.JUMP,
    ActionType.DODGE,
    ActionType.USE_PICKUP,
)


def _state() -> StatePacket:
    return StatePacket(
        scene=SceneInfo(id=SCENE_WORLD, confidence=0.9),
        player=PlayerInfo(party=(), hp_ratio=1.0, hp_source="unit"),
        resource=ResourceInfo(perception_ok=True),
        state_version=1,
    )


def _executor(**kw) -> IntentExecutor:
    return IntentExecutor(dry_run=True, guard=SafetyGuard(SafetyConfig()), **kw)


def _run(ex: IntentExecutor, action: ActionType):
    intent = IntentPacket.make(action, params={}, state_version=1)
    return ex.execute(intent, _state(), now_ms=int(time.time() * 1000))


# ==================================================================
# 键位表里的两种类型
# ==================================================================

def test_default_keymap_values_are_vk_members() -> None:
    """内置默认表给的是 ``VK`` 成员（枚举，有 .name）。

    留这一条是为了把「两种来源」写进测试里 ——
    修完之后执行器内部已经统一成 KeyId，但**来源**仍是两种。
    """
    assert isinstance(DEFAULT_KEYMAP["USE_BURST"], VK)


def test_loaded_keymap_values_are_keyid_without_name() -> None:
    """★ 读文件得到的是 ``KeyId``，它**没有** ``.name``。

    这就是崩的根因 —— 把它钉成断言，以后谁把 KeyId 改成带 .name 的
    类型，这个测试会提醒：依赖点已经变了。
    """
    kid = KeyId.from_name("Q")
    assert isinstance(kid, KeyId)
    assert not hasattr(kid, "name")
    assert kid.display_name == "Q"


# ==================================================================
# 标签函数
# ==================================================================

def test_key_label__works_for_keyid() -> None:
    assert _key_label(KeyId.from_name("Q")) == "Q"


def test_key_label__works_for_vk_member() -> None:
    """两种类型都要能过 —— 这是这个函数存在的全部理由。"""
    assert _key_label(VK.Q) == "Q"


def test_key_label__works_for_plain_int() -> None:
    assert _key_label(int(VK.E)) == "E"


def test_key_label__sentinels() -> None:
    assert _key_label(KeyId.NONE) == "NONE"
    assert _key_label(KeyId.UNKNOWN) == "UNKNOWN"


def test_key_label__unknown_code_does_not_raise() -> None:
    """查不到键名也要给个能看的东西，不能抛。"""
    assert _key_label(0x1234)


# ==================================================================
# ★ 回归：执行动作不许抛 AttributeError
# ==================================================================

@pytest.mark.parametrize("action", KEYED_ACTIONS, ids=lambda a: a.value)
def test_execute__keyed_action_never_raises_attribute_error(action) -> None:
    """★ 用**文件加载**的键位表跑每一个会读键的动作。

    修之前：``AttributeError: 'KeyId' object has no attribute 'name'``。
    修之后：动作正常走完，outcome 是 OK。
    """
    km = load_keymap()          # 读插件内那份 autopilot/config/keymap.toml
    assert any(isinstance(v, KeyId) for v in km.values()), \
        "这份夹具必须真的是 KeyId —— 否则测的不是这条 bug"

    ex = _executor(keymap=km)
    try:
        result = _run(ex, action)
    except AttributeError as exc:      # pragma: no cover - 修完不该走到
        pytest.fail(f"{action.value} 抛了 AttributeError：{exc}")
    assert result.outcome.value == "OK", (
        f"{action.value} 没走通：{result.outcome.value} {result.reason}")


@pytest.mark.parametrize("action", KEYED_ACTIONS, ids=lambda a: a.value)
def test_execute__keyed_action_reaches_the_injection(action) -> None:
    """★ 不只是「不抛」—— 还要**真的按出去了**。

    只断言「不抛」是不够的：报错可能被吞掉、动作可能被静默跳过，
    那样连招还是打不出来。steps 非空 = 走到了 simulator 注入。
    """
    km = load_keymap()
    ex = _executor(keymap=km)
    result = _run(ex, action)
    assert result.steps, f"{action.value} 没有任何注入步骤：{result.reason}"


def test_execute__burst_and_skill_press_the_right_keys() -> None:
    """★ 精确断言：Q / E。

    标签取错键名（比如把 Q 显示成 E）也意味着这里查错了键。
    """
    km = load_keymap()
    ex = _executor(keymap=km)
    burst = _run(ex, ActionType.USE_BURST)
    skill = _run(ex, ActionType.USE_SKILL)
    assert any("Q" in s for s in burst.steps), burst.steps
    assert any("E" in s for s in skill.steps), skill.steps


# ==================================================================
# 键位表类型统一
# ==================================================================

def test_executor_normalizes_keymap_to_keyid() -> None:
    """★ 执行器内部的键位表**只有一种类型**。

    混着 ``VK`` 和 ``KeyId`` 就是这次崩溃的土壤：下游只能靠
    「这个值恰好是哪个」来判断，于是写 ``key_id.name`` 的人
    在读文件那条路上必然踩空。
    """
    ex = _executor(keymap=DEFAULT_KEYMAP)   # 全是 VK
    assert all(isinstance(v, KeyId) for v in ex.keymap.values())

    ex2 = _executor(keymap=load_keymap())   # 全是 KeyId
    assert all(isinstance(v, KeyId) for v in ex2.keymap.values())


def test_resolve_key__none_sentinel_is_rejected_by_value() -> None:
    """★ 显式 NONE 必须被拦下。

    ``KeyId.NONE`` 是类体里的普通 int（0），而
    ``KeyId.from_name("NONE")`` 造出来的是 ``KeyId(0)`` ——
    两者 ``==`` 但 ``is`` 为 False。所以判 NONE **必须按值比**：

        >>> KeyId.from_name("NONE") is KeyId.NONE
        False

    写成 ``is`` 的话这个判断对文件加载的键位**永远不成立**，
    动作会一路掉到 ``to_vk()`` 才炸 —— 报错信息差一大截，
    而「这个动作就是不发按键」这个本意完全看不出来。
    """
    ex = _executor(keymap={"WAIT": KeyId.from_name("NONE")})
    with pytest.raises(Exception) as ei:
        ex.resolve_key("WAIT")
    assert "NONE" in str(ei.value)
