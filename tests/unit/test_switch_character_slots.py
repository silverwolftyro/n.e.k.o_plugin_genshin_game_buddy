"""``SWITCH_CHARACTER`` 的 char_id -> slot 解析（2026-10-13 修 bug）。

★ 这个 bug 长什么样
  ``_slot_of()`` 读 ``getattr(self, "_char_slots", {})``，
  而**全插件没有任何一处给 ``_char_slots`` 赋值** —— 它恒为 ``{}``，
  于是 ``_slot_of()`` 恒返回 None，走
  ``executor.py`` 的「跳过（无可切的位）」分支：

      规则判定「该切人」 -> 执行器收到意图 -> **什么都没做** -> 不报错

  症状是「AI 偶尔不切人」，看起来像规则问题或游戏状态问题；
  根因（一个从未被赋值的属性）却隔了意图生成 -> 执行器 -> keymap 三层。

★ 另一层「死代码」
  旧代码还写了 ``self.keymap.get("__char_slots__")`` —— 但
  ``load_keymap()`` **只接受** ``DEFAULT_KEYMAP`` 里的动作名，未知名字
  一律 ``KeymapError``（拼错要当场报错是故意的）。所以那个键永远进不来，
  赋了值从没读过。
"""

from __future__ import annotations

import pytest

from autopilot.actuator.executor import IntentExecutor
from autopilot.actuator.keymap import KeymapError, load_char_slots

pytestmark = pytest.mark.unit


def _ex(**kw) -> IntentExecutor:
    return IntentExecutor(dry_run=True, **kw)


# ==================================================================
# _slot_of
# ==================================================================

def test_slot_of__resolves_a_configured_character() -> None:
    assert _ex(char_slots={"raiden": 2})._slot_of("raiden") == 2


def test_slot_of__unknown_character_returns_none() -> None:
    """查不到就说不知道 —— 不能猜一个默认槽位（那会切错人）。"""
    assert _ex(char_slots={"raiden": 2})._slot_of("barbara") is None


def test_slot_of__no_config_returns_none_instead_of_crashing() -> None:
    """★ 没配编队时必须返回 None，而不是抛异常或返回 1。

    这是本 bug 修复前后的分水岭：以前返回的是 None 但**原因错了**
    （读一个空属性），现在返回 None 是因为**真的没配**。
    """
    assert _ex()._slot_of("raiden") is None


def test_slot_of__string_slot_is_coerced() -> None:
    """配置里写成字符串也要能用（用户手改 TOML 的常见情形）。"""
    ex = _ex()
    ex.set_char_slots({"raiden": "3"})       # type: ignore[dict-item]
    assert ex._slot_of("raiden") == 3


def test_slot_of__garbage_slot_returns_none() -> None:
    ex = _ex()
    ex._char_slots["raiden"] = "abc"        # type: ignore[assignment]
    assert ex._slot_of("raiden") is None


def test_set_char_slots__replaces_the_mapping() -> None:
    ex = _ex(char_slots={"raiden": 2})
    ex.set_char_slots({"barbara": 1})
    assert ex._slot_of("raiden") is None
    assert ex._slot_of("barbara") == 1


def test_set_keymap__does_not_wipe_char_slots() -> None:
    """★★ 两张表必须互相独立。

    旧代码把 ``__char_slots__`` 混进 keymap，这条断言就是钉住那个
    设计错误的反面：换键位不该让人重新配编队。
    """
    ex = _ex(char_slots={"raiden": 2})
    ex.set_keymap({"USE_SKILL": ex.resolve_key("USE_SKILL")})
    assert ex._slot_of("raiden") == 2


# ==================================================================
# 端到端：SWITCH_CHARACTER 不再被静默丢弃
# ==================================================================

def _switch_intent(char_id: str, slot=None):
    from autopilot.contracts.intent_packet import ActionType, IntentPacket

    params = {"char_id": char_id}
    if slot is not None:
        params["slot"] = slot
    return IntentPacket(action=ActionType.SWITCH_CHARACTER, params=params)


def _steps(ex: IntentExecutor, intent) -> list[str]:
    """跑那条真正决定「按哪个数字键」的分支。

    ``execute()`` 里会先过校验 / 熔断 / 急停 / 焦点四道门，
    每道门各有先决条件；本组要验的是**槽位解析**这一步本身，
    所以直接调 ``_run_sequence``（``execute`` 内部也是用它）。
    它还需要一个 ``ExecutionResult`` 当 deadline 基准 —— 手工造一个，
    ``start_mono`` 给当前时刻，避免一进去就超时。
    """
    import time as _time

    from autopilot.actuator.executor import ExecutionResult, Outcome
    from autopilot.actuator.traits import traits_of

    res = ExecutionResult(outcome=Outcome.OK, reason="", intent_id="t",
                          action=intent.action.value, started_ms=0)
    res.start_mono = _time.monotonic()
    return ex._run_sequence(intent, traits_of(intent.action.value),
                            ex.resolve_key("SWITCH_CHARACTER"), res)


def test_switch__with_slot_configured_is_executed() -> None:
    """★ 本次修复的核心断言：配了编队就真的会切。"""
    ex = _ex(char_slots={"raiden": 2})
    joined = " ".join(_steps(ex, _switch_intent("raiden")))
    assert "跳过" not in joined, joined
    assert "slot 2" in joined, joined


def test_switch__explicit_slot_still_works_without_config() -> None:
    """意图自带 slot 时不需要编队配置 —— 那条路本来就没坏。"""
    ex = _ex()
    joined = " ".join(_steps(ex, _switch_intent("raiden", slot=2)))
    assert "跳过" not in joined, joined
    assert "slot 2" in joined, joined


def test_switch__unresolvable_is_skipped_but_recorded() -> None:
    """★ 仍然跳过的场景必须**留下痕迹**。

    这不是「静默」：steps 里明写「跳过（无可切的位）」，
    排障时能看到 AI 为什么没切人。真正的静默是「什么都没发生且无从查起」。
    """
    ex = _ex()
    joined = " ".join(_steps(ex, _switch_intent("raiden")))
    assert "跳过" in joined and "无可切的位" in joined, joined


# ==================================================================
# [char_slots] 配置加载
# ==================================================================

def test_load_char_slots__missing_file_is_empty(tmp_path) -> None:
    assert load_char_slots(tmp_path / "不存在.toml") == {}


def test_load_char_slots__missing_section_is_empty(tmp_path) -> None:
    f = tmp_path / "keymap.toml"
    f.write_text('[keymap]\nUSE_SKILL = "E"\n', encoding="utf-8")
    assert load_char_slots(f) == {}


def test_load_char_slots__reads_the_table(tmp_path) -> None:
    f = tmp_path / "keymap.toml"
    f.write_text('[char_slots]\nraiden = 2\nbarbara = 1\n', encoding="utf-8")
    assert load_char_slots(f) == {"raiden": 2, "barbara": 1}


def test_load_char_slots__out_of_range_is_rejected(tmp_path) -> None:
    """★ 队伍只有 4 个位置，5 是写错了 —— 响亮失败比静默跳过来得好。"""
    f = tmp_path / "keymap.toml"
    f.write_text("[char_slots]\nraiden = 5\n", encoding="utf-8")
    with pytest.raises(KeymapError, match="1~4"):
        load_char_slots(f)


def test_load_char_slots__non_integer_is_rejected(tmp_path) -> None:
    f = tmp_path / "keymap.toml"
    f.write_text('[char_slots]\nraiden = "第二个"\n', encoding="utf-8")
    with pytest.raises(KeymapError):
        load_char_slots(f)


def test_load_char_slots__broken_toml_is_tolerated(tmp_path) -> None:
    """键位表本身已经由 ``load_keymap`` 报过错了，这里不重复刷屏。"""
    f = tmp_path / "keymap.toml"
    f.write_text("这不是 TOML [[[", encoding="utf-8")
    assert load_char_slots(f) == {}


def test_load_char_slots__feeds_the_executor(tmp_path) -> None:
    """端到端：配置 -> 注入 -> 真的切人。"""
    f = tmp_path / "keymap.toml"
    f.write_text("[char_slots]\nraiden = 2\n", encoding="utf-8")
    ex = _ex(char_slots=load_char_slots(f))
    assert "slot 2" in " ".join(_steps(ex, _switch_intent("raiden")))
