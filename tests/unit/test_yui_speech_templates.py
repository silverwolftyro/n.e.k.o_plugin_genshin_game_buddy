"""YUI 台词渲染的占位符契约（2026-10-13）。

★ 一段差点被当成 bug 的东西
  ``EVENT_TO_SPEECH["ENEMY_DETECTED"] = "有怪！{count}个，小心点"``，
  而某次离线检查里渲染出的是 ``"有怪！个，小心点"`` —— ``{count}`` 消失了。

  查下来**生产代码是对的**：

  * 写方 :meth:`Event.enemy_detected` 构造的 context 一定含
    ``{"count": int(count)}``（``contracts/events.py``）；
  * 带真实 context 渲染 -> ``"有怪！3个，小心点"``。

  断句出现在**手工构造、没有 context 的合成事件**上 —— 也就是测试/演示自己
  造的假数据。所以这里**不改模板**，而是把两种行为都钉成断言：
  真实路径必须填上占位符；缺 context 时退化成什么样，也要写明，
  免得下次又有人把它当成 bug 去"修"。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autopilot.contracts.events import Event
from event_to_yui import (BUSINESS_CODES, EVENT_TO_SPEECH, EventToYui)

pytestmark = pytest.mark.unit


def _render(payloads: list[dict]) -> list[str]:
    """把事件写进真格式的流，再让真 ``EventToYui`` 追尾渲染。"""
    import tempfile

    said: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        rt = Path(td)
        (rt / "run_current.txt").write_text("t", encoding="utf-8")
        with (rt / "t.stream.jsonl").open("w", encoding="utf-8") as fh:
            for p in payloads:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        EventToYui(
            push_fn=lambda **kw: (
                said.append("".join(x.get("text", "") for x in kw["parts"]))
                or {"submitted": True}),
            runtime_dir=rt).drain_once()
    return said


# ==================================================================
# ★ 占位符确实会被替换（真实路径）
# ==================================================================

def test_enemy_detected__count_is_substituted_with_a_real_context() -> None:
    """★★ 真实事件路径：``{count}`` 一定被填上。

    写方 :meth:`Event.enemy_detected` 恒带 ``count``，所以线上不会断句。
    """
    got = _render([{"code": "ENEMY_DETECTED",
                    "message": "附近有 3 个敌人",
                    "context": {"count": 3, "nearest_m": 120.5}}])
    assert got == ["有怪！3个，小心点"], got


def test_position_changed__dist_is_substituted() -> None:
    got = _render([{"code": "POSITION_CHANGED", "message": "x",
                    "context": {"x": 1.0, "z": 2.0, "dist_m": 30.0}}])
    assert got == ["跑了30.0米了"], got


def test_count_zero_is_dropped_instead_of_saying_zero_enemies() -> None:
    """★★ ``count=0`` **不播报**（2026-10-16 改）。

    ⚠️ 这条断言**方向在本轮翻了过来**。原断言是「``count=0`` 是有效值，
    不能被当成缺失而清空」，渲染出 ``有怪！0个，小心点``。

    那是 demo 出厂前粗糙度审计的 A1 级问题：真机 2026-10-14 run 的
    ``ENEMY_DETECTED`` count 序列是 ``0→2→1→1→1→0``，两次 count=0
    都会原样播出去——**「有怪！0个，小心点」**。这是全片唯一的亮点台词，
    说自己看见 0 个怪，比不说伤害大得多。

    所以现在区分两种 0：

    * ``count=0``（**明说有 0 个**）-> 不播。观众不需要被告知「没怪」。
    * ``count`` **缺失**（见下一个测试）-> 仍走退化路径，
      因为那是数据源问题，要修写方而不是改模板。

    两者刻意不同：一个是「说的内容是废话」，一个是「说的内容是残句」，
    后者由 ``_SafeDict`` 兜住，不在本次范围内。
    """
    got = _render([{"code": "ENEMY_DETECTED", "message": "没有敌人",
                    "context": {"count": 0}}])
    assert got == [], got


def test_negative_count_is_also_dropped() -> None:
    """★ count 为负（异常值）同样不播 —— 「有怪！-1个」比 0 更荒唐。"""
    got = _render([{"code": "ENEMY_DETECTED", "message": "?",
                    "context": {"count": -1}}])
    assert got == [], got


def test_count_zero_does_not_consume_the_repeat_budget() -> None:
    """★ count=0 被压掉**不该**占用重复上限的配额。

    否则一次故障里几十条 ``count=0`` 会把「有怪」这条的预算吃光，
    真看见 3 只怪时反而播不出来。
    """
    assert _render([
        {"code": "ENEMY_DETECTED", "message": "0", "context": {"count": 0}},
        {"code": "ENEMY_DETECTED", "message": "3", "context": {"count": 3}},
    ]) == ["有怪！3个，小心点"]


def test_producer_always_sets_count() -> None:
    """★ 从**写方**再确认一次：契约不是靠约定，而是代码保证。"""
    ev = Event.enemy_detected(count=5, nearest_m=10.0)
    assert ev.context["count"] == 5
    assert ev.context["nearest_m"] == 10.0


def test_producer_omits_nearest_m_when_unknown() -> None:
    ev = Event.enemy_detected(count=2, nearest_m=-1.0)
    assert ev.context["count"] == 2
    assert "nearest_m" not in ev.context, "距离不可知时不该塞一个假值"


# ==================================================================
# 已知退化：缺 context 时占位符变空串（**当前行为，不是 bug**）
# ==================================================================

def test_missing_context_degrades_to_a_grammatical_hole() -> None:
    """★ 缺 context -> ``_SafeDict`` 回空串 -> ``"有怪！个，小心点"``。

    这是**已知退化**，不是缺陷：只有手工造的事件才会走到这里
    （真实写方恒带 ``count``）。把它钉住，是为了下次再看到时
    **知道该修数据源、而不是去改模板**。
    """
    got = _render([{"code": "ENEMY_DETECTED", "message": "测试"}])
    assert got == ["有怪！个，小心点"], got


def test_every_business_template_renders_without_raising() -> None:
    """★ 任何业务码、任何 context，渲染都不许抛。"""
    for code in BUSINESS_CODES:
        assert _render([{"code": code, "message": "m", "context": {}}]) is not None


def test_unknown_code_falls_back_without_raising() -> None:
    """★ 表外 code 必须走兜底而不是崩 —— EventCode 是开放集合。"""
    assert _render([{"code": "TOTALLY_NEW_CODE", "message": "某件事"}]) is not None


def test_templates_never_leave_a_visible_placeholder() -> None:
    """★★ 所有模板都不允许在正常渲染后露出 ``{xxx}``。"""
    ctx = {"count": 3, "dist_m": 30.0, "x": 1.0, "z": 2.0, "nearest_m": 9.0,
           "message": "消息"}
    for code, tpl in EVENT_TO_SPEECH.items():
        if code in ("ENEMY_DETECTED", "POSITION_CHANGED"):
            continue
        try:
            text = tpl.format_map(type("D", (dict,),
                                      {"__missing__": lambda s, k: ""})(ctx))
        except (IndexError, ValueError):
            pytest.fail(f"{code} 的模板渲染失败：{tpl!r}")
        assert "{" not in text, (code, tpl, text)
