"""2026-11-10 接线：known_suppress 接到主动提示层（"玩家已知的话题不重复主动提"）。

背景
----
``terms/genshin/scenes.toml`` 的 18 个攻略场景每个都带 ``known_suppress = true``：
"这个场景的攻略内容一旦给玩家看过，就别再主动提"。上一轮（2026-11-09）只把它做成
**可读可查**（``scene_store.SceneEntry``），这一轮接到主动提示层，形成闭环::

    记录侧（场景内容注入成功）──▶ KnownSuppressRegistry ──▶ 消费侧（主动提示决策）

本文件钉住 6 组契约：

A. 登记簿纯单元   —— 窗口内 True / 窗口外 False / known_suppress=False 恒 False
                     / 窗口关闭恒 False / 剪枝 / forget / clear
B. 配置解析       —— 默认 600 / 0=关闭 / clamp [0, 86400] / 非整数回落默认
C. 消费侧         —— ``_should_push_proactive`` 命中抑制 → False + INFO 日志
                     + **不更新** ``_last_pushed_hash``（否则窗口过后同文本被永久挡死）
D. 记录侧         —— ``_record_scene_injection`` 取 ``SceneEntry.known_suppress`` 快照
                     —— ``_push_context`` 注入成功后才登记（失败不登记）
E. 用户提问路径   —— 注入过后用户主动提问，术语卡仍推（抑制只管"主动"路径）
F. 端到端证据     —— 同场景连续注入两次 → 第二次主动提示被抑制；用户手动提问仍应答

⚠️ 关于"引擎场景 id" vs "攻略场景 id"
   ``chat_state.py`` / ``event_to_yui.py`` 用的是**引擎场景**（WORLD/COMBAT/IDLE/UNKNOWN），
   与 ``scenes.toml`` 的**攻略场景**（深境螺旋等）是两套不相交的 id 空间。
   把攻略场景 id 塞进引擎场景空间会造成真实误判，所以消费侧接在
   ``__init__.py::_should_push_proactive``（攻略场景的主动路径）上。
"""

from __future__ import annotations

import threading
import time
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from conftest import FakeLogger
from plugin.plugins.genshin_game_buddy import GenshinGameBuddyPlugin
from plugin.plugins.genshin_game_buddy.game_registry import PluginOptions
from plugin.plugins.genshin_game_buddy.known_suppress import (
    DEFAULT_WINDOW_SECONDS,
    KnownSuppressRegistry,
)
from plugin.plugins.genshin_game_buddy.scene_store import SceneEntry
from plugin.plugins.genshin_game_buddy.session import GameSession
from plugin.plugins.genshin_game_buddy.term_store import TermEntry

#: scenes.toml 里用作"真接线"断言的场景（known_suppress = true）。
PROBE_SCENE = "深境螺旋"

#: core.toml 里的真实术语键——用户提问路径用它做命中。
PROBE_TERM = "元素反应"

#: characters.toml 里的真实术语键。
#: ⚠️ 端到端用例必须用它（而不是 PROBE_TERM）问"术语卡有没有推出来"：
#:   core 维度的术语在 ``set_game`` 时就随语境块激活了（``_scan_and_activate_query``
#:   的契约是"已激活的不重推"），拿它提问不会产生新的术语卡。角色维度的术语
#:   不进自动语境块，问它才会走"新命中 → 推卡"这条路径。
PROBE_CHARACTER = "纳西妲"


# =============================================================================
# 夹具
# =============================================================================


def _make_stub(*, window: int = 600, logger: FakeLogger | None = None) -> GenshinGameBuddyPlugin:
    """最小可用 stub——只填主动提示决策与记录侧需要的字段。

    不调 ``__init__``（要 ctx），用 ``__new__`` 跳过。形状与
    ``tests/unit/test_scene_switch.py`` 的 stub 保持一致。
    """
    plugin = GenshinGameBuddyPlugin.__new__(GenshinGameBuddyPlugin)
    plugin._options = PluginOptions(known_suppress_window_s=window)
    plugin._last_pushed_hash = ""
    plugin._last_proactive_at = 0.0
    plugin._last_scene_switch_at = 0.0
    plugin._last_scene_for_proactive = ""
    plugin._manager = MagicMock()
    plugin._manager.library = None  # 跳过术语密度检查
    plugin.logger = logger or FakeLogger()
    plugin._known_suppress = KnownSuppressRegistry(window_seconds=float(window))
    plugin._scene_store = {}
    return plugin


def _make_query_stub(*, window: int = 600) -> GenshinGameBuddyPlugin:
    """能驱动用户提问路径（``on_chat_message`` → ``_scan_and_activate_query``）的 stub。"""
    plugin = _make_stub(window=window)
    plugin._state_lock = threading.Lock()
    plugin._activation = MagicMock()
    plugin._activation.get_active_keys = MagicMock(return_value=[])
    plugin._manager.current = GameSession(game_id="genshin", display_name="原神")
    entry = TermEntry(key=PROBE_TERM, kind="机制", brief="元素反应机制说明")
    library = MagicMock()
    library.by_key = {PROBE_TERM: entry}
    library.get = MagicMock(return_value=entry)
    plugin._manager.library = library
    plugin.push_message = MagicMock(return_value={"submitted": True})
    return plugin


# =============================================================================
# A. 登记簿纯单元
# =============================================================================


@pytest.mark.unit
def test_registry__within_window_suppresses() -> None:
    """窗口内被注入过 → 应抑制。"""
    registry = KnownSuppressRegistry(window_seconds=600.0)
    registry.record(PROBE_SCENE, known_suppress=True, now=1000.0)
    assert registry.should_suppress(PROBE_SCENE, now=1000.0) is True
    assert registry.should_suppress(PROBE_SCENE, now=1000.0 + 599.0) is True


@pytest.mark.unit
def test_registry__after_window_allows_again() -> None:
    """超过窗口 → 不再抑制（玩家可以重新被提醒）。"""
    registry = KnownSuppressRegistry(window_seconds=600.0)
    registry.record(PROBE_SCENE, known_suppress=True, now=1000.0)
    assert registry.should_suppress(PROBE_SCENE, now=1000.0 + 600.0) is False
    assert registry.should_suppress(PROBE_SCENE, now=1000.0 + 3600.0) is False


@pytest.mark.unit
def test_registry__known_suppress_false_never_suppresses() -> None:
    """场景自己标了 known_suppress=False → 恒不抑制（标记才是"已知"的唯一依据）。"""
    registry = KnownSuppressRegistry(window_seconds=600.0)
    registry.record("普通场景", known_suppress=False, now=1000.0)
    assert registry.should_suppress("普通场景", now=1000.0) is False
    assert registry.should_suppress("普通场景", now=1000.0 + 10.0) is False


@pytest.mark.unit
def test_registry__window_zero_disables_suppression() -> None:
    """窗口 = 0 → 关闭抑制（回到改造前行为）。"""
    registry = KnownSuppressRegistry(window_seconds=0.0)
    registry.record(PROBE_SCENE, known_suppress=True, now=1000.0)
    assert registry.should_suppress(PROBE_SCENE, now=1000.0) is False


@pytest.mark.unit
def test_registry__negative_window_becomes_zero() -> None:
    """负窗口没有意义 → 归零。"""
    registry = KnownSuppressRegistry(window_seconds=-5.0)
    assert registry.window_seconds == 0.0


@pytest.mark.unit
def test_registry__unknown_scene_never_suppresses() -> None:
    """没注入过的场景 → 不抑制。"""
    registry = KnownSuppressRegistry(window_seconds=600.0)
    assert registry.should_suppress("没注入过的场景", now=1000.0) is False


@pytest.mark.unit
def test_registry__empty_scene_id_ignored() -> None:
    """空 scene_id 不进登记簿（防止把"无场景"记成一次注入）。"""
    registry = KnownSuppressRegistry(window_seconds=600.0)
    assert registry.record("", known_suppress=True, now=1000.0) is False
    assert registry.record("   ", known_suppress=True, now=1000.0) is False
    assert registry.should_suppress("", now=1000.0) is False


@pytest.mark.unit
def test_registry__prunes_expired_records() -> None:
    """窗口外的旧记录被剪掉——登记簿保持有界，不会随会话无限增长。"""
    registry = KnownSuppressRegistry(window_seconds=600.0)
    registry.record("旧场景", known_suppress=True, now=1000.0)
    registry.record("新场景", known_suppress=True, now=2000.0)
    assert registry.last_injected_at("旧场景") is None
    assert registry.last_injected_at("新场景") == 2000.0


@pytest.mark.unit
def test_registry__forget_and_clear() -> None:
    """forget 忘掉单个场景；clear 清空全部。"""
    registry = KnownSuppressRegistry(window_seconds=600.0)
    registry.record("甲", known_suppress=True, now=1000.0)
    registry.record("乙", known_suppress=True, now=1000.0)
    assert registry.forget("甲") is True
    assert registry.forget("甲") is False
    assert registry.should_suppress("甲", now=1000.0) is False
    registry.clear()
    assert registry.should_suppress("乙", now=1000.0) is False


@pytest.mark.unit
def test_registry__set_window_applies_immediately() -> None:
    """配置热更新改窗口后立即生效（不用重启插件）。"""
    registry = KnownSuppressRegistry(window_seconds=600.0)
    registry.record(PROBE_SCENE, known_suppress=True, now=1000.0)
    registry.set_window(60.0)
    assert registry.window_seconds == 60.0
    assert registry.should_suppress(PROBE_SCENE, now=1030.0) is True
    assert registry.should_suppress(PROBE_SCENE, now=1100.0) is False


@pytest.mark.unit
def test_registry__known_suppress_flag_snapshot() -> None:
    """登记簿保存的是"注入那一刻"的 known_suppress 快照。"""
    registry = KnownSuppressRegistry(window_seconds=600.0)
    registry.record("甲", known_suppress=True, now=1000.0)
    registry.record("乙", known_suppress=False, now=1000.0)
    assert registry.known_suppress_flag("甲") is True
    assert registry.known_suppress_flag("乙") is False
    assert registry.known_suppress_flag("不存在") is False


@pytest.mark.unit
def test_registry__default_window_is_ten_minutes() -> None:
    """默认窗口 = 600 秒（10 分钟）。"""
    assert DEFAULT_WINDOW_SECONDS == 600.0
    assert KnownSuppressRegistry().window_seconds == 600.0


# =============================================================================
# B. 配置解析
# =============================================================================


@pytest.mark.unit
def test_options__known_suppress_window_default_is_600() -> None:
    """不配置时默认 600 秒。"""
    assert PluginOptions().known_suppress_window_s == 600
    assert PluginOptions.from_section({}).known_suppress_window_s == 600


@pytest.mark.unit
def test_options__known_suppress_window_reads_config() -> None:
    """配置里给了就按配置的来。"""
    assert PluginOptions.from_section({"known_suppress_window_s": 120}).known_suppress_window_s == 120


@pytest.mark.unit
def test_options__known_suppress_window_clamped() -> None:
    """越界值被 clamp：负数 → 0（关闭），超大 → 86400（1 天）。"""
    assert PluginOptions.from_section({"known_suppress_window_s": -1}).known_suppress_window_s == 0
    assert PluginOptions.from_section({"known_suppress_window_s": 999999}).known_suppress_window_s == 86400


@pytest.mark.unit
def test_options__known_suppress_window_garbage_falls_back_to_default() -> None:
    """非整数 → 回落默认 600（不静默变成 0，否则会意外关掉整个功能）。"""
    assert PluginOptions.from_section({"known_suppress_window_s": "abc"}).known_suppress_window_s == 600
    assert PluginOptions.from_section({"known_suppress_window_s": 1.5}).known_suppress_window_s == 600


@pytest.mark.unit
def test_plugin_toml__declares_the_key() -> None:
    """plugin.toml 的 [genshin_game_buddy] 段声明了这个键（配置项要能被看见）。"""
    manifest = Path(__file__).resolve().parents[2] / "plugin.toml"
    with manifest.open("rb") as stream:
        payload = tomllib.load(stream)
    section = payload.get("genshin_game_buddy", {})
    assert section.get("known_suppress_window_s") == 600


# =============================================================================
# C. 消费侧：_should_push_proactive
# =============================================================================


@pytest.mark.unit
def test_consumer__suppressed_after_injection() -> None:
    """★ 核心断言：注入过 → 主动提示被抑制（返回 False，不说话）。"""
    plugin = _make_stub()
    plugin._known_suppress.record(PROBE_SCENE, known_suppress=True)
    assert plugin._should_push_proactive("深境螺旋 第12层", scene_id=PROBE_SCENE) is False


@pytest.mark.unit
def test_consumer__not_suppressed_without_injection() -> None:
    """没注入过 → 主动提示照旧（不误伤）。"""
    plugin = _make_stub()
    assert plugin._should_push_proactive("深境螺旋 第12层", scene_id=PROBE_SCENE) is True


@pytest.mark.unit
def test_consumer__scene_without_known_suppress_is_not_suppressed() -> None:
    """场景自己标 known_suppress=False → 不抑制。"""
    plugin = _make_stub()
    plugin._known_suppress.record("普通场景", known_suppress=False)
    assert plugin._should_push_proactive("随便什么文本", scene_id="普通场景") is True


@pytest.mark.unit
def test_consumer__no_scene_id_means_no_suppression() -> None:
    """没带 scene_id（老调用方）→ 不抑制，保持向后兼容。"""
    plugin = _make_stub()
    plugin._known_suppress.record(PROBE_SCENE, known_suppress=True)
    assert plugin._should_push_proactive("任意文本") is True


@pytest.mark.unit
def test_consumer__window_zero_disables_suppression() -> None:
    """窗口 = 0 → 抑制功能整体关闭（配置可关）。"""
    plugin = _make_stub(window=0)
    plugin._known_suppress.record(PROBE_SCENE, known_suppress=True)
    assert plugin._should_push_proactive("深境螺旋 第12层", scene_id=PROBE_SCENE) is True


@pytest.mark.unit
def test_consumer__suppression_is_logged() -> None:
    """★ 抑制时必须有明确日志（排查"为什么没说话"）。"""
    logger = FakeLogger()
    plugin = _make_stub(logger=logger)
    plugin._known_suppress.record(PROBE_SCENE, known_suppress=True)
    plugin._should_push_proactive("深境螺旋 第12层", scene_id=PROBE_SCENE)
    infos = logger.lines("info")
    assert any("suppressed by known_suppress" in text for text in infos), infos
    assert any(PROBE_SCENE in text for text in infos), infos


@pytest.mark.unit
def test_consumer__suppression_does_not_update_hash() -> None:
    """★ 抑制时**不更新** ``_last_pushed_hash``。

    否则窗口过后同一段文本会被 hash 永久挡死，再也主动不了。
    """
    plugin = _make_stub()
    text = "深境螺旋 第12层"
    plugin._known_suppress.record(PROBE_SCENE, known_suppress=True)
    assert plugin._should_push_proactive(text, scene_id=PROBE_SCENE) is False
    assert plugin._last_pushed_hash == ""
    # 让那条记录过期（改成 601 秒前）→ 同一段文本重新可主动
    plugin._known_suppress.record(PROBE_SCENE, known_suppress=True, now=time.monotonic() - 601.0)
    assert plugin._should_push_proactive(text, scene_id=PROBE_SCENE) is True


@pytest.mark.unit
def test_consumer__scene_switch_path_is_also_suppressed() -> None:
    """场景切换路径同样受抑制约束——"玩家已经知道"不该被切换绕过。"""
    plugin = _make_stub()
    plugin._last_scene_switch_at = time.monotonic() - 100.0  # 冷却已过
    plugin._known_suppress.record(PROBE_SCENE, known_suppress=True)
    assert plugin._should_push_proactive("深境螺旋 第12层", scene_switched=True, scene_id=PROBE_SCENE) is False


# =============================================================================
# D. 记录侧
# =============================================================================


@pytest.mark.unit
def test_record_side__uses_scene_entry_snapshot() -> None:
    """记录侧按 ``SceneEntry.known_suppress`` 打快照，而不是一律 True。"""
    plugin = _make_stub()
    plugin._scene_store = {
        PROBE_SCENE: SceneEntry(name=PROBE_SCENE, known_suppress=True),
        "普通场景": SceneEntry(name="普通场景", known_suppress=False),
    }
    plugin._record_scene_injection(PROBE_SCENE)
    plugin._record_scene_injection("普通场景")
    assert plugin._known_suppress.should_suppress(PROBE_SCENE) is True
    assert plugin._known_suppress.should_suppress("普通场景") is False


@pytest.mark.unit
def test_record_side__unknown_scene_is_recorded_as_not_suppressing() -> None:
    """场景不在 ``_scene_store`` 里 → 记成不抑制（宁可多说也不要误吞）。"""
    plugin = _make_stub()
    plugin._scene_store = {}
    plugin._record_scene_injection("没登记的场景")
    assert plugin._known_suppress.should_suppress("没登记的场景") is False


@pytest.mark.unit
def test_record_side__empty_scene_id_is_ignored() -> None:
    """空 scene_id 不记录。"""
    plugin = _make_stub()
    plugin._scene_store = {PROBE_SCENE: SceneEntry(name=PROBE_SCENE, known_suppress=True)}
    plugin._record_scene_injection("")
    assert plugin._known_suppress.should_suppress("") is False


@pytest.mark.unit
def test_record_side__missing_registry_does_not_raise() -> None:
    """登记簿还没建好（启动早期 / 测试 stub）→ 什么都不记，也不抛异常。"""
    plugin = _make_stub()
    plugin._known_suppress = None  # type: ignore[assignment]
    plugin._record_scene_injection(PROBE_SCENE)  # 不抛


@pytest.mark.unit
def test_record_side__push_context_success_records_injection() -> None:
    """★ 语境块注入成功 → 登记"玩家已经知道这个场景了"。"""
    plugin = _make_stub()
    plugin._scene_store = {PROBE_SCENE: SceneEntry(name=PROBE_SCENE, known_suppress=True)}
    plugin._last_scene_pushed = PROBE_SCENE
    plugin._manager.library = MagicMock(tone="")
    plugin._build_activated_context = MagicMock(return_value="")
    plugin._t = MagicMock(side_effect=lambda key, **_kwargs: key)
    plugin.push_message = MagicMock(return_value={"submitted": True})
    session = GameSession(game_id="genshin", display_name="原神")
    assert plugin._push_context(session, announce_switch=False) is True
    assert plugin._known_suppress.should_suppress(PROBE_SCENE) is True


@pytest.mark.unit
def test_record_side__push_context_failure_does_not_record() -> None:
    """语境块注入**失败** → 不登记（没送到玩家眼前的东西不算"玩家已知"）。"""
    plugin = _make_stub()
    plugin._scene_store = {PROBE_SCENE: SceneEntry(name=PROBE_SCENE, known_suppress=True)}
    plugin._last_scene_pushed = PROBE_SCENE
    plugin._manager.library = MagicMock(tone="")
    plugin._build_activated_context = MagicMock(return_value="")
    plugin._t = MagicMock(side_effect=lambda key, **_kwargs: key)
    plugin.push_message = MagicMock(return_value={"submitted": False, "reason": "quota"})
    session = GameSession(game_id="genshin", display_name="原神")
    assert plugin._push_context(session, announce_switch=False) is False
    assert plugin._known_suppress.should_suppress(PROBE_SCENE) is False


# =============================================================================
# E. 用户提问路径不受影响
# =============================================================================


@pytest.mark.unit
async def test_user_query__still_answered_after_injection() -> None:
    """★ 用户主动提问 → 术语卡照推（抑制只管"主动"路径，不拦截应答）。"""
    plugin = _make_query_stub()
    plugin._scene_store = {PROBE_SCENE: SceneEntry(name=PROBE_SCENE, known_suppress=True)}
    plugin._record_scene_injection(PROBE_SCENE)
    assert plugin._known_suppress.should_suppress(PROBE_SCENE) is True

    await plugin.on_chat_message(text=f"{PROBE_TERM}怎么打？")

    assert plugin.push_message.called, "用户提问路径被抑制了——这是 bug"
    hint = plugin.push_message.call_args.kwargs["parts"][0]["text"]
    assert PROBE_TERM in hint


@pytest.mark.unit
async def test_user_query__does_not_touch_suppression_registry() -> None:
    """用户提问路径**不**消费登记簿（不会把"问过"当成"被告知过"）。"""
    plugin = _make_query_stub()
    plugin._scene_store = {PROBE_SCENE: SceneEntry(name=PROBE_SCENE, known_suppress=True)}
    await plugin.on_chat_message(text=f"{PROBE_TERM}怎么打？")
    assert plugin._known_suppress.last_injected_at(PROBE_SCENE) is None


# =============================================================================
# F. 端到端可复现证据
# =============================================================================


@pytest.mark.integration
async def test_end_to_end__second_injection_suppresses_proactive_but_query_still_answers(
    build_plugin: Any,
) -> None:
    """★ 可复现证据（真实插件实例 + 真实场景表 + 真实术语库）：

    1. 玩家走进「深境螺旋」→ 攻略语境注入成功 → 登记"玩家已经知道"；
    2. 同场景再要主动搭话 → **被抑制**（这次不说话）；
    3. 玩家手动提问「元素反应怎么打？」→ **仍然得到应答**（术语卡照推）。
    """
    plugin = build_plugin()
    await plugin.set_game(game="原神")
    session = plugin._manager.current
    assert session is not None
    assert PROBE_SCENE in plugin._scene_store
    assert plugin._scene_store[PROBE_SCENE].known_suppress is True

    # ── ① 第一次注入：攻略语境进模型上下文 ──
    plugin._last_scene_pushed = PROBE_SCENE
    assert plugin._push_context(session, announce_switch=False) is True
    assert plugin._known_suppress.should_suppress(PROBE_SCENE) is True

    # ── ② 同一场景的主动提示：被抑制 ──
    plugin._last_pushed_hash = ""
    plugin._last_proactive_at = 0.0
    assert plugin._should_push_proactive("深境螺旋 第12层", scene_id=PROBE_SCENE) is False
    assert any(
        "suppressed by known_suppress" in text for text in plugin.logger.lines("info")
    ), plugin.logger.text

    # ── ③ 用户手动提问：仍然应答 ──
    before = len(plugin.ctx.pushed)
    await plugin.on_chat_message(text=f"{PROBE_CHARACTER}怎么配队？")
    assert len(plugin.ctx.pushed) > before, "用户提问没有得到应答"
    card = plugin.ctx.pushed[-1]
    assert card["ai_behavior"] == "read"
    assert PROBE_CHARACTER in card["parts"][0]["text"]


@pytest.mark.integration
async def test_end_to_end__window_expiry_restores_proactive_push(build_plugin: Any) -> None:
    """窗口过后 → 主动提示恢复（抑制不是永久的）。"""
    plugin = build_plugin()
    await plugin.set_game(game="原神")
    plugin._last_scene_pushed = PROBE_SCENE
    plugin._last_pushed_hash = ""
    plugin._last_proactive_at = 0.0

    # 把注入时间推到 601 秒前（窗口 600s）→ 已过期
    plugin._known_suppress.record(PROBE_SCENE, known_suppress=True, now=time.monotonic() - 601.0)
    assert plugin._should_push_proactive("深境螺旋 第12层", scene_id=PROBE_SCENE) is True


@pytest.mark.integration
async def test_end_to_end__config_window_is_applied_from_plugin_toml(build_plugin: Any) -> None:
    """配置（默认 600s）真的落到了登记簿上。"""
    plugin = build_plugin()
    await plugin.set_game(game="原神")
    assert plugin._options.known_suppress_window_s == 600
    assert plugin._known_suppress.window_seconds == 600.0
