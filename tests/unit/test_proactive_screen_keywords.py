"""P1-1：主动提示只引用真实屏幕来源关键词，不再把常驻锚点当屏幕事实。

背景
----
``_push_proactive`` 过去拿 ``self._activation.get_active_keys()[:12]`` 当
"当前屏幕上出现了这些关键词"念给模型。那个集合里混着 **core 层常驻锚点**
（抽卡 / 体力 / 欧 / 非 / 歪了 / 毕业 / 肝 / 裂开 ...）——它们在 ``set_game``
时就注册好了，**从来不是屏幕 OCR 的结果**。把它们说成"屏幕上出现了"是对模型
的事实误报：模型完全可能据此编造"我看到了抽卡界面"。

修复后：关键词子句只列 source ∈ {screen, screen_slang, query} 的激活键；
过滤后为空 → 改走中性表述，**不硬凑**。

本文件钉住四件事：
  1. ``get_active_keys_by_sources`` 按来源**精确**筛选（``scene:<name>`` 不被
     ``"scene"`` 误伤）；
  2. core 层常驻锚点一个都不进关键词子句；
  3. 过滤后为空 → 中性表述，不硬凑关键词；
  4. 用户主动提问路径（query 激活 + 语境块）**不受影响**。
"""
from __future__ import annotations

import re
from typing import Any

import pytest
from plugin.plugins.genshin_game_buddy.activation import ActivationState

#: 关键词子句的形状："当前屏幕上出现了这些关键词：X / Y。"
KEYWORD_CLAUSE_RE = re.compile(r"当前屏幕上出现了这些关键词：(?P<keys>.+?)。")

#: core 层常驻锚点里的代表词——用来做"整句里都不该出现"的强断言。
#: 刻意选两个**两字**词：单字锚点（欧 / 非）会撞上"非常好"这类无关文案，
#: 不能用来做整句断言（keyword 子句断言另有其法，见下）。
CORE_ANCHOR_PROBES = ("抽卡", "体力")

#: 场景名。选"深境螺旋"是因为它**本身也是 core 锚点**——正好用来验证
#: "场景名出现在动作子句 ≠ core 锚点混进关键词子句"。
PROBE_SCENE = "深境螺旋"

#: 屏幕真命中的探针词。刻意避开 core 锚点（运行时断言会再确认一次）。
SCREEN_PROBES = ["胡桃", "甘雨"]


# =============================================================================
# 数据层：get_active_keys_by_sources
# =============================================================================


@pytest.mark.unit
def test_get_active_keys_by_sources__only_exact_source_matches() -> None:
    """按来源筛选：只认**精确** source，``scene:<name>`` 不被 ``"scene"`` 命中。"""
    state = ActivationState()
    state.activate_terms(["抽卡", "体力"], source="core_layer", ttl_seconds=float("inf"))
    state.activate_terms(["胡桃"], source="screen", ttl_seconds=60.0)
    state.activate_terms(["甘雨"], source="query", ttl_seconds=60.0)
    state.activate_scene(PROBE_SCENE, [PROBE_SCENE])

    assert state.get_active_keys_by_sources(("screen", "screen_slang", "query")) == ["胡桃", "甘雨"]
    assert state.get_active_keys_by_sources(("core_layer",)) == ["抽卡", "体力"]
    # scene:<name> 是带前缀的 source，精确匹配语义下 "scene" 匹配不到它
    assert state.get_active_keys_by_sources(("scene",)) == []
    assert state.get_active_keys_by_sources(("scene:" + PROBE_SCENE,)) == [PROBE_SCENE]
    # 空集合 / 未知 source → 空列表，不抛异常
    assert state.get_active_keys_by_sources(()) == []
    assert state.get_active_keys_by_sources(("不存在的来源",)) == []


@pytest.mark.unit
def test_get_active_keys_by_sources__preserves_activation_order() -> None:
    """筛选后保持激活顺序（文案里词的先后与激活先后一致）。"""
    state = ActivationState()
    state.activate_terms(["胡桃"], source="screen", ttl_seconds=60.0)
    state.activate_terms(["抽卡"], source="core_layer", ttl_seconds=float("inf"))
    state.activate_terms(["甘雨"], source="screen", ttl_seconds=60.0)
    assert state.get_active_keys_by_sources(("screen",)) == ["胡桃", "甘雨"]


@pytest.mark.unit
def test_get_active_keys_by_sources__does_not_mutate_activation_state() -> None:
    """只读查询：不新增、不改 source、不动淘汰状态。"""
    state = ActivationState()
    state.activate_terms(["抽卡"], source="core_layer", ttl_seconds=float("inf"))
    state.activate_terms(["胡桃"], source="screen", ttl_seconds=60.0)

    before = dict(state._active)
    assert state.get_active_keys_by_sources(("screen",)) == ["胡桃"]
    assert state.get_active_keys_by_sources(("core_layer",)) == ["抽卡"]
    assert dict(state._active) == before


# =============================================================================
# 文案层：_push_proactive
# =============================================================================


async def _plugin_with_scene(build_plugin: Any) -> Any:
    """构造插件 → 登记「原神」（core 层锚点随之激活）→ 激活一个场景。"""
    plugin = build_plugin()
    await plugin.set_game(game="原神")
    plugin._activation.activate_scene(PROBE_SCENE, [PROBE_SCENE])
    return plugin


def _last_pushed_text(plugin: Any) -> str:
    return plugin.ctx.pushed[-1]["parts"][0]["text"]


@pytest.mark.integration
async def test_push_proactive__no_screen_hit__no_keyword_clause(build_plugin: Any) -> None:
    """★ P1-1 主断言：屏幕没命中任何词 → **关键词子句整个不出现**。

    过滤后为空时改走中性表述，**不硬凑**——更不能把 core 锚点凑进去。
    """
    plugin = await _plugin_with_scene(build_plugin)
    core_keys = plugin._activation.get_active_keys_by_sources(("core_layer",))
    assert core_keys, "core 层常驻锚点应已激活"
    for probe in CORE_ANCHOR_PROBES:
        assert probe in core_keys, probe

    assert plugin._push_proactive([PROBE_SCENE], [], []) is True

    text = _last_pushed_text(plugin)
    assert KEYWORD_CLAUSE_RE.search(text) is None
    assert "当前屏幕上出现了这些关键词" not in text
    for probe in CORE_ANCHOR_PROBES:
        assert probe not in text
    # 真实事实（切换了场景）必须保留——只是不说"屏幕上出现了什么"
    assert "切换到了「" + PROBE_SCENE + "」场景" in text


@pytest.mark.integration
async def test_push_proactive__screen_hit__lists_only_screen_words(build_plugin: Any) -> None:
    """屏幕真命中时：关键词子句**只**列这几个词，core 锚点一个都不混进来。"""
    plugin = await _plugin_with_scene(build_plugin)
    core_keys = set(plugin._activation.get_active_keys_by_sources(("core_layer",)))
    assert not (core_keys & set(SCREEN_PROBES)), "探针词撞上 core 锚点了，换个词"

    plugin._activation.activate_terms(SCREEN_PROBES, source="screen", ttl_seconds=60.0)

    assert plugin._push_proactive([PROBE_SCENE], [], []) is True

    text = _last_pushed_text(plugin)
    match = KEYWORD_CLAUSE_RE.search(text)
    assert match is not None, text
    listed = match.group("keys").split(" / ")
    assert listed == SCREEN_PROBES
    assert not (set(listed) & core_keys)
    for probe in CORE_ANCHOR_PROBES:
        assert probe not in text


@pytest.mark.integration
async def test_push_proactive__every_listed_key_is_a_real_screen_source(build_plugin: Any) -> None:
    """关键词子句里的**每一个**词都必须是 screen / screen_slang / query 来源。

    这条是 P1-1 的核心不变式：子句里出现的词 ⇒ 屏幕上真的出现过（或用户问过）。
    """
    plugin = await _plugin_with_scene(build_plugin)
    plugin._activation.activate_terms(SCREEN_PROBES, source="screen", ttl_seconds=60.0)
    plugin._activation.activate_terms(["刻晴"], source="query", ttl_seconds=60.0)
    plugin._activation.activate_terms(["行秋"], source="screen_slang", ttl_seconds=60.0)

    assert plugin._push_proactive([PROBE_SCENE], [], []) is True

    text = _last_pushed_text(plugin)
    match = KEYWORD_CLAUSE_RE.search(text)
    assert match is not None, text
    listed = set(match.group("keys").split(" / "))

    real_sources = set(
        plugin._activation.get_active_keys_by_sources(("screen", "screen_slang", "query"))
    )
    assert listed == real_sources
    assert listed <= real_sources


@pytest.mark.integration
async def test_push_proactive__core_anchors_absent_from_keyword_clause(build_plugin: Any) -> None:
    """全量 core 锚点逐条比对：没有任何一条出现在关键词子句里。

    只对**子句**做这个断言——场景名"深境螺旋"本身也是 core 锚点，
    它出现在动作子句（"切换到了「深境螺旋」场景"）是**事实**，不算误报。
    """
    plugin = await _plugin_with_scene(build_plugin)
    core_keys = set(plugin._activation.get_active_keys_by_sources(("core_layer",)))
    assert len(core_keys) >= 10, core_keys

    plugin._activation.activate_terms(SCREEN_PROBES, source="screen", ttl_seconds=60.0)

    assert plugin._push_proactive([PROBE_SCENE], [], []) is True

    text = _last_pushed_text(plugin)
    match = KEYWORD_CLAUSE_RE.search(text)
    assert match is not None, text
    listed = set(match.group("keys").split(" / "))
    leaked = listed & core_keys
    assert leaked == set(), f"core 锚点混进了关键词子句：{sorted(leaked)}"


@pytest.mark.integration
async def test_push_proactive__source_filter_is_what_keeps_anchors_out(
    build_plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★ 拔牙对照：同一个插件状态，**拆掉**来源过滤再**装回去**，两次文案各断言一次。

    为什么要这条：上面那些"不含 core 锚点"的断言，如果不是因为过滤真的在起作用，
    就可能只是断言写虚了（永远绿）。这里把过滤这一环**临时拆掉**（monkeypatch 换掉
    ``get_active_keys_by_sources``，让关键词子句退回"全量激活键"的旧逻辑）：

    * 拆掉时 → 文案里**会**漏出 core 锚点（抽卡 / 体力）；
    * 装回时 → 同样的状态、同样两个词，文案里**不含**这些锚点。

    于是"绿"这件事被钉成可执行的事实：**是过滤让它变绿的**。
    任何人以后把过滤换成 ``get_active_keys()``，上面几条断言立刻变红。

    ★ 全程只 monkeypatch **运行时对象**，不改任何源码文件。
    """
    plugin = await _plugin_with_scene(build_plugin)
    activation = plugin._activation
    original = activation.get_active_keys_by_sources

    # ① 退回旧逻辑：关键词子句 = 全量激活键（core 锚点会被当成屏幕事实）
    monkeypatch.setattr(
        activation,
        "get_active_keys_by_sources",
        lambda sources: activation.get_active_keys(),
    )
    assert plugin._push_proactive([PROBE_SCENE], [], []) is True
    old_text = _last_pushed_text(plugin)
    assert KEYWORD_CLAUSE_RE.search(old_text) is not None, old_text
    for probe in CORE_ANCHOR_PROBES:
        assert probe in old_text, f"旧逻辑下 {probe} 应当漏进文案（否则这测试就没有拔牙力）"

    # ② 装回真实过滤：同样的状态，锚点一个都不漏
    monkeypatch.setattr(activation, "get_active_keys_by_sources", original)
    assert plugin._push_proactive([PROBE_SCENE], [], []) is True
    new_text = _last_pushed_text(plugin)
    assert KEYWORD_CLAUSE_RE.search(new_text) is None, new_text
    for probe in CORE_ANCHOR_PROBES:
        assert probe not in new_text, probe


# =============================================================================
# 用户主动提问路径：不受影响
# =============================================================================


@pytest.mark.integration
async def test_query_path__query_term_activated_and_card_pushed(build_plugin: Any) -> None:
    """用户主动提问 → query 激活照旧、术语卡照旧推。

    P1-1 只改**主动提示**里那句"屏幕上出现了什么"，
    用户主动问的那条路径（激活 + 推卡）一个字都没动。
    """
    plugin = build_plugin()
    await plugin.set_game(game="原神")
    core_before = set(plugin._activation.get_active_keys_by_sources(("core_layer",)))
    assert core_before

    plugin._scan_and_activate_query("胡桃怎么养")

    assert "胡桃" in plugin._activation.get_active_keys_by_sources(("query",))
    # 语境块走的是**未过滤**的 get_active_keys()——core 锚点照旧在场
    all_keys = set(plugin._activation.get_active_keys())
    assert core_before <= all_keys
    # 术语卡照旧推出去
    texts = [entry["parts"][0]["text"] for entry in plugin.ctx.pushed]
    assert any("胡桃" in text for text in texts), texts


@pytest.mark.integration
async def test_context_block__still_carries_core_anchors(build_plugin: Any) -> None:
    """语境块仍带 core 锚点——P1-1 没把常驻锚点从上下文里摘掉。

    语境块是"术语速查表"，锚点在里面是对的；错的是把它们说成"屏幕看到了"。
    """
    plugin = build_plugin()
    await plugin.set_game(game="原神")
    core_keys = plugin._activation.get_active_keys_by_sources(("core_layer",))
    assert core_keys

    session = plugin._manager.current
    assert session is not None
    block = plugin._build_activated_context(session)
    assert block.strip()
    assert any(key in block for key in core_keys), block
