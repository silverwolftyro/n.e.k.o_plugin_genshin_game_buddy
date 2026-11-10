"""2026-11-09 接线：场景级 avoid / known_suppress 真正被消费。

背景
----
``terms/genshin/scenes.toml`` 里 18 个场景每个都带 ``avoid``（剧透清单）与
``known_suppress = true``，但此前 ``scene_store.SceneEntry`` 不暴露这两个字段，
数据是**死的**。本文件钉住"接线真的通了"：

1. **数据侧**：18 个场景加载后 ``avoid`` 非空、``known_suppress is True``；
2. **装配侧**：``build_context_block`` 把 avoid 渲染成**一句**约束文本进语境块；
   没有 avoid 时输出与改动前**逐字一致**（老行为不变）；
3. **接线侧**：``GenshinGameBuddyPlugin._push_context`` 会把当前场景的 avoid
   真的送进 push_message 的语境块（不是只加了字段）。

第 3 条是最关键的一条：前两条只证明"数据对得上"，只有它证明"消费方接上了"。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from plugin.plugins.genshin_game_buddy.context_pack import (
    build_context_block,
    render_scene_avoid,
)
from plugin.plugins.genshin_game_buddy.scene_store import SceneEntry, load_scenes

#: 标签文本由调用方传入（i18n 在插件层，本模块不硬编码）。
AVOID_LABEL = "本场景避免："

#: 18 个场景里用来做"真接线"断言的那一个（scenes.toml 的第一个场景）。
PROBE_SCENE = "深境螺旋"
PROBE_AVOID_ITEM = "不要剧透当期深渊敌人的配招与后续版本改动"


def _plugin_dir() -> Path:
    # tests/unit/ → tests/ → genshin_game_buddy/
    return Path(__file__).resolve().parents[2]


# =============================================================================
# 装配侧：avoid 渲染成一句
# =============================================================================


@pytest.mark.unit
def test_render_scene_avoid__joins_with_dun_hao_into_one_line() -> None:
    """多条 avoid 用「、」连接，渲染成**一句**（不拆成多行）。"""
    rendered = render_scene_avoid(["不要剧透", "不要照抄数值"], label=AVOID_LABEL)
    assert rendered == "本场景避免：不要剧透、不要照抄数值"
    assert "\n" not in rendered


@pytest.mark.unit
def test_render_scene_avoid__empty_list_yields_empty_string() -> None:
    """avoid 为空 → 空串（调用方据此不产生空行）。"""
    assert render_scene_avoid([], label=AVOID_LABEL) == ""
    assert render_scene_avoid((), label=AVOID_LABEL) == ""


@pytest.mark.unit
def test_render_scene_avoid__no_label_yields_empty_string() -> None:
    """没有标签 → 空串（标签是调用方责任，本模块不硬编码中文文案）。"""
    assert render_scene_avoid(["不要剧透"], label="") == ""


@pytest.mark.unit
def test_render_scene_avoid__drops_blank_items() -> None:
    """空白项不参与渲染（防御脏数据）。"""
    rendered = render_scene_avoid(["  不要剧透  ", "   ", ""], label=AVOID_LABEL)
    assert rendered == "本场景避免：不要剧透"


# =============================================================================
# 装配侧：avoid 真的进语境块
# =============================================================================


@pytest.mark.unit
def test_build_context_block__scene_avoid__reaches_the_output() -> None:
    """avoid 真的进了语境块输出——不是只加了个字段。"""
    block = build_context_block(
        header="【当前陪聊游戏】原神",
        terms_header="常用术语：",
        term_lines=["- 元素反应：两种元素相互作用产生的额外效果。"],
        scene_avoid=[PROBE_AVOID_ITEM, "不要照抄游戏内可读的 buff 数值"],
        scene_avoid_label=AVOID_LABEL,
        max_chars=800,
    )
    assert AVOID_LABEL + PROBE_AVOID_ITEM in block
    assert "不要照抄游戏内可读的 buff 数值" in block
    # 一句约束，不拆行
    avoid_lines = [line for line in block.splitlines() if AVOID_LABEL in line]
    assert len(avoid_lines) == 1


@pytest.mark.unit
def test_build_context_block__no_scene_avoid__output_unchanged() -> None:
    """没有 scene_avoid 时，输出与改动前**逐字一致**（老行为不变）。"""
    kwargs = {
        "header": "【当前陪聊游戏】原神",
        "tone_line": "口吻：熟悉提瓦特大陆",
        "terms_header": "常用术语：",
        "term_lines": ["- 元素反应：两种元素相互作用产生的额外效果。"],
        "empty_note": "（这款游戏暂未提供常驻术语。）",
        "hint": "需要更细的术语时调用 lookup_game_term 查询。",
        "max_chars": 800,
    }
    block = build_context_block(**kwargs)
    assert block == "\n".join(
        [
            kwargs["header"],
            kwargs["tone_line"],
            kwargs["terms_header"],
            kwargs["term_lines"][0],
            kwargs["hint"],
        ]
    )
    assert AVOID_LABEL not in block


@pytest.mark.unit
def test_build_context_block__scene_avoid__dropped_whole_when_over_budget() -> None:
    """预算不够时**整行丢** avoid 约束（沿用既有语义，不切半个词）。"""
    common = {
        "header": "【当前陪聊游戏】原神",
        "terms_header": "常用术语：",
        "term_lines": ["- 元素反应：两种元素相互作用产生的额外效果。"],
        "scene_avoid": [PROBE_AVOID_ITEM],
        "scene_avoid_label": AVOID_LABEL,
    }
    roomy = build_context_block(**common, max_chars=800)
    assert AVOID_LABEL in roomy

    # 预算只够放标题 + 术语区：avoid 行放不下 → 整行消失
    tight = build_context_block(**common, max_chars=40)
    assert AVOID_LABEL not in tight
    assert PROBE_AVOID_ITEM not in tight
    assert len(tight) <= 40


# =============================================================================
# 数据侧：18 个场景的 avoid / known_suppress 被解析出来
# =============================================================================


@pytest.mark.unit
def test_scenes__all_18_expose_avoid_and_known_suppress() -> None:
    """18 个场景加载后 avoid 非空、known_suppress is True。"""
    scenes = load_scenes(_plugin_dir(), None, "genshin", "terms/genshin")
    assert len(scenes) >= 18
    for name, scene in scenes.items():
        assert scene.avoid, f"{name}: avoid 为空"
        assert all(item.strip() for item in scene.avoid), f"{name}: avoid 含空白项"
        assert scene.known_suppress is True, f"{name}: known_suppress 不是 True"


# =============================================================================
# 接线侧：_push_context 真的把 avoid 送进语境块
# =============================================================================


async def _registered_plugin(build_plugin: Any) -> Any:
    """构造插件实例并登记「原神」——场景表随之加载（走真实的 set_game 路径）。"""
    plugin = build_plugin()
    await plugin.set_game(game="原神")
    return plugin


@pytest.mark.integration
async def test_push_context__scene_avoid__reaches_pushed_block(build_plugin: Any) -> None:
    """★ 接线断言：_push_context 送出的语境块里**真的有**当前场景的 avoid 约束。

    没有这一步，前面的断言都只是"数据对得上"——证明不了消费方接上了。
    """
    plugin = await _registered_plugin(build_plugin)
    # 场景表已加载（set_game → _load_scenes_for_current）
    assert PROBE_SCENE in plugin._scene_store

    # 模拟 OCR 命中场景后推过 scene prompt（_ocr_perceive 里的状态位）
    plugin._last_scene_pushed = PROBE_SCENE
    session = plugin._manager.current
    assert session is not None

    assert plugin._push_context(session, announce_switch=False) is True

    block = plugin.ctx.pushed[-1]["parts"][0]["text"]
    assert AVOID_LABEL in block
    assert PROBE_AVOID_ITEM in block


@pytest.mark.integration
async def test_push_context__no_scene__no_avoid_line(build_plugin: Any) -> None:
    """没有命中场景时，语境块**不含** avoid 行（不产生空行/悬空标签）。"""
    plugin = await _registered_plugin(build_plugin)
    plugin._last_scene_pushed = ""

    session = plugin._manager.current
    assert session is not None
    assert plugin._push_context(session, announce_switch=False) is True

    block = plugin.ctx.pushed[-1]["parts"][0]["text"]
    assert AVOID_LABEL not in block


@pytest.mark.integration
async def test_push_context__empty_scene_store__does_not_raise(build_plugin: Any) -> None:
    """场景表为空（启动早期 _load_scenes_for_current 还没跑完）→ 防御式降级，不抛异常。"""
    plugin = await _registered_plugin(build_plugin)
    plugin._scene_store = {}
    plugin._last_scene_pushed = PROBE_SCENE

    session = plugin._manager.current
    assert session is not None
    assert plugin._push_context(session, announce_switch=False) is True

    block = plugin.ctx.pushed[-1]["parts"][0]["text"]
    assert AVOID_LABEL not in block


@pytest.mark.unit
def test_scene_entry__avoid_is_tuple_and_hashable() -> None:
    """avoid 用 tuple（不是 list）——SceneEntry 是 frozen dataclass，必须保持 hashable。"""
    entry = SceneEntry(name="X", avoid=("不要剧透",))
    assert entry.avoid == ("不要剧透",)
    assert isinstance(entry.avoid, tuple)
    assert hash(entry)
