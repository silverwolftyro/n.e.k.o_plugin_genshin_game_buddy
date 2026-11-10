"""2026-11-10：查询路径的命中精度门槛（单字术语）。

背景
----
``_scan_and_activate_query`` 原来只用"术语是用户原文的子串"判命中，
**没有长度下限**（屏幕路径 2.0.63 加过 ``len >= 2``，查询路径漏了）。
于是问"风鹰剑和天空之刃哪个好？"会因为"天空之**空**刃"里的"空"字，
推出一张完全无关的术语卡：

    用户刚提到：空。相关条目：
    - 空：可变元素 · 单手剑 · 异界的旅人

推错卡比不推更糟 —— 它把无关条目塞进模型上下文。

这一组钉住三件事：

1. **夹在汉字中间的单字不算命中**（"天空之刃"里的"空"）；
2. **真问单字术语时照常命中**（"空是谁"）；
3. **命中集合里混着单字 term 且不止一条时只激活不推卡**。

判定规则（``_standalone_single_char``）：单字左右**至少一侧不是汉字**
才算独立成词 —— 夹在两个汉字中间说明它属于更长的词；
站在句首或紧邻标点/英文才算用户真在指代它。
"""
from __future__ import annotations

from typing import Any

import pytest
from plugin.plugins.genshin_game_buddy import GenshinGameBuddyPlugin


async def _genshin(build_plugin: Any) -> Any:
    plugin = build_plugin()
    await plugin.set_game(game="原神")
    return plugin


def _pushed_texts(plugin: Any) -> list[str]:
    """插件已 push 出去的文本（按顺序）。"""
    return [
        "\n".join(str(part.get("text", "")) for part in item.get("parts") or [])
        for item in plugin.ctx.pushed
    ]


# =============================================================================
# 1. 夹在汉字中间的单字不算命中
# =============================================================================


@pytest.mark.integration
async def test_query_gate__sky_blade_does_not_hit_single_char_kong(build_plugin: Any) -> None:
    """"风鹰剑和天空之刃哪个好？"不再推出"空"的卡。"""
    plugin = await _genshin(build_plugin)
    before = len(plugin.ctx.pushed)

    await plugin.on_chat_message(text="风鹰剑和天空之刃哪个好？", sender="user")

    new_texts = _pushed_texts(plugin)[before:]
    assert not any("空：可变元素" in text for text in new_texts), new_texts
    assert not any("用户刚提到：空" in text for text in new_texts), new_texts
    # 整串里不该有任何一张术语卡（真没命中什么）
    assert new_texts == [], new_texts


@pytest.mark.integration
async def test_query_gate__sky_wing_query_does_not_hit_kong(build_plugin: Any) -> None:
    """"天空之翼和风鹰剑哪个好？"同样不命中"空"（天/空 两侧都是汉字）。"""
    plugin = await _genshin(build_plugin)
    before = len(plugin.ctx.pushed)

    await plugin.on_chat_message(text="天空之翼和风鹰剑哪个好", sender="user")

    new_texts = _pushed_texts(plugin)[before:]
    assert not any("用户刚提到：空" in text for text in new_texts), new_texts


@pytest.mark.integration
async def test_query_gate__kong_is_not_activated(build_plugin: Any) -> None:
    """不命中 → 连激活都不该有（不是"激活了但不推卡"）。"""
    plugin = await _genshin(build_plugin)

    await plugin.on_chat_message(text="风鹰剑和天空之刃哪个好？", sender="user")

    assert "空" not in plugin._activation.get_active_keys()


# =============================================================================
# 2. 真问单字术语时照常命中
# =============================================================================


@pytest.mark.integration
async def test_query_gate__kong_shi_shui_still_hits(build_plugin: Any) -> None:
    """"空是谁"→ 照常命中"空"并推出术语卡（不是"单字一律不命中"）。"""
    plugin = await _genshin(build_plugin)
    before = len(plugin.ctx.pushed)

    await plugin.on_chat_message(text="空是谁", sender="user")

    new_texts = _pushed_texts(plugin)[before:]
    assert any("用户刚提到：空" in text for text in new_texts), new_texts
    assert any("空：可变元素" in text for text in new_texts), new_texts
    assert "空" in plugin._activation.get_active_keys()


@pytest.mark.integration
async def test_query_gate__multi_char_term_unaffected(build_plugin: Any) -> None:
    """多字术语的既有行为不变（"胡桃怎么养？"照常推卡）。"""
    plugin = await _genshin(build_plugin)
    before = len(plugin.ctx.pushed)

    await plugin.on_chat_message(text="胡桃怎么养？", sender="user")

    new_texts = _pushed_texts(plugin)[before:]
    assert any("用户刚提到：胡桃" in text for text in new_texts), new_texts


# =============================================================================
# 3. 单字 term 混进多条命中 → 只激活不推卡
# =============================================================================


@pytest.mark.integration
async def test_query_gate__single_char_mixed_with_other__no_card(build_plugin: Any) -> None:
    """命中里混着单字 term 且不止一条 → 只激活，不推卡。

    "空和雷电将军"：空（单字，句首独立）+ 雷电将军（多字，characters 维度，
    平时会推卡）。混着单字 → 整张卡都不推。
    """
    plugin = await _genshin(build_plugin)
    before = len(plugin.ctx.pushed)

    await plugin.on_chat_message(text="空和雷电将军", sender="user")

    new_texts = _pushed_texts(plugin)[before:]
    assert new_texts == [], new_texts
    # 但两个词都激活了（只是不推卡）
    active = plugin._activation.get_active_keys()
    assert "空" in active
    assert "雷电将军" in active


# =============================================================================
# 4. 判定规则本身的边界
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    ("char", "text", "expected"),
    [
        ("空", "空是谁", True),          # 句首 → 独立
        ("空", "天空之刃哪个好", False),  # 两侧都是汉字 → 属于更长的词
        ("空", "天空之刃", False),        # 同上
        ("欧", "这波很欧", True),          # 句尾（边界算"非汉字侧"）→ 独立
        ("空", "空", True),              # 就一个字
        ("欧", "今天太欧了", False),      # 两侧都是汉字 → 不独立（已知取舍，见下）
    ],
)
def test_standalone_single_char__boundaries(char: str, text: str, expected: bool) -> None:
    """单字"独立成词"判定的边界。

    判定式：**左右至少一侧不是汉字**（串首/串尾的边界也算"非汉字侧"）→ 独立。
    两侧都是汉字 → 该字属于更长的词 → 不独立。

    ★ 已知取舍："今天太欧了"里的"欧"两侧都是汉字 → 不独立 → 查询路径**漏命中**。
      影响可控：欧 / 非 / 肝 是 slang 维度的**常驻**术语（core 层，TTL=inf），
      早已在语境块里，查询路径漏激活它们没有实际损失。
      反过来"天空之刃"里的"空"必须被拒 —— 它是假阳性，会推出无关卡片。
    """
    assert GenshinGameBuddyPlugin._standalone_single_char(char, text) is expected


@pytest.mark.unit
def test_query_match_kind__distinguishes_single_and_multi() -> None:
    """命中种类：多字 key/alias → "multi"；独立单字 → "single"。"""
    from plugin.plugins.genshin_game_buddy.term_store import TermEntry

    multi = TermEntry(key="胡桃", kind="角色", brief="", aliases=("火化",))
    single = TermEntry(key="空", kind="角色", brief="", aliases=("空哥",))

    assert GenshinGameBuddyPlugin._query_match_kind(multi, "胡桃怎么养？") == "multi"
    assert GenshinGameBuddyPlugin._query_match_kind(multi, "今天火化了") == "multi"
    assert GenshinGameBuddyPlugin._query_match_kind(single, "空是谁") == "single"
    assert GenshinGameBuddyPlugin._query_match_kind(single, "天空之刃") == ""


@pytest.mark.unit
def test_screen_path__single_char_still_not_matched() -> None:
    """屏幕路径的老规则不动：单字 term 在屏幕侧本来就不命中。"""
    from plugin.plugins.genshin_game_buddy.term_store import TermEntry, TermLibrary

    entry = TermEntry(key="空", kind="角色", brief="")
    library = TermLibrary(
        game_id="genshin",
        display_name="原神",
        tone="",
        entries=(entry,),
        by_key={"空": entry},
    )
    screen, slang = GenshinGameBuddyPlugin._scan_screen_activate_terms("天空之刃 空", library)
    assert screen == []
    assert slang == []
