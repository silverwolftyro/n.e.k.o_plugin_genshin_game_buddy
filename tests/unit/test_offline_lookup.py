"""2026-11-10：离线攻略数据（武器 / 圣遗物 / 材料）接入检索层。

背景
----
``terms/genshin/{weapons,artifacts,materials}.toml`` 三张表早已落地，
但 ``game_data_store`` 此前**不在运行时 import 链**里 —— 数据在盘上、
检索层够不着。端到端验证里三问插件侧零供给：

* "风鹰剑和天空之刃哪个好？" → 只推了张无关的"空"卡；
* "绝缘之旗印给谁用？" → 0 条输出；
* "精锻用魔矿怎么获得？" → 0 条输出。

这一组钉住"真的接上了"：

1. **能查到**：问武器 / 圣遗物 / 材料，``lookup_game_term`` 返回对应条目且字段正确；
2. **术语库优先**：问"胡桃"仍走术语库，不被离线表抢走；
3. **跨游戏隔离**：鸣潮下查不到原神的离线数据；
4. **契约不变**：返回体形状与既有调用方（main_server 读的 llm_tool 返回体）期望一致；
5. **运行时零网络**：``game_data_store`` 现在**在** import 链里，且仍不含任何网络模块。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest
from conftest import make_config


def _plugin_dir() -> Path:
    # tests/unit/ → tests/ → genshin_game_buddy/
    return Path(__file__).resolve().parents[2]


async def _genshin(build_plugin: Any) -> Any:
    """构造插件实例并登记「原神」——载入真实术语库 + 真实场景表。"""
    plugin = build_plugin()
    await plugin.set_game(game="原神")
    return plugin


# =============================================================================
# 1. 能查到
# =============================================================================


@pytest.mark.integration
async def test_lookup__offline__weapon_is_found(build_plugin: Any) -> None:
    """问武器 → 命中离线武器表，字段正确。"""
    plugin = await _genshin(build_plugin)

    result = await plugin.lookup_game_term(query="风鹰剑")
    assert result["is_error"] is False
    output = result["output"]
    assert output["found"] is True
    assert output["count"] == 1

    entry = output["entries"][0]
    assert entry["key"] == "风鹰剑"
    assert entry["kind"] == "武器"
    assert entry["brief"]
    detail = entry["detail"]
    assert detail["rarity"] == 5
    assert detail["weapon_type"] == "单手剑"
    assert detail["base_atk"] == 47.537
    assert detail["passive"] == "西风之鹰的抗争"
    # 文本块里模型能直接读到特性与突破材料
    assert "风鹰剑" in output["text"]
    assert "西风之鹰的抗争" in output["text"]


@pytest.mark.integration
async def test_lookup__offline__artifact_is_found(build_plugin: Any) -> None:
    """问圣遗物 → 命中离线圣遗物表，2/4 件套效果都在。"""
    plugin = await _genshin(build_plugin)

    result = await plugin.lookup_game_term(query="绝缘之旗印")
    output = result["output"]
    assert output["found"] is True

    entry = output["entries"][0]
    assert entry["key"] == "绝缘之旗印"
    assert entry["kind"] == "圣遗物"
    detail = entry["detail"]
    assert "元素充能效率" in detail["effect_2pc"]
    assert "元素爆发" in detail["effect_4pc"]
    assert "2件套" in output["text"]
    assert "4件套" in output["text"]


@pytest.mark.integration
async def test_lookup__offline__material_is_found(build_plugin: Any) -> None:
    """问材料 → 命中离线材料表，获取途径在字段里。"""
    plugin = await _genshin(build_plugin)

    result = await plugin.lookup_game_term(query="精锻用魔矿")
    output = result["output"]
    assert output["found"] is True

    entry = output["entries"][0]
    assert entry["key"] == "精锻用魔矿"
    assert entry["kind"] == "材料"
    assert "锻造获得" in entry["detail"]["sources"]
    assert "获取途径" in output["text"]


@pytest.mark.integration
async def test_lookup__offline__natural_language_query_also_hits(build_plugin: Any) -> None:
    """自然语言问句里也能命中（不是只认"恰好等于名字"的查询）。"""
    plugin = await _genshin(build_plugin)

    result = await plugin.lookup_game_term(query="风鹰剑和天空之刃哪个好？")
    output = result["output"]
    assert output["found"] is True
    keys = [entry["key"] for entry in output["entries"]]
    assert "风鹰剑" in keys
    # "天空之刃"在 genshin-db 最小集里不存在（表里是"天空之翼"）——不该凭空捏造
    assert "天空之刃" not in keys


# =============================================================================
# 2. 术语库优先
# =============================================================================


@pytest.mark.integration
async def test_lookup__term_library_takes_priority_over_offline(build_plugin: Any) -> None:
    """问"胡桃"仍走术语库（kind=角色），不被离线表抢走。"""
    plugin = await _genshin(build_plugin)

    result = await plugin.lookup_game_term(query="胡桃")
    output = result["output"]
    assert output["found"] is True
    entry = output["entries"][0]
    assert entry["key"] == "胡桃"
    assert entry["kind"] == "角色"
    assert "detail" not in entry  # 术语条目没有离线表的 detail 字段


# =============================================================================
# 3. 跨游戏隔离
# =============================================================================


@pytest.mark.integration
async def test_lookup__offline__other_game_is_isolated(build_plugin: Any) -> None:
    """切到鸣潮 → 查不到原神的离线数据（鸣潮没有那三张表）。"""
    plugin = build_plugin()
    await plugin.set_game(game="鸣潮")

    result = await plugin.lookup_game_term(query="风鹰剑")
    assert result["output"]["found"] is False


# =============================================================================
# 4. 返回契约不变
# =============================================================================


@pytest.mark.integration
async def test_lookup__offline__contract_shape_unchanged(build_plugin: Any) -> None:
    """离线命中沿用既有返回体形状：found / game / count / entries / text。"""
    plugin = await _genshin(build_plugin)

    output = (await plugin.lookup_game_term(query="绝缘之旗印"))["output"]
    assert set(output) >= {"found", "game", "count", "entries", "text"}
    assert output["game"] == "原神"
    assert output["count"] == len(output["entries"])
    for entry in output["entries"]:
        # 术语条目形状兼容：key / kind / brief 三个字段一定在
        assert {"key", "kind", "brief"} <= set(entry)


@pytest.mark.integration
async def test_lookup__offline__card_respects_char_budget(build_plugin: Any) -> None:
    """离线卡片同样受 ``lookup_max_chars_per_term`` 约束（与术语卡一致）。"""
    plugin = build_plugin(config=make_config(options={"lookup_max_chars_per_term": 40}))
    await plugin.set_game(game="原神")

    output = (await plugin.lookup_game_term(query="绝缘之旗印"))["output"]
    assert output["found"] is True
    for card in output["text"].split("\n\n")[1:]:
        assert len(card) <= 40


@pytest.mark.integration
async def test_lookup__offline__respects_max_matches(build_plugin: Any) -> None:
    """命中条数受 ``lookup_max_matches`` 约束。"""
    plugin = build_plugin(config=make_config(options={"lookup_max_matches": 1}))
    await plugin.set_game(game="原神")

    output = (await plugin.lookup_game_term(query="风鹰剑"))["output"]
    assert output["found"] is True
    assert output["count"] <= 1


# =============================================================================
# 5. 运行时零网络 + 真的进了 import 链
# =============================================================================


@pytest.mark.unit
def test_game_data_store__now_in_runtime_import_chain() -> None:
    """``game_data_store`` 现在**在**插件的运行时 import 链里。

    这是 P0-1 的根因断言：之前它不在链上 → 三张表运行时不可达。
    """
    import plugin.plugins.genshin_game_buddy  # noqa: F401  - 触发 import

    assert "plugin.plugins.genshin_game_buddy.game_data_store" in sys.modules


@pytest.mark.unit
def test_game_data_store__imports_no_network_module() -> None:
    """``game_data_store`` 不 import 任何网络模块（运行时零网络）。"""
    source = (_plugin_dir() / "game_data_store.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.append(node.module.split(".")[0])
    network = {"requests", "urllib", "urllib3", "http", "httpx", "socket", "aiohttp"}
    assert not (set(roots) & network), set(roots) & network
