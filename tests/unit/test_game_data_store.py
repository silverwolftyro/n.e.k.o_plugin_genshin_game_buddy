"""任务 B：原神离线数据（武器 / 圣遗物 / 材料）的落地与查询回归。

覆盖三件事：

1. **数据集形状**：条目数、字段完整性、文件头版本锚点；
2. **查询能力**：按中文名 / slug 精确查、按关键字模糊查；
3. **运行时零网络**：查询模块不 import 任何网络库，运行时模块不 import 查询模块，
   取数脚本带"仅开发期使用"标记且不在运行时 import 链上。
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest
from conftest import PLUGIN_DIR, PLUGIN_MODULES, read_plugin_source
from plugin.plugins.genshin_game_buddy.game_data_store import (
    GameData,
    default_terms_dir,
    load_game_data,
    normalize_key,
)

TERMS_DIR = PLUGIN_DIR / "terms" / "genshin"
DATA_FILES = {
    "weapons": TERMS_DIR / "weapons.toml",
    "artifacts": TERMS_DIR / "artifacts.toml",
    "materials": TERMS_DIR / "materials.toml",
}

#: 运行时**绝不允许**出现的网络相关 import。
FORBIDDEN_NETWORK_MODULES = {
    "urllib",
    "urllib.request",
    "urllib.error",
    "http",
    "http.client",
    "httplib",
    "requests",
    "socket",
    "ssl",
    "aiohttp",
    "httpx",
    "ftplib",
    "telnetlib",
    "smtplib",
    "xmlrpc",
    "asyncio",
}


@pytest.fixture(scope="module")
def data() -> GameData:
    return load_game_data()


@pytest.fixture(scope="module")
def payloads() -> dict:
    return {
        kind: tomllib.loads(path.read_text(encoding="utf-8"))
        for kind, path in DATA_FILES.items()
    }


# =============================================================================
# 1. 数据集形状
# =============================================================================


def test_dataset__entry_counts(data: GameData) -> None:
    """最小数据集的条目数：武器 10 / 圣遗物 5 / 材料 20。"""
    assert len(data.weapons) == 10
    assert len(data.artifacts) == 5
    assert len(data.materials) == 20


@pytest.mark.parametrize("kind", sorted(DATA_FILES))
def test_dataset__file_header_has_version_anchor(payloads: dict, kind: str) -> None:
    """三个文件都带 schema_version / generated_at / source（可追溯）。"""
    payload = payloads[kind]
    assert payload["schema_version"] == 1
    assert payload["generated_at"].strip()
    assert payload["source"].strip()
    assert payload["source_repo"].strip()
    assert payload["source_commit"].strip()


def test_dataset__source_commit_is_pinned(payloads: dict) -> None:
    """数据源固定到某个 commit，重跑可复现。"""
    commits = {payload["source_commit"] for payload in payloads.values()}
    assert len(commits) == 1, f"三个文件的 source_commit 不一致：{commits}"
    assert all(len(commit) == 40 for commit in commits)


def test_dataset__names_are_unique(data: GameData) -> None:
    """同名条目会让"按名称查"变得有歧义，必须唯一。"""
    for label, entries in (
        ("weapons", data.weapons),
        ("artifacts", data.artifacts),
        ("materials", data.materials),
    ):
        names = [entry.name for entry in entries]
        assert len(names) == len(set(names)), f"{label} 存在重名：{names}"
        slugs = [entry.slug for entry in entries]
        assert len(slugs) == len(set(slugs)), f"{label} 存在重复 slug：{slugs}"


def test_dataset__weapon_fields_are_complete(data: GameData) -> None:
    """每把武器至少有名称 / 稀有度 / 类型 / 基础攻击 / 主属性。"""
    for weapon in data.weapons:
        assert weapon.name.strip(), f"{weapon.slug} 缺名称"
        assert weapon.rarity >= 1, f"{weapon.name} 缺稀有度"
        assert weapon.weapon_type.strip(), f"{weapon.name} 缺武器类型"
        assert weapon.weapon_type_code.strip(), f"{weapon.name} 缺武器类型码"
        assert weapon.base_atk > 0, f"{weapon.name} 缺基础攻击"
        assert weapon.main_stat.strip(), f"{weapon.name} 缺主属性"
        assert weapon.main_stat_value.strip(), f"{weapon.name} 缺主属性数值"


def test_dataset__artifact_fields_are_complete(data: GameData) -> None:
    """每套圣遗物至少有套装名 / 2 件套 / 4 件套效果。"""
    for artifact in data.artifacts:
        assert artifact.name.strip(), f"{artifact.slug} 缺套装名"
        assert artifact.rarity, f"{artifact.name} 缺可用稀有度"
        assert artifact.effect_2pc.strip(), f"{artifact.name} 缺 2 件套效果"
        assert artifact.effect_4pc.strip(), f"{artifact.name} 缺 4 件套效果"
        assert len(artifact.pieces) == 5, f"{artifact.name} 部位数不对：{artifact.pieces}"


def test_dataset__material_fields_are_complete(data: GameData) -> None:
    """每种材料至少有名称 / 类别，且"用途说明"与"获取途径"至少有一项。"""
    for material in data.materials:
        assert material.name.strip(), f"{material.slug} 缺名称"
        assert material.category.strip(), f"{material.name} 缺类别码"
        assert material.category_text.strip(), f"{material.name} 缺类别名"
        assert material.sources or material.description.strip(), f"{material.name} 既无用途也无获取途径"


def test_dataset__material_rarity_zero_means_unlabelled(data: GameData) -> None:
    """``rarity = 0`` 表示上游未标注，不是"0 星"——必须有条目真的落到 0。"""
    assert any(material.rarity == 0 for material in data.materials)


# =============================================================================
# 2. 查询能力
# =============================================================================


def test_lookup__weapon_by_chinese_name(data: GameData) -> None:
    """按中文名精确查武器。"""
    weapon = data.weapon("风鹰剑")
    assert weapon is not None
    assert weapon.slug == "aquilafavonia"
    assert weapon.weapon_type == "单手剑"
    assert weapon.rarity == 5
    assert weapon.passive == "西风之鹰的抗争"
    assert weapon.base_atk > 0


def test_lookup__artifact_by_chinese_name(data: GameData) -> None:
    """按中文名精确查圣遗物。"""
    artifact = data.artifact("绝缘之旗印")
    assert artifact is not None
    assert artifact.slug == "emblemofseveredfate"
    assert artifact.rarity == (4, 5)
    assert "元素充能效率" in artifact.effect_2pc
    assert artifact.effect_4pc.strip()


def test_lookup__material_by_chinese_name(data: GameData) -> None:
    """按中文名精确查材料。"""
    material = data.material("原石")
    assert material is not None
    assert material.slug == "primogem"
    assert material.category == "ADSORBATE"
    assert material.category_text == "稀有货币"


def test_lookup__by_slug(data: GameData) -> None:
    """slug 也能查（同一批条目的稳定标识）。"""
    assert data.weapon("aquilafavonia") is not None
    assert data.artifact("emblemofseveredfate") is not None
    assert data.material("primogem") is not None


def test_lookup__unknown_name_returns_none(data: GameData) -> None:
    """查不到就返回 None，不抛异常、不瞎猜。"""
    assert data.weapon("并不存在的武器") is None
    assert data.artifact("并不存在的套装") is None
    assert data.material("并不存在的材料") is None
    assert data.weapon("") is None


def test_lookup__is_case_and_width_insensitive(data: GameData) -> None:
    """查询键归一化：全角 / 半角、大小写、空白都不影响命中。"""
    assert normalize_key("　风鹰剑　") == "风鹰剑"
    assert normalize_key("FENGYING") == "fengying"
    assert data.weapon("　风鹰剑　") is not None
    assert data.weapon("风鹰剑".lower()) is not None


def test_search__by_keyword(data: GameData) -> None:
    """关键字模糊查：武器名 / 材料类别 / 套装效果。"""
    assert [w.name for w in data.find_weapons("天空")] == ["天空之翼"]
    assert any(m.name == "风之印" for m in data.find_materials("印"))
    assert any(m.name == "好感经验" for m in data.find_materials("经验"))
    assert any(a.name == "绝缘之旗印" for a in data.find_artifacts("充能"))


def test_search__empty_keyword_returns_nothing(data: GameData) -> None:
    """空关键字不返回全量（避免误当成"列出全部"）。"""
    assert data.find_weapons("") == ()
    assert data.find_artifacts("   ") == ()
    assert data.find_materials("") == ()


def test_summary__renders_one_line(data: GameData) -> None:
    """summary() 给 LLM 直接用：一行、含关键字段。"""
    weapon = data.weapon("风鹰剑")
    assert weapon is not None
    line = weapon.summary()
    assert "\n" not in line
    assert "风鹰剑" in line
    assert "单手剑" in line

    artifact = data.artifact("绝缘之旗印")
    assert artifact is not None
    assert "2件套" in artifact.summary()
    assert "4件套" in artifact.summary()

    material = data.material("兽肉")
    assert material is not None
    assert "兽肉" in material.summary()


def test_load__default_dir_is_package_relative() -> None:
    """默认数据目录相对本模块定位（不含机器绝对路径）。"""
    resolved = default_terms_dir().resolve()
    assert resolved == (PLUGIN_DIR / "terms" / "genshin").resolve()
    for path in DATA_FILES.values():
        assert path.is_file(), f"{path} 不存在"


# =============================================================================
# 3. 运行时零网络
# =============================================================================


def _imported_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_runtime__query_module_imports_no_network_library() -> None:
    """查询模块不得 import 任何网络库（运行时零网络的第一道闸）。"""
    source = (PLUGIN_DIR / "game_data_store.py").read_text(encoding="utf-8")
    imported = _imported_modules(source)
    offenders = {
        name
        for name in imported
        if name.split(".")[0] in FORBIDDEN_NETWORK_MODULES
    }
    assert not offenders, f"game_data_store.py 引用了网络库：{sorted(offenders)}"


def test_runtime__query_module_has_no_network_calls() -> None:
    """再堵一层：源码里不得出现发请求的调用名。"""
    source = (PLUGIN_DIR / "game_data_store.py").read_text(encoding="utf-8")
    for token in ("urlopen", "urlretrieve", "requests.", "socket.", "http.client"):
        assert token not in source, f"game_data_store.py 疑似发起网络请求：{token}"


def test_runtime__no_runtime_module_imports_query_module() -> None:
    """查询模块不在运行时 import 链上——插件运行行为零变化。"""
    for module in PLUGIN_MODULES:
        source = read_plugin_source(module)
        assert "game_data_store" not in source, f"{module} 引用了 game_data_store"


def test_runtime__fetch_script_is_dev_only_and_out_of_runtime_chain() -> None:
    """取数脚本在 tools/ 下、带"仅开发期"标记，且不被运行时模块 import。"""
    script = PLUGIN_DIR / "tools" / "fetch_genshin_db.py"
    assert script.is_file(), "缺少 tools/fetch_genshin_db.py"
    source = script.read_text(encoding="utf-8")
    assert "仅开发期使用，不随包运行" in source, "取数脚本缺少'仅开发期使用，不随包运行'标记"
    for module in PLUGIN_MODULES:
        assert "fetch_genshin_db" not in read_plugin_source(module), f"{module} 引用了取数脚本"
