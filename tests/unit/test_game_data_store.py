"""任务 B：原神离线数据（武器 / 圣遗物 / 材料）的落地与查询回归。

覆盖三件事：

1. **数据集形状**：条目数、字段完整性、文件头版本锚点；
2. **查询能力**：按中文名 / slug 精确查、按关键字模糊查；
3. **运行时零网络**：查询模块不 import 任何网络库、不发起网络调用；
   取数脚本带"仅开发期使用"标记且不在运行时 import 链上。

★ 2026-11-10（P0-1）修正：查询模块**已接入**运行时 import 链（只有入口
  ``__init__.py`` 引用它）——"数据落地但不接线"让三张表运行时够不着。
  零网络这条没变，只是"不 import 查询模块"换成了"import 查询模块但不联网"。

★ 2026-11-11（数据扩全量）：三张表从**最小集**（武器 10 / 圣遗物 5 / 材料 20）
  扩到**上游全量**（武器 256→去重 254 / 圣遗物 63 / 材料 931→去重 916）。
  本文件随之做了三类改动：

  1. 写死条目数改成**下界**断言（全量只会变多，不写脆弱的等号）；
  2. 字段完整性断言按**上游真实缺口**放宽（见各用例注释），
     不给上游没有的字段编造值；
  3. 新增"全量抽样查询"用例，覆盖最小集里查不到的条目。
"""

from __future__ import annotations

import ast
import tomllib

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

#: 各表条数的**下界**（上游全量规模：武器 256 / 圣遗物 63 / 材料 931，
#: 同名去重后 254 / 63 / 916）。只断言下界，上游继续扩充时不会误报。
MIN_ENTRY_COUNTS = {"weapons": 200, "artifacts": 50, "materials": 800}

#: 上游 ``prayer*`` 系列是"祈愿"伪套装：只有头冠一件，没有 2/4 件套效果。
UPSTREAM_PSEUDO_ARTIFACT_PREFIX = "prayer"

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
    """全量数据集的条目数：断言**下界**而不是脆弱的等号。

    ★ 2026-11-11：数据从最小集（10 / 5 / 20）扩到上游全量
      （256 / 63 / 931，去重后 254 / 63 / 916）。上游继续扩充时
      这条用例不应误报，所以钉下界而不是具体数字。
    """
    assert len(data.weapons) >= MIN_ENTRY_COUNTS["weapons"]
    assert len(data.artifacts) >= MIN_ENTRY_COUNTS["artifacts"]
    assert len(data.materials) >= MIN_ENTRY_COUNTS["materials"]


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
    """每把武器至少有名称 / 稀有度 / 类型 / 基础攻击 / 主属性。

    ★ 2026-11-11（全量数据）：1★ / 2★ 的**训练用武器**上游**没有**主属性——
      ``mainStatText`` / ``baseStatText`` 这两个键在原始 JSON 里整个不存在
      （已逐条核对上游：10 条 rarity<=2 的武器全部如此，rarity>=3 全部有）。
      这与游戏内"1/2★ 武器没有属性条"一致，所以 rarity<=2 允许主属性为空，
      rarity>=3 必须有。不给上游没有的字段编造值。
    """
    for weapon in data.weapons:
        assert weapon.name.strip(), f"{weapon.slug} 缺名称"
        assert weapon.rarity >= 1, f"{weapon.name} 缺稀有度"
        assert weapon.weapon_type.strip(), f"{weapon.name} 缺武器类型"
        assert weapon.weapon_type_code.strip(), f"{weapon.name} 缺武器类型码"
        assert weapon.base_atk > 0, f"{weapon.name} 缺基础攻击"
        if weapon.rarity >= 3:
            assert weapon.main_stat.strip(), f"{weapon.name} 缺主属性"
            assert weapon.main_stat_value.strip(), f"{weapon.name} 缺主属性数值"


def test_dataset__artifact_fields_are_complete(data: GameData) -> None:
    """每套圣遗物至少有套装名 / 2 件套 / 4 件套效果。

    ★ 2026-11-11（全量数据）：上游 ``prayer*`` 系列（祭水之人 / 祭光之冠…）
      在游戏里**不是套装**，只是祈愿展示；上游只给了头冠一件
      （``flower``/``plume``/``sands``/``goblet`` 都是 null）、也没有套装效果。
      这类条目按上游事实记录，不编造效果；其余 59 套是真正的套装。
    """
    for artifact in data.artifacts:
        assert artifact.name.strip(), f"{artifact.slug} 缺套装名"
        assert artifact.rarity, f"{artifact.name} 缺可用稀有度"
        if artifact.slug.startswith(UPSTREAM_PSEUDO_ARTIFACT_PREFIX):
            assert len(artifact.pieces) == 1, f"{artifact.name} 上游伪套装只给了一件"
            continue
        assert artifact.effect_2pc.strip(), f"{artifact.name} 缺 2 件套效果"
        assert artifact.effect_4pc.strip(), f"{artifact.name} 缺 4 件套效果"
        assert len(artifact.pieces) == 5, f"{artifact.name} 部位数不对：{artifact.pieces}"


def test_dataset__material_fields_are_complete(data: GameData) -> None:
    """每种材料至少有名称 / 类别码，且"用途说明"与"获取途径"至少有一项。

    ★ 2026-11-11（全量数据）：上游有 29 条材料**根本没有** ``typeText`` 字段
      （纪游经验、店铺饰物、炼金台扩容…，已逐条核对原始 JSON：``type`` 与
      ``typeText`` 都不存在），所以 ``category_text`` 允许为空；
      但 ``category`` 枚举码上游始终给，必须有。
    """
    for material in data.materials:
        assert material.name.strip(), f"{material.slug} 缺名称"
        assert material.category.strip(), f"{material.name} 缺类别码"
        assert material.sources or material.description.strip(), f"{material.name} 既无用途也无获取途径"


def test_dataset__material_category_text_is_upstream_optional(data: GameData) -> None:
    """``category_text`` 缺失是**上游事实**（不是清洗丢字段）。

    多数材料有中文类目、少数没有；没有的那些**仍必须**有类别枚举码，
    否则运行时按类别查不到。
    """
    with_text = [m for m in data.materials if m.category_text.strip()]
    without_text = [m for m in data.materials if not m.category_text.strip()]
    assert len(with_text) > len(without_text), "多数材料的 category_text 应该存在"
    assert without_text, "上游确实存在没有 typeText 的材料"
    assert all(m.category.strip() for m in without_text), "缺中文类别的材料仍必须有类别码"


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
    """关键字模糊查：武器名 / 材料类别 / 套装效果。

    ★ 2026-11-11（全量数据）：``find_weapons("天空")`` 从 1 条变成 5 条
      （天空之卷 / 天空之刃 / 天空之翼 / 天空之傲 / 天空之脊），
      所以这里改成"包含"断言，不再钉等值列表。
    """
    sky = [w.name for w in data.find_weapons("天空")]
    assert "天空之翼" in sky
    assert "天空之刃" in sky
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


def test_summary__material_without_category_text_keeps_balanced_brackets(data: GameData) -> None:
    """上游没给 ``typeText`` 的材料，摘要不能掉成半个括号。

    ★ 2026-11-11（全量数据暴露）：``MaterialEntry.summary()`` 原本对标签做了
      ``.rstrip("】")``，在 ``category_text`` 为空时会把闭合括号也去掉，
      渲染成 ``【药材研析仪``。上游有 29 条材料没有 ``typeText``，
      全量数据一落地就会命中这个坑。
    """
    material = next(m for m in data.materials if not m.category_text.strip())
    line = material.summary()
    assert "\n" not in line
    assert f"【{material.name}】" in line


# =============================================================================
# 3. 全量抽样查询（2026-11-11 数据扩全量）
# =============================================================================


def test_lookup__weapon_sampled_from_full_set(data: GameData) -> None:
    """全量武器抽样：若水（5★ 弓）。最小集里没有。"""
    weapon = data.weapon("若水")
    assert weapon is not None
    assert weapon.slug == "aquasimulacra"
    assert weapon.weapon_type == "弓"
    assert weapon.rarity == 5
    assert weapon.main_stat == "暴击伤害"
    assert weapon.passive.strip()
    assert weapon.ascension_materials


def test_lookup__weapon_skyward_blade_present(data: GameData) -> None:
    """「天空之刃」在**最小集**里没有，全量后必须查得到。

    这是端到端问题「风鹰剑和天空之刃哪个好？」能被回答的前置条件：
    最小集只收录了「天空之翼」，上游其实有整套"天空"系列。
    """
    weapon = data.weapon("天空之刃")
    assert weapon is not None
    assert weapon.slug == "skywardblade"
    assert weapon.weapon_type == "单手剑"
    assert weapon.rarity == 5
    assert weapon.main_stat.strip()


def test_lookup__artifact_sampled_from_full_set(data: GameData) -> None:
    """全量圣遗物抽样：追忆之注连（稻妻 4/5 件套）。最小集里没有。"""
    artifact = data.artifact("追忆之注连")
    assert artifact is not None
    assert artifact.slug == "shimenawasreminiscence"
    assert artifact.rarity == (4, 5)
    assert artifact.effect_2pc.strip()
    assert artifact.effect_4pc.strip()
    assert len(artifact.pieces) == 5


def test_lookup__artifact_second_sample(data: GameData) -> None:
    """全量圣遗物抽样二：悠古的磐岩。"""
    artifact = data.artifact("悠古的磐岩")
    assert artifact is not None
    assert artifact.slug == "archaicpetra"
    assert artifact.rarity == (4, 5)
    assert len(artifact.pieces) == 5


def test_lookup__material_silkflower_sampled(data: GameData) -> None:
    """全量材料抽样：霓裳花（璃月区域特产）。最小集里没有。"""
    material = data.material("霓裳花")
    assert material is not None
    assert material.slug == "silkflower"
    assert material.category == "EXCHANGE"
    assert material.sources
    assert any("采集" in source for source in material.sources)


def test_lookup__material_qingxin_sampled(data: GameData) -> None:
    """全量材料抽样二：清心（璃月区域特产）。"""
    material = data.material("清心")
    assert material is not None
    assert material.slug == "qingxin"
    assert material.sources
    assert any("采集" in source for source in material.sources)


def test_lookup__new_entries_are_reachable_by_fuzzy_search(data: GameData) -> None:
    """新增条目在模糊查询里也够得着（检索层真的用得上，不只是数据在盘上）。"""
    assert any(w.name == "若水" for w in data.find_weapons("若水"))
    assert any(a.name == "追忆之注连" for a in data.find_artifacts("注连"))
    assert any(m.name == "霓裳花" for m in data.find_materials("霓裳"))


def test_lookup__skyward_series_all_present(data: GameData) -> None:
    """上游"天空"系列 5 把全都在（这正是最小集答不了第 2 问的原因）。"""
    names = {w.name for w in data.weapons}
    assert {"天空之卷", "天空之刃", "天空之翼", "天空之傲", "天空之脊"} <= names


# =============================================================================
# 4. 运行时零网络
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


def test_runtime__query_module_is_in_runtime_chain_via_entry_only() -> None:
    """2026-11-10（P0-1）：查询模块**已接入**运行时 import 链。

    ★ 这条断言的**方向变了**。它原本钉的是"查询模块不在运行时 import 链上
      —— 插件运行行为零变化"。那是上一轮接入离线数据时的**有意设计**：
      数据先落地、不接线，留给后面某轮。

      但端到端验证证明了这个设计的代价：数据在盘上、检索层却够不着，
      "风鹰剑""绝缘之旗印""精锻用魔矿"三问插件侧零供给。于是本轮把它
      接进 ``lookup_game_term``（术语库优先，未命中再回落离线表）。

    现在钉住的是"接上了，但不越界"：

    * 确实在链上（否则三张表永远查不到）；
    * **只有** ``__init__.py`` 引用它（入口一处，其余模块不直接依赖）；
    * 仍然零网络（上面两条用例已钉）。
    """
    referrers = [
        module for module in PLUGIN_MODULES if "game_data_store" in read_plugin_source(module)
    ]
    assert referrers == ["__init__.py"], referrers


def test_runtime__fetch_script_is_dev_only_and_out_of_runtime_chain() -> None:
    """取数脚本在 tools/ 下、带"仅开发期"标记，且不被运行时模块 import。"""
    script = PLUGIN_DIR / "tools" / "fetch_genshin_db.py"
    assert script.is_file(), "缺少 tools/fetch_genshin_db.py"
    source = script.read_text(encoding="utf-8")
    assert "仅开发期使用，不随包运行" in source, "取数脚本缺少'仅开发期使用，不随包运行'标记"
    for module in PLUGIN_MODULES:
        assert "fetch_genshin_db" not in read_plugin_source(module), f"{module} 引用了取数脚本"
