"""原神攻略数据（武器 / 圣遗物 / 材料）的**离线**查询模块。

数据来源：``terms/genshin/{weapons,artifacts,materials}.toml``，
由 ``tools/fetch_genshin_db.py`` 在**开发期**从社区数据源 ``genshin-db``
拉取、清洗后落地（上游 commit 固定在文件头，可复现）。

**运行时零网络**：本模块只读本地 TOML 文件。它**不 import** 任何网络库
（``urllib`` / ``requests`` / ``socket`` / ``http`` 一律不碰），
也不发起任何 IO 以外的动作；数据拉取是开发期一次性动作，已经完成。

**模块边界**：本模块是"按需调用的离线查询工具"。

★ 2026-11-10：本模块**已接入**运行时 import 链（由插件入口 ``__init__.py``
  引用），``lookup_game_term`` 在术语库未命中时回落到这里的武器 / 圣遗物 /
  材料表。此前它"不被任何运行时入口 import"，导致三张表数据在盘上、检索层
  却够不着（回归见 ``tests/unit/test_offline_lookup.py``）。
  **仍然零网络**：接入的是 import 关系，不是 IO。

覆盖层（用户侧改写）机制目前只作用于 ``term_store`` 的四个术语维度与
``scenes.toml``；本模块读取的是包内只读数据文件，不支持覆盖层。

★ 2026-11-11：数据从最小集扩到**上游全量**（武器 254 / 圣遗物 63 / 材料 916）。
查询接口未变；只有 ``MaterialEntry.summary()`` 的标签渲染修了一处
（上游 29 条材料没有 ``typeText``，旧的 ``.rstrip("】")`` 会渲染出半个括号）。

用法::

    from game_data_store import load_game_data

    data = load_game_data()
    data.weapon("风鹰剑")
    data.artifact("绝缘之旗印")
    data.material("原石")
    data.find_weapons("天空")
"""

from __future__ import annotations

import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "ArtifactEntry",
    "GameData",
    "MaterialEntry",
    "WeaponEntry",
    "default_terms_dir",
    "load_game_data",
    "normalize_key",
]


# ---------------------------------------------------------------------------
# 条目模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeaponEntry:
    """一把武器。

    ⚠️ 上游 ``genshin-db`` 的 ``ChineseSimplified/weapons/*.json``
    **没有**副属性（affix）列表，因此本模型也没有 ``affix`` 字段——
    不造上游没有的数据。``base_atk`` 是上游给出的浮点基础攻击值。
    """

    name: str
    slug: str
    rarity: int
    weapon_type: str
    weapon_type_code: str
    base_atk: float
    main_stat: str
    main_stat_value: str
    passive: str
    passive_detail: str
    ascension_materials: tuple[str, ...] = ()

    def summary(self) -> str:
        """一行摘要，给 LLM 直接用。"""
        parts = [f"【{self.name}】{self.weapon_type}·{self.rarity}星"]
        parts.append(f"基础攻击 {self.base_atk:g}")
        if self.main_stat:
            parts.append(f"{self.main_stat} {self.main_stat_value}".strip())
        if self.passive:
            parts.append(f"特性：{self.passive}")
        return "，".join(parts)


@dataclass(frozen=True)
class ArtifactEntry:
    """一套圣遗物。"""

    name: str
    slug: str
    rarity: tuple[int, ...]
    effect_2pc: str
    effect_4pc: str
    pieces: tuple[str, ...] = ()

    def summary(self) -> str:
        """一行摘要，给 LLM 直接用。"""
        parts = [f"【{self.name}】"]
        if self.rarity:
            parts.append("/".join(f"{value}星" for value in self.rarity))
        if self.effect_2pc:
            parts.append(f"2件套：{self.effect_2pc}")
        if self.effect_4pc:
            parts.append(f"4件套：{self.effect_4pc}")
        return "，".join(parts)


@dataclass(frozen=True)
class MaterialEntry:
    """一种材料。

    ``rarity`` 为 ``0`` 表示上游未标注稀有度（不是"0 星"）。
    """

    name: str
    slug: str
    rarity: int
    category: str
    category_text: str
    sources: tuple[str, ...] = ()
    description: str = ""

    def summary(self) -> str:
        """一行摘要，给 LLM 直接用。

        ★ 2026-11-11：标签不再 ``.rstrip("】")``。上游有 29 条材料**没有**
          ``typeText``，``rstrip`` 会把闭合括号一并去掉、渲染成 ``【药材研析仪``。
          现在空类别时保留成对括号 ``【药材研析仪】``。
        """
        parts = [f"【{self.name}】{self.category_text}"]
        if self.sources:
            parts.append("获取途径：" + "、".join(self.sources))
        elif self.description:
            parts.append(self.description.splitlines()[0])
        return "，".join(parts)


@dataclass(frozen=True)
class GameData:
    """一次性载入的武器 / 圣遗物 / 材料快照。"""

    weapons: tuple[WeaponEntry, ...] = ()
    artifacts: tuple[ArtifactEntry, ...] = ()
    materials: tuple[MaterialEntry, ...] = ()
    generated_at: str = ""
    source: str = ""
    source_commit: str = ""

    # -- 精确查询 -----------------------------------------------------------

    def weapon(self, name: str) -> WeaponEntry | None:
        """按名称或 slug 精确查一把武器。"""
        return self._find_one(self.weapons, name)

    def artifact(self, name: str) -> ArtifactEntry | None:
        """按名称或 slug 精确查一套圣遗物。"""
        return self._find_one(self.artifacts, name)

    def material(self, name: str) -> MaterialEntry | None:
        """按名称或 slug 精确查一种材料。"""
        return self._find_one(self.materials, name)

    # -- 模糊查询 -----------------------------------------------------------

    def find_weapons(self, keyword: str) -> tuple[WeaponEntry, ...]:
        """名称/类型/特性里含关键字的武器。"""
        return self._find_many(self.weapons, keyword)

    def find_artifacts(self, keyword: str) -> tuple[ArtifactEntry, ...]:
        """名称/套装效果里含关键字的圣遗物。"""
        return self._find_many(self.artifacts, keyword)

    def find_materials(self, keyword: str) -> tuple[MaterialEntry, ...]:
        """名称/类别/说明里含关键字的材料。"""
        return self._find_many(self.materials, keyword)

    # -- 内部实现 -----------------------------------------------------------

    @staticmethod
    def _find_one(entries: Iterable[object], needle: str):
        key = normalize_key(needle)
        if not key:
            return None
        for entry in entries:
            if key in (normalize_key(entry.name), normalize_key(entry.slug)):
                return entry
        return None

    @staticmethod
    def _find_many(entries: Iterable[object], keyword: str) -> tuple[object, ...]:
        key = normalize_key(keyword)
        if not key:
            return ()
        hits = []
        for entry in entries:
            haystack = " ".join(
                normalize_key(part)
                for part in (
                    entry.name,
                    getattr(entry, "slug", ""),
                    getattr(entry, "weapon_type", ""),
                    getattr(entry, "passive", ""),
                    getattr(entry, "category_text", ""),
                    getattr(entry, "description", ""),
                    getattr(entry, "effect_2pc", ""),
                    getattr(entry, "effect_4pc", ""),
                )
                if part
            )
            if key in haystack:
                hits.append(entry)
        return tuple(hits)


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------


def normalize_key(text: str) -> str:
    """查询键归一化：NFKC（兼容全角）+ 去首尾空白 + 大小写折叠。"""
    return unicodedata.normalize("NFKC", str(text or "")).strip().casefold()


def default_terms_dir() -> Path:
    """包内数据目录（相对本模块定位，不含任何机器绝对路径）。"""
    return Path(__file__).resolve().parent / "terms" / "genshin"


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _load_weapons(payload: dict) -> tuple[WeaponEntry, ...]:
    return tuple(
        WeaponEntry(
            name=str(item.get("name") or ""),
            slug=str(item.get("slug") or ""),
            rarity=int(item.get("rarity") or 0),
            weapon_type=str(item.get("weapon_type") or ""),
            weapon_type_code=str(item.get("weapon_type_code") or ""),
            base_atk=float(item.get("base_atk") or 0.0),
            main_stat=str(item.get("main_stat") or ""),
            main_stat_value=str(item.get("main_stat_value") or ""),
            passive=str(item.get("passive") or ""),
            passive_detail=str(item.get("passive_detail") or ""),
            ascension_materials=_as_str_tuple(item.get("ascension_materials")),
        )
        for item in payload.get("weapons") or []
    )


def _load_artifacts(payload: dict) -> tuple[ArtifactEntry, ...]:
    return tuple(
        ArtifactEntry(
            name=str(item.get("name") or ""),
            slug=str(item.get("slug") or ""),
            rarity=tuple(int(value) for value in (item.get("rarity") or [])),
            effect_2pc=str(item.get("effect_2pc") or ""),
            effect_4pc=str(item.get("effect_4pc") or ""),
            pieces=_as_str_tuple(item.get("pieces")),
        )
        for item in payload.get("artifacts") or []
    )


def _load_materials(payload: dict) -> tuple[MaterialEntry, ...]:
    return tuple(
        MaterialEntry(
            name=str(item.get("name") or ""),
            slug=str(item.get("slug") or ""),
            rarity=int(item.get("rarity") or 0),
            category=str(item.get("category") or ""),
            category_text=str(item.get("category_text") or ""),
            sources=_as_str_tuple(item.get("sources")),
            description=str(item.get("description") or ""),
        )
        for item in payload.get("materials") or []
    )


def load_game_data(terms_dir: Path | str | None = None) -> GameData:
    """载入三份数据文件。

    :param terms_dir: 数据目录，默认取包内 ``terms/genshin``。
    :raises FileNotFoundError: 目录或文件缺失。
    """
    root = Path(terms_dir).resolve() if terms_dir is not None else default_terms_dir()
    weapons_payload = _read_toml(root / "weapons.toml")
    artifacts_payload = _read_toml(root / "artifacts.toml")
    materials_payload = _read_toml(root / "materials.toml")
    return GameData(
        weapons=_load_weapons(weapons_payload),
        artifacts=_load_artifacts(artifacts_payload),
        materials=_load_materials(materials_payload),
        generated_at=str(weapons_payload.get("generated_at") or ""),
        source=str(weapons_payload.get("source") or ""),
        source_commit=str(weapons_payload.get("source_commit") or ""),
    )
