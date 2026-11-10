"""从社区数据源 ``genshin-db`` 拉取并清洗原神武器 / 圣遗物 / 材料数据，落地为本地 TOML。

⚠️ **仅开发期使用，不随包运行。**

- 本脚本**不是**插件运行时的一部分：它不被任何运行时模块 import，
  插件运行时**零网络**（运行时的查询入口 ``game_data_store.py`` 只读本地 TOML，
  连 ``urllib`` 都不 import）。
- 拉取是**开发期一次性动作**：跑一次 → 产出 ``terms/genshin/*.toml`` → 提交进仓。
  之后插件运行、离线查表都不再需要网络。
- 数据源：``theBowja/genshin-db``（npm 包 ``genshin-db`` 的上游仓库），
  **固定到某个 commit**，保证任何人重跑都得到同一份数据。

用法::

    python tools/fetch_genshin_db.py                          # 全量，落到 terms/genshin/
    python tools/fetch_genshin_db.py --scope minimal          # 最小集（原先的选品）
    python tools/fetch_genshin_db.py --dry-run                # 只读报告，不落盘、不打印全文
    python tools/fetch_genshin_db.py --dry-run --print-content  # 只读 + 打印将要写出的 TOML
    python tools/fetch_genshin_db.py --out-dir /tmp/xx        # 落到别处（便于对比）
    python tools/fetch_genshin_db.py --material-categories whitelist  # 材料只留实体材料

取数通道（按顺序回退，都是只读 GET）：

0. ``codeload.github.com/<repo>/tar.gz/<commit>`` —— **首选**：一次请求拿到 pinned
   commit 的**完整源码快照**，既不受 GitHub REST API 60 次/小时的匿名限额约束，
   也不需要逐条 CDN 请求（1000+ 次）。只读 gzip 流，不落盘、不做路径提取。
1. ``fastly.jsdelivr.net/gh/<repo>@<commit>/...`` —— CDN 镜像，无速率限制；
2. ``api.github.com/repos/<repo>/contents/...`` —— 官方 API，有速率限制。

1 与 2 返回的是**同一份** JSON（已逐字段比对过），只是渠道不同。

目录列举（tarball 不可用时的兜底，只请求 3 次）：

1. ``api.github.com/repos/<repo>/contents/src/data/ChineseSimplified/<kind>?ref=<commit>``；
2. ``api.github.com/repos/<repo>/git/trees/<commit>?recursive=1``。

详情抓取走 jsDelivr 并发（默认 8 线程），GitHub API 只在 CDN 失败时兜底，
避免触发 60 次/小时的未授权 API 限额。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# 上游坐标（固定 commit = 可复现）
# ---------------------------------------------------------------------------

UPSTREAM_REPO = "theBowja/genshin-db"
PINNED_COMMIT = "8e5badd60b5d894284ccc927a56c9683529d2592"
UPSTREAM_LICENSE = "MIT（上游仓库 LICENSE；本插件只取名称/数值/效果描述等事实性数据）"

JSDELIVR_BASE = f"https://fastly.jsdelivr.net/gh/{UPSTREAM_REPO}@{PINNED_COMMIT}/src/data"
GITHUB_BASE = f"https://api.github.com/repos/{UPSTREAM_REPO}/contents/src/data"

#: 源码 tarball 通道：一次请求拿到 **pinned commit 的完整快照**。
#:
#: 不用 GitHub REST API，因此不受 60 次/小时的匿名限额约束；
#: 也不用逐条 CDN 请求（1000+ 次），一次压缩包就够。
CODELOAD_BASE = f"https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/{PINNED_COMMIT}"

#: tarball 体积上限（防止异常响应把内存吃满）。
MAX_TARBALL_BYTES = 256 * 1024 * 1024

#: 拉取失败的重试次数与间隔（秒）。
FETCH_TRIES = 5
FETCH_RETRY_WAIT = 10.0

USER_AGENT = "genshin-game-buddy-dev-fetch"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "terms" / "genshin"

# ---------------------------------------------------------------------------
# 选品：常见、稳定的条目，不追求覆盖
# ---------------------------------------------------------------------------

#: 武器 10 把（均为 5★）
WEAPON_SLUGS: tuple[str, ...] = (
    "aquilafavonia",          # 风鹰剑
    "staffofhoma",            # 护摩之杖
    "mistsplitterreforged",   # 雾切之回光
    "skywardharp",            # 天空之翼
    "amosbow",                # 阿莫斯之弓
    "amenomakageuchi",        # 天目影打刀
    "engulfinglightning",     # 薙草之稻光
    "calamityqueller",        # 息灾
    "primordialjadewingedspear",  # 和璞鸢
    "primordialjadecutter",   # 磐岩结绿
)

#: 圣遗物套装 5 套
ARTIFACT_SLUGS: tuple[str, ...] = (
    "emblemofseveredfate",         # 绝缘之旗印
    "heartofdepth",                # 沉沦之心
    "adaycarvedfromrisingwinds",   # 风起之日
    "noblesseoblige",              # 昔日宗室之仪
    "gladiatorsfinale",            # 角斗士的终幕礼
)

#: 材料 20 种（覆盖货币 / 经验 / 突破 / 采集 / 地域特产 / 神瞳 / 印）
MATERIAL_SLUGS: tuple[str, ...] = (
    "primogem",                # 原石
    "mora",                    # 摩拉
    "companionshipexp",        # 好感经验
    "bep",                     # 纪行经验
    "heroswit",                # 大英雄的经验
    "adventurersexperience",   # 冒险家的经验
    "mysticenhancementore",   # 精锻用魔矿
    "rawmeat",                # 兽肉
    "fowl",                    # 禽肉
    "fish",                    # 鱼肉
    "anemoculus",              # 散失的风神瞳
    "geoculus",                # 散失的岩神瞳
    "anemosigil",              # 风之印
    "electrosigil",            # 雷之印
    "jadebranchofadistantsea",  # 远海夷地的琼枝
    "coralbranchofadistantsea",  # 远海夷地的瑚枝
    "nakuweed",                # 鸣草
    "everflameseed",           # 常燃火种
    "cloudseamscale",          # 凝云鳞甲
    "lightningprism",          # 雷光棱镜
)

#: 圣遗物 5 个部位在 JSON 里的字段名（按 花→羽→沙→杯→冠 顺序）
ARTIFACT_PIECE_FIELDS = ("flower", "plume", "sands", "goblet", "circlet")

# ---------------------------------------------------------------------------
# 全量选品：由上游目录列举得到，不写死
# ---------------------------------------------------------------------------

#: 全量模式下三类的上游目录名（``src/data/ChineseSimplified/<kind>/``）。
FULL_FOLDERS = ("weapons", "artifacts", "materials")

#: 材料 ``category`` 白名单（上游英文枚举码），**默认不启用**。
#:
#: 只保留"可采集 / 可购买 / 可合成"的**实体**材料；纯剧情道具、货币、测试项排除。
#: 取值来自上游 JSON 的 ``category`` 字段（英文枚举码），不是 ``typeText`` 中文名
#: （上游 ``typeText`` 是**细子类目**，同一个 ``category`` 下会有多个不同中文名，
#:  例如 ``EXCHANGE`` 下面既有鱼也有螃蟹）。
#:
#: 上游 931 条材料的完整 category 分布见 ``--dry-run`` 的输出；
#: 排除项是 ``ADSORBATE``（通用货币：原石 / 摩拉）与 ``ITEM_VIRTUAL``（系统虚拟项）。
#: 用 ``--material-categories whitelist`` 启用这份白名单。
MATERIAL_CATEGORY_WHITELIST: frozenset[str] = frozenset(
    {
        "AVATAR_MATERIAL",  # 角色与武器培养素材
        "CONSUME",  # 消耗品
        "ELEM_CRYSTAL",  # 元素结晶 / 冒险道具
        "EXP_FRUIT",  # 角色经验素材
        "FISH_BAIT",  # 鱼饵
        "FISH_ROD",  # 鱼竿
        "NOTICE_ADD_HP",  # 食物
        "WEAPON_EXP_STONE",  # 武器强化素材
        "WOOD",  # 木材
        "EXCHANGE",  # 兑换物（鱼 / 肉 / 贝壳等采集物）
    }
)

#: 并发抓取线程数（jsDelivr 侧无速率限制，8 已足够）。
DEFAULT_WORKERS = 8


# ---------------------------------------------------------------------------
# 拉取
# ---------------------------------------------------------------------------


def _http_get(url: str, *, timeout: int = 60) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _fetch_from_github(path: str) -> dict:
    payload = json.loads(_http_get(f"{GITHUB_BASE}/{path}?ref={PINNED_COMMIT}"))
    if payload.get("encoding") == "base64":
        return json.loads(base64.b64decode(payload["content"]).decode("utf-8"))
    return payload


def fetch_item(folder: str, slug: str, *, tries: int = 4) -> dict:
    """取 ``ChineseSimplified/<folder>/<slug>.json``，两个通道都试，带退避重试。"""
    path = f"ChineseSimplified/{folder}/{slug}.json"
    last_error: Exception | None = None
    for attempt in range(tries):
        for loader in (
            lambda: json.loads(_http_get(f"{JSDELIVR_BASE}/{path}")),
            lambda: _fetch_from_github(path),
        ):
            try:
                return loader()
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 404:
                    break  # 这条路径确实不存在，换通道也没用
            except Exception as exc:  # noqa: BLE001 —— 网络层什么都可能抛
                last_error = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"拉取失败：{path}（{last_error}）")


# ---------------------------------------------------------------------------
# 全量发现：列出上游目录下的全部 slug
# ---------------------------------------------------------------------------


def _list_from_contents(folder: str) -> list[str]:
    """用 contents API 列目录。GitHub 单目录上限 1000 条，超出视为截断。"""
    payload = json.loads(_http_get(f"{GITHUB_BASE}/ChineseSimplified/{folder}?ref={PINNED_COMMIT}"))
    if not isinstance(payload, list):
        raise RuntimeError(f"contents API 返回了非列表（folder={folder}）")
    names = [
        str(item.get("name") or "")
        for item in payload
        if item.get("type") == "file" and str(item.get("name") or "").endswith(".json")
    ]
    if len(names) >= 1000:
        raise RuntimeError(f"contents API 疑似被截断（folder={folder}，{len(names)} 条）")
    return sorted(name[:-5] for name in names)


def _list_from_tree(folder: str) -> list[str]:
    """用 git tree API 兜底列举；``truncated`` 时宁可报错也不静默少数据。"""
    payload = json.loads(
        _http_get(f"https://api.github.com/repos/{UPSTREAM_REPO}/git/trees/{PINNED_COMMIT}?recursive=1")
    )
    if payload.get("truncated"):
        raise RuntimeError(f"tree API 返回 truncated（folder={folder}），拒绝使用不完整清单")
    prefix = f"src/data/ChineseSimplified/{folder}/"
    names = []
    for node in payload.get("tree") or []:
        path = str(node.get("path") or "")
        if node.get("type") == "blob" and path.startswith(prefix) and path.endswith(".json"):
            names.append(path[len(prefix) : -5])
    return sorted(names)


def list_slugs(folder: str) -> list[str]:
    """列出上游 ``ChineseSimplified/<folder>/`` 下的全部 slug（已按名称排序）。"""
    problems: list[str] = []
    for loader in (_list_from_contents, _list_from_tree):
        try:
            slugs = loader(folder)
        except Exception as exc:  # noqa: BLE001 —— 网络层什么都可能抛
            problems.append(f"{loader.__name__}: {exc}")
            continue
        if slugs:
            return slugs
        problems.append(f"{loader.__name__}: 目录为空")
    raise RuntimeError(f"无法列举上游目录 {folder}：{'; '.join(problems)}")


# ---------------------------------------------------------------------------
# 源码 tarball：一次请求拿到 pinned commit 的完整快照
# ---------------------------------------------------------------------------


def _http_get_bytes(url: str, *, timeout: int = 180, max_bytes: int = MAX_TARBALL_BYTES) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RuntimeError(f"响应超过体积上限（{max_bytes} 字节）：{url}")
    return data


def download_tarball(*, tries: int = FETCH_TRIES, wait: float = FETCH_RETRY_WAIT) -> bytes:
    """下载 pinned commit 的源码 tarball（gzip 字节），失败按固定间隔重试。"""
    last_error: Exception | None = None
    for attempt in range(tries):
        try:
            return _http_get_bytes(CODELOAD_BASE)
        except Exception as exc:  # noqa: BLE001 —— 网络层什么都可能抛
            last_error = exc
            if attempt < tries - 1:
                time.sleep(wait)
    raise RuntimeError(f"下载源码 tarball 失败（重试 {tries} 次）：{last_error}")


class TarballIndex:
    """tarball 内的 ``ChineseSimplified/{weapons,artifacts,materials}/*.json`` 索引。

    只读不落盘：直接从 gzip 流里读成员字节，不做路径提取，
    因此不存在 tar 路径穿越 / 覆盖本地文件的风险。
    """

    def __init__(self, payload: bytes) -> None:
        self.entries: dict[str, dict[str, bytes]] = {folder: {} for folder in FULL_FOLDERS}
        prefix = "src/data/ChineseSimplified/"
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                parts = member.name.split("/")
                relative = "/".join(parts[1:])
                if not relative.startswith(prefix):
                    continue
                folder, _, filename = relative[len(prefix) :].partition("/")
                if folder not in self.entries or "/" in filename:
                    continue
                if not filename.endswith(".json"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                self.entries[folder][filename[:-5]] = handle.read()
        missing = [folder for folder, items in self.entries.items() if not items]
        if missing:
            raise RuntimeError(f"源码 tarball 里没有这些目录：{missing}")

    def slugs(self, folder: str) -> list[str]:
        return sorted(self.entries.get(folder, {}))

    def raw(self, folder: str, slug: str) -> bytes:
        return self.entries[folder][slug]


def fetch_many(
    kind: str,
    slugs: list[str],
    *,
    workers: int = DEFAULT_WORKERS,
    tries: int = 3,
) -> dict[str, dict]:
    """并发拉取一批条目，返回 ``{slug: raw_json}``；任一条失败则整体失败。"""
    results: dict[str, dict] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch_item, kind, slug, tries=tries): slug for slug in slugs}
        for done, future in enumerate(as_completed(futures), start=1):
            slug = futures[future]
            try:
                results[slug] = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{slug}: {exc}")
            if done % 50 == 0 or done == len(slugs):
                print(f"  [{kind}] 已拉取 {done}/{len(slugs)}")
    if failures:
        raise RuntimeError(f"拉取 {kind} 失败 {len(failures)}/{len(slugs)} 条，例：{failures[:3]}")
    return results


# ---------------------------------------------------------------------------
# 清洗：只留"查询和攻略提示真正需要"的字段
# ---------------------------------------------------------------------------


def clean_weapon(raw: dict, slug: str) -> dict:
    """武器 → 名称 / 类型 / 稀有度 / 基础攻击 / 主属性 / 武器特性 / 突破材料。

    ⚠️ 上游**没有**副属性（affix）列表：``ChineseSimplified/weapons/*.json``
    的键只有 ``baseAtkValue`` / ``mainStatText`` / ``baseStatText`` 等，
    不含游戏内武器的副词条。故此处不造"副属性"字段。
    """
    costs = raw.get("costs") or {}
    ascension: list[str] = []
    for stage in ("ascend1", "ascend2", "ascend3", "ascend4", "ascend5"):
        for item in costs.get(stage) or []:
            name = str(item.get("name") or "").strip()
            if name and name not in ascension:
                ascension.append(name)

    rank1 = raw.get("r1") or {}
    return {
        "name": str(raw.get("name") or "").strip(),
        "slug": slug,
        "rarity": int(raw.get("rarity") or 0),
        "weapon_type": str(raw.get("weaponText") or "").strip(),
        "weapon_type_code": str(raw.get("weaponType") or "").strip(),
        "base_atk": float(raw.get("baseAtkValue") or 0.0),
        "main_stat": str(raw.get("mainStatText") or "").strip(),
        "main_stat_value": str(raw.get("baseStatText") or "").strip(),
        "passive": str(raw.get("effectName") or "").strip(),
        "passive_detail": str(rank1.get("description") or "").strip(),
        "ascension_materials": ascension,
    }


def clean_artifact(raw: dict, slug: str) -> dict:
    """圣遗物 → 套装名 / 可用稀有度 / 2 件套 / 4 件套 / 五件部位名。"""
    pieces: list[str] = []
    for field in ARTIFACT_PIECE_FIELDS:
        node = raw.get(field) or {}
        name = str(node.get("name") or "").strip()
        if name:
            pieces.append(name)
    return {
        "name": str(raw.get("name") or "").strip(),
        "slug": slug,
        "rarity": [int(value) for value in (raw.get("rarityList") or [])],
        "effect_2pc": str(raw.get("effect2Pc") or "").strip(),
        "effect_4pc": str(raw.get("effect4Pc") or "").strip(),
        "pieces": pieces,
    }


def clean_material(raw: dict, slug: str) -> dict:
    """材料 → 名称 / 稀有度 / 类别 / 用途说明 / 获取途径。

    ``rarity`` 上游缺省时记 ``0``（表示"上游未标注"，不是"0 星"）。
    """
    sources = [str(value).strip() for value in (raw.get("sources") or []) if str(value).strip()]
    return {
        "name": str(raw.get("name") or "").strip(),
        "slug": slug,
        "rarity": int(raw.get("rarity") or 0),
        "category": str(raw.get("category") or "").strip(),
        "category_text": str(raw.get("typeText") or "").strip(),
        "sources": sources,
        "description": str(raw.get("description") or "").strip(),
    }


# ---------------------------------------------------------------------------
# TOML 输出（stdlib 没有 writer，这里写一个够用且确定的最小实现）
# ---------------------------------------------------------------------------


def _toml_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"不支持的 TOML 值类型：{type(value)!r}")


def render_toml(*, header: dict, table: str, rows: list[dict]) -> str:
    """渲染成 TOML 文本：顶层键 + 一张 ``[[table]]`` 数组表。"""
    lines: list[str] = []
    for key, value in header.items():
        lines.append(f"{key} = {_toml_value(value)}")
    lines.append("")
    for row in rows:
        lines.append(f"[[{table}]]")
        for key, value in row.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def build_header(kind: str, generated_at: str) -> dict:
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "source": f"genshin-db ({UPSTREAM_REPO})",
        "source_repo": f"https://github.com/{UPSTREAM_REPO}",
        "source_commit": PINNED_COMMIT,
        "source_path": f"src/data/ChineseSimplified/{kind}/*.json",
        "source_license": UPSTREAM_LICENSE,
    }


# ---------------------------------------------------------------------------
# 过滤与去重
# ---------------------------------------------------------------------------


def dedupe_by_name(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """按 ``name`` 去重，保留**首条**；返回 (去重后行, 被去掉的重复名列表)。

    同名条目会让运行时"按名称精确查"产生歧义（``game_data_store._find_one``），
    所以三张表都按名称去重；材料表最常见（上游同一材料会拆成多个 slug）。
    """
    seen: set[str] = set()
    kept: list[dict] = []
    duplicates: list[str] = []
    for row in rows:
        name = row["name"]
        if name in seen:
            duplicates.append(name)
            continue
        seen.add(name)
        kept.append(row)
    return kept, duplicates


def filter_by_category(
    rows: list[dict], whitelist: frozenset[str] | None
) -> tuple[list[dict], dict[str, int]]:
    """按材料 ``category`` 白名单过滤；返回 (保留行, 各类别被排除条数)。

    ``whitelist`` 为 ``None`` 表示不过滤。
    """
    if not whitelist:
        return list(rows), {}
    kept: list[dict] = []
    excluded: dict[str, int] = {}
    for row in rows:
        code = row.get("category") or ""
        if code in whitelist:
            kept.append(row)
        else:
            excluded[code] = excluded.get(code, 0) + 1
    return kept, excluded


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def collect(
    kind: str,
    slugs: list[str],
    generated_at: str,
    *,
    workers: int,
    index: TarballIndex | None,
) -> dict:
    """拉取一批条目并清洗，返回 ``{"header", "rows", "stats"}``（未过滤、未去重）。"""
    cleaners = {
        "weapons": clean_weapon,
        "artifacts": clean_artifact,
        "materials": clean_material,
    }
    clean = cleaners[kind]
    if index is not None:
        raw_map = {slug: json.loads(index.raw(kind, slug).decode("utf-8")) for slug in slugs}
    else:
        raw_map = fetch_many(kind, slugs, workers=workers)
    rows: list[dict] = []
    for slug in slugs:
        row = clean(raw_map[slug], slug)
        if not row["name"]:
            raise RuntimeError(f"上游条目没有名称：{kind}/{slug}")
        rows.append(row)

    stats: dict = {"upstream_count": len(slugs), "cleaned_count": len(rows)}
    if kind == "materials":
        histogram: dict[str, int] = {}
        texts: dict[str, str] = {}
        for row in rows:
            code = row.get("category") or "(空)"
            histogram[code] = histogram.get(code, 0) + 1
            texts.setdefault(code, row.get("category_text") or "")
        stats["category_histogram"] = histogram
        stats["category_texts"] = texts
    return {"header": build_header(kind, generated_at), "rows": rows, "stats": stats}


def _resolve_plan(scope: str, index: TarballIndex | None) -> list[tuple[str, list[str]]]:
    """按 scope 决定每一类要拉哪些 slug。"""
    if scope == "minimal":
        return [
            ("weapons", list(WEAPON_SLUGS)),
            ("artifacts", list(ARTIFACT_SLUGS)),
            ("materials", list(MATERIAL_SLUGS)),
        ]
    plan: list[tuple[str, list[str]]] = []
    for folder in FULL_FOLDERS:
        if index is not None:
            slugs = index.slugs(folder)
        else:
            slugs = list_slugs(folder)
        print(f"=== 列举上游 {folder}：{len(slugs)} 条 ===")
        plan.append((folder, slugs))
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 genshin-db 并落地本地 TOML（仅开发期使用）")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="输出目录，默认 terms/genshin",
    )
    parser.add_argument(
        "--scope",
        choices=("minimal", "full"),
        default="full",
        help="minimal=写死的最小集；full=列举上游全量目录（默认）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读：抓取并报告条目数/体积/类别分布，不落盘",
    )
    parser.add_argument(
        "--print-content",
        action="store_true",
        help="配合 --dry-run 打印将要写出的 TOML 全文",
    )
    parser.add_argument(
        "--material-categories",
        default="all",
        help="材料 category 过滤：'all'（默认，不过滤）/ 'whitelist'（脚本内置白名单）/ 逗号分隔的枚举码",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并发抓取线程数（默认 {DEFAULT_WORKERS}）",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "tarball", "http"),
        default="auto",
        help="auto=优先源码 tarball，失败回退 HTTP；tarball=只用 tarball；http=只用逐条 HTTP",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S+08:00", time.localtime())

    if args.material_categories is None or args.material_categories.strip().lower() == "all":
        whitelist: frozenset[str] | None = None
    elif args.material_categories.strip().lower() == "whitelist":
        whitelist = MATERIAL_CATEGORY_WHITELIST
    else:
        whitelist = frozenset(
            part.strip() for part in args.material_categories.split(",") if part.strip()
        )

    filenames = {"weapons": "weapons.toml", "artifacts": "artifacts.toml", "materials": "materials.toml"}
    report: list[dict] = []
    total_full_bytes = 0
    total_final_bytes = 0

    index: TarballIndex | None = None
    if args.scope == "full" and args.source in ("auto", "tarball"):
        print(f"=== 下载源码 tarball（pinned commit {PINNED_COMMIT[:12]}…）===")
        try:
            payload = download_tarball()
            index = TarballIndex(payload)
            print(
                "  tarball 就绪："
                + "，".join(f"{folder} {len(index.slugs(folder))} 条" for folder in FULL_FOLDERS)
                + f"（{len(payload)} 字节）"
            )
        except Exception as exc:  # noqa: BLE001
            if args.source == "tarball":
                raise
            print(f"  [warn] tarball 不可用（{exc}），回退到目录列举 + 逐条 HTTP 抓取")
            index = None

    for kind, slugs in _resolve_plan(args.scope, index):
        print(f"=== 拉取 {kind}（{len(slugs)} 条）===")
        result = collect(kind, slugs, generated_at, workers=args.workers, index=index)
        header = result["header"]
        rows = result["rows"]
        stats = result["stats"]

        full_text = render_toml(header=header, table=kind, rows=rows)
        full_bytes = len(full_text.encode("utf-8"))

        if kind == "materials":
            filtered, excluded = filter_by_category(rows, whitelist)
        else:
            filtered, excluded = rows, {}
        deduped, duplicates = dedupe_by_name(filtered)
        final_text = render_toml(header=header, table=kind, rows=deduped)
        final_bytes = len(final_text.encode("utf-8"))

        total_full_bytes += full_bytes
        total_final_bytes += final_bytes
        report.append(
            {
                "kind": kind,
                "upstream_count": stats["upstream_count"],
                "cleaned_count": stats["cleaned_count"],
                "full_bytes": full_bytes,
                "final_count": len(deduped),
                "final_bytes": final_bytes,
                "excluded": excluded,
                "duplicates": duplicates,
                "stats": stats,
                "final_text": final_text,
            }
        )

        print(f"  上游 {stats['upstream_count']} 条 -> 清洗后 {stats['cleaned_count']} 条")
        if kind == "materials":
            histogram = stats.get("category_histogram", {})
            texts = stats.get("category_texts", {})
            print(f"  category 分布（{len(histogram)} 类）：")
            for code, count in sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    {code:<28} {count:>5}  {texts.get(code, '')}")
            if excluded:
                print(f"  白名单外被排除 {sum(excluded.values())} 条：")
                for code, count in sorted(excluded.items(), key=lambda kv: (-kv[1], kv[0])):
                    print(f"    {code:<28} {count:>5}  {texts.get(code, '')}")
        if duplicates:
            print(f"  同名去重：{len(filtered)} -> {len(deduped)}，重复名 {duplicates}")
        print(f"  体积：全量 {full_bytes} 字节 / 落盘范围 {final_bytes} 字节")

    print("")
    print("=== 汇总 ===")
    for item in report:
        print(
            f"  {item['kind']:<10} 上游 {item['upstream_count']:>4} 条"
            f"  全量 {item['full_bytes']:>9} 字节"
            f"  落盘 {item['final_count']:>4} 条 / {item['final_bytes']:>9} 字节"
        )
    print(f"  合计：全量 {total_full_bytes} 字节 / 落盘范围 {total_final_bytes} 字节")

    if args.dry_run:
        if args.print_content:
            for item in report:
                print(f"--- {item['kind']} ---")
                print(item["final_text"])
        print("[dry-run] 未写任何文件")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for item in report:
        kind = item["kind"]
        target = out_dir / filenames[kind]
        target.write_text(item["final_text"], encoding="utf-8", newline="\r\n")
        print(f"  -> 写入 {target}（{item['final_count']} 条，{item['final_bytes']} 字节）")
    print("[OK] 三个数据文件已落地")
    return 0


if __name__ == "__main__":
    sys.exit(main())
