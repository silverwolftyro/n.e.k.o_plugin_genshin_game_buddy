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

    python tools/fetch_genshin_db.py                     # 落到 terms/genshin/
    python tools/fetch_genshin_db.py --dry-run           # 只打印，不落盘
    python tools/fetch_genshin_db.py --out-dir /tmp/xx   # 落到别处（便于对比）

拉取通道（按顺序回退，都是只读 GET）：

1. ``fastly.jsdelivr.net/gh/<repo>@<commit>/...`` —— CDN 镜像，无速率限制；
2. ``api.github.com/repos/<repo>/contents/...`` —— 官方 API，有速率限制。

两者返回的是**同一份** JSON（已逐字段比对过），只是渠道不同。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# 上游坐标（固定 commit = 可复现）
# ---------------------------------------------------------------------------

UPSTREAM_REPO = "theBowja/genshin-db"
PINNED_COMMIT = "8e5badd60b5d894284ccc927a56c9683529d2592"
UPSTREAM_LICENSE = "MIT（上游仓库 LICENSE；本插件只取名称/数值/效果描述等事实性数据）"

JSDELIVR_BASE = f"https://fastly.jsdelivr.net/gh/{UPSTREAM_REPO}@{PINNED_COMMIT}/src/data"
GITHUB_BASE = f"https://api.github.com/repos/{UPSTREAM_REPO}/contents/src/data"

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
# 主流程
# ---------------------------------------------------------------------------


def collect(kind: str, slugs: tuple[str, ...], generated_at: str) -> tuple[dict, list[dict]]:
    """拉一组条目并清洗，返回 (header, rows)。"""
    cleaners = {
        "weapons": clean_weapon,
        "artifacts": clean_artifact,
        "materials": clean_material,
    }
    clean = cleaners[kind]
    rows: list[dict] = []
    for slug in slugs:
        raw = fetch_item(kind, slug)
        row = clean(raw, slug)
        if not row["name"]:
            raise RuntimeError(f"上游条目没有名称：{kind}/{slug}")
        rows.append(row)
        print(f"  [{kind}] {row['name']}  <- {slug}")
    return build_header(kind, generated_at), rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 genshin-db 并落地本地 TOML（仅开发期使用）")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="输出目录，默认 terms/genshin",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要写出的内容，不落盘",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S+08:00", time.localtime())

    plan = (
        ("weapons", WEAPON_SLUGS, "weapons.toml"),
        ("artifacts", ARTIFACT_SLUGS, "artifacts.toml"),
        ("materials", MATERIAL_SLUGS, "materials.toml"),
    )

    for kind, slugs, filename in plan:
        print(f"=== 拉取 {kind}（{len(slugs)} 条）===")
        header, rows = collect(kind, slugs, generated_at)
        text = render_toml(header=header, table=kind, rows=rows)
        if args.dry_run:
            print(text)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / filename
        target.write_text(text, encoding="utf-8", newline="\r\n")
        print(f"  -> 写入 {target}（{len(rows)} 条，{len(text.encode('utf-8'))} 字节）")

    if args.dry_run:
        print("[dry-run] 未写任何文件")
    else:
        print("[OK] 三个数据文件已落地")
    return 0


if __name__ == "__main__":
    sys.exit(main())
