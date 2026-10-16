"""传送点数据的加载入口。

★ 本模块**不含任何坐标数据**
--------------------------
只做三件事：找文件、读 JSON、交给 :mod:`_tp_converter` 转换。
具体的点在哪里、叫什么、有多少个，全在数据文件里。

★ 数据来源与合规
----------------
本项目**不自带**传送点数据（BetterGI 是 GPL-3.0，见
``recognition/notes/LICENSING.md``）。加载器按以下优先级找：

    1. 显式传入的 ``user_path``
    2. ``$NEKO_TELEPORT_POINTS`` 环境变量（可指文件，也可指目录）
    3. 自有数据 ``$RECOGNITION_ROOT/data/teleport_points.json``
    4. **本机 BetterGI 安装** —— 按 ``BETTERGI_RELPATH`` 精确拼到
       ``GameTask/AutoTrackPath/Assets/tp.json``

第 4 条用的是 :func:`detect_bettergi`，不是猜文件名。
更宽的安装位置靠 ``NEKO_RECOGNITION_ROOT`` 之外的环境变量不适用，
所以同时提供 :func:`add_bettergi_candidate` 供上层注入。

四条都找不到就抛 —— 不静默返回空表。空表会让下游"没有传送点可用"和
"读不到文件"表现成同一个症状，排查时无从下手。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..paths import RECOGNITION_ROOT
from ._tp_converter import convert_bettergi_format

logger = logging.getLogger(__name__)

#: 自有数据相对 ``RECOGNITION_ROOT`` 的路径。
#: ★ 不再用 ``_PLUGIN_ROOT.parent / ...`` 上溯 —— 插件装在哪都不影响，
#: 认准感知层的资产根常量即可（支持 ``NEKO_RECOGNITION_ROOT`` 覆盖）。
OWN_DATA_RELPATH = Path("data") / "teleport_points.json"

#: 环境变量名
ENV_VAR = "NEKO_TELEPORT_POINTS"

#: BetterGI 内部 tp.json 相对**安装根**的路径。
#: ★ 关键：BetterGI 的文件名叫 ``tp.json``，不是 ``teleport_points.json``，
#:   而且埋在三层子目录下。曾经这里被当成"根目录 + 统一文件名"去拼，
#:   结果永远命中不了 —— 这就是 resolve_path 找不到 BetterGI 的原因。
BETTERGI_RELPATH = Path("GameTask") / "AutoTrackPath" / "Assets" / "tp.json"

#: 本机 BetterGI 可能的安装位置。**只用来"找到没有"**，不预置内容。
#: 是列表而不是常量：换机器 / 多开时用 :func:`add_bettergi_candidate` 追加。
BETTERGI_CANDIDATES: list[Path] = [
    Path(r"D:\NEKO\现成的自动化工具参考\参考：better GI\BetterGI"),
    Path(r"C:\Program Files\BetterGI"),
]


class TeleportPointSource(str):
    """数据来源标记。"""

    EXPLICIT = "explicit"
    ENV = "env"
    OWN = "own"
    BETTERGI = "bettergi"


def add_bettergi_candidate(root: str | Path) -> None:
    """追加一个 BetterGI 安装根候选。**只记位置，不读内容。**"""
    p = Path(root)
    if p not in BETTERGI_CANDIDATES:
        BETTERGI_CANDIDATES.append(p)


def own_data_path() -> Path:
    """自有数据的绝对路径。"""
    return RECOGNITION_ROOT / OWN_DATA_RELPATH


def _env_candidates() -> list[Path]:
    """环境变量指向的候选文件。

    变量可以指**文件**（直接用），也可以指**目录**（在里面按
    自有文件名、BetterGI 相对路径各试一次）。
    """
    env = os.environ.get(ENV_VAR, "").strip()
    if not env:
        return []
    p = Path(env)
    if p.is_dir():
        return [p / OWN_DATA_RELPATH.name, p / BETTERGI_RELPATH.name]
    return [p]


def _candidate_files() -> list[Path]:
    """按优先级列出候选**文件**（不是目录）。顺序即优先级。"""
    return _env_candidates() + [own_data_path()]


def _classify(p: Path, user_path: str | Path | None = None) -> TeleportPointSource:
    """判断一个已命中的文件来自哪一级。"""
    if user_path and p == Path(user_path):
        return TeleportPointSource.EXPLICIT
    if p == own_data_path():
        return TeleportPointSource.OWN
    if p in _env_candidates():
        return TeleportPointSource.ENV
    return TeleportPointSource.BETTERGI


def detect_bettergi() -> Path | None:
    """探测本机 BetterGI 安装。**找不到返回 None**，不抛。

    ★ 只做存在性判断
      不读内容、不解析、不缓存 —— 是不是 BetterGI 的格式，
      交给加载后转换时的异常去说。
    """
    for root in BETTERGI_CANDIDATES:
        cand = root / BETTERGI_RELPATH
        try:
            if cand.is_file():
                logger.info("探测到 BetterGI 传送点数据：%s", cand)
                return cand
        except OSError as exc:
            logger.debug("探测 %s 失败（忽略）：%s", cand, exc)
    logger.info("未探测到 BetterGI 安装（已查 %d 个候选位置）",
                len(BETTERGI_CANDIDATES))
    return None


def resolve_path(user_path: str | Path | None = None) -> Path | None:
    """决定用哪个数据文件。找不到返回 None。

    优先级：显式路径 > 环境变量 > 自有数据 > 本机 BetterGI。
    ★ 最后一级必须走 :func:`detect_bettergi`；统一按"根目录 +
      固定文件名"去拼是命中不了的（BetterGI 的文件名不是那个）。
    """
    if user_path:
        p = Path(user_path)
        return p if p.is_file() else None
    for cand in _candidate_files():
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return detect_bettergi()


def _not_found_detail(user_path: str | Path | None) -> str:
    """找不到时，把**每一级具体查了什么**列出来。

    只说"没找到"没用 —— 排查时要能一眼看出是没配环境变量、
    还是自有数据没生成、还是 BetterGI 装在别处。
    """
    lines: list[str] = []
    if user_path:
        lines.append(f"  [1] 显式指定: {user_path}   "
                     f"<- 不存在，或不是文件")
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        lines.append(f"  [2] 环境变量 {ENV_VAR} = {env}")
        for c in _env_candidates():
            lines.append(f"        - {c}")
    else:
        lines.append(f"  [2] 环境变量 {ENV_VAR}: 未设置")
    lines.append(f"  [3] 自有数据: {own_data_path()}")
    lines.append(f"  [4] BetterGI 探测（{len(BETTERGI_CANDIDATES)} 个候选根）:")
    for root in BETTERGI_CANDIDATES:
        lines.append(f"        - {root / BETTERGI_RELPATH}")
    lines.append(f"  资产根 RECOGNITION_ROOT = {RECOGNITION_ROOT}  "
                 f"（可用 NEKO_RECOGNITION_ROOT 覆盖）")
    return "\n".join(lines)


def load_teleport_points(user_path: str | Path | None = None) -> dict[str, Any]:
    """加载并转换成自有 schema。

    :param user_path: 显式指定数据文件；``None`` 时按优先级自动找
    :returns: ``{"version", "source", "regions": [...]}``
    :raises FileNotFoundError: 任何来源都没找到数据
    :raises ValueError: 文件不是合法 JSON / 顶层不是对象

    ★ 为什么"找不到"要抛而不是返回空表
      空表会让下游把"没数据"和"文件读不到"当成同一件事，
      而这两种情况的处置方式完全不同。
    """
    p = resolve_path(user_path)
    if p is None:
        raise FileNotFoundError(
            "找不到传送点数据。已按优先级逐级查过：\n"
            f"{_not_found_detail(user_path)}\n"
            "（本项目不内置该数据 —— BetterGI 是 GPL-3.0，见 "
            "recognition/notes/LICENSING.md）")
    try:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{p} 不是合法 JSON：{exc}") from exc

    # 自有格式的文件顶层已经是 {version, source, regions}，不必再转。
    # 判据：有自己的 regions，且**没有**源格式的 data。
    if isinstance(raw, dict) and "regions" in raw and "data" not in raw:
        logger.info("读到自有格式的传送点数据：%s", p)
        return raw

    out = convert_bettergi_format(raw)
    n = sum(len(r.get("points", [])) for r in out["regions"])
    logger.info("已从 %s 转换 %s：%d 个区域 / %d 个传送点",
                _classify(p, user_path), p, len(out["regions"]), n)
    return out


__all__ = [
    "BETTERGI_CANDIDATES",
    "BETTERGI_RELPATH",
    "ENV_VAR",
    "OWN_DATA_RELPATH",
    "TeleportPointSource",
    "add_bettergi_candidate",
    "detect_bettergi",
    "load_teleport_points",
    "own_data_path",
    "resolve_path",
]
