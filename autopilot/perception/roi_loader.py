"""读 ``rois.toml``，把归一化坐标变成可用的矩形。

★ 为什么要归一化
----------------
ROI 写的是 0~1 的相对坐标，这样同一份配置在 1280×720 和 2560×1600 上都成立。
分辨率一变，只要乘上新的宽高就行，不用维护两套配置。

★ 命名扁平化
------------
``rois.toml`` 里有两层结构::

    [rois]
    minimap = [0.02, 0.01, 0.14, 0.19]

    [rois.party]
    "1" = [...]

本模块把它们拍平成 ``{"minimap": Rect(...), "party_1": Rect(...)}``。
调用方只关心"叫什么"，不关心它在 TOML 里挂在哪一层 —— 否则每加一层
嵌套，所有调用点都要跟着改。
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import ROIS_TOML

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rect:
    """一个 ROI。四个边都是 **0~1 归一化**的浮点数。"""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """归一化 -> 像素 ``(x1, y1, x2, y2)``。

        ★ 为什么要夹边界
          numpy 的负数下标不是"报错"，而是**从末尾往前数** ——
          ``img[-50:100]`` 悄悄变成空数组，或者更糟，切出一段完全无关的图。
          配置里一个手滑的 ``-0.01`` 就能触发，而且**不报错**。
          所以这里显式夹到 [0, size]，宁可少几个像素也不要拿到错的区域。
        """
        x1 = _clamp(self.left, 0.0, 1.0, width)
        y1 = _clamp(self.top, 0.0, 1.0, height)
        x2 = _clamp(self.right, 0.0, 1.0, width)
        y2 = _clamp(self.bottom, 0.0, 1.0, height)
        return (x1, y1, max(x1, x2), max(y1, y2))

    def crop(self, frame: Any) -> Any:
        """从图像（numpy 数组）里裁出这个区域。

        :param frame: ``numpy.ndarray``，形状 ``(H, W)`` 或 ``(H, W, 3)``。
        :returns: 裁剪后的**视图**（不复制内存）。尺寸不对时返回原图。
        """
        h, w = _frame_shape(frame)
        if h is None or w is None:
            logger.warning("crop 收到非图像对象，原样返回")
            return frame
        x1, y1, x2, y2 = self.to_pixels(w, h)
        return frame[y1:y2, x1:x2]


def _clamp(value: float, lo: float, hi: float, size: int) -> int:
    """0~1 的相对值 -> 0~size 的整数，夹在 [0, size] 内。"""
    px = int(round(value * size))
    return max(0, min(size, px))


def _frame_shape(frame: Any) -> tuple[int | None, int | None]:
    shape = getattr(frame, "shape", None)
    if not isinstance(shape, tuple) or len(shape) < 2:
        return None, None
    return int(shape[0]), int(shape[1])


def _as_rect(value: Any) -> Rect | None:
    """把 TOML 里的 ``[l, t, r, b]`` 转成 :class:`Rect`；不合法返回 None。

    刻意**不抛异常**：一个写错格式的 ROI 不该让整张配置全废 ——
    那会让用户改了无关的一行，整个感知层直接起不来。跳过 + 记日志。
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    # 右/下边界必须**严格大于**左/上，否则宽高为 0 甚至负数
    if not (0.0 <= left < right <= 1.0):
        return None
    if not (0.0 <= top < bottom <= 1.0):
        return None
    return Rect(left, top, right, bottom)


def _parse_table(table: dict[str, Any]) -> dict[str, Rect]:
    """拍平一层嵌套。

    ``{"minimap": [...], "party": {"1": [...]}}`` ->
    ``{"minimap": Rect, "party_1": Rect}``
    """
    out: dict[str, Rect] = {}
    for key, value in table.items():
        if isinstance(value, dict):
            for sub, coords in value.items():
                rect = _as_rect(coords)
                if rect is None:
                    logger.warning("ROI %s_%s 格式不合法，已跳过", key, sub)
                    continue
                out[f"{key}_{sub}"] = rect
        else:
            rect = _as_rect(value)
            if rect is None:
                logger.warning("ROI %s 格式不合法，已跳过", key)
                continue
            out[key] = rect
    return out


@lru_cache(maxsize=1)
def _load_toml(path: str) -> dict[str, Any]:
    """读并缓存整个 TOML。``path`` 传 str 是为了让 lru_cache 可哈希。"""
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"找不到 ROI 配置：{p}\n"
            f"它应当在 {ROIS_TOML}。"
            f"确认 recognition 目录已建好、配置已迁移。") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"ROI 配置 TOML 语法错误：{p} —— {exc}") from exc


@lru_cache(maxsize=1)
def load_rois(path: str | None = None) -> dict[str, Rect]:
    """读全部 ROI。**结果缓存**（进程内只解析一次）。

    :param path: 指定配置路径；默认用 :data:`paths.ROIS_TOML`。
    :returns: ``{"minimap": Rect, "party_1": Rect, ...}``
    :raises FileNotFoundError: 配置不存在（**响亮失败** —— 静默返回空 dict
        会让上层一路带着"没有任何 ROI"跑下去，症状是匹配全部落空，
        排查起来比直接报错难十倍）。
    """
    target = str(Path(path) if path else ROIS_TOML)
    data = _load_toml(target)
    rois = data.get("rois")
    if not isinstance(rois, dict):
        raise ValueError(f"{target} 里没有 [rois] 段")
    result = _parse_table(rois)
    logger.info("已加载 %d 个 ROI（来自 %s）", len(result), target)
    return result


def reload_rois(path: str | None = None) -> dict[str, Rect]:
    """强制清缓存重读。**调完 rois.toml 用这个**，否则改了不生效。"""
    _load_toml.cache_clear()
    load_rois.cache_clear()
    logger.info("ROI 缓存已清空，重新加载")
    return load_rois(path)


@lru_cache(maxsize=1)
def load_meta(path: str | None = None) -> dict[str, Any]:
    """读 ``[meta]`` 段（基准分辨率、参考帧）。"""
    target = str(Path(path) if path else ROIS_TOML)
    data = _load_toml(target)
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return {}
    return meta


def load_thresholds(path: str | None = None) -> dict[str, float]:
    """读 ``[thresholds]`` 段。"""
    target = str(Path(path) if path else ROIS_TOML)
    data = _load_toml(target)
    raw = data.get("thresholds")
    if not isinstance(raw, dict):
        return {}
    return {k: float(v) for k, v in raw.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def load_templates_dir(path: str | None = None) -> str:
    """``[templates].dir`` 里声明的目录**原文**返回。

    ⚠ 那一段现在是**过期路径**（还指向迁移前的 ``_screenshots/templates``）。
      本函数只如实返回，不替用户改 —— 路径该指向哪，是人的决定。
      要用模板请走 :data:`paths.TEMPLATES_UI`。
    """
    target = str(Path(path) if path else ROIS_TOML)
    data = _load_toml(target)
    tmpl = data.get("templates")
    if isinstance(tmpl, dict) and isinstance(tmpl.get("dir"), str):
        return tmpl["dir"]
    return ""


__all__ = [
    "Rect",
    "load_meta",
    "load_rois",
    "load_templates_dir",
    "load_thresholds",
    "reload_rois",
]
