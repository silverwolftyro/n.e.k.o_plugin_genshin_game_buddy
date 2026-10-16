"""金字塔 L1~L4 整图加载器（常驻缓存）。

★ 与 L0 分块的区别
------------------
L0 太大（648 MiB）不能常驻，切成 144 块按需加载；
L1~L4 最大只有 162 MiB（L1），**全部常驻内存一次读盘**。

* L1  18432x9216   162 MiB
* L2  9216x4608     40.5 MiB
* L3  4608x2304     10.1 MiB
* L4  2304x1152      2.5 MiB

四层合计 215 MiB，是搜索时真正会被反复扫的部分。

★ 缩放比
--------
``level`` 是**降采样倍数**（1=L0 全分辨率），``get_level_scale(1) = 0.5``。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

__all__ = ["PyramidLoader", "LEVELS", "MIN_LEVEL", "MAX_LEVEL"]

#: 可加载的层（不含 L0，L0 走 chunk_loader）
LEVELS = (1, 2, 3, 4)
MIN_LEVEL, MAX_LEVEL = 1, 4


def _sibling(name: str):
    """加载同目录模块。**兼容"作为包导入"和"独立按文件加载"两种情况。**

    ``spec_from_file_location`` 直接加载本文件时没有包上下文，
    ``from .chunk_loader import ...`` 会报 ImportError。
    """
    try:
        import importlib
        return importlib.import_module(f".{name}", __package__)
    except (ImportError, ValueError, TypeError):
        import importlib.util
        import sys
        p = Path(__file__).with_name(f"{name}.py")
        s = importlib.util.spec_from_file_location(f"_nk_{name}", p)
        m = importlib.util.module_from_spec(s)
        sys.modules[f"_nk_{name}"] = m
        s.loader.exec_module(m)
        return m


def _pyramid_dir():
    return _sibling("chunk_loader").PYRAMID_DIR


class PyramidLoader:
    """按层加载 L1~L4 整图，**每层只读一次盘**。

    :param pyramid_dir: ``P0_pyramid`` 目录
    :param preload: 构造时是否立刻把四层全读进内存（``True`` /
        ``False`` / 只读某几层的元组）
    """

    def __init__(self, pyramid_dir: Optional[str | Path] = None,
                 preload: bool | tuple = True):
        self.root = Path(pyramid_dir) if pyramid_dir else _pyramid_dir()
        self._imgs: dict[int, object] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.loads = 0
        self.base_size: Optional[tuple[int, int]] = None
        if preload:
            for lv in (preload if isinstance(preload, tuple) else LEVELS):
                self.get_level(lv)

    # -------------------------------------------------- 基础
    def get_level_scale(self, level: int) -> float:
        """该层相对 L0 的缩放比。``level=1`` -> 0.5。"""
        lv = int(level)
        if lv < 0 or lv > MAX_LEVEL:
            raise ValueError(
                f"level 必须在 0~{MAX_LEVEL}，收到 {level}"
                f"（0 = L0 全分辨率，4 = 1/16）")
        return 1.0 / (2 ** lv)

    def get_level(self, level: int):
        """返回第 ``level`` 层的灰度整图（ndarray）。**首次调用读盘，之后常驻。**"""
        lv = int(level)
        if lv == 0:
            raise ValueError("level=0 是 L0 全分辨率，太大不能常驻；"
                             "请用 chunk_loader.ChunkLoader.load_l0_window()")
        if lv < 0 or lv > MAX_LEVEL:
            raise ValueError(f"level 必须在 1~{MAX_LEVEL}，收到 {level}")
        with self._lock:
            if lv in self._imgs:
                self._imgs[lv]        # 触碰即算命中
                self.hits += 1
                return self._imgs[lv]
        import cv2

        from .image_io import imread_unicode
        p = self.root / f"L{lv}_full.png"
        # ⚠️ 不用 cv2.imread：数据根路径含中文时会静默读不到
        im = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"读不到金字塔层: {p}")
        with self._lock:
            self._imgs[lv] = im
            self.loads += 1
            if self.base_size is None:
                self.base_size = (im.shape[1] * (2 ** lv),
                                  im.shape[0] * (2 ** lv))
        return im

    def level_size(self, level: int) -> tuple[int, int]:
        im = self.get_level(level)
        return im.shape[1], im.shape[0]

    def clear(self) -> None:
        with self._lock:
            self._imgs.clear()
