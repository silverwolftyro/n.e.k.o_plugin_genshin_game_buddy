"""L0 分块加载器（含跨块补拉）。

★ 为什么需要补拉
----------------
L0 块是 3072x1536 的规整网格，**不重叠**。如果按"玩家所在块"裁窗口，
而窗口中心离块边只有几十像素，窗口就会跨到邻块 ——
直接裁会拿到黑边，NCC 分数被拉低、定位偏移，而且**只在边缘帧发生**，
极难排查。

所以：**跨块时把邻块也拉进来拼接**。12x12 规整网格下，
邻块定位是纯算术（``r±1`` / ``c±1``），几行代码。

★ 底图外
--------
窗口部分落在底图之外时，用黑填充并返回 —— **这是合法边界，不是错误**
（地图边缘本来就可能是空的）。
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

__all__ = ["ChunkLoader", "PYRAMID_DIR"]

#: 金字塔根目录（可被环境变量覆盖）
PYRAMID_DIR = Path(
    __import__("os").environ.get(
        "NEKO_P0_PYRAMID",
        r"D:\NEKO\recognition\data\mihoyo_maps\P0_pyramid"))

_DEFAULT_CACHE = 8


class ChunkLoader:
    """按需加载 L0 分块，带 LRU 缓存和跨块补拉。

    :param pyramid_dir: ``P0_pyramid`` 目录
    :param cache_size: 同时驻留的块数（0 = 不缓存）
    """

    def __init__(self, pyramid_dir: Optional[str | Path] = None,
                 cache_size: int = _DEFAULT_CACHE):
        self.root = Path(pyramid_dir) if pyramid_dir else PYRAMID_DIR
        idx = self.root / "chunks_index.json"
        if not idx.is_file():
            raise FileNotFoundError(
                f"找不到分块索引: {idx}\n"
                f"（先跑 build/slice_p0_pyramid.py 切片）")
        meta = json.loads(idx.read_text(encoding="utf-8"))
        self.meta = meta
        self.width, self.height = meta["source_size"]
        self.cw, self.ch = meta["chunk_size"]
        self.gx, self.gy = meta["grid"]
        self._cache: "OrderedDict[tuple[int, int], object]" = OrderedDict()
        self._lock = threading.Lock()
        self._cache_size = int(cache_size)
        self.stats = {"load": 0, "hit": 0}

    # -------------------------------------------------- 块坐标
    def cell_of(self, px: float, py: float) -> tuple[int, int]:
        """像素坐标 -> (row, col)。越界返回 (-1,-1)。"""
        c = int(px // self.cw)
        r = int(py // self.ch)
        if 0 <= r < self.gy and 0 <= c < self.gx:
            return r, c
        return -1, -1

    def _read(self, r: int, c: int):
        key = (r, c)
        if self._cache_size <= 0:
            return self._read_uncached(r, c)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.stats["hit"] += 1
                return self._cache[key]
        img = self._read_uncached(r, c)
        with self._lock:
            self.stats["load"] += 1
            self._cache[key] = img
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return img

    def _read_uncached(self, r: int, c: int):
        import cv2

        from .image_io import imread_unicode
        p = self.root / "L0" / f"r{r:02d}_c{c:02d}.png"
        # ⚠️ 不用 cv2.imread：数据根路径含中文时会静默读不到
        im = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"读不到 L0 块: {p}")
        return im

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def warm(self, center_px, radius_cells: int = 1) -> int:
        """预拉中心附近的 (2r+1)² 个块。

        冷缓存下定位要拉 4 块 ≈ 90 ms，贴着 L1 的 100 ms 预算线。
        第一次定位后立刻预热周边，之后落点小幅移动就都是热缓存。

        :param radius_cells: 1 -> 3x3（9 块），2 -> 5x5（25 块）
        :returns: 实际预热的块数
        """
        r, c = self.cell_of(float(center_px[0]), float(center_px[1]))
        if r < 0 or c < 0:
            return 0
        n = 0
        for rr in range(max(0, r - radius_cells),
                        min(self.gy, r + radius_cells + 1)):
            for cc in range(max(0, c - radius_cells),
                            min(self.gx, c + radius_cells + 1)):
                self._read(rr, cc)
                n += 1
        return n

    def cell_bbox(self, r: int, c: int) -> tuple[int, int, int, int]:
        """(row, col) -> L0 bbox ``(x0, y0, x1, y1)``。"""
        if not (0 <= r < self.gy and 0 <= c < self.gx):
            raise ValueError(f"块 ({r},{c}) 越界：网格 {self.gy}x{self.gx}")
        return (c * self.cw, r * self.ch,
                min(self.width, (c + 1) * self.cw),
                min(self.height, (r + 1) * self.ch))

    # -------------------------------------------------- 主入口
    def load_l0_window(self, center_px: tuple[float, float],
                       window_size: int) -> "object":
        """以 ``center_px`` 为中心取 ``window_size`` 见方的 L0 灰度窗口。

        跨块自动补拉邻块拼接；落在底图外的部分用黑填充。

        :param center_px: 窗口中心在 **L0** 坐标系里的位置
        :param window_size: 边长（像素）
        :returns: ``window_size x window_size`` 的 uint8 灰度图
        """
        import numpy as np
        half = window_size // 2
        cx, cy = float(center_px[0]), float(center_px[1])
        wx0 = int(round(cx)) - half          # 窗口左上（可能为负）
        wy0 = int(round(cy)) - half
        out = np.zeros((window_size, window_size), dtype=np.uint8)

        # 窗口覆盖到的块范围
        r0 = int(np.floor(wy0 / self.ch))
        r1 = int(np.floor((wy0 + window_size - 1) / self.ch))
        c0 = int(np.floor(wx0 / self.cw))
        c1 = int(np.floor((wx0 + window_size - 1) / self.cw))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if not (0 <= r < self.gy and 0 <= c < self.gx):
                    continue                      # 底图外，留黑
                blk = self._read(r, c)
                # 该块在窗口坐标系里的位置
                dx = c * self.cw - wx0
                dy = r * self.ch - wy0
                sx0 = max(0, -dx)
                sy0 = max(0, -dy)
                ax0 = max(0, dx)
                ay0 = max(0, dy)
                w = min(blk.shape[1] - sx0, window_size - ax0)
                h = min(blk.shape[0] - sy0, window_size - ay0)
                if w > 0 and h > 0:
                    out[ay0:ay0 + h, ax0:ax0 + w] = blk[sy0:sy0 + h,
                                                         sx0:sx0 + w]
        return out

    # -------------------------------------------------- 诊断
    def describe(self, center_px, window_size: int) -> dict:
        """这次窗口跨了几个块 / 有多少落在底图外。**给测试和排查用。**"""
        import numpy as np
        half = window_size // 2
        wx0 = int(round(float(center_px[0]))) - half
        wy0 = int(round(float(center_px[1]))) - half
        r0, r1 = int(np.floor(wy0 / self.ch)), \
            int(np.floor((wy0 + window_size - 1) / self.ch))
        c0, c1 = int(np.floor(wx0 / self.cw)), \
            int(np.floor((wx0 + window_size - 1) / self.cw))
        inside = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if 0 <= r < self.gy and 0 <= c < self.gx]
        total = (r1 - r0 + 1) * (c1 - c0 + 1)
        # 底图内的合法像素数
        gx0, gy0 = max(0, wx0), max(0, wy0)
        gx1 = min(self.width, wx0 + window_size)
        gy1 = min(self.height, wy0 + window_size)
        legal = max(0, gx1 - gx0) * max(0, gy1 - gy0)
        return {"touched_cells": total, "inside_cells": len(inside),
                "cross_chunk": total > 1, "out_of_map_px":
                    window_size * window_size - legal}
