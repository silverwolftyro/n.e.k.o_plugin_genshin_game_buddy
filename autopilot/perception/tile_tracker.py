"""任务驱动定位 · 瓦片追踪器。

★ 它做什么
----------
把"相位相关测出的帧间位移"累积成世界坐标，并回答两个问题：

    1. 我现在在哪块瓦片？
    2. 我在这块瓦片里的哪个像素？

输入是**小地图像素位移**（``cv2.phaseCorrelate`` 的输出，见
``scripts/probe/exp_phase_corr.py``），不是图像。
所以本模块**不碰任何图像**，纯几何。

★ 三条必须守住的不变量
----------------------
1. **跳变不静默累加。** session_a 实测帧间位移 p95 = 2.01 px，
   而传送/场景切换会产生 40~125 px 的跳变。把这种跳变累加进去，
   轨迹会瞬间飞到地图另一头，而且**之后再也纠不回来**。
   默认直接抛 :class:`TeleportJump` —— 宁可炸，不可静默走错。
2. **起点是外部给定的，本模块不猜。** 起点来自传送 OCR + tp.json
   （见 ``notes/teleport_contract.md``）。没有起点就没有本模块。
3. **漂移上界随时间开根号增长。** 逐帧噪声 sigma 独立同分布，
   累加 n 帧的随机游走误差是 ``sigma * sqrt(n)``。
   把这个上界**和位置一起返回**，调用方才知道这个位置有多可信。

★ 为什么不直接存"当前块"
------------------------
存全局像素坐标、每次 :meth:`TileTracker.tile_of` 现算。
好处是：块边界判定只有一处实现，跨块逻辑不会和坐标状态不一致；
将来换瓦片尺寸（金字塔层不同）只改索引，不改追踪逻辑。

★ 坐标约定（已用雅可比实测确认）
---------------------------------
1. 底图像素 -> 世界坐标：直接用 ``coord_transform``，
   它已经把"底图相对世界坐标是**转置**的"（px~world_z, py~world_x）
   吸收进仿射矩阵。**不要试图"纠正"轴向**，会得到 4762 像素残差。
2. **轴向是转置的**（两点差分实测雅可比）：
   ``d(px)/d(world_z) = -0.9995``、``d(py)/d(world_x) = -0.9983``。
   小地图正北朝上（用户确认"固定地图方向"）=> +x = 底图 +x = -world_z，
   +y = 底图 +y = -world_x。所以**横移小地图改的是 world z，纵移改的是
   world x** —— 与直觉相反。:meth:`TileTracker.update` 内部直接按
   ``dwx = -dy*scale, dwz = -dx*scale`` 换算。
   （曾因忽略转置写成 ``dwx=dx, dwz=dy``，导致走 +y 60 步仍在同一瓦片内。）
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

__all__ = [
    "DEFAULT_SIGMA_PX",
    "MAX_STEP_PX",
    "OVER_STREAK_TO_REJECT",
    "TeleportJump",
    "TileGrid",
    "TileInfo",
    "TileTracker",
    "TrackerState",
]

#: 帧间位移上限（小地图像素）。
#:
#: ★ 为什么从 3.0 提到 50.0（2026-10-04）
#: -----------------------------------------
#: 旧值 3.0 的依据是「session_a 1511 对全量实测 p95 = 2.01 px」。
#: **p95 这个统计量本身就是错的**：跑步帧在数据集里占比不到 5%，
#: 全被 p95 截掉了 —— 而 ``configs/minimap_scale.toml`` 自己就写着
#: 「段3 5.69 HIGH(含冲刺)」，说明**跑步帧当时就在数据里**，只是被
#: 分位数掩盖了。后果是 Shift+W 一上线，跑步位移（5~6 px/帧）被
#: **逐帧**判成传送而拒收，寻路全程 ``|win| = 0``、``θ_m`` 永不出现。
#:
#: 现在按真实运动学分档取阈值（实测）：
#:   静止噪声   0.109 px
#:   走路      0.3~0.7 px/帧
#:   **跑步    5~6 px/帧**      <- 旧阈值在这里全被拒
#:   冲刺       8~10 px/帧
#:   真传送    100~5000 px
#: 50 px 落在冲刺(10) 与传送(100) 之间，两侧都有 5~10 倍余量。
#:
#: ⚠️ 单帧阈值只是第一道防线，**真跳变靠连续计数**（见
#:    :data:`OVER_STREAK_TO_REJECT`）—— 50px 单帧也可能是相关毛刺。
MAX_STEP_PX: float = 50.0

#: 连续多少帧都超 :data:`MAX_STEP_PX` 才判定为真跳变（抛 TeleportJump）。
#:
#: ★ 第 1 帧超阈值时**丢弃该帧**（位置不动、不累加），不抛异常。
#:   这样单帧相关毛刺既不会污染位置，也不会误报传送。
#:   连续第 2 帧仍超 -> 认定真跳变，抛 :class:`TeleportJump`。
#:
#: 为什么不「单帧超就抛」：真传送往往**一帧内**跳 100~5000 px。
#: 若第 1 帧就累加进去，位置瞬间错 50~99 px（× city_px_to_world 0.389
#: ≈ 20~38 m）且**事后无法纠正** —— 这正是本模块不变量 1 要防的事。
OVER_STREAK_TO_REJECT: int = 2

#: 逐帧测量噪声（小地图像素），来自 session_a 静止基线 579 帧 p99。
#: 累加 n 帧的随机游走误差 = 本值 * sqrt(n)，换算成世界单位再乘比例。
DEFAULT_SIGMA_PX: float = 0.109


class TeleportJump(RuntimeError):
    """连续多帧帧间位移超过 :data:`MAX_STEP_PX` —— 判定为传送/场景切换。

    :param step_px: 观测到的帧间位移模长
    :param limit_px: 判定阈值
    :param dx: 小地图 +x 位移（屏幕向右为正）。**可选**，给了才进报错文案。
    :param dy: 小地图 +y 位移（屏幕向下为正）。**可选**。
    """

    def __init__(self, step_px: float, limit_px: float = MAX_STEP_PX,
                 dx: Optional[float] = None, dy: Optional[float] = None):
        self.step_px = float(step_px)
        self.limit_px = float(limit_px)
        self.dx = None if dx is None else float(dx)
        self.dy = None if dy is None else float(dy)
        # ★ 把**方向**带进异常：光有模长分不清「向前猛冲」和「场景瞬切」，
        #   有了 (dx,dy) 就能一眼看出位移是否同向、量级是否合理。
        where = ("" if self.dx is None or self.dy is None else
                 f" 方向(dx,dy)=({self.dx:+.2f},{self.dy:+.2f})")
        super().__init__(
            f"连续 {OVER_STREAK_TO_REJECT} 帧帧间位移超阈值 —— 本次 "
            f"{self.step_px:.2f} px > {self.limit_px:.2f} px{where}。"
            f"判定为传送/场景切换。**不要累加**，应调用 "
            f"TileTracker.teleport_to() 用新起点重置。")


def _sibling(name: str):
    """加载同目录模块。**兼容"作为包导入"和"独立按文件加载"两种情况。**

    ``spec_from_file_location`` 直接加载本文件时没有包上下文，
    ``from .chunk_loader import ...`` 会报 ImportError。
    """
    try:
        return importlib.import_module(f".{name}", __package__)
    except (ImportError, ValueError, TypeError):
        p = Path(__file__).with_name(f"{name}.py")
        s = importlib.util.spec_from_file_location(f"_nk_{name}", p)
        m = importlib.util.module_from_spec(s)
        sys.modules[f"_nk_{name}"] = m
        s.loader.exec_module(m)
        return m


def _paths():
    """取 :mod:`paths`（全仓唯一路径出处），加载不到就用环境变量兜底。"""
    try:
        return _sibling("paths")
    except Exception:                                    # noqa: BLE001
        import os
        root = Path(os.environ.get("NEKO_RECOGNITION_ROOT", "").strip()
                    or r"D:\NEKO\recognition")
        return type("_P", (), {
            "RECOGNITION_ROOT": root,
            "MINIMAP_SCALE_TOML": root / "configs" / "minimap_scale.toml",
            "COORD_TRANSFORM_PY": (root / "exports" / "benchmark" / "scripts"
                                   / "misc" / "coord_transform.py"),
        })()


def _scale_toml_path() -> Path:
    """小地图比例配置路径。"""
    return Path(_paths().MINIMAP_SCALE_TOML)


# ── 瓦片网格 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TileInfo:
    """一块瓦片的静态信息。"""

    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def contains(self, px: float, py: float) -> bool:
        """像素是否落在本块内（**右/下边界开区间**，与 ``bbox`` 语义一致）。

        边界像素归属哪一块必须唯一，否则同一像素会算出两个 tile_id。
        这里统一取"左上闭、右下开"。
        """
        return self.x0 <= px < self.x1 and self.y0 <= py < self.y1


class TileGrid:
    """底图瓦片网格。

    :param index_path: ``chunks_index.json`` 路径。``None`` 时走
        :class:`~autopilot.perception.chunk_loader.ChunkLoader` 的默认目录。

    :raises FileNotFoundError: 索引文件不存在
    :raises ValueError: 索引格式不对（缺 chunks / bbox 长度不为 4）
    """

    def __init__(self, index_path: Optional[Path] = None):
        if index_path is None:
            index_path = Path(_sibling("chunk_loader").PYRAMID_DIR) \
                / "chunks_index.json"
        import json
        p = Path(index_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"找不到瓦片索引: {p}\n"
                f"（先跑 build/slice_p0_pyramid.py 切片）")
        meta = json.loads(p.read_text(encoding="utf-8"))
        if "chunks" not in meta or not meta["chunks"]:
            raise ValueError(f"{p} 缺 [chunks] 段或为空")
        self.source_size = tuple(meta["source_size"])
        self.chunk_size = tuple(meta["chunk_size"])
        self.grid = tuple(meta["grid"])
        self._tiles: dict[str, TileInfo] = {}
        for c in meta["chunks"]:
            bb = c["bbox"]
            if len(bb) != 4:
                raise ValueError(
                    f"{p}: {c.get('id')} 的 bbox 长度 {len(bb)} != 4")
            self._tiles[str(c["id"])] = TileInfo(
                str(c["id"]), int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        if not self._tiles:
            raise ValueError(f"{p} 里没有任何合法瓦片")

    # -------------------------------------------------- 查询
    def __len__(self) -> int:
        return len(self._tiles)

    def __contains__(self, tile_id: str) -> bool:
        return tile_id in self._tiles

    def get(self, tile_id: str) -> TileInfo:
        """按 id 取瓦片。**找不到就抛**，不返回 None。"""
        try:
            return self._tiles[tile_id]
        except KeyError:
            raise KeyError(
                f"没有这块瓦片: {tile_id!r}（现有 {len(self._tiles)} 块，"
                f"grid={self.grid}）") from None

    def tile_of(self, px: float, py: float) -> Optional[TileInfo]:
        """像素坐标 -> 所在瓦片。**出图返回 None**（不抛）。

        :param px: 底图 L0 像素 x
        :param py: 底图 L0 像素 y
        """
        w, h = self.source_size
        if not (0 <= px < w and 0 <= py < h):
            return None
        # 直接定位到网格行列，比线性扫 144 块快且不会漏
        cw, ch = self.chunk_size
        ncols, nrows = self.grid
        col = min(int(px // cw), ncols - 1)
        row = min(int(py // ch), nrows - 1)
        t = self._tiles.get(f"r{row:02d}_c{col:02d}")
        if t is None:
            return None
        return t if t.contains(px, py) else None

    def neighbors(self, tile_id: str) -> list[str]:
        """中心块 + 8 邻居，**按行列顺序**（左上 -> 右下），不含自身重复。

        网格边缘的块邻居自然更少 —— 不做环绕，底图边界外没有瓦片。

        :raises KeyError: ``tile_id`` 不存在
        """
        t = self.get(tile_id)
        cw, ch = self.chunk_size
        ncols, nrows = self.grid
        row = t.y0 // ch
        col = t.x0 // cw
        out: list[str] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if 0 <= r < nrows and 0 <= c < ncols:
                    tid = f"r{r:02d}_c{c:02d}"
                    if tid in self._tiles and tid not in out:
                        out.append(tid)
        return out


# ── 追踪状态 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrackerState:
    """一次 :meth:`TileTracker.update` 的输出。

    :ivar world_pos: 世界坐标 ``(world_x, world_z)``
    :ivar pixel: 底图 L0 像素 ``(px, py)``
    :ivar tile_id: 当前瓦片 id，如 ``"r04_c06"``；出图为 ``None``
    :ivar in_tile_px: 块内像素 ``(x, y)``，原点为块左上角；出图为 ``None``
    :ivar tile_changed: 本次 update 是否跨了块
    :ivar step_index: 起点以来的 update 次数
    :ivar path_len: 累计路径长度（**世界单位**，非净位移）
    :ivar drift_bound: 累计漂移**上界**（世界单位），= ``sigma*sqrt(n)*scale``
    """

    world_pos: tuple[float, float]
    pixel: tuple[float, float]
    tile_id: Optional[str]
    in_tile_px: Optional[tuple[float, float]]
    tile_changed: bool
    step_index: int
    path_len: float
    drift_bound: float


# ── 追踪器 ────────────────────────────────────────────────────────────────

class TileTracker:
    """从已知起点累积相位相关位移，实时回答"在哪块瓦片 + 块内坐标"。

    :param start_world_pos: 起点世界坐标 ``(world_x, world_z)``。
        来源必须是传送 OCR + tp.json（见 ``notes/teleport_contract.md``），
        **本模块不猜起点**。
    :param grid: 瓦片网格。``None`` 时按默认路径构造。
    :param px_to_world: 小地图像素 -> 世界单位。``None`` 时读
        ``configs/minimap_scale.toml`` 的 ``scale.city_px_to_world``。
    :param sigma_px: 逐帧测量噪声（小地图像素），用于给漂移上界。
    :param max_step_px: 帧间位移上限，超过抛 :class:`TeleportJump`。

    :raises FileNotFoundError: 比例配置或瓦片索引不存在
    :raises ValueError: 比例值非正
    """

    def __init__(self,
                 start_world_pos: tuple[float, float],
                 grid: Optional[TileGrid] = None,
                 px_to_world: Optional[float] = None,
                 sigma_px: float = DEFAULT_SIGMA_PX,
                 max_step_px: float = MAX_STEP_PX):
        self.grid = grid if grid is not None else TileGrid()
        self.px_to_world = (self._load_scale() if px_to_world is None
                            else float(px_to_world))
        if self.px_to_world <= 0:
            raise ValueError(f"px_to_world 必须为正，收到 {self.px_to_world}")
        if max_step_px <= 0:
            raise ValueError(f"max_step_px 必须为正，收到 {max_step_px}")
        self.sigma_px = float(sigma_px)
        self.max_step_px = float(max_step_px)
        #: 连续超阈值帧数（见 :data:`OVER_STREAK_TO_REJECT`）
        self._over_streak = 0
        self._w2p: Callable[[float, float], tuple[float, float]] = \
            self._load_world_to_pixel()
        self.reset(start_world_pos)

    # -------------------------------------------------- 配置
    @staticmethod
    def _load_scale() -> float:
        p = _scale_toml_path()
        if not p.is_file():
            raise FileNotFoundError(
                f"找不到小地图比例配置: {p}\n"
                f"（它是比例的唯一出处，缺了就不猜默认值）")
        with p.open("rb") as fh:
            data = tomllib.load(fh)
        try:
            v = data["scale"]["city_px_to_world"]
        except KeyError:
            raise KeyError(
                f"{p} 缺 [scale].city_px_to_world。"
                f"现有 [scale] 键: {sorted(data.get('scale', {}))}") from None
        return float(v)

    @staticmethod
    def _load_world_to_pixel() -> Callable[[float, float],
                                            tuple[float, float]]:
        """取底图像素 <-> 世界坐标的换算。

        优先用已加载的 ``coord_transform``，否则按 :mod:`paths` 给的
        绝对路径加载。它**不在本包内**（是离线脚本），所以只能按文件加载。

        找不到就**抛**，不自己重抄一遍仿射系数 —— 抄两遍迟早会漂。
        """
        m = sys.modules.get("coord_transform")
        if m is None or not hasattr(m, "world_to_pixel"):
            p = Path(_paths().COORD_TRANSFORM_PY)
            if not p.is_file():
                raise FileNotFoundError(
                    f"找不到 coord_transform: {p}\n"
                    f"没有它就无法做世界坐标 <-> 底图像素换算。\n"
                    f"（它把'底图相对世界坐标是转置的'吸收进了仿射矩阵，"
                    f"自己重抄系数一定会错）")
            s = importlib.util.spec_from_file_location("coord_transform", p)
            m = importlib.util.module_from_spec(s)
            sys.modules["coord_transform"] = m
            s.loader.exec_module(m)
        if not hasattr(m, "world_to_pixel"):
            raise AttributeError(
                f"{getattr(m, '__file__', m)} 里没有 world_to_pixel()")
        return m.world_to_pixel

    # -------------------------------------------------- 状态
    def reset(self, start_world_pos: tuple[float, float]) -> TrackerState:
        """用新起点重置，**清空所有累积量**。"""
        wx, wz = float(start_world_pos[0]), float(start_world_pos[1])
        px, py = self._w2p(wx, wz)
        self._world = (wx, wz)
        self._pixel = (float(px), float(py))
        self._steps = 0
        self._path_len = 0.0
        self._over_streak = 0          # ★ 超阈值计数也要清，否则传送后第一帧就误抛
        return self._state(tile_changed=False)

    #: 传送后的重置入口。语义与 :meth:`reset` 相同，单独命名是为了让
    #: 调用方（传送契约的 WORLD_RESTORED 分支）一眼看懂意图。
    teleport_to = reset

    @property
    def world_pos(self) -> tuple[float, float]:
        return self._world

    @property
    def pixel(self) -> tuple[float, float]:
        return self._pixel

    @property
    def step_count(self) -> int:
        return self._steps

    @property
    def drift_bound(self) -> float:
        """累计漂移上界（世界单位）= ``sigma_px * sqrt(n) * scale``。"""
        return self.sigma_px * math.sqrt(self._steps) * self.px_to_world

    def _state(self, tile_changed: bool) -> TrackerState:
        t = self.grid.tile_of(*self._pixel)
        in_tile = (self._pixel[0] - t.x0, self._pixel[1] - t.y0) if t else None
        return TrackerState(
            world_pos=self._world, pixel=self._pixel,
            tile_id=(t.tile_id if t else None), in_tile_px=in_tile,
            tile_changed=tile_changed, step_index=self._steps,
            path_len=self._path_len, drift_bound=self.drift_bound)

    # -------------------------------------------------- 主入口
    def update(self, dx_px: float, dy_px: float) -> TrackerState:
        """输入本帧相对上帧的**小地图像素位移**，返回当前状态。

        :param dx_px: 小地图 +x 位移（屏幕向右为正）
        :param dy_px: 小地图 +y 位移（**屏幕向下为正**）
        :returns: :class:`TrackerState`
        :raises TeleportJump: 位移模长超过 ``max_step_px``。
            这是**故意不吞**的：传送产生的 40~125 px 跳变一旦累加，
            轨迹会瞬间错位且无法事后纠正。

        .. note::
           **轴向是转置的**（已用雅可比实测确认，见下方注释）。
           小地图正北朝上 => +x = 底图 +x = -world_z，
           +y = 底图 +y = -world_x。所以横移小地图会改变 world **z**，
           纵移会改变 world **x** —— 和"直觉的非转置"相反。
        """
        dx, dy = float(dx_px), float(dy_px)
        step = math.hypot(dx, dy)
        if step > self.max_step_px:
            # ★ 连续计数，不是单帧即判（见 OVER_STREAK_TO_REJECT 注释）。
            #   第 1 帧：丢弃但**不抛** —— 位置一动不动，既不污染轨迹，
            #   也不把单帧相关毛刺误报成传送。
            self._over_streak += 1
            if self._over_streak >= OVER_STREAK_TO_REJECT:
                raise TeleportJump(step, self.max_step_px, dx, dy)
            return self._state(tile_changed=False)
        self._over_streak = 0

        # 小地图像素 -> 世界单位。
        # ★ **转置**：底图 px 轴 = 世界 z 轴，py 轴 = 世界 x 轴（雅可比实测
        #   d(px)/d(world_z) = -0.9995, d(py)/d(world_x) = -0.9983）。
        #   小地图正北朝上 => 小地图 +x = 底图 +x = -world_z，
        #                       小地图 +y = 底图 +y = -world_x。
        #   **不能写成 dwx=dx, dwz=dy** —— 那会走错轴（曾导致走 +y 60 步
        #   仍在同一瓦片内，因为 +y 走的是块内另一个轴）。
        dwx = -dy * self.px_to_world
        dwz = -dx * self.px_to_world
        wx = self._world[0] + dwx
        wz = self._world[1] + dwz

        px, py = self._w2p(wx, wz)
        prev_tile = self.grid.tile_of(*self._pixel)
        self._world = (wx, wz)
        self._pixel = (float(px), float(py))
        self._steps += 1
        self._path_len += step * self.px_to_world
        new_tile = self.grid.tile_of(*self._pixel)
        changed = (getattr(prev_tile, "tile_id", None)
                   != getattr(new_tile, "tile_id", None))
        return self._state(tile_changed=changed)

    # -------------------------------------------------- 预加载
    def nearby_tiles(self) -> list[str]:
        """当前块 + 8 邻居（用于预加载）。

        出了底图（``tile_id is None``）时返回空列表 —— **不编造**。
        """
        cur = self.grid.tile_of(*self._pixel)
        if cur is None:
            return []
        return self.grid.neighbors(cur.tile_id)

    def __repr__(self) -> str:                            # pragma: no cover
        t = self.grid.tile_of(*self._pixel)
        return (f"TileTracker(world=({self._world[0]:.1f}, {self._world[1]:.1f}) "
                f"tile={getattr(t, 'tile_id', None)!r} steps={self._steps} "
                f"drift<={self.drift_bound:.2f})")
