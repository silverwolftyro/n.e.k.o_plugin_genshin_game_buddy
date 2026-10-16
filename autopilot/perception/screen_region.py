"""ImageRegion：链式裁剪的图像区域，坐标可回推。

对应架构文档 4.1。为什么要这么设计：
    感知层会反复裁剪（整帧 -> 顶栏 -> 某个图标），识别结果往往需要
    换算回全屏坐标才能点/画框。ImageRegion 记录 origin 与变换历史，
    让坐标**双向可推**。

一条硬规则（来自 BetterGI 的教训）：
    **在裁剪过深的 Region 上禁止使用参考画布锚定。** 锚定需要知道
    "我在整图的哪一块"，深层裁剪后这个语义已经丢失。本类的
    ``supports_reference_anchor`` 属性把这条规则变成可检查的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass
class ImageRegion:
    """带坐标变换历史的图像区域。

    ``origin`` 是**在父区域自身图像坐标系**中的左上角偏移（即裁剪量本身），
    不是相对根的绝对偏移。这样 ``to_root_coords`` 沿链上溯时每个节点只加一次，
    不会出现重复累加。

    ``parent`` 串起完整历史，``to_screen_coords`` 沿链上溯再叠加根区域的屏幕原点。
    """

    image: np.ndarray
    origin: tuple[int, int] = (0, 0)
    parent: "ImageRegion | None" = field(default=None, repr=False)
    name: str = ""

    # -- 基本属性 --------------------------------------------------------

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.image.shape)

    @property
    def depth(self) -> int:
        """裁剪层数。根区域为 0。"""
        d = 0
        cur: ImageRegion | None = self.parent
        while cur is not None:
            d += 1
            cur = cur.parent
        return d

    @property
    def supports_reference_anchor(self) -> bool:
        """是否还能用参考画布锚定。

        根区域与它的一层直接派生可以（对齐 BetterGI 的 CanUseReferenceSearch：
        只允许整图或 DeriveTo1080P 的直接派生）。更深就必须用 ROI 表达式。
        """
        return self.depth <= 1

    # -- 裁剪 ------------------------------------------------------------

    def derive_crop(self, x: int, y: int, w: int, h: int, *, name: str = "") -> "ImageRegion":
        """裁剪出子区域。**越界会被裁剪并记录**（UI 元素可能部分在屏外）。"""
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            raise ValueError(f"裁剪尺寸非法: w={w}, h={h}")
        cx1, cy1 = max(0, x), max(0, y)
        cx2, cy2 = min(self.width, x + w), min(self.height, y + h)
        if cx2 <= cx1 or cy2 <= cy1:
            raise ValueError(
                f"裁剪区域完全在图像外: ({x},{y},{w},{h}) vs {self.width}x{self.height}"
            )
        sub = self.image[cy1:cy2, cx1:cx2]
        return ImageRegion(
            image=np.ascontiguousarray(sub),
            origin=(cx1, cy1),          # ★ 只记「在父图里的裁剪量」，不叠加父的 origin
            parent=self,
            name=name,
        )

    def derive_by_scale(self, scale: float, *, name: str = "") -> "ImageRegion":
        """按比例缩放（如 YOLO 前处理的降采样）。

        ⚠️ 缩放会改变坐标系，但本类**不跟踪缩放因子** —— origin 保持 (0,0)，
        坐标回推在缩放层以下是近似的。**需要精确回推时，请在缩放前记录
        ``scale``，或改用 ROI 表达式重新定位。** 这是刻意的取舍：
        缩放只用于喂模型，模型的输出框不需要回推（builder 会换算）。
        """
        import cv2  # noqa: PLC0415

        if scale <= 0:
            raise ValueError(f"scale 必须为正: {scale}")
        new_w = max(1, int(round(self.width * scale)))
        new_h = max(1, int(round(self.height * scale)))
        resized = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return ImageRegion(image=resized, origin=(0, 0), parent=self,
                           name=name or f"scale{scale:g}")

    def derive_1080p(self, *, name: str = "1080p") -> "ImageRegion":
        """缩放到高度 1080（对齐 BetterGI 的 DeriveTo1080P）。

        缩放后 origin 的换算需要反比例，下游用 to_screen_coords 时
        会按链上溯并考虑各级比例 —— 这里用一个保守近似：origin 按比例缩放，
        坐标回推时再除回来。**深链上的精确回推请改用 ROI 表达式重新定位。**
        """
        return self.derive_by_scale(1080.0 / max(1, self.height), name=name)

    # -- 坐标回推 --------------------------------------------------------

    def to_parent_coords(self, x: int, y: int) -> tuple[int, int]:
        """本区域坐标 -> 父区域坐标。"""
        return (int(x) + self.origin[0], int(y) + self.origin[1])

    def to_root_coords(self, x: int, y: int) -> tuple[int, int]:
        """本区域坐标 -> 根区域（全帧）坐标。"""
        cx, cy = int(x), int(y)
        cur: ImageRegion | None = self
        while cur is not None and cur.parent is not None:
            cx, cy = cur.to_parent_coords(cx, cy)
            cur = cur.parent
        return (cx, cy)

    def to_screen_coords(self, x: int, y: int) -> tuple[int, int]:
        """本区域坐标 -> 屏幕坐标。

        需要根 Region 持有 ``screen_origin``。``CapturedFrame`` 包装出来的
        根 Region 会带上它；纯离屏区域返回 None。
        """
        cx, cy = self.to_root_coords(x, y)
        origin = getattr(self, "_screen_origin", None)
        if origin is None:
            cur: ImageRegion | None = self
            while cur is not None:
                origin = getattr(cur, "_screen_origin", None)
                if origin is not None:
                    break
                cur = cur.parent
        if origin is None:
            return (cx, cy)
        return (int(cx + origin[0]), int(cy + origin[1]))

    def attach_screen_origin(self, screen_x: int, screen_y: int) -> "ImageRegion":
        """给根区域打上屏幕原点标记（由 StateBuilder 调用）。"""
        self._screen_origin = (int(screen_x), int(screen_y))  # type: ignore[attr-defined]
        return self

    # -- 匹配结果回推 ----------------------------------------------------

    def box_to_root(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """(x, y, w, h) 从本区域换算到根区域。"""
        x, y, w, h = box
        rx, ry = self.to_root_coords(x, y)
        return (rx, ry, int(w), int(h))

    def point_to_root(self, point: tuple[int, int]) -> tuple[int, int]:
        return self.to_root_coords(point[0], point[1])

    # -- 杂项 ------------------------------------------------------------

    def copy(self) -> np.ndarray:
        return self.image.copy()

    def iter_chain(self) -> Iterator["ImageRegion"]:
        """从根到当前依次产出，用于调试打印变换历史。"""
        chain: list[ImageRegion] = []
        cur: ImageRegion | None = self
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        yield from reversed(chain)

    def __repr__(self) -> str:
        return (
            f"ImageRegion(name={self.name!r}, {self.width}x{self.height}, "
            f"origin={self.origin}, depth={self.depth})"
        )


def region_of_frame(image: np.ndarray, screen_x: int = 0, screen_y: int = 0) -> ImageRegion:
    """把一帧裸图包成带屏幕原点的根 Region。"""
    r = ImageRegion(image=image, origin=(0, 0), parent=None, name="root")
    return r.attach_screen_origin(screen_x, screen_y)


__all__ = ["ImageRegion", "region_of_frame"]
