"""小地图圆形掩码。

★ 为什么需要
------------
小地图所在的那个矩形里，除了圆盘本身还有**方向标、缩放档位数字、
坐标文本、边框**等干扰物。做 NCC 模板匹配时这些像素会拉低分数。
本模块把矩形裁出来再套圆形掩码，**只留圆盘内容**。

★ 坐标约定
----------
* ``cx`` / ``cy``：圆心，按**当前帧宽高各自归一化**（除以 w / h）
* ``r_norm_by_width``：半径**按宽度归一化**（除以 w），**不用对角线** ——
  对角线归一化会随画面比例漂移；按宽度则保证圆在任何分辨率下
  横向切得一样准

★ 分辨率无关
------------
所有换算都基于**当前帧的实际宽高**，不硬编码 720。
1280x720 和 1152x720 都能用。
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Optional

__all__ = [
    "load_circle_cfg",
    "make_minimap_mask",
    "minimap_rect",
]


def _default_cfg_path() -> Path:
    """rois.toml 的位置：``perception/paths.py`` 里已经定义好了。"""
    try:
        from .paths import ROIS_TOML
        return Path(ROIS_TOML)
    except Exception:                          # noqa: BLE001
        # 独立跑（不作为包导入）时的兜底路径
        here = Path(__file__).resolve()
        for up in here.parents:
            cand = up / "recognition" / "configs" / "rois.toml"
            if cand.is_file():
                return cand
        return Path(r"D:\NEKO\recognition\configs\rois.toml")


def load_circle_cfg(cfg_path: Optional[str | Path] = None) -> dict:
    """从 rois.toml 读 ``[rois.minimap_circle]``。

    :raises FileNotFoundError: 配置文件不存在
    :raises KeyError: 配置里没有该段（**响亮失败**，不静默用默认值）
    """
    p = Path(cfg_path) if cfg_path else _default_cfg_path()
    if not p.is_file():
        raise FileNotFoundError(f"rois.toml 不存在: {p}")
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    try:
        return data["rois"]["minimap_circle"]
    except KeyError as exc:
        raise KeyError(
            f"{p} 里缺少 [rois.minimap_circle] 段。\n"
            f"当前 rois 段有哪些键: {sorted(data.get('rois', {}))}") from exc


def minimap_rect(frame, cfg: Optional[dict] = None) -> tuple[int, int, int, int]:
    """小地图**外接矩形**（含圆盘），按当前帧尺寸缩放。

    :returns: ``(x0, y0, x1, y1)`` 左闭右开，可直接切片
    """
    cfg = cfg or load_circle_cfg()
    h, w = frame.shape[:2]
    cx = float(cfg["cx"]) * w
    cy = float(cfg["cy"]) * h
    r = float(cfg["r_norm_by_width"]) * w
    return (max(0, int(math.floor(cx - r))),
            max(0, int(math.floor(cy - r))),
            min(w, int(math.ceil(cx + r))),
            min(h, int(math.ceil(cy + r))))


def make_minimap_mask(frame, rois: Optional[dict] = None, pad: int = 0):
    """生成与输入同尺寸的 uint8 圆形掩码。**圆内 255，圆外 0。**

    :param frame: 任意尺寸的 BGR 画面（单通道也行）
    :param rois: ``[rois.minimap_circle]`` 段；``None`` 时读 rois.toml
    :param pad: 半径外扩（正数）/ 内缩（负数）**像素**。默认 0。
        外扩用来抗边缘抗锯齿，内缩用来只取最中心区域。
    :returns: ``np.ndarray``，dtype ``uint8``，形状与 ``frame`` 前两维一致
    :raises ValueError: 画面尺寸退化，或半径缩放后 <= 0
    """
    import numpy as np
    frame = np.asarray(frame)
    if frame.ndim not in (2, 3) or frame.shape[0] < 4 or frame.shape[1] < 4:
        raise ValueError(
            f"画面过小或格式不支持: shape={frame.shape}（需要 HxW 或 HxWx3，"
            f"且 H,W >= 4）")
    cfg = rois or load_circle_cfg()
    h, w = frame.shape[:2]
    cx = float(cfg["cx"]) * w
    cy = float(cfg["cy"]) * h
    r = float(cfg["r_norm_by_width"]) * w + float(pad)
    if r <= 0:
        raise ValueError(
            f"半径缩放后 <= 0：归一化半径 {cfg['r_norm_by_width']} * 宽 {w} "
            f"+ pad {pad} = {r}")
    return make_circle_mask(h, w, cx, cy, r)


def make_circle_mask(h: int, w: int, cx: float, cy: float, r: float):
    """按像素坐标直接画圆盘掩码（不走归一化）。"""
    import numpy as np
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return (d2 <= r * r).astype(np.uint8) * 255


#: ``make_minimap_roi`` 的默认半径补偿（像素，正=外扩）。
#: 手工标注的圆盘边界偏内，+3 用于吃掉小地图外环的暗边。
#: 负值内缩 / 需要更大覆盖就传别的 pad。
DEFAULT_ROI_PAD = 3


def make_minimap_roi(frame, rois: Optional[dict] = None,
                     pad: int = DEFAULT_ROI_PAD):
    """**推荐入口**：一步拿到"小地图裁剪图 + 同尺寸圆形掩码"。

    :returns: ``(crop, mask)``，两者尺寸一致，``mask`` 圆内 255 / 圆外 0

    ⚠️ 别自己写 ``make_minimap_mask(crop)``
    ---------------------------------------
    归一化是相对**整帧**的。如果先裁剪再对裁剪图调用
    ``make_minimap_mask``，圆心/半径会按裁剪图尺寸重算 ——
    261x261 的裁剪图上 r 会变成 ``0.0563*261 = 14.7`` 而不是 129.7，
    圆会缩到左上角一小块。**必须先在整帧上算掩码，再按同一个 rect 裁。**
    本函数就是这个顺序。
    """
    frame = np.asarray(frame) if False else frame
    rect = minimap_rect(frame, rois)
    x0, y0, x1, y1 = rect
    full_mask = make_minimap_mask(frame, rois, pad=pad)
    return frame[y0:y1, x0:x1], full_mask[y0:y1, x0:x1]
