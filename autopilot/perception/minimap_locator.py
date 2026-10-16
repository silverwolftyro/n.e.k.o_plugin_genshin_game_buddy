# @deprecated —— 外观匹配定位路线已证伪（规划文档 §14），改用 LocalizationService
"""小地图定位核心：掩码 NCC + 金字塔多层级搜索。

★★ 为什么「已废弃」的文件还在包里（2026-10-13 记）
  --------------------------------------------------
  质量扫描把它判成死代码（全仓零引用，建议删整个文件）。**那个结论只对
  「本仓库」成立** —— 它是被 ``recognition\exports\benchmark\scripts\`` 下
  7 个**开发期探针**按路径加载的：

      probe/diag_locator_deep.py       probe/validate_locator_real.py
      probe/diag_ncc_real.py          probe/diag_orb_feasibility.py
      probe/exp_e_orb_l2_locate.py    probe/exp_phase_corr.py
      test/test_minimap_locator.py

  它们加载方式是 ``_load("minimap_locator", PERC / "minimap_locator.py")``
  —— 直接按文件路径读，**不是 import**。所以本仓库删掉它不会让
  ``autopilot`` 少一块拼图，但会让上面 7 个脚本在跑之前就
  ``FileNotFoundError``。

  因此本轮选择**保留原地**而不是删，也不移到 ``_deprecated/`` ——
  移动同样会让那 7 条按路径的引用失效，而收益只是「目录好看点」。

  **代价**：一个已证伪的 286 行模块继续随包分发。
  **规则**：可以读、可以参考证伪过程，**不要在新代码里 import 或扩展它**。
  真要清理，请连同上面 7 个 recognition 探针脚本一起处理。

.. deprecated::
    **本模块已废弃，不要在新代码里调用。**

    原因：这套"截小地图 → 在底图里搜最像的位置"的外观匹配路线
    **已被实测证伪**（规划文档 §14）。session_a 152 帧 × L2 全图穷举，
    **0 帧有解**：

    * 真实小地图在底图上的最佳内点数只有 **5~7**，且峰值位置跨帧不一致
      → 这是噪声地板，不是"再调调参数就能用"
    * 根因一：底图自相似（水陆交界到处长得都一样）
    * 根因二：即使换成 ORB，同源同尺度也只有 1187 内点，跨尺度 348/401，
      且 OpenCV 4.11 的 ``knnMatch`` 训练集超过约 20 万描述子会断言失败

    当前定位走**任务驱动**路线：起点由传送 OCR + tp.json 给出，
    之后靠**帧间位移累积**（相位相关），终点由 TileTracker 做瓦片查询。
    见 :mod:`autopilot.perception.localization_service`。

    **保留本文件不删**：底层函数（``match_ncc`` / 层级精化逻辑）仍是
    可读的参考实现，且没有其他模块 import 它（已 grep 验证），
    标注废弃不会破坏任何现存 import。

★ 匹配为什么用 SQDIFF 定位、再手算 NCC
--------------------------------------
``cv2.matchTemplate`` 只有 ``TM_SQDIFF`` / ``TM_CCORR_NORMED`` 支持
``mask`` 参数，**``TM_CCOEFF_NORMED`` 不支持**。而我们要的正是
"相关系数 + 圆形掩码"。所以：

1. 用 ``TM_SQDIFF`` + mask 找最相似位置（圆内像素的平方差最小）
2. 在该位置**手算**真正的 NCC（只用掩膜内的像素）

这样既快（SQDIFF 走 FFT）又精确（NCC 按定义算）。

★ 层级流程
----------
::

    L4 (1/16)  全局粗搜  —— 2304x1152 一次扫完
      ↓
    L3 (1/8)   上一级位置附近小窗精化
      ↓
    L2 (1/4)
      ↓
    L1 (1/2)
      ↓
    L0 (1/1)   chunk_loader 局部窗口（跨块自动补拉）
      ↓
    coord_transform.pixel_to_world()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

__all__ = [
    "LocateResult", "SearchRange", "SCALE_CITY", "SCALE_FIELD",
    "MATCH_THRESHOLD", "locate_minimap", "locate_with_hint", "match_ncc",
]

#: PEP 702 废弃标记。静态检查器 / IDE 据此提示。
#: 替代方案见 :mod:`autopilot.perception.localization_service`。
__deprecated__ = "外观匹配定位已证伪（规划文档 §14）；改用 LocalizationService"

#: 城��� / 野外两档。数值 = 小地图模板在 L0 上要放大多少倍
SCALE_CITY = 2.0
SCALE_FIELD = 1.0

#: 采信阈值（规划文档 §4.4 用户定的工程标准）
MATCH_THRESHOLD = 0.98

#: 每层精化的搜索半径（该层像素）
_REFINE_RADIUS = 24
_COARSE_MIN = 8          # L4 上最小搜索边长


class SearchRange(Enum):
    L1 = "L1"
    L1_PLUS = "L1+"
    L1_PLUS_PLUS = "L1++"
    L2 = "L2"

    @property
    def cells(self) -> int:
        """搜索半径（块数）。L2 = 全局，特殊处理。"""
        return {SearchRange.L1: 0, SearchRange.L1_PLUS: 1,
                SearchRange.L1_PLUS_PLUS: 2, SearchRange.L2: 99}[self]


@dataclass
class LocateResult:
    ok: bool
    pixel: Optional[Tuple[float, float]] = None
    world: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    scale_used: Optional[str] = None
    level_reached: int = -1
    elapsed_ms: float = 0.0
    note: str = ""
    steps: list = field(default_factory=list)


# ================================================== NCC
def match_ncc(search_img, template, template_mask=None):
    """掩码归一化互相关匹配。

    :param search_img: 大图（灰度）
    :param template: 模板（灰度，须小于 search）
    :param template_mask: 与 template 同尺寸的 uint8 掩码（255=参与）。
        ``None`` 表示全参与。
    :returns: ``(最高分, (x, y))``，找不到返回 ``(-1.0, (-1, -1))``
    """
    import cv2
    import numpy as np
    s = np.asarray(search_img)
    t = np.asarray(template)
    if s.ndim == 3:
        s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    if t.ndim == 3:
        t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    th, tw = t.shape[:2]
    sh, sw = s.shape[:2]
    if th > sh or tw > sw:
        return -1.0, (-1, -1)

    if template_mask is None:
        res = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        return float(mx), (int(loc[0]), int(loc[1]))

    m = np.asarray(template_mask)
    if m.shape != t.shape[:2]:
        raise ValueError(f"mask 尺寸 {m.shape[:2]} 与模板 {t.shape[:2]} 不符")
    if m.max() > 1:
        m = (m > 127).astype(np.uint8)
    nz = m > 0
    if nz.sum() < 16:                       # 有效像素太少，NCC 无意义
        return -1.0, (-1, -1)

    # ① 快速定位：掩码平方差
    res = cv2.matchTemplate(s, t, cv2.TM_SQDIFF, mask=m)
    _, _, minloc, _ = cv2.minMaxLoc(res)
    x, y = int(minloc[0]), int(minloc[1])

    # ② 手算该位置的 NCC（只用掩码内像素）
    tv = t[nz].astype(np.float64)
    mu_t = tv.mean()
    sd_t = tv.std()
    best_score, best_loc = -1.0, (x, y)
    if sd_t < 1e-9:
        return 0.0, (x, y)
    for dy in range(max(0, y - 1), min(sh - th, y + 1) + 1):
        for dx in range(max(0, x - 1), min(sw - tw, x + 1) + 1):
            win = s[dy:dy + th, dx:dx + tw][nz].astype(np.float64)
            mu_w = win.mean()
            sd_w = win.std()
            if sd_w < 1e-9:
                continue
            ncc = float(((tv - mu_t) * (win - mu_w)).mean() / (sd_t * sd_w))
            if ncc > best_score:
                best_score, best_loc = ncc, (dx, dy)
    return best_score, best_loc


# ================================================== 定位
def _sibling(name: str):
    """加载同目录模块。**兼容"作为包导入"和"独立按文件加载"。**"""
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


def _load_coord_transform():
    """加载 coord_transform.py。

    路径可用环境变量 ``NEKO_COORD_TRANSFORM`` 覆盖 ——
    那个文件在 recognition 树里，不在插件树里，靠相对层数推不稳。
    """
    import importlib.util
    import os
    import sys
    p = Path(os.environ.get(
        "NEKO_COORD_TRANSFORM",
        r"D:\NEKO\recognition\exports\benchmark\scripts\misc"
        r"\coord_transform.py"))
    if not p.is_file():
        raise FileNotFoundError(
            f"找不到 coord_transform.py: {p}\n"
            f"用环境变量 NEKO_COORD_TRANSFORM 指定路径。")
    s = importlib.util.spec_from_file_location("neko_coord_transform", p)
    m = importlib.util.module_from_spec(s)
    sys.modules["neko_coord_transform"] = m
    s.loader.exec_module(m)
    return m


def _to_pyr(arr, size: int):
    import cv2
    import numpy as np
    a = np.asarray(arr)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if a.shape[0] == size and a.shape[1] == size:
        return a
    interp = (cv2.INTER_AREA if size < a.shape[0]
              else cv2.INTER_CUBIC)
    return cv2.resize(a, (size, size), interpolation=interp)


def _make_template(mini_gray, minimap_mask, scale, ctx):
    """按档位把小地图缩放成模板 + 掩码。"""
    import cv2
    import numpy as np
    d = int(round(mini_gray.shape[0] * scale))
    t = cv2.resize(mini_gray, (d, d), interpolation=cv2.INTER_CUBIC)
    mk = cv2.resize(minimap_mask, (d, d), interpolation=cv2.INTER_NEAREST)
    mk = (mk > 127).astype(np.uint8) * 255
    return t, mk


def _search_level(pyr_img, templ, tmpl_mask, center=None,
                  radius=_REFINE_RADIUS, level=0, out=None):
    """在某一层搜。``center=None`` -> 全局；否则只搜 center 附近半径内。"""
    sh, sw = pyr_img.shape[:2]
    th, tw = templ.shape[:2]
    if center is None:
        roi = pyr_img
        ox = oy = 0
    else:
        cx, cy = int(round(center[0])), int(round(center[1]))
        R = max(_COARSE_MIN, radius)
        x0, y0 = max(0, cx - R), max(0, cy - R)
        x1, y1 = min(sw, cx + R + tw), min(sh, cy + R + th)
        if x1 - x0 < tw or y1 - y0 < th:
            return None
        roi = pyr_img[y0:y1, x0:x1]
        ox, oy = x0, y0
    if roi.shape[0] < th or roi.shape[1] < tw:
        return None
    sc, loc = match_ncc(roi, templ, tmpl_mask)
    if loc[0] < 0:
        return None
    gx = loc[0] + ox + tw / 2.0
    gy = loc[1] + oy + th / 2.0
    if out is not None:
        out.append(f"L{level} score={sc:.4f} @({gx:.0f},{gy:.0f})")
    return gx, gy, sc


def locate_minimap(frame_720, last_known_pos=None, hint_scale="auto",
                   coord_transform=None, chunk_loader=None,
                   pyramid=None, max_range=SearchRange.L2,
                   thresholds=(MATCH_THRESHOLD,)):
    """定位小地图在世界坐标中的位置。

    :param frame_720: 整帧画面（BGR）
    :param last_known_pos: 上一帧世界坐标 ``(wx, wz)``，给了就做局部优先
    :param hint_scale: ``"auto"`` / ``"field"`` / ``"city"``
    :param thresholds: 采信阈值，可给多个做双档尝试
    :returns: :class:`LocateResult`（失败时 ``ok=False``）
    """
    t0 = time.perf_counter()

    chunk_loader = chunk_loader or _sibling("chunk_loader").ChunkLoader()
    pyramid = pyramid or _sibling("pyramid_loader").PyramidLoader()
    make_minimap_roi = _sibling("minimap_mask").make_minimap_roi
    if coord_transform is None:
        coord_transform = _load_coord_transform()

    res = LocateResult(ok=False)
    # 1) 小地图裁图 + 圆形掩码
    mini, mask = make_minimap_roi(frame_720)
    import cv2
    mini_gray = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)

    scales = ([(SCALE_FIELD, "field"), (SCALE_CITY, "city")]
              if hint_scale == "auto" else
              [(SCALE_FIELD, "field")] if hint_scale == "field"
              else [(SCALE_CITY, "city")])

    for sc, nm in scales:
        templ, tmpl_mask = _make_template(mini_gray, mask, sc, None)
        L0_size = templ.shape[0]
        last_ok = None
        for lv in (4, 3, 2, 1, 0):
            if lv == 0:
                break
            img = pyramid.get_level(lv)
            k = pyramid.get_level_scale(lv)      # 该层 / L0
            t_size = max(8, int(round(L0_size * k)))
            t_lv, m_lv = _to_pyr(templ, t_size), \
                cv2.resize(tmpl_mask, (t_size, t_size),
                           interpolation=cv2.INTER_NEAREST)
            center = None
            if last_known_pos is not None and lv < 4:
                wx, wz = last_known_pos
                wpx = coord_transform.world_to_pixel(wx, wz)[0]
                wpy = coord_transform.world_to_pixel(wx, wz)[1]
                center = (wpx * k, wpy * k)
            r = _search_level(img, t_lv, m_lv, center, _REFINE_RADIUS, lv,
                              res.steps)
            if r is None:
                break
            gx, gy, score = r
            if score < max(thresholds) and lv == 4 and \
                    last_known_pos is None:
                last_ok = (gx, gy, score)
                break
            if score < max(thresholds):
                last_ok = (gx, gy, score)
                break
            last_ok = (gx, gy, score)
            if lv == 1:
                break
        if last_ok is None:
            continue
        gx, gy, score = last_ok
        if score < min(thresholds):
            continue
        # 4) 落到 L0 精化
        k1 = pyramid.get_level_scale(1)
        px0 = gx / k1
        py0 = gy / k1
        win = chunk_loader.load_l0_window((px0, py0), L0_size)
        sc0, loc0 = match_ncc(win, templ, tmpl_mask)
        if sc0 >= 0:
            px = px0 - L0_size / 2.0 + loc0[0] + templ.shape[1] / 2.0
            py = py0 - L0_size / 2.0 + loc0[1] + templ.shape[0] / 2.0
            if sc0 >= score:
                score, px0, py0 = sc0, px, py
        res.ok = score >= min(thresholds)
        res.pixel = (px0, py0)
        res.world = coord_transform.pixel_to_world(px0, py0)
        res.confidence = float(score)
        res.scale_used = nm
        res.level_reached = 0
        res.elapsed_ms = (time.perf_counter() - t0) * 1000
        return res
    res.elapsed_ms = (time.perf_counter() - t0) * 1000
    res.note = "所有档位/层级都没过阈值"
    return res


def locate_with_hint(frame_720, last_known_pos, max_range,
                     **kw):
    """带搜索范围约束的定位（范围不够时自动升一级）。"""
    order = {SearchRange.L1: (4, 3, 2, 1),
             SearchRange.L1_PLUS: (4, 3, 2, 1),
             SearchRange.L1_PLUS_PLUS: (4, 3, 2, 1),
             SearchRange.L2: (4, 3, 2, 1)}
    return locate_minimap(frame_720, last_known_pos=last_known_pos,
                          max_range=max_range, **kw)
