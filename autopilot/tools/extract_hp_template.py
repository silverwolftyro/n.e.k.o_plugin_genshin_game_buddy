"""从一张真机截图里提取血条参考资产（离线，不开游戏）。

> 2026-10-12 首次落地。产出见 ``autopilot/perception/templates/``。

★ 为什么需要脚本，而不是把坐标写死在代码里
  ----------------------------------------
  血条 ROI 依赖**分辨率**。本脚本在 2560x1600 上测出的是
  ``(1076, 1500, 408, 15)``；换一台 1920x1080 的机器这个框就完全错位。
  所以这里**先自动检测**血条，再把结果写成 JSON 资产 —— 换机器时重跑一遍本脚本
  就行，而不是去改代码里的常量。

★ 三个产出物，性质完全不同，别混为一谈
  ------------------------------------
  1. ``full_hp_template.png``   —— **真实**。逐像素原样裁剪，未做任何修改。
  2. ``hp_bar_reference.json``  —— **实测事实**。ROI、填充色、外框色，全部是量出来的。
  3. ``empty_hp_template.png``  —— **合成**。⚠️ 满血帧里根本没有空条区域，
     空条底色无法从真实数据测得。这里用**声明的**合成色生成，
     只作单测夹具，**不是**真机采集，**不能**当成「空血长什么样」的证据。

★ 为什么满血帧测不出 ``background_color``
  --------------------------------------
  满血时条的内部是一整片纯色 —— 它和「一块纯色补丁」在像素上完全无法区分。
  所以 :func:`~autopilot.perception.hp_estimator.estimate_fill_ratio` 在没有显式
  传入 ``background_color`` 时会返回 ``None``（不猜）。
  幸而这**只影响满血/空血两档**：部分血量的条内部有填充与空白的分界，
  不需要任何额外参数就能测准。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

#: 血条 ROI 相对 2560x1600 参考分辨率的默认值（真机实测）。
REF_WIDTH = 2560
REF_HEIGHT = 1600

#: 合成空条用的底色。**这是我们选的，不是量出来的** ——
#: 见模块 docstring「为什么满血帧测不出 background_color」。
SYNTH_EMPTY_BG = (25, 25, 25)


# ==================================================================
# 自动检测
# ==================================================================

def detect_hp_bar(
    frame: np.ndarray,
    search: tuple[int, int, int, int] = (980, 1460, 1620, 1540),
) -> tuple[int, int, int, int] | None:
    """在 ``search`` 区域里找血条，返回 ``(x, y, w, h)``。

    :param search: ``(x0, y0, x1, y1)`` 搜索框。默认只扫底部中间 ——
        满屏扫会把小地图、地形绿化都算进来。
    :returns: 找不到返回 ``None``（**不猜默认框**）。
    """
    x0, y0, x1, y1 = search
    h, w = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    sub = frame[y0:y1, x0:x1]

    b, g, r = (sub[:, :, i].astype(np.int16) for i in range(3))
    # 血条绿：G 显著高于 B，且 R 居中。三个条件同时满足才算是条，
    # 少一个都会把地形绿化 / 草地吃进来。
    mask = (g > 150) & (g - b > 80) & (r > 60) & (r < 230)
    ys, xs = np.nonzero(mask)
    if ys.size < 50:
        return None

    bx0, bx1 = xs.min(), xs.max()
    by0, by1 = ys.min(), ys.max()
    w_box, h_box = bx1 - bx0 + 1, by1 - by0 + 1
    # 血条是扁长条：宽高比至少 10。高宽比不对说明框到了别的东西。
    if w_box < h_box * 10 or h_box > 60:
        return None
    return int(x0 + bx0), int(y0 + by0), int(w_box), int(h_box)


def dominant_color(patch: np.ndarray) -> tuple[int, int, int]:
    """块里的主色（出现次数最多的那一个），BGR 元组。"""
    if patch.size == 0:
        return (0, 0, 0)
    colors, counts = np.unique(patch.reshape(-1, 3), axis=0, return_counts=True)
    return tuple(int(v) for v in colors[int(np.argmax(counts))])


def measure(bar: np.ndarray) -> dict[str, Any]:
    """量一根条，产出**实测**信息。"""
    h, w = bar.shape[:2]
    # 外框色取首列（最外一圈像素）
    frame_color = dominant_color(bar[:, 0:1])
    # 填充色 = 整块的主色（满血时它占多数）
    fill_color = dominant_color(bar)
    # 白色像素占比 —— 血条上常压着 "17640/17640" 这种文字，
    # 它会被几何投影误当成「更满」，所以必须量出来记账。
    white = int(np.sum(np.all(bar >= 200, axis=2)))
    return {
        "width": w,
        "height": h,
        "fill_bgr": list(fill_color),
        "frame_bgr": list(frame_color),
        "white_overlay_px": white,
        "white_overlay_ratio": round(white / (h * w), 4),
    }


# ==================================================================
# 合成空条（明确标注为合成）
# ==================================================================

def synth_empty(
    frame_bgr: tuple[int, int, int],
    fill_bgr: tuple[int, int, int],
    width: int,
    height: int,
    border: int = 1,
    bg: tuple[int, int, int] = SYNTH_EMPTY_BG,
) -> np.ndarray:
    """合成一根**空**条。

    ⚠️ 这不是真机采集物。满血帧里没有空条区域可裁，本函数用声明的 ``bg``
    画出单测夹具。用它验证「空血应当读出 0.0」是合理的；
    用它宣称「真机空血就是这个颜色」是错的。
    """
    a = np.empty((height, width, 3), np.uint8)
    a[:, :] = frame_bgr
    a[border:height - border, border:width - border] = bg
    return a


# ==================================================================
# 主流程
# ==================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从截图提取血条参考资产")
    ap.add_argument("--image", required=True, help="真机截图路径")
    ap.add_argument("--out", default=None, help="输出目录（默认 perception/templates）")
    ap.add_argument("--roi", default=None,
                    help="手动指定 ROI 'x,y,w,h'；不给就自动检测")
    args = ap.parse_args(argv)

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"[X] 截图不存在：{img_path}")
        return 1
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[X] 解码失败（不是图片？）：{img_path}")
        return 1
    h, w = frame.shape[:2]
    print(f"截图 {w}x{h}")

    if args.roi:
        x, y, rw, rh = (int(v) for v in args.roi.split(","))
        print(f"用手动 ROI {x},{y},{rw},{rh}")
    else:
        found = detect_hp_bar(frame)
        if found is None:
            print("[X] 没检测到血条。角色满血 + 无队友时可能是这个原因；")
            print("    也可以用 --roi 手动指定 x,y,w,h。")
            return 2
        x, y, rw, rh = found
        print(f"自动检测到血条 ROI：({x}, {y}, {rw}, {rh})")

    bar = frame[y:y + rh, x:x + rw]
    m = measure(bar)

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / \
        "perception" / "templates"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 真实满血条 —— 逐像素原样
    cv2.imwrite(str(out_dir / "full_hp_template.png"), bar)

    # 2) 合成空条 —— 明确标注
    empty = synth_empty(tuple(m["frame_bgr"]), tuple(m["fill_bgr"]), rw, rh)
    cv2.imwrite(str(out_dir / "empty_hp_template.png"), empty)

    # 3) 实测事实
    ref = {
        "_comment": "由 autopilot/tools/extract_hp_template.py 实测生成，勿手改。"
                    "ROI 依赖分辨率，换机器请重跑脚本。",
        "measured_from": str(img_path),
        "frame_size": [w, h],
        "reference_resolution": [REF_WIDTH, REF_HEIGHT],
        "roi_xywh": [x, y, rw, rh],
        "roi_frac": [round(x / w, 6), round(y / h, 6),
                     round(rw / w, 6), round(rh / h, 6)],
        "fill_bgr": m["fill_bgr"],
        "frame_bgr": m["frame_bgr"],
        "white_overlay_ratio": m["white_overlay_ratio"],
        "empty_background_bgr": list(SYNTH_EMPTY_BG),
        "empty_template_is_synthetic": True,
        "notes": [
            "fill_bgr / frame_bgr / white_overlay_ratio 为真机实测。",
            "empty_background_bgr 是我们选的合成色，不是量出来的 —— "
            "满血帧里没有空条区域可测。",
            "hp_estimator 在没有显式 background_color 时，"
            "对满血/空血返回 None（不猜）；部分血量不需要该参数。",
        ],
    }
    (out_dir / "hp_bar_reference.json").write_text(
        json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n填充色 BGR {tuple(m['fill_bgr'])}   外框色 BGR {tuple(m['frame_bgr'])}")
    print(f"条上白色文字占比 {m['white_overlay_ratio']:.1%}"
          f"（会被几何投影误当成更满，已记账）")
    print(f"\n写出 -> {out_dir}")
    for n in ("full_hp_template.png", "empty_hp_template.png",
              "hp_bar_reference.json"):
        p = out_dir / n
        print(f"  {p.name:26s} {p.stat().st_size:>7} B")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())