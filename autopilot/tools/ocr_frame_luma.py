"""OCR 抓帧体检 —— 回答「OCR 到底拿到了什么」。

    python -m autopilot.tools.ocr_frame_luma
    python -m autopilot.tools.ocr_frame_luma --save
    python -m autopilot.tools.ocr_frame_luma --title 原神 --delay 8

★ 为什么要有这个脚本（和 capture_diagnose 的区别）
--------------------------------------------------
``capture_diagnose`` 体检的是**引擎**的通道（WGC/BitBlt/DXcam/mss）。
但 OCR 走的是**另一套**、完全独立的代码：

    __init__.py:ocr_perceive
        -> screen_capture.capture_active_frame()
            -> capture_window_base64()   # 降级链 PrintWindow -> mss -> bridge

两套链**没有共用任何一行**，引擎那套修好的 WGC 对 OCR 一点帮助都没有。
所以「引擎看得到画面、OCR 说是黑屏」是完全可能同时成立的。

★ 怎么读结果
------------
``luma`` 列是全图平均亮度（0-255），但**光看它会被骗**：

``screen_capture.is_black_frame`` 的判据是 ``luma < 12`` 就判黑。
问题是**残帧**：通道只 blit 到画面的一部分（比如窗口刚改过分辨率、
GDI DC 尺寸对不上），另一半是纯黑 padding。这时全图 luma 轻松上到 18~25，
**顺利越过 12 这道门槛被判成「有画面」**，然后原样送进 OCR。
OCR 看到的就只有左上角那一块，识不出东西 —— 这就是
「实际上有画面、但 OCR 说是黑屏」。

所以本脚本同时打两列：

  * ``luma``      —— 平均亮度，OCR 路径自己的判据只看这个
  * ``dead%``     —— 亮度 < 12 的像素占比。**这一列 > 20% 就是残帧**
  * ``corner``    —— 四角亮度。用来看黑块在哪半边（判断是
                     「右/下半边没抓到」还是「整张都没抓到」）

★ --delay 是给「只能全屏」的游戏准备的
--------------------------------------
双击脚本那一刻游戏还是最小化的，客户区退化成 0x0，
那不是 bug。加 ``--delay 8``，它会每秒打一行倒计时，
你趁这 8 秒 alt-tab 回游戏。
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from pathlib import Path

from ._bootstrap import C, bootstrap, line

#: ``screen_capture.BLACK_FRAME_LUMA_THRESHOLD`` 的值。
#: 这里写死而不是 import，是为了让脚本在插件 import 失败时也能报数。
LUMA_BLACK = 12.0

#: dead% 超过它就判「残帧」。40% 是实测出来的：
#: 2026-10-13 capmatrix 里 BitBlt 的 normal/minimize/offscreen
#: 三张全部落在 53.9%~55.6%，而真正的完整帧是 0.0%~2.8%。
DEAD_FRAC = 0.20

OUT_DIR = Path(__file__).resolve().parents[1] / "_runtime" / "ocrprobe"


# ----------------------------------------------------------------------
# 帧体检
# ----------------------------------------------------------------------
def _probe(img) -> dict:
    """一张 RGB 图像 -> luma / dead% / corner 四个数。"""
    import numpy as np

    a = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = a.shape
    dead = float((a < LUMA_BLACK).mean())

    # 四角各取 16x16 的均值 —— 纯 padding 角的 luma 会精确接近 0
    k = max(1, min(h, w) // 16)
    corners = {
        "左上": float(a[:k, :k].mean()),
        "右上": float(a[:k, -k:].mean()),
        "左下": float(a[-k:, :k].mean()),
        "右下": float(a[-k:, -k:].mean()),
    }
    return {
        "w": w, "h": h,
        "luma": float(a.mean()),
        "dead": dead,
        "corners": corners,
    }


def _verdict(p: dict) -> str:
    """把四个数翻译成人话。"""
    if p["luma"] < LUMA_BLACK:
        return f"{C.ERR}整帧黑{C.OFF}（守卫会拦下来）"
    if p["dead"] > DEAD_FRAC:
        dark = [k for k, v in p["corners"].items() if v < LUMA_BLACK]
        where = ("黑角=" + ",".join(dark)) if dark else "黑块在中间"
        return (f"{C.ERR}残帧{C.OFF} luma 越过 {LUMA_BLACK:.0f} 门槛，"
                f"守卫拦不住 → OCR 只能看到一部分（{where}）")
    return f"{C.OK}完整帧{C.OFF}"


def _fmt(p: dict) -> str:
    c = p["corners"]
    return (f"{p['w']}x{p['h']:<5} luma={p['luma']:6.1f}  "
            f"dead={p['dead'] * 100:5.1f}%  "
            f"角[{c['左上']:.0f}/{c['右上']:.0f}/{c['左下']:.0f}/{c['右下']:.0f}]")


def _from_b64(b64: str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _save(img, tag: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"ocrprobe_{tag}.png"
    img.save(p)
    return p


# ----------------------------------------------------------------------
# 三段体检
# ----------------------------------------------------------------------
def _probe_ocr_path(title: str) -> dict | None:
    """第一段：**OCR 实际走的那条路**。这是主问题。"""
    print(f"{C.BOLD}【1】OCR 实际路径 screen_capture.capture_active_frame(){C.OFF}")
    print(line())

    import screen_capture as sc

    target = sc.find_genshin_window() if title is None else None
    if title is not None:
        target = {"title": title}
    print(f"  find_genshin_window -> "
          f"{(str(target.get('title'))[:40] if target else 'None（会走 foreground 兜底')}")

    try:
        res, path = sc.capture_active_frame()
    except Exception as exc:
        print(f"  {C.ERR}抛异常：{type(exc).__name__}: {exc}{C.OFF}")
        return None

    print(f"  capture_path = {path}")
    print(f"  ok={res.ok}  method={res.method}  error={res.error!r}  "
          f"size={res.width}x{res.height}")
    if not res.ok or not res.image_base64:
        print(f"  {C.WARN}没有拿到图 —— OCR 这一轮会走 bus.frames 兜底或直接早退{C.OFF}")
        print(f"  {C.DIM}这不算「黑屏」，是「没抓到」。黑屏是抓到了但是黑的。{C.OFF}")
        return None

    img = _from_b64(res.image_base64)
    p = _probe(img)
    print(f"  {_fmt(p)}")
    print(f"  is_black_frame() 判定 -> "
          f"{'BLACK' if sc.is_black_frame(res.image_base64) else 'not-black（会送去 OCR）'}")
    print(f"  结论：{_verdict(p)}")
    return img


def _probe_chain_routes(title: str) -> None:
    """第二段：把降级链每一路**单独**跑一遍，看是谁在交黑帧。"""
    print()
    print(f"{C.BOLD}【2】降级链逐路体检 PrintWindow -> mss -> bridge{C.OFF}")
    print(line())

    import screen_capture as sc

    target = sc.find_genshin_window()
    if not target:
        print(f"  {C.WARN}找不到原神窗口，跳过（降级链走的是本地窗口分支）{C.OFF}")
        return
    print(f"  窗口：{str(target.get('title'))[:50]}")
    print(f"  矩形：{target.get('rect')}")
    print()

    routes = (
        ("printwindow", lambda: sc._capture_printwindow(target.get("hwnd"), target.get("rect"))),
        ("mss", lambda: sc._capture_mss(target.get("rect"))),
        ("bridge", lambda: sc._capture_bridge(target.get("pid"), target.get("title", ""))),
    )
    for name, fn in routes:
        try:
            res = fn()
        except Exception as exc:
            print(f"  {name:<12} {C.ERR}抛异常 {type(exc).__name__}{C.OFF}")
            continue
        if not res.ok or not res.image_base64:
            print(f"  {name:<12} {C.DIM}抓不到（error={res.error}）{C.OFF}")
            continue
        p = _probe(_from_b64(res.image_base64))
        flag = "" if p["dead"] <= DEAD_FRAC else f"  {C.ERR}<-- 交的是残帧{C.OFF}"
        print(f"  {name:<12} {_fmt(p)}{flag}")


def _probe_engine_path(title: str) -> None:
    """第三段：引擎那套（WGC 优先）作对照。"""
    print()
    print(f"{C.BOLD}【3】对照：引擎的 Capturer（默认链 WGC -> BitBlt -> DXcam -> mss）{C.OFF}")
    print(line())

    try:
        from autopilot.perception.capturer import CaptureChannel, Capturer
    except Exception as exc:
        print(f"  {C.WARN}引擎 capturer 不可用：{exc}{C.OFF}")
        return

    import numpy as np

    for ch in (CaptureChannel.WGC, CaptureChannel.BITBLT,
               CaptureChannel.DXCAM, CaptureChannel.MSS):
        cap = Capturer(window_title=title or "原神", channels=[ch])
        try:
            frame = cap.grab()
        except Exception as exc:
            print(f"  {ch.value:<12} {C.DIM}抓不到（{type(exc).__name__}: {str(exc)[:40]}）{C.OFF}")
            continue
        finally:
            try:
                cap.close()
            except Exception:
                pass
        img = frame.image
        luma = float(img.mean())
        dead = float((img[:, :, :3].mean(axis=2) < LUMA_BLACK).mean())
        flag = "" if dead <= DEAD_FRAC else f"  {C.ERR}<-- 残帧{C.OFF}"
        print(f"  {ch.value:<12} {img.shape[1]}x{img.shape[0]:<5} "
              f"luma={luma:6.1f}  dead={dead * 100:5.1f}%{flag}")
        del np


# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="OCR 抓帧体检")
    parser.add_argument("--title", default=None, help="窗口标题子串（默认自动找）")
    parser.add_argument("--delay", type=int, default=0,
                        help="开跑前倒计时秒数（alt-tab 回游戏用）")
    parser.add_argument("--save", action="store_true", help="把抓到的帧存成 png")
    parser.add_argument("--engine", action="store_true",
                        help="额外跑引擎那套通道做对照（慢）")
    args = parser.parse_args()

    if sys.platform == "win32":
        C.disable()

    bootstrap()

    if args.delay > 0:
        for i in range(args.delay, 0, -1):
            print(f"  {i}s —— 切回游戏", end="\r", flush=True)
            time.sleep(1.0)
        print(" " * 30, end="\r")

    img = _probe_ocr_path(args.title)
    if img is not None and args.save:
        print(f"  已存：{_save(img, 'ocrpath')}")

    _probe_chain_routes(args.title)

    if args.engine:
        _probe_engine_path(args.title)

    print()
    print(line())
    print(f"{C.BOLD}怎么判读{C.OFF}")
    print(f"  · 第 1 段 dead> {DEAD_FRAC * 100:.0f}%  → OCR 路径抓的是残帧，")
    print(f"    根因在 screen_capture 的降级链，引擎那边看不到（两套代码）。")
    print(f"  · 第 1 段 luma< {LUMA_BLACK:.0f} 且 ok=True → 守卫有洞（走 foreground 时）。")
    print(f"  · 第 2 段哪一路 dead 高 → 就是那一路在交黑帧，删掉它或把 WGC 提到链首。")
    print(f"  · 第 2 段全好但第 1 段坏 → 目标窗口没找对，capture_active_frame 走了 foreground。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
