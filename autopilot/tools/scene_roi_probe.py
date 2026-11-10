"""场景 ROI 探针 —— 把判定树真正看的那块肉画出来并算分。

★ 为什么需要它
--------------
场景判定（``coord_scene_tree``）**只用血条 ROI 一个特征源**：
``edge_density`` 与 ``mean_s``。浮窗显示「场景: 未知」就是
``is_world=False``，往下只剩两条路：

1. ROI 框歪了（裁到别的地方去了）
2. ROI 框对，但那一块的内容不满足阈值

肉眼看浮窗分不出来。这个脚本**直接把框画在截图上**并算出两个
特征值 —— 框歪了一眼能看见，框对而分数低一眼也能看见。

★ 用法
------
::

    python autopilot/tools/scene_roi_probe.py
    python autopilot/tools/scene_roi_probe.py --image _runtime/capdiag_wgc.png
    python autopilot/tools/scene_roi_probe.py --rois D:\\workspace\\N.E.K.O\\NEKO-genshin\\recognition\\configs\\rois.toml

默认输出 ``autopilot/_runtime/scene_roi_debug.png``。

★ 框的颜色约定
--------------
* **红** ``minimap``   —— 小地图
* **绿** ``hp_player`` —— 血条（★ 唯一真正参与判定的 ROI）
* **蓝** ``eb_main``   —— 技能栏（元素爆发）

另外三个角色框（``key_e`` / ``key_z`` / 派蒙）用细框标出，
不参与判定，纯粹给人看方位。

⚠️ 本脚本**只读不写业务状态**：不改 rois.toml，不动判定树。
阈值对比照抄 ``coord_scene_tree`` 的常量，不重新训练。
"""

from __future__ import annotations

import sys

# ---- console encoding ----------------------------------------------
# chcp 65001 只改代码页；Python 仍按自己的编码写字节
# （中文 Windows 上是 cp936/GBK），中文会变乱码。两个流都强制 UTF-8。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):   # 被重定向到 StringIO -> 没这个方法
        _s.reconfigure(encoding="utf-8", errors="replace")

import argparse
import tomllib
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import AUTOPILOT_ROOT, C, bootstrap, line, setup_logging

bootstrap()

# bootstrap() 之后 autopilot 包才可导入，所以这行必须放在它后面
from autopilot.perception.image_io import imread_unicode, imwrite_unicode  # noqa: E402

#: 默认截图：WGC 通道（保真度最高）
DEFAULT_IMAGE = AUTOPILOT_ROOT / "_runtime" / "capdiag_wgc.png"
#: 默认 ROI 配置
DEFAULT_ROIS = Path(r"D:\workspace\N.E.K.O\NEKO-genshin\recognition\configs\rois.toml")
#: 默认输出图
DEFAULT_OUT = AUTOPILOT_ROOT / "_runtime" / "scene_roi_debug.png"

#: 参与判定树的 ROI 及其判定特征（照抄 coord_scene_tree.py）
DECISION_ROI = "hp_player"

#: 判定阈值 —— 与 coord_scene_tree.EDGE_DENSITY_MAX / MEAN_S_MAX 一致
EDGE_DENSITY_MAX = 0.1177
MEAN_S_MAX = 104.3326

#: 画框样式：名字 -> (BGR 颜色, 线宽, 中文标签)
STYLES = {
    "minimap":     ((0, 0, 255),   3, "小地图 minimap"),
    "hp_player":   ((0, 255, 0),   3, "血条 hp_player ★判定"),
    "eb_main":     ((255, 160, 0), 3, "技能栏 eb_main"),
}
#: 不参与判定的辅助框（细框，只标方位）
AUX_STYLES = {
    "key_e":         (0, 215, 255),
    "key_z":         (0, 215, 255),
    "paimon_button": (200, 200, 200),
    "avatar":        (200, 200, 200),
}
#: 圈出小地图圆盘（照抄 rois.toml 的 minimap_circle 换算）
CIRCLE_KEY = "minimap_circle"


def to_px(roi: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    """归一化 ROI -> 像素半开区间 (px0, py0, px1, py1)。"""
    x0, y0, x1, y1 = roi
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def describe_px(roi: list[float], w: int, h: int) -> str:
    px0, py0, px1, py1 = to_px(roi, w, h)
    return (f"norm[{roi[0]:.4f},{roi[1]:.4f},{roi[2]:.4f},{roi[3]:.4f}] "
            f"px=({px0},{py0})-({px1},{py1}) "
            f"{px1 - px0}x{py1 - py0}px "
            f"@右侧余{100 * (1 - px1 / w):.2f}% 底部余{100 * (1 - py1 / h):.2f}%")


def roi_stats(name: str, crop: np.ndarray) -> dict:
    """按判定树的方式算特征，外加一组给人看的描述统计。"""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    edge_density = float((cv2.Canny(g, 60, 160) > 0).mean())
    return {
        "name": name,
        "w": crop.shape[1],
        "h": crop.shape[0],
        "edge_density": edge_density,
        "mean_s": float(s.mean()),
        "mean_v": float(v.mean()),
        "black_ratio": float((v < 30).mean()),
        "bright_ratio": float((v > 200).mean()),
        "mean_bgr": [float(crop[:, :, i].mean()) for i in range(3)],
    }


def verdict_hp(st: dict) -> tuple[str, str]:
    """血条 ROI 的判定结论。返回 (颜色, 文字)。"""
    if st["edge_density"] <= EDGE_DENSITY_MAX:
        return C.ERR, (
            f"edge_density({st['edge_density']:.4f}) <= {EDGE_DENSITY_MAX}"
            f"  ->  非大世界（UNKNOWN）")
    if st["mean_s"] <= MEAN_S_MAX:
        return C.ERR, (
            f"edge ok({st['edge_density']:.4f}) 但 "
            f"mean_s({st['mean_s']:.2f}) <= {MEAN_S_MAX}"
            f"  ->  非大世界（UNKNOWN）")
    return C.OK, (
        f"edge({st['edge_density']:.4f}) > {EDGE_DENSITY_MAX} 且 "
        f"mean_s({st['mean_s']:.2f}) > {MEAN_S_MAX}  ->  大世界（WORLD）")


def draw(img: np.ndarray, roi: list[float], color, thick: int,
         label: str, w: int, h: int) -> None:
    px0, py0, px1, py1 = to_px(roi, w, h)
    cv2.rectangle(img, (px0, py0), (px1, py1), color, thick)
    # 标签压在框的上沿；顶到屏幕上边就挪到框内侧
    ty = py0 - 6 if py0 > 18 else py0 + 18
    cv2.putText(img, label, (px0, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把场景判定用的 ROI 画到截图上并算分")
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                    help=f"截图路径（默认 {DEFAULT_IMAGE.name}）")
    ap.add_argument("--rois", type=Path, default=DEFAULT_ROIS,
                    help="rois.toml 路径")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"输出图路径（默认 {DEFAULT_OUT.name}）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(args.verbose)

    if not args.image.is_file():
        print(f"{C.ERR}✗ 截图不存在: {args.image}{C.OFF}")
        print(f"  先跑 capture_diagnose.py 生成一张。")
        return 2
    if not args.rois.is_file():
        print(f"{C.ERR}✗ ROI 配置不存在: {args.rois}{C.OFF}")
        return 2

    # ⚠️ 不用 cv2.imread：它在 Windows 上按 ANSI 码页解路径，
    #    用户传一个中文名的截图就会静默读不到（详见 image_io 模块）。
    img = imread_unicode(args.image)
    if img is None:
        print(f"{C.ERR}✗ 读不出图像（文件损坏？）: {args.image}{C.OFF}")
        return 2
    h, w = img.shape[:2]

    with open(args.rois, "rb") as fh:
        cfg = tomllib.load(fh)
    rois: dict = cfg["rois"]
    meta: dict = cfg.get("meta", {})

    ref = meta.get("reference_size", [])
    tgt = meta.get("target_resolution", [])
    ref_ar = f"{ref[0] / ref[1]:.4f}" if len(ref) == 2 else "?"
    cur_ar = f"{w / h:.4f}"
    print(line("="))
    print(f"{C.BOLD}场景 ROI 探针{C.OFF}")
    print(line("="))
    print(f"  截图     {args.image}")
    print(f"  分辨率   {w}x{h}  宽高比 {cur_ar}")
    if len(ref) == 2:
        same_ar = abs(float(ref_ar) - float(cur_ar)) < 0.01
        flag = f"{C.OK}一致{C.OFF}" if same_ar else f"{C.ERR}不一致{C.OFF}"
        print(f"  基准     {ref[0]}x{ref[1]}  宽高比 {ref_ar}")
        print(f"  配置目标 {tgt}  宽高比 {flag}")
        if ref_ar != cur_ar:
            print(f"  {C.WARN}⚠️ 宽高比不一致：归一化 ROI 会横向/纵向拉伸，"
                  f"框可能落在错的地方{C.OFF}")
        else:
            print(f"  {C.OK}✓ 宽高比一致 —— 归一化 ROI 直接缩放即正确，"
                  f"不是分辨率不匹配的锅{C.OFF}")
    print(line("-"))

    canvas = img.copy()

    # ── 辅助细框（不参与判定，先画免得被主框盖住） ──────────────
    for name, color in AUX_STYLES.items():
        if name in rois:
            draw(canvas, rois[name], color, 1, name, w, h)

    # ── 三个主 ROI ──────────────────────────────────────────────
    stats: dict[str, dict] = {}
    for name, (color, thick, label) in STYLES.items():
        if name not in rois:
            print(f"{C.WARN}△ rois.toml 里没有 {name}，跳过{C.OFF}")
            continue
        draw(canvas, rois[name], color, thick, label, w, h)
        px0, py0, px1, py1 = to_px(rois[name], w, h)
        if px1 - px0 < 2 or py1 - py0 < 2:
            print(f"{C.ERR}✗ {name} ROI 退化（{px1-px0}x{py1-py0}px），"
                  f"画面太小或配置坏了{C.OFF}")
            continue
        # ★ 取样必须用**原图 img**，不能用 canvas ——
        #   canvas 上已经画了框，框线本身会抬高 edge_density 和 mean_s。
        stats[name] = roi_stats(name, img[py0:py1, px0:px1])

    # ── 小地图圆盘 ──────────────────────────────────────────────
    circ = rois.get(CIRCLE_KEY)
    if isinstance(circ, dict) and "cx" in circ:
        cx = int(circ["cx"] * w)
        cy = int(circ["cy"] * h)
        # 半径**按宽度**归一（配置里的约定），不是按对角线
        rn = circ.get("r_norm_by_width")
        r = int(rn * w) if rn else int(circ.get("r_norm", 0.05) * w)
        cv2.circle(canvas, (cx, cy), r, (0, 0, 255), 2)
        cv2.putText(canvas, "minimap circle", (cx + r + 4, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)

    # ── 顶部信息条（不盖住任何 ROI） ────────────────────────────
    verdict_txt = "未知"
    if DECISION_ROI in stats:
        color, verdict_txt = verdict_hp(stats[DECISION_ROI])
    banner = f"scene={verdict_txt}"
    cv2.rectangle(canvas, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(canvas, banner, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "R=minimap  G=hp_player(decision)  B=eb_main",
                (8 + cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX,
                                     0.6, 1)[0][0] + 16, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                cv2.LINE_AA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ 不用 cv2.imwrite：--out 含中文时会静默存成乱码名
    if not imwrite_unicode(args.out, canvas):
        print(f"{C.ERR}✗ 写不出图: {args.out}{C.OFF}")
        return 2

    # ── 打印每框明细 ────────────────────────────────────────────
    print(f"{C.BOLD}ROI 明细{C.OFF}")
    for name in STYLES:
        if name not in rois:
            continue
        print(line("-"))
        mark = f"{C.BOLD}★ 判定特征{C.OFF}" if name == DECISION_ROI else ""
        print(f"[{name}] {mark}")
        print(f"  {describe_px(rois[name], w, h)}")
        st = stats.get(name)
        if not st:
            print(f"  {C.ERR}(ROI 退化，无法取样){C.OFF}")
            continue
        print(f"  尺寸     {st['w']}x{st['h']}px"
              f"  均值BGR=({st['mean_bgr'][0]:.1f},"
              f"{st['mean_bgr'][1]:.1f},{st['mean_bgr'][2]:.1f})")
        print(f"  edge_density {st['edge_density']:.4f}"
              f"   阈值 {EDGE_DENSITY_MAX}")
        print(f"  mean_s       {st['mean_s']:.2f}"
              f"       阈值 {MEAN_S_MAX}")
        print(f"  mean_v       {st['mean_v']:.2f}"
              f"   全黑像素占比 {st['black_ratio']*100:.1f}%"
              f"   高亮占比 {st['bright_ratio']*100:.1f}%")
        if name == DECISION_ROI:
            color, txt = verdict_hp(st)
            print(f"  {C.BOLD}判定 -> {color}{txt}{C.OFF}")
    if isinstance(circ, dict) and "cx" in circ:
        print(line("-"))
        print("[minimap_circle]")
        print(f"  圆心 ({circ['cx']:.5f}, {circ['cy']:.5f}) "
              f"-> px ({int(circ['cx']*w)}, {int(circ['cy']*h)})")
        print(f"  半径 {circ.get('r_norm_by_width')} x 宽 = "
              f"{int(circ.get('r_norm_by_width', 0)*w)}px")

    print(line("="))
    print(f"{C.BOLD}结论{C.OFF}  {verdict_txt}")
    if DECISION_ROI in stats:
        st = stats[DECISION_ROI]
        if st["black_ratio"] > 0.9:
            print(f"  {C.ERR}血条 ROI {st['black_ratio']*100:.0f}% 全黑 —— "
                  f"HUD 淡出或框到了纯黑区（已知边界情况 #1）{C.OFF}")
        elif st["edge_density"] <= EDGE_DENSITY_MAX:
            print(f"  血条 ROI 边缘太少（{st['edge_density']:.4f}）。"
                  f"若框画出来明显不在血条上 -> ROI 偏移；"
                  f"若框确实在血条上 -> 该分辨率下血条渲染更细，"
                  f"需要重标阈值。")
        elif st["mean_s"] <= MEAN_S_MAX:
            print(f"  边缘够但饱和度低（{st['mean_s']:.2f}）。"
                  f"血条区域偏灰/偏白 -> 满血条本身没颜色，"
                  f"或框到了血条旁边的白底。")
    print(f"\n已写出: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
