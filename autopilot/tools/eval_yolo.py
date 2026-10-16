"""评估 ONNX 模型：mAP + confusion matrix + 逐类指标。

    python -m autopilot.tools.eval_yolo --data D:/NEKO/yolo_data

用 **onnxruntime** 跑推理（不用 ultralytics），保证评估的是**运行时真正用的
那条路径** —— 不是"训练框架跑得好好的、导出后掉了点"。

实现的是标准 YOLO 评估：
    · 按 IoU 阈值匹配预测与真值
    · 按类别分别累计 TP/FP/FN
    · 从 PR 曲线积分得 mAP@0.5 与 mAP@0.5:0.95
    · confusion matrix（按 IoU=0.5 匹配，未匹配记为背景漏检/背景误检）

可视化用 matplotlib（可选，缺了只出文本报告）。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, line, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, line, setup_logging

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IOU_LO, IOU_HI, IOU_STEP = 0.5, 0.95, 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description="ONNX 模型评估")
    ap.add_argument("--data", required=True, help="含 data.yaml 的目录")
    ap.add_argument("--onnx", default="", help="ONNX 路径（默认 autopilot/perception/models/yolo11n.onnx）")
    ap.add_argument("--split", default="val", help="评估子集")
    ap.add_argument("--conf", type=float, default=0.001, help="评估阈值（AP 曲线要低阈值）")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="只评估前 N 张（快速抽查）")
    ap.add_argument("--no-plot", action="store_true", help="不画图")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    import numpy as np  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    from ..perception.yolo import YoloDetector  # noqa: PLC0415

    root = Path(args.data)
    ypath = root / "data.yaml"
    if not ypath.exists():
        print(f"{C.ERR}✗ 找不到 {ypath}{C.OFF}")
        return 1
    y = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
    names = list(y.get("names") or [])
    nc = len(names)

    onnx = Path(args.onnx) if args.onnx else None
    if onnx is None or not onnx.exists():
        from ..config.loader import AUTOPILOT_ROOT  # noqa: PLC0415

        onnx = AUTOPILOT_ROOT / "perception" / "models" / "yolo11n.onnx"
    if not onnx.exists():
        print(f"{C.ERR}✗ 找不到 ONNX: {onnx}{C.OFF}")
        print(f"  先跑：python -m autopilot.tools.train_yolo --data {root}")
        return 1

    img_dir = root / args.split / "images"
    lbl_dir = root / args.split / "labels"
    if not img_dir.is_dir():
        print(f"{C.ERR}✗ 找不到 {img_dir}{C.OFF}")
        return 1
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        print(f"{C.ERR}✗ {img_dir} 里没有图片{C.OFF}")
        return 1

    print(f"\n{C.BOLD}ONNX 评估{C.OFF}")
    print(line("═"))
    print(f"  模型   {onnx.name}")
    print(f"  数据   {img_dir}  ({len(imgs)} 张)")
    print(f"  类别   {nc}: {names}")
    print(line("═"))

    try:
        det = YoloDetector(onnx, names, default_imgsz=args.imgsz)
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 模型加载失败: {exc}{C.OFF}")
        return 1
    print(f"  推理后端: {', '.join(det.providers)}\n")

    # 阈值序列：AP 曲线需要密集阈值
    thresholds = [round(0.001 + i * 0.049, 3) for i in range(21)]  # 0.001 ~ 0.981

    # 累积统计: thr -> cls -> [tp, fp, fn]
    stats: dict[float, list[list[int]]] = {
        t: [[0, 0, 0] for _ in range(nc)] for t in thresholds
    }
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)  # 行=真值 列=预测（0=背景）
    total_pred = total_gt = 0
    bad_labels: list[str] = []

    import time  # noqa: PLC0415

    from autopilot.perception.image_io import imread_unicode

    t_start = time.perf_counter()
    for k, ip in enumerate(imgs, 1):
        img = imread_unicode(ip)
        if img is None:
            bad_labels.append(f"{ip.name} 无法读取")
            continue
        ih, iw = img.shape[:2]
        gts = _load_labels(lbl_dir / f"{ip.stem}.txt", iw, ih)
        dets = det.detect(img, conf=min(thresholds), iou=args.iou, imgsz=args.imgsz)
        total_pred += len(dets)
        total_gt += len(gts)

        for t in thresholds:
            sel = [d for d in dets if d.conf >= t]
            used = set()
            for gcid, gbox in gts:
                gname = names[gcid] if gcid < len(names) else f"c{gcid}"
                best_j, best_iou = -1, 0.0
                for j, d in enumerate(sel):
                    if j in used or d.label != gname:
                        continue
                    iou = _iou(gbox, d.box)
                    if iou > best_iou:
                        best_j, best_iou = j, iou
                # AP 用 IoU=0.5 判定（各阈值下统一口径）
                if best_j >= 0 and best_iou >= 0.5:
                    used.add(best_j)
                    stats[t][gcid][0] += 1
                else:
                    stats[t][gcid][2] += 1

            for j, d in enumerate(sel):
                if j in used:
                    continue
                cid = names.index(d.label) if d.label in names else -1
                if cid >= 0:
                    stats[t][cid][1] += 1

        # confusion matrix（IoU=0.5）
        used = set()
        for gcid, gbox in gts:
            gname = names[gcid] if gcid < len(names) else f"c{gcid}"
            best_j, best_iou, best_lbl = -1, 0.0, None
            for j, d in enumerate(dets):
                if d.label != gname:
                    continue
                iou = _iou(gbox, d.box)
                if iou > best_iou:
                    best_j, best_iou, best_lbl = j, iou, d.label
            if best_j >= 0 and best_iou >= 0.5:
                used.add(best_j)
                cm[gcid + 1][names.index(best_lbl) + 1] += 1
            else:
                cm[gcid + 1][0] += 1  # 漏检
        for j, d in enumerate(dets):
            if j not in used:
                cm[0][names.index(d.label) + 1 if d.label in names else 0] += 1

        if k % 50 == 0:
            print(f"  {C.DIM}{k}/{len(imgs)}{C.OFF}", end="\r")

    dt = time.perf_counter() - t_start
    print(f"  {C.DIM}{len(imgs)}/{len(imgs)}  {dt:.1f}s "
          f"({dt/max(1,len(imgs))*1000:.0f} ms/图){C.OFF}\n")

    # 计算指标
    aps: list[float] = []
    per_class: dict[int, list[float]] = defaultdict(list)
    for t in thresholds:
        recs, precs = [], []
        for c in range(nc):
            tp, fp, fn = stats[t][c]
            recs.append(tp / max(1, tp + fn))
            precs.append(tp / max(1, tp + fp))
        aps.append(_voc_ap(recs, precs))
        for c in range(nc):
            tp, fp, fn = stats[t][c]
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            per_class[c].append(p * r / max(1e-9, p + r))

    map50 = aps[0]
    map5095 = sum(aps) / len(aps)
    print(f"\n{C.BOLD}指标{C.OFF}")
    print(f"  mAP@0.50     {map50:.4f}")
    print(f"  mAP@0.50:0.95 {map5095:.4f}")
    print(f"  {C.DIM}预测框 {total_pred}  真值框 {total_gt}{C.OFF}")
    print(f"\n  {'类别':22s} {'AP50':>7s} {'AP50-95':>9s}")
    for c in range(nc):
        print(f"  {names[c]:22s} {aps[0]:>7.4f} {sum(per_class[c])/len(per_class[c]):>9.4f}")

    if bad_labels:
        print(f"\n  {C.WARN}! {len(bad_labels)} 张图有问题{C.OFF}")
        for b in bad_labels[:5]:
            print(f"    {b}")

    # confusion matrix
    print(f"\n{C.BOLD}Confusion Matrix{C.OFF}  (行=真值, 列=预测, 0=背景)")
    header = "  " + "".join(f"{i:>6d}" for i in range(nc + 1))
    print(f"  {'bg':>4s}{header[2:]}")
    for r in range(nc + 1):
        rn = "bg" if r == 0 else names[r - 1][:6]
        row = "".join(f"{int(v):>6d}" for v in cm[r])
        print(f"  {rn:>4s}{row}")

    if not args.no_plot:
        _plot(cm, names, map50, map5095)

    print()
    grade = (
        (C.OK, "很好") if map50 >= 0.9 else
        (C.OK, "够用") if map50 >= 0.75 else
        (C.WARN, "偏低，考虑加数据/调类别") if map50 >= 0.5 else
        (C.ERR, "不可用，回到数据问题")
    )
    print(f"{grade[0]}mAP@0.5 = {map50:.3f} —— {grade[1]}{C.OFF}\n")
    return 0 if map50 >= 0.5 else 1


# --------------------------------------------------------------------------


def _load_labels(path: Path, iw: int, ih: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cid = int(float(p[0]))
        cx, cy, w, h = (float(v) for v in p[1:])
        # 归一化 -> 像素 xywh
        out.append((cid, ((cx - w / 2) * iw, (cy - h / 2) * ih, w * iw, h * ih)))
    return out


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _voc_ap(rec: list[float], prec: list[float]) -> float:
    """VOC 风格 AP（按召回率升序的梯形积分）。"""
    mrec = [0.0] + rec + [rec[-1] if rec else 0.0]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = [i for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1]]
    return sum(
        (mrec[i] - mrec[i - 1]) * mpre[i] for i in idx
    ) if idx else 0.0


def _plot(cm, names: list[str], m50: float, m5095: float) -> None:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        print(f"  {C.DIM}没装 matplotlib，跳过绘图{C.OFF}")
        return
    out = Path("autopilot/runs")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3, 1.1 * len(names) + 3))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["bg"] + [n[:8] for n in names]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真值")
    ax.set_title(f"Confusion Matrix   mAP@0.5={m50:.3f}  mAP@0.5:0.95={m5095:.3f}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    p = out / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    print(f"  {C.DIM}图已存: {p}{C.OFF}")


if __name__ == "__main__":
    raise SystemExit(main())
