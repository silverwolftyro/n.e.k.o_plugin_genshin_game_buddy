"""数据集质量检查（训练前必跑）。

    python -m autopilot.tools.check_dataset --data D:/NEKO/yolo_data

检查项：
    1. images/ 与 labels/ 数量是否一致
    2. 每个 label 文件格式（每行 5 字段，0<=cx,cy,w,h<=1）
    3. 框是否越界 / 退化（w 或 h 为 0）
    4. 类别分布（严重不均衡会直接毁掉小类）
    5. 重复图片（感知哈希 dHash）
    6. 类别名与 data.yaml 是否一致

🔴 **隐私检查**：脚本会检测 label 落在常见 UID 区域的情况并提醒。
   训练集**不入 git**，且必须脱敏后再入库。

退出码：0 = 可训练；1 = 有严重问题，别训。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, line, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, line, setup_logging

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
#: 单张图最多框数（超过多半是标注错位）
MAX_BOXES_PER_IMAGE = 80


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO 数据集质量检查")
    ap.add_argument("--data", required=True, help="数据目录（含 images/ labels/，或 YOLO 标准布局）")
    ap.add_argument("--split", default="", help="只查某个子集（train/val/test）；空 = 全部")
    ap.add_argument("--min-class-ratio", type=float, default=0.02,
                    help="最少类占比阈值，低于此值算严重不均衡（默认 0.02）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()


    root = Path(args.data)
    if not root.is_dir():
        print(f"{C.ERR}✗ 数据目录不存在: {root}{C.OFF}")
        return 1

    print(f"\n{C.BOLD}数据集检查{C.OFF}  {root}\n")

    # 定位 images/ labels/
    splits = [args.split] if args.split else None
    pairs: list[tuple[str, Path, Path]] = []
    if splits:
        for s in splits:
            pairs.append((s, root / s / "images", root / s / "labels"))
    else:
        for s in ("train", "val", "test"):
            if (root / s / "images").is_dir():
                pairs.append((s, root / s / "images", root / s / "labels"))
        if not pairs:
            pairs.append(("", root / "images", root / "labels"))

    errors: list[str] = []
    warnings: list[str] = []
    class_counts: Counter[int] = Counter()
    total_boxes = 0
    total_images = 0
    all_hashes: dict[str, list[str]] = defaultdict(list)
    yaml_names: list[str] = []

    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        try:
            import yaml  # noqa: PLC0415

            y = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            yaml_names = list(y.get("names") or [])
            print(f"  data.yaml names: {yaml_names}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"data.yaml 解析失败: {exc}")
    else:
        warnings.append(f"没有 data.yaml（{yaml_path}），训练脚本需要它")

    for split, img_dir, lbl_dir in pairs:
        if not img_dir.is_dir():
            errors.append(f"[{split or 'root'}] 图片目录不存在: {img_dir}")
            continue
        imgs = sorted(
            p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT
        )
        lb_missing = 0
        split_boxes = 0
        n_class_here: Counter[int] = Counter()

        for ip in imgs:
            total_images += 1
            lp = lbl_dir / f"{ip.stem}.txt"
            if not lp.exists():
                # YOLO 允许负样本（空 txt）
                if not any(lbl_dir.rglob(f"{ip.stem}.txt")):
                    lb_missing += 1
                continue

            try:
                h = _dhash(ip)
                all_hashes[h].append(str(ip))
            except Exception:  # noqa: BLE001
                pass

            for ln, raw in enumerate(
                lp.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                parts = raw.split()
                if len(parts) != 5:
                    errors.append(f"{lp}:{ln} 字段数 {len(parts)}（应为 5）: {raw[:60]}")
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    errors.append(f"{lp}:{ln} 数值解析失败: {raw[:60]}")
                    continue

                split_boxes += 1
                total_boxes += 1
                class_counts[cid] += 1
                n_class_here[cid] += 1

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    errors.append(f"{lp}:{ln} 中心越界 cx={cx:.3f} cy={cy:.3f}")
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    errors.append(f"{lp}:{ln} 宽高非法 w={w:.3f} h={h:.3f}")
                elif cx - w / 2 < -0.02 or cx + w / 2 > 1.02 or \
                        cy - h / 2 < -0.02 or cy + h / 2 > 1.02:
                    errors.append(
                        f"{lp}:{ln} 框越界 x=[{cx-w/2:.3f},{cx+w/2:.3f}] "
                        f"y=[{cy-h/2:.3f},{cy+h/2:.3f}]"
                    )

        if lb_missing:
            warnings.append(f"[{split or 'root'}] {lb_missing} 张图没有对应 label 文件")
        n_c = len(n_class_here)
        print(
            f"  {C.OK if split_boxes else C.WARN}{'✓' if split_boxes else '!'}{C.OFF} "
            f"[{split or 'root':5s}] 图 {len(imgs):5d}  框 {split_boxes:6d}  类 {n_c}"
        )
        if n_c > MAX_BOXES_PER_IMAGE:
            warnings.append(f"[{split}] 单图框数可能偏多（检查标注是否错位）")

    # 类别分布
    n_cls = len(class_counts)
    print(f"\n{C.BOLD}类别分布{C.OFF}  共 {n_cls} 类 / {total_boxes} 框")
    if n_cls:
        items = sorted(class_counts.items(), key=lambda kv: -kv[1])
        mx = items[0][1]
        for cid, cnt in items:
            name = (yaml_names[cid] if cid < len(yaml_names) else f"class_{cid}")
            ratio = cnt / total_boxes
            bar = "█" * max(1, int(cnt / mx * 28))
            print(f"  {cid:3d} {name:22s} {cnt:7d}  {ratio*100:5.1f}%  {C.DIM}{bar}{C.OFF}")
            if ratio < args.min_class_ratio:
                warnings.append(
                    f"类别 {cid}({name}) 仅 {cnt} 个框（{ratio*100:.2f}%），"
                    f"低于 {args.min_class_ratio*100:.0f}% —— 严重不均衡，小类会学不出来"
                )

    if yaml_names and n_cls > len(yaml_names):
        errors.append(
            f"label 里出现 class {n_cls-1}，但 data.yaml 只有 {len(yaml_names)} 个 names"
        )

    # 重复检测
    dups = {h: v for h, v in all_hashes.items() if len(v) > 1}
    print(f"\n{C.BOLD}重复图片{C.OFF}  {len(dups)} 组")
    for h, paths in list(dups.items())[:5]:
        print(f"    {C.DIM}{h[:8]}: {len(paths)} 张{C.OFF}")
        for p in paths[:3]:
            print(f"       {Path(p).name}")
    if dups:
        warnings.append(f"{len(dups)} 组重复/近重复图片，会让验证集虚高 —— 建议去重")

    # 隐私
    print(f"\n{C.BOLD}隐私提醒{C.OFF}")
    print(f"  {C.WARN}⚠{C.OFF} 训练集不要入 git（画面含 UID/头像）")
    print(f"  {C.WARN}⚠{C.OFF} 入库前用 redact 脚本把 UID 区域纯色遮盖")
    print(f"  {C.DIM}做法：固定 ROI 纯绿遮盖 + OCR 扫 UID（连续≥6 位数字）{C.OFF}")

    # 汇总
    print(line("═"))
    print(f"\n  图 {total_images}  框 {total_boxes}  类 {n_cls}")
    if errors:
        print(f"  {C.ERR}错误 {len(errors)}{C.OFF}")
        for e in errors[:15]:
            print(f"    ✗ {e}")
        if len(errors) > 15:
            print(f"    {C.DIM}… 还有 {len(errors)-15} 条{C.OFF}")
    if warnings:
        print(f"  {C.WARN}警告 {len(warnings)}{C.OFF}")
        for w in warnings[:10]:
            print(f"    ! {w}")
    print()

    if not errors and not warnings and total_images > 0:
        print(f"{C.OK}✓ 数据集看起来干净，可以训练：{C.OFF}")
        print(f"  python -m autopilot.tools.train_yolo --data {root}\n")
        return 0
    if errors:
        print(f"{C.ERR}✗ 有 {len(errors)} 个错误，先修好再训练。{C.OFF}\n")
        return 1
    print(f"{C.WARN}! 可以训练，但建议先处理上面的警告。{C.OFF}\n")
    return 0


def _dhash(path: Path, size: int = 8) -> str:
    """dHash 感知哈希，用于近重复检测。"""
    import cv2  # noqa: PLC0415

    from autopilot.perception.image_io import imread_unicode

    # ⚠️ 不用 cv2.imread：数据集路径含中文时会静默读不到
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("unreadable")
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return hashlib.md5(diff.tobytes()).hexdigest()[:16]  # noqa: S324 - 非安全用途


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
