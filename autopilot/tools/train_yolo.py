"""训练 YOLO11n 并导出 ONNX。

    python -m autopilot.tools.train_yolo --data D:/NEKO/yolo_data

⚠️ **许可证说明（重要）**：
    本脚本 **import ultralytics**（AGPL-3.0）。这是**允许的**，因为：
      · 它是开发/训练工具，**不分发**，不进 N.E.K.O. 运行时
      · 它的**产出物是 ONNX 文件**（权重数据，不受代码许可证传染）
      · 运行时 perception/yolo.py 只用 onnxruntime（MIT）加载 ONNX
    前提：**绝不要**把 ultralytics 作为依赖写进 requirements-autopilot.txt，
    绝不要在运行时 import 它。

产出：
    runs/detect/train/weights/best.pt
    autopilot/perception/models/yolo11n.onnx   ← 复制到这里供运行时用

训练前建议先跑：python -m autopilot.tools.check_dataset --data <目录>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import AUTOPILOT_ROOT, C, line, setup_logging  # noqa: E402
else:
    from ._bootstrap import AUTOPILOT_ROOT, C, line, setup_logging

MODEL_OUT_DIR = AUTOPILOT_ROOT / "perception" / "models"


def main() -> int:
    ap = argparse.ArgumentParser(description="训练 YOLO11n 并导出 ONNX")
    ap.add_argument("--data", required=True, help="data.yaml 路径或含它的目录")
    ap.add_argument("--weights", default="yolo11n.pt", help="初始权重")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16, help="5070 12G 建议 16~32")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="0=GPU, cpu=CPU")
    ap.add_argument("--patience", type=int, default=20, help="早停 patience")
    ap.add_argument("--project", default="", help="输出目录（默认 autopilot/runs）")
    ap.add_argument("--name", default="detect", help="实验名")
    ap.add_argument("--resume", action="store_true", help="从上次中断继续")
    ap.add_argument("--no-export", action="store_true", help="只训练不导出 ONNX")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup_logging()
    if not sys.stdout.isatty():
        C.disable()

    data = Path(args.data)
    if data.is_dir():
        data = data / "data.yaml"
    if not data.exists():
        print(f"{C.ERR}✗ 找不到 data.yaml: {data}{C.OFF}")
        print(f"  需要这样的结构：\n"
              f"    root/\n      data.yaml\n      images/train ...\n      labels/train ...")
        return 1

    print(f"\n{C.BOLD}训练 YOLO11n → ONNX{C.OFF}")
    print(line("═"))
    print(f"  data      {data}")
    print(f"  weights   {args.weights}")
    print(f"  epochs    {args.epochs}   batch {args.batch}   imgsz {args.imgsz}   device {args.device}")
    print(line("═"))
    print(f"{C.WARN}提示{C.OFF} 训练前建议先跑 check_dataset，避免脏数据浪费一晚上。\n")

    try:
        from ultralytics import YOLO  # noqa: PLC0415 - 仅训练用，AGPL 不进运行时
    except ImportError:
        print(f"{C.ERR}✗ 未安装 ultralytics{C.OFF}")
        print(f"  pip install ultralytics")
        print(f"  {C.DIM}（仅训练需要；运行时不需要，见 requirements-autopilot.txt）{C.OFF}")
        return 1

    project = Path(args.project) if args.project else (AUTOPILOT_ROOT / "runs")
    project.mkdir(parents=True, exist_ok=True)

    try:
        model = YOLO(args.weights)
        if args.resume:
            last = project / args.name / "weights" / "last.pt"
            if not last.exists():
                print(f"{C.ERR}✗ --resume 但找不到 {last}{C.OFF}")
                return 1
            print(f"{C.DIM}从 {last} 继续{C.OFF}\n")
            model = YOLO(str(last))

        results = model.train(
            data=str(data),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            patience=args.patience,
            project=str(project),
            name=args.name,
            workers=args.workers,
            seed=args.seed,
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            # 小数据集的关键超参
            cos_lr=True,
            close_mosaic=10,
            degrees=5.0,        # 轻微旋转
            translate=0.1,
            scale=0.4,
            fliplr=0.5,         # ★ UI 左右翻转通常不合理，默认关；这里给 0.5 需自行确认
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.4,
            # 验证
            val=True,
            plots=True,
            verbose=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n{C.ERR}✗ 训练失败: {type(exc).__name__}: {exc}{C.OFF}")
        return 1

    run_dir = Path(getattr(results, "save_dir", project / args.name))
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"{C.ERR}✗ 训练结束但没找到 {best}{C.OFF}")
        return 1

    print(f"\n{C.OK}✓{C.OFF} 训练完成: {best}")
    if args.no_export:
        print(f"  {C.DIM}已跳过导出{C.OFF}")
        return 0

    # ---- 导出 ONNX ----
    print(f"\n{C.BOLD}导出 ONNX{C.OFF}")
    try:
        m = YOLO(str(best))
        out = m.export(format="onnx", opset=args.opset, simplify=True, dynamic=False)
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ ONNX 导出失败: {exc}{C.OFF}")
        print(f"  可以单独跑：python -m autopilot.tools.convert_to_onnx --weights {best}")
        return 1

    src = Path(out)
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODEL_OUT_DIR / "yolo11n.onnx"
    shutil.copy2(src, dst)
    size_mb = dst.stat().st_size / 1024 / 1024

    print(f"\n{C.OK}✓ ONNX 已就位{C.OFF}")
    print(f"  {src.name}  ({size_mb:.1f} MB)")
    print(f"  → {dst}")
    print(line("═"))
    print("下一步：")
    print(f"  1. 在 defaults.toml 里设 perception.yolo.enabled = true")
    print(f"     并确认 class_names 与 data.yaml 的 names 顺序**完全一致**")
    print(f"  2. 验证：")
    print(f"     python -m autopilot.tools.check_env --title 原神")
    print(f"     python -m autopilot.tools.autopilot_demo --title 原神")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
