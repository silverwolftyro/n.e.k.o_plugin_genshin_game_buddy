"""单独把训练好的权重导出成 ONNX。

    python -m autopilot.tools.convert_to_onnx --weights runs/detect/train/weights/best.pt

train_yolo.py 已经会自动导出；本工具用于：
    · 换 opset（onnxruntime 版本低时降 opset）
    · 只重新导出、不想重训
    · 导出到自定义路径

⚠️ 同样 import ultralytics（AGPL），**仅开发工具，不进运行时**。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import AUTOPILOT_ROOT, C, setup_logging  # noqa: E402
else:
    from ._bootstrap import AUTOPILOT_ROOT, C, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 ONNX")
    ap.add_argument("--weights", required=True, help=".pt 权重路径")
    ap.add_argument("--out", default="", help="输出路径（默认 autopilot/perception/models/yolo11n.onnx）")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--dynamic", action="store_true", help="动态尺寸（推理端需自行处理，一般不用）")
    ap.add_argument("--half", action="store_true", help="FP16 导出（需 GPU）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    w = Path(args.weights)
    if not w.exists():
        print(f"{C.ERR}✗ 权重不存在: {w}{C.OFF}")
        return 1

    try:
        from ultralytics import YOLO  # noqa: PLC0415 - 仅工具用
    except ImportError:
        print(f"{C.ERR}✗ 未安装 ultralytics{C.OFF}\n  pip install ultralytics")
        return 1

    print(f"\n{C.BOLD}导出 ONNX{C.OFF}  {w.name}")
    print(f"  opset={args.opset}  imgsz={args.imgsz}  simplify={not args.no_simplify}  "
          f"dynamic={args.dynamic}  half={args.half}\n")

    try:
        m = YOLO(str(w))
        out = m.export(
            format="onnx",
            opset=args.opset,
            imgsz=args.imgsz,
            simplify=not args.no_simplify,
            dynamic=args.dynamic,
            half=args.half,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 导出失败: {type(exc).__name__}: {exc}{C.OFF}")
        return 1

    src = Path(out)
    dst = Path(args.out) if args.out else (AUTOPILOT_ROOT / "perception" / "models" / "yolo11n.onnx")
    if src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"\n{C.OK}✓{C.OFF} {dst}  ({size_mb:.1f} MB)")

    # 冒烟：能否被 onnxruntime 加载
    print(f"{C.DIM}冒烟测试 onnxruntime 加载...{C.OFF}")
    try:
        import onnxruntime as ort  # noqa: PLC0415

        s = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        i = s.get_inputs()[0]
        print(f"  输入: {i.name} shape={i.shape}")
        print(f"  输出: {[o.name for o in s.get_outputs()]}")
        import numpy as np  # noqa: PLC0415

        sz = args.imgsz
        s.run(None, {i.name: np.zeros((1, 3, sz, sz), dtype=np.float32)})
        print(f"  {C.OK}✓ 推理通过{C.OFF}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {C.ERR}✗ 加载/推理失败: {exc}{C.OFF}")
        print(f"  试试降低 opset：--opset 11")
        return 1

    print(f"\n下一步：在 defaults.toml 设 perception.yolo.enabled = true 并填 class_names\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
