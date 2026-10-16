"""最小可跑：只截图 + 显示。

    python -m autopilot.tools.capture_demo --title 原神

按 ESC 或 Q 退出。跑通它说明截图链路 OK，可以进下一步。

常用：
    --save out/frame.png      存一张当前帧（用于裁模板图）
    --interval 0              不限速（默认 0 = 尽可能快）
    --no-window               不开预览窗口，只统计性能（无显示环境用）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description="截图演示")
    ap.add_argument("--title", default="原神", help="目标窗口标题片段")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="每帧最小间隔秒数，0 = 不限速")
    ap.add_argument("--save", default="", help="保存一张当前帧到该路径")
    ap.add_argument("--no-window", action="store_true", help="不开预览窗口")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    import cv2  # noqa: PLC0415

    from autopilot.perception.capturer import CaptureError, Capturer  # noqa: PLC0415

    try:
        cap = Capturer(args.title)
        cap.open()
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 截图器初始化失败{C.OFF}\n  {exc}")
        print(f"  {C.DIM}先跑 python -m autopilot.tools.check_env --title {args.title}{C.OFF}")
        return 1

    print(f"\n{C.BOLD}截图演示{C.OFF}  通道={cap.get_active_channel().value}  "
          f"窗口={cap.window_rect}")
    print(f"{C.DIM}ESC 或 Q 退出{C.OFF}\n")

    samples: list[float] = []
    saved = not args.save
    last = time.perf_counter()
    n = 0

    try:
        with cap:
            while True:
                t0 = time.perf_counter()
                try:
                    frame = cap.grab()
                except CaptureError as exc:
                    print(f"{C.ERR}截图失败: {exc}{C.OFF}")
                    break

                dt = (time.perf_counter() - t0) * 1000
                samples.append(dt)
                n += 1

                if not saved:
                    p = Path(args.save)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    # ⚠️ 不用 cv2.imwrite：--save 是**用户给的路径**，
                    #    含中文时它会静默存成乱码名（详见 image_io 模块）。
                    from autopilot.perception.image_io import imwrite_unicode

                    imwrite_unicode(p, frame.image)
                    print(f"{C.OK}✓{C.OFF} 已保存 {p}  ({frame.width}x{frame.height})")
                    saved = True
                    if args.no_window:
                        break

                if not args.no_window:
                    canvas = frame.image
                    hud = [
                        f"seq={frame.frame_seq}  {dt:.1f}ms  "
                        f"{frame.width}x{frame.height}  {frame.channel.value}",
                    ]
                    for i, t in enumerate(hud):
                        cv2.putText(canvas, t, (8, 22 + i * 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow("capture_demo", canvas)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                        break
                else:
                    if n % 60 == 0:
                        print(f"  {n} 帧  平均 {sum(samples[-60:]) / len(samples[-60:]):.1f}ms")

                elapsed = time.perf_counter() - last
                if args.interval > 0 and elapsed < args.interval:
                    time.sleep(args.interval - elapsed)
                last = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_window:
            cv2.destroyAllWindows()

    if samples:
        s = sorted(samples)
        print(f"\n{C.BOLD}统计{C.OFF}  共 {n} 帧")
        print(f"  P50  {s[len(s)//2]:.1f} ms")
        print(f"  P95  {s[int(len(s)*0.95)]:.1f} ms")
        print(f"  max  {s[-1]:.1f} ms")
        verdict = "✓ 满足感知层预算" if s[int(len(s) * 0.95)] < 30 else (
            "⚠ 偏慢，建议装 dxcam" if s[int(len(s) * 0.95)] < 80
            else "✗ 太慢，tick 会超时"
        )
        print(f"  {C.DIM}{verdict}{C.OFF}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
