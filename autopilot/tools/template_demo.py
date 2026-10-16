"""模板匹配演示：验证 ROI 表达式 + 模板匹配是否正确。

    python -m autopilot.tools.template_demo --title 原神

从 config/defaults.toml 读模板与 ROI 表达式，在实时截图上匹配，画框 + 分数。
**这是裁模板图的辅助工具**：看到框对了，就说明 ROI 表达式和阈值都对了。

裁图流程：
    1. python -m autopilot.tools.capture_demo --save work/shot.png
    2. 用 X-AnyLabeling / 画图工具裁出目标，存到
       autopilot/perception/templates/<name>.png
    3. 在 defaults.toml 的 [autopilot.templates.<name>] 里配好 path / threshold / roi
    4. 跑本工具验证

无模板时（默认就是）会提示怎么裁。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, line, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, line, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description="模板匹配演示")
    ap.add_argument("--title", default="原神")
    ap.add_argument("--config", default="", help="配置文件路径（默认 config/defaults.toml）")
    ap.add_argument("--only", default="", help="只测指定模板名，逗号分隔")
    ap.add_argument("--no-stable", action="store_true", help="关闭稳定计数，单帧即命中")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    import cv2  # noqa: PLC0415

    from autopilot.config.loader import AUTOPILOT_ROOT, load_config  # noqa: PLC0415
    from autopilot.perception.capturer import Capturer  # noqa: PLC0415
    from autopilot.perception.roi_expr import RoiExprEvaluator  # noqa: PLC0415
    from autopilot.perception.screen_region import region_of_frame  # noqa: PLC0415
    from autopilot.perception.template import TemplateMatcher  # noqa: PLC0415

    cfg_path = Path(args.config) if args.config else AUTOPILOT_ROOT / "config" / "defaults.toml"
    try:
        cfg = load_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 配置加载失败{C.OFF}\n  {exc}")
        return 1

    specs = cfg.template_specs()
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        specs = {k: v for k, v in specs.items() if k in want}

    missing = [n for n, s in specs.items() if not s.path.exists()]
    usable = {n: s for n, s in specs.items() if s.path.exists()}

    if not usable:
        print(f"\n{C.WARN}没有可用模板（配置在 {cfg_path}）{C.OFF}\n")
        print("模板目录: " + str(AUTOPILOT_ROOT / "perception" / "templates"))
        print(f"缺失: {', '.join(missing) or '（配置里就没写模板）'}\n")
        print(line("═"))
        print("裁图步骤：")
        print("  1. python -m autopilot.tools.capture_demo --title <窗口> --save work/shot.png")
        print("  2. 裁出目标，存到 autopilot/perception/templates/paimon_menu.png")
        print("  3. 在 defaults.toml 里配 [autopilot.templates.PaimonMenu] 的 path/roi")
        print("  4. 再跑本工具\n")
        return 0

    if missing:
        print(f"{C.WARN}跳过 {len(missing)} 个缺失模板: {', '.join(missing)}{C.OFF}")

    matcher = TemplateMatcher()
    ev = RoiExprEvaluator(cfg.perception.ref_size)
    try:
        roi_exprs = cfg.roi_expressions()
        var_vals = cfg.roi_var_values()
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ ROI 表达式求值失败{C.OFF}\n  {exc}")
        return 1

    print(f"\n{C.BOLD}模板匹配演示{C.OFF}  启用 {len(usable)} 个模板")
    for n, s in usable.items():
        roi = roi_exprs.get(n, "(全图)")
        print(f"  · {n:16s} 阈值={s.threshold:.2f}  稳定={s.stable_count}  roi={roi}")
    print(f"{C.DIM}ESC / Q 退出{C.OFF}\n")

    try:
        cap = Capturer(args.title)
        cap.open()
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 截图器初始化失败{C.OFF}\n  {exc}")
        return 1

    try:
        with cap:
            while True:
                frame = cap.grab()
                region = region_of_frame(frame.image, frame.rect[0], frame.rect[1])
                canvas = frame.image.copy()
                t0 = time.perf_counter()
                hits, misses = [], []

                for name, spec in usable.items():
                    box = None
                    src = roi_exprs.get(name)
                    if src:
                        if src.strip().startswith("@"):
                            src = cfg.resolve_region(src)
                        compiled = ev.compile(src, where=name)
                        box = ev.eval(compiled, region.width, region.height, vars=var_vals)
                    m = matcher.find(
                        region, spec, search_box=box, use_stability=not args.no_stable
                    )
                    if m is None:
                        acc = matcher.accumulator(name)
                        detail = ""
                        if acc and acc.max_consecutive_count:
                            detail = (f"  {C.WARN}最高连续 {acc.max_consecutive_count}"
                                      f"/{acc.required_count} 最低分 {acc.last_score:.2f}{C.OFF}")
                        misses.append(f"    ✗ {name}{detail}")
                        continue
                    rx, ry, rw, rh = m.box
                    cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 220, 220), 2)
                    cv2.putText(canvas, f"{name} {m.score:.2f}", (rx, max(14, ry - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)
                    if box:
                        cv2.rectangle(canvas, box[:2],
                                      (box[0] + box[2], box[1] + box[3]), (120, 120, 120), 1)
                    hits.append(f"    ✓ {name} score={m.score:.3f} box={m.box}")

                ms = (time.perf_counter() - t0) * 1000
                print(f"\r  seq={frame.frame_seq} {ms:.1f}ms  命中 {len(hits)}/{len(usable)}   ",
                      end="", flush=True)
                if misses:
                    print("\n" + "\n".join(misses), flush=True)
                for h in hits:
                    print(h, flush=True)

                cv2.imshow("template_demo", canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    print(f"\n{C.DIM}提示：长期 LOWER 说明阈值太高；长期无命中先查 ROI 表达式对不对{C.OFF}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
