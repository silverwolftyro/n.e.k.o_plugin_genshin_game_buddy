"""完整感知循环：截图 → YOLO → 模板 → 状态包 → 叠加层。

    python -m autopilot.tools.autopilot_demo --title 原神

**本工具不执行任何输入**（第 1 期没有执行层）。它做的是：
    · 实时叠加层显示识别框
    · 状态包以 JSONL 写入日志文件
    · 事件流打印到终端

用法：
    看屏幕上的框对不对 → 识别是否准
    看 JSONL → 状态包字段是否够用
    看事件 → 哪些识别在持续失败（事件带诊断数据）
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, setup_logging

_running = True


def _stop(*_a) -> None:
    global _running
    _running = False


def main() -> int:
    ap = argparse.ArgumentParser(description="完整感知循环（无执行层）")
    ap.add_argument("--title", default="原神")
    ap.add_argument("--config", default="")
    ap.add_argument("--out", default="", help="状态包 JSONL 输出路径")
    ap.add_argument("--interval", type=float, default=0.1, help="tick 间隔秒（默认 0.1 = 10Hz）")
    ap.add_argument("--max-frames", type=int, default=0, help="跑多少帧后退出（0 = 无限）")
    ap.add_argument("--no-yolo", action="store_true", help="强制关闭 YOLO")
    ap.add_argument("--no-overlay", action="store_true", help="不开叠加层")
    ap.add_argument("--scene", default="", help="强制场景（跳过 YOLO 粗判）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()
    signal.signal(signal.SIGINT, _stop)

    import cv2  # noqa: PLC0415

    from autopilot.config.loader import AUTOPILOT_ROOT, load_config  # noqa: PLC0415
    from autopilot.contracts.events import Severity  # noqa: PLC0415
    from autopilot.perception.builder import StateBuilder  # noqa: PLC0415
    from autopilot.perception.capturer import Capturer  # noqa: PLC0415
    from autopilot.perception.debug_overlay import DebugOverlay  # noqa: PLC0415
    from autopilot.perception.ocr import OcrEngine  # noqa: PLC0415
    from autopilot.perception.template import TemplateMatcher  # noqa: PLC0415
    from autopilot.perception.yolo import YoloDetector, YoloUnavailableError  # noqa: PLC0415
    from autopilot.state import WorldStateStore  # noqa: PLC0415

    cfg_path = Path(args.config) if args.config else AUTOPILOT_ROOT / "config" / "defaults.toml"
    try:
        cfg = load_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 配置校验失败{C.OFF}\n  {exc}")
        return 1

    # YOLO
    yolo = None
    if cfg.perception.yolo.enabled and not args.no_yolo:
        ypath = Path(cfg.perception.yolo.onnx_path)
        if not ypath.is_absolute():
            ypath = (AUTOPILOT_ROOT / ypath).resolve()
        try:
            yolo = YoloDetector(
                ypath, cfg.perception.yolo.class_names,
                default_imgsz=cfg.perception.yolo.imgsz,
            )
            yolo.warmup()
            print(f"{C.OK}✓{C.OFF} YOLO: {ypath.name} providers={yolo.providers}")
        except YoloUnavailableError as exc:
            print(f"{C.WARN}!{C.OFF} YOLO 不可用，退化为模板探测: {exc}")
            yolo = None
    else:
        print(f"{C.WARN}!{C.OFF} YOLO 未启用 —— 场景识别会退化为单模板探测")

    # OCR（复用 2.0 的 WinRT 引擎）
    ocr = None
    if cfg.perception.ocr.enabled:
        legacy = _load_legacy_ocr()
        ocr = OcrEngine(
            legacy_engine=legacy,
            fail_threshold=cfg.perception.ocr.fail_threshold,
            allow_rapid_fallback=cfg.perception.ocr.allow_rapid_fallback,
        )
        ocr.warmup()
        print(f"{C.OK}✓{C.OFF} OCR 引擎: {'2.0 WinRT' if legacy else '（无 2.0 引擎，用 RapidOCR/不可用）'}")

    try:
        cap = Capturer(
            cfg.capture.window_title_substr if args.title == "原神" else args.title,
            channel_order=[c.value for c in cfg.capture.channel_order],
            fail_threshold=cfg.capture.fail_threshold,
        )
        cap.open()
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 截图器初始化失败{C.OFF}\n  {exc}")
        return 1

    builder = StateBuilder(
        capturer=cap, yolo=yolo,
        template=TemplateMatcher(), ocr=ocr,
        game_id=cfg.game_id,
        scene_map=dict(cfg.scene_map) or None,
        scene_routes=dict(cfg.scene_routes) or None,
    )
    try:
        builder.set_templates(cfg.template_specs())
        builder.set_roi(cfg.roi_expressions(), cfg.roi_var_values())
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 模板/ROI 配置失败{C.OFF}\n  {exc}")
        return 1

    store = WorldStateStore()
    out_f = None
    if args.out:
        op = Path(args.out)
        op.parent.mkdir(parents=True, exist_ok=True)
        out_f = op.open("w", encoding="utf-8")
        print(f"{C.DIM}状态包 -> {op}{C.OFF}")

    overlay = None if args.no_overlay else DebugOverlay()
    print(f"\n{C.BOLD}感知循环运行中{C.OFF}  {cap.get_active_channel().value}  "
          f"tick={args.interval}s  {C.DIM}Ctrl+C 退出{C.OFF}\n")

    n = 0
    t_start = time.perf_counter()
    last_print = 0.0
    try:
        with cap:
            while _running:
                if args.max_frames and n >= args.max_frames:
                    break
                loop_t0 = time.perf_counter()
                try:
                    frame = cap.grab()
                except Exception as exc:  # noqa: BLE001
                    print(f"{C.ERR}截图失败: {exc}{C.OFF}")
                    break

                state = builder.build(frame, forced_scene=args.scene or None)
                events = builder.drain_events()
                ws = store.update(state)
                store.bump_events(len(events))

                if out_f:
                    out_f.write(state.to_json() + "\n")

                for ev in events:
                    if ev.severity in (Severity.WARN, Severity.ERROR):
                        color = C.WARN if ev.severity is Severity.WARN else C.ERR
                        print(f"  {color}[{ev.severity.value}]{C.OFF} {ev}")

                # 调试叠加层复用本帧真实产出（原则 4：工具=运行时）
                if overlay is not None:
                    key = overlay.render(
                        frame, state, builder.last_detections, builder.last_matches
                    )
                    if key in ("ESC", "q"):
                        break

                n += 1
                now = time.perf_counter()
                if now - last_print > 1.0:
                    el = now - t_start
                    print(f"  {C.DIM}帧 {n} | {n/el:.1f} fps | scene={state.scene.id} "
                          f"({state.scene.confidence:.2f}) | tick={state.resource.tick_ms}ms "
                          f"| ok={state.resource.perception_ok}{C.OFF}")
                    last_print = now

                sleep = args.interval - (time.perf_counter() - loop_t0)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        if out_f:
            out_f.close()
        if overlay is not None:
            overlay.close()
        cv2.destroyAllWindows()

    el = time.perf_counter() - t_start
    print(f"\n{C.BOLD}结束{C.OFF}  {n} 帧 / {el:.1f}s = {n/el:.1f} fps")
    if n:
        print(f"  {C.DIM}最后一帧: {json.dumps(store.get().current_state.to_dict(), ensure_ascii=False)[:300]}...{C.OFF}")
    print(f"\n{C.DIM}下一步：状态包字段够用吗？识别对吗？看 JSONL 和上面的事件。{C.OFF}\n")
    return 0


def _load_legacy_ocr():
    """尝试加载 2.0 的 OcrEngine 以复用 WinRT。失败返回 None。"""
    try:
        from ..ocr_engine import OcrEngine as LegacyOcr  # type: ignore

        return LegacyOcr("genshin_game_buddy")
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    raise SystemExit(main())
