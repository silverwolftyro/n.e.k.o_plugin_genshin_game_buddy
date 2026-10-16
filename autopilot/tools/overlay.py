"""autopilot overlay —— 独立进程的状态浮窗。

    python -m autopilot.tools.overlay
    python -m autopilot.tools.overlay --reset-pos
    python -m autopilot.tools.overlay --x 1500 --y 140 --alpha 0.8
    python -m autopilot.tools.overlay --scale 0.8     # 小两成，少挡游戏画面
    python -m autopilot.tools.overlay --mode dev     # 开发者版：事件流 + 分析摘要

A 轨：独立于 N.E.K.O. 插件，只通过 autopilot/run/ 下的文件与 run_loop 通信。
两个进程互不知道对方存在 —— overlay 关掉不影响 run_loop，
run_loop 崩了 overlay 显示「离线」。

⚠ 默认**不发任何按键**，它连执行层都不 import。唯一能控制 run_loop 的
    是那个 ⏹ 按钮，它写的是协议里的 KILL 命令。

依赖：仅标准库 tkinter + 已有的 pydantic。**不新增第三方依赖**。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, setup_logging


def declare_dpi_aware() -> None:
    """在**任何 Tk() 创建之前**强制声明进程 DPI 感知。

    ★ 为什么必须显式声明
    ------------------
    在不同 DPI 感知状态的宿主进程里启动，tkinter 看到的屏幕尺寸不一样：

        终端直接跑        -> 未声明 -> winfo_screenwidth() 返回 1707（逻辑像素）
        主控台(PySide6)起  -> 继承 Qt 的 DPI 感知 -> 返回 2560（物理像素）

    同一份 ``overlay_pos.json`` 里的 (x, y) 在这两种情况下会被放到
    屏幕上的**不同物理位置**，用户看到的就是"位置没归位"。

    声明之后两条路径都返回物理像素，坐标就一致了。

    ★ 位置很关键：必须在 ``tk.Tk()`` **之前**调用。Tk 创建后再声明
      来不及 —— 窗口几何在创建那一刻就按当时的 DPI 上下文算好了。

    ★ 失败静默：声明失败不该挡住启动，大不了退回"坐标可能不准"，
      而不是"程序起不来"。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes  # noqa: PLC0415 - 只在 Windows 上需要

        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        # 首选 shcore（Win 8.1+，能区分多显示器各自的缩放）
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            # 老系统（Win 7 / 8.0）没有 shcore，退回 System-DPI-Aware
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001 - 声明失败不能挡住启动
        pass


def main() -> int:
    # ★ 第一件事就是声明 DPI —— 早于 Tk() 创建，早于任何窗口几何计算
    declare_dpi_aware()

    ap = argparse.ArgumentParser(
        description="autopilot 状态浮窗（独立进程，不发按键）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--x", type=int, default=None, help="覆盖默认 X（像素）")
    ap.add_argument("--y", type=int, default=None, help="覆盖默认 Y（像素）")
    ap.add_argument("--alpha", type=float, default=None,
                    help="覆盖默认透明度 0..1（默认 0.72）")
    ap.add_argument("--scale", type=float, default=None,
                    help="卡片大小倍率 0.5~3.0（默认 1.0 = 系统默认大小；"
                         "调大更清楚但更挡画面，调小更不挡但更小）")
    ap.add_argument("--reset-pos", action="store_true",
                    help="清除保存的位置，回到默认比例坐标")
    # ★ 2026-10-10 新增。开发者版是**同一套窗口的第二种排版**，
    #   不是另起一个进程 —— 位置/缩放/按钮/拖动全部复用（见 dev_panel.py）。
    ap.add_argument("--mode", default="product",
                    choices=["product", "dev"],
                    help="product=5 行精简面板（默认）；"
                         "dev=开发者版：顶部状态行 + 事件流 + 分析摘要，"
                         "面板更宽、刷新 500ms")
    ap.add_argument("--event-rows", type=int, default=10,
                    help="开发者版事件流占几行（默认 10，仅 --mode dev 有效）")
    ap.add_argument("--theme", default="",
                    help="颜色主题：dark（默认）/ light / high-contrast。"
                         "运行中按 Alt+T 也能轮换")
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARN", "ERROR"])
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.log_level == "DEBUG")
    if not sys.stdout.isatty():
        C.disable()

    if args.alpha is not None and not (0.0 <= args.alpha <= 1.0):
        print(f"{C.ERR}✗ --alpha 必须在 0..1 之间（收到 {args.alpha}）{C.OFF}",
              file=sys.stderr)
        return 2

    try:
        from autopilot.ui.overlay.app import build  # noqa: PLC0415
    except ImportError as exc:
        # tkinter 在某些精简的 Python 发行版里没编进去
        print(f"{C.ERR}✗ 无法加载 overlay：{exc}{C.OFF}", file=sys.stderr)
        print(f"{C.DIM}  这个 Python 可能没编译 tkinter。"
              f"标准安装包都带，conda/embed 发行版有时不带。{C.OFF}")
        return 2

    # ★ --scale 的区间校验放在这里，而不是跟 --alpha 挤在一起：
    #   边界定义在 theme 里，而 theme 必须在 declare_dpi_aware() **之后**
    #   才导入（BASE_SCALE 依赖那个顺序）。上面的 app 导入已经把 theme
    #   带进来了，这里直接取边界，避免重复写死一份数字。
    if args.scale is not None:
        from autopilot.ui.overlay import theme as _theme  # noqa: PLC0415

        lo, hi = _theme.USER_SCALE_MIN, _theme.USER_SCALE_MAX
        if not math.isfinite(args.scale) or not (lo <= args.scale <= hi):
            print(f"{C.ERR}✗ --scale 必须是 {lo} ~ {hi} 之间的数字"
                  f"（收到 {args.scale}）{C.OFF}", file=sys.stderr)
            print(f"{C.DIM}  1.0 = 系统默认大小。调大更清楚但更挡游戏画面，"
                  f"调小更不挡但字更小。{C.OFF}")
            return 2

    if args.event_rows < 1:
        print(f"{C.ERR}✗ --event-rows 至少 1（收到 {args.event_rows}）{C.OFF}",
              file=sys.stderr)
        return 2

    build_kw = {}
    if args.mode == "dev":
        build_kw["event_rows"] = args.event_rows

    if args.theme:
        # ★ 必须在 build() **之前**应用：widget 的颜色是构造时烤进去的，
        #   建完再切主题就得逐控件重画（app 里 _force_theme_repaint 干那活）。
        from autopilot.ui.overlay import theme as _theme  # noqa: PLC0415

        _theme.apply_theme(args.theme)

    app = build(x=args.x, y=args.y, alpha=args.alpha,
                scale=args.scale, reset_pos=args.reset_pos,
                mode=args.mode, **build_kw)
    print(f"{C.DIM}overlay 已启动（{args.mode}），Ctrl+C 关闭{C.OFF}")
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
