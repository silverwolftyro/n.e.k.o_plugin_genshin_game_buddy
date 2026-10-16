"""WGC 冒烟测试 —— 只回答一个问题：**WGC 能不能抓到原神**。

2026-10-10：BitBlt / DXcam / mss 在原神全屏下全抓黑帧。整个 WGC 方案成立与否
全押在这一条上，所以先用一个一次性脚本验证，验证不过就不写集成代码。

    python -m autopilot.tools.wgc_smoke --delay 10

为什么这么写（三条都是踩过的坑，不是风格选择）：

1. **必须 ``start_free_threaded()``**，不能用 ``start()`` —— 后者进 Win32 消息
   循环且不返回，会把主线程直接卡死。
2. **``@capture.event`` 是按函数名分派的**（``windows_capture/__init__.py:295``
  ）：函数名必须正好是 ``on_frame_arrived`` / ``on_closed``，别的名字直接
   ``ValueError``。而且两个 handler 都必须在 ``start_*`` **之前**注册，
   少一个就是 ``RuntimeError``。
3. **``draw_border`` 必须显式关掉**：WGC 默认给捕获内容加一圈**黄色边框**，
   不关的话每帧都带黄框，直接毁掉模板匹配和亮度统计。``cursor_capture`` 同理。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._bootstrap import C, bootstrap, line  # noqa: F401

#: 存 PNG 的目录，和 capture_diagnose 放一起（排障时一起看）
_RUNTIME = Path(__file__).resolve().parents[1] / "_runtime"


def main() -> int:
    # ⚠️ GBK 控制台编不了 ✓/✗ 这类符号（run_loop --help 就这么崩过一次）。
    #   errors="replace" 必须在任何 print **之前**生效。
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description="WGC 冒烟测试（一次性）")
    ap.add_argument("--title", default="原神", help="窗口标题关键字")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="开跑前倒计时几秒（切回游戏用）")
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="最多等几秒收 3 帧")
    ap.add_argument("--save", action="store_true", help="存 capdiag_wgc.png")
    args = ap.parse_args()

    for i in range(int(args.delay), 0, -1):
        print(f"  {i} 秒后开始 —— 请切回游戏并保持前台", flush=True)
        time.sleep(1.0)

    bootstrap()
    import numpy as np

    from autopilot.perception.capturer import _find_window

    hwnd, rect = _find_window(args.title, False)
    if not hwnd:
        print(f"{C.ERR}✗ 找不到标题含 {args.title!r} 的窗口 —— 游戏开着吗？{C.OFF}",
              file=sys.stderr)
        return 2
    print(f"  窗口 hwnd={hwnd} 客户区={rect}")

    try:
        from windows_capture import WindowsCapture
    except ImportError:
        print(f"{C.ERR}✗ 没装 windows-capture：{C.OFF}\n"
              f"    uv pip install --python <venv>\\Scripts\\python.exe windows-capture",
              file=sys.stderr)
        return 2

    seen: list[tuple[int, int, float, int, object]] = []

    cap = WindowsCapture(window_hwnd=hwnd, cursor_capture=False, draw_border=False)

    @cap.event
    def on_frame_arrived(frame, ctrl):  # noqa: ANN001
        # 统计口径与 capture_diagnose **完全一致**，两边数字才能直接对比。
        # ⚠️ 不要用 frame.convert_to_bgr() 的返回值：那是一个**新的 Frame**，
        #   它的 frame_buffer 只是原 buffer 的**视图**，而真正持有原生
        #   mapped frame（内存所有者）的是**传进来的这个 frame**。
        #   一旦原 Frame 被回收，视图就悬空 —— 所以统计直接走
        #   frame.frame_buffer，并且把 frame 本体存进 seen 保活。
        bgr = frame.frame_buffer[:, :, :3]
        quant = (bgr.astype(np.uint16) >> 4).reshape(-1, 3)
        kinds = len({tuple(v) for v in quant[:: max(1, len(quant) // 20000)]})
        luma = float(bgr.mean())
        seen.append((frame.width, frame.height, luma, kinds, frame))
        print(f"  帧{len(seen)}: {frame.width}x{frame.height} "
              f"luma={luma:.1f} 颜色种类={kinds}", flush=True)
        # WGC 是**画面变了才推帧**，不是定时的。收到 3 帧就主动收工，
        # 免得一个不动的窗口把主循环的轮询打满（第一版就这么收了 11 帧）。
        if len(seen) >= 3:
            ctrl.stop()

    @cap.event
    def on_closed():  # noqa: ANN001, ANN202
        print(f"{C.WARN}!! 采集会话被关闭（窗口没了？）{C.OFF}", flush=True)

    print(f"\n{C.BOLD}=== WGC ==={C.OFF} hwnd={hwnd}")
    try:
        ctl = cap.start_free_threaded()
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ WGC 起不来：{type(exc).__name__}: {exc}{C.OFF}",
              file=sys.stderr)
        return 1

    deadline = time.monotonic() + args.timeout
    while len(seen) < 3 and time.monotonic() < deadline:
        if ctl.is_finished():
            print(f"{C.WARN}  采集线程已退出（只拿到 {len(seen)} 帧）{C.OFF}")
            break
        time.sleep(0.5)
    # ⚠️ stop/wait 必须各自兜住异常：采集线程一旦在 handler 里炸了，
    #   stop() 会把「真正的错误」重新抛成 "Failed to stop the capture
    #   thread: <原异常>"，而那个原异常才是病因（第一次跑就踩到了）。
    for fn in (ctl.stop, ctl.wait):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001, PERF203
            print(f"{C.DIM}  {fn.__name__}() 报错（已忽略）："
                  f"{type(exc).__name__}: {exc}{C.OFF}")

    if args.save and seen:
        _RUNTIME.mkdir(parents=True, exist_ok=True)
        out = _RUNTIME / "capdiag_wgc.png"

        from autopilot.perception.image_io import imwrite_unicode

        # ⚠️ 不用 cv2.imwrite：Windows 上路径含中文时会静默存成乱码名
        imwrite_unicode(out, seen[-1][4].frame_buffer[:, :, :3])
        print(f"  已存 {out}")

    print(line())
    if len(seen) < 3:
        print(f"{C.ERR}✗ 只收到 {len(seen)}/3 帧 —— WGC 在这个窗口上抓不到东西{C.OFF}")
        return 1
    luma = seen[-1][2]
    kinds = seen[-1][3]
    if luma < 8.0 or kinds < 3:
        print(f"{C.ERR}✗ 抓到帧但是全黑（luma={luma:.1f} 种类={kinds}）—— "
              f"和 bitblt 一样瞎{C.OFF}")
        return 1
    print(f"{C.OK}✓ WGC 可用：luma={luma:.1f} 颜色种类={kinds}{C.OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())