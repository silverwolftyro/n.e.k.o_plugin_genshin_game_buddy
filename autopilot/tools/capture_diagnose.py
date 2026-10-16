"""三通道抓图体检 —— 在**游戏前台显示**时跑一次，看哪个通道能拿到亮画面。

    python -m autopilot.tools.capture_diagnose
    python -m autopilot.tools.capture_diagnose --title 原神 --save
    python -m autopilot.tools.capture_diagnose --delay 8 --save

★ ``--delay`` 是给「只能全屏」的游戏准备的
  ----------------------------------------------
  双击脚本的那一刻游戏还是最小化的，客户区退化成 0x0，三条通道全都
  「打不开」—— 那**不是 bug**，是你还没切回去。加 ``--delay 8``，
  它会每秒打一行倒计时，你趁这 8 秒 alt-tab 回游戏。
  与 ``run_loop --delay`` 同一套约定。

★ 什么时候需要跑它
------------------
2026-10-10 真机教训：引擎报 ``CAPTURE_FAILED 截图重试 25 次仍未成功：
PrintWindow 返回失败``，而当时的降级链最后一步是 PrintWindow ——
**那条路在 Win11 + 硬件加速下本来就不该存在**。BitBlt 顶替它之后，
又出现「抓到的是黑帧」。这个脚本就是用来回答那个更基础的问题：

    在**当前这台机器、当前这个窗口**上，三条通道各自到底能抓到什么？

★ 怎么读结果
------------
``平均亮度`` 这一列最关键：

  * ``>= 8``   有画面（游戏内容通常在 30~120）
  * ``< 8``    黑帧/近黑 —— 通道能跑但**抓不到内容**
  * ``None``   抓不到 / 通道不可用

抓得到但全黑，比直接报错更危险：引擎会以为自己在工作。
本脚本额外统计**颜色种类数**，纯黑图的颜色数只有 1~2，
这一项能把「黑帧」和「真的拍到了暗场景」分开。

★ 为什么要单独一个脚本而不是复用 check_env
------------------------------------------
``check_env.py`` 是**环境体检**，覆盖面广（YOLO / 模板 / OCR …），
一轮要几十秒；这里只做一件事、每个通道取 3 帧、几秒出结果，
适合"我怀疑截图有问题"这种时刻随手敲。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._bootstrap import C, bootstrap, line  # noqa: F401

#: 每个通道取几帧。只取 1 帧会撞上 dxcam 的间歇空帧（模块 docstring
#: 明写它会「间歇性返回 None」），3 帧能看出是偶发还是稳定失败。
FRAMES_PER_CHANNEL = 3

#: 判定"有内容"的亮度下限。低于它就是黑帧。
LUMA_OK = 8.0

#: 判定"纯黑"的颜色种类上限。纯黑图的 unique 颜色通常只有 1~3。
COLOR_KINDS_DEAD = 8


def _describe(img) -> str:
    """一帧图的体检摘要。"""
    import numpy as np

    if img is None or getattr(img, "size", 0) == 0:
        return "空帧(None)"
    h, w = img.shape[:2]
    luma = float(img.mean())
    # 每个像素压成 16bit 再数种类：全黑图会掉到 1~3，
    # 而真拍到的暗场景（夜晚/洞里）仍然有几十上百种。
    quant = (img[:, :, :3].astype(np.uint16) >> 4).reshape(-1, 3)
    kinds = len({tuple(v) for v in quant[:: max(1, len(quant) // 20000)]})
    verdict = "有画面" if luma >= LUMA_OK else "★黑/近黑"
    if kinds <= COLOR_KINDS_DEAD and luma >= LUMA_OK:
        verdict += "(但颜色种类极少，可能仍是死画面)"
    return f"{w}x{h}  亮度={luma:6.1f}  颜色种类≈{kinds:4d}  {verdict}"


def main() -> int:
    bootstrap()
    # ★ GBK 控制台编不了 `✗`(U+2717) 这类字符，而这个脚本**必然会**
    #   打到「打不开」分支（游戏没开时），于是崩溃在 UnicodeEncodeError 上，
    #   恰好把最需要看到的那条错误给吞了。只加 errors="replace"、不改编码。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")      # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass

    ap = argparse.ArgumentParser(
        description="三通道抓图体检（游戏前台显示时跑）")
    ap.add_argument("--title", default="原神", help="窗口标题关键字")
    ap.add_argument("--save", action="store_true",
                    help="把每通道最后一帧存成 PNG 到 _runtime/，方便肉眼确认")
    # ★ 2026-10-14 新增。与 run_loop 的 --delay 同一套约定：双击脚本 ->
    #   引擎/本脚本还没起来 -> 此刻游戏必然是最小化 -> 立刻开抓图器就会
    #   因客户区退化成 0x0 而 WindowNotFoundError。延迟给你时间 alt-tab 回游戏。
    ap.add_argument("--delay", type=float, default=0.0,
                    help="开始抓图前先等 N 秒，期间打倒计时（默认 0 = 不等）。"
                         "★ 给「只能全屏/最小化」的游戏用：等这段时间里切回游戏")
    args = ap.parse_args()

    # ---- 延迟启动（--delay）----
    #
    # ★ 为什么必须放在**窗口查找之前**（也就是 Capturer.open() 之前）
    #   ------------------------------------------------------------
    #   ``Capturer.open()`` 会立刻去找窗口并探测通道；全屏游戏一旦失去前台
    #   就最小化，客户区退化成 0x0，那一步直接抛 WindowNotFoundError。
    #   所以"等用户切回游戏"必须发生在**抓图器打开之前**，
    #   否则等多久都没用。
    #
    # ★ 为什么打倒计时而不是静默 sleep
    #   ------------------------------
    #   用户双击之后**看不到任何反馈**，会以为程序卡死或根本没启动，
    #   于是又双击一次 —— 第二个实例立刻跟第一个抢同一个游戏窗口。
    #   每秒一行让「它还活着，还有 N 秒」这件事是确定的。
    if args.delay < 0:
        print(f"{C.ERR}✗ --delay 不能为负（{args.delay}）{C.OFF}", file=sys.stderr)
        return 2
    if args.delay > 0:
        total = int(round(args.delay))
        print(f"{C.WARN}▶ 等 {total} 秒后开始 —— 现在请切回游戏窗口！{C.OFF}",
              flush=True)
        try:
            for left in range(total, 0, -1):
                print(f"  {C.DIM}{left:>3} 秒后开始抓图（切回游戏）{C.OFF}",
                      flush=True)
                time.sleep(1.0)
        except KeyboardInterrupt:
            print(f"\n{C.ERR}✗ 等待期间被中断{C.OFF}", file=sys.stderr)
            return 130
        print(f"{C.OK}✓ 等待结束，开始抓图{C.OFF}", flush=True)

    try:
        from autopilot.perception.capturer import CaptureChannel, Capturer
    except ImportError as exc:
        print(f"{C.ERR}✗ 无法导入 capturer：{exc}{C.OFF}", file=sys.stderr)
        return 2

    out_dir = Path(__file__).resolve().parents[1] / "_runtime"
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print(" 三通道抓图体检 —— 请确保游戏**在前台显示**（不是最小化）")
    print("=" * 72)

    verdicts: dict[str, str] = {}

    for ch in CaptureChannel:
        label = ch.value
        print()
        print(f"=== {label} ===")
        cap = None
        try:
            # ★ 关键字是 window_title_substr，不是 title —— 写错会
            #   TypeError，而且报错离病因很远。
            cap = Capturer(window_title_substr=args.title, force_channel=label)
            cap.open()
            active = cap.get_active_channel().value
            print(f"  激活通道 = {active}")
            if active != label:
                print(f"  {C.WARN}⚠ 请求 {label} 但实际激活 {active} —— "
                      f"该通道不可用，已自动降级{C.OFF}")
        except Exception as exc:                      # noqa: BLE001
            print(f"  {C.ERR}✗ 打开失败：{type(exc).__name__}: {exc}{C.OFF}")
            verdicts[label] = "打不开"
            continue

        got_any = False
        best = None
        for i in range(FRAMES_PER_CHANNEL):
            try:
                frame = cap.grab()
            except Exception as exc:                  # noqa: BLE001
                print(f"  第 {i + 1} 帧 抛异常：{type(exc).__name__}: {exc}")
                continue
            img = getattr(frame, "image", None)      # CapturedFrame.image
            desc = _describe(img)
            print(f"  第 {i + 1} 帧  {desc}")
            if img is not None and getattr(img, "size", 0) > 0:
                got_any = True
                best = img
            time.sleep(0.2)

        if not got_any:
            verdicts[label] = "空帧"
        elif "★黑" in _describe(best):
            verdicts[label] = "黑帧"
        else:
            verdicts[label] = "可用"
            if args.save:
                from autopilot.perception.image_io import imwrite_unicode
                p = out_dir / f"capdiag_{label}.png"
                # ⚠️ 不用 cv2.imwrite：--out 含中文时会静默存成乱码名
                imwrite_unicode(p, best)
                print(f"  {C.OK}已存 {p}{C.OFF}")
        try:
            cap.close()
        except Exception:                             # noqa: BLE001
            pass

    print()
    print("=" * 72)
    print(" 结论")
    print("=" * 72)
    for label, v in verdicts.items():
        mark = {C.OK: "✓", C.WARN: "!", C.ERR: "✗"}.get(
            C.OK if v == "可用" else (C.WARN if v == "黑帧" else C.ERR), "?")
        print(f"  {mark} {label:<8} {v}")

    usable = [k for k, v in verdicts.items() if v == "可用"]
    if not usable:
        print()
        print(f"{C.ERR}没有一条通道拿到有内容的画面。{C.OFF}")
        print(f"{C.DIM}  逐条排查：")
        print("    1) 游戏是不是最小化了？最小化时客户区退化成 0x0，"
              "三条通道全都拿不到（这**不是** bug）。")
        print("    2) 有没有开 HDR / 滤镜？会改变合成结果。")
        print("    3) BitBlt 抓黑屏是 Win11 已知问题：")
        print(f"       apply_win11_bitblt_fix(apply=True) 需在**游戏启动前**执行。{C.OFF}")
        return 1

    print()
    print(f"{C.OK}可用通道：{', '.join(usable)}{C.OFF}")
    print(f"{C.DIM}  如果不止一条可用，让引擎用哪条：{C.OFF}")
    print(f"{C.DIM}    在 autopilot/config/defaults.toml 的 "
          f"[autopilot.capture] channel_order 里把它排到前面。{C.OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())