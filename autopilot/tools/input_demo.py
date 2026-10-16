"""输入演示：验证「按键真的发得出去」。

    python -m autopilot.tools.input_demo              # 干跑，只打印不发送（默认）
    python -m autopilot.tools.input_demo --live       # 真发输入 ⚠
    python -m autopilot.tools.input_demo --live --title 原神   # 顺带校验前台焦点

演示序列（对应任务书验收）：**等 3 秒 → 按住 W 1 秒 → 按 E → 按 1 → 释放**。

**为什么默认是干跑**：这个工具会往你的真实桌面发键鼠输入。默认发出去
等于「跑一条命令就动了你的鼠标」，不该是默认行为。要真发必须显式 `--live`。

全程可用 **F12 急停**（不是 Ctrl+C —— 那一刻你手可能在游戏里，
Ctrl+C 需要切窗口，而 F12 是无焦点的）。急停检查每 20ms 一次。

⚠ `--live` 的常见失败（报错会明确告诉你是哪一种）：
  · 权限不足 → 以管理员身份运行本脚本
  · 被安全软件拦截 → 关掉 360 / 火绒 的「键鼠模拟」防护
  · UIPI → 目标窗口以更高权限运行时本进程无法注入
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
    ap = argparse.ArgumentParser(
        description="输入演示（默认干跑，加 --live 才真发）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--live", action="store_true",
                    help="真的发送输入。⚠ 会控制你的键鼠")
    ap.add_argument("--title", default="",
                    help="同时校验该标题窗口在前台（默认不校验）")
    ap.add_argument("--countdown", type=float, default=3.0,
                    help="开始前倒计时秒数（给你时间切窗口）")
    ap.add_argument("--hold", type=float, default=1.0,
                    help="W 键按住秒数")
    ap.add_argument("--skip-release", action="store_true",
                    help="故意不调 release_all —— 用来验证「卡住的按键」长什么样")
    ap.add_argument("--diagnose", action="store_true",
                    help="只打印 backend 诊断（平台/DLL/权限/完整性级别），"
                         "不发送任何输入")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    from autopilot.actuator import (  # noqa: PLC0415
        DEFAULT_KEYMAP, DryRunBackend, InputSimulator, InvalidStateError,
        PANIC_KEY, VK, Win32Backend,
    )
    from autopilot.actuator.safety import SafetyGuard  # noqa: PLC0415

    # ★ 不要写 `InputSimulator(backend=None)`：dataclass 的 default_factory
    #   只在**不传该参数**时生效，显式 None 会覆盖它，于是拿到字面量 None，
    #   直到第一次 send() 才炸成 "'NoneType' object has no attribute 'send'"。
    #
    # --diagnose 必须探 **Win32 真实路径**：干跑后端永远报告 ok，
    # 用它诊断等于什么都没查。
    want_win32 = args.live or args.diagnose
    try:
        sim = InputSimulator(Win32Backend() if want_win32 else DryRunBackend())
    except InvalidStateError as exc:
        print(f"{C.ERR}✗ 无法初始化输入后端{C.OFF}\n\n{exc}", file=sys.stderr)
        return 2

    # ---- --diagnose：只读检查，不发任何输入 ----
    if args.diagnose:
        rep = sim.diagnose()
        print(f"\n{C.BOLD}输入后端诊断{C.OFF}")
        print(f"  平台          : {'Windows' if rep.is_windows else '非 Windows'}")
        print(f"  backend 类型  : {rep.backend_type}")
        print(f"  user32.dll    : {'✓ 已加载' if rep.user32_loaded else '✗ 加载失败'}")
        print(f"  SendInput     : {'✓ 可调用' if rep.sendinput_callable else '✗ 不可调用'}")
        print(f"  进程完整性    : {rep.integrity_label or '未知'}"
              f"{f' (rid=0x{rep.integrity_rid:04X})' if rep.integrity_rid else ''}")
        print(f"  VK 表条目     : {rep.vk_count}")
        print(f"  总体          : {'✓ 可用' if rep.ok else '✗ 不可用'}")
        if rep.problems:
            print(f"\n{C.ERR}问题：{C.OFF}")
            for i, p in enumerate(rep.problems, 1):
                print(f"  {i}. {p}")
        print()
        return 0 if rep.ok else 2

    mode = f"{C.ERR}LIVE{C.OFF}" if args.live else f"{C.OK}DRY-RUN{C.OFF}"
    print(f"\n{C.BOLD}输入演示{C.OFF}  模式={mode}")
    if args.live:
        print(f"  {C.WARN}⚠ 接下来会真的控制你的键鼠。随时按 F12 停手。{C.OFF}")
    else:
        print(f"  {C.DIM}干跑：只记录不发送。确认无误后加 --live 真跑。{C.OFF}")
    print(f"  急停键 = {PANIC_KEY.name}")

    # 急停线程：F12 用独立线程轮询，主循环只查标志位。
    # 不做成后台线程的话，长按 W 期间没人看标志，停下就要等 1 秒。
    guard = SafetyGuard()

    import threading  # noqa: PLC0415

    stop = threading.Event()

    def watch_panic() -> None:
        while not stop.wait(0.02):
            if guard.panic_active:
                stop.set()

    if args.live:
        w = threading.Thread(target=watch_panic, daemon=True, name="panic-watch")
        w.start()

    sent: list[str] = []

    def step(desc: str, fn, *fn_args) -> bool:
        """跑一步。返回 False = 被急停 / 失败，应立刻收手。"""
        if stop.is_set():
            print(f"  {C.ERR}✗ 急停生效，剩余步骤不再执行{C.OFF}")
            return False
        before = len(sim.history)
        try:
            fn(*fn_args)
        except Exception as exc:  # noqa: BLE001
            print(f"  {C.ERR}✗ {desc} 失败: {exc}{C.OFF}")
            return False
        got = sim.history[before:]
        summary = ", ".join(_fmt(e) for e in got) or "(无事件)"
        sent.append(f"{desc}: {summary}")
        print(f"  {C.OK}✓{C.OFF} {desc:<22} {summary}")
        return True

    def hold_w(seconds: float) -> None:
        """按住 W —— 分片循环，每片都查急停，否则停手要等满一秒。"""
        vk = DEFAULT_KEYMAP["MOVE_FORWARD"]
        sim.key_down(vk)
        end = time.monotonic() + seconds
        try:
            while time.monotonic() < end:
                if stop.is_set():
                    return
                time.sleep(0.02)
        finally:
            sim.key_up(vk)

    print(f"\n{line()}")
    try:
        # ---- ① 倒计时：给用户时间切到游戏窗口 ----
        for i in range(int(args.countdown), 0, -1):
            if stop.is_set():
                return 130
            sys.stdout.write(f"\r  {C.DIM}倒计时 {i} …{C.OFF}")
            sys.stdout.flush()
            time.sleep(1.0)
        sys.stdout.write("\r" + " " * 30 + "\r")

        # ---- ② 前台校验（可选）----
        if args.title:
            guard.set_foreground_title(args.title)
            if not guard.is_foreground_ok():
                print(f"  {C.ERR}✗ 窗口「{args.title}」不在前台，已中止{C.OFF}")
                print(f"    {C.DIM}先把游戏切到前台再跑。{C.OFF}")
                return 1
            print(f"  {C.OK}✓{C.OFF} 前台校验通过：{args.title}")

        # ---- ③ 演示序列 ----
        if not step(f"按住 W {args.hold}s", hold_w, args.hold):
            return 130
        if not step("按 E", sim.key_press, DEFAULT_KEYMAP["USE_SKILL"], 30):
            return 130
        if not step("按 1（切人）", sim.key_press, VK.ONE, 30):
            return 130
        if not step("等 0.3s", lambda: time.sleep(0.3)):
            return 130

        # ---- ④ release_all ----
        if args.skip_release:
            print(f"\n  {C.WARN}⚠ 已跳过 release_all。live 模式下可能真的卡住按键，"
                  f"请手动按一遍 W/S/A/D 解除。{C.OFF}")
        else:
            before = len(sim.history)
            released = sim.release_all()
            got = sim.history[before:]
            summary = ", ".join(_fmt(e) for e in got) or "(无按键被按下，无需释放)"
            print(f"  {C.OK}✓{C.OFF} {'release_all':<22} {summary}")
            print(f"    {C.DIM}释放了 {len(released)} 个键码{C.OFF}")

    except KeyboardInterrupt:
        print(f"\n  {C.WARN}Ctrl+C 中断{C.OFF}")
        return 130
    finally:
        # ★ 无条件释放 —— 与 executor 同一条铁律
        stop.set()
        try:
            sim.release_all()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.02)   # 让后台线程看到 stop 标志退出

    # ---- ⑤ 汇总 ----
    print(f"\n{line()}")
    total = len(sim.history)
    print(f"{C.BOLD}共发出 {total} 个输入事件{C.OFF}")
    if not args.live:
        print(f"{C.DIM}（干跑：这些只进了记录，没有真的发出去）{C.OFF}")
        print(f"  下一步：python -m autopilot.tools.input_demo --live\n")
    else:
        print(f"  如果游戏里什么都没发生，看上面 FAILED 那行的具体原因。\n")
    return 0


def _fmt(e) -> str:
    if e.kind == "key":
        return f"{e.name}{'' if e.down else ' up'}"
    return f"{e.name}({e.dx},{e.dy})"


if __name__ == "__main__":
    raise SystemExit(main())
