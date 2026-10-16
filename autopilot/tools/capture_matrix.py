"""截图通道 × 窗口状态 矩阵采集（不需要原神）。

    python -m autopilot.tools.capture_matrix

用 tkinter 造一个内容**高对比、状态可辨**的测试窗口，然后把它摆成三种状态，
每个状态用每个通道各抓一帧，最后打印 MD5 与亮度并校验「9 张互不相同」。

★ 为什么必须校验 MD5
------------------
2026-10-12 的实测打假：磁盘上曾躺着一批叫
``cap3_A_正常显示 / cap3_B_屏幕最小化 / cap3_C_最小化`` 的截图，
看起来是「受控矩阵」，结果 **A/B/C 三组 MD5 完全相同** —— 是同一帧
复制了三份。照文件名验黑帧会得出「最小化没判成 BLACK」的错误结论。

**文件名会骗人，MD5 不会。** 所以本工具每次采集后强制自检：
9 张里只要有两张 MD5 相同就报「采集无效」。

⚠️ 必须**每个状态跑在独立进程**里
--------------------------------
最小化 / 移出屏幕这些 Win32 状态是**进程外**的全局状态。同一进程里连续
摆三种状态，前一个的残留（窗口矩形缓存、通道黑名单）会污染后一个的结果。
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, line, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, line, setup_logging

#: 测试窗口标题（ASCII，避免和文件名的 Unicode 问题纠缠）
TITLE = "ZZCAPTEST"

#: 输出目录 = ``_runtime`` **本身**，文件名平铺、带 ``capmatrix_`` 前缀。
#:
#: ⚠️ 刻意**不建子目录**：``_runtime/.gitignore`` 内容是 ``*``，
#:    打包器能按**文件**排除（那 60+ 个 .jsonl 就是这么被跳过的），
#:    但**目录**排除不掉 —— 建一个 ``cap_matrix/`` 子目录会让打包器报
#:    ``[WARN] staged tree contains directories that cannot survive
#:    packaging (['_runtime'])``，进而**跳过 metadata 推导**，
#:    打出来的包没有 plugin.meta.json。
#:    平铺写就绕开了这个问题，而且**不需要**动打包规则。
OUT = Path(__file__).resolve().parents[1] / "_runtime"

#: 输出文件名前缀
PREFIX = "capmatrix_"

STATES = ("normal", "offscreen", "minimize")

STATE_CN = {
    "normal": "正常显示",
    "offscreen": "拖到屏幕外",
    "minimize": "最小化",
}

#: 子进程里跑的窗口代码：**刻意画成逐列字母 + 大号文字**，
#: 这样「有没有抓到内容」一眼可辨，也不会和纯色块混淆。
#:
#: ⚠️ 颜色直接写 ``#rrggbb``，不要为了躲外层的 ``% TITLE`` 格式化而写成
#:    ``%23rrggbb`` —— 那样外层 ``%`` 展开后 Tk 收到的是 ``%23000000``，
#:    直接 ``TclError: unknown color name``，窗口起不来（踩过）。
CHILD_SRC = (
    "import tkinter as tk\n"
    "root = tk.Tk()\n"
    "root.title(%r)\n"
    "root.geometry('900x600+300+200')\n"
    "root.configure(bg='#101010')\n"
    "cv = tk.Canvas(root, width=900, height=600, highlightthickness=0,\n"
    "               bg='#101010')\n"
    "cv.pack()\n"
    "for i in range(0, 900, 30):\n"
    "    cv.create_rectangle(i, 0, i+30, 600, fill='#202830',\n"
    "                       outline='')\n"
    "    cv.create_text(i+15, 300, text=chr(65+i//30), fill='#ffffff',\n"
    "                   font=('Consolas', 18, 'bold'))\n"
    "cv.create_text(450, 120, text='CAPTURE MATRIX', fill='#00ff88',\n"
    "               font=('Consolas', 40, 'bold'))\n"
    "cv.create_text(450, 480, text='ZZCAPTEST', fill='#ff8800',\n"
    "               font=('Consolas', 30, 'bold'))\n"
    "root.update()\n"
    "root.mainloop()\n"
) % TITLE


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


def _u32_setup() -> None:
    u = ctypes.windll.user32
    u.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    u.ShowWindow.restype = ctypes.c_bool
    u.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + \
        [ctypes.c_int] * 4 + [ctypes.c_uint]
    u.SetWindowPos.restype = ctypes.c_bool
    u.IsIconic.argtypes = [ctypes.c_void_p]
    u.IsIconic.restype = ctypes.c_bool


def _find_window(title: str) -> int | None:
    u = ctypes.windll.user32
    found: list[int] = []
    proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def _cb(h, _e):
        buf = ctypes.create_unicode_buffer(512)
        u.GetWindowTextW(ctypes.c_void_p(h), buf, 512)
        if buf.value == title:
            found.append(int(h))
        return True

    u.EnumWindows(proc(_cb), None)
    return found[0] if found else None


def _window_rect(hwnd: int) -> tuple[int, int, int, int]:
    rc = _RECT()
    ctypes.windll.user32.GetWindowRect(ctypes.c_void_p(hwnd), ctypes.byref(rc))
    return (rc.left, rc.top, rc.right, rc.bottom)


def _child_for(state: str) -> None:
    """子进程：开窗 -> 摆成 state -> 三通道各抓一帧 -> 写盘。"""
    u32 = ctypes.windll.user32
    _u32_setup()
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001
        pass

    from autopilot.perception import capturer as cap_mod
    from autopilot.perception.builder import BLACK_FRAME_LUMA, _luma
    from autopilot.perception.image_io import imwrite_unicode

    OUT.mkdir(parents=True, exist_ok=True)

    child = subprocess.Popen([sys.executable, "-c", CHILD_SRC])
    hwnd = None
    for _ in range(80):
        time.sleep(0.25)
        hwnd = _find_window(TITLE)
        if hwnd:
            break
    if not hwnd:
        print("  测试窗口没起来")
        child.kill()
        raise SystemExit(1)
    time.sleep(1.2)

    hp = ctypes.c_void_p(hwnd)
    if state == "minimize":
        u32.ShowWindow(hp, 6)                    # SW_MINIMIZE
    elif state == "offscreen":
        u32.ShowWindow(hp, 9)                    # SW_RESTORE
        time.sleep(0.5)
        u32.SetWindowPos(hp, None, -3200, -3200, 0, 0,
                         0x0001 | 0x0004 | 0x0010)  # NOSIZE|NOZORDER|NOMOVE
        u32.SetWindowPos(hp, None, -3200, -3200, 900, 600,
                         0x0004 | 0x0010)         # NOZORDER|NOMOVE
    time.sleep(2.0)

    left, top, right, bot = _window_rect(hwnd)
    print(f"  状态={state:<10} 窗口矩形=({left},{top},{right-left}x{bot-top}) "
          f"IsIconic={bool(u32.IsIconic(hp))}")
    print()

    for channel in cap_mod.CaptureChannel:
        label = channel.value
        t0 = time.perf_counter()
        try:
            cap = cap_mod.Capturer(TITLE, force_channel=label)
            cap.open()
            active = cap.get_active_channel().value
            frame = cap.grab()
            ms = (time.perf_counter() - t0) * 1000
            img = getattr(frame, "image", None)
            if img is None or getattr(img, "size", 0) == 0:
                print(f"    {label:<7} 激活={active:<7} 抓取=空   {ms:.0f}ms")
            else:
                L = _luma(img)
                p = OUT / f"{PREFIX}{state}_{label}.png"
                imwrite_unicode(p, img)
                md5 = hashlib.md5(p.read_bytes()).hexdigest()[:16] \
                    if p.exists() else "MISSING"
                print(f"    {label:<7} 激活={active:<7} "
                      f"{img.shape[1]}x{img.shape[0]:<5} luma={L:6.1f} "
                      f"{'BLACK' if L < BLACK_FRAME_LUMA else 'ok   '} "
                      f"{ms:5.0f}ms  md5={md5}")
            cap.close()
        except Exception as exc:  # noqa: BLE001
            ms = (time.perf_counter() - t0) * 1000
            print(f"    {label:<7} 失败 {type(exc).__name__}: "
                  f"{str(exc)[:80]}  {ms:.0f}ms")

    child.kill()


def _verify() -> int:
    """校验：所有 ``capmatrix_*.png`` 齐全且互不相同。不通过返回非 0。"""
    files = sorted(OUT.glob(f"{PREFIX}*.png"))
    digests: dict[str, list[str]] = {}
    print(line("═"))
    print(f"{C.BOLD}MD5 校验{C.OFF}  （{len(files)} 张）")
    for p in files:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        digests.setdefault(h, []).append(p.name)
        print(f"  {h[:16]}  {p.name}")

    dupes = {h: names for h, names in digests.items() if len(names) > 1}
    print(line("═"))
    if not files:
        print(f"{C.ERR}没采到任何图 —— 采集无效{C.OFF}")
        return 2
    if dupes:
        print(f"{C.ERR}✗ 采集无效：{len(dupes)} 组 MD5 相同{C.OFF}")
        for h, names in dupes.items():
            print(f"    {h[:16]}  {names}")
        print("  这些不是不同状态的画面 —— 别拿它们当回归基线。")
        return 1
    print(f"{C.OK}✓ {len(files)} 张全部互不相同 —— 采集有效{C.OFF}")
    return 0


def main() -> int:
    global OUT          # 必须在任何一次 OUT 的**读**之前声明
    ap = argparse.ArgumentParser(
        description="截图通道 × 窗口状态 矩阵采集（用 tkinter 窗口，不需要原神）")
    ap.add_argument("--out", default="", help=f"输出目录（默认 {OUT}）")
    ap.add_argument("--keep", action="store_true", help="采集后不清空旧图")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.keep:
        for old in OUT.glob(f"{PREFIX}*.png"):
            old.unlink()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    here = str(Path(__file__).resolve())
    print(line("═"))
    print(f"{C.BOLD}通道 × 状态 矩阵采集{C.OFF}")
    print(f"  输出：{OUT}")
    print(f"  状态：{', '.join(f'{s}({STATE_CN[s]})' for s in STATES)}")
    print(line("═"))

    for st in STATES:
        print(f"{C.BOLD}# 独立进程：{st}（{STATE_CN[st]}）{C.OFF}")
        r = subprocess.run([sys.executable, here, "--child", st],
                           env={**__import__("os").environ,
                                "NEKO_CAPMATRIX_CHILD": "1"})
        if r.returncode:
            print(f"  子进程退出码 {r.returncode}")

    rc = _verify()
    print()
    if rc == 0:
        print(f"下一步：把这些图喂给黑帧判定回归（保存目录 {OUT}）")
    return rc


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        _child_for(sys.argv[2])
        raise SystemExit(0)
    raise SystemExit(main())
