"""YUI 插件侧的「一键拉起」接口。

★ 这是给**宿主**用的，不给用户用
------------------------------
用户双击的是发行包根目录下的 ``一键启动.bat``（那只是两行）。
YUI（宿主插件）在自己的按钮里调这里面的函数，就能做到同样的事，
而且不用起一个 cmd 窗口。

★ 为什么不是 ``subprocess.run``
------------------------------
``run`` 会等子进程退出。而 ``NEKO-Console.exe`` 起来之后是**常驻的**
（托盘图标），等它退出等于永远阻塞。用 ``Popen`` + 立刻返回，
才是「点一下就走」的语义。

⚠️ 一个必须知道的事实：第一次会弹 UAC
--------------------------------------
发行包 ``config.toml`` 里 ``require_admin = true``，截屏 API 需要提权。
所以 ``--autopilot`` 那个进程会经 ``ShellExecuteW(runas)`` **重新拉起
自己一次**。这意味着：

    · 用户会看到一次 UAC —— 这是预期的，不是 bug；
    · 新进程拿到的 argv 与原来**完全一致**（``elevation.py`` 会转发
      ``sys.argv[1:]``，2026-10-19 修的）；
    · 所以**不要**在这个 ``Popen`` 上等返回码来判成败 ——
      它退出得很快（提权后就 SystemExit），那不是「启动失败」。

★ 已有一个实例在跑时怎么办
--------------------------
Console 的 ``single_instance`` 会把第二个实例挡在 IPC 外面。
那种情况下新进程会通过 ``console_commands.jsonl`` 把「起引擎 / 起卡片」
转交给旧实例，然后自己安静退出。所以重复点是安全的。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

__all__ = [
    "CONSOLE_EXE_NAME",
    "find_console_exe",
    "launch",
    "main",
]

#: 发行包里主控台 exe 的文件名。
CONSOLE_EXE_NAME = "NEKO-Console.exe"


def find_console_exe(start: str | Path | None = None) -> Path | None:
    """从 ``start`` 往上找 ``NEKO-Console.exe``。

    :param start: 起点。默认取本文件所在目录。
    :returns: 找不到返回 ``None`` —— **不抛**。
        宿主没装发行包是常态，报异常会让 YUI 的按钮直接崩掉。

    ★ 为什么往上找而不是只查本目录
      源码形态下本文件在 ``plugin/plugins/genshin_game_buddy/``，
      exe 在 ``<发行包>/``；编译形态下它们根本不在一个盘符的同一个树里。
      往上逐级找是唯一能同时罩住「就地放一个 exe」和
      「仓库开发时指向 stage 目录」两种摆放的办法。
    """
    here = Path(start).resolve() if start else Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / CONSOLE_EXE_NAME
        if cand.is_file():
            return cand
    return None


def build_args(*, autopilot: bool = True, overlay: bool = True,
               minimized: bool = True, dry_run: bool = False) -> list[str]:
    """拼参数。**与 :mod:`neko_console.cli` 一一对应**。"""
    args: list[str] = []
    if autopilot:
        args.append("--autopilot")
    if dry_run:
        args.append("--dry-run")
    if overlay:
        args.append("--overlay")
    if minimized:
        args.append("--minimized")
    return args


def launch(exe: str | Path | None = None, *, autopilot: bool = True,
           overlay: bool = True, minimized: bool = True,
           dry_run: bool = False,
           creationflags: int | None = None) -> subprocess.Popen | None:
    """拉起主控台，**立刻返回**。

    :param exe: 主控台 exe。``None`` = 用 :func:`find_console_exe` 找。
    :returns: ``Popen``；找不到 exe 返回 ``None``（**不抛**）。
    :raises RuntimeError: exe 存在但拉不起来（权限 / 格式不对）。
    """
    target = Path(exe) if exe else find_console_exe()
    if target is None or not target.is_file():
        return None
    args = [str(target), *build_args(autopilot=autopilot, overlay=overlay,
                                     minimized=minimized, dry_run=dry_run)]
    # ★ DETACHED_PROCESS：主控台是常驻的，不能让它的控制台跟着
    #   YUI 的命令行窗口一起消失。宿主是 GUI 程序时这几乎不会发生，
    #   但从 cmd 里跑 launch.py 就会 —— 那是本脚本自己的一条路径。
    flags = creationflags
    if flags is None and sys.platform == "win32":
        flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    return subprocess.Popen(  # noqa: S603 - 参数是列表，无 shell
        args, cwd=str(target.parent),
        creationflags=flags or 0)


def main() -> int:
    """命令行自测入口：``python launch_autopilot_buddy.py``。"""
    exe = find_console_exe()
    if exe is None:
        print(f"没找到 {CONSOLE_EXE_NAME}。\n"
              f"请把它放在发行包根目录（与 {CONSOLE_EXE_NAME} 同级的脚本），"
              f"或显式传路径。")
        return 2
    proc = launch(exe)
    if proc is None:
        print(f"拉起失败：{exe}")
        return 1
    print(f"已拉起（pid={proc.pid}）：{exe}")
    print("参数：--autopilot --overlay --minimized")
    if sys.platform == "win32" and not os.environ.get("WT_SESSION"):
        print("提示：第一次会弹一次 UAC，点「是」即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
