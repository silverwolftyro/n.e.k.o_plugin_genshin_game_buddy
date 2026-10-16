"""从浮窗唤起 / 启动 NEKO-Console（2026-10-05）。

★ 为什么不直接 launch exe
------------------------
主控台的单例机制在 ``neko_console/app.py:620-636``：第二个实例启动时
**先弹一个三选一模态框**（「打开已有实例 / 关闭旧实例并重开 / 取消」）。
所以「点一下按钮就 launch」在主控台已经运行时，会让用户每点一次
就吃一个模态框 —— 那不是「简单」，是把保护逻辑变成了打扰。

正确做法是两段式：
    主控台在跑 -> 走命名管道发一条 ``SHOW``，**静默**弹到前台（无对话框）
    主控台没跑 -> 才启动一个

★ 为什么用 ctypes 而不是 import PySide6
---------------------------------------
管道那一端是 Qt 的 :class:`QLocalServer`。overlay 是 tkinter 的独立进程，
让它为了发 4 个字节去 import PySide6，等于给一个几十行的浮窗拖上一个
几百 MB 的 GUI 框架，还制造一个「overlay 也依赖 Qt」的假耦合。
``CreateFileW`` 是 Win32 自带的，ctypes 是标准库 —— **不引入任何新依赖**。

★ 管道名是 Qt 的内部约定，不是公开 API
------------------------------------
Qt 在 Windows 上把 ``QLocalServer(name)`` 实现成命名管道
``\\\\.\\pipe\\<name>``，服务名见 ``neko_console/app.py:54``
（``IPC_SERVER_NAME = "NekoConsole.IPC"``）。Qt 若哪天改了命名规则，
这里会**静默**失效 —— 所以每一步失败都降级到「启动一个」，
并把降级原因写进日志，而不是让用户对着一个不响的按钮发呆。

★ 崩溃恢复为什么天然成立
----------------------
命名管道是**内核对象**：主控台进程一死（含崩溃 / 强杀）句柄立刻关闭、
名字随即消失。所以 ``CreateFileW`` 连不上天然等价于「没有活着的实例」，
不需要额外的存活探测。
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ 常量

#: Qt 侧的服务名（``neko_console/app.py:54``）。**改名会让正在运行的
#: 旧实例失联** —— 两边必须一致。
CONSOLE_IPC_SERVER_NAME = "NekoConsole.IPC"

#: 上面这个名字在 Windows 上对应的**实际**命名管道路径。
#: Qt 的 QLocalServer 在 Windows 上就是拿它当 CreateNamedPipe 的名字。
CONSOLE_IPC_PIPE = r"\\.\pipe\NekoConsole.IPC"

#: 命令字（``neko_console/app.py:57`` 的 ``CMD_SHOW``）。
#: 接收端见 ``app.py:252``，处理方式是 ``MainWindow.restore_window()``
#: —— 从托盘弹出并置顶。
CONSOLE_IPC_COMMAND = "SHOW"

#: 主控台 exe 路径的环境变量名。由主控台 spawn 浮窗时注入 ——
#: **正方向**注入，因为只有主控台自己知道自己是 frozen 还是源码模式。
CONSOLE_EXE_ENV = "NEKO_CONSOLE_EXE"

#: 环境变量没给时的兜底。仅在浮窗被单独跑起来时用得上。
CONSOLE_EXE_FALLBACK = Path(r"D:\NEKO\NEKO-Console\build\Console.exe")

#: 报错时提一句文件名，比甩一整条路径好懂。
CONSOLE_EXE_NAME = "Console.exe"

#: :func:`launch_console` 的成功返回值。**不是给人看的** —— 是给
#: ``app.py`` 映射 toast 文案用的。分清「怎么成功的」比只给一句
#: 「好了」有用：走管道说明主控台本来就在跑，走启动说明刚刚才起来。
RESULT_IPC = "ipc"
RESULT_SPAWN = "spawn"


def _kernel32():
    """拿到 ``kernel32``。**非 Windows / 加载失败都返回 None**。

    ★ 为什么不放模块顶层
      ------------------
      ``ctypes.WinDLL`` 只在 ``os.name == 'nt'`` 时存在。放顶层就是
      「import 这个模块」直接炸 —— 而浮窗真正需要它的只有点击那一下。
      这里返回 ``None``，让 :func:`_ipc_show_console` 干净地返回
      ``False``，然后走「启动一个」的兜底。
    """
    import ctypes                                    # noqa: PLC0415

    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return None
    try:
        return win_dll("kernel32", use_last_error=True)
    except OSError as exc:
        logger.warning("加载 kernel32 失败，走启动兜底：%s", exc)
        return None


def _ipc_show_console() -> bool:
    """主控台已在运行 -> 静默弹到前台。**没在跑返回 False。**

    ★ 为什么不用超时参数
      ------------------
    ``CreateFileW`` 打开命名管道是**同步**的：服务端没调
    ``ConnectNamedPipe`` / ``CreateNamedPipe`` 时它立刻返回
    ``ERROR_FILE_NOT_FOUND``，不会阻塞。主控台侧
    ``IPC_CONNECT_TIMEOUT_MS = 300`` 是 Qt 的 ``waitForConnected``，
    那是**客户端**超时，跟这里不是一回事 —— 照抄一个 300ms 进来
    只会让人误以为这里也在等。

    ★ 返回 False 意味着什么
      -------------------
      只有一种情况能走到这：**管道不存在 = 主控台没在跑**。
      「在跑但拒绝连接」在 Windows 命名管道上不会表现为
      ``INVALID_HANDLE_VALUE``（那会是别的错误码），所以这里
      不去细分错误码 —— 细分了也改变不了「那就启动一个」。
    """
    k32 = _kernel32()
    if k32 is None:
        return False

    import ctypes                                    # noqa: PLC0415
    from ctypes import wintypes                      # noqa: PLC0415

    GENERIC_WRITE = 0x40000000
    OPEN_EXISTING = 3
    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

    handle = k32.CreateFileW(
        CONSOLE_IPC_PIPE, GENERIC_WRITE, 0, None,
        OPEN_EXISTING, 0, None)
    if handle is None or handle == INVALID_HANDLE_VALUE:
        # 绝大多数情况下这就是「主控台没在跑」，属于**正常路径**，
        # 不记 warning —— 否则每次点按钮日志里都躺一条，看着像故障。
        logger.debug("主控台管道不存在（=%s没在运行）", CONSOLE_EXE_NAME)
        return False

    try:
        payload = CONSOLE_IPC_COMMAND.encode("utf-8")
        written = wintypes.DWORD()
        ok = k32.WriteFile(handle, payload, len(payload),
                           ctypes.byref(written), None)
        if not ok:
            logger.warning("写主控台管道失败，错误码=%d",
                           ctypes.get_last_error())
            return False
        # ★ 短写也要当失败。主控台那边是 ``payload.strip().upper()``
        #   整条比较，收到半个 "SH" 会被当成空命令**默默丢掉** ——
        #   用户点了按钮、什么也没发生、也没有任何日志。
        if written.value != len(payload):
            logger.warning("主控台管道短写：%d/%d 字节",
                           written.value, len(payload))
            return False
        return True
    finally:
        k32.CloseHandle(handle)


def _console_exe() -> tuple[Path | None, str]:
    """定出主控台 exe 路径。返回 ``(路径, 失败原因)``。"""
    raw = (os.environ.get(CONSOLE_EXE_ENV) or "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return p, ""
        logger.warning("%s 指向的不是文件：%s", CONSOLE_EXE_ENV, p)

    if CONSOLE_EXE_FALLBACK.is_file():
        return CONSOLE_EXE_FALLBACK, ""

    return None, (f"找不到 {CONSOLE_EXE_NAME}：{CONSOLE_EXE_ENV} 未设置，"
                  f"兜底路径 {CONSOLE_EXE_FALLBACK} 也不存在")


def _spawn_console() -> tuple[bool, str]:
    """启动一个主控台。返回 ``(成功, 详情)``。

    ★ 三个标志位为什么都要
      --------------------
      ``DETACHED_PROCESS``：不继承浮窗的控制台 —— 否则主控台会把
      浮窗的标准句柄当成自己的，浮窗一关它也跟着完蛋。
      ``CREATE_NEW_PROCESS_GROUP``：让它自己的 Ctrl+C / Ctrl+Break
      只作用于自己，别和浮窗抢。
      三条 ``DEVNULL``：**建了管道却没人读 = 管道写满后子进程阻塞** ——
      进程活着、窗口不动、用户以为启动失败（主控台那边踩过一次同样的坑）。

    ★ cwd 必须设到 exe 所在目录
      -----------------------
      Nuitka onefile 的 ``base_dir()`` 靠 ``__compiled__.containing_dir``
      推导；cwd 不对的话它可能把 config.toml / runtime 写到别处去。
    """
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200

    exe, why = _console_exe()
    if exe is None:
        return False, why
    try:
        subprocess.Popen(
            [str(exe)],
            cwd=str(exe.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    except OSError as exc:
        return False, f"启动 {CONSOLE_EXE_NAME} 失败：{exc}"
    logger.info("已启动主控台：%s", exe)
    return True, str(exe)


def launch_console() -> str:
    """把主控台叫出来。**永远不抛异常。**

    :returns: :data:`RESULT_IPC` / :data:`RESULT_SPAWN`，
        或一句可直接显示给用户的失败原因。

    ★ 为什么不抛
      ----------
    调用方是 Tk 的按钮回调，异常会一路冒到 tkinter 的事件循环，
    打印一串 traceback 之后**按钮就再也不响了**。这里把一切都
    变成返回值，UI 层只需要决定文案。
    """
    try:
        if _ipc_show_console():
            return RESULT_IPC
    except Exception as exc:                          # noqa: BLE001
        # 管道这条路是「优化」，它炸了不该连累启动那条路。
        logger.warning("唤起主控台管道异常，降级为启动：%s", exc)
    try:
        ok, detail = _spawn_console()
    except Exception as exc:                          # noqa: BLE001
        return f"启动主控台异常：{exc}"
    return RESULT_SPAWN if ok else detail


__all__ = [
    "CONSOLE_EXE_ENV",
    "CONSOLE_EXE_FALLBACK",
    "CONSOLE_IPC_COMMAND",
    "CONSOLE_IPC_PIPE",
    "CONSOLE_IPC_SERVER_NAME",
    "RESULT_IPC",
    "RESULT_SPAWN",
    "launch_console",
]
