"""孤儿守卫：主控台不在了，卡片自己走。

★ 要解决什么
------------
卡片是**独立进程**（防崩溃隔离，不能合并进主控台）。主控台退出时只
terminate **它自己 spawn 的那个** handle（``OverlayManager._proc``）。
于是这三种卡片它根本看不见，也就永远留着：

    · 用户手动 ``python -m autopilot.tools.overlay`` 起的
    · 上一个主控台实例起的，那个实例崩了没清理
    · N.E.K.O 插件起的

结果就是**孤儿**：一个没有主人的置顶浮窗，永远浮在游戏画面上。

★ 为什么是"卡片自己判断"，而不是"主控台通知"
----------------------------------------------
主控台**通知不到**才是问题的根 —— 孤儿要么是它没 spawn 的，要么是它
自己先没了。两条路都走不通，只能让卡片**反向观察**主控台的死活。

★ 两条独立信号（都到了才算数，任意一条即可触发退出）
---------------------------------------------------
    1. **优雅退出**：主控台 ``quit_app()`` 退出前会先写一次
       ``console.running=false`` 进桥接文件（app.py:611）。卡片读到就
       **立即**退出 —— 不用等宽限期。
    2. **崩溃 / 任务管理器强杀**：桥接文件不会再更新，但父 PID 已经
       不在了。累计超过宽限期（默认 5s）才退出。

为什么崩溃要宽限、优雅不要：
    崩溃场景下宽限期是**唯一**的防线，而 :func:`pid_alive` 可能因为
    权限/PID 短暂查不到而误报 —— 误报的代价是**误杀一张还在正常显示
    的卡片**，比多留 5 秒孤儿严重得多。反过来，优雅路径的信号是主控台
    亲笔写下的 false，误报概率为零，没必要等。

★ 为什么手动启动的卡片**不**自杀
--------------------------------
``python -m autopilot.tools.overlay`` 本来就是受支持的用法（没有主控台
也能单跑）。拿"桥接文件不更新"当死亡判据会把它立刻杀掉。所以**只有
显式拿到主控台 PID 时才上膛**（``NEKO_CONSOLE_PID``），否则整个守卫
空转，永远不管。

本模块**不 import tkinter**，方便单测。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: 主控台 spawn 卡片时注入自己的 PID。**只有它存在才武装守卫**。
CONSOLE_PID_ENV = "NEKO_CONSOLE_PID"

#: 桥接文件名。必须与 ``neko_console/bridge.py::BRIDGE_FILE_NAME`` 一致。
BRIDGE_FILE_NAME = "console_bridge.json"

#: 崩溃后到自杀的宽限秒数。
#:
#: ★ 为什么是 3s 而不是 5s
#:   ------------------
#:   验收标准是「主控台被强杀后，卡片 **5 秒内**消失」。宽限取 5 的话，
#:   卡片正好在第 5 秒才决定自杀，再加一个轮询间隔和窗口销毁，落地就
#:   变成 5.1s —— **踩线即失败**。留 2s 余量，取 3s。
#:
#:   3s 够短吗？够。宽限要挡的不是「主控台重启」，而是 :func:`pid_alive`
#:   的**瞬时误报**（权限抖动、PID 查询竞争）。那类事件是毫秒级的，
#:   3s 有三个数量级的余量；真正的风险是 PID 被别的进程复用，而那会让
#:   我们**误判为活着**（漏杀），方向正好相反、不会误杀。
#:
#: 想要更长/更短就调配置 —— 但改了记得同步上面那句验收标准。
DEFAULT_GRACE_S = 3.0

#: :meth:`ParentWatchdog.check` 的返回理由
REASON_GRACEFUL = "console_running_false"
REASON_PARENT_DEAD = "parent_dead"


def pid_alive(pid: int) -> bool:
    """PID 对应的进程是否还活着。

    ★ Windows 上**不能**用 ``os.kill(pid, 0)``
    ------------------------------------------
    CPython 的 ``posixmodule.c`` 在 MS_WINDOWS 分支里只特判了
    ``CTRL_C_EVENT`` / ``CTRL_BREAK_EVENT``，其余取值一律
    ``TerminateProcess(handle, sig)``。也就是说在某些 3.x 版本上
    ``os.kill(pid, 0)`` 会**直接把目标杀掉**。

    本机 3.11.16 恰好加了 sig==0 的纯查询分支（实测不会杀），但那是
    **实现细节不是契约**，换个 Python 小版本就可能变成误杀。这里用
    ``OpenProcess`` 自己查，写死行为，不赌 CPython 的实现。

    :returns: 进程存在返回 True；不存在返回 False。查询被拒（权限）时
        返回 **True** —— 宁可当成活着，也绝不误杀。
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - 只在 Windows 分支需要
        from ctypes import wintypes  # noqa: PLC0415

        # ★ 必须 use_last_error=True
        #   -----------------------------------
        #   ``ctypes.windll`` 的默认调用**不会**设置线程的 last error，
        #   于是 ``GetLastError()`` 返回的是**上一次碰巧留下的陈旧值**。
        #   实测踩过：进程已经被 TerminateProcess 掉了，OpenProcess 返回
        #   NULL，陈旧的 err 恰好是 5(ACCESS_DENIED) -> 误判成"还活着"，
        #   卡片就永远不会自杀。用 WinDLL(..., use_last_error=True) 才能
        #   读到本次调用真实的错误码。
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # ★ 显式 argtypes/restypes
        #   ----------------------
        #   ctypes 默认把返回值当 c_int（32 位）。kernel32 的 HANDLE 在
        #   64 位下是 64 位指针，不声明的话高位会被截断 —— 句柄值 32 位
        #   刚好落在低位时截断无害，但这是靠运气，不能留。argtypes 同样
        #   决定 ``None`` 传进 ``LPDWORD`` 时是 NULL 还是空指针：写死
        #   POINTER(DWORD) 之后传 byref(buf) 才有正确的 ABI。
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                         wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE,
                                                ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        ERROR_ACCESS_DENIED = 5
        STILL_ACTIVE = 259

        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                      False, int(pid))
        if not handle:
            err = ctypes.get_last_error()
            # 查不动但进程在（受保护 / 提权差异）-> 当活着，宁可漏杀不误杀
            return err == ERROR_ACCESS_DENIED

        # 已拿到句柄：GetExitCodeProcess 最准 —— 退出的进程立刻带上退出码，
        # STILL_ACTIVE(259) 才表示还活着。
        code = wintypes.DWORD()
        ok = bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code)))
        kernel32.CloseHandle(handle)
        return ok and code.value == STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _read_bridge_console_running(path: Path | None) -> bool | None:
    """读桥接文件里的 ``console.running``。

    :returns: True=主控台在跑 / False=明确写了在退 / None=读不到（别猜）
    """
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    console = data.get("console")
    if not isinstance(console, dict):
        return None
    running = console.get("running")
    return running if isinstance(running, bool) else None


class ParentWatchdog:
    """盯主控台的死活，决定自己要不要退出。

    :param console_pid: 主控台 PID。**None/0 = 手动启动，不上膛**
    :param bridge_path: ``console_bridge.json`` 路径，用于优雅信号
    :param follow_exit: 关掉就完全不设防（用户自己要用常驻卡片时）
    :param grace_s: 崩溃信号的宽限秒数
    """

    def __init__(self, console_pid: int | None = None,
                 bridge_path: Path | None = None,
                 follow_exit: bool = True,
                 grace_s: float = DEFAULT_GRACE_S) -> None:
        self.console_pid = int(console_pid) if console_pid else 0
        self.bridge_path = Path(bridge_path) if bridge_path else None
        self.follow_exit = bool(follow_exit)
        self.grace_s = float(grace_s)
        self._dead_since: float | None = None
        self._last_reason: str | None = None

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, cmd_dir: Path | None, *, follow_exit: bool = True,
                 grace_s: float = DEFAULT_GRACE_S) -> ParentWatchdog:
        """按主控台注入的环境变量自建一个守卫。

        :param cmd_dir: :func:`autopilot.ui.overlay.app.console_cmd_dir` 的结果
        """
        raw = os.environ.get(CONSOLE_PID_ENV, "").strip()
        pid = 0
        if raw.isdigit():
            pid = int(raw)
        bridge = (Path(cmd_dir) / BRIDGE_FILE_NAME) if cmd_dir else None
        return cls(pid, bridge, follow_exit, grace_s)

    # ------------------------------------------------------------------
    @property
    def armed(self) -> bool:
        """是否真的在盯。手动启动 / 用户关掉开关时为 False。"""
        return self.follow_exit and self.console_pid > 0

    def check(self, now: float | None = None) -> str | None:
        """跑一次判定。

        :returns: 该退了就返回理由（:data:`REASON_*`），没该退返回 None
        """
        if not self.armed:
            return None
        if now is None:
            now = _monotonic()

        # ① 优雅信号：主控台亲笔写了 running=false —— 立刻走，不等宽限
        if _read_bridge_console_running(self.bridge_path) is False:
            return self._fire(REASON_GRACEFUL,
                              "主控台已写入 console.running=false")

        # ② 崩溃信号：父进程没了，累计到宽限期才走
        if pid_alive(self.console_pid):
            # 活着的每一轮都要把计时清掉，否则"短暂死一下再活"会累积
            if self._dead_since is not None:
                logger.info("主控台 %d 恢复存活，取消退出计时",
                            self.console_pid)
            self._dead_since = None
            return None

        if self._dead_since is None:
            self._dead_since = now
            # 宽限为 0 = "一发现就走"。单独拎出来是因为通用分支要**再轮**
            # 一次才会 fire（轮询间隔 100ms，差一秒不到，但语义上不该
            # 让"零宽限"还要等下一拍）。
            if self.grace_s <= 0:
                return self._fire(REASON_PARENT_DEAD,
                                  "主控台 %d 已消失（零宽限）"
                                  % self.console_pid)
            logger.warning("主控台 %d 消失，%.1fs 内不回来就自杀",
                           self.console_pid, self.grace_s)
            return None

        waited = now - self._dead_since
        if waited >= self.grace_s:
            return self._fire(REASON_PARENT_DEAD,
                              "主控台 %d 已消失 %.1fs（超过 %.1fs 宽限）"
                              % (self.console_pid, waited, self.grace_s))
        return None

    def _fire(self, reason: str, detail: str) -> str:
        logger.warning("孤儿守卫触发：%s", detail)
        self._last_reason = reason
        return reason

    @property
    def last_reason(self) -> str | None:
        """最近一次触发的理由（没触发过是 None）。纯给日志和单测看。"""
        return self._last_reason


def _monotonic() -> float:
    """单调时钟。``time.time()`` 会被 NTP 校时拽走，宽度判断不能用它。"""
    import time  # noqa: PLC0415

    return time.monotonic()


__all__ = [
    "BRIDGE_FILE_NAME",
    "CONSOLE_PID_ENV",
    "DEFAULT_GRACE_S",
    "REASON_GRACEFUL",
    "REASON_PARENT_DEAD",
    "ParentWatchdog",
    "pid_alive",
]
