"""窗口查找兜底 + 自诊断（2026-10-10）。

★ 为什么**全部用 ctypes**，不依赖 pywin32 / psutil
------------------------------------------------
决定性证据（2026-10-10 16:25 真机）::

    [窗口普查 timer] 无法导入 win32gui：ModuleNotFoundError: No module named 'win32gui'

插件跑在 **N.E.K.O.exe（Electron）打包出来的运行时**里，那个环境
**没有 pywin32**，而且它是随应用分发的、不可写 —— ``pip install`` 装不进去。
同一台机器上独立跑（``.venv\\Scripts\\python.exe``）却完全正常，于是出现
"独立进程找得到、插件宿主永远找不到"。

ctypes 是标准库，任何 Python 环境都有，**宿主环境再干净也能跑**。
本模块因此不 import 任何第三方包。

★ 为什么它单独一个文件
--------------------
红线 ``test_redline__no_undeclared_third_party_imports`` 只允许根模块
import 标准库 / plugin / __future__ / winrt（``win32gui``、``psutil``
都不在内），而它检查的是 ``node.level == 0``（**绝对**导入）。
:mod:`__init__` 用**相对导入** ``from .window_probe import ...`` 取本模块，
``level > 0`` 不触发；本模块自己也不在 ``PLUGIN_MODULES`` 扫描列表里。
两边都不破线。

★ 这套代码存在的理由
-------------------
``screen_capture.find_genshin_window()`` 的窗口回调整个包在 ``try`` 里，
而它取进程名靠 ``psutil.Process(pid).name()`` —— 宿主里 pywin32 **和** psutil
都没有，于是 ``import`` 就抛，**每个窗口都被 except 跳过**，结果恒 None。
外层只看到"没找到游戏窗口"，与"游戏没开"无法区分。

本模块提供一条**只认标题、零第三方依赖**的查找作为兜底，并配一份用
**宿主自己 logger** 打出的普查——凌晨那版普查写在 ``screen_capture`` 里，
用子 logger，真机日志里**一条都没有**：宿主只给
``plugin.genshin_game_buddy`` 这一个名字装了 handler（今天 17937 行日志里
logger 名只有它一种），子 logger 往 root 冒泡时没人接，记录被吃掉了。
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: 窗口普查里最多记多少个标题样本 —— 够看出规律，又不至于把日志淹了。
_CENSUS_SAMPLE_MAX = 12

#: 每进程只普查一次。宿主里"没找到窗口"是 5s 一次的常态，
#: 不设闸的话普查会跟着每 60s 刷一条，把真正的日志冲掉。
_CENSUS_DONE = False

_ENUM_PROC = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


# --------------------------------------------------------------------------
# ctypes 版 Win32 封装
# --------------------------------------------------------------------------

def _user32():
    return ctypes.windll.user32


def _is_visible(hwnd) -> bool:
    return bool(_user32().IsWindowVisible(hwnd))


def _window_title(hwnd) -> str:
    buf = ctypes.create_unicode_buffer(512)
    _user32().GetWindowTextW(hwnd, buf, 512)
    return buf.value or ""


def _window_pid(hwnd) -> int:
    pid = wintypes.DWORD(0)
    _user32().GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _client_rect_on_screen(hwnd) -> tuple[int, int, int, int]:
    """客户区 -> 屏幕坐标。与 ``capturer._find_window`` 同算法。"""
    rc = _RECT()
    if not _user32().GetClientRect(hwnd, ctypes.byref(rc)):
        return (0, 0, 0, 0)
    pt = _POINT(0, 0)
    _user32().ClientToScreen(hwnd, ctypes.byref(pt))
    return (int(pt.x), int(pt.y), int(rc.right - rc.left), int(rc.bottom - rc.top))


def _process_name(pid: int) -> Any:
    """进程名。**ctypes 版**，不碰 psutil。

    :returns: 名字字符串，或 ``"★取不到 <原因>"``。
        主路径就死在这一步，失败原因必须能被看见。
    """
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    try:
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return "★OpenProcess 失败(errno=%d)" % ctypes.get_last_error()
        try:
            size = wintypes.DWORD(260)
            buf = ctypes.create_unicode_buffer(size.value)
            if not k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return "★QueryFullProcessImageNameW 失败(errno=%d)" % ctypes.get_last_error()
            return buf.value.rsplit("\\", 1)[-1]
        finally:
            k32.CloseHandle(h)
    except Exception as exc:  # noqa: BLE001
        return "★%s: %s" % (type(exc).__name__, exc)


def _enum_visible_titled(
    visit: Callable[[Any, str], None],
) -> None:
    """枚举所有**可见且有标题**的窗口，逐个交给 ``visit(hwnd, title)``。"""
    proc = _ENUM_PROC(lambda hwnd, _lparam: (
        (visit(hwnd, _window_title(hwnd))
         if _is_visible(hwnd) and _window_title(hwnd) else None)
        or True))

    try:
        _user32().EnumWindows(proc, 0)
    finally:
        # 回调对象必须在 EnumWindows 返回前保持存活，否则解释器回收后
        # 段错误。这里显式持有一份引用直到枚举结束。
        del proc


# --------------------------------------------------------------------------
# 对外
# --------------------------------------------------------------------------

def find_window_by_title(title_substr: str = "原神") -> Optional[dict]:
    """只按**窗口标题子串**找游戏窗口。**零第三方依赖**。

    返回 ``{"hwnd", "pid", "title", "rect"}``，找不到返回 ``None``。

    :param title_substr: 标题子串。原神窗口标题就是「原神」。
    """
    found: list[tuple[Any, tuple[int, int, int, int], str]] = []

    def _visit(hwnd, title):
        if title_substr in title:
            found.append((hwnd, _client_rect_on_screen(hwnd), title))

    try:
        _enum_visible_titled(_visit)
    except Exception:  # noqa: BLE001
        return None
    if not found:
        return None
    # 取面积最大的 —— 避免命中工具提示之类的小窗（与引擎侧同策略）
    hwnd, rect, title = max(found, key=lambda t: t[1][2] * t[1][3])
    return {"hwnd": int(hwnd), "pid": _window_pid(hwnd),
            "title": title, "rect": rect}


def log_window_census(log: logging.Logger, trigger: str,
                      title_substr: str = "原神") -> bool:
    """打一份窗口普查。**用调用方给的 logger**，不自己造。进程内只打一次。

    覆盖三层：窗口枚举总数 / 逐窗口判定 / **进程名取不取得到**。
    第三层是本次事故的关键 —— 主路径就死在 psutil/pywin32 上。

    :returns: 实际是否打了普查（False = 本进程已打过）。
    """
    global _CENSUS_DONE
    if _CENSUS_DONE:
        return False
    _CENSUS_DONE = True

    stats = {"total": 0, "visible": 0, "titled": 0, "matched": 0}
    samples: list[str] = []
    procname: Any = "<没跑到标题命中的窗口>"

    def _visit(hwnd, title):
        nonlocal procname
        stats["total"] += 1
        stats["visible"] += 1          # 枚举器只回调"可见且有标题"的
        stats["titled"] += 1
        hit = title_substr in title
        if hit:
            stats["matched"] += 1
        if len(samples) < _CENSUS_SAMPLE_MAX:
            samples.append(("★" if hit else "") + title[:34])
        if hit and isinstance(procname, str):
            # 主路径就死在这一步：它取进程名，再拿名字匹配游戏。
            procname = _process_name(_window_pid(hwnd))

    try:
        _enum_visible_titled(_visit)
    except Exception as exc:  # noqa: BLE001
        log.warning("genshin_game_buddy: [窗口普查 %s] 枚举失败：%s: %s",
                    trigger, type(exc).__name__, exc)
        return True

    log.warning(
        "genshin_game_buddy: [窗口普查 %s] 可见有标题=%d 标题命中「%s」=%d "
        "标题样本=%r 命中窗口进程名=%r python=%r",
        trigger, stats["titled"], title_substr, stats["matched"],
        samples, procname, sys_executable())
    return True


def sys_executable() -> str:
    """当前解释器路径。**排查宿主问题必需** ——
    2026-10-10 整场排查最费劲的就是"宿主到底用的哪个 Python"，
    而宿主日志里从不打印它。"""
    import sys  # noqa: PLC0415
    return sys.executable


__all__ = ["find_window_by_title", "log_window_census", "sys_executable"]