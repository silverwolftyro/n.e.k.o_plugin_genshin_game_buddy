"""genshin_game_buddy · 游戏生命周期守卫（前台检测 → 感知栈休眠）。

职责
----
用一个**极轻量**的后台线程判断「原神现在是不是在前台跑」，把结论喂给
感知栈（YOLO 推理循环 / OCR 轮询）：

* 游戏进程**不存在** → 挂起感知栈（不再抓屏、不再推理、不再 OCR）
* 游戏进程在但**不在前台**（最小化 / 被别的窗口盖住 / Alt-Tab 走了）
  → 挂起感知栈，**除非**用户显式打开了「允许后台运行」
* 游戏回到前台 → **自动唤醒**，感知栈下一拍就恢复

★ 为什么需要它（不是重复造轮子）
  插件本来就有一条感知状态机（``UNKNOWN → IN_GAME → AWAY → OUT_OF_GAME``），
  但它是**从 OCR 结果反推**的：判断「你不在游戏里」必须先做一次 OCR。
  于是「游戏已经关了」这件事，要等 3 次 miss（AWAY）+ 3 次 miss（OUT_OF_GAME）
  才停 —— 拍板 2.0.68/2.0.71 的档位下就是 **12 秒左右的空转**：
  抓屏 + OCR 全在跑，GPU/CPU 白烧。

  本守卫改成**进程级主动探测**：看一眼操作系统就知道游戏在不在，
  不用等 OCR 兜底，**零延迟**挂起。

★ 为什么仍然保留低功耗 Webhook 监听
  BetterGI 的事件通道是**独立线程**（``bettergi.webhook_server``），
  感知栈休眠时它照常监听 —— 否则「游戏刚退出，BetterGI 报了最后一条事件」
  就会丢，用户反而看不到收尾台词。

零依赖
------
* 进程枚举：**psutil 优先**，没有就用 Win32 ``CreateToolhelp32Snapshot``。
* 前台窗口：``GetForegroundWindow`` + ``GetWindowThreadProcessId``（ctypes）。

★ 为什么 psutil 走「可选」而不是写进 ``pyproject.toml``
  插件的第三方依赖必须 vendor 进包里（``validate_cmd`` 会硬校验），
  而 psutil 有 4 MB+ 且宿主运行时**本来就带**（引擎自己就用它做进程管理）。
  写成硬依赖只会让包白白变大；写成软依赖则「有就用，没有就用 Win32」，
  两条路都通。
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from typing import Callable, Dict, Iterable, Optional, Set, Tuple

__all__ = [
    "LifecycleGuard",
    "STATE_ABSENT",
    "STATE_BACKGROUND",
    "STATE_FOREGROUND",
    "STATE_UNKNOWN",
    "GAME_PROCESSES",
    "DEFAULT_POLL_INTERVAL_S",
]

logger = logging.getLogger(__name__)

#: 游戏进程不在（压根没启动）
STATE_ABSENT = "ABSENT"
#: 游戏进程在，但窗口不在前台（最小化 / 被遮挡 / Alt-Tab 走了）
STATE_BACKGROUND = "BACKGROUND"
#: 游戏窗口在前台
STATE_FOREGROUND = "FOREGROUND"
#: 探测失败（权限不足 / 非 Windows / 出错）—— **fail-open，不挂起**
STATE_UNKNOWN = "UNKNOWN"

#: 受守卫管理的游戏 -> 其进程名（小写）。
#:
#: ★ 为什么要按 game_id 分组而不是只盯一个进程
#:   本插件是多游戏的（原神 / 鸣潮 / …）。只认原神进程的话，
#:   玩鸣潮时守卫会说「原神没开 → 挂起」，把鸣潮的感知也一起干掉。
#:   ⇒ 只有**当前会话所属的游戏**在表里，守卫才对它生效。
GAME_PROCESSES: Dict[str, Tuple[str, ...]] = {
    "genshin": ("yuanshen.exe", "genshinimpact.exe"),
}

#: 探测周期（秒）。默认 5 秒：一次进程枚举 + 一次前台查询，耗时在毫秒级，
#: 5 秒一次对 CPU 完全没有可测量的影响，但「关掉游戏后多久挂起」只差 5 秒。
DEFAULT_POLL_INTERVAL_S = 5.0


# ── Win32 前台查询 ──────────────────────────────────────────────────────

def _foreground_pid() -> Optional[int]:
    """返回当前前台窗口所属的进程 id；取不到返回 ``None``。

    ★ 为什么要「前台窗口的 pid」而不是「进程是不是前台」
      Windows 没有「前台进程」这个概念，只有「前台**窗口**」。
      所以标准做法是：拿前台窗口句柄 → 反查它属于哪个进程。
    """
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        pid = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value) or None
    except Exception:  # noqa: BLE001 - 取不到就当未知，fail-open
        return None


def _iter_process_names_windows() -> Iterable[Tuple[int, str]]:
    """用 Win32 ``CreateToolhelp32Snapshot`` 枚举 ``(pid, exe 名)``。

    这是 psutil 缺席时的兜底路径：只用 ctypes，不需要任何第三方包。
    """
    if sys.platform != "win32":
        return
    try:
        from ctypes import wintypes

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32FirstW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        ]
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        ]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        TH32CS_SNAPPROCESS = 0x00000002
        INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

        snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snapshot == INVALID_HANDLE_VALUE or not snapshot:
            return
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                return
            while True:
                yield int(entry.th32ProcessID), str(entry.szExeFile).lower()
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
        finally:
            kernel32.CloseHandle(snapshot)
    except Exception:  # noqa: BLE001 - 枚举失败就当没找到，fail-open
        return


def _find_processes(names: Iterable[str]) -> Set[int]:
    """找出名字命中 ``names``（小写）的进程 id 集合。

    优先用 psutil（更快、信息更全），没有就退回 Win32 toolhelp。
    """
    wanted = {n.lower() for n in names}
    if not wanted:
        return set()

    try:
        import psutil  # noqa: PLC0415 - 可选依赖，装了才用
    except Exception:  # noqa: BLE001 - 没装就走 Win32 兜底
        psutil = None  # type: ignore[assignment]

    if psutil is not None:
        found: Set[int] = set()
        try:
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    name = (proc.info.get("name") or "").lower()
                except Exception:  # noqa: BLE001 - 进程可能已经退了
                    continue
                if name in wanted:
                    found.add(int(proc.info["pid"]))
        except Exception:  # noqa: BLE001 - 权限不足等
            return _find_processes_win32(wanted)
        return found

    return _find_processes_win32(wanted)


def _find_processes_win32(wanted: Set[str]) -> Set[int]:
    """Win32 toolhelp 版进程查找。"""
    found: Set[int] = set()
    for pid, name in _iter_process_names_windows():
        if name in wanted:
            found.add(pid)
    return found


class LifecycleGuard:
    """原神进程 / 前台状态守卫。

    用法::

        guard = LifecycleGuard(logger=logger, active_game_fn=lambda: "genshin")
        guard.start()
        ...
        if guard.suspended:
            return          # 感知栈跳过本轮
        ...
        guard.stop()
    """

    def __init__(
        self,
        *,
        logger_: Any = None,
        active_game_fn: Optional[Callable[[], str]] = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        allow_background: bool = False,
        game_processes: Optional[Dict[str, Tuple[str, ...]]] = None,
        on_state_change: Optional[Callable[[str, str], None]] = None,
        probe_fn: Optional[Callable[[], str]] = None,
    ) -> None:
        """
        :param logger_: 可选 logger。
        :param active_game_fn: 返回当前会话的 game_id（没有会话返回 ``""``）。
            只有它在 :data:`GAME_PROCESSES` 表里，守卫才对当前游戏生效。
        :param poll_interval_s: 探测周期（秒）。
        :param allow_background: 游戏在后台时**不**挂起感知栈。
        :param game_processes: 覆盖默认的游戏 → 进程名表（测试用）。
        :param on_state_change: 状态迁移回调 ``(旧状态, 新状态)``。
        :param probe_fn: 覆盖整个探测过程（测试用；返回状态字符串）。
        """
        self.logger = logger_ or logger
        self._active_game_fn = active_game_fn
        self._poll_interval_s = max(0.5, float(poll_interval_s))
        self._allow_background = bool(allow_background)
        self._game_processes = game_processes or GAME_PROCESSES
        self._on_state_change = on_state_change
        self._probe_fn = probe_fn

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._state = STATE_UNKNOWN
        self._last_probe_at = 0.0
        self._probe_count = 0
        self._suspended = False

    # ── 配置 ──────────────────────────────────────────────────────────
    @property
    def allow_background(self) -> bool:
        """游戏在后台时是否仍保持感知栈运行。"""
        return self._allow_background

    def set_allow_background(self, value: bool) -> None:
        """运行期改「允许后台运行」（配置面板热更新用）。

        ★ 必须**立刻**重算挂起状态
          改了开关却要等下一个探测周期（默认 5s）才生效的话，用户在面板上
          勾了「允许后台运行」，感知栈却还停着 —— 看起来就像开关没用。
          所以这里就地按当前状态重算一次。
        """
        with self._lock:
            self._allow_background = bool(value)
            self._suspended = self._compute_suspended(self._state)
        self._log("info", "bettergi lifecycle guard: allow_background=%s", value)

    def set_poll_interval(self, seconds: float) -> None:
        """运行期改探测周期。"""
        with self._lock:
            self._poll_interval_s = max(0.5, float(seconds))

    # ── 状态 ──────────────────────────────────────────────────────────
    @property
    def state(self) -> str:
        """最近一次探测到的状态。"""
        with self._lock:
            return self._state

    @property
    def suspended(self) -> bool:
        """感知栈**现在应该挂起**吗。

        这是感知栈唯一要问的问题。``True`` 表示：不该抓屏、不该推理、不该 OCR。
        """
        with self._lock:
            return self._suspended

    @property
    def running(self) -> bool:
        """守卫线程是否在跑。"""
        return self._thread is not None and self._thread.is_alive()

    # ── 生命周期 ──────────────────────────────────────────────────────
    def start(self) -> None:
        """启动后台探测线程（幂等）。

        ★ 第一次探测是**同步**做的：这样插件刚起来就知道当前状态，
          不用等第一个周期才生效。

        ★ 但同步探测**必须**包在 try/except 里
          探测一旦抛异常就直接冒泡的话，守卫线程根本起不来 ——
          而「守卫起不来」在 fail-open 语义下应该是「不挂起、继续跑」，
          不是「插件启动失败」。所以异常一律吞成 ``STATE_UNKNOWN``。
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            # 同步先探一次，避免「起来后 5 秒内还在空转」
            try:
                self._state = self._probe()
            except Exception as exc:  # noqa: BLE001 - 探测失败 = 不知道 = 不挂起
                self._log("warning", "bettergi lifecycle guard: first probe failed: %r", exc)
                self._state = STATE_UNKNOWN
            self._suspended = self._compute_suspended(self._state)
            self._last_probe_at = time.monotonic()
            self._probe_count = 1
            thread = threading.Thread(
                target=self._loop,
                daemon=True,
                name="neko-lifecycle-guard",
            )
            self._thread = thread
        thread.start()
        self._log(
            "info",
            "bettergi lifecycle guard: started state=%s suspended=%s poll=%.1fs",
            self.state,
            self.suspended,
            self._poll_interval_s,
        )

    def stop(self) -> None:
        """停止后台探测线程（幂等）。"""
        with self._lock:
            thread, self._thread = self._thread, None
            self._stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._log("info", "bettergi lifecycle guard: stopped")

    # ── 探测 ──────────────────────────────────────────────────────────
    def _loop(self) -> None:
        """后台线程主体：睡 → 探 → 睡 → 探。

        ★ 用 ``Event.wait`` 而不是 ``time.sleep``
          前者能被打断（``stop()`` 立刻返回），后者要多等一个周期。
        """
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._poll_interval_s):
                break
            try:
                state = self._probe()
            except Exception as exc:  # noqa: BLE001 - 探测失败不能拖垮插件
                self._log("warning", "bettergi lifecycle guard: probe failed: %r", exc)
                continue
            self._apply_state(state)

    def _apply_state(self, state: str) -> None:
        """把新状态写进去，变了就通知。"""
        with self._lock:
            previous = self._state
            suspended = self._compute_suspended(state)
            self._state = state
            self._suspended = suspended
            self._last_probe_at = time.monotonic()
            self._probe_count += 1
        if previous != state:
            self._log(
                "info",
                "bettergi lifecycle guard: state %s -> %s (suspended=%s)",
                previous,
                state,
                suspended,
            )
            if self._on_state_change is not None:
                try:
                    self._on_state_change(previous, state)
                except Exception as exc:  # noqa: BLE001 - 回调失败不影响守卫
                    self._log(
                        "warning",
                        "bettergi lifecycle guard: on_state_change failed: %r",
                        exc,
                    )

    def _probe(self) -> str:
        """探测一次，返回状态。

        ★ fail-open 原则
          任何探测失败都返回 :data:`STATE_UNKNOWN`，而 ``STATE_UNKNOWN``
          **不挂起**感知栈 —— 守卫失灵时宁可多烧一点 CPU，
          也不能让插件整个哑掉。
        """
        if self._probe_fn is not None:
            return str(self._probe_fn())

        game_id = self._current_game_id()
        names = self._game_processes.get(game_id)
        if not names:
            # 当前会话的游戏不在守卫表里（多游戏插件的常态）→ 不管它。
            return STATE_UNKNOWN

        pids = _find_processes(names)
        if not pids:
            return STATE_ABSENT

        fg_pid = _foreground_pid()
        if fg_pid is not None and fg_pid in pids:
            return STATE_FOREGROUND
        return STATE_BACKGROUND

    def _current_game_id(self) -> str:
        """取当前会话的 game_id（取不到返回空串）。"""
        if self._active_game_fn is None:
            return ""
        try:
            return str(self._active_game_fn() or "").strip().lower()
        except Exception:  # noqa: BLE001 - 取不到就当没有会话
            return ""

    def _compute_suspended(self, state: str) -> bool:
        """按状态算出「该不该挂起感知栈」。

        规则：
        * ``ABSENT`` → 挂起（游戏没开）
        * ``BACKGROUND`` → 挂起，**除非**开了「允许后台运行」
        * ``FOREGROUND`` / ``UNKNOWN`` → 不挂起
        """
        if state == STATE_ABSENT:
            return True
        if state == STATE_BACKGROUND:
            return not self._allow_background
        return False

    # ── 辅助 ──────────────────────────────────────────────────────────
    def _log(self, level: str, message: str, *args: Any) -> None:
        log = getattr(self.logger, level, None)
        if log is None:
            return
        try:
            log(message, *args)
        except Exception:  # pragma: no cover
            pass
