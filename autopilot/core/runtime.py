"""主循环运行时状态。

⚠ 刻意**不叫 state.py** —— 仓库根已经有 ``autopilot/state.py``
（WorldState 快照）。两个都叫 state.py 时，IDE 跳转、
``from .state import ...`` 的自我怀疑、以及「到底是哪个 state」这种
三分钟的困惑都是白给。

    autopilot/state.py          WorldState  —— 世界的一次快照（不可变数据）
    autopilot/core/runtime.py   RuntimeState —— 程序跑到哪一步了（控制流）

一个描述「世界长什么样」，一个描述「程序在干什么」。混在一起就没法
分开测试：想测世界快照不需要启循环，想测状态机不需要造假数据。

**为什么状态机必须是显式的**
----------------------------
「跑 / 停 / 暂停」如果只用 ``while self._running`` 表达，会出现这些烂摊子：

* 按了暂停但当前 tick 正在执行按键 —— 用户以为停了，其实这一帧还在按
* 停止要等当前 tick 跑完 —— 而 MOVE_FORWARD 的 traits 有 60s 上限，
  用户以为没响应，其实卡在一个 60 秒的 tick 里
* 急停和 stop 混为一谈 —— 前者要立刻松键，后者要走完收尾

所以显式五态，每个态只允许特定的转移，越界的转移直接抛。
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any

from ..state import ConnectionState, WorldState, WorldStateStore
from .events import EventBus
from .stats import LoopStats


class RuntimeState(str, Enum):
    """主循环状态。

    合法转移::

        IDLE     -> RUNNING   （start）
        RUNNING  -> PAUSED    （pause）
        RUNNING  -> STOPPING  （stop / F12）
        PAUSED   -> RUNNING   （resume）
        PAUSED   -> STOPPING  （stop / F12）
        STOPPING -> STOPPED   （收尾完成）
        STOPPED  -> IDLE      （reset）
    """

    IDLE = "IDLE"          # 已构造，未启动
    RUNNING = "RUNNING"    # tick 中，全链路可执行
    PAUSED = "PAUSED"      # 用户暂停：感知继续，执行层不发按键
    STOPPING = "STOPPING"  # 正在收尾：松键、关截图器
    STOPPED = "STOPPED"    # 已停止，可 reset 重来

    @property
    def accepts_input(self) -> bool:
        """这个状态下**允许**发送输入吗。

        ★ PAUSED 也返回 False —— 暂停的意思是「别碰我的游戏」。
          若暂停期间还在按键，那叫「慢速运行」，不是暂停。
        """
        return self is RuntimeState.RUNNING

    @property
    def runs_perception(self) -> bool:
        """这个状态下**继续感知**吗。

        ★ PAUSED 返回 True 是刻意的：暂停时用户可能切回桌面干别的，
          几分钟后想恢复 —— 那时如果感知也停了，恢复后第一帧是陈旧的，
          会基于旧状态做决策。所以「暂停」只停执行，不停眼睛。
        """
        return self in (RuntimeState.RUNNING, RuntimeState.PAUSED)

    @property
    def alive(self) -> bool:
        return self in (RuntimeState.RUNNING, RuntimeState.PAUSED,
                        RuntimeState.STOPPING)


#: 合法转移表。**不在表里的转移直接抛** —— 状态机的价值就在于
#: 「非法转移会立刻暴露」，而不是让它悄悄发生。
_ALLOWED: dict[RuntimeState, frozenset[RuntimeState]] = {
    RuntimeState.IDLE: frozenset({RuntimeState.RUNNING}),
    RuntimeState.RUNNING: frozenset({
        RuntimeState.PAUSED, RuntimeState.STOPPING}),
    RuntimeState.PAUSED: frozenset({
        RuntimeState.RUNNING, RuntimeState.STOPPING}),
    RuntimeState.STOPPING: frozenset({RuntimeState.STOPPED}),
    RuntimeState.STOPPED: frozenset({RuntimeState.IDLE}),
}


class IllegalTransition(RuntimeError):
    """非法的状态转移。**响亮失败**，不做静默忽略。"""


def check_transition(src: RuntimeState, dst: RuntimeState) -> None:
    """校验转移合法性，不合法就抛。"""
    if dst in _ALLOWED.get(src, frozenset()):
        return
    raise IllegalTransition(
        f"非法状态转移 {src.value} -> {dst.value}。"
        f"{src.value} 只能转到 "
        f"{sorted(s.value for s in _ALLOWED.get(src, frozenset()))}"
    )


# --------------------------------------------------------------------------
# 运行时上下文
# --------------------------------------------------------------------------


class RuntimeContext:
    """三层实例 + 世界状态 + 统计 + 事件总线。

    刻意做成**可注入的容器**而不是全局单例：主循环因此可以在测试里
    塞假的 capturer / policy / executor 跑完整流程，不碰实机
    （原则 4：工具 = 运行时）。

    与统计/总线分开成 ``stats.py`` / ``events.py``，是为了让这两个
    可被单独测试 —— 它们是纯聚合与纯投递，不该和"循环怎么跑"混在一起。
    """

    def __init__(
        self,
        *,
        capturer: Any = None,
        builder: Any = None,
        policy: Any = None,
        executor: Any = None,
        guard: Any = None,
        bus: EventBus | None = None,
        store: WorldStateStore | None = None,
        stats: LoopStats | None = None,
        connection: ConnectionState | None = None,
        tick_rate_hz: float = 5.0,
        max_consecutive_capture_failures: int = 10,
    ) -> None:
        # 三层实例
        self.capturer = capturer
        self.builder = builder
        self.policy = policy
        self.executor = executor
        self.guard = guard

        # 聚合
        self.store = store or WorldStateStore()
        self.stats = stats or LoopStats()
        self.bus = bus or EventBus()
        self.connection = connection or ConnectionState()

        # ── 节奏 ───────────────────────────────────────────────
        #: tick 目标频率（Hz）。默认 **5 Hz（200ms 一帧）**。
        #: 感知层单帧 30-80ms，10 Hz 会让 60% 时间花在想怎么凑间隔上，
        #: 而且决策层对 100ms 内重复的动作没有额外价值（规则有 cooldown）。
        #: 战斗场景想更快就调高 —— 接口已留好。
        self.tick_rate_hz = float(tick_rate_hz)
        if self.tick_rate_hz < 0:
            raise ValueError(f"tick_rate_hz 不能为负（{tick_rate_hz}）")

        #: 感知层连续失败多少次后自动暂停。**不是无限重试** ——
        #: 游戏窗口被关掉时继续重试只会刷屏，而且用户以为程序还活着。
        self.max_consecutive_capture_failures = max_consecutive_capture_failures

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    @property
    def target_interval_s(self) -> float:
        """目标 tick 间隔（秒）。0 = 不限速。"""
        if self.tick_rate_hz <= 0:
            return 0.0
        return 1.0 / self.tick_rate_hz

    def set_tick_rate(self, hz: float) -> None:
        """运行时调频率（战斗场景提速用）。"""
        if hz < 0:
            raise ValueError(f"tick_rate_hz 不能为负（{hz}）")
        self.tick_rate_hz = float(hz)

    # -- 世界状态 --------------------------------------------------------
    def world(self) -> WorldState:
        return self.store.get()

    def current_state(self):
        return self.store.get().current_state

    # -- 关闭底层资源 ----------------------------------------------------
    def close_resources(self) -> dict[str, bool]:
        """优雅关闭：截图器 / 执行器 / 事件总线。

        ★ 顺序有讲究：先停总线（不再接新事件）→ 再释放按键 →
          最后关截图器。反过来的话关截图器时可能还有事件在发。
        """
        result: dict[str, bool] = {}
        result["bus"] = self._close_bus()
        result["release"] = self._release_keys()
        result["capturer"] = self._close_capturer()
        return result

    def _close_bus(self) -> bool:
        try:
            self.bus.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _release_keys(self) -> bool:
        sim = getattr(self.executor, "sim", None)
        release = getattr(sim, "release_all", None)
        if release is None:
            return True
        try:
            release()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _close_capturer(self) -> bool:
        close = getattr(self.capturer, "close", None)
        if close is None:
            return True
        try:
            close()
            return True
        except Exception:  # noqa: BLE001
            return False


__all__ = [
    "IllegalTransition",
    "RuntimeContext",
    "RuntimeState",
    "check_transition",
]
