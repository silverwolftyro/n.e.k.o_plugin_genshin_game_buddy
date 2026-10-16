"""安全兜底：急停、焦点检查、熔断、动作上限、黑名单。

对应架构文档 6.7 的 SafetyGuard。

**这一层是执行层的最后一道闸**。它不关心"做什么"，只关心"现在允不允许做"。
所有检查**返回具体原因**，不静默返回 False（原则 1）。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ..contracts.events import Event, EventCategory, Severity
from ..contracts.state_packet import StatePacket
from .simulator import VK

logger = logging.getLogger(__name__)

#: 急停键（硬编码，用户要求）
PANIC_KEY = VK.F12

#: 默认会话动作上限
DEFAULT_MAX_ACTIONS = 5000

#: 「硬停机」原因码前缀。
#:
#: ★ 这些前缀是本模块与 executor 之间的**契约**。reason 字符串带变量
#: （如 ``BREAKER_OPEN(剩 1234ms)``），上层**不要拿整个字符串做相等比较** ——
#: 用 `is_hard_stop()` 判前缀，格式改动才不会静默改变上层行为。
REASON_PANIC = "PANIC_KEY_PRESSED"
REASON_BREAKER = "BREAKER_OPEN"


def is_hard_stop(reason: str) -> bool:
    """这个拒绝原因是不是「硬停机」（急停 / 熔断）。

    与「这条意图本身不合法」区分开：硬停机的含义是**整个执行层被叫停**，
    上层应当记 INTERRUPTED 并保留现场，而不是当成一条坏意图丢掉。
    """
    return reason.startswith(REASON_PANIC) or reason.startswith(REASON_BREAKER)


class BreakerState(str, Enum):
    CLOSED = "CLOSED"        # 正常
    OPEN = "OPEN"            # 已熔断，停止执行
    HALF_OPEN = "HALF_OPEN"  # 冷却后试探


@dataclass
class SafetyConfig:
    """安全策略配置。"""

    #: 急停键是否启用
    panic_key_enabled: bool = True
    #: 连续失败多少次后熔断
    breaker_threshold: int = 5
    #: 熔断后多久进入 HALF_OPEN（毫秒）
    breaker_cooldown_ms: int = 30_000
    #: 单次会话总动作上限
    max_actions: int = DEFAULT_MAX_ACTIONS
    #: 同一动作的最小间隔（毫秒），跨意图生效
    rate_limit_ms: int = 50
    #: 是否要求目标窗口在前台
    require_foreground: bool = True
    #: 检查窗口焦点的轮询间隔（秒）—— 每次检查是一次 GetForegroundWindow，很便宜
    foreground_poll_s: float = 0.25

    #: 动作黑名单：set[(scene, action)]
    blacklisted: set[tuple[str, str]] = field(default_factory=set)
    #: 动作白名单之外还额外禁止的（场景无关）
    forbidden_actions: set[str] = field(default_factory=set)


@dataclass
class BreakerTrip:
    """一次熔断的记录。"""

    at_ms: int
    reason: str
    consecutive_failures: int


class SafetyGuard:
    """执行层安全闸。

    典型用法::

        guard = SafetyGuard(cfg, state_version=0)
        verdict = guard.check(intent, state, now_ms)
        if not verdict.ok:
            return                      # ← 带原因，不静默
        ...
        guard.mark_success()
        guard.mark_failure("INPUT_SEND_FAIL")
    """

    def __init__(
        self,
        config: SafetyConfig | None = None,
        state_version: int = 0,
    ) -> None:
        self.cfg = config or SafetyConfig()
        self._version = state_version
        self._breaker = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_ms = 0
        self._trips: deque[BreakerTrip] = deque(maxlen=20)
        self._action_count = 0
        self._last_action_ms: dict[str, int] = {}
        self._panic_pressed = False
        self._fg_cache: tuple[str, float] = ("", 0.0)

    # -- 状态 ------------------------------------------------------------

    @property
    def breaker_state(self) -> BreakerState:
        return self._breaker

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def trips(self) -> list[BreakerTrip]:
        return list(self._trips)

    @property
    def panic_active(self) -> bool:
        """F12 是否被按住。**每次检查动作前都要问这个。**"""
        if not self.cfg.panic_key_enabled:
            return False
        from .simulator import _user32

        try:
            pressed = bool(_user32().GetAsyncKeyState(int(PANIC_KEY)) & 0x8000)
        except Exception:  # noqa: BLE001
            return False
        if pressed:
            self._panic_pressed = True
        elif self._panic_pressed:
            # 松开后自动复位，允许继续（避免一次误按永久锁死）
            self._panic_pressed = False
            logger.info("急停键已松开，恢复执行")
        return pressed

    def reset_panic(self) -> None:
        self._panic_pressed = False

    def reset_breaker(self) -> None:
        self._breaker = BreakerState.CLOSED
        self._consecutive_failures = 0
        logger.info("熔断器已手动复位")

    def reset_session(self) -> None:
        """重置整场会话计数（换任务时调）。"""
        self._action_count = 0
        self._last_action_ms.clear()
        self._breaker = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._panic_pressed = False

    # -- 主检查 ----------------------------------------------------------

    def check(self, intent, state: StatePacket, now_ms: int) -> tuple[bool, str]:
        """执行前的全部检查。**返回 (是否放行, 原因)**。

        顺序是刻意的：先查最致命的（急停/熔断），再查便宜的条件。
        """
        # ① 急停（最致命，任何时候都优先）
        if self.panic_active:
            return False, REASON_PANIC

        # ② 熔断器
        ok, reason = self._check_breaker(now_ms)
        if not ok:
            return False, reason

        # ③ 动作黑名单
        action = intent.action.value
        if action in self.cfg.forbidden_actions:
            return False, f"ACTION_FORBIDDEN:{action}"
        if (state.scene.id, action) in self.cfg.blacklisted:
            return False, f"BLACKLISTED:{state.scene.id}/{action}"

        # ④ 场景许可（traits）
        from .traits import traits_of

        t = traits_of(action)
        if not t.allows_scene(state.scene.id):
            return False, f"SCENE_NOT_ALLOWED:{action}@ {state.scene.id}"

        # ⑤ 焦点
        if (self.cfg.require_foreground or t.requires_foreground) and \
                not self.is_foreground_ok():
            return False, "WINDOW_NOT_FOREGROUND"

        # ⑥ 频率限制
        last = self._last_action_ms.get(action)
        if last is not None and self.cfg.rate_limit_ms > 0:
            gap = now_ms - last
            if gap < self.cfg.rate_limit_ms:
                return False, f"RATE_LIMITED:{action}({gap}<{self.cfg.rate_limit_ms}ms)"
        if t.min_gap_ms > 0 and last is not None and (now_ms - last) < t.min_gap_ms:
            return False, f"MIN_GAP:{action}({now_ms-last}<{t.min_gap_ms}ms)"

        # ⑦ 会话动作上限
        if self._action_count >= self.cfg.max_actions:
            return False, f"MAX_ACTIONS_REACHED({self._action_count})"

        return True, "OK"

    def _check_breaker(self, now_ms: int) -> tuple[bool, str]:
        if self._breaker is BreakerState.OPEN:
            if now_ms - self._last_failure_ms >= self.cfg.breaker_cooldown_ms:
                self._breaker = BreakerState.HALF_OPEN
                logger.warning("熔断器进入 HALF_OPEN，试探一次")
                return True, "HALF_OPEN"
            remain = self.cfg.breaker_cooldown_ms - (now_ms - self._last_failure_ms)
            return False, f"{REASON_BREAKER}(剩 {remain}ms)"
        return True, "OK"

    # -- 焦点 ------------------------------------------------------------

    def set_foreground_title(self, title: str) -> None:
        """告诉 guard 目标窗口的标题子串，用于焦点比对。

        ★ 注意这会把 TTL 缓存的「检查时刻」设成**现在**
          于是紧接着的第一次 :meth:`is_foreground_ok` 必然命中缓存、
          **一次 Win32 都不调**就返回 True。这不是 bug（检查会在 TTL
          过期后照常发生），但意味着**「刚配好标题」的那一次焦点校验
          是假的**。真要立刻校验，得先把缓存时刻设成 0。
        """
        self._fg_cache = (title, time.perf_counter())

    def is_foreground_ok(self) -> bool:
        """目标窗口是否在前台。带短 TTL 缓存 —— 一次检查一次 Win32 调用，
        每帧都查会浪费。

        ⚠️ **这里是整个 guard 里唯一 fail-open 的门**
          （①熔断 ④场景 ⑥频率 全部 fail-closed）。查不到窗口状态时
          返回 True = 「不拦」，于是**按键会被注入到一个我们没能确认
          在前台的窗口**里 —— 用户切到聊天软件时按的 W 会打进聊天框。

          刻意这么选是有道理的：``GetForegroundWindow`` 极少抛，
          而把它 fail-closed 会带来「pywin32 抽风 -> 整个引擎瘫掉」。
          但**至少要留下痕迹** —— 原来这里一行日志都没有，
          导致「这道门自己关掉了」完全不可见（2026-10-13 审计发现）。
        """
        import win32gui  # noqa: PLC0415

        title, checked_at = self._fg_cache
        if not title:
            return True          # 没配标题就不拦
        if time.perf_counter() - checked_at < self.cfg.foreground_poll_s:
            return True
        try:
            fg = win32gui.GetForegroundWindow()
            cur = win32gui.GetWindowText(fg) or ""
        except Exception as exc:  # noqa: BLE001
            # ★ fail-open，但**必须留痕**：否则这道门悄悄失效时没人知道
            logger.warning(
                "焦点校验失败（%s: %s）—— 本次**不拦**，按键可能打进非游戏窗口。"
                "若你在切窗口后遇到过按键打到别处，先看这条日志。",
                type(exc).__name__, exc)
            return True
        self._fg_cache = (title, time.perf_counter())
        return title in cur

    # -- 记账 ------------------------------------------------------------

    def mark_executed(self, action: str, now_ms: int) -> None:
        self._action_count += 1
        self._last_action_ms[action] = now_ms
        if self._breaker is BreakerState.HALF_OPEN:
            self._breaker = BreakerState.CLOSED
            self._consecutive_failures = 0
            logger.info("熔断器试探成功，已闭合")

    def mark_success(self) -> None:
        self._consecutive_failures = 0

    def mark_failure(self, reason: str, now_ms: int | None = None) -> Event | None:
        """记一次失败。达到阈值就熔断并返回事件。"""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        self._consecutive_failures += 1
        self._last_failure_ms = now
        if self._consecutive_failures < self.cfg.breaker_threshold:
            return None
        self._breaker = BreakerState.OPEN
        self._trips.append(
            BreakerTrip(now, reason, self._consecutive_failures)
        )
        logger.error(
            "熔断器开启：连续失败 %d 次，最后原因 %s",
            self._consecutive_failures, reason,
        )
        return Event(
            severity=Severity.ERROR,
            category=EventCategory.EXECUTION,
            code="BREAKER_OPENED",
            message=(
                f"连续 {self._consecutive_failures} 次执行失败，熔断器开启。"
                f"最后原因：{reason}"
            ),
            context={
                "consecutive_failures": self._consecutive_failures,
                "threshold": self.cfg.breaker_threshold,
                "cooldown_ms": self.cfg.breaker_cooldown_ms,
                "last_reason": reason[:200],
                "action_count": self._action_count,
            },
            state_version=self._version,
        )

    # -- 诊断 ------------------------------------------------------------

    def health(self) -> dict:
        return {
            "breaker": self._breaker.value,
            "consecutive_failures": self._consecutive_failures,
            "action_count": self._action_count,
            "max_actions": self.cfg.max_actions,
            "panic_pressed": self._panic_pressed,
            "trips": len(self._trips),
        }


__all__ = [
    "DEFAULT_MAX_ACTIONS",
    "PANIC_KEY",
    "BreakerState",
    "BreakerTrip",
    "SafetyConfig",
    "SafetyGuard",
]
