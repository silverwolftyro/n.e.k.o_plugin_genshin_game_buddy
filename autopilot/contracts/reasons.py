"""拒绝原因的**共享契约**（2026-10-18）。

★ 为什么单独一个模块
--------------------
「拒绝原因」是**执行层（actuator）产出**、**多方消费**的东西：

* ``actuator/safety.py`` —— 产出它
* ``actuator/executor.py`` —— 按它分 outcome
* ``core/stats.py`` —— 按它分类计数（Console 的红牌靠这个）
* ``core/loop.py`` —— 记进事件流
* ``event_to_yui.py`` —— 按它决定**播什么话**

放在 ``actuator/safety.py`` 里的话，``core``（上层）就得 import
``actuator``（下层）—— 方向反了。所以放 ``contracts``：
它本来就是**跨层约定**，不是某一层的实现细节。

★ 前缀 vs 整串
--------------
reason 串带变量::

    WINDOW_NOT_FOREGROUND
    MIN_GAP:USE_PICKUP(205<800ms)
    BREAKER_OPEN(剩 30000ms)
    ACTION_FORBIDDEN:NORMAL_ATTACK

**前缀**是稳定的分类维度，整串不是。所以：
分类/计数/播报一律用前缀；整串只给人看。
"""

from __future__ import annotations

#: 急停键被按下。整个执行层停机。
REASON_PANIC = "PANIC_KEY_PRESSED"

#: 熔断器打开。
REASON_BREAKER = "BREAKER_OPEN"

#: 「用户正在操作，引擎主动让」（2026-10-17）。
#:
#: ★ 这**不是**安全拦截，是**礼让**。
#:   与 ``MIN_GAP`` 同类：动作本来合法，只是「现在不该做」。
#:   读侧必须把它和「安全拦下」分开播报 ——
#:   说「我没敢做」是骗人，引擎是**主动让**的。
REASON_USER_TAKEOVER = "USER_TAKEOVER"

#: 「输出限流」的原因前缀（2026-10-17）。
#:
#: 也是**礼让**性质：动作合法，只是间隔太密。
REASON_THROTTLE_PREFIXES: tuple[str, ...] = ("MIN_GAP", "RATE_LIMITED")

#: 「权限/焦点不足」的原因前缀（2026-10-17）。
#:
#: ★ 这一类的用户可读含义是「**权限不足**」，不是「动作非法」。
#:   ``WINDOW_NOT_FOREGROUND`` 的真实成因有两个，**都**指向权限：
#:     1. 主控台/引擎**不是管理员** -> Windows UIPI 拦下了注入
#:     2. 游戏窗口不在前台 -> 焦点检查主动拒绝
#:   用户看到「没敢做」会以为规则错了；看到「权限不足」才知道该
#:   用管理员身份重跑。
PERMISSION_REASON_PREFIXES: tuple[str, ...] = (
    "WINDOW_NOT_FOREGROUND",
    "UIPI",
    "ACCESS_DENIED",
)


def reason_head(reason: str) -> str:
    """取 reason 的**分类前缀**。

    ``"MIN_GAP:USE_PICKUP(205<800ms)"`` -> ``"MIN_GAP"``
    ``"WINDOW_NOT_FOREGROUND"``          -> ``"WINDOW_NOT_FOREGROUND"``
    """
    if not reason:
        return ""
    return reason.split(":", 1)[0].split("(", 1)[0].strip()


def is_permission_problem(reason: str) -> bool:
    """这个拒绝原因是「权限/焦点不足」吗。"""
    return any(reason.startswith(p) for p in PERMISSION_REASON_PREFIXES)


def is_user_takeover(reason: str) -> bool:
    """这个拒绝原因是「用户接管，引擎主动让」吗。"""
    return reason.startswith(REASON_USER_TAKEOVER)


def is_throttle(reason: str) -> bool:
    """这个拒绝原因是「输出限流」吗。"""
    return any(reason.startswith(p) for p in REASON_THROTTLE_PREFIXES)


__all__ = [
    "REASON_PANIC", "REASON_BREAKER", "REASON_USER_TAKEOVER",
    "REASON_THROTTLE_PREFIXES", "PERMISSION_REASON_PREFIXES",
    "reason_head", "is_permission_problem", "is_user_takeover",
    "is_throttle",
]
