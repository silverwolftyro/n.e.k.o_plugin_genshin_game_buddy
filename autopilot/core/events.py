"""事件总线：有界队列 + 后台分发，**永不阻塞主循环**。

为什么必须异步分发
------------------
订阅者将来是 SQLite 写入 / UI 面板 / 通知推送。任何一个在 tick 线程里
同步执行，主循环就会被拖住：

* SQLite 一次 commit 几毫秒，盘慢了是几百毫秒 —— 直接吃掉整个帧预算
* UI 刷新要是同步的，主循环会被窗口重绘的耗时绑在一起
* 某个订阅者抛异常或死循环，主循环跟着完蛋

所以 ``publish()`` 只做一件事：**append 到有界队列 + 置位 Event**。
分发在独立线程做。这是 O(1)、不持锁等待、不会因订阅者慢而堆积。

满了怎么办
----------
丢**最旧**的。这是有意的：新事件比旧事件更有价值 —— 用户关心的是
「刚才发生了什么」，不是三分钟前那次。丢掉时计数 ``dropped``，
绝不静默（原则 1）。

去重
----
主循环会为「同一件事」反复发事件（比如每帧都在说感知不健康）。
``publish`` 接受 ``dedupe_key``：同一个 key 在 ``dedupe_window_s`` 内
只发一次，计数 ``suppressed``。这解决的是「同一事件不重复打」。

⚠ 这不是"日志去重"那种事后处理，而是**在源头就不产生** —— 日志里
    出现 300 条一样的 ERROR 和出现 1 条带计数 300 的 ERROR，
    排障效率差一个量级。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from ..contracts.events import Event

logger = logging.getLogger(__name__)

Subscriber = Callable[[Event], None]


class EventBus:
    """事件总线。

    ::

        bus = EventBus(maxlen=1000)
        bus.subscribe(save_to_sqlite)     # 返回取消订阅的函数
        bus.start()                        # 起分发线程
        ...
        bus.close()
    """

    def __init__(self, *, maxlen: int = 1000, name: str = "event-bus") -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen 必须 >= 1（当前 {maxlen}）")
        self._name = name
        self._maxlen = maxlen
        self._queue: deque[Event] = deque(maxlen=maxlen)
        self._wake = threading.Event()
        self._subscribers: dict[int, tuple[str, Subscriber]] = {}
        self._next_id = 0
        self._lock = threading.RLock()

        self._thread: threading.Thread | None = None
        self._closed = False

        #: key -> 最近一次发布时间（同 key 的去重窗口）
        self._seen: dict[str, float] = {}

        # 计数（排障时先看这些）
        self.published = 0
        self.dropped = 0
        self.suppressed = 0
        self.delivered = 0
        self.subscriber_errors = 0
        self.last_error = ""

    # ==================================================================
    # 订阅
    # ==================================================================

    def subscribe(self, fn: Subscriber, *, name: str = "") -> Callable[[], None]:
        """订阅。返回**取消订阅的函数**（比要求调用方自己 unsubscribe 好用 ——
        忘记退订的订阅者会在总线关闭后继续收事件，静默泄漏）。"""
        if not callable(fn):
            raise TypeError(f"订阅者必须可调用，收到 {type(fn).__name__}")
        with self._lock:
            self._next_id += 1
            sid = self._next_id
            self._subscribers[sid] = (name or getattr(fn, "__name__", "?"), fn)

        def unsubscribe() -> None:
            self.unsubscribe(sid)

        return unsubscribe

    def unsubscribe(self, sid: int) -> None:
        with self._lock:
            self._subscribers.pop(sid, None)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscriber_names(self) -> list[str]:
        with self._lock:
            return [n for n, _ in self._subscribers.values()]

    # ==================================================================
    # 发布（主循环唯一调用的方法，必须快且不阻塞）
    # ==================================================================

    def publish(self, ev: Event, *, dedupe_key: str = "",
                dedupe_window_s: float = 5.0) -> bool:
        """投递一条事件。**永不阻塞、永不抛异常。**

        :param dedupe_key: 非空时，同 key 在窗口内只发一次
        :returns: 是否真的入队（被去重抑制时返回 False）
        """
        if self._closed:
            return False
        if dedupe_key:
            now = time.monotonic()
            last = self._seen.get(dedupe_key)
            if last is not None and (now - last) < dedupe_window_s:
                self.suppressed += 1
                return False
            self._seen[dedupe_key] = now

        # deque(maxlen=) 满了自动丢最旧的 —— 这里不用手写判断
        if len(self._queue) == self._maxlen:
            self.dropped += 1
        self._queue.append(ev)
        self.published += 1
        self._wake.set()
        return True

    # ==================================================================
    # 分发（后台线程）
    # ==================================================================

    def start(self) -> None:
        """起分发线程。**幂等** —— 已经起了就什么都不做。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._closed = False
        self._thread = threading.Thread(
            target=self._dispatch, name=self._name, daemon=True)
        self._thread.start()
        logger.debug("事件总线已启动（容量 %d）", self._maxlen)

    def _dispatch(self) -> None:
        while not self._closed:
            # 空闲时等唤醒（有界超时，保证 close() 能及时退出）
            self._wake.wait(0.25)
            self._wake.clear()
            while True:
                try:
                    ev = self._queue.popleft()
                except IndexError:
                    break
                self._deliver(ev)

    def _deliver(self, ev: Event) -> None:
        with self._lock:
            targets = list(self._subscribers.values())
        for name, fn in targets:
            try:
                fn(ev)
                self.delivered += 1
            except Exception as exc:  # noqa: BLE001 - 一个订阅者挂了不影响别的
                self.subscriber_errors += 1
                self.last_error = f"{name}: {type(exc).__name__}: {exc}"
                logger.error("事件订阅者 %s 异常：%s", name, exc)

    def drain_now(self) -> int:
        """**同步**把队列排空，返回投递条数。

        只给测试用 —— 生产路径靠分发线程。给「跑完 30 秒要立刻把
        统计刷出来」这种收尾场景用。
        """
        n = 0
        while True:
            try:
                ev = self._queue.popleft()
            except IndexError:
                return n
            self._deliver(ev)
            n += 1

    def close(self, timeout: float = 2.0) -> None:
        """停止分发并把**队列里剩下的全部投完**。

        丢事件是不可接受的 —— 用户停掉程序后，最后那几条"场景切换"
        "意图失败"恰恰是最该看到的。
        """
        self._closed = True
        self._wake.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        # 收尾：不管线程状态如何，剩余的都要送出去
        left = self.drain_now()
        if left:
            logger.debug("关闭前补投 %d 条事件", left)
        self._thread = None
        logger.info("事件总线已关闭（发布 %d / 投递 %d / 丢弃 %d / 抑制 %d）",
                    self.published, self.delivered, self.dropped,
                    self.suppressed)

    # ==================================================================
    # 读
    # ==================================================================

    @property
    def pending(self) -> int:
        return len(self._queue)

    def recent(self, n: int = 20) -> list[Event]:
        """最近的 n 条（队尾方向）。"""
        return list(self._queue)[-n:]

    def stats(self) -> dict[str, Any]:
        return {
            "published": self.published,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "suppressed": self.suppressed,
            "subscriber_errors": self.subscriber_errors,
            "pending": len(self._queue),
            "capacity": self._maxlen,
            "subscribers": self.subscriber_names(),
            "last_error": self.last_error,
        }

    # 上下文管理
    def __enter__(self) -> "EventBus":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# 常用订阅者
# --------------------------------------------------------------------------


class LogSubscriber:
    """把事件打到 logging。

    ``dedupe`` 用 ``(code, message)`` 做 key —— 同一条错误连报 300 次时
    只打第一次，其余记在 ``suppressed`` 里。这直接对应验收第 6 条
    「同一事件不重复打」。
    """

    def __init__(self, *, level_warn: int = logging.WARNING,
                 dedupe_window_s: float = 10.0, logger_: logging.Logger
                 | None = None) -> None:
        self._warn = level_warn
        self._window = dedupe_window_s
        self._log = logger_ or logging.getLogger("autopilot.event")
        self.counts: dict[str, int] = {}
        self.printed: dict[str, int] = {}

    def __call__(self, ev: Event) -> None:
        key = f"{ev.code}|{ev.message}"
        self.counts[key] = self.counts.get(key, 0) + 1
        n = self.counts[key]
        if n > 1:
            # 重复只提示一次，且提示带上累计次数
            if self.printed.get(key, 0) == 0:
                self.printed[key] = n
                self._log.warning(
                    "[%s] %s（此后同类事件只计总数，本次为第 %d 次）",
                    ev.code, ev.message, n)
            else:
                self._log.debug("[%s] x%d（已抑制）", ev.code, n)
            return
        self.printed[key] = 1
        lvl = self._log.warning if ev.severity.value in ("WARN", "ERROR") \
            else self._log.info
        lvl("[%s] %s", ev.code, ev.message)

    def summary(self) -> list[tuple[str, int]]:
        return sorted(self.counts.items(), key=lambda x: -x[1])


__all__ = ["EventBus", "LogSubscriber", "Subscriber"]
