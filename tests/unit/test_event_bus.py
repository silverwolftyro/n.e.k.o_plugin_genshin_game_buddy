"""``core/events.py`` 事件总线的单测（2026-10-12 任务 C）。

总线是感知→决策→执行三层与 YUI/日志之间的唯一出口，它的三条性质是
**安全边界**，必须钉死：

1. **publish 永不阻塞、永不抛** —— 它在 5Hz 主循环里被调用，
   一次阻塞就是按键延迟，一次抛异常就是整条 tick 崩。
2. **去重计数必须与真实次数分开** —— 10-11 的教训：有人拿去重后的
   事件条数当真实次数，得出「闸门没生效」的错误结论。
3. **close() 必须把队列里剩的全投完** —— 用户停掉程序后，
   最后那几条「场景切换」「意图失败」恰恰是最该看到的。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.contracts.events import (  # noqa: E402
    Event,
    EventCategory,
    Severity,
)
from autopilot.core.events import EventBus, LogSubscriber  # noqa: E402


def _ev(code: str = "TEST", sev: str = "INFO", msg: str = "x") -> Event:
    # ⚠️ ``context`` **没有默认值** —— 契约要求每条事件都带上下文。
    return Event(severity=Severity(sev), category=EventCategory.SYSTEM,
                 code=code, message=msg, context={})


# ==================================================================
# 订阅
# ==================================================================

@pytest.mark.unit
def test_bus__subscriber_receives_published_events() -> None:
    bus, seen = EventBus(), []
    bus.subscribe(seen.append, name="collector")
    bus.publish(_ev())
    assert bus.drain_now() == 1
    assert len(seen) == 1


@pytest.mark.unit
def test_bus__subscribe_returns_a_working_unsubscribe() -> None:
    """返回值是取消订阅的**闭包** —— 拿它退订比记 sid 更不容易出错。

    ⚠️ 注意投递顺序：退订发生在**投递之前**时，队列里那条也不会被送达
    （订阅者已从快照里消失）。所以要断言「退订后不再收到」，
    必须先 drain 掉之前那条 —— 否则测的是另一件事。
    """
    bus, seen = EventBus(), []
    off = bus.subscribe(seen.append, name="c")
    bus.publish(_ev())
    bus.drain_now()
    assert len(seen) == 1

    off()
    bus.publish(_ev())
    bus.drain_now()
    assert len(seen) == 1, "退订后不该再收到"


@pytest.mark.unit
def test_bus__unsubscribe_twice_is_harmless() -> None:
    """退订钩子可能被调两次（异常路径 + 正常路径）。"""
    bus = EventBus()
    off = bus.subscribe(lambda e: None, name="c")
    off()
    off()


@pytest.mark.unit
def test_bus__unsubscribe_by_sid_tolerates_an_unknown_id() -> None:
    """``subscribe`` 返回的是闭包而不是 sid，所以按 sid 退订只可能拿到外部
    存下来的数字（跨进程重连时）。传一个不存在的 sid 必须**无害**，不能抛。"""
    bus, seen = EventBus(), []
    bus.subscribe(seen.append, name="c")
    bus.unsubscribe(999999)
    assert bus.subscriber_count == 1, "退订不存在的 id 不该误伤别人"
    bus.unsubscribe(999999)


@pytest.mark.unit
def test_bus__subscriber_count_tracks_live_subscriptions() -> None:
    # ⚠️ subscriber_count 是 @property，不是方法 —— 加括号会得到
    #    "'int' object is not callable"
    bus = EventBus()
    assert bus.subscriber_count == 0
    off = bus.subscribe(lambda e: None, name="c")
    assert bus.subscriber_count == 1
    off()
    assert bus.subscriber_count == 0


@pytest.mark.unit
def test_bus__subscriber_names_are_listed() -> None:
    """名字是给日志/诊断看的 —— 「哪个订阅者挂了」要能一眼看出。"""
    bus = EventBus()
    bus.subscribe(lambda e: None, name="alpha")
    bus.subscribe(lambda e: None, name="beta")
    assert sorted(bus.subscriber_names()) == ["alpha", "beta"]


# ==================================================================
# 投递隔离
# ==================================================================

@pytest.mark.unit
def test_bus__one_broken_subscriber_does_not_block_the_others() -> None:
    """★ 一个订阅者挂了，其余的照收。"""
    bus, seen = EventBus(), []

    def boom(_e):
        raise RuntimeError("挂了")

    bus.subscribe(boom, name="bad")
    bus.subscribe(seen.append, name="good")
    bus.publish(_ev())
    assert bus.drain_now() == 1
    assert len(seen) == 1


@pytest.mark.unit
def test_bus__records_the_last_subscriber_error() -> None:
    """出错了要留痕 —— 否则排查时只看到「事件没到」。"""
    def boom(_e):
        raise ValueError("坏订阅者")

    bus = EventBus()
    bus.subscribe(boom, name="bad")
    bus.publish(_ev())
    bus.drain_now()
    st = bus.stats()
    assert st["subscriber_errors"] == 1
    assert "bad" in st["last_error"]


@pytest.mark.unit
def test_bus__delivery_iterates_a_snapshot_of_subscribers() -> None:
    """订阅者在回调里退订自己，不该让迭代炸掉。"""
    bus = EventBus()
    box = {}

    def once(_e):
        box["off"]()

    box["off"] = bus.subscribe(once, name="once")
    bus.publish(_ev())
    assert bus.drain_now() == 1
    bus.publish(_ev())
    bus.drain_now()          # 不抛就是过


# ==================================================================
# 去重
# ==================================================================

@pytest.mark.unit
def test_bus__dedupe_key_suppresses_within_the_window() -> None:
    bus, seen = EventBus(), []
    bus.subscribe(seen.append, name="c")
    assert bus.publish(_ev(), dedupe_key="k") is True
    assert bus.publish(_ev(), dedupe_key="k") is False
    bus.drain_now()
    assert len(seen) == 1


@pytest.mark.unit
def test_bus__dedupe_window_is_configurable() -> None:
    """窗口是调用方说了算 —— 10-11 那次就是默认 5s 造成了误读。"""
    bus = EventBus()
    assert bus.publish(_ev(), dedupe_key="k") is True
    time.sleep(0.05)
    assert bus.publish(_ev(), dedupe_key="k", dedupe_window_s=0.01) is True


@pytest.mark.unit
def test_bus__different_keys_do_not_suppress_each_other() -> None:
    bus = EventBus()
    assert bus.publish(_ev(), dedupe_key="a") is True
    assert bus.publish(_ev(), dedupe_key="b") is True


@pytest.mark.unit
def test_bus__no_key_means_everything_gets_through() -> None:
    """不给 key 就是「每条都要发」—— 关键事件不能被静默吞掉。"""
    bus = EventBus()
    for _ in range(5):
        assert bus.publish(_ev()) is True


@pytest.mark.unit
def test_bus__suppressed_count_is_tracked_separately() -> None:
    """★ 10-11 的教训：被抑制的条数必须能被单独读出来。

    有人拿去重后的条数当真实次数，误判「min_gap 没生效」。
    计数器分开，统计才不会骗人。
    """
    bus = EventBus()
    bus.publish(_ev(), dedupe_key="k")
    for _ in range(4):
        bus.publish(_ev(), dedupe_key="k")
    st = bus.stats()
    assert st["published"] == 1
    assert st["suppressed"] == 4


# ==================================================================
# 容量
# ==================================================================

@pytest.mark.unit
def test_bus__queue_is_bounded_and_drops_the_oldest() -> None:
    """满了丢最旧的 —— 但**必须记账**，否则丢事件不可见。"""
    bus = EventBus(maxlen=3)
    for i in range(5):
        bus.publish(_ev(code=f"E{i}"))
    assert bus.pending == 3
    assert bus.stats()["dropped"] == 2


@pytest.mark.unit
def test_bus__recent_returns_from_the_tail() -> None:
    bus = EventBus()
    for i in range(5):
        bus.publish(_ev(code=f"E{i}"))
    assert [e.code for e in bus.recent(2)] == ["E3", "E4"]


# ==================================================================
# 生命周期
# ==================================================================

@pytest.mark.unit
def test_bus__publish_after_close_is_a_no_op() -> None:
    """关掉之后再 publish 不该崩 —— 收尾路径常常还在发事件。"""
    bus = EventBus()
    bus.close()
    assert bus.publish(_ev()) is False


@pytest.mark.unit
def test_bus__close_delivers_everything_still_queued() -> None:
    """★ 停程序时最后那几条恰恰最该看到，不能丢。"""
    bus, seen = EventBus(), []
    bus.subscribe(seen.append, name="c")
    for i in range(4):            # 没起分发线程，全压在队列里
        bus.publish(_ev(code=f"E{i}"))
    bus.close()
    assert len(seen) == 4


@pytest.mark.unit
def test_bus__start_is_idempotent() -> None:
    """start 被调两次不该起两个线程（5Hz tick 里很容易重复调）。"""
    bus = EventBus()
    bus.start()
    bus.start()
    t1 = bus._thread
    bus.start()
    assert bus._thread is t1


@pytest.mark.unit
def test_bus__start_actually_spawns_a_dispatcher() -> None:
    bus, seen = EventBus(), []
    bus.subscribe(seen.append, name="c")
    bus.start()
    try:
        bus.publish(_ev())
        deadline = time.time() + 2.0
        while not seen and time.time() < deadline:
            time.sleep(0.01)
        assert len(seen) == 1
    finally:
        bus.close()


@pytest.mark.unit
def test_bus__context_manager_starts_and_closes() -> None:
    with EventBus() as bus:
        assert bus._thread is not None
    assert bus._thread is None


@pytest.mark.unit
def test_bus__stats_expose_every_counter() -> None:
    """排障时全靠这张表 —— 少一个字段就得靠猜。"""
    st = EventBus().stats()
    for key in ("published", "delivered", "dropped", "suppressed",
                "subscriber_errors", "pending", "capacity",
                "subscribers", "last_error"):
        assert key in st, f"stats 少了 {key}"


# ==================================================================
# LogSubscriber
# ==================================================================

@pytest.mark.unit
def test_log_subscriber__is_callable_and_handles_every_severity() -> None:
    # ⚠️ 枚举成员是 WARN 不是 WARNING —— 写错会在构造时抛 ValueError
    sub = LogSubscriber()
    sub(_ev(sev="INFO"))
    sub(_ev(sev="WARN"))
    sub(_ev(sev="ERROR"))


@pytest.mark.unit
def test_log_subscriber__summary_counts_by_code() -> None:
    """summary 给「这轮播了什么」的聚合视图。"""
    sub = LogSubscriber()
    sub(_ev(code="A"))
    sub(_ev(code="A"))
    sub(_ev(code="B"))
    total = sum(n for _code, n in sub.summary())
    assert total == 3


@pytest.mark.unit
def test_bus__publish_from_another_thread_is_safe() -> None:
    """publish 会在分发线程之外被调用 —— 竞争不得炸。"""
    bus = EventBus()
    errs: list[BaseException] = []

    def worker():
        try:
            for i in range(50):
                bus.publish(_ev(code=f"E{i}"))
        except BaseException as exc:       # noqa: BLE001
            errs.append(exc)

    ts = [threading.Thread(target=worker) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=5)
    assert not errs, f"并发 publish 抛了：{errs}"
    assert bus.stats()["published"] == 150
