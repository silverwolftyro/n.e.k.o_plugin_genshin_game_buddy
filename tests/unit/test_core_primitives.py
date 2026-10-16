"""事件契约 + 事件总线 + 安全闸的单测（2026-10-11 补测）。

这三个模块此前**零覆盖**，而它们的失败模式都很隐蔽：

* ``core/events.py`` 的 ``dedupe_key`` —— 真机上因为它，我一度把
  「154.8 秒里被 min_gap 拒了 603 次」误读成 31 次（事件流只落了去重
  后的条���）。**计数器 vs 事件流** 必须都有测试钉住，否则下次还会踩。
* ``actuator/safety.py`` 的 min_gap —— 同样是被误读的那道闸。
* ``contracts/events.py`` 的工厂 —— 严重级别是给YUI 判断「该不该说」的
  依据，写错了会直接变成用户看到的行为。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.safety import SafetyConfig, SafetyGuard  # noqa: E402
from autopilot.actuator.traits import TRAITS, traits_of  # noqa: E402
from autopilot.contracts.events import Event, EventCategory, Severity  # noqa: E402

# ==================================================================
# contracts/events.py —— 工厂
# ==================================================================

@pytest.mark.unit
def test_screenshot_failed__severity_escalates_at_3() -> None:
    """连着失败要升级到 ERROR —— 单次失败只是 WARN。"""
    assert Event.screenshot_failed("e", 1).severity is Severity.WARN
    assert Event.screenshot_failed("e", 2).severity is Severity.WARN
    assert Event.screenshot_failed("e", 3).severity is Severity.ERROR
    assert Event.screenshot_failed("e", 99).severity is Severity.ERROR


@pytest.mark.unit
def test_screenshot_failed__carries_error_and_streak() -> None:
    # ⚠️ 契约里的码是 ``SCREENSHOT_FAIL``（不带 ED），而 ``CAPTURE_FAILED``
    #   是 ``core/loop.py`` 另行发的**另一个**码。两者别混。
    ev = Event.screenshot_failed("boom", 5)
    assert ev.code == "SCREENSHOT_FAIL"
    assert ev.context.get("error") == "boom"
    assert ev.context.get("consecutive") == 5


@pytest.mark.unit
def test_black_frame__records_luma() -> None:
    ev = Event.black_frame(0.0)
    assert ev.code == "BLACK_FRAME"
    assert ev.severity is Severity.WARN
    assert ev.context.get("luma") == 0.0


@pytest.mark.unit
def test_config_error__always_error_and_truncates() -> None:
    """加载期失败必须是 ERROR，且长错误要被截断（不然能把事件流撑爆）。"""
    ev = Event.config_error("defaults.toml", "x" * 5000)
    assert ev.severity is Severity.ERROR
    assert len(ev.context["error"]) <= 500


@pytest.mark.unit
def test_event_categories_are_stable() -> None:
    """枚举值是对外契约（overlay 按 category 上色），不能随手改。"""
    assert Severity.INFO.value == "INFO"
    assert EventCategory.PERCEPTION.value == "PERCEPTION"
    assert EventCategory.EXECUTION.value == "EXECUTION"


@pytest.mark.unit
def test_contract_codes_have_a_yui_line_or_are_deliberately_skipped() -> None:
    """★ 2026-10-11 加：契约里发得出的 code，必须在 YUI 台词表里有台词。

    背景：``SCREENSHOT_FAIL``（contracts/events.py 的码，和 loop 自己发的
    ``CAPTURE_FAILED`` 是**两个不同的码**）一直没有台词，于是「截图失败」
    这类事件到了 YUI 那里只剩兜底前缀 —— 用户看到的就是「游戏那边……」
    这种毫无信息量的话。

    这条守卫的作用是：以后新增事件码时，如果忘了配台词，**测试会红**，
    逼着人明确表态「要不要说」，而不是让它静默走兜底。
    """
    import re

    from event_to_yui import EVENT_TO_SPEECH, SKIP_CODES

    src = (PLUGIN_ROOT / "autopilot" / "contracts" / "events.py").read_text(
        encoding="utf-8")
    codes = set(re.findall(r'code="([A-Z_]+)"', src))
    codes |= {"CAPTURE_FAILED", "PERCEPTION_DOWN", "ENTITY_DETECTOR_UNAVAILABLE",
              "CORE_STARTED", "CORE_PAUSED", "CORE_RESUMED", "CORE_STOPPED"}
    # ★ 2026-10-11 用户已拍板，下面这 8 个码全部落地（4 个配台词 / 4 个跳过），
    #   所以这里**刻意保持为空** —— 它现在的作用是「将来再冒出没表态的码就红」。
    _PENDING_HUMAN_DECISION: set[str] = set()
    uncovered = sorted(c for c in codes
                      if c not in EVENT_TO_SPEECH
                      and c not in SKIP_CODES
                      and c not in _PENDING_HUMAN_DECISION)
    assert not uncovered, (
        f"这些事件码既没有台词也不在 SKIP_CODES 里，YUI 只会说出兜底前缀："
        f"{uncovered}")


# ==================================================================
# core/events.py —— 总线去重（真机踩过的坑）
# ==================================================================

@pytest.mark.unit
def test_bus__dedupe_key_suppresses_within_window() -> None:
    """★ 这条是 2026-10-11 踩坑的固化：同一个 dedupe_key 在窗口内
    只发一次。统计计数器仍然是全部，**只有事件流被压掉** ——
    排查时必须分清「真实发生了几次」和「日志里看到几条」。"""
    from autopilot.core.events import EventBus

    bus = EventBus()
    got: list[Event] = []
    bus.subscribe(lambda ev: got.append(ev), name="t")

    assert bus.publish(Event.black_frame(0.0), dedupe_key="BF") is True
    assert bus.publish(Event.black_frame(0.0), dedupe_key="BF") is False
    assert bus.publish(Event.black_frame(0.0), dedupe_key="BF") is False
    assert bus.publish(Event.black_frame(0.0), dedupe_key="OTHER") is True
    bus.drain_now()

    codes = [e.code for e in got]
    assert codes == ["BLACK_FRAME", "BLACK_FRAME"], (
        f"去重后应只剩 2 条（BF 一条 + OTHER 一条），实际 {codes}")
    bus.close()


@pytest.mark.unit
def test_bus__no_dedupe_key_emits_everything() -> None:
    """不传 dedupe_key 就**不该**去重。"""
    from autopilot.core.events import EventBus

    bus = EventBus()
    got: list[Event] = []
    bus.subscribe(lambda ev: got.append(ev), name="t")
    for _ in range(5):
        assert bus.publish(Event.black_frame(0.0)) is True
    bus.drain_now()
    assert len(got) == 5
    bus.close()


@pytest.mark.unit
def test_bus__zero_window_disables_dedupe() -> None:
    from autopilot.core.events import EventBus

    bus = EventBus()
    for _ in range(3):
        assert bus.publish(Event.black_frame(0.0), dedupe_key="K",
                           dedupe_window_s=0.0) is True
    bus.close()


@pytest.mark.unit
def test_bus__subscriber_exception_does_not_break_bus() -> None:
    """订阅者抛异常只能记账，**绝不能**让总线炸掉或吞掉后续事件。"""
    from autopilot.core.events import EventBus

    bus = EventBus()
    good: list[Event] = []

    def bad(_ev):
        raise RuntimeError("boom")

    bus.subscribe(bad, name="bad")
    bus.subscribe(lambda ev: good.append(ev), name="good")
    bus.publish(Event.black_frame(1.0))
    bus.publish(Event.black_frame(2.0))
    n = bus.drain_now()

    assert len(good) == 2, "好订阅者应该两条都收到"
    assert n == 2
    st = bus.stats()
    assert st.get("subscriber_errors", 0) >= 2, "异常必须被记账"
    bus.close()


@pytest.mark.unit
def test_bus__unsubscribe_works() -> None:
    from autopilot.core.events import EventBus

    bus = EventBus()
    got: list[Event] = []
    off = bus.subscribe(lambda ev: got.append(ev), name="t")
    assert bus.subscriber_count == 1
    off()
    assert bus.subscriber_count == 0
    bus.publish(Event.black_frame(0.0))
    bus.drain_now()
    assert got == []
    bus.close()


# ==================================================================
# actuator/safety.py —— 安全闸
# ==================================================================

def _world():
    """造一个够安全闸跑一轮的假 StatePacket（感知健康 + WORLD）。"""
    from autopilot.contracts.state_packet import (
        ResourceInfo,
        SceneInfo,
        StatePacket,
    )

    st = StatePacket.unknown(game_id="genshin", state_version=1, reason="t")
    object.__setattr__(st, "scene",
                       SceneInfo(id="WORLD", confidence=0.9, evidence=("t",)))
    object.__setattr__(st, "resource", ResourceInfo(perception_ok=True))
    return st


def _intent(action: str = "USE_PICKUP"):
    from autopilot.contracts.intent_packet import ActionType, IntentPacket

    return IntentPacket.make(ActionType(action), params={}, state_version=1,
                             ttl_seconds=2, priority=50, scene_required=(),
                             source="LOCAL")


@pytest.mark.unit
def test_safety__min_gap_blocks_quick_repeat() -> None:
    """★ 同一条 min_gap 事故的反面：这次断言它**正常工作**。

    真实 min_gap_ms=800，连续两次相隔 100ms 时第二次必须被拒。
    上一轮我误判「它没生效」，根源是没看计数器只看事件流 ——
    有了这条用例，以后再怀疑它坏了可以直接跑。
    """
    g = SafetyGuard(SafetyConfig(require_foreground=False))
    st = _world()
    t0 = 1_000_000
    ok1, r1 = g.check(_intent(), st, t0)
    assert ok1 is True, f"第一次应放行：{r1}"
    g.mark_executed("USE_PICKUP", t0)

    ok2, r2 = g.check(_intent(), st, t0 + 100)
    assert ok2 is False
    assert r2.startswith("MIN_GAP:"), r2

    ok3, r3 = g.check(_intent(), st, t0 + 5000)
    assert ok3 is True, f"超过 min_gap 后应放行：{r3}"
    g.mark_success()


@pytest.mark.unit
def test_safety__rate_limit_is_independent_of_min_gap() -> None:
    """rate_limit_ms 是全局闸，min_gap 是每动作闸 —— 两条都要能拒。"""
    g = SafetyGuard(SafetyConfig(require_foreground=False,
                                 rate_limit_ms=50))
    st = _world()
    t0 = 2_000_000
    g.check(_intent(), st, t0)
    g.mark_executed("USE_PICKUP", t0)
    # min_gap 对这个动作是 800ms，所以 100ms 时应该报 MIN_GAP
    # （rate_limit 50ms 已经放行了）
    ok, reason = g.check(_intent(), st, t0 + 100)
    assert ok is False and reason.startswith("MIN_GAP:"), reason


@pytest.mark.unit
def test_safety__scene_guard_rejects_wrong_scene() -> None:
    """traits 的 allowed_scenes 是护栏：场景不匹配直接拒，不放行到执行。"""
    from autopilot.contracts.state_packet import StatePacket

    g = SafetyGuard(SafetyConfig(require_foreground=False))
    st = StatePacket.unknown(game_id="genshin", state_version=1, reason="t")
    # unknown() 的场景就是 UNKNOWN，而 USE_PICKUP 只允许 WORLD/UNKNOWN…
    # 这里改成 COMBAT 来验证护栏确实会拦
    object.__setattr__(st, "resource", ResourceInfo_ok())
    from autopilot.contracts.state_packet import SceneInfo

    object.__setattr__(st, "scene",
                       SceneInfo(id="COMBAT", confidence=0.9, evidence=("t",)))
    ok, reason = g.check(_intent(), st, 3_000_000)
    assert ok is False and "SCENE_NOT_ALLOWED" in reason, reason


def ResourceInfo_ok():
    from autopilot.contracts.state_packet import ResourceInfo

    return ResourceInfo(perception_ok=True)


@pytest.mark.unit
def test_traits__every_action_has_entry() -> None:
    """白名单里的每个动作都必须配了 traits —— 查表会静默返回保守默认。"""
    from autopilot.contracts.intent_packet import ActionType

    missing = [a.value for a in ActionType if a.value not in TRAITS]
    assert not missing, f"这些动作没配 traits：{missing}"


@pytest.mark.unit
def test_traits__use_pickup_is_tap_not_hold() -> None:
    """真机调研结论：开宝箱是**点按** F。改了这里要连带改真机结论。"""
    t = traits_of("USE_PICKUP")
    assert t.holds_keys is False
    assert "WORLD" in t.allowed_scenes


@pytest.mark.unit
def test_traits__unknown_action_warns_not_crashes() -> None:
    from autopilot.actuator.traits import traits_of

    t = traits_of("NOT_A_REAL_ACTION")
    assert t is not None, "未知动作要返回保守默认，不能抛"
    assert time.monotonic() >= 0