"""YUI 播报层的两道门控（2026-10-17）。

覆盖两件事：

1. **拒绝语义分流** —— ``INTENT_REJECTED`` 原来只有一句「这个动作我没敢做」。
   引擎连点太快被 ``MIN_GAP`` 限流时也这么说，是**误导**：
   动作本来合法，只是节流了。现在限流说「手速太快啦，慢一点」，
   真正的安全拦截（场景不对 / 急停 / 熔断…）才说「没敢做」。

2. **刷怪防抖** —— ``ENEMY_DETECTED`` 在真机 count 1↔2↔1↔2 抖，
   每次变化都发一条，5 秒节流压不住。改成 15 秒窗口 + 数量阈值。

★ 两条都**只在播报层**。决策侧 ``world_attack_only_with_enemy`` 读的是
  ``has_nearby_enemy()``，与 ``event_to_yui`` 零关系 ——
  :func:`test_decision_layer_never_imports_the_speech_adapter` 把这点钉住。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_to_yui import (
    ENEMY_COUNT_DELTA,
    ENEMY_DEBOUNCE_SECONDS,
    EVENT_TO_SPEECH_POOLS,
    PERMISSION_SPEECH,
    EventToYui,
)

PLUGIN_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit

#: ``ENEMY_DETECTED`` 的随机池（``_PROMPT_GUIDE.md`` §6.3）。
#: 台词会在池子里轮换，所以断言要认「池子里的任意一句 + 数量对得上」，
#: 不能钉死某一句 —— 钉死等于把「不复读」这个特性测没了。
ENEMY_LINES = EVENT_TO_SPEECH_POOLS["ENEMY_DETECTED"]


def _assert_enemy_line(said: list[str], count: int) -> None:
    """断言「只说了一句、数量对得上、句式取自随机池」。

    ★ 比的是**渲染后**的台词
      池子里存的是**模板**（``"有{count}个坏东西来了…"``），
      播出去的是填好占位符的成品。
    """
    assert len(said) == 1, said
    rendered = {line.format(count=count) for line in ENEMY_LINES}
    assert said[0] in rendered, (said, sorted(rendered))


# ================================================================== 夹具
class _Adapter:
    """真适配器 + 假事件流 + 假 push。"""

    def __init__(self, throttle: float = 0.0) -> None:
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="e2y_debounce_"))
        self.run_id = "t"
        (self.dir / "run_current.txt").write_text(self.run_id, encoding="utf-8")
        self.stream = self.dir / f"{self.run_id}.stream.jsonl"
        self.said: list[str] = []
        self.a = EventToYui(
            push_fn=lambda **kw: (
                self.said.append("".join(x.get("text", "") for x in kw["parts"]))
                or {"submitted": True}),
            runtime_dir=self.dir,
            throttle_seconds=throttle,
            # ★ 场景组冷却**关掉**
            #   这份测试验的是「拒绝语义分流 + 刷怪防抖」两层，
            #   场景组冷却（Companion Mode）是第三层 —— 它若开着，
            #   同组内第二条 ``ENEMY_DETECTED`` 会被它挡掉，
            #   测到的就不是被测的那一层了。
            #   场景组冷却由 ``test_yui_companion_mode.py`` 单独钉。
            scene_group_cooldown_s=0.0)

    def write(self, code: str, message: str = "", **ctx) -> None:
        with self.stream.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"schema_version": "1.0", "run_id": self.run_id, "seq": 0,
                 "ts_ms": 0, "code": code, "severity": "INFO",
                 "message": message, "context": ctx},
                ensure_ascii=False) + "\n")

    def drain(self) -> list[str]:
        before = len(self.said)
        self.a.drain_once()
        return self.said[before:]

    def close(self) -> None:
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)


@pytest.fixture()
def adapter():
    a = _Adapter()
    yield a
    a.close()


# ==================================================================
# A · INTENT_REJECTED 的语义分流
# ==================================================================
def test_min_gap_rejection_is_silent() -> None:
    """★★ 被 ``MIN_GAP`` 限流时**一句都不说**（2026-10-19 改）。

    真机 2026-10-17：引擎连点太快（同一 run 754 次 USE_PICKUP），
    被 ``safety.py`` 的 ``min_gap_ms`` 拦下，YUI 却一脸紧张地说
    「这个动作我没敢做」—— 用户会以为急停/熔断在拦它。

    2026-10-19 再进一步：从「换一句不误导的话」改成**彻底闭嘴**。

    ★ 为什么沉默比换一句话好
      限流是引擎**自己**在节流，动作本身完全合法。连招跑一轮能触发
      十几次限流 —— 每 5 秒顶一句「手速太快啦」，用户听着像引擎在
      抱怨；更糟的是它会**强化「引擎不敢动」这个错觉**：
      用户本来就在怀疑引擎是不是坏了，再听它自己喊「太快了」，
      只会更不敢放手。

    真正的信息量在别处：连招进度、抢权提示、真正的故障。
    """
    a = _Adapter()
    try:
        a.write("INTENT_REJECTED",
                message="USE_PICKUP REJECTED: MIN_GAP:USE_PICKUP(205<800ms)",
                reason="MIN_GAP:USE_PICKUP(205<800ms)")
        said = a.drain()
        assert said == [], f"限流不该播报：{said}"
        assert not any("没敢做" in s for s in said), said
    finally:
        a.close()


def test_rate_limited_is_also_silent() -> None:
    """``RATE_LIMITED`` 与 ``MIN_GAP`` 同类：都是输出节流，都闭嘴。"""
    a = _Adapter()
    try:
        a.write("INTENT_REJECTED",
                message="USE_PICKUP REJECTED: RATE_LIMITED:USE_PICKUP(10<50ms)",
                reason="RATE_LIMITED:USE_PICKUP(10<50ms)")
        said = a.drain()
        assert said == [], f"限流不该播报：{said}"
    finally:
        a.close()


def test_old_engine_message_only_is_also_silent() -> None:
    """★★ 兼容**在跑的旧发行包**：reason 只在 message 里。

    旧包（``context`` 里没有 ``reason``）的 message 形如
    ``"USE_PICKUP REJECTED: MIN_GAP:USE_PICKUP(205<800ms)"``。
    两边都扫才能判对 —— 而判对之后的动作是**沉默**。
    """
    a = _Adapter()
    try:
        a.write("INTENT_REJECTED",
                message="USE_PICKUP REJECTED: MIN_GAP:USE_PICKUP(205<800ms)")
        said = a.drain()
        assert said == [], f"限流不该播报：{said}"
    finally:
        a.close()


@pytest.mark.parametrize("reason", [
    "SCENE_NOT_ALLOWED:NORMAL_ATTACK@ WORLD",
    "ACTION_FORBIDDEN:NORMAL_ATTACK",
    "BLACKLISTED:WORLD/NORMAL_ATTACK",
    "PANIC_KEY_PRESSED",
    "BREAKER_OPEN(剩 30000ms)",
    "MAX_ACTIONS_REACHED(200)",
])
def test_real_safety_blocks_keep_saying_i_dared_not(reason: str) -> None:
    """★ 真正的安全拦截**保留**「没敢做」—— 那句话是对的。

    参数化覆盖 ``safety.py::check()`` 的大部分分支。

    ⚠ ``WINDOW_NOT_FOREGROUND`` **不在**这里：它是权限问题不是策略问题，
      播「没敢做」等于让用户去猜自己哪里设置错了（2026-10-18 用户实机反馈）。
    """
    a = _Adapter()
    try:
        a.write("INTENT_REJECTED",
                message=f"NORMAL_ATTACK REJECTED: {reason}", reason=reason)
        said = a.drain()
        assert said == ["这个动作我没敢做"], (reason, said)
    finally:
        a.close()


def test_window_not_foreground_says_permission_not_dared_not() -> None:
    """★ 权限拦截必须说「权限不足」，不能混进「没敢做」那档。"""
    a = _Adapter()
    try:
        a.write("INTENT_REJECTED",
                message="NORMAL_ATTACK REJECTED: WINDOW_NOT_FOREGROUND",
                reason="WINDOW_NOT_FOREGROUND")
        said = a.drain()
        assert said == [PERMISSION_SPEECH], said
    finally:
        a.close()


def test_rejection_without_reason_keeps_the_old_wording() -> None:
    """没有 reason 信息时**按老话术**走（信息不足就别改口）。"""
    a = _Adapter()
    try:
        a.write("INTENT_REJECTED")
        said = a.drain()
        assert said == ["这个动作我没敢做"], said
    finally:
        a.close()


def test_throttled_rejection_produces_no_lines_at_all() -> None:
    """★ 限流**一条都不产生**（2026-10-19）。

    以前是「受 MAX_LINES_PER_RUN 约束」—— 也就是说它**确实会播**，
    只是最多 N 句。现在改成彻底沉默，所以连一句都没有。

    ★ 为什么这条比「限流台词也有上限」更值得钉住
      「有上限」防的是**复读**；「完全沉默」防的是**误导**。
      用户听到引擎自己喊「手速太快」，会以为引擎在拒绝做事 ——
      而它其实只是自己踩了刹车。这条断言守住的是**语义**。
    """
    a = _Adapter()
    try:
        for _ in range(40):
            a.write("INTENT_REJECTED",
                    message="USE_PICKUP REJECTED: MIN_GAP:USE_PICKUP(205<800ms)",
                    reason="MIN_GAP:USE_PICKUP(205<800ms)")
        said = a.drain()
        assert said == [], f"限流一条都不该播：{said}"
    finally:
        a.close()


# ==================================================================
# B · ENEMY_DETECTED 防抖
# ==================================================================
def test_enemy_first_sighting_always_speaks() -> None:
    """首次发现当然要说。"""
    a = _Adapter(throttle=5.0)
    try:
        a.write("ENEMY_DETECTED", count=2, nearest_m=30.0)
        said = a.drain()
        _assert_enemy_line(said, 2)
    finally:
        a.close()


def test_enemy_small_count_change_within_window_is_suppressed() -> None:
    """★★ 15 秒内「2 变 3」不播 —— 用户点名要压的就是这个。"""
    a = _Adapter(throttle=5.0)
    try:
        a.write("ENEMY_DETECTED", count=2)
        a.drain()
        # 通用节流是 5s，这里靠改内部时间戳跳过它，单独验证防抖层
        a.a._last_spoke.pop("ENEMY_DETECTED", None)
        a.write("ENEMY_DETECTED", count=3)
        said = a.drain()
        assert said == [], said
    finally:
        a.close()


def test_enemy_big_count_change_within_window_still_speaks() -> None:
    """★ 数量**真的**变了（2 -> 9）就该说 —— 那是新情况。"""
    a = _Adapter(throttle=5.0)
    try:
        a.write("ENEMY_DETECTED", count=2)
        a.drain()
        a.a._last_spoke.pop("ENEMY_DETECTED", None)
        a.write("ENEMY_DETECTED", count=9)
        said = a.drain()
        _assert_enemy_line(said, 9)
    finally:
        a.close()


def test_enemy_speaks_again_after_the_window() -> None:
    """★ 超过 15 秒没播报 -> 重新发现就该再播一次。"""
    a = _Adapter(throttle=5.0)
    try:
        a.write("ENEMY_DETECTED", count=2)
        a.drain()
        # 把「上次播报」推到窗口之外
        a.a._last_enemy_time -= (ENEMY_DEBOUNCE_SECONDS + 1.0)
        a.a._last_spoke.pop("ENEMY_DETECTED", None)
        a.write("ENEMY_DETECTED", count=2)
        said = a.drain()
        _assert_enemy_line(said, 2)
    finally:
        a.close()


def test_enemy_oscillation_1_2_1_2_is_fully_suppressed() -> None:
    """★★ 真机的抖动形态：1↔2↔1↔2，每跳都只差 1 —— 一次都不该播。"""
    a = _Adapter(throttle=5.0)
    try:
        a.write("ENEMY_DETECTED", count=1)
        first = a.drain()
        assert first == ["有1个坏东西来了，主人小心~"], first
        said: list[str] = []
        for n in (2, 1, 2, 1):
            a.a._last_spoke.pop("ENEMY_DETECTED", None)
            a.write("ENEMY_DETECTED", count=n)
            said.extend(a.drain())
        assert said == [], said
    finally:
        a.close()


def test_enemy_debounce_does_not_consume_the_repeat_budget() -> None:
    """被防抖压住的事件**不**该影响别的配额。"""
    a = _Adapter(throttle=5.0)
    try:
        a.write("ENEMY_DETECTED", count=2)
        a.drain()
        a.a._last_spoke.pop("ENEMY_DETECTED", None)
        a.write("ENEMY_DETECTED", count=3)
        assert a.drain() == []
        # 抑制后状态没被写坏：真正的安全拦截仍能正常播
        a.write("INTENT_REJECTED",
                message="NORMAL_ATTACK REJECTED: SCENE_NOT_ALLOWED:NORMAL_ATTACK@ WORLD",
                reason="SCENE_NOT_ALLOWED:NORMAL_ATTACK@ WORLD")
        assert a.drain() == ["这个动作我没敢做"]
    finally:
        a.close()


def test_enemy_debounce_resets_across_runs() -> None:
    """★ 换 run 要清防抖状态 —— 否则第二次跑一声不吭。

    ⚠ 只清**防抖**状态，通用节流（``_last_spoke``）是**跨 run 保留**的 ——
      那是既有设计（适配器是常驻单例，「多久能再说一次」本就该跨场次）。
      所以这里要把它摘掉，否则测到的会是通用节流而不是防抖。
    """
    a = _Adapter(throttle=5.0)
    try:
        a.write("ENEMY_DETECTED", count=2)
        a.drain()
        # 换 run：改 run_current.txt
        (a.dir / "run_current.txt").write_text("t2", encoding="utf-8")
        a.stream = a.dir / "t2.stream.jsonl"
        a.a._last_spoke.pop("ENEMY_DETECTED", None)
        a.write("ENEMY_DETECTED", count=2)
        said = a.drain()
        _assert_enemy_line(said, 2)
    finally:
        a.close()


def test_enemy_debounce_is_off_when_throttle_is_zero() -> None:
    """★ ``throttle_seconds=0`` = 「别再压我的播报」，防抖一并关。

    既有测试 ``test_t2_throttle_zero_disables_dedup`` 依赖这个语义。
    """
    a = _Adapter(throttle=0.0)
    try:
        a.write("ENEMY_DETECTED", count=1)
        a.write("ENEMY_DETECTED", count=2)
        said = a.drain()
        assert len(said) == 2, said
    finally:
        a.close()


def test_enemy_delta_threshold_is_two() -> None:
    """把阈值钉住 —— 改它是有意行为，必须有人知道。"""
    assert ENEMY_COUNT_DELTA == 2
    # ★ 2026-10-19：15 -> 30。真机复盘 run 20261019_000606（273 秒）
    #   里播了 8 句「发现N个敌人」—— 感知 5Hz 下数量抖动远比 15 秒密，
    #   15 秒窗仍然拦不住。加倍之后配合干跑静默，这条基本只在真接管时说。
    assert ENEMY_DEBOUNCE_SECONDS == 30.0


# ==================================================================
# C · 隔离性：防抖不许影响决策
# ==================================================================
def test_decision_layer_never_imports_the_speech_adapter() -> None:
    """★★ 决策层**零引用** ``event_to_yui`` —— 防抖动不了决策。

    ``world_attack_only_with_enemy``（``rules/genshin_combat.toml:31``）
    读的是 ``has_nearby_enemy()``，走 policy context。
    播报层改再多，规则命中与否也不变。
    """
    import re

    offenders: list[str] = []
    for rel in ("autopilot/policy", "autopilot/rules", "autopilot/actuator",
                "autopilot/core", "autopilot/contracts",
                "autopilot/perception"):
        base = PLUGIN_ROOT / rel
        for py in base.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="replace")
            if re.search(r"^\s*(from|import)\s+event_to_yui", text,
                         re.MULTILINE):
                offenders.append(str(py.relative_to(PLUGIN_ROOT)))
    assert offenders == [], f"决策层引用了播报层：{offenders}"


def test_enemy_count_still_reaches_the_policy_context() -> None:
    """★ 防抖只压**播报**，不压**数据流** —— 事件本身照样进流。

    引擎侧的 ``ENEMY_DETECTED`` 事件仍会写进 stream.jsonl，
    决策侧读的是 policy context（``has_nearby_enemy()``），
    与本适配器无关。这里证明事件没被适配器吞掉。
    """
    a = _Adapter(throttle=5.0)
    try:
        a.write("ENEMY_DETECTED", count=2)
        a.drain()
        a.write("ENEMY_DETECTED", count=3)
        a.drain()
        lines = [json.loads(x) for x in
                 a.stream.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(lines) == 2, lines
        assert [x["context"]["count"] for x in lines] == [2, 3], lines
    finally:
        a.close()
