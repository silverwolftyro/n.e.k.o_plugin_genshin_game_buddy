"""物理输入抢占 + YUI 语义分流（2026-10-18）。

覆盖三件事：

1. **物理输入抢占** —— 用户按的键**必须**优先于引擎的虚拟按键。
2. **YUI 台词分流** —— 限流 / 权限不足 / 用户让权 / 真安全拦截，
   四件事说四种话。
3. **权限计数** —— 权限类拦截要被 ``LoopStats`` 记下来，
   Console 的红牌靠它。

★ 战斗规则与识别模型**一律没动**（用户约束）。
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.physical_input import (  # noqa: E402
    POLL_INTERVAL_S, TAKEOVER_IDLE_S, WATCHED_KEYS, PhysicalInputWatcher,
    TakeoverState)
from autopilot.actuator.safety import (  # noqa: E402
    SafetyConfig, SafetyGuard, is_permission_problem, is_user_takeover)
from autopilot.actuator.simulator import VK  # noqa: E402
from autopilot.contracts.events import Event  # noqa: E402
from autopilot.contracts.reasons import (  # noqa: E402
    PERMISSION_REASON_PREFIXES, REASON_USER_TAKEOVER, reason_head)
from autopilot.core.stats import LoopStats  # noqa: E402
from event_to_yui import (  # noqa: E402
    PERMISSION_REASON_PREFIXES as YUI_PERMISSION_PREFIXES,
    PERMISSION_SPEECH, USER_TAKEOVER_SPEECH,
    EventToYui)


# ==================================================================
# A · 物理输入抢占
# ==================================================================
class _FakeKeyState:
    """可编程的 GetAsyncKeyState 替身。"""

    def __init__(self) -> None:
        self.down: set[int] = set()

    def __call__(self, vk: int) -> int:
        return 0x8000 if vk in self.down else 0

    def press(self, vk: VK) -> None:
        self.down.add(int(vk))

    def release(self, vk: VK) -> None:
        self.down.discard(int(vk))


def test_watched_keys_cover_the_usual_game_keys() -> None:
    """★ 被监听的键必须包含用户操作游戏最常用的那些。"""
    names = {vk.name for vk in WATCHED_KEYS}
    for must in ("W", "A", "S", "D", "Q", "E", "LBUTTON", "RBUTTON"):
        assert must in names, f"没监听 {must}"
    # ★ F12 是急停键，**不能**在这里（语义完全错位）
    assert "F12" not in names


def test_poll_interval_is_cheap_but_fast_enough() -> None:
    """50Hz：够抓住 100ms 的轻点，又不至于烧 CPU。"""
    assert 0.0 < POLL_INTERVAL_S <= 0.05
    assert 1.0 / POLL_INTERVAL_S >= 20


def test_takeover_idle_grace_is_one_second() -> None:
    """★ 让权不再是「死等 3 秒」，而是**跟着用户的手指走**。

    规则（见 TakeoverState）：
      · 用户还按着键   -> 一直让权
      · 用户松手了     -> 再宽限 TAKEOVER_IDLE_S 秒
      · 宽限也过了     -> 立刻恢复接管
    """
    assert TAKEOVER_IDLE_S == 1.0


def test_takeover_follows_the_users_finger() -> None:
    """★ 核心行为：按着不放就一直让权，松手 1 秒后才恢复。

    旧实现是「一按就让权 3 秒」—— 用户点一下走位要干等 3 秒，
    而按住 W 走 5 秒时 3 秒窗口一过引擎就来抢方向。两种都不对。
    """
    st = TakeoverState(idle_s=1.0)

    # ① 按下 -> 让权
    st.note_press("W")
    st.note_hold("W")
    assert st.active is True

    # ② 按住 2 秒（远超空闲宽限）-> **仍然**让权
    time.sleep(0.15)
    assert st.active is True, "用户还按着键，引擎不该抢"

    # ③ 松手 -> 仍然让权（1 秒宽限）
    st.note_release("W")
    assert st.active is True, "松手瞬间不该立刻抢"

    # ④ 宽限用完 -> 恢复接管
    st.idle_s = 0.05
    time.sleep(0.12)
    assert st.active is False


def test_takeover_resumes_quickly_after_a_tap() -> None:
    """★ 用户点一下走位：1 秒后就该恢复接管，不是干等 3 秒。"""
    st = TakeoverState(idle_s=0.1)
    st.note_press("E")
    st.note_hold("E")
    st.note_release("E")
    assert st.active is True
    time.sleep(0.2)
    assert st.active is False


def test_watcher_reports_release_edge() -> None:
    """★ 松开也要报 —— 「用户还按着」是动态让权的第一条规则的数据源。

    只报按下的话，引擎永远不知道用户松手了，
    就会一直以为对方还在按住 W 而不敢接管。
    """
    fake = _FakeKeyState()
    pressed: list[str] = []
    released: list[str] = []
    watcher = PhysicalInputWatcher(pressed.append, on_release=released.append)
    watcher._poll_once(fake)          # 先跑一次建立基线

    fake.press(VK.W)
    watcher._poll_once(fake)
    assert pressed == ["W"], pressed
    assert released == [], released

    fake.release(VK.W)
    watcher._poll_once(fake)
    assert released == ["W"], released

    # 松开不该重复报
    for _ in range(5):
        watcher._poll_once(fake)
    assert released == ["W"], released


def test_guard_takeover_follows_hold_and_release() -> None:
    """★ 安全闸这一侧也要是动态的：按着就一直拦，松手 1 秒后才放行。"""
    guard = SafetyGuard(SafetyConfig())
    guard.takeover.idle_s = 0.05
    intent = _intent()
    state = _state()

    guard.note_user_input("E", 0.05, still_held=True)
    ok, reason = guard.check(intent, state, int(time.time() * 1000))
    assert not ok
    assert reason.startswith(REASON_USER_TAKEOVER), reason

    # 松手 -> 仍有宽限
    guard.note_user_release("E")
    ok, _ = guard.check(intent, state, int(time.time() * 1000))
    assert not ok, "松手后还有空闲宽限，不该立刻放行"

    # 宽限用完 -> 放行
    time.sleep(0.12)
    ok, reason = guard.check(intent, state, int(time.time() * 1000))
    assert ok, f"宽限用完应该恢复接管：{reason}"


def test_watcher_fires_on_new_press_only() -> None:
    """★ 边沿检测：按住不放**只报一次**。

    长按 W 会在 20ms 一次的重触发下把让权窗口无限续期，
    引擎就永远不恢复。
    """
    fake = _FakeKeyState()
    seen: list[str] = []
    watcher = PhysicalInputWatcher(seen.append)
    watcher._poll_once(fake)          # 先跑一次建立基线
    assert seen == []

    fake.press(VK.E)
    watcher._poll_once(fake)
    assert seen == ["E"], seen

    # 一直按着
    for _ in range(20):
        watcher._poll_once(fake)
    assert seen == ["E"], f"按住不该重复报：{seen}"

    # 松开再按 -> 再报一次
    fake.release(VK.E)
    watcher._poll_once(fake)
    fake.press(VK.E)
    watcher._poll_once(fake)
    assert seen == ["E", "E"], seen


def test_watcher_does_not_fire_for_already_held_keys_at_start() -> None:
    """★ 引擎刚起、用户正好按着 W —— 不该白白触发一次让权。"""
    fake = _FakeKeyState()
    fake.press(VK.W)
    seen: list[str] = []
    watcher = PhysicalInputWatcher(seen.append)
    watcher._down = {vk for vk in WATCHED_KEYS if fake(int(vk)) & 0x8000}
    watcher._poll_once(fake)
    assert seen == []


def test_watcher_callback_exception_does_not_kill_the_thread() -> None:
    """回调抛异常不能让监听线程死掉（否则用户就再也抢不回来了）。"""
    watcher = PhysicalInputWatcher(lambda name: (_ for _ in ()).throw(
        RuntimeError("boom")))
    watcher._fire(VK.E)              # 不抛就算过


def test_guard_blocks_execution_during_takeover() -> None:
    """★ 让权窗口内，安全闸**不放行**任何意图。"""
    guard = SafetyGuard(SafetyConfig())
    intent = _intent()
    state = _state()
    ok, _ = guard.check(intent, state, int(time.time() * 1000))
    assert ok, "没让权时应该放行"

    guard.note_user_input("E", 3.0)
    assert guard.user_takeover_active is True
    ok, reason = guard.check(intent, state, int(time.time() * 1000))
    assert not ok
    assert reason.startswith(REASON_USER_TAKEOVER), reason


def test_takeover_window_expires() -> None:
    guard = SafetyGuard(SafetyConfig())
    guard.note_user_input("E", 0.05)
    assert guard.user_takeover_active is True
    time.sleep(0.12)
    assert guard.user_takeover_active is False


def test_takeover_is_not_a_hard_stop() -> None:
    """★ 让权**不是**熔断/急停 —— 熔断器不能被用户按几下键打开。"""
    from autopilot.actuator.safety import is_hard_stop

    assert not is_hard_stop(f"{REASON_USER_TAKEOVER}:E")


def test_user_interrupt_event_is_info_not_error() -> None:
    """★ 让权是**礼让**，不是故障 —— 严重级别不能是 ERROR。"""
    from autopilot.contracts.events import Severity

    ev = Event.user_interrupt("E", 2, 3.0)
    assert ev.code == "USER_INTERRUPT"
    assert ev.severity is Severity.INFO
    assert ev.context["key"] == "E"
    assert ev.context["released_keys"] == 2
    assert ev.context["pause_s"] == 3.0


def test_reason_prefixes_match_between_contract_and_safety() -> None:
    """★ 抄一份可以，**不一致不行**。"""
    from autopilot.actuator import safety

    assert set(safety.PERMISSION_REASON_PREFIXES) == \
        set(PERMISSION_REASON_PREFIXES)
    assert safety.REASON_USER_TAKEOVER == REASON_USER_TAKEOVER


def test_reason_head_splits_prefix() -> None:
    assert reason_head("MIN_GAP:USE_PICKUP(205<800ms)") == "MIN_GAP"
    assert reason_head("WINDOW_NOT_FOREGROUND") == "WINDOW_NOT_FOREGROUND"
    assert reason_head("BREAKER_OPEN(剩 30000ms)") == "BREAKER_OPEN"
    assert reason_head("") == ""


# ==================================================================
# B · YUI 台词分流
# ==================================================================
class _Adapter:
    def __init__(self, throttle: float = 0.0) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="e2y_perm_"))
        self.run_id = "t"
        (self.dir / "run_current.txt").write_text(self.run_id, encoding="utf-8")
        self.stream = self.dir / f"{self.run_id}.stream.jsonl"
        self.said: list[str] = []
        self.a = EventToYui(
            push_fn=lambda **kw: (
                self.said.append("".join(x.get("text", "") for x in kw["parts"]))
                or {"submitted": True}),
            runtime_dir=self.dir, throttle_seconds=throttle,
            # ★ 场景组冷却关掉 —— 这份测试验的是「拒绝语义 + 台词内容」，
            #   场景组冷却（Companion Mode）由 ``test_yui_companion_mode.py`` 钉。
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


def test_permission_rejection_says_permission_not_dared_not(adapter) -> None:
    """★★ 没管理员权限时，YUI 应该说「权限不足」。

    之前一律说「这个动作我没敢做」—— 用户会以为规则写错了，
    而真正该做的是**用管理员身份重跑**。
    """
    adapter.write("INTENT_REJECTED",
                  message="NORMAL_ATTACK REJECTED: WINDOW_NOT_FOREGROUND",
                  reason="WINDOW_NOT_FOREGROUND")
    said = adapter.drain()
    assert said == [PERMISSION_SPEECH], said
    assert not any("没敢做" in s for s in said), said


def test_permission_rejection_detected_from_message_only(adapter) -> None:
    """★ 旧发行包没有 context["reason"] —— 也能判出来。"""
    adapter.write("INTENT_REJECTED",
                  message="NORMAL_ATTACK REJECTED: WINDOW_NOT_FOREGROUND")
    said = adapter.drain()
    assert said == [PERMISSION_SPEECH], said


def test_user_takeover_rejection_is_silent(adapter) -> None:
    """★ 让权导致的拒绝**不播报** —— USER_INTERRUPT 已经说过了。

    再补一句「我没敢做」纯属噪音，还会让人以为引擎坏了。
    """
    adapter.write("INTENT_REJECTED",
                  message="USE_SKILL REJECTED: USER_TAKEOVER:E",
                  reason="USER_TAKEOVER:E")
    assert adapter.drain() == []


def test_user_interrupt_event_has_its_own_line(adapter) -> None:
    """★ 让权本身有话说：「你先来，我不抢」。"""
    adapter.write("USER_INTERRUPT", message="用户按下 E，引擎让权 3 秒",
                  key="E", released_keys=1, pause_s=3.0)
    said = adapter.drain()
    assert said == [USER_TAKEOVER_SPEECH], said


def test_real_safety_block_still_says_dared_not(adapter) -> None:
    """★ 真正的安全拦截**保留**「没敢做」—— 那句话是对的。"""
    adapter.write("INTENT_REJECTED",
                  message="NORMAL_ATTACK REJECTED: SCENE_NOT_ALLOWED:NORMAL_ATTACK@ WORLD",
                  reason="SCENE_NOT_ALLOWED:NORMAL_ATTACK@ WORLD")
    said = adapter.drain()
    assert said == ["这个动作我没敢做"], said


def test_throttle_is_silent(adapter) -> None:
    """★ 限流**不播报**（2026-10-19）。

    限流是引擎自己在节流，动作完全合法。播出来只会让用户以为
    引擎「不敢动」或出故障了 —— 而它现在连招跑一轮能触发十几次。

    真正该让用户听见的是「你按了键，引擎让权了」和真的故障。
    """
    adapter.write("INTENT_REJECTED",
                  message="USE_PICKUP REJECTED: MIN_GAP:USE_PICKUP(205<800ms)",
                  reason="MIN_GAP:USE_PICKUP(205<800ms)")
    said = adapter.drain()
    assert said == [], f"限流不该播报：{said}"


def test_enemy_line_has_no_species(adapter) -> None:
    """★★ YUI 台词里**不许有种族信息**（2026-10-18 真机：把盗宝团说成丘丘人）。

    引擎**根本不知道**敌人种类（YOLO 的 7 类是功能类别：
    health_bar / enemy_identify / drops / ore / unmined_ore / chest /
    locked_chest），所以台词里只能有数量。
    """
    adapter.write("ENEMY_DETECTED", count=3, nearest_m=30.0)
    said = adapter.drain()
    assert said, "该播报"
    line = said[0]
    assert "3" in line
    for banned in ("丘丘人", "盗宝团", "史莱姆", "野猪", "丘丘"):
        assert banned not in line, f"台词里出现了种族名 {banned}：{line}"
    # ★ 敌人只能**泛指**（2026-10-29 人设改造后说的是「坏东西」）
    assert "坏东西" in line or "敌人" in line, line


def test_enemy_line_leaves_no_species_to_guess(adapter) -> None:
    """★ 台词必须让模型**无从猜出种族**。

    ★ 2026-10-29：原来的「没看清是哪种」被换成了「坏东西」
      （``_PROMPT_GUIDE.md`` §6.3 的人设改造）。

      那句「没看清」本来是对 LLM 的硬约束，但真正起作用的是
      **台词里根本没有种族信息** —— 只要没有，模型就没有可编造的素材。
      所以这里把判据从「有没有说没看清」换成「有没有任何种族词」，
      并把种族词表**扩大**。
    """
    adapter.write("ENEMY_DETECTED", count=2)
    line = adapter.drain()[0]
    for banned in ("丘丘", "盗宝团", "史莱姆", "野猪", "深渊", "法师",
                   "弓手", "元素精灵", "遗迹守卫", "派蒙"):
        assert banned not in line, line
    # 数量必须留着 —— 引擎只报数量，不报种类
    assert "2" in line, line


def test_no_monster_species_names_anywhere_in_the_adapter() -> None:
    """★ 全文件扫一遍：台词表里不许出现任何具体种族名。"""
    src = (PLUGIN_ROOT / "event_to_yui.py").read_text(encoding="utf-8")
    for banned in ("丘丘人", "盗宝团", "史莱姆", "深渊法师", "丘丘岩"):
        assert banned not in src, f"event_to_yui.py 里出现了 {banned}"


# ==================================================================
# C · 权限计数
# ==================================================================
def test_stats_count_permission_blocks() -> None:
    """★ 权限类拦截要被单独计数 —— Console 的红牌靠它。"""
    st = LoopStats()
    st.record_rejection("WINDOW_NOT_FOREGROUND")
    st.record_rejection("WINDOW_NOT_FOREGROUND")
    st.record_rejection("MIN_GAP:USE_PICKUP(205<800ms)")
    st.record_rejection("SCENE_NOT_ALLOWED:X@ WORLD")
    assert st.permission_blocks == 2
    assert st.by_reason["WINDOW_NOT_FOREGROUND"] == 2
    assert st.by_reason["MIN_GAP"] == 1
    assert st.by_reason["SCENE_NOT_ALLOWED"] == 1
    snap = st.snapshot()
    assert snap["permission_blocks"] == 2
    assert snap["by_reason"]["WINDOW_NOT_FOREGROUND"] == 2


def test_stats_ignore_empty_reason() -> None:
    st = LoopStats()
    st.record_rejection("")
    assert st.permission_blocks == 0
    assert st.by_reason == {}


def test_is_permission_problem_classifier() -> None:
    assert is_permission_problem("WINDOW_NOT_FOREGROUND") is True
    assert is_permission_problem("MIN_GAP:USE_PICKUP(1<800ms)") is False
    assert is_user_takeover("USER_TAKEOVER:E") is True
    assert is_user_takeover("WINDOW_NOT_FOREGROUND") is False


def test_yui_permission_prefixes_match_the_contract() -> None:
    """★ 适配器里抄的那份前缀必须与契约一致。"""
    assert set(YUI_PERMISSION_PREFIXES) == set(PERMISSION_REASON_PREFIXES)


# ==================================================================
# 夹具
# ==================================================================
def _intent():
    from autopilot.contracts.intent_packet import ActionType, IntentPacket

    return IntentPacket(action=ActionType.NORMAL_ATTACK, state_version=1)


def _state():
    from autopilot.contracts.state_packet import SceneInfo, StatePacket

    return StatePacket(state_version=1, scene=SceneInfo(id="WORLD", confidence=1.0))
