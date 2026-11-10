"""Companion Mode 的四道新闸门（2026-10-29）。

★ 这份测试在验什么
------------------
「不能像复读机（少一分），也不能像指导教练（多一分）」——
这句话拆成四条**互相独立**的规则，每条都必须能被单独触发：

1. **场景门控**（场景 × 档位）：战斗态只留 tier 1 关键异常。
2. **同场景组冷却**（90s）：同一场景组内非 tier 1 台词只准一句。
3. **挂机主动开口**（180s + 600s）：主人不动时 YUI 主动找话题。
4. **台词随机池**：同一件事换个说法说，且第一句固定可断言。

对应 ``_PROMPT_GUIDE.md`` §9 的自检清单 1~4、6。

★ 为什么用假时钟
----------------
180s / 600s 这种阈值**不能靠 sleep 去测**（一次 10 分钟），
所以 ``ChatState`` 支持注入时钟（``clock=``），
这里注入一个能手推的假时钟 —— 阈值逻辑因此可以被精确断言。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from autopilot.ui.chat_state import (
    GROUP_COMBAT,
    GROUP_MENU,
    GROUP_TRANSITION,
    GROUP_WORLD,
    IDLE_PROACTIVE_MIN_GAP_S,
    IDLE_SECONDS,
    SCENE_GROUP,
    STATE_COMBAT,
    STATE_IDLE,
    STATE_UNKNOWN,
    STATE_WORLD,
    ChatState,
    group_for_scene,
    state_for_scene,
)
from event_to_yui import (
    ACTION_SPEECH,
    ACTION_SPEECH_POOLS,
    EVENT_TO_SPEECH,
    EVENT_TO_SPEECH_POOLS,
    IDLE_PROACTIVE_LINES,
    SCENE_GROUP_COOLDOWN_S,
    SCENE_TIER_LIMIT,
    SKIP_CODES,
    TIER_CRITICAL,
    TIER_NORMAL,
    TIER_VERBOSE,
    EventToYui,
    tier_of,
)

pytestmark = pytest.mark.unit


# ================================================================== 夹具
class _Clock:
    """能手推的假时钟。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class _Rig:
    """真适配器 + 假事件流 + 假 push + 假时钟。"""

    def __init__(self, *, throttle: float = 0.0, clock: _Clock | None = None,
                 group_cooldown: float | None = None) -> None:
        self.clock = clock or _Clock()
        self.dir = Path(tempfile.mkdtemp(prefix="e2y_companion_"))
        self.run_id = "t"
        (self.dir / "run_current.txt").write_text(self.run_id, encoding="utf-8")
        self.stream = self.dir / f"{self.run_id}.stream.jsonl"
        self.said: list[str] = []
        self.state = ChatState(clock=self.clock)
        kwargs = {}
        if group_cooldown is not None:
            kwargs["scene_group_cooldown_s"] = group_cooldown
        self.a = EventToYui(
            push_fn=lambda **kw: (
                self.said.append("".join(x.get("text", "") for x in kw["parts"]))
                or {"submitted": True}),
            runtime_dir=self.dir,
            throttle_seconds=throttle,
            state=self.state,
            # ★ 同一个假时钟喂给适配器和状态机 ——
            #   场景组冷却窗口与状态判定必须同源。
            clock=self.clock,
            **kwargs)

    def clear_group_window(self) -> None:
        """清掉场景组冷却窗口 —— 让用例只测被测的那一层。"""
        self.a._last_group_spoke.clear()

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
def rig():
    r = _Rig()
    yield r
    r.close()


# ==================================================================
# A · 状态机本身（纯逻辑）
# ==================================================================
class TestStateMachine:
    def test_initial_state_is_world(self) -> None:
        """★ 初值取 ``WORLD`` 而不是 ``UNKNOWN``。

        「还没开始看」不等于「看不见」—— 取 UNKNOWN 的话，
        插件刚起来的那几十秒（事件最密的时候）会整体静音，
        等于「加了陪玩功能」变成了「开场先哑三十秒」。
        """
        st = ChatState(clock=_Clock())
        assert st.state == STATE_WORLD
        assert st.base_state == STATE_WORLD

    def test_scene_changed_moves_between_states(self) -> None:
        st = ChatState(clock=_Clock())
        assert st.observe("SCENE_CHANGED", {"to": "COMBAT"}) == STATE_COMBAT
        assert st.observe("SCENE_CHANGED", {"to": "WORLD"}) == STATE_WORLD
        assert st.observe("SCENE_CHANGED", {"to": "COMBAT_RESULT"}) == STATE_COMBAT
        assert st.observe("SCENE_CHANGED", {"to": "BAG_INVENTORY"}) == STATE_WORLD

    def test_unknown_scene_id_is_unknown_state(self) -> None:
        """★ 不认识的场景 = 不知道自己在哪 = 不该乱说话。"""
        st = ChatState(clock=_Clock())
        assert st.observe("SCENE_CHANGED", {"to": "SOMEWHERE_ELSE"}) == STATE_UNKNOWN
        assert state_for_scene("SOMETHING_NEW") == STATE_UNKNOWN

    def test_perception_down_is_unknown_state(self) -> None:
        st = ChatState(clock=_Clock())
        assert st.observe("PERCEPTION_DOWN", {}) == STATE_UNKNOWN
        st2 = ChatState(clock=_Clock())
        assert st2.observe("CAPTURE_FAILED", {}) == STATE_UNKNOWN

    def test_position_change_resets_the_idle_timer(self) -> None:
        """★ 玩家动了 -> 挂机计时归零。"""
        clock = _Clock()
        st = ChatState(clock=clock)
        st.observe("SCENE_CHANGED", {"to": "WORLD"})
        clock.advance(IDLE_SECONDS + 10)
        assert st.state == STATE_IDLE
        st.observe("POSITION_CHANGED", {"x": 1.0, "y": 2.0, "z": 3.0})
        assert st.state == STATE_WORLD

    def test_idle_after_exactly_180_seconds(self) -> None:
        clock = _Clock()
        st = ChatState(clock=clock)
        st.observe("SCENE_CHANGED", {"to": "WORLD"})
        clock.advance(IDLE_SECONDS - 1)
        assert st.state == STATE_WORLD, "179s 还不是挂机"
        clock.advance(1)
        assert st.state == STATE_IDLE, "180s 必须算挂机"

    def test_combat_never_becomes_idle(self) -> None:
        """★ 战斗中没有「挂机」这回事。"""
        clock = _Clock()
        st = ChatState(clock=clock)
        st.observe("SCENE_CHANGED", {"to": "COMBAT"})
        clock.advance(IDLE_SECONDS * 10)
        assert st.state == STATE_COMBAT

    def test_scene_group_table_covers_all_thirteen_ids(self) -> None:
        """★ 13 个真实场景 id 一个不漏地落进 4 个组。"""
        assert len(SCENE_GROUP) == 13, SCENE_GROUP
        assert set(SCENE_GROUP.values()) == {
            GROUP_WORLD, GROUP_COMBAT, GROUP_MENU, GROUP_TRANSITION}
        assert SCENE_GROUP["WORLD"] == GROUP_WORLD
        assert SCENE_GROUP["COMBAT"] == GROUP_COMBAT
        assert SCENE_GROUP["COMBAT_RESULT"] == GROUP_COMBAT
        for menu_scene in ("MAIN_MENU", "DIALOG", "BAG_INVENTORY", "MAP_BIG",
                           "GAME_MENU", "NOTICE", "TELEPORT_MAP"):
            assert SCENE_GROUP[menu_scene] == GROUP_MENU, menu_scene
        for transition_scene in ("LOADING", "CUTSCENE", "UNKNOWN"):
            assert SCENE_GROUP[transition_scene] == GROUP_TRANSITION, transition_scene
        # 不认识的场景兜底到 TRANSITION（防刷屏的最保守选择）
        assert group_for_scene("SOMETHING_NEW") == GROUP_TRANSITION


# ==================================================================
# B · 场景门控（§9 自检清单 1、3）
# ==================================================================
class TestSceneGate:
    def test_combat_mutes_tier2_and_tier3(self) -> None:
        """★★ §9-1：``COMBAT`` 态下 20 条 ``ENEMY_DETECTED`` 一句都不播。"""
        r = _Rig()
        try:
            r.write("SCENE_CHANGED", to="COMBAT")
            r.drain()
            for _ in range(20):
                r.write("ENEMY_DETECTED", count=3)
            assert r.drain() == [], "战斗态不该出声"
        finally:
            r.close()

    def test_combat_never_mutes_tier1(self) -> None:
        """★★ §9-3：tier 1 异常在 ``COMBAT`` 态下**必须播**。"""
        r = _Rig()
        try:
            r.write("SCENE_CHANGED", to="COMBAT")
            r.drain()
            r.write("PANIC_TRIGGERED")
            said = r.drain()
            assert said, "急停必须播 —— 静音档的用户也要知道"
            assert tier_of("PANIC_TRIGGERED") == TIER_CRITICAL
        finally:
            r.close()

    def test_world_lets_tier2_through(self) -> None:
        r = _Rig()
        try:
            r.write("SCENE_CHANGED", to="WORLD")
            r.drain()
            r.clear_group_window()
            r.write("ENEMY_DETECTED", count=3)
            assert r.drain(), "大世界里该说的要说"
        finally:
            r.close()

    def test_unknown_state_only_lets_tier1_through(self) -> None:
        r = _Rig()
        try:
            r.write("PERCEPTION_DOWN")
            r.drain()
            r.write("ENEMY_DETECTED", count=3)
            assert r.drain() == [], "看不见的时候不该喊有怪"
            r.write("CAPTURE_FAILED")
            assert r.drain(), "但故障必须说"
        finally:
            r.close()

    def test_scene_tier_limits_are_the_documented_ones(self) -> None:
        """★ 把表钉住 —— 改它是有意行为，必须有人知道。"""
        assert SCENE_TIER_LIMIT[STATE_COMBAT] == TIER_CRITICAL
        assert SCENE_TIER_LIMIT[STATE_UNKNOWN] == TIER_CRITICAL
        assert SCENE_TIER_LIMIT[STATE_WORLD] == TIER_VERBOSE
        assert SCENE_TIER_LIMIT[STATE_IDLE] == TIER_NORMAL

    def test_state_is_observed_before_any_suppression(self) -> None:
        """★ 状态机必须**先**观察，否则被压掉的事件不会更新状态。

        症状：战斗中却在按大世界的话说话。
        """
        r = _Rig()
        try:
            # SCENE_CHANGED 是 tier3 —— 在 WORLD 态它会被场景组冷却压住，
            # 但**状态机必须已经看到它**。
            r.write("SCENE_CHANGED", to="COMBAT")
            r.drain()
            assert r.state.state == STATE_COMBAT, "场景变更没被状态机看到"
        finally:
            r.close()


# ==================================================================
# C · 同场景组冷却（§9 自检清单 2）
# ==================================================================
class TestSceneGroupCooldown:
    def test_twenty_position_changes_speak_at_most_once(self) -> None:
        """★★ §9-2：``WORLD`` 态下 20 条 ``POSITION_CHANGED`` 最多播 1 句。"""
        r = _Rig()
        try:
            r.write("SCENE_CHANGED", to="WORLD")
            r.drain()
            # 只清掉「场景切换」自己占掉的那个窗口 —— 下面要验的是
            # 20 条移动事件之间的互相压制。
            r.a._last_group_spoke.clear()
            for i in range(20):
                r.write("POSITION_CHANGED", x=float(i), y=0.0, z=0.0, dist_m=1.5)
            said = r.drain()
            assert len(said) <= 1, f"播了 {len(said)} 句：{said}"
        finally:
            r.close()

    def test_the_lock_is_shared_across_codes_in_the_same_group(self) -> None:
        """★ 同一场景组内**跨事件码**也只准一句。

        ``domain.start`` / ``task.error`` / ``domain.end`` 属于同一段场景，
        它们被同一把锁压住（``_PROMPT_GUIDE.md`` §5）。
        """
        r = _Rig()
        try:
            r.write("SCENE_CHANGED", to="WORLD")
            r.drain()
            r.a._last_group_spoke.clear()
            r.write("PATHFIND_START")
            first = r.drain()
            assert len(first) == 1, first
            r.write("POSITION_CHANGED", x=1.0, y=0.0, z=0.0, dist_m=2.0)
            assert r.drain() == [], "同组内第二条不该播"
        finally:
            r.close()

    def test_different_groups_have_independent_locks(self) -> None:
        """★ 场景组之间**各有一把锁**，互不影响。"""
        r = _Rig()
        try:
            r.write("SCENE_CHANGED", to="WORLD")
            r.drain()
            r.clear_group_window()
            r.write("PATHFIND_START")
            assert len(r.drain()) == 1
            # 换到菜单组 —— 那是另一把锁
            r.write("SCENE_CHANGED", to="BAG_INVENTORY")
            r.drain()
            r.clear_group_window()
            r.write("PATHFIND_START")
            assert len(r.drain()) == 1, "菜单组该有自己的窗口"
        finally:
            r.close()

    def test_the_lock_expires(self) -> None:
        """★ 冷却不是永久静音。"""
        clock = _Clock()
        r = _Rig(clock=clock)
        try:
            r.write("SCENE_CHANGED", to="WORLD")
            r.drain()
            r.a._last_group_spoke.clear()
            r.write("PATHFIND_START")
            assert len(r.drain()) == 1
            clock.advance(SCENE_GROUP_COOLDOWN_S - 1)
            r.write("POSITION_CHANGED", x=1.0, y=0.0, z=0.0, dist_m=2.0)
            assert r.drain() == [], "窗口内不该播"
            clock.advance(1)
            r.write("POSITION_CHANGED", x=2.0, y=0.0, z=0.0, dist_m=2.0)
            assert len(r.drain()) == 1, "窗口过了该放行"
        finally:
            r.close()

    def test_tier1_is_not_subject_to_the_lock(self) -> None:
        """★ 关键异常不该被「上一句还没凉」压住。"""
        r = _Rig()
        try:
            r.write("SCENE_CHANGED", to="WORLD")
            r.drain()
            r.a._last_group_spoke.clear()
            r.write("PATHFIND_START")
            assert len(r.drain()) == 1
            r.write("PANIC_TRIGGERED")
            assert len(r.drain()) == 1, "急停必须能插队"
        finally:
            r.close()

    def test_a_suppressed_event_does_not_consume_the_window(self) -> None:
        """★ 「只查不占」—— 被压住的事件不能宣称「刚播过」。"""
        r = _Rig()
        try:
            r.write("SCENE_CHANGED", to="WORLD")
            r.drain()
            r.a._last_group_spoke.clear()
            r.write("PATHFIND_START")
            assert len(r.drain()) == 1
            # 这一条被组冷却压住
            r.write("POSITION_CHANGED", x=1.0, y=0.0, z=0.0, dist_m=2.0)
            assert r.drain() == []
            assert len(r.a._last_group_spoke) == 1, "窗口不该被推进"
        finally:
            r.close()

    def test_the_lock_can_be_switched_off(self) -> None:
        """★ 调成 0 = 退回「只按事件码节流」的老行为。"""
        r = _Rig(group_cooldown=0.0)
        try:
            r.write("SCENE_CHANGED", to="WORLD")
            r.drain()
            r.write("PATHFIND_START")
            r.write("POSITION_CHANGED", x=1.0, y=0.0, z=0.0, dist_m=2.0)
            assert len(r.drain()) == 2, "关掉后两句都该说"
        finally:
            r.close()

    def test_the_lock_resets_across_runs(self) -> None:
        """★ 换 run = 新一段场景，上一场的窗口不该压住这一场。"""
        r = _Rig()
        try:
            r.write("SCENE_CHANGED", to="WORLD")
            r.drain()
            r.clear_group_window()
            r.write("PATHFIND_START")
            assert len(r.drain()) == 1
            (r.dir / "run_current.txt").write_text("t2", encoding="utf-8")
            r.stream = r.dir / "t2.stream.jsonl"
            r.write("PATHFIND_START")
            assert len(r.drain()) == 1, "第二个 run 的第一句不该被压住"
        finally:
            r.close()


# ==================================================================
# D · 挂机主动开口（§9 自检清单 4）
# ==================================================================
class TestIdleProactive:
    def test_speaks_once_after_180s_then_stays_quiet_for_600s(self) -> None:
        """★★ §9-4：IDLE 满 180s 后主动开口一次，此后 600s 内不再开口。"""
        clock = _Clock()
        r = _Rig(clock=clock)
        try:
            clock.advance(IDLE_SECONDS - 1)
            assert r.a.check_idle_proactive() is False, "还没满 180s"
            clock.advance(1)
            assert r.state.state == STATE_IDLE
            assert r.a.check_idle_proactive() is True, "满 180s 该主动开口"
            assert len(r.said) == 1

            clock.advance(IDLE_PROACTIVE_MIN_GAP_S - 1)
            assert r.a.check_idle_proactive() is False, "600s 内不该再开口"
            assert len(r.said) == 1

            clock.advance(1)
            assert r.a.check_idle_proactive() is True, "600s 之后可以再说一次"
            assert len(r.said) == 2
        finally:
            r.close()

    def test_never_speaks_proactively_outside_idle(self) -> None:
        """★ 大世界里**玩家还在动** -> 不是挂机；战斗里永远不是挂机。"""
        clock = _Clock()
        r = _Rig(clock=clock)
        try:
            # ① 大世界里，但玩家每分钟都在动
            for i in range(5):
                clock.advance(60.0)
                r.write("POSITION_CHANGED", x=float(i), y=0.0, z=0.0, dist_m=1.0)
                r.drain()
            assert r.state.state == STATE_WORLD, "玩家一直在动，不该算挂机"
            assert r.a.check_idle_proactive() is False, "没挂机就不该找话题"

            # ② 战斗里永远不是挂机
            r.write("SCENE_CHANGED", to="COMBAT")
            r.drain()
            clock.advance(IDLE_SECONDS * 5)
            assert r.state.state == STATE_COMBAT
            assert r.a.check_idle_proactive() is False, "战斗里更不许"
        finally:
            r.close()

    def test_never_speaks_proactively_while_blind(self) -> None:
        """★ 看不见的时候不许找话题 —— 说了主人也不知道。"""
        clock = _Clock()
        r = _Rig(clock=clock)
        try:
            r.write("PERCEPTION_DOWN")
            r.drain()
            clock.advance(IDLE_SECONDS * 5)
            assert r.state.state == STATE_UNKNOWN
            assert r.a.check_idle_proactive() is False, "看不见的时候不该找话题"
        finally:
            r.close()

    def test_proactive_line_goes_through_the_companion_channel(self) -> None:
        """★ 走 ``COMPANION_LINE`` 通路，不新增 code。"""
        clock = _Clock()
        r = _Rig(clock=clock)
        try:
            clock.advance(IDLE_SECONDS)
            assert r.a.check_idle_proactive() is True
            assert r.said, "应该有一句"
            assert any(line in r.said[0] for line in IDLE_PROACTIVE_LINES), r.said
        finally:
            r.close()

    def test_idle_lines_have_no_placeholders_or_directives(self) -> None:
        """★ 每句都要能被 YUI **原样复述**。"""
        for line in IDLE_PROACTIVE_LINES:
            assert "{" not in line and "}" not in line, line
            assert "（" not in line and "）" not in line, line
            assert "不要" not in line, line


# ==================================================================
# E · 台词随机池（§9 自检清单 6）
# ==================================================================
class TestPersonaPools:
    def test_pool_first_entry_equals_the_dictionary(self) -> None:
        """★★ 不变量：池子的第一句 = 字典里那一句。

        「只读字典」的读侧永远拿得到一句能播的话，
        不会因为池子抽空了而失声。
        """
        for code, pool in EVENT_TO_SPEECH_POOLS.items():
            assert pool, f"{code} 的池子不能为空"
            assert pool[0] == EVENT_TO_SPEECH[code], code

    def test_every_pool_has_at_least_two_lines(self) -> None:
        """★ 一句就谈不上「随机池」。"""
        for code, pool in EVENT_TO_SPEECH_POOLS.items():
            assert len(pool) >= 2, f"{code} 的池子只有 {len(pool)} 句"
        for action, pool in ACTION_SPEECH_POOLS.items():
            assert len(pool) >= 2, f"{action} 的池子只有 {len(pool)} 句"

    def test_action_pool_first_entry_equals_the_dictionary(self) -> None:
        for action, pool in ACTION_SPEECH_POOLS.items():
            assert pool[0] == ACTION_SPEECH[action], action

    def test_no_engineering_jargon_in_any_line(self) -> None:
        """★★ §3 硬边界第 1 条：台词里不许出现工程黑话。

        这些词一旦念出来，观众听到的是「某个模块报了错」，
        而不是 YUI 在跟他说话。

        ★ 黑名单里的 code（``SKIP_CODES``）**不参与**这次检查
          它们根本不会被播出 —— 念出来的是「引擎起来了」这种老台词，
          而 ``CORE_STARTED`` / ``CORE_STOPPED`` 早就进了黑名单。
          （它们的模板还留着，是给历史日志看的。）
        """
        banned = ("检测", "感知", "激活", "意图", "阈值", "provider", "fps",
                  "场景换成了", "引擎", "主循环", "模块")
        spoken = {code: text for code, text in EVENT_TO_SPEECH.items()
                  if code not in SKIP_CODES}
        lines = list(spoken.values()) + list(ACTION_SPEECH.values())
        for code, pool in EVENT_TO_SPEECH_POOLS.items():
            if code in SKIP_CODES:
                continue
            lines.extend(pool)
        for pool in ACTION_SPEECH_POOLS.values():
            lines.extend(pool)
        lines.extend(IDLE_PROACTIVE_LINES)
        for line in lines:
            for word in banned:
                assert word not in line, f"{word!r} 出现在台词 {line!r}"

    def test_scene_id_is_never_spoken(self) -> None:
        """★ 场景 id 是**工程枚举**，念出来等于暴露内部实现。"""
        assert "{to}" not in EVENT_TO_SPEECH["SCENE_CHANGED"]
        for code, pool in EVENT_TO_SPEECH_POOLS.items():
            for line in pool:
                assert "{to}" not in line, code

    def test_the_picker_does_not_repeat_the_same_line_back_to_back(self) -> None:
        """★ 池子小的时候纯随机会有一半概率抽回上一句 —— 那就是复读。"""
        r = _Rig()
        try:
            r.write("SCENE_CHANGED", to="WORLD")
            r.drain()
            picked = []
            for _ in range(8):
                r.a._last_group_spoke.clear()
                r.a._last_spoke.clear()
                r.write("PATHFIND_START")
                got = r.drain()
                if got:
                    picked.append(got[0])
            assert len(picked) >= 3, picked
            for a, b in zip(picked, picked[1:]):
                assert a != b, f"连着说了同一句：{a}"
        finally:
            r.close()

    def test_custom_templates_disable_the_pool(self) -> None:
        """★ 注入了自定义模板的调用方要的是「逐字可控」。

        池子会让同一句话随机变脸 —— 那正是它们在躲的事。
        """
        r = _Rig()
        try:
            a = EventToYui(
                push_fn=lambda **kw: {"submitted": True},
                runtime_dir=r.dir,
                templates={"PATHFIND_START": "固定的一句"})
            assert a._pools == {}, "自定义模板时默认池子必须作废"
        finally:
            r.close()
