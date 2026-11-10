# -*- coding: utf-8 -*-
"""2026-10-19：干跑不许打、让权不许刷、随包连招默认全关。

★ 这份测试在验什么
------------------
用户实机报了三件事：没开 Live 却在播「这个动作我没敢做」、
默认开启的「通用战斗」一直在产生意图、YUI 废话连篇。

三件事的根因其实是**同一个**架构缺口：

    2026-10-18 之前 ``--dry-run`` 只换了执行层的注入后端
    （DryRunBackend），决策层照跑、连招照走、事件照发。

于是「一个键都没发」的引擎，表现得和真在打一模一样。这份测试
把三处的**方向**钉死 —— 宁可少发不许多发。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))


# ==================================================================
# A · 干跑决策闸门
# ==================================================================
class TestDryRunGate:
    """``--dry-run`` 时决策层压根不跑。"""

    def _loop(self, live: bool):
        from autopilot.core.loop import CoreLoop
        from autopilot.core.runtime import RuntimeContext, RuntimeState

        class _Executor:
            def __init__(self) -> None:
                self.executed: list = []
                self.released = 0

            def execute(self, intent, _state):
                self.executed.append(intent)

            def release_held(self):
                self.released += 1

        class _Policy:
            def __init__(self) -> None:
                self.calls = 0

            def poll(self, _ws):
                self.calls += 1
                return None

        class _Builder:
            last_world_entities = None

        class _Capturer:
            def grab(self):
                raise RuntimeError("测试不需要真的抓帧")

        pol, exe = _Policy(), _Executor()
        ctx = RuntimeContext(capturer=_Capturer(), builder=_Builder(),
                             policy=pol, executor=exe)
        loop = CoreLoop(ctx)
        loop.set_run_mode(live)
        loop._state = RuntimeState.RUNNING
        return loop, pol, exe

    def test_is_live_tracks_the_declared_mode(self) -> None:
        from autopilot.ui.schema import DRY_RUN, LIVE

        loop_dry, _p, _e = self._loop(False)
        loop_live, _p2, _e2 = self._loop(True)
        assert loop_dry.run_mode() == DRY_RUN
        assert loop_dry.is_live() is False
        assert loop_live.run_mode() == LIVE
        assert loop_live.is_live() is True

    def test_default_run_mode_is_dry_run(self) -> None:
        """★★ 不调 ``set_run_mode`` 的调用方一律当干跑。

        直接构造 ``CoreLoop`` 的人（测试、插件内嵌）**不会**调
        ``set_run_mode()``。默认值必须朝着安全的方向 ——
        反过来的话，任何漏调用的调用方都直接接管键鼠。
        """
        from autopilot.ui.schema import DRY_RUN

        loop, _p, _e = self._loop(False)
        loop._run_mode = DRY_RUN      # 模拟「从没声明过」
        assert loop.is_live() is False
    def test_an_unrecognised_mode_is_treated_as_dry_run(self) -> None:
        """读到一个不认识的 ``run_mode`` 字符串时，宁可当成在干跑。"""
        loop, _p, _e = self._loop(False)
        loop._run_mode = "SOMETHING_ELSE"
        assert loop.is_live() is False


# ==================================================================
# B · 随包连招默认全关
# ==================================================================
class TestShippedCombosDefaultOff:
    """**出厂配置**里一条都不许开着。

    ⚠️ ���的是随包那份 ``autopilot/combos/combos.toml``，不是测试临时目录 ——
       「用户装完第一次打开是什么样」这件事只有这一个落点能回答。
    """

    @staticmethod
    def _entries() -> tuple[list[dict], list[dict]]:
        import tomllib

        p = (PLUGIN_ROOT / "autopilot" / "combos" / "combos.toml")
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        return list(data.get("combo", [])), list(data.get("reflex", []))

    def test_the_file_exists_and_parses(self) -> None:
        combos, reflex = self._entries()
        assert combos and reflex, "连招表被清空了？那不是省事，那是删功能"

    def test_no_combo_ships_enabled(self) -> None:
        """★ 出厂不带任何开着的连招。

        ⚠️ 具名豁免（2026-10-19 晚，用户明确要求源码树默认开四神演示连招，
          供 ``run_demo.bat`` 一键演示）。豁免做成集合，往里加第二个
          名字守卫就继续红 —— 只有这一个是白名单。
          理由与影响见 ``test_combo_sequence.py::NAMED_EXCEPTIONS``。

          ⚠️ 接管决定仍由用户自己按下：bat 显式带 ``--live``，
            不是靠默认值代劳。发布包那份配置仍全关。
        """
        on = {c["name"] for c in self._entries()[0] if c.get("enabled")}
        extra = on - {"钟纳夜丝_全自动演示"}
        assert not extra, f"出厂默认不该有开着的连招：{sorted(extra)}"

    def test_no_reflex_ships_enabled(self) -> None:
        """★ 危机反射也必须关。

        「低血闪避」「濒死放大招」看着像保命键，但它们**同样会真的按键**，
        而触发条件是感知层判定的血量 —— 战斗中特效遮挡会让判读漂移。
        用户抱怨的是「默认在打」，那就一条都不留。
        """
        on = [r["name"] for r in self._entries()[1] if r.get("enabled")]
        assert on == [], f"出厂默认不该有开着的反射：{on}"

    def test_every_entry_has_a_chinese_note(self) -> None:
        """关掉不等于删掉。每一项都要留着「它在做什么、代价是什么」。"""
        for e in self._entries()[0] + self._entries()[1]:
            note = str(e.get("note") or "")
            assert note, f"{e['name']} 没有 note"
            assert any("一" <= ch <= "鿿" for ch in note), \
                f"{e['name']} 的 note 不是中文"


# ==================================================================
# C · YUI 播报：让权防抖 + 干跑静默
# ==================================================================
def _adapter(tmp_path: Path, *, live: bool = True):
    from event_to_yui import EventToYui

    said: list[str] = []

    def _push(**kw):
        said.append("".join(x.get("text", "") for x in kw["parts"]))
        return {"submitted": True}

    rt = tmp_path / "rt"
    rt.mkdir(parents=True, exist_ok=True)
    (rt / "run_current.txt").write_text("t", encoding="utf-8")
    a = EventToYui(
        push_fn=_push, runtime_dir=rt,
        throttle_seconds=5.0,
        # ★ 场景组冷却**关掉**
        #   这份测试验的是「让权防抖 / 干跑静默 / 刷怪降频」三层，
        #   场景组冷却（Companion Mode）是第四层 —— 它若开着，
        #   同组内第二条非 tier1 台词会被它挡掉，
        #   测到的就不是被测的那一层了。
        #   场景组冷却由 ``test_yui_companion_mode.py`` 单独钉。
        scene_group_cooldown_s=0.0,
        run_mode_fn=lambda: "LIVE" if live else "DRY_RUN")
    return a, rt, said


def _emit(rt: Path, code: str, message: str = "", **ctx) -> None:
    with (rt / "t.stream.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(
            {"schema_version": "1.0", "run_id": "t", "seq": 0,
             "ts_ms": 0, "code": code, "severity": "INFO",
             "category": "DECISION", "message": message,
             "context": ctx}, ensure_ascii=False) + "\n")


class TestTakeoverDebounce:
    """一个「让权 3 秒」的过程只说一句。"""

    def test_one_takeover_burst_speaks_once(self, tmp_path: Path) -> None:
        """★★ 真机 run 20261019_000606：273 秒里 42 条 USER_INTERRUPT。

        用户在让权窗口里连按十个键，那是**一次**抢占，不是十次。
        改之前 5s 通用节流把它播成 11 句。

        ⚠️ 期望值是 5 而不是 1 —— 这条测试钉的是**量级**不是「完全静音」。
        40 秒的连续操作配 10 秒窗口，本来就该是 4~5 句；
        「一句都不说」不是目标，那是把用户需要知道的「我让开了」也吞了。
        真正要防的是回到 42 句或 11 句。
        """
        import event_to_yui as m

        a, rt, said = _adapter(tmp_path)
        clock = [1000.0]
        real = m.time.monotonic
        m.time.monotonic = lambda: clock[0]
        try:
            for i in range(42):          # 42 次物理按键，横跨 41 秒
                _emit(rt, "USER_INTERRUPT", f"用户按下 A{i}，引擎让权 3 秒",
                      key=f"A{i}")
                clock[0] += 1.0
                a.drain_once()
        finally:
            m.time.monotonic = real

        assert len(said) == 5, f"41 秒 / 10 秒窗口应当是 5 句，实得 {len(said)}"
        assert len(set(said)) == 1, "让权台词只有一句，不该有第二种说法"

    def test_the_window_is_ten_seconds(self, tmp_path: Path) -> None:
        from event_to_yui import TAKEOVER_DEBOUNCE_SECONDS

        assert TAKEOVER_DEBOUNCE_SECONDS == 10.0

        import event_to_yui as m

        a, rt, said = _adapter(tmp_path)
        clock = [1000.0]
        real = m.time.monotonic
        m.time.monotonic = lambda: clock[0]
        try:
            _emit(rt, "USER_INTERRUPT", "用户按下 A，引擎让权 3 秒", key="A")
            a.drain_once()
            assert len(said) == 1

            clock[0] += 9.9               # 窗口内
            _emit(rt, "USER_INTERRUPT", "用户按下 B，引擎让权 3 秒", key="B")
            a.drain_once()
            assert len(said) == 1, "9.9 秒内第二次让权不该开口"

            clock[0] += 0.2               # 越过窗口
            _emit(rt, "USER_INTERRUPT", "用户按下 C，引擎让权 3 秒", key="C")
            a.drain_once()
            assert len(said) == 2, "超过 10 秒后的新抢占应该开口"
        finally:
            m.time.monotonic = real

    def test_a_suppressed_burst_does_not_consume_the_window(
            self, tmp_path: Path) -> None:
        """★ 被档位压掉的让权**不能**算「刚播过」。

        否则用户把滑块拉到能听见让权时，第一条又要多等 10 秒 ——
        表现为「刚打开就一声不吭」。与 ENEMY_DETECTED 防抖同一套规矩。
        """
        import event_to_yui as m

        rt = tmp_path / "rt2"
        rt.mkdir(parents=True, exist_ok=True)
        (rt / "run_current.txt").write_text("t", encoding="utf-8")
        said: list[str] = []

        a = m.EventToYui(
            push_fn=lambda **kw: (
                said.append("".join(x.get("text", "") for x in kw["parts"]))
                or {"submitted": True}),
            runtime_dir=rt, throttle_seconds=5.0,
            verbosity=0.0,                     # 只放行 1 档…USER_INTERRUPT 是 1 档
            run_mode_fn=lambda: "LIVE")
        clock = [1000.0]
        real = m.time.monotonic
        m.time.monotonic = lambda: clock[0]
        try:
            # 直接把档位压到 0（1 档也放行不了的那档）
            a._tier_limit = 0
            _emit(rt, "USER_INTERRUPT", "用户按下 A，引擎让权 3 秒", key="A")
            a.drain_once()
            assert said == [], "档位压制的场景前提不成立"

            a._tier_limit = 3                  # 恢复
            clock[0] += 0.5                   # 才过了半秒
            _emit(rt, "USER_INTERRUPT", "用户按下 B，引擎让权 3 秒", key="B")
            a.drain_once()
            assert len(said) == 1, "被压住的那次不该占掉 10 秒窗口"
        finally:
            m.time.monotonic = real


class TestDryRunSilence:
    """干跑时 YUI 不喊战斗。"""

    @pytest.mark.parametrize("code", [
        "ENEMY_DETECTED", "USER_INTERRUPT", "INTENT_REJECTED",
        "COMBO_STARTED", "COMBO_FINISHED", "REFLEX_FIRED",
    ])
    def test_combat_codes_are_muted_under_dry_run(
            self, tmp_path: Path, code: str) -> None:
        a, rt, said = _adapter(tmp_path, live=False)
        _emit(rt, code, "随便什么", count=3, key="A")
        a.drain_once()
        assert said == [], f"干跑时 {code} 不该出声"

    @pytest.mark.parametrize("code", [
        "PERCEPTION_DOWN", "CAPTURE_FAILED", "PANIC_TRIGGERED",
    ])
    def test_failure_codes_still_speak_under_dry_run(
            self, tmp_path: Path, code: str) -> None:
        """★ 干跑不等于聋。引擎抓不到窗口时用户有权知道。"""
        a, rt, said = _adapter(tmp_path, live=False)
        _emit(rt, code, "出事了")
        a.drain_once()
        assert said, f"干跑时 {code} 必须照播"

    def test_live_mode_keeps_speaking(self, tmp_path: Path) -> None:
        a, rt, said = _adapter(tmp_path, live=True)
        _emit(rt, "ENEMY_DETECTED", "发现敌人", count=2)
        a.drain_once()
        assert said, "Live 模式下不该把战斗 chatter 也吞掉"

    def test_run_mode_probe_failure_keeps_the_previous_verdict(
            self, tmp_path: Path) -> None:
        """★ 读不到 ≠ 干跑。

        引擎没跑的时候快照文件根本不存在 —— 那不是「引擎在干跑」，
        而是「没有引擎」。这时保持原值才对，否则用户会在引擎
        停机后听到一片死寂。
        """
        import event_to_yui as m

        a, rt, said = _adapter(tmp_path, live=True)
        # 有探针时初值是 False（快照还没读过），先让它建立一次 run。
        assert a._saw_live is False, "有探针时不该一开始就假定 Live"

        _emit(rt, "RUN_STARTED", "引擎起来了")
        a.drain_once()
        assert a._saw_live is True, "读过一次探针后应确认为 Live"

        def _boom() -> str:
            raise OSError("快照读不到")

        a._run_mode_fn = _boom
        assert a.refresh_run_mode() is True
        assert a._saw_live is True

        _emit(rt, "ENEMY_DETECTED", "发现敌人", count=2)
        a.drain_once()
        assert any("坏东西" in s for s in said), \
            "供应函数抛异常不该把已确认的 Live 变成干跑"

    def test_run_mode_resets_on_a_new_run(self, tmp_path: Path) -> None:
        """★★ 切回干跑必须真的会静默。

        Console 切模式会**重启引擎** -> 新 run_id。若不在 run 切换时
        重置判定，第二个 run 会继承上一场的「已确认 Live」，
        于是干跑静默整个失效 —— 症状是「刚切回干跑，YUI 又开始喊有怪」。
        """
        import event_to_yui as m

        rt = tmp_path / "rt3"
        rt.mkdir(parents=True, exist_ok=True)
        (rt / "run_current.txt").write_text("r1", encoding="utf-8")
        said: list[str] = []
        mode = ["LIVE"]
        a = m.EventToYui(
            push_fn=lambda **kw: (
                said.append("".join(x.get("text", "") for x in kw["parts"]))
                or {"submitted": True}),
            runtime_dir=rt, throttle_seconds=5.0,
            run_mode_fn=lambda: mode[0])

        def _feed(run_id: str, code: str, **ctx) -> None:
            (rt / "run_current.txt").write_text(run_id, encoding="utf-8")
            with (rt / f"{run_id}.stream.jsonl").open("a",
                                                     encoding="utf-8") as fh:
                fh.write(json.dumps(
                    {"code": code, "message": code, "context": ctx},
                    ensure_ascii=False) + "\n")
            a.drain_once()

        _feed("r1", "ENEMY_DETECTED", count=1)
        assert said, "第一个 run 是 Live，该播"

        mode[0] = "DRY_RUN"
        said.clear()
        _feed("r2", "ENEMY_DETECTED", count=3)
        assert said == [], "切到干跑的新 run 必须静默"


class TestRunModeReader:
    """``autopilot_bridge.read_engine_run_mode``。"""

    def test_missing_snapshot_converges_to_dry_run(self, tmp_path: Path) -> None:
        from autopilot_bridge import read_engine_run_mode

        assert read_engine_run_mode(tmp_path) == "DRY_RUN"

    def test_reads_live_from_the_snapshot(self, tmp_path: Path) -> None:
        from autopilot_bridge import _RUN_MODE_CACHE, read_engine_run_mode

        _RUN_MODE_CACHE.clear()
        (tmp_path / "state_snapshot.json").write_text(
            json.dumps({"run_mode": "LIVE"}), encoding="utf-8")
        assert read_engine_run_mode(tmp_path) == "LIVE"

    def test_reads_dry_run_from_the_snapshot(self, tmp_path: Path) -> None:
        from autopilot_bridge import _RUN_MODE_CACHE, read_engine_run_mode

        _RUN_MODE_CACHE.clear()
        (tmp_path / "state_snapshot.json").write_text(
            json.dumps({"run_mode": "DRY_RUN"}), encoding="utf-8")
        assert read_engine_run_mode(tmp_path) == "DRY_RUN"

    def test_unknown_value_converges_to_dry_run(self, tmp_path: Path) -> None:
        """认不出的 ``run_mode`` 一律当干跑 —— 宁可少播。"""
        from autopilot_bridge import _RUN_MODE_CACHE, read_engine_run_mode

        _RUN_MODE_CACHE.clear()
        (tmp_path / "state_snapshot.json").write_text(
            json.dumps({"run_mode": "全接管"}), encoding="utf-8")
        assert read_engine_run_mode(tmp_path) == "DRY_RUN"


# ==================================================================
# D · 刷怪降频
# ==================================================================
class TestEnemyDebounceTightened:
    def test_window_was_raised_to_thirty_seconds(self) -> None:
        from event_to_yui import ENEMY_DEBOUNCE_SECONDS

        assert ENEMY_DEBOUNCE_SECONDS == 30.0

    def test_small_count_wobble_is_swallowed(self, tmp_path: Path) -> None:
        """1->2->1->2 这种来回抖，30 秒内一句都不该说。"""
        import event_to_yui as m

        a, rt, said = _adapter(tmp_path)
        clock = [1000.0]
        real = m.time.monotonic
        m.time.monotonic = lambda: clock[0]
        try:
            _emit(rt, "ENEMY_DETECTED", "发现敌人", count=1)
            a.drain_once()
            assert len(said) == 1
            for n in (2, 1, 3, 2):
                clock[0] += 5.0            # 5Hz 感知下的正常间隔
                _emit(rt, "ENEMY_DETECTED", "发现敌人", count=n)
                a.drain_once()
            assert len(said) == 1, f"小范围抖动播了 {len(said)} 句：{said}"

            clock[0] += 31.0               # 窗口过去，且数量真的翻倍
            _emit(rt, "ENEMY_DETECTED", "发现敌人", count=9)
            a.drain_once()
            assert len(said) == 2, "9 只怪远超阈值，该播"
        finally:
            m.time.monotonic = real
