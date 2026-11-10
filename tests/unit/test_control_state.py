"""接管状态（run_mode / user_takeover）从引擎一路传到卡片的三态显示。

★ 这份测试在验什么
------------------
2026-10-18 之前，「它现在到底动不动我的键」这件事**在 UI 上完全看不见**：

    · ``--dry-run`` 和 ``--live`` 的日志、快照、卡片长得**一模一样**；
    · 用户按下键让权 3 秒，卡片颜色纹丝不动 ——
      于是「我刚抢了一把，它怎么还在打」看着像 bug。

现在这两个值从 ``run_loop`` 的 argparse → :class:`CoreLoop` → 快照 →
卡片 headline 一路传下来。这里把**每一段**都钉住，重点是：

    · 方向 —— 漏了「真发按键」时永远显示成「仅观察」，反之更糟；
    · 向后兼容 —— 老快照文件读进来不能把 overlay 打死；
    · 重绘 —— 指纹必须含 control，否则绿点不会变蓝（这正是症状本身）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.ui.overlay import layout, theme  # noqa: E402
from autopilot.ui.schema import (  # noqa: E402
    DRY_RUN, LIVE, RUN_MODES, CaptureStat, Health, RuntimePhase, StateSnapshot,
)


def _snap(**kw) -> StateSnapshot:
    """一份「引擎正在正常跑」的快照。默认 dry-run —— 与发行包一致。"""
    d = dict(state=RuntimePhase.RUNNING, health=Health.OK, scene="WORLD",
             capture=CaptureStat(active="dxcam"))
    d.update(kw)
    return StateSnapshot(**d)


# ==================================================================
# 1 · 快照契约
# ==================================================================
class TestSnapshotContract:
    def test_defaults_are_the_safe_side(self) -> None:
        """★ 默认必须是 DRY_RUN。

        ``StateSnapshot()`` 到处都在被构造（ChannelReader 读不到文件时
        就给一个空的）。默认值朝「不动用户的键」这个方向，才不会
        在任何一条降级路径上骗人说「接管中」。
        """
        s = StateSnapshot()
        assert s.run_mode == DRY_RUN
        assert s.user_takeover is False
        assert s.is_live is False
        assert s.control_state == DRY_RUN

    def test_run_modes_are_exactly_two(self) -> None:
        assert RUN_MODES == (DRY_RUN, LIVE)

    def test_live_is_live(self) -> None:
        s = _snap(run_mode=LIVE)
        assert s.is_live is True
        assert s.control_state == LIVE

    def test_takeover_wins_over_live(self) -> None:
        """★ 用户按键的那一刻，引擎当下就是不动 —— 哪怕它跑在 LIVE。

        把让权排在模式**后面**会让用户在自己抢权时看到一片绿，
        那恰恰是「我按了它怎么还在打」的来源。
        """
        assert _snap(run_mode=LIVE, user_takeover=True).control_state == "TAKEOVER"
        # 干跑时让权也一样要显示 —— 否则「我按了」没反应更像坏了
        assert _snap(run_mode=DRY_RUN,
                     user_takeover=True).control_state == "TAKEOVER"

    def test_unknown_mode_falls_back_to_dry_run(self) -> None:
        """★ 不认识的 run_mode **绝不能**当成 LIVE。

        将来加了第三种模式（REPLAY / PROBE），旧读侧拿到的是陌生值。
        那个方向的默认值必须是「不接管」——
        显示成「完全接管」而引擎其实不动手，是最坏的撒谎。
        """
        assert _snap(run_mode="WHATEVER").is_live is False
        assert _snap(run_mode="WHATEVER").control_state == DRY_RUN
        assert _snap(run_mode="").is_live is False
        assert _snap(run_mode=None).is_live is False

    def test_old_snapshot_without_the_fields_still_loads(self) -> None:
        """★ 向后兼容：上一代引擎写的快照文件读进来不能炸。

        overlay 可能拿到**没重启**的老引擎写的文件，那时这两个键
        根本不存在。pydantic 校验失败会让卡片**整个崩掉** ——
        用户连急停键都看不见了。
        """
        old = _snap().model_dump()
        old.pop("run_mode")
        old.pop("user_takeover")
        s = StateSnapshot.model_validate(old)
        assert s.run_mode == DRY_RUN
        assert s.user_takeover is False


# ==================================================================
# 2 · CoreLoop 是唯一真值来源
# ==================================================================
class TestCoreLoopRunMode:
    def test_default_is_dry_run(self) -> None:
        """★ 直接 new CoreLoop 的人不会调 set_run_mode。

        默认 LIVE 会让**任何**漏调用的调用方接管键鼠。
        """
        from autopilot.core.loop import CoreLoop

        loop = CoreLoop(_fake_ctx())
        assert loop.run_mode() == DRY_RUN
        assert loop.runtime_info() == (DRY_RUN, False)

    def test_set_run_mode(self) -> None:
        from autopilot.core.loop import CoreLoop

        loop = CoreLoop(_fake_ctx())
        loop.set_run_mode(True)
        assert loop.run_mode() == LIVE
        loop.set_run_mode(False)
        assert loop.run_mode() == DRY_RUN

    def test_runtime_info_reads_takeover_live(self) -> None:
        """★ 两个字段必须来自**同一时刻**。

        分两次读的话，让权窗口可能在两次之间到期 —— 快照记下
        「不在让权中」而 UI 刚读到 True，那种不一致极难复现。
        """
        from autopilot.core.loop import CoreLoop

        loop = CoreLoop(_fake_ctx())
        loop.set_run_mode(True)
        # 让权是**动态**的（2026-10-19）：按着就一直让权。
        loop._takeover.note_press("E")
        loop._takeover.note_hold("E")
        assert loop.runtime_info() == (LIVE, True)
        loop._takeover.clear()
        assert loop.runtime_info() == (LIVE, False)


def _fake_ctx():
    """最小 RuntimeContext —— CoreLoop 只在 __init__ 里存引用，不调用。"""

    class _Bus:
        def subscribe(self, *_a, **_k) -> None:
            pass

        def publish(self, *_a, **_k) -> None:
            pass

    class _C:
        bus = _Bus()
        executor = None
        policy = None
        capturer = None
        stats = None
        active_slots = None

    return _C()


# ==================================================================
# 3 · 卡片三态
# ==================================================================
class TestOverlayThreeStates:
    def _head(self, **kw):
        m = layout.build_panel(_snap(**kw))
        return m.lines[0]

    def test_three_styles_exist(self) -> None:
        assert set(layout.CONTROL_STYLE) == {"TAKEOVER", "LIVE", "DRY_RUN"}

    def test_each_state_has_its_own_color(self) -> None:
        """★ 三种颜色必须**互不相同**。

        颜色相同的两个状态在卡片上完全无法区分 —— 那这个功能等于没做。
        """
        colors = {k: v[1] for k, v in layout.CONTROL_STYLE.items()}
        assert len(set(colors.values())) == 3, colors

    def test_live_is_green_dry_run_is_yellow_takeover_is_blue(self) -> None:
        assert layout.CONTROL_STYLE["LIVE"][1] == theme.ACCENT_RUN
        assert layout.CONTROL_STYLE["DRY_RUN"][1] == theme.ACCENT_PAUSE
        assert layout.CONTROL_STYLE["TAKEOVER"][1] == theme.ACCENT_IDLE

    def test_headline_text_names_the_mode(self) -> None:
        """★ 光靠颜色不够 —— 色觉障碍用户和截图都要能读出文字。"""
        assert "完全接管" in self._head(run_mode=LIVE).text
        assert "仅观察" in self._head().text
        assert "用户接管中" in self._head(run_mode=LIVE,
                                        user_takeover=True).text

    def test_headline_dot_carries_the_control_color(self) -> None:
        """★ 圆点（accent）用接管色，文字用 head_color。

        两者刻意分开：head_color 承载「无会话 / 已完成 / 已崩溃」这套
        诊断语义（demo 粗糙度审计 A6 加的），接管色承载「能不能动我的键」。
        合成一个颜色会让其中一个语义消失。
        """
        assert self._head(run_mode=LIVE).accent == theme.ACCENT_RUN
        assert self._head().accent == theme.ACCENT_PAUSE
        assert self._head(run_mode=LIVE,
                          user_takeover=True).accent == theme.ACCENT_IDLE

    def test_green_to_blue_changes_the_fingerprint(self) -> None:
        """★★ 卡片必须**立刻**从绿变蓝。

        overlay 只在 ``content_key()`` 变了才重绘（省电的关键，见 app.py
        模块文档）。control **不进指纹**的话，这次重绘正好被比对挡掉 ——
        于是「快照已经变了、卡片还是绿的」，而那正是本功能要治的病。
        """
        green = layout.build_panel(_snap(run_mode=LIVE))
        blue = layout.build_panel(_snap(run_mode=LIVE, user_takeover=True))
        assert green.content_key() != blue.content_key()
        # 不只是圆点：headline 文字本身也变了，双保险
        assert green.lines[0].text != blue.lines[0].text

    def test_row_count_and_panic_line_are_unchanged(self) -> None:
        """★ 不能为了塞模式而挪动「急停」那一行。

        ``layout.py`` 里写明「第 5 行是用户的肌肉记忆」。加一行会把
        慌乱时的误按风险往上引 —— 模式显示不值那个代价，
        所以模式**并进 headline**，不新增行。
        """
        m = layout.build_panel(_snap(run_mode=LIVE))
        assert len(m.lines) == 5
        assert m.lines[-1].text.startswith("急停")

    def test_control_is_exposed_on_the_model(self) -> None:
        """app.py 靠它决定要不要把整块卡片描成蓝色。"""
        assert layout.build_panel(_snap(run_mode=LIVE)).control == "LIVE"
        assert layout.build_panel(
            _snap(run_mode=LIVE, user_takeover=True)).control == "TAKEOVER"

    def test_old_snapshot_object_without_control_state_survives(self) -> None:
        """★ 没有 control_state 属性的对象（老引擎的解析结果）不能打死卡片。

        这里刻意传一个**裸对象**，不是 StateSnapshot ——
        对应「overlay 读到上一代引擎写的文件」那条真实路径。
        """
        class _Old:
            state = RuntimePhase.RUNNING.value
            health = Health.OK.value
            scene = "WORLD"
            tick_hz = 5.0
            tick_p95_ms = 10.0
            panic_key = "F12"
            intent = None
            capture = None

        m = layout.build_panel(_Old())
        assert m.control == "DRY_RUN"       # 兜底成保守值
        assert "仅观察" in m.lines[0].text

    def test_offline_panel_does_not_claim_a_control_state(self) -> None:
        """★ 离线时不谈模式。

        引擎没跑的时候显示「完全接管」毫无意义，而用户会读成
        「它接管了但没动静」。
        """
        m = layout.build_panel(StateSnapshot())
        assert m.control is None
        assert "离线" in m.lines[0].text


# ==================================================================
# 4 · run_loop 真的把值填进快照
# ==================================================================
class TestRunLoopWiring:
    def test_snapshot_carries_both_fields(self) -> None:
        """★ 端到端：CoreLoop 的状态 -> ``_ui_snapshot`` -> StateSnapshot。

        这是唯一一处证明「字段加了但没人填」的测试。前面几条全绿
        而这里红的话，UI 上就永远是默认值 —— 而症状是
        「卡片永远写仅观察，用户选了 Live 也没用」，极难自查。
        """
        from autopilot.tools.run_loop import _ui_snapshot
        from autopilot.core.loop import CoreLoop

        loop = CoreLoop(_fake_ctx())
        loop.set_run_mode(True)
        loop._takeover.note_press("E")
        loop._takeover.note_hold("E")

        class _Ctx:
            stats = None
            capturer = None
            executor = None

        snap = _ui_snapshot(_Ctx(), loop, "run_x", "WORLD", "USE_SKILL")
        assert snap.run_mode == LIVE
        assert snap.user_takeover is True
        assert snap.control_state == "TAKEOVER"

    def test_snapshot_falls_back_when_loop_is_broken(self) -> None:
        """★ 拿不到 loop 就退回保守值，不许抛。

        UI 线程死了的话 overlay 会一直显示「离线」，而引擎明明
        还好好跑着 —— 那比少显示一个模式严重得多。
        """
        from autopilot.tools.run_loop import _ui_snapshot

        class _Ctx:
            stats = None
            capturer = None
            executor = None

        class _Bad:
            def runtime_info(self):
                raise RuntimeError("boom")

        snap = _ui_snapshot(_Ctx(), _Bad(), "run_x", "WORLD", "")
        assert snap.run_mode == DRY_RUN
        assert snap.user_takeover is False


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
