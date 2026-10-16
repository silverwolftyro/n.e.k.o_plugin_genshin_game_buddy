"""传送契约 ① SELECT 的接线（2026-10-15）。

★★ 这组测试守护的是一个**曾经根本不存在**的接线
-------------------------------------------------
``LocalizationService.on_teleport_ocr()`` 在 2026-10-15 之前的调用方
是 **0 个**。后果不是报错，而是：

* ``locator.enabled = true`` 照常生效，服务照常被构造；
* ``builder._locate`` 照常每帧调 ``on_frame()``；
* 但状态机永远停在 ``UNINITIALIZED``（``localization_service.py:327``
  「UNINITIALIZED 原地不动」）；
* ``world_pos`` 恒 ``None`` -> ``PositionInfo.lost()`` -> ``quality="LOST"``。

**用户传送了十几次，position 全程 LOST，而日志里一行 OCR 都没有** ——
因为 ``teleport_ocr.recognize_teleport`` 同样没人调。

这类「静默不生效」的洞，单测是唯一能钉住它的东西：它不抛异常、不打
日志，只表现为一个恒定的值。
"""

from __future__ import annotations

import numpy as np
import pytest

from autopilot.contracts.state_packet import SCENE_UNKNOWN, SCENE_WORLD
from autopilot.perception import teleport_ocr as tp
from autopilot.perception.builder import StateBuilder

pytestmark = pytest.mark.unit

_IMG = np.zeros((720, 1280, 3), dtype=np.uint8)


class _FakeTarget:
    def __init__(self, name="枫丹廷", tran=(1.0, 2.0, 3.0)):
        self.matched_id = "tp-1"
        self.name = name
        self.position = (0.0, 0.0, 0.0)
        self.tran_position = tran
        self.confidence = 0.91


class _FakeLocator:
    """鸭子类型定位服务，带 ``on_teleport_ocr``。"""

    def __init__(self) -> None:
        self.feed: list = []
        self.frames: list = []

    def on_frame(self, image, ts_sec):
        self.frames.append((image, ts_sec))
        return None

    def on_teleport_ocr(self, target) -> None:
        self.feed.append(target)


class _LocatorNoHook:
    """**没有** ``on_teleport_ocr`` 的鸭子类型 —— 接上之前的老形态。"""

    def __init__(self) -> None:
        self.frames: list = []

    def on_frame(self, image, ts_sec):
        self.frames.append((image, ts_sec))
        return None


def _builder(locator, **kw) -> tuple[StateBuilder, list]:
    got: list = []
    b = StateBuilder(locator=locator, **kw)
    b._emit = got.append          # type: ignore[method-assign]
    return b, got


class TestWiring:
    def test_disabled_when_no_locator(self) -> None:
        b, ev = _builder(None)
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 100.0, 1)
        assert ev == [], "没有 locator 却发事件了"

    def test_duck_typed_locator_without_hook_is_skipped(self) -> None:
        """★ 老 locator 不受影响 —— 与接线之前**完全一致**。"""
        loc = _LocatorNoHook()
        b, ev = _builder(loc)
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 100.0, 1)
        assert ev == []

    def test_world_scene_never_runs_ocr(self, monkeypatch) -> None:
        """大世界里信息框根本不存在，跑 OCR 纯属浪费。"""
        called = []
        monkeypatch.setattr(tp, "recognize_teleport",
                            lambda img: called.append(img) or None)
        loc = _FakeLocator()
        b, _ = _builder(loc)
        b._feed_teleport_ocr(_IMG, SCENE_WORLD, 100.0, 1)
        assert called == []
        assert loc.feed == []

    def test_hit_feeds_the_locator(self, monkeypatch) -> None:
        tgt = _FakeTarget()
        monkeypatch.setattr(tp, "recognize_teleport", lambda img: tgt)
        loc = _FakeLocator()
        b, ev = _builder(loc)
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 100.0, 1)
        assert loc.feed == [tgt], "契约① 没被触发"
        assert any(getattr(e, "code", "") == "TELEPORT_OCR_HIT" for e in ev)

    def test_throttle_skips_repeated_frames(self, monkeypatch) -> None:
        """地图开着几十秒，每 0.5s 跑一次 OCR 就够了。"""
        n = []
        monkeypatch.setattr(tp, "recognize_teleport",
                            lambda img: n.append(1) or None)
        b, _ = _builder(_FakeLocator(), teleport_ocr_interval_s=0.5)
        for k in range(10):                       # 10 帧 = 2 秒（0.2s/帧）
            b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 100.0 + k * 0.2, 1)
        assert len(n) == 4, f"2 秒内跑了 {len(n)} 次 OCR（应为 4）"

    def test_same_target_not_reported_twice(self, monkeypatch) -> None:
        """菜单重绘会反复命中同一个点 —— 不能刷屏。"""
        tgt = _FakeTarget()
        monkeypatch.setattr(tp, "recognize_teleport", lambda img: tgt)
        loc = _FakeLocator()
        b, ev = _builder(loc)
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 100.0, 1)
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 101.0, 2)   # 换目标
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 102.0, 3)
        hits = [e for e in ev if getattr(e, "code", "") == "TELEPORT_OCR_HIT"]
        assert len(hits) == 1, f"同一目标报了 {len(hits)} 次"
        assert len(loc.feed) == 3, "但三帧都要喂（契约每次都重新挂起）"

    def test_switching_target_reports_again(self, monkeypatch) -> None:
        monkeypatch.setattr(tp, "recognize_teleport",
                            lambda img: _FakeTarget("A"))
        loc = _FakeLocator()
        b, ev = _builder(loc)
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 100.0, 1)
        monkeypatch.setattr(tp, "recognize_teleport",
                            lambda img: _FakeTarget("B"))
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 101.0, 2)
        hits = [e for e in ev if getattr(e, "code", "") == "TELEPORT_OCR_HIT"]
        assert len(hits) == 2


class TestFailurePaths:
    """传送识别不到是**常态**（加载中、背包、任何弹窗）—— 不能炸。"""

    def test_no_hit_is_silent(self, monkeypatch) -> None:
        monkeypatch.setattr(tp, "recognize_teleport", lambda img: None)
        loc = _FakeLocator()
        b, ev = _builder(loc)
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 100.0, 1)
        assert loc.feed == []
        assert ev == [], "没命中不该发事件（否则加载屏上会刷屏）"

    def test_ocr_exception_is_contained(self, monkeypatch) -> None:
        def boom(img):
            raise FileNotFoundError("找不到传送点数据")
        monkeypatch.setattr(tp, "recognize_teleport", boom)
        loc = _FakeLocator()
        b, ev = _builder(loc)
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 100.0, 1)   # 不抛
        assert loc.feed == []
        assert any(getattr(e, "code", "") == "TELEPORT_OCR_FAILED"
                   for e in ev)

    def test_invalid_target_is_contained(self, monkeypatch) -> None:
        """目标没有三轴落点 -> 响亮失败但**不炸感知**。"""
        bad = _FakeTarget(tran=None)
        monkeypatch.setattr(tp, "recognize_teleport", lambda img: bad)

        class _Strict(_FakeLocator):
            def on_teleport_ocr(self, target):
                raise ValueError("目标没有可用的落点坐标")

        b, ev = _builder(_Strict())
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 100.0, 1)   # 不抛
        assert any(getattr(e, "code", "") == "TELEPORT_TARGET_INVALID"
                   for e in ev)


class TestStackAssembly:
    """★ 这一组是为了堵一个**真实的洞**（2026-10-15）。

    ``build_perception`` 里写了 ``loc_cfg``，但那是 :func:`build_locator`
    的**局部变量** —— 本函数里根本没定义。结果是工具一启动就
    ``NameError: name 'loc_cfg' is not defined``，而**全量单测照样全绿**。

    为什么会全绿？因为在那之前**没有任何测试真的调用过
    ``build_perception``**。``py_compile`` 也查不出来（名字在运行时才解析）。

    所以这里直接把它调起来。不碰游戏：``open_capturer=False``。
    """

    @staticmethod
    def _cfg(**loc_kw):
        from autopilot.config.loader import load_config

        cfg = load_config()
        cfg.locator.enabled = bool(loc_kw.pop("enabled", True))
        for k, v in loc_kw.items():
            setattr(cfg.locator, k, v)
        return cfg

    def test_runs_with_shipped_config(self) -> None:
        """不管 ``defaults.toml`` 当前怎么写，它都得能组装出来。"""
        from autopilot.config.loader import load_config
        from autopilot.tools._stack import build_perception

        st = build_perception(load_config(), open_capturer=False)
        assert st.builder is not None
        assert st.capturer is not None

    def test_locator_enabled_wires_teleport_ocr(self) -> None:
        from autopilot.tools._stack import build_perception

        st = build_perception(
            self._cfg(enabled=True, teleport_ocr_enabled=True,
                      teleport_ocr_interval_s=0.25),
            open_capturer=False)
        assert st.locator is not None
        assert st.builder._teleport_ocr is True
        assert st.builder._tp_ocr_interval_s == 0.25

    def test_teleport_ocr_kill_switch_is_respected(self) -> None:
        from autopilot.tools._stack import build_perception

        st = build_perception(
            self._cfg(enabled=True, teleport_ocr_enabled=False),
            open_capturer=False)
        assert st.locator is not None
        assert st.builder._teleport_ocr is False

    def test_locator_off_turns_teleport_ocr_off_too(self) -> None:
        from autopilot.tools._stack import build_perception

        st = build_perception(self._cfg(enabled=False), open_capturer=False)
        assert st.locator is None
        assert st.builder._teleport_ocr is False


class TestKillSwitch:
    def test_teleport_ocr_can_be_turned_off(self, monkeypatch) -> None:
        called = []
        monkeypatch.setattr(tp, "recognize_teleport",
                            lambda img: called.append(1) or _FakeTarget())
        loc = _FakeLocator()
        b, _ = _builder(loc, teleport_ocr=False)
        b._feed_teleport_ocr(_IMG, SCENE_UNKNOWN, 100.0, 1)
        assert called == []
        assert loc.feed == []

    def test_default_is_on(self) -> None:
        """开了 locator 就该是能用的 —— 默认关等于把这个 bug 留着。"""
        b = StateBuilder(locator=_FakeLocator())
        assert b._teleport_ocr is True
