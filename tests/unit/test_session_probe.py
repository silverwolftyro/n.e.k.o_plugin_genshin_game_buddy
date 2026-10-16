"""会话自动建立探针（3.0-a）的单元测试。

背景：`ocr_perceive` 在 `self._manager.current is None` 时直接 early-return，
真机实测 803 次 tick / 803 次空转。根因是会话唯一写入者是 LLM 工具
`set_game`，全新安装时没人调它，`restore()` 就永远返回 None。

这里覆盖新加的 `_probe_game_for_title` / `_probe_and_activate` /
`session_probe` / `activate_game_entry`，全部**不碰真实窗口**：
`find_genshin_window` 一律 monkeypatch 掉，所以这些用例在任何机器上
（包括没有 pywin32 的）都能跑。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from plugin.plugins.genshin_game_buddy import GenshinGameBuddyPlugin
from plugin.plugins.genshin_game_buddy.game_registry import GameEntry, GameRegistry, PluginOptions
from plugin.sdk.plugin import Err

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------ 夹具

def _entry(game_id: str, *, display: str, aliases: tuple[str, ...],
           enabled: bool = True) -> GameEntry:
    return GameEntry(
        game_id=game_id,
        display_name=display,
        aliases=aliases,
        terms_dir=f"terms/{game_id}",
        enabled=enabled,
    )


def _registry(*entries: GameEntry) -> GameRegistry:
    return GameRegistry(games=tuple(entries))


GENS = _entry("genshin", display="原神",
              aliases=("原神", "Genshin", "Genshin Impact", "genshin"))
WUWA = _entry("wuthering_waves", display="鸣潮",
              aliases=("鸣潮", "Wuthering Waves", "WuWa", "wuthering"))


class _FakeSession:
    def __init__(self, game_id: str, display_name: str) -> None:
        self.game_id = game_id
        self.display_name = display_name


class _FakeManager:
    def __init__(self, current: Any = None) -> None:
        self.current = current


def _plugin(
    *,
    registry: GameRegistry | None = None,
    options: PluginOptions | None = None,
    manager: _FakeManager | None = None,
) -> Any:
    """绕过 __init__ 造一个只带探针所需字段的插件实例。

    和 tests/unit/test_perception_state.py 同一个路子：`__init__` 会连
    OCR 引擎、线程池一起建起来，而这里要测的三个方法一个都用不到。
    """
    p = GenshinGameBuddyPlugin.__new__(GenshinGameBuddyPlugin)
    p._registry = registry if registry is not None else _registry(GENS, WUWA)
    p._options = options if options is not None else PluginOptions()
    p._manager = manager if manager is not None else _FakeManager(None)
    p.logger = logging.getLogger("test.session_probe")
    p._probe_last_log_at = 0.0
    p._probe_last_reason = ""
    p._probe_activated = ""
    p._scenes_loaded = 0
    return p


def _stub_window(p: Any, monkeypatch: pytest.MonkeyPatch, window: Any) -> None:
    """把**两条**找窗路径都 stub 掉。

    ★ 为什么必须两条（2026-10-10 实测踩到）
      ---------------------------------------
      原来只 stub 了主路径 ``find_genshin_window``，**漏了标题兜底路径**
      ``_find_window_by_title``（``__init__.py:91`` 导入、``:1114`` 调用）。
      当时没出事，纯粹是因为兜底路径是坏的 —— ``win32gui.
      GetWindowThreadProcessId`` 写错了模块（应为 win32process），恒返回
      None。那个 typo 修好之后，**真机原神一开**，兜底路径就真的命中了：

          会话探测（timer）主路径没找到，已由标题兜底路径命中：title='原神' pid=64580

      于是 ``_probe_and_activate`` 走进成功分支，碰一个 ``__new__`` 绕过
      ``__init__`` 造出来的 stub 实例 —— 它压根没有 ``_state_lock`` ——
      报出一个跟本用例毫无关系的 AttributeError。测试红的原因和它自己
      宣称要测的东西完全无关。

      教训：**"这个用例能过"不等于"它在测它宣称的东西"**。只要它偷偷吃了
      真实世界的输入，它测的就是环境而不是代码。
    """
    monkeypatch.setattr(
        "plugin.plugins.genshin_game_buddy.find_genshin_window",
        lambda: window,
    )
    # 第二条：标题兜底。漏了它，真机开着游戏就会串进来（见 docstring）。
    monkeypatch.setattr(
        "plugin.plugins.genshin_game_buddy._find_window_by_title",
        lambda: window,
    )


# ======================================================== _probe_game_for_title

def test_title_matches_genshin_chinese():
    p = _plugin()
    assert p._probe_game_for_title("原神") == ("genshin", "title")


def test_title_matches_genshin_english_with_space():
    """标题 "Genshin Impact" 必须命中 —— 两边都过 normalize_key 去空白。"""
    p = _plugin()
    assert p._probe_game_for_title("Genshin Impact") == ("genshin", "title")


def test_title_matches_genshin_english_no_space():
    """窗口标题常常已经是无空格形式（进程名/内部标题）。"""
    p = _plugin()
    assert p._probe_game_for_title("GenshinImpact 3.4") == ("genshin", "title")


def test_title_matches_wuthering_waves():
    p = _plugin()
    assert p._probe_game_for_title("鸣潮") == ("wuthering_waves", "title")


def test_unmatched_title_falls_back_to_default_game():
    p = _plugin(options=PluginOptions(default_game="wuthering_waves"))
    game_id, rule = p._probe_game_for_title("某个不认识的窗口")
    assert game_id == "wuthering_waves"
    assert rule == "default_game"


def test_unmatched_title_and_no_default_falls_back_to_first_enabled():
    p = _plugin()
    game_id, rule = p._probe_game_for_title("记事本")
    assert game_id == "genshin"
    assert rule == "fallback:first_enabled"


def test_disabled_game_is_never_auto_registered():
    """下架的游戏不能被兜底逻辑复活 —— enabled 是硬门槛。"""
    registry = _registry(
        _entry("genshin", display="原神", aliases=("原神",), enabled=False),
        WUWA,
    )
    p = _plugin(registry=registry)
    game_id, rule = p._probe_game_for_title("记事本")
    assert game_id == "wuthering_waves"
    assert rule == "fallback:first_enabled"


def test_no_enabled_games_returns_empty():
    registry = _registry(
        _entry("genshin", display="原神", aliases=("原神",), enabled=False),
    )
    p = _plugin(registry=registry)
    game_id, rule = p._probe_game_for_title("原神")
    assert game_id == ""
    assert rule == "fallback:none_enabled"


# ========================================================= _probe_and_activate

def test_existing_session_skips_window_scan(monkeypatch):
    """已有会话时**一个窗口都不扫** —— 探测是兜底，不该抢占用户手动切的游戏。"""
    p = _plugin(manager=_FakeManager(_FakeSession("wuthering_waves", "鸣潮")))

    called = []
    monkeypatch.setattr(
        "plugin.plugins.genshin_game_buddy.find_genshin_window",
        lambda: called.append(1) or {"title": "原神"},
    )
    ok, msg = asyncio.run(p._probe_and_activate("timer"))
    assert ok is False
    assert "已有会话" in msg and "鸣潮" in msg
    assert called == []          # 关键：没扫窗口
    assert p._probe_activated == "wuthering_waves"


def test_no_window_is_throttled(monkeypatch):
    """没找到窗口是常态；60s 内只记一次日志，不能每 5s 刷一行。"""
    p = _plugin()
    _stub_window(p, monkeypatch, None)
    records: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    h = _Cap()
    p.logger.addHandler(h)
    try:
        asyncio.run(p._probe_and_activate("timer"))
        asyncio.run(p._probe_and_activate("timer"))
    finally:
        p.logger.removeHandler(h)

    assert len(records) == 1
    assert "没找到游戏窗口" in records[0]
    assert p._probe_last_reason == "no_window"


def test_no_window_throttle_reopens_after_window(monkeypatch):
    """节流窗口过去后，"没找到窗口"这条日志必须还能再记一次。

    ★ 这里必须 stub 窗口，不能吃真机的 find_genshin_window()
      （2026-10-10 加的）。原来这条用例没 stub，直接调真函数 —— 它能过，
      **只是因为真函数是坏的**：win32gui.GetWindowThreadProcessId 写错了模块，
      恒返回 None。一旦有人把那个 typo 修好、原神又正好开着，它就会反过来
      挂在这里，报一个跟本用例毫无关系的 _state_lock 错误。
      依赖"真机恰好没开游戏"的用例，测的其实不是它自己宣称的东西。
    """
    p = _plugin()
    _stub_window(p, monkeypatch, None)

    ok, msg = asyncio.run(p._probe_and_activate("timer"))
    assert ok is False
    assert "没找到游戏窗口" in msg
    # 节流窗口之后仍可再次记录（下面把时间推后）
    p._probe_last_log_at -= GenshinGameBuddyPlugin._PROBE_LOG_THROTTLE_S + 1
    ok2, _ = asyncio.run(p._probe_and_activate("timer"))
    assert ok2 is False


def test_window_found_registers_and_logs_session(monkeypatch, caplog):
    p = _plugin()
    _stub_window(p, monkeypatch, {"pid": 4242, "title": "原神", "rect": (0, 0, 1, 1)})

    registered: list[str] = []

    async def _fake_register(raw: object) -> tuple[str, Any]:
        registered.append(str(raw))
        class R:
            session = _FakeSession("genshin", "原神")
        return "ok", R()

    p._register = _fake_register
    p._load_scenes_for_current = lambda: setattr(p, "_scenes_loaded", 1)

    with caplog.at_level(logging.INFO, logger="test.session_probe"):
        ok, msg = asyncio.run(p._probe_and_activate("timer"))

    assert ok is True
    assert "会话已建立" in msg and "原神" in msg
    assert registered == ["genshin"]
    assert p._probe_activated == "genshin"
    # 场景表必须跟着建，否则 switch 后 scenes=0（拍板 2.0.72 的坑）
    assert p._scenes_loaded == 1
    assert any("会话已建立" in r.getMessage() for r in caplog.records)


def test_register_failure_is_reported_not_raised(monkeypatch):
    p = _plugin()
    _stub_window(p, monkeypatch, {"pid": 1, "title": "原神", "rect": (0, 0, 1, 1)})

    async def _bad(raw: object) -> tuple[str, Any]:
        return "terms_load_failed", None

    p._register = _bad
    ok, msg = asyncio.run(p._probe_and_activate("timer"))
    assert ok is False
    assert "terms_load_failed" in msg


def test_scene_load_failure_does_not_rollback_session(monkeypatch):
    """场景表挂了不该把已经建好的会话一起弄没。"""
    p = _plugin()
    _stub_window(p, monkeypatch, {"pid": 1, "title": "原神", "rect": (0, 0, 1, 1)})

    async def _fake_register(raw: object) -> tuple[str, Any]:
        class R:
            session = _FakeSession("genshin", "原神")
        return "ok", R()

    def _boom() -> None:
        raise RuntimeError("scenes.toml 坏了")

    p._register = _fake_register
    p._load_scenes_for_current = _boom

    ok, _ = asyncio.run(p._probe_and_activate("timer"))
    assert ok is True                      # 会话仍然建立成功
    assert p._probe_activated == "genshin"


def test_window_scan_exception_is_contained(monkeypatch):
    """扫窗口抛异常绝不能拖死 timer 线程。"""
    p = _plugin()

    def _boom() -> None:
        raise OSError("win32gui 炸了")

    monkeypatch.setattr(
        "plugin.plugins.genshin_game_buddy.find_genshin_window", _boom)
    ok, msg = asyncio.run(p._probe_and_activate("timer"))
    assert ok is False
    assert "扫窗口失败" in msg


def test_window_without_registerable_game(monkeypatch):
    registry = _registry(
        _entry("genshin", display="原神", aliases=("原神",), enabled=False),
    )
    p = _plugin(registry=registry)
    _stub_window(p, monkeypatch, {"pid": 1, "title": "原神", "rect": (0, 0, 1, 1)})
    ok, msg = asyncio.run(p._probe_and_activate("timer"))
    assert ok is False
    assert "没有可登记的游戏" in msg


# ============================================================ session_probe

def test_session_probe_swallows_crash(monkeypatch):
    """探针崩了不能让 @timer_interval 停摆。"""
    p = _plugin()

    async def _boom(trigger: str) -> tuple[bool, str]:
        raise RuntimeError("炸了")

    p._probe_and_activate = _boom
    # 不抛异常即通过；logger.exception 走的是 logger，不影响断言
    asyncio.run(p.session_probe())


def test_session_probe_is_registered_with_host_timer():
    """必须挂在宿主 timer 上 —— 生命周期跟 ocr_perceive 一致，shutdown 不用管。"""
    timer_ids = {
        getattr(getattr(v, "func", v), "__name__", "")
        for v in vars(GenshinGameBuddyPlugin).values()
    }
    assert "session_probe" in timer_ids


# ======================================================= activate_game_entry

def test_activate_game_entry_success(monkeypatch):
    p = _plugin()
    _stub_window(p, monkeypatch, {"pid": 1, "title": "原神", "rect": (0, 0, 1, 1)})

    async def _fake_register(raw: object) -> tuple[str, Any]:
        class R:
            session = _FakeSession("genshin", "原神")
        return "ok", R()

    p._register = _fake_register
    p._load_scenes_for_current = lambda: None

    out = asyncio.run(p.activate_game_entry())
    assert not isinstance(out, Err), f"unexpected Err: {out}"
    assert out.value["activated"] is True
    assert "会话已建立" in out.value["summary"]


def test_activate_game_entry_no_window(monkeypatch):
    p = _plugin()
    _stub_window(p, monkeypatch, None)
    out = asyncio.run(p.activate_game_entry())
    # entry 本身成功；业务上"没激活"是正常结果，不是错误
    assert not isinstance(out, Err), f"unexpected Err: {out}"
    assert out.value["activated"] is False
    assert "没找到游戏窗口" in out.value["summary"]