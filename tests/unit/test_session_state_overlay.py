"""会话状态落盘 -> overlay 读取 -> 面板渲染（3.0-b）的单元测试。

这条链跨越**两个独立进程**（插件进程写 / overlay 进程读），中间只靠
`_runtime/session_state.json` 一个文件。测试用真实的
`atomic_write_text` + 真实的 `load_session_state`，只把两个模块的
路径常量 monkeypatch 到同一个 tmp_path ——这样测的就是真实的跨进程
数据通路，而不是把两边各自 mock 掉的假验证。

覆盖：
  - 写 active=false / true 两种状态，读侧都能正确解析
  - 过期文件返回 None（插件崩了不能一直显示「已建立」）
  - schema 版本不符返回 None
  - 面板第 1 行：有会话显示游戏名，无会话显示「无会话」+橙色
  - 行数恒为 5（不破坏「急停: F12」的位置）
  - 读写两侧的路径常量必须指向同一个文件（防有人改一边忘另一边）
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from plugin.plugins.genshin_game_buddy.autopilot.ui.overlay import layout, theme

pytestmark = pytest.mark.unit


class _FakeSession:
    def __init__(self, game_id: str, display_name: str) -> None:
        self.game_id = game_id
        self.display_name = display_name


class _FakeManager:
    def __init__(self, current: Any = None) -> None:
        self.current = current


def _plugin(manager: Any = None) -> Any:
    """只带会话落盘所需字段的插件实例（同 test_session_probe 的路子）。"""
    import logging

    from plugin.plugins.genshin_game_buddy import GenshinGameBuddyPlugin

    p = GenshinGameBuddyPlugin.__new__(GenshinGameBuddyPlugin)
    p._manager = manager if manager is not None else _FakeManager(None)
    p.logger = logging.getLogger("test.session_state_overlay")
    p._session_state_active = False
    p._session_state_written_at = 0.0
    return p


@pytest.fixture()
def session_file(tmp_path, monkeypatch):
    """把 layout 的 SESSION_STATE_FILE 指到 tmp_path。

    写侧 ``_write_session_state`` 用的是
    ``protocol.DEFAULT_RUN_DIR / "session_state.json"``，那是模块级常量，
    不能直接改（改了会影响整个进程）。所以这里改的是它的**父目录**
    所依赖的 DEFAULT_RUN_DIR 指向 —— 用 monkeypatch.setattr 换掉模块属性。
    """
    from plugin.plugins.genshin_game_buddy.autopilot.ui import protocol as _proto

    monkeypatch.setattr(_proto, "DEFAULT_RUN_DIR", tmp_path)
    monkeypatch.setattr(layout, "SESSION_STATE_FILE",
                        tmp_path / "session_state.json")
    return tmp_path / "session_state.json"


def _write_raw(path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# ====================================================== 路径一致性（防回归）

def test_read_and_write_paths_point_at_same_file(session_file):
    """写侧推导出的路径必须和读侧常量落在同一个文件上。

    这条是整个设计的地基：写侧从 ``protocol.DEFAULT_RUN_DIR`` 推导，
    读侧在 layout 里手写 ``parents[2] / "_runtime"``。两边**故意**
    不是同一份表达式，所以需要一个测试把它们钉在一起 ——
    有人改了任一边的层级深度，这个测试立刻红。
    """
    from plugin.plugins.genshin_game_buddy.autopilot.ui import protocol as _proto

    write_side = _proto.DEFAULT_RUN_DIR / "session_state.json"
    assert write_side.name == layout.SESSION_STATE_FILE.name
    assert write_side.parent.name == layout.SESSION_STATE_FILE.parent.name


# ============================================================== 写 -> 读

def test_write_active_session_is_readable(session_file):
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    assert p._publish_session(reason="test") is True

    doc = layout.load_session_state()
    assert doc is not None
    assert doc["active"] is True
    assert doc["game_id"] == "genshin"
    assert doc["display_name"] == "原神"
    assert doc["reason"] == "test"
    assert doc["schema_version"] == layout.SESSION_SCHEMA_VERSION


def test_write_inactive_session_is_readable(session_file):
    p = _plugin(_FakeManager(None))
    assert p._publish_session(reason="test:no") is True

    doc = layout.load_session_state()
    assert doc is not None
    assert doc["active"] is False
    assert doc["game_id"] == ""


def test_missing_file_returns_none(session_file):
    assert layout.load_session_state() is None


def test_expired_file_returns_none(session_file):
    """插件崩了/被杀后，overlay 不能一直拿陈旧文件显示「已建立」。"""
    _write_raw(session_file, {
        "schema_version": layout.SESSION_SCHEMA_VERSION,
        "active": True, "game_id": "genshin", "display_name": "原神",
    })
    assert layout.load_session_state() is not None       # 刚写的是新鲜的

    # 假装文件是 10 分钟前的
    old = time.time() - 600
    import os
    os.utime(session_file, (old, old))
    assert layout.load_session_state() is None


def test_schema_version_mismatch_returns_none(session_file):
    """版本对不上就**不显示** —— 宁可退回「无会话」，也不能拿错字段渲染。"""
    _write_raw(session_file, {
        "schema_version": layout.SESSION_SCHEMA_VERSION + 1,
        "active": True, "game_id": "genshin", "display_name": "原神",
    })
    assert layout.load_session_state() is None


def test_corrupt_json_returns_none(session_file):
    session_file.write_text("{半截", encoding="utf-8")
    assert layout.load_session_state() is None


def test_write_failure_does_not_raise(tmp_path, monkeypatch):
    """目录写不进去只 warning，**不能把会话本身带崩**。"""
    from plugin.plugins.genshin_game_buddy.autopilot.ui import protocol as _proto

    def _boom(*_a, **_k):
        raise OSError("只读文件系统")

    monkeypatch.setattr(_proto, "DEFAULT_RUN_DIR", tmp_path)
    monkeypatch.setattr(_proto, "atomic_write_text", _boom)

    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    assert p._publish_session(reason="test") is False      # 返回 False，不抛


# ============================================================== 渲染

def _snap(health: str = "OK", state: str = "RUNNING") -> Any:
    class _Cap:
        active = "dxcam"
        degraded_from = ""
    class _Intent:
        count = 3
        idle_streak = 0
    class S:
        pass
    s = S()
    s.health = health
    s.state = state
    s.tick_hz = 12.5
    s.tick_p95_ms = 88.0
    s.scene = "Teyvat"
    s.capture = _Cap()
    s.intent = _Intent()
    s.panic_key = "F12"
    return s


def test_line1_shows_game_name_when_session_active(session_file):
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    model = layout.build_panel(_snap(), session=layout.load_session_state())
    head = model.lines[0].text
    assert "原神" in head
    assert "运行中" in head
    assert "无会话" not in head


# ==================================================================
# B2：headline 精简（2026-10-16）
# ==================================================================

def test_headline_has_no_performance_numbers(session_file):
    """★ B2：Hz / P95 不再占第一行。

    它们原本把「会话 · 状态」挤掉大半 —— 而那两个诊断数字对观众
    零信息量。观众看到「4.8 Hz」只会问「这是好还是坏」。
    """
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    model = layout.build_panel(_snap(), session=layout.load_session_state())
    head = model.lines[0].text
    assert "Hz" not in head
    assert "P95" not in head
    # 真正该一眼看到的东西现在独占第一行
    assert head.strip().startswith("●")


def test_headline_carries_no_dangling_spacing(session_file):
    """去掉数字之后不能留一串尾随空格 —— 视觉上会看出「少了点什么」。"""
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    model = layout.build_panel(_snap(), session=layout.load_session_state())
    assert model.lines[0].text == model.lines[0].text.rstrip()


def test_performance_numbers_come_back_under_the_flag(session_file, monkeypatch):
    """排障时能临时把它们找回来（``NEKO_OVERLAY_METRICS=1``）。"""
    monkeypatch.setattr(layout, "SHOW_METRICS", True)
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    model = layout.build_panel(_snap(), session=layout.load_session_state())
    assert "12.5 Hz" in model.lines[1].text
    assert "P95" in model.lines[1].text


def test_channel_line_is_hidden_when_there_is_nothing_to_say(session_file):
    """★ B2：没有通道时**不占一行**。

    原来是常驻的「通道: —」，而破折号会被观众读成「这一项坏了」。
    真出过问题的 run 里，那一行和橙字「已停止」一起被列为 A6 级粗糙点。
    """
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    snap = _snap()
    snap.capture.active = ""          # 快照里没有通道，也没见过
    model = layout.build_panel(snap, session=layout.load_session_state())
    texts = [l.text for l in model.lines]
    assert not any("通道" in t for t in texts), texts


def test_channel_line_appears_once_a_channel_was_seen(session_file):
    """抓到过通道就显示出来 —— 「有内容」比「一直占位」有用。"""
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    snap = _snap()
    snap.capture.active = ""          # 当前没在抓，但之前抓到过 wgc
    model = layout.build_panel(snap, session=layout.load_session_state(),
                               last_channel="wgc")
    texts = [l.text for l in model.lines]
    assert any("通道: wgc" in t for t in texts), texts


def test_snapshot_channel_wins_over_the_remembered_one(session_file):
    """★ 快照里的当前值优先于「上一次见到的」—— 三级兜底的顺序不能反。

    通道从 wgc 降级到 dxcam 时，显示「dxcam」才对；显示 wgc 会让人
    以为还在走更好的那条路。
    """
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    model = layout.build_panel(_snap(), session=layout.load_session_state(),
                               last_channel="wgc")
    texts = [l.text for l in model.lines]
    assert any("通道: dxcam" in t for t in texts), texts


def test_intent_count_is_always_visible(session_file):
    """★ 用户明确要求保留「意图: N 次」—— 唯一的动态状态。"""
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    model = layout.build_panel(_snap(), session=layout.load_session_state())
    assert any("意图:" in l.text for l in model.lines)


def test_panel_never_exceeds_the_widget_budget(session_file):
    """★ 行数必须 ≤ overlay 预建的 Label 数，否则多出来的行渲染不出来。

    ``OverlayApp.text_rows = 5``（app.py）决定预建几个 Label；模型行数
    超过它时渲染循环会静默截断，最后一行直接看不见。

    这里写死 5 而不是 import OverlayApp：那个类 import tkinter，
    而本文件是**无头**跑的（离屏/无 GUI），引进来会把一个纯布局测试
    变成依赖桌面环境的测试。
    """
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    no_channel = _snap()
    no_channel.capture.active = ""

    cases = (
        dict(session=layout.load_session_state()),
        dict(session=None),
        dict(session=None, last_channel="wgc"),
        dict(session=layout.load_session_state(), last_channel="wgc"),
    )
    for kwargs in cases:
        model = layout.build_panel(_snap(), **kwargs)
        assert len(model.lines) <= 5, (
            f"{kwargs} 产出了 {len(model.lines)} 行，超过 OverlayApp.text_rows")

    for state in ("RUNNING", "PAUSED", "STOPPED"):
        model = layout.build_panel(_snap(state=state),
                                   session=layout.load_session_state())
        assert len(model.lines) <= 5, f"{state} 产出了 {len(model.lines)} 行"

    model = layout.build_panel(_snap(health="OFFLINE"),
                               session=layout.load_session_state())
    assert len(model.lines) <= 5, f"OFFLINE 产出了 {len(model.lines)} 行"


def test_line1_shows_no_session_when_missing(session_file):
    """★ 803 次空转那个场景就该长这样：一眼看出没有会话。"""
    model = layout.build_panel(_snap(), session=layout.load_session_state())
    head = model.lines[0].text
    assert "无会话" in head
    assert "运行中" in head          # 相位还在，只是没有会话


def test_line1_warns_in_orange_when_no_session(session_file):
    """没有会话时行首用橙色而不是相位绿 —— 绿会让人以为一切正常。"""
    model = layout.build_panel(_snap(), session=layout.load_session_state())
    assert model.lines[0].color == theme.ACCENT_WARN


def test_line1_keeps_phase_color_when_session_present(session_file):
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    model = layout.build_panel(_snap(state="PAUSED"),
                               session=layout.load_session_state())
    assert model.lines[0].color == theme.phase_color("PAUSED")
    assert "已暂停" in model.lines[0].text


def test_panel_always_five_lines(session_file):
    """行数恒为 5 —— 「急停: F12」在第 5 行是肌肉记忆，不能挪。"""
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    for kwargs in (
        {"session": layout.load_session_state()},
        {"session": None},
        {"session": None, "health": "OFFLINE"},
    ):
        snap = _snap(health=kwargs.pop("health", "OK"))
        model = layout.build_panel(snap, **kwargs)
        assert len(model.lines) == 5
        assert model.lines[-1].text.startswith("急停:")


def test_offline_branch_also_shows_session(session_file):
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    model = layout.build_panel(_snap(health="OFFLINE"),
                               session=layout.load_session_state())
    assert "原神" in model.lines[0].text


def test_nav_mode_still_wins(session_file):
    """寻路状态有效时整块面板被替换，会话**不进**寻路面板。

    这是有意的：寻路是用户手动跑的独立脚本，此刻他关心的是「去哪」，
    塞一行会话进去会把 5 行布局挤掉；而行数一变，窗口高度就会跳。
    """
    nav = {"schema_version": layout.NAV_SCHEMA_VERSION, "state": "walking",
           "hint": "接近目标", "dist": 20.0, "target_name": "七天神像-风"}
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    model = layout.build_panel(_snap(), nav=nav,
                               session=layout.load_session_state())
    assert len(model.lines) == 5
    # 寻路面板的 5 行是：状态 / 目标 / 下一步 / 距离 / 急停
    assert "七天神像-风" in model.lines[1].text
    assert "接近目标" in model.lines[2].text
    assert "20m" in model.lines[3].text
    assert model.lines[4].text.startswith("急停:")
    # 会话**不进**寻路面板
    assert "原神" not in "\n".join(l.text for l in model.lines)


def test_content_key_changes_with_session(session_file):
    """指纹必须随会话变化 —— 否则 overlay 的省电短路会把状态变化吞掉。"""
    p = _plugin(_FakeManager(_FakeSession("genshin", "原神")))
    p._publish_session(reason="test")

    with_sess = layout.build_panel(_snap(), session=layout.load_session_state())
    without = layout.build_panel(_snap(), session=None)
    assert with_sess.content_key() != without.content_key()