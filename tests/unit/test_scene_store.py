"""scene_store 单元测试：scenes.toml 解析 / 空文件 / 覆盖层合并。"""
from __future__ import annotations

from pathlib import Path

from plugin.plugins.genshin_game_buddy.scene_store import load_scenes


def test_load_scenes_normal(scratch: Path) -> None:
    (scratch / "terms" / "demo").mkdir(parents=True)
    (scratch / "terms" / "demo" / "scenes.toml").write_text(
        '[scenes."深境螺旋"]\nsignals = ["深境螺旋", "第12层"]\ncontext_terms = ["元素反应"]\n',
        encoding="utf-8",
    )
    scenes = load_scenes(scratch, None, "demo", "terms/demo")
    assert "深境螺旋" in scenes
    assert scenes["深境螺旋"].signals == ("深境螺旋", "第12层")
    assert scenes["深境螺旋"].context_terms == ("元素反应",)


def test_load_scenes_missing_file(scratch: Path) -> None:
    scenes = load_scenes(scratch, None, "demo", "terms/demo")
    assert scenes == {}


def test_load_scenes_empty_file(scratch: Path) -> None:
    (scratch / "terms" / "demo").mkdir(parents=True)
    (scratch / "terms" / "demo" / "scenes.toml").write_text("", encoding="utf-8")
    scenes = load_scenes(scratch, None, "demo", "terms/demo")
    assert scenes == {}


def test_load_scenes_missing_fields(scratch: Path) -> None:
    (scratch / "terms" / "demo").mkdir(parents=True)
    (scratch / "terms" / "demo" / "scenes.toml").write_text(
        '[scenes."场景1"]\nsignals = []\ncontext_terms = []\n',
        encoding="utf-8",
    )
    scenes = load_scenes(scratch, None, "demo", "terms/demo")
    assert scenes["场景1"].signals == ()
    assert scenes["场景1"].context_terms == ()


def test_load_scenes_override_merge(scratch: Path) -> None:
    (scratch / "default" / "terms" / "demo").mkdir(parents=True)
    (scratch / "override" / "terms" / "demo").mkdir(parents=True)
    (scratch / "default" / "terms" / "demo" / "scenes.toml").write_text(
        '[scenes."A"]\nsignals = ["a"]\ncontext_terms = ["x"]\n',
        encoding="utf-8",
    )
    (scratch / "override" / "terms" / "demo" / "scenes.toml").write_text(
        '[scenes."A"]\nsignals = ["b"]\ncontext_terms = ["y"]\n',
        encoding="utf-8",
    )
    scenes = load_scenes(scratch / "default", scratch / "override", "demo", "terms/demo")
    assert scenes["A"].signals == ("b",)
    assert scenes["A"].context_terms == ("y",)


# =============================================================================
# 2026-11-09 接线：avoid / known_suppress 的解析与向后兼容
# =============================================================================


def test_load_scenes_legacy_entry_without_avoid_fields(scratch: Path) -> None:
    """旧式条目（没有 avoid / known_suppress）照旧能加载。

    向后兼容是硬要求：用户覆盖层的旧 scenes.toml 里不可能有这两个 key。
    """
    (scratch / "terms" / "demo").mkdir(parents=True)
    (scratch / "terms" / "demo" / "scenes.toml").write_text(
        '[scenes."场景1"]\nsignals = ["甲"]\ncontext_terms = ["乙"]\n',
        encoding="utf-8",
    )
    scenes = load_scenes(scratch, None, "demo", "terms/demo")
    entry = scenes["场景1"]
    assert entry.signals == ("甲",)
    assert entry.context_terms == ("乙",)
    assert entry.avoid == ()
    assert entry.known_suppress is False


def test_load_scenes_parses_avoid_and_known_suppress(scratch: Path) -> None:
    """avoid 走 _clean_str_list；known_suppress 只认真布尔 true。"""
    (scratch / "terms" / "demo").mkdir(parents=True)
    (scratch / "terms" / "demo" / "scenes.toml").write_text(
        '[scenes."场景1"]\n'
        'signals = ["甲"]\n'
        'avoid = ["不要剧透", "  不要照抄数值  "]\n'
        'known_suppress = true\n',
        encoding="utf-8",
    )
    scenes = load_scenes(scratch, None, "demo", "terms/demo")
    entry = scenes["场景1"]
    assert entry.avoid == ("不要剧透", "不要照抄数值")
    assert entry.known_suppress is True


def test_load_scenes_known_suppress_false_when_missing(scratch: Path) -> None:
    """缺 known_suppress key → False。"""
    (scratch / "terms" / "demo").mkdir(parents=True)
    (scratch / "terms" / "demo" / "scenes.toml").write_text(
        '[scenes."场景1"]\nsignals = ["甲"]\n',
        encoding="utf-8",
    )
    scenes = load_scenes(scratch, None, "demo", "terms/demo")
    assert scenes["场景1"].known_suppress is False


def test_load_scenes_known_suppress_non_bool_is_false(scratch: Path) -> None:
    """非布尔值一律 False——不能用 bool(...)（那样字符串 "false" 也会变 True）。"""
    (scratch / "terms" / "demo").mkdir(parents=True)
    (scratch / "terms" / "demo" / "scenes.toml").write_text(
        '[scenes."字符串true"]\nsignals = ["甲"]\nknown_suppress = "true"\n'
        '\n'
        '[scenes."字符串false"]\nsignals = ["甲"]\nknown_suppress = "false"\n'
        '\n'
        '[scenes."数字1"]\nsignals = ["甲"]\nknown_suppress = 1\n'
        '\n'
        '[scenes."布尔false"]\nsignals = ["甲"]\nknown_suppress = false\n',
        encoding="utf-8",
    )
    scenes = load_scenes(scratch, None, "demo", "terms/demo")
    assert scenes["字符串true"].known_suppress is False
    assert scenes["字符串false"].known_suppress is False
    assert scenes["数字1"].known_suppress is False
    assert scenes["布尔false"].known_suppress is False


def test_load_scenes_avoid_wrong_type_is_empty(scratch: Path) -> None:
    """avoid 不是列表 → 空 tuple（复用 _clean_str_list 的既有防御）。"""
    (scratch / "terms" / "demo").mkdir(parents=True)
    (scratch / "terms" / "demo" / "scenes.toml").write_text(
        '[scenes."场景1"]\nsignals = ["甲"]\navoid = "不是列表"\n',
        encoding="utf-8",
    )
    scenes = load_scenes(scratch, None, "demo", "terms/demo")
    assert scenes["场景1"].avoid == ()


def test_load_scenes_avoid_drops_blank_items(scratch: Path) -> None:
    """avoid 里的空白项被剔除（沿用 _clean_str_list 语义）。"""
    (scratch / "terms" / "demo").mkdir(parents=True)
    (scratch / "terms" / "demo" / "scenes.toml").write_text(
        '[scenes."场景1"]\nsignals = ["甲"]\navoid = ["不要剧透", "   ", 42]\n',
        encoding="utf-8",
    )
    scenes = load_scenes(scratch, None, "demo", "terms/demo")
    assert scenes["场景1"].avoid == ("不要剧透",)
