"""``actuator/keymap.py`` 的单测（2026-10-12 任务 C）。

这个模块的价值全在**加载期严格失败**：一个拼错的键位如果被悄悄忽略，
角色就永远按不出技能，而症状是「规则触发了但没反应」，极难排查。

所以本文件的重点不是「能正常加载」，而是**每种错都能指名道姓地报出来**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.executor import DEFAULT_KEYMAP  # noqa: E402
from autopilot.actuator.keymap import (  # noqa: E402
    KeymapError,
    dump_keymap,
    keymap_path,
    load_keymap,
    save_keymap,
)
from autopilot.actuator.simulator import KeyId  # noqa: E402


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "keymap.toml"
    p.write_text(body, encoding="utf-8")
    return p


# ==================================================================
# 回落
# ==================================================================

@pytest.mark.unit
def test_load_keymap__missing_file_falls_back_to_defaults(tmp_path) -> None:
    got = load_keymap(tmp_path / "nope.toml")
    assert got == dict(DEFAULT_KEYMAP)


@pytest.mark.unit
def test_load_keymap__fallback_is_a_copy_not_the_shared_dict(tmp_path) -> None:
    """改返回值不能污染 ``DEFAULT_KEYMAP`` —— 它是进程级常量。"""
    got = load_keymap(tmp_path / "nope.toml")
    got["MOVE_FORWARD"] = KeyId.NONE
    assert DEFAULT_KEYMAP["MOVE_FORWARD"] != KeyId.NONE


@pytest.mark.unit
def test_load_keymap__all_values_are_keyid_instances(tmp_path) -> None:
    """两条路径（回落 / 读文件）必须返回**同一种**类型。

    踩过的坑：回落路径曾经直接 ``dict(DEFAULT_KEYMAP)``，值是 ``VK`` 成员，
    而读文件那条路是 ``KeyId``。``VK`` 没有 ``to_vk()``，
    于是「没配 keymap.toml 的新用户」拿到的键位根本没法注入按键 ——
    偏偏这正是默认路径。
    """
    got = load_keymap(tmp_path / "nope.toml")
    assert all(isinstance(v, KeyId) for v in got.values()), (
        "混进 VK 成员的话，下游 KeyId 专属方法会时有时无")


@pytest.mark.unit
def test_keymap_path__prefers_the_env_override(monkeypatch, tmp_path) -> None:
    target = tmp_path / "custom.toml"
    monkeypatch.setenv("NEKO_KEYMAP", str(target))
    assert keymap_path() == target


@pytest.mark.unit
def test_keymap_path__without_env_falls_back_to_a_real_path() -> None:
    monkey = keymap_path()
    assert isinstance(monkey, Path)
    assert monkey.name == "keymap.toml"


# ==================================================================
# 正常加载
# ==================================================================

@pytest.mark.unit
def test_load_keymap__partial_config_overrides_only_named_actions(tmp_path) -> None:
    p = _write(tmp_path, '[keymap]\nUSE_SKILL = "Q"\n')
    got = load_keymap(p)
    assert got["USE_SKILL"] == KeyId.from_name("Q")
    assert got["MOVE_FORWARD"] == KeyId(int(DEFAULT_KEYMAP["MOVE_FORWARD"]))


@pytest.mark.unit
def test_load_keymap__accepts_none_to_disable_an_action(tmp_path) -> None:
    """NONE = 这个动作不发按键 —— 和 UNKNOWN 语义完全不同。"""
    p = _write(tmp_path, '[keymap]\nUSE_BURST = "NONE"\n')
    assert load_keymap(p)["USE_BURST"] == KeyId.NONE


@pytest.mark.unit
def test_load_keymap__accepts_hex_vk_codes(tmp_path) -> None:
    p = _write(tmp_path, '[keymap]\nUSE_SKILL = "0x51"\n')
    assert int(load_keymap(p)["USE_SKILL"]) == 0x51


# ==================================================================
# 加载期严格失败
# ==================================================================

@pytest.mark.unit
def test_load_keymap__missing_keymap_table_is_rejected(tmp_path) -> None:
    with pytest.raises(KeymapError) as ei:
        load_keymap(_write(tmp_path, "USE_SKILL = 'E'\n"))
    assert "[keymap]" in str(ei.value)


@pytest.mark.unit
def test_load_keymap__toml_syntax_error_is_rejected(tmp_path) -> None:
    with pytest.raises(KeymapError) as ei:
        load_keymap(_write(tmp_path, "[keymap\nUSE_SKILL = "))
    assert "TOML" in str(ei.value)


@pytest.mark.unit
def test_load_keymap__unknown_action_name_lists_legal_ones(tmp_path) -> None:
    """MOVE_FROWARD 这种拼错必须报，且要告诉用户合法的名字。"""
    p = _write(tmp_path, '[keymap]\nMOVE_FROWARD = "W"\n')
    with pytest.raises(KeymapError) as ei:
        load_keymap(p)
    msg = str(ei.value)
    assert "MOVE_FROWARD" in msg
    assert "MOVE_FORWARD" in msg, "错误信息里应给出最接近的合法名"


@pytest.mark.unit
def test_load_keymap__unknown_key_name_is_rejected(tmp_path) -> None:
    p = _write(tmp_path, '[keymap]\nUSE_SKILL = "NOT_A_KEY"\n')
    with pytest.raises(KeymapError) as ei:
        load_keymap(p)
    assert "USE_SKILL" in str(ei.value)


@pytest.mark.unit
def test_load_keymap__non_string_value_is_rejected(tmp_path) -> None:
    """值写成数字（很常见的手滑）也必须报，而不是悄悄 int() 掉。"""
    p = _write(tmp_path, "[keymap]\nUSE_SKILL = 69\n")
    with pytest.raises(KeymapError) as ei:
        load_keymap(p)
    assert "字符串" in str(ei.value) or "str" in str(ei.value)


@pytest.mark.unit
def test_load_keymap__unknown_sentinel_cannot_be_written(tmp_path) -> None:
    """UNKNOWN 是「未配置」哨兵，写进配置等于让角色永远按不出技能。

    用 ``0xFF`` 写（``KeyId.UNKNOWN`` 的值）—— ``from_name`` 只认 VK 名 /
    十六进制 / ``NONE``，直接写 ``"UNKNOWN"`` 会在解析阶段就报，
    测不到哨兵这一层的检查。
    """
    p = _write(tmp_path, f'[keymap]\nUSE_SKILL = "0x{int(KeyId.UNKNOWN):02X}"\n')
    with pytest.raises(KeymapError) as ei:
        load_keymap(p)
    assert "UNKNOWN" in str(ei.value)


# ==================================================================
# 序列化 / 回写
# ==================================================================

@pytest.mark.unit
def test_dump_keymap__round_trips_through_load(tmp_path) -> None:
    """dump 出来的文本必须能被 load 读回去 —— 否则编辑器会毁配置。"""
    src = _write(tmp_path, '[keymap]\nUSE_SKILL = "Q"\nUSE_BURST = "NONE"\n')
    got = load_keymap(src)
    dumped = dump_keymap(got)
    p2 = tmp_path / "again.toml"
    p2.write_text(dumped, encoding="utf-8")
    assert load_keymap(p2) == got


@pytest.mark.unit
def test_dump_keymap__is_deterministic(tmp_path) -> None:
    """同样的输入必须生成逐字节相同的文件，否则改动会淹没在噪声里。"""
    got = load_keymap(tmp_path / "nope.toml")
    assert dump_keymap(got) == dump_keymap(got)


@pytest.mark.unit
def test_dump_keymap__writes_none_as_a_literal_string(tmp_path) -> None:
    got = load_keymap(tmp_path / "nope.toml")
    got["USE_BURST"] = KeyId.NONE
    assert 'USE_BURST = "NONE"' in dump_keymap(got)


@pytest.mark.unit
def test_save_keymap__writes_a_reloadable_file(tmp_path) -> None:
    src = _write(tmp_path, '[keymap]\nUSE_SKILL = "Q"\n')
    got = load_keymap(src)
    out = save_keymap(got, tmp_path / "saved.toml")
    assert out.exists()
    assert load_keymap(out) == got


@pytest.mark.unit
def test_save_keymap__creates_parent_dirs(tmp_path) -> None:
    got = load_keymap(tmp_path / "nope.toml")
    deep = tmp_path / "a" / "b" / "keymap.toml"
    save_keymap(got, deep)
    assert deep.exists()


@pytest.mark.unit
def test_save_keymap__rejects_unknown_action_before_touching_disk(tmp_path) -> None:
    """先校验再落盘 —— 不让非法配置污染磁盘。"""
    got = load_keymap(tmp_path / "nope.toml")
    got["NOT_AN_ACTION"] = KeyId.from_name("W")
    target = tmp_path / "out.toml"
    with pytest.raises(KeymapError):
        save_keymap(got, target)
    assert not target.exists(), "校验失败就不该留下文件"


@pytest.mark.unit
def test_save_keymap__rejects_the_unknown_sentinel(tmp_path) -> None:
    got = load_keymap(tmp_path / "nope.toml")
    got["USE_SKILL"] = KeyId.UNKNOWN
    with pytest.raises(KeymapError):
        save_keymap(got, tmp_path / "out.toml")


@pytest.mark.unit
def test_save_keymap__rejects_a_non_positive_vk(tmp_path) -> None:
    """VK 码 <= 0 会被 SendInput 当成无效输入。

    注意别用 ``KeyId(0)`` —— 那正好等于 ``KeyId.NONE``，会走「禁用」分支。
    """
    got = load_keymap(tmp_path / "nope.toml")
    got["USE_SKILL"] = KeyId(-1)
    with pytest.raises(KeymapError):
        save_keymap(got, tmp_path / "out.toml")


# ==================================================================
# 三级解耦契约
# ==================================================================

@pytest.mark.unit
def test_keymap__covers_every_action_the_traits_declare() -> None:
    """动作表与 traits 必须同步 —— 否则新动作没有键位可用。

    ``TRAITS`` 是 ``{动作名: ActionTraits}``，所以这里取的是键。
    """
    from autopilot.actuator.traits import TRAITS
    missing = sorted(set(TRAITS) - set(DEFAULT_KEYMAP))
    assert not missing, f"traits 里有动作没有键位：{missing}"


@pytest.mark.unit
def test_keymap__values_all_resolve_to_a_vk() -> None:
    """每个非哨兵键位都要能 ``to_vk()`` —— 归一化没做干净的话这里会炸。"""
    got = load_keymap(Path(__file__).parent / "does-not-exist.toml")
    for action, kid in got.items():
        if int(kid) == KeyId.NONE:
            continue
        assert kid.to_vk() is not None, f"{action} 解析不出 VK"


@pytest.mark.unit
def test_keymap__sentinels_refuse_to_become_real_keystrokes() -> None:
    """哨兵不能被拿去发按键 —— NONE/UNKNOWN 都不是真实键。"""
    from autopilot.actuator.simulator import KeyIdError
    for sentinel in (KeyId(int(KeyId.NONE)), KeyId(int(KeyId.UNKNOWN))):
        with pytest.raises(KeyIdError):
            sentinel.to_vk()
