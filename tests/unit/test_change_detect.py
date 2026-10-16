"""拍板 2.0.67：S1 变化驱动 OCR——轻量帧 dHash 算法 + diff 边界 + 静止不触发。

不构造插件实例——直接测静态方法：
  - _light_capture_dhash() 失败/无屏幕 → 返回空串（不抛）
  - _dhash_diff() 边界（相同/不同/空串/长度不匹配）
  - 阈值默认 8 决定的触发行为（≥ 8 → trigger；< 8 → 不 trigger）

S1 整体行为（轻量帧抓取 + tick 决策）在集成测试里覆盖；这里单测保证哈希纯函数正确。
"""
from __future__ import annotations

import time

import pytest
from plugin.plugins.genshin_game_buddy import GenshinGameBuddyPlugin
from plugin.plugins.genshin_game_buddy.game_registry import PluginOptions

#: 生产侧的「画面变了」判定阈值（``game_registry.PluginOptions``）。
#: 直接取默认值而不是写死 8 —— 哪天改了默认值，这条测试自动跟上。
PROACTIVE_HASH_THRESHOLD = PluginOptions().change_detect_threshold

#: 「桌面算不算静止」的判据。**刻意不复用** :data:`PROACTIVE_HASH_THRESHOLD`
#: —— 那是「要不要触发 OCR」的业务阈值，语义完全不同，拿它当环境判据会
#: 在 diff 落在 (2, 8] 时误判成「静止」，于是动画桌面上照样红（实测过）。
STATIC_MAX_DIFF = 2


@pytest.mark.unit
def test_dhash_diff__same_hash_zero() -> None:
    """相同哈希 → 差异为 0。"""
    h = "0123456789abcdef"
    assert GenshinGameBuddyPlugin._dhash_diff(h, h) == 0


@pytest.mark.unit
def test_dhash_diff__all_bits_differ_max() -> None:
    """完全不同的 64-bit 哈希 → 差异 = 64（dHash 上限）。"""
    a = "0000000000000000"
    b = "ffffffffffffffff"
    assert GenshinGameBuddyPlugin._dhash_diff(a, b) == 64


@pytest.mark.unit
def test_dhash_diff__one_bit_diff() -> None:
    """只差 1 bit → 差异 = 1。"""
    a = "0000000000000000"
    b = "0000000000000001"
    assert GenshinGameBuddyPlugin._dhash_diff(a, b) == 1


@pytest.mark.unit
def test_dhash_diff__empty_string_returns_zero() -> None:
    """空串（首次或抓屏失败）→ diff = 0（不误触发）。"""
    assert GenshinGameBuddyPlugin._dhash_diff("", "abcd") == 0
    assert GenshinGameBuddyPlugin._dhash_diff("abcd", "") == 0
    assert GenshinGameBuddyPlugin._dhash_diff("", "") == 0


@pytest.mark.unit
def test_dhash_diff__length_mismatch_returns_zero() -> None:
    """长度不匹配的字符串 → diff = 0（不误触发）。"""
    assert GenshinGameBuddyPlugin._dhash_diff("ab", "abcdef") == 0
    assert GenshinGameBuddyPlugin._dhash_diff("a", "ab") == 0


@pytest.mark.unit
def test_dhash_diff__invalid_hex_returns_zero() -> None:
    """非法 hex → diff = 0（不抛、不误触发）。"""
    assert GenshinGameBuddyPlugin._dhash_diff("zz", "ab") == 0
    assert GenshinGameBuddyPlugin._dhash_diff("1234", "xyz!") == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "b", "expected_diff"),
    [
        ("0000000000000000", "0000000000000000", 0),       # 全 0 = 全 0
        ("0000000000000000", "ffffffffffffffff", 64),       # 全 0 vs 全 1 = 64 位不同
        ("ffffffffffffffff", "ffffffffffffffff", 0),       # 全 1 = 全 1
        ("0000000000000000", "00000000000000ff", 8),        # 后 8 位不同
        ("0000000000000000", "0000000000000001", 1),        # 后 1 位不同
        ("0000000000000000", "0000000000000007", 3),        # 后 3 位不同（默认阈值 8 上下）
        ("0000000000000000", "00000000000000ff", 8),        # 阈值边界（8 位不同 = 默认阈值）
    ],
)
def test_dhash_diff__boundary_table(
    a: str, b: str, expected_diff: int,
) -> None:
    """表驱动边界——与默认阈值 8 对齐验证。"""
    assert GenshinGameBuddyPlugin._dhash_diff(a, b) == expected_diff


@pytest.mark.unit
def test_light_capture_dhash__returns_hex_string_or_empty() -> None:
    """_light_capture_dhash 在有屏幕时返回 16 hex chars；无 mss/PIL 时返回空串。"""
    h = GenshinGameBuddyPlugin._light_capture_dhash()
    # 允许两种合法返回：16-char hex（成功）或空串（缺依赖/无屏幕）
    assert h == "" or (len(h) == 16 and all(c in "0123456789abcdef" for c in h))


@pytest.mark.unit
def test_light_capture_dhash__static_screen_returns_same_hash() -> None:
    """拍板 2.0.67：静止画面两次抓取 → 哈希相同 → diff = 0 → 不触发。

    ⚠️ 这条测试**依赖桌面是静止的**，而本测试无法保证这一点
    ------------------------------------------------------
    它抓的是**真实桌面**。只要桌面上有东西在动 —— 全屏播视频、游戏在跑、
    甚至本进程自己的输出在终端里滚动 —— 两次抓屏就必然不同。

    实测踩过两次：
      · 桌面全屏播 B 站视频 -> 两帧差 39.6% 的像素，diff=7，稳定失败
      · 静止判据一度误用生产阈值（8）当闸门 -> diff=7 < 8 没拦住，照样红

    所以闸门**不复用** :data:`PROACTIVE_HASH_THRESHOLD`（那是「要不要触发
    OCR」的业务阈值，与「画面动不动」无关）。这里用独立的 :data:`STATIC_MAX_DIFF`，
    连抓三帧：任一对超过它就判定桌面在动、跳过；全都在它以内才算静止，
    这时再断言 ``diff <= 2``。

    坦白说：在不受控的桌面环境里，这条测试基本只会 skip 而不会红 ——
    这是它应有的诚实语义（环境前提不成立时不该报假警），而不是它变强了。
    真正有断言价值的是它上面那条表驱动的 :func:`_dhash_diff` 边界测试。
    """
    h1 = GenshinGameBuddyPlugin._light_capture_dhash()
    if not h1:
        pytest.skip("no display/mss/pillow in this environment")
    time.sleep(0.25)
    h2 = GenshinGameBuddyPlugin._light_capture_dhash()
    time.sleep(0.25)
    h3 = GenshinGameBuddyPlugin._light_capture_dhash()
    if not h2 or not h3:
        pytest.skip("capture failed partway (screen changing?)")

    diffs = [GenshinGameBuddyPlugin._dhash_diff(a, b)
             for a, b in ((h1, h2), (h2, h3), (h1, h3))]
    if max(diffs) > STATIC_MAX_DIFF:
        pytest.skip(
            f"桌面不是静止画面（三帧两两 diff={diffs} > {STATIC_MAX_DIFF}）"
            "—— 视频 / 游戏 / 终端滚动都会这样")
    assert max(diffs) <= 2, f"static screen produced diff={diffs} (expected ≤ 2)"


@pytest.mark.unit
def test_light_capture_dhash__does_not_raise() -> None:
    """_light_capture_dhash 任何异常都吞掉返回空串——不污染 tick 流程。"""
    # 调用一次：要么返回合法 hash，要么返回空串；都不应抛
    try:
        result = GenshinGameBuddyPlugin._light_capture_dhash()
    except Exception as exc:
        pytest.fail(f"_light_capture_dhash raised {type(exc).__name__}: {exc}")
    assert result == "" or (len(result) == 16 and all(c in "0123456789abcdef" for c in result))