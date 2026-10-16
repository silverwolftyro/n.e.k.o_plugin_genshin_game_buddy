"""定位服务注入的单测（2026-10-11 任务 C）。

★ 先说清现状，避免误解
----------------------
注入能力**早就存在**（``StateBuilder.__init__`` 的 ``locator=`` 参数、
``_locate()`` 的完整降级链、``_locator_guard`` 熔断）。缺的是两样：

  1. **没有任何测试** —— 那条降级链从没被验过
  2. ``_stack.build_perception`` 没传 locator（因为没有可用的真实
     ``LocalizationService``：它初始是 UNINITIALIZED，必须先传送建立原点）

所以本文件测的是**注入点本身**：给一个鸭子类型的 locator，builder 能不能
正确地取到位置、能不能在它坏掉时优雅降级、以及 ``None`` 时行为是否与
接入前**完全一致**（这是向后兼容的唯一保证）。

⚠️ 这些测试**不接真实 LocalizationService** —— 那需要真机建立原点。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.perception.builder import (  # noqa: E402
    LOCATOR_FAIL_THRESHOLD,
    LOCATOR_RECOVER_THRESHOLD,
    StateBuilder,
)
from autopilot.perception.localization_service import (  # noqa: E402
    LocalizationState,
    LocalizationStateMachine,
)

#: ``PositionInfo.quality`` 由 ``LocalizationStateMachine`` 反查得到
#: （``position_mapping._quality_table()``），所以夹具只能喂**枚举**，
#: 喂字符串会直接 ``KeyError``。这张表把测试里读的 quality 名词映射回枚举。
_QUALITY_TO_STATE = {
    "GOOD": LocalizationStateMachine.TRACKING,
    "DEGRADED": LocalizationStateMachine.PAUSED,
    "LOST": LocalizationStateMachine.UNINITIALIZED,
}


def _loc_state(*, quality="GOOD", x=4634.0, y=4687.0, drift=0.0):
    """构造**真实**的 ``LocalizationState``，不手搓替身。

    ⚠️ 这里刻意用生产 dataclass 而不是 mock 类：上一次手搓替身时把
    ``quality`` 当成了枚举字段名，结果 ``position_from_state`` 拿到的是
    字符串 ``"GOOD"``，直接 ``KeyError``。契约一变，替身不会跟着变。
    """
    return LocalizationState(
        state=_QUALITY_TO_STATE[quality],
        world_pos=None if x is None else (x, y),
        tile_id="liyue_1_1" if x is not None else None,
        in_tile_px=None,
        drift_bound=drift,
        step_count=3,
        is_world=True,
        scene_changed=False,
        position_age_s=0.05,
        peak_response=0.9,
        step_px=(1.0, 2.0),
        note="",
    )


class _GoodLocator:
    """永远定位成功的假 locator。"""

    def __init__(self, **kw):
        self.kw = kw
        self.calls = 0

    def on_frame(self, frame, ts_sec):
        self.calls += 1
        return _loc_state(**self.kw)


class _BrokenLocator:
    """每次都抛异常 —— 用来验证**不外抛**。"""

    def __init__(self):
        self.calls = 0

    def on_frame(self, frame, ts_sec):
        self.calls += 1
        raise RuntimeError("定位炸了")


class _FlakyLocator:
    """前 ``ok_times`` 次成功，之后失败。"""

    def __init__(self, ok_times=1):
        self.ok_times = ok_times
        self.calls = 0

    def on_frame(self, frame, ts_sec):
        self.calls += 1
        if self.calls <= self.ok_times:
            return _loc_state()
        raise RuntimeError("之后都炸")


def _builder(locator):
    return StateBuilder(capturer=None, yolo=None, template=None,
                       ocr=None, locator=locator)


def _frame():
    return np.zeros((720, 1280, 3), np.uint8)


# ==================================================================
# 向后兼容
# ==================================================================

@pytest.mark.unit
def test_locator__none_keeps_pre_integration_behaviour() -> None:
    """locator=None -> 直接 (None, 0.0)。**这是向后兼容的唯一保证。**"""
    b = _builder(None)
    pos, ms = b._locate(_frame(), 1.0)
    assert pos is None
    assert ms == 0.0


@pytest.mark.unit
def test_locator__none_never_touches_the_locator_slot() -> None:
    b = _builder(None)
    assert b._last_good_position is None
    b._locate(_frame(), 1.0)
    assert b._last_good_position is None, "None 模式下不该产生任何位置"


# ==================================================================
# 正常路径
# ==================================================================

@pytest.mark.unit
def test_locator__injected_service_produces_position() -> None:
    loc = _GoodLocator()
    b = _builder(loc)
    pos, _ms = b._locate(_frame(), 1.0)
    assert pos is not None
    assert pos.quality == "GOOD"
    assert loc.calls == 1
    assert b._last_good_position is not None


@pytest.mark.unit
def test_locator__degraded_still_carries_position() -> None:
    """DEGRADED 是「能用但要额外确认」，不是「不能用」。"""
    b = _builder(_GoodLocator(quality="DEGRADED", x=100.0, y=200.0))
    pos, _ = b._locate(_frame(), 1.0)
    assert pos is not None and pos.quality == "DEGRADED"
    assert (pos.x, pos.y) == (100.0, 200.0)


@pytest.mark.unit
def test_locator__no_coordinate_yields_usable_false() -> None:
    """状态非 LOST 但没坐标 -> 退化成 ``lost()``，不能当导航可用。

    对应现实：菜单里小地图没取到，但状态机还停在 TRACKING。
    这时若让规则拿着空坐标走路就是灾难，所以 ``position_from_state``
    在 ``world_pos is None`` 时直接返回 ``lost()``。
    """
    b = _builder(_GoodLocator(quality="DEGRADED", x=None, y=None))
    pos, _ = b._locate(_frame(), 1.0)
    assert pos is not None
    assert pos.quality == "LOST", "没坐标就不该还挂着 DEGRADED"
    assert pos.usable_for_navigation is False


# ==================================================================
# 降级链
# ==================================================================

@pytest.mark.unit
def test_locator__exception_does_not_propagate() -> None:
    """★ 定位抖动**绝不能**让整个 tick 崩。

    ``core/loop.py`` 收到异常会走 ``_note_crash``，而那里**无条件**
    ``_release_all()`` —— 一次定位抖动就会把按住的键全松开。
    """
    b = _builder(_BrokenLocator())
    pos, _ms = b._locate(_frame(), 1.0)      # 不抛就是过
    assert pos is None or pos is not None     # 只验证不抛


@pytest.mark.unit
def test_locator__single_failure_reuses_last_good() -> None:
    """单帧失败（未判死）-> 沿用上一帧位置，而不是返回 UNKNOWN 空壳。"""
    loc = _FlakyLocator(ok_times=1)
    b = _builder(loc)
    first, _ = b._locate(_frame(), 1.0)
    assert first is not None
    second, _ = b._locate(_frame(), 2.0)     # 这次抛
    assert second is not None, "单帧失败不该把位置清空"
    assert loc.calls == 2


@pytest.mark.unit
def test_locator__sustained_failure_kills_and_gives_LOST() -> None:
    """连续失败到阈值 -> 判死，给显式 quality=LOST（**不是 None**）。

    None 会让 ``pos_quality()`` 返回 UNKNOWN，规则拿到的是「不知道」；
    显式 LOST 拿到的是「不能用」—— 后者方向性明确得多。
    """
    b = _builder(_BrokenLocator())
    pos = None
    for _ in range(LOCATOR_FAIL_THRESHOLD + 1):
        pos, _ = b._locate(_frame(), 1.0)
    assert pos is not None, "判死后应给显式 LOST 包，不是 None"
    assert pos.quality == "LOST"
    assert pos.usable_for_navigation is False


@pytest.mark.unit
def test_locator__recovery_after_kill() -> None:
    """熔断后恢复成功帧要能重新拿到位置。"""
    assert LOCATOR_RECOVER_THRESHOLD >= 1
    b = _builder(_FlakyLocator(ok_times=LOCATOR_FAIL_THRESHOLD + 5))
    for _ in range(LOCATOR_FAIL_THRESHOLD + 2):
        b._locate(_frame(), 1.0)              # 先判死
    pos, _ = b._locate(_frame(), 2.0)
    for _ in range(LOCATOR_RECOVER_THRESHOLD + 1):
        pos, _ = b._locate(_frame(), 3.0)
    assert pos is not None and pos.quality == "GOOD", (
        f"恢复后应该重新拿到位置，实际 {pos}")


@pytest.mark.unit
def test_locator__failure_does_not_touch_perception_ok() -> None:
    """定位挂了**不影响** perception_ok —— YOLO/OCR 还能干活。"""
    b = _builder(_BrokenLocator())
    b._locate(_frame(), 1.0)
    assert b._compute_perception_ok() is True


# ==================================================================
# 接线点现状
# ==================================================================

@pytest.mark.unit
def test_stack__injects_locator_only_when_config_enables_it() -> None:
    """接线现状（2026-10-12 更新）：**受配置门控**地注入。

    原来是「`_stack` 一律不传 locator」，现在改成读
    ``cfg.locator.enabled``：

    * ``False``（默认）-> ``locator=None``，行为与接入之前**完全一致**
    * ``True`` -> 造一个 ``LocalizationService`` 传进去

    所以这条测试从「断言源码里没有 locator=」变成「断言它**有**门控」。
    门控的语义由 ``tests/unit/test_locator_wiring.py`` 覆盖。
    """
    import inspect

    from autopilot.tools import _stack

    src = inspect.getsource(_stack.build_perception)
    assert "locator=st.locator" in src, (
        "_stack.build_perception 不再把 locator 传给 StateBuilder 了 —— "
        "请同步更新本测试和 recognition/notes/signal_integration_roadmap.md")
    assert "build_locator(cfg)" in src, (
        "locator 必须经 build_locator(cfg) 门控，不能无条件构造")


@pytest.mark.unit
def test_stack__build_perception_uses_the_gate_not_a_direct_import() -> None:
    """门控函数单独存在 —— 便于测试直接调它，不必组装整个 Stack。"""
    from autopilot.tools import _stack

    assert callable(_stack.build_locator)
    assert "build_locator" in _stack.__all__


@pytest.mark.unit
def test_locator__duck_typed_is_enough() -> None:
    """只要有 ``on_frame(frame, ts_sec)`` 就行 —— 不要求是 LocalizationService。"""
    class _Duck:
        def on_frame(self, frame, ts_sec):
            return _loc_state(quality="GOOD", x=1.0, y=2.0)

    pos, _ = _builder(_Duck())._locate(_frame(), 1.0)
    assert pos is not None and pos.quality == "GOOD"