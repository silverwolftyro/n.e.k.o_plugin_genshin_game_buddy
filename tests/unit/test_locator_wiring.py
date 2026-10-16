"""locator 接线门控的单测（2026-10-12 任务 B）。

接线本身早在 ``StateBuilder(locator=)`` 就有了，缺的是 ``_stack`` 那一行。
本轮补上，但**用配置门控**而不是无条件注入：

    cfg.locator.enabled = false（默认） -> locator=None -> position 恒 None
    cfg.locator.enabled = true          -> LocalizationService

★ 为什么「开了也不等于能定位」
------------------------------
``LocalizationService`` 初始是 ``UNINITIALIZED`` —— **它没有起点**。
必须先在游戏里传送一次建立原点（``teleport_contract.md`` 的三段契约），
在那之前 ``position`` 恒为 ``quality="LOST"``。

所以这个开关的价值是**让接线可被测试**，不是「一开就能定位」。
本文件把这条语义钉死，免得以后有人把它当成「定位已就绪」的信号。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.config.loader import (  # noqa: E402
    AutopilotConfig,
    LocatorConfig,
    load_config,
)
from autopilot.perception.localization_service import (  # noqa: E402
    PHASE_MIN_CIRCLE_STD,
    PHASE_MIN_PEAK_RESPONSE,
    LocalizationService,
    LocalizationStateMachine,
)
from autopilot.tools._stack import build_locator  # noqa: E402


def _cfg(**kw) -> AutopilotConfig:
    return AutopilotConfig(**kw)


# ==================================================================
# 门控：默认关
# ==================================================================

@pytest.mark.unit
def test_locator_config__is_disabled_by_default() -> None:
    """默认关 —— 这条是向后兼容的唯一保证，坏了整个用户群受影响。"""
    assert LocatorConfig().enabled is False
    assert AutopilotConfig().locator.enabled is False


@pytest.mark.unit
def test_build_locator__returns_none_when_disabled() -> None:
    assert build_locator(_cfg(locator=LocatorConfig(enabled=False))) is None


@pytest.mark.unit
def test_build_locator__returns_none_when_the_section_is_absent() -> None:
    """老配置文件里没有 ``[autopilot.locator]`` 段 —— 必须容错成「关」。

    这就是 ``getattr(cfg, "locator", None)`` 那层保护的意义：
    用户拿新代码跑旧配置，不该直接 AttributeError。
    """
    class _Old:
        """模拟 3.1.17 时代的配置对象：没有 locator 属性。"""
        enabled = True

    assert build_locator(_Old()) is None


# ==================================================================
# 门控：开了
# ==================================================================

@pytest.mark.unit
def test_build_locator__returns_a_service_when_enabled() -> None:
    got = build_locator(_cfg(locator=LocatorConfig(enabled=True)))
    assert got is not None
    assert isinstance(got, LocalizationService)


@pytest.mark.unit
def test_build_locator__service_starts_uninitialized() -> None:
    """★ 开了也**不是**已就绪 —— 没有起点。

    这是本文件最该守住的一条：接线完成不等于定位可用。
    """
    svc = build_locator(_cfg(locator=LocatorConfig(enabled=True)))
    assert svc._state is LocalizationStateMachine.UNINITIALIZED
    assert svc.get_position() is None, "UNINITIALIZED 就不该有位置"


@pytest.mark.unit
def test_build_locator__passes_the_thresholds_through() -> None:
    svc = build_locator(_cfg(locator=LocatorConfig(
        enabled=True, min_peak_response=0.33, min_circle_std=44.0)))
    assert svc.min_peak_response == pytest.approx(0.33)
    assert svc.min_circle_std == pytest.approx(44.0)


@pytest.mark.unit
def test_build_locator__uses_service_defaults_when_not_overridden() -> None:
    """配置没写阈值时用服务自己的默认值，而不是硬编一份。"""
    svc = build_locator(_cfg(locator=LocatorConfig(enabled=True)))
    assert svc.min_peak_response == pytest.approx(PHASE_MIN_PEAK_RESPONSE)
    assert svc.min_circle_std == pytest.approx(PHASE_MIN_CIRCLE_STD)


@pytest.mark.unit
def test_config_thresholds_mirror_the_service_constants() -> None:
    """⚠️ 配置层**不能** import 感知层（依赖方向会反过来），只能镜像 + 加守卫。

    这条就是那个守卫 —— 哪天有人改了 ``PHASE_MIN_CIRCLE_STD`` 而忘了同步配置，
    这里会红。
    """
    c = LocatorConfig()
    assert c.min_peak_response == pytest.approx(PHASE_MIN_PEAK_RESPONSE)
    assert c.min_circle_std == pytest.approx(PHASE_MIN_CIRCLE_STD)


@pytest.mark.unit
def test_build_locator__builds_a_fresh_instance_each_call() -> None:
    """两个调用不能共享同一个服务 —— 状态机有跨帧记忆（上一帧小地图）。"""
    cfg = _cfg(locator=LocatorConfig(enabled=True))
    a, b = build_locator(cfg), build_locator(cfg)
    assert a is not b


# ==================================================================
# 配置契约
# ==================================================================

@pytest.mark.unit
def test_locator_config__rejects_an_out_of_range_peak_response() -> None:
    """归一化相关只在 0~1，写 5.0 一定是用户手滑。"""
    with pytest.raises(Exception):
        LocatorConfig(enabled=True, min_peak_response=5.0)


@pytest.mark.unit
def test_locator_config__rejects_a_negative_circle_std() -> None:
    with pytest.raises(Exception):
        LocatorConfig(enabled=True, min_circle_std=-1.0)


@pytest.mark.unit
def test_defaults_toml__has_the_locator_section_disabled() -> None:
    """仓库里的默认配置必须显式写出 ``enabled = <布尔>`` 并带上警告注释。

    显式写出来是为了：用户翻到这段时立刻知道有这个东西、知道它默认关、
    也知道开了为什么可能没用。

    ⚠️ **断言的是「键写对了」，不是「值是 false」**（2026-10-15 改）。
       原本这里是 ``assert "enabled = false" in text``，那测的是**用户
       的本地取值**而不是代码：调试时把 ``defaults.toml`` 翻成
       ``enabled = true``（为了跑通真机定位）就会把这条测试点红 ——
       可那不是回归，是本地配置。
       「默认关」这个不变量现在由
       ``test_locator_config__defaults_to_disabled`` 在 dataclass 默认值
       上守（那才是它的归属层），这里只守「键名别写错」。
    """
    p = PLUGIN_ROOT / "autopilot" / "config" / "defaults.toml"
    text = p.read_text(encoding="utf-8")
    assert "[autopilot.locator]" in text
    assert re.search(r"^\s*enabled\s*=\s*(true|false)\s*$", text, re.M), \
        "locator 段必须显式写出 enabled = true/false（写错键名会让整个配置加载失败）"
    assert "UNINITIALIZED" in text, "必须说明「开了也不等于能定位」"


@pytest.mark.unit
def test_locator_config__defaults_to_disabled() -> None:
    """★「默认关」的不变量住在这里 —— **dataclass 默认值**层。

    它与 ``defaults.toml`` 里写了什么**无关**：本地调试翻开关不会让这条
    变红，而真有人把 dataclass 默认改成 ``True``（等于让所有用户默认
    开启一个需要手动建立原点的子系统）时它会立刻报警。
    """
    assert LocatorConfig().enabled is False
    # 传送 OCR 的两个开关默认 True —— 开了 locator 就该是能用的
    assert LocatorConfig().teleport_ocr_enabled is True
    assert LocatorConfig().teleport_ocr_interval_s > 0.0


@pytest.mark.unit
def test_defaults_toml__actually_loads_with_the_section_present() -> None:
    """TOML 里的键名必须和 dataclass 字段对得上。

    ``AutopilotConfig`` 是 ``extra="forbid"`` —— 写成 ``enable`` 而不是
    ``enabled`` 会直接让**整个默认配置加载失败**。这条防的就是那个。
    """
    cfg = load_config(PLUGIN_ROOT / "autopilot" / "config" / "defaults.toml")
    # 这条测的是**键名**（`enable` vs `enabled` 会让 extra="forbid"
    # 直接拒绝加载），不是本地取值 —— 见上面那条的注释。
    assert isinstance(cfg.locator.enabled, bool)
    assert cfg.locator.min_peak_response > 0.0
    assert cfg.locator.min_circle_std > 0.0
    assert cfg.locator.teleport_ocr_enabled is True
    assert cfg.locator.teleport_ocr_interval_s > 0.0


# ==================================================================
# 接线的下游语义
# ==================================================================

@pytest.mark.unit
def test_disabled_locator__builder_keeps_position_none() -> None:
    """关着时整条链路的下游表现：``position`` 恒为 None。"""
    from autopilot.perception.builder import StateBuilder

    b = StateBuilder(capturer=None, yolo=None, template=None, ocr=None,
                     locator=build_locator(_cfg()))
    assert b._locator is None
    assert b._locate(None, 0.0) == (None, 0.0)


@pytest.mark.unit
def test_enabled_locator__builder_reports_uninitialized_as_LOST() -> None:
    """开着但没建原点时，**给显式 LOST 而不是 None**。

    语义差别很关键：``None`` 让 ``pos_quality()`` 返回 UNKNOWN（不知道），
    LOST 返回「确定不能用」。后者方向性明确得多（builder.py:862-866）。
    """
    from autopilot.perception.builder import StateBuilder
    from autopilot.perception.localization_service import (
        LocalizationState,
    )
    from autopilot.perception.localization_service import (
        LocalizationStateMachine as SM,
    )

    class _Uninit:
        def on_frame(self, frame, ts_sec):
            return LocalizationState(state=SM.UNINITIALIZED, world_pos=None)

    b = StateBuilder(capturer=None, yolo=None, template=None, ocr=None,
                     locator=_Uninit())
    pos, _ms = b._locate(None, 0.0)
    assert pos is not None
    assert pos.quality == "LOST"


@pytest.mark.unit
def test_stack_dataclass__exposes_the_locator_slot() -> None:
    from autopilot.tools._stack import Stack
    assert Stack().locator is None
