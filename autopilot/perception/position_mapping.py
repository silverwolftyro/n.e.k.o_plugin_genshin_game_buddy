"""定位状态 → ``PositionInfo`` 的**纯映射**。

★ 为什么要单独一个函数
----------------------
``LocalizationService`` 在 ``perception`` 内、``StatePacket`` 在 ``contracts``
里，两边都不该互相 import。把映射抽成纯函数有三个好处：

1. **可单测**：不需要游戏、不需要截图、不需要底图，构造一个
   :class:`LocalizationState` 就能验映射表。
2. **不污染依赖方向**：``contracts`` 零内部依赖这条铁律不能破。
3. **纯度可审计**：无全局状态、无 I/O、无时间读取。同样的输入永远同样输出。

★ quality 映射（用户 2026-10-02 拍板）
--------------------------------------
=========================  ================  ===================================
``LocalizationState``     ``quality``       理由
=========================  ================  ===================================
``TRACKING``              ``GOOD``          位移在正常累积
``PAUSED``                ``DEGRADED``      位置冻结但有效，可用于导航
``RESETTING``             ``DEGRADED``      位置冻结在传送前的旧值
``UNINITIALIZED``         ``LOST``          还没有起点
=========================  ================  ===================================

⚠️ **已知盲区（用户已确认本轮接受）**
------------------------------------
``PositionInfo`` **没有"位置有多旧"的字段**。``LocalizationState`` 上的
``position_age_s`` 在映射时丢失。后果：菜单里冻结 5 分钟的位置，与刚追踪到的
位置，在 ``PositionInfo`` 里**完全同形**（同样 DEGRADED + 同样 x/y）——
``actuator/validator.py:_check_position`` 无从区分。
补它需要改 ``contracts/state_packet.py``，属于改契约，另议。

⚠️ ``heading_deg`` 恒为 ``None``
--------------------------------
朝向提取**已证伪**（规划文档 §7：朝向 σ 中位 17.3°，两段失控 R=0.388/0.402）。
宁可不填，也不填一个假的。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

__all__ = ["position_from_state", "confidence_from_drift",
           "QUALITY_BY_STATE", "POSITION_SOURCE"]

#: 本映射产出的 ``PositionInfo.source``。区别于旧的外观匹配路线（已证伪）。
POSITION_SOURCE = "task_driven"


def _sibling(name: str):
    """加载同目录模块。**兼容包导入和按文件加载两种情况。**"""
    try:
        return importlib.import_module(f".{name}", __package__)
    except (ImportError, ValueError, TypeError):
        p = Path(__file__).with_name(f"{name}.py")
        s = importlib.util.spec_from_file_location(f"_nk_{name}", p)
        m = importlib.util.module_from_spec(s)
        sys.modules[f"_nk_{name}"] = m
        s.loader.exec_module(m)
        return m


def _position_info_cls():
    """拿 ``contracts.state_packet.PositionInfo``。

    生产走包内相对 import；单测/离线脚本没有包上下文时按文件路径加载。
    ``contracts/state_packet.py`` 只依赖标准库，两种方式都安全。
    """
    try:
        from ..contracts.state_packet import PositionInfo
        return PositionInfo
    except ImportError:
        p = Path(__file__).resolve().parents[1] / "contracts" / "state_packet.py"
        s = importlib.util.spec_from_file_location("_nk_state_packet", p)
        m = importlib.util.module_from_spec(s)
        sys.modules["_nk_state_packet"] = m
        s.loader.exec_module(m)
        return m.PositionInfo


def _quality_table() -> dict:
    """状态机枚举 → quality。**每次现取**，不缓存模块级状态。"""
    sm = _sibling("localization_service").LocalizationStateMachine
    return {
        sm.TRACKING: "GOOD",
        sm.PAUSED: "DEGRADED",
        sm.RESETTING: "DEGRADED",
        sm.UNINITIALIZED: "LOST",
    }


def confidence_from_drift(drift_bound: float) -> float:
    """漂移上界 → 置信度。``confidence = 1 / (1 + drift_bound)``。

    ⚠️ **未标定**。半衰常数取 1.0 是拍脑袋的 —— 实测 ``drift_bound`` 只在
    session_a 的 3 段纯小地图走路里走过（0.44~0.47 世界单位），没有覆盖
    长时间漫游 / 多次传送的分布。传送硬重置时 ``drift_bound == 0`` → 1.0，
    与传送契约 §4 的"传送是硬事实、无不确定性"一致。

    :param drift_bound: 漂移上界，世界单位。非负。
    :returns: ``(0, 1]`` 区间内的置信度
    :raises ValueError: ``drift_bound`` 为负或 NaN
    """
    d = float(drift_bound)
    if d != d or d < 0.0:                       # NaN 或负数
        raise ValueError(f"drift_bound 非法: {drift_bound}（不能为负或 NaN）")
    return 1.0 / (1.0 + d)


def position_from_state(loc_state) -> "object":
    """``LocalizationState`` → ``PositionInfo``。**纯函数**。

    :param loc_state: :class:`localization_service.LocalizationState`
    :returns: :class:`contracts.state_packet.PositionInfo`

    ⚠️ **总是返回 ``PositionInfo``，不返回 ``None``**（除非入参本身是 None）。
    没有位置时返回 ``quality="LOST"`` 且 ``x/y=None`` 的包 —— 对齐
    ``PositionInfo.lost()``。理由：``LOST`` 与 ``None`` 在
    ``validator._check_position`` 里都被拒，但包形式能同时表达
    "无位置"和"当前状态"，对日志与调试更有用。

    :raises ValueError: ``drift_bound`` 非法（会向上传播）
    """
    if loc_state is None:
        return None

    quality = _quality_table()[loc_state.state]
    pos = loc_state.world_pos
    if pos is None:
        return _position_info_cls().lost()

    wx, wz = float(pos[0]), float(pos[1])
    return _position_info_cls()(
        quality=quality,
        zone=loc_state.tile_id,
        x=wx,
        y=wz,
        heading_deg=None,          # 朝向提取已证伪（§7），宁缺勿造
        confidence=confidence_from_drift(loc_state.drift_bound),
        source=POSITION_SOURCE,
    )
