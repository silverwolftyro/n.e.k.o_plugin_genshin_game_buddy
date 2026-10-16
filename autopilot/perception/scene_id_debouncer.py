"""场景 ID 时序去抖器（多类别）。

★ 为什么需要它：``SceneDetector`` 只回答"是不是大世界"（二值），回答不了"我在哪个
场景"。builder 需区分 7 个场景，其中 **6 个是非大世界**，故 ``SceneDetector.changed``
当不了场景切换触发源 —— ``MAIN_MENU → BAG_INVENTORY`` 全程 ``is_world=False``。

阈值与 ``scene_detector`` 平行（沿用其 session_a 实测标定的 N/M）：首次切换需 3.0s，
后续只需 1.0s。**冷启动第一帧直接采信**原始 id（同 ``SceneDetector.is_first``），否则
builder 开机后有 3 秒拿不到场景 id、整条模板路由空转；冷启动**不计为一次切换**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

__all__ = ["DebouncedScene", "SceneIdDebouncer",
           "DEFAULT_N_SECONDS", "DEFAULT_M_SECONDS"]

DEFAULT_N_SECONDS: float = 3.0
DEFAULT_M_SECONDS: float = 1.0

#: 阈值比较的浮点容差：逐帧累加会系统性偏小，没容差会在边界上差一帧。
_EPS = 1e-9


@dataclass(frozen=True)
class DebouncedScene:
    """单次 :meth:`SceneIdDebouncer.update` 的输出。"""

    scene_id: str          #: 去抖后的稳定场景 ID
    changed: bool          #: 这一步是否刚完成一次切换
    pending_seconds: float #: 累计的"待切换"秒数（稳定态恒 0.0）
    is_first: bool = False #: 是否是首次调用


class SceneIdDebouncer:
    """多类别场景 ID 的时序去抖状态机。"""

    def __init__(self, n_seconds: float = DEFAULT_N_SECONDS,
                 m_seconds: float = DEFAULT_M_SECONDS):
        for name, v in (("n_seconds", n_seconds), ("m_seconds", m_seconds)):
            if v < 0 or math.isnan(v):
                raise ValueError(f"{name} 非法: {v}（不能为负或 NaN）")
        self.n_seconds = float(n_seconds)
        self.m_seconds = float(m_seconds)
        self.reset()

    def reset(self) -> None:
        """清空全部状态。下一次 ``update()`` 视为首次调用。"""
        self._stable_id: Optional[str] = None
        self._pending_id: Optional[str] = None
        self._pending_seconds: float = 0.0
        self._last_ts: Optional[float] = None
        self._switch_count: int = 0

    @property
    def stable_id(self) -> Optional[str]:
        """当前稳定场景 ID。未初始化过为 None。"""
        return self._stable_id

    def update(self, scene_id: str, timestamp: float) -> DebouncedScene:
        """喂一帧的原始场景 ID，推进状态机。

        :raises ValueError: **时间戳倒退**。响亮失败不钳零 —— 钳零会让去抖计时错乱。
        """
        if timestamp is None:
            raise TypeError("timestamp 不能为 None")
        ts = float(timestamp)
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError(
                f"时间戳倒退: {ts:.6f}s < 上一次 {self._last_ts:.6f}s"
                f"（响亮失败：钳到 0 步长会让去抖计时错乱）")
        dt = 0.0 if self._last_ts is None else ts - self._last_ts
        self._last_ts = ts

        if self._stable_id is None:              # 冷启动：首帧直接采信
            self._stable_id = scene_id
            return DebouncedScene(scene_id, False, 0.0, True)

        changed = False
        if scene_id == self._stable_id:
            self._pending_id = None
            self._pending_seconds = 0.0
        else:
            if scene_id != self._pending_id:     # 候选换了，计时从头开始
                self._pending_id = scene_id
                self._pending_seconds = 0.0
            self._pending_seconds += dt
            need = self.n_seconds if self._switch_count == 0 else self.m_seconds
            if self._pending_seconds >= need - _EPS:
                self._stable_id = scene_id
                self._pending_id = None
                self._pending_seconds = 0.0
                self._switch_count += 1
                changed = True

        return DebouncedScene(self._stable_id, changed, self._pending_seconds)
