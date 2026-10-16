"""场景检测器（时序去抖 · 状态机版）。

★ 为什么需要时序去抖
--------------------
底层单帧判定 ``coord_scene_tree`` 是**无状态**的，它有 4 个已知误判
（见 ``recognition/notes/scene_detector_report.md``）：

1. HUD 淡出态（冲刺 / 交互时游戏隐藏 HUD）
   血条 ROI 全黑 -> edge_density 约等于 0 -> 误判"非大世界"
   实测：``a:f00071`` / ``a:f00079`` / ``a:f00083``
2. 阈值贴边（浮点抖动）
   实测：``b:f00108``

这两种误判**都是短时间现象**（HUD 淡出是几帧，浮点抖动是单帧）。
**连续 N 帧一致才切换状态**就能同时过滤掉它们。

★ 状态机
--------
::

    [大世界] --连续 N 秒"非大世界"--> [非大世界]
    [非大世界] --连续 M 秒"大世界"--> [大世界]

N=3.0 秒（进入去抖，保守：宁可晚进菜单，不可早进）
M=1.0 秒（退出去抖，激进：确认还在大世界就尽快恢复）

**时间基准由调用方传入**（``update(frame, timestamp)``），
不用 ``time.time()``。这样离线回放、单元测试、变速帧率下行为一致。

★ 职责边界
----------
**本检测器只输出状态，不做任何动作** —— 不通知小地图定位器、
不暂停传送、不发按键。``SceneState.changed`` 置 True 时，
上层拿 ``changed`` 自己决定怎么做。

★ 实测标定（session_a，2026-10-01）
------------------------------------
录制内容：12 段操作、11 次状态切换。
数据：1512 帧 / 151.2 秒 / 10 fps，**每帧带真实时间戳**。

N/M 扫描 15 组（N ∈ {1,1.5,2,2.5,3}s × M ∈ {0.3,0.5,1.0}s）：
**全部得到 11 次切换，与真实操作次数一致** → 结果对参数不敏感。

* 最短真实大世界段 **1.9 秒**（t=74.2~76.0）
* 最短真实非世界段 **8.6 秒**（t=65.6~74.1，退出大地图的过渡态）

据此确认 **N=3.0s / M=1.0s**：

* N=3.0s —— 最短非世界段 8.6s，有 2.8 倍余量；段内 0.2s 的
  raw 抖动被自然滤掉（合并后仍是完整一段）
* M=1.0s —— 最短大世界段 1.9s，1.0s 抓得到；0.4s 以下的抖动滤掉

已验证场景：

* 大世界 ↔ 背包 / 大地图 / 派蒙菜单（各 2~3 次切换）
* 放大招 / 冲刺的 HUD 淡出（被 N 过滤）
* 退出大地图的过渡态（8.6 秒）

⚠️ 这段录屏里**没有短于 1.9 秒的真实段**。若实际存在
"开背包后 0.8 秒就关掉"这类操作，M=1.0s 会漏掉它（保守失败，不危险）。

时间线图：``recognition/exports/benchmark/visualizations/ui_probe/
session_a_timeline.png``

★ 不做什么
----------
* 不做多分类（底层就是二分类）
* 不缓存图像（每帧只用一次）
* 不线程安全（单线程调用；多线程请各自持有一份实例）
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

__all__ = [
    "SceneDetector",
    "SceneState",
    "DEFAULT_M_SECONDS",
    "DEFAULT_N_SECONDS",
    "set_classifier_module_path",
]

#: 进入去抖秒数：连续判"非大世界"达到这么久才切换
DEFAULT_N_SECONDS = 3.0
#: 退出去抖秒数：连续判"大世界"达到这么久才切换
DEFAULT_M_SECONDS = 1.0

#: 阈值比较的浮点容差。远小于任何合理帧间隔（1e-9 秒）。
_EPS = 1e-9

#: 底层单帧判定模块路径。可用环境变量 NEKO_SCENE_TREE 覆盖。
_TREE_PATH = Path(os.environ.get(
    "NEKO_SCENE_TREE",
    r"D:\NEKO\recognition\exports\benchmark\scripts\misc\coord_scene_tree.py"))

_TREE = None


def set_classifier_module_path(path: str | Path) -> None:
    """改底层判定模块路径并清缓存。**下次调用时才会重新加载。**"""
    global _TREE, _TREE_PATH
    _TREE_PATH = Path(path)
    _TREE = None


def _tree():
    """加载底层判定模块。**找不到就抛，不静默退化。**"""
    global _TREE
    if _TREE is None:
        if not _TREE_PATH.is_file():
            raise FileNotFoundError(
                f"找不到底层场景判定模块: {_TREE_PATH}\n"
                f"用 set_classifier_module_path() 或环境变量 NEKO_SCENE_TREE 指定。\n"
                f"（响亮失败：找不到它就静默返回 False 会让检测器永远判"
                f"『非大世界』，比报错危险得多）")
        spec = importlib.util.spec_from_file_location(
            "neko_coord_scene_tree", _TREE_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 {_TREE_PATH}")
        mod = importlib.util.module_from_spec(spec)
        # ★ 必须先注册进 sys.modules，否则模块内的 @dataclass 会在
        #   cls.__module__ 查找时拿到 None 而炸。
        sys.modules["neko_coord_scene_tree"] = mod
        spec.loader.exec_module(mod)
        _TREE = mod
    return _TREE


def _default_classifier(frame) -> bool:
    """默认判定：BGR 整帧 -> 是否大世界。"""
    return bool(_tree().classify_frame(frame).is_world)


@dataclass
class SceneState:
    """单次 ``update()`` 的输出。"""

    is_world: bool              # 去抖后的稳定状态
    changed: bool               # 这一步是否刚发生状态切换
    raw_prediction: bool         # 未去抖的原始单帧判定
    pending_seconds: float       # 当前累计的"待切换"秒数
    is_first: bool = False      # 是否是首次调用

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (f"SceneState(is_world={self.is_world}, "
                f"changed={self.changed}, raw={self.raw_prediction}, "
                f"pending={self.pending_seconds:.2f}s)")


class SceneDetector:
    """带时序去抖的场景状态机。

    :param n_seconds: 进入去抖秒数（连续"非大世界"）
    :param m_seconds: 退出去抖秒数（连续"大世界"）
    :param classifier: 可注入的判定函数 ``frame -> bool``，
        用于单元测试。``None`` 时用默认的 ``coord_scene_tree``。

    用法::

        det = SceneDetector()
        st = det.update(frame_720, timestamp_sec)
        if st.changed:
            handle_state_switch(st.is_world)
    """

    def __init__(self, n_seconds: float = DEFAULT_N_SECONDS,
                 m_seconds: float = DEFAULT_M_SECONDS,
                 classifier: Optional[Callable[[object], bool]] = None):
        if n_seconds < 0 or m_seconds < 0:
            raise ValueError(
                f"去抖秒数不能为负: n={n_seconds} m={m_seconds}")
        self.n_seconds = float(n_seconds)
        self.m_seconds = float(m_seconds)
        self._classify = classifier or _default_classifier
        self.reset()

    def reset(self) -> None:
        """清空全部状态。下一次 ``update()`` 视为首次调用。"""
        self._stable: Optional[bool] = None    # None = 尚未初始化
        self._world_pending = 0.0
        self._nonworld_pending = 0.0
        self._last_ts: Optional[float] = None

    # ---------------------------------------------------------- 内部
    def _dt(self, timestamp: float) -> float:
        """本次时间步长。首次为 0；**时间戳倒退时钳到 0，不产生负累积**。"""
        if self._last_ts is None:
            self._last_ts = timestamp
            return 0.0
        dt = float(timestamp) - self._last_ts
        self._last_ts = float(timestamp)
        return dt if dt > 0.0 else 0.0

    # ---------------------------------------------------------- 主入口
    def update(self, frame, timestamp: float) -> SceneState:
        """喂一帧，推进状态机。

        :param frame: BGR 画面。**归一化 ROI 与分辨率无关**，
            720 基准或原始分辨率都可以。
        :param timestamp: 秒级时间戳。要求**单调不减**；倒退会被钳成 0 步长。
        :returns: :class:`SceneState`
        """
        if timestamp is None:
            raise TypeError("timestamp 不能为 None")
        timestamp = float(timestamp)

        raw = bool(self._classify(frame))
        dt = self._dt(timestamp)

        first = self._stable is None
        if first:
            # 首次调用直接采信，不等去抖 —— 否则冷启动要干等 N 秒
            self._stable = raw
            self._world_pending = 0.0
            self._nonworld_pending = 0.0
            return SceneState(is_world=raw, changed=False, raw_prediction=raw,
                              pending_seconds=0.0, is_first=True)

        # 累计两个方向的待切换时长
        if raw:
            self._world_pending += dt
            self._nonworld_pending = 0.0
        else:
            self._nonworld_pending += dt
            self._world_pending = 0.0

        changed = False
        # ★ 浮点容忍
        #   dt 逐帧累加会系统性偏小（30 次 +0.1 在 IEEE754 下是
        #   2.9999999999999996），直接 >= n 会让切换被推迟一帧。
        #   加一个远小于帧间隔的容差，保证"累计到 N 秒"这件事
        #   在数学上成立时不会因为最后一位的舍入而漏掉。
        if self._stable:
            # 稳定态是"大世界"，要切走需要连续 N 秒"非大世界"
            if self._nonworld_pending >= self.n_seconds - _EPS:
                self._stable = False
                changed = True
                self._nonworld_pending = 0.0
                pending = 0.0
            else:
                pending = self._nonworld_pending
        else:
            # 稳定态是"非大世界"，要切回需要连续 M 秒"大世界"
            if self._world_pending >= self.m_seconds - _EPS:
                self._stable = True
                changed = True
                self._world_pending = 0.0
                pending = 0.0
            else:
                pending = self._world_pending

        return SceneState(is_world=self._stable, changed=changed,
                          raw_prediction=raw, pending_seconds=pending,
                          is_first=False)

    # ---------------------------------------------------------- 观测
    @property
    def stable_state(self) -> Optional[bool]:
        """当前稳定状态；``None`` 表示还没喂过帧。"""
        return self._stable

    @property
    def world_pending_seconds(self) -> float:
        return self._world_pending

    @property
    def nonworld_pending_seconds(self) -> float:
        return self._nonworld_pending

    def __repr__(self) -> str:  # pragma: no cover
        return (f"SceneDetector(n={self.n_seconds}, m={self.m_seconds}, "
                f"stable={self._stable}, "
                f"world_pending={self._world_pending:.2f}, "
                f"nonworld_pending={self._nonworld_pending:.2f})")
