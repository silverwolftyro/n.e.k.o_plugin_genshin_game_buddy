"""导航决策层（**纯函数库**）。

这里只放「目标点 -> 转向/行走建议」的**算法**，不含任何 IO、不含线程、
不含进程间通信。调用方（未来的 ``policy/pathfind.py`` 或 ``run_loop``）
负责拿状态、按下发键盘鼠标。

为什么先把它做成库（而不是直接并进 ``core/loop.py``）
------------------------------------------------------
``e2e_pathfind.py`` 与引擎是两套编排、抢同一套键鼠。合并的第一步**不是**
搬代码，而是先把算法从编排里剥出来 —— 剥出来才能单测，而这段转向律的
每个常数都是真机标定出来的，恰恰最需要单测钉死。

本模块的每一个默认值都来自 ``e2e_pathfind.py`` 的同名常量，
抽出来时**逐条对照**，没有做任何重新标定。

⚠️ 关于名字
----------
子模块叫 :mod:`autopilot.nav.steer`，里面的顶层决策函数也叫 ``steer``。
所以本包**不**在 ``__all__`` 里用裸名 ``steer`` 转发那个函数 ——
那会把子模块属性覆盖掉，``from autopilot.nav import steer`` 拿到的将是
函数而不是模块（本项目已经踩过一次）。
要用那个函数请写::

    from autopilot.nav.steer import steer          # 函数
    from autopilot.nav import steer_plan           # 同函数的别名
    from autopilot.nav import steer as steer_mod   # 模块
"""

from __future__ import annotations

from . import steer
from .steer import (
    DEFAULT_CONFIG,
    SteerConfig,
    SteerPlan,
    bearing_deg,
    circ_mean_deg,
    dynamic_max_step_px,
    heading_deg,
    split_turn,
    stand_turn_step,
    turn_plan,
    wrap_deg,
)

#: :func:`autopilot.nav.steer.steer` 的别名（避免与子模块同名）。
steer_plan = steer.steer

__all__ = [
    "DEFAULT_CONFIG",
    "SteerConfig",
    "SteerPlan",
    "bearing_deg",
    "circ_mean_deg",
    "dynamic_max_step_px",
    "heading_deg",
    "split_turn",
    "stand_turn_step",
    "steer",
    "steer_plan",
    "turn_plan",
    "wrap_deg",
]
