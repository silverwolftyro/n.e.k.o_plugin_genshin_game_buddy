"""autopilot 双轨 UI · A 轨（独立 overlay）的公共层。

本包是**协议层**，不含任何 tkinter / IO 实现细节。协议先定死，
实现（v1 文件、v2 socket）才能换。

依赖方向是单向的::

    core (run_loop) ──写──> protocol <──读── overlay
                              │
                        schema（只描述数据，不描述传输）

overlay 与 run_loop 是**两个独立进程，互不知道对方存在**，
只通过 autopilot/run/ 下的文件通信。

⚠ 不 import ultralytics（AGPL）。本包只用 pydantic + 标准库。
"""

from __future__ import annotations

__all__: list[str] = []
