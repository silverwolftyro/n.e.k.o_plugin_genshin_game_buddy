"""工具共用引导：把插件目录挂到 sys.path，让 `python autopilot/tools/x.py` 能直接跑。

**R8 红线**：模块顶层不做副作用以外的事 —— 这里只在被导入时设置路径，
不建连接、不读配置、不改事件循环。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

#: autopilot 包根（tools/ 的上一级）
AUTOPILOT_ROOT = Path(__file__).resolve().parents[1]
#: 插件根（autopilot 的上一级）
PLUGIN_ROOT = AUTOPILOT_ROOT.parent
#: 仓库根（插件的上两级）
REPO_ROOT = PLUGIN_ROOT.parents[1]


def bootstrap() -> Path:
    """把插件根加入 sys.path，使 ``autopilot.*`` 可导入。返回插件根。"""
    p = str(PLUGIN_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return PLUGIN_ROOT


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 第三方库降噪
    for noisy in ("PIL", "urllib3", "matplotlib", "numba", "mss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


#: 终端着色（Windows 10+ 支持）
class C:
    OK = "\033[92m"
    WARN = "\033[93m"
    ERR = "\033[91m"
    DIM = "\033[90m"
    BOLD = "\033[1m"
    OFF = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        for k in ("OK", "WARN", "ERR", "DIM", "BOLD", "OFF"):
            setattr(cls, k, "")


def line(char: str = "─", width: int = 64) -> str:
    return char * width


__all__ = ["AUTOPILOT_ROOT", "C", "PLUGIN_ROOT", "REPO_ROOT", "bootstrap", "line", "setup_logging"]
