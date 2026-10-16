"""Autopilot B 轨 · 插件侧桥接：读主控台状态 + 发命令给主控台。

★ 与主控台的协议（两边必须一致）
----------------------------------
    主控台 -> 插件   ``console_bridge.json``     本文件 :func:`read_bridge`
    插件 -> 主控台   ``console_commands.jsonl``  本文件 :func:`send_command`

状态文件的字段与命令行的形状由 ``NEKO-Console/neko_console/bridge.py``
唯一定义。改格式要**两边同时改**，否则面板会静默失联。

★ 红线：模块顶层不做任何 IO
--------------------------
不读文件、不建连接、不起定时器。所有 IO 都在函数被调用时发生。
插件在 import 期被宿主加载，那一刻动文件系统是不允许的。

本模块**故意不 import** NEKO-Console 的任何代码 —— 那是另一个仓库、
另一个进程，甚至可能是另一个 Python 环境。唯一的耦合是上面那两个
文件名和它们的 JSON 形状。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 插件版本号。**与 plugin.toml 的 version 字段保持一致**。
#:
#:   这里刻意复制一份而不是去读 plugin.toml：模块顶层不做 IO（插件红线 R8），
#:   而 plugin.toml 里那份是给宿主看的。两份不一致时以 plugin.toml 为准，
#:   所以**改版本号时这两个地方都要改**。
PLUGIN_VERSION = "3.1.16"


def plugin_version() -> str:
    """插件版本号，供 @ui.context 透出给面板显示。"""
    return PLUGIN_VERSION

#: ── 协议常量（与主控台 bridge.py 一一对应）──────────────────────────
BRIDGE_FILE_NAME = "console_bridge.json"
COMMANDS_FILE_NAME = "console_commands.jsonl"
BRIDGE_SCHEMA_VERSION = 1
COMMANDS_SCHEMA_VERSION = 1

#: 主控台超过这个时间没更新，就认为它不在了。
#: 主控台每 200ms 写一次，3 秒的容忍度约等于 15 个周期 ——
#: 足以扛住磁盘卡顿或一次 GC，又不至于让面板一直显示"运行中"。
CONSOLE_ALIVE_TIMEOUT_MS = 3000

#: 命令全集。**只发这几个**，不要自己扩展。
VALID_COMMANDS = (
    "START_ENGINE",
    "STOP_ENGINE",
    "START_OVERLAY",
    "STOP_OVERLAY",
    "QUIT_CONSOLE",
)

#: ── 默认路径 ────────────────────────────────────────────────────────
#: 主控台仓库根。换机器改这两个常量，或用 :func:`configure` 覆盖。
DEFAULT_CONSOLE_DIR = r"D:\NEKO\NEKO-Console"
DEFAULT_BRIDGE_PATH = str(Path(DEFAULT_CONSOLE_DIR) / "runtime" / BRIDGE_FILE_NAME)

# ── 运行时覆盖（由插件在启动时调一次）────────────────────────────────
_bridge_path_override: str | None = None
_console_dir_override: str | None = None

# ── 读缓存（mtime 短路用）────────────────────────────────────────────
_cache_path: Path | None = None
_cache_mtime_ns: int | None = None
_cache_size: int | None = None
_cache_data: dict[str, Any] | None = None


def configure(*, bridge_path: str | None = None,
              console_dir: str | None = None) -> None:
    """由插件启动时调用一次，覆盖默认路径。

    ★ 为什么不做成"每次去读插件配置"
      那意味着每次读桥接都要再开一次 plugin.toml。插件配置的读取应该
      发生在启动阶段（宿主已经把 config 交给插件了），而不是每 200ms 一次。
    """
    global _bridge_path_override, _console_dir_override
    _bridge_path_override = bridge_path or None
    _console_dir_override = console_dir or None


def bridge_path() -> Path:
    """桥接状态文件路径。"""
    return Path(_bridge_path_override or DEFAULT_BRIDGE_PATH)


def commands_path() -> Path:
    """命令文件路径（插件写、主控台读）。"""
    base = Path(_console_dir_override or DEFAULT_CONSOLE_DIR)
    return base / "runtime" / COMMANDS_FILE_NAME


def read_bridge() -> dict[str, Any] | None:
    """读主控台状态。**读不到 / 不认得一律返回 None**。

    ★ mtime 短路
      面板可能每秒调好几次，而主控台 200ms 才写一次。不看 mtime 就等于
      每个调用都开一次文件、解析一次 JSON。现在文件没变就返回上次结果。
      同时比 ``st_size`` —— 极端情况下同一 mtime 下内容变了（比如某些
      文件系统 mtime 精度只有 1 秒），size 能兜住。
    """
    global _cache_path, _cache_mtime_ns, _cache_size, _cache_data

    p = bridge_path()
    try:
        st = p.stat()
    except OSError:
        # 文件不存在 = 主控台没跑。顺手把缓存清掉，否则主控台退出后
        # 面板还会一直显示它最后一份状态 —— 那是最误导的一种"假活"。
        _cache_path, _cache_mtime_ns, _cache_size, _cache_data = None, None, None, None
        return None

    if (p == _cache_path and st.st_mtime_ns == _cache_mtime_ns
            and st.st_size == _cache_size):
        return _cache_data

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        # 主控台正在 replace 的那一瞬间可能读到半截 JSON —— 等下一拍
        logger.debug("桥接状态暂时读不了：%s", exc)
        return _cache_data

    if not isinstance(data, dict):
        return _cache_data
    # ★ 版本不认得就当读不了。宁可显示"未知"，也不要按自己的理解去解析
    #   一个未来的字段布局，那会显示出一堆看似合理其实编造的数据。
    if data.get("schema_version") != BRIDGE_SCHEMA_VERSION:
        logger.warning("桥接状态版本不认识：%r（期望 %s）",
                       data.get("schema_version"), BRIDGE_SCHEMA_VERSION)
        return _cache_data

    _cache_path, _cache_mtime_ns, _cache_size, _cache_data = (
        p, st.st_mtime_ns, st.st_size, data)
    return data


def is_console_alive(bridge: dict[str, Any] | None) -> bool:
    """主控台是不是活着。

    两条都满足才算活：
      1. ``console.running == True``
      2. ``written_at_ms`` 距今 < 3 秒

    ★ 为什么第 2 条不能省
      1 条只看 ``running`` 的话，主控台**崩溃**时最后一份状态里
      ``running`` 仍然是 true，面板会一直显示"运行中" —— 而实际上
      那个进程早就没了，用户点「完全退出」也点不动。
      心跳超时是唯一能发现"进程没了"的信号。
    """
    if not isinstance(bridge, dict):
        return False
    console = bridge.get("console")
    if not isinstance(console, dict) or console.get("running") is not True:
        return False
    # ★★ `written_at_ms` 在 **bridge 顶层**，不在 console 里面 ★★
    #   主控台 app.py::_write_bridge 写的是：
    #       {"schema_version":…, "written_at_ms":…, "console": {running,version,pid},
    #        "engine":…, "overlay":…, "stats":…}
    #   心跳时间戳描述的是**整份文件**的写入时刻，和 console 自己无关，
    #   所以它是顶层字段。
    #
    #   曾经写成 console.get("written_at_ms") —— console 里根本没有这个键，
    #   isinstance 判定恒 False，于是 is_console_alive 恒返回 False，
    #   面板永远显示「主控台未启动」，哪怕主控台好好地开着。
    #
    #   ★ 不要为了"省事"往 console 里再塞一份 written_at_ms：
    #     那就是两份真相源，主控台和插件迟早会漂移。以主控台实际写的为准。
    ts = bridge.get("written_at_ms")
    if not isinstance(ts, (int, float)):
        return False
    return (int(time.time() * 1000) - int(ts)) < CONSOLE_ALIVE_TIMEOUT_MS


def console_version(bridge: dict[str, Any] | None) -> str | None:
    """主控台版本号；读不到返回 None。"""
    if not isinstance(bridge, dict):
        return None
    console = bridge.get("console")
    if not isinstance(console, dict):
        return None
    v = console.get("version")
    return v if isinstance(v, str) else None


def send_command(cmd: str, args: dict[str, Any] | None = None) -> str:
    """往命令文件追加一行。返回一条给用户看的中文结果。

    :raises ValueError: 命令不在 :data:`VALID_COMMANDS` 里 —— 这是
        调用方的 bug（前端按钮写错了 id），必须响亮地失败而不是发出去
        让主控台拒绝。

    ★ 为什么要先校验再写
      命令是对另一个进程的控制指令。主控台侧会拒掉不认识的命令（记
      WARN 后跳过），但那样用户只会看到"点了没反应"。在这里就拒掉，
      至少插件侧能给出明确原因。
    """
    if cmd not in VALID_COMMANDS:
        raise ValueError(
            f"未知命令 {cmd!r}；可用：{', '.join(VALID_COMMANDS)}")

    line = json.dumps({
        "schema_version": COMMANDS_SCHEMA_VERSION,
        "cmd_id": uuid.uuid4().hex,
        "ts_ms": int(time.time() * 1000),
        "cmd": cmd,
        "args": args or {},
    }, ensure_ascii=False)

    p = commands_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # ★ 追加必须是**单次** open+write。
        #   主控台正在按 offset 读这个文件，分两次写的话，它可能正好读到
        #   "{" 之后换行之前那半截。JSONL 的一行要么完整要么不存在 ——
        #   中间态只能靠"读到换行才算一条命令完整"来兜（主控台侧已做）。
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
    except OSError as exc:
        logger.warning("发命令失败：%s", exc)
        return f"发送失败：{exc}"

    logger.info("已发命令 %s -> %s", cmd, p)
    return f"已发送 {cmd}"


__all__ = [
    "BRIDGE_FILE_NAME",
    "BRIDGE_SCHEMA_VERSION",
    "COMMANDS_FILE_NAME",
    "COMMANDS_SCHEMA_VERSION",
    "CONSOLE_ALIVE_TIMEOUT_MS",
    "DEFAULT_BRIDGE_PATH",
    "DEFAULT_CONSOLE_DIR",
    "PLUGIN_VERSION",
    "VALID_COMMANDS",
    "bridge_path",
    "commands_path",
    "configure",
    "console_version",
    "is_console_alive",
    "plugin_version",
    "read_bridge",
    "send_command",
]
