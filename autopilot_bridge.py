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
PLUGIN_VERSION = "3.1.18"


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

#: 读不到主控台配置时的播报档位。
#:
#: ★ 与 ``event_to_yui.DEFAULT_VERBOSITY`` 取值一致（都是 1.0），
#:   但**故意各写一份**：本模块的红线是「不 import 插件内任何东西」，
#:   而 event_to_yui 是插件内的模块。两份必须同值这件事由
#:   ``test_default_verbosity_matches_event_to_yui`` 钉住。
#:
#: ★ 选 1.0（详尽）而不是 0.0
#:   主控台没跑的时候，播报不该突然闭嘴 —— 那会被误解成
#:   「我刚把播报关了」。宁吵不漏。
DEFAULT_YUI_VERBOSITY = 1.0

#: 宿主配置里的「话痨度预设」→ 播报档位（0.0~1.0）。
#:
#: ★ 三档与 Console 滑块的档位口径**逐字对齐**
#:   （Console 滑块直接给的就是 0.0 / 0.5 / 1.0 这三个数）。
#:
#:   quiet  = 0.0 -> tier 1（只播关键）
#:   normal = 0.5 -> tier 1 + 2
#:   chatty = 1.0 -> tier 1 + 2 + 3
#:
#: ★ 刻意各写一份而不 import event_to_yui：模块的红线是「不 import 插件内
#:   任何东西」。两份必须同值这件事由
#:   ``test_verbosity_preset_matches_event_to_yui`` 钉住。
VERBOSITY_PRESETS: dict[str, float] = {
    "quiet": 0.0,
    "normal": 0.5,
    "chatty": 1.0,
}

#: 预设名「等于这个值」时视为**用户没管**，交回给 Console 滑块。
#:
#: ★ 为什么需要这一条
#:   宿主把清单默认值一起写进用户配置副本（``<用户数据>/plugins/<id>/config/
#:   plugin.toml``），所以清单里写着的默认值**永远在文件里**。
#:   若"键存在即覆盖"，Console 滑块就永远轮不到说话 —— 而滑块恰恰是
#:   用户最常用的即时开关。于是这里把「等于清单默认值」解释成
#:   「我没管它」；想用宿主配置压过滑块，选 quiet / normal。
DEFAULT_PRESET_NAME = "chatty"

#: 宿主配置给出的档位。``None`` = 宿主没给（交回给 Console 滑块）。
#:
#: ★ 谁写它
#:   插件在 ``_reload_config()`` 里读到生效配置后调
#:   :func:`set_host_verbosity_preset` 写进来。宿主改配置会推 CONFIG_UPDATE
#:   -> ``config_change`` 生命周期 -> 重新读配置 -> 再写一次，
#:   所以**改宿主配置不需要重启插件**。
_host_verbosity: float | None = None

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
DEFAULT_CONSOLE_DIR = r"D:\workspace\N.E.K.O\NEKO-genshin\NEKO-Console"
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


def set_host_verbosity_preset(name: object | None) -> None:
    """记下宿主配置里的「话痨度预设」。

    由插件在读到生效配置后调用（``__init__.py`` 的 ``_reload_config``）。
    传 ``None`` / 空串 = 宿主没给（交回给 Console 滑块）。

    ★ 认不出来的取值**不抛异常**，只当没给
      这是一个「用户手滑写错了一个词」的场景。让播报线程炸掉没有意义，
      而静默忽略又太隐蔽 —— 所以 warn 一次，然后按「没给」处理。

    :param name: 预设名（``"quiet"`` / ``"normal"`` / ``"chatty"``）或 ``None``
    """
    global _host_verbosity

    if name is None:
        _host_verbosity = None
        return
    key = str(name).strip().lower()
    if not key:
        _host_verbosity = None
        return
    if key not in VERBOSITY_PRESETS:
        logger.warning(
            "不认识的话痨度预设：%r（可选 %s；忽略，继续用 Console 档位）",
            name, "/".join(sorted(VERBOSITY_PRESETS)))
        _host_verbosity = None
        return
    if key == DEFAULT_PRESET_NAME:
        # 清单默认值 =「我没管它」→ 交回给 Console 滑块（见常量注释）
        _host_verbosity = None
        return
    _host_verbosity = VERBOSITY_PRESETS[key]


def host_verbosity_preset() -> float | None:
    """宿主配置当前给的档位。**没给返回 None**。"""
    return _host_verbosity


def read_yui_verbosity(bridge: dict[str, Any] | None = None,
                      default: float | None = None) -> float:
    """播报档位（0.0~1.0）。

    ★ 三级优先（2026-10-29 落地 P0）
    --------------------------------
    1. **宿主配置** —— ``plugin.toml`` 的 ``[genshin_game_buddy]
       yui_verbosity_preset``，由插件读到后写进
       :func:`set_host_verbosity_preset`。
    2. **Console 滑块** —— 桥接状态里的 ``ui.yui_verbosity``。
    3. **默认值** —— :data:`DEFAULT_YUI_VERBOSITY`（1.0）。

    :param bridge: 已经读好的桥接状态。**不传就自己读一次** ——
        mtime 短路让重复读几乎免费。
    :param default: 读不到时给什么。**不传就用模块里的
        :data:`DEFAULT_YUI_VERBOSITY`**，且会 warn 一次。

    ★ 为什么读不到时**回落到默认值**而不是静音
      主控台没跑 / 文件正在被替换 / 版本不认得 —— 这三种情况下
      静默是最坏的选择：用户会以为「我把播报关了」，而实际上只是
      文件没读到。与其让人困惑，不如照常播。
    """
    # ① 宿主配置优先
    host = _host_verbosity
    if host is not None:
        return host
    # ② Console 滑块
    if bridge is None:
        bridge = read_bridge()
    if not isinstance(bridge, dict):
        return _fallback_verbosity(default)
    ui = bridge.get("ui")
    if not isinstance(ui, dict):
        return _fallback_verbosity(default)
    v = ui.get("yui_verbosity")
    if v is None:
        return _fallback_verbosity(default)
    try:
        f = float(v)
    except (TypeError, ValueError):
        logger.warning("播报档位不是数字：%r（忽略）", v)
        return _fallback_verbosity(default)
    # 越界夹回而不是报错 —— 拖滑块拖出 [0,1] 是 UI 层的问题，
    # 不该让播报整个哑掉。
    return min(1.0, max(0.0, f))


def _fallback_verbosity(default: float | None) -> float:
    v = DEFAULT_YUI_VERBOSITY if default is None else default
    if v != DEFAULT_YUI_VERBOSITY:
        logger.debug("读不到播报档位，用默认 %.2f", v)
    return v


#: 引擎快照文件名。与 ``autopilot/ui/protocol.py`` 的 :data:`SNAPSHOT_NAME`
#: 逐字一致 —— **刻意复制而不 import**：本模块的红线是「顶层不做任何
#: 事」（见模块 docstring），多一个跨模块 import 就多一条要走得通的
#: 路径，而这条常量的生命周期比本插件还长。
_SNAPSHOT_NAME = "state_snapshot.json"

#: 与 ``autopilot/ui/schema.py`` 的 :data:`LIVE` / :data:`DRY_RUN` 逐字一致。
_LIVE = "LIVE"
_DRY_RUN = "DRY_RUN"


def read_engine_run_mode(run_dir: Path | str | None = None) -> str:
    """引擎当前的执行模式（``"LIVE"`` / ``"DRY_RUN"``）。

    给 YUI 播报器当「干跑时闭嘴」的判据用（2026-10-19）。

    ★ 为什么要读快照而不是让事件流带这个字段
      事件流是**追加**写的，引擎改一次行为就得等下一个版本；
      快照 ``state_snapshot.json`` 是引擎每 200ms 覆盖写的，
      ``run_mode`` 字段从 2026-10-18 起就在里面（``ui/schema.py``）。
      读快照 = 不用等引擎重新发版就能知道它现在是不是在干跑。

    ★ 为什么读不到时返回 ``"DRY_RUN"``
      读不到的三种情况：引擎没跑 / 快照正在被替换 / 老版本引擎没有这个
      字段。三种都**按干跑处理** —— 宁可少播，不在用户不知情的状态下
      替他喊战斗。反过来默认 LIVE 的话，老引擎会被当成在真打，
      而它可能正在干跑 —— 那正是要治的病。

    :param run_dir: 运行目录。不传就用协议层的默认运行目录。
    :returns: ``"LIVE"`` 或 ``"DRY_RUN"``；**永远不会是 None**。
    """
    try:
        base = Path(run_dir) if run_dir is not None else _default_run_dir()
        snap = base / _SNAPSHOT_NAME
        mtime = snap.stat().st_mtime
        cached = _RUN_MODE_CACHE.get("mtime")
        if cached is not None and cached == mtime:
            return str(_RUN_MODE_CACHE["mode"])
        data = json.loads(snap.read_text(encoding="utf-8"))
        mode = str(data.get("run_mode") or _DRY_RUN).strip().upper()
        if mode != _LIVE:
            mode = _DRY_RUN         # 不认识的取值一律当干跑（宁可少播）
    except Exception:  # noqa: BLE001
        # ★ 读快照是**旁路**能力，不该拖垮播报线程。
        #   文件不存在是常态（引擎没跑），抛别的异常也是常态
        #   （正写到一半、编码损坏）—— 两种都退回收敛值。
        return _DRY_RUN
    _RUN_MODE_CACHE["mtime"] = mtime
    _RUN_MODE_CACHE["mode"] = mode
    return mode


#: 快照 mtime -> 模式。快照 200ms 覆盖写一次，不缓存就是每轮 poll
#: 都去 stat + read + json.loads。
_RUN_MODE_CACHE: dict[str, Any] = {"mtime": None, "mode": _DRY_RUN}


def _default_run_dir() -> Path:
    """引擎默认的运行目录。**惰性 import** —— 别把协议层拉进模块级。"""
    from .autopilot.ui import protocol  # noqa: PLC0415

    return protocol.run_dir()


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
    "DEFAULT_YUI_VERBOSITY",
    "PLUGIN_VERSION",
    "VALID_COMMANDS",
    "bridge_path",
    "commands_path",
    "configure",
    "console_version",
    "is_console_alive",
    "plugin_version",
    "read_bridge",
    "read_yui_verbosity",
    "send_command",
]
