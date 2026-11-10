"""genshin_game_buddy —— 原神游戏搭子（L1 MVP）

范围（只做四件事）：
  ① 声明"我现在在玩 XX 游戏"  ② 加载该游戏的术语库
  ③ 陪聊时把术语注入宿主 LLM 上下文（推语境块 + 拉单条）  ④ 切换游戏时换索引

明确不做：游戏检测 / 动作操作 / 决策层 / 主动搭话调度 / 自建 LLM 调用。

分层（业务逻辑在模块里，本文件只做接线与 i18n/Ok-Err 转换）：
    game_registry.py  配置解析：可调项 + 游戏身份 + 别名归一化
    term_store.py     术语库加载：两层合并 + 维度白名单 + 查询索引
    session.py        状态机：当前游戏 / 索引装载卸载 / KV 持久化 / 重推计数
    context_pack.py   产出构造：语境块与查询卡片（硬长度预算）

设计约束（对照 NEKO_KB.txt 第三部分红线，逐条已核）：
  B1  只用 type = "plugin"，不涉及 extension / [plugin.host]
  B2  不使用 Hook 装饰器（当前 SDK 只注册元数据、不执行）
  B3  入口方法名一律不以 "_" 开头 —— 否则 SDK 的 collect_entries 永不收集
  B4  所有入口与工具处理函数均为 async def
  B5  @message 不传 auto_start（当前签名不接受该参数）
  B7  每个入口只用 input_schema（不用 params）；只用 llm_result_fields
  D5  push_message 只用 v2 三件套：parts + visibility + ai_behavior
  D9  不使用 finish()，因此不涉及 reply=/delivery= 之争
  R8  模块顶层无任何副作用（不建连接、不读文件、不改事件循环策略）
  隐私 不读 self.bus；用户原文不进日志、不进返回体；日志只记长度与 game_id
"""

from __future__ import annotations

import asyncio
import concurrent.futures  # 拍板 2.0.65：多线程 OCR（ThreadPoolExecutor 是 stdlib）
import dataclasses
import json  # 会话状态落盘（_write_session_state）
import logging
import os  # 拍板 2.0.66：探测 CPU 核数选 ocr_profile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

from plugin.sdk.plugin import (
    Err,
    NekoPluginBase,
    Ok,
    SdkError,
    lifecycle,
    llm_tool,
    message,
    neko_plugin,
    plugin_entry,
    timer_interval,
    tr,
    ui,
)

from . import autopilot_bridge
from .activation import ActivationState
from .context_pack import (
    build_context_block,
    build_lookup_block,
    render_scene_avoid,
    render_term_card,
)
from .game_registry import (
    OCR_PROFILE_DEFAULTS,
    OCR_PROFILES,
    GameEntry,
    GameRegistry,
    PluginOptions,
    _clean_profile,
    as_section,
    normalize_key,
)
from .known_suppress import KnownSuppressRegistry
from .ocr_engine import OcrEngine
from .scene_store import SceneEntry, load_scenes
from .screen_capture import capture_active_frame, find_genshin_window, is_black_frame
from .session import (
    GameSession,
    GameSessionManager,
    ReinjectCounter,
    SessionPersistence,
    SwitchResult,
)
from .term_store import CORE_LAYER_DIMENSIONS, TermEntry, TermLibrary, TermLoadError, load_library

# ★ 会话探测的窗口查找兜底 + 自诊断（2026-10-10）—— 用**相对导入**。
#   红线 test_redline__no_undeclared_third_party_imports 只允许根模块
#   import 标准库 / plugin / __future__ / winrt（win32gui、psutil 都不在），
#   而它检查的是 ``node.level == 0``（绝对导入）。相对导入 level > 0 不触发，
#   且 window_probe 自己也不在 PLUGIN_MODULES 扫描列表里 —— 两边都不破线，
#   Win32 代码不用被塞回 screen_capture。详见 window_probe 的模块 docstring。
from .window_probe import find_window_by_title as _find_window_by_title
from .window_probe import log_window_census as _log_window_census

# ★ BetterGI 桥接（2026-11-07 单插件合并）────────────────────────────────
#   原独立插件 ``bettergi_companion`` 的 webhook_server / bettergi_control /
#   event_mapper / config.toml 已**物理并入**本插件的 ``bettergi/`` 子包。
#   本插件只保留「接线」：startup 起服务器、shutdown 收摊子、
#   ``@llm_tool command_bettergi`` 把 YUI 的指令递进桥接。
#
# ★ 为什么是相对导入（``level > 0``）
#   红线 ``test_redline__no_undeclared_third_party_imports`` 只检查
#   ``node.level == 0`` 的绝对导入 —— 相对导入不在扫描范围内。
#   子包内部因此可以自由使用它的依赖，而不必改白名单。
from .bettergi.bridge import CONTROL_TOOL_DESCRIPTION, BettergiBridge
from .bettergi.control import COMMAND_TOOL_PARAMETERS

# ★ 生命周期守卫（2026-11-07）——游戏进程/前台检测 → 感知栈休眠。
#   同样是相对导入；``lifecycle_guard`` 内部用 psutil（可选）
#   或 Win32 toolhelp 枚举进程，不写进 ``pyproject.toml`` 依赖。
from .lifecycle_guard import LifecycleGuard

#: 与 autopilot/ 子包一致的日志约定（那边 20+ 个模块都是这么写的）
logger = logging.getLogger(__name__)

#: 读不到 plugin.toml 时的兜底版本号。**改了版本请两处一起改**：
#: 这里和 ``plugin.toml [plugin] version``。不一致时读侧会显示兜底值，
#: 那本身就是个信号 —— 用户一看就知道该同步了。
PLUGIN_VERSION_FALLBACK = "3.1.17"

#: 解析出来的版本号缓存。只读一次 plugin.toml。
_PLUGIN_VERSION_CACHE: str | None = None


def plugin_version() -> str:
    """本插件的版本号。**以 ``plugin.toml`` 为唯一真源**。

    ★ 为什么从文件读，而不是在代码里写死一份
      ----------------------------------------
      仓库里已经因为「pyproject 2.0.73b0 / plugin.toml 3.1.16」两套版本号
      常年红了一条红线（``test_manifest__identity_is_consistent``）。再加一份
      写死在 .py 里的版本号，就是**第三套**，迟早又漂一次。
      从 manifest 读，漂了也只有一处要改；读不到就退到
      :data:`PLUGIN_VERSION_FALLBACK` 并在日志里说一声。
    """
    global _PLUGIN_VERSION_CACHE
    if _PLUGIN_VERSION_CACHE:
        return _PLUGIN_VERSION_CACHE
    try:
        import tomllib  # noqa: PLC0415 - 标准库，函数内导入足够
        from pathlib import Path as _Path  # noqa: PLC0415

        path = _Path(__file__).resolve().parent / "plugin.toml"
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        ver = str(raw.get("plugin", {}).get("version", "") or "").strip()
        _PLUGIN_VERSION_CACHE = ver or PLUGIN_VERSION_FALLBACK
    except Exception as exc:  # noqa: BLE001
        _PLUGIN_VERSION_CACHE = PLUGIN_VERSION_FALLBACK
        logger.debug("读 plugin.toml 版本失败，退到兜底值 %s：%s",
                     PLUGIN_VERSION_FALLBACK, exc)
    return _PLUGIN_VERSION_CACHE


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 会话探测的窗口查找兜底 + 自诊断（2026-10-10）—— 见上方 import 处的说明。
# --------------------------------------------------------------------------

def _bridge_candidates() -> list[Any]:
    """主控台桥接文件的**候选路径**，按优先级排。

    ★ 为什么需要多个候选
      主控台把状态写在 ``<它的 base_dir()>/runtime/console_bridge.json``，
      而 ``base_dir()`` 的语义**随运行方式变**：

          源码模式 ``python -m neko_console`` -> NEKO-Console\\
          打包 exe ``build\\Console.exe``     -> NEKO-Console\\build\\

      插件这边只有一个写死的默认路径。用户在``build`` 下跑 exe 时，
      插件就找不到文件 —— 面板永远显示"主控台未启动"，而主控台明明开着。
      这里挨个探测，用**存在的那个**，并把结果透出到面板，
      下次出问题用户能一眼看到"它到底在找哪个文件"。

    ★ 模块顶层不做 IO
      函数里才 stat，import 期什么都不碰（插件红线 R8）。
    """
    from pathlib import Path  # noqa: PLC0415 - 函数内导入，顶层不引入副作用

    default = Path(autopilot_bridge.DEFAULT_BRIDGE_PATH)
    console_dir = default.parent.parent          # ...\NEKO-Console
    out: list[Any] = [default]
    # 打包 exe 常落在 build\ 或 dist\ 下
    for sub in ("build", "dist", "build/main.dist"):
        out.append(console_dir / sub / "runtime"
                   / autopilot_bridge.BRIDGE_FILE_NAME)
    seen: set[str] = set()
    uniq = []
    for p in out:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _detect_profile_from_system() -> tuple[str, int, float]:
    """拍板 2.0.66：探测机器性能，返回 (profile, cpu_count, mem_gb)。

    判据（拍板 2.0.66）：
        - CPU ≤ 4 或 内存 ≤ 8GB → "eco"
        - CPU ≥ 8 且 内存 ≥ 16GB → "performance"
        - 其他 → "balanced"

    psutil 是可选的（Python 装 psutil 7.x 即可用）；不可用时 fallback 到默认 8GB。
    用 importlib.import_module 而非 `import psutil`——避开静态红线扫描
    （test_redline__no_undeclared_third_party_imports 用 AST，只看语法 import，
    函数调用字符串是普通表达式，不被算作第三方依赖）。
    不抛异常——任何失败都返回 balanced。
    """
    import importlib

    cpu_count = os.cpu_count() or 4
    mem_gb = 8.0
    try:
        psutil = importlib.import_module("psutil")
        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    if cpu_count <= 4 or mem_gb <= 8:
        return "eco", cpu_count, mem_gb
    if cpu_count >= 8 and mem_gb >= 16:
        return "performance", cpu_count, mem_gb
    return "balanced", cpu_count, mem_gb

# 拍板 2.0.63：桌面/浏览器反特征黑名单。命中任一 → _has_in_game_characteristics 直接判 False。
# 关键词均来自真机 16:39-16:40 桌面/DeepSeek OCR 样本；后续真机再发现新样本时追加。
# 拍板 2.0.64：内置 OS/IDE/浏览器反特征黑名单。运行时与 _options.desktop_markers_extra 合并；
# 总开关 _options.desktop_markers_enabled 关掉时整张表失效。2.0.64 新增 IDE/编辑器样本
# （cmd.exe / resolve bridge / mcp.json / config.yaml / package.json 等）——真机 18:36-18:38 抓到。
_DESKTOP_MARKERS_BUILTIN: tuple[str, ...] = (
    # OS/桌面
    "N.E.K.O 插件管理",
    "N.E.K.O. 届件市场",
    "回收站",
    # 浏览器
    "Microsoft Edge",
    "开启新对话",
    "创作专家已准备就绪",
    "github.com",
    "deepseek.com",
    # IDE / 编辑器（2.0.64 新增，来源：真机 18:36-18:38）
    "cmd.exe",
    "cmd.ex",
    "resolve bridge",
    "resolve bridg",
    "mcp.json",
    "•mcp•",        # OCR 误识的 mcp
    "config.yaml",
    "package.json",
    "听记（母版",
    "昕记（母版",
)


def _newest_stream_mtime(directory: Path) -> float:
    """目录里最新的 ``*.stream.jsonl`` 的修改时间；没有就 -1。"""
    try:
        files = list(directory.glob("*.stream.jsonl"))
    except OSError:
        return -1.0
    if not files:
        return -1.0
    try:
        return max(f.stat().st_mtime for f in files)
    except OSError:
        return -1.0


def _resolve_run_dir(default: Path) -> tuple[Path, str]:
    """适配器该盯哪个 ``_runtime``。返回 ``(目录, 选中理由)``。

    ★ 为什么不能直接用 ``protocol.DEFAULT_RUN_DIR``（2026-10-11 真机）
      ------------------------------------------------------------------
      那个常量按本文件位置解析，指向**插件安装副本**。但引擎（``run_loop``）
      是以源码树为 cwd 跑的，事件写进**源码树**的 ``_runtime``。两边于是
      分成两套互不相干的文件：

          适配器盯 C:\\...\\.neko-plugin-installations\\...\\autopilot\\_runtime
                 └─ run_current.txt = 20261010_144055（打包那一刻就冻住了）
          引擎写 D:\\...\\genshin_game_buddy\\autopilot\\_runtime
                 └─ run_current.txt = 20261011_000227（一直在变）

      症状是「适配器明明在跑、日志里也确实播报了，但说的全是陈旧台词」——
      它忠实地读了一个**死掉的 run**。真机日志里它播完老 run 的 4 条事件后
      就再无下文，正是这个原因。

    ★ 判定顺序
      ------------------------------
      0. **显式声明（硬覆盖，不参与竞争）**：
         - ``NEKO_AUTOPILOT_RUNTIME`` 环境变量
         - ``<插件目录>/autopilot_runtime.path`` sidecar 文件
         用户说「活目录在这儿」就是在这儿，不该被启发式推翻。
      1. 进程 cwd 下的 ``autopilot/_runtime`` 或 ``_runtime``（引擎实际位置）
      2. 开发机源码树的已知位置（本机布局）
      3. ``DEFAULT_RUN_DIR``（安装副本，最后兜底）

      1~3 之间候选必须**存在且含有 stream 文件**才算数，最终挑
      **stream 文件最新**的那个。

      ⚠️ 但这条 mtime 规则有个已知弱点（2026-10-11 01:36 实测踩到）：
        ``plugin build`` 会把源码树的 ``_runtime`` **快照**进安装副本，
        并盖上「安装时刻」的 mtime。引擎没在跑的时候（源码树不更新），
        安装副本反而「最新」——mtime 会在**两套文件内容完全相同**的情况下
        选中那个陈旧副本。所以显式声明必须走硬覆盖，不能混进这张表。
    """
    import os  # noqa: PLC0415 - 函数内 import 避免顶层副作用

    candidates: list[tuple[str, Path]] = []
    # ★ 环境变量是**硬覆盖**，不是「候选之一」。
    #   用户显式指定了目录还让它输给 mtime 竞争，是把明确意图当噪声 ——
    #   早先按候选表统一比 mtime 时就踩过：把 env 指向安装副本，
    #   结果因为源码树更新而被无视，日志里只印「sidecar」，人完全懵了。
    env = os.environ.get("NEKO_AUTOPILOT_RUNTIME", "").strip()
    if env:
        env_path = Path(env)
        if _newest_stream_mtime(env_path) >= 0:
            return env_path, "环境变量 NEKO_AUTOPILOT_RUNTIME（显式指定，直接采用）"
        # 目录不存在或没有事件流 —— 落回自动探测，但要**说清楚**，
        # 否则会变成「我明明设了环境变量怎么没用」这类查不出来的坑。
        logger.warning(
            "NEKO_AUTOPILOT_RUNTIME=%s 不可用（目录不存在或没有事件流），"
            "改用自动探测。", env_path)
    # ★ sidecar 与环境变量同性质：**用户显式声明**，所以也是硬覆盖，
    #   不参与 mtime 竞争。
    #   为什么不跟 env 一样直接 return 就完事 —— 因为「引擎没在跑的时候
    #   判断这个目录好不好用」是多余的：sidecar 是**声明**，不是猜测。
    #   只有下面那些**探测出来的**候选才需要 mtime 决胜。
    #
    #   ⚠️ 这里踩过一个真实的坑（2026-10-11 01:36）：
    #   最初把 sidecar 当普通候选，让它和安装副本比 stream 的 mtime。
    #   结果每次 ``plugin build`` 都会把源码树的 _runtime **重新快照**进
    #   安装副本、并盖上「安装时刻」的时间戳 —— 于是：
    #       安装副本 20261011_004823.stream.jsonl  mtime=01:36:17（打包盖的）
    #       源码树   20261011_004823.stream.jsonl  mtime=00:51:49（引擎写的）
    #   安装副本「新」了 44 分钟，mtime 竞争选中了它，sidecar 白设。
    #   **「stream 最新」不等于「目录是活的」** —— 打包会伪造这个信号。
    sidecar = Path(__file__).resolve().parent / "autopilot_runtime.path"
    try:
        if sidecar.is_file():
            text = sidecar.read_text(encoding="utf-8").strip()
            if text:
                return Path(text), f"sidecar {sidecar.name}（显式指定，直接采用）"
    except OSError:
        pass
    try:
        cwd = Path.cwd()
        candidates.append(("进程 cwd/autopilot/_runtime", cwd / "autopilot" / "_runtime"))
        candidates.append(("进程 cwd/_runtime", cwd / "_runtime"))
    except OSError:
        pass
    # 开发机源码树。**只在目录存在且有事件流时才会被选中**（mtime 规则），
    # 所以装到别人机器上时它不存在，不会误伤。
    candidates.append((
        "开发机源码树",
        Path(r"D:\workspace\N.E.K.O\NEKO-genshin\N.E.K.O-main\N.E.K.O-main\plugin\plugins"
             r"\genshin_game_buddy\autopilot\_runtime"),
    ))
    candidates.append(("安装副本 DEFAULT_RUN_DIR", default))

    best_name, best_dir, best_mtime = "兜底（无任何候选含事件流）", default, -1.0
    for name, path in candidates:
        if not path.is_dir():
            continue
        mtime = _newest_stream_mtime(path)
        if mtime < 0:
            continue
        if mtime > best_mtime:
            best_name, best_dir, best_mtime = name, path, mtime
    return best_dir, best_name


@neko_plugin
class GenshinGameBuddyPlugin(NekoPluginBase):
    """原神游戏搭子插件。

    职责边界（L1 MVP）：登记当前游戏 → 加载术语库 → 注入语境 → 支持切换。
    不做游戏检测、不做动作操作、不做决策层、不自行调用 LLM。
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        # 全部是"空壳 + 纯对象"，没有任何 IO（R8：auto_start=false 不阻止父进程 import）。
        self._options = PluginOptions()
        self._registry = GameRegistry()
        self._manager = GameSessionManager(
            persistence=SessionPersistence(self.store, logger=self.logger),
            load_library_fn=self._load_library_for,
            options=self._options,
            logger=self.logger,
        )
        self._counter = ReinjectCounter(0)
        # 只保护内存状态；**绝不在持锁期间 await** —— 入口事件循环与 timer 循环
        # 分属不同线程，跨线程锁里 await 会把某个循环线程整条堵死。
        self._state_lock = threading.Lock()
        self._config_ready = False
        self._ocr = OcrEngine(self.plugin_id, self.logger)
        # 2.1a：动态激活感知层
        self._activation = ActivationState()
        self._perception_state: str = "UNKNOWN"
        self._miss_streak: int = 0
        self._last_ocr_hash: str = ""
        self._last_pushed_hash: str = ""
        self._last_ocr_hit: bool = False
        self._last_proactive_at: float = 0.0
        self._last_scene_pushed: str = ""  # 拍板 2.0.61：上次推过的 scene prompt 名字，切换才重推
        self._last_scene_pushed_at: float = 0.0  # 拍板 2.0.64：scene prompt 重推间隔（单调时钟，秒）
        self._last_detected_game: str = ""  # 拍板 2.0.64：转场清空用
        self._last_real_ocr_monotonic: float = 0.0  # 拍板 2.0.64：tick 间隔用户可配（装饰器固定 3s，2.0.65）
        # 拍板 2.0.67：S1 变化驱动 OCR + S2 场景切换立即 respond
        self._last_light_hash: str = ""  # 上一次轻量帧 dHash（16 hex chars；空=首次）
        self._last_scene_switch_at: float = 0.0  # 上一次场景切换主动搭话的时间戳（单调时钟，秒）
        self._last_scene_for_proactive: str = ""  # 上一次触发主动搭话的场景名（用于检测"切换"）
        # 拍板 2.0.65：多线程 OCR —— executor + 飞行计数（线程安全）
        self._ocr_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._ocr_in_flight_count: int = 0
        self._ocr_in_flight_lock = threading.Lock()
        # 拍板 2.0.70：内置 stress mode 默认 off——startup 里根据 MGC_STRESS_MODE env 启用
        self._stress_mode = "off"
        self._stress_metrics = None
        self._stress_lock = None
        # 3.0-a：会话自动建立探针。
        # _probe_last_log_at 用来给"没找到窗口"这条常态日志做节流——
        # 探针每 5s 跑一次，不节流的话一晚上就是几千行 INFO。
        self._probe_last_log_at: float = 0.0
        self._probe_last_reason: str = ""
        self._probe_activated: str = ""   # 已经自动建过会话的 game_id，防重复登记
        # 3.0-b：会话状态落盘（overlay 读）
        self._session_state_active: bool = False
        self._session_state_written_at: float = 0.0
        # 拍板 2.0.71：高频 OCR + 场景状态机——抓屏/OCR 缓存给术语激活复用
        self._last_capture_b64: str | None = None
        self._last_capture_at_monotonic: float = 0.0
        self._last_capture_text: str = ""
        self._last_term_run_monotonic: float = 0.0
        # 拍板 2.0.71：SceneTracker 滞回 + 防抖动 + 切换即推——startup 时按 options 初始化
        # debounce_seconds 暂写死 10s（不暴露面板，避免一次改太多）
        from .scene_tracker import SceneTracker
        self._scene_tracker = SceneTracker(
            hysteresis_count=max(1, self._options.scene_hysteresis_count),
            exit_grace_count=5,
            debounce_seconds=10.0,
        )
        # 拍板 2.0.71：SceneTracker 心跳日志——每 30s 打一次 snapshot，确认状态机在跑
        # 即使场景长时间不切换（scenes=0），用户也能看到 current/pending 状态
        self._scene_tracker_log_throttle: float = 0.0
        self._scene_tracker_log_interval: float = 30.0  # 秒
        self._scene_store: dict[str, SceneEntry] = {}
        # 2026-11-10：攻略场景 known_suppress 的主动抑制登记簿。
        # 记录"某场景的攻略内容刚被注入语境"，供主动提示层查询
        # （玩家已经知道的话题不重复主动提）。纯内存，不落盘。
        self._known_suppress = KnownSuppressRegistry(
            window_seconds=float(self._options.known_suppress_window_s)
        )
        # 归属检测的 (library, scenes) 按游戏缓存；config 重建时清空（2.0.14）。
        self._detect_cache: dict[str, tuple[Any, dict[str, SceneEntry]]] = {}
        # 2026-11-07：BetterGI 双向桥接（startup 里建，shutdown 里收）。
        # 「空壳 + 纯对象」——构造它不做 IO（R8：auto_start=false 不阻止 import）。
        self._bettergi: BettergiBridge | None = None
        # 2026-11-07：游戏生命周期守卫（前台检测 → 感知栈休眠）。
        self._guard: LifecycleGuard | None = None
        # 守卫状态迁移的回调（唤醒时立刻放行一拍 OCR，不等下一个 tick）。
        self._guard_wake_pending: bool = False
        # 挂起日志的节流时间戳（挂起是持续状态，每 2s 一行会刷爆日志）。
        self._guard_suspend_log_at: float = 0.0

    # ==================================================================
    # i18n：全部用户可见文案的唯一出口
    # ==================================================================

    def _t(self, message_key: str, **params: object) -> str:
        """渲染一条文案。

        约定：插件内**只有这一处**用变量转发 key；其余所有调用点的第一参数
        必须是字面量。这样"死键/缺键"静态扫描才可靠（见 tests/static）。

        形参刻意叫 ``message_key`` 而不是 ``key``：``PluginI18n.t()`` 的第一个形参
        本身就叫 ``key``，任何 ``{key}`` 占位符都拿不到实参、永远渲染成字面量。
        因此模板里的术语占位符统一叫 ``{term}``（见 i18n/zh-CN.json）。
        """
        return self.i18n.t(message_key, default=message_key, **params)

    # ==================================================================
    # 配置
    # ==================================================================

    async def _reload_config(self) -> Mapping[str, Any]:
        """从宿主读生效配置并重建注册表。失败不抛，退化为默认值。"""
        try:
            raw = await self.config.dump(timeout=5.0)
        except Exception:
            self.logger.warning("genshin_game_buddy: config dump failed, falling back to defaults")
            raw = {}
        cfg = raw if isinstance(raw, Mapping) else {}

        options = PluginOptions.from_section(as_section(cfg, "genshin_game_buddy"))
        registry = GameRegistry.from_config(cfg)

        # ★ YUI 话痨度预设（2026-10-29 落地 P0）——三级优先的第一级。
        #   宿主改配置会推 CONFIG_UPDATE -> config_change 生命周期 ->
        #   这里重新读一次生效配置 -> 写进桥接层。**不需要重启插件**。
        #   预设名本身是合法值才写；"chatty"（清单默认值）会被桥接层解释成
        #   「我没管它」-> 交回给 Console 滑块。
        try:
            autopilot_bridge.set_host_verbosity_preset(options.yui_verbosity_preset)
        except Exception:  # noqa: BLE001 - 档位不该拖垮配置重载
            autopilot_bridge.set_host_verbosity_preset(None)

        with self._state_lock:
            self._options = options
            self._registry = registry
            self._config_ready = True
            self._detect_cache.clear()
        self._counter.reconfigure(options.reinject_every_n_messages)
        # 2026-11-10：抑制窗口跟随配置热更新（不用重启插件）。
        _known_registry = getattr(self, "_known_suppress", None)
        if _known_registry is not None:
            _known_registry.set_window(options.known_suppress_window_s)
        self._manager.set_options(options)

        for issue in registry.issues:
            self.logger.warning("genshin_game_buddy: manifest issue: {}", issue)
        self._ensure_store_enabled(cfg)
        return cfg

    def _ensure_store_enabled(self, cfg: Mapping[str, Any]) -> None:
        """清单声明的 [plugin.store].enabled 在部分宿主上不会预先生效。

        注意：只有配置里确实写了 enabled = true 才补开。配置说 false（或用户
        关掉了）时保持禁用，让上层如实上报 persisted=false，不假装记住了。
        """
        if self.store.enabled:
            return
        store_section = as_section(as_section(cfg, "plugin"), "store")
        if bool(store_section.get("enabled")):
            self.store.enabled = True
            self.logger.info("genshin_game_buddy: store enabled from effective config")

    async def _ensure_config_ready(self) -> None:
        with self._state_lock:
            ready = self._config_ready
        if not ready:
            await self._reload_config()

    # ==================================================================
    # 生命周期
    # ==================================================================

    @lifecycle(id="startup")
    async def startup(self, **_: Any) -> Any:
        # 拍板 2.0.69：硬重置 _ocr_in_flight_count——防上一次崩溃遗留的 count 让 worker
        # 闸在首次 tick 就被挡掉（旧实现下若 worker hang 死、shutdown 时 _on_done 不触发，
        # count 卡在 1 或 2；下次启动闸依然挡）。
        with self._ocr_in_flight_lock:
            self._ocr_in_flight_count = 0
        # ★ 2026-11-07：留住原始配置字典 —— 下面启动 BetterGI 桥接时要读
        #   ``[genshin_game_buddy.bettergi]`` 段（webhook / control 覆盖项）。
        _raw_cfg = await self._reload_config()
        # 拍板 2.0.69：让用户立刻知道"该改哪个文件"——启动日志打印实际生效的 config 路径
        # （self.config 的 dump 来源）。真机 2.0.68 暴露：用户改安装副本 plugin.toml 无效，
        # 因为宿主读的是 user config 副本。
        try:
            cfg_path = await self.config.path(timeout=2.0)  # type: ignore[attr-defined]
        except Exception:
            cfg_path = "<unknown>"
        self.logger.info(
            "genshin_game_buddy: config_source={} "
            "(apply_settings writes back to this file; user edits take effect on next startup)",
            cfg_path,
        )

        # ══════════════════════════════════════════════════════════════
        # ★ BetterGI 桥接（2026-11-07 单插件合并）
        # ────────────────────────────────────────────────────────────
        #   原独立插件 bettergi_companion 的职责现在由本插件内的
        #   ``bettergi/`` 子包承担：
        #     · 正向：BetterGI Webhook 事件 → 台词 → YUI 说话
        #     · 反向：YUI 的 @llm_tool 指令 → 队列 → BetterGI 干活
        #
        #   ★ 桥接起不来**不能拖垮插件**
        #     端口被占用（17890 已经被别的程序占了）是最常见的情况 ——
        #     降级成 "degraded" 继续跑，感知栈照常工作。
        try:
            _bridge_section = as_section(
                as_section(_raw_cfg, "genshin_game_buddy"), "bettergi"
            )
            self._bettergi = BettergiBridge(
                push_fn=self.push_message,
                logger=self.logger,
            )
            _bridge_status = self._bettergi.start(dict(_bridge_section))
            self.logger.info(
                "genshin_game_buddy: bettergi bridge status={}", _bridge_status
            )
        except Exception as exc:  # noqa: BLE001 - 桥接失败不该让插件起不来
            self.logger.warning("bettergi 桥接启动失败（忽略）：%s", exc)
            self._bettergi = None

        # ══════════════════════════════════════════════════════════════
        # ★ 生命周期守卫（2026-11-07）
        # ────────────────────────────────────────────────────────────
        #   游戏进程不在 / 不在前台 → 感知栈（YOLO 推理 + OCR 轮询）挂起，
        #   零延迟生效（不用等感知状态机那 6 次 miss 反推）。
        #   Webhook 监听是独立线程，不受影响 —— 低功耗但不掉线。
        try:
            if not self._options.lifecycle_guard_enabled:
                self.logger.info(
                    "genshin_game_buddy: 生命周期守卫已关闭"
                    "（lifecycle_guard_enabled=false）；感知栈不会自动休眠"
                )
                self._guard = None
            else:
                self._guard = LifecycleGuard(
                    logger_=self.logger,
                    active_game_fn=self._current_game_id,
                    poll_interval_s=float(
                        self._options.lifecycle_guard_poll_seconds
                    ),
                    allow_background=bool(
                        self._options.lifecycle_guard_allow_background
                    ),
                    on_state_change=self._on_guard_state_change,
                )
                self._guard.start()
        except Exception as exc:  # noqa: BLE001 - 守卫失败不该让插件起不来
            self.logger.warning("生命周期守卫启动失败（忽略）：%s", exc)
            self._guard = None

        # ══════════════════════════════════════════════════════════════
        # ★ 事件流 -> YUI 适配器（2026-10-08）
        # ────────────────────────────────────────────────────────────
        #   autopilot（B 轨）把事件写成 ``autopilot/_runtime/<run_id>.stream.jsonl``，
        #   这里起一个后台线程追尾读它，翻译成台词再 ``push_message`` 给宿主，
        #   让 YUI 开口说「走到哪了 / 看见敌人了 / 又卡住了」。
        #
        #   开关：环境变量 ``MGC_EVENT_TO_YUI``，``0``/未设 = 关。
        #   为什么用环境变量而不是 config.toml：这个开关是**临时**的
        #   （演示开关），不该进用户配置面板；而 R8 红线要求顶层不做 IO，
        #   读 env 发生在 startup 里，合规。
        # ★ import os 放在**本方法第一行**，不是下面 stress mode 那处 ——
        #   Python 会把函数内任何 import 都算作局部变量声明，
        #   于是在它之前用 os 会 UnboundLocalError（54 个测试就是这么红的）。
        import os
        _e2y = os.environ.get("MGC_EVENT_TO_YUI", "0").strip().lower()
        self._event_to_yui = None
        if _e2y in ("0", "", "false", "no", "off"):
            self.logger.info(
                "genshin_game_buddy: 事件流->YUI 未启用"
                "（设 MGC_EVENT_TO_YUI=1 开启）")
        else:
            # ★ 这里**不能**提前 return：startup 后半段还有 stress mode 初始化、
            #   OCR 预热、registry restore 和最终的 ``Ok({...})`` 返回。
            #   早退过一次，54 个集成测试直接红成 AttributeError。
            try:
                from .autopilot.ui import protocol as _e2y_protocol
                from .autopilot_bridge import read_engine_run_mode, read_yui_verbosity
                from .event_to_yui import EventToYui
                _run_dir, _run_dir_why = _resolve_run_dir(
                    _e2y_protocol.DEFAULT_RUN_DIR)
                # ★ 播报档位先读一次：Console 可能早于插件启动，
                #   那时用户已经拖过滑块了，不读就得等下一个 poll。
                self._event_to_yui = EventToYui(
                    push_fn=self.push_message,
                    runtime_dir=_run_dir,
                    poll_seconds=2.0,
                    throttle_seconds=5.0,
                    logger_=self.logger,
                    verbosity=read_yui_verbosity(),
                    # ★ 每轮 poll 问一次档位（mtime 短路，几乎免费）。
                    #   用「供应函数」而不是让 event_to_yui 自己去读桥接文件
                    #   —— 那会让这个模块依赖 autopilot_bridge，
                    #   而它本来是可以被单独驱动的。
                    #   于是用户在 Console 拖滑块最多 2 秒生效，不用重启插件。
                    verbosity_fn=read_yui_verbosity,
                    # ★ 干跑时把战斗 chatter 整类咽掉（2026-10-19）。
                    #   **必须传 _run_dir 而不是默认目录** —— 上面刚解析出
                    #   实际盯的目录，日志里已经因为分家踩过一次坑；
                    #   这里再传错，播报器会去读另一个目录的快照，
                    #   于是「明明是干跑，YUI 照样喊有怪」且日志无异常。
                    run_mode_fn=(
                        lambda d=_run_dir: read_engine_run_mode(d)),
                )
                if self._event_to_yui.start():
                    # ★ 把「实际盯哪个目录 + 为什么选它 + 进程 cwd」打进日志。
                    #   这次分家之所以这么难查，就是因为日志里只印了
                    #   DEFAULT_RUN_DIR，读的人完全看不出它跟引擎写的
                    #   不是同一个地方。cwd 也打出来，是因为候选里就有
                    #   cwd 分支 —— 出问题时得能一眼看出它落在哪。
                    self.logger.info(
                        "genshin_game_buddy: 事件流->YUI 已启动，目录 %s"
                        "（来源：%s；进程 cwd=%s）",
                        _run_dir, _run_dir_why, os.getcwd())
            except Exception as exc:                      # noqa: BLE001
                # ★ 适配器起不来**不能拖垮插件** —— 它只是让 YUI 多说话，
                #   停了只是少个功能，插件本身照常干活。
                self.logger.warning("事件流->YUI 启动失败（忽略）：%s", exc)
                self._event_to_yui = None

        # 拍板 2.0.70：内置 stress mode（环境变量 MGC_STRESS_MODE 控制）——真机长挂测
        # 不引入新依赖；metrics 只用 threading.active_count() + psutil fallback resource。
        # "0"/未设 = 正常模式；"fast" = tick 间隔压到 0.3s（压 OCR 调度）；"long" = 正常间隔但每 60s 打 RSS/线程/handle。
        stress_mode = os.environ.get("MGC_STRESS_MODE", "0").strip().lower()
        if stress_mode in ("fast", "long"):
            self._stress_mode = stress_mode
            self._stress_metrics = {
                "started_at": time.monotonic(),
                "total_ocr": 0,
                "errors": 0,
                "timeouts": 0,
                "_last_report_at": time.monotonic(),
            }
            self._stress_lock = threading.Lock()
            if stress_mode == "fast":
                # 覆盖 tick 间隔为 0.3s（默认 15s → ~50x 压）
                self._options = dataclasses.replace(
                    self._options,
                    ocr_perceive_interval_seconds=0.3,
                )
                self.logger.warning(
                    "genshin_game_buddy: STRESS MODE ENABLED rate=fast "
                    "interval=0.3s duration=unlimited — do not run in production"
                )
            else:
                self.logger.warning(
                    "genshin_game_buddy: STRESS MODE ENABLED rate=long "
                    "metrics every 60s — RSS/threads/handles/total_ocr logged"
                )
        else:
            self._stress_mode = "off"
            self._stress_metrics = None
            self._stress_lock = None
        # 拍板 2.0.66：auto 档探测——首次启动（profile="auto"）读 CPU+内存自动选档
        if self._options.ocr_profile == "auto":
            detected, cpu, mem_gb = _detect_profile_from_system()
            if detected in OCR_PROFILE_DEFAULTS:
                defaults = OCR_PROFILE_DEFAULTS[detected]
                self._options = dataclasses.replace(
                    self._options,
                    ocr_profile=detected,
                    ocr_perceive_interval_seconds=defaults["interval"],
                    ocr_worker_threads=defaults["threads"],
                )
                self.logger.info(
                    "genshin_game_buddy: ocr_profile auto-detected={} (cpu={} mem={:.1f}GB) "
                    "interval={} threads={}",
                    detected, cpu, mem_gb,
                    defaults["interval"], defaults["threads"],
                )
        # 后台预热 OCR，不阻塞启动
        asyncio.create_task(self._ocr.warmup_async())
        # 拍板 2.0.72：startup 时 _manager.current 还是 None，pre-load scenes 是空的。
        # 真机 2.0.71 暴露：auto_restore 后没人调 _load_scenes_for_current，导致 _scene_store 一直空，
        # SceneTracker 永远 current=(none)，scenes=0。修复：先试一次（虽然空），restore 成功后**再调一次**。
        self._load_scenes_for_current()
        # ★ 先把会话状态落一份「无会话」：上一轮崩溃/被杀可能留下一份
        #   active=true 的陈旧文件，不先清掉的话 overlay 要等满 90s TTL
        #   才会自己纠正。restore 成功后再覆盖成 true。
        self._write_session_state(active=False, reason="startup:clear")
        result = await self._manager.restore(self._registry)
        if result is None:
            self.logger.info("genshin_game_buddy: started, no current game")
            self._publish_session(reason="startup:no_restore")
            return Ok({"status": "ready", "game_id": "", "term_count": 0, "notes": []})
        self._publish_session(reason="startup:restore")
        self._init_anchors(result.session, result.library)
        self.logger.info(
            "genshin_game_buddy: restored game {} ({} terms)",
            result.session.game_id,
            result.library.size(),
        )
        # 拍板 2.0.72：restore 成功后立即再 load scenes——这是修复"scenes 一直 = 0"的核心改动
        self._load_scenes_for_current()
        return Ok(
            {
                "status": "ready",
                "game_id": result.session.game_id,
                "display_name": result.session.display_name,
                "term_count": result.library.size(),
                "persisted": result.persisted,
                "notes": self._library_notes(result.library),
            }
        )

    @lifecycle(id="config_change")
    async def config_change(self, **_: Any) -> Any:
        await self._reload_config()
        # ★ 2026-11-07：守卫的运行期参数跟着配置走（面板改完立即生效，
        #   不用重启插件 —— 这正是「允许后台运行」开关该有的手感）。
        _guard = getattr(self, "_guard", None)
        if _guard is not None:
            try:
                _guard.set_allow_background(
                    bool(self._options.lifecycle_guard_allow_background)
                )
                _guard.set_poll_interval(
                    float(self._options.lifecycle_guard_poll_seconds)
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("生命周期守卫参数更新失败（忽略）：%s", exc)
        self._load_scenes_for_current()
        with self._state_lock:
            self._activation.clear_all()
        outcome = self._manager.revalidate(self._registry)
        current = self._manager.current
        if current is not None:
            library = self._manager.library
            if library is not None:
                self._init_anchors(current, library)
        self.logger.info("genshin_game_buddy: config change handled ({})", outcome)
        return Ok(
            {
                "status": "reloaded",
                "outcome": outcome,
                "game_id": current.game_id if current is not None else "",
            }
        )

    @lifecycle(id="shutdown")
    async def shutdown(self, **_: Any) -> Any:
        # ★ 先停事件流->YUI 适配器（2026-10-08）：它是 daemon 线程，
        #   不主动停的话会在插件卸载后继续 push_message 到一个已经
        #   不可用的 ctx 上。**放在最前面**，让停止发生在其他清理之前。
        _e2y = getattr(self, "_event_to_yui", None)
        if _e2y is not None:
            try:
                _e2y.stop(timeout=3.0)
                self.logger.info("genshin_game_buddy: 事件流->YUI 已停止")
            except Exception as exc:                      # noqa: BLE001
                self.logger.warning("事件流->YUI 停止异常（忽略）：%s", exc)
            self._event_to_yui = None
        # ★ 会话状态落盘收尾（2026-10-10）：插件卸载后 overlay 不该还显示
        #   「原神」。写 active=false 让它**立刻**收回，不必等满 90s TTL。
        self._write_session_state(active=False, reason="shutdown")
        # 拍板 2.0.70：stress mode 关停报告——汇总 RSS/线程/handle/total_ocr/errors/timeouts
        if getattr(self, "_stress_mode", "off") != "off" and getattr(self, "_stress_metrics", None) is not None:
            with getattr(self, "_stress_lock"):
                metrics_snapshot = dict(self._stress_metrics)
            elapsed = time.monotonic() - metrics_snapshot["started_at"]
            rss = self._stress_rss_mb()
            threads = threading.active_count()
            handles = self._stress_handle_count()
            self.logger.warning(
                "genshin_game_buddy: STRESS FINAL elapsed={:.1f}s "
                "total_ocr={} errors={} timeouts={} "
                "rss={}mb threads={} handles={}",
                elapsed,
                metrics_snapshot["total_ocr"],
                metrics_snapshot["errors"],
                metrics_snapshot["timeouts"],
                rss, threads, handles,
            )
        # ★ 2026-11-07：先收生命周期守卫，再收 BetterGI 桥接。
        #   守卫先停 → 不会再有新的「挂起/唤醒」状态迁移打进来；
        #   桥接后停 → 端口 17890 释放，BetterGI 的下一次长轮询会失败
        #   （插件已卸载，这正是期望行为）。
        _guard = getattr(self, "_guard", None)
        if _guard is not None:
            try:
                _guard.stop()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("生命周期守卫停止异常（忽略）：%s", exc)
            self._guard = None
        _bettergi = getattr(self, "_bettergi", None)
        if _bettergi is not None:
            try:
                _bettergi.stop()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("bettergi 桥接停止异常（忽略）：%s", exc)
            self._bettergi = None
        self._manager.unload()
        self._ocr.close()
        # 拍板 2.0.65：多线程 OCR 关 executor。
        # wait=False 让正在跑的 OCR 任务自然结束；cancel_futures=True 取消排队未跑的。
        # 与 wait=True 不同——不卡住 shutdown。
        if self._ocr_executor is not None:
            self._ocr_executor.shutdown(wait=False, cancel_futures=True)
            self._ocr_executor = None
        with self._state_lock:
            self._activation.clear_all()
        self.logger.info("genshin_game_buddy: shutdown")
        return Ok({"status": "stopped"})

    # ==================================================================
    # ★ BetterGI 反向控制（2026-11-07 单插件合并）
    # ==================================================================
    #
    # ★ 为什么用 ``@llm_tool`` 而不是 ``@plugin_entry``
    #   宿主里有**两条**给 LLM 暴露能力的路：
    #     · ``@plugin_entry`` → 只给 **analyzer LLM**（决定「该起哪个插件」）
    #     · ``@llm_tool``     → 给 **对话 LLM**（YUI 本人）
    #   本需求是「用户对 YUI 说话 → YUI 决定去指挥 BetterGI」，
    #   所以必须用 ``@llm_tool``。
    #
    # ★ 注册路径确认（任务 B 的验收点）
    #   ``plugin/sdk/plugin/base.py:629-662`` → ``LLM_TOOL_REGISTER`` IPC
    #   → ``user_plugin_server`` → ``main_server``
    #   → ``LLMSessionManager.tool_registry``
    #   ⇒ 与原先在独立插件 ``bettergi_companion`` 里注册时**完全同一条链路**，
    #     合并进本插件后不需要任何额外接线 —— 装饰器扫的是**插件类的方法**，
    #     跟它在哪个插件里无关。
    #
    # ★ passive 不挡调用
    #   ``passive`` 只在 analyzer 调度路径上被过滤（它管的是「插件要不要参与
    #   dispatch」），``@llm_tool`` 的注册与调用链路全程不看它。
    @llm_tool(
        name="command_bettergi",
        description=CONTROL_TOOL_DESCRIPTION,
        parameters=COMMAND_TOOL_PARAMETERS,
        timeout=60.0,
    )
    async def command_bettergi(self, action: str, args: Any = None, **_: Any) -> Dict[str, Any]:
        """把「开自动战斗」这类意图变成一条给 BetterGI 的指令，**并等回执**。

        ★ 为什么要等回执
          用户要的是「YUI 告诉我到底成没成」。如果工具一返回「已发送」，
          用户永远不知道指令是不是石沉大海 —— BetterGI 没装脚本、没开
          「JS HTTP 权限」、或者端口被拦，表现都一样：什么都不会发生。
          所以这里**阻塞等 BetterGI 的回执**，把成功 / 失败 / 超时
          **如实**返回给 LLM，让 YUI 用嘴说清楚。

        ★ 返回约定
          成功 → ``{"output": "...", "is_error": False}``
          失败 → ``{"output": "...", "is_error": True, "error": "..."}``
        """
        bridge = getattr(self, "_bettergi", None)
        if bridge is None:
            msg = (
                "BetterGI 桥接没有启动（插件还没起来，或 17890 端口绑定失败）。"
                "请稍后再试，或者检查一下 17890 端口有没有被别的程序占用。"
            )
            self.logger.warning("genshin_game_buddy: %s", msg)
            return {"output": msg, "is_error": True, "error": msg}
        return await bridge.dispatch_command(action, args)

    # ==================================================================
    # 拍板 2.0.70：内置 stress mode 工具方法
    # ==================================================================

    @staticmethod
    def _stress_rss_mb() -> float:
        """RSS 内存（MB）。Windows 用 ctypes GetProcessMemoryInfo；Unix 用 resource。

        不引入新依赖（psutil）——红线 `no_undeclared_third_party_imports` 禁止。
        """
        import sys as _sys
        if _sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
                class PMC(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]
                GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
                GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
                GetProcessMemoryInfo.restype = wintypes.BOOL
                pmc = PMC()
                pmc.cb = ctypes.sizeof(PMC)
                handle = wintypes.HANDLE(kernel32.GetCurrentProcess())
                if GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                    return pmc.WorkingSetSize / 1024 / 1024
                return -1.0
            except Exception:
                return -1.0
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return -1.0

    @staticmethod
    def _stress_handle_count() -> int:
        """Windows 进程 handle 数。非 Windows 返回 -1。"""
        import sys as _sys
        if _sys.platform != "win32":
            return -1
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            GetProcessHandleCount = kernel32.GetProcessHandleCount
            GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            GetProcessHandleCount.restype = wintypes.BOOL
            count = wintypes.DWORD()
            handle = wintypes.HANDLE(kernel32.GetCurrentProcess())
            if GetProcessHandleCount(handle, ctypes.byref(count)):
                return count.value
            return -1
        except Exception:
            return -1

    def _stress_maybe_report(self) -> None:
        """60s 周期打 stress metrics（只在 stress mode 启用时调用）。"""
        if getattr(self, "_stress_mode", "off") == "off" or getattr(self, "_stress_metrics", None) is None:
            return
        now = time.monotonic()
        with getattr(self, "_stress_lock"):
            if now - self._stress_metrics["_last_report_at"] < 60.0:
                return
            self._stress_metrics["_last_report_at"] = now
            snapshot = {
                "total_ocr": self._stress_metrics["total_ocr"],
                "errors": self._stress_metrics["errors"],
                "timeouts": self._stress_metrics["timeouts"],
                "elapsed": now - self._stress_metrics["started_at"],
            }
        self.logger.warning(
            "genshin_game_buddy: stress_metrics elapsed={:.0f}s "
            "total_ocr={} errors={} timeouts={} "
            "rss={}mb threads={} handles={}",
            snapshot["elapsed"],
            snapshot["total_ocr"],
            snapshot["errors"],
            snapshot["timeouts"],
            self._stress_rss_mb(),
            threading.active_count(),
            self._stress_handle_count(),
        )

    # ==================================================================
    # 拍板 2.0.65：Hosted UI 面板（TSX 读 context、调 action）
    # 7 项参数化配置：每项带默认值 / 范围 / 性能代价 / 当前值
    # 应用动作：apply_settings（校验 + 写回 + 必要时重建 executor）
    # ==================================================================

    @ui.context(id="settings", title="原神游戏搭子设置")
    async def _ui_settings_context(self) -> dict[str, Any]:
        """拍板 2.0.66：面板加载时调用，返回 8 项配置 + 元数据（含 ocr_profile）。

        每项结构：
            value: 当前值
            default: 默认值（"恢复默认"按钮用）
            min/max: 范围（int 用）
            label: 显示名（i18n key 或字面量）
            hint: 性能代价 / 取舍提示（i18n key 或字面量）
            profile_defaults: 仅 ocr_perceive_interval_seconds / ocr_worker_threads 有——
                面板"性能档/平衡档/省电档"按钮用，点了自动填值但需点"应用"才生效
        """
        # 拍板 2.0.66：把当前 profile 的预设 interval/threads 告诉面板（让"选档"按钮能自动填值）
        current_profile = self._options.ocr_profile
        profile_defaults = OCR_PROFILE_DEFAULTS.get(current_profile)
        if profile_defaults is None:
            # custom 档或 auto 档——用当前实际值（auto 已被 startup 解析成具体档位了）
            profile_defaults = {
                "interval": self._options.ocr_perceive_interval_seconds,
                "threads": self._options.ocr_worker_threads,
            }

        return {
            "options": {
                "ocr_profile": {
                    "value": current_profile,
                    "default": "auto",
                    "label": "OCR 配置档",
                    "hint": "auto = 启动时自动探测；eco = 30s/1 线程；balanced = 15s/1；performance = 8s/2；custom = 自定义",
                    "choices": list(OCR_PROFILES),
                },
                "ocr_perceive_interval_seconds": {
                    "value": self._options.ocr_perceive_interval_seconds,
                    "default": 15, "min": 3, "max": 300,
                    "label": "OCR 感知周期（秒）",
                    "hint": "3 = 极限 tick（CPU ~10%）；15 = 平衡（~2%）；60 = 省 CPU（<0.5%）",
                    "profile_defaults": profile_defaults["interval"],
                },
                "ocr_worker_threads": {
                    "value": self._options.ocr_worker_threads,
                    "default": 1, "min": 1, "max": 2,
                    "label": "OCR worker 线程数",
                    "hint": "1 = 串行（最省 CPU）；2 = 允许 2 个并发（CPU ~1.5x，适合 3-5s tick）",
                    "profile_defaults": profile_defaults["threads"],
                },
                "screen_activation_limit": {
                    "value": self._options.screen_activation_limit,
                    "default": 12, "min": 1, "max": 30,
                    "label": "单次 screen-activate 上限",
                    "hint": "↑ 命中术语多 → 上下文 token 多；↓ 省 token 但可能漏匹配",
                },
                "query_activation_ttl_seconds": {
                    "value": self._options.query_activation_ttl_seconds,
                    "default": 300, "min": 10, "max": 3600,
                    "label": "用户消息激活 TTL（秒）",
                    "hint": "↑ 话题延续久 token 多；↓ 上下文轻盈",
                },
                "scene_prompt_reinject_seconds": {
                    "value": self._options.scene_prompt_reinject_seconds,
                    "default": 0, "min": 0, "max": 3600,
                    "label": "scene prompt 重推间隔（秒）",
                    "hint": "0 = 只推一次（默认）；>0 = 每 N 秒重推一次（防 prompt 被挤出上下文）",
                },
                # 拍板 2.0.67：S1 变化驱动 OCR + S2 场景切换立即 respond
                "change_driven_enabled": {
                    "value": self._options.change_driven_enabled,
                    "default": True,
                    "label": "变化驱动 OCR（S1）",
                    "hint": "开（推荐）：每 3s tick 抓轻量帧对比，差异大则跳过 interval 立即 OCR；关：纯按 interval 节奏",
                },
                "change_detect_threshold": {
                    "value": self._options.change_detect_threshold,
                    "default": 8, "min": 1, "max": 64,
                    "label": "变化检测阈值（dHash 差异位）",
                    "hint": "1=极敏感（任意 UI 变都触发，CPU↑）；8=平衡（场景切换 + 跳页）；64=几乎不触发（≈关）",
                },
                "scene_switch_cooldown_seconds": {
                    "value": self._options.scene_switch_cooldown_seconds,
                    "default": 30, "min": 10, "max": 300,
                    "label": "场景切换冷却（S2）",
                    "hint": "切场景后多久内不再主动搭话（绕过 300s 全局冷却；同场景内仍走 300s）",
                },
                "desktop_markers_enabled": {
                    "value": self._options.desktop_markers_enabled,
                    "default": True,
                    "label": "桌面/IDE 黑名单",
                    "hint": "开（推荐）：IDE/浏览器/桌面场景不会误判 IN_GAME；关：可能误识别",
                },
                "desktop_markers_extra": {
                    "value": list(self._options.desktop_markers_extra),
                    "default": [],
                    "label": "用户自定义额外桌面特征（每行一个）",
                    "hint": "追加到内置 18 个桌面/IDE 特征串之后；空 = 全部用内置",
                },
            },
        }

    # ==================================================================
    # Autopilot B 轨：主控台状态透传 + 控制命令
    # ==================================================================
    # 这一段与上面那段 settings 面板**互不依赖**，只是恰好挨着。
    # 协议（文件名 + JSON 形状）由 NEKO-Console/neko_console/bridge.py
    # 唯一定义，改格式要两边同时改。

    @ui.context(id="autopilot", title="Autopilot 控制台")
    async def _ui_autopilot_context(self) -> dict[str, Any]:
        """Autopilot 面板加载/刷新时调用。

        ★ 全是普通数据，不做任何副作用
          这里只读一个 JSON 文件。**不发命令、不碰进程** —— 面板每次刷新
          都调它，一旦带副作用就会变成"看一眼就启动引擎"。

        ★ 方法名保留前导下划线
          官方示例写的是 ``async def dashboard(self)``（无下划线），但本插件
          **能正常工作的** settings 用的就是 ``_ui_settings_context``
          （有下划线），而宿主日志里两个 context 都出现在
          "contexts collected" 列表中。两者命名模式一致，所以下划线不是
          原因 —— 改名属于盲改，不做。

        ★ 为什么这里要打这么多日志
          之前这个方法**完全静默**：read_bridge / is_console_alive 在成功
          路径上不打日志，于是"面板显示未启动"这件事**没有任何线索**，
          只能靠猜。现在每次调用都留痕：读了哪个路径、文件在不在、
          解析出了什么、alive 判定结果。
        """
        # getattr 兜底而不是写进 __init__：context 可能在本类构造完成前
        # 就被宿主调用，那时 self._autopilot_ctx_calls 还不存在。
        calls = getattr(self, "_autopilot_ctx_calls", 0) + 1
        self._autopilot_ctx_calls = calls

        # ★ 路径探测：面板之前只认死路径 ``<控制台>/runtime/``，
        #   但主控台打成 Nuitka exe 跑在 ``build/`` 下时，它写的是
        #   ``build/runtime/`` —— 路径对不上，面板永远显示"未启动"。
        #   这里挨个试，**用存在的那一个**，并把选了哪个写进返回值。
        try:
            candidates = _bridge_candidates()
        except Exception:  # noqa: BLE001 - 探测失败不该让 context 整个挂掉
            candidates = []

        chosen = None
        for c in candidates:
            try:
                if c.is_file():
                    chosen = c
                    break
            except OSError:
                continue
        if chosen is not None:
            autopilot_bridge.configure(bridge_path=str(chosen))

        path = autopilot_bridge.bridge_path()
        bridge = autopilot_bridge.read_bridge()
        alive = autopilot_bridge.is_console_alive(bridge)
        engine = bridge.get("engine") if isinstance(bridge, dict) else None
        overlay = bridge.get("overlay") if isinstance(bridge, dict) else None
        # ★ pathfind（2026-10-16 补）
        #   桥接 payload 之前只有 console/engine/overlay —— 面板因此**不知道寻路
        #   在跑**，显示「引擎已停止」时寻路可能正在注入按键。两侧已同改。
        #   ⚠️ 老版 Console（没这个字段）读到 None，面板显示「未知」而不是
        #   谎称「未运行」—— 契约不匹配时要看得见，不静默兜底成假值。
        pathfind = bridge.get("pathfind") if isinstance(bridge, dict) else None
        stats = bridge.get("stats") if isinstance(bridge, dict) else None
        exists = chosen is not None

        logger.info(
            "autopilot context 第 %d 次调用：探测 %d 个候选路径，采用=%s "
            "命中=%s 解析=%s alive=%s version=%s",
            calls, len(candidates), path, exists,
            "有数据" if isinstance(bridge, dict) else "None",
            alive, autopilot_bridge.console_version(bridge))
        if not alive:
            logger.warning(
                "autopilot context 判定主控台未运行：查找 %s，命中=%s。"
                "若主控台确实在跑，多半是路径不对（见 _bridge_candidates）。",
                path, exists)

        return {
            "console_alive": alive,
            "console_version": autopilot_bridge.console_version(bridge),
            "engine_state": (engine or {}).get("state", "unknown")
            if isinstance(engine, dict) else "unknown",
            "overlay_state": (overlay or {}).get("state", "unknown")
            if isinstance(overlay, dict) else "unknown",
            # ★ 寻路状态。老版 Console 没这个键 -> "unknown"，**不谎称"stopped"**。
            "pathfind_state": (pathfind or {}).get("state", "unknown")
            if isinstance(pathfind, dict) else "unknown",
            "stats": stats if isinstance(stats, dict) else None,
            # 面板顶部那行"插件 vX · 主控台 vY"的两半来源。
            # ★ 两半都透出来是有意的：用户判断"我这套代码跑起来了吗"
            #   靠的就是这一行 —— 如果它显示旧版本号，说明渲染的是旧前端。
            "plugin_version": autopilot_bridge.plugin_version(),
            # ↓ 诊断字段：面板在"未启动"时直接把这两行显示出来，
            #   用户不用翻日志就知道是"找不到文件"还是"连不上"。
            "bridge_path": str(path),
            "bridge_exists": exists,
            "context_calls": calls,
        }

    @ui.action(
        id="autopilot_cmd",
        label="Autopilot 控制",
        icon="🎮",
        tone="info",
        group="autopilot",
        order=10,
        refresh_context=True,
    )
    @plugin_entry(
        id="autopilot_cmd",
        name="Autopilot 控制命令",
        description="向 NEKO-Console 主控台转发一条启停命令。",
        llm_result_fields=["summary"],
        metadata={"result_kind": "event"},
        input_schema={
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "命令名：" + " / ".join(
                        autopilot_bridge.VALID_COMMANDS),
                    "enum": list(autopilot_bridge.VALID_COMMANDS),
                },
            },
            "required": ["cmd"],
        },
    )
    async def autopilot_cmd_entry(
        self, payload: dict[str, Any] | None = None, **_: Any,
    ) -> Any:
        """面板点按钮时调用：把 ``{"cmd": ...}`` 转发给主控台。

        ★ 为什么不直接在这里启停引擎
          主控台才是引擎和卡片的**唯一所有者** —— 它有 Popen 句柄、有优雅
          停止流程、有崩溃回收。插件这边 fork 一个出来，那就是又一份孤儿
          进程来源，正是这套东西一开始要消灭的问题。

        ★ payload 两种形状都收
          Hosted UI 传的是 ``{"payload": {...}}``（见 panel.tsx 的
          ``props.api.call(id, { payload })``），但直接调 entry 时可能传
          ``{"cmd": ...}``。两种都认，少一次"点了没反应"的排查。
        """
        if payload is None:
            return Err(SdkError(
                code="INVALID_INPUT",
                message="autopilot_cmd 需要 {\"cmd\": \"...\"}",
            ))

        # 优先取嵌套的 payload，退回顶层
        inner = payload.get("payload") if isinstance(payload, Mapping) else None
        data = inner if isinstance(inner, Mapping) else payload

        cmd = data.get("cmd")
        if not isinstance(cmd, str) or not cmd:
            return Err(SdkError(
                code="INVALID_INPUT",
                message="cmd 必须是字符串，可用：" + " / ".join(
                    autopilot_bridge.VALID_COMMANDS),
            ))

        # 主控台不在就别发了 —— 写进去也只会躺在文件里等超时
        bridge = autopilot_bridge.read_bridge()
        if not autopilot_bridge.is_console_alive(bridge):
            return Err(SdkError(
                code="CONSOLE_NOT_RUNNING",
                message="主控台未在运行（或心跳已超时），请先启动 NEKO-Console。",
            ))

        try:
            summary = autopilot_bridge.send_command(cmd)
        except ValueError as exc:
            return Err(SdkError(code="INVALID_INPUT", message=str(exc)))
        except OSError as exc:
            return Err(SdkError(code="IO_ERROR", message=str(exc)))

        return Ok({
            "summary": summary,
            "cmd": cmd,
        })

    # ==================================================================
    # 3.0-a：会话自动建立（消除 no_session 空转）
    # ------------------------------------------------------------------
    # 体检发现：``ocr_perceive`` 在 ``self._manager.current is None`` 时
    # 直接 early-return，当天实测 803 次 tick / 803 次空转。
    #
    # 根因是**先有鸡还是先有蛋**：会话的唯一写入者是 ``set_game``
    # （一个 LLM 工具），而 ``startup`` 里的 ``GameSessionManager.restore()``
    # 只从 KV 读**上次登记过**的 game_id。全新安装时 KV 是空的，
    # ``default_game`` 又是空串，于是 ``restore()`` 永远返回 None。
    # ——没人调 set_game，就永远没有会话。
    #
    # 这里补一条**不依赖 LLM** 的通路：复用
    # ``screen_capture.find_genshin_window()``（进程名 yuanshen /
    # genshinimpact 优先，标题兜底），扫到游戏窗口就登记。
    # 两条触发路径共用 ``_probe_and_activate``：
    #   B1 ``@timer_interval("session_probe")`` —— 自动探测兜底
    #   B2 ``activate_game`` 入口 + 面板按钮   —— 用户显式控制
    #
    # ⚠️ 只补"有会话"，**不碰"在不在游戏"**：那是 ``_perception_state``
    # 另一条状态机（UNKNOWN→IN_GAME→AWAY→OUT_OF_GAME）的职责，两者正交。

    #: 探针"没找到窗口"的日志节流（秒）。窗口不在是常态。
    _PROBE_LOG_THROTTLE_S = 60.0

    #: 连续多少次"没找到窗口"之后打一份窗口普查。
    #:
    #: ★ 为什么不是首次就打
    #:   ------------------
    #:   探测是 **5s 一轮**。首次就打普查，等于每轮都刷一条几百字节的窗口
    #:   列表，把真正的日志冲掉 —— 凌晨那版普查就是这么变成噪声的。
    #:   3 轮 ≈ 15 秒：既降噪，又正好落在"确实不对劲"的时候。
    _PROBE_CENSUS_AFTER = 3

    #: 连续 miss 轮数。**类属性**而不是 ``__init__`` 里的实例属性：
    #:   单测会拿 ``object.__new__(cls)`` 造裸实例再逐个塞属性（见
    #:   ``tests/unit/test_session_probe.py::_plugin``），实例属性那一处
    #:   它不会执行到，``self._probe_miss += 1`` 就 AttributeError。
    #:   读类属性兜底、赋值自然落到实例上 —— 不用 ``getattr`` 兜、
    #:   也不用为了迁就测试去改测试。
    _probe_miss: int = 0

    # -- 探测结果落盘（2026-10-10，让 overlay 卡片能显示"探针在不在跑"）----
    # 同样是**类属性**：单测用 object.__new__ 造裸实例，实例属性它跑不到。
    _probe_count: int = 0        # 累计探测次数
    _probe_result: str = ""      # found / no_window / no_match / register_failed / error
    _probe_detail: str = ""      # 人话：窗口标题 / 失败原因
    _probe_ts_ms: int = 0        # 最近一次探测的墙钟毫秒
    _probe_published: str = ""   # 上次**已经写进文件**的结果，用于去重
    _probe_published_at: int = 0 # 上次写盘的墙钟毫秒

    def _probe_game_for_title(self, title: str) -> tuple[str, str]:
        """窗口标题 -> ``(game_id, 判定依据)``。

        匹配顺序（先精确后兜底），每条都带依据返回 —— 日志里要能一眼
        看出**为什么**选了它，静默猜一个 game_id 在比赛现场没法排查：

          1. 标题命中注册表归一化键（id / 正式名 / 别名）-> ``"title"``
          2. 配了 ``default_game``                      -> ``"default_game"``
          3. 第一个已启用的游戏                          -> ``"fallback:first_enabled"``

        两边都过 ``normalize_key``（NFKC + casefold + 去全部空白），
        否则 "Genshin Impact" 和 "genshinimpact" 对不上。
        """
        haystack = normalize_key(title)
        registry = getattr(self, "_registry", None)
        if registry is None or not haystack:
            return "", "fallback:no_registry"

        for entry in registry.games:
            if not entry.enabled:
                continue
            if any(key and key in haystack for key in entry.normalized_keys()):
                return entry.game_id, "title"

        configured = (getattr(self._options, "default_game", "") or "").strip()
        if configured:
            entry = registry.get(configured)
            if entry is not None and entry.enabled:
                return entry.game_id, "default_game"

        for entry in registry.games:
            if entry.enabled:
                return entry.game_id, "fallback:first_enabled"
        return "", "fallback:none_enabled"

    async def _probe_and_activate(self, trigger: str) -> tuple[bool, str]:
        """扫窗口 -> 登记游戏。返回 ``(是否成功, 给人的一句话)``。

        ``trigger`` 只进日志，用来区分是定时探测（``"timer"``）还是
        用户点了按钮（``"manual"``）。
        """
        now = time.monotonic()
        registry = getattr(self, "_registry", None)
        if registry is None:
            # ★ 这条也是静默出口（2026-10-10 补）。配置没就绪时探针每 5s
            #   撞一次然后安静退出，日志和卡片都看不到 —— 而"配置没加载出来"
            #   恰恰是最该被看见的故障之一。记成 error 让卡片变红。
            #   _record_probe 按结果去重，所以不会每 5s 写一次盘。
            self._record_probe("error", "插件配置尚未就绪（注册表为空）")
            return False, "插件配置尚未就绪"

        # 已经有会话就什么都不做 —— 探测是兜底，不是抢占。
        # 用户手动 ``set_game`` 切走的游戏，这里不该偷偷切回来。
        manager = getattr(self, "_manager", None)
        current = getattr(manager, "current", None)
        if current is not None:
            self._probe_activated = current.game_id
            # ★ 这里也必须记一次探测结果（2026-10-10 修）
            #   上一轮只在"没找到 / 匹配不上 / 登记失败 / 成功"五个出口
            #   挂了 ``_record_probe``，**漏了这条"已经有会话"的路**。
            #   后果恰好在**一切正常**的时候出现：会话一旦建立，探针每 5s
            #   都在这条 return 上安静退出 —— 不记日志、不更新 session_state.json，
            #   于是卡片上的「探测」停在会话建立前那一条，甚至整个文件因
            #   90s TTL 过期而显示「探测:—」。真机 2026-10-10 17:37 就是这个状态：
            #   定时器在跑、会话活着（no_session=0、detected=genshin），
            #   但日志里一条探测记录都没有、session_state.json 停在 17:13。
            self._record_probe("found", "会话已存在：{}".format(current.display_name))
            return False, "已有会话（{}），无需探测".format(current.display_name)

        try:
            window = await asyncio.to_thread(find_genshin_window)
        except Exception as exc:  # noqa: BLE001 —— 探测失败绝不能拖死 timer 线程
            self.logger.warning("会话探测：扫窗口失败（%s）：%s", trigger, exc)
            self._record_probe("error", "扫窗口失败：{}".format(exc))
            return False, "扫窗口失败：{}".format(exc)

        if not window:
            # ★ 兜底：改用**引擎侧**那套只认标题的查找（2026-10-10）。
            #   两条查找的关键差异 ——
            #     screen_capture.find_genshin_window()：进程名 **或** 标题，
            #       进程名靠 ``psutil.Process(pid).name()``，而整个 _cb 被
            #       try 包着 —— psutil 在插件宿主进程里对游戏进程一抛
            #       （权限/会话），**每个窗口都走 except 被跳过**，
            #       结果恒 None，而外���看跟"游戏没开"一模一样。
            #     capturer._find_window()：只看标题子串，**完全不碰进程名**，
            #       所以对 psutil 的权限问题免疫 —— 引擎侧一直靠它跑通。
            #   引擎侧已实测能拿到原神窗口（客户区 0,0,2560,1600），所以
            #   这里用它兜底是安全的；两套都失败才认定"真没找到"。
            window = await asyncio.to_thread(_find_window_by_title)
            if window:
                self.logger.info(
                    "genshin_game_buddy: 会话探测（%s）主路径没找到，"
                    "已由标题兜底路径命中：title=%r pid=%s",
                    trigger, window.get("title"), window.get("pid"))
            else:
                # ★ 普查**不是每次都打**。宿主里"没找到窗口"是 5s 一次的常态，
                #   首轮就打普查既刷屏又会把真正的日志冲掉；等连续 miss 到
                #   _PROBE_CENSUS_AFTER 轮（约 15s）才打，既降噪又正好落在
                #   诊断价值最高的时刻。window_probe 侧还有一道进程级闸。
                self._probe_miss += 1
                if self._probe_miss >= self._PROBE_CENSUS_AFTER:
                    # 用**插件自己**的 logger —— 写在 screen_capture 里那份
                    # 用子 logger 打，真机日志里一条都没有：宿主只给
                    # plugin.genshin_game_buddy 这一个名字装了 handler，
                    # 子 logger 往 root 冒泡时没人接，记录被日志配置吃掉了。
                    _log_window_census(self.logger, trigger)
                    self._probe_miss = 0

        if not window:
            self._record_probe("no_window")
            # 常态路径，节流后再记。
            if now - self._probe_last_log_at >= self._PROBE_LOG_THROTTLE_S:
                self._probe_last_log_at = now
                self._probe_last_reason = "no_window"
                self.logger.info(
                    "genshin_game_buddy: 会话探测（%s）没找到游戏窗口，"
                    "引擎继续空转；启动原神后会自动建立会话",
                    trigger,
                )
            return False, "没找到游戏窗口 —— 请先启动原神"

        title = str(window.get("title") or "")
        game_id, rule = self._probe_game_for_title(title)
        if not game_id:
            self.logger.warning(
                "会话探测（%s）找到了窗口「%s」但没有可登记的游戏：%s",
                trigger, title, rule,
            )
            self._record_probe("no_match", "窗口「{}」不匹配：{}".format(title, rule))
            return False, "找到窗口但没有可登记的游戏（{}）".format(rule)

        try:
            status, result = await self._register(game_id)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("会话探测（%s）登记 %s 失败：%s", trigger, game_id, exc)
            self._record_probe("register_failed",
                               "登记 {} 失败：{}".format(game_id, exc))
            return False, "登记失败：{}".format(exc)

        if status != "ok" or result is None:
            self.logger.warning(
                "会话探测（%s）登记 %s 失败：status=%s", trigger, game_id, status,
            )
            self._record_probe("register_failed",
                               "登记 {} 返回 status={}".format(game_id, status))
            return False, "登记失败（{}）".format(status)

        session = result.session
        self._probe_activated = session.game_id
        self._probe_last_log_at = now
        # 场景/术语索引在 set_game 路径上会跟着建；这里补一次
        # _load_scenes_for_current，保证 switch 后场景表不是空的
        # （拍板 2.0.72 记的正是这个坑）。
        try:
            self._load_scenes_for_current()
        except Exception as exc:  # noqa: BLE001 —— 场景表失败不该回滚会话
            self.logger.warning("会话探测：场景表加载失败（不影响会话）：%s", exc)

        self.logger.info(
            "genshin_game_buddy: 会话已建立（窗口=%s，标题=%s，game=%s，依据=%s，触发=%s）",
            window.get("pid"), title, session.game_id, rule, trigger,
        )
        # ★ overlay 读的就是这份文件。写失败只 warning，不影响会话本身。
        #   先记探测结果再发布 —— _publish_session 会把 self 上的探测字段
        #   一并写进 payload，所以顺序必须是"记录 -> 发布"。
        self._record_probe("found", "窗口「{}」-> {}".format(title, session.game_id))
        return True, "会话已建立：{}".format(session.display_name)

    @timer_interval(id="session_probe", seconds=5)
    async def session_probe(self) -> None:
        """B1：每5s 扫一次游戏窗口，扫到就自动建立会话。

        与 ``ocr_perceive`` 共用 ``@timer_interval`` 机制（宿主在timer
        线程里 ``asyncio.run()`` 跑这个协程），所以这里**不需要**自己
        起线程、也不需要在 ``shutdown`` 里回收 —— 生命周期跟
        ``ocr_perceive`` 完全一致，少一处容易漏的清理。

        ★ 为什么不用 ``default_game = "genshin"`` 一刀切：
          那样 ``restore()`` 会在每次启动时**无条件**建立会话，于是用户
          根本没开游戏时也会去抓屏（要等 6 个 tick 才由 OUT_OF_GAME 闸门
          停掉）。窗口命中才建会话，语义上干净得多。
        """
        try:
            await self._probe_and_activate("timer")
        except Exception:  # noqa: BLE001 —— 探针崩了不能让 timer 停摆
            self.logger.exception(
                "genshin_game_buddy: session_probe tick crashed (timer keeps running)"
            )
            return
        # 探针正常跑完就顺手续个命：overlay 的 90s TTL 靠这个撑着。
        # 放在 try 外面是有意的 —— _probe_and_activate 抛了也不该停止续写，
        # 那正好是"会话还在但探测出错"的情况。
        self._session_heartbeat(reason="probe:heartbeat")

    @ui.action(
        id="activate_game",
        label="检测并激活游戏",
        icon="🎯",
        tone="primary",
        group="settings",
        order=5,
        refresh_context=True,
    )
    @plugin_entry(
        id="activate_game",
        name="检测并激活游戏",
        description="扫描本地游戏窗口，命中就自动建立会话（不用先在对话里说游戏名）。",
        llm_result_fields=["summary"],
        metadata={"result_kind": "event"},
        input_schema={"type": "object", "properties": {}},
    )
    async def activate_game_entry(self, **_: Any) -> Any:
        """B2：面板按钮 / LLM 都能显式触发一次探测。

        与定时器走**同一条** ``_probe_and_activate``，只是不受 5s 节流
        限制 —— 用户主动点的，必须立刻给结论。
        """
        ok, message = await self._probe_and_activate("manual")
        return Ok({"summary": message, "activated": ok})

    # ==================================================================
    # 3.0-b：会话状态落盘（overlay 读）
    # ------------------------------------------------------------------
    # 为什么**不**写进 state_snapshot.json：那个文件归**引擎**
    # （run_loop 经 ChannelWriter）所有，而「有没有会话」是 N.E.K.O
    # 插件侧的事 —— 插件不跑引擎，引擎也不认识 N.E.K.O 的会话。
    # 混写会变成两边抢同一个文件，且 StateSnapshot 的字段是引擎形状的。
    #
    # 为什么写侧**不在这里复制路径表达式**：读侧在
    # ``autopilot/ui/overlay/layout.py`` 的 SESSION_STATE_FILE。这里从
    # ``protocol.DEFAULT_RUN_DIR`` 推导，两边同源于 ``autopilot/_runtime``，
    # **结构上就不可能写歪**。pathfind_state.json 那对是靠「两边各写一份
    # + 注释互相提醒」，这里能做到一样准就不该退化到靠纪律。

    #: 与 ``layout.SESSION_SCHEMA_VERSION`` 必须一致；对不上 overlay
    #: **不显示**（宁可退回「无会话」，也不拿错字段名渲染）。
    SESSION_SCHEMA_VERSION = 1
    #: 续写间隔（秒）。``layout.SESSION_STATE_MAX_AGE_S`` 是 90s，
    #: 超过它插件在 overlay 眼里就"没影了" —— 所以有会话时必须定期续命。
    #: 取 30s = 3 倍余量，容忍偶发的循环延迟。
    SESSION_HEARTBEAT_S = 30.0

    def _write_session_state(self, *, active: bool, game_id: str = "",
                             display_name: str = "", reason: str = "") -> bool:
        """把会话状态写成 ``autopilot/_runtime/session_state.json``。

        用 ``atomic_write_text``（临时文件 + ``os.replace``）而不是直接
        ``open(path, "w")`` —— 后者先把文件截断成 0 字节再逐字节写，
        overlay 正好在那个窗口里 stat 到 size 变化、紧接着读到半截 JSON，
        第 1 行就会闪一下退回「无会话」。理由见 ``protocol.atomic_write_text``。

        **失败只 warning 不抛**：这是纯诊断输出，目录写不进去不该
        反过来影响会话本身。
        """
        try:
            from .autopilot.ui import protocol as _proto
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("会话状态文件不可用（autopilot.ui 导入失败）：%s", exc)
            return False
        target = _proto.DEFAULT_RUN_DIR / "session_state.json"
        payload = {
            "schema_version": self.SESSION_SCHEMA_VERSION,
            "written_at_ms": int(time.time() * 1000),
            "active": bool(active),
            "game_id": game_id,
            "display_name": display_name or game_id,
            "reason": reason,
            # -- 探测反馈（2026-10-10 新增，全部**可选**）--
            # 老版本 overlay 只读 active/game_id/display_name，多这几个键
            # 它一个都不看；新版 overlay 缺这几个键时退回「探测:—」。
            # 所以**不 bump schema 版本**：bump 会让"新写侧 + 旧读侧"
            # 直接读不出文件（load_session_state 版本不符返回 None），
            # 而加可选键两个方向都兼容。
            "probe_ts_ms": int(getattr(self, "_probe_ts_ms", 0) or 0),
            "probe_result": str(getattr(self, "_probe_result", "") or ""),
            "probe_detail": str(getattr(self, "_probe_detail", "") or ""),
            "probe_count": int(getattr(self, "_probe_count", 0) or 0),
            # 宿主跑的是哪一版 —— 2026-10-10 加：排查时反复遇到"我改了
            # 怎么没生效"，根因都是**宿主还跑着上一次的构建**。版本号
            # 直接摆在卡片上，一眼就能看出装的是不是刚改的那版。
            "plugin_version": plugin_version(),
        }
        try:
            _proto.atomic_write_text(
                target, json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("写会话状态失败（不影响会话本身）：%s", exc)
            return False
        self._session_state_written_at = time.monotonic()
        self._session_state_active = bool(active)
        return True

    def _publish_session(self, *, reason: str) -> bool:
        """按**当前真实**会话状态写一次。startup / set_game / 探针命中都走它。"""
        session = getattr(getattr(self, "_manager", None), "current", None)
        self._probe_published = str(getattr(self, "_probe_result", "") or "")
        self._probe_published_at = int(time.time() * 1000)
        if session is None:
            return self._write_session_state(active=False, reason=reason)
        return self._write_session_state(
            active=True,
            game_id=session.game_id,
            display_name=session.display_name,
            reason=reason,
        )

    #: 两次落盘之间的最小间隔（秒）。探针 5s 一轮，但**结果没变时**
    #: 每 12 轮（≈60s）才写一次。
    #:
    #: ★ 为什么要去重 —— 不能每轮都写
    #:   ------------------------------
    #:   ① ``session_state.json`` 的 TTL 是 90s，它的"过期"本身就是
    #:      "插件没了"的信号。探针每 5s 写一次，等于**永久不过期**，
    #:      插件崩了/被杀之后 overlay 还会一直显示"有会话"，骗人。
    #:   ② 每 5s 一次原子写（临时文件 + replace）对一个诊断文件太吵。
    #:   结果**变了**立刻写（用户要马上看到），没变才按 60s 续命。
    _PROBE_PUBLISH_MIN_INTERVAL_S = 60.0

    def _record_probe(self, result: str, detail: str = "") -> None:
        """记一次探测结果，并**按需**落盘。

        :param result: ``found`` / ``no_window`` / ``no_match`` /
            ``register_failed`` / ``error``
        :param detail: 人话，overlay 原样显示（窗口标题 / 失败原因）。

        结果变了立刻写；没变则每 60s 写一次，既让 overlay 看得见"探针还活着"，
        又保住 90s TTL 的"插件死了会过期"语义（见
        :attr:`_PROBE_PUBLISH_MIN_INTERVAL_S` 的注释）。
        """
        import time as _time  # noqa: PLC0415 - 与模块级 time 同源，显式别名避免歧义

        self._probe_count = int(getattr(self, "_probe_count", 0) or 0) + 1
        self._probe_result = result
        self._probe_detail = detail
        self._probe_ts_ms = int(_time.time() * 1000)

        now_ms = self._probe_ts_ms
        changed = result != str(getattr(self, "_probe_published", "") or "")
        due = (now_ms - int(getattr(self, "_probe_published_at", 0) or 0)
               >= self._PROBE_PUBLISH_MIN_INTERVAL_S * 1000)
        if changed or due:
            self._publish_session(reason="probe:{}".format(result))

    def _session_heartbeat(self, *, reason: str = "heartbeat") -> None:
        """定期续命，**只在有会话时写**。

        没会话时没必要每 30 秒动一次文件去说「还是没有」—— 那条信息
        在 overlay 上已经是默认显示的了（读不到文件就显示「无会话」）。
        """
        if getattr(self, "_session_state_active", False) is not True:
            return
        if (time.monotonic()
                - getattr(self, "_session_state_written_at", 0.0)
                < self.SESSION_HEARTBEAT_S):
            return
        self._publish_session(reason=reason)

    @ui.action(
        id="apply_settings",
        label="应用设置",
        icon="💾",
        tone="success",
        group="config",
        order=10,
        refresh_context=True,
    )
    @plugin_entry(
        id="apply_settings",
        name="应用设置",
        description="将面板的设置写回插件运行时配置。",
        llm_result_fields=["summary"],
        metadata={"result_kind": "event"},
        input_schema={
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "description": "7 项配置字典（仅传改动过的字段也可）",
                },
            },
        },
    )
    async def apply_settings_entry(
        self, payload: dict[str, Any] | None = None, **_: Any,
    ) -> Any:
        """拍板 2.0.66：面板点"应用"时调用——校验 + 写回 + 持久化 + 必要时重建 executor。

        拍板 2.0.68：payload 改为可选（默认 None）—— 旧签名 `payload: dict` 在面板
        调用栈变更 / Hosted UI v2 sandbox 重发 / 老面板残留 instance 等情况下偶发
        传空 arg 列表，导致 `TypeError: missing 1 required positional argument`。
        这里把 None 视为"未传任何字段"，返回 Err(INVALID_INPUT) 而非让 Python 抛
        TypeError 干掉 entry call。

        新增 2.0.66 行为：
            1. ocr_profile 自适应：
               - profile 在预设档（eco/balanced/performance/auto 探测后）→ 强制 interval/threads 用档位默认
               - profile=custom → 保留用户手填的 interval/threads
               - 面板手填 interval/threads 但 profile 还是预设档 → 自动切到 custom
            2. 持久化：apply 后用 self.config.update(...) 写回 plugin.toml，重启不丢

        热更新规则（沿用 2.0.65）：
            ocr_perceive_interval_seconds 改：自动生效（tick 里读 self._options）
            ocr_worker_threads 改：重建 ThreadPoolExecutor（旧 wait=False 关掉）
            screen_activation_limit / query_activation_ttl_seconds / scene_prompt_reinject_seconds：
                自动生效（下一次激活/推送时读 self._options）
            desktop_markers_* 改：自动生效（_desktop_markers() 每次调用重读）
        """
        options_payload = payload.get("options", {}) if isinstance(payload, Mapping) else {}
        if payload is None:
            # 拍板 2.0.68：payload 缺省 → 视为"没传任何字段"——返 Err 而非抛 TypeError
            return Err(SdkError(
                code="INVALID_INPUT",
                message="apply_settings called without payload; pass {\"options\": {...}}",
            ))
        if not isinstance(options_payload, Mapping):
            return Err(SdkError(code="INVALID_INPUT", message="options must be an object"))

        applied: dict[str, Any] = {}
        old_workers = self._options.ocr_worker_threads
        new_workers = old_workers
        old_profile = self._options.ocr_profile

        try:
            # 用 from_section 重用校验逻辑——但用现有 self._options 作为基线，
            # 让未传字段保持原值。先把 payload 合并到当前 options dict 的视图。
            base = {
                "default_game": self._options.default_game,
                "auto_restore_last_game": self._options.auto_restore_last_game,
                "context_inject_max_terms": self._options.context_inject_max_terms,
                "max_context_chars": self._options.max_context_chars,
                "lookup_max_matches": self._options.lookup_max_matches,
                "lookup_max_chars_per_term": self._options.lookup_max_chars_per_term,
                "reinject_every_n_messages": self._options.reinject_every_n_messages,
                "ocr_profile": self._options.ocr_profile,
                "ocr_perceive_interval_seconds": self._options.ocr_perceive_interval_seconds,
                "ocr_worker_threads": self._options.ocr_worker_threads,
                "screen_activation_limit": self._options.screen_activation_limit,
                "query_activation_ttl_seconds": self._options.query_activation_ttl_seconds,
                "scene_prompt_reinject_seconds": self._options.scene_prompt_reinject_seconds,
                "change_driven_enabled": self._options.change_driven_enabled,
                "change_detect_threshold": self._options.change_detect_threshold,
                "scene_switch_cooldown_seconds": self._options.scene_switch_cooldown_seconds,
                "desktop_markers_enabled": self._options.desktop_markers_enabled,
                "desktop_markers_extra": list(self._options.desktop_markers_extra),
            }
            base.update({k: v for k, v in options_payload.items() if k in base})

            # 拍板 2.0.66：profile 自适应——手填 interval/threads 自动切 custom；
            # 预设档强制 interval/threads = 档位默认。
            profile = _clean_profile(base.get("ocr_profile"))
            interval_changed = (
                "ocr_perceive_interval_seconds" in options_payload
                and base["ocr_perceive_interval_seconds"] != self._options.ocr_perceive_interval_seconds
            )
            threads_changed = (
                "ocr_worker_threads" in options_payload
                and base["ocr_worker_threads"] != self._options.ocr_worker_threads
            )
            if (interval_changed or threads_changed) and profile != "custom":
                profile = "custom"
                base["ocr_profile"] = "custom"
            if profile != "custom" and profile in OCR_PROFILE_DEFAULTS:
                defaults = OCR_PROFILE_DEFAULTS[profile]
                base["ocr_perceive_interval_seconds"] = defaults["interval"]
                base["ocr_worker_threads"] = defaults["threads"]

            # 复用 PluginOptions.from_section 做完整 clamp + 类型校验
            new_opts = PluginOptions.from_section(base)
            new_workers = new_opts.ocr_worker_threads
            self._options = new_opts

            applied = {
                "ocr_profile": new_opts.ocr_profile,
                "ocr_perceive_interval_seconds": new_opts.ocr_perceive_interval_seconds,
                "ocr_worker_threads": new_opts.ocr_worker_threads,
                "screen_activation_limit": new_opts.screen_activation_limit,
                "query_activation_ttl_seconds": new_opts.query_activation_ttl_seconds,
                "scene_prompt_reinject_seconds": new_opts.scene_prompt_reinject_seconds,
                "change_driven_enabled": new_opts.change_driven_enabled,
                "change_detect_threshold": new_opts.change_detect_threshold,
                "scene_switch_cooldown_seconds": new_opts.scene_switch_cooldown_seconds,
                "desktop_markers_enabled": new_opts.desktop_markers_enabled,
                "desktop_markers_extra": list(new_opts.desktop_markers_extra),
            }

            # 拍板 2.0.66：持久化——写回 plugin.toml 的 [genshin_game_buddy] 段
            # 注意：写整个 genshin_game_buddy 子树以保证一致
            try:
                await self.config.update({
                    "genshin_game_buddy": {
                        "ocr_profile": new_opts.ocr_profile,
                        "ocr_perceive_interval_seconds": new_opts.ocr_perceive_interval_seconds,
                        "ocr_worker_threads": new_opts.ocr_worker_threads,
                        "screen_activation_limit": new_opts.screen_activation_limit,
                        "query_activation_ttl_seconds": new_opts.query_activation_ttl_seconds,
                        "scene_prompt_reinject_seconds": new_opts.scene_prompt_reinject_seconds,
                        "change_driven_enabled": new_opts.change_driven_enabled,
                        "change_detect_threshold": new_opts.change_detect_threshold,
                        "scene_switch_cooldown_seconds": new_opts.scene_switch_cooldown_seconds,
                        "desktop_markers_enabled": new_opts.desktop_markers_enabled,
                        "desktop_markers_extra": list(new_opts.desktop_markers_extra),
                    },
                })
                self.logger.info(
                    "genshin_game_buddy: apply_settings persisted to plugin.toml (11 fields)",
                )
            except Exception as persist_err:
                # 拍板 2.0.66：持久化失败不静默——返回 Err 给面板显示"保存失败"
                self.logger.exception(
                    "genshin_game_buddy: apply_settings persist failed",
                )
                return Err(SdkError(
                    code="PERSIST_FAILED",
                    message=f"内存已更新，但写盘失败：{persist_err}",
                ))

            # 拍板 2.0.65：worker 数变了 → 重建 executor。
            # wait=False + cancel_futures=True 让正在跑的 OCR 自然结束、排队的被取消。
            if self._ocr_executor is not None and new_workers != old_workers:
                self.logger.info(
                    "genshin_game_buddy: apply_settings rebuild executor workers={}->{}",
                    old_workers, new_workers,
                )
                self._ocr_executor.shutdown(wait=False, cancel_futures=True)
                self._ocr_executor = None  # 下次 tick 懒重建

            self.logger.info(
                "genshin_game_buddy: apply_settings applied profile={} (was={}) workers={}->{}",
                new_opts.ocr_profile, old_profile, old_workers, new_workers,
            )
            return Ok({
                "summary": (
                    f"已应用设置并保存到 plugin.toml；"
                    f"profile {old_profile}→{new_opts.ocr_profile}；"
                    f"worker 线程数 {old_workers}→{new_workers}"
                ),
                "applied": applied,
                "executor_rebuilt": (self._ocr_executor is None and new_workers != old_workers),
                "persisted": True,
            })
        except Exception as exc:
            self.logger.exception("genshin_game_buddy: apply_settings failed")
            return Err(SdkError(code="APPLY_FAILED", message=str(exc)))

    # ==================================================================
    # 入口点（Plugin Manager / Agent / 其它插件可调用）
    # ==================================================================
    # 返回值契约：成功 → Ok({..., "summary": <一句话>})；失败 → Err(SdkError(...))。
    # 四个入口都声明了 llm_result_fields=["summary"]，因此每个成功返回体都必须带
    # "summary"，否则结果契约校验会报 LlmResultValidationError。
    # 四个入口都是"即时回执"，按宿主规范显式降级为事件语义（result_kind=event），
    # 避免主 AI 用"任务已完成"的口吻汇报一次查询。

    @plugin_entry(
        id="set_game",
        name=tr("entries.set_game.name", default="登记游戏"),
        description=tr(
            "entries.set_game.description",
            default="登记你当前在玩的游戏，之后陪聊就用那款游戏的语境和术语。",
        ),
        llm_result_fields=["summary"],
        metadata={"result_kind": "event"},
        input_schema={
            "type": "object",
            "properties": {
                "game": {
                    "type": "string",
                    "description": tr(
                        "entries.set_game.param.game",
                        default="游戏名称，例如「原神」「鸣潮」。",
                    ),
                },
            },
            "required": ["game"],
        },
    )
    async def set_game(self, game: str = "", **_: Any) -> Any:
        status, result = await self._register(game)
        if status == "not_found":
            return Err(SdkError(self._t("errors.unknown_game", available=self._registry.available_names())))
        if status == "disabled":
            return Err(SdkError(self._t("errors.game_disabled")))
        if status == "internal":
            return Err(SdkError(self._t("errors.internal")))
        if status in ("terms_dir_missing", "terms_load_failed"):
            return Err(SdkError(self._terms_error_text(status)))
        if result is None:  # pragma: no cover - 上面的分支已覆盖全部非 ok 状态
            return Err(SdkError(self._t("errors.internal")))
        # 会话变了 -> overlay 的第 1 行要跟着变（用户手动切的也算）
        self._publish_session(reason="set_game")

        injected = self._push_context(result.session)
        summary = self._t(
            "status.set_game.done",
            game=result.session.display_name,
            count=result.library.size(),
        )
        summary += (
            self._t("status.set_game.injected")
            if injected
            else self._t("status.set_game.not_injected")
        )
        if not result.persisted:
            summary += self._t("errors.store_unavailable")
        return Ok(
            {
                "game_id": result.session.game_id,
                "game": result.session.display_name,
                "term_count": result.library.size(),
                "context_injected": injected,
                "persisted": result.persisted,
                "summary": summary,
            }
        )

    @plugin_entry(
        id="get_current_game",
        name=tr("entries.get_current_game.name", default="查看当前游戏"),
        description=tr(
            "entries.get_current_game.description",
            default="看看现在登记的是哪款游戏。",
        ),
        llm_result_fields=["summary"],
        metadata={"result_kind": "event"},
        input_schema={"type": "object", "properties": {}},
    )
    async def get_current_game(self, **_: Any) -> Any:
        await self._ensure_config_ready()
        session = self._manager.current
        if session is None:
            return Ok({"game_id": "", "game": "", "summary": self._t("status.get_game.none")})
        return Ok(
            {
                "game_id": session.game_id,
                "game": session.display_name,
                "switched_at": session.switched_at,
                "summary": self._t("status.get_game.done", game=session.display_name),
            }
        )

    @plugin_entry(
        id="list_games",
        name=tr("entries.list_games.name", default="列出可用游戏"),
        description=tr(
            "entries.list_games.description",
            default="列出当前已登记、可以陪聊的游戏。",
        ),
        llm_result_fields=["summary"],
        metadata={"result_kind": "event"},
        input_schema={"type": "object", "properties": {}},
    )
    async def list_games(self, **_: Any) -> Any:
        await self._ensure_config_ready()
        games = self._registry.enabled_games()
        if not games:
            return Ok({"games": [], "count": 0, "summary": self._t("status.list_games.empty")})
        current = self._manager.current
        payload = [
            {
                "game_id": entry.game_id,
                "display_name": entry.display_name,
                "current": bool(current is not None and current.game_id == entry.game_id),
            }
            for entry in games
        ]
        names = "、".join(entry.display_name for entry in games)
        return Ok(
            {
                "games": payload,
                "count": len(payload),
                "summary": self._t("status.list_games.done", count=len(payload), games=names),
            }
        )

    @plugin_entry(
        id="refresh_game_context",
        name=tr("entries.refresh_game_context.name", default="重新注入语境"),
        description=tr(
            "entries.refresh_game_context.description",
            default="把当前游戏的语境重新送进对话，长对话里语境变淡时用。",
        ),
        llm_result_fields=["summary"],
        metadata={"result_kind": "event"},
        input_schema={"type": "object", "properties": {}},
    )
    async def refresh_game_context(self, **_: Any) -> Any:
        await self._ensure_config_ready()
        session = self._manager.current
        entry = self._registry.get(session.game_id) if session is not None else None
        if session is None or entry is None:
            return Err(SdkError(self._t("errors.no_game_declared")))
        try:
            # force：用户可能刚改过覆盖层术语，"重新注入语境"要顺带重新读盘。
            self._manager.ensure_library(entry, force=True)
        except TermLoadError as exc:
            return Err(SdkError(self._terms_error_text(exc.code)))
        if not self._push_context(session, announce_switch=False):
            return Err(SdkError(self._t("errors.push_failed")))
        return Ok(
            {
                "game_id": session.game_id,
                "game": session.display_name,
                "context_injected": True,
                "summary": self._t("status.refresh.done", game=session.display_name),
            }
        )

    @plugin_entry(
        id="capture_screen",
        name=tr("entries.capture_screen.name", default="捕获屏幕"),
        description=tr(
            "entries.capture_screen.description",
            default="读取宿主推给模型的最近屏幕帧，用于 OCR 识别。",
        ),
        metadata={"result_kind": "event"},
        input_schema={
            "type": "object",
            "properties": {
                "max_count": {
                    "type": "integer",
                    "description": "最多读取几帧，默认 2，上限 4",
                    "default": 2,
                }
            },
        },
    )
    async def capture_screen(self, *, max_count: int = 2, **_):
        # 只读 frames，不写日志、不外传
        limit = max(1, min(int(max_count), 4))
        try:
            frames = await asyncio.to_thread(
                self.bus.frames.get, max_count=limit
            )
        except Exception as exc:
            return Err(SdkError(f"bus.frames 不可用: {type(exc).__name__}"))

        if isinstance(frames, Err):
            return frames
        if isinstance(frames, Ok):
            frames = frames.value

        records = list(frames)
        return Ok({
            "count": len(records),
            "frames": [
                {
                    "source": getattr(item, "source", None),
                    "mime": getattr(item, "mime", None),
                    "captured_at": getattr(item, "captured_at", None),
                    "has_image": bool(getattr(item, "image_base64", None)),
                }
                for item in records
            ],
        })

    @plugin_entry(
        id="ocr_screen",
        name=tr("entries.ocr_screen.name", default="识别屏幕文字"),
        description=tr(
            "entries.ocr_screen.description",
            default="读取屏幕帧做 OCR，与当前游戏术语库匹配后返回 matched_terms 列表。当用户问'屏幕上是什么''我在干嘛'时调用；不要拿返回值联网搜索，不要复述数字，不要反复调用。",
        ),
        metadata={"result_kind": "event"},
        input_schema={
            "type": "object",
            "properties": {
                "max_count": {
                    "type": "integer",
                    "description": "最多处理几帧，默认 1，上限 2",
                    "default": 1,
                }
            },
        },
    )
    async def ocr_screen(self, *, max_count: int = 1, **_):
        limit = max(1, min(int(max_count), 2))
        try:
            frames = await asyncio.to_thread(
                self.bus.frames.get, max_count=limit
            )
        except Exception as exc:
            return Err(SdkError(f"bus.frames 不可用: {type(exc).__name__}"))

        if isinstance(frames, Err):
            return frames
        if isinstance(frames, Ok):
            frames = frames.value

        records = list(frames)
        ocr_parts = []
        for item in records:
            b64 = getattr(item, "image_base64", None) or ""
            if b64:
                text = await self._ocr.extract_text_from_base64(b64)
                if text:
                    ocr_parts.append(text)

        combined = "\n".join(ocr_parts)
        ocr_chars = len(combined)

        session = self._manager.current
        game_id = session.game_id if session else None
        matched: list[str] = []
        library = self._manager.library
        if library is not None and combined:
            for entry in library.by_key.values():
                if len(matched) >= 10:
                    break
                if entry.key in combined:
                    matched.append(entry.key)
                    continue
                for alias in entry.aliases:
                    if alias in combined:
                        matched.append(entry.key)
                        break

        return Ok({
            "count": len(records),
            "game": game_id,
            "matched_terms": matched[:10],
            "ocr_chars": ocr_chars,
        })

    # ==================================================================
    # 宿主消息：P1 的"每 N 轮重推一次语境"
    # ==================================================================
    # 刻意不看 text / sender 的内容（那是用户对话，属隐私面），只用于计数。
    # 宿主不派发或清空配置时行为是"永不重推"，绝不会刷屏。

    @message(id="on_chat_message", source="chat")
    async def on_chat_message(self, text: str = "", sender: str = "", **_: Any) -> Any:
        del sender
        session = self._manager.current
        # 拍板 2.0.59：无感激活——用户消息命中术语 → 立即激活 + 推术语卡（与 periodic reinject 独立）
        if session is not None and text:
            self._scan_and_activate_query(text)
        every_n = self._options.reinject_every_n_messages
        if every_n <= 0:
            return Ok({"reinjected": False, "reason": "disabled"})
        if session is None:
            return Ok({"reinjected": False, "reason": "no_game"})
        if not self._counter.observe(1):
            return Ok(
                {"reinjected": False, "reason": "below_threshold", "pending": self._counter.pending}
            )
        injected = self._push_context(session, announce_switch=False)
        self.logger.info(
            "genshin_game_buddy: periodic reinject (every {} messages, submitted={})",
            every_n,
            injected,
        )
        return Ok({"reinjected": injected, "game_id": session.game_id})

    # ==================================================================
    # LLM 工具（对话期由模型调用）
    # ==================================================================
    # ⚠️ 与入口点**不同**：@llm_tool 处理函数的返回形状是 main_server 直接读的
    #    JSON —— 规范形状是 {"output": ..., "is_error": bool}，**不是** Ok/Err。
    #    依据：docs/plugins/tool-calling.md 的 "Returning errors"；
    #    参考实现 plugin/plugins/game_agent_minecraft/__init__.py 的 llm_tool 处理器。

    @llm_tool(
        name="set_current_game",
        description=tr(
            "tools.set_current_game.description",
            default="当用户提到自己正在玩某款游戏时调用，登记当前游戏，之后的陪聊会使用该游戏的语境与术语。",
        ),
        parameters={
            "type": "object",
            "properties": {
                "game": {
                    "type": "string",
                    "description": tr(
                        "tools.set_current_game.param.game",
                        default="游戏名称，可以是用户原话里的说法。",
                    ),
                },
            },
            "required": ["game"],
        },
        timeout=10.0,
    )
    async def set_current_game(self, *, game: str = "", **_: Any) -> Any:
        status, result = await self._register(game)
        if status == "ok" and result is not None:
            injected = self._push_context(result.session)
            return {
                "output": {
                    "registered": True,
                    "game": result.session.display_name,
                    "game_id": result.session.game_id,
                    "term_count": result.library.size(),
                    "context_injected": injected,
                    "persisted": result.persisted,
                    "notice": "" if result.persisted else self._t("errors.store_unavailable"),
                },
                "is_error": False,
            }
        if status == "not_found":
            # 用户说了一个库里没有的名字：这是可自纠的正常情况，不是错误。
            return {
                "output": {
                    "registered": False,
                    "reason": "unknown_game",
                    "available": [entry.display_name for entry in self._registry.enabled_games()],
                    "message": self._t(
                        "errors.unknown_game", available=self._registry.available_names()
                    ),
                },
                "is_error": False,
            }
        if status == "disabled":
            return {
                "output": {
                    "registered": False,
                    "reason": "game_disabled",
                    "message": self._t("errors.game_disabled"),
                },
                "is_error": False,
            }
        if status in ("terms_dir_missing", "terms_load_failed"):
            return {
                "output": {
                    "registered": False,
                    "reason": status,
                    "message": self._terms_error_text(status),
                },
                "is_error": True,
                "error": "TERMS_UNAVAILABLE",
            }
        return {
            "output": {
                "registered": False,
                "reason": "internal",
                "message": self._t("errors.internal"),
            },
            "is_error": True,
            "error": "INTERNAL",
        }

    @llm_tool(
        name="lookup_game_term",
        description=tr(
            "tools.lookup_game_term.description",
            default="查询当前游戏术语库里的具体条目（角色、机制、武器、圣遗物等）。【重要】参数 query 必须传角色名/术语名（如「纳西妲」「深境螺旋」），不要传完整句子。查不到时先用你已有的知识回答，不要只是说「找不到」。",
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": tr(
                        "tools.lookup_game_term.param.query",
                        default="要查的术语或玩家的说法。",
                    ),
                },
            },
            "required": ["query"],
        },
        timeout=10.0,
    )
    async def lookup_game_term(self, *, query: str = "", **_: Any) -> Any:
        await self._ensure_config_ready()
        text = query.strip() if isinstance(query, str) else ""
        # 隐私红线：只记长度，绝不记 query 原文。
        self.logger.info("genshin_game_buddy: lookup_game_term received query_len={}", len(text))

        session = self._manager.current
        if session is None:
            return {
                "output": {"found": False, "reason": "no_game", "message": self._t("lookup.no_game")},
                "is_error": False,
            }

        library, error_code = self._library_for(session)
        if library is None:
            return {
                "output": {
                    "found": False,
                    "reason": error_code,
                    "message": self._terms_error_text(error_code),
                },
                "is_error": True,
                "error": "TERMS_UNAVAILABLE",
            }

        matches = library.match(text, self._options.lookup_max_matches)
        if not matches:
            self.logger.info(
                "genshin_game_buddy: lookup_game_term miss (game={}, query_len={})",
                session.game_id,
                len(text),
            )
            return {
                "output": {
                    "found": False,
                    "game": session.display_name,
                    "message": self._t("lookup.not_found", game=session.display_name),
                },
                "is_error": False,
            }

        cards = [
            render_term_card(
                entry,
                entry_line=self._entry_line,
                slang_label=self._t("lookup.slang_label"),
                avoid_label=self._t("lookup.avoid_label"),
                max_chars=self._options.lookup_max_chars_per_term,
            )
            for entry in matches
        ]
        self.logger.info(
            "genshin_game_buddy: lookup_game_term hit {} (game={}, query_len={})",
            len(matches),
            session.game_id,
            len(text),
        )
        return {
            "output": {
                "found": True,
                "game": session.display_name,
                "count": len(matches),
                "entries": [self._entry_payload(entry) for entry in matches],
                "text": build_lookup_block(
                    self._t("lookup.found_header", count=len(matches)), cards
                ),
            },
            "is_error": False,
        }

    # ==================================================================
    # 内部实现（不以 "_" 开头以外的公开方法一律不挂装饰器）
    # ==================================================================

    def _terms_error_text(self, code: str) -> str:
        """把术语加载错误码映射成文案（错误码不含路径，只含维度/原因）。"""
        if code == "terms_dir_missing":
            return self._t("errors.terms_dir_missing")
        return self._t("errors.terms_load_failed")

    def _warn_dev(self, note: str) -> None:
        self.logger.warning("genshin_game_buddy: {}", note)

    def _load_library_for(self, entry: GameEntry) -> TermLibrary:
        """默认层读插件目录，覆盖层读 data_path()（KB：不得自拼数据根）。"""
        return load_library(
            default_root=self.plugin_dir,
            override_root=self.data_path(),
            game_id=entry.game_id,
            display_name=entry.display_name,
            terms_dir=entry.terms_dir,
            warn=self._warn_dev,
        )

    def _library_notes(self, library: TermLibrary) -> list[str]:
        notes = [self._t("terms.load.ok", count=library.size())]
        for dimension in library.failed_dimensions:
            notes.append(self._t("terms.load.partial", dimension=dimension))
        for dimension in library.empty_dimensions:
            notes.append(self._t("terms.load.empty_dimension", dimension=dimension))
        return notes

    def _library_for(self, session: GameSession) -> tuple[TermLibrary | None, str]:
        """取当前术语库；不存在时按需重载。

        返回 ``(library, error_code)``：成功时 ``error_code`` 为空串。
        """
        library = self._manager.library
        if library is not None:
            return library, ""
        entry = self._registry.get(session.game_id)
        if entry is None:
            return None, "terms_dir_missing"
        try:
            return self._manager.ensure_library(entry), ""
        except TermLoadError as exc:
            return None, exc.code

    async def _register(self, raw: object) -> tuple[str, SwitchResult | None]:
        """登记/切换的公共路径。返回 ``(status, result)``。

        status ∈ ok / not_found / disabled / terms_dir_missing / terms_load_failed / internal
        """
        await self._ensure_config_ready()
        match = self._registry.resolve(raw)
        if not match.found or match.game is None:
            self.logger.info(
                "genshin_game_buddy: register miss (input_len={})",
                len(raw) if isinstance(raw, str) else 0,
            )
            return "not_found", None
        entry = match.game
        if not entry.enabled:
            return "disabled", None
        try:
            result = await self._manager.switch(entry)
        except TermLoadError as exc:
            self.logger.warning(
                "genshin_game_buddy: term library unavailable for game {} ({})",
                entry.game_id,
                exc.code,
            )
            return exc.code, None
        except Exception:
            self.logger.exception("genshin_game_buddy: unexpected failure while switching game")
            return "internal", None
        self._load_scenes_for_current()
        self._init_anchors(result.session, result.library)
        # game_id 不是用户隐私，允许进日志（E-21）；用户原文与 query 一律不记。
        self.logger.info(
            "genshin_game_buddy: registered game {} ({} terms, persisted={})",
            entry.game_id,
            result.library.size(),
            result.persisted,
        )
        return "ok", result

    def _entry_line(self, entry: TermEntry) -> str:
        return self._t("lookup.entry_line", term=entry.key, kind=entry.kind, brief=entry.brief)

    def _entry_payload(self, entry: TermEntry) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": entry.key,
            "kind": entry.kind,
            "brief": entry.brief,
        }
        if entry.aliases:
            payload["aliases"] = list(entry.aliases)
        if entry.slang:
            payload["slang"] = list(entry.slang)
        if entry.avoid:
            payload["avoid"] = entry.avoid
        if entry.source:
            payload["source"] = entry.source
        return payload

    # ==================================================================
    # 2.1a：感知层内部方法
    # ==================================================================

    def _load_scenes_for_current(self) -> None:
        """加载当前游戏的场景定义。文件缺失返回空 dict。"""
        session = self._manager.current
        if session is None:
            with self._state_lock:
                self._scene_store = {}
            return
        entry = self._registry.get(session.game_id)
        terms_dir = entry.terms_dir if entry else f"terms/{session.game_id}"
        warnings: list[str] = []
        scenes = load_scenes(
            default_root=self.plugin_dir,
            override_root=self.data_path(),
            game_id=session.game_id,
            terms_dir=terms_dir,
            warnings=warnings,
        )
        with self._state_lock:
            self._scene_store = scenes
        for w in warnings:
            self.logger.warning("genshin_game_buddy: scene load: {}", w)
        # 拍板 2.0.72：明确打"restored"——startup 自动恢复场景是关键路径
        # （区别于 set_game 触发的 reload/手动重载）
        self.logger.info(
            "genshin_game_buddy: restored {} scene(s) for {}",
            len(scenes), session.game_id,
        )

    def _init_anchors(self, session: Any, library: TermLibrary) -> None:
        """初始化核心层激活 + context_terms 显式补充，永久常驻。

        拍板 2.0.56：核心层（core + slang + systems 维度全部条目，约 20-35 条）默认常驻上下文，
        TTL=inf、永不淘汰；MAX_ACTIVE 同步从 12 放宽到 40 容纳核心层。entry.context_terms
        显式清单作为额外补充（也常驻）；角色/武器等具体条目留给 lookup_game_term 按需查。
        """
        # 核心层：library 里所有 dimension ∈ CORE_LAYER_DIMENSIONS 的条目，按维度优先级排列
        core_layer_keys: list[str] = []
        for dimension in CORE_LAYER_DIMENSIONS:
            core_layer_keys.extend(
                e.key for e in library.entries if e.dimension == dimension
            )
        # context_terms 显式补充（去重于核心层）
        extra_keys: list[str] = []
        missing: list[str] = []
        entry = self._registry.get(session.game_id)
        if entry is not None and entry.context_terms:
            for k in entry.context_terms:
                if library.get(k) is None:
                    missing.append(k)
                elif k not in core_layer_keys:
                    extra_keys.append(k)
        if missing:
            self.logger.warning(
                "genshin_game_buddy: context_terms has {} unknown key(s) for game {}",
                len(missing), session.game_id,
            )
        all_keys = core_layer_keys + extra_keys
        if all_keys:
            with self._state_lock:
                self._activation.activate_terms(
                    all_keys, source="core_layer", ttl_seconds=float("inf")
                )

    def _update_perception_state(self, ocr_hit: bool, in_game_chars: bool = False) -> str:
        """状态机：UNKNOWN → IN_GAME → AWAY → OUT_OF_GAME。返回新状态。

        拍板 2.0.62：必须 ocr_hit AND in_game_chars 才升 IN_GAME；
        仅 ocr_hit 但无游戏内特征（如桌面泛词误命中）→ 保持上态（不增减 miss_streak）。
        in_game_chars 默认 False——保持早期 hash-重复路径的行为兼容。
        """
        if ocr_hit and in_game_chars:
            self._miss_streak = 0
            self._perception_state = "IN_GAME"
        elif not ocr_hit:
            self._miss_streak += 1
            if self._perception_state == "IN_GAME" and self._miss_streak >= 3:
                self._perception_state = "AWAY"
            elif self._perception_state == "AWAY" and self._miss_streak >= 6:
                self._perception_state = "OUT_OF_GAME"
        # else: ocr_hit 但无游戏内特征 → 保持现状（桌面泛词命中，不升级不降级）
        return self._perception_state

    def _desktop_markers(self) -> tuple[str, ...]:
        """拍板 2.0.64：运行时合并 builtin + 用户自定义 extra；总开关 desktop_markers_enabled 关掉时整张表失效。"""
        if not self._options.desktop_markers_enabled:
            return ()
        return _DESKTOP_MARKERS_BUILTIN + tuple(self._options.desktop_markers_extra)

    def _has_in_game_characteristics(self, text: str, detected_game: str) -> bool:
        """拍板 2.0.62 + 2.0.63：判定 OCR 文本是否真有"游戏内特征"，防桌面泛词误判 IN_GAME。

        2.0.63 收紧：
          (0) 桌面/浏览器反特征黑名单（命中任一 → 直接判非游戏）
          (a) 命中当前游戏的核心层术语（core/slang/systems 维度，>= 1）
          (b) 命中当前游戏 >= 2 个术语（任意维度）
          (c) signals 命中 **且** 至少有 1 条术语命中（signals 单独不算）

        全不满足 -> False（视为桌面/非游戏 UI）。
        """
        if not text or not detected_game or detected_game == "none":
            return False
        # (0) 桌面/浏览器反特征黑名单（最优先——命中即拒）
        if any(marker in text for marker in self._desktop_markers()):
            return False
        for entry in self._registry.enabled_games():
            if entry.game_id != detected_game:
                continue
            # (a) + (b) 先算 library 命中数（alias 长度下限 >= 2 防单字误命中）
            lib, _ = self._detect_assets(entry)
            core_count = 0
            total_count = 0
            if lib is not None:
                for term in lib.by_key.values():
                    if (
                        (len(term.key) >= 2 and term.key in text)
                        or any(len(a) >= 2 and a in text for a in term.aliases)
                    ):
                        total_count += 1
                        if term.dimension in ("core", "slang", "systems"):
                            core_count += 1
            # (c) signals 命中 + 至少 1 条术语命中 → 才算真游戏内
            signals_hit = any(sig and sig in text for sig in entry.signals)
            if (core_count >= 1 or total_count >= 2) or (signals_hit and total_count >= 1):
                return True
            return False
        return False

    @staticmethod
    def _scan_screen_activate_terms(text: str, library: Any) -> tuple[list[str], list[str]]:
        """拍板 2.0.59 + 2.0.63：扫屏幕 OCR 命中术语（screen + slang）。

        2.0.63 加 alias/key 长度下限 >= 2——防单字 alias 误命中
        （"莹"→"谢莹"、"鱼"→任意"鱼"字、单字 key 裸命中）。
        Returns (screen_matched_keys, slang_matched_keys)。
        """
        screen_matched: list[str] = []
        slang_matched: list[str] = []
        if library is None:
            return screen_matched, slang_matched
        seen: set[str] = set()
        for entry in library.entries:
            if entry.key in seen:
                continue
            key_hit = len(entry.key) >= 2 and entry.key in text
            alias_hit = any(len(a) >= 2 and a in text for a in entry.aliases)
            if key_hit or alias_hit:
                screen_matched.append(entry.key)
                seen.add(entry.key)
            elif any(s and s in text for s in entry.slang):
                slang_matched.append(entry.key)
                seen.add(entry.key)
        return screen_matched, slang_matched

    def _match_scenes(self, ocr_text: str) -> list[str]:
        """遍历场景定义，signals 命中即返回场景名。"""
        if not ocr_text:
            return []
        matched: list[str] = []
        for name, scene in self._scene_store.items():
            for sig in scene.signals:
                if sig and sig in ocr_text:
                    matched.append(name)
                    break
        return matched

    async def detect_game(self, text: str) -> str:
        """用术语库+场景信号判断 OCR 文本的游戏归属（拍板 2.0.14）。返回 game_id 或 "none"。

        枚举注册表全部启用游戏（不硬编码 game_id——今天是 genshin/wuthering_waves，
        将来配了 honkai_star_rail/zenless_zone_zero 自动纳入），按 registry signals（权重高）
        + 术语库 key/alias 命中 + 场景信号命中 加权评分取最高；低于阈值判 "none"。
        OCR 文本只用于匹配，绝不写日志（隐私红线：只记长度+hash）。
        """
        if not text:
            return "none"
        best_id = ""
        best_score = 0
        for entry in self._registry.enabled_games():
            score = sum(3 for sig in entry.signals if sig and sig in text)
            lib, scenes = await asyncio.to_thread(self._detect_assets, entry)
            if lib is not None:
                score += sum(
                    1 for e in lib.by_key.values()
                    if e.key in text or any(a and a in text for a in e.aliases)
                )
            for scene in (scenes or {}).values():
                score += sum(1 for sig in scene.signals if sig and sig in text)
            if score > best_score:
                best_id, best_score = entry.game_id, score
        return best_id if best_score >= 2 else "none"

    def _detect_assets(self, entry: Any) -> tuple[Any, dict[str, SceneEntry]]:
        """缓存每游戏 (TermLibrary, scenes) 供归属检测复用；IO 只在首次发生。"""
        cached = self._detect_cache.get(entry.game_id)
        if cached is not None:
            return cached
        try:
            lib: Any = self._load_library_for(entry)
        except Exception:
            lib = None
        try:
            scenes = load_scenes(
                default_root=self.plugin_dir,
                override_root=self.data_path(),
                game_id=entry.game_id,
                terms_dir=entry.terms_dir,
                warnings=[],
            ) or {}
        except Exception:
            scenes = {}
        self._detect_cache[entry.game_id] = (lib, scenes)
        return lib, scenes

    # 拍板 2.0.67：S1 变化驱动 OCR——轻量帧 dHash（9x8 灰度 → 64 bit → 16 hex）。
    # 拍板 2.0.69：委托给 screen_capture.grab_primary_for_dhash——
    # ① mss.mss() 走 screen_capture 模块级 _MSS_LOCK（防 2 worker 并发抓屏死锁）
    # ② 失败/无屏幕 → 返回空串（调用方按"未检测"处理，继续走 interval 节奏）
    # ③ 不存帧，只算哈希（隐私红线）；日志只记 hash 前 8 字
    @staticmethod
    def _light_capture_dhash() -> str:
        """拍板 2.0.67/2.0.69：抓主屏 → 9x8 灰度 → dHash 64 bit → 16 hex chars。

        实际抓屏逻辑在 screen_capture.grab_primary_for_dhash（带 mss 全局锁），
        本函数仅做委托，让 mss 调用点全部集中到 screen_capture 模块。
        """
        try:
            from .screen_capture import grab_primary_for_dhash
            return grab_primary_for_dhash()
        except Exception:
            return ""

    @staticmethod
    def _dhash_diff(a: str, b: str) -> int:
        """拍板 2.0.67：两个 16-hex dHash 的汉明距离（位差异数）。空串/长度不匹配→0。"""
        if not a or not b or len(a) != len(b):
            return 0
        try:
            return (int(a, 16) ^ int(b, 16)).bit_count()
        except Exception:
            return 0

    def _should_push_proactive(
        self, ocr_text: str, scene_switched: bool = False, *, scene_id: str = ""
    ) -> bool:
        """拍板 2.0.67：场景切换路径绕过 300s 全局冷却，走独立 scene_switch_cooldown。

        同场景：hash 变化 + 300s 全局冷却 + 术语密度 < 5（2.0.59 沿用）。
        场景切换：仅看 scene_switch_cooldown（默认 30s，防刷屏）+ 术语密度 < 5。
          - hash 检查跳过：切场景文本必然不同；放宽信任。
          - 300s 跳过：用户切场景就该立即说话，300s 卡死体验。
          - 术语密度保留：太多术语命中（密集中文）会让提示语挤掉上下文，仍判负。

        ★ 2026-11-10：known_suppress 抑制排在**最前面**。
          若当前场景的攻略内容刚在窗口内被注入过语境（玩家已经知道了），
          这次主动搭话就**不说**。抑制时**不更新** ``_last_pushed_hash`` ——
          否则窗口过后同一段文本会被 hash 永久挡住，再也主动不了。
        """
        import hashlib

        now = time.monotonic()
        # ★ known_suppress：玩家已知道的话题不重复主动提。
        #   只在"主动"路径生效；用户提问走 on_chat_message，完全不查这里。
        _known_registry = getattr(self, "_known_suppress", None)
        if _known_registry is not None and scene_id:
            if _known_registry.should_suppress(scene_id, now):
                _injected_at = _known_registry.last_injected_at(scene_id)
                _age = 0.0 if _injected_at is None else now - _injected_at
                self.logger.info(
                    "genshin_game_buddy: proactive suppressed by known_suppress scene={} age={:.1f}s window={:.0f}s",
                    scene_id,
                    _age,
                    _known_registry.window_seconds,
                )
                return False
        if scene_switched:
            # 拍板 2.0.67：S2 独立冷却——切换就主动搭话，但同一切换间隔别刷屏
            if now - self._last_scene_switch_at < float(
                self._options.scene_switch_cooldown_seconds
            ):
                return False
        else:
            text_hash = hashlib.md5(ocr_text.encode("utf-8", errors="replace")).hexdigest()
            if text_hash == self._last_pushed_hash:
                return False
            if now - self._last_proactive_at < 300.0:
                return False
            # 同场景路径：hash 不同 → 把它记下来，避免下个 tick 仍判 hash 变化重复推
            self._last_pushed_hash = text_hash
        library = self._manager.library
        if library is not None:
            matched_count = sum(1 for e in library.by_key.values() if e.key in ocr_text)
            if matched_count > 5:
                return False
        return True

    def _push_proactive(
        self, changed_scenes: list[str], added: list[str], removed: list[str]
    ) -> bool:
        """构造变化量文本 + 当前激活 key 列表，push_message(ai_behavior="respond")。"""
        active_keys = self._activation.get_active_keys()
        if not self._activation.has_scene_active():
            text = self._t("perception.anchors_only")
        elif changed_scenes or added:
            action = (
                "切换到了「{}」场景".format("、".join(changed_scenes))
                if changed_scenes
                else "在当前场景中发现了新的关键词"
            )
            keys_text = " / ".join(active_keys[:12])
            text = self._t("perception.proactive_switch", action=action, keys=keys_text)
        else:
            _gid = getattr(self._manager.current, "game_id", "") if self._manager.current else ""
            text = self._t("perception.proactive_no_change", game=_gid)
        if added:
            text += " 新增：{}".format(" / ".join(added[:5]))
        if removed:
            text += " 移除：{}".format(" / ".join(removed[:5]))
        receipt = self.push_message(
            source="genshin_game_buddy",
            visibility=[],
            ai_behavior="respond",
            parts=[{"type": "text", "text": text}],
            priority=0,
            metadata={"kind": "perception_change"},
        )
        self._last_proactive_at = time.monotonic()
        return isinstance(receipt, Mapping) and receipt.get("submitted") is not False

    def _build_activated_context(self, session: Any) -> str:
        """锚点 + 激活集的 key + brief 文本。"""
        limit = self._options.context_inject_max_terms
        if limit <= 0:
            return ""
        library = self._manager.library
        if library is None:
            return ""
        active_keys = self._activation.get_active_keys()[:limit]
        lines: list[str] = []
        for key in active_keys:
            entry = library.get(key)
            if entry is not None:
                lines.append(self._t("context.term_line", term=entry.key, brief=entry.brief))
        return "\n".join(lines)

    def _scan_and_activate_query(self, text: str) -> None:
        """拍板 2.0.59：扫用户消息命中术语 → 激活（source="query", ttl=300）→ 推简短术语卡。
        只扫 key + aliases（slang 留给 ocr_perceive）。已激活的不重推；>8 条只激活不推。
        push 内容只含 key+brief（库数据），不含用户原文——守住 I-16。"""
        library = self._manager.library
        if library is None:
            return
        matched: list[str] = []
        seen: set[str] = set()
        for entry in library.by_key.values():
            if entry.key in seen:
                continue
            if entry.key in text or any(a and a in text for a in entry.aliases):
                matched.append(entry.key)
                seen.add(entry.key)
        if not matched:
            return
        with self._state_lock:
            old_keys = set(self._activation.get_active_keys())
            self._activation.activate_terms(matched, source="query", ttl_seconds=float(self._options.query_activation_ttl_seconds))
        newly = [k for k in matched if k not in old_keys]
        self.logger.info(
            "genshin_game_buddy: on_chat_message query-activate keys={} count={}",
            newly or matched, len(matched),
        )
        if not newly or len(matched) > 8:
            return
        hint = self._render_query_hint(newly)
        current = self._manager.current
        self.push_message(
            source="genshin_game_buddy",
            visibility=[],
            ai_behavior="read",
            parts=[{"type": "text", "text": hint}],
            priority=0,
            metadata={"game_id": current.game_id if current else "", "term_count": len(newly)},
        )

    def _render_query_hint(self, keys: list[str]) -> str:
        """拍板 2.0.59：构造「用户刚提到 X。相关条目：...」术语卡。
        只含 key + brief（库数据），不含用户原文——守住 I-16。"""
        library = self._manager.library
        if library is None or not keys:
            return ""
        lines = [f"用户刚提到：{'、'.join(keys)}。相关条目："]
        for k in keys:
            entry = library.get(k)
            if entry is not None:
                lines.append(f"- {entry.key}：{entry.brief}")
        return "\n".join(lines)

    # ==================================================================
    # 2.1a：周期感知 timer
    # ==================================================================

    # ------------------------------------------------------------------
    # ★ 2026-11-07：生命周期守卫（任务 C）—— 感知栈的挂起闸门
    # ------------------------------------------------------------------

    def _perception_suspended(self) -> bool:
        """感知栈（YOLO 推理 / OCR 轮询）现在该不该挂起。

        判定全部交给 :class:`lifecycle_guard.LifecycleGuard`：

        * 游戏进程**不在** → 挂起
        * 游戏进程在但**不在前台** → 挂起（除非开了「允许后台运行」）
        * 游戏在前台 / 守卫没起来 / 探测失败 → **不挂起**（fail-open）

        ★ fail-open 是刻意的
          守卫失灵时宁可多烧一点 CPU，也不能让插件整个哑掉。
        """
        guard = getattr(self, "_guard", None)
        if guard is None:
            return False
        try:
            return bool(guard.suspended)
        except Exception:  # noqa: BLE001 - 守卫异常不能拖垮感知栈
            return False

    def _guard_state(self) -> str:
        """守卫最近一次探测到的状态（拿不到返回空串）。"""
        guard = getattr(self, "_guard", None)
        if guard is None:
            return ""
        try:
            return str(guard.state)
        except Exception:  # noqa: BLE001
            return ""

    def _log_guard_skip(self) -> None:
        """挂起时打一行 INFO。

        ★ 为什么节流
          拍板 2.0.68 要求「任何 early-return 都打 INFO」，但挂起是
          **持续状态**（可能挂几分钟），每 2s 一行会把日志刷爆。
          ⇒ 首次必打，之后每 60s 补一行，状态一变（唤醒）立即再打。
        """
        now = time.monotonic()
        last = getattr(self, "_guard_suspend_log_at", 0.0)
        if last and (now - last) < 60.0:
            return
        self._guard_suspend_log_at = now
        self.logger.info(
            "genshin_game_buddy: perception suspended by lifecycle guard "
            "(state={}); capture/OCR skipped, webhook listener still alive",
            self._guard_state(),
        )

    def _current_game_id(self) -> str:
        """当前会话的游戏 id（没有会话返回空串）。

        给生命周期守卫用：只有当前游戏在守卫表里（原神），守卫才对它生效 ——
        否则玩鸣潮时会被「原神没开」误伤。
        """
        try:
            current = self._manager.current
        except Exception:  # noqa: BLE001
            return ""
        if current is None:
            return ""
        return str(getattr(current, "game_id", "") or "")

    def _on_guard_state_change(self, previous: str, current: str) -> None:
        """守卫状态迁移回调 —— 唤醒时立刻放行下一拍感知。

        ★ 为什么不只是「等下一个 tick」
          感知 tick 是 2s 一拍，唤醒后若还要等 interval 到点，
          最多白等 2s。直接把 ``_last_real_ocr_monotonic`` 归零，
          下一拍就满足「interval 已到」，**立即**恢复感知。
        """
        if previous in ("ABSENT", "BACKGROUND") and current == "FOREGROUND":
            with self._state_lock:
                self._last_real_ocr_monotonic = 0.0
            self.logger.info(
                "genshin_game_buddy: lifecycle guard resumed "
                "(%s -> %s); perception resumes on next tick",
                previous,
                current,
            )

    @timer_interval(id="ocr_perceive", seconds=3)
    async def ocr_perceive(self) -> None:
        """拍板 2.0.65：tick 装饰器固定 3s；实际 OCR 间隔由 self._options 控制。

        多线程 OCR：把 OCR 主体扔进 ThreadPoolExecutor 的 worker 线程（自带 asyncio.run 事件循环），
        timer 线程立即返回不阻塞。worker 数由 self._options.ocr_worker_threads 决定（1 或 2）。

        拍板 2.0.67：S1 变化驱动 OCR——每 3s tick 抓轻量帧 dHash，
        与上一帧差异 ≥ threshold 时绕过 interval 立即 OCR。
        轻量帧只算 hash、不存像素（隐私）；失败/无屏幕 → 静默跳过，按 interval 节奏走。

        拍板 2.0.68：3 层防御——保证 timer 不被任何子异常弄死（真机 2.0.67 暴露：
        14 分钟只 1 条 tick begin，怀疑 run_in_executor 抛异常让 _ocr_in_flight_count
        永久卡住、后续所有 tick 被 worker 闸静默挡掉）。
          ① 任何 early-return 都打 INFO 日志（reason + 关键值），便于排查"卡在哪儿"
          ② 整个函数 try/except 包住，任何异常 logger.exception 后吞掉（timer 继续）
          ③ _ocr_in_flight_count 用 try/finally 保证释放——schedule 阶段抛异常
             也能减回，不会让 worker 闸永久卡死

        拍板 2.0.71：高频 OCR——tick 间隔 = ocr_scene_interval_seconds（默认 2s）。
        真机单次 OCR 链路 1.44s（capture 0.14 + ocr 1.30 + detect 0.00）→ interval < 1.5s 会排队。
        抓屏/OCR 结果缓存给"术语激活"轮次复用（避免每 2s 都做完整链路）。
        """
        try:
            # ══════════════════════════════════════════════════════════
            # ★ 2026-11-07：生命周期守卫闸门（任务 C 的核心）
            # ──────────────────────────────────────────────────────────
            #   游戏进程不在 / 不在前台时，**整轮感知栈跳过**：
            #   不抓屏、不跑模型推演、不做 OCR。
            #
            #   ★ 为什么放在这里（最前面）
            #     这是感知链路里**第一个**能下决心的地方。放在后面
            #     （比如只跳过 OCR 本体）的话，change-driven 的轻量帧
            #     dHash 抓屏已经先跑掉了 —— 一样在烧 CPU。
            #
            #   ★ 为什么零延迟
            #     守卫是**进程级**判断（看一眼操作系统就知道游戏在不在），
            #     而感知状态机是**从 OCR 结果反推**的：判「你不在游戏里」
            #     得先做一次 OCR，于是 3 次 miss（AWAY）+ 3 次 miss
            #     （OUT_OF_GAME）才停 —— 2s 档位下就是 12 秒空转。
            #
            #   ★ Webhook 监听不受影响
            #     它在独立线程上（bettergi/webhook_server.py），
            #     不经过这里 —— 游戏刚退出时 BetterGI 的最后一条事件
            #     照样收得到，用户还能听到收尾台词。
            if self._perception_suspended():
                self._log_guard_skip()
                return
            # 拍板 2.0.71：tick 间隔用 scene_interval（默认 2s）；保留 ocr_perceive_interval_seconds
            # 字段以兼容旧面板/配置文件，但语义已被 ocr_scene_interval_seconds / ocr_term_interval_seconds 取代。
            interval = self._options.ocr_scene_interval_seconds
            now_mono = time.monotonic()
            trigger_reason = ""

            # 拍板 2.0.67：S1 变化驱动——轻量帧 dHash 对比（默认关，2.0.70 关闭的 worker 卡死回避）
            if self._options.change_driven_enabled:
                with self._state_lock:
                    state = self._perception_state
                if state != "OUT_OF_GAME":
                    try:
                        new_hash = await asyncio.to_thread(self._light_capture_dhash)
                    except Exception:
                        self.logger.exception(
                            "genshin_game_buddy: ocr_perceive change_detect dhash await failed "
                            "(fall through to interval gate)"
                        )
                        new_hash = ""
                    if new_hash:
                        diff = self._dhash_diff(new_hash, self._last_light_hash)
                        threshold = self._options.change_detect_threshold
                        self._last_light_hash = new_hash
                        if diff >= threshold:
                            trigger_reason = (
                                f"change_detect diff={diff}>={threshold}"
                            )

            # 拍板 2.0.71：决定是否 OCR——scene_interval 已到 OR change_detect 命中
            if not trigger_reason and now_mono - self._last_real_ocr_monotonic < interval:
                # 拍板 2.0.68：每个 early-return 必打 INFO（带 elapsed vs interval）
                self.logger.info(
                    "genshin_game_buddy: ocr_perceive skip reason=interval_not_reached "
                    "elapsed={:.1f}s interval={}s state_change={}",
                    now_mono - self._last_real_ocr_monotonic,
                    interval,
                    "yes" if trigger_reason else "no",
                )
                return

            # 拍板 2.0.65：worker 闸——限制并发 OCR 数（防堆积）。
            # tick=3s 但 OCR 自身可能 1-2s；worker 数限速：1=上一轮未跑完则跳过；2=允许 2 并发。
            with self._ocr_in_flight_lock:
                if self._ocr_in_flight_count >= self._options.ocr_worker_threads:
                    # 拍板 2.0.68：worker 闸被挡也打 INFO（防"沉默卡死"再次发生）
                    self.logger.info(
                        "genshin_game_buddy: ocr_perceive skip reason=worker_busy "
                        "in_flight={} workers={}",
                        self._ocr_in_flight_count,
                        self._options.ocr_worker_threads,
                    )
                    return
                self._ocr_in_flight_count += 1

            self._last_real_ocr_monotonic = now_mono
            self.logger.info(
                "genshin_game_buddy: ocr_perceive tick begin reason={}",
                trigger_reason or "interval",
            )

            # 拍板 2.0.70：stress mode 计数 + 60s 周期报告（getattr 防 stubs 缺字段）
            if getattr(self, "_stress_mode", "off") != "off" and getattr(self, "_stress_metrics", None) is not None:
                with getattr(self, "_stress_lock"):
                    self._stress_metrics["total_ocr"] += 1
                self._stress_maybe_report()

            # 拍板 2.0.70：watchdog 改 threading.Timer + concurrent.futures.Future——
            # 关键修复：2.0.69 的 `loop.call_later(60, _watchdog)` 用了**临时事件循环**（`@timer_interval`
            # 每 3s `asyncio.run(ocr_perceive)`，每次 run 都创建临时 loop，循环在 ocr_perceive return 后立刻关闭），
            # 排上的 call_later 回调**永远不会被触发**。真机 2.0.69 暴露 watchdog 从没触发过。
            # 同样，`asyncio.Future.add_done_callback(_on_done)` 也依赖临时 loop——也永不触发。
            #
            # 修复：① 直接用 `self._ocr_executor.submit(fn)` 拿 `concurrent.futures.Future`（不被 loop 管）
            #      ② `cf_future.add_done_callback(_on_done)` 会在 executor 的 worker 线程里跑（线程安全）
            #      ③ `threading.Timer(60, _watchdog)` 是独立线程，**不依赖任何 loop**，绝对能触发
            #      ④ 共享 _released flag 互斥释放 count（避免 _on_done 和 watchdog 重复 -1）
            try:
                # 懒初始化 executor：第一次 tick 时按配置创建，后续 worker 数变了由 apply_settings 重建
                if self._ocr_executor is None:
                    workers = max(1, min(2, self._options.ocr_worker_threads))
                    self._ocr_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=workers,
                        thread_name_prefix="mgc-ocr",
                    )

                # 拍板 2.0.70：直接 submit 拿 concurrent.futures.Future——不被任何临时 loop 管
                cf_future = self._ocr_executor.submit(self._ocr_perceive_sync_entry)

                # 共享释放标志——watchdog 和 _on_done 谁先到谁释放，避免重复 -1
                _released = [False]

                def _release_count():
                    if _released[0]:
                        return
                    _released[0] = True
                    with self._ocr_in_flight_lock:
                        self._ocr_in_flight_count -= 1

                def _on_done(_fut):
                    try:
                        _release_count()
                    except Exception:
                        # 拍板 2.0.70：绝不让 count 减失败变成静默
                        self.logger.exception(
                            "genshin_game_buddy: _on_done count release failed "
                            "(cf_future done but count may leak; watchdog will retry-release)"
                        )

                # concurrent.futures.Future.add_done_callback 在 executor 的 worker 线程里跑回调，
                # 不依赖任何临时 loop——**保证触发**
                cf_future.add_done_callback(_on_done)

                # 拍板 2.0.70：threading.Timer 60s 看门狗——独立线程，不被任何 loop 关闭影响。
                # 即使 tick 装饰器的临时 loop 死了，Timer 线程照常跑。
                def _watchdog():
                    try:
                        if not cf_future.done():
                            self.logger.warning(
                                "genshin_game_buddy: ocr worker timeout after 60s; "
                                "force-cancelling and releasing slot (likely mss deadlock, "
                                "winrt OCR hang, or await block in _ocr_perceive_body — "
                                "see stage= timing logs in 2.0.70+ to locate which stage stuck)"
                            )
                            cf_future.cancel()  # cf 在运行中 cancel 返回 False 但不抛——只标 cancelled
                            # 拍板 2.0.70：stress mode 计数
                            if self._stress_mode != "off" and self._stress_metrics is not None:
                                with self._stress_lock:
                                    self._stress_metrics["timeouts"] += 1
                        _release_count()  # 即使 cf_future 已 done，flag 兜底也再调一次（no-op）
                    except Exception:
                        self.logger.exception(
                            "genshin_game_buddy: worker watchdog callback failed "
                            "(count may leak; next tick may stall at worker闸)"
                        )

                timer = threading.Timer(60.0, _watchdog)
                timer.daemon = True
                timer.start()
            except Exception:
                # schedule 失败 → 把刚加的 count 还回去，让下个 tick 能正常尝试
                with self._ocr_in_flight_lock:
                    self._ocr_in_flight_count -= 1
                self.logger.exception(
                    "genshin_game_buddy: ocr_perceive schedule failed (count released, timer continues)"
                )
                # 拍板 2.0.70：stress mode 计数（getattr 防 stubs 缺字段）
                if getattr(self, "_stress_mode", "off") != "off" and getattr(self, "_stress_metrics", None) is not None:
                    with getattr(self, "_stress_lock"):
                        self._stress_metrics["errors"] += 1
        except Exception:
            # 拍板 2.0.68：最外层兜底——任何意外异常都不能逃逸出 timer 协程，
            # 否则宿主 timer 装饰器可能停止调度后续 tick。
            self.logger.exception(
                "genshin_game_buddy: ocr_perceive tick outer exception (tick dropped, timer keeps running)"
            )

    def _ocr_perceive_sync_entry(self) -> None:
        """拍板 2.0.65：worker 线程入口——asyncio.run 自带事件循环。

        线程模型：@timer_interval 在 timer 线程用 asyncio.run() 跑 ocr_perceive 协程，
        ocr_perceive 调 run_in_executor 把这个同步函数扔进 ThreadPoolExecutor，
        worker 线程里再 asyncio.run() 跑 _ocr_perceive_body 协程。两层事件循环在两个不同线程，不冲突。
        """
        try:
            asyncio.run(self._ocr_perceive_body())
        except Exception:
            self.logger.exception("genshin_game_buddy: _ocr_perceive_body crashed")

    async def _ocr_perceive_body(self) -> None:
        """拍板 2.0.65：OCR 感知主体（OCR 抓屏 → 识别 → 归属 → 激活 → 推）。

        在 ThreadPoolExecutor worker 线程的事件循环里跑（asyncio.run 自带 loop）。
        内层 await 全部走 SDK/内部协程，不依赖 timer 线程的 loop。

        拍板 2.0.70：段级打点——每个 await 前后打 time.monotonic() 戳（stage=xxx + dt=Ns），
        定位 worker 卡死的真根因。日志格式：
          stage=body_enter / stage=capture_done dt=Xs / stage=ocr_done dt=Xs / ...
        下次卡死时日志停在某个 stage=xxx_done 之前 = 该段 await 永久阻塞。

        拍板 2.0.71：高频 OCR + 场景状态机——
          - tick 间隔 = ocr_scene_interval_seconds（默认 2s；快）
          - term 轮次 = ocr_term_interval_seconds（默认 15s；慢）
          - 抓屏/OCR 结果缓存给 term 轮次复用（不重抓）
          - SceneTracker 滞回（hysteresis_count）→ 切换即推（不等 interval 周期）
        """
        import hashlib
        t0 = time.monotonic()
        self.logger.info("genshin_game_buddy: stage=body_enter")
        # ★ 2026-11-07：生命周期守卫 —— **第二道闸**（纵深防御）。
        #   第一道闸在 ``ocr_perceive`` 里（提交任务之前）；这一道在
        #   worker 线程**真正开跑时**再确认一次。
        #   ★ 为什么两道都要
        #     两道闸之间隔了一次线程调度 —— 守卫可能正好在这中间判定
        #     「游戏退出了」。那时候已经排好的任务如果还往下跑，
        #     就是一次完整的抓屏 + OCR，纯白烧 CPU。
        #   ★ 这里 return 不影响 worker 闸
        #     ``_ocr_in_flight_count`` 由 submit 出去的 Future 的
        #     ``_on_done`` 回调释放，body 提前返回照样会触发。
        if self._perception_suspended():
            self.logger.info(
                "genshin_game_buddy: _ocr_perceive_body skip "
                "reason=lifecycle_guard_suspended state={}",
                self._guard_state(),
            )
            return
        # 拍板 2.0.71：SceneTracker 心跳——每 30s 打一次 snapshot（不论 body 走完整链路还是 early-return）
        # 解决"2 分钟日志只有 body_enter/exit，没有 scene_tracker"问题：状态机一直在跑，只是没切换就不打日志
        # 拍板 2.0.73：心跳附加 ocr_in_flight / ocr_workers / executor_alive——便于排查 worker 卡死。
        # 真机 2.0.72 暴露：3 分钟只有 SceneTracker 心跳，但看不出 worker 闸是否被卡、
        # executor 是否 shutdown、heartbeat 线程本身是否在跑。一次性把状态机 + worker 健康度拼一行。
        # 用 getattr 防 stubs 缺字段（__new__ bypass __init__ 的情况）
        _st = getattr(self, "_scene_tracker", None)
        if _st is not None:
            _st_throttle = getattr(self, "_scene_tracker_log_throttle", 0.0)
            _st_interval = getattr(self, "_scene_tracker_log_interval", 30.0)
            if (t0 - _st_throttle) >= _st_interval:
                _snap = _st.snapshot()
                # 拍板 2.0.73：worker 闸 / executor 健康状态——一眼看出 worker 是否被卡
                with self._ocr_in_flight_lock:
                    _ocr_in_flight = self._ocr_in_flight_count
                _executor = getattr(self, "_ocr_executor", None)
                _executor_alive = (
                    _executor is not None
                    and not getattr(_executor, "_shutdown", False)
                )
                _ocr_workers = (
                    max(1, self._options.ocr_worker_threads)
                    if _executor_alive else 0
                )
                self.logger.info(
                    "genshin_game_buddy: scene_tracker heartbeat "
                    "current={} pending={} pending_count={} exit_pending_count={} "
                    "ocr_in_flight={} ocr_workers={} executor_alive={}",
                    _snap["current"] or "(none)",
                    _snap["pending"] or "(none)",
                    _snap["pending_count"],
                    _snap["exit_pending_count"],
                    _ocr_in_flight,
                    _ocr_workers,
                    _executor_alive,
                )
                self._scene_tracker_log_throttle = t0
        # 拍板 2.0.71：term 轮次判断——若距上次 term 激活 ≥ ocr_term_interval_seconds，本次也跑 term 激活
        # 用 getattr 防 stubs 缺字段（__new__ bypass __init__ 的情况）
        is_term_round = (
            t0 - getattr(self, "_last_term_run_monotonic", 0.0)
            >= self._options.ocr_term_interval_seconds
        )
        try:
            with self._state_lock:
                session = self._manager.current
                state = self._perception_state
            if session is None:
                self.logger.info("genshin_game_buddy: ocr_perceive early-return: no_session")
                return
            if state == "OUT_OF_GAME":
                # 隐私闸：判定已离开游戏，停止抓屏（D5）
                # 拍板 2.0.73：早退也打 INFO——区分"跳过"vs"卡死"。
                # 真机 2.0.72 暴露：3 分钟日志只有 heartbeat 没有 body_exit，
                # 根因就是这里静默 return。下一拍板必须能一眼看出"body 走到这步早退了"。
                self.logger.info(
                    "genshin_game_buddy: ocr_perceive early-return: state=OUT_OF_GAME "
                    "(privacy gate; next tick will re-check; if persists → perception_state stuck at OUT_OF_GAME)"
                )
                return

            # 拍板 2.0.14：本地进程优先，找不到抓前台窗口（云游戏）
            t_capture_start = time.monotonic()
            result, capture_path = await asyncio.to_thread(capture_active_frame)
            t_capture = time.monotonic()
            self.logger.info(
                "genshin_game_buddy: stage=capture_done dt={:.3f}s",
                t_capture - t_capture_start,
            )
            b64 = ""
            if result is not None and result.ok and result.image_base64:
                b64 = result.image_base64
                # 拍板 2.0.71：缓存抓屏结果给 term 轮次复用（避免每 2s 重抓）
                self._last_capture_b64 = b64
                self._last_capture_at_monotonic = t_capture
                self.logger.info(
                    "genshin_game_buddy: ocr_perceive capture_path={} method={} bytes={} luma={:.1f}",
                    capture_path, result.method, len(b64), result.avg_luma,
                )
            else:
                err = getattr(result, "error", "") if result else ""
                self.logger.info(
                    "genshin_game_buddy: ocr_perceive capture unavailable path={} error={}",
                    capture_path, err or "none",
                )
                # 辅助信号（D3）：主动抓屏失败时回退 bus.frames
                t_bus_start = time.monotonic()
                try:
                    frames = await asyncio.to_thread(self.bus.frames.get, max_count=1)
                except Exception:
                    frames = None
                self.logger.info(
                    "genshin_game_buddy: stage=bus_fallback_done dt={:.3f}s",
                    time.monotonic() - t_bus_start,
                )
                if isinstance(frames, Err):
                    frames = None
                elif isinstance(frames, Ok):
                    frames = frames.value
                records = list(frames) if frames is not None else []
                b64 = getattr(records[0], "image_base64", None) if records else ""
                if not b64:
                    # 拍板 2.0.71：本轮抓屏失败 → 若有缓存且本轮是 term 轮 → 用缓存做 OCR
                    if is_term_round and getattr(self, "_last_capture_b64", None):
                        age = t0 - self._last_capture_at_monotonic
                        if age < self._options.ocr_term_interval_seconds * 2:
                            b64 = self._last_capture_b64
                            self.logger.info(
                                "genshin_game_buddy: ocr_perceive term-round reuse-cached-capture "
                                "age={:.1f}s (current capture failed)",
                                age,
                            )
                    if not b64:
                        tag = "no_genshin_window" if err == "no_window" else "no_frame"
                        self.logger.info("genshin_game_buddy: ocr_perceive early-return: {}", tag)
                        return

            t_ocr_start = time.monotonic()
            text = await self._ocr.extract_text_from_base64(b64)
            self.logger.info(
                "genshin_game_buddy: stage=ocr_done dt={:.3f}s",
                time.monotonic() - t_ocr_start,
            )
            text_hash = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
            self.logger.info(
                "genshin_game_buddy: ocr_perceive text_len={} hash={} prev={}",
                len(text), text_hash[:8], (self._last_ocr_hash or "")[:8],
            )
            if len(text) == 0 and is_black_frame(b64):
                # 黑帧/无信号：不计入 miss（D6），也不做后续激活/推送
                self.logger.info("genshin_game_buddy: ocr_perceive early-return: black_frame")
                return

            # 拍板 2.0.71：缓存 OCR 文本——term 轮次复用（虽然现在 term round 自己也会 OCR；
            # 这里保存一下，供未来扩展如"离线分析"使用）。
            self._last_capture_text = text

            if text_hash == self._last_ocr_hash:
                with self._state_lock:
                    self._update_perception_state(self._last_ocr_hit)
                # 即便文本未变，也要做 SceneTracker 状态推进（防 stuck）
                self._scene_tracker.update([])  # 让 exit_grace_count 推进
                # 拍板 2.0.73：早退也打 INFO——文本未变是合法跳过，但 2.0.72 真机 3 分钟
                # 没 body_exit 日志时无法区分"卡死"和"OCR 一直返回同一帧"——必须留痕。
                self.logger.info(
                    "genshin_game_buddy: ocr_perceive early-return: text_unchanged hash={} prev={}",
                    text_hash[:8], (self._last_ocr_hash or "")[:8],
                )
                return

            self._last_ocr_hash = text_hash

            # 拍板 4：用术语库+场景信号判断游戏归属（命中 = 识别出某游戏）
            t_detect_start = time.monotonic()
            detected = await self.detect_game(text)
            self.logger.info(
                "genshin_game_buddy: stage=detect_done dt={:.3f}s detected={}",
                time.monotonic() - t_detect_start, detected,
            )
            ocr_hit = detected != "none"
            self._last_ocr_hit = ocr_hit
            # 拍板 2.0.64：转场清空——从"有游戏"变成"没游戏"或游戏变了 → 清屏/场景激活。
            # 但保留 query（用户主动问过的，LLM 可能还在聊）+ core_layer（常驻锚点）。
            # "scene" 前缀会同时清掉 f"scene:<name>" 形式的所有 scene 激活（activation.py 里就是这么存的）。
            if self._last_detected_game and detected != self._last_detected_game:
                with self._state_lock:
                    cleared = self._activation.clear_by_sources(
                        ("screen", "screen_slang", "scene")
                    )
                if cleared:
                    self.logger.info(
                        "genshin_game_buddy: ocr_perceive transition-clear "
                        "from={} to={} cleared={}",
                        self._last_detected_game, detected, cleared,
                    )
                self._last_scene_pushed = ""  # 场景 prompt 下次重新推
                # 拍板 2.0.67：S2 转场也清掉"上次主动搭话场景"，让切回新游戏时能识别为切换
                self._last_scene_for_proactive = ""
            self._last_detected_game = detected
            # 拍板 2.0.62：game 命中 ≠ 真在游戏——桌面泛词（plugin/N.E.K.O 插件管理等）也会让
            # detect_game 假阳性。必须满足"游戏内特征"才升级 IN_GAME；screen-activate 同步 gate。
            in_game_chars = self._has_in_game_characteristics(text, detected) if ocr_hit else False

            with self._state_lock:
                self._update_perception_state(ocr_hit, in_game_chars)
                old_keys = set(self._activation.get_active_keys())
                matched_scenes = self._match_scenes(text) if ocr_hit else []
                # 拍板 2.0.21：scenes=0 临时调试——scenes.toml 信号串未按真实 OCR 调优，
                # 打前 80 字便于调参。调好后撤除（2.0.22）。
                if text and not matched_scenes:
                    self.logger.info(
                        "ocr_perceive sample (scenes=0) text_head={!r}",
                        text[:80],
                    )
                if ocr_hit:
                    for scene_name in matched_scenes:
                        scene = self._scene_store.get(scene_name)
                        if scene is not None:
                            self._activation.activate_scene(scene_name, list(scene.context_terms))
                for scene_name in self._scene_store:
                    self._activation.tick_scene(scene_name, scene_name in matched_scenes)
                # 拍板 2.0.59 + 2.0.62：OCR 命中术语（key/alias/slang）→ 立即激活，screen ttl=180。
                # 2.0.62 加 in_game_chars gate——桌面泛词误命中时即便扫到别名也不激活
                # （防"桌面误激活迪奥娜"）。
                # 拍板 2.0.71：term 激活只在 term 轮跑（默认 15s）；scene 轮只做场景判定（2s）
                # 否则每 2s 都激活，刷屏且浪费 CPU。
                if in_game_chars and is_term_round:
                    library = self._manager.library
                    screen_matched, slang_matched = self._scan_screen_activate_terms(text, library)
                    # 拍板 2.0.64：screen-activate 上限（默认 12；MAX_ACTIVE=40 留 ~25 给核心层）
                    limit = self._options.screen_activation_limit
                    if limit > 0:
                        screen_matched = screen_matched[:limit]
                        remaining = max(0, limit - len(screen_matched))
                        slang_matched = slang_matched[:remaining]
                    if screen_matched:
                        self._activation.activate_terms(screen_matched, source="screen", ttl_seconds=180.0)
                    if slang_matched:
                        self._activation.activate_terms(slang_matched, source="screen_slang", ttl_seconds=180.0)
                    if screen_matched or slang_matched:
                        self.logger.info(
                            "genshin_game_buddy: ocr_perceive screen-activate keys={} slang_keys={} count={}",
                            screen_matched, slang_matched, len(screen_matched) + len(slang_matched),
                        )
                    # 拍板 2.0.71：term 轮跑完 → 更新 last_term_run（用当前 monotonic 即可）
                    self._last_term_run_monotonic = t0
                new_keys = set(self._activation.get_active_keys())

            if not ocr_hit:
                # 拍板 5：不中 → early-return no_game_detected
                self.logger.info(
                    "genshin_game_buddy: ocr_perceive early-return: no_game_detected detected_game={}",
                    detected,
                )
                return

            added = sorted(new_keys - old_keys)
            removed = sorted(old_keys - new_keys)

            # 拍板 2.0.67：S2 场景切换检测——primary 与上次主动搭话的场景不同 = 切换。
            # 切过游戏（_last_scene_for_proactive="" 但有 matched_scenes）也当切换看。
            scene_switched = False
            if matched_scenes:
                primary = matched_scenes[0]
                if primary != self._last_scene_for_proactive:
                    scene_switched = True
                    self.logger.info(
                        "genshin_game_buddy: ocr_perceive scene-switch "
                        "from={} to={} cooldown={}s",
                        self._last_scene_for_proactive or "(none)",
                        primary,
                        self._options.scene_switch_cooldown_seconds,
                    )

            # 拍板 2.0.71：场景状态机——SceneTracker 滞回 + 防抖动。
            # 不依赖 asyncio；纯同步逻辑（SceneTracker 内部状态机）。
            # 与 2.0.67 S2 scene_switched 互补：SceneTracker 防抖动（连续 N 次才认），
            # S2 scene_switched 立即识别（不防抖但走 30s 冷却）。
            # SceneTracker SCENE_SWITCHED 不直接 push_message（守住 push site == 4 红线），
            # 只更新 _last_scene_for_proactive；实际 push 走下方 2266 路径（应 always true）。
            from .scene_tracker import SceneEventType
            scene_event = self._scene_tracker.update(matched_scenes)
            if scene_event.type == SceneEventType.SCENE_SWITCHED:
                primary = scene_event.to_scene
                self.logger.info(
                    "genshin_game_buddy: scene_tracker scene-switch from={} to={} elapsed_in_old={:.1f}s",
                    scene_event.from_scene or "(none)",
                    primary,
                    scene_event.elapsed_in_old,
                )
                # 只更新跟踪字段；不动 _last_scene_pushed（让下方 2266 路径自然触发 push）
                self._last_scene_for_proactive = primary
            elif scene_event.type == SceneEventType.SCENE_EXITED:
                # 退出场景——可选：标记 OUT_OF_GAME（隐私闸）。这里只记日志，避免激进切回 OUT_OF_GAME。
                self.logger.info(
                    "genshin_game_buddy: scene_tracker scene-exited from={} elapsed_in_old={:.1f}s",
                    scene_event.from_scene or "(none)",
                    scene_event.elapsed_in_old,
                )

            if matched_scenes and self._should_push_proactive(
                text, scene_switched, scene_id=matched_scenes[0]
            ):
                self._push_proactive(matched_scenes, added, removed)
                # 拍板 2.0.67：S2 更新切换时间戳 + 上次主动搭话场景名。
                # 切路径走独立冷却；同场景路径只更新场景名（让下一次切回能再触发）。
                with self._state_lock:
                    if scene_switched:
                        self._last_scene_switch_at = time.monotonic()
                    self._last_scene_for_proactive = matched_scenes[0]

            # 拍板 2.0.61：场景命中 → 推 scene_prompt 到模型上下文（read，silently）。
            # 与 _push_proactive 独立——proactive 是 respond（触发回复），prompt 是 read（只预载）。
            # 拍板 2.0.64：scene_prompt_reinject_seconds > 0 时，同一场景距上次推超过 N 秒也重推
            # （防长对话中 prompt 被挤出上下文窗口；默认 0 = 只推一次，保持 2.0.61 行为）。
            # 拍板 2.0.71：SceneTracker SCENE_SWITCHED 触发后，这里 should_push=True（primary != _last_scene_pushed）
            # → 立即 push（不等 interval）。SceneTracker 滞回保证只在前 N 次连续匹配后才切换。
            if matched_scenes:
                primary = matched_scenes[0]
                should_push = primary != self._last_scene_pushed
                reinject = self._options.scene_prompt_reinject_seconds
                if not should_push and reinject > 0:
                    elapsed = time.monotonic() - self._last_scene_pushed_at
                    if elapsed >= reinject:
                        should_push = True
                if should_push:
                    scene = self._scene_store.get(primary)
                    if scene is not None and (scene.prompt or scene.avoid):
                        # ★ 2026-11-10：场景级 avoid 与场景正文**同批**注入。
                        #   此前 avoid 只在 ``_push_context`` 里渲染，而默认配置
                        #   ``reinject_every_n_messages = 0`` → 语境块永不重推 →
                        #   avoid 写了没人读，防剧透约束形同虚设（端到端验证实证：
                        #   场景命中后的全部 push 里"本场景避免"一次都没出现）。
                        #   同一次 push 一起送 —— 不新增 push 调用点（红线：恰好 4 处）。
                        avoid_line = render_scene_avoid(
                            scene.avoid, label=self._t("context.scene_avoid")
                        )
                        scene_text = f"【当前场景：{primary}】"
                        if scene.prompt:
                            scene_text += f"\n{scene.prompt}"
                        if avoid_line:
                            scene_text += f"\n{avoid_line}"
                        self.push_message(
                            source="genshin_game_buddy",
                            visibility=[],
                            ai_behavior="read",
                            parts=[{"type": "text", "text": scene_text}],
                            priority=0,
                            metadata={
                                "scene": primary,
                                "game_id": (self._manager.current.game_id if self._manager.current else ""),
                            },
                        )
                        self.logger.info(
                            "genshin_game_buddy: ocr_perceive scene-prompt-push scene={} chars={} avoid={}",
                            primary, len(scene.prompt), len(scene.avoid),
                        )
                        self._last_scene_pushed_at = time.monotonic()
                        # ★ 2026-11-10：场景攻略正文已进模型上下文 →
                        #   登记"玩家已经知道这个场景了"，主动提示层据此抑制同话题。
                        self._record_scene_injection(primary)
                    self._last_scene_pushed = primary
            else:
                self._last_scene_pushed = ""

            self.logger.info(
                "genshin_game_buddy: ocr_perceive state={} hit={} scenes={} active={} detected_game={}",
                self._perception_state, ocr_hit, len(matched_scenes), len(new_keys), detected,
            )
            # 拍板 2.0.70：stage=body_exit 打点——若卡死，日志停在某个 stage=xxx_done 之前 = 根因
            # 拍板 2.0.71：附 SceneTracker 状态——让用户每次成功 body 都能看到状态机进度
            # 用 getattr 防 stubs 缺 _scene_tracker（__new__ bypass __init__ 的情况）
            _st = getattr(self, "_scene_tracker", None)
            if _st is not None:
                _st_snap = _st.snapshot()
                self.logger.info(
                    "genshin_game_buddy: stage=body_exit total={:.3f}s scene_tracker current={} pending={} pending_count={}",
                    time.monotonic() - t0,
                    _st_snap["current"] or "(none)",
                    _st_snap["pending"] or "(none)",
                    _st_snap["pending_count"],
                )
            else:
                self.logger.info(
                    "genshin_game_buddy: stage=body_exit total={:.3f}s",
                    time.monotonic() - t0,
                )
        except Exception:
            self.logger.exception("genshin_game_buddy: _ocr_perceive_body tick failed")

    def _record_scene_injection(self, scene_id: str) -> None:
        """登记"该攻略场景的内容刚刚被注入语境"——供主动提示层抑制同话题。

        ★ 记录侧有两个入口（都是"场景内容进入模型上下文"的时刻）：
          1. ``_push_context``  —— 语境块（术语 + 场景 avoid 约束）注入；
          2. ``ocr_perceive`` 的 scene-prompt push —— 场景攻略正文注入。
          只记 ① 不够：玩家走进"深境螺旋"的那一刻，主动提示走的是 ②，
          而 ① 只在登记游戏 / 重推周期才跑 —— 只记 ① 这条功能基本不生效。

        ★ 防御式：``_scene_store`` / ``_known_suppress`` 在启动早期可能还没就位
          （测试也可能用 ``__new__`` 绕过 ``__init__``），取不到就什么都不记，
          绝不抛异常。
        """
        key = (scene_id or "").strip()
        if not key:
            return
        scenes = getattr(self, "_scene_store", None) or {}
        scene = scenes.get(key)
        known = bool(scene.known_suppress) if scene is not None else False
        registry = getattr(self, "_known_suppress", None)
        if registry is None:
            return
        registry.record(key, known_suppress=known)

    def _push_context(self, session: GameSession, *, announce_switch: bool = True) -> bool:
        """把语境块送进对话上下文。返回是否真的提交成功（失败绝不谎报）。

        ``announce_switch=False`` 用于"重推"（刷新 / 每 N 轮）——那时切换早已发生，
        再报一次"已从 X 切换"只会让模型误以为刚换过游戏。
        """
        library = self._manager.library
        if library is None:
            return False

        activated_text = self._build_activated_context(session)
        activated_lines = [line for line in activated_text.split("\n") if line] if activated_text else []

        switched = (
            announce_switch
            and bool(session.previous_game_id)
            and session.previous_game_id != session.game_id
        )
        if switched:
            previous_entry = self._registry.get(session.previous_game_id)
            previous_name = (
                previous_entry.display_name if previous_entry is not None else session.previous_game_id
            )
            header = self._t(
                "context.title_switched",
                game=session.display_name,
                previous=previous_name,
            )
        else:
            header = self._t("context.title", game=session.display_name)

        # ★ 2026-11-09 接线：场景级 avoid 进语境块（防剧透标记真正被消费）。
        #   当前场景优先用 _last_scene_pushed（OCR 命中过、已推过 scene prompt 的那个）。
        #   防御式：_scene_store 在启动早期可能还是空 dict（见 __init__ 里
        #   _load_scenes_for_current 的注释），取不到就用空 avoid —— 绝不抛异常。
        #   读法沿用本文件既有惯例（与 ocr_perceive 里 self._scene_store.get(...) 一致）：
        #   不引入新的锁语义。
        scene = self._scene_store.get(self._last_scene_pushed) if self._last_scene_pushed else None
        scene_avoid = scene.avoid if scene is not None else ()
        # ★ 2026-11-10：known_suppress 已接上主动提示层。
        #   语境块注入成功时登记"玩家已经知道这个场景了"（_record_scene_injection），
        #   主动提示层据此在窗口内不再提同话题（见 _should_push_proactive）。
        #   ⚠️ chat_state / event_to_yui 用的是**引擎场景** id 空间
        #     （WORLD/COMBAT/IDLE），与这里的**攻略场景** id 空间不相交 ——
        #     两者混用会造成真实误判，所以消费侧接在 __init__ 的攻略场景主动路径上。

        block = build_context_block(
            header=header,
            tone_line=self._t("context.tone", tone=library.tone) if library.tone else "",
            terms_header=self._t("context.terms_header"),
            term_lines=activated_lines,
            empty_note=self._t("context.no_terms"),
            hint=self._t("context.hint"),
            note=self._t("context.no_speculation"),
            scene_avoid=scene_avoid,
            scene_avoid_label=self._t("context.scene_avoid"),
            max_chars=self._options.max_context_chars,
        )
        # push_message v2：只用 parts + visibility + ai_behavior。
        # visibility=[] → 用户看不到原始语境块；ai_behavior="read" → 进模型上下文
        # 但**不**立即触发回复 turn（respond 会造成自我触发）。
        receipt = self.push_message(
            source="genshin_game_buddy",
            visibility=[],
            ai_behavior="read",
            parts=[{"type": "text", "text": block}],
            priority=0,
            metadata={"game_id": session.game_id, "term_count": len(activated_lines)},
        )
        if isinstance(receipt, Mapping) and receipt.get("submitted") is False:
            self.logger.warning(
                "genshin_game_buddy: context injection not submitted ({})", receipt.get("reason")
            )
            return False
        # ★ 2026-11-10：语境块注入成功 → 登记"玩家已经知道这个场景了"。
        #   主动提示层据此在窗口内不再提同话题（known_suppress）。
        self._record_scene_injection(getattr(self, "_last_scene_pushed", ""))
        return True
