"""genshin_game_buddy · BetterGI 桥接 —— 事件映射模块。

职责：把 BetterGI Webhook 事件码映射成一句台词。

★ 2026-11-07：从独立插件 ``bettergi_companion`` **物理并入** ``genshin_game_buddy``。
  逻辑逐字保留（只改了日志前缀与本段文档）。

设计约束
--------
* **零第三方依赖** —— 只用 Python 标准库（``tomllib`` / ``threading`` / ``time`` / ``random``）。
* **不 import autopilot 子包** —— 虽然本模块现在就在 ``genshin_game_buddy`` 里，
  仍刻意不碰 ``autopilot.*``：它要在**不装宿主**的情况下单独跑离线测试，
  保持零耦合比「少复制几行」重要。
* 配置从本目录内的 ``config.toml`` 读取（相对 ``__file__`` 定位，与安装位置无关）；
  文件缺失或损坏时**回落到内置默认映射**，绝不因配置问题让插件起不来。
* 限流（同一事件 N 秒内不重复）与防复读（同一句台词 N 秒内不重复）都在线程锁内完成，
  因为 HTTP 服务器是多线程的。
"""

from __future__ import annotations

import random
import threading
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "DEFAULT_CONFIG",
    "EventMapper",
    "load_config",
    "PLUGIN_DIR",
    "DEFAULT_CONFIG_PATH",
    # 2026-10-29 Companion Mode：场景组冷却
    "SCENE_GROUP", "A_EVENT_GROUP", "A_EVENT_TIER", "TIER_CRITICAL",
    "GROUP_WORLD", "GROUP_COMBAT", "GROUP_MENU", "GROUP_TRANSITION",
    "DEFAULT_SCENE_GROUP_COOLDOWN_S",
    "group_for_scene",
]

#: 插件目录（本文件所在目录）
PLUGIN_DIR = Path(__file__).resolve().parent

#: 默认配置文件位置
DEFAULT_CONFIG_PATH = PLUGIN_DIR / "config.toml"

#: 防复读窗口：同一句台词在这个秒数内不重复说
DEFAULT_ANTI_REPEAT_S = 30.0

#: 台词最大长度（防止 BetterGI 传来超长文本把对话刷屏）
MAX_LINE_LENGTH = 200

# ══════════════════════════════════════════════════════════════════════
# 场景组冷却（Companion Mode，2026-10-29）
# ══════════════════════════════════════════════════════════════════════
#: 同场景组的最小间隔（秒）。**非 tier1 台词**在同组内 90s 只准一句。
#:
#: ★ 为什么要有这一层
#:   限流（``cooldown_s``）是**按事件码**的 —— ``domain.start`` 连发 10 次
#:   压成 1 次，但 ``domain.start`` + ``domain.end`` + ``task.error`` 各来一次
#:   就是三句，观众听到的是「开打 / 打完了 / 没成」连珠炮。
#:   场景组把粒度提到「同一段场景里」，这才叫「同场景不刷屏」。
#:
#: ★ tier1 不受此限
#:   关键异常不该被「上一句还没凉」压住 —— 漏报一次比多说一句严重得多。
#:
#: ★ 可调
#:   见 ``config.toml`` 的 ``[yui].scene_group_cooldown_s``（默认 90）。
#:   调到 0 = 关掉这一层（退回改造前的行为）。
DEFAULT_SCENE_GROUP_COOLDOWN_S = 90.0

#: 场景组。四组，与 B 轨 ``autopilot/ui/chat_state.py`` 完全一致。
GROUP_WORLD = "WORLD_GROUP"
GROUP_COMBAT = "COMBAT_GROUP"
GROUP_MENU = "MENU_GROUP"
GROUP_TRANSITION = "TRANSITION_GROUP"

#: 13 个真实场景 id -> 场景组。
#:
#: ★ 这里是**抄一份**，不是 import
#:   本模块刻意不 import ``autopilot.ui.chat_state``（见模块文档「设计约束」）——
#:   抄一份的代价是可能漂移，代价由 ``tests/integration/test_bettergi_bridge.py`` 里的
#:   ``test_scene_group_table_covers_all_thirteen_ids`` 钉住 ——
#:   抄可以，**不一致不行**。
#:
#: ★ 为什么 MENU 组有 7 个场景
#:   玩家在背包 / 地图 / 通知之间来回点时，共享一把 90s 锁 ——
#:   不该被播成「换场景了」「又换场景了」。
SCENE_GROUP: Dict[str, str] = {
    "WORLD": GROUP_WORLD,
    "COMBAT": GROUP_COMBAT,
    "COMBAT_RESULT": GROUP_COMBAT,
    "MAIN_MENU": GROUP_MENU,
    "DIALOG": GROUP_MENU,
    "BAG_INVENTORY": GROUP_MENU,
    "MAP_BIG": GROUP_MENU,
    "GAME_MENU": GROUP_MENU,
    "NOTICE": GROUP_MENU,
    "TELEPORT_MAP": GROUP_MENU,
    "LOADING": GROUP_TRANSITION,
    "CUTSCENE": GROUP_TRANSITION,
    "UNKNOWN": GROUP_TRANSITION,
}

#: A 轨事件码 -> 场景组。
#:
#: ★ 为什么 ``domain.*`` 和 ``task.error`` 归**同一组**
#:   ``_PROMPT_GUIDE.md`` §5 原文：「``domain.start`` 属于 ``COMBAT_GROUP``，
#:   连着来的 ``domain.start`` / ``task.error`` / ``domain.end``
#:   会被同一把 90s 锁压住。」——
#:   一次战斗就是「开打 ->（可能失败）-> 打完了」，
#:   它是**同一段场景**，不该被播成三句。
A_EVENT_GROUP: Dict[str, str] = {
    "domain.start": GROUP_COMBAT,
    "domain.end": GROUP_COMBAT,
    "task.error": GROUP_COMBAT,
    "js.custom": GROUP_WORLD,
    "notify.test": GROUP_WORLD,
}

#: A 轨事件码 -> 档位。**1 = 关键**（不受组冷却限制）。
#:
#: ★ 为什么 ``js.custom`` / ``notify.test`` 是关键档
#:   这两个都是**人主动触发**的：``js.custom`` 是用户自己推的台词，
#:   ``notify.test`` 是手动点「测试 WebHook 通知」按钮。
#:   用户要听就得听到 —— 被 90s 锁挡掉会让人以为通道坏了。
#:   （与 B 轨 ``COMPANION_LINE`` 同一套逻辑：用户推的台词按 tier1 走。）
#:
#: ★ 未登记的事件码按 **2** 算（非关键）
#:   宁可被锁一下，也不要让一个没人想过的新事件在耳边连响。
A_EVENT_TIER: Dict[str, int] = {
    "domain.start": 2,
    "domain.end": 2,
    "task.error": 2,
    "js.custom": 1,
    "notify.test": 1,
}

#: 关键档（tier 1）—— 不受场景组冷却限制。
TIER_CRITICAL = 1


def group_for_scene(scene_id: str) -> str:
    """场景 id -> 场景组。**不认识的一律落到 TRANSITION 组**。

    ★ 兜底方向与 B 轨一致
      场景组是「防刷屏」的，兜底到 TRANSITION 最保守 ——
      加载 / 过场本来就是「一句话都该少说」的地方。
    """
    scene = (scene_id or "").strip().upper()
    return SCENE_GROUP.get(scene, GROUP_TRANSITION)


def _extract_scene(data: Any) -> str:
    """从事件负载里抠出场景 id（如果有）。

    ★ 为什么要有这一步
      A 轨的事件码（``domain.*`` / ``task.error``）只覆盖战斗族，
      用户在 ``config.toml`` 里自己加的事件码则**不属于任何已知场景**。
      如果负载里带了 ``scene``（BetterGI 的自定义脚本可以塞任意字段），
      就能把它归到正确的组里，用上那张 13 场景表。
    """
    if isinstance(data, dict):
        for key in ("scene", "scene_id", "sceneId"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
    return ""

#: 内置默认配置 —— 当 ``config.toml`` 缺失/损坏时使用。
#: 结构与 ``config.toml`` 完全一致。
DEFAULT_CONFIG: Dict[str, Any] = {
    "webhook": {
        "host": "127.0.0.1",
        "port": 17890,
        "path": "/bettergi",
        "replay_window_s": 300,
    },
    # ── 反向控制（YUI → BetterGI）───────────────────────────────────
    # ★ 默认**开着** —— 反向控制已经是这个插件的核心功能，
    #   让用户先去翻 config.toml 才能用属于「配置地狱」。
    #   真没装 BetterGI 侧代理脚本也不要紧：启动自检会发现通道没连上，
    #   让 YUI 主动提醒用户跑一下「一键配置.bat」（见 __init__.py）。
    "control": {
        "enabled": True,
        "ack_timeout_s": 15,
        "poll_hold_s": 5,
        # 启动自检：等多久开始查 BetterGI 有没有连上来
        "self_check_delay_s": 15,
        # 没连上时每隔多久再提醒一次
        "self_check_repeat_s": 120,
        # 最多提醒几次（0 = 完全不提醒）
        "self_check_max": 2,
    },
    # ── 陪玩参数（Companion Mode，2026-10-29）────────────────────────
    "yui": {
        # 同场景组内非 tier1 台词的最小间隔（秒）。0 = 关掉这一层。
        "scene_group_cooldown_s": DEFAULT_SCENE_GROUP_COOLDOWN_S,
    },
    "events": {
        # ★ 台词按 ``_PROMPT_GUIDE.md`` §6.1 改写：每条至少 3 句随机池。
        #   原文是系统口吻（「开打开打，主人掩护我！」），新的是陪玩口吻。
        "domain.start": {
            "enabled": True,
            "lines": [
                "来了来了，主人尽管冲，我在旁边看着~",
                "打架了打架了！……主人小心点呀。",
                "开打！输了我可不哄你哦~",
                "自动打手上线啦，主人快找好位置看戏~",
            ],
            "cooldown_s": 10,
        },
        "domain.end": {
            "enabled": True,
            "lines": [
                "打完啦~ 主人歇会儿吧，喘口气。",
                "呼——总算结束了，主人喝水去。",
                "辛苦了笨蛋主人，我给你记着战绩呢。",
                "自动打手完工啦，主人快来验收战绩~",
            ],
            "cooldown_s": 10,
        },
        "task.error": {
            "enabled": True,
            "lines": [
                "哎呀，这次没成……主人再试一次嘛。",
                "唔……失败了。要不换个打法？",
                "又扑空了。主人是不是走神啦~",
                "哎呀，自动打手卡壳了，主人快去帮帮它。",
            ],
            "cooldown_s": 30,
        },
        "js.custom": {
            "enabled": True,
            "passthrough": True,
            "cooldown_s": 10,
        },
        "notify.test": {
            "enabled": True,
            "lines": ["收到收到，通道通了！"],
            "cooldown_s": 10,
        },
    },
}


def _deep_copy(obj: Any) -> Any:
    """深拷贝一份默认配置，避免调用方改到模块级常量。"""
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj


#: 一条事件规则里可能出现的键。用来区分「规则」和「规则表」。
_RULE_KEYS = ("enabled", "lines", "cooldown_s", "passthrough")


def _iter_event_rules(events: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """把 TOML 的 ``[events.*]`` 摊平成 ``(事件码, 规则)`` 列表。

    ★ 为什么要摊平
      TOML 里 ``[events.domain.start]`` 解析出来是**嵌套**的::

          {"events": {"domain": {"start": {...}}, "js": {"custom": {...}}}}

      而规则表期望的是**平铺**的::

          {"events": {"domain.start": {...}, "js.custom": {...}}}

      以前直接遍历 ``events.items()`` 的话，拿到的 ``event_name`` 是
      ``"domain"``、``rule`` 是 ``{"start": {...}}`` —— 两者都不是规则，
      于是**整个 ``[events.*]` 段一条都没读进来**。
      表现是「改了 config.toml 的台词没反应」，
      而因为内置默认与配置文件写着同样的内容，这个 bug 一直没被发现。

    ★ 两种写法都认
      · ``[events.domain.start]``   -> 嵌套表（摊平后 ``domain.start``）
      · ``[events."domain.start"]`` -> 平铺键（本来就是 ``domain.start``）
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, value in events.items():
        if not isinstance(value, dict):
            continue
        if any(key in value for key in _RULE_KEYS):
            out.append((str(name), value))
            continue
        # 不是规则本身 -> 是一张「规则表」，往下再走一层
        for sub_name, sub_rule in _iter_event_rules(value):
            out.append((f"{name}.{sub_name}", sub_rule))
    return out


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """读取 ``config.toml``；失败时返回内置默认配置。

    :param config_path: 配置文件路径，``None`` 表示用插件目录下的默认位置。
    :returns: 合并后的配置字典（默认值 + 文件覆盖）。
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    merged = _deep_copy(DEFAULT_CONFIG)
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return merged
    if not isinstance(raw, dict):
        return merged

    # [webhook]
    webhook = raw.get("webhook")
    if isinstance(webhook, dict):
        for key in ("host", "port", "path", "replay_window_s"):
            if key in webhook:
                merged["webhook"][key] = webhook[key]

    # [control]
    # ★ 这一段以前**根本没被读进来** —— 反向控制开关永远读不到，
    #   于是 `[control].enabled` 写 true 也等于没写。
    #   （`__init__.py` 的 startup 取的是 `self._mapper.config["control"]`，
    #     而这里以前只合并 [webhook] 和 [events.*]。）
    control = raw.get("control")
    if isinstance(control, dict):
        for key in ("enabled",):
            if key in control:
                merged["control"][key] = bool(control[key])
        for key in ("ack_timeout_s", "poll_hold_s", "self_check_delay_s", "self_check_repeat_s"):
            if key in control:
                try:
                    merged["control"][key] = float(control[key])
                except (TypeError, ValueError):
                    pass
        if "self_check_max" in control:
            try:
                merged["control"]["self_check_max"] = int(control["self_check_max"])
            except (TypeError, ValueError):
                pass

    # [yui] —— 陪玩参数（Companion Mode）
    yui = raw.get("yui")
    if isinstance(yui, dict):
        if "scene_group_cooldown_s" in yui:
            try:
                merged["yui"]["scene_group_cooldown_s"] = max(
                    0.0, float(yui["scene_group_cooldown_s"]))
            except (TypeError, ValueError):
                pass

    # [events.*]
    events = raw.get("events")
    if isinstance(events, dict):
        for event_name, rule in _iter_event_rules(events):
            if not isinstance(rule, dict):
                continue
            target = merged["events"].setdefault(event_name, {})
            if "enabled" in rule:
                target["enabled"] = bool(rule["enabled"])
            if "passthrough" in rule:
                target["passthrough"] = bool(rule["passthrough"])
            if "cooldown_s" in rule:
                try:
                    target["cooldown_s"] = float(rule["cooldown_s"])
                except (TypeError, ValueError):
                    pass
            lines = rule.get("lines")
            if isinstance(lines, list):
                cleaned = [str(x) for x in lines if isinstance(x, str) and x.strip()]
                if cleaned:
                    target["lines"] = cleaned
    return merged


def sanitize(text: Any) -> str:
    """把任意值清洗成一句干净的单行台词。"""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
    if len(text) > MAX_LINE_LENGTH:
        text = text[:MAX_LINE_LENGTH]
    return text


class EventMapper:
    """事件 → 台词 的映射器（含限流与防复读）。

    线程安全：所有状态变更都在 ``_lock`` 内完成。
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        *,
        anti_repeat_s: float = DEFAULT_ANTI_REPEAT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        :param config_path: 配置文件路径；``None`` 用插件目录下的 ``config.toml``。
        :param anti_repeat_s: 同一句台词的最小重复间隔（秒）。
        :param clock: 时钟函数，测试时可注入假时钟。
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = load_config(self.config_path)
        self.anti_repeat_s = float(anti_repeat_s)
        self._clock = clock
        self._lock = threading.Lock()
        # 事件 -> 最近一次触发时间（限流）
        self._last_event_at: Dict[str, float] = {}
        # 台词 -> 最近一次说出时间（防复读）
        self._last_line_at: Dict[str, float] = {}
        # 场景组 -> 最近一次说出非 tier1 台词的时间（场景组冷却）
        self._last_group_spoke: Dict[str, float] = {}
        # 当前场景 id。初值 "WORLD"（乐观）—— 与 B 轨状态机同一个约定。
        self._scene: str = "WORLD"
        # 统计信息，便于排查
        self.stats: Dict[str, int] = {
            "mapped": 0,
            "dropped_unknown_event": 0,
            "dropped_disabled": 0,
            "dropped_cooldown": 0,
            "dropped_anti_repeat": 0,
            "dropped_empty": 0,
            "dropped_scene_group": 0,
        }

    # ── 场景组冷却 ────────────────────────────────────────────────────
    @property
    def scene_group_cooldown_s(self) -> float:
        """同场景组内非 tier1 台词的最小间隔（秒）。0 = 关掉这一层。"""
        try:
            return max(0.0, float(
                self.config.get("yui", {}).get(
                    "scene_group_cooldown_s",
                    DEFAULT_SCENE_GROUP_COOLDOWN_S)))
        except (TypeError, ValueError):
            return DEFAULT_SCENE_GROUP_COOLDOWN_S

    @property
    def scene(self) -> str:
        """当前场景 id。"""
        return self._scene

    def scene_group_times(self) -> Dict[str, float]:
        """场景组冷却窗口的副本（诊断用）。"""
        with self._lock:
            return dict(self._last_group_spoke)

    def _group_for(self, event: str, data: Any) -> str:
        """这个事件属于哪个场景组。

        判定顺序：
        1. 负载里显式带了场景 id -> 走 13 场景表
        2. 事件码自带归属（``domain.*`` / ``task.error`` -> 战斗组）
        3. 兜底 -> 当前场景
        """
        scene = _extract_scene(data)
        if scene:
            self._scene = scene
            return group_for_scene(scene)
        group = A_EVENT_GROUP.get(event)
        if group is not None:
            # ★ 场景跟着事件走
            #   ``domain.start`` = 进入战斗，``domain.end`` = 回到大世界。
            #   这让「用户在 config.toml 里自己加的事件码」也能落到
            #   正确的组里（它们不在 :data:`A_EVENT_GROUP` 里，走兜底分支）。
            if event == "domain.start":
                self._scene = "COMBAT"
            elif event == "domain.end":
                self._scene = "WORLD"
            return group
        return group_for_scene(self._scene)

    # ── 配置读取 ──────────────────────────────────────────────────────
    @property
    def webhook_config(self) -> Dict[str, Any]:
        """``[webhook]`` 段（host / port / path）。"""
        return self.config.get("webhook", DEFAULT_CONFIG["webhook"])

    def event_rule(self, event: str) -> Optional[Dict[str, Any]]:
        """取某个事件码的规则；未登记返回 ``None``。"""
        events = self.config.get("events", {})
        rule = events.get(event)
        return rule if isinstance(rule, dict) else None

    def known_events(self) -> List[str]:
        """当前已登记的事件码列表。"""
        return sorted(self.config.get("events", {}).keys())

    # ── 核心映射 ──────────────────────────────────────────────────────
    def map_event(
        self,
        event: str,
        message: str = "",
        data: Any = None,
        result: Any = None,
    ) -> Optional[str]:
        """把一个 BetterGI 事件映射成一句台词。

        :param event: 事件码，例如 ``"domain.start"``。
        :param message: BetterGI 传来的原始消息。
        :param data: 事件附带的结构化数据。
        :param result: 任务结果（BetterGI 的 ``Success`` / ``Fail`` 等）。
        :returns: 要念的台词；不该念时返回 ``None``。
        """
        if not event or not isinstance(event, str):
            with self._lock:
                self.stats["dropped_unknown_event"] += 1
            return None

        rule = self.event_rule(event)
        if rule is None:
            # 没登记的事件一律忽略 —— BetterGI 以后加新事件码时不会乱播。
            with self._lock:
                self.stats["dropped_unknown_event"] += 1
            return None

        if not rule.get("enabled", True):
            with self._lock:
                self.stats["dropped_disabled"] += 1
            return None

        now = self._clock()

        with self._lock:
            # ① 限流：同一事件在 cooldown_s 内不重复
            cooldown_s = float(rule.get("cooldown_s", 10))
            last = self._last_event_at.get(event)
            if last is not None and (now - last) < cooldown_s:
                self.stats["dropped_cooldown"] += 1
                return None

            # ② 场景组冷却（Companion Mode）
            #   同一场景组内，非 tier1 台词 90s 只准一句。
            #   ★ tier1（用户主动推的 ``js.custom`` / 手动测试）不受此限。
            #   ★ 只查不占 —— 窗口在**真的开口之后**才推进。
            group = self._group_for(event, data)
            tier = A_EVENT_TIER.get(event, 2)
            group_cd = self.scene_group_cooldown_s
            if tier > TIER_CRITICAL and group_cd > 0.0:
                last_g = self._last_group_spoke.get(group)
                if last_g is not None and (now - last_g) < group_cd:
                    self.stats["dropped_scene_group"] += 1
                    return None

            # ③ 选台词
            if rule.get("passthrough"):
                text = sanitize(data) or sanitize(message)
            else:
                text = self._pick_line(rule.get("lines") or [], now)

            if not text:
                self.stats["dropped_empty"] += 1
                if text is None:
                    self.stats["dropped_anti_repeat"] += 1
                return None

            self._last_event_at[event] = now
            self._last_line_at[text] = now
            # ★ 场景组窗口**只在真的开口之后**推进
            #   被限流 / 被组冷却压住的事件不能宣称「刚播过」。
            if tier > TIER_CRITICAL and group_cd > 0.0:
                self._last_group_spoke[group] = now
            self.stats["mapped"] += 1
            return text

    def _pick_line(self, lines: List[str], now: float) -> Optional[str]:
        """从台词池里挑一句，避开防复读窗口内刚说过的。

        全部台词都在窗口内时返回 ``None`` —— 宁可不说，也不要复读。
        """
        pool = [sanitize(x) for x in lines if isinstance(x, str) and x.strip()]
        if not pool:
            return None
        fresh = [x for x in pool if (now - self._last_line_at.get(x, -1e18)) >= self.anti_repeat_s]
        if not fresh:
            return None
        return random.choice(fresh)

    # ── 维护 ──────────────────────────────────────────────────────────
    def reset_cooldowns(self) -> None:
        """清空限流/防复读/场景组冷却状态（测试与手动重置用）。"""
        with self._lock:
            self._last_event_at.clear()
            self._last_line_at.clear()
            self._last_group_spoke.clear()

    def snapshot(self) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        """返回 ``(统计, 事件时间表, 台词时间表)`` 的副本。"""
        with self._lock:
            return dict(self.stats), dict(self._last_event_at), dict(self._last_line_at)
