"""genshin_game_buddy · BetterGI 桥接 —— 适配层。

职责
----
把 :mod:`bettergi.webhook_server` / :mod:`bettergi.control` /
:mod:`bettergi.event_mapper` 这三个零依赖模块接到 NEKO 宿主上：
正向（BetterGI 事件 → 台词 → YUI 说话）与反向（YUI 指令 → BetterGI 干活）。

★ 2026-11-07：从独立插件 ``bettergi_companion`` **物理并入** ``genshin_game_buddy``。
  原先这些逻辑住在 ``bettergi_companion/__init__.py`` 里，与 NEKO 耦合的部分
  （``push_message`` / ``self.config`` / ``@llm_tool``）混在插件类上。
  拆开之后：**SDK 无关的部分全在本文件与三个兄弟模块里，插件类只剩接线**。

设计约束
--------
* **不 import NEKO 插件 SDK** —— 这样本模块可以脱离宿主单独跑离线测试；
  宿主相关的东西全部通过构造参数注入（``push_fn`` / ``logger``）。
* **零第三方依赖** —— 只用 Python 标准库。

两条链路
--------
正向::

    BetterGI ──(HTTP POST /bettergi)──▶ webhook_server ──▶ event_mapper
                                                                  │
                                       push_fn(parts=..., visibility=..., ...)
                                                                  ▼
                                                            YUI 说话

反向::

    YUI(对话 LLM) ──@llm_tool command_bettergi──▶ dispatch_command
        ──▶ CommandQueue ──▶ BetterGI 的 JS 脚本 ──dispatcher.RunTask(...)
                              │
                       ◀──(POST /bettergi/ack) 回执
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .control import (
    ACTION_CATALOG,
    ACTION_NAMES,
    ACTION_SUMMARY,
    CommandQueue,
)
from .event_mapper import EventMapper, load_config
from .webhook_server import WebhookServer

__all__ = [
    "BettergiBridge",
    "CONTROL_TOOL_DESCRIPTION",
    "CONTROL_SELF_CHECK_REMINDER",
    "COMPANION_LINE_PREFIX",
    "COMPANION_LINE_VISIBILITY",
    "PUSH_SOURCE",
]


#: 推送来源标识（会写进 ``push_message(source=...)``）。
#:
#: ★ 为什么不是 "bettergi_companion"
#:   单插件架构下这个插件只有一个 id（``genshin_game_buddy``）。
#:   来源标识沿用旧名会让宿主日志里出现一个**已不存在的插件**，
#:   排查时容易误以为装了两个插件。
PUSH_SOURCE = "genshin_game_buddy"

#: 台词送给 LLM 前的**硬指令前缀**（强制第一人称、原样说出来）。
#:
#: ★ 机制出处
#:   同款做法见 ``genshin_game_buddy/event_to_yui.py`` 的
#:   ``COMPANION_LINE_PREFIX``（2026-10-22 落地，实测有效）。
#:
#: ★ 为什么台词要用【】包住
#:   给模型一个明确的「要复述的边界」。少了括号，模型经常把指令本身
#:   也复述出来 —— 观众会看到「你现在就是 YUI 本人…」整句。
COMPANION_LINE_PREFIX = (
    "（你现在就是 YUI 本人。请完全以第一人称直接、自然地对主人说出下面这句话，"
    "不要做任何补充，也不要说这是插件说的。直接说：）\n"
)

#: 台词的**显示通道**（``push_message`` 的 ``visibility``）。
#:
#: ★ 为什么是空列表（= 不直接上屏）
#:   硬指令 + 台词整条塞进 ``visibility=["chat"]`` 的话，聊天框里会出现
#:   「（你现在就是 YUI 本人。…）\n【冲鸭！】」这种工程腔 ——
#:   观众看到的应该是 YUI 说的话，不是我们的内部指令。
#:   空列表 = 台词**只当 LLM 上下文**，观众只看 YUI 说的那一句。
#:
#:   官方语义见 ``plugin/sdk/shared/core/push_message_schema.py:5-8``。
#:   合法值只有 ``"chat"`` / ``"hud"`` —— **没有** ``"log"`` 这种通道。
COMPANION_LINE_VISIBILITY: list[str] = []

#: 反向控制工具的 ``description`` —— **这是给 LLM 看的说明书**，
#: 写得好坏直接决定 LLM 会不会在对的时机调用它。
#:
#: ★ 为什么要写得这么啰嗦
#:   LLM 只根据这段文字判断「现在该不该调用这个工具」。如果不明确写出
#:   「只有主人**明确要求**前台自动助手干活时才调用」，模型很容易在普通聊天里
#:   也去调它，主人就会看到 YUI 无缘无故让自动助手开始打怪。
#:
#: ★ 为什么要明确写「失败要如实告诉主人」
#:   本工具会**阻塞等回执**，返回给模型的是「成功 / 失败 / 超时」的实情。
#:   如果不叮嘱模型，失败时它可能会含糊地说「好的，我这就试试」——
#:   那就是欺骗主人。
CONTROL_TOOL_DESCRIPTION = (
    "当主人要求前台自动助手（BetterGI）干活时调用。"
    "它是主人召唤的游戏辅助程序，你（YUI）负责替主人向它下达指令 —— "
    "当主人**明确要求**自动助手干活时才调用这个工具，例如："
    "「帮我把自动战斗打开」「开始战斗」「先停一下」「开始自动秘境」"
    "「去钓鱼」「帮我伐木」「BetterGI 现在在干嘛」。"
    "可用 action：" + ACTION_SUMMARY + "。"
    "其中 combat_script 需要额外参数 args："
    '{"script": "attack,e,q", "avatar": "可莉"}（脚本串与角色名均可省略）。'
    "调用后本工具会等待自动助手的执行回执，并把成功或失败**如实**告诉你 —— "
    "拿到失败或超时时，请直接告诉主人「指令没发出去 / 没收到回执」以及可能的原因，"
    "不要假装成功。"
    "注意：普通聊天、闲聊、或者主人只是提到「战斗」两个字时，**不要**调用这个工具。"
)

#: 反向通道**没接上**时，让 YUI 主动提醒用户去跑一键配置的话。
#:
#: ★ 判定依据
#:   「有没有人取过指令」。BetterGI 的代理脚本一旦跑起来就会不停
#:   ``GET <path>/command``；一次都没人来取 = 通道没接上。
#:   比「BetterGI 进程在不在」准得多（装了 ≠ 配好了）。
CONTROL_SELF_CHECK_REMINDER = (
    "主人，BetterGI 那边的控制通道好像还没连好哦，"
    "双击插件目录里的 `一键配置.bat` 试试吧~"
)

#: 自检的默认参数（``[control]`` 段可以覆盖）
DEFAULT_SELF_CHECK_DELAY_S = 15.0
DEFAULT_SELF_CHECK_REPEAT_S = 120.0
DEFAULT_SELF_CHECK_MAX = 2


def _as_float(value: Any, fallback: float) -> float:
    """把配置里的值转成 float；转不动就用 fallback。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _current_loop() -> Optional[asyncio.AbstractEventLoop]:
    """拿到当前正在跑的事件循环（没有就返回 ``None``）。"""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class BettergiBridge:
    """BetterGI 双向桥接（SDK 无关）。

    用法::

        bridge = BettergiBridge(push_fn=plugin.push_message, logger=plugin.logger)
        bridge.start(host_section)      # 监听 127.0.0.1:17890/bettergi
        ...
        bridge.stop()
    """

    def __init__(
        self,
        *,
        push_fn: Optional[Callable[..., Any]] = None,
        logger: Any = None,
        config_path: Optional[Path] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """
        :param push_fn: 推送函数，签名对齐 SDK 的 ``push_message``（v2）。
            传 ``None`` 时桥接仍可运行，只是不会真的出声。
        :param logger: 可选 logger（需有 debug/info/warning/error 方法）。
        :param config_path: 桥接配置文件；``None`` 用 ``bettergi/config.toml``。
        :param loop: 自检定时器要回到的事件循环；``None`` 表示「当前循环」。
        """
        self._push_fn = push_fn
        self.logger = logger
        self._config_path = Path(config_path) if config_path else None
        self._loop = loop

        self._server: Optional[WebhookServer] = None
        self._mapper: Optional[EventMapper] = None
        #: 反向控制的指令队列（``[control].enabled = true`` 时才建）
        self._control_queue: Optional[CommandQueue] = None
        #: 等回执的超时（秒）
        self._control_ack_timeout_s = 15.0
        #: 取指令时的长轮询保持时长（秒）
        self._control_poll_hold_s = 5.0
        # ── 启动自检 ──────────────────────────────────────────────
        self._control_self_check_delay_s = DEFAULT_SELF_CHECK_DELAY_S
        self._control_self_check_repeat_s = DEFAULT_SELF_CHECK_REPEAT_S
        self._control_self_check_max = DEFAULT_SELF_CHECK_MAX
        self._control_self_check_sent = 0
        self._self_check_timer: Optional[threading.Timer] = None

    # ── 生命周期 ──────────────────────────────────────────────────────
    def start(self, host_section: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """启动 Webhook 服务器。

        :param host_section: 宿主配置面板里的覆盖段（``webhook`` / ``control`` 子段）。
        :returns: 状态字典（``status`` = running / degraded）。
        """
        section = host_section if isinstance(host_section, dict) else {}
        self._mapper = EventMapper(self._config_path)

        webhook = dict(self._mapper.webhook_config)
        host_override = section.get("webhook")
        if isinstance(host_override, dict):
            for key in ("host", "port", "path"):
                if key in host_override:
                    webhook[key] = host_override[key]

        host = str(webhook.get("host", "127.0.0.1"))
        port = int(webhook.get("port", 17890))
        path = str(webhook.get("path", "/bettergi"))
        try:
            replay_window_s = float(webhook.get("replay_window_s", 300))
        except (TypeError, ValueError):
            replay_window_s = 300.0

        # ── 反向控制通道（YUI → BetterGI）────────────────────────────
        # `[control].enabled` 默认 **true** —— 反向控制已经是核心功能，
        # 不该让用户先翻配置文件才能用。
        control_cfg: Dict[str, Any] = {}
        raw_cfg = getattr(self._mapper, "config", None)
        if isinstance(raw_cfg, dict) and isinstance(raw_cfg.get("control"), dict):
            control_cfg = dict(raw_cfg["control"])

        # 宿主配置面板里的段优先于插件目录的 config.toml
        # （重新导入插件包时 config.toml 可能被安装流程替换，面板里的更稳）。
        host_control = section.get("control")
        if isinstance(host_control, dict):
            control_cfg.update(host_control)

        control_hold_s = 5.0
        if bool(control_cfg.get("enabled", True)):
            self._control_ack_timeout_s = _as_float(
                control_cfg.get("ack_timeout_s"), 15.0
            )
            control_hold_s = _as_float(control_cfg.get("poll_hold_s"), 5.0)
            self._control_poll_hold_s = control_hold_s
            self._control_self_check_delay_s = _as_float(
                control_cfg.get("self_check_delay_s"), DEFAULT_SELF_CHECK_DELAY_S
            )
            self._control_self_check_repeat_s = _as_float(
                control_cfg.get("self_check_repeat_s"), DEFAULT_SELF_CHECK_REPEAT_S
            )
            self._control_self_check_max = max(
                0,
                int(
                    _as_float(
                        control_cfg.get("self_check_max"), DEFAULT_SELF_CHECK_MAX
                    )
                ),
            )
            self._control_queue = CommandQueue(logger=self.logger)
        else:
            self._control_queue = None

        self._server = WebhookServer(
            host=host,
            port=port,
            path=path,
            on_event=self.on_event,
            replay_window_s=replay_window_s,
            logger=self.logger,
            control=self._control_queue,
            control_hold_s=control_hold_s,
        )
        try:
            bound = self._server.start()
        except OSError as exc:
            # 端口被占用等 —— 不让插件整体崩掉，降级为「未监听」
            self._log("error", f"bettergi: 端口绑定失败 {host}:{port} —— {exc}")
            self._server = None
            return {"status": "degraded", "reason": "bind_failed", "detail": str(exc)}

        # ── 启动自检：反向通道到底通没通 ──────────────────────────
        if self._loop is None:
            self._loop = _current_loop()
        self.schedule_self_check()

        control_note = (
            f"，反向控制已启用（取指令 {bound}{self._server.command_path}）"
            if self._control_queue is not None
            else ""
        )
        self._log(
            "info",
            f"bettergi 桥接已启动：监听 {host}:{bound}{path}，"
            f"已登记事件 {len(self._mapper.known_events())} 个{control_note}",
        )
        return {
            "status": "running",
            "host": host,
            "port": bound,
            "path": path,
            "control_enabled": self._control_queue is not None,
        }

    def stop(self) -> None:
        """停止 Webhook 服务器与自检定时器（可重复调用）。"""
        self.cancel_self_check()
        if self._server is not None:
            self._server.stop()
            self._server = None
        if self._control_queue is not None:
            self._control_queue.clear()
        self._log("info", "bettergi 桥接已停止")

    def reload(self) -> None:
        """重载桥接配置（不改监听地址）。"""
        if self._mapper is not None:
            # ★ ``load_config`` 是**模块级函数**，不是 EventMapper 的方法。
            #   写成 ``type(self._mapper).load_config(...)`` 会在运行时
            #   抛 AttributeError —— 一旦有人调 reload 就炸。
            self._mapper.config = load_config(self._mapper.config_path)
        self._log("info", "bettergi 桥接配置已重载")

    # ── 事件处理 ──────────────────────────────────────────────────────
    def on_event(self, event: str, message: str, data: Any, result: Any) -> bool:
        """Webhook 回调：事件 → 台词 → 推送。

        :returns: 是否真的推送了（被限流/未登记时返回 ``False``）。
        """
        if self._mapper is None:
            self._log("warning", "bettergi: 收到事件但映射器尚未初始化")
            return False

        try:
            line = self._mapper.map_event(event, message=message, data=data, result=result)
        except Exception as exc:  # noqa: BLE001 - 映射异常不能拖垮服务器
            self._log("error", f"bettergi: 事件映射异常 {exc!r}")
            return False

        if not line:
            self._log(
                "debug", f"bettergi: 事件 {event} 无台词（未登记/限流/防复读）"
            )
            return False

        return self.push_line(line, event=event, message=message, result=result)

    def push_line(
        self, line: str, *, event: str, message: str = "", result: Any = None
    ) -> bool:
        """把一句台词推给宿主，让 YUI 开口。

        用的是 v2 参数（``parts`` / ``visibility`` / ``ai_behavior``），
        不用任何已弃用的 v1 字段。

        ★ 推给 LLM 的文本 = **硬指令前缀 + 【台词】**
          目的是让 YUI 以第一人称**原样说出**这句台词，而不是把它当成
          「插件发来的通知」去点评。

        ★ ``visibility=[]``（不直接上屏）
          内部指令不该出现在观众眼前；观众只看 YUI 说的那一句。

        ★ 日志里记的仍然是**干净的台词**（不含前缀）
          排查时一眼能看出「我们到底说了什么」。
        """
        if self._push_fn is None:
            self._log("error", "bettergi: 没有配置推送函数（push_fn is None）")
            return False

        # 硬指令 + 边界括号（只在推给 LLM 时加，不污染日志）
        speech = f"{COMPANION_LINE_PREFIX}【{line}】"

        try:
            result_obj = self._push_fn(
                source=PUSH_SOURCE,
                visibility=list(COMPANION_LINE_VISIBILITY),
                ai_behavior="respond",
                parts=[{"type": "text", "text": speech}],
                priority=5,
                metadata={
                    "event": event,
                    "bettergi_message": message,
                    "bettergi_result": result,
                    "plugin_id": "genshin_game_buddy",
                    "speech_style": "first_person_verbatim",
                },
            )
        except Exception as exc:  # noqa: BLE001 - 推送失败不能拖垮服务器
            self._log("error", f"bettergi: push_message 失败 {exc!r}")
            return False

        submitted = True
        if isinstance(result_obj, dict):
            submitted = result_obj.get("submitted") is True
            if not submitted:
                self._log(
                    "warning",
                    f"bettergi: 推送被拒 reason={result_obj.get('reason')!r}",
                )

        self._log("info", f"bettergi: 已推送 [{event}] {line}")
        return submitted

    # ── 反向控制：YUI 指挥 BetterGI ──────────────────────────────────
    async def dispatch_command(self, action: str, args: Any = None) -> Dict[str, Any]:
        """把「开自动战斗」这类意图变成一条给 BetterGI 的指令，**并等回执**。

        ★ 为什么要等回执
          用户要的是「YUI 告诉我到底成没成」。如果工具一返回「已发送」，
          用户永远不知道指令是不是石沉大海 —— BetterGI 没装脚本、没开
          「JS HTTP 权限」、或者端口被拦，表现都一样：什么都不会发生。
          所以这里**阻塞等 BetterGI 的回执**（``[control].ack_timeout_s`` 秒），
          把成功 / 失败 / 超时**如实**返回给 LLM，让 YUI 用嘴说清楚。

        ★ 返回约定
          成功 → ``{"output": "...", "is_error": False}``
          失败 → ``{"output": "...", "is_error": True, "error": "..."}``
        """
        queue = self._control_queue
        if queue is None:
            msg = (
                "反向控制通道没有启用（config.toml 里 [control].enabled = false，"
                "或者宿主配置面板里把它关了），所以现在没法指挥 BetterGI。"
            )
            self._log("warning", f"bettergi: {msg}")
            return {"output": msg, "is_error": True, "error": msg}

        name = str(action or "").strip()
        if name not in ACTION_CATALOG:
            msg = f"未知动作 {name!r}，可用动作：{', '.join(ACTION_NAMES)}"
            self._log("warning", f"bettergi: {msg}")
            return {"output": msg, "is_error": True, "error": msg}

        payload = dict(args) if isinstance(args, dict) else {}

        try:
            cmd = queue.enqueue(name, payload)
        except ValueError as exc:
            msg = str(exc)
            self._log("warning", f"bettergi: {msg}")
            return {"output": msg, "is_error": True, "error": msg}

        timeout = max(1.0, float(self._control_ack_timeout_s))
        deadline = time.monotonic() + timeout
        result = None
        while True:
            result = queue.take_result(cmd.command_id)
            if result is not None:
                break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.15)

        if result is None:
            msg = (
                f"指令「{ACTION_CATALOG[name].label}」已经发出去了，"
                f"但 {timeout:.0f} 秒内没收到 BetterGI 的回执 —— "
                "很可能指令根本没送到。请检查："
                "① BetterGI 的调度器里「NEKO代理」配置组是不是正在跑；"
                "② 那个项目的「JS HTTP 权限」是不是设成了**启用**；"
                "③ 插件的 HTTP 端口有没有被防火墙拦。"
                "（双击插件目录里的「一键配置.bat」可以自动检查一遍。）"
            )
            self._log("warning", f"bettergi: 指令超时未回执 {name}")
            return {"output": msg, "is_error": True, "error": msg, "action": name}

        if result.ok:
            text = result.message or f"BetterGI 已经{ACTION_CATALOG[name].label}了。"
            self._log("info", f"bettergi: 指令成功 {name}")
            return {
                "output": text,
                "is_error": False,
                "action": name,
                "detail": result.detail,
            }

        text = result.message or f"BetterGI 执行「{ACTION_CATALOG[name].label}」失败了。"
        self._log("warning", f"bettergi: 指令失败 {name}")
        return {
            "output": text,
            "is_error": True,
            "error": text,
            "action": name,
            "detail": result.detail,
        }

    # ── 启动自检：反向通道（YUI → BetterGI）到底通没通 ────────────────
    #
    # ★ 为什么用「有没有人来取过指令」当判据
    #   BetterGI 的 JS 代理脚本一旦跑起来，就会不停
    #   ``GET <path>/command`` 长轮询。桥接在 ``_serve_next_command``
    #   里每被取一次指令就记一笔（``WebhookServer.note_control_poll``）。
    #   ⇒ 计数为 0 = 没有任何人在取 = 通道没接上。
    def schedule_self_check(self) -> None:
        """安排第一次「反向通道通没通」的自检。"""
        if self._control_queue is None or self._server is None:
            return
        if self._control_self_check_max <= 0 or self._control_self_check_delay_s <= 0:
            return
        self._start_self_check_timer(self._control_self_check_delay_s)

    def _start_self_check_timer(self, delay: float) -> None:
        """起一个守护定时器，到点后触发一次自检。"""
        self.cancel_self_check()
        timer = threading.Timer(max(0.5, float(delay)), self._self_check_tick)
        timer.daemon = True
        self._self_check_timer = timer
        timer.start()

    def cancel_self_check(self) -> None:
        """取消还没触发的自检定时器。"""
        timer, self._self_check_timer = self._self_check_timer, None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:  # pragma: no cover
                pass

    def _self_check_tick(self) -> None:
        """定时器线程的入口 —— 把检查甩回事件循环线程再做。"""
        loop = self._loop
        if loop is None or loop.is_closed():
            # 没有可用的事件循环（离线测试 / 宿主没给循环）→ 直接做
            self.do_self_check()
            return
        try:
            loop.call_soon_threadsafe(self.do_self_check)
        except RuntimeError:
            # 循环刚关掉
            self.do_self_check()

    def do_self_check(self) -> bool:
        """真正的一次自检（跑在事件循环线程上）。

        :returns: 反向通道是否已经接通。
        """
        try:
            if self._control_queue is None or self._server is None:
                return False
            if self._server.control_poll_count > 0:
                self._log(
                    "info",
                    "bettergi: 反向通道已接通（BetterGI 的代理脚本在轮询）",
                )
                return True
            if self._control_self_check_sent >= self._control_self_check_max:
                return False
            self._control_self_check_sent += 1
            self._log(
                "warning",
                "bettergi: 没检测到 BetterGI 的代理脚本在取指令（反向通道未接上）",
            )
            self.push_line(
                CONTROL_SELF_CHECK_REMINDER,
                event="control.self_check",
                message="反向控制通道没有检测到 BetterGI 的 JS 代理脚本",
            )
            # 还没到提醒上限 → 过一会儿再看一眼
            if self._control_self_check_sent < self._control_self_check_max:
                self._start_self_check_timer(self._control_self_check_repeat_s)
            return False
        except Exception as exc:  # noqa: BLE001 - 自检失败不能拖垮插件
            self._log("error", f"bettergi: 启动自检异常 {exc!r}")
            return False

    # ── 辅助 ──────────────────────────────────────────────────────────
    def _log(self, level: str, message: str) -> None:
        """写日志（logger 可能不存在于桩件环境）。"""
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        fn = getattr(logger, level, None)
        if fn is None:
            return
        try:
            fn(message)
        except Exception:  # pragma: no cover
            pass

    # ── 只读属性（供测试与调试）────────────────────────────────────────
    @property
    def mapper(self) -> Optional[EventMapper]:
        """当前事件映射器。"""
        return self._mapper

    @property
    def server(self) -> Optional[WebhookServer]:
        """当前 Webhook 服务器。"""
        return self._server

    @property
    def control_queue(self) -> Optional[CommandQueue]:
        """当前指令队列（反向控制关时是 ``None``）。"""
        return self._control_queue

    @property
    def is_running(self) -> bool:
        """Webhook 服务器是否在跑。"""
        return self._server is not None and self._server.is_running
