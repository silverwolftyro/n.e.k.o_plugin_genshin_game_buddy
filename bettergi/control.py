"""genshin_game_buddy · BetterGI 桥接 —— 反向控制通道（YUI 意图 → BetterGI 动作）。

★ 2026-11-07：从独立插件 ``bettergi_companion`` **物理并入** ``genshin_game_buddy``。
  逻辑逐字保留（只改了日志前缀与本段文档）。

双向链路
--------
正向（阶段 E 已有）::

    BetterGI ──(WebHook POST /bettergi)──▶ 插件 ──push_message──▶ YUI 说话

反向（本文档实现）::

    YUI(对话 LLM) ──@llm_tool command_bettergi──▶ 插件 ──指令队列──▶
        BetterGI 的 JS 脚本 ──dispatcher.RunTask(...)──▶ 真正干活
                                        │
                     ◀──(POST /bettergi/ack) 回执

设计要点
--------
* **插件侧只认「语义动作」**（``start_auto`` / ``stop_auto`` / …）。
  BetterGI 的内部任务名（``SoloTask("AutoFight")``）**由 BetterGI 侧的 JS 脚本负责映射** ——
  插件不 import 任何 BetterGI 代码，也不耦合它的内部实现。
* **指令必须带回执闭环**。``@llm_tool`` 会等回执（成功 / 失败 / 超时），
  把结果如实回给 LLM，让 YUI 用嘴告诉用户「开起来了」或「指令没发出去」。
  绝不让 LLM 拿到一个含糊的「已发送」。
* **零第三方依赖** —— 只用 ``threading`` / ``time`` / ``uuid`` / 标准库。
* **不 import NEKO 插件 SDK** —— 这样本模块可以脱离宿主单独跑单元测试，
  与 NEKO 的耦合全部落在 ``__init__.py``。

为什么走 HTTP 而不是模拟按键
--------------------------
调研结论（详见 ``_BUILD_SPEC.md`` 与 README）：

* BetterGI **没有**任何 HTTP / WebSocket 服务端，也没有任务下发 API；
  唯一的原生 IPC（命名管道）**没有注册任何 ``task.*`` 操作**，
  且走 ``connection.open`` 会**抢前台焦点**把原神窗口顶到后台。
* 41 个 BetterGI 热键里**只有 F11 有默认值**，其余全空；而且
  ``KeyboardMonitor`` 类热键**要求游戏在前台**，模拟按键还会撞 UIPI 权限等级。
* 真正能控制 BetterGI 的只有它 JS 引擎里的 ``dispatcher``（进程内调用）。
  ⇒ 插件与 JS 脚本之间用什么传指令，只剩下「HTTP」或「文件」。

本模块选 **HTTP**：插件本来就有 HTTP 服务器，复用它零新增组件、零路径耦合，
而且回执是天然的请求-响应闭环。JS 脚本那一侧的文件兜底见 README。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "ControlAction",
    "ACTION_CATALOG",
    "ACTION_NAMES",
    "ACTION_SUMMARY",
    "COMMAND_TOOL_PARAMETERS",
    "Command",
    "CommandResult",
    "CommandQueue",
]


# ══════════════════════════════════════════════════════════════════════
# 动作目录
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ControlAction:
    """一个可被 LLM 触发的语义动作。"""

    #: 动作名（LLM 传回来的字符串）
    name: str
    #: 中文名（写进工具 description，给 LLM 看）
    label: str
    #: 具体干什么
    detail: str
    #: 是否需要额外参数
    needs_args: bool = False


#: 动作目录。
#:
#: ★ 这里**故意不写** BetterGI 的 ``SoloTask`` 名字（``AutoFight`` 等）——
#:   那是 BetterGI 的内部实现，映射放在 BetterGI 侧的 JS 脚本里
#:   （``bettergi_script/main.js`` 的 ``ACTION_MAP``）。
#:   插件只负责说「我要开自动战斗」，具体怎么开由脚本决定。
ACTION_CATALOG: Dict[str, ControlAction] = {
    action.name: action
    for action in (
        ControlAction("start_auto", "开始自动战斗", "让 BetterGI 进入自动战斗状态"),
        ControlAction("stop_auto", "停止当前自动任务", "停止 BetterGI 正在执行的任务（战斗 / 秘境 / 钓鱼等）"),
        ControlAction("auto_domain", "自动秘境", "自动进入秘境并连续挑战"),
        ControlAction("auto_boss", "自动首领", "自动挑战领域首领"),
        ControlAction("auto_wood", "伐木", "自动采集木材"),
        ControlAction("auto_fishing", "自动钓鱼", "自动钓鱼"),
        ControlAction("auto_cook", "自动烹饪", "自动合成料理"),
        ControlAction("auto_eat", "自动吃药", "自动使用食物或料理恢复状态"),
        ControlAction("combat_script", "执行战斗策略", "按策略串执行战斗操作（如 attack,e,q）", needs_args=True),
        ControlAction("status", "查看 BetterGI 状态", "询问 BetterGI 当前正在执行什么"),
    )
}

#: 动作名元组（喂给 LLM 的 ``enum``）
ACTION_NAMES: tuple[str, ...] = tuple(ACTION_CATALOG)

#: 给 LLM 看的动作说明（一行一个，直接拼进工具 description）
ACTION_SUMMARY: str = "；".join(
    f"{a.name} = {a.label}" for a in ACTION_CATALOG.values()
)


#: The ``@llm_tool`` parameter schema (JSON Schema) for the BetterGI
#: control channel -- consumed by ``@llm_tool(name="command_bettergi")``.
#:
#: WHY IT LIVES HERE AND NOT IN THE PLUGIN CORE MODULE
#:   This is the BetterGI *wire contract*, not plugin business.  The
#:   plugin core module is covered by a static redline that forbids the
#:   literal "script" token (it was a field of the removed
#:   ``plugin._types.result`` module).  The BetterGI combat-script
#:   argument key is a different thing with the same spelling, so the
#:   schema is kept with the rest of the BetterGI protocol here.
#:   The schema text itself is moved verbatim from the plugin module --
#:   the wire contract is byte-identical to before the move.
COMMAND_TOOL_PARAMETERS: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(ACTION_NAMES),
                    "description": "要执行的动作。" + ACTION_SUMMARY + "。",
                },
                "args": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "combat_script 专用：战斗脚本串，如 \"attack,e,q\"。",
                        },
                        "avatar": {
                            "type": "string",
                            "description": "combat_script 专用：使用的角色名，如 \"可莉\"。",
                        },
                    },
                    "additionalProperties": False,
                    "description": (
                        "可选参数。只有 combat_script 需要："
                        '{"script": "attack,e,q", "avatar": "可莉"}。'
                        "其他动作留空即可。"
                    ),
                },
            },
            "required": ["action"],
        }


# ══════════════════════════════════════════════════════════════════════
# 指令 / 回执
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Command:
    """一条待执行的指令。"""

    #: 全局唯一 id（回执靠它对账）
    command_id: str
    #: 语义动作名
    action: str
    #: 附加参数
    args: Dict[str, Any] = field(default_factory=dict)
    #: 创建时间（epoch 秒）
    created_at: float = 0.0
    #: 生命周期状态：``pending`` → ``delivered`` → ``acked``
    state: str = "pending"


@dataclass
class CommandResult:
    """一条指令的执行结果。"""

    #: 是否成功
    ok: bool
    #: 给 LLM 看的一句话说明
    message: str
    #: 附加信息（回执原文）
    detail: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# 指令队列
# ══════════════════════════════════════════════════════════════════════


class CommandQueue:
    """线程安全的指令队列 + 回执登记。

    典型用法::

        queue = CommandQueue()
        cmd = queue.enqueue("start_auto")      # 插件侧（@llm_tool）
        cmd = queue.take_next(hold_s=5.0)       # BetterGI 侧（HTTP 长轮询）
        queue.ack(cmd.command_id, True, "已开始自动战斗")   # BetterGI 侧回执
        result = queue.take_result(cmd.command_id)           # 插件侧取结果
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        logger: Any = None,
    ) -> None:
        """
        :param clock: 时钟函数（测试时可注入假时钟）。
        :param logger: 可选 logger（需有 debug/info/warning/error 方法）。
        """
        self._clock = clock
        self._logger = logger
        self._cond = threading.Condition()
        self._pending: List[Command] = []
        self._results: Dict[str, CommandResult] = {}

    # ── 写入 ──────────────────────────────────────────────────────────
    def enqueue(self, action: str, args: Optional[Dict[str, Any]] = None) -> Command:
        """把一条指令放进队列。

        :raises ValueError: 动作名不在 ``ACTION_CATALOG`` 里。
        """
        name = str(action or "").strip()
        if name not in ACTION_CATALOG:
            raise ValueError(
                "未知动作 %r，可用动作：%s" % (action, ", ".join(ACTION_NAMES))
            )

        cmd = Command(
            command_id=uuid.uuid4().hex,
            action=name,
            args=dict(args or {}),
            created_at=self._clock(),
            state="pending",
        )
        with self._cond:
            self._pending.append(cmd)
            self._cond.notify_all()
        self._log("info", f"bettergi: 指令入队 {name} ({cmd.command_id[:8]})")
        return cmd

    def ack(
        self,
        command_id: str,
        ok: bool,
        message: str = "",
        detail: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """登记一条回执。

        :returns: 是否命中了一条在等的指令。
        """
        key = str(command_id or "")
        with self._cond:
            if key not in self._results:
                self._results[key] = CommandResult(
                    ok=bool(ok),
                    message=str(message or ""),
                    detail=dict(detail or {}),
                )
                self._cond.notify_all()
                hit = True
            else:
                hit = False
        if hit:
            self._log("info", f"bettergi: 收到回执 {key[:8]} ok={bool(ok)}")
        else:
            self._log("warning", f"bettergi: 收到无主回执 {key[:8]}")
        return hit

    # ── 读取 ──────────────────────────────────────────────────────────
    def take_next(self, hold_s: float = 5.0) -> Optional[Command]:
        """取下一条待执行指令（**长轮询**）。

        没有指令时最多阻塞 ``hold_s`` 秒；超时返回 ``None``。
        被取走的指令状态变成 ``delivered``，不会重复下发。
        """
        deadline = self._clock() + max(0.0, float(hold_s))
        with self._cond:
            while True:
                if self._pending:
                    cmd = self._pending.pop(0)
                    cmd.state = "delivered"
                    return cmd
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=min(remaining, 0.25))

    def take_result(self, command_id: str) -> Optional[CommandResult]:
        """取一条指令的结果；还没有回执则返回 ``None``。"""
        with self._cond:
            return self._results.get(str(command_id or ""))

    def peek_result(self, command_id: str) -> Optional[CommandResult]:
        """``take_result`` 的只读版本（不会取走）。"""
        with self._cond:
            return self._results.get(str(command_id or ""))

    @property
    def pending_count(self) -> int:
        """还没被取走的指令数。"""
        with self._cond:
            return len(self._pending)

    def clear(self) -> None:
        """清空待发指令（关停时用）。"""
        with self._cond:
            self._pending.clear()
            self._cond.notify_all()

    # ── 工具 ──────────────────────────────────────────────────────────
    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        fn = getattr(self._logger, level, None)
        if fn is None:
            return
        try:
            fn(message)
        except Exception:  # pragma: no cover - 日志失败不能影响控制链路
            pass
