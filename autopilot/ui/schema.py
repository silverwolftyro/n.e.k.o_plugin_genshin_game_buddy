"""数据通道协议（**只描述数据，不描述传输**）。

这一层是双轨制的地基：

    run_loop（核心）  ──写──┐
                          ├──> autopilot/run/  ──读──> overlay（A 轨，独立进程）
    N.E.K.O. 面板（B 轨） ──┘

★ 为什么 schema 必须独立于实现
------------------------------
v1 用文件（`state_snapshot.json` / `state_stream.jsonl` / `commands.jsonl`），
v2 换成 socket 或 named pipe。**换了传输，这三个模型的字段、一个都不能动** ——
否则 A 轨（overlay）和 B 轨（插件面板）要跟着改两遍。

所以这里：
* 只用 pydantic 描述**数据长什么样**，不出现任何路径、文件、socket 字样
* 传输细节全在 ``protocol.py``
* 未知字段用 ``extra="ignore"`` 容忍（v2 加字段不该让 v1 的读侧崩）
* 未知 ``code`` / ``cmd`` **允许**，由读侧按 INFO / warning 处理

三个模型对应三条流向：
* :class:`StateSnapshot` —— 覆盖写，当前态（overlay 每 100ms 读一次）
* :class:`StateEvent`   —— 追加写，事件流（overlay 追尾）
* :class:`Command`      —— 追加写，反向控制（overlay → run_loop）
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: 协议版本。**任何字段级不兼容改动都要 +1**，且读侧要能识别旧版本。
SCHEMA_VERSION = "1.0"


class _Base(BaseModel):
    """所有通道模型的公共基类。

    ``extra="ignore"`` 而不是 ``"forbid"``：读侧遇到**更新的**写侧加的
    字段应该静默忽略，而不是整个崩掉。反过来（写侧遇到老读侧）由
    ``schema_version`` 负责判断。
    """

    model_config = ConfigDict(extra="ignore")


# --------------------------------------------------------------------------
# 1. 状态快照（覆盖写）
# --------------------------------------------------------------------------


class RuntimePhase(str, Enum):
    """主循环状态。取值与 ``autopilot.core.RuntimeState`` **一一对应**。

    这里重新定义而**不直接复用** core 的枚举：协议层不能依赖运行层，
    否则把 core 拆出去时协议就跟着塌了。写侧负责转换。
    """

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"       # overlay 启动时还没读到快照


class Health(str, Enum):
    """健康度。取值来自 ``LoopStats.health`` 的粗粒度归类。

    写侧做归并（"正常" -> OK，"长期空转" -> IDLE_LONG…），
    **不要**把中文原样塞进来 —— 协议里只放稳定的英文枚举。
    """

    OK = "OK"
    OFFLINE = "OFFLINE"          # 没有 run_loop 在跑
    IDLE_LONG = "IDLE_LONG"      # 长期空转（规则没覆盖当前场景）
    DEGRADED = "DEGRADED"        # 感知/截图有问题但还在跑
    CRASHED = "CRASHED"          # 连续崩溃


class IntentStat(_Base):
    """意图相关的聚合计数。"""

    count: int = Field(default=0, ge=0, description="本次运行累计执行的意图数")
    idle_streak: int = Field(
        default=0, ge=0, description="当前连续空转轮数（命中即归零）")
    last_action: str = Field(default="", description="最近一次意图的动作名")


class CaptureStat(_Base):
    """截图通道状态。

    ``degraded_from`` 记录"从哪个通道降级过来的"—— 用户会问
    "为什么变慢了"，答案通常就是这个字段。
    """

    active: str = Field(default="", description="当前通道名 dxcam/bitblt/mss")
    degraded_from: str = Field(default="", description="降级前的通道名，空=未降级")


class StateSnapshot(_Base):
    """当前状态。**覆盖写**（临时文件 + rename），读侧永远读到完整的一份。

    不做增量/补丁：状态快照的语义就是"此刻是什么样"，
    补丁会让读侧必须维护一个可能与写侧不同步的累积状态。
    """

    schema_version: str = SCHEMA_VERSION
    run_id: str = Field(default="", description="run_loop 每次启动生成，形如 20260928_170541")
    written_at_ms: int = Field(default=0, description="快照写入时刻（墙钟毫秒）")
    uptime_s: float = Field(default=0.0, ge=0.0)

    state: RuntimePhase = RuntimePhase.UNKNOWN
    health: Health = Health.OFFLINE

    tick_hz: float = Field(default=0.0, ge=0.0, description="实际 tick 频率")
    tick_avg_ms: float = Field(default=0.0, ge=0.0)
    tick_p95_ms: float = Field(default=0.0, ge=0.0)
    tick_count: int = Field(default=0, ge=0)

    scene: str = Field(default="UNKNOWN", description="最近一帧的场景 ID")
    intent: IntentStat = Field(default_factory=IntentStat)
    capture: CaptureStat = Field(default_factory=CaptureStat)

    panic_key: str = Field(default="F12")
    panic_active: bool = False

    #: 原始 stats 透传。**读侧不要依赖它的结构** —— 它是诊断用的，
    #: schema 演进时可能变；上面那些扁平字段才是稳定契约。
    raw_stats: dict[str, Any] = Field(default_factory=dict)

    def is_stale(self, now_ms: int, max_age_ms: int = 10_000) -> bool:
        """快照是否过期（run_loop 崩了没机会更新时会这样）。

        overlay 用它显示「最后更新于 N 秒前」，而不是假装一切正常。
        """
        if not self.written_at_ms:
            return True
        return (now_ms - self.written_at_ms) > max_age_ms


# --------------------------------------------------------------------------
# 2. 事件流（追加写）
# --------------------------------------------------------------------------


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class EventCode(str, Enum):
    """v1 最小事件集。

    ★ **这是开放集合**：读侧遇到表外的 code **必须**按 INFO 处理而不是
      崩溃 —— v2 加新事件时，v1 的 overlay 还要继续能用。
    """

    # 生命周期
    RUN_STARTED = "RUN_STARTED"
    RUN_STOPPING = "RUN_STOPPING"
    RUN_STOPPED = "RUN_STOPPED"
    # 通道
    CHANNEL_DEGRADED = "CHANNEL_DEGRADED"
    CHANNEL_RECOVERED = "CHANNEL_RECOVERED"
    # 急停
    PANIC_TRIGGERED = "PANIC_TRIGGERED"
    PANIC_CLEARED = "PANIC_CLEARED"
    # 熔断
    BREAKER_OPENED = "BREAKER_OPENED"
    BREAKER_CLOSED = "BREAKER_CLOSED"
    # 意图
    INTENT_EXECUTED = "INTENT_EXECUTED"
    INTENT_REJECTED = "INTENT_REJECTED"
    # 场景 / 故障
    SCENE_CHANGED = "SCENE_CHANGED"
    TICK_CRASH = "TICK_CRASH"
    # 命令（写侧在**收到**命令时回一条，让读侧知道命令被采纳了）
    COMMAND_RECEIVED = "COMMAND_RECEIVED"
    # ---- 业务事件（2026-10-08，给「事件流 -> YUI 说话」用）----
    # ★ 加它们**不需要** bump SCHEMA_VERSION：设计之初就写明
    #   「这是开放集合」，v2 加新事件时 v1 的 overlay 还要继续能用。
    POSITION_CHANGED = "POSITION_CHANGED"
    ENEMY_DETECTED = "ENEMY_DETECTED"
    PATHFIND_START = "PATHFIND_START"
    PATHFIND_ARRIVED = "PATHFIND_ARRIVED"
    PATHFIND_STUCK = "PATHFIND_STUCK"
    ESCAPE_SUCCESS = "ESCAPE_SUCCESS"


class StateEvent(_Base):
    """一条事件。**追加写**，一行一条 JSON。"""

    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    seq: int = Field(default=0, ge=0, description="run 内单调递增")
    ts_ms: int = Field(default=0)
    #: 开放枚举：写进表外的字符串也接受（读侧按 INFO 处理）
    code: str = ""
    severity: Severity = Severity.INFO
    message: str = ""
    context: dict[str, Any] = Field(default_factory=dict)

    @property
    def known(self) -> bool:
        """code 是否在 v1 已知集合内。overlay 用它决定要不要高亮。"""
        try:
            EventCode(self.code)
        except ValueError:
            return False
        return True


# --------------------------------------------------------------------------
# 3. 命令（追加写，反向）
# --------------------------------------------------------------------------


class CommandName(str, Enum):
    """v1 支持的命令。

    STOP / KILL 的区别是**是否走清理流程**，不是"快慢"：
    ``STOP`` 会 ``release_all()`` + 关截图器 + 关总线；
    ``KILL`` 直接 ``os._exit()``，什么都来不及 —— 万不得已才用
    （比如按键卡死、STOP 卡在某个 tick 里出不来）。
    """

    STOP = "STOP"          # 软停：释放按键、关资源
    KILL = "KILL"          # 强杀：立即退出，跳过一切清理
    PAUSE = "PAUSE"        # 暂停执行层，保留感知层
    RESUME = "RESUME"      # 恢复
    NUDGE = "NUDGE"        # 轻推一下（v1 不实现命令体，留接口）


class Command(_Base):
    """一条命令。overlay / 插件面板 -> run_loop。"""

    schema_version: str = SCHEMA_VERSION
    cmd_id: str = Field(default="", description="cmd_000001 形式，run 内唯一")
    ts_ms: int = Field(default=0)
    #: 开放枚举：v1 未实现的命令要被**忽略并记 warning**，不是报错
    cmd: str = ""
    args: dict[str, Any] = Field(default_factory=dict)

    @property
    def supported(self) -> bool:
        try:
            CommandName(self.cmd)
        except ValueError:
            return False
        return self.cmd != CommandName.NUDGE.value


def new_cmd_id(seq: int) -> str:
    return f"cmd_{seq:06d}"


__all__ = [
    "SCHEMA_VERSION",
    "CaptureStat",
    "Command",
    "CommandName",
    "EventCode",
    "Health",
    "IntentStat",
    "RuntimePhase",
    "Severity",
    "StateEvent",
    "StateSnapshot",
    "new_cmd_id",
]
