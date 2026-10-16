"""policy 层对外唯一契约。

对应架构文档 6.6。这个文件是**将来可能闭源化的分界线** ——
主线（core loop）只允许 `from autopilot.policy import PolicySource`，
具体实现（本地规则树 / 将来可能的云端）随时可换，主循环一行不改。

依赖边界（硬约束）
--------------------
本模块及整个 policy 包**只依赖** ``contracts`` 与 ``state``：

    ✅ autopilot.contracts.state_packet
    ✅ autopilot.contracts.intent_packet
    ✅ autopilot.state
    ❌ autopilot.perception   （识别是它的单向职责，policy 不反向依赖）
    ❌ autopilot.actuator     （执行层被 policy **驱动**，不是它的依赖）

唯一例外是 ``rule_loader`` 为做加载期白名单校验会**惰性** import
``actuator.traits.TRAITS``。那是纯校验、纯运行期的单向引用，不产生
行为耦合；已在其模块内注明原因。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from ..contracts.intent_packet import IntentPacket
from ..contracts.state_packet import StatePacket

#: 决策延迟档位。两条决策回路（realtime 每帧 / strategic 低频）共用同一个
#: 接口，靠这个字段区分负载特征，而不是靠 isinstance 判断实现类型 ——
#: 否则「换实现」就变成了「改主循环」，可替换性直接失效。
LatencyClass = Literal["realtime", "strategic"]


class IntentSourceKind(str, Enum):
    """策略来源类型。

    与 `IntentPacket.source`（裸字符串）分开：那边是**线上报的**值，
    冻结在契约里；这边是**配置选择的**实现类别，带 DISABLED。

    DISABLED 是合法且必要的一档：autopilot 关闭时主循环仍要能构造一个
    合法 PolicySource，而不是 None —— 否则每个调用点都得写
    `if policy is not None`，可替换性又漏回主循环了。
    """

    LOCAL = "LOCAL"
    CLOUD = "CLOUD"
    DISABLED = "DISABLED"

    @classmethod
    def parse(cls, raw: "str | IntentSourceKind") -> "IntentSourceKind":
        if isinstance(raw, cls):
            return raw
        key = str(raw).strip().upper()
        for m in cls:
            if m.value == key:
                return m
        raise PolicyConfigError(
            f"未知的策略来源 {raw!r}。合法值：{[m.value for m in cls]}"
        )


class PolicySourceError(RuntimeError):
    """policy 层一切错误的基类。"""


class PolicyConfigError(PolicySourceError):
    """配置非法 / 指定的实现不存在。"""


@runtime_checkable
class PolicySource(Protocol):
    """决策源。

    任何东西只要满足这四样，就能被主循环驱动：换一个实现不需要改主循环
    一行代码。这是 policy 可替换性的全部依据，所以**这个协议要尽可能小**。

    约定
    ----
    * ``poll`` 必须在有限时间内返回 —— 它跑在感知主循环里，一次卡住
      就是整个 tick 卡住。具体实现需要长耗时必须自己开线程 + 排队。
    * ``poll`` 返回 ``None`` 表示「这一帧没有该做的事」。**空转是合法结果**，
      不是错误（原则 3：一轮只出一个动作，没命中就不发）。
    * ``poll`` 不得抛异常到主循环。内部错误应转成事件 + ``None``，
      但**不能静默** —— 见 ``last_error``。
    """

    #: 人类可读的来源名，进日志与事件。
    name: str

    #: 延迟档位，见 ``LatencyClass``。
    latency_class: LatencyClass

    def poll(self, state: StatePacket) -> IntentPacket | None:
        """根据一帧状态给出一个意图，或 ``None`` 表示空转。

        :param state: 本帧的不可变状态快照
        :returns: 最多一个意图。命中即返回，不做多步规划（原则 3）。
        """
        ...

    def available(self) -> bool:
        """当前是否可用。不可用时主循环会跳过它（降级而非崩溃）。"""
        ...

    @property
    def last_error(self) -> str:
        """最近一次内部错误的原因，空串表示没出错。

        非空**必须**伴随一条 Event —— 决策层出错是必须被看见的故障，
        不能只换个返回值就当没事发生（原则 1）。
        """
        ...


def describe_source(src: PolicySource) -> dict[str, Any]:
    """把任意 PolicySource 打成一份诊断字典。

    故意只吃 Protocol 不吃具体类型 —— 这样闭源实现也能被诊断。
    """
    return {
        "name": getattr(src, "name", "<unnamed>"),
        "type": type(src).__name__,
        "latency_class": getattr(src, "latency_class", "?"),
        "available": bool(src.available()),
        "last_error": getattr(src, "last_error", ""),
    }


__all__ = [
    "IntentSourceKind",
    "LatencyClass",
    "PolicyConfigError",
    "PolicySource",
    "PolicySourceError",
    "StatePacket",
    "describe_source",
]