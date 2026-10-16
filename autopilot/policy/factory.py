"""策略源工厂 —— **唯一**决定用哪个实现的地方。

主循环只写：

    from autopilot.policy import create_policy_source
    policy = create_policy_source(config.policy)

永远不会出现 ``from autopilot.policy.local_tree import LocalPolicyTree``。
将来把本地实现闭源化 / 换成云端，主循环一行不改 —— 可替换性就靠这一条
守住。

具体实现**惰性导入**：``import autopilot.policy`` 不会连带 import
``local_tree`` / ``rule_loader`` / ``dsl``。好处有二：

1. 真正关掉策略时（DISABLED），那三个模块根本不会被加载。
2. 将来闭源化时，替换 ``_build_local`` 里的分支即可，其余文件不动。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts.intent_packet import IntentPacket
from ..contracts.state_packet import StatePacket
from .base import (
    IntentSourceKind,
    LatencyClass,
    PolicyConfigError,
    PolicySource,
    describe_source,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------


@dataclass
class PolicyConfig:
    """策略层配置。

    刻意做成普通 dataclass 而不是 pydantic 模型：policy 是**可替换层**，
    主循环不应该因为换了个策略实现就被新配置模型的校验规则绊住。
    校验责任落在 ``__post_init__`` 上，且失败即抛。
    """

    kind: IntentSourceKind = IntentSourceKind.DISABLED
    #: 规则目录（仅 LOCAL 用）。None = 用内置 rules/
    rules_dir: Path | str | None = None
    #: 规则文件名过滤，None = 目录内全部 *.toml
    rule_files: tuple[str, ...] | None = None
    #: 未命中任何规则时的最大静默轮询次数。超过就发一条 WARN 事件 ——
    #: 「一条规则都不命中」在战斗场景里通常是识别出了问题，不是真的没事干。
    idle_warn_after: int = 300
    #: 给实现起名（进日志/事件）。None = 用实现自报的名字。
    name: str | None = None

    def __post_init__(self) -> None:
        self.kind = IntentSourceKind.parse(self.kind)
        if self.rules_dir is not None:
            self.rules_dir = Path(self.rules_dir)
        if self.idle_warn_after < 0:
            raise PolicyConfigError(
                f"idle_warn_after 不能为负（当前 {self.idle_warn_after}）"
            )

    @classmethod
    def coerce(cls, raw: Any) -> "PolicyConfig":
        """从 dict / 已有配置 / None 构造。

        容忍 dict 是为了能直接吃 TOML 段落，而不必先定义一堆模型类。
        """
        if raw is None:
            return cls()
        if isinstance(raw, PolicyConfig):
            return raw
        if isinstance(raw, Mapping):
            known = {f for f in cls.__dataclass_fields__}
            unknown = set(raw) - known
            if unknown:
                raise PolicyConfigError(
                    f"[policy] 里有未知配置项 {sorted(unknown)}。"
                    f"合法项：{sorted(known)}"
                )
            return cls(**dict(raw))
        # duck typing：任何有 .kind 的对象都当配置用
        if hasattr(raw, "kind"):
            return cls(
                kind=raw.kind,
                rules_dir=getattr(raw, "rules_dir", None),
                rule_files=getattr(raw, "rule_files", None),
                idle_warn_after=getattr(raw, "idle_warn_after", 300),
                name=getattr(raw, "name", None),
            )
        raise PolicyConfigError(
            f"无法把 {type(raw).__name__} 当作 PolicyConfig。"
            f"需要 None / Mapping / 有 .kind 属性的对象。"
        )


# --------------------------------------------------------------------------
# DISABLED —— 空实现
# --------------------------------------------------------------------------


class DisabledSource:
    """策略关闭。

    **不是 None**：主循环不该为了「可能没有 policy」到处写
    ``if policy is not None``。返回空转的合法实现，可替换性才守得住。
    """

    name = "disabled"
    latency_class: LatencyClass = "realtime"

    def __init__(self, reason: str = "策略已禁用") -> None:
        self._reason = reason
        self.last_error = ""

    def poll(self, state: StatePacket) -> IntentPacket | None:
        return None

    def available(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"DisabledSource(reason={self._reason!r})"


# --------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------

Builder = Callable[[PolicyConfig], PolicySource]
_REGISTRY: dict[IntentSourceKind, Builder] = {}


def register(kind: IntentSourceKind, builder: Builder) -> None:
    """注册一个实现。**云端实现将来从这里接入**，不需要改本文件的逻辑。"""
    _REGISTRY[IntentSourceKind.parse(kind)] = builder


def _build_disabled(cfg: PolicyConfig) -> PolicySource:
    return DisabledSource()


# ==========================================================================
# ⚠ 以下是唯一「知道具体实现」的地方。闭源化时改这里就够了。
# ==========================================================================


def _build_local(cfg: PolicyConfig) -> PolicySource:
    # 惰性导入：避免 import autopilot.policy 就把整套规则引擎拖进来
    from .local_tree import LocalPolicyTree  # noqa: PLC0415

    return LocalPolicyTree(
        rules_dir=cfg.rules_dir,
        rule_files=cfg.rule_files,
        idle_warn_after=cfg.idle_warn_after,
        name=cfg.name,
    )


def _build_cloud(cfg: PolicyConfig) -> PolicySource:
    # 预留位。**故意现在不实现** —— 一个连不上的云端实现留在代码里，
    # 只会让人以为有回退路径（原则 1）。真要做时在这里注册即可。
    raise PolicyConfigError(
        "CLOUD 策略尚未实现。\n"
        "  本期只交付本地策略。要接云端请在 policy/cloud.py 实现\n"
        "  满足 PolicySource 协议的四项，然后在 factory 里 register()。\n"
        "  接口已就位，接入不需要改主循环。"
    )


register(IntentSourceKind.DISABLED, _build_disabled)
register(IntentSourceKind.LOCAL, _build_local)
register(IntentSourceKind.CLOUD, _build_cloud)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def create_policy_source(config: Any = None) -> PolicySource:
    """创建策略源。**这是 policy 层对外唯一的构造入口。**

    :param config: ``PolicyConfig`` / dict（含 ``kind``）/ None / 有 ``.kind`` 的对象
    :returns: 满足 ``PolicySource`` 协议的实现，永不返回 None
    :raises PolicyConfigError: 配置非法或指定的实现不可用

    不做「静默回落」：CLOUD 不可用时**不**悄悄退回 LOCAL。
    悄悄换掉决策来源 = 用户以为在用 A 其实在用 B（原则 1）。
    要降级请由调用方显式处理。
    """
    cfg = PolicyConfig.coerce(config)
    builder = _REGISTRY.get(cfg.kind)
    if builder is None:
        raise PolicyConfigError(
            f"策略类型 {cfg.kind.value} 没有注册实现。"
            f"已注册：{sorted(k.value for k in _REGISTRY)}"
        )
    src = builder(cfg)
    if not isinstance(src, PolicySource):  # runtime_checkable Protocol
        raise PolicyConfigError(
            f"{type(src).__name__} 不满足 PolicySource 协议 —— "
            f"{describe_source(src)}"
        )
    logger.info("策略源已创建：%s（kind=%s, latency=%s）",
                type(src).__name__, cfg.kind.value, src.latency_class)
    return src


def registered_kinds() -> tuple[str, ...]:
    """已注册的策略类型（诊断用）。"""
    return tuple(sorted(k.value for k in _REGISTRY))


__all__ = [
    "DisabledSource",
    "PolicyConfig",
    "create_policy_source",
    "register",
    "registered_kinds",
]
