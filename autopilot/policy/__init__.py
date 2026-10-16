"""policy 层（决策层）。

**这层的全部价值是「可替换」**。

    现在 : 本地规则树，开源
    将来 : 本地规则树闭源（收费） / 云端策略（增值）

两种情况下主循环都只写这一行：

    from autopilot.policy import PolicySource, create_policy_source
    policy = create_policy_source(config.policy)

所以本 ``__init__`` **只导出三样东西**：

    PolicySource            协议 —— 唯一契约
    IntentSourceKind        枚举
    create_policy_source    工厂

``LocalPolicyTree`` / ``DslEvaluator`` / ``Rule`` 等具体实现**一律不导出**。
调用方一旦写了 ``from autopilot.policy.local_tree import ...``，
可替换性就破了 —— 这属于架构违规，跟 import 了别的模块不一样。

想诊断具体实现，用 ``autopilot.tools.policy_diagnose``，
它从工厂拿对象，不绕过包边界。

依赖边界
--------
只依赖 ``contracts`` + ``state``。不反向依赖 perception / actuator
（唯一例外是 rule_loader 为加载期白名单校验惰性引用 actuator.traits，
已在其模块内注明）。

**不 import ultralytics**（AGPL）—— policy 层连 ONNX 都不碰。
"""

from __future__ import annotations

from .base import IntentSourceKind, PolicySource
from .factory import PolicyConfig, create_policy_source, registered_kinds

__all__ = [
    # ↓↓↓ 对外的全部契约，就这几样 ↓↓↓
    "IntentSourceKind",
    "PolicySource",
    "create_policy_source",
    # registered_kinds() 回答的是「配置里 kind 能填什么」——
    # 那是**契约级**信息（用户据此知道有哪些可选实现），不是某个实现类的
    # 内部细节，所以放出来。LocalPolicyTree 之类仍然是封装的。
    "registered_kinds",
    # PolicyConfig 是 create_policy_source 的入参类型，不导出的话调用方
    # 无法类型标注；它不含任何行为，只是数据。
    "PolicyConfig",
]
