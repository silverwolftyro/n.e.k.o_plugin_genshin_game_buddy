"""本地 / 云端意图仲裁。

**当前状态：只交付接口与仲裁逻辑，不实现真实云端调用。**
用户已拍板「暂不隔离、暂不做云端」。这里写的是将来接入时的地基 ——
接口、校验、仲裁顺序现在定死，接云端时只补 transport，不改决策。

仲裁顺序（不可协商）
--------------------
    1. 本地有合法意图              -> 用本地
    2. 本地为 None + 云端合法      -> 用云端
    3. 其余（都无 / 云端不合法）    -> None（空转）

本地优先是刻意的：本地零延迟、零依赖、行为完全可复现。云端只在本地
**确实没有决策**时补位，而不是"云端优先级高就听云端的" ——
那意味着网络抖动能直接改变机器人行为。

云端意图的四道校验，任一不过即丢弃
----------------------------------
    TTL / state_version 漂移 / 签名 / scene_required

复用 ``IntentPacket.is_valid_for``（contracts 里的权威实现），**不重写**
—— 执行层校验的是同一套规则，决策层和执行层对"什么是合法意图"的理解
必须完全一致，否则会出现"决策层认为合法、执行层拒绝"的诡异现场。

签名：默认**无验证器即拒绝**（fail-closed）
------------------------------------------
云端意图来自网络，签名缺失等于没有来源校验。此时放行 = 任何能往管道里
塞数据的人都能指挥你的机器人按键。所以 ``signature_verifier=None`` 时
**一律拒绝云端意图**，而不是"没配就先信着"。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..contracts.events import Event, EventCategory, Severity
from ..contracts.intent_packet import IntentPacket
from ..contracts.state_packet import StatePacket

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 签名验证
# --------------------------------------------------------------------------


@runtime_checkable
class SignatureVerifier(Protocol):
    """签名验证器。**将来接云端时实现它**，本模块不含任何密钥逻辑。"""

    def verify(self, intent: IntentPacket) -> tuple[bool, str]:
        """返回 ``(是否可信, 原因)``。"""
        ...


class NullSignatureVerifier:
    """永远拒绝的验证器（默认）。

    存在的意义是让"没配验证器"这件事**显式且响亮**：配了它，云端意图
    一律带 ``SIGNATURE_REJECTED`` 被丢弃；不配它走的是同一条路，只是
    原因写得更直白。
    """

    def verify(self, intent: IntentPacket) -> tuple[bool, str]:
        return False, "SIGNATURE_UNVERIFIABLE: 未配置签名验证器"


# --------------------------------------------------------------------------
# 结果
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeResult:
    """仲裁结果。**永远带原因**（原则 1）。"""

    intent: IntentPacket | None
    #: "LOCAL" / "CLOUD" / "NONE"
    source: str
    reason: str
    #: 云端意图被丢弃的原因（为空表示云端没参与或被采用）
    cloud_rejected: str = ""
    #: 采纳云端时的校验细节，便于排障
    cloud_checks: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.intent is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.intent.action.value if self.intent else None,
            "source": self.source,
            "reason": self.reason,
            "cloud_rejected": self.cloud_rejected,
            "cloud_checks": self.cloud_checks,
        }


# --------------------------------------------------------------------------
# 云端校验
# --------------------------------------------------------------------------


def validate_cloud_intent(
    intent: IntentPacket,
    state: StatePacket,
    *,
    now_ms: int | None = None,
    signature_verifier: SignatureVerifier | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """校验云端意图。返回 ``(是否可用, 原因, 明细)``。

    明细逐项给出四项校验的结论，便于 ``policy_diagnose`` / ``replay``
    解释「为什么这条云端意图没被采纳」。
    """
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    checks: dict[str, Any] = {
        "ttl_ms": now - intent.issued_at_ms,
        "ttl_limit_ms": intent.ttl_seconds * 1000,
        "state_version": intent.state_version,
        "current_state_version": state.state_version,
        "scene_required": list(intent.target.scene_required),
        "current_scene": state.scene.id,
        "signature": bool(intent.signature),
    }

    # ① 签名 —— **先验签，再看别的**。
    #    未验签的消息不该先被逐项分析：万一它既伪造又过期，先报 EXPIRED
    #    会把"有人伪造指令"误导成"网络延迟"（原则 1：要指向真正的病根）。
    if not intent.signature:
        checks["signature_check"] = "MISSING"
        return False, "SIGNATURE_MISSING", checks
    verifier = signature_verifier or NullSignatureVerifier()
    sig_ok, sig_reason = verifier.verify(intent)
    checks["signature_check"] = sig_reason
    if not sig_ok:
        return False, f"SIGNATURE_REJECTED: {sig_reason}", checks

    # ② TTL + state_version + scene_required —— 复用 contracts 的权威实现
    ok, reason = intent.is_valid_for(state, now)
    checks["is_valid_for"] = reason
    if not ok:
        return False, reason, checks

    return True, "OK", checks


# --------------------------------------------------------------------------
# 仲裁
# --------------------------------------------------------------------------


def merge_intents(
    local_intent: IntentPacket | None,
    cloud_intent: IntentPacket | None,
    state: StatePacket,
    *,
    now_ms: int | None = None,
    signature_verifier: SignatureVerifier | None = None,
    emit_event: Callable[[Event], None] | None = None,
) -> MergeResult:
    """合并本地与云端意图，**最多产出一个**。

    :param local_intent: 本地策略的产物（可能 None = 空转）
    :param cloud_intent: 云端策略的产物（本期恒为 None，只是接口位）
    :param signature_verifier: 签名验证器；None = 一律拒绝云端（fail-closed）

    ⚠ **调用顺序契约**（否则安全闸会退化成优先级建议）::

        intent = merge_intents(local, cloud, state)     # ① 先合并
        result = executor.execute(intent, state)         # ② 再执行
        # ③ 执行被 SafetyGuard 拒绝时 —— 到此为止。
        #    **不要**把 local 拿掉再 merge 一次"试试云端的"。

    为什么：SafetyGuard 拒绝的理由（急停 / 熔断 / 场景不许 / 频率限制）
    对本地和云端**一视同仁**。绕过它去用云端，等于说「本地这条不合法，
    那换个来源的同一条动作就不合法了」—— 同一个动作、同一个场景，
    只因来源不同就放行，安全闸形同虚设。
    """
    now = now_ms if now_ms is not None else int(time.time() * 1000)

    # ① 本地优先
    if local_intent is not None:
        return MergeResult(
            intent=local_intent, source="LOCAL",
            reason="本地策略已给出意图（本地优先，云端不参与）",
        )

    # ② 本地空转 -> 看云端
    if cloud_intent is None:
        return MergeResult(
            intent=None, source="NONE",
            reason="本地空转且无云端意图 —— 本轮不产生输入（原则 3）",
        )

    ok, reason, checks = validate_cloud_intent(
        cloud_intent, state, now_ms=now,
        signature_verifier=signature_verifier,
    )
    if not ok:
        _emit(emit_event, Event(
            severity=Severity.WARN,
            category=EventCategory.DECISION,
            code="CLOUD_INTENT_REJECTED",
            message=f"云端意图被丢弃：{reason}",
            context={
                "action": cloud_intent.action.value,
                "reason": reason,
                **checks,
            },
            state_version=state.state_version,
        ))
        logger.warning("云端意图被丢弃：%s (%s)",
                       reason, cloud_intent.action.value)
        return MergeResult(
            intent=None, source="NONE",
            reason=f"本地空转，但云端意图不合法（{reason}）—— 宁可空转",
            cloud_rejected=reason, cloud_checks=checks,
        )

    return MergeResult(
        intent=cloud_intent, source="CLOUD",
        reason="本地空转，采用通过校验的云端意图",
        cloud_checks=checks,
    )


def _emit(fn: Callable[[Event], None] | None, ev: Event) -> None:
    if fn is not None:
        fn(ev)
    else:
        logger.warning("[%s] %s", ev.code, ev.message)


__all__ = [
    "MergeResult",
    "NullSignatureVerifier",
    "SignatureVerifier",
    "merge_intents",
    "validate_cloud_intent",
]
