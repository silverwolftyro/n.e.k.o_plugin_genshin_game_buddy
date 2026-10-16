"""意图校验：执行前的完整合法性检查。

对应架构文档 5.2 与 6.3。

**所有失败都返回具体 reason**，不静默返回 False（原则 1）。上层拿到
`"RATE_LIMITED:USE_SKILL(30<150ms)"` 才知道该等多久，而不是面对一个
`False` 去猜。

**去重与重放**：``seen_intents`` 是一个有界 set，带 TTL 清理。云端重放同一个
`intent_id` 会被这里拦下。
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from ..contracts.intent_packet import MAX_STATE_VERSION_DRIFT, ActionType, IntentPacket
from ..contracts.state_packet import StatePacket
from .traits import traits_of

logger = logging.getLogger(__name__)

#: 去重表保留多少条
DEFAULT_SEEN_SIZE = 2000
#: 去重记录保留时长（毫秒）
DEFAULT_SEEN_TTL_MS = 300_000


@dataclass
class ValidationResult:
    """校验结果。**带上原因与可读描述**，方便日志与 UI。"""

    ok: bool
    reason: str
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        return self.reason if self.ok else f"{self.reason}: {self.detail}"


@dataclass
class SeenIntentStore:
    """有界的 intent_id 去重表。"""

    capacity: int = DEFAULT_SEEN_SIZE
    ttl_ms: int = DEFAULT_SEEN_TTL_MS
    _seen: "OrderedDict[str, int]" = field(default_factory=OrderedDict, repr=False)

    def seen(self, intent_id: str, now_ms: int) -> bool:
        """是否已见过（并顺手清理过期项）。"""
        self._evict(now_ms)
        return intent_id in self._seen

    def remember(self, intent_id: str, now_ms: int) -> None:
        self._seen[intent_id] = now_ms
        self._seen.move_to_end(intent_id)
        self._evict(now_ms)

    def _evict(self, now_ms: int) -> None:
        while self._seen and (now_ms - next(iter(self._seen.values())) > self.ttl_ms
                              or len(self._seen) > self.capacity):
            self._seen.popitem(last=False)

    def __len__(self) -> int:
        return len(self._seen)


class IntentValidator:
    """意图合法性检查。**无状态**（去重表是可选注入的）。"""

    def __init__(
        self,
        *,
        seen_store: SeenIntentStore | None = None,
        allow_unsigned_cloud: bool = False,
    ) -> None:
        self._seen = seen_store if seen_store is not None else SeenIntentStore()
        self._allow_unsigned_cloud = allow_unsigned_cloud

    # -- 主入口 ----------------------------------------------------------

    def validate(
        self,
        intent: IntentPacket,
        state: StatePacket,
        now_ms: int | None = None,
        *,
        commit: bool = False,
    ) -> ValidationResult:
        """校验意图。

        ``commit=False``（默认）只检查不登记；真正执行前再调一次
        ``commit=True`` 把它写进去。这样"先看一眼能不能做"和"真的要做了"
        之间不会重复占位。
        """
        now = now_ms if now_ms is not None else int(time.time() * 1000)

        r = self._check_action(intent)
        if not r.ok:
            return r
        r = self._check_id(intent, now)
        if not r.ok:
            return r
        r = self._check_ttl(intent, now)
        if not r.ok:
            return r
        r = self._check_state_version(intent, state)
        if not r.ok:
            return r
        r = self._check_perception(state)
        if not r.ok:
            return r
        r = self._check_scene(intent, state)
        if not r.ok:
            return r
        r = self._check_position(intent, state)
        if not r.ok:
            return r
        r = self._check_signature(intent)
        if not r.ok:
            return r
        r = self._check_params(intent)
        if not r.ok:
            return r

        if commit:
            self._seen.remember(intent.intent_id, now)
        return ValidationResult(True, "OK")

    # -- 各项检查 --------------------------------------------------------

    def _check_action(self, intent: IntentPacket) -> ValidationResult:
        """action 必须在白名单内。非法值**在这里就拒**（原则 1）。"""
        try:
            action = ActionType(intent.action)
        except ValueError as exc:
            return ValidationResult(
                False, "ACTION_NOT_IN_WHITELIST",
                f"{intent.action!r} 不在 ActionType 白名单内: {exc}",
            )
        if intent.action_kind and intent.action_kind != action.kind:
            return ValidationResult(
                False, "ACTION_KIND_MISMATCH",
                f"{action.value} 实际属于 {action.kind}，但声明成 {intent.action_kind}",
            )
        if not intent.action_kind:
            logger.warning("意图 %s 缺 action_kind，按 %s 处理",
                           intent.intent_id, action.kind)
        return ValidationResult(True, "OK")

    def _check_id(self, intent: IntentPacket, now_ms: int) -> ValidationResult:
        if not intent.intent_id or not intent.intent_id.strip():
            return ValidationResult(False, "INTENT_ID_EMPTY", "intent_id 不能为空")
        # ⚠️ 必须用 ``validate()`` 传进来的 ``now_ms``，不能在这里另取
        #   ``time.time()`` —— 那会让**两条时钟混用**：调用方传合成时间戳
        #   （回放 / 确定性测试 / 合并层用状态时间戳）时，去重的 TTL 比较
        #   会对不上，重复意图因此漏过。整条校验链只认同一个 now。
        if self._seen.seen(intent.intent_id, now_ms):
            return ValidationResult(
                False, "INTENT_ID_DUPLICATE",
                f"intent_id {intent.intent_id} 已执行过（重复或重放）",
            )
        return ValidationResult(True, "OK")

    def _check_ttl(self, intent: IntentPacket, now_ms: int) -> ValidationResult:
        if intent.ttl_seconds <= 0:
            return ValidationResult(
                False, "TTL_INVALID", f"ttl_seconds={intent.ttl_seconds} 非法")
        if intent.ttl_seconds > 300:
            return ValidationResult(
                False, "TTL_TOO_LONG", f"ttl_seconds={intent.ttl_seconds} 超过上限 300")
        age = now_ms - intent.issued_at_ms
        if age < 0:
            # 时钟不同步：给一个宽限，但别当成正常
            if age < -60_000:
                return ValidationResult(
                    False, "CLOCK_SKEW",
                    f"意图的 issued_at_ms 比当前时间晚 {-age}ms，可能时钟不同步")
        elif age > intent.ttl_seconds * 1000:
            return ValidationResult(
                False, "INTENT_EXPIRED",
                f"意图已过期 {age}ms（ttl={intent.ttl_seconds}s）")
        return ValidationResult(True, "OK")

    def _check_state_version(self, intent: IntentPacket, state: StatePacket) -> ValidationResult:
        """状态漂移检查。

        为什么不能只用 TTL：云端往返 2 秒，期间世界可能从背包变成战斗，
        只看 TTL 会出现"在战斗里点背包按钮"。``state_version`` 让意图知道
        自己是为哪个状态生成的。
        """
        drift = abs(intent.state_version - state.state_version)
        if drift > MAX_STATE_VERSION_DRIFT:
            return ValidationResult(
                False, "STATE_VERSION_STALE",
                f"意图基于 sv={intent.state_version}，当前 sv={state.state_version}，"
                f"漂移 {drift} 帧（上限 {MAX_STATE_VERSION_DRIFT}）",
            )
        return ValidationResult(True, "OK")

    def _check_perception(self, state: StatePacket) -> ValidationResult:
        if not state.resource.perception_ok:
            return ValidationResult(
                False, "PERCEPTION_DOWN",
                "感知层不健康 —— 不知道自己在哪就不该动手（原则 1）")
        return ValidationResult(True, "OK")

    def _check_scene(self, intent: IntentPacket, state: StatePacket) -> ValidationResult:
        scene = state.scene.id
        if intent.target.scene_required and scene not in intent.target.scene_required:
            return ValidationResult(
                False, "SCENE_MISMATCH",
                f"意图要求场景 {list(intent.target.scene_required)}，当前 {scene}")
        if scene in intent.constraints.forbid_during:
            return ValidationResult(
                False, "SCENE_FORBIDDEN",
                f"意图禁止在 {scene} 执行")
        # traits 层面的场景许可
        t = traits_of(intent.action.value)
        if not t.allows_scene(scene):
            return ValidationResult(
                False, "SCENE_NOT_ALLOWED",
                f"动作 {intent.action.value} 的 traits 不允许场景 {scene}；"
                f"允许: {list(t.allowed_scenes)}")
        return ValidationResult(True, "OK")

    def _check_position(self, intent: IntentPacket, state: StatePacket) -> ValidationResult:
        if not intent.target.world_position:
            return ValidationResult(True, "OK")
        pos = state.position
        if pos is None or not pos.usable_for_navigation:
            q = pos.quality if pos else "NONE"
            return ValidationResult(
                False, "POSITION_UNAVAILABLE",
                f"意图带 world_position，但定位质量为 {q}（禁止用预测值填补）")
        return ValidationResult(True, "OK")

    def _check_signature(self, intent: IntentPacket) -> ValidationResult:
        if intent.source != "CLOUD":
            return ValidationResult(True, "OK")
        if not self._allow_unsigned_cloud and not intent.signature:
            return ValidationResult(
                False, "SIGNATURE_MISSING",
                "云端意图必须带签名（未配置验签时按拒绝处理）")
        return ValidationResult(True, "OK")

    def _check_params(self, intent: IntentPacket) -> ValidationResult:
        """参数完整性 —— **加载期/校验期就该发现的错，不要留到执行时**。"""
        p = intent.params or {}
        action = intent.action.value

        if action == "SWITCH_CHARACTER" and not p.get("char_id"):
            return ValidationResult(
                False, "PARAM_MISSING", "SWITCH_CHARACTER 需要 params.char_id")
        if action == "USE_BURST" and p.get("char_id") is None and not p.get("current", True):
            return ValidationResult(
                False, "PARAM_INVALID", "USE_BURST 指定了 non-current 角色却没有 char_id")
        if action == "WAIT":
            ms = p.get("ms", 1000)
            if not isinstance(ms, (int, float)) or ms < 0:
                return ValidationResult(
                    False, "PARAM_INVALID", f"WAIT.ms={ms} 非法（需 >= 0 的数字）")
            if ms > 60_000:
                return ValidationResult(
                    False, "PARAM_INVALID", f"WAIT.ms={ms} 超过上限 60000")
        if action == "CLICK_ELEMENT" and not (p.get("element") or
                                             (p.get("x") is not None and
                                              p.get("y") is not None)):
            # 两种给法都支持：按元素名（交给上层找框），或直接给屏幕坐标
            return ValidationResult(
                False, "PARAM_MISSING",
                "CLICK_ELEMENT 需要 params.element（元素名）或 params.x + params.y（屏幕坐标）")
        if intent.constraints.max_attempts < 1:
            return ValidationResult(
                False, "CONSTRAINT_INVALID",
                f"max_attempts={intent.constraints.max_attempts}，至少为 1")
        if intent.priority < 0 or intent.priority > 100:
            return ValidationResult(
                False, "PRIORITY_OUT_OF_RANGE",
                f"priority={intent.priority}，应在 0-100")
        return ValidationResult(True, "OK")

    # -- 诊断 ------------------------------------------------------------

    def seen_count(self) -> int:
        return len(self._seen)

    def forget(self, intent_id: str) -> None:
        self._seen._seen.pop(intent_id, None)  # noqa: SLF001


__all__ = [
    "DEFAULT_SEEN_SIZE",
    "DEFAULT_SEEN_TTL_MS",
    "IntentValidator",
    "SeenIntentStore",
    "ValidationResult",
]
