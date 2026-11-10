"""本地策略树：规则 -> 意图。

决策循环只有一句话：

    按 priority 顺序遍历规则，**命中即返回**（一轮只出一个），全不命中返回 None。

三条不可动摇的规则
------------------
1. **一轮只出一个**（原则 3）。命中就 return，绝不"再多跑一条看看"。
   多步规划看起来更强，但那样"这一帧为什么按了 E"就答不上来了。
2. **空转是合法结果**。全不命中 -> ``None``，不产生任何输入。
   一条规则都不命中在战斗场景里通常意味着识别坏了，所以会发一条
   WARN 事件（:attr:`LocalPolicyTree.idle_warn_after` 帧之后），
   但**不编造**一个动作来"填补"空转。
3. **单条规则出错不毁整棵树**。``RuleEvalError`` 转成事件然后继续下一条
   —— 既不静默（事件必须发），也不让一条写坏的规则让机器人整场发呆。

场景 UNKNOWN 的额外约束
-----------------------
识别不出场景时，**只允许安全意图**（WAIT / SCREENSHOT / NOOP / CLOSE /
CANCEL）。其余动作全部拒绝 —— 在不知道身处何地时按技能，是纯粹的赌博。

连招序列（2026-10-18）
---------------------
:class:`~autopilot.policy.combo_manager.ComboRunner` 挂在本树**外面**，
由 :meth:`LocalPolicyTree.poll` 在 ``decide`` 之后拦截。契约没变：
**一轮仍然只出一个 IntentPacket**。分工是

    decide()  ->  规则命中了谁（含优先级）
    runner    ->  连招说：这一轮该出它那串里的哪一个

两者的仲裁只有三条：

1. 命中的规则 priority **小于**当前连招的 ``break_below_priority``
   -> **打断连招**，这一轮走规则（吃药、闪避、感知不可信）。
2. 否则连招说了算 —— 连招在跑就一直跑，连招空闲且 trigger 命中就启动。
3. 连招没配（``combo_runner=None``）-> 本文件行为与接入前**逐字节一致**。

⚠️ 为什么拦截点在 ``poll`` 而不是 ``decide``
------------------------------------------
``decide`` 是**纯判定**（不发意图、不记账），离线工具 ``rule_test`` /
``replay`` / ``policy_diagnose`` 全靠它。连招是有状态的跨 tick 逻辑，
塞进去会让「拿一帧状态跑一次 decide」这个契约失效。所以
:meth:`decide` **完全不知道连招存在**，只有 :meth:`poll` 知道。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..actuator.humanizer import CadenceConfig, CadenceGate
from ..contracts.events import Event, EventCategory, Severity
from ..contracts.intent_packet import ActionType, IntentPacket
from ..contracts.state_packet import SCENE_UNKNOWN, StatePacket
from .context import (
    PICK_MARK_COOLDOWN_S,
    PickMark,
    PolicyContext,
    PolicySignals,
    _entity_screen_pos,
)
from .combo_manager import ComboRunner
from .dsl import DslEvaluator, EvalOutcome, RuleEvalError
from .rule_loader import Rule, load_rules_from_dir

logger = logging.getLogger(__name__)

#: 场景 UNKNOWN 时仍允许的动作。**全是"停下来"或"看一眼"的动作** ——
#: 没有一个会改变游戏世界状态。
SAFE_ACTIONS: frozenset[ActionType] = frozenset({
    ActionType.WAIT,
    ActionType.SCREENSHOT,
    ActionType.NOOP,
    ActionType.CLOSE,
    ActionType.CANCEL,
})

#: 内置规则目录
DEFAULT_RULES_DIR = Path(__file__).resolve().parents[1] / "rules"

EmitFn = Callable[[Event], None]


# --------------------------------------------------------------------------
# 诊断
# --------------------------------------------------------------------------


@dataclass
class RuleTrace:
    """单条规则这一轮的判定过程。给 rule_test / policy_diagnose 看。"""

    rule: str
    action: str
    priority: int
    verdict: str          # MATCHED / SKIPPED / ERRORED / NOT_EVALUATED
    reason: str = ""
    unknowns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "action": self.action,
            "priority": self.priority,
            "verdict": self.verdict,
            "reason": self.reason,
            "unknowns": list(self.unknowns),
        }


@dataclass
class DecisionTrace:
    """一轮决策的完整痕迹。**命中后不再评估后面的规则**。"""

    matched: RuleTrace | None = None
    entries: list[RuleTrace] = field(default_factory=list)
    skipped_by_shortcircuit: int = 0
    #: 命中的规则本体。避免调用方拿名字再反查一遍（查不到还得处理 None）。
    matched_rule: Rule | None = field(default=None, repr=False, compare=False)

    @property
    def hit(self) -> bool:
        return self.matched is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit": self.hit,
            "matched": self.matched.to_dict() if self.matched else None,
            "entries": [e.to_dict() for e in self.entries],
            "skipped_by_shortcircuit": self.skipped_by_shortcircuit,
        }


# --------------------------------------------------------------------------
# 本地策略树
# --------------------------------------------------------------------------


class LocalPolicyTree:
    """规则树策略实现。

    ⚠ 这是**具体实现**，刻意不从 `autopilot.policy` 导出。
    主循环用 ``create_policy_source()`` 拿它，不直接 import 本类 ——
    否则"将来闭源化"就成了"改主循环"。
    """

    name = "local_tree"
    latency_class = "realtime"

    def __init__(
        self,
        *,
        rules_dir: Path | str | None = None,
        rule_files: tuple[str, ...] | None = None,
        idle_warn_after: int = 300,
        name: str | None = None,
        evaluator: DslEvaluator | None = None,
        emit_event: EmitFn | None = None,
        signals: PolicySignals | None = None,
        now_fn: Callable[[], int] | None = None,
        cadence: CadenceConfig | None = None,
        combo_runner: ComboRunner | None = None,
    ) -> None:
        if name:
            self.name = name
        self._dir = Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR
        self._idle_warn_after = idle_warn_after
        self._eval = evaluator or DslEvaluator()
        self._emit_fn = emit_event
        self._signals = signals or PolicySignals()
        self._now = now_fn or (lambda: int(time.time() * 1000))
        #: 本帧的世界实体（F2 方案：结果在 builder 上，不进 StatePacket）。
        #: ``core/loop.py`` 在 poll 之前从 ``builder.last_world_entities``
        #: 写进来。**None = 没跑实体检测** → 派生属性返回 UNKNOWN
        #: （规则保守不触发）；空 tuple = 跑了但确实没有 → False。
        self.last_world_entities: tuple | None = None

        #: ★ 加载期失败直接抛（RuleLoadError），不降级到空规则集。
        #:   "规则加载失败"和"没有规则"必须区分：前者是故障，要修；
        #:   后者是配置。悄悄退化成空规则集 = 机器人永远不动且没人知道。
        self._rules: tuple[Rule, ...] = tuple(
            load_rules_from_dir(self._dir, only=rule_files, evaluator=self._eval)
        )
        self._last_fire_ms: dict[str, int] = {}
        #: ★ 连击节奏门（2026-10-16，B1）。同样**必须存在这里而不是 PolicyContext**
        #:   ——理由与下面 _pick_marks 完全一样：PolicyContext 每 tick 重建。
        #:   真机实测连击标准差只有 0.089s，看着像节拍器；这个门把
        #:   「恒 200ms」变成「多数 200、偶尔 400、偶尔 600+」。
        self._cadence = CadenceGate(config=cadence)
        #: 「刚按过 F 的位置」标记。**必须存在这里而不是 PolicyContext** ——
        #:   PolicyContext 每 tick 由 from_state 重建一次，实例变量活不过一轮。
        #:   LocalPolicyTree 由 create_policy_source 建一次、用到进程结束。
        self._pick_marks: list[PickMark] = []
        self._idle_streak = 0
        self._idle_warned = False
        self._last_error = ""
        self._polls = 0

        #: ★ 连招状态机（2026-10-18）。**None = 没配连招 = 纯反射模式**，
        #:   此时本类的行为与接入连招之前完全一致（单测钉住这一点）。
        #:   非 None 时只在 :meth:`poll` 里被问，:meth:`decide` 永远看不见它。
        self._combos = combo_runner

        logger.info("本地策略树已就绪：%d 条规则，目录 %s%s",
                    len(self._rules), self._dir,
                    f"，连招 {len(combo_runner.book)} 条"
                    f"（开着 {len(combo_runner.book.enabled)} 条）"
                    if combo_runner is not None else "，未接连招（纯反射模式）")

    # -- PolicySource ----------------------------------------------------
    @property
    def last_error(self) -> str:
        return self._last_error

    def available(self) -> bool:
        """有规则（或者有连招）就可用。全被 enabled=false 关掉也算不可用 ——
        那说明配置有问题，调用方应该知道而不是白等。"""
        if any(r.enabled for r in self._rules):
            return True
        return self._combos is not None and bool(self._combos.book.enabled)

    @property
    def combo_runner(self) -> ComboRunner | None:
        """连招状态机。``None`` = 没配连招（纯反射模式）。"""
        return self._combos

    def abort_combo(self, reason: str) -> bool:
        """**立刻**中断正在进行的连招。返回是否真的中断了。

        给主循环用 —— 物理输入抢占（`core/loop.py::_on_physical_input`）与
        F12 急停（`_on_panic`）都会走这里。没接连招时返回 False，调用方
        不用先问一遍。
        """
        return self._combos.abort(reason) if self._combos is not None else False

    def notify_combo_done(self, *, ok: bool, error: str = "",
                          reason: str = "") -> bool:
        """转述给连招：「上一个动作真的做完了」。

        主循环拿到的是**执行结果**，它不认识连招；这一层就是那条通路。
        没有接连招时返回 ``False``，主循环那边是无害的空操作。

        :param reason: 执行层给的**详细**拒绝原因（``MIN_GAP:...`` /
            ``KEYMAP: ...`` 之类）。连招靠它分辨「该重试」还是「该中止」——
            只看 ``error``（"REJECTED"）是分不出来的。
        """
        if self._combos is None:
            return False
        return self._combos.notify_done(ok=ok, error=error, reason=reason)

    def poll(self, state: StatePacket) -> IntentPacket | None:
        """决策一轮。最多返回一个意图，全不命中返回 None。"""
        trace = self.decide(state)
        self._polls += 1

        # ★ 危机反射在最前面（2026-10-19）。
        #   顺序是刻意的：**连招 → 规则 → 反射** 才是错的 ——
        #   反射存在的全部意义就是「不管现在轮到谁，我先救这一下」。
        #   放在 decide 之后是必须的（它要复用 decide 的结果吗？不，
        #   它有自己的 when），但放在 ``_combo_step`` 之前是必须的。
        if self._combos is not None:
            reflex = self._combos.reflex_step(
                self._make_ctx(state, self._now()))
            if reflex is not None:
                self._idle_streak = 0
                self._idle_warned = False
                return reflex

        # ★ 连招拦截（2026-10-18）。**在 decide 之后、规则产出之前** ——
        #   因为「能不能打断连招」的判据就是 decide 的结果（命中规则的
        #   priority）。反过来放的话就没有任何东西能让连招让路了。
        if self._combos is not None:
            combo_intent = self._combo_step(state, trace)
            if combo_intent is not None:
                self._idle_streak = 0
                self._idle_warned = False
                return combo_intent

        if trace.hit and trace.matched_rule is not None:
            self._idle_streak = 0
            self._idle_warned = False
            self._last_fire_ms[trace.matched_rule.name] = self._now()
            return self._build_intent(trace.matched_rule, state)

        self._on_idle(state)
        return None

    def _combo_step(
        self, state: StatePacket, trace: DecisionTrace
    ) -> IntentPacket | None:
        """连招这一轮要不要出场。返回 ``None`` = 这一轮归规则管。

        两条让路条件，**顺序不能反**：

        1. 感知不健康 —— 连招的每一步都是「盲按技能」，看不见画面时
           连蓄力方向都不该有。和 :meth:`decide` 的短路同一方向。
        2. 命中了 priority 小于 ``break_below_priority`` 的规则 ——
           恢复类（感知不健康/OCR 失败/场景没把握）、被围后退、
           开锁宝箱。这些是「不打会出事」，连招必须让路
           （用户要的「没血了立刻中断」就落在这条上）。
        """
        runner = self._combos
        assert runner is not None          # 只由 poll 在非 None 时调用

        if not state.resource.perception_ok:
            runner.abort("感知层不健康")
            return None

        rule = trace.matched_rule
        if rule is not None and rule.priority < runner.break_priority:
            runner.abort(f"应急规则 {rule.name}（priority {rule.priority}）抢占")
            return None

        # ⚠️ 这里 ``decide`` 可能已经命中了某条规则，而连招仍然抢走这一轮。
        #   那条规则的 ``cooldown_ms`` **不会**被记账（``_last_fire_ms``
        #   只在下面那条路径里写）—— 所以它不消耗任何冷却额度。
        #   唯一被消耗的是 ``CadenceGate``（它在 decide 里就记账），
        #   影响范围是「同一种动作在连招期间被稍微往后推一点」，
        #   方向上是**对的**：连招期间本来就不该让通用规则插手。
        return runner.next_intent(self._make_ctx(state, self._now()))

    # -- 诊断接口 --------------------------------------------------------
    def decide(self, state: StatePacket) -> DecisionTrace:
        """同 :meth:`poll`，但返回完整痕迹。**不发意图**，供离线工具用。

        刻意把"判定"和"产意图"分开：这样 rule_test 可以在**不接执行层**的
        前提下看清每条规则的判断过程（原则 4：工具 = 运行时）。
        """
        trace = DecisionTrace()
        now = self._now()

        # ① 感知层不健康 -> 一律不产意图。契约里写明了这一点（ResourceInfo）
        if not state.resource.perception_ok:
            trace.skipped_by_shortcircuit = len(self._rules)
            self._emit(
                Event(
                    severity=Severity.WARN,
                    category=EventCategory.DECISION,
                    code="PERCEPTION_DOWN_NO_DECISION",
                    message="感知层不健康，本轮不产出任何意图",
                    context={"scene": state.scene.id},
                    state_version=state.state_version,
                )
            )
            return trace

        ctx = self._make_ctx(state, now)

        for idx, rule in enumerate(self._rules):
            t = RuleTrace(rule.name, rule.action.value, rule.priority,
                          verdict="SKIPPED")

            if not rule.enabled:
                t.reason = "规则被 enabled=false 禁用"
                trace.entries.append(t)
                continue

            if not rule.matches_scene(state.scene.id):
                t.reason = f"场景不匹配（当前 {state.scene.id}）"
                trace.entries.append(t)
                continue

            if (state.scene.id == SCENE_UNKNOWN
                    and rule.action not in SAFE_ACTIONS):
                t.reason = (
                    f"场景 UNKNOWN，只允许安全意图"
                    f"（{sorted(a.value for a in SAFE_ACTIONS)}）"
                )
                trace.entries.append(t)
                continue

            last = self._last_fire_ms.get(rule.name)
            if rule.cooldown_ms > 0 and last is not None \
                    and (now - last) < rule.cooldown_ms:
                t.reason = (f"冷却中（{now - last}ms < "
                            f"{rule.cooldown_ms}ms）")
                trace.entries.append(t)
                continue

            # ★ 节奏门（2026-10-16，B1）。位置**紧跟冷却之后、求值之前**：
            #   顺序有讲究 —— 先挡掉明显不该发的（冷却），再挡节奏，
            #   最后才跑 DSL 求值。反过来的话每个 tick 都要白求值一次。
            #
            #   ⚠️ 与冷却是两个独立的闸门，**不是**同一个机制：
            #     冷却是规则作者写的硬下限（有它的道理：技能 CD），
            #     节奏门是拟人化的分布。前者不能被后者改写，反之亦然。
            #   这里只做「纯判断」，真正的记账在命中之后。
            action_name = rule.action.value
            if not self._cadence.allows(action_name, now):
                due = self._cadence.next_due_ms(action_name) or now
                t.reason = f"节奏门控中（还需 {max(0, due - now)}ms）"
                trace.entries.append(t)
                continue

            # ★ 求值。异常转事件后继续，绝不静默也绝不让一条坏规则毁整棵树
            try:
                out: EvalOutcome = self._eval.evaluate(
                    rule.compiled, ctx,
                    rule_name=rule.name,
                    source_file=rule.source_file,
                    line_no=rule.line_no,
                )
            except RuleEvalError as exc:
                t.verdict = "ERRORED"
                t.reason = str(exc)
                trace.entries.append(t)
                self._last_error = str(exc)
                self._emit(
                    Event(
                        severity=Severity.ERROR,
                        category=EventCategory.DECISION,
                        code="RULE_EVAL_ERROR",
                        message=f"规则 {rule.name} 求值失败：{exc.message}",
                        context=exc.to_dict(),
                        state_version=state.state_version,
                    )
                )
                logger.error("规则 %s 求值失败：%s", rule.name, exc)
                continue

            t.unknowns = out.unknowns
            if not out.value:
                t.reason = out.reason or "条件为假"
                if out.unknowns:
                    # 未知导致的未命中要区分出来 —— 它通常意味着识别缺口
                    t.reason += f"（未知字段 {len(out.unknowns)} 个）"
                trace.entries.append(t)
                continue

            t.verdict = "MATCHED"
            t.reason = "命中"
            trace.entries.append(t)
            trace.matched = t
            trace.matched_rule = rule
            # ★ 拾取标记（2026-10-11 真机事故修复）。
            #   规则**命中**就意味着「对某个东西按了 F」，所以标记记在这里 ——
            #   而不是等执行层回报成功。那样做需要改 core（引擎核心），
            #   而这里的效果等价：命中即尝试，没命中的自然没按。
            #   位置取「刚被这条规则瞄上的那个实体」—— has_nearby_* 命中的
            #   不一定只有一个，按最近拾取的可交互分组实体记。
            if rule.action.value == "USE_PICKUP":
                self._remember_pick(ctx, now)
            # ★ 节奏门记账（2026-10-16，B1）。**命中才记账**，与上面的
            #   pick_mark 同一个理由：条件为假的那几轮不该消耗掉一次机会，
            #   否则敌人一消失，下次打出来就莫名多等一个间隔，手感变慢。
            self._cadence.note_fired(action_name, now)
            # ★ 命中即停：后面的规则这一轮不再评估（原则 3）
            trace.skipped_by_shortcircuit = len(self._rules) - idx - 1
            break

        return trace

    # -- 拾取标记 ------------------------------------------------------
    #: 这些分组的实体值得「记住按过了」
    _PICKABLE_GROUPS = ("chest", "ore", "drops")

    def _remember_pick(self, ctx, now_ms: int) -> None:
        """把刚被 USE_PICKUP 瞄上的实体位置记下来。"""
        ents = getattr(ctx, "world_entities", None) or ()
        best: PickMark | None = None
        for e in ents:
            group = getattr(e, "group", getattr(e, "kind", ""))
            if group not in self._PICKABLE_GROUPS:
                continue
            pos = _entity_screen_pos(e)
            if pos is None:
                continue
            cand = PickMark(group=str(group), x=pos[0], y=pos[1], ts_ms=now_ms)
            # 同一帧命中多个时记**第一个** —— 下一帧它们都会因为这个标记
            # 被压住，取哪个对「别重复按」这个目标是等价的。
            if best is None:
                best = cand
        if best is not None:
            self._pick_marks.append(best)
            # 顺手淘汰过期的，列表不会随运行时间线性增长
            cutoff = now_ms - int(PICK_MARK_COOLDOWN_S * 1000) * 2
            self._pick_marks = [m for m in self._pick_marks if m.ts_ms >= cutoff]
            logger.debug("已记拾取标记 %s @ (%.0f, %.0f)",
                         best.group, best.x, best.y)

    # -- 内部 ------------------------------------------------------------
    def _make_ctx(self, state: StatePacket, now: int) -> PolicyContext:
        """造这一帧的 DSL 上下文。

        ★ 抽成方法是给连招用的：连招的 trigger 与每步的 when 要用
          **同一个**上下文求值（同一个 ``last_world_entities``、同一批
          ``pick_marks``）。两边各造一个不是不行，但那样「规则看到的
          世界」和「连招看到的世界」就可能不是同一个了 —— 那种不一致
          排查起来最难，因为它表现为「条件明明一样，一边真一边假」。
        """
        return PolicyContext.from_state(
            state, now_ms=now, signals=self._signals,
            world_entities=self.last_world_entities,
            pick_marks=tuple(self._pick_marks),
        )

    def _build_intent(self, rule: Rule, state: StatePacket) -> IntentPacket:
        return IntentPacket.make(
            rule.action,
            params=rule.params,
            state_version=state.state_version,
            ttl_seconds=rule.ttl_seconds,
            priority=rule.priority,
            scene_required=rule.scenes,
            source="LOCAL",
            constraints={"max_attempts": rule.max_attempts},
        )

    def _on_idle(self, state: StatePacket) -> None:
        """连续空转告警。**只告警，不补动作。**"""
        self._idle_streak += 1
        if self._idle_warned or self._idle_warn_after <= 0:
            return
        if self._idle_streak < self._idle_warn_after:
            return
        self._idle_warned = True
        self._emit(
            Event(
                severity=Severity.WARN,
                category=EventCategory.DECISION,
                code="POLICY_IDLE",
                message=(
                    f"连续 {self._idle_streak} 轮没有任何规则命中。"
                    f"若是战斗场景，通常是识别出了问题而不是真的没事干。"
                ),
                context={
                    "scene": state.scene.id,
                    "rules_evaluated": len(self._rules),
                    "streak": self._idle_streak,
                },
                state_version=state.state_version,
            )
        )
        logger.warning("策略连续空转 %d 轮（场景 %s）",
                       self._idle_streak, state.scene.id)

    def _emit(self, ev: Event) -> None:
        """发事件。没接 sink 就退化成日志 —— 但**绝不静默丢弃**。"""
        if self._emit_fn is not None:
            self._emit_fn(ev)
        elif ev.severity is Severity.ERROR:
            logger.error("[%s] %s", ev.code, ev.message)
        else:
            logger.info("[%s] %s", ev.code, ev.message)

    # -- 只读属性 --------------------------------------------------------
    @property
    def rules(self) -> tuple[Rule, ...]:
        """已加载规则（按 priority 升序）。"""
        return self._rules

    @property
    def rules_dir(self) -> Path:
        return self._dir

    def stats(self) -> dict[str, Any]:
        by_scene: dict[str, int] = {}
        by_action: dict[str, int] = {}
        for r in self._rules:
            for s in (r.scenes or ("*",)):
                by_scene[s] = by_scene.get(s, 0) + 1
            by_action[r.action.value] = by_action.get(r.action.value, 0) + 1
        return {
            "total": len(self._rules),
            "enabled": sum(1 for r in self._rules if r.enabled),
            "polls": self._polls,
            "idle_streak": self._idle_streak,
            "rules_dir": str(self._dir),
            "by_scene": dict(sorted(by_scene.items())),
            "by_action": dict(sorted(by_action.items())),
            "last_error": self._last_error,
            "combos": self._combos.stats() if self._combos is not None else None,
        }


__all__ = [
    "DEFAULT_RULES_DIR",
    "SAFE_ACTIONS",
    "DecisionTrace",
    "LocalPolicyTree",
    "RuleTrace",
]
