"""连招被 ``MIN_GAP`` 限流时的**重试**语义（2026-10-19）。

★ 背景：真机「连招卡死中断」的完整因果链

    连招配置里写了显式的 ``SWITCH_CHARACTER`` 步骤
      -> 带 ``char`` 的动作步**自己就会隐式切人**
      -> 于是连发两次切人指令
      -> 第二次离上一次太密，被 traits 的 ``min_gap_ms=200`` 拦成
         ``REJECTED: MIN_GAP``
      -> 切人**没按出去** -> 执行层没记账 -> 在场槽位没变
      -> 连招卡在 ``PHASE_WAIT_SWITCH`` 等「切到 X 生效」
      -> 3 秒后 ``switch_timeout_ms`` 到点 -> **整条连招被中断**

    日志里只有一句「切到 夜兰 等了 3000ms 还没生效」——
    完全看不出是被限流了，更看不出该怎么改。

★ 两个修法，缺一不可

    1. **配置层**：删掉显式的 ``SWITCH_CHARACTER`` 步骤。
       隐式切人本来就够用，多写一遍只会连发两次。
    2. **引擎层**：``MIN_GAP`` 是**节流**不是失败 ——
       动作完全合法，只是「现在太快了」。所以该**重试**，
       不该跳过、也不该等 3 秒超时。

★ 为什么第 2 点不能省

    就算配置改对了，隐式切人**仍然可能**撞上 MIN_GAP：
    纳西妲的 E 是点按（47ms + 120ms 空档 = 167ms），
    下一个切人离上一个只有 167ms < 200ms —— 还是会被拦。
    所以重试是**兜底**，配置是**治本**，两个都要。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.contracts.intent_packet import ActionType  # noqa: E402
from autopilot.policy.combo_manager import (  # noqa: E402
    _is_throttled_reason,
    load_combos,
)

# ``tests`` 不是包（没有 __init__.py），按包名 import 会直接
# ModuleNotFoundError —— 而那是在 collection 阶段炸，一个用例都没跑。
# 按**文件名** import 是唯一稳的写法（test_combo_flow_control 也这么干）。
_UNIT = Path(__file__).resolve().parent
if str(_UNIT) not in sys.path:
    sys.path.insert(0, str(_UNIT))

from test_combo_sequence import (  # noqa: E402
    COMBOS_FILE,
    Harness,
    _book_from_toml,
)

pytestmark = pytest.mark.unit


def _book(body: str, tmp_path: Path):
    return _book_from_toml(body, tmp_path)


#: 一个**需要切人**的连招：玩家在 1 号位，第一步要 2 号位。
NEEDS_SWITCH = """\
[[combo]]
name = "要切人"
trigger = "has_nearby_enemy()"
sequence = [
    { slot = 2, action = "USE_SKILL", when = "e_ready()", wait = "ready" },
]
"""

#: 一个**不需要切人**的连招：玩家就在 1 号位。
NO_SWITCH = """\
[[combo]]
name = "不切人"
trigger = "has_nearby_enemy()"
sequence = [
    { slot = 1, action = "USE_SKILL", when = "e_ready()", wait = "ready" },
]
"""


class _RejectingHarness(Harness):
    """能「拒绝执行」的 Harness。

    真实执行层在动作间隔太密时会返回 ``REJECTED: MIN_GAP:...``。
    旧 Harness 只会成功，所以它**测不出**限流路径 ——
    而那正是真机出问题的地方。
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        #: 接下来几次执行要被拒（-1 = 不拒）
        self.reject_next: int = 0
        #: 拒绝动作时也要拒吗（False = 只拒切人）
        self.reject_any: bool = False
        self.reject_reason: str = "MIN_GAP:SWITCH_CHARACTER(167<200ms)"
        #: 被拒掉的意图
        self.rejected: list = []
        #: 连招发出的事件
        self.events: list = []
        self.runner._emit_fn = self.events.append

    def _execute(self, intent) -> None:
        is_switch = intent.action is ActionType.SWITCH_CHARACTER
        if self.reject_next > 0 and (self.reject_any or is_switch):
            self.reject_next -= 1
            self.rejected.append(intent)
            # ★ 回报**带 reason** —— 与真实主循环一致（loop.py 传 res.reason）。
            #   只报 "REJECTED" 的话连招分不出「限流」和「配置错」。
            self.tree.notify_combo_done(ok=False, error="REJECTED",
                                        reason=self.reject_reason)
            return
        super()._execute(intent)


# ==================================================================
# 1 · 限流识别
# ==================================================================

def test_throttle_reasons_are_recognised() -> None:
    assert _is_throttled_reason("MIN_GAP:SWITCH_CHARACTER(167<200ms)") is True
    assert _is_throttled_reason("RATE_LIMITED:USE_SKILL(10<50ms)") is True


def test_real_failures_are_not_throttles() -> None:
    """★ 限流要重试；真失败不能重试（重试也没用）。"""
    assert _is_throttled_reason("KEYMAP: 动作 USE_SKILL 没有配置按键") is False
    assert _is_throttled_reason("USER_TAKEOVER:E") is False
    assert _is_throttled_reason("WINDOW_NOT_FOREGROUND") is False
    assert _is_throttled_reason("") is False


# ==================================================================
# 2 · 切人被限流 -> 重试，不中断
# ==================================================================

def test_throttled_switch_is_retried_not_abandoned(tmp_path) -> None:
    """★★ 切人被 MIN_GAP 拦 -> **重发一次**，连招不中断。

    旧行为：切人没记账 -> 卡在 PHASE_WAIT_SWITCH -> 3 秒超时 -> 整条中断。
    """
    h = _RejectingHarness(combos=_book(NEEDS_SWITCH, tmp_path))
    h.set_onfield(1)
    h.reject_next = 1                 # 第一次切人被限流

    first = h.tick(cd={"skill": True})
    assert first is not None
    assert first.action is ActionType.SWITCH_CHARACTER
    assert len(h.rejected) == 1, "这一次切人应该被限流拒掉"

    # ★ 下一 tick 必须**重发切人**，而不是干等
    second = h.tick(cd={"skill": True})
    assert second is not None, "被限流后连招不该卡住"
    assert second.action is ActionType.SWITCH_CHARACTER, \
        f"被限流后应该重发切人，实际是 {second.action}"

    # 重发的那次应该成功（记账器更新 -> 在场变成 2 号位）
    assert h.slot == 2, f"重试后应该切到 2 号位，实际 {h.slot}"
    assert h.runner.active, "连招不该被中断"


def test_throttled_switch_emits_a_throttle_event_not_a_failure(tmp_path) -> None:
    """★ 限流要**响亮但温和**（INFO + COMBO_STEP_THROTTLED）。

    报成 WARN / COMBO_STEP_NOT_EXECUTED 会让人以为连招出故障了。
    """
    h = _RejectingHarness(combos=_book(NEEDS_SWITCH, tmp_path))
    h.set_onfield(1)
    h.reject_next = 1
    h.tick(cd={"skill": True})

    codes = [getattr(e, "code", "") for e in h.events]
    assert "COMBO_STEP_THROTTLED" in codes, codes
    assert "COMBO_STEP_NOT_EXECUTED" not in codes, \
        "限流不是失败，不该报成「没按出去」"


def test_real_switch_failure_aborts_immediately_with_the_reason(tmp_path) -> None:
    """★ 非限流的切人失败 -> **立刻**中断并说清原因。

    旧行为是干等 3 秒 ``switch_timeout_ms`` 再报一句
    「切到 X 等了 3000ms 还没生效」—— 用户完全看不出是配置错了。
    """
    h = _RejectingHarness(combos=_book(NEEDS_SWITCH, tmp_path))
    h.set_onfield(1)
    h.reject_next = 1
    h.reject_reason = "KEYMAP: 动作 SWITCH_CHARACTER 没有配置按键"
    h.tick(cd={"skill": True})

    assert not h.runner.active, "配置错就该立刻中断，而不是等 3 秒"
    prog = h.runner.progress()
    assert "KEYMAP" in str(prog.get("last_reason") or ""), prog


# ==================================================================
# 3 · 动作被限流 -> 重试，不跳过
# ==================================================================

def test_throttled_action_is_retried_not_skipped(tmp_path) -> None:
    """★★ 技能被限流 -> **重发**，不是「当它做过了」接着往下走。

    旧行为：跳过这一步 —— 连招以为技能放过了，其实**根本没按出去**。
    """
    h = _RejectingHarness(combos=_book(NO_SWITCH, tmp_path))
    h.set_onfield(1)
    h.reject_next = 1
    h.reject_any = True              # 连技能也拒

    first = h.tick(cd={"skill": True})
    assert first is not None
    assert first.action is ActionType.USE_SKILL
    assert len(h.rejected) == 1

    # ★ 游标退回去了 -> 下一 tick 重发**同一个**动作
    second = h.tick(cd={"skill": True})
    assert second is not None, "被限流后连招不该跳过这一步"
    assert second.action is ActionType.USE_SKILL, \
        f"被限流后应该重发同一个动作，实际是 {second.action}"


def test_throttled_action_does_not_emit_not_executed(tmp_path) -> None:
    """★ 限流**不是**「没按出去」—— 那个事件留给真失败。"""
    h = _RejectingHarness(combos=_book(NO_SWITCH, tmp_path))
    h.set_onfield(1)
    h.reject_next = 1
    h.reject_any = True
    h.tick(cd={"skill": True})

    codes = [getattr(e, "code", "") for e in h.events]
    assert "COMBO_STEP_NOT_EXECUTED" not in codes, codes


def test_real_action_failure_still_skips_loudly(tmp_path) -> None:
    """★ 真失败仍然是「响亮跳过」—— 旧契约不能丢。"""
    h = _RejectingHarness(combos=_book(NO_SWITCH, tmp_path))
    h.set_onfield(1)
    h.reject_next = 1
    h.reject_any = True
    h.reject_reason = "WINDOW_NOT_FOREGROUND"
    h.tick(cd={"skill": True})

    codes = [getattr(e, "code", "") for e in h.events]
    assert "COMBO_STEP_NOT_EXECUTED" in codes, codes


# ==================================================================
# 4 · 不知道谁在场时也要切人
# ==================================================================

def test_unknown_current_slot_still_switches(tmp_path) -> None:
    """★★ 「不知道现在是谁在场」**不能**导致整步被跳过。

    ★ 为什么这条必须有
      以前「不知道目标是谁」和「不知道现在是谁在场」是一起跳过的。
      于是**删掉显式切人步之后**，连招第一步会因为「不知道在场是谁」
      被整个跳过 —— 钟离的盾根本放不出来。

      目标已知时切人**不靠猜**：按目标号的数字键，切完在场的一定是
      目标角色（显式切人步也是这么做的，它压根不需要知道在场是谁）。
    """
    h = _RejectingHarness(combos=_book(NEEDS_SWITCH, tmp_path))
    h.tracker.reset()                # 现在不知道谁在场
    h.slot = 1

    intent = h.tick(cd={"skill": True})
    assert intent is not None, "不知道在场是谁时不该整步跳过"
    assert intent.action is ActionType.SWITCH_CHARACTER, \
        f"应该照常切人，实际是 {intent.action}"


# ==================================================================
# 6 · ★ 先切人、后判守卫
# ==================================================================

def test_guard_is_evaluated_after_the_switch(tmp_path) -> None:
    """★★ 带 ``char`` 的动作步：**先切人，再判守卫**。

    ★ 为什么是这个顺序（2026-10-19）
      ``when = "e_ready()"`` 问的是**在场角色**的 E 好了没
      （``context._active_cd`` 读 ``state.player.active_member``）。
      而 ``cd_ready`` **只挂在在场角色身上** —— 它读的是**右下角 E/Q 图标**，
      那显示的是当前在场角色的技能栏。

      目标角色**还没切上场**时，他的 CD **根本读不到**（UNKNOWN -> 判为假）。

      所以先判守卫的后果是：上一个角色的 E 刚进冷却 ->
      「下一个角色那一步」的守卫读的是**上一个角色**的 CD -> 为假 ->
      那一步被**整个跳过** —— 技能**根本没放出来**，
      而日志里一条线索都没有。

    ★ 切过去之后再判，守卫读到的就是**对的人**的 CD。
      这也正是人类的操作顺序：先切过去，再看 E 亮不亮。
    """
    # 第 1 步在 1 号位（当前就在场）-> 直接判守卫
    # 第 2 步要 2 号位（不在场）  -> 先切人，下一拍才判守卫
    h = _RejectingHarness(combos=_book(NEEDS_SWITCH, tmp_path))
    h.set_onfield(1)

    # 第 1 拍：隐式切人（守卫还没判 —— 因为目标不在场）
    first = h.tick(cd={"skill": True})
    assert first is not None
    assert first.action is ActionType.SWITCH_CHARACTER, \
        f"目标不在场时应该先切人，实际是 {first.action}"

    # 第 2 拍：目标已经在场了 -> 这时才判守卫 -> 放技能
    second = h.tick(cd={"skill": True})
    assert second is not None
    assert second.action is ActionType.USE_SKILL, \
        f"切过去之后才该放技能，实际是 {second.action}"


def test_switch_happens_even_when_the_guard_is_false(tmp_path) -> None:
    """★ 守卫为假时我们**已经切过去了** —— 这是诚实的代价。

    ★ 为什么不能反过来（先判守卫再切）
      目标角色的 CD 在切过去之前**读不到**（见上一条的注释），
      所以「先判」只会读到**上一个角色**的 CD，判出来的是错的值。
      真人也是这么做的：切过去发现 E 是灰的，再切走。
    """
    h = _RejectingHarness(combos=_book(NEEDS_SWITCH, tmp_path))
    h.set_onfield(1)

    first = h.tick(cd={"skill": True})
    assert first.action is ActionType.SWITCH_CHARACTER

    # cd 里没有 "skill" -> e_ready() 是 UNKNOWN -> 守卫为假 -> 跳过
    second = h.tick(cd={"burst": True})
    assert second is not None
    assert second.action != ActionType.USE_SKILL, \
        "守卫为假就不该放技能"
    # ★ 但连招**继续往下走**（这一步被跳过，不是卡住）
    assert h.runner.active or h.runner.progress().get("phase") != "STEP"


# ==================================================================
# 5 · 发布的那份连招配置
# ==================================================================

def test_shipped_demo_combo_has_no_explicit_switch_steps() -> None:
    """★★ 发布的连招里**不许**再有显式的 SWITCH_CHARACTER 步骤。

    带 ``char`` 的动作步自己就会隐式切人；多写一遍 = 连发两次切人
    = 第二次撞 MIN_GAP = 连招卡死中断。
    """
    book = load_combos(COMBOS_FILE)
    seq = next(s for s in book if s.name == "钟纳夜丝_全自动演示")
    explicit = [s for s in seq.steps if s.is_switch]
    assert not explicit, f"还有显式切人步：{explicit}"
    # 而且每一步都指定了「对谁做」—— 隐式切人才知道切给谁
    assert all(s.char for s in seq.steps), \
        f"每一步都得指定角色：{[s.char for s in seq.steps]}"


def test_shipped_demo_combo_still_has_the_zhongli_long_press() -> None:
    """★ 精简配置时**不能**把钟离的长按弄丢了。"""
    book = load_combos(COMBOS_FILE)
    seq = next(s for s in book if s.name == "钟纳夜丝_全自动演示")
    step = next(s for s in seq.steps
                if s.action is ActionType.USE_SKILL and s.char == "钟离")
    assert step.hold_ms == 1500, f"钟离的长按丢了：{step.hold_ms}"
