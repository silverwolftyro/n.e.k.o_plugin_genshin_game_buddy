"""动态连招的流程控制与危机反射。

★ 这份测试在验什么
------------------
四个新原语每一个都对应一个**「错了也不会报错」**的故障：

``wait = "ready"``
    少了它 -> 技能被切人取消。**日志一条错误都没有**，症状是
    「大招按了没反应」，而玩家会去怪手柄、怪键位、怪版本。
``until``
    少了它 -> 只证明了「键按出去了」，证明不了「技能放出来了」。
``jump_to``
    少��它 -> 没有循环，攒球这类需求写不出来。
``[[reflex]]``
    少了它 -> 危险时连招还在打，等它走完人已经死了。

另外还有两个**必须响亮**的输入错误：
``jump_to`` 越界（写 5 而只有 3 步）和反射写成环形的冷却为 0
（会让引擎每 tick 闪一次）。

⚠️ ``notify_done`` 不是可选的礼貌
  ``IntentExecutor.execute`` 是**同步**的，返回即「按完键 + hold/gap
  都走完」。那是唯一一个不靠估算的「这一步真做完了」信号。
  这份测试里的 Harness 必须回报它，否则带 wait 的步骤会一路等到
  兜底超时 —— 测出来的是「等满 2 秒」而不是「等执行层返回」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.policy.combo_manager import (  # noqa: E402
    COMBO_INTENT_PRIORITY,
    DEFAULT_REFLEX_COOLDOWN_MS,
    DEFAULT_UNTIL_TIMEOUT_MS,
    MAX_JUMPS,
    REFLEX_INTENT_PRIORITY,
    ComboLoadError,
    load_combos,
)

# 复用同一份 Harness：位置语义 / 队伍 / 记账器全都在那边定义过了，
# 抄一份只会在改动时漂掉。
#
# ⚠️ 用**文件路径**导入而不是 ``from tests.unit.test_combo_sequence import``
#   ``tests`` 不是包（没有 ``__init__.py``），按包名 import 会直接
#   ModuleNotFoundError —— 而那是在 collection 阶段炸，一个用例都没跑。
#   pytest 自己往 ``sys.path`` 里塞的是 ``tests/unit``（rootdir 模式），
#   所以按文件名 import 是唯一稳的写法。
_UNIT = Path(__file__).resolve().parent
if str(_UNIT) not in sys.path:
    sys.path.insert(0, str(_UNIT))

from test_combo_sequence import (  # noqa: E402
    COMBOS_FILE,
    Harness,
    _book_from_toml,
)

pytestmark = pytest.mark.unit


def _toml(body: str, tmp_path: Path):
    return _book_from_toml(body, tmp_path)


def _combo_actions(h: Harness) -> list[str]:
    """**只**看连招发出的动作；别的一律记 ``-``。

    ★ 为什么必须按 priority 过滤而不是看动作名
      「连招让路」不等于「引擎闭嘴」。连招不产出的那一 tick，
      通用规则（priority 80 的「E 亮了放 E」）会**正常接手** ——
      那正是设计里写着的。所以 :attr:`Harness.actions` 里混着规则意图，
      而「这一步是不是被门挡住了」只能靠 priority 判定。
      不用它的话，「被挡住了」会被误读成「又放了一次」。
    """
    out = []
    for i in h.log:
        if i is None or i.priority != COMBO_INTENT_PRIORITY:
            out.append("-")
        else:
            out.append(i.action.value)
    return out


# ================================================================== wait
class TestWaitReady:
    """``wait = "ready"`` = 执行完成门。"""

    BOOK = """\
[[combo]]
name = "开盾"
trigger = "has_nearby_enemy()"
sequence = [
    { slot = 1, action = "USE_SKILL", when = "e_ready()", wait = "ready" },
    { slot = 2, action = "NORMAL_ATTACK" },
]
"""

    def test_does_not_advance_until_done_arrives(self, tmp_path) -> None:
        """★★ 核心语义：**没回报就不往下走**。

        没有这条，「开盾 -> 切人」会在套盾后摇里把人切走。
        """
        h = Harness(combos=_toml(self.BOOK, tmp_path))
        h.set_onfield(1)
        h.auto_done = False                     # 执行层还不回报
        for _ in range(3):
            h.tick(cd={"skill": True})
        assert _combo_actions(h) == ["USE_SKILL", "-", "-"], _combo_actions(h)

    def test_advances_on_the_tick_after_done(self, tmp_path) -> None:
        h = Harness(combos=_toml(self.BOOK, tmp_path))
        h.set_onfield(1)
        h.auto_done = False
        h.tick(cd={"skill": True})              # 发出去，不回报
        assert h.runner.awaiting_done
        h.runner.notify_done(ok=True)           # 执行层返回
        # 第二步是 slot = 2，而我们现在站在 1 号位 —— 所以先切人再普攻。
        h.tick(cd={"skill": True})
        got = _combo_actions(h)
        assert got[-1] == "SWITCH_CHARACTER", got

    def test_no_wait_means_no_gate(self, tmp_path) -> None:
        """对照组：不写 wait 就是不设门，每 tick 推进一步。"""
        h = Harness(combos=_toml(
            self.BOOK.replace(', wait = "ready"', ""), tmp_path))
        h.set_onfield(1)
        h.auto_done = False
        h.tick(cd={"skill": True})
        assert not h.runner.awaiting_done
        h.tick(cd={"skill": True})
        got = _combo_actions(h)
        assert got[-1] == "SWITCH_CHARACTER", got

    def test_events_report_the_step_number_users_can_look_up(
            self, tmp_path) -> None:
        """★★ 事件里的步号必须与 ``combos.toml`` 的**第 N 步**对得上。

        出队时游标已经 +1 了，所以完成那一刻 ``_step_idx`` 就是刚做完
        那一歩的 1 起编号。再多加一个 1 的话，报出来会**系统性地大 1** ——
        而用户是拿着这条事件去 TOML 里对照的，对不上会以为引擎认错了步骤。
        """
        seen: list = []
        book = _toml(
            '[[combo]]\nname = "t"\ntrigger = "has_nearby_enemy()"\n'
            'sequence = [\n'
            '    { onfield = true, action = "WAIT", wait = "ready" },\n'
            '    { onfield = true, action = "WAIT", wait = "ready" },\n'
            '    { onfield = true, action = "WAIT", wait = "ready" },\n'
            ']\n', tmp_path)
        h = Harness(combos=book)
        h.runner._emit_fn = seen.append
        for _ in range(3):
            h.tick()
        # 每次回报带一步；这里靠 stats 的 done_waits 确认它确实走了三步
        assert h.runner.stats()["done_waits"] == 3, h.runner.stats()
        assert h.runner.stats()["done_timeouts"] == 0

    def test_done_with_no_wait_is_a_noop(self, tmp_path) -> None:
        """★ 多回报一次是**无害的空操作**。

        主循环对每个意图都回报，而规则的意图也走那条路。
        回报一个没人等的门必须是 False 且什么都不做。
        """
        h = Harness(combos=_toml(
            self.BOOK.replace(', wait = "ready"', ""), tmp_path))
        assert h.runner.notify_done(ok=True) is False

    def test_executor_failure_skips_loudly(self, tmp_path) -> None:
        """★★ 执行层拒绝 -> 跳过该步并**记一笔**。

        症状是「连招在打，但某一步从来没生效过」—— 日志里没这行
        就完全查不出来。
        """
        seen: list = []
        book = _toml(self.BOOK, tmp_path)
        h = Harness(combos=book)
        h.runner._emit_fn = seen.append
        h.set_onfield(1)
        h.exec_ok = False
        h.tick(cd={"skill": True})
        assert "COMBO_STEP_NOT_EXECUTED" in [e.code for e in seen], seen

    def test_executor_crash_does_not_wedge_the_combo(self, tmp_path) -> None:
        """★★ 执行层**不回报**时的兜底：到点放行，不能永久卡住。

        卡住的连招比打得烂的连招更糟：它连「什么都没做」都不是，
        而是永远停在那一步，而现象看起来像「引擎死了」。
        """
        seen: list = []
        book = _toml(self.BOOK, tmp_path)
        h = Harness(combos=book)
        h.runner._emit_fn = seen.append
        h.set_onfield(1)
        h.auto_done = False
        h.tick(cd={"skill": True})               # 执行层卡了
        for _ in range(DEFAULT_UNTIL_TIMEOUT_MS // 200 + 3):
            h.tick(cd={"skill": True})
        assert "COMBO_DONE_TIMEOUT" in [e.code for e in seen], \
            [e.code for e in seen]
        assert h.runner.stats()["done_timeouts"] == 1
        got = [a for a in _combo_actions(h) if a != "-"]
        assert "SWITCH_CHARACTER" in got, "放行后应该继续往下走"

    def test_repeat_does_not_gate_every_shot(self, tmp_path) -> None:
        """★ ``repeat`` 的中间几次**不挂门**。

        挂了就是「等一次再做下一次」，repeat=5 会被拖成 5 倍时长。
        """
        h = Harness(combos=_toml(
            '[[combo]]\nname = "t"\ntrigger = "has_nearby_enemy()"\n'
            'sequence = [\n'
            '    { onfield = true, action = "NORMAL_ATTACK", repeat = 3, wait = "ready" },\n'
            ']\n', tmp_path))
        h.auto_done = False
        h.tick()
        h.tick()
        assert h.actions[:2] == ["NORMAL_ATTACK", "NORMAL_ATTACK"], h.actions
        assert not h.runner.awaiting_done, "中间那次不该挂门"
        h.tick()
        assert h.runner.awaiting_done, "最后一次才挂门"


class TestWaitValidation:
    @pytest.mark.parametrize("bad", ["anim", "done", "READY", "就绪"])
    def test_unknown_value_is_loud(self, bad, tmp_path) -> None:
        with pytest.raises(ComboLoadError, match="不认识"):
            _toml('[[combo]]\nname = "t"\ntrigger = "has_nearby_enemy()"\n'
                  f'sequence = [ {{ action = "USE_SKILL", wait = "{bad}" }} ]\n',
                  tmp_path)

    def test_error_lists_the_only_legal_value(self, tmp_path) -> None:
        with pytest.raises(ComboLoadError, match="ready"):
            _toml('[[combo]]\nname = "t"\ntrigger = "has_nearby_enemy()"\n'
                  'sequence = [ { action = "USE_SKILL", wait = "anim" } ]\n',
                  tmp_path)


# ================================================================== until
class TestUntil:
    """``until`` = 「这一下真的生效了吗」。"""

    #: 按完 E 之后，感知会在**下一个 tick** 看到技能进冷却。
    BOOK = """\
[[combo]]
name = "确认放出来"
trigger = "has_nearby_enemy()"
sequence = [
    { slot = 1, action = "USE_SKILL", when = "e_ready()", wait = "ready", until = "not e_ready()" },
    { slot = 2, action = "USE_BURST", when = "q_ready()" },
]
"""

    def test_waits_across_ticks_for_perception(self, tmp_path) -> None:
        """★★★ ``until`` **不是**在执行完成那一刻就地求值的。

        按完键那一刻 ``e_ready()`` 还是按之前的值 —— 技能进冷却要等
        下一帧的感知。就地求值等于「刚按完 E 就问 E 亮不亮」，
        永远答「亮」，后置条件形同虚设。
        """
        h = Harness(combos=_toml(self.BOOK, tmp_path))
        h.set_onfield(1)
        h.tick(cd={"skill": True})              # 发出去 + 回报完成
        h.tick(cd={"skill": True})              # 感知还没看到进冷却
        assert h.runner.active is True, "观察窗里连招不该继续"
        assert _combo_actions(h)[:2] == ["USE_SKILL", "-"], _combo_actions(h)

    def test_advances_once_perception_confirms(self, tmp_path) -> None:
        h = Harness(combos=_toml(self.BOOK, tmp_path))
        h.set_onfield(1)
        h.tick(cd={"skill": True})              # 发出去
        # 现在把 cd 翻掉 = 「游戏里真的放出来了」。
        # 第 2 步是 slot = 2，而我们在 1 号位 —— 所以先隐式切人，
        # **再**放大招。两个 tick，不是一个。
        h.tick(cd={"burst": True})
        h.tick(cd={"burst": True})
        got = [a for a in _combo_actions(h) if a != "-"]
        assert got == ["USE_SKILL", "SWITCH_CHARACTER", "USE_BURST"], got

    def test_giving_up_is_loud_not_silent(self, tmp_path) -> None:
        """★★ 观察窗到期仍不成立 -> ``COMBO_STEP_NO_EFFECT`` + 跳过。

        这就是「技能被动画取消了」的可见痕迹 —— 没有它，用户看到的
        只是「大招按了没反应」。
        """
        seen: list = []
        book = _toml(self.BOOK, tmp_path)
        h = Harness(combos=book)
        h.runner._emit_fn = seen.append
        h.set_onfield(1)
        h.tick(cd={"skill": True})
        for _ in range(DEFAULT_UNTIL_TIMEOUT_MS // 200 + 4):
            h.tick(cd={"skill": True})          # cd 一直亮 = 没生效
        assert "COMBO_STEP_NO_EFFECT" in [e.code for e in seen], \
            [e.code for e in seen]

    def test_no_until_means_no_observation_window(self, tmp_path) -> None:
        """★★ 对照：去掉 ``until``，同一份配置**不再被观察窗挡住**。

        两份配置只差一个 ``until``，跑同样的几拍：
        带 ``until`` 的还停在观察窗里，不带的已经走完并让路给规则。
        这条对照是为了让人一眼看出「多写那六个字换来了什么」。

        ★ 为什么是 3 拍不是 2 拍（2026-10-19）
          第 2 步是 ``slot = 2``，而我们在 1 号位 —— 所以引擎**先隐式切人**，
          **下一拍**才判 ``when = "q_ready()"``。
          这正是「先切人、后判守卫」的新顺序（见 combo_manager 的 ③）。
        """
        book = _toml(self.BOOK.replace(', until = "not e_ready()"', ""),
                     tmp_path)
        h = Harness(combos=book)
        h.set_onfield(1)
        h.tick(cd={"skill": True})     # 第 1 拍：放 E
        h.tick(cd={"skill": True})     # 第 2 拍：切到 2 号位
        h.tick(cd={"skill": True})     # 第 3 拍：判 q_ready()（没亮）-> 跳过走完
        # 第 2 步带 when = "q_ready()"，而这里 q 没亮 -> 走完并让路
        assert h.runner.active is False, "没 until 还被挡住了"

    def test_bad_expression_is_a_load_error(self, tmp_path) -> None:
        with pytest.raises(ComboLoadError, match="until 不合法"):
            _toml('[[combo]]\nname = "t"\ntrigger = "has_nearby_enemy()"\n'
                  'sequence = [ { action = "USE_SKILL", until = "nope()" } ]\n',
                  tmp_path)


# ================================================================== jump
class TestJumpTo:
    """``jump_to`` = 唯一的控制流。"""

    #: 2 号位放大招亮了就放；不亮就去普攻攒球，跳回第 2 步重试。
    BOOK = """\
[[combo]]
name = "攒球"
trigger = "has_nearby_enemy()"
sequence = [
    { slot = 2, action = "USE_BURST", when = "q_ready()" },
    { onfield = true, action = "NORMAL_ATTACK", when = "not q_ready()", jump_to = 1 },
]
"""

    def test_jumps_back_when_the_guard_is_false(self, tmp_path) -> None:
        """★★ 「能量不够先去攒球」的标准写法。"""
        h = Harness(combos=_toml(self.BOOK, tmp_path))
        h.set_onfield(2)
        for _ in range(8):
            h.tick(cd={})                       # 大招一直不亮
        # 攒球 -> 跳回第 1 步 -> 大招还是不行 -> 再攒……
        # 连招**从不结束**，所以这条 run 会一直跑下去。
        assert h.runner.active is True, "攒球循环把连招跑完了 = 没跳回去"
        assert _combo_actions(h).count("NORMAL_ATTACK") >= 3, _combo_actions(h)

    def test_jump_is_one_based(self, tmp_path) -> None:
        """★ ``jump_to = 1`` 是**第 1 步**，不是「往前 1 步」。

        两种理解只差一个 off-by-one，而症状是「跳回去了但还打错一步」，
        极难从动作序列里看出来。加载期就把范围验掉。
        """
        h = Harness(combos=_toml(
            '[[combo]]\nname = "t"\ntrigger = "has_nearby_enemy()"\n'
            'sequence = [\n'
            '    { onfield = true, action = "USE_BURST" },\n'
            '    { onfield = true, action = "WAIT" },\n'
            '    { onfield = true, action = "NORMAL_ATTACK", jump_to = 1 },\n'
            ']\n', tmp_path))
        for _ in range(4):
            h.tick(cd={"burst": True})
        got = [a for a in _combo_actions(h) if a != "-"]
        assert got == ["USE_BURST", "WAIT", "NORMAL_ATTACK", "USE_BURST"], got

    @pytest.mark.parametrize("bad", [3, 99, -1])
    def test_out_of_range_is_a_load_error(self, bad, tmp_path) -> None:
        """★★ 越界在**加载期**炸，不留到运行时。

        运行时才炸的话症状是「连招跑到某一步突然停了」，而用户
        不会想到是自己写错了步号。
        """
        with pytest.raises(ComboLoadError, match="jump_to"):
            _toml('[[combo]]\nname = "t"\ntrigger = "has_nearby_enemy()"\n'
                  'sequence = [\n'
                  f'    {{ onfield = true, action = "WAIT", jump_to = {bad} }},\n'
                  ']\n', tmp_path)

    def test_self_loop_is_cut_off_loudly(self, tmp_path) -> None:
        """★★ 写 ``jump_to`` 指回自己 = 死循环，必须有结构性兜底。

        ★ 为什么要写成**两步**而不是一步自指
          ``jump_to = 本步`` 在只有一步时根本成不了环 ——
          游标出队时 +1 就越过末尾，下一 tick 直接「走完」。
          真正的环是「A 走到末尾 -> B 跳回 A」，而那条 run
          **永远不会走完**，于是 ``timeout_ms`` 要等几十秒才兜得住。
          跳计数是结构性的：64 次跳转在任何配置下都远超正常轮转。
        """
        seen: list = []
        book = _toml(
            '[[combo]]\nname = "环"\ntrigger = "has_nearby_enemy()"\n'
            'timeout_ms = 300000\n'
            # ★ cooldown 要给长 —— 默认 0 的话中止后**下一个 tick 就重启**，
            #   于是断言 active is False 时它早就又跑起来了。看起来像
            #   「兜底没生效」，真因是这条配置自己跑得太勤。
            'cooldown_ms = 300000\n'
            'sequence = [\n'
            '    { onfield = true, action = "USE_BURST" },\n'
            '    { onfield = true, action = "NORMAL_ATTACK", jump_to = 1 },\n'
            ']\n', tmp_path)
        h = Harness(combos=book)
        h.runner._emit_fn = seen.append
        # ★ 要跑 **2 × MAX_JUMPS** 个 tick：一次跳转占两个 tick
        #   （跳回第 1 步 -> 出队第 1 步 -> 再跳）。只跑 MAX_JUMPS 个
        #   的话跳计数才到一半，看着像「兜底没生效」。
        for _ in range(MAX_JUMPS * 2 + 8):
            h.tick(cd={"burst": True})
        assert "COMBO_ABORTED" in [e.code for e in seen], [e.code for e in seen]
        assert h.runner.active is False, "跳了上限还没停 = 没兜住"


# ================================================================== reflex
REFLEX_DODGE = """\
[[reflex]]
name = "低血闪避"
action = "DODGE"
when = "hp() < 0.35"
cooldown_ms = 4000
lock_ms = 350

[[combo]]
name = "打"
trigger = "has_nearby_enemy()"
sequence = [
    { onfield = true, action = "NORMAL_ATTACK" },
]
"""


def _dodge_harness(tmp_path, hp=0.5):
    """血量走 ``tick(hp=...)``，**不是** ``PolicySignals``。

    ``hp()`` 读的是 ``StatePacket.player.hp_ratio``，而 ``PolicySignals``
    是「契约尚未承载的信号」—— 里面没有血量，且它是个 dataclass，
    不能用下标赋值。留 ``None`` 的话 ``hp()`` 返回 UNKNOWN，
    而 UNKNOWN 在 ``when`` 里判为假 -> 「低血闪避」永远不触发，
    且日志里一条线索都没有。
    """
    return Harness(combos=_toml(REFLEX_DODGE, tmp_path))


class TestReflex:
    def test_fires_above_everything(self, tmp_path) -> None:
        """★★ 反射的 priority 高于连招。

        低了的话「连招正好在放技能」就会把闪避挤掉 —— 而那正是
        最需要闪避的时刻。
        """
        h = _dodge_harness(tmp_path)
        i = h.tick(cd={}, hp=0.2)
        assert i is not None and i.action.value == "DODGE"
        assert i.priority == REFLEX_INTENT_PRIORITY

    def test_does_not_fire_when_the_guard_is_false(self, tmp_path) -> None:
        h = _dodge_harness(tmp_path)
        i = h.tick(cd={}, hp=0.9)
        assert i is None or i.action.value != "DODGE"

    def test_interrupts_a_running_combo(self, tmp_path) -> None:
        """★★ 命中即**中止**正在进行的连招 —— 这是它存在的理由。"""
        h = _dodge_harness(tmp_path)
        h.tick(cd={}, hp=0.9)                     # 连招开打
        assert h.runner.active is True
        seen: list = []
        h.runner._emit_fn = seen.append
        h.tick(cd={}, hp=0.1)                     # 危险了
        assert h.runner.active is False, "危险了连招还在打"
        assert "COMBO_ABORTED" in [e.code for e in seen]

    def test_cooldown_stops_it_repeating(self, tmp_path) -> None:
        """★★ 条件持续成立时**不能**每 tick 都闪。

        症状是「贴着怪疯狂翻滚」，比不动糟得多。
        """
        h = _dodge_harness(tmp_path)
        for _ in range(12):
            h.tick(cd={}, hp=0.1)
        fired = h.runner.stats()["reflex_fired"]
        assert 0 < fired <= 2, f"12 tick 里触发了 {fired} 次"

    def test_lock_silences_the_combo_too(self, tmp_path) -> None:
        """★★ 静默锁期间**连招也不许出手**。

        闪避刚按下去就接着放技能，等于把无敌帧浪费掉了。
        """
        h = _dodge_harness(tmp_path)
        assert h.tick(cd={}, hp=0.1).action.value == "DODGE"
        h.tick(cd={}, hp=0.1)
        h.tick(cd={}, hp=0.1)
        assert h.runner.active is False, "锁期间连招又起来了"

    def test_lock_expires(self, tmp_path) -> None:
        h = _dodge_harness(tmp_path)
        h.tick(cd={}, hp=0.1)
        for _ in range(5):
            h.tick(cd={}, hp=0.1)
        assert h.runner.active is True, "锁过了还不恢复 = 永久哑火"

    def test_never_switches_just_to_reach_its_slot(self, tmp_path) -> None:
        """★★★ 反射**绝不**为了够到目标槽位而切人。

        ★ 曾经的实现是「先切一步，下一帧再发技能」，那是**错的**：
          切人那一帧之后这条反射已经进了冷却，而真正要发的
          闪避跟着一起丢掉 —— 于是配置写了「用 3 号位大招躲」，
          实际表现是「永远只切人，不放技能」，而且**一条错误都没有**。
          症状像「反射不生效」，真因是一次机会被换成了两步。
        """
        seen: list = []
        book = _toml(
            '[[reflex]]\nname = "大招躲"\naction = "USE_BURST"\n'
            'slot = 3\nwhen = "hp() < 0.2"\ncooldown_ms = 4000\n\n'
            '[[combo]]\nname = "打"\ntrigger = "has_nearby_enemy()"\n'
            'sequence = [ { onfield = true, action = "NORMAL_ATTACK" } ]\n',
            tmp_path)
        h = Harness(combos=book)
        h.runner._emit_fn = seen.append
        h.set_onfield(1)                        # 站在 1 号位，目标 3 号位
        got = [h.tick(cd={"burst": True}, hp=0.1) for _ in range(3)]
        assert all(g is None or g.action.value != "SWITCH_CHARACTER"
                   for g in got), [g.action.value for g in got]
        assert "REFLEX_NO_TARGET" in [e.code for e in seen], \
            [e.code for e in seen]

    def test_fires_when_already_on_its_slot(self, tmp_path) -> None:
        """对照组：人就在那一格时，反射正常出手。"""
        book = _toml(
            '[[reflex]]\nname = "大招躲"\naction = "USE_BURST"\n'
            'slot = 3\nwhen = "hp() < 0.2"\ncooldown_ms = 4000\n',
            tmp_path)
        h = Harness(combos=book)
        h.set_onfield(3)
        i = h.tick(cd={"burst": True}, hp=0.1)
        assert i is not None and i.action.value == "USE_BURST", i


class TestReflexValidation:
    def test_missing_when_is_loud(self, tmp_path) -> None:
        with pytest.raises(ComboLoadError, match="必须写 when"):
            _toml('[[reflex]]\nname = "x"\naction = "DODGE"\n', tmp_path)

    def test_zero_cooldown_is_rejected(self, tmp_path) -> None:
        """★★ 冷却必须有**下界**。

        0 会让「条件成立 -> 每 tick 闪一次」。默认给 3000 就是为了
        堵这个，而显式写 0 是「我知道我在干什么吗」。
        """
        with pytest.raises(ComboLoadError, match="cooldown_ms"):
            _toml('[[reflex]]\nname = "x"\naction = "DODGE"\n'
                  'when = "hp() < 0.5"\ncooldown_ms = 0\n', tmp_path)

    def test_switch_is_not_allowed(self, tmp_path) -> None:
        """★★ 反射**不做隐式切人**。

        「闪避」的意义是立刻解决眼前这一下。切过去再闪就是延迟。
        """
        with pytest.raises(ComboLoadError, match="不能用"):
            _toml('[[reflex]]\nname = "x"\naction = "SWITCH_CHARACTER"\n'
                  'when = "hp() < 0.5"\n', tmp_path)

    def test_unknown_field_is_loud(self, tmp_path) -> None:
        with pytest.raises(ComboLoadError, match="不认识的字段"):
            _toml('[[reflex]]\nname = "x"\naction = "DODGE"\n'
                  'when = "hp() < 0.5"\ncooldownm_s = 3000\n', tmp_path)

    def test_disabled_reflex_needs_a_chinese_note(self, tmp_path) -> None:
        """★ 关着的反射**必须**说明为什么关。

        它比连招更需要 —— 一条「关掉了但没说为什么」的闪避，
        是最难在真机上发现的东西。
        """
        with pytest.raises(ComboLoadError, match="没写中文 note"):
            _toml('[[reflex]]\nname = "x"\naction = "DODGE"\n'
                  'when = "hp() < 0.5"\nenabled = false\n', tmp_path)

    def test_bad_scene_is_loud(self, tmp_path) -> None:
        with pytest.raises(ComboLoadError, match="场景"):
            _toml('[[reflex]]\nname = "x"\naction = "DODGE"\n'
                  'when = "hp() < 0.5"\nscenes = ["NOPE"]\n', tmp_path)


class TestShippedReflex:
    def test_the_shipped_file_has_reflexes(self) -> None:
        book = load_combos(COMBOS_FILE)
        assert len(book.reflexes) >= 2, book.reflexes

    def test_no_reflex_ships_enabled(self) -> None:
        """★★ 出厂配置**一条反射都不许开着**（2026-10-19）。

        「低血闪避」「濒死放大招」看着像保命键，但它们**同样会真的按键**，
        而触发条件是感知层判定的血量 —— 战斗中特效遮挡会让判读漂移。
        用户抱怨的是「默认就在打」，那就一条都不留。

        守卫的**方向**：开着的数量恒为 0。将来有人再加一条默认开着的
        危机反射，这条立刻红。
        """
        book = load_combos(COMBOS_FILE)
        assert len(book.enabled_reflexes) == 0, (
            "出厂默认不该有开着的反射："
            f"{[r.name for r in book.enabled_reflexes]}")

    def test_shipped_reflexes_name_no_character(self) -> None:
        """★★ 与连招同一条红线：反射也不许写死角色名。"""
        book = load_combos(COMBOS_FILE)
        for r in book.reflexes:
            assert not r.name.count("钟"), r.name
            assert r.slot == 0 or 1 <= r.slot <= 4, r.name

    def test_every_shipped_reflex_has_a_cooldown(self) -> None:
        """★ 默认值在这里也必须是安全的那个。"""
        assert DEFAULT_REFLEX_COOLDOWN_MS >= 1000
        for r in load_combos(COMBOS_FILE).reflexes:
            assert r.cooldown_ms > 0, r.name
