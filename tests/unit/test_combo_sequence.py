"""连招序列（``combos/combos.toml`` + ``policy/combo_manager.py``）行为测试。

★ 这份测试在验什么
------------------
不是「配置能不能解析」—— 那由 :class:`ComboLoadError` 的响亮失败保证。
它验的是**五件错了都不会报错的事**：

1. **顺序对**：4 步序列必须按「钟离 E → 夜兰 Q → 纳西妲 E → 绫华普攻」
   的顺序发意图。写错顺序不会报错，只会静默打一套错误的连招。
2. **隐式切人插在正确位置**：写 ``{ char = "夜兰", action = "USE_BURST" }``
   时引擎要先切人。切早了会「钟离的大招」、切晚了会「切完人就站着」。
3. **一 tick 一个意图**：连招不是「一次发一串」。发一串会绕过
   执行层的限频、安全闸和按键释放。
4. **抢占立刻清空**：用户一按键，半条连招必须作废。
5. **没配置就退回纯反射**：``combo_runner=None`` 时决策层行为不变。

★ 为什么从**发布的那份** ``combos.toml`` 加载
------------------------------------------
同 ``test_combo_ladder.py`` 的理由：另抄一份的话两边迟早漂移，
而漂移的正是最该被测的那部分（玩家真正写的那份）。要测变体时用
:func:`_book_from_toml` 在 ``tmp_path`` 里现写。

覆盖链路：rois.toml -> 实体注入 -> has_nearby_enemy()
          CharSlotIndex（名册）+ ActiveSlotTracker（记账）
            -> 隐式切人 -> onfield/PartyMember -> e_ready()/q_ready()
              -> LocalPolicyTree.poll -> ComboRunner -> IntentPacket
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.active_slot import (  # noqa: E402
    ActiveSlotTracker, CharSlotIndex,
)
from autopilot.actuator.validator import IntentValidator  # noqa: E402
from autopilot.contracts.intent_packet import (  # noqa: E402
    ActionType, IntentPacket,
)
from autopilot.contracts.state_packet import (  # noqa: E402
    PartyMember, PlayerInfo, ResourceInfo, SceneInfo, StatePacket,
    SCENE_UNKNOWN, SCENE_WORLD,
)
from autopilot.policy.combo_manager import (  # noqa: E402
    COMBO_INTENT_PRIORITY, ComboBook, ComboLoadError, ComboRunner,
    load_combos, load_optional_combos,
)
from autopilot.policy.context import PolicySignals  # noqa: E402
from autopilot.policy.dsl import DslEvaluator  # noqa: E402
from autopilot.policy.local_tree import LocalPolicyTree  # noqa: E402
from autopilot.policy.rule_loader import load_rules_from_dir  # noqa: E402

RULES_DIR = PLUGIN_ROOT / "autopilot" / "rules"
COMBAT_FILE = RULES_DIR / "genshin_combat.toml"
COMBOS_FILE = PLUGIN_ROOT / "autopilot" / "combos" / "combos.toml"

#: ★ 随包配置守卫的**具名豁免**（2026-10-19 晚，用户明确要求）。
#:
#: 用户要求源码树默认开启一条四神演示连招，供 ``run_demo.bat`` 一键演示。
#: 它**必然**违反三条出厂守卫：
#:   1. 出厂一条都不许开着；
#:   2. 配置里不许写死角色名；
#:   3. 每一步要么 ``slot`` 要么 ``onfield``。
#:
#: 为什么豁免而**不是删掉守卫**：
#:   这三条守的是真实事故 —— 「随手加一条写死角色名的连招」只对
#:   A_120^4 种队伍中的一种成立，其余 20 多万种会「永远不触发」或
#:   「对着队友按技能」。豁免做成**集合**之后，往里加第二个名字就会
#:   让守卫继续红；只有这一个名字是白名单。
#:
#: ⚠️ 这条例外只在**源码树**成立。发布包 ``combos.toml`` 仍然全关 ——
#:   ``verify_release_zip.py`` 第 14 段查的就是那个，不受这里影响。
#: ⚠️ 接管决定仍然由用户自己按下：``run_demo.bat`` 显式带 ``--live``，
#:   不是靠默认值代劳。
NAMED_EXCEPTIONS: set[str] = {"钟纳夜丝_全自动演示"}

#: 队伍（也是 keymap.toml 注释里那份示例）
PARTY = [(1, "钟离"), (2, "夜兰"), (3, "纳西妲"), (4, "神里绫华")]
SLOT_OF = {name: slot for slot, name in PARTY}
#: ``slot`` 式配置的反查表。**故意和上面那份不一样** ——
#: 见 ``TestSlotStyle::test_slot_resolves_through_the_roster``。
NAME_OF = {slot: name for slot, name in PARTY}

#: 一帧 200ms（5Hz 主循环）
TICK_MS = 200


# ---------------------------------------------------------------- 夹具
@dataclass
class FakeEnemy:
    """够用的假敌人实体。

    ``is_enemy_evidence`` 走判据 B：血条 + 上半屏 + **高度达标**
    （``ENEMY_HEALTH_BAR_H_MIN = 0.015``，2026-10-18 加，用来挡 UI 细条）。
    这里用**实测**的真敌人血条尺寸
    （``notes/a5_attack_gating_root_cause.md`` §2.3），不随手编 ——
    随手编成 h=0.01 的话会被尺寸下限挡掉，症状是「连招永远不触发」，
    而原因看起来像 trigger 写错了。
    """
    #: 实测：真敌人血条 bbox=(0.30, 0.22, 0.05, 0.02)
    bbox: tuple = (0.30, 0.22, 0.05, 0.02)
    kind: str = "health_bar"
    group: str = "combat"
    confidence: float = 0.9
    screen_pos: tuple = (360, 260)


class Harness:
    """把「决策 + 执行」的一个 tick 复现出来。

    ★ 为什么要连执行层一起模拟
      切人意图发出去**不会**立刻改「谁在场」—— 记账发生在执行层按下
      数字键**之后**（``executor.py`` 的 ``_note_switch`` 在 ``check()``
      之后）。只跑决策不跑执行，测出来的「切完人下一 tick 就能放大招」
      是一种现实里不存在的时序。
    """

    def __init__(self, *, combos: ComboBook | None = None,
                 enable: bool = True,
                 rules_dir: Path | None = RULES_DIR,
                 rule_files: tuple[str, ...] | None = ("genshin_combat.toml",),
                 signals: PolicySignals | None = None,
                 static_slots: dict | None = None) -> None:
        self.clock = {"ms": 0}
        self.signals = signals or PolicySignals()
        self.tracker = ActiveSlotTracker(now_fn=lambda: self.clock["ms"])
        self.index = CharSlotIndex()
        for slot, name in PARTY:
            self.index.note_name(slot, name)
        #: 名册里查不到人的集合（模拟「OCR 还没填满」）
        self.missing: set[str] = set()
        self.slot = 1
        self.tracker.note_switch(1, source="engine")

        if combos is None:
            # ★ 默认是**测试自己**的梯子，不是发布的那份 ——
            #   理由见 :data:`LADDER_TOML`。
            combos = _ladder_book()
        if not enable:
            combos = ComboBook(
                sequences=tuple(dataclasses.replace(s, enabled=False)
                                for s in combos),
                reflexes=combos.reflexes,
                source_file=combos.source_file)
        self.book = combos
        self.runner = ComboRunner(
            combos, now_fn=lambda: self.clock["ms"],
            active_slots=self.tracker, slot_index=self._index_view(),
            static_slots=static_slots)

        self.tree = LocalPolicyTree(
            rules_dir=rules_dir, rule_files=rule_files,
            now_fn=lambda: self.clock["ms"], signals=self.signals,
            combo_runner=self.runner)
        #: 本次跑出来的意图（按 tick 顺序）
        self.log: list = []
        #: 执行层这次返回 ok 还是失败。默认成功。
        self.exec_ok = True
        #: 执行层会不会回报完成。关掉它就能测「执行层卡住了」那条兜底。
        self.auto_done = True

    def _index_view(self):
        """名册视图。``missing`` 里的角色当查不到 —— 模拟 OCR 没填满。"""
        outer = self

        class _View:
            @staticmethod
            def slot_of(char_id: str):
                if char_id in outer.missing:
                    return None
                return SLOT_OF.get(char_id)

            # ★ ``name_of`` 是**方法**不是 property（CharSlotIndex 就是
            #   这样）。``ComboRunner._name_of`` 拿到它之后是 ``fn(key)``，
            #   不是 ``fn.key`` —— 写成后者会拿到 bound method，
            #   那一步的 TypeError 被 except 吃掉，变成「永远查不到」。
            @staticmethod
            def name_of(slot: int):
                return NAME_OF.get(slot)
        return _View()

    # -- 一帧 --------------------------------------------------------
    def set_onfield(self, slot: int) -> None:
        """把「现在站着谁」挪到 ``slot``。

        ★ 必须**连记账器一起挪**
          ``ComboRunner._current_slot()`` 问的是
          :class:`ActiveSlotTracker`，不是 ``self.slot``。
          只改 ``h.slot`` 的话连招看到的还是 1 号位 —— 于是
          「已经在 2 号位了就不该切人」这条会拿到一个切人意图，
          测出来的失败原因指向引擎，而真因在测试。
        """
        self.slot = slot
        self.tracker.note_switch(slot, source="engine")

    def tick(self, *, enemy: bool = True, scene: str = SCENE_WORLD,
             cd: dict | None = None, perception_ok: bool = True,
             hp: float = 1.0, execute: bool = True):
        """走一 tick，返回 ``IntentPacket`` 或 None，并记进 :attr:`log`。

        :param hp: 血量比例。**必须显式给一个值**（默认满血）——
            留 ``None`` 的话 ``hp()`` 在 DSL 里是 UNKNOWN，
            而 UNKNOWN 在 ``when`` 里判为假 —— 于是「低血闪避」
            这类反射**永远不触发**，而日志里一条线索都没有。
        """
        self.clock["ms"] += TICK_MS
        self.tree.last_world_entities = (FakeEnemy(),) if enemy else ()
        intent = self.tree.poll(self._state(scene, cd, perception_ok, hp))
        self.log.append(intent)
        if intent is not None and execute:
            self._execute(intent)
        return intent

    def _execute(self, intent) -> None:
        """模拟执行层。**只做切人记账** —— 别的动作不影响决策。

        切人记账之后还要**回报执行完成**（``notify_combo_done``）——
        这是 ``wait = "ready"`` 唯一的信号源。不回报的话，带 wait 的
        那一步会一路等到兜底超时才放行，测出来的是「等满 2 秒」，
        而不是「等执行层返回」。
        """
        if intent.action is not ActionType.SWITCH_CHARACTER:
            pass
        else:
            want = intent.params.get("char_id")
            slot = self.runner._slot_of(want or "")
            if slot is None:
                return                      # 执行层会跳过并记一笔
            self.slot = slot
            self.tracker.note_switch(slot, source="engine")
        if self.auto_done:
            self.tree.notify_combo_done(ok=self.exec_ok,
                                        error="" if self.exec_ok else "REJECTED")

    def _state(self, scene: str, cd: dict | None,
               perception_ok: bool, hp: float = 1.0) -> StatePacket:
        members = []
        for slot, name in PARTY:
            is_active = slot == self.slot
            members.append(PartyMember(
                slot=slot, char_id=name, name=name, active=is_active,
                # ★ cd 只挂在在场角色身上 —— 与 builder 的实际接线一致。
                cd_ready=dict(cd or {}) if is_active else {},
            ))
        return StatePacket(
            scene=SceneInfo(id=scene, confidence=0.9),
            player=PlayerInfo(party=tuple(members), hp_ratio=float(hp),
                              hp_source="test"),
            resource=ResourceInfo(perception_ok=perception_ok),
            state_version=self.clock["ms"],
        )

    # -- 断言辅助 ----------------------------------------------------
    @property
    def actions(self) -> list[str]:
        """每个 tick 的动作名。None（空转）记成 ``"-"``。"""
        return [i.action.value if i is not None else "-" for i in self.log]

    @property
    def tags(self) -> list[str]:
        """``动作:角色``，隐式切人看得见切的是谁。"""
        out = []
        for i in self.log:
            if i is None:
                out.append("-")
                continue
            who = i.params.get("char_id") or ""
            out.append(f"{i.action.value}:{who}" if who else i.action.value)
        return out


def _enabled_shipped_book() -> ComboBook:
    """从**发布的那份** combos.toml 加载，并把 enabled 翻成 True。

    发布的那份故意是关着的，而测试要的是「**同样那条连招**打开之后的
    行为」，不是另抄一条打开的。
    """
    book = load_combos(COMBOS_FILE)
    return ComboBook(
        sequences=tuple(dataclasses.replace(s, enabled=True) for s in book),
        reflexes=book.reflexes,
        source_file=book.source_file)


#: ★ 测试**自己**的四人梯子（带写死的角色名）。
#:
#: 为什么不能让 :class:`Harness` 默认用发布的那份
#:   2026-10-19 之前发布的那份**就是**这条梯子，于是 ``Harness()``
#:   等于「用发布配置跑」。发布配置一改（这次把写死的角色名全删了），
#:   十几条测「顺序 / 抢占 / 打断 / repeat」的行为测试集体变红 ——
#:   而它们验的东西跟「发布的那条连招叫什么」**毫无关系**。
#:
#:   顺序、抢占、打断这些语义属于 :class:`ComboRunner`，不属于
#:   某一份配置文件。所以梯子搬到测试里，发布配置只留给
#:   :class:`TestShippedConfig` 专门验。
#:
#: 这里仍用 ``char =`` 写法是**故意的**：它守着「老配置不炸」这条
#:   兼容承诺。位置语义在 :class:`TestSlotStyle` / :class:`TestOnfieldStyle`。
LADDER_TOML = """\
[[combo]]
name = "四人梯子"
trigger = "has_nearby_enemy()"
cooldown_ms = 12000
sequence = [
    { char = "钟离",     action = "USE_SKILL" },
    { char = "夜兰",     action = "USE_BURST", when = "q_ready()" },
    { char = "纳西妲",   action = "USE_SKILL" },
    { char = "神里绫华", action = "NORMAL_ATTACK", repeat = 5 },
]
"""


@lru_cache(maxsize=1)
def _ladder_book() -> ComboBook:
    """那条四人梯子。缓存一份 —— 它是 frozen 的，跑多少次都一样。"""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="neko_combo_ladder_"))
    return _book_from_toml(LADDER_TOML, d)


def _book_from_toml(text: str, tmp_path: Path) -> ComboBook:
    """把一段 TOML 写成真文件再加载 —— 走的是和发布那份**同一条**解析路径。"""
    p = tmp_path / "combos.toml"
    p.write_text(text, encoding="utf-8")
    return load_combos(p)


def _rules():
    return tuple(load_rules_from_dir(RULES_DIR, only=("genshin_combat.toml",)))


def _emergency_rule(template):
    """造一条 priority 5 的应急规则，条件 ``enemy_count() >= 2``。

    ``template`` 只是拿来提供别的字段（source_file 之类），
    ``dataclasses.replace`` 保得住 frozen 与 compiled 之外的东西。
    """
    when = "enemy_count() >= 2"
    return replace(template, name="emergency_when_swarmed",
                   action=ActionType.WAIT, when=when, priority=5,
                   cooldown_ms=0, enabled=True,
                   params={}, scenes=("WORLD",),
                   compiled=DslEvaluator().compile(when, rule_name="emergency"))


# ==================================================================
# 1 · 发布的那份配置本身
# ==================================================================
class TestShippedConfig:
    def test_shipped_file_parses(self) -> None:
        book = load_combos(COMBOS_FILE)
        assert len(book) >= 1

    def test_the_shipped_config_names_no_character(self) -> None:
        """★★ 发布的那份配置**一个角色名都不许写死**。

        这是 2026-10-19 这一轮改动的**唯一硬目标**。原来那条「钟夜」
        对 ``A_120^4`` 种队伍只对一种成立 —— 其余 20 多万种组合下，
        引擎会去找一个不存在的名字，表现为「连招永远不触发」或者
        「对着队友按技能」。

        守卫写成**扫全文**而不是「检查某几步」：新加一条连招、
        在注释里留个名字，都会被这条抓住。

        ⚠️ 唯一豁免见 :data:`NAMED_EXCEPTIONS`。豁免是**具名**的 ——
          删掉守卫会让「随手再写一条写死名字的连招」畅通无阻，
          而那正是这条当初要挡的东西。
        """
        book = load_combos(COMBOS_FILE)
        offenders = [
            f"{s.name} 第 {i + 1} 步 char={st.char!r}"
            for s in book if s.name not in NAMED_EXCEPTIONS
            for i, st in enumerate(s.steps) if st.char
        ]
        assert offenders == [], f"发布配置里还有写死的角色名：{offenders}"

    def test_every_shipped_step_is_positional(self) -> None:
        """★ 发布的每一步要么 ``slot`` 要么 ``onfield``。

        ``char`` 仍被解析器接受（老配置不炸），但**发布的那份不许用** ——
        兼容字段留在代码里，才不会逼着已有配置立刻改。

        ⚠️ 同上，豁免 :data:`NAMED_EXCEPTIONS`。
        """
        for s in load_combos(COMBOS_FILE):
            if s.name in NAMED_EXCEPTIONS:
                continue
            for i, st in enumerate(s.steps):
                assert st.is_positional, (
                    f"{s.name} 第 {i + 1} 步既没写 slot 也没写 onfield："
                    f"{st.describe()}")

    def test_every_disabled_combo_explains_why(self) -> None:
        """关着的连招必须有中文 note，否则「临时关一下」会变成垃圾桶。"""
        for s in load_combos(COMBOS_FILE):
            if not s.enabled:
                assert s.note.strip(), s.name
                assert any("一" <= ch <= "鿿" for ch in s.note), s.name

    def test_every_disabled_combo_in_the_file_explains_why(self, tmp_path) -> None:
        """加载器在**加载期**就拒「enabled=false 但没有 note」。"""
        with pytest.raises(ComboLoadError, match="没有 note"):
            _book_from_toml(
                '[[combo]]\nname = "x"\nenabled = false\n'
                'trigger = "has_nearby_enemy()"\n'
                'sequence = [{ char = "钟离", action = "USE_SKILL" }]\n',
                tmp_path)

    #: ★ 随包那份「通用战斗」连招。2026-10-19 用户要求「让用户能直接拿去演示」
    #: ——但**不是**原来的「钟夜」。「钟夜」写死了两个角色名，对
    #: A_120^4 种队伍只对一种成立。换成「通用战斗」：它从不问队伍是谁，
    #: E 亮就按、没亮就普攻，**对任何队伍都成立**。
    #:
    #: ⚠️ 2026-10-19：它**不再默认开启**（见下面的出厂全关断言）。
    #:   下面三条「它长什么样」的守卫因此改成按名字找，而不是按 enabled 找 ——
    #:   关着不等于删掉，它仍是「用户想开就开」的那一条。
    SHIPPED_DEMO = "通用战斗"

    def _demo(self):
        """随包那份「通用战斗」。

        ★ 为什么用「找不到就报错」而不是 ``next(..., None)``
          这条连招是「用户想开就开」的唯一入口。它要是被改名或删了，
          下面的三条守卫就该**红着**，而不是静悄悄全 skip。
        """
        book = load_combos(COMBOS_FILE)
        got = [s for s in book if s.name == self.SHIPPED_DEMO]
        assert len(got) == 1, (
            f"随包配置里找不到唯一的「{self.SHIPPED_DEMO}」"
            f"（现有：{[s.name for s in book]}）")
        return got[0]

    def test_nothing_ships_enabled(self) -> None:
        """★★ 出厂配置**一条都不许开着**（2026-10-19）。

        这条断言的**方向**是本轮最重要的一次翻转。原来这里豁免了
        「通用战斗」，理由是「让用户能直接拿去演示」。

        实机把它打穿了：run 20261019_000606 是**干跑**，用户从没点过
        「完全接管」，而「通用战斗」照样产出了 203 条 INTENT_EXECUTED
        —— 引擎一个键都没发，却像真在打，YUI 照着念战斗播报。
        「演示」不能靠「默认替用户按下这个决定」来实现。

        要演示就两下：把 enabled 改 true，再去 Console 点「完全接管」。
        而这两件事在引擎侧是**同一个开关**（DRY_RUN 决策闸门），
        所以「改了配置但没点接管」这个最危险的组合压根不会成立。

        ⚠️ 2026-10-19 晚：用户明确要求**源码树**默认开着四神演示连招
          （``run_demo.bat`` 一键演示用）。于是这条出现一个**具名**
          豁免 :data:`NAMED_EXCEPTIONS` —— 只放行那一条，不放行「任何」。
          理由：这条守的是「别替用户按下接管决定」，而源码树演示由
          ``run_demo.bat`` 显式带 ``--live`` 启动，接管决定仍是
          用户自己按下的那一下，没有被默认值代劳。
        """
        book = load_combos(COMBOS_FILE)
        enabled = {s.name for s in book.enabled}
        extra = enabled - NAMED_EXCEPTIONS
        assert not extra, (
            f"出厂默认不该有开着的连招（除具名例外 {sorted(NAMED_EXCEPTIONS)}）："
            f"{sorted(extra)}")

    def test_every_combo_now_needs_a_note(self) -> None:
        """★ 全关之后，note 就成了这条连招**唯一**的说明书。

        原来只有「关着的」需要 note（``test_every_disabled_combo_explains_why``）。
        现在全关，那条断言已经覆盖全部 —— 这条留着是为了在它被放宽时
        有人会注意到。
        """
        for s in load_combos(COMBOS_FILE):
            assert s.note.strip(), f"{s.name} 缺 note"

    def test_the_demo_combo_explains_its_risk(self) -> None:
        """★ 那条留给用户去开的连招必须自带 note 说明代价。

        「关着」不等于「不用负责」—— 用户点开它的那一秒就该知道
        它在什么情况下会打错。
        """
        note = self._demo().note or ""
        assert "误报" in note, "默认开启的连招必须在 note 里写明误报风险"
        assert len(note) >= 20, "note 太短，说不清代价"

    def test_demo_combo_touches_nobody_in_particular(self) -> None:
        """★★ 随包那份连招必须**只**用 ``onfield``。

        这是「不写死角色名」的落地形式：不指定人，只指定「现在站着的人」。
        它对队伍是谁**零假设**，所以在任何队伍下都连贯。

        对比一下就明白为什么必须是这样：原来那条「钟夜」两步分别写死
        了钟离和夜兰，换成任意一队，只要这两个人不在编队里，
        连招就永远等在「名册里查不到」上 —— 表现为「开了但不打」。
        """
        seq = self._demo()
        assert all(st.onfield for st in seq.steps), (
            "通用战斗里出现了非 onfield 的步骤："
            f"{[st.describe() for st in seq.steps]}")
        assert not any(st.slot for st in seq.steps), (
            "通用战斗不该指定位置 —— 一旦指定 slot = N，它就只对"
            "「N 号位站着一个恰好 E 亮的角色」成立")

    def test_demo_combo_never_switches(self) -> None:
        """★ onfield 的步骤**永不切人**。

        如果它会切人，「通用战斗」就会退化成「按队伍顺序挨个切」——
        那正是写死队伍要避免的事，而且会让普攻打断切人窗口。
        """
        seq = self._demo()
        assert all(not st.is_switch for st in seq.steps)

    def test_no_legacy_combo_rules_are_left(self) -> None:
        """旧的 7 条 combo_* 规则必须已经删干净。

        留着它们会有一个很难发现的症状：两套机制同时抢同一件事，
        priority 62~68 的规则会在连招之前先命中 —— 而它们本来就是
        连招要取代的东西。
        """
        left = [r.name for r in _rules() if r.name.startswith("combo_")]
        assert left == [], f"旧连招规则还在：{left}"


# ==================================================================
# 2 · 解析器响亮失败（配错了一定要报错，不能静默跳过）
# ==================================================================
class TestLoudFailure:
    def test_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        """★ 没配 combos.toml = 正常配置 -> None（纯反射模式）。"""
        assert load_optional_combos(tmp_path / "nope.toml") is None

    def test_broken_file_is_an_error(self, tmp_path: Path) -> None:
        """★ 存在但坏了 = 故障，必须抛。吞成 None 的话玩家只会看到
        「引擎不认我的连招」，屏幕和日志上什么都没有。"""
        p = tmp_path / "combos.toml"
        p.write_text("[[combo]]\nname = 1\n", encoding="utf-8")
        with pytest.raises(ComboLoadError):
            load_optional_combos(p)

    def test_unknown_action_lists_the_whitelist(self, tmp_path: Path) -> None:
        with pytest.raises(ComboLoadError, match="不在白名单"):
            _book_from_toml(
                '[[combo]]\nname = "x"\ntrigger = "has_nearby_enemy()"\n'
                'sequence = [{ char = "钟离", action = "USE_ULT" }]\n',
                tmp_path)

    def test_switch_without_char_is_rejected(self, tmp_path: Path) -> None:
        """SWITCH_CHARACTER 没 char 会在执行层被拒 —— 那时连招已经启动了。"""
        with pytest.raises(ComboLoadError, match="必须写 char"):
            _book_from_toml(
                '[[combo]]\nname = "x"\ntrigger = "has_nearby_enemy()"\n'
                'sequence = [{ action = "SWITCH_CHARACTER" }]\n',
                tmp_path)

    def test_unknown_step_field_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ComboLoadError, match="未知字段"):
            _book_from_toml(
                '[[combo]]\nname = "x"\ntrigger = "has_nearby_enemy()"\n'
                'sequence = [{ char = "钟离", action = "USE_SKILL", "delay" = 1 }]\n',
                tmp_path)

    def test_unknown_top_level_table_is_rejected(self, tmp_path: Path) -> None:
        """``[[combos]]``（复数）会被整个静默忽略 —— 必须报出来。"""
        with pytest.raises(ComboLoadError, match="未知表"):
            _book_from_toml('[[combos]]\nname = "x"\n', tmp_path)

    def test_bad_trigger_dsl_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ComboLoadError, match="trigger 表达式不合法"):
            _book_from_toml(
                '[[combo]]\nname = "x"\ntrigger = "nosuchfunc()"\n'
                'sequence = [{ char = "钟离", action = "USE_SKILL" }]\n',
                tmp_path)

    def test_bad_step_when_dsl_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ComboLoadError, match="when 不合法"):
            _book_from_toml(
                '[[combo]]\nname = "x"\ntrigger = "has_nearby_enemy()"\n'
                'sequence = [{ char = "钟离", action = "USE_SKILL",'
                ' when = "hp.__class__" }]\n',
                tmp_path)

    def test_empty_sequence_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ComboLoadError, match="非空数组"):
            _book_from_toml(
                '[[combo]]\nname = "x"\ntrigger = "has_nearby_enemy()"\n'
                "sequence = []\n", tmp_path)

    def test_duplicate_name_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ComboLoadError, match="重复"):
            _book_from_toml(
                '[[combo]]\nname = "x"\ntrigger = "has_nearby_enemy()"\n'
                'sequence = [{ char = "钟离", action = "USE_SKILL" }]\n'
                '[[combo]]\nname = "x"\ntrigger = "has_nearby_enemy()"\n'
                'sequence = [{ char = "夜兰", action = "USE_BURST" }]\n',
                tmp_path)

    def test_out_of_range_repeat_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ComboLoadError, match="repeat"):
            _book_from_toml(
                '[[combo]]\nname = "x"\ntrigger = "has_nearby_enemy()"\n'
                'sequence = [{ char = "钟离", action = "NORMAL_ATTACK",'
                ' repeat = 9999 }]\n', tmp_path)


# ==================================================================
# 3 · ★ 4 步序列，10 个 tick，顺序断言
# ==================================================================
class TestSequenceOrder:
    def test_four_steps_in_ten_ticks_in_order(self) -> None:
        """★ 用户要的那条断言：4 步序列跑 10 个 tick，顺序必须对。

        真实时序（每 tick 200ms，动作在**决策之后**才执行）::

            tick 0  钟离在场 -> 开盾
            tick 1  夜兰不在场 -> **隐式先切夜兰**
            tick 2  切完 -> 夜兰开大
            tick 3  纳西妲不在场 -> 隐式切纳西妲
            tick 4  切完 -> 纳西妲挂草
            tick 5  绫华不在场 -> 隐式切绫华
            tick 6~  绫华连续普攻（repeat = 5）

        所以 10 个 tick 里发出的 10 个意图是
        ``USE_SKILL, SWITCH(夜兰), USE_BURST, SWITCH(纳西妲),
        USE_SKILL, SWITCH(绫华), NORMAL_ATTACK x4``
        —— 4 个步骤的动作全部出现且顺序正确。
        """
        h = Harness()
        cd = {"skill": True, "burst": True}

        for _ in range(10):
            h.tick(cd=cd)

        assert h.tags == [
            "USE_SKILL",                     # ① 钟离
            "SWITCH_CHARACTER:夜兰",          #   ② 隐式切人
            "USE_BURST",                     # ③ 夜兰
            "SWITCH_CHARACTER:纳西妲",
            "USE_SKILL",                     # ④ 纳西妲
            "SWITCH_CHARACTER:神里绫华",
            "NORMAL_ATTACK", "NORMAL_ATTACK", "NORMAL_ATTACK", "NORMAL_ATTACK",
        ]
        assert h.slot == 4, "应该站在神里绫华身上"

    def test_combo_preempts_the_generic_rules(self) -> None:
        """★ 连招必须压过 80 / 85 那两条**默认开着**的通用规则。

        它们在有敌人 + E 亮着时每 tick 都命中。不压过去的话，
        玩家打开连招后会看到「什么也没变」—— 因为 80 号规则先赢了。
        """
        h = Harness()
        for _ in range(6):
            intent = h.tick(cd={"skill": True, "burst": True})
        assert intent is not None
        assert intent.priority == COMBO_INTENT_PRIORITY, (
            f"这一 tick 出的是规则意图（p{intent.priority}），不是连招")
        # 而规则那边确实一直在命中（否则这条测试就证明不了任何事）
        assert any(r.name == "world_use_skill_if_ready" and r.enabled
                   for r in _rules())

    def test_one_intent_per_tick_never_a_batch(self) -> None:
        """★ 一 tick 只能有一个意图。

        「一次发一串」会绕过执行层的 min_gap 限频、安全闸和长按键释放 ——
        那是连招唯一可能越权的地方，所以钉死它。
        """
        h = Harness()
        for _ in range(15):
            intent = h.tick(cd={"skill": True, "burst": True})
            assert intent is None or isinstance(intent, IntentPacket)
        assert len(h.log) == 15

    def test_full_run_finishes_and_clears_state(self) -> None:
        """走完之后状态清空，重新等 trigger。

        ⚠️ 「走完」发生在**最后一个动作之后的下一 tick**：第 11 个 tick
          打出第 5 次普攻时序列就算数完了，但清空要等下一次调用才发生
          （那一 tick 返回 None，因为没有动作可发了）。所以是 12 tick。
          这不是漏拍 —— 是「一轮一个意图」的直接推论。
        """
        h = Harness()
        cd = {"skill": True, "burst": True}
        for _ in range(11):
            h.tick(cd=cd)
        assert h.runner.active is True, "第 11 tick 刚打完最后一次普攻"
        # ⚠️ 第 12 tick 连招已经清空并让路 —— 这一拍出的是**规则**意图
        #   （80 放 E），不是 None。连招让路不等于引擎闭嘴。
        last = h.tick(cd=cd)
        assert last is not None and last.priority != COMBO_INTENT_PRIORITY
        assert h.runner.active is False, h.runner.progress()
        assert h.runner.stats()["finished"] == 1
        # 冷却期内不该重新启动
        before = len(h.log)
        h.tick(cd=cd)
        assert h.runner.stats()["started"] == 1
        assert len(h.log) == before + 1

    def test_repeat_is_exactly_five(self) -> None:
        h = Harness()
        for _ in range(11):
            h.tick(cd={"skill": True, "burst": True})
        assert h.actions.count("NORMAL_ATTACK") == 5, h.actions

    def test_step_guard_skips_instead_of_stalling(self) -> None:
        """★ 大招没充能时**跳过**那一步并切走，不是卡在原地等。

        这是原来 ``combo_yelan_burst`` 的行为（``q_ready()`` 为假就不放，
        让切人的那条规则接手）。序列化之后必须保持。
        """
        h = Harness()
        cd = {"skill": True, "burst": True}
        h.tick(cd=cd)                      # 钟离开盾
        h.tick(cd=cd)                      # 切夜兰
        # 夜兰大招没充能 -> 这一步被跳过，直接切纳西妲
        h.tick(cd={"skill": True, "burst": False})
        assert h.tags[-1] == "SWITCH_CHARACTER:纳西妲", h.tags

    def test_switches_only_move_forward_within_a_run(self) -> None:
        """★ 一轮里切人顺序必须是 夜兰 -> 纳西妲 -> 绫华，绝不回头。

        旧梯子最容易出的错是双向守卫 -> 乒乓，而乒乓**不报错**，
        只表现为机器人在两个人之间无限横跳。
        """
        h = Harness()
        cd = {"skill": True, "burst": True}
        for _ in range(11):
            h.tick(cd=cd)
        switches = [i.params["char_id"] for i in h.log
                    if i is not None and i.action is ActionType.SWITCH_CHARACTER]
        assert switches == ["夜兰", "纳西妲", "神里绫华"], switches

    def test_a_finished_combo_restarts_from_the_top_after_cooldown(self) -> None:
        """★ 冷却到期后重新触发时，**从第 1 步重来**，不是接着上次跑。

        这就是「循环连招」的全部机制 —— 不需要代码里的 while：
        走完 + 冷却到期 + trigger 还成立 = 再来一遍。
        """
        h = Harness()
        cd = {"skill": True, "burst": True}
        for _ in range(12):
            h.tick(cd=cd)
        assert h.runner.stats()["finished"] == 1
        # 冷却 12s / 每 tick 200ms -> 再等 60 拍
        h.clock["ms"] += 13_000
        h.tick(cd=cd)
        assert h.runner.stats()["started"] == 2
        assert h.runner.progress()["step"] == 1, "必须从头开始，不是接着上次"


# ==================================================================
# 4 · ★ 中断：物理抢占 / 应急规则 / 感知不健康 / 切人超时
# ==================================================================
class TestInterrupts:
    def test_physical_takeover_clears_the_remaining_steps(self) -> None:
        """★ 用户要求的那条：中途物理抢占，后续步骤必须被清空。

        「后续步骤被清空」是**状态层面**的保证 —— runner 回到空闲、
        进度归零，而不是停在第 3 步等着下次唤醒接着打。
        （抢占之后规则仍然可以照常出手：连招让路给一切非应急规则。
          这里要验的是「连招不再继续」，不是「整个引擎停摆」。）
        """
        h = Harness()
        cd = {"skill": True, "burst": True}
        h.tick(cd=cd)                       # 开盾
        h.tick(cd=cd)                       # 切夜兰
        h.tick(cd=cd)                       # 夜兰开大
        assert h.runner.active is True
        assert h.runner.progress()["step"] == 3

        # ★ 用户按了键 —— core/loop.py::_on_physical_input 走的就是这条路
        assert h.runner.abort("用户按了 E") is True
        assert h.runner.active is False
        prog = h.runner.progress()
        assert prog["active"] is False and prog["phase"] == "-"

        # 冷却期内不该原地复活：一个连招意图都不许再发
        mark = len(h.log)
        h.clock["ms"] += 1_000
        for _ in range(5):
            h.tick(cd=cd)
        assert [i for i in h.log[mark:] if i is not None
                and i.priority == COMBO_INTENT_PRIORITY] == []
        assert h.runner.stats()["started"] == 1
        assert "用户按了 E" in h.runner.stats()["last_reason"]

    def test_abort_is_idempotent_and_returns_false_when_idle(self) -> None:
        h = Harness()
        assert h.runner.abort("没事") is False
        h.tick(cd={"skill": True, "burst": True})
        assert h.runner.abort("第一次") is True
        assert h.runner.abort("第二次") is False

    def test_policy_forwards_abort(self) -> None:
        """主循环拿到的 policy 对象上就有这个方法（loop 靠 getattr 调它）。"""
        h = Harness()
        h.tick(cd={"skill": True, "burst": True})
        assert h.tree.abort_combo("F12 急停") is True

    def test_policy_without_runner_degrades_quietly(self) -> None:
        """没接连招时 abort_combo 返回 False，不抛 —— loop 用 getattr 兜。"""
        tree = LocalPolicyTree(rules_dir=RULES_DIR,
                               rule_files=("genshin_combat.toml",))
        assert tree.combo_runner is None
        assert tree.abort_combo("F12") is False

    def test_emergency_rule_interrupts_and_wins_that_tick(self) -> None:
        """★ priority < break_below_priority 的规则**打断连招并接管那一轮**。

        这是「没血了立刻中断」的落点：吃药 / 闪避 / 感知不可信
        都不能排在一条开盾连招后面。

        应急条件用 ``enemy_count() >= 2``（走 :class:`PolicySignals`），
        这样能在连招跑到一半时**才**把它点亮 —— 如果写成恒真条件，
        连招连启动都启动不了，测的就不是「打断」而是「一开始就没有」。
        """
        sig = PolicySignals(enemy_count=1)
        h = Harness(signals=sig)
        h.tree._rules = (_emergency_rule(h.tree.rules[0]),) + h.tree.rules

        cd = {"skill": True, "burst": True}
        h.tick(cd=cd)
        h.tick(cd=cd)
        assert h.runner.active is True, "前两轮不该被打断"

        # ⚠️ signals 是 LocalPolicyTree 在**构造时**存下来的引用，
        #   所以换对象要连 tree 上的那份一起换（from_state 每帧重建
        #   ctx，但 ctx 拿的是 self._signals）。
        big = PolicySignals(enemy_count=3)
        h.signals = big
        h.tree._signals = big
        intent = h.tick(cd=cd)

        assert h.runner.active is False, "应急规则命中后连招必须被清掉"
        assert intent is not None and intent.action is ActionType.WAIT
        assert intent.priority == 5

    def test_perception_down_aborts_and_emits_nothing(self) -> None:
        """★ 感知不健康 -> 连招立刻中断，这一轮不发任何意图。"""
        h = Harness()
        h.tick(cd={"skill": True, "burst": True})
        assert h.runner.active is True
        intent = h.tick(cd={"skill": True, "burst": True},
                        perception_ok=False)
        assert intent is None
        assert h.runner.active is False
        assert "感知层不健康" in h.runner.stats()["last_reason"]

    def test_switch_that_never_completes_times_out(self) -> None:
        """★ 切人等不到记账器更新 -> 超时中断整条连招。

        ``execute=False`` 模拟「执行层跳过了这次切人」（名册里查不到人
        时真实就是这样）。继续往下打的话，夜兰的大招会打在钟离身上。
        """
        h = Harness()
        h.tick(cd={"skill": True, "burst": True}, execute=False)
        for _ in range(20):
            h.tick(cd={"skill": True, "burst": True}, execute=False)
        assert h.runner.active is False, h.runner.progress()
        assert "还没生效" in h.runner.stats()["last_reason"]

    def test_scene_change_aborts(self) -> None:
        """场景掉出连招允许的范围 -> 中断（不是继续盲按）。"""
        h = Harness()
        h.tick(cd={"skill": True, "burst": True})
        assert h.runner.active is True
        h.tick(cd={"skill": True, "burst": True}, scene=SCENE_UNKNOWN)
        assert h.runner.active is False
        assert "场景变成" in h.runner.stats()["last_reason"]


# ==================================================================
# 5 · ★ 兼容：没配 combos.toml 就退回纯反射模式
# ==================================================================
class TestPureReflectionFallback:
    def test_no_runner_means_no_combo_intents(self) -> None:
        """没接连招状态机时，决策层只发规则意图。"""
        tree = LocalPolicyTree(rules_dir=RULES_DIR,
                               rule_files=("genshin_combat.toml",),
                               now_fn=lambda: 0)
        assert tree.combo_runner is None
        assert tree.stats()["combos"] is None

    def test_disabled_book_produces_no_intent_and_falls_through(self) -> None:
        """★ 连招配置在但全是 enabled=false -> 行为等同纯反射。

        这一条是**兼容性的真正含义**：发行的那份 combos.toml 就是关着的，
        所以「发布状态下引擎的行为」必须与接入连招之前一模一样 ——
        命中的是 80/85 那两条通用规则，不是连招。
        """
        book = _enabled_shipped_book()
        off = ComboBook(
            sequences=tuple(dataclasses.replace(s, enabled=False)
                            for s in book),
            source_file=book.source_file)
        runner = ComboRunner(off, now_fn=lambda: 0)
        tree = LocalPolicyTree(rules_dir=RULES_DIR,
                               rule_files=("genshin_combat.toml",),
                               now_fn=lambda: 0, combo_runner=runner)

        st = StatePacket(
            scene=SceneInfo(id=SCENE_WORLD, confidence=0.9),
            player=PlayerInfo(party=()),
            state_version=1)
        tree.last_world_entities = (FakeEnemy(),)
        intent = tree.poll(st)
        assert intent is not None
        assert intent.priority == 85, "该走的是普攻兜底那条通用规则"

    def test_decide_is_never_affected_by_combos(self) -> None:
        """★ :meth:`LocalPolicyTree.decide` **完全不知道连招存在**。

        离线工具（rule_test / replay / policy_diagnose）靠它。
        连招是有状态的跨 tick 逻辑，塞进去会让「拿一帧跑一次 decide」
        这个契约失效。
        """
        h = Harness()
        st = h._state(SCENE_WORLD, {"skill": True}, True)
        ents = (FakeEnemy(),)
        h.tree.last_world_entities = ents
        plain = LocalPolicyTree(rules_dir=RULES_DIR,
                                rule_files=("genshin_combat.toml",),
                                now_fn=lambda: 0)
        plain.last_world_entities = ents
        assert ([e.to_dict() for e in plain.decide(st).entries]
                == [e.to_dict() for e in h.tree.decide(st).entries])
        assert h.runner.active is False, "decide 不该启动连招"
        assert h.runner.stats()["started"] == 0

    def test_missing_combos_file_is_reported_as_absent(self, tmp_path) -> None:
        assert load_optional_combos(tmp_path / "nope.toml") is None


# ==================================================================
# 6 · 安全审查没有被绕过
# ==================================================================
class TestSafetyUnchanged:
    def test_combo_intent_goes_through_the_same_validator(self) -> None:
        """★ 连招的每一步必须**照常**过 ``validator``。

        用真实 validator 跑一遍：合法意图过；把 priority 改成越界的
        那个必须被拒（``PRIORITY_OUT_OF_RANGE``）—— 说明连招产出的
        意图走的是同一条校验路径，没有旁路。
        """
        from autopilot.actuator.validator import IntentValidator

        h = Harness()
        intent = h.tick(cd={"skill": True, "burst": True})
        assert intent is not None
        st = h._state(SCENE_WORLD, {"skill": True}, True)

        v = IntentValidator()
        res = v.validate(intent, st)
        assert res.ok, getattr(res, "reason", res)

        broken = replace(intent, priority=999)
        res2 = v.validate(broken, st)
        assert not res2.ok and res2.reason == "PRIORITY_OUT_OF_RANGE"

    def test_switch_intent_carries_char_id(self) -> None:
        """隐式切人产出的意图必须带 char_id，否则执行层无从按哪个数字键。"""
        h = Harness()
        h.tick(cd={"skill": True, "burst": True})
        sw = h.tick(cd={"skill": True, "burst": True})
        assert sw is not None
        assert sw.action is ActionType.SWITCH_CHARACTER
        assert sw.params["char_id"] == "夜兰"


# ==================================================================
# 7 · 不知道就不做
# ==================================================================
class TestUnknownMeansDoNothing:
    def test_step_is_skipped_when_the_roster_does_not_know_the_char(self) -> None:
        """★ 名册里查不到 -> 跳过这一步，**不猜**。

        猜错的后果是「夜兰的大招打在钟离身上」—— 比不做更糟
        （不做只是这一轮没输出）。
        """
        h = Harness()
        h.missing.add("夜兰")
        for _ in range(4):
            h.tick(cd={"skill": True, "burst": True})
        assert h.tags == [
            "USE_SKILL",                    # 钟离（名册里有）
            "SWITCH_CHARACTER:纳西妲",         # 夜兰被跳过 -> 直接去纳西妲
            "USE_SKILL",
            "SWITCH_CHARACTER:神里绫华",
        ], h.tags

    def test_unknown_onfield_skips_rather_than_sending_the_wrong_char_skill(self) -> None:
        """★ 名册整个是空的时候，连招**一步都不发**。

        没有 active_member 时不能靠猜补一手 —— 猜错就是放错人的技能。
        ⚠️ 断言的是「没有一个意图来自连招」，**不是**「这一轮没有意图」：
          连招让路之后，通用规则（80 放 E）接手是完全正常的 ——
          连招不负责让引擎闭嘴。
        """
        h = Harness()
        h.missing.update({"钟离", "夜兰", "纳西妲", "神里绫华"})
        for _ in range(6):
            h.tick(cd={"skill": True, "burst": True})
        assert [i for i in h.log if i is not None
                and i.priority == COMBO_INTENT_PRIORITY] == []
        assert h.runner.stats()["finished"] == 1, "应该整条跳过并走完"
        assert h.runner.active is False

    def test_no_enemy_no_combo(self) -> None:
        """没有敌人时连招一条都不能起（trigger 就该挡住）。"""
        h = Harness()
        for _ in range(5):
            h.tick(enemy=False, cd={"skill": True, "burst": True})
        assert h.runner.active is False
        assert h.runner.stats()["started"] == 0


class TestStaticCharSlots:
    """★ ``keymap.toml`` 的 ``[char_slots]`` 对连招**必须有效**。

    文档让玩家填它的唯一理由是绕开启动后 ~20 秒的名册空窗。如果连招
    只认 OCR 名册，那份配置对连招一点用没有 —— 而文档还让它填。
    那种「文档在骗人」的 bug 只能靠测试挡住。
    """

    _SLOTS = {"钟离": 1, "夜兰": 2, "纳西妲": 3, "神里绫华": 4}

    def test_static_slots_rescue_a_completely_empty_roster(self) -> None:
        """★ 名册一格都没填 + 手填了 [char_slots] -> 连招照跑。"""
        h = Harness(static_slots=self._SLOTS)
        h.missing.update({"钟离", "夜兰", "纳西妲", "神里绫华"})
        cd = {"skill": True, "burst": True}
        for _ in range(4):
            h.tick(cd=cd)
        assert h.tags == ["USE_SKILL", "SWITCH_CHARACTER:夜兰",
                          "USE_BURST", "SWITCH_CHARACTER:纳西妲"], h.tags

    def test_static_slots_win_over_the_ocr_roster(self) -> None:
        """★ 两者打架时**静态优先**，与执行层 ``_slot_of`` 同口径。

        反过来的话，玩家手填纠正 OCR 错名字时会变成「规则切对了人、
        连招切错了人」—— 同一个错名字在两边表现不一致，最难查。
        """
        h = Harness(static_slots={"夜兰": 4})     # OCR 名册里夜兰 = 2
        assert h.runner._slot_of("夜兰") == 4
        assert h.runner._slot_of("纳西妲") == 3, "没配静态就该走 OCR 名册"

    def test_explicit_switch_is_skipped_when_the_slot_is_unknown(
            self, tmp_path: Path) -> None:
        """★ 显式切人查不到槽位 -> 跳过这一步，**不发**那个一定会被
        执行层跳过的意图（否则白占一个 tick，日志里还留一条假记录）。
        """
        book = _book_from_toml(
            '[[combo]]\nname = "换人"\ntrigger = "has_nearby_enemy()"\n'
            'sequence = [\n'
            '  { action = "SWITCH_CHARACTER", char = "魈" },\n'
            '  { char = "钟离", action = "USE_SKILL" },\n'
            ']\n', tmp_path)
        h = Harness(combos=book)      # 名册里没有「魈」
        cd = {"skill": True}
        h.tick(cd=cd)
        h.tick(cd=cd)
        assert h.tags == ["USE_SKILL", "USE_SKILL"], h.tags
        assert "SWITCH" not in h.actions, "查不到槽位还发切人意图"
        # ★ 跳一步 ≠ 整条作废：剩下那一步照打，最后是**走完**不是中断
        assert h.runner.stats()["finished"] == 1
        assert h.runner.stats()["aborted"] == 0


# ==================================================================
# 6 · 位置语义（slot / onfield）—— 不写死角色名的落地
# ==================================================================
class TestSlotAndOnfieldParsing:
    """``slot`` / ``onfield`` 的解析与互斥校验。

    ★ 这里验的是「配错一定要响」
      静默取一个的表现是「我明明改了配置但行为没变」，而玩家不会
      去怀疑是配置与位置互斥 —— 那是一条要重读三遍的报错。
    """

    OK = ('[[combo]]\nname = "t"\ntrigger = "has_nearby_enemy()"\n'
          'sequence = [ %s ]\n')

    @pytest.mark.parametrize("text,slot,onfield", [
        ('{ slot = 1, action = "USE_SKILL" }', 1, False),
        ('{ slot = 4, action = "USE_SKILL" }', 4, False),
        ('{ onfield = true, action = "USE_SKILL" }', 0, True),
        ('{ action = "NORMAL_ATTACK" }', 0, False),
        ('{ char = "某某", action = "USE_SKILL" }', 0, False),
    ])
    def test_valid_spellings(self, text, slot, onfield, tmp_path) -> None:
        st = _book_from_toml(self.OK % text, tmp_path).sequences[0].steps[0]
        assert (st.slot, st.onfield) == (slot, onfield), st.describe()
        assert st.char == "" or not st.is_positional

    @pytest.mark.parametrize("text,frag", [
        # ★ 互斥：静默取一个 = 用户改了半条却不知道
        ('{ slot = 1, onfield = true, action = "USE_SKILL" }', "互斥"),
        ('{ char = "某某", slot = 1, action = "USE_SKILL" }', "二选一"),
        ('{ char = "某某", onfield = true, action = "USE_SKILL" }', "二选一"),
        # 值域：队伍只有 4 格
        ('{ slot = 0, action = "USE_SKILL" }', "越界"),
        ('{ slot = 5, action = "USE_SKILL" }', "越界"),
        ('{ slot = -1, action = "USE_SKILL" }', "越界"),
        # 类型：TOML 的 true 是 bool，bool 是 int 的子类
        ('{ slot = true, action = "USE_SKILL" }', "1~4 的整数"),
        ('{ slot = "2", action = "USE_SKILL" }', "1~4 的整数"),
        ('{ slot = 1.5, action = "USE_SKILL" }', "1~4 的整数"),
    ])
    def test_bad_spellings_raise(self, text, frag, tmp_path) -> None:
        with pytest.raises(ComboLoadError, match=frag):
            _book_from_toml(self.OK % text, tmp_path)

    def test_omitting_slot_is_not_slot_zero(self, tmp_path) -> None:
        """★★ 回归：``raw.get("slot", 0)`` 把「没写」变成「0 号位」。

        0 是 int，于是掉进 ``1 <= slot <= 4`` 的越界校验里炸掉 ——
        表现是**只写 onfield 的一步永远加载不了**，而发布的那条
        「通用战斗」两步全是 onfield，也就是整份配置直接读不出来。
        """
        st = _book_from_toml(
            self.OK % '{ onfield = true, action = "USE_SKILL" }',
            tmp_path).sequences[0].steps[0]
        assert st.slot == 0 and st.onfield is True
        assert st.is_positional

    def test_slot_zero_while_a_step_only_wants_the_active_one(
            self, tmp_path) -> None:
        """★ 「没写 slot」与「写了 slot = 0」必须**表现不同**。

        前者是合法配置，后者是笔误。所以不能靠 ``slot == 0`` 判
        「没写」—— 那正是这个 bug 的成因。
        """
        with pytest.raises(ComboLoadError, match="越界"):
            _book_from_toml(self.OK % '{ slot = 0, action = "WAIT" }',
                            tmp_path)

    def test_describe_shows_the_position_not_the_name(self,
                                                     tmp_path) -> None:
        """诊断输出必须能直接看懂是谁 —— 玩家不看 slot 字段。"""
        book = _book_from_toml(
            '[[combo]]\nname = "t"\ntrigger = "has_nearby_enemy()"\n'
            'sequence = [\n'
            '  { slot = 2, action = "USE_BURST" },\n'
            '  { onfield = true, action = "NORMAL_ATTACK" },\n'
            '  { char = "某某", action = "WAIT" },\n'
            ']\n', tmp_path)
        got = [s.describe() for s in book.sequences[0].steps]
        assert "2 号位" in got[0], got
        assert "在场" in got[1], got
        assert "某某" in got[2], got

    def test_to_dict_round_trips(self, tmp_path) -> None:
        book = _book_from_toml(
            self.OK % '{ slot = 3, action = "USE_SKILL", gap_ms = 250 }',
            tmp_path)
        d = book.sequences[0].steps[0].to_dict()
        assert d["slot"] == 3 and d["onfield"] is False
        assert d["char"] == ""


class TestSlotStyle:
    """``slot = N`` 怎么落到一次真实的按键上。"""

    BOOK = ('[[combo]]\nname = "接力"\ntrigger = "has_nearby_enemy()"\n'
            'sequence = [\n'
            '  { slot = 2, action = "USE_BURST", when = "q_ready()" },\n'
            ']\n')

    def test_slot_resolves_through_the_roster(self, tmp_path) -> None:
        """★★ ``slot = 2`` -> 运行时反查出名字 -> SWITCH_CHARACTER。

        执行层的 ``SWITCH_CHARACTER`` 收的是**名字**（``params.char_id``），
        它自己再翻成数字键。所以「按第 2 号位」这条链路上，
        必须有一个时刻是拿着名字的 —— 那个名字只能来自 OCR 名册。
        """
        book = _book_from_toml(self.BOOK, tmp_path)
        h = Harness(combos=book)
        h.slot = 1                                   # 现在站着 1 号位
        i = h.tick(cd={"burst": True})
        assert i is not None and i.action is ActionType.SWITCH_CHARACTER
        assert i.params["char_id"] == NAME_OF[2], (
            "slot=2 没有反查成名字，而是把槽位当名字发出去了")
        assert i.params["char_id"] != 2

    def test_slot_step_skips_switch_when_already_there(self,
                                                       tmp_path) -> None:
        """★ 已经在 2 号位了就直接放大招，不浪费一个 tick 切自己。"""
        book = _book_from_toml(self.BOOK, tmp_path)
        h = Harness(combos=book)
        h.set_onfield(2)
        i = h.tick(cd={"burst": True})
        assert i is not None and i.action is ActionType.USE_BURST

    def test_unresolved_slot_is_skipped_loudly_not_guessed(
            self, tmp_path) -> None:
        """★★ 名册里没这个人 -> 跳过 + 发 ``COMBO_STEP_UNKNOWN``。

        猜一个的后果是「2 号位站着谁就打给谁」—— 实际是打给站在
        那个位置的人，但**发出去的意图带着错误的名字**，后面每次
        记账都会跟真实在场错开。不做只是这一轮没输出，好得多。
        """
        seen: list = []
        book = _book_from_toml(self.BOOK, tmp_path)
        h = Harness(combos=book)
        h.runner._emit_fn = lambda ev: seen.append(ev)
        h.set_onfield(1)
        # 让 2 号位查不到：整个名册置空（模拟 OCR 还没填满）
        h.runner._index = _EmptyRoster()
        for _ in range(3):
            h.tick(cd={"burst": True})
        codes = [e.code for e in seen]
        assert "COMBO_STEP_UNKNOWN" in codes, codes
        # ★ 响亮跳过的**对价**：不能发出任何带错误名字的切人意图。
        assert not any(e.code.endswith("_SWITCH") for e in seen), codes
        # ⚠️ 断的是「连招不许出手」，**不是**「引擎不许出手」——
        #   连招让路之后 85 号普攻兜底照常接手。
        assert [i for i in h.log if i is not None
                and i.priority == COMBO_INTENT_PRIORITY] == [], h.actions

    def test_static_char_slots_win_over_the_roster(self, tmp_path) -> None:
        """★ 静态 ``[char_slots]`` 反查优先，且**与 _slot_of 同序**。

        反过来的话，玩家手填纠正 OCR 错名字时会变成「规则切对了人、
        连招切错了人」—— 同一个错名字在两边表现不一致，最难查。
        """
        book = _book_from_toml(self.BOOK, tmp_path)
        h = Harness(combos=book, static_slots={"魈": 2})
        h.slot = 1
        i = h.tick(cd={"burst": True})
        assert i is not None
        assert i.params["char_id"] == "魈", (
            "静态配置里的「魈=2 号位」没有被 slot=2 反查到")


class _EmptyRoster:
    """名册一个名字都没有（模拟 OCR 还没填满）。"""

    @staticmethod
    def name_of(slot: int):
        return None

    @staticmethod
    def slot_of(char_id: str):
        return None


class TestOnfieldStyle:
    """``onfield = true``：谁在场上就动谁，**永不切人**。"""

    BOOK = ('[[combo]]\nname = "通用"\ntrigger = "has_nearby_enemy()"\n'
            'sequence = [\n'
            '  { onfield = true, action = "USE_SKILL", when = "e_ready()" },\n'
            '  { onfield = true, action = "NORMAL_ATTACK", repeat = 2 },\n'
            ']\n')

    def test_acts_on_whichever_slot_is_active(self, tmp_path) -> None:
        """★★ 连贯性的来源：换队伍不用改配置。

        同一份连招，站在 1 号位打一遍、站在 4 号位再打一遍，
        动作序列完全相同 —— 因为它从来没问过「队伍是谁」。
        """
        seqs = []
        for who in (1, 4):
            h = Harness(combos=_book_from_toml(self.BOOK, tmp_path))
            h.slot = who
            h.tick(cd={"skill": True})
            h.tick()
            h.tick()
            seqs.append([i.action.value for i in h.log])
        assert seqs[0] == seqs[1] == ["USE_SKILL", "NORMAL_ATTACK",
                                      "NORMAL_ATTACK"]

    def test_never_switches_character(self, tmp_path) -> None:
        """★ onfield 的步骤**一个切人意图都不许发**。

        它会切人就等于退化成「按队伍顺序挨个切」，那正是写死队伍
        要避免的，而且切人会打断普攻节奏。
        """
        h = Harness(combos=_book_from_toml(self.BOOK, tmp_path))
        h.slot = 3
        for _ in range(8):
            h.tick(cd={"skill": True})
        assert "SWITCH_CHARACTER" not in h.actions, h.actions

    def test_needs_no_roster_at_all(self, tmp_path) -> None:
        """★ 名册全空也要能打 —— 这是它与 slot 式最本质的差别。

        蒙德里鲍姆式的盲打：E 亮按 E，没亮普攻。对 2 亿种队伍
        全部成立，代价只是「不知道自己在按谁」。
        """
        book = _book_from_toml(self.BOOK, tmp_path)
        h = Harness(combos=book)
        h.runner._index = _EmptyRoster()          # 一个名字都没有
        h.slot = 1
        h.tick(cd={"skill": True})
        assert h.actions[0] == "USE_SKILL"

    def test_when_false_moves_on_to_normal_attack(self, tmp_path) -> None:
        """★★ E 不亮就普攻 —— 「通用战斗」不靠 E 活着。

        没有这一步，E 永远不亮的角色（比如大招循环角色）会让连招
        **空转到超时**，表现是「开着但什么都不做」。

        ★ 注意是**同一个 tick** 里就普攻，不是「这一拍空着，下一拍普攻」
          ``when`` 为假的语义是「跳过这一步接着走下一步」，跟
          ``has_nearby_enemy()`` 为假时整个连招不启动是两回事。
          写成「先空一拍」的话，一轮连招会白白少打一下。
        """
        h = Harness(combos=_book_from_toml(self.BOOK, tmp_path))
        h.set_onfield(1)
        h.tick(cd={})                              # E 没亮
        h.tick()
        assert h.actions[:2] == ["NORMAL_ATTACK", "NORMAL_ATTACK"], h.actions

    def test_shipped_general_combat_needs_no_roster_at_all(self) -> None:
        """★★ 端到端：随包那条「通用战斗」在**名册全空**时照样打。

        这是「不写死角色名」最硬的证据形式 —— 它不是「换了队伍也
        行」（那还可能残留着别的默认名），而是「连队伍是**谁**都
        不知道，照样按键」。
        """
        h = Harness(combos=_enabled_shipped_book())
        h.runner._index = _EmptyRoster()
        h.slot = 1
        h.tick(cd={"skill": True})
        h.tick()
        h.tick()
        assert h.actions[0] == "USE_SKILL", h.actions
        assert "SWITCH_CHARACTER" not in h.actions

    def test_shipped_general_combat_has_no_names_to_get_wrong(
            self) -> None:
        """★ 跑起来时意图里也不该冒出任何 ``char_id``。

        :meth:`TestShippedConfig.test_the_shipped_config_names_no_character`
        守的是「配置里没写死名字」；这里守的是另一半 ——
        「引擎也没在别处偷偷替玩家决定一个人」。
        """
        h = Harness(combos=_enabled_shipped_book())
        h.tick(cd={"skill": True})
        assert all(
            not (i.params.get("char_id") if i else None)
            for i in h.log), [i.params for i in h.log if i]


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
