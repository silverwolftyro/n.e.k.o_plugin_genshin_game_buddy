"""规则表结构 + 活跃/待接入分离的单测（2026-10-11）。

这一版比上一版多了一条最关键的守卫：

    **活跃目录里不许存在 when 依赖「今天恒 UNKNOWN 的信号」的规则。**

上一版只查「动作在白名单里」「DSL 函数注册了」，那些规则**结构上**都合法，
但结构合法不等于跑得起来 —— `bag_hold_for_upgrade` 的 when 就是字面量
`false`，它通过了所有结构检查，却永远不会触发。真正该守的是**可达性**。
"""

from __future__ import annotations

import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.contracts.intent_packet import ActionType  # noqa: E402
from autopilot.policy.context import FUNCTION_NAMES  # noqa: E402

RULES = PLUGIN_ROOT / "autopilot" / "rules"
FUTURE = RULES / "_future"

#: 今天恒 UNKNOWN 的信号 —— 依赖它们的规则进了 _future/ 就等于关掉。
#: 另外 scene() 是**二值**的（WORLD / UNKNOWN），比别的不在列表里就永远不匹配。
#:
#: ⚠️ 三个**不要**加进来的：
#:   ``enemy_count`` —— 2026-10-11 已接血条兜底，是活信号。
#:   ``since``      —— builder.py:592 **一直**在设 ``scene.entered_ms``，
#:   所以 since() 在生产里是真值。之前我用合成的 StatePacket 探测（没带
#:   entered_ms 字段），得出了「since 恒 UNKNOWN」的**错误结论**，还据此把
#:   combat_settle_on_enter 挪进了 _future/。这条已经改正。
#:   ——教训：探测用的夹具必须**逐字段复刻真实状态**，否则会把「夹具缺字段」
#:   误判成「生产没接」。
#:   ``e_ready``    —— 2026-10-18 变了。它不再是「恒 UNKNOWN」，而是
#:   **配置门控的条件活信号**：[autopilot.skill_cd] enabled=true 时
#:   perception.detectors.key_e 会往 PartyMember.cd_ready["skill"] 填三态。
#:   ⚠️ 但它**默认关着**，所以「今天跑会不会触发」仍取决于配置 ——
#:   这正是本文件看不见的东西。硬判它死会把刚接上的规则逼进 _future/
#:   （等于把功能悄悄关掉），硬判它活又会漏掉「接线被回滚」。
#:   → 改判成 :func:`test_conditional_signal_has_a_live_wiring` 那条守卫：
#:     只要活跃目录里有规则用 e_ready()，就必须存在能填它的配置 + 接线。
#:
#:   ``onfield``    —— 2026-10-18 **第二次**变了，现在**不是死的**。
#:     视觉路线仍然不通（296 帧素材标不出「谁在场」），但换成了
#:     **记账**：``actuator/active_slot.ActiveSlotTracker`` 记住
#:     「引擎按了哪个数字键 / 用户按了哪个数字键」，builder 据此把对应
#:     ``PartyMember.active`` 置 True。于是 ``active_member`` 有了，
#:     ``onfield()`` 能答出 True/False —— 前提是有人切过人。
#:     剩下的门是**名字**：[autopilot.character_id] 关着时 ``name`` 恒为
#:     ``""``，``onfield('钟离')`` 恒 False，所以它进了
#:     :data:`CONDITIONAL_FUNCS`。
#:     ⚠️ 记账的**唯一失效场景**是「角色阵亡自动切人」—— 引擎无从发现。
#:
#:   ``q_ready``    —— 2026-10-18 变了，不再死。
#:     [autopilot.burst_cd] enabled=true 时 ``detectors/key_q.py`` 往
#:     ``cd_ready["burst"]`` 填就绪态（296 帧实测：15 命中 0 误报 0 漏报）。
DEAD_FUNCS = {"hp", "energy", "resin", "pos_quality", "pos_usable",
              "has_enemy",
              "target_locked", "slot"}
LIVE_SCENES = {"WORLD", "UNKNOWN"}

#: **条件活**信号：默认可能关着，但开着就能出真值。
#:
#: 这些不能进 :data:`DEAD_FUNCS`（它们能被接线救活），也不能当活信号看待
#: （接线被回滚时没有任何静态检查会红）。对应的守卫见
#: ``test_conditional_signal_has_a_live_wiring``。
#:
#: 值是 ``(配置段, PolicyContext 里必须还能读到的源码标记)``。
#: 两个函数读的是不同东西 —— ``e_ready``/``q_ready`` 走 ``_active_cd()``
#: 读 ``cd_ready``，``onfield`` 直接读 ``active_member``。用同一个断言
#: 去卡它们会把其中一个永久判红。
CONDITIONAL_FUNCS: dict[str, tuple[str, tuple[str, ...]]] = {
    "e_ready": ("skill_cd", ("_active_cd(",)),
    "q_ready": ("burst_cd", ("_active_cd(",)),
    "onfield": ("character_id", ("active_member",)),
}

#: 本轮新增的规则文件（仍在活跃目录里）
NEW_FILES = ("genshin_combat_advanced.toml", "genshin_interaction.toml",
             "genshin_recovery.toml", "genshin_bag_advanced.toml",
             "genshin_team.toml")


def _load(path: Path) -> list[dict]:
    return tomllib.loads(path.read_text(encoding="utf-8")).get("rules", [])


def _active() -> list[tuple[str, dict]]:
    """只扫活跃目录 —— 加载器是 ``glob("*.toml")``，非递归。"""
    return [(p.name, r) for p in sorted(RULES.glob("*.toml"))
            for r in _load(p)]


def _pending() -> list[tuple[str, dict]]:
    return [(p.name, r) for p in sorted(FUTURE.glob("*.toml"))
            for r in _load(p)]


def _dead_reasons(when: str) -> list[str]:
    out = []
    for f in re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", when):
        if f in DEAD_FUNCS:
            out.append(f"{f}() 恒 UNKNOWN")
    for sc in re.findall(r"scene\(\)\s*==\s*'([A-Z_]+)'", when):
        if sc not in LIVE_SCENES:
            out.append(f"scene() 永不产出 {sc}")
    if when.strip() == "false":
        out.append("when 是字面量 false（占位）")
    return out


# ==================================================================
# 分离结构
# ==================================================================

@pytest.mark.unit
def test_loader__directory_scan_is_non_recursive() -> None:
    """★ 直接钉住**机制**，而不是靠「当前没被加载」这个结果。

    上面那些守卫（``test_future_dir__*``）比对的是「加载到的规则名」与
    ``_future/`` 里的规则名 —— 它们能发现「死规则复活」，但前提是
    ``DEAD_FUNCS`` / ``LIVE_SCENES`` 这两张手工清单没漏。

    漏了新信号的场景下，清单会安静地放过一条新死规则，于是
    ``test_active_dir__has_no_unreachable_rules`` 不会红 ——
    但根因（加载器递归）其实就在那儿。

    所以这里直接看源码：``load_rules_from_dir`` 里出现 ``rglob``
    就是一个足以让整个分离方案静默失效的单字符改动。
    """
    import inspect

    from autopilot.policy import rule_loader

    src = inspect.getsource(rule_loader.load_rules_from_dir)
    assert ".rglob(" not in src, (
        "加载器改用 rglob 了 —— rules/_future/ 里的死规则会**原地复活**"
        "且不报错。请把它改回 glob(\"*.toml\")，或者给 _future/ 加显式排除。")
    assert '.glob("*.toml")' in src, (
        "加载器的扫描方式变了 —— 请确认它仍然是非递归的 *.toml")


@pytest.mark.unit
def test_loader__a_nested_dir_is_behaviourally_invisible(tmp_path) -> None:
    """★ 用真实的目录树证明「子目录里的 toml 不会被加载」。

    比读源码更强：不管实现怎么写，只要行为变了这条就红。
    """
    from autopilot.policy.rule_loader import load_rules_from_dir

    def _rule(name: str) -> str:
        return (
            '[[rules]]\n'
            f'name = "{name}"\n'
            'action = "WAIT"\n'
            'when = "true"\n'
            'priority = 50\n'
            'scenes = ["WORLD"]\n'
            'enabled = false\n'
            'note = "probe"\n'
        )

    (tmp_path / "live.toml").write_text(_rule("probe_active"), encoding="utf-8")
    (tmp_path / "_future").mkdir()
    (tmp_path / "_future" / "dead.toml").write_text(
        _rule("probe_pending"), encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deeper.toml").write_text(
        _rule("probe_nested"), encoding="utf-8")

    loaded = {getattr(r, "name", None) for r in load_rules_from_dir(tmp_path)}
    assert "probe_active" in loaded, "顶层文件必须被加载"
    assert "probe_pending" not in loaded, "子目录里的规则被加载了"
    assert "probe_nested" not in loaded, "更深一层的规则被加载了"


@pytest.mark.unit
def test_future_dir__exists_and_is_not_loaded() -> None:
    """``_future/`` 必须存在，且**加载器看不到它**（glob 非递归）。

    这是整个分离方案的地基：如果加载器哪天改成 rglob，20 条死规则会
    原地复活，而且没有任何报错。
    """
    assert FUTURE.is_dir(), "rules/_future/ 不见了"
    from autopilot.policy.rule_loader import load_rules_from_dir

    src = load_rules_from_dir(RULES)
    loaded = {getattr(r, "name", None) for r in getattr(src, "rules", ())}
    pending = {r["name"] for _f, r in _pending()}
    assert pending, "_future 里应该有东西"
    assert not (loaded & pending), (
        f"_future 里的规则被加载了：{sorted(loaded & pending)}")


@pytest.mark.unit
def test_active_dir__has_no_unreachable_rules() -> None:
    """★★ 核心守卫：活跃目录里不许有「结构合法但永远不触发」的规则。

    ⚠ ``enabled = false`` 的规则**不参与**这条检查（2026-10-18）
    --------------------------------------------------------------
    一条 ``enabled = false`` 的规则**明确就是关着的** —— 它不会误触发、
    不会抢风，把它按「死规则」处理只会逼所有人把「写好但先关着」的
    规则挪进 ``_future/``（那就等于把规则删了，只是换了个目录）。

    真正要防的是另一种情况：**先关着、悄悄就忘了打开**。所以下面
    :func:`test_disabled_rules_say_why_they_are_off` 补了一条强制要求 ——
    关着的规则**必须**写清为什么关着。这让「临时关一下」和
    「永远没人打开」在代码里可区分。

    ⚠ 这条口子**不能**被拿来藏接线回滚：``e_ready()`` 这类
    「配置门控的活信号」走的是 :func:`test_conditional_signal_has_a_live_wiring`，
    那里不看 ``enabled``，只看接线还在不在。
    """
    offenders = []
    for f, r in _active():
        if not r.get("enabled", True):
            continue
        for why in _dead_reasons(r["when"]):
            offenders.append(f"{f}:{r['name']}  ({why})")
    assert not offenders, (
        "这些**启用中**的规则依赖今天恒 UNKNOWN 的信号，"
        "应该挪到 rules/_future/ 或设 enabled = false：\n  "
        + "\n  ".join(offenders))


@pytest.mark.unit
def test_disabled_rules_say_why_they_are_off() -> None:
    """★★ 关着的规则必须写清「为什么关着」。

    :func:`test_active_dir__has_no_unreachable_rules` 放过了
    ``enabled = false`` 的规则 —— 这是必要的（关着的规则本来就不该被
    判死），但它开了一个口子：**把规则设成关着，就可以永久绕过死信号检查**。

    所以要求每条关着的规则都带 ``note``，并且 note 里至少有一个中文字符
    —— 后者是刻意的：项目里所有 note 都是中文说明，写成
    ``note = "disabled"`` 的那条几乎肯定是随手糊的。

    ⚠ 这**不是**形式主义。它是让下面这件事在 code review 里看得见：
    「这条规则为什么还关着、什么时候会打开」。
    """
    offenders = []
    for f, r in _active():
        if r.get("enabled", True):
            continue
        note = (r.get("note") or "").strip()
        if not note:
            offenders.append(f"{f}:{r['name']}  (enabled=false 但 note 为空)")
        elif not any("\u4e00" <= ch <= "\u9fff" for ch in note):
            offenders.append(
                f"{f}:{r['name']}  (note 没有中文说明：{note!r})")
    assert not offenders, (
        "关着的规则必须写清为什么关着（note），否则等于偷偷藏一条永不触发的规则：\n  "
        + "\n  ".join(offenders))


@pytest.mark.unit
def test_conditional_signal_has_a_live_wiring() -> None:
    """★★ :data:`CONDITIONAL_FUNCS` 的配套守卫（取代把它们塞进 DEAD_FUNCS）。

    :data:`DEAD_FUNCS` 判的是「信号今天恒 UNKNOWN」。但 ``e_ready()`` 现在
    是**配置门控**的：``[autopilot.skill_cd] enabled=true`` 就能出真值。
    两种一刀切的判法都会出事：

    * 判它死 -> 刚接上的规则被逼进 ``_future/``，等于把功能悄悄关掉，
      而且测试**是绿的**（守卫和规则一起搬到「正确」的位置）。
    * 判它活 -> 有人把接线回滚了（builder 不再填 cd_ready、配置段被删），
      这条规则又变成死规则，而**没有任何静态检查会红**。

    所以这里改成一条**双向**检查：活跃目录里只要有规则用了条件信号，
    就必须同时存在 ① 对应的配置段 ② 感知侧的填充路径 ③ 真值能打通到规则。
    三者缺一，这条规则就是死的 —— 明确报错，而不是安静地不触发。

    ⚠️ 这里刻意**不**去查配置默认值。默认关着不是缺陷（关着 = 与接入之前
      完全一致，是刻意的安全姿态）；缺陷是「关着的时候没人知道它关着」。
      那是 ``tools/_stack.py`` 的 warnings 的活。
    """
    used = [(f, r["name"]) for f, r in _active()
            if any(fn + "(" in r["when"] for fn in CONDITIONAL_FUNCS)]
    if not used:
        return                      # 没人用，条件自然满足

    from autopilot.config.loader import load_config
    cfg = load_config()

    for fname, rname in used:
        for fn, (attr, markers) in CONDITIONAL_FUNCS.items():
            if f"{fn}(" not in next(r["when"] for f, r in _active()
                                    if r["name"] == rname):
                continue

            # ① 配置段还在
            section = getattr(cfg, attr, None)
            assert section is not None, (
                f"{fname}:{rname} 用了 {fn}()，但 autopilot.{attr} 配置段不见了。"
                f"把它删掉会让这条规则永远不触发，且不会报错。")
            assert hasattr(section, "enabled"), (
                f"autopilot.{attr} 的配置类少 enabled 字段 —— "
                f"_stack.py 靠它决定接不接线。")

            # ② 决策侧还认这个信号
            import inspect

            from autopilot.policy.context import PolicyContext

            src = inspect.getsource(getattr(PolicyContext, fn))
            assert any(m in src for m in markers), (
                f"{fn}() 不再读 {markers}，autopilot.{attr} 填的东西没人读了。")

    # ③ 端到端：真值确实能打通到**某个消费者**（不是只验了各个零件都在）
    #    详细三态用例在 tests/unit/test_skill_cd_wiring.py 与
    #    tests/unit/test_burst_cd_wiring.py，这里只钉住
    #    「确实有一个规则 / 连招会因为它而活过来」。
    #
    # ★ 2026-10-18：这里原来硬编码了两条探针规则
    #   （``world_use_skill_if_ready`` / ``combo_yelan_burst``）。
    #   后者随 7 条 ``combo_*`` 规则一起删掉了 —— q_ready 的消费者
    #   搬到了 ``combos/combos.toml``。改成「在规则**和连招**里都找一遍」，
    #   钉的还是同一件事（有活着的消费者），但不再被某一次搬家打脸。
    consumers = {fn: [r["name"] for _f, r in _active()
                      if f"{fn}(" in (r.get("when") or "")]
                  for fn in CONDITIONAL_FUNCS}
    for seq in _combos():
        name = seq.name
        if any(f"{fn}(" in seq.trigger for fn in CONDITIONAL_FUNCS):
            for fn in CONDITIONAL_FUNCS:
                if f"{fn}(" in seq.trigger:
                    consumers[fn].append(f"combo:{name}(trigger)")
        for i, st in enumerate(seq.steps, 1):
            for fn in CONDITIONAL_FUNCS:
                if f"{fn}(" in st.when:
                    consumers[fn].append(f"combo:{name}#{i}")

    for fn, why in (("e_ready", "配置门控 skill_cd"),
                    ("q_ready", "配置门控 burst_cd")):
        assert consumers[fn], (
            f"{fn}()（{why}）已经没有任何规则或连招在用了 —— "
            f"接线要么被挪走了（功能就废了），要么全被删/改名字了。"
            f"如果不打算再用，就把 autopilot.{CONDITIONAL_FUNCS[fn][0]} "
            f"一起清掉；否则补一个消费者。")

    # ★ ``onfield()`` **不要求**仓库内有消费者 —— 这是 2026-10-18 迁移连招
    #   之后的真实状态，不是遗漏：它原本唯一的消费者是 7 条 ``combo_*`` 规则，
    #   那些规则删掉之后就没有了。
    #
    #   但它是**给玩家用的公共 DSL 函数**（docs/combat_architecture.md 里有），
    #   玩家在自己的 rules/*.toml 里写 onfield('夜兰') 完全正常。所以判它
    #   「必须有人用」会逼着仓库留一条假规则来凑数 —— 那比没有消费者更糟。
    #
    #   它的接线仍然被上面 ① 配置段 ② ``PolicyContext.onfield`` 读
    #   ``active_member`` 两条守着；端到端三态用例在
    #   tests/unit/test_character_id_wiring.py。
    assert CONDITIONAL_FUNCS["onfield"][0] == "character_id"


def _combos():
    """发布的那份连招配置（没有就当空）。"""
    from autopilot.policy.combo_manager import (  # noqa: PLC0415
        DEFAULT_COMBOS_PATH, load_optional_combos,
    )

    book = load_optional_combos(DEFAULT_COMBOS_PATH)
    return list(book) if book is not None else []


@pytest.mark.unit
def test_future_dir__only_contains_unreachable_rules() -> None:
    """反向守卫：``_future/`` 里不该混进其实能触发的规则。

    ★ 2026-10-18 修正：判「能不能触发」**不能只看 when 里的信号**
      ``_dead_reasons`` 只看信号，不看 ``enabled`` / ``scenes``。于是
      ``combat_skill_ready``（``enabled = false`` + ``scenes = ["COMBAT"]``）
      在 ``e_ready()`` 变成条件活信号之后被判成「能触发」——
      **假阳性**：它既关着、又锁在一个永不产出的场景上，两条都够它躺一辈子。

      留着 :func:`_structurally_parked` 一起判：只要它还有**任何一条**
      「结构上就发不出去」的理由，就该继续待在 ``_future/``。
      反过来，两边都判不出理由才报错。
    """
    reachable = []
    for f, r in _pending():
        if not _dead_reasons(r["when"]) and not _structurally_parked(r):
            reachable.append(f"{f}:{r['name']}")
    assert not reachable, (
        f"这些规则其实能触发，不该待在 _future/：{reachable}")


def _structurally_parked(rule: dict) -> list[str]:
    """与信号无关、光靠规则自身就发不出去的理由。

    只给 :func:`test_future_dir__only_contains_unreachable_rules` 用 ——
    活跃目录那条守卫**故意不看**这些：那里放一条 ``enabled = false``
    是「先摆着别删」的正常写法（``combat_settle_on_enter`` 就是），
    拿这条去卡活跃目录会把正常代码判成违规。
    """
    out = []
    if rule.get("enabled") is False:
        out.append("enabled = false")
    scenes = rule.get("scenes") or []
    # "*" / 空 = 不限场景；否则只看是不是全部落在产出不到的场景里
    dead = [s for s in scenes if s not in LIVE_SCENES]
    if scenes and "*" not in scenes and len(dead) == len(scenes):
        out.append(f"scenes={scenes} 全是产出不到的场景")
    return out


# ==================================================================
# 结构合法性（原有守卫保留）
# ==================================================================

@pytest.mark.unit
def test_rules__all_files_load() -> None:
    for p in list(RULES.glob("*.toml")) + list(FUTURE.glob("*.toml")):
        for r in _load(p):
            for key in ("name", "action", "when", "priority", "scenes",
                        "enabled", "note"):
                assert key in r, f"{p.name}:{r.get('name')} 缺字段 {key}"


@pytest.mark.unit
def test_rules__actions_are_in_whitelist() -> None:
    """动作不在 ActionType 里 -> 规则**加载期**就失败，不是运行期。"""
    valid = {a.value for a in ActionType}
    bad = [(f, r["name"], r["action"]) for f, r in _active()
           if r["action"] not in valid]
    assert not bad, f"这些规则的动作不在白名单里：{bad}"


@pytest.mark.unit
def test_rules__dsl_functions_are_registered() -> None:
    unknown = []
    for f, r in _active():
        for m in re.finditer(r"\b([a-z_][a-z0-9_]*)\s*\(", r["when"]):
            name = m.group(1)
            if name in ("and", "or", "not"):
                continue
            if name not in FUNCTION_NAMES:
                unknown.append((f, r["name"], name))
    assert not unknown, f"这些规则引用了未注册的 DSL 函数：{unknown}"


@pytest.mark.unit
def test_rules__new_rules_all_disabled() -> None:
    """新规则必须全部 enabled=false —— 一个都没真机验过。"""
    for name in NEW_FILES:
        p = RULES / name
        if not p.is_file():
            continue            # bag_advanced 已挪进 _future（那条也永远不触发）
        for r in _load(p):
            assert r["enabled"] is False, f"{name}:{r['name']} 被打开了"


@pytest.mark.unit
def test_rules__no_heavy_priority_collision() -> None:
    """同一场景同一优先级不超过 2 条。

    「命中即停」下同优先级会互相遮蔽，而遮蔽顺序取决于**文件加载顺序**。
    """
    buckets = defaultdict(list)
    for f, r in _active():
        for sc in r["scenes"]:
            buckets[(sc, r["priority"])].append(f"{f}:{r['name']}")
    heavy = {k: v for k, v in buckets.items() if len(v) > 2}
    assert not heavy, f"这些 (场景, 优先级) 挤了 3 条以上：{heavy}"


@pytest.mark.unit
def test_rules__world_collision_is_known() -> None:
    """把 PRIORITY_MAP 里记录的冲突固化成测试。"""
    buckets = defaultdict(list)
    for f, r in _active():
        if "WORLD" in r["scenes"]:
            buckets[r["priority"]].append(r["name"])
    collisions = {p: n for p, n in buckets.items() if len(n) > 1}
    known = {30, 50}
    unexpected = set(collisions) - known
    assert not unexpected, (
        f"PRIORITY_MAP.md 没记录的新冲突："
        f"{ {p: collisions[p] for p in unexpected} }")


# ==================================================================
# since() —— 2026-10-11 阶段一信号
# ==================================================================

@pytest.mark.unit
def test_since__returns_ms_since_scene_entry() -> None:
    """场景刚进来 -> 小值；场景持续 -> 大值。

    ★ 这条曾经被**误判**成死信号：探测用的合成 StatePacket 没带
      ``scene.entered_ms``，于是 ``since()`` 恒 UNKNOWN。但
      ``builder.py:592`` 的 ``_track_scene_enter`` **一直**在设它 ——
      是夹具缺字段，不是生产没接。教训见 test 文件顶部的 DEAD_FUNCS 注释。
    """
    from autopilot.contracts.state_packet import SceneInfo, StatePacket
    from autopilot.policy.context import PolicyContext as _PC

    def _ctx(entered_ms, now_ms):
        st = StatePacket.unknown(game_id="g", state_version=1, reason="t")
        object.__setattr__(
            st, "scene",
            SceneInfo(id="WORLD", confidence=0.9, evidence=("p",),
                      entered_ms=entered_ms))
        return _PC.from_state(st, now_ms=now_ms)

    assert _ctx(1_000_000, 1_000_000).since() == 0, "刚进入应该是 0"
    assert _ctx(1_000_000, 1_003_500).since() == 3_500
    # 时钟倒退不能给出负数
    assert _ctx(1_000_000, 500_000).since() == 0, "倒退要被钳到 0"


@pytest.mark.unit
def test_since__unknown_when_entered_ms_missing() -> None:
    """真的没设 entered_ms 时仍是 UNKNOWN —— 不是 0。"""
    from autopilot.contracts.state_packet import StatePacket
    from autopilot.policy.context import PolicyContext as _PC
    from autopilot.policy.context import UnknownValue

    st = StatePacket.unknown(game_id="g", state_version=1, reason="t")
    v = _PC.from_state(st, now_ms=1).since()
    assert isinstance(v, UnknownValue)


@pytest.mark.unit
def test_since__builder_actually_sets_entered_ms() -> None:
    """★ 守「builder 一直在设 entered_ms」这个事实。

    上面两条测的是 PolicyContext。**真正让它变成活信号的是 builder 那一行**
    —— 哪天有人把 `_track_scene_enter` 拆了，since() 会静默退化成 UNKNOWN，
    而所有依赖它的规则也会跟着静默失效，且没有任何报错。
    """
    import inspect

    from autopilot.perception.builder import StateBuilder

    src = inspect.getsource(StateBuilder._track_scene_enter)
    assert "self._scene_entered_ms" in src, (
        "_track_scene_enter 不再写入 _scene_entered_ms —— since() 会退化成 "
        "UNKNOWN，依赖它的规则会静默失效")
    build_src = inspect.getsource(StateBuilder.build)
    assert "_track_scene_enter" in build_src, (
        "build() 不再调用 _track_scene_enter")


# ==================================================================
# 新增的 DSL 函数
# ==================================================================

@pytest.mark.unit
def test_rules__new_dsl_functions_work() -> None:
    """4 个新函数：有实体给真值，没实体给 UNKNOWN。"""

    class _E:
        def __init__(self, kind, group, pos=(640.0, 360.0), bbox=None):
            self.kind = kind
            self.group = group
            self.screen_pos = pos
            # ⚠️ h 必须 >= ENEMY_HEALTH_BAR_H_MIN（0.015），否则
            #    is_enemy_evidence 会按「UI 细条」挡掉（2026-10-18）。
            #    这里的 0.02 是**实测**值，不是随手写的 ——
            #    notes/a5_attack_gating_root_cause.md §2.3 记的唯一一条
            #    真实敌人血条 bbox=(0.30, 0.22, 0.05, 0.02)。
            self.bbox = bbox or (0.5, 0.4, 0.1, 0.02)
            self.conf = 0.9

    from autopilot.contracts.state_packet import StatePacket
    from autopilot.policy.context import PolicyContext as _PC
    from autopilot.policy.context import UnknownValue

    ents = (_E("health_bar", "combat"), _E("health_bar", "combat"),
            _E("chest", "chest"), _E("locked_chest", "chest", pos=(300.0, 300.0)),
            _E("ore", "ore"), _E("drops", "drops"))
    st = StatePacket.unknown(game_id="genshin", state_version=1, reason="t")
    ctx = _PC.from_state(st, now_ms=1_000, world_entities=ents)

    assert ctx.enemy_bar_count() == 2
    assert ctx.has_locked_chest() is True
    assert ctx.nearby_count("chest") == 2
    assert ctx.nearby_count("drops") == 1
    assert ctx.nearest_group_px("chest") < 400.0
    assert ctx.nearest_group_px("ore") == pytest.approx(0.0)

    ctx2 = _PC.from_state(st, now_ms=1_000, world_entities=None)
    for fn, arg in (("enemy_bar_count", None), ("has_locked_chest", None),
                    ("nearby_count", "chest"), ("nearest_group_px", "ore")):
        v = getattr(ctx2, fn)() if arg is None else getattr(ctx2, fn)(arg)
        assert isinstance(v, UnknownValue), (
            f"{fn}() 在没有实体信息时返回 {v!r}，必须是 UnknownValue")


@pytest.mark.unit
def test_rules__enemy_count_falls_back_to_health_bars() -> None:
    """★ 任务 C：`enemy_count()` 不再恒 UNKNOWN。

    它先读 ``PolicySignals``（云端/上层给的权威值）；没给时**退回**
    ``enemy_bar_count()`` —— 世界实体检测已经在数血条了，只是没人用。
    仍然读不到（比如没注入检测器）时保持 UNKNOWN，绝不返回 0：
    返回 0 会让「附近一个敌人都没有」变成默认假设。
    """
    from autopilot.contracts.state_packet import StatePacket
    from autopilot.policy.context import PolicyContext as _PC
    from autopilot.policy.context import UnknownValue

    class _E:
        kind = group = "combat"
        # h=0.02 = 实测真敌人血条高度（见 is_enemy_evidence 的常量注释）。
        # 原来的 0.01 是随手写的，被 2026-10-18 的尺寸下限判掉了。
        bbox = (0.5, 0.4, 0.1, 0.02)
        screen_pos = (640.0, 360.0)
        conf = 0.9

    st = StatePacket.unknown(game_id="genshin", state_version=1, reason="t")
    with_bars = _PC.from_state(st, now_ms=1_000,
                               world_entities=(_E(), _E(), _E()))
    assert with_bars.enemy_count() == 3

    without = _PC.from_state(st, now_ms=1_000, world_entities=None)
    assert isinstance(without.enemy_count(), UnknownValue), (
        "没有实体信息时 enemy_count() 必须还是 UNKNOWN，不能退化成 0")


@pytest.mark.unit
def test_rules__policy_tree_loads_active_only() -> None:
    """策略树加载的条数必须等于活跃目录的条数，且不含 _future。"""
    from autopilot.tools._stack import build_policy

    src, rules_dir, count = build_policy(None)
    active = sum(len(_load(p)) for p in rules_dir.glob("*.toml"))
    assert count == active, f"加载了 {count} 条，活跃目录只有 {active} 条"
    names = {getattr(r, "name", None) for r in getattr(src, "rules", ())}
    assert "recovery_hold_when_perception_down" in names
    assert "bag_watchdog_close" not in names, "已死的 bag 规则不该被加载"