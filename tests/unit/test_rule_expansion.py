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
#: ⚠️ 两个**不要**加进来的：
#:   ``enemy_count`` —— 2026-10-11 已接血条兜底，是活信号。
#:   ``since``      —— builder.py:592 **一直**在设 ``scene.entered_ms``，
#:   所以 since() 在生产里是真值。之前我用合成的 StatePacket 探测（没带
#:   entered_ms 字段），得出了「since 恒 UNKNOWN」的**错误结论**，还据此把
#:   combat_settle_on_enter 挪进了 _future/。这条已经改正。
#:   ——教训：探测用的夹具必须**逐字段复刻真实状态**，否则会把「夹具缺字段」
#:   误判成「生产没接」。
DEAD_FUNCS = {"hp", "energy", "resin", "pos_quality", "pos_usable",
              "q_ready", "e_ready", "has_enemy",
              "target_locked", "onfield", "slot"}
LIVE_SCENES = {"WORLD", "UNKNOWN"}

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
    """★★ 核心守卫：活跃目录里不许有「结构合法但永远不触发」的规则。"""
    offenders = []
    for f, r in _active():
        for why in _dead_reasons(r["when"]):
            offenders.append(f"{f}:{r['name']}  ({why})")
    assert not offenders, (
        "这些规则依赖今天恒 UNKNOWN 的信号，应该挪到 rules/_future/：\n  "
        + "\n  ".join(offenders))


@pytest.mark.unit
def test_future_dir__only_contains_unreachable_rules() -> None:
    """反向守卫：``_future/`` 里不该混进其实能触发的规则。"""
    reachable = []
    for f, r in _pending():
        if not _dead_reasons(r["when"]):
            reachable.append(f"{f}:{r['name']}")
    assert not reachable, (
        f"这些规则其实能触发，不该待在 _future/：{reachable}")


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
            self.bbox = bbox or (0.5, 0.4, 0.1, 0.01)
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
        bbox = (0.5, 0.4, 0.1, 0.01)
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