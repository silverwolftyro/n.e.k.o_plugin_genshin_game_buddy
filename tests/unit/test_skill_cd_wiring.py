"""E 技能就绪感知**接线**的单测（2026-10-18）。

覆盖这条链::

    rois.toml[key_e] -> detectors.key_e.read_skill_cd()
                     -> StateBuilder._build_player()
                     -> PlayerInfo.party[0].cd_ready["skill"]
                     -> PolicyContext.e_ready()
                     -> 规则 world_use_skill_if_ready

★ 全程**不需要视觉**、不需要游戏窗口、不需要模型
  感知那一段用 monkeypatch 换掉，规则那一段用合成 StatePacket。
  这就是「离线可验」的意思：契约与接线能在没有真机的情况下证明是通的。

★ 三条底线（任何一条破了都是功能回退）
  1. ``cd_ready={"skill": True}``  -> ``e_ready() is True``，规则**命中** USE_SKILL
  2. ``cd_ready={"skill": False}`` -> ``e_ready() is False``，规则**不命中** USE_SKILL
  3. ``cd_ready={}`` / party 为空   -> ``e_ready()`` 是 UnknownValue，规则**不命中**

第 3 条是安全兜底：感知没开、没读到、读挂了，都必须落在这里而不是第 1 条。
把 UNKNOWN 当成「就绪」会白按 E；当成「CD 中」会让人以为规则写错了。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autopilot.config.loader import SkillCdConfig, load_config
from autopilot.contracts.state_packet import (
    PartyMember,
    PlayerInfo,
    ResourceInfo,
    SceneInfo,
    StatePacket,
)
from autopilot.perception import builder as B
from autopilot.perception.detectors import key_e
from autopilot.perception.detectors.key_e import CD_KEY, KeyEState, read_skill_cd
from autopilot.perception.world_entities import WorldEntity
from autopilot.policy.context import PolicyContext, UnknownValue

# ==================================================================
# 夹具
# ==================================================================
#: 合成帧尺寸 —— ROI 是**比例**，尺寸只要不是 0 就行
FRAME_H, FRAME_W = 1600, 2560


def _frame() -> np.ndarray:
    """一张什么都读不出来的整帧（纯中灰）。

    ★ 为什么要**中灰**而不是纯黑：纯黑 S=0，三个阈值都不满足，
    :func:`detect_key_e` 会走 UI 判定返回 UNKNOWN —— 那是「看不出是
    不是 UI」。中灰同样 UNKNOWN，但走的是「饱和度落在灰带」这条路，
    两条路都该归 UNKNOWN，都归在本文件的测���范围里。
    """
    return np.full((FRAME_H, FRAME_W, 3), 128, np.uint8)


def _sb(**kw) -> B.StateBuilder:
    """最小 StateBuilder：只测接线，不建模型、不建捕获器。"""
    return B.StateBuilder(**kw)


def _member(skill: bool | None | object = ...) -> PartyMember:
    """造一个「在场 + 有 cd_ready」的队员。

    ``...``（省略号哨兵）= ``cd_ready={}`` —— 即「记了成员但没记 CD」，
    这与「压根没有 active 成员」是两种不同的 UNKNOWN，分开测。
    """
    cd: dict[str, bool | None] = {} if skill is ... else {CD_KEY: skill}  # type: ignore[dict-item]
    return PartyMember(slot=1, char_id="", name="", active=True, cd_ready=cd)


def _state(skill: bool | None | object = ...) -> StatePacket:
    return StatePacket(
        scene=SceneInfo(id="WORLD", confidence=1.0),
        player=PlayerInfo(party=(_member(skill),)),
        resource=ResourceInfo(perception_ok=True),
        state_version=1,
    )


def _enemy() -> tuple[WorldEntity, ...]:
    """一个够格当「附近有敌人」的实体。

    判据 B：``health_bar``（group=combat）且 bbox 的 y < 0.5
    （自己的血条在 y > 0.9，见 context.has_nearby_enemy 的实测表），
    **且 h >= ENEMY_HEALTH_BAR_H_MIN**（2026-10-18 加的尺寸下限，
    用来挡掉 NPC 绿条和 UI 细条）。

    ⚠️ h 必须是 **0.02**（实测值，见
      ``notes/a5_attack_gating_root_cause.md`` §2.3），不能随手写 0.01 ——
      那个值落在实测误报区间 [0.005, 0.011] 里面，会被下限挡掉，
      而失败信息会指向「敌人不存在」而不是「夹具尺寸写错了」，
      排查方向直接被带偏。
    """
    return (WorldEntity(kind="health_bar", bbox=(0.3, 0.22, 0.08, 0.02),
                        confidence=0.9, screen_pos=(360, 260)),)


def _ctx(state: StatePacket, *, enemy: bool = True) -> PolicyContext:
    return PolicyContext.from_state(
        state,
        now_ms=1_000_000,
        world_entities=_enemy() if enemy else (),
    )


# ==================================================================
# 1 · 配置层
# ==================================================================
class TestConfig:
    def test_default_is_on(self) -> None:
        """2026-10-18 起**默认打开** —— 用户确认要放技能。

        ⚠️ 打开的**前提**是「阈值可能不准」，所以它同时意味着：
        冷却中也可能判就绪。要恢复旧行为把这里改回 ``False`` 即可，
        ``_read_skill_cd`` 会直接跳过，``party`` 回到 ``()``。
        """
        assert SkillCdConfig().enabled is True

    def test_defaults_toml_ships_it_on(self) -> None:
        """发行包里带的 defaults.toml 与代码默认值必须一致。

        两者不一致时症状极其难查：源码里 ``True`` 看着像接通了，
        真机上却因为包里的 toml 是 ``False`` 而永远 UNKNOWN。
        """
        cfg = load_config()
        assert cfg.skill_cd.enabled is SkillCdConfig().enabled

    def test_disabling_restores_the_pre_integration_behaviour(self) -> None:
        """★ 逃生阀：关掉就完全回到接入之前，一个字节的额外行为都不剩。"""
        cfg = SkillCdConfig(enabled=False)
        p = _sb(skill_cd=cfg)._build_player("WORLD", None, [], [], 1,
                                           frame=_frame())
        assert p.party == ()

    def test_toml_parses_the_section(self) -> None:
        """``[autopilot.skill_cd]`` 真能被解析（拼错键名会在这里炸）。"""
        cfg = load_config()
        assert isinstance(cfg.skill_cd, SkillCdConfig)


# ==================================================================
# 2 · 契约层：cd_ready -> e_ready()
# ==================================================================
class TestEReady:
    def test_true_reads_true(self) -> None:
        got = _ctx(_state(True)).e_ready()
        assert got is True, got

    def test_false_reads_false(self) -> None:
        got = _ctx(_state(False)).e_ready()
        assert got is False, got

    def test_none_value_is_unknown(self) -> None:
        """显式 None = 「看见了图标但读不出」-> UNKNOWN。

        ⚠ 不是 False：False 会让规则**确定地**不触发，而 UNKNOWN 会被
        记进 ``unknown_hits``，``rule_test.py`` 能说出「为什么没命中」。
        """
        ctx = _ctx(_state(None))
        got = ctx.e_ready()
        assert isinstance(got, UnknownValue), got
        assert ctx.unknown_hits, "UNKNOWN 必须被记下来，否则排查时是瞎的"

    def test_empty_cd_ready_is_unknown(self) -> None:
        assert isinstance(_ctx(_state()).e_ready(), UnknownValue)

    def test_no_party_is_unknown(self) -> None:
        """没有 active 成员（感知没开时的默认状态）。"""
        st = StatePacket(scene=SceneInfo(id="WORLD", confidence=1.0),
                         resource=ResourceInfo(perception_ok=True))
        assert isinstance(_ctx(st).e_ready(), UnknownValue)

    def test_unknown_never_compares_true(self) -> None:
        """★ UNKNOWN 的 ``__eq__``/``__bool__`` 恒 False —— 这是安全底座。

        规则里写 ``e_ready() == True`` 和写 ``e_ready()`` 都必须不触发；
        万一哪天有人把 ``__eq__`` 改成 True，这两条会一起红。
        """
        unk = _ctx(_state()).e_ready()
        assert (unk == True) is False          # noqa: E712 - 故意写 True 字面量
        assert bool(unk) is False


# ==================================================================
# 3 · 规则层：三种 cd_ready 各自的结果
# ==================================================================
_RULES_DIR = Path(__file__).resolve().parents[2] / "autopilot" / "rules"


class TestRuleWiring:
    """直接加载真实的 ``rules/`` 目录 —— 不 mock 规则文件。

    为什么要用真文件：规则的 ``when`` 表达式、priority 排序、
    ``scenes`` 匹配这三样只有真加载才验得到。用临时 TOML 造夹具的话，
    「写错了函数名」这类错误会一路绿灯到真机。
    """

    @pytest.fixture
    def tree(self):
        """**函数级**夹具：LocalPolicyTree 带共享状态。

        :class:`CadenceGate` 会记「上一条动作什么时候打的」，命中即记账；
        共用一棵树的话，第二条用例会被「节奏门控中（还需 xxx ms）」挡掉，
        症状看起来像规则坏了 —— 实际是夹具串味。
        """
        from autopilot.policy.local_tree import LocalPolicyTree

        t = LocalPolicyTree(rules_dir=_RULES_DIR)
        # ★ 世界实体是**树上的状态**，不是每次调用的参数 ——
        #   core/loop.py 每 tick 从 builder.last_world_entities 写进去。
        #   不设的话 has_nearby_enemy() 恒 UNKNOWN，两条战斗规则都不触发。
        t.last_world_entities = _enemy()
        return t

    def _fire(self, tree, skill: bool | None | object) -> tuple[str | None, object]:
        tr = tree.decide(_state(skill))
        if not tr.hit:
            return None, tr
        return tr.matched_rule.name, tr

    def test_ready_fires_use_skill(self, tree) -> None:
        """★ 用户要的核心断言：E 好就放 E。"""
        name, tr = self._fire(tree, True)
        assert name == "world_use_skill_if_ready", (name, tr)
        assert tr.matched_rule.action.value == "USE_SKILL"
        # decide() 与 poll() 共用同一条路径：真正产出的意图也是同一个动作
        assert tree.poll(_state(True)).action.value == "USE_SKILL"

    def test_cooldown_does_not_fire_skill(self, tree) -> None:
        """E 在 CD：技能规则不触发，应落到普攻规则。"""
        name, tr = self._fire(tree, False)
        assert name != "world_use_skill_if_ready", (name, tr)
        assert name == "world_attack_only_with_enemy", (name, tr)

    def test_unknown_does_not_fire_skill(self, tree) -> None:
        """★ 安全兜底：感知给不出结论时，**绝不能**放 E。

        这条最要紧 —— 放错了是「对着 CD 按 E」，白按且会让
        ``stats.by_reason`` 里堆出一堆无意义的拒因。
        """
        name, tr = self._fire(tree, ...)
        assert name != "world_use_skill_if_ready", (name, tr)
        assert name == "world_attack_only_with_enemy", (name, tr)

    def test_explicit_none_does_not_fire_skill(self, tree) -> None:
        name, _ = self._fire(tree, None)
        assert name != "world_use_skill_if_ready", name

    def test_no_enemy_fires_no_combat_rule(self, tree) -> None:
        """没有敌人时两条战斗规则都**不能命中**。

        ⚠ ``trace.entries`` 里含全部被评估过的规则（含 SKIPPED 的），
        所以不能直接断言「名字不在 entries 里」—— 那样断言的是
        「规则被加载了」，不是「规则没触发」。要看 verdict。
        """
        tree.last_world_entities = ()
        tr = tree.decide(_state(True))
        fired = {e.rule for e in tr.entries if e.verdict != "SKIPPED"}
        assert "world_use_skill_if_ready" not in fired, tr.entries
        assert "world_attack_only_with_enemy" not in fired, tr.entries

    def test_perception_down_short_circuits_everything(self, tree) -> None:
        """感知不健康时一条战斗规则都不产出（``decide`` 的第 ① 道短路）。

        这条保证「感知挂了」时不会因为上次缓存的 cd_ready 而乱按。
        """
        st = _state(True)
        st = StatePacket(
            scene=st.scene, player=st.player,
            resource=ResourceInfo(perception_ok=False),
            state_version=st.state_version,
        )
        tr = tree.decide(st)
        assert tr.hit is False
        assert tr.skipped_by_shortcircuit > 0

    def test_priority_puts_skill_before_attack(self, tree) -> None:
        """★ 技能必须排在普攻**前面**。

        规则是升序遍历、首个命中即停（``local_tree.decide``），
        数字越小越优先。所以这条断言钉的是「80 < 85」这个事实 ——
        哪天有人把技能规则写成 90，它就会永远被普攻挡住，而且**不报错**。
        """
        by_name = {r.name: r for r in tree.rules}
        assert "world_use_skill_if_ready" in by_name, sorted(by_name)
        assert by_name["world_use_skill_if_ready"].priority \
            < by_name["world_attack_only_with_enemy"].priority

    def test_rule_is_enabled_and_world_only(self, tree) -> None:
        r = {x.name: x for x in tree.rules}["world_use_skill_if_ready"]
        assert r.enabled is True
        assert r.scenes == ("WORLD",)


# ==================================================================
# 4 · 感知层接线：StateBuilder -> PartyMember
# ==================================================================
class TestBuilderWiring:
    def test_no_config_means_no_party(self) -> None:
        """★ 接入前的行为必须原样保留：party 恒空。"""
        p = _sb()._build_player("WORLD", None, [], [], 1, frame=_frame())
        assert p.party == ()

    def test_disabled_config_means_no_party(self) -> None:
        cfg = SkillCdConfig(enabled=False)
        p = _sb(skill_cd=cfg)._build_player("WORLD", None, [], [], 1,
                                           frame=_frame())
        assert p.party == ()

    def test_disabled_config_never_calls_the_detector(self, monkeypatch) -> None:
        """关着的时候**一次都不该**碰 ROI（省掉每帧的 HSV 开销）。"""
        calls: list[int] = []

        def _boom(frame):
            calls.append(1)
            return True, 1.0, "key_e"

        monkeypatch.setattr(B, "read_skill_cd", _boom)
        _sb(skill_cd=SkillCdConfig(enabled=False))._build_player(
            "WORLD", None, [], [], 1, frame=_frame())
        assert calls == []

    def test_enabled_fills_cd_ready(self, monkeypatch) -> None:
        monkeypatch.setattr(B, "read_skill_cd",
                            lambda frame: (True, 128.0, "key_e"))
        p = _sb(skill_cd=SkillCdConfig(enabled=True))._build_player(
            "WORLD", None, [], [], 1, frame=_frame())
        assert len(p.party) == 1
        assert p.party[0].active is True
        assert p.party[0].cd_ready == {"skill": True}

    def test_unknown_still_creates_the_member(self, monkeypatch) -> None:
        """★ 读不出来也要有成员（值 None）。

        因为 ``e_ready()`` 走的是 ``active_member.cd_ready``：
        没有成员 -> 报「队伍没有 active 成员」；
        有成员但值 None -> 报「skill 状态未知」。**后者更接近真相**，
        排查时能一眼看出「是感知没读到」，而不是「队伍感知压根没接」。
        """
        monkeypatch.setattr(B, "read_skill_cd",
                            lambda frame: (None, 12.0, "none"))
        p = _sb(skill_cd=SkillCdConfig(enabled=True))._build_player(
            "WORLD", None, [], [], 1, frame=_frame())
        assert len(p.party) == 1
        assert p.party[0].cd_ready == {"skill": None}

    def test_detector_exception_degrades_to_unknown(self, monkeypatch) -> None:
        """感知抛异常不能带崩整条 tick。"""

        def _raise(frame):
            raise RuntimeError("cv2 抽风")

        monkeypatch.setattr(B, "read_skill_cd", _raise)
        p = _sb(skill_cd=SkillCdConfig(enabled=True))._build_player(
            "WORLD", None, [], [], 1, frame=_frame())
        assert p.party[0].cd_ready == {"skill": None}

    def test_char_label_still_wins_for_identity(self, monkeypatch) -> None:
        """``char:`` 标签在的时候，身份照旧，cd 照样填进去。"""
        from autopilot.perception.yolo import Detection

        monkeypatch.setattr(B, "read_skill_cd",
                            lambda frame: (False, 5.0, "key_e"))
        det = [Detection("char:钟离", (10, 10, 20, 20), 0.9)]
        p = _sb(skill_cd=SkillCdConfig(enabled=True))._build_player(
            "WORLD", None, [], det, 1, frame=_frame())
        assert p.party[0].name == "钟离"
        assert p.party[0].cd_ready == {"skill": False}

    def test_no_frame_means_no_party(self, monkeypatch) -> None:
        """没给帧就什么都不做 —— 不拿上一帧的旧数据冒充本帧。"""
        monkeypatch.setattr(B, "read_skill_cd",
                            lambda frame: (True, 1.0, "key_e"))
        p = _sb(skill_cd=SkillCdConfig(enabled=True))._build_player(
            "WORLD", None, [], [], 1, frame=None)
        assert p.party == ()


# ==================================================================
# 5 · read_skill_cd 本身：三态映射 + 兜底
# ==================================================================
class TestReadSkillCd:
    @pytest.mark.parametrize("state, want", [
        (KeyEState.READY, True),
        (KeyEState.COOLDOWN, False),
        (KeyEState.UNKNOWN, None),
    ])
    def test_maps_every_state(self, monkeypatch, state, want) -> None:
        monkeypatch.setattr(key_e, "detect_key_e",
                            lambda frame: (state, 42.0))
        got, conf, src = read_skill_cd(_frame())
        assert got is want
        assert conf == 42.0
        assert src == ("none" if want is None else "key_e")

    def test_exception_becomes_unknown(self, monkeypatch) -> None:
        def _raise(frame):
            raise ValueError("roi 不在帧内")

        monkeypatch.setattr(key_e, "detect_key_e", _raise)
        assert read_skill_cd(_frame()) == (None, 0.0, "none")

    def test_empty_frame_is_unknown_not_crash(self) -> None:
        """空帧 / 小帧 —— 感知层不该崩，也不该给出 True。"""
        for bad in (np.zeros((0, 0, 3), np.uint8),
                    np.zeros((5, 5, 3), np.uint8)):
            got, _, _ = read_skill_cd(bad)
            assert got is None, bad.shape

    def test_cd_key_matches_what_context_reads(self) -> None:
        """★ 契约一致性：``CD_KEY`` 必须就是 ``e_ready()`` 硬编码的那一把。

        ``policy/context.py::e_ready()`` 里是 ``_active_cd("skill")`` 这个
        **字符串字面量**，改名不会报错、只会静默失效（规则永远 UNKNOWN）。
        这里用源码钉死。
        """
        import inspect

        from autopilot.policy.context import PolicyContext

        assert f'"{CD_KEY}"' in inspect.getsource(PolicyContext.e_ready)


# ==================================================================
# 6 · ROI 配置
# ==================================================================
class TestRoi:
    def test_key_e_roi_is_shipped(self) -> None:
        """``rois.toml`` 里必须真的有 ``key_e`` —— 缺了会静默退化成 UNKNOWN。"""
        from autopilot.perception.roi_loader import ROIS_TOML, load_rois

        rois = load_rois()
        assert "key_e" in rois, sorted(rois)
        r = rois["key_e"]
        # Rect 是 (left, top, right, bottom) 的归一化坐标，不是 (x,y,w,h)
        for v in (r.left, r.top, r.right, r.bottom):
            assert 0.0 <= v <= 1.0, (r, v)
        assert r.right > r.left and r.bottom > r.top, r
        # E 图标在右下角：top 必须在下半屏，否则读的是别的东西
        assert r.top > 0.5, r
        assert ROIS_TOML.is_file(), ROIS_TOML

    def test_roi_round_trips_on_a_real_frame(self) -> None:
        """★ ROI 在 2560x1600 上要真的能裁出东西（不是出框的 0 像素）。"""
        from autopilot.perception.roi_loader import load_rois

        crop = load_rois()["key_e"].crop(_frame())
        assert crop is not None and crop.size > 0, crop
