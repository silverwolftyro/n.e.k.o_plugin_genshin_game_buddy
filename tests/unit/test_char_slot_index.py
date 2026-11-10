"""免配置切人（``CharSlotIndex``）+ 头像分类器降级路径的测试。

★ 这份测试在守什么
------------------
1. **免配置闭环**：用户什么都不填，引擎靠大世界 OCR 自动建立
   「槽位 -> 角色名」，然后 ``SWITCH_CHARACTER(char_id='夜兰')`` 按对数字键。
2. **用户配置优先**：手填了 ``[char_slots]`` 就以用户为准 ——
   OCR 可能误识，而用户是**故意**写的。
3. **失败不许撒谎**：查不到槽位时跳过并在 ``steps`` 里说明原因，
   绝不按一个猜出来的键。
4. **头像分类器默认关**，且置信度不够时必须报 UNKNOWN ——
   **错名字比没有名字更糟**：``onfield()`` 会拿着它去判连招。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.actuator.active_slot import (  # noqa: E402
    ActiveSlotTracker, CharSlotIndex,
)
from autopilot.actuator.executor import IntentExecutor  # noqa: E402
from autopilot.actuator.simulator import DryRunBackend, InputSimulator  # noqa: E402
from autopilot.contracts.intent_packet import IntentPacket  # noqa: E402
from autopilot.contracts.state_packet import (  # noqa: E402
    SCENE_WORLD, SceneInfo, StatePacket,
)
from autopilot.perception.builder import StateBuilder  # noqa: E402

PARTY = {1: "纳西妲", 2: "钟离", 3: "夜兰", 4: "丝柯克"}


def _state() -> StatePacket:
    return StatePacket(scene=SceneInfo(id=SCENE_WORLD, confidence=0.9))


def _switch(name: str):
    return IntentPacket.make("SWITCH_CHARACTER", params={"char_id": name})


def _executor(**kw) -> IntentExecutor:
    sim = InputSimulator(backend=DryRunBackend())
    return IntentExecutor(simulator=sim, **kw)


# ==================================================================
# 1 · 名册索引
# ==================================================================
class TestCharSlotIndex:
    def test_round_trip(self) -> None:
        idx = CharSlotIndex()
        for slot, name in PARTY.items():
            assert idx.note_name(slot, name) is True
        for slot, name in PARTY.items():
            assert idx.name_of(slot) == name
            assert idx.slot_of(name) == slot
        assert idx.complete is True

    def test_mapping_shape_is_name_to_slot(self) -> None:
        """给 ``IntentExecutor(char_slots=...)`` 用的方向不能搞反。"""
        idx = CharSlotIndex()
        idx.note_name(3, "夜兰")
        assert idx.mapping() == {"夜兰": 3}

    def test_empty_name_is_ignored_and_keeps_the_old_one(self) -> None:
        """★ OCR 在切屏/开菜单时整栏读不到 —— 那时**不能**清空名册。

        清空会让「已知队伍」闪成空，执行层突然切不动人。
        """
        idx = CharSlotIndex()
        idx.note_name(2, "钟离")
        assert idx.note_name(2, "") is False
        assert idx.note_name(2, "   ") is False
        assert idx.name_of(2) == "钟离"

    def test_unchanged_name_reports_false(self) -> None:
        """轮转 OCR 每轮都会报同样的名字 —— 那时不该一直刷「名册更新」。"""
        idx = CharSlotIndex()
        assert idx.note_name(1, "纳西妲") is True
        assert idx.note_name(1, "纳西妲") is False

    def test_illegal_slot_rejected(self) -> None:
        idx = CharSlotIndex()
        assert idx.note_name(0, "钟离") is False
        assert idx.note_name(5, "钟离") is False
        assert idx.slot_of("钟离") is None

    def test_unknown_name_returns_none(self) -> None:
        idx = CharSlotIndex()
        idx.note_name(2, "钟离")
        assert idx.slot_of("甘雨") is None
        assert idx.slot_of("") is None

    def test_complete_is_false_when_partial(self) -> None:
        idx = CharSlotIndex()
        idx.note_name(1, "纳西妲")
        assert idx.complete is False

    def test_change_of_slot_updates_the_name(self) -> None:
        idx = CharSlotIndex()
        idx.note_name(1, "纳西妲")
        idx.note_name(1, "钟离")
        assert idx.name_of(1) == "钟离"
        assert idx.slot_of("纳西妲") is None


# ==================================================================
# 2 · 免配置闭环：OCR 建名册 -> 执行层切人
# ==================================================================
class TestZeroConfigSwitching:
    def test_end_to_end_without_any_user_config(self) -> None:
        """★ 用户什么都不填，引擎自己认队伍、自己切人。"""
        idx, trk = CharSlotIndex(), ActiveSlotTracker(stale_after_ms=0)
        sb = StateBuilder(active_slots=trk, char_slot_index=idx)
        # ⚠️ 每步换一个 executor：这只是为了让两步之间不被**限流器**
        #   挡掉（见下面 test_skip_also_consumes_the_rate_limit_budget）。
        #   共享的是 idx / trk —— 那才是被测对象。
        ex1 = _executor(char_slots={}, active_slot_tracker=trk, slot_index=idx)

        # 第一步：名册还是空的 -> 跳过，且**说清楚为什么**
        r = ex1.execute(_switch("夜兰"), state=_state())
        assert "夜兰" in " ".join(r.steps)
        assert any("跳过" in s for s in r.steps), r.steps
        assert trk.current() is None

        # 第二步：OCR 把 4 格都读出来了
        for slot, name in PARTY.items():
            sb._publish_char_slots({slot: name})
        assert idx.mapping() == {"纳西妲": 1, "钟离": 2, "夜兰": 3, "丝柯克": 4}

        # 第三步：同一个规则现在能切到正确的数字键
        ex2 = _executor(char_slots={}, active_slot_tracker=trk, slot_index=idx)
        r = ex2.execute(_switch("夜兰"), state=_state())
        assert any("slot 3" in s for s in r.steps), r.steps
        assert trk.current() == 3, "切完人必须记账，否则 onfield() 跟不上"

    def test_skip_also_consumes_the_rate_limit_budget(self) -> None:
        """★ 一条容易被当成 bug 的**真实行为**，钉在这里免得以后当 bug 修。

        ``safety.SafetyGuard`` 在 ``check()`` 里就写了 ``_last_action_ms``，
        而**跳过发生在 check() 之后** —— 所以一次「查不到槽位」的跳过
        照样会消耗 ``rate_limit_ms``（50 ms）的额度。

        后果：名册刚填满的那一刻，如果上一 tick 恰好跳过过，
        这一 tick 的切人会被限流挡掉，下一 tick（200 ms 后）才成。
        **只慢一拍，不出错**。所以没改 —— 改「跳过不计入限流」会削弱
        输出节流，而那是有安全含义的。
        """
        idx, trk = CharSlotIndex(), ActiveSlotTracker(stale_after_ms=0)
        ex = _executor(char_slots={}, active_slot_tracker=trk, slot_index=idx)
        r1 = ex.execute(_switch("夜兰"), state=_state())
        assert any("跳过" in s for s in r1.steps)
        idx.note_name(3, "夜兰")
        r2 = ex.execute(_switch("夜兰"), state=_state())
        assert r2.ok is False
        assert "RATE_LIMITED" in (r2.reason or "")

    def test_user_config_beats_ocr(self) -> None:
        """★ 手填 ``[char_slots]`` 时以用户为准。

        OCR 有误识可能（实测 296 帧里就有 6 条 1 字名误吸附），
        用户是**故意**写的，听用户的。
        """
        idx = CharSlotIndex()
        for slot, name in PARTY.items():
            idx.note_name(slot, name)
        ex = _executor(char_slots={"夜兰": 1}, slot_index=idx)
        r = ex.execute(_switch("夜兰"), state=_state())
        assert any("slot 1" in s for s in r.steps), r.steps

    def test_illegal_user_value_is_not_silently_replaced_by_ocr(self) -> None:
        """``[char_slots]`` 写错值（比如 ``"夜兰" = 9``）**不能**悄悄
        回退到 OCR —— 那会把用户的笔误藏起来，按错人还不报错。"""
        idx = CharSlotIndex()
        idx.note_name(3, "夜兰")
        ex = _executor(char_slots={"夜兰": 9}, slot_index=idx)
        r = ex.execute(_switch("夜兰"), state=_state())
        assert any("跳过" in s for s in r.steps), r.steps

    def test_unknown_character_is_skipped(self) -> None:
        idx = CharSlotIndex()
        idx.note_name(1, "纳西妲")
        ex = _executor(char_slots={}, slot_index=idx)
        r = ex.execute(_switch("甘雨"), state=_state())
        assert any("跳过" in s for s in r.steps), r.steps

    def test_no_index_no_switch(self) -> None:
        """没接索引、也没静态表 -> 切不动，但**说得出来**。"""
        ex = _executor(char_slots={})
        r = ex.execute(_switch("夜兰"), state=_state())
        assert any("跳过" in s for s in r.steps), r.steps


# ==================================================================
# 3 · 名册 -> PartyMember.active -> onfield()
# ==================================================================
class TestRosterFeedsActiveMember:
    def test_ocr_names_reach_the_builder(self) -> None:
        trk, idx = ActiveSlotTracker(stale_after_ms=0), CharSlotIndex()
        sb = StateBuilder(active_slots=trk, char_slot_index=idx)
        trk.note_switch(3, source="engine")
        for slot, name in PARTY.items():
            sb._publish_char_slots({slot: name})

        # 没有 character_id 时名字来自 OCR 缓存（本测试直接注入）
        sb._char_detector = None
        sb.last_character_names = dict(PARTY)
        info = sb._build_player("WORLD", None, [], [], 1, frame=None)
        active = info.active_member
        assert active is not None
        assert active.slot == 3
        assert active.active is True

    def test_publish_failure_does_not_break_the_tick(self) -> None:
        """名册索引挂了不该掀翻感知这条 tick。"""

        class _Boom:
            def note_name(self, slot, name):
                raise RuntimeError("index 挂了")

        sb = StateBuilder(char_slot_index=_Boom())
        sb._publish_char_slots({1: "纳西妲"})      # 不抛就算过


# ==================================================================
# 4 · 头像分类器（降级路径）
# ==================================================================
class TestAvatarClassifierFallback:
    def test_defaults_to_off(self) -> None:
        from autopilot.config.loader import load_config

        cfg = load_config().avatar_classifier
        assert cfg.enabled is False, "实测准确率在随机水平，默认必须关"
        assert cfg.min_confidence > 0, "没有置信度闸 = 每次都输出一个名字"

    def test_no_classifier_means_no_fallback(self) -> None:
        sb = StateBuilder()
        assert sb._avatar_fallback(np.zeros((720, 1280, 3), np.uint8)) == {}

    def test_missing_model_file_returns_none(self) -> None:
        """可选功能缺模型时保持关闭，**不抛**。"""
        from autopilot.perception.detectors.avatar_classifier import (
            AvatarClassifier,
        )

        assert AvatarClassifier.load("D:/不存在/character_classifier.onnx") is None

    def test_low_confidence_is_reported_as_unknown(self) -> None:
        """★ 错名字比没有名字更糟 —— 必须有置信度闸。"""
        from autopilot.perception.detectors.avatar_classifier import (
            AvatarClassifier,
        )

        class _Stub(AvatarClassifier):
            def __init__(self, conf):                      # noqa: D107
                self._conf = conf

            def classify(self, crop, *, top_k=5):
                from autopilot.perception.detectors.avatar_classifier import (
                    AvatarPrediction,
                )
                return AvatarPrediction(name="钟离", confidence=self._conf)

        from autopilot.perception.detectors.character_detector import (
            CharacterDetector,
        )

        frame = np.zeros((1440, 2304, 3), np.uint8)
        det = CharacterDetector(object(), classifier=_Stub(0.10))
        pred = det.classify_slot(frame, 1, min_confidence=0.35)
        assert pred.known is False, "置信度 0.10 < 0.35 却报了名字"
        assert pred.confidence == 0.10, "置信度要透出来供排查"

        det2 = CharacterDetector(object(), classifier=_Stub(0.90))
        assert det2.classify_slot(frame, 1, min_confidence=0.35).known is True

    def test_builder_fallback_uses_confidence_gate(self) -> None:
        from autopilot.config.loader import AvatarClassifierConfig
        from autopilot.perception.detectors.avatar_classifier import (
            AvatarPrediction,
        )
        from autopilot.perception.detectors.character_detector import (
            CharacterDetector,
        )

        class _Stub:
            def classify(self, crop, *, top_k=5):
                return AvatarPrediction(name="夜兰", confidence=0.9)

        cfg = AvatarClassifierConfig(enabled=True, min_confidence=0.35)
        sb = StateBuilder(avatar_classifier=_Stub(), avatar_classifier_cfg=cfg)
        sb._char_detector = CharacterDetector(object(), classifier=_Stub())
        sb._avatar_classifier = _Stub()
        got = sb._avatar_fallback(np.zeros((1440, 2304, 3), np.uint8))
        # 4 个槽位全部命中（用同一张全黑帧，stub 无所谓）
        assert set(got) == {1, 2, 3, 4}, got
        assert set(got.values()) == {"夜兰"}

    def test_fallback_error_is_swallowed(self) -> None:
        """分类器逐帧失败不该让 OCR 那条（已经够用的那条）一起失效。"""

        class _Boom:
            def classify_slot(self, *a, **k):
                raise RuntimeError("推理炸了")

        from autopilot.config.loader import AvatarClassifierConfig

        sb = StateBuilder(avatar_classifier=object(),
                          avatar_classifier_cfg=AvatarClassifierConfig())
        sb._char_detector = _Boom()
        sb._avatar_classifier = object()
        assert sb._avatar_fallback(np.zeros((10, 10, 3), np.uint8)) == {}


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
