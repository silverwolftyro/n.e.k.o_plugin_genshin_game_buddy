"""OCR 场景分类器的单测（2026-10-11）。

样本是**原神界面上真实会出现的字**，不是编的。用它量准确率，
才有资格讨论「要不要接进主流程」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.perception.scene_from_ocr import (  # noqa: E402
    SceneOcrRefiner,
    classify_scene_from_text,
    normalize,
)

# (OCR 文本, 期望场景)。★ 分三档，因为**单弱词本来就不该定性**：
#
#   正样本：多个词叠加，或出现强词（「圣遗物」「传送锚点」…）
#   负样本：只有一个通用词 —— 期望回落 WORLD，不是具体场景
#   噪声  ：完全认不出来 —— 同样回落
#
# 负样本单列是因为它们恰恰是**真实会出事**的那批：「地图」「兑换」「道具」
# 这类词在大世界里也可能飘过，定性了就等于「野外乱开背包」。
SAMPLES_POS = [
    ("背包  道具  圣遗物  武器", "BAG_INVENTORY"),   # 4 个词叠加
    ("行囊", "BAG_INVENTORY"),                       # 强词
    ("圣遗物", "BAG_INVENTORY"),                     # 强词
    ("传送锚点", "MAP_BIG"),                          # 强词
    ("七天神像", "MAP_BIG"),                          # 强词
    ("继续   跳过", "DIALOG"),                        # 2 个词叠加
    ("获得  奖励  离开", "COMBAT_RESULT"),            # 3 个词叠加
    ("商店  购买", "SHOP"),                           # 2 个词叠加
    ("派蒙  设置  队伍", "MAIN_MENU"),                # 3 个词叠加
    ("地图  探索度 87%", "MAP_BIG"),                  # 叠加才过
]
SAMPLES_WEAK = [
    ("地图", "WORLD"),        # 单个通用词 —— 不足以定性
    ("兑换", "WORLD"),
    ("道具", "WORLD"),
    ("探索度 87%", "WORLD"),  # 大地图右上角常驻，最容易出事的一个
]
SAMPLES_NOISE = ["", "   ", "\n\n", "12345", "!!??", "abc"]


@pytest.mark.unit
def test_ocr_scene__positive_samples() -> None:
    wrong = []
    for text, want in SAMPLES_POS:
        g = classify_scene_from_text(text)
        if g.scene_id != want:
            wrong.append(f"{text!r} -> {g.scene_id}（期望 {want}）")
    assert not wrong, "\n".join(wrong)


@pytest.mark.unit
def test_ocr_scene__single_weak_word_does_not_classify() -> None:
    """单个通用词**不该**定性 —— 这正是防误判的那道闸。"""
    wrong = []
    for text, want in SAMPLES_WEAK:
        g = classify_scene_from_text(text, fallback="WORLD")
        if g.scene_id != want:
            wrong.append(f"{text!r} -> {g.scene_id}（期望回落 {want}）")
    assert not wrong, "\n".join(wrong)


@pytest.mark.unit
def test_ocr_scene__wild_text_falls_back_not_guesses() -> None:
    """★ 最关键的一条：**乱判的代价不对称**。

    把大世界误判成 BAG_INVENTORY，引擎就会在野外疯狂按 B 键开背包 ——
    症状离病因极远。判不出来（回落）几乎没有代价。
    """
    for noise in SAMPLES_NOISE:
        g = classify_scene_from_text(noise, fallback="WORLD")
        assert g.scene_id == "WORLD", f"{noise!r} 不该判出具体场景，却判成 {g.scene_id}"
        assert g.confidence == 0.0


@pytest.mark.unit
def test_ocr_scene__low_confidence_falls_back() -> None:
    """单个弱关键词不够 —— 宁可回落。

    「兑换」单独出现 0.35 < 0.55 阈值，但「商店」+「兑换」叠加才过。
    """
    weak = classify_scene_from_text("兑换", fallback="WORLD")
    assert weak.scene_id == "WORLD", "单个弱词不该定性"
    strong = classify_scene_from_text("商店 兑换", fallback="WORLD")
    assert strong.scene_id == "SHOP", "两个弱词叠加应该定性"


@pytest.mark.unit
def test_ocr_scene__normalize_strips_ocr_noise() -> None:
    assert normalize(" 背包 \n\t道具　圣遗物 ") == "背包道具圣遗物"
    assert normalize(None) == ""


@pytest.mark.unit
def test_ocr_scene__always_carries_evidence() -> None:
    """证据链必须存在 —— 判错了要能查为什么。"""
    for text in ("背包 道具", "", "zzz", "商店"):
        g = classify_scene_from_text(text)
        assert g.evidence, f"{text!r} 没有证据"


# ==================================================================
# 去抖
# ==================================================================

@pytest.mark.unit
def test_refiner__first_call_adopts() -> None:
    r = SceneOcrRefiner(n_seconds=1.0)
    assert r.update("背包 道具").scene_id == "BAG_INVENTORY"


@pytest.mark.unit
def test_refiner__requires_sustained_evidence_to_switch() -> None:
    """★ 单帧漏字不该让场景横跳。

    直接逐帧用结果的话，OCR 把「背包」认错一帧，场景就变；
    而 traits 的 allowed_scenes 是按场景过滤的 —— 横跳会让规则
    「时而能触发时而不能」，看起来像随机失效。

    ⚠️ 注意终点是 **WORLD** 不是 MAP_BIG：「探索度 87%」单独出现
    达不到定性阈值（见 test_ocr_scene__single_weak_word_does_not_classify），
    所以正确行为是回落到二值检测器给的 WORLD。
    """
    r = SceneOcrRefiner(n_seconds=1.0)
    assert r.update("背包 道具", now=0.0).scene_id == "BAG_INVENTORY"

    # 大世界画面回来，但只持续 0.5s < 1.0s
    got = r.update("探索度 87%", now=0.1)
    assert got.scene_id == "BAG_INVENTORY", "短暂抖动不该切场景"
    got = r.update("探索度 87%", now=0.6)
    assert got.scene_id == "BAG_INVENTORY", "累计 0.6s 仍不够"

    # 持续够久才切；且切到的是回落值 WORLD（单弱词不定性）
    got = r.update("探索度 87%", now=1.2)
    assert got.scene_id == "WORLD", "持续超过 N 秒才切；终点是回落值"


@pytest.mark.unit
def test_refiner__sustained_map_keyword_does_switch_to_map() -> None:
    """真的定性时，切过去的就是那个具体场景。

    ⚠️ 注意必须**喂多帧**：去抖靠的是「持续观测到多久」，
    一次调用即使 now 很大也只是把候选刚挂上，计时从那一刻才开始 ——
    这正是它该有的行为（不能因为某一帧偶然读全了就立刻切）。
    """
    r = SceneOcrRefiner(n_seconds=1.0)
    r.update("背包 道具", now=0.0)
    for now in (0.1, 0.5, 0.9, 1.2):
        got = r.update("地图 传送锚点", now=now)
    assert got.scene_id == "MAP_BIG"


@pytest.mark.unit
def test_refiner__pending_resets_when_candidate_flips() -> None:
    """候选换了，计时必须**从头来**。"""
    r = SceneOcrRefiner(n_seconds=1.0)
    r.update("背包 道具", now=0.0)
    r.update("地图 传送锚点", now=0.2)      # 候选 = MAP_BIG
    got = r.update("商店 购买", now=0.5)      # 候选换成 SHOP -> 计时重来
    assert got.scene_id == "BAG_INVENTORY"
    for now in (0.6, 1.0, 1.6):
        got = r.update("商店 购买", now=now)
    assert got.scene_id == "SHOP", "SHOP 从 0.5 攒够 1.0s（即 1.5 之后）才该切"


@pytest.mark.unit
def test_refiner__reset_returns_to_uninitialized() -> None:
    r = SceneOcrRefiner(n_seconds=1.0)
    r.update("背包 道具", now=0.0)
    r.reset()
    assert r.update("地图 传送锚点", now=1.0).scene_id == "MAP_BIG"


@pytest.mark.unit
def test_refiner__rejects_negative_debounce() -> None:
    with pytest.raises(ValueError):
        SceneOcrRefiner(n_seconds=-1.0)


# ==================================================================
# 准确率
# ==================================================================

@pytest.mark.unit
def test_ocr_scene__accuracy_on_sample_set() -> None:
    """量出当前样本集上的准确率。

    正样本 10 条必须全对；弱词 4 条必须**全部回落**（宁可判不出来）。
    两个数都写死在断言里 —— 将来改关键词，这里会立刻反映是进步还是退步。
    """
    ok = sum(1 for t, w in SAMPLES_POS
             if classify_scene_from_text(t).scene_id == w)
    assert ok == len(SAMPLES_POS), f"正样本准确率 {ok}/{len(SAMPLES_POS)}"

    safe = sum(1 for t, w in SAMPLES_WEAK
               if classify_scene_from_text(t, fallback="WORLD").scene_id == w)
    assert safe == len(SAMPLES_WEAK), f"弱词防误判 {safe}/{len(SAMPLES_WEAK)}"


@pytest.mark.unit
def test_ocr_scene__exploration_percent_alone_is_weak() -> None:
    """「探索度」在大地图右上角**常驻**，单靠它定性会把野外误判成地图。

    这是这批关键词里最容易出事的一个 —— 单列一条测试盯着。
    """
    g = classify_scene_from_text("探索度 87%", fallback="WORLD")
    assert g.confidence < 0.55, "单个「探索度」不该够格定性"
    g2 = classify_scene_from_text("地图 探索度 87%", fallback="WORLD")
    assert g2.scene_id == "MAP_BIG", "叠上「地图」才定性"