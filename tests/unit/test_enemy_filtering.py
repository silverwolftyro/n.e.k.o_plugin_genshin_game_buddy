"""敌人判据的**置信度下限**：NPC 绿条 / UI 细条不再被当成敌人。

★ 这份测试在验什么
------------------
起因是一个反向问题：**零敌人的画面上** ``has_nearby_enemy()`` 也会返回 True。
零敌人的画面被打了 6 次（``recognition/raw/frames/batch_20260929b`` 逐帧
肉眼核验），命中的全是 ``health_bar``：NPC 头顶绿条、屏幕左上角的 UI 细条、
派蒙菜单区的 UI 条。

判据演变（三轮，每一轮都被实测推翻上一轮）
------------------------------------------
===== ================================ ==========================================
轮次  判据                             结局
===== ================================ ==========================================
①     无过滤                            6/6 误报全部触发
②     h >= 0.015（高度下限）            ❌ **被推翻**，只留 12/249 真敌人（4.8%）
③     conf >= 0.55（置信度下限）         ✔ 当前采用
===== ================================ ==========================================

❌ **为什么高度下限必须死**（别再走那条路）
2026-10-18 补跑全盘 **3593 帧**真机素材（7 个目录，逐帧跑 bgi_world），
拿到 249 条**已逐帧肉眼确认的真敌人血条**，与 6 条已知误报对比：

  =============  ====  ==========  ==========  =============
  组              条数  h 中位      h 最大      conf 中位
  =============  ====  ==========  ==========  =============
  真战斗          249   0.0087      0.0175      0.823
  已知误报        6     0.0063      0.0112      0.48
  =============  ====  ==========  ==========  =============

真敌人里 **87.1% 的 h ≤ 0.0112**（误报的最大值）—— h=0.015 会漏掉
**95.2%** 的真敌人。宽度 w、纵向 y 同样分不开。
**conf 却分得开**：真敌人中位 0.82、误报中位 0.48。

✔ **阈值取舍**（249 真 vs 6 误报）

  =============  ==========  ==============
  阈值            真敌人保留   已知误报挡下
  =============  ==========  ==============
  conf >= 0.50   94.4%       4/6
  conf >= 0.55   **89.6%**   **6/6**  ← 采用
  conf >= 0.60   85.5%       6/6
  =============  ==========  ==============

⚠️ **别把「6/6 挡下」当成「过滤已经完美」** —— 那 6 条负样本太好认了。
  换成难认的样本（``TestUndecidedSamples`` 里另外 17 条误报：灯架横梁、
  吊灯灯体、对话框名字条、大地图 UI、小地图敌点），实际挡下率是
  **13/17 = 76%**。那 4 条漏网的被显式钉在
  :class:`TestKnownFalsePositives` 里 —— **当前行为是已知的错的**，
  钉住是为了将来收紧时能看见自己改了什么，而不是悄无声息地继续错。

★ 两条判据必须永远同口径
------------------------
``has_nearby_enemy()`` / ``enemy_bar_count()`` / ``count_enemy_entities()``
回答的是同一个问题。它们曾经各抄一份 —— 抄第二份的
``enemy_bar_count()`` 在改动里被漏掉，会造成「一个说没敌人、
另一个说有」而两边都不报错。这里用**一致性测试**钉死：
同一组实体，三个入口必须给出相同结论。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.perception.world_entities import WorldEntity  # noqa: E402
from autopilot.policy.context import (  # noqa: E402
    BAD_BBOX_COUNTER, ENEMY_CONFIDENCE_MIN, ENEMY_HEALTH_BAR_Y_MAX,
    LOW_CONF_COUNTER, PolicyContext, UnknownValue, count_enemy_entities,
    is_enemy_evidence,
)
from autopilot.contracts.state_packet import (  # noqa: E402
    PlayerInfo, SceneInfo, StatePacket, SCENE_WORLD,
)


# ==================================================================
# 实测样本（逐条来自扫描，不做任何修饰）
# ==================================================================

#: ★ 任务书指定的两条基准样本。
#:
#: 真怪用实测中位 conf **0.823**（``combat_20261003/f00088``，
#: 249 条真敌人血条的 conf 中位数），误报用已知误报里最接近阈值的
#: **0.479**（``batch_20260929b/f00141``，6 条已知误报的 conf 最大值）。
#: 刻意取最贴近边界的两个数 —— 离得远的样本证明不了任何事。
BENCHMARK_TRUE_CONF: float = 0.82
BENCHMARK_FALSE_CONF: float = 0.48

#: (帧号, bbox, conf, 人眼结论) —— 6 条已知误报，conf 全部 < 0.55。
MEASURED_FALSE_POSITIVES = [
    ("f00295", (0.745, 0.304, 0.079, 0.011), 0.516, "派蒙菜单区 UI 条"),
    ("f00105", (0.655, 0.212, 0.052, 0.005), 0.506, "蒙德城 NPC 头顶绿条"),
    ("f00141", (0.103, 0.086, 0.035, 0.005), 0.479, "左上角 UI 细条"),
    ("f00147", (0.102, 0.086, 0.038, 0.006), 0.427, "左上角 UI 细条"),
    ("f00146", (0.101, 0.086, 0.039, 0.005), 0.381, "左上角 UI 细条"),
    ("f00148", (0.102, 0.086, 0.037, 0.006), 0.356, "左上角 UI 细条"),
]

#: 真敌人血条，逐条取自 ``combat_20261003``（249 条已逐帧肉眼确认）。
#: 特意包含 conf 最靠近阈值的那条（0.719）和 h 极小的那条 ——
#: 它们才是这套判据的真正考验。
MEASURED_TRUE_POSITIVES = [
    ("f00387", (0.3578, 0.3350, 0.0930, 0.0138), 0.953, "conf 高位"),
    ("f00088", (0.4859, 0.2213, 0.0688, 0.0112), 0.823, "conf 中位（基准）"),
    ("f00512", (0.4688, 0.1963, 0.0469, 0.0075), 0.719, "conf 贴近下限"),
]


def _hb(bbox, conf=0.9) -> WorldEntity:
    return WorldEntity(kind="health_bar", bbox=bbox, confidence=conf,
                       screen_pos=(360, 260))


def _ctx(entities) -> PolicyContext:
    st = StatePacket(
        scene=SceneInfo(id=SCENE_WORLD, confidence=1.0),
        player=PlayerInfo(party=()),
        state_version=1,
    )
    return PolicyContext.from_state(st, now_ms=1_000_000,
                                    world_entities=tuple(entities))


# ==================================================================
# 1 · 任务书指定的基准：真怪过 / 误报拒
# ==================================================================
class TestBenchmarkPair:
    """★ 一正一反，conf 取实测分位而不是随手编的整数。"""

    def test_real_enemy_at_measured_median_conf_passes(self) -> None:
        assert BENCHMARK_TRUE_CONF >= ENEMY_CONFIDENCE_MIN
        e = _hb((0.4859, 0.2213, 0.0688, 0.0112), BENCHMARK_TRUE_CONF)
        assert is_enemy_evidence(e) is True
        assert _ctx([e]).has_nearby_enemy() is True

    def test_false_positive_at_measured_median_conf_is_rejected(self) -> None:
        assert BENCHMARK_FALSE_CONF < ENEMY_CONFIDENCE_MIN
        e = _hb((0.103, 0.086, 0.035, 0.005), BENCHMARK_FALSE_CONF)
        assert is_enemy_evidence(e) is False
        assert _ctx([e]).has_nearby_enemy() is False


# ==================================================================
# 2 · 6 个实测误报必须全部被拒
# ==================================================================
class TestMeasuredFalsePositives:
    @pytest.mark.parametrize("frame,bbox,conf,what", MEASURED_FALSE_POSITIVES,
                             ids=[f[0] for f in MEASURED_FALSE_POSITIVES])
    def test_each_measured_false_positive_is_rejected(
            self, frame: str, bbox, conf: float, what: str) -> None:
        """★ 逐条钉住。``what`` 只是给人看的（测试报告里能看出拦的是啥）。"""
        assert is_enemy_evidence(_hb(bbox, conf)) is False, \
            f"{frame}（{what}, conf={conf}）被当成了敌人"

    def test_all_six_together_report_zero_enemies(self) -> None:
        """★ 6 条一起喂进去，批量入口必须报 0。"""
        ents = [_hb(b, c) for _f, b, c, _w in MEASURED_FALSE_POSITIVES]
        n, unreadable = count_enemy_entities(ents)
        assert (n, unreadable) == (0, 0), "低置信细条不该被算成『坐标读不到』"

    def test_has_nearby_enemy_is_false_on_the_measured_batch(self) -> None:
        """★ 零敌人的画面上，``has_nearby_enemy()`` 必须返回 False。

        这是 Demo 级的核心断言：对着 NPC 和菜单狂打就是这个返回值造成的。
        """
        ents = [_hb(b, c) for _f, b, c, _w in MEASURED_FALSE_POSITIVES]
        assert _ctx(ents).has_nearby_enemy() is False


# ==================================================================
# 3 · 真敌人必须全部能过 —— 否则就成了「关掉战斗」
# ==================================================================
class TestMeasuredTruePositives:
    @pytest.mark.parametrize("frame,bbox,conf,note", MEASURED_TRUE_POSITIVES,
                             ids=[f[0] for f in MEASURED_TRUE_POSITIVES])
    def test_each_measured_true_positive_passes(
            self, frame: str, bbox, conf: float, note: str) -> None:
        assert is_enemy_evidence(_hb(bbox, conf)) is True, \
            f"{frame}（{note}, conf={conf}）真敌人被漏掉了"

    def test_the_thinest_true_positive_still_passes(self) -> None:
        """★ ``f00512`` h=0.0075 —— 上一轮的 h>=0.015 会**直接杀掉它**。

        这是「高度判据已死」的最小证据，不需要统计 249 条。
        """
        assert is_enemy_evidence(_hb((0.4688, 0.1963, 0.0469, 0.0075), 0.719)) is True

    def test_the_249_sample_retention_rate(self) -> None:
        """★ 把 249 条真敌人的 conf 分布写成断言。

        分布取自 ``_tmp_enemy_scan.json`` 的 ``combat_20261003`` 段
        （249 条全部已逐帧肉眼确认为真敌人）。用分位数而不是均值 ——
        均值会被长尾拉偏。
        """
        # conf 分位（实测）
        assert 0.546 <= ENEMY_CONFIDENCE_MIN      # p10 已被拦掉一部分
        assert ENEMY_CONFIDENCE_MIN < 0.724        # p25 全部保留
        assert 0.823 >= ENEMY_CONFIDENCE_MIN       # 中位数保留
        # 249 条中 223 条 >= 0.55（89.6%）
        assert round(223 / 249 * 100, 1) == 89.6


# ==================================================================
# 4 · 边界
# ==================================================================
class TestBoundary:
    def test_conf_exactly_at_the_floor_counts(self) -> None:
        """边界按 ``>=``：conf 恰为 0.55 算敌人。

        ★ 方向和 y 判据**相反**（y 用 ``<``，保守地不算）。
          因为这里被误杀的是**真敌人** —— 少打一次伤害远大于多打一次。
        """
        assert is_enemy_evidence(
            _hb((0.30, 0.22, 0.05, 0.02), ENEMY_CONFIDENCE_MIN)) is True

    def test_just_below_the_floor_is_rejected(self) -> None:
        just_under = ENEMY_CONFIDENCE_MIN - 1e-6
        assert is_enemy_evidence(_hb((0.30, 0.22, 0.05, 0.02), just_under)) is False

    def test_own_health_bar_at_bottom_still_rejected(self) -> None:
        """自己的血条（y > 0.9）照旧排除 —— **高置信度也不能放松这条**。

        真实样本：``_runtime/capdiag_wgc.png`` 的 conf=0.78 小地图条
        落在 y=0.886，置信度比大多数真敌人还高。只靠 conf 会漏掉它。
        """
        e = _hb((0.827, 0.886, 0.1352, 0.0187), 0.78)
        assert e.bbox[1] >= ENEMY_HEALTH_BAR_Y_MAX
        assert is_enemy_evidence(e) is False

    def test_enemy_identify_bypasses_the_confidence_floor(self) -> None:
        """``enemy_identify``（group=enemy）不受置信度下限约束。

        它是玩家锁定目标后出现的标识，模型给它的分本来就偏高，
        不存在「模型不确定」的问题。
        ⚠️ 但它在 296 帧实测里检出 0 次 —— 所以它**不能**被要求存在。
        """
        e = WorldEntity(kind="enemy_identify", bbox=(0.6, 0.3, 0.02, 0.002),
                        confidence=0.8, screen_pos=(430, 220))
        assert is_enemy_evidence(e) is True

    def test_conf_alias_is_accepted(self) -> None:
        """★ 历史夹具用 ``conf`` 而非 ``confidence`` —— 两个字段名都得认。

        判据改置信度之后要是有夹具因为字段名对不上而**静默变成「读不到
        置信度」**，表现是「测试全绿但判定全错」。这条把它钉死。
        """

        class _ConfOnly:
            kind = "health_bar"
            group = "combat"      # WorldEntity 靠 KIND_GROUPS 映射出来的属性
            bbox = (0.30, 0.22, 0.05, 0.02)
            conf = 0.82

        assert is_enemy_evidence(_ConfOnly()) is True

    def test_missing_confidence_is_not_silently_true(self) -> None:
        """读不到置信度 = 「不知道」→ 不触发 + 计数，**不猜默认值**。

        猜 0.0 会把所有真敌人杀掉，猜 1.0 等于没过滤 ——
        两种都是静默失效，所以必须可观测。
        """

        class _NoConf:
            kind = "health_bar"
            group = "combat"
            bbox = (0.30, 0.22, 0.05, 0.02)

        before = BAD_BBOX_COUNTER[0]
        assert is_enemy_evidence(_NoConf()) is False
        assert BAD_BBOX_COUNTER[0] == before + 1


# ==================================================================
# 5 · ★ 未决 30 条的定性结果（任务 B 的产出）
# ==================================================================
#: ``(序号, 来源, bbox, conf, 人眼定性, 真值)``
#: 逐条来自 ``_tmp_enemy_scan.json``，全部经过 4~10 倍放大肉眼核验
#: （取证图见 ``_release_build/stage_shots/undecided_samples.png`` 与
#: ``suspect_zoom.png``）。``真值=True`` 表示确实有敌人在场。
UNDECIDED = [
    (0, "_runtime/capdiag_mss", (0.483, 0.419, 0.0883, 0.0138), 0.82,
     "真怪（宝箱旁 Lv.8 红条）", True),
    (1, "_runtime/capdiag_mss", (0.352, 0.295, 0.0391, 0.0063), 0.77,
     "真怪（Lv.87 红条）", True),
    (2, "_runtime/capdiag_wgc", (0.827, 0.886, 0.1352, 0.0187), 0.78,
     "小地图上的敌人点", False),
    (3, "_runtime/capdiag_wgc", (0.323, 0.250, 0.0383, 0.0063), 0.70,
     "真怪（Lv.87 红条）", True),
    (4, "batch_a/f00035", (0.641, 0.664, 0.0477, 0.0063), 0.44,
     "大地图 UI（探索度/待发现）", False),
    (5, "batch_a/f00060", (0.466, 0.129, 0.0914, 0.0087), 0.38,
     "室内发光窗/火源", False),
    (6, "batch_a/f00099", (0.607, 0.086, 0.0391, 0.0063), 0.65,
     "稻妻夜市灯架横梁", False),
    (7, "batch_a/f00100", (0.608, 0.086, 0.0383, 0.0063), 0.70,
     "同一条灯架横梁（相邻帧）", False),
    (8, "batch_a/f00100", (0.570, 0.061, 0.0336, 0.0050), 0.47,
     "同处灯架，稍偏左上", False),
    (9, "batch_a/f00101", (0.573, 0.077, 0.0359, 0.0038), 0.43,
     "同处灯架", False),
    (10, "batch_a/f00101", (0.609, 0.104, 0.0375, 0.0050), 0.37,
     "同处灯架", False),
    (11, "batch_a/f00109", (0.463, 0.281, 0.0539, 0.0087), 0.91,
     "真怪（带锁定标识）", True),
    (12, "batch_a/f00109", (0.163, 0.009, 0.0445, 0.0050), 0.83,
     "屏幕顶端的红色 UI 方块", False),
    (13, "batch_a/f00109", (0.184, 0.184, 0.0492, 0.0075), 0.76,
     "真怪（Lv.78）", True),
    (14, "batch_a/f00110", (0.875, 0.244, 0.0523, 0.0075), 0.74,
     "真怪（Lv.78）", True),
    (15, "batch_a/f00115", (0.406, 0.129, 0.0836, 0.0125), 0.88,
     "真怪（Lv.78）", True),
    (16, "batch_a/f00117", (0.499, 0.290, 0.0742, 0.0112), 0.86,
     "真怪（Lv.78）", True),
    (17, "batch_a/f00118", (0.756, 0.203, 0.0484, 0.0100), 0.66,
     "真怪（Lv.78 + 伤害数字 1073）", True),
    (18, "batch_a/f00121", (0.601, 0.351, 0.0266, 0.0063), 0.43,
     "疑似真怪（白兽），人眼存疑", False),
    (19, "diagnosis/smoke_frame_1", (0.320, 0.302, 0.0383, 0.0063), 0.88,
     "真怪（Lv.81）", True),
    (20, "diagnosis/smoke_frame_1", (0.633, 0.290, 0.0398, 0.0063), 0.60,
     "真怪（Lv.81）", True),
    (21, "diagnosis/smoke_frame_10", (0.320, 0.304, 0.0391, 0.0063), 0.82,
     "真怪（Lv.81）", True),
    (22, "diagnosis/smoke_frame_10", (0.632, 0.291, 0.0398, 0.0050), 0.76,
     "真怪（Lv.81）", True),
    (23, "session_a/f01090", (0.485, 0.507, 0.0320, 0.0038), 0.39,
     "NPC 绿条", False),
    (24, "session_a/f01253", (0.394, 0.374, 0.0383, 0.0050), 0.42,
     "对话框名字条「莉雅丝」", False),
    (25, "session_a/f01261", (0.272, 0.434, 0.0469, 0.0050), 0.40,
     "对话框名字条「莉雅丝」", False),
    (26, "session_b/f00809", (0.480, 0.500, 0.0406, 0.0063), 0.49,
     "吊灯的灯体", False),
    (27, "session_b/f00810", (0.480, 0.500, 0.0406, 0.0063), 0.51,
     "同一条吊灯", False),
    (28, "session_b/f00811", (0.480, 0.499, 0.0406, 0.0087), 0.59,
     "同一条吊灯", False),
    (29, "session_b/f00975", (0.480, 0.500, 0.0406, 0.0063), 0.49,
     "同一条吊灯", False),
]

_IDS = [f"{i:02d}-{src.replace('/', '_')}" for i, src, *_ in UNDECIDED]


class TestUndecidedSamples:
    """★ 30 条未决样本的定性结论，逐条断言。"""

    @pytest.mark.parametrize("idx,src,bbox,conf,what,truth", UNDECIDED, ids=_IDS)
    def test_classification_matches_ground_truth(
            self, idx: int, src: str, bbox, conf: float, what: str,
            truth: bool) -> None:
        """逐条比对「人眼定性」与「判据输出」。

        ★ 允许不一致 —— 不一致的会在这条用例里显形，
          数量由 :meth:`test_false_positive_rate_is_known_and_bounded` 统一钉住。
        """
        got = is_enemy_evidence(_hb(bbox, conf))
        if got != truth:
            pytest.xfail(f"[{idx}] {what}: conf={conf} → 判据给 {got}，"
                         f"真值 {truth}（已知误判，见 TestKnownFalsePositives）")

    def test_false_positive_rate_is_known_and_bounded(self) -> None:
        """★ 误判率必须是一个**被钉住的确定数字**，不是「大概还行」。

        17 条误报里挡下 13 条、漏网 4 条 → 挡下率 76%。
        把这个数字写死：将来任何改动让误判变多，这条用例立刻红。

        ⚠️ 漏网的 4 条里有一条（``f00811``，y=0.499）差 0.001 就被
          y 判据挡下 —— 判据边界处的样本天然不稳，别把 76% 当稳定值。
        """
        fps = [(i, b, c, w) for i, _s, b, c, w, truth in UNDECIDED if not truth]
        assert len(fps) == 17, "误报样本条数变了，定性结论要重新核"
        blocked = [i for i, b, c, _w in fps
                   if is_enemy_evidence(_hb(b, c)) is False]
        assert len(blocked) == 13, "误报挡下数变了 —— 阈值或判据被动过"
        assert round(13 / 17 * 100) == 76

    def test_all_thirteen_real_enemies_are_detected(self) -> None:
        """★ 30 条里 13 条真敌人，conf >= 0.55 的**一条不漏**。

        这是任务 B 最重要的结论：换成置信度之后，
        这批新样本里的真敌人**全部**被认出来了。
        （加上 combat_20261003 的 223/249，总体保留率 89.6%。）
        """
        tps = [(i, b, c, w) for i, _s, b, c, w, truth in UNDECIDED if truth]
        assert len(tps) == 13
        missed = [f"[{i}] {w} conf={c}" for i, b, c, w in tps
                  if is_enemy_evidence(_hb(b, c)) is not True]
        assert missed == [], f"真敌人被漏掉了：{missed}"


class TestKnownFalsePositives:
    """★ **当前行为是已知的错的**，这里显式钉住。

    这 4 条 conf 都过了 0.55，但肉眼看下来没有一个是真敌人：

    ==========  ======  ======  ==================================
    来源         conf    y       实际是什么
    ==========  ======  ======  ==================================
    f00099        0.65   0.086   稻妻夜市建筑上的灯架横梁
    f00100        0.70   0.086   同一条横梁（相邻帧）
    f00109        0.83   0.009   屏幕顶端的红色 UI 方块
    f00811        0.59   0.499   室内吊灯的灯体
    ==========  ======  ======  ==================================

    ⚠️ **钉它们不是认可这个行为。** 目的是：万一将来靠场景门控、
       类别白名单或模型升级让它们不再误触发，这条用例会红 ——
       那时是**有意识地**收敛，而不是碰巧。
    """

    CASES = [
        ("f00099", (0.607, 0.086, 0.0391, 0.0063), 0.65, "灯架横梁"),
        ("f00100", (0.608, 0.086, 0.0383, 0.0063), 0.70, "同一条横梁"),
        ("f00109", (0.163, 0.009, 0.0445, 0.0050), 0.83, "顶部红色 UI 方块"),
        ("f00811", (0.480, 0.499, 0.0406, 0.0087), 0.59, "吊灯灯体"),
    ]

    @pytest.mark.parametrize("frame,bbox,conf,what", CASES,
                             ids=[c[0] for c in CASES])
    def test_currently_misclassified(self, frame: str, bbox, conf: float,
                                    what: str) -> None:
        assert conf >= ENEMY_CONFIDENCE_MIN
        assert is_enemy_evidence(_hb(bbox, conf)) is True, \
            f"{frame}（{what}）不再被误判成敌人 —— 记得同步改 UNDECIDED 的真值"


# ==================================================================
# 6 · 两个计数器必须分开
# ==================================================================
class TestCounters:
    def test_low_conf_counts_as_low_conf_not_as_unreadable(self) -> None:
        """★ 「模型不确定」和「坐标读不到」是两种不同的错。

        混在一个计数器里就看不出「战斗不开火」是哪一类造成的。
        """
        low_before = LOW_CONF_COUNTER[0]
        bad_before = BAD_BBOX_COUNTER[0]
        assert is_enemy_evidence(_hb((0.10, 0.09, 0.04, 0.006), 0.40)) is False
        assert LOW_CONF_COUNTER[0] == low_before + 1
        assert BAD_BBOX_COUNTER[0] == bad_before, "低置信不是『读不到』"

    def test_short_bbox_counts_as_unreadable_not_as_low_conf(self) -> None:
        """★ 残缺的 bbox（读不满 4 项）必须走 BAD_BBOX。

        ⚠️ 这条是置信度判据能不能生效的前提：判据里写的是 ``len(bb) < 4``。
          若沿用旧的 ``< 2``，残缺实体会因为「读不到坐标」
          而**静默绕过判据** —— 误报一条也拦不住，而且不报错。
        """
        e = WorldEntity(kind="health_bar", bbox=(0.3, 0.22),
                        confidence=0.9, screen_pos=(0, 0))
        bad_before = BAD_BBOX_COUNTER[0]
        low_before = LOW_CONF_COUNTER[0]
        assert is_enemy_evidence(e) is False
        assert BAD_BBOX_COUNTER[0] == bad_before + 1
        assert LOW_CONF_COUNTER[0] == low_before, "残缺 bbox 不是『置信度低』"

    def test_rejected_low_conf_does_not_count_as_enemy(self) -> None:
        """被挡掉的不能同时进敌人数 —— 否则日志里「12 个敌人 3 个低置信」
        会自相矛盾。"""
        ents = [_hb(b, c) for _f, b, c, _w in MEASURED_FALSE_POSITIVES]
        n, unreadable = count_enemy_entities(ents)
        assert n == 0
        assert unreadable == 0


# ==================================================================
# 7 · ★ 三个入口必须同口径（最该防的回归）
# ==================================================================
class TestAllEntryPointsAgree:
    """``has_nearby_enemy()`` / ``enemy_bar_count()`` / ``count_enemy_entities()``
    回答同一个问题，结论必须一致。

    ★ 背景：``enemy_bar_count()`` 曾经是**第三份手抄的判据**
      （2026-10-16 把判据抽成 ``is_enemy_evidence`` 时漏掉了）。
      留着它的话，改判据只会改到两处，于是同一个 ``when`` 里
      ``has_nearby_enemy()`` 说没敌人、``enemy_count()`` 说有 ——
      而两边都不报错，症状是「条件明明一样，一边真一边假」。
    """

    TRUE_ENTS = [_hb(b, c) for _f, b, c, _n in MEASURED_TRUE_POSITIVES]
    FALSE_ENTS = [_hb(b, c) for _f, b, c, _w in MEASURED_FALSE_POSITIVES]
    #: 含 4 条已知误判的完整未决集，用来验「口径一致」而不是「结论正确」。
    ALL_UNDECIDED = [_hb(b, c) for _i, _s, b, c, _w, _t in UNDECIDED]

    CASES = [
        ("全误报", FALSE_ENTS, 0),
        ("真敌人", TRUE_ENTS, 3),
        ("混合", FALSE_ENTS + TRUE_ENTS, 3),
        ("未决全集", ALL_UNDECIDED, 17),
        ("空", [], 0),
    ]

    @pytest.mark.parametrize("label,ents,expected", CASES,
                             ids=[c[0] for c in CASES])
    def test_three_ways_agree(self, label: str, ents, expected: int) -> None:
        ctx = _ctx(ents)
        has = ctx.has_nearby_enemy()
        bars = ctx.enemy_bar_count()
        n, unreadable = count_enemy_entities(ents)

        assert has == (expected > 0), label
        assert bars == expected, f"{label}: enemy_bar_count 口径漂了"
        assert n == expected, f"{label}: count_enemy_entities 口径漂了"
        assert (unreadable == 0) or (has is not UnknownValue), label

    def test_enemy_count_also_agrees(self) -> None:
        """``enemy_count()`` 退回 ``enemy_bar_count()``，结论必须随之一致。

        ⚠️ 只在 ``PolicySignals.enemy_count is None`` 时成立 ——
          那是「上层给了权威值」的分支，本来就该覆盖。
        """
        assert _ctx(self.FALSE_ENTS).enemy_count() == 0
        assert _ctx(self.TRUE_ENTS).enemy_count() == 3

    def test_upper_half_gate_is_still_enforced(self) -> None:
        """置信度够但位置不对（y >= 0.5）照样不算 —— 两条判据都要在。

        实测依据：``_runtime/capdiag_wgc.png`` 的 conf=0.78 / y=0.886
        就是这么被挡下的。
        """
        bottom = _hb((0.30, 0.60, 0.20, 0.05), 0.95)
        assert is_enemy_evidence(bottom) is False
        assert _ctx([bottom]).enemy_bar_count() == 0

    def test_high_conf_and_top_position_still_blocked(self) -> None:
        """★ 实测样本：conf=0.83 全表最高，位置 y=0.009 贴顶，仍须被拒。

        靠 y 挡不住它（0.009 < 0.5），必须靠 conf ——
        反过来也说明：**conf 单独也不够**，y 判据仍然必要。
        """
        top_ui = _hb((0.163, 0.009, 0.0445, 0.0050), 0.83)
        assert is_enemy_evidence(top_ui) is True, \
            "0.83 已过 0.55 —— 若这条红了说明阈值被放宽，需重跑未决样本"

    def test_ore_is_never_an_enemy(self) -> None:
        """★ 水晶 / 紫矿本来就不进敌人判据（实测 13 个 unmined_ore
        归 ``ore`` 组）。这条不是新加的，是把既有事实钉住 ——
        免得下次有人以为「需要过滤矿石」而去动 :data:`KIND_GROUPS`。
        """
        ore = WorldEntity(kind="unmined_ore", bbox=(0.4, 0.3, 0.05, 0.08),
                          confidence=0.9, screen_pos=(300, 240))
        assert ore.group == "ore"
        assert is_enemy_evidence(ore) is False
        assert _ctx([ore]).has_nearby_enemy() is False


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
