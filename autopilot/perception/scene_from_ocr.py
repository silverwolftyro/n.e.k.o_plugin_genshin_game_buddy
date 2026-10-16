"""从 OCR 文字推断场景标签（2026-10-11 新增，**默认不接入主流程**）。

★ 为什么值得做
-------------
路线图（``recognition/notes/signal_integration_roadmap.md``）把「场景识别
多值化」列为解锁 **10 条**规则的最大单一来源，同时标注难度 = 复杂，因为
「需要一个训练好的分类器」。

但那条路不是唯一解。场景本来就可以从 **OCR 文字**里读出来 ——
而且**已经在读了**：builder 每一帧都跑 OCR，文字就在手里，只是被
``_resolve_scene_binary`` 直接丢掉了，只留下二值的 WORLD / UNKNOWN。

所以这里做一个**纯函数**的后处理器：给它一段 OCR 文字，它还你一个场景标签。
不依赖模型、不需要训练、不碰主流程 —— 先单独验证准确率。

★ 为什么**默认不接入**
-------------------
误判的代价不对称：

    把 WORLD 误判成 BAG_INVENTORY -> 引擎在野外疯狂按 B 键开背包
    把 BAG_INVENTORY 误判成 WORLD   -> 引擎在大世界乱按 F

第二种几乎无害，第一种很难排查（症状是「AI 一直开菜单」）。
所以接进主流程之前必须先有真实样本量出准确率，并给置信度设阈值。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: 场景 -> 关键词。**全部小写**（匹配前会 normalize）。
#:
#: ★ 关键词来源
#:   取自原神 PC 界面上**真实会出现**的字，不是猜的：
#:     背包页标题是「背包」，顶部页签是「道具 / 圣遗物 / 武器 / 角色」
#:     地图界面是「地图」，右上角是原粹树脂与传送锚点
#:     对话框底部是「继续」/「跳过」，选项列表是「选项」
#:     商店是「商店」，旁边有「购买」
#:     结算界面是「获得」+「离开」
SCENE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "BAG_INVENTORY": ("背包", "道具", "圣遗物", "武器", "行囊"),
    "MAP_BIG": ("地图", "传送锚点", "七天神像", "世界", "探索度"),
    "COMBAT_RESULT": ("获得", "离开", "再来一次", "奖励"),
    "SHOP": ("商店", "购买", "兑换"),
    "MAIN_MENU": ("派蒙", "设置", "队伍", "好友"),
    "DIALOG": ("继续", "跳过", "选项"),
}

#: 权重：出现即判定为该场景的关键词。**长词比短词可靠**
#: （「圣遗物」比「背包」更不会误伤；「传送锚点」比「世界」可靠得多）。
_STRONG = {"圣遗物", "传送锚点", "七天神像", "再来一次", "行囊"}

#: 二值检测器的两个取值 —— 判不出具体场景时的回落目标。
FALLBACKS = ("WORLD", "UNKNOWN")


def normalize(text: str) -> str:
    """去掉 OCR 常见的噪声：空白、全角空格、换行。"""
    return re.sub(r"[\s　]+", "", str(text or ""))


@dataclass(frozen=True)
class SceneGuess:
    """一次推断的结果。**永远带证据**，便于回溯为什么这么判。"""

    scene_id: str
    confidence: float
    evidence: tuple[str, ...]
    matched: str = ""


def classify_scene_from_text(
    text: str,
    *,
    fallback: str = "WORLD",
    min_confidence: float = 0.55,
) -> SceneGuess:
    """从一段 OCR 文字里推断场景。

    :param fallback: 判不出来时返回什么（通常是二值检测器的结果）
    :param min_confidence: 低于它就退回 fallback —— **宁可保守**，
        因为「把野外误判成背包」的代价远大于「判不出来」。
    """
    norm = normalize(text)
    if not norm:
        return SceneGuess(fallback, 0.0, ("ocr:空文本",), "")

    hits: list[tuple[str, str]] = []      # (scene, keyword)
    for scene, words in SCENE_KEYWORDS.items():
        for w in words:
            if normalize(w) in norm:
                hits.append((scene, w))

    if not hits:
        return SceneGuess(fallback, 0.0, ("ocr:无关键词",), "")

    # 同场景内多个关键词叠加 —— 两个独立关键词同时出现，比一个可信
    per_scene: dict[str, list[str]] = {}
    for scene, w in hits:
        per_scene.setdefault(scene, []).append(w)

    def _score(words: list[str]) -> float:
        s = 0.0
        for w in words:
            s += 0.55 if w in _STRONG else 0.35
        return min(0.98, s)

    scored = [(scene, _score(words), words)
              for scene, words in per_scene.items()]
    scored.sort(key=lambda t: (-t[1], t[0]))          # 分数高者优先，同分按名字定序
    scene, score, words = scored[0]

    if score < min_confidence:
        return SceneGuess(fallback, score,
                          (f"ocr:最高分 {scene} 仅 {score:.2f} < {min_confidence}",
                           f"命中={words}"), words[0])
    return SceneGuess(scene, score,
                      (f"ocr:命中 {words}",), words[0])


class SceneOcrRefiner:
    """带去抖的包装 —— **不要**每帧都改判。

    单帧 OCR 漏字很常见（「背包」被认成「背句」）。直接逐帧用结果会让
    场景在 WORLD / BAG_INVENTORY 之间**疯狂横跳**，而 traits 的
    ``allowed_scenes`` 是按场景过滤的 —— 横跳的后果是规则时而能触发、
    时而不能，看起来像随机失效。

    去抖参数沿用 ``SceneDetector`` 的形状（连续 N 秒才算切换），
    保持两个检测器的语义一致。
    """

    def __init__(self, n_seconds: float = 1.5,
                 min_confidence: float = 0.55) -> None:
        if n_seconds < 0:
            raise ValueError("n_seconds 不能为负")
        self._n = float(n_seconds)
        self._min_conf = float(min_confidence)
        self._stable: str | None = None
        self._pending: str | None = None
        self._pending_since: float = 0.0

    def reset(self) -> None:
        self._stable = None
        self._pending = None
        self._pending_since = 0.0

    def update(self, text: str, *, fallback: str = "WORLD",
               now: float | None = None) -> SceneGuess:
        """喂一段 OCR 文字，返回（去抖后的）场景。"""
        t = time.monotonic() if now is None else now
        g = classify_scene_from_text(text, fallback=fallback,
                                     min_confidence=self._min_conf)

        if self._stable is None:
            self._stable = g.scene_id
            return g
        if g.scene_id == self._stable:
            self._pending = None
            return g

        if self._pending != g.scene_id:
            self._pending = g.scene_id
            self._pending_since = t
            return SceneGuess(self._stable, g.confidence,
                              (f"ocr:候选 {g.scene_id} 计时中",) + g.evidence,
                              g.matched)

        if (t - self._pending_since) >= self._n:
            self._stable = g.scene_id
            self._pending = None
            return SceneGuess(self._stable, g.confidence,
                              (f"ocr:切到 {g.scene_id}",) + g.evidence, g.matched)
        return SceneGuess(self._stable, g.confidence,
                          (f"ocr:去抖中 {self._pending}",), g.matched)


__all__ = [
    "FALLBACKS",
    "SCENE_KEYWORDS",
    "SceneGuess",
    "SceneOcrRefiner",
    "classify_scene_from_text",
    "normalize",
]