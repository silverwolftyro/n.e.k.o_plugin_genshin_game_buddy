"""队伍栏角色名识别 —— **OCR 路线**。

★ 为什么是 OCR 而不是头像图像匹配（本模块存在的理由）
--------------------------------------------------------
原计划是拿 BetterGI 的 123 张角色头像提特征（pHash / DCT / 颜色直方图）
建一个 <1MB 的特征库，做模板匹配。**真机实测把它否掉了**：

``recognition/exports/benchmark/scripts/probe_avatar_match.py`` 用真机
``party_1..4`` 的裁切去匹配 BetterGI 的 ``characterimage`` 素材：

* 真机那一格是**圆形裁切的 95x87 小头像**（在 2304x1440 参考帧上），
  BetterGI 的是 64x64~79x79 的**满方块立绘**，另一套构图；
* 6 种描述子（pHash64 / DCT12 / HSV 直方图 / 8x8 网格 / DCT+HSV / 网格+HSV）
  的 top-1 与 top-2 相似度差**只有 0.004**；
* 同一 slot 在不同帧里 top-1 还会变（帧内稳定 0~2/4）。

那不是「识别得不准」，那是**随机噪声**。继续在特征库方向上调参，
只会得到一个看起来像样、真机上乱指的组件。

★ 真正可用的信号：游戏自己画的字
--------------------------------
大世界右侧队伍栏**每一格头像左边就印着角色名**（白字）：

    slot1 纳西妲    slot2 钟离    slot3 夜兰    slot4 丝柯克

rapidocr 在这种小图上都能读对（实测见
``recognition/exports/benchmark/scripts/eval_party_ocr.py``）。
更关键的是它**零第三方资产** —— 不分发任何 BetterGI 像素，
许可问题直接消失。

★ 形近字靠白名单纠，不靠调参
------------------------------
OCR 会把「妲」认成「姐」（实测稳定发生）。角色名是**封闭集合**，
所以在集合内取最小编辑距离是**消除歧义**，不是「凑准确率」：

    「纳西姐」不在任何角色名里 -> 唯一合法解就是「纳西妲」

白名单取自 **``terms/genshin/characters.toml``**（125 条，
由 ``tools/sync_mihoyo_terms.py`` 从 miao-plugin 生成）——
**自有 Apache-2.0 资产，与 BetterGI 无关**。

★ active（谁在场）**故意不做**
------------------------------
``PlayerInfo.active_member`` 需要知道当前操控的是谁。本批 296 帧
**无法标定**，理由逐条写在这里，别再猜一遍：

1. 296 帧是**同一个队伍、同一个在场角色**，没有正负样本对。
2. 4 格头像的**渲染尺寸完全一致**（``probe_party_active.py``），
   早期"slot4 更大"的目视结论是看错了，已用跨帧拼图推翻。
3. 数字键提示框的**不透明度**随 UI 淡入淡出剧烈波动：
   296 帧里 195 帧「最亮格 vs 次亮格」的分隔度 < 0.10，
   判据分不开（``probe_party_badge.py``）。
4. 左下角 HUD **没有元素/角色标识**，只有一个武器类型图标，
   而这批帧的 5 个采样里全是单手剑，零区分度。

所以 :func:`read_active` 恒返回 ``None``（UNKNOWN），且本模块**不**提供
猜测路径。要恢复它，先录一段**含角色切换**的素材：按 1/2/3/4 各切几次，
让「在场」这个标签有正负样本 —— 与
:mod:`autopilot.perception.detectors.party_avatar` 对死亡检测的冻结说明
是同一套处理方式。
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..roi_loader import Rect, load_rois
from .avatar_classifier import AvatarPrediction

logger = logging.getLogger(__name__)

#: 队伍栏槽位。与 ``rois.toml`` 的 ``[rois.party]`` 一致。
SLOTS: tuple[int, ...] = (1, 2, 3, 4)

#: 角色名 ROI 的键名。``rois.toml`` 的 ``[rois.party_name]`` 会被
#: :func:`roi_loader.load_rois` 拍平成 ``party_name_1..4``。
NAME_ROI_PREFIX = "party_name_"

#: 角色名白名单的 TOML（相对插件根）。**随插件包分发**，
#: ``build_engine.py`` 的 CHARACTER_DATA 会把它带进编译后的 exe。
_ROSTER_TOML = Path(__file__).resolve().parents[3] / "terms" / "genshin" / "characters.toml"

#: OCR 前的放大倍数。真机名字区只有 ~33x19 px（2304x1440 下），
#: rapidocr 的检测器在小图上容易漏检，放大 3 倍后实测稳定。
#: ⚠️ 实测放大 1/2/3/4 倍**耗时一样**（都是 ~1.3 s，见类文档），
#:    所以这里按「识别率更稳」而不是「更快」来选。
UPSCALE = 3

#: 白名单纠错的最大编辑距离。**超过它就是「读不出来」**，
#: 硬纠会把无关文本变成某个角色名 —— 那比留空危险得多。
#: 「纳西姐」->「纳西妲」距离 1；2 字名与 2 字名的距离普遍 >= 2。
MAX_EDIT_DIST = 1

#: 「压根没找到候选」时用的哨兵距离。角色名最长 5 字，真实距离上限
#: 就是 5，所以 99 足够且一眼可辨（不会跟真实距离混淆）。
_UNREACHABLE_DIST = 99


def max_dist_for(text: str) -> int:
    """这段 OCR 文本**允许**的纠错距离。

    ★ 为什么距离要随长度收紧（实测逼出来的）
    -----------------------------------------
    白名单里有 4 个**单字**角色名：``空`` / ``琴`` / ``荧`` / ``魈``
    （流浪者、优菈的武器、主角、仙人）。于是一字 OCR 结果
    （「塔」「西」「9」「二」）只要距离 1 就会被纠成它们 ——
    实测 7 条误识里 **5 条**是这一种。它不是偶发，是**系统性**的：
    一字文本的邻域覆盖了几千个常用字里的很大一块。

    而 3 字名的邻域小得多：「纳西姐」->「纳西妲」距离 1，
    是全表里唯一稳定发生的真实混淆。

    所以阈值取 ``len - 2``（再被 :data:`MAX_EDIT_DIST` 封顶）：

        「塔」（1 字）-> 允许 0  -> 只认精确匹配
        「夜兰」（2 字）-> 允许 0  -> 2 字名没有纠错空间
        「纳西姐」（3 字）-> 允许 1 -> 放行真实的形近字

    这不是调参凑准确率，是**按信息量定邻域**：
    同样的距离在 1 字空间和 3 字空间里含义完全不同。
    """
    return max(0, min(MAX_EDIT_DIST, len(text) - 2))


class CharacterIdError(RuntimeError):
    """角色名白名单不可用。**响亮失败** —— 静默退化成「不认角色」最难查。"""


@lru_cache(maxsize=1)
def load_roster() -> tuple[str, ...]:
    """读角色名白名单。进程内只读一次。

    :raises CharacterIdError: 文件不存在 / 语法错 / 一条名字都没有
    """
    p = _ROSTER_TOML
    try:
        payload = tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CharacterIdError(
            f"找不到角色名白名单：{p}\n"
            "它随插件包分发（build_engine.py 的 CHARACTER_DATA 负责带进 exe）。"
            "缺了它队伍栏识别会整体失效，所以这里直接抛，不静默降级。"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise CharacterIdError(f"{p} 语法错误：{exc}") from exc

    terms = payload.get("terms")
    if not isinstance(terms, dict) or not terms:
        raise CharacterIdError(f"{p} 里没有非空的 [terms] 表")

    names = tuple(sorted(k for k in terms if isinstance(k, str) and k.strip()))
    if not names:
        raise CharacterIdError(f"{p} 的 [terms] 里没有可用名字")
    logger.info("角色名白名单 %d 条（来自 %s）", len(names), p.name)
    return names


def _edit_distance(a: str, b: str) -> int:
    """经典 DP。名字都是 2~5 字，O(len^2) 完全够用。"""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def snap_to_roster(text: str, roster: tuple[str, ...] | None = None) -> tuple[str, int]:
    """把 OCR 文本纠到最近的角色名。

    :returns: ``(角色名, 编辑距离)``。**距离 > :data:`MAX_EDIT_DIST` 时
        返回 ``("", 距离)``** —— 即「认不出来」，而不是最像的那个。
        返回 ``""`` 是有意义的：上层据此把该槽标成 UNKNOWN，
        规则不触发；填一个猜出来的名字会让机器人对着错误角色放技能。
    """
    raw = (text or "").strip()
    if not raw:
        return "", 99
    names = roster if roster is not None else load_roster()
    if raw in names:                      # 快路径：本来就对
        return raw, 0
    # ⚠️ 阈值随文本长度收紧，见 :func:`max_dist_for`。用固定的
    #   ``MAX_EDIT_DIST`` 会让「塔」「西」这类一字噪声被纠成
    #   单字角色名「琴」「空」—— 实测 7 条误识里 5 条是这一种。
    limit = max_dist_for(raw)
    # ⚠️ 初值必须是「无穷大」而不是 ``limit + 1``：
    #   用后者会让返回值**被截断**在 limit+1 —— 「距离 2」和「距离 5」
    #   都报成同一个数。判定不受影响（都 > limit），
    #   但 ``SlotName.edit_dist`` 是诊断字段，报错会误导排查。
    best, best_d = "", _UNREACHABLE_DIST
    for name in names:
        # 长度剪枝：角色名 2~5 字，长度差超过 2 的不用算
        if abs(len(name) - len(raw)) > 2:
            continue
        d = _edit_distance(raw, name)
        if d < best_d:
            best, best_d = name, d
    if best_d > limit:
        return "", best_d
    return best, best_d


@dataclass(frozen=True)
class SlotName:
    """一个槽位当前的识别结果。"""

    slot: int
    name: str = ""          # 角色名；``""`` = UNKNOWN（读不出来）
    raw: str = ""           # OCR 原始文本（诊断用）
    edit_dist: int = 99     # 到白名单的编辑距离
    confidence: float = 0.0

    @property
    def known(self) -> bool:
        """是否认出了**一个具体**角色名。"""
        return bool(self.name)


class CharacterDetector:
    """队伍栏角色名识别器。

    ★ 为什么**每次只读一格**（``slots_per_call``）
    -----------------------------------------------
    单次 OCR 实测 **1.0~1.5 秒**（见
    ``recognition/exports/benchmark/scripts/eval_party_ocr.py``），
    是 200 ms tick 预算的 **5~7 倍**。所以这里做成**轮转**：
    每次调用只读 ``slots_per_call`` 格，读过的结果缓存下来，
    返回**当前已知全量快照**。队伍组成很少变，慢一点完全可以接受；
    一次顶掉 4 格则会让整条控制回路抖动。

    ⚠️ 这 1 秒**与图像大小无关**（实测放大 1/2/3/4 倍都是 ~1.3 s），
      是 rapidocr 每次调用的固定开销（检测 + 方向分类 + 识别三个模型，
      加上 DML provider 的调度），不是推理量的问题。
      也就是说**调小 :data:`UPSCALE` 省不到时间**，只影响小图上的识别率。

    ⚠️ 唯一实测有效的提速是关掉方向分类（``use_cls=False``，省 ~32%），
      但那是 :class:`~autopilot.perception.ocr.OcrEngine` 的**全局**配置，
      会连带影响传送点 OCR 等所有调用方 —— 那些图里可能有斜排文字。
      所以**没有关**。要提速得换 OCR 方案，不是调这里的参数。
    """

    def __init__(
        self,
        ocr: Any,
        *,
        roster: tuple[str, ...] | None = None,
        slots_per_call: int = 1,
        upscale: int = UPSCALE,
        rois: dict[str, Rect] | None = None,
        classifier: Any = None,
    ) -> None:
        if slots_per_call < 1:
            raise ValueError(f"slots_per_call 必须 >= 1，收到 {slots_per_call}")
        self._ocr = ocr
        self._roster = roster
        self._slots_per_call = slots_per_call
        self._upscale = upscale
        self._rois = rois if rois is not None else load_rois()
        #: 头像分类器（``detectors.avatar_classifier.AvatarClassifier``）。
        #: ``None`` = 不走降级路径（**默认**，实测准确率不过线时应当如此）。
        self._classifier = classifier
        self._cache: dict[int, SlotName] = {}
        self._next_slot = 0
        #: 上次 :meth:`update` 实际读了几格（0 = 全走了缓存）。
        self.last_read_count = 0

    # -- 内部 ---------------------------------------------------------

    def _name_rect(self, slot: int) -> Rect | None:
        return self._rois.get(f"{NAME_ROI_PREFIX}{slot}")

    def read_slot(self, frame: np.ndarray, slot: int) -> SlotName:
        """读**一个**槽位。纯读，不碰缓存。"""
        rect = self._name_rect(slot)
        if rect is None:
            logger.warning("rois.toml 里没有 %s%d，跳过", NAME_ROI_PREFIX, slot)
            return SlotName(slot=slot)
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise TypeError("frame 应为 (H, W, 3) 的 BGR 图像")
        crop = rect.crop(frame[:, :, :3])
        if crop is None or getattr(crop, "size", 0) == 0:
            return SlotName(slot=slot)

        k = max(1, int(self._upscale))
        big = _resize3(crop, k)
        raw, conf = self._ocr_text(big)
        name, dist = snap_to_roster(raw, self._roster)
        return SlotName(slot=slot, name=name, raw=raw, edit_dist=dist,
                        confidence=conf)

    def _ocr_text(self, img: np.ndarray) -> tuple[str, float]:
        """跑 OCR，拿到 ``(拼接文本, 平均置信度)``。

        ⚠️ ``self._ocr`` 既可以是 :class:`~autopilot.perception.ocr.OcrEngine`
           （``.read() -> OcrResult``），也可以是裸的
           ``RapidOCR``（``(boxes, elapse) -> (res, None)``）——
           后者是离线评测脚本用的 duck type，两种都支持是为了让
           :mod:`eval_party_ocr` 与生产走**同一段**文本拼接逻辑。

        ⚠️ 走 :class:`OcrEngine` 时**必须** ``count_failure=False``：
           队伍栏在切屏/开菜单时本来就没字，rapidocr 返回空是**正确
           结果**。若按失败计，连续 5 次之后 ``OcrEngine._available``
           变 False —— 整个 OCR 通道被判死，传送点识别一起废掉。
        """
        obj = self._ocr
        if hasattr(obj, "read"):
            try:
                r = obj.read(img, count_failure=False)
            except TypeError:
                # 鸭子类型的老 OcrEngine 没有这个参数；那时退回默认计数。
                r = obj.read(img)
            return (getattr(r, "text", "") or ""), float(getattr(r, "confidence", 0.0))

        res, _ = obj(img)
        if not res:
            return "", 0.0
        lines = [x[1] for x in res if len(x) > 1]
        scores = [float(x[2]) for x in res if len(x) > 2]
        return "".join(lines), (sum(scores) / len(scores) if scores else 0.0)

    # -- 对外 ---------------------------------------------------------

    def update(self, frame: np.ndarray) -> dict[int, SlotName]:
        """轮转读 ``slots_per_call`` 格，返回**缓存的全量快照**。

        ⚠️ 某一格读失败（UNKNOWN）**不清缓存** —— 队伍栏在切屏、
        打开菜单时会整体虚化，此时 OCR 全灭。若失败就清缓存，
        快照会从「已知的队伍」闪成空，规则跟着每帧抖动。
        所以：成功才更新，读不到就沿用上次的结果。
        """
        self.last_read_count = 0
        for _ in range(self._slots_per_call):
            slot = SLOTS[self._next_slot % len(SLOTS)]
            self._next_slot += 1
            try:
                res = self.read_slot(frame, slot)
            except Exception:  # noqa: BLE001 - 逐帧抖动不该掀翻整轮 tick
                logger.warning("槽位 %d 角色名识别异常，沿用上次结果", slot,
                               exc_info=True)
                continue
            if res.known:
                self._cache[slot] = res
            self.last_read_count += 1
        return dict(self._cache)

    def reset(self) -> None:
        """清空缓存。换队伍 / 切到别的游戏时调用。"""
        self._cache.clear()
        self._next_slot = 0

    # -- 降级路径：头像分类器（2026-10-18）---------------------------

    #: 头像 ROI 前缀（``rois.toml`` 的 ``[rois.party]``）。
    AVATAR_ROI_PREFIX = "party_"

    def classify_slot(self, frame: np.ndarray, slot: int,
                      *, min_confidence: float = 0.35):
        """用**头像**认角色。**降级路径**，只在 OCR 读不到时才用。

        :returns: :class:`~autopilot.perception.detectors.avatar_classifier.AvatarPrediction`
            ；没挂分类器 / 没 ROI / 置信度不够，都返回 ``known`` 为假的实例。

        ⚠️ **置信度闸是必需的，不是可选的**
          分类器**一定**会给出一个 argmax。没有阈值就等于每帧都输出
          一个名字，哪怕它 90% 不确定 —— 而**错名字比没有名字更糟**：
          ``onfield()`` 会拿着它去判连招，按错技能、切错人。
        """
        if getattr(self, "_classifier", None) is None:
            return AvatarPrediction()
        rect = self._rois.get(f"{self.AVATAR_ROI_PREFIX}{slot}")
        if rect is None:
            return AvatarPrediction()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise TypeError("frame 应为 (H, W, 3) 的 BGR 图像")
        crop = rect.crop(frame[:, :, :3])
        if crop is None or getattr(crop, "size", 0) == 0:
            return AvatarPrediction()
        pred = self._classifier.classify(crop)
        if pred.known and pred.confidence < min_confidence:
            logger.info("槽位 %d 头像分类置信度 %.2f < %.2f，按 UNKNOWN 处理",
                        slot, pred.confidence, min_confidence)
            return AvatarPrediction(confidence=pred.confidence, top5=pred.top5)
        return pred


def _resize3(img: np.ndarray, k: int) -> np.ndarray:
    """放大 k 倍。``INTER_CUBIC``：小图放大要平滑，双线性会糊掉笔画。"""
    import cv2  # noqa: PLC0415

    return cv2.resize(img, (img.shape[1] * k, img.shape[0] * k),
                      interpolation=cv2.INTER_CUBIC)


def read_active(names: dict[int, SlotName]) -> None:
    """当前在场角色 —— **恒返回 None**。

    保留这个函数是为了让「为什么不填 ``active``」有个**可以被调用、
    可以被测试**的出口，而不是散落在注释里的一句话。

    要实现它，先按模块 docstring 的第 4 条录一段**含角色切换**的素材。
    """
    return None


__all__ = [
    "MAX_EDIT_DIST",
    "NAME_ROI_PREFIX",
    "SLOTS",
    "UPSCALE",
    "CharacterDetector",
    "CharacterIdError",
    "SlotName",
    "load_roster",
    "max_dist_for",
    "read_active",
    "snap_to_roster",
]
