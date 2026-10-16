"""``key_z``（小道具）检测器 —— 模板匹配。

★ 为什么能用模板匹配，而 key_e 不能
---------------------------------
``key_z.png`` 是**跨角色不变**的那类图标：它属于角色（工具/弩箭）之外的
固定 UI 元素，所以同一个模板在所有队伍配置下都长得一样（实测跨帧分数
0.985 / 0.928，稳）。

而 ``key_e`` 是**角色技能**图标 —— 每个角色长得都不一样，拿单个模板去
匹配，遇到不同角色必然掉到 0.2 以下。所以 key_e 走的是另一条路
（只看明暗，不看图案），不在本文件里。

★ 状态语义
----------
    READY     分数 >= 0.90   图标与模板一致，小道具可用
    UNKNOWN   分数 <  0.50   ROI 里压根不是这个图标（画面变了 / 角色不同
                            / UI 被遮挡）—— **不能当成 CD**，那会让机器人
                            以为在等一个永远不会结束的冷却
    COOLDOWN  介于两者之间    图标还在，但被冷却遮罩压过，分数中等

★ 模块顶层不读文件
-----------------
模板在**首次调用时**才加载并缓存 —— import 这个包不能有任何 IO 副作用。
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ..paths import TEMPLATES_UI
from ..roi_loader import load_rois

logger = logging.getLogger(__name__)

ROI_NAME = "key_z"
TEMPLATE_NAME = "key_z.png"

#: 达到这个分数算 READY
SCORE_READY = 0.90
#: 低于这个分数算 UNKNOWN（不是 CD —— 见模块 docstring）
SCORE_UNKNOWN = 0.50

#: 模板缓存。``None`` = 还没加载过；``False`` = 试过但加载失败（别反复重试）
_template_cache: np.ndarray | None | bool = None
#: ROI 比模板小，已对称扩边补足。只警告一次。
_padded_warned = False


class KeyZState(Enum):
    """小道具的三态。"""

    READY = "ready"
    COOLDOWN = "cooldown"
    UNKNOWN = "unknown"


def _template() -> np.ndarray | None:
    """取模板灰度图。**懒加载 + 缓存**，import 期不做任何 IO。"""
    global _template_cache
    # ★ 必须显式判 None。用 ``if _template_cache`` 会踩
    #   "truth value of an array is ambiguous"。
    if _template_cache is not None:
        return _template_cache if _template_cache is not False else None
    p = Path(TEMPLATES_UI) / TEMPLATE_NAME
    from ..image_io import imread_unicode
    img = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error("模板加载失败：%s（key_z 检测将恒为 UNKNOWN）", p)
        _template_cache = False
        return None
    _template_cache = img
    logger.info("key_z 模板已加载：%s  %dx%d", p, img.shape[1], img.shape[0])
    return img


def _roi_crop(frame: np.ndarray) -> np.ndarray | None:
    rois = load_rois()
    r = rois.get(ROI_NAME)
    if r is None:
        raise KeyError(f"rois.toml 里没有 {ROI_NAME!r}")
    return r.crop(frame[:, :, :3])


def _crop_fitting(crop: np.ndarray, tw: int, th: int) -> np.ndarray | None:
    """把 ROI 裁剪调整到**恰好** ``tw x th``。

    ★ 实测：模板 99x92、ROI 98x93 —— 模板比 ROI 宽 1 像素（两边取整
      口径不同造成的）。``cv2.matchTemplate`` 要求模板不大于搜索图，
      否则直接抛 cv2.error。

    ★ 为什么用缩放而不是**对称扩边 + 复制边缘**
      扩边补出来的是 BORDER_REPLICATE 复制的边缘像素，与模板里的真实
      边缘对不上，会**系统性压低匹配分** —— 实测这一项就把最高分从
      0.98 附近压到了 0.8867，直接导致 READY 永远不触发。

      1 像素的缩放是几乎无损的，而且让搜索区正好等于模板的覆盖范围。
      差异超过 10% 才放弃（那说明不是取整误差，是 ROI 配错了）。
    """
    global _padded_warned
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    h, w = crop.shape[:2]
    if (h, w) == (th, tw):
        return crop
    if abs(tw - w) > 0.10 * w or abs(th - h) > 0.10 * h:
        logger.error(
            "模板 %dx%d 与 ROI %dx%d 相差超过 10%%，多半是 ROI 配错了，"
            "不做缩放直接判 UNKNOWN", tw, th, w, h)
        return None
    if not _padded_warned:
        logger.info("ROI %dx%d 与模板 %dx%d 不等，已缩放到模板尺寸", w, h, tw, th)
        _padded_warned = True
    interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_LINEAR
    return cv2.resize(crop, (tw, th), interpolation=interp)


def match_score(frame: np.ndarray) -> float:
    """模板匹配得分（0~1）。ROI/模板任一缺失时返回 0.0。"""
    tpl = _template()
    if tpl is None:
        return 0.0
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise TypeError("frame 应为 (H, W, 3) 的 BGR 图像")
    crop = _roi_crop(frame)
    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    th, tw = tpl.shape[:2]
    crop = _crop_fitting(crop, tw, th)
    if crop is None or crop.shape[0] < th or crop.shape[1] < tw:
        logger.error("模板 %dx%d 无法在 ROI 内搜索", tw, th)
        return 0.0
    res = cv2.matchTemplate(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                            tpl, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


def detect_key_z(frame: np.ndarray) -> tuple[KeyZState, float]:
    """检测小道具状态。

    :param frame: BGR 整帧
    :returns: ``(状态, 置信度)``，置信度就是模板匹配分数
    """
    score = match_score(frame)
    if score >= SCORE_READY:
        return KeyZState.READY, score
    if score < SCORE_UNKNOWN:
        return KeyZState.UNKNOWN, score
    return KeyZState.COOLDOWN, score


def reload_template() -> bool:
    """强制重载模板。**替换了 templates/ 下的图之后调它。**"""
    global _template_cache
    _template_cache = None
    return _template() is not None


__all__ = [
    "ROI_NAME",
    "SCORE_READY",
    "SCORE_UNKNOWN",
    "TEMPLATE_NAME",
    "KeyZState",
    "detect_key_z",
    "match_score",
    "reload_template",
]
