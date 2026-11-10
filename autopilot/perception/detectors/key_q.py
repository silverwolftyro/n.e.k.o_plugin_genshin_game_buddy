"""``key_q``（元素爆发 / 大招）就绪态检测器 —— **环形青色占比**。

★ 为什么不用模板匹配
--------------------
实测：拿 f00001 的 Q 裁切当模板全帧匹配，真正有 Q 的 15 帧 0.966~1.000、
其余全部 ≤0.413，间隔 0.55，干净得不像话。**但它不能用** ——
Q 图标是**角色专属**的（每角色一个爆发立绘），模板分数换角色就废。
这正是 :mod:`key_e` 早就写在模块文档里的那条教训（E 模板分数随角色在
0.914/0.218 之间跳）。拿单角色模板去判四角色，等于给三个角色判错。

★ 为什么用「环形」
------------------
要找**角色无关**的那个部分。就绪态的爆发按钮外面套着一圈
**青色发光环**（见 ``stage_shots/key_q_zoom.png`` 的 f00001），
它属于「就绪」这件事，不属于「这是谁」。所以取 ROI 外侧一圈做统计，
而不是像 ``key_e`` 那样取中心方块 —— 后者实测被背景主导（r = 0.88）。

★ 阈值是量出来的，不是拍的
--------------------------
296 帧遗留素材（``recognition/raw/frames/batch_20260929b``）上跑
``recognition/exports/benchmark/scripts/probe_key_q_ring.py``：

    就绪 15 帧   cyan_ratio = 0.822 ~ 0.824
    不就绪 281 帧 cyan_ratio ≤ 0.091
    ─────────────────────────────────
    间隔 0.731

    用**本模块**跑同一批素材（不是探针脚本）：15 命中 / 0 误报 / 0 漏报。

同一次运行里的**对照组**，用来证明「取环形」这一步是必要的::

    sat  就绪最小 137.9 / 不就绪最大 167.3  -> 不可分（夜空 sat 167）
    val  就绪最小 246.1 / 不就绪最大 254.5  -> 不可分
    b_r  就绪最小 132.9 / 不就绪最大  89.8  -> 可分，但余量只有 43

``cyan_ratio`` 同时用**蓝度差**和**亮度**两个闸门再取占比：
只卡蓝度会选中夜空（f00268 那个位置就是纯天空，sat 高达 167），
只卡亮度会选中亮沙地。两条一起卡，再取「占环形多少」，余量拉到 0.731。

🔴 这个检测器**没有 COOLDOWN** 这个状态 —— 这是故意的
-------------------------------------------------------
肉眼核查全部 296 帧（``key_q_sheet1..3.png`` / ``key_q_zoom.png``）：
那 281 个「不就绪」帧里，Q 位置分别是 HUD 淡出后的**空按钮圆盘**
（圆盘在、**里面没有图标**）、纯沙地、纯夜空、**大地图**、菜单页。
**一个「徽章在冷却」的帧都没有。**

所以报 ``COOLDOWN`` 等于宣称「我看见冷却读条了」——那是编的。
按现状，就绪和不就绪能分（负样本 282 个，够），冷却和淡出不能分。
好在**这两者对规则完全等价**：``q_ready()`` 都要返回假。
代价是：不就绪时报 ``UNKNOWN`` 而不是 ``False``，于是
``out.unknowns`` 里会出现 ``cd`` —— 这是**特性不是 bug**，
它让人一眼看出「大招读不出来」，而不是「系统在正常地等 CD」。

⚠️ 仍未验证的一件事（写在这里，别让它被忘掉）
------------------------------------------
标定样本**只有 1 个角色**（f00001 的骰子图标）。发光环按设计是
角色无关的，但**没有拿第二个角色的爆发按钮验过**。
第一次上真机如果发现「换角色就不准了」，先怀疑这里，
别去调 ``CYAN_RATIO_READY`` —— 余量有 0.731，调阈值救不了系统性错。

★ 模块顶层不读文件、不建 Tk
--------------------------
纯 numpy / cv2 计算，import 这个包没有任何副作用。
"""

from __future__ import annotations

import logging
from enum import Enum

import cv2
import numpy as np

from ..roi_loader import load_rois

logger = logging.getLogger(__name__)

ROI_NAME = "key_q"

#: 环形内外半径，占 ROI **短边**的比例。
#:
#: ★ 0.30~0.46 是量出来的：f00001 的青色发光环正好落在这段，
#:   再往里是蓝盘（角色相关，图标在里面），再往外是背景。
#:   内半径不能小于 0.30 —— 那会吃进角色图标，特征就不角色无关了。
R_IN, R_OUT = 0.30, 0.46

#: 「这是青光」的判据，**两条都要**。
#:
#: ⚠ 只卡蓝度（``CYAN_B_MIN``）会选中夜空 —— 实测 f00268/f00270/f00271
#:   那个位置就是纯蓝色夜空，sat 167、b_r 89.8，比就绪帧的 137.9 还高。
#: 只卡亮度会选中亮沙地。两条一起卡才有 0.731 的余量。
CYAN_B_MIN = 55.0      # B 至少比 R 高这么多（蓝）
CYAN_V_MIN = 150.0     # 且亮度至少这么多（排除暗蓝）

#: 环形里青光像素的占比 ≥ 此值 -> 就绪。
#:
#: ★ 就绪 [0.822, 0.824] / 不就绪 ≤ 0.091。0.45 取的是这段死区的中点。
#:   具体取多少**不重要** —— 中间 0.091~0.822 之间一个样本都没有，
#:   阈值放在哪都一样。写 0.45 只是因为它离两端都远。
CYAN_RATIO_READY = 0.45


class KeyQState(Enum):
    """大招就绪态。**刻意没有 COOLDOWN**，理由见模块文档的红框。"""

    READY = "ready"
    UNKNOWN = "unknown"


def annulus_mask(h: int, w: int) -> np.ndarray:
    """环形掩膜。用**相对**半径 -> 与分辨率无关。"""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    short = float(min(h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(yy - cy, xx - cx) / short
    return (d >= R_IN) & (d <= R_OUT)


def ring_stats(frame: np.ndarray) -> dict[str, float | bool]:
    """Q ROI 环形的统计量。供批量分析与调参用。

    :returns: ``{"cyan_ratio", "sat", "val", "b_r", "ok"}``

    ★ ``ok`` 与 :func:`key_e.center_stats` 同义 —— 「读到了」和
      「压根没读到」在最坏情况下数值完全一样（空 ROI 与纯黑都是 0），
      少了这个标记，:func:`detect_key_q` 会把「画面是空的」自信地报成
      「大招没就绪」，而不是 UNKNOWN。
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise TypeError("frame 应为 (H, W, 3) 的 BGR 图像")
    rect = load_rois().get(ROI_NAME)
    if rect is None:
        raise KeyError(f"rois.toml 里没有 {ROI_NAME!r}")
    crop = rect.crop(frame[:, :, :3])
    if crop is None or getattr(crop, "size", 0) == 0:
        return {"cyan_ratio": 0.0, "sat": 0.0, "val": 0.0,
                "b_r": 0.0, "ok": False}

    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return {"cyan_ratio": 0.0, "sat": 0.0, "val": 0.0,
                "b_r": 0.0, "ok": False}

    m = annulus_mask(h, w)
    b = crop[:, :, 0].astype(np.float32)
    r = crop[:, :, 2].astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    cyan = (b > r + CYAN_B_MIN) & (val >= CYAN_V_MIN)

    return {
        "cyan_ratio": float(cyan[m].mean()),
        "sat": float(hsv[:, :, 1][m].mean()),
        "val": float(val[m].mean()),
        "b_r": float((b - r)[m].mean()),
        "ok": True,
    }


def detect_key_q(frame: np.ndarray) -> tuple[KeyQState, float]:
    """检测大招就绪态。

    :param frame: BGR 整帧
    :returns: ``(状态, 置信度)``。置信度 = 环形青光占比。

    ★ 返回 UNKNOWN 时**不代表没就绪**，只代表「没看见亮青环」。
      可能的原因有三个且当前**分不开**：真的在冷却 / HUD 淡出了 /
      那个位置压根不是按钮（地图、菜单、天空）。
      对规则来说三者等价（不放大招），所以不必强行分。
    """
    st = ring_stats(frame)
    if not st["ok"]:
        return KeyQState.UNKNOWN, 0.0
    ratio = st["cyan_ratio"]
    if ratio >= CYAN_RATIO_READY:
        return KeyQState.READY, ratio
    return KeyQState.UNKNOWN, ratio


#: ``PartyMember.cd_ready`` 里大招对应的键。
#:
#: ★ 必须与 ``policy/context.py::q_ready()`` 里硬编码的 ``"burst"`` 一致。
#:   改这里不同步改那里 -> 填了值规则也读不到，症状是「接上了但还是 UNKNOWN」。
CD_KEY = "burst"


def read_burst_cd(frame: np.ndarray) -> tuple[bool | None, float, str]:
    """读大招就绪态，**直接给契约要的形态**。

    :param frame: BGR 整帧
    :returns: ``(cd_ready, confidence, source)``
        * ``cd_ready`` —— ``True``=确认就绪 / ``None``=UNKNOWN。
          **没有 ``False``**：这个检测器判不出「在冷却」（见模块文档）。
          对 ``q_ready()`` 来说 ``None`` 与 ``False`` 行为相同（都不触发），
          区别只在排查时 ``unknowns`` 里有没有 ``cd``。
        * ``confidence`` —— 环形青光占比（诊断用）
        * ``source`` —— ``"key_q"``（真读到亮环）/ ``"none"``（UNKNOWN）

    ★ 任何异常都降级成 ``(None, 0.0, "none")``，理由同
      :func:`key_e.read_skill_cd`：调用方不需要写 try/except。
    """
    try:
        state, conf = detect_key_q(frame)
    except Exception:  # noqa: BLE001 - ROI 缺失 / 帧形状怪 / cv2 抽风
        logger.warning("大招检测异常，本帧降级为 UNKNOWN", exc_info=True)
        return None, 0.0, "none"

    if state is KeyQState.READY:
        return True, conf, "key_q"
    return None, conf, "none"


__all__ = [
    "CD_KEY",
    "CYAN_B_MIN",
    "CYAN_RATIO_READY",
    "CYAN_V_MIN",
    "R_IN",
    "R_OUT",
    "ROI_NAME",
    "KeyQState",
    "annulus_mask",
    "detect_key_q",
    "read_burst_cd",
    "ring_stats",
]
