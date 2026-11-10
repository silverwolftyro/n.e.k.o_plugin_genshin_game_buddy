"""``key_e``（元素战技）检测器 —— 中心方块饱和度 + 明度双闸门。

★ 为什么不用模板匹配
--------------------
E 是**角色技能**图标，每个角色图案都不同。实测模板分数随角色在
0.914 / 0.218 之间跳，换角色就废。只能靠"外观状态"而非"图案内容"。

★ 状态信号是饱和度，不是明度
----------------------------
就绪 = 彩色（金/角色元素色）；CD 中 = 去饱和的灰白。
两者的**明度几乎一样**（实测都在 190~205），动的是饱和度。
所以判据建在饱和度上，明度只用来识别"这里压根不是 E 图标"。

🔴 2026-10-18 实测推翻：这个特征**测的是背景，不是技能状态**
--------------------------------------------------------------
296 帧遗留训练集（``recognition/raw/frames/batch_20260929b``）跑下来，
把中心方块 sat 与**同一帧**紧邻 ROI 外圈的背景环 sat 做相关::

    r = 0.88        内 = 0.65 * 外 - 1.04

原因看图就知道：E 徽章是**半透明**的（圆形底 + 中间角色剪影），
内部能透出后面的场景 —— 沙漠偏黄、草地偏绿、夜里偏暗。
于是"中心方块平均饱和度"实际是背景光照的代理。

⚠️ **不要**通过调 ``SAT_READY`` 来"修"这个问题。绝对阈值作用在一个
   被背景主导的量上，调它只会在"亮场景永远按 E"和"暗场景永远不放 E"
   之间二选一，两边都错。必须换特征（对比量 / ONNX 分类器）。

★ 直方图里那道"谷底"是假的
--------------------------
按 bin=5 看，``[25,35)`` 只有 3 个样本，``SAT_READY = 30`` 正好落在
里面 —— 看着像"早就标定好了"。**那是背景光照的分界**（暗地形 vs 亮沙地）。
扣掉背景之后（``inside - k*ring``）这道谷被抹平，分布变成主峰
``[-5,0)`` 加一条长右尾。真有两态的话扣背景只会更清楚。

复现：``recognition/exports/benchmark/scripts/calibrate_key_e.py``
（只读，会打印上面那些数）

★ 三条分支的来源（都实测过）
----------------------------
1. 大地图/面板覆盖 —— ``center_sat_ratio >= 0.30``
   阈值 0.3 精确命中 9 帧，且这 9 帧的 sat/val/hue 几乎相同
   （sat 114~161、val 56~121、hue 74~101），是**同一类画面**。
   其中 2 帧是人工确认的大地图，另外 7 帧特征完全同源。
   阈值下面一刀切得很干净：ratio >= 0.2 时是 10 帧，>= 0.3 时是 9 帧，
   0.29 附近**没有**样本，落点有余量。

2. 活动页/非战斗 UI —— ``center_val >= 218 且 center_sat < 27``
   ⚠ 这条是**照着 3 帧（f00210/214/218）拟的**，而那 3 帧是连续帧、
      内容几乎一样。**样本量不足，泛化性未验证**。宁可误判也不能
      让机器人对着活动页按战技。

   ✅ 2026-10-18 复核：这 5 帧 non-E（含 f00108/f00158 的地图分支）
      **全部仍被正确判成 unknown**，两条守卫都不用改。
      问题只在第 3 条。

3. 其余按 ``center_sat`` 分 READY / COOLDOWN
   ⚠ **这条的输入不承载技能状态**（见上面的红色框）。
     保留它是为了让管线仍然跑得通、且失败方向是安全的
     （读不清 -> UNKNOWN -> 规则不触发 -> 退回只普攻）。
     但**不要**把它的输出当成可信的技能状态。

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

ROI_NAME = "key_e"

#: 中心方块边长占 ROI 短边的比例。
#: 整块 ROI 会被边缘透入污染（实测 f00086/87/89 圆圈边缘透进橙色地面），
#: 取中心 50% 只剩 44x44，但那是图标本体所在，够用。
CENTER_FRAC = 0.5

#: ① 大地图/面板：中心里 S>100 的像素占比下限
SATRATIO_MAP = 0.30
#: ② 活动页：明度很高但饱和度不高
VAL_UI_MIN = 218.0
SAT_UI_MAX = 27.0
#: ③ 就绪/未就绪的饱和度分界（**未标定，见模块 docstring**）
SAT_READY = 30.0


class KeyEState(Enum):
    """元素战技的三态。"""

    READY = "ready"
    COOLDOWN = "cooldown"
    UNKNOWN = "unknown"


def _center(crop: np.ndarray, frac: float = CENTER_FRAC):
    """取中心正方形并转 HSV，返回 ``(sat, val)`` 两个通道。"""
    h, w = crop.shape[:2]
    side = int(round(min(h, w) * frac))
    if side < 4:
        return None, None
    y0, x0 = (h - side) // 2, (w - side) // 2
    hsv = cv2.cvtColor(crop[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1], hsv[:, :, 2]


def center_stats(frame: np.ndarray) -> dict[str, float | bool]:
    """中心方块的三个统计量。供批量分析与调参用。

    :returns: ``{"center_sat", "center_sat_ratio", "center_val", "ok"}``

    ★ ``ok``（2026-10-18 新增）—— **"读到了" 与 "根本没读到" 必须能区分**
      两者在最坏情况下数值完全一样（空 ROI 与纯黑 ROI 都是 sat=0/val=0），
      只看数值分不出来。少了这个标记，:func:`detect_key_e` 会把「画面是空的」
      当成「E 在 CD」报出去 —— 那是**自信地报错**，比返回 UNKNOWN 糟得多：
      规则里 ``e_ready() == False`` 和 UNKNOWN 都不触发，但前者让
      ``unknown_hits`` 是空的，排查时看不出「感知压根没工作」。
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise TypeError(f"frame 应为 (H, W, 3) 的 BGR 图像")
    rect = load_rois().get(ROI_NAME)
    if rect is None:
        raise KeyError(f"rois.toml 里没有 {ROI_NAME!r}")
    crop = rect.crop(frame[:, :, :3])
    if crop is None or getattr(crop, "size", 0) == 0:
        return {"center_sat": 0.0, "center_sat_ratio": 0.0,
                "center_val": 0.0, "ok": False}
    sat, val = _center(crop)
    if sat is None:
        return {"center_sat": 0.0, "center_sat_ratio": 0.0,
                "center_val": 0.0, "ok": False}
    return {
        "center_sat": float(sat.mean()),
        "center_sat_ratio": float((sat > 100).mean()),
        "center_val": float(val.mean()),
        "ok": True,
    }


def detect_key_e(frame: np.ndarray) -> tuple[KeyEState, float]:
    """检测元素战技状态。

    :param frame: BGR 整帧
    :returns: ``(状态, 置信度)``。置信度 = 中心方块平均饱和度
        （READY/COOLDOWN 都在这条轴上，UNKNOWN 时参考意义有限）。

    ★ 读不到就 UNKNOWN，**绝不**当成 CD
      画面是空的/ROI 出框时返回 COOLDOWN 会让规则「确定地不按 E」，
      而真相是「不知道」。判据 B 的 ``val >= 0`` 会顺带命中，所以必须
      在最前面显式挡掉（见 ``center_stats`` 的 ``ok``）。
    """
    st = center_stats(frame)
    if not st["ok"]:
        return KeyEState.UNKNOWN, 0.0

    sat, ratio, val = st["center_sat"], st["center_sat_ratio"], st["center_val"]

    # ① 大地图 / 面板盖上来：中心几乎全是高饱和地形色
    if ratio >= SATRATIO_MAP:
        return KeyEState.UNKNOWN, sat
    # ② 活动页之类的亮底 UI
    if val >= VAL_UI_MIN and sat < SAT_UI_MAX:
        return KeyEState.UNKNOWN, sat
    # ③ 正常图标：彩色 = 就绪，灰白 = CD
    return (KeyEState.READY if sat >= SAT_READY else KeyEState.COOLDOWN), sat


__all__ = [
    "CENTER_FRAC",
    "CD_KEY",
    "ROI_NAME",
    "SATRATIO_MAP",
    "SAT_READY",
    "SAT_UI_MAX",
    "VAL_UI_MIN",
    "KeyEState",
    "center_stats",
    "detect_key_e",
    "read_skill_cd",
]


#: ``PartyMember.cd_ready`` 里 E 技能对应的键。
#:
#: ★ 这个名字**必须**和 ``policy/context.py::e_ready()`` 读的那一把一致
#:   （它硬编码 ``_active_cd("skill")``）。改这里就要同步改那里，
#:   否则填了值规则也读不到 —— 症状是「接上了但还是 UNKNOWN」。
CD_KEY = "skill"


def read_skill_cd(frame: np.ndarray) -> tuple[bool | None, float, str]:
    """读 E 技能就绪态，**直接给契约要的形态**。

    :param frame: BGR 整帧
    :returns: ``(cd_ready, confidence, source)``
        * ``cd_ready`` —— ``True``=确认就绪 / ``False``=确认在 CD /
          ``None``=UNKNOWN。**这就是 :attr:`PartyMember.cd_ready` 要的值**，
          不用上层再做映射。
        * ``confidence`` —— 中心方块平均饱和度（诊断用）
        * ``source`` —— ``"key_e"``（真读到了）/ ``"none"``（UNKNOWN）

    ★ 任何异常都降级成 ``(None, 0.0, "none")``
      感知层一次抖动不该让整条 tick 崩掉，更不该让规则拿到一个猜的值。
      这里比 :func:`detect_key_e` 更"厚"一层，就是为了让调用方**不需要写
      try/except**。
    """
    try:
        state, conf = detect_key_e(frame)
    except Exception:  # noqa: BLE001 - ROI 缺失 / 帧形状怪 / cv2 抽风，一律 UNKNOWN
        logger.warning("E 技能检测异常，本帧降级为 UNKNOWN", exc_info=True)
        return None, 0.0, "none"

    if state is KeyEState.READY:
        return True, conf, "key_e"
    if state is KeyEState.COOLDOWN:
        return False, conf, "key_e"
    return None, conf, "none"
