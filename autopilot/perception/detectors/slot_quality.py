"""队伍圆环（eb_1/2/3）的**可用性**判定 —— 标定自 25 帧人工标注。

★ 25 帧坏帧其实是三种物理失效模式
----------------------------------
人工在 ``eb_1_clean_wall.png`` 上标的 25 个"不可用"帧，特征上分成三组：

  ① 纯白空槽（8 帧）f00068/69, f00109/10/11, f00162/63/64
     mean_sat = 0.000（精确）  hue_concentration = 1.000  mean_val = 255.000
     —— 那个位置根本没人，整个 ROI 是纯白。
     ★ 好帧里 mean_sat==0 的**一个都没有** → 这条判据零误杀，余量 47。

  ② 紫/灰色块（5 帧）f00014/15/16/17, f00161
     saturated_ratio = 0.000（前三帧）、mean_sat 20~81、ring_sat_delta ≈ −20
     —— UI 面板挡住了圆环，露出底下的紫色块。

  ③ 场景内容（12 帧）f00209~f00220
     mean_sat ≈ 93.35、ring_sat_delta ≈ −31.43、saturated_ratio ≈ 0.605
     —— 画面根本不是队伍界面（爬楼梯、贴着墙），那个位置是木板/石砖。
     ★ 这 11 帧特征**逐位相同**，是同一段连续静止画面。

★ 为什么用分支而不是一条 AND
---------------------------
最紧的 AND（``ring<=0.088 AND mean_sat<=93.367``）能 25/25，但阈值窗口
只有 [0.09, 2.0] 这条窄缝 —— f00161 的 ring 恰好是 +0.0899，擦边过关。
换一批数据大概率漏。分支写法把每种模式的判据**对准它自己的物理成因**，
余量宽得多。

★ ⚠ 已知的两处不完美 ★
--------------------
1. 规则会误杀 f00221 / f00222 —— 但它们的特征和被人工标为坏帧的
   f00210~f00220 **完全相同**（sat=93.35, ring=−31.43, conc=0.277）。
   所以这大概率是**标注漏了两帧**，而不是规则错了。等复查确认。

2. 阈值是**对着 eb_1 标定**的。eb_2 / eb_3 的圆环可能是别的元素、
   别的亮度，换槽位不一定适用 —— 所以 :func:`check_slot_quality` 把
   阈值暴露成模块常量，换槽位时应先各自验证。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── ① 纯白 / 空槽 ────────────────────────────────────────────────────
#: mean_sat 低于它 = 整个 ROI 一点颜色都没有。
#: 坏帧是 0.000，好帧最低 48.2，**余量 47**，这一条几乎不可能误杀。
SAT_EMPTY = 1.0

# ── ② 紫 / 灰块：饱和像素几乎为零 ──────────────────────────────────────
#: 紫块那几帧 saturated_ratio 精确为 0.000，好帧最低 0.035。
SATRATIO_FLAT = 0.02

# ── ③ 场景内容：圆环的"边缘比中心艳"特征消失 ──────────────────────────
#: 真正的圆环 ring_sat_delta 是正的（中心实心、边缘是背景）。
#: 露出木板/石砖时它变成大负值。
#: 阈值 2.0 而非贴着数据的 0.088 —— f00161 是 +0.0899，贴边会不稳。
RING_MAX = 2.0
#: 配合上一条：场景画面整体饱和度偏低。坏帧最高 93.37，好帧最低 48.2。
#: 取 95 而不是 93.37，同样是为了留余量。
SAT_SCENE_MAX = 95.0


@dataclass(frozen=True)
class SlotQuality:
    """一次质量判定的结果。"""

    usable: bool
    reason: str

    def __bool__(self) -> bool:
        return self.usable


def check_slot_quality(feats) -> SlotQuality:
    """判定某个圆环槽位当前是否可用。

    :param feats: :class:`~autopilot.perception.detectors.eb_main.EbMainFeatures`
        或含同名属性的对象
    :returns: :class:`SlotQuality`

    ★ 返回 False **不等于"这个槽位没角色"** —— 它只表示"这个 ROI 里没有
      可靠的圆环特征"，原因可能是没人、也可能是 UI 挡住了、也可能是画面
      根本不是队伍界面。三者在特征上无法进一步区分，交给上层决定。
    """
    sat = float(feats.mean_sat)
    satr = float(feats.saturated_ratio)
    ring = float(feats.ring_sat_delta)

    if sat < SAT_EMPTY:
        return SlotQuality(False, "empty_or_blank")
    if satr < SATRATIO_FLAT:
        return SlotQuality(False, "flat_or_occluded")
    if ring <= RING_MAX and sat <= SAT_SCENE_MAX:
        return SlotQuality(False, "not_a_team_screen")
    return SlotQuality(True, "ok")


__all__ = [
    "RING_MAX",
    "SATRATIO_FLAT",
    "SAT_EMPTY",
    "SAT_SCENE_MAX",
    "SlotQuality",
    "check_slot_quality",
]
