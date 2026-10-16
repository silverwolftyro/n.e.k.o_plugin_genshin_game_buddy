"""``party_1..4``（角色头像）存活判定 —— **已冻结，只做两态**。

★ ★ 冻结说明：为什么不判「死亡」 ★ ★
------------------------------------
本模块**故意没有** DEAD 状态。这不是漏做，是主动搁置：

1. ``batch_20260929b`` 全 223 帧里**没有一帧真正的角色死亡**。
   没有正样本时写阈值，只会产出死代码或疯狂误报。
2. 误报代价高于漏报 —— 误判「不可用」会让机器人拒绝行动，
   而漏判死亡只是少一个优化项。

要恢复死亡检测，先按
``exports/benchmark/notes/death_detection_todo.md`` 录一段含死亡的素材。

★ ★ 冻结说明：这里测的根本不是「死没死」 ★ ★
--------------------------------------------
``check_available`` 实际测的是「**队伍栏有没有被 UI 盖住**」。

两种情况在数值上撞车：
    · UI 整块盖住队伍栏  -> 四个头像同时变纯灰
    · 角色真的死了        -> 只有一个头像变灰

真实死亡**永远不会**让四槽同时归零，所以这个规则对死亡天然不敏感。

★ 当前的已知缺口（**可接受**）★
------------------------------
在 ``scene_detection_todo.md`` 的方案落地之前，
下面这 7 帧会被误判成「可用」：

    f00157  f00159  f00208    大地图覆盖，头像格里是地形
    f00210  f00215  f00216  f00220    背包页覆盖，头像格是虚化色块

它们的 ``mean_sat`` 读数是 49~123，压根不是灰的 —— 现有特征覆盖不到。

**取舍依据**：8/15 命中、**0 误报**。
另一条「三血条全为 0」的规则能到 13/15，但误报 25 帧（11% 的正常画面），
已否决 —— 误报会让机器人无故拒绝行动。

**恢复死亡检测 / 修这三类遮挡的完整需求见**
``exports/benchmark/notes/scene_detection_todo.md``。
"""

from __future__ import annotations

import logging
from enum import Enum

import cv2
import numpy as np

from ..roi_loader import load_rois

logger = logging.getLogger(__name__)

#: 队伍栏不可信的饱和度上限。低于它 = 这一格是纯灰，没有头像信息。
#: 来自 batch_20260929b：被整块遮挡的 8 帧该值精确为 0，
#: 正常帧最低约 20 —— 20 落在两者之间，**没有余量**，是折中值。
SAT_GRAY = 20

#: 头像是"满员"的槽位数。rois.toml 里配了 1..4。
SLOTS = (1, 2, 3, 4)


class PartyState(Enum):
    """队伍栏的可用性。**故意只有两态** —— 见模块 docstring。"""

    ALIVE = "alive"      # 队伍栏可读
    UNKNOWN = "unknown"  # 被 UI 遮挡 / 不可信


def read_sat(frame: np.ndarray, slot: int) -> float:
    """一个头像槽位的平均饱和度（0~255）。

    ★ 注意命名
      ``rois.toml`` 里**没有** ``party_avatar_*``。头像就是 ``party_1..4``：

          eb_1       元素圆环  (x≈0.822~0.862)
          party_1    角色头像  (x≈0.926~0.967)   <- 本函数读它
          hp_party_1 后台血条  (x≈0.880~0.918, y 略低)
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise TypeError(f"frame 应为 (H, W, 3) 的 BGR 图像")
    rect = load_rois().get(f"party_{slot}")
    if rect is None:
        raise KeyError(f"rois.toml 里没有 party_{slot}")
    crop = rect.crop(frame[:, :, :3])
    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1].mean())


def read_all_sat(frame: np.ndarray) -> dict[int, float]:
    """全部槽位的平均饱和度。"""
    return {s: read_sat(frame, s) for s in SLOTS}


def check_available(frame: np.ndarray) -> tuple[PartyState, dict[int, float]]:
    """判断队伍栏当前是否可读。

    :returns: ``(状态, 每槽平均饱和度)``

    ★ 判据：4 槽**全部**低于 :data:`SAT_GRAY` 才判 UNKNOWN
      「全灰」是 UI 整块遮挡的特征；单个槽位低饱和**不**作为依据 ——
      那是角色的图标配色差异，不是状态。
    """
    sats = read_all_sat(frame)
    all_gray = all(v < SAT_GRAY for v in sats.values())
    return (PartyState.UNKNOWN if all_gray else PartyState.ALIVE), sats


__all__ = [
    "SAT_GRAY",
    "SLOTS",
    "PartyState",
    "check_available",
    "read_all_sat",
    "read_sat",
]
