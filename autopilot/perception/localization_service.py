"""任务驱动定位 · 集成服务。

★ 它只做"串起来"，不含任何新算法
--------------------------------
``phaseCorrelate`` 来自 ``cv2``，圆心掩码来自 :mod:`minimap_mask`，
场景判定来自 :mod:`scene_detector`，位移累积与瓦片查询来自
:mod:`tile_tracker`，坐标换算来自 ``coord_transform``。
本模块负责**编排**和**状态机**，一行算法都没新写。

★ 状态机
--------
::

    UNINITIALIZED --on_teleport_ocr()--> RESETTING
    TRACKING --scene 非大世界--> PAUSED --scene 恢复--> TRACKING
    任意态 --on_teleport_ocr()--> RESETTING
    RESETTING --②LOADING + ③RESTORED--> TRACKING(新起点)
    RESETTING --30s 超时--> 作废，回进入前状态（旧位置保留）

**传送严格三段全过才刷新位置**（见 notes/teleport_contract.md）：
    ① SELECT        on_teleport_ocr 挂起目标，**不提交坐标**
    ② LOADING       scene 连续非大世界（检测器自带 N=3.0s 去抖）
    ③ RESTORED      scene 回大世界 且 changed==True -> 此时才 teleport_to
任一段未满足（含超时）-> **整次作废，保留旧位置**。
初版在 ① 就提交坐标，会跳过整个契约，传送失败时位置已污染且无法回退。

★ 三条不可省的约定
--------------------
1. **PAUSED 期间上一帧缓存必须作废。** 菜单里的小地图是 UI 图案，
   回到大世界后拿它和上一帧做相位相关，会得到一个**巨大的假位移**。
   所以 scene 从非大世界恢复时，丢弃缓存的上一帧小地图，
   下一帧重新起步（代价是丢 1 帧位移，可忽略）。
2. **传送必须由外部显式通知。** 本服务**不猜**"这是不是传送" ——
   它只认 :meth:`LocalizationService.on_teleport_ocr`。
   原因：§16.3 实测，相位相关的帧间位移里混着大量 >3px 的跳变，
   靠阈值区分"传送"和"玩家跑得快"不可靠。判据交给传送契约。
3. **返回的是"最后已知位置"，不是"实时位置"。**
   非大世界时位置是冻结的。调用方必须读 ``state.state`` 才知道
   这个位置有多旧。

★ 比例与符号的来源
--------------------
``city_px_to_world`` 读 ``configs/minimap_scale.toml``（由 TileTracker
负责）。**底图轴向转置**已在 TileTracker 内处理（``dwx=-dy``,
``dwz=-dx``），本模块**不再做任何坐标换算**，避免两处都转置一遍。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple

__all__ = [
    "LocalizationService",
    "LocalizationState",
    "LocalizationStateMachine",
    "PHASE_MIN_PEAK_RESPONSE",
    "PHASE_MIN_CIRCLE_STD",
]

#: 相位相关峰值响应下限。低于它说明两帧小地图不像（场景突变/黑屏）。
#: 依据：session_a 静止基线 579 帧的 p99 |d| = 0.109 px，
#: 大世界段峰值响应 p25 = 0.89。
PHASE_MIN_PEAK_RESPONSE: float = 0.10

#: 圆内标准差下限。低于它说明圆内是纯色（黑屏/加载/纯水），
#: 相位相关的结果没有意义。依据：大世界段 p25 = 39.1。
PHASE_MIN_CIRCLE_STD: float = 20.0

#: 传送契约 RESETTING 阶段超时（秒）。超时 -> 整次传送作废，保留旧位置。
#: 30s 远大于一次传送动画（3~5s），留足余量。
RESET_TIMEOUT_S: float = 30.0


def _sibling(name: str):
    """加载同目录模块。**兼容包导入和按文件加载两种情况。**"""
    try:
        return importlib.import_module(f".{name}", __package__)
    except (ImportError, ValueError, TypeError):
        p = Path(__file__).with_name(f"{name}.py")
        s = importlib.util.spec_from_file_location(f"_nk_{name}", p)
        m = importlib.util.module_from_spec(s)
        sys.modules[f"_nk_{name}"] = m
        s.loader.exec_module(m)
        return m


class LocalizationStateMachine(str, Enum):
    """定位服务的四个状态。"""

    UNINITIALIZED = "UNINITIALIZED"
    TRACKING = "TRACKING"
    PAUSED = "PAUSED"
    RESETTING = "RESETTING"


@dataclass(frozen=True)
class LocalizationState:
    """一帧的定位输出。"""

    state: LocalizationStateMachine
    world_pos: Optional[Tuple[float, float]]
    tile_id: Optional[str] = None
    in_tile_px: Optional[Tuple[float, float]] = None
    drift_bound: float = 0.0
    step_count: int = 0
    is_world: bool = False
    scene_changed: bool = False
    #: 位置最后一次**真正更新**的时间戳。PAUSED 时它不前进。
    position_age_s: float = 0.0
    #: 本帧的相位相关峰值响应；未做相关时为 None
    peak_response: Optional[float] = None
    #: 本帧的帧间位移（小地图像素）；未做相关时为 None
    step_px: Optional[Tuple[float, float]] = None
    note: str = ""


def _prep_minimap(crop, mask):
    """小地图裁块 -> 相位相关用的 float64 图（减均值 + Hanning 窗）。"""
    import cv2
    import numpy as np
    h, w = crop.shape[:2]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    m = (mask > 0).astype(np.float64)
    inside = m > 0
    if inside.sum() < 100:
        return None, 0.0
    v = g.astype(np.float64) * m
    v -= v[inside].mean()
    win = cv2.createHanningWindow((w, h), cv2.CV_64F)
    v *= win
    return v, float(g[inside].std())


class LocalizationService:
    """输入一帧画面 -> 输出世界坐标。

    :param scene_detector: 场景检测器。``None`` 时用默认参数构造。
    :param tracker: 预置追踪器。``None`` 表示懒加载（收到首个传送目标才建）。
    :param minimap_fn: 小地图裁剪函数 ``(frame) -> (crop, mask)``。
        ``None`` 时用 :func:`minimap_mask.make_minimap_roi` 的默认 pad。
    :param phase_fn: 相位相关函数 ``(prev, curr) -> (dx, dy, response)``。
        ``None`` 时用 ``cv2.phaseCorrelate``。**注入它是为了单测
        不必真去贴图**，也让"位移从哪来"只有一个入口。

    用法::

        svc = LocalizationService()
        svc.on_teleport_ocr(target)          # SELECT 阶段
        for frame, ts in stream:
            st = svc.on_frame(frame, ts)     # 每帧
            if st.state is LocalizationStateMachine.TRACKING:
                print(st.world_pos, st.tile_id, st.drift_bound)
    """

    def __init__(self,
                 scene_detector=None,
                 tracker=None,
                 minimap_fn: Optional[Callable] = None,
                 phase_fn: Optional[Callable] = None,
                 min_peak_response: float = PHASE_MIN_PEAK_RESPONSE,
                 min_circle_std: float = PHASE_MIN_CIRCLE_STD):
        self.scene = (scene_detector if scene_detector is not None
                      else _sibling("scene_detector").SceneDetector())
        self.tracker = tracker
        self._minimap_fn = minimap_fn
        self._phase_fn = phase_fn or self._default_phase
        self.min_peak_response = float(min_peak_response)
        self.min_circle_std = float(min_circle_std)

        self._state = LocalizationStateMachine.UNINITIALIZED
        self._last_mini = None            # 上一帧的小地图（float64）
        self._last_timestamp: Optional[float] = None
        self._pos_timestamp: Optional[float] = None
        # ---- 传送契约三段（①SELECT 已由 on_teleport_ocr 挂起）
        self._pending_target = None
        self._pending_pos: Optional[Tuple[float, float]] = None
        self._reset_prev_state = LocalizationStateMachine.UNINITIALIZED
        self._loading_seen = False        # ② LOADING 是否已确认
        self._reset_deadline: Optional[float] = None
        self._note = ""
        self._circle_cfg = None
        self._last_tracker_state = None
        self._peak = None
        self._step = None

    # -------------------------------------------------- 相位相关
    @staticmethod
    def _default_phase(prev, curr):
        import cv2
        (dx, dy), resp = cv2.phaseCorrelate(prev, curr)
        return float(dx), float(dy), float(resp)

    def _minimap(self, frame):
        """帧 -> (裁块, 掩码)。**优先用注入的函数。**"""
        if self._minimap_fn is not None:
            return self._minimap_fn(frame)
        mm = _sibling("minimap_mask")
        if self._circle_cfg is None:
            self._circle_cfg = mm.load_circle_cfg()
        return mm.make_minimap_roi(frame, self._circle_cfg)

    # -------------------------------------------------- 传送
    def on_teleport_ocr(self, target) -> None:
        """**① SELECT** —— OCR 识别到传送目标。

        ★ 这里**不提交坐标**，只是把目标挂起。
          位置刷新必须等 ② LOADING 和 ③ WORLD_RESTORED 都过（见
          :meth:`_advance_reset`）。初版在这一行就 ``teleport_to``，
          会跳过整个契约 —— 传送失败/取消时位置已经被污染且无法回退。

        :param target: :class:`teleport_ocr.TeleportTarget`。
            落点取 **``tran_position``**（传送后角色实际站的位置），
            不是 ``position``（传送点本体）—— 两者平均差 7.5 米。
        :raises ValueError: 目标没有可用的三轴落点坐标
        """
        pos = getattr(target, "tran_position", None) or \
            getattr(target, "position", None)
        if not (isinstance(pos, (list, tuple)) and len(pos) >= 3):
            raise ValueError(
                f"传送目标没有可用的落点坐标: {target!r}\n"
                f"需要 TeleportTarget.tran_position（传送后实际落点）")
        self._pending_target = target
        self._pending_pos = (float(pos[0]), float(pos[2]))
        # ⚠️ 只有**首次** SELECT 才记回退目标和超时锚点。
        #   RESETTING 期间 OCR 再次命中（菜单重绘、重复轮询）时，
        #   若照写不误，_reset_prev_state 会变成 RESETTING —— 超时 abort
        #   回到 RESETTING，而 deadline 又被重新武装，服务永久卡死。
        #   所以这里只**换目标**，不重置时钟。
        if self._state is not LocalizationStateMachine.RESETTING:
            self._reset_prev_state = self._state
            # 已在 PAUSED = 已经确认过"非大世界 3s"，② LOADING 天然满足
            self._loading_seen = (self._state is LocalizationStateMachine.PAUSED)
            self._reset_deadline = self._last_timestamp + RESET_TIMEOUT_S \
                if self._last_timestamp is not None else None
        self._state = LocalizationStateMachine.RESETTING
        # ★ 上一帧小地图立即作废：传送期间小地图内容无意义，
        #   拿它做相位相关会得到跨传送的巨大假位移。
        self._last_mini = None
        self._peak = None
        self._step = None

    def _abort_reset(self, reason: str) -> None:
        """传送作废：丢掉待提交目标，**保留旧位置**，回到进入前状态。"""
        self._pending_target = None
        self._pending_pos = None
        self._loading_seen = False
        self._reset_deadline = None
        self._state = self._reset_prev_state
        self._last_mini = None
        self._note = f"teleport aborted: {reason}"

    def _commit_reset(self) -> None:
        """三段全过：位置刷新（**替换**，不是更新），转 TRACKING。"""
        wx, wz = self._pending_pos
        if self.tracker is None:
            tt = _sibling("tile_tracker")
            self.tracker = tt.TileTracker((wx, wz))
        else:
            self.tracker.teleport_to((wx, wz))
        self._last_tracker_state = self.tracker._state(tile_changed=False)
        self._pos_timestamp = self._last_timestamp
        self._pending_target = None
        self._pending_pos = None
        self._loading_seen = False
        self._reset_deadline = None
        self._state = LocalizationStateMachine.TRACKING
        self._last_mini = None
        self._note = "teleport committed"

    def _advance_reset(self, is_world: bool, changed: bool) -> bool:
        """推进契约 ② LOADING / ③ WORLD_RESTORED。返回是否已提交。"""
        if not is_world:
            # 场景检测器自身有 N=3.0s 去抖，走到这里说明已连续非大世界 3s
            self._loading_seen = True
            return False
        if changed and self._loading_seen:
            # ③ WORLD_RESTORED：判据必须是 changed，不是"当前 is_world"
            self._commit_reset()
            return True
        return False

    # -------------------------------------------------- 主循环
    def on_frame(self, frame, timestamp: float) -> LocalizationState:
        """每帧调用一次。

        :param frame: BGR 画面。分辨率任意（掩码按当前帧尺寸归一化）。
        :param timestamp: **秒级时间戳**，要求单调不减。
        :returns: :class:`LocalizationState`
        """
        if timestamp is None:
            raise TypeError("timestamp 不能为 None")
        ts = float(timestamp)
        self._last_timestamp = ts
        st = self.scene.update(frame, ts)
        is_world = bool(st.is_world)
        changed = bool(st.changed)
        self._note = ""

        # ---- RESETTING：只认契约，**期间不累积位移**
        if self._state is LocalizationStateMachine.RESETTING:
            if self._reset_deadline is None:
                self._reset_deadline = ts + RESET_TIMEOUT_S
            elif ts > self._reset_deadline:
                self._abort_reset(f"timeout after {RESET_TIMEOUT_S:.0f}s")
            if self._state is LocalizationStateMachine.RESETTING:
                self._advance_reset(is_world, changed)
            self._last_mini = None
            return self._snapshot(
                is_world, changed, now=ts,
                note=self._note or "teleport pending: position frozen")

        # ---- 常规场景转移
        if self._state is LocalizationStateMachine.TRACKING:
            if not is_world:
                self._state = LocalizationStateMachine.PAUSED
        elif self._state is LocalizationStateMachine.PAUSED:
            if is_world:
                # ★ 菜单里的小地图是 UI 图案，必须作废缓存
                self._last_mini = None
                self._state = LocalizationStateMachine.TRACKING
        # UNINITIALIZED 原地不动，等 on_teleport_ocr

        # ---- 非大世界：冻结位置，**不做相位相关**
        if not is_world:
            self._last_mini = None
            return self._snapshot(is_world, changed, now=ts,
                                  note="non-world: position frozen")

        # ---- UNINITIALIZED：没有起点
        if self._state is LocalizationStateMachine.UNINITIALIZED or \
                self.tracker is None:
            self._capture_mini(frame)
            return self._snapshot(is_world, changed, now=ts,
                                  note="awaiting first teleport target")

        # ---- 大世界：相位相关 + 位移累积
        _, mimg, cstd = self._capture_mini(frame, want=True)
        if mimg is not None and self._last_mini is not None:
            if cstd is not None and cstd < self.min_circle_std:
                self._step, self._peak = None, None
                self._last_mini = mimg
                return self._snapshot(is_world, changed, now=ts,
                                      note="circle too flat, phase skipped")
            try:
                dx, dy, resp = self._phase_fn(self._last_mini, mimg)
            except Exception as exc:                  # noqa: BLE001
                self._last_mini = mimg
                return self._snapshot(is_world, changed, now=ts,
                                      note=f"phase failed: {exc}")
            if resp < self.min_peak_response:
                self._step, self._peak = (dx, dy), resp
                self._last_mini = mimg
                return self._snapshot(
                    is_world, changed, now=ts,
                    note=f"peak {resp:.2f} below threshold, step skipped")
            try:
                tstate = self.tracker.update(dx, dy)
            except Exception as exc:                  # noqa: BLE001
                # ★ TeleportJump 及其它：位置不动，跳过这一帧
                self._last_mini = mimg
                self._step, self._peak = (dx, dy), resp
                return self._snapshot(is_world, changed, now=ts,
                                      note=f"tracker rejected step: {exc}")
            self._last_tracker_state = tstate
            self._pos_timestamp = ts
            self._step, self._peak = (dx, dy), resp
        self._last_mini = mimg
        return self._snapshot(is_world, changed, now=ts)

    def _capture_mini(self, frame, want=False):
        """取本帧小地图并预处理。``want=False`` 时不返回浮点图（省 CPU）。"""
        crop, mask = self._minimap(frame)
        if not want:
            self._last_mini = None
            return crop, None, None
        v, cstd = _prep_minimap(crop, mask)
        return crop, v, cstd

    # -------------------------------------------------- 输出
    def get_position(self) -> Optional[Tuple[float, float]]:
        """当前世界坐标。**未初始化返回 None。**"""
        if self.tracker is None:
            return None
        return tuple(self.tracker.world_pos)

    def nearby_tiles(self) -> list:
        """当前块 + 8 邻居。**未初始化或出图返回空列表。**"""
        if self.tracker is None:
            return []
        return self.tracker.nearby_tiles()

    def _snapshot(self, is_world, changed, note="", now=None) -> LocalizationState:
        """组装输出。``now`` 是本帧时间戳，用来算位置"有多旧"。"""
        age = 0.0
        if self._pos_timestamp is not None and now is not None:
            age = max(0.0, float(now) - self._pos_timestamp)
        t = self._last_tracker_state
        wp = self.get_position()
        return LocalizationState(
            state=self._state,
            world_pos=wp,
            tile_id=(t.tile_id if t is not None else None),
            in_tile_px=(t.in_tile_px if t is not None else None),
            drift_bound=(self.tracker.drift_bound if self.tracker else 0.0),
            step_count=(self.tracker.step_count if self.tracker else 0),
            is_world=is_world, scene_changed=changed,
            position_age_s=age, peak_response=self._peak, step_px=self._step,
            note=note or self._note)

    def __repr__(self) -> str:                        # pragma: no cover
        p = self.get_position()
        ps = "None" if p is None else f"({p[0]:.1f}, {p[1]:.1f})"
        return f"LocalizationService({self._state.value}, pos={ps})"
