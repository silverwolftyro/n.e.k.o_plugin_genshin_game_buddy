"""转向律：把「我在哪 / 目标在哪 / 我朝哪」变成「转多少、怎么走」。

★★ 契约
--------
**纯函数**：无 IO、无线程、无全局可变状态、无时间源。
所有输入都显式传进来，所有输出都是返回值。
因此每一个函数都能被单测逐条钉死 —— 而这一段的常数**全是真机标定的**，
最怕的就是「跑着跑着悄悄漂了」。

角度约定（全模块统一，改一处就全错）
--------------------------------------
* 方位角一律 ``degrees(atan2(dz, dx))``，即 **+x 为 0°，逆时针为正**。
* ``delta_deg``（Δ）= **目标方位角 − 当前朝向**。
* **Δ > 0 = 目标在左边**（要往左转）。
* 下发的鼠标位移是 ``-fraction × Δ``：Δ 为正时下发负像素，
  即「鼠标往右推 = 视角往左转」，所以两者符号相反（见 :func:`turn_plan`）。

出处
----
逐条对照 ``recognition/exports/benchmark/scripts/misc/e2e_pathfind.py``：
``_turn_plan`` / ``_split_turn`` / ``_stand_turn_step`` /
``_dynamic_max_step_px`` / ``_circ_mean_deg`` / ``_heading_deg``。
抽出来时**没有重新标定任何常数**，只是把它们变成了可注入的 ``cfg``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DEFAULT_CONFIG",
    "SteerConfig",
    "SteerPlan",
    "bearing_deg",
    "circ_mean_deg",
    "dynamic_max_step_px",
    "heading_deg",
    "split_turn",
    "stand_turn_step",
    "steer",
    "turn_plan",
    "wrap_deg",
]


# ==================================================================
# 配置
# ==================================================================


@dataclass(frozen=True)
class SteerConfig:
    """转向律的全部可调常数。

    默认值 = ``e2e_pathfind.py`` 的模块级常量，逐条对应（括号内是出处）::

        mouse_deg_per_px   = MOUSE_DEG_PER_PX
        turn_fraction      = TURN_FRACTION
        turn_max_deg       = TURN_MAX_DEG
        turn_max_deg_big   = TURN_MAX_DEG_BIG
        turn_big_err_deg   = TURN_BIG_ERR_DEG
        turn_deadzone_deg  = TURN_DEADZONE_DEG
        turn_stand_deg     = TURN_STAND_DEG
        turn_resume_deg    = TURN_RESUME_DEG
        turn_stand_segments= TURN_STAND_SEGMENTS
        min_turn_px        = MIN_TURN_PX
        mouse_saturation_px= MOUSE_SATURATION_PX
        split_max_segments = SPLIT_MAX_SEGMENTS
        speed_run_mps      = SPEED_RUN_MPS
        step_limit_safety  = STEP_LIMIT_SAFETY
        step_limit_min_px  = STEP_LIMIT_MIN_PX
        step_limit_max_px  = STEP_LIMIT_MAX_PX

    :param arrive_radius_m: 默认到达半径（米）。原脚本是 ``--arrive`` 命令行
        参数的默认值 1.5。
    :param climb_eps_m: 单段位移低于它就算「没动」，方向估计里跳过。
    :param window_move_min_m: 窗口净位移低于它就**判不出朝向**（返回 None）。
    """

    # -- 鼠标 -> 角度 ------------------------------------------------
    #: 游戏内每 1px 鼠标相对位移转多少度。**改游戏内灵敏度就必须重标**。
    mouse_deg_per_px: float = 0.07
    #: 每帧最多纠正 |Δ| 的这个比例（无量纲比例控制，不是绝对增益）。
    turn_fraction: float = 0.45
    #: 单帧转向的角度上限（小误差时）。
    turn_max_deg: float = 45.0
    #: |Δ| >= turn_big_err_deg 时放宽到的角度上限（**仍有界**）。
    turn_max_deg_big: float = 60.0
    turn_big_err_deg: float = 90.0
    #: |Δ| 小于它就不转（死区）。
    turn_deadzone_deg: float = 8.0
    #: 小于 1px 的移动没有意义（会被取整成 0，白发一次）。
    min_turn_px: int = 1

    # -- 大角度站定（滞回）------------------------------------------
    #: 进入站定的夹角阈值。
    turn_stand_deg: float = 60.0
    #: 退出站定的夹角阈值。**与进入阈值差 40° 就是滞回的全部意义** ——
    #: 单阈值必然在阈值附近震荡。
    turn_resume_deg: float = 20.0
    #: 站定期间每帧把转向拆成几段（加速转）。
    turn_stand_segments: int = 8
    #: ★ 站定阶段愿意**开环**花掉的总转角，占进入时 |Δ| 的比例
    #:   （2026-10-14）。
    #:
    #: 取 0.95 而不是 1.0：预算花完那一刻我们**看不到**真实剩余误差
    #: （朝向冻结），留 5% 宁可少转一点、下一轮重测，也不要冲过去。
    stand_budget_ratio: float = 0.95

    # -- 鼠标饱和 ----------------------------------------------------
    #: 单次注入的像素上限。真机分桶：40~60px 达成率 74%，60~80 掉到 55%。
    mouse_saturation_px: int = 40
    #: 常态每帧拆几段。**段数少只是「发得少」，不是「不设上限」**。
    split_max_segments: int = 1

    # -- 位移跳变检测 ------------------------------------------------
    #: 跑步最高速（世界单位/秒），自拍定标，Sprint 约 6~11 m/s 取上界。
    speed_run_mps: float = 11.0
    #: 动态阈值相对「正常运动一帧位移」的安全倍数。
    step_limit_safety: float = 3.0
    #: 动态阈值的下限（保住走路/静止的裕量）。
    step_limit_min_px: float = 50.0
    #: 动态阈值的上限（远低于真传送的 100~5000px）。
    step_limit_max_px: float = 500.0

    # -- 到达 / 方向估计 ---------------------------------------------
    arrive_radius_m: float = 1.5
    climb_eps_m: float = 0.02
    window_move_min_m: float = 1.0

    def stand_step_cap_deg(self) -> float:
        """站定期间**单帧**允许发出的转角上限（度）。

        = 段数 × 单段像素上限 × 每像素度数。默认
        ``8 × 40 × 0.07 = 22.4°``。转向预算就靠它按帧扣减。
        """
        return (self.turn_stand_segments * self.mouse_saturation_px
                * self.mouse_deg_per_px)


#: 模块级默认配置。**只读使用**；要改就 :func:`dataclasses.replace` 造新的。
DEFAULT_CONFIG = SteerConfig()


# ==================================================================
# 角度工具
# ==================================================================


def wrap_deg(a: float) -> float:
    """把角度归一化到 ``[-180, 180)``。

    ``atan2`` 出来的差值可能跨 ±π（+179 与 -179 其实是几乎同向）。
    """
    return (float(a) + 180.0) % 360.0 - 180.0


def circ_mean_deg(angles_deg: list[float] | tuple[float, ...]) -> float | None:
    """一组角度的**圆周**均值（度）。样本 <2 或向量和退化返回 ``None``。

    ★ 不能用算术平均：角度跨 ±180° 时（比如 +179 和 -179）算术平均会得到
      0°，完全错。必须先转单位向量、相加、最后 atan2。
    """
    if len(angles_deg) < 2:
        return None
    sx = sum(math.cos(math.radians(a)) for a in angles_deg)
    sy = sum(math.sin(math.radians(a)) for a in angles_deg)
    if math.hypot(sx, sy) < 1e-9:
        return None
    return math.degrees(math.atan2(sy, sx))


def bearing_deg(pos_x: float, pos_z: float,
                target_x: float, target_z: float) -> float:
    """从 ``(pos_x, pos_z)`` 指向 ``(target_x, target_z)`` 的方位角（度）。

    约定：**+x 为 0°，逆时针为正**（``atan2(dz, dx)``）。
    """
    return math.degrees(math.atan2(float(target_z) - float(pos_z),
                                   float(target_x) - float(pos_x)))


def heading_deg(hist: list[tuple[float, float, float]],
                now: float,
                window_s: float,
                cfg: SteerConfig = DEFAULT_CONFIG) -> float | None:
    """"进入方向" = 最近 ``window_s`` 秒内各段位移方向的圆周均值（度）。

    :param hist: ``(t, x, z)`` 时间戳窗口，**按时间升序**。
    :param now: 当前时刻。
    :param window_s: 窗口宽度（秒）。
    :returns: 朝向（度）；窗口内有效段 <2 时返回 ``None``。

    ★ 用**滑动窗口平均**，不用「最后一帧方向」—— 停下之后窗口位移归零、
      方向冻结，拿它判朝向等于判噪声。
    ★ 单段位移 <= ``climb_eps_m`` 就跳过，避免停在原地时把噪声当方向。
    ★ 判不出朝向时返回 ``None``，**由调用方决定**（原脚本是「按距离放行」，
      本库把它显式化成 :func:`steer` 的 ``walk_unaligned`` 策略）。
    """
    segs: list[float] = []
    t_floor = now - window_s
    for i in range(1, len(hist)):
        ta, xa, za = hist[i - 1]
        tb, xb, zb = hist[i]
        if ta < t_floor:
            continue
        dx, dz = xb - xa, zb - za
        if math.hypot(dx, dz) <= cfg.climb_eps_m:
            continue
        segs.append(math.degrees(math.atan2(dz, dx)))
    return circ_mean_deg(segs)


# ==================================================================
# 转向律
# ==================================================================


def turn_plan(d_deg: float,
              cfg: SteerConfig = DEFAULT_CONFIG) -> tuple[float, float, float]:
    """把夹角变成 ``(想转多少度, 限幅度数, 实际下发像素)``。

    ★ 这是**无量纲比例控制**，不是绝对增益::

          want = -fraction × Δ        限幅按**度数**做（不按像素做）

    ★ **符号相反**：Δ > 0 表示目标在左边，要往左转；而「鼠标往右推 =
      视角往左转」，所以下发的是**负**像素。这一处搞反了整个寻路会往
      反方向画圈 —— 单测里有专门的用例钉它。

    :param d_deg: 夹角（度），= 目标方位角 − 当前朝向。
    """
    d = float(d_deg)
    want = -cfg.turn_fraction * d
    cap = (cfg.turn_max_deg_big if abs(d) >= cfg.turn_big_err_deg
           else cfg.turn_max_deg)
    want = max(-cap, min(cap, want))
    px = want / cfg.mouse_deg_per_px
    if 0 < abs(px) < cfg.min_turn_px:
        # 取整会变 0 的那一小撮：宁可发 1px 也不要白发一次。
        px = float(cfg.min_turn_px if px > 0 else -cfg.min_turn_px)
    return want, cap, px


def split_turn(turn_px: float,
               max_segments: int | None = None,
               cfg: SteerConfig = DEFAULT_CONFIG
               ) -> tuple[list[int], int]:
    """把一次转向拆成若干段，**每段都不超过饱和阈值**。

    :param turn_px: 本帧总共要下发的像素（带符号）。
    :param max_segments: 本次最多拆几段。``None`` = 用
        ``cfg.split_max_segments``（常态 1）。站定加速转传
        ``cfg.turn_stand_segments``。
    :returns: ``(本帧要发的段列表, 发不掉的余量像素)``

    ★★ 「不拆」与「不限幅」是**两件事**。旧代码把它们混成一条捷径
    （``SPLIT_MAX_SEGMENTS <= 1 就直接返回原值``），而默认值恰好是 1，
    于是「不拆」= 「不限幅」= 857px 直接落进饱和区（达成率 < 23%）。
    这里两个量彻底解耦：发几次 = 段数，一次多少 = 饱和阈值（**永远**生效）。

    ★ 余量必须**退回队列**而不是丢弃或硬塞：丢掉则转向永远差一截；
      硬塞进最后一段又回到「单段超阈值 -> 收益打折」的老问题。
    """
    total = int(round(turn_px))
    if total == 0:
        return [], 0
    sgn = -1 if total < 0 else 1
    # ★ 饱和阈值 × 段数，且**没有**「段数<=1 就不限幅」的旁路。
    room = cfg.mouse_saturation_px * max(
        1, cfg.split_max_segments if max_segments is None else int(max_segments))
    take = min(abs(total), room)
    n = max(1, math.ceil(take / cfg.mouse_saturation_px))
    base = take // n
    segs = [sgn * base] * n
    segs[-1] += sgn * (take - base * n)      # 尾段吸收取整余数
    segs = [p for p in segs if p != 0]
    return segs, total - sgn * take


def stand_turn_step(active: bool, d_deg: float,
                    cfg: SteerConfig = DEFAULT_CONFIG) -> tuple[bool, str]:
    """大角度站定转的**纯状态机**，每帧推进一步。

    语义::

        进入: |Δ| > turn_stand_deg     (60°)
        退出: |Δ| <= turn_resume_deg    (20°)  <- 滞回，带宽 40°

    :param active: 本帧之前是否处于站定模式。
    :param d_deg: 本帧的 Δ（**必须**有值）。
    :returns: ``(新 active, 事件)``，事件取 ``"start"`` / ``"done"`` / ``""``。

    ⚠️ 这里**没有**窗口安全阀：安全阀要能在 ``d_deg is None`` 时也触发
    （窗口见底那一刻正是没有 Δ 的时候），所以它必须留在调用方 ——
    放进这个纯函数就变成了永远走不到的死代码。
    """
    if not active:
        return ((True, "start") if abs(d_deg) > cfg.turn_stand_deg
                else (False, ""))
    # 已在站定：只有降到恢复阈值才退出。中间那一大段（20°~60°）**绝不退出**
    # —— 这就是滞回防抖的全部意义。
    if abs(d_deg) <= cfg.turn_resume_deg:
        return False, "done"
    return True, ""


def dynamic_max_step_px(px_to_world: float, dt_avg: float,
                        cfg: SteerConfig = DEFAULT_CONFIG) -> float:
    """按**实测帧间隔**算这一帧允许的帧间位移上限（小地图像素）。

    ★ 为什么必须动态
    ----------------
    ``TileTracker.MAX_STEP_PX`` 是**每帧**上限，而「每帧位移」正比于帧
    间隔：同样跑 11 m/s，200ms 一帧是 5.7px，1.6fps 一帧就是 17.7px。
    写死 50 在已观测帧率下够用，但主循环再掉就会顶到天花板 ——
    而掉帧恰恰是本项目反复遇到的事。

    ★ 比例读不出来（<=0）时走**保守**分支取下限。反过来（放行到上限）
    会让一个坏配置把跳变检测整个放开。
    """
    if not px_to_world or float(px_to_world) <= 0.0:
        return cfg.step_limit_min_px
    px_per_s = cfg.speed_run_mps / float(px_to_world)
    raw = px_per_s * max(1e-6, float(dt_avg)) * cfg.step_limit_safety
    return max(cfg.step_limit_min_px, min(cfg.step_limit_max_px, raw))


# ==================================================================
# 顶层决策
# ==================================================================


@dataclass(frozen=True)
class SteerPlan:
    """一帧的转向/行走建议。**纯数据**，不含动作。"""

    #: ``"ARRIVED"`` / ``"TURN"`` / ``"WALK"`` / ``"BLIND_WALK"``
    action: str
    #: 到目标的距离（米）。
    dist_m: float
    #: 目标方位角（度）。
    bearing: float
    #: 夹角 Δ（度）= 目标方位角 − 当前朝向。判不出朝向时为 ``None``。
    delta_deg: float | None
    #: 本帧总共该下发的鼠标像素（带符号；``action != "TURN"`` 时为 0）。
    turn_px: float
    #: 本帧要发的分段。
    segments: tuple[int, ...]
    #: 发不掉的余量像素，**退回队列**，下一帧继续。
    owed_px: int
    #: 该不该按住前进键。
    walk: bool
    #: 该不该按住冲刺键。
    sprint: bool
    #: **下一帧**的站定状态（跨帧记忆，调用方负责存回去）。
    stand_active: bool
    #: 本帧的状态机事件：``"start"`` / ``"done"`` / ``""``。
    event: str
    #: 站定期间是否处于安全阀触发状态（窗口快见底，该走一帧）。
    #: 纯计算，**是否真走一帧由调用方决定**（见 :func:`stand_turn_step` 警告）。
    window_blind: bool
    #: ★ 站定阶段的**剩余转向预算**（度）。``0`` = 不在站定 / 预算已花完。
    #:
    #: 调用方必须把它**跨帧传回** :func:`steer` 的 ``stand_budget_deg``，
    #: 否则每帧都会重新按当前（站定期间是陈旧的）Δ 重算预算 —— 那正是
    #: 2026-10-14 修掉的振荡：站定时角色不动 -> heading 冻结 -> Δ 不变
    #: -> 每帧都发满转向 -> 冲过头 -> 反向 -> 振荡。
    stand_budget_deg: float = 0.0
    #: 预算花完、强制结束站定的那一帧为 ``True``。此时 ``walk`` 必然为
    #: ``True`` —— 走一帧才能让朝向重新可观测。
    budget_exhausted: bool = False

    @property
    def segment_total(self) -> int:
        """本帧实际发出的总像素（各段之和）。"""
        return sum(self.segments)


def steer(pos_x: float, pos_z: float,
          target_x: float, target_z: float,
          heading: float | None,
          *,
          arrive_radius_m: float | None = None,
          stand_active: bool = False,
          want_sprint: bool = False,
          window_move_m: float | None = None,
          stand_budget_deg: float = 0.0,
          cfg: SteerConfig = DEFAULT_CONFIG) -> SteerPlan:
    """"我在哪 / 目标在哪 / 我朝哪" -> "这一帧怎么走"。

    :param pos_x: 当前 x（世界单位 / 米）。
    :param pos_z: 当前 z。
    :param target_x: 目标 x。
    :param target_z: 目标 z。
    :param heading: 当前朝向（度，``+x 为 0°``）。``None`` = **判不出**
        （位移窗口见底），此时按距离盲走（``BLIND_WALK``）。
    :param arrive_radius_m: 到达半径；``None`` 用 ``cfg.arrive_radius_m``。
    :param stand_active: 上一帧的站定状态（**必须由调用方跨帧传回来**）。
    :param want_sprint: 调用方是否想冲刺（角色是否够体力它自己知道）。
    :param window_move_m: 最近窗口的净位移（米）；给 ``None`` 时安全阀
        不触发。用于站定期的「窗口保命」出口。
    :param stand_budget_deg: 上一帧 :attr:`SteerPlan.stand_budget_deg`。
        **站定阶段的关键状态 —— 必须由调用方跨帧传回。**

    ★ 站定为什么必须**开环**
    ---------------------
    站定时角色**不动**，而 ``heading`` 只能从位移窗口反推 ->
    站定期间 heading 是**冻结**的。若每帧都按 ``Δ = 目标方位 - 陈旧朝向``
    重算转向量，整个站定期的 Δ 几乎不变，于是每帧都发同样大的转向，
    角色必然冲过头；等它走起来、朝向终于刷新时，早就转过头了
    -> 反向再转 -> **振荡**。

    所以：进入站定时按当时的 |Δ| 算一次**总预算**，之后每帧只从预算里
    扣，扣完就**强制走一帧**重新观测。非站定的转向是新鲜 heading 上的
    闭环反馈，不受预算约束。
    """
    radius = cfg.arrive_radius_m if arrive_radius_m is None \
        else float(arrive_radius_m)

    dx = float(target_x) - float(pos_x)
    dz = float(target_z) - float(pos_z)
    dist = math.hypot(dx, dz)
    bear = math.degrees(math.atan2(dz, dx))

    if dist <= radius:
        return SteerPlan(
            action="ARRIVED", dist_m=dist, bearing=bear, delta_deg=None,
            turn_px=0.0, segments=(), owed_px=0, walk=False, sprint=False,
            stand_active=False, event="", window_blind=False,
            stand_budget_deg=0.0)

    # ── 判不出朝向：盲走，但**不静默当它对准了** ──────────────────
    if heading is None:
        return SteerPlan(
            action="BLIND_WALK", dist_m=dist, bearing=bear, delta_deg=None,
            turn_px=0.0, segments=(), owed_px=0,
            walk=True, sprint=bool(want_sprint),
            stand_active=False, event="", window_blind=True,
            stand_budget_deg=0.0)

    delta = wrap_deg(bear - float(heading))

    # ── 站定状态机（纯滞回，无 IO）───────────────────────────────
    now_stand, event = stand_turn_step(stand_active, delta, cfg)

    blind = (window_move_m is not None
             and float(window_move_m) < cfg.window_move_min_m)

    # ── 死区内：直行 ────────────────────────────────────────────
    if abs(delta) <= cfg.turn_deadzone_deg:
        return SteerPlan(
            action="WALK", dist_m=dist, bearing=bear, delta_deg=delta,
            turn_px=0.0, segments=(), owed_px=0,
            walk=True, sprint=bool(want_sprint),
            stand_active=now_stand, event=event, window_blind=blind,
            stand_budget_deg=0.0)

    # ── 需要转 ─────────────────────────────────────────────────
    _want, _cap, px = turn_plan(delta, cfg)

    # ★ 站定：从预算里扣，而不是每帧重算（见函数 docstring）。
    budget = float(stand_budget_deg)
    if now_stand:
        if event == "start":
            # 刚进站定：按**此刻**的 |Δ| 开一次预算。
            budget = abs(delta) * cfg.stand_budget_ratio
        # ⚠️⚠️ **单位**：``px`` 是像素、``budget`` 与 ``cap_deg`` 是度。
        #    这里曾经写成 ``min(abs(px), allow_deg)``，于是 578px 被
        #    22.4「度」截成 22.4**px** —— 每帧只转 1.5° 而不是 22.4°，
        #    转向慢了 15 倍。**度 -> 像素必须除以 mouse_deg_per_px。**
        cap_px = cfg.stand_step_cap_deg() / cfg.mouse_deg_per_px
        allow_px = 0.0 if budget <= 0.0 else min(
            cap_px,          # 单帧上限
            budget / cfg.mouse_deg_per_px,   # 预算上限
            abs(px),         # 本帧本来就没打算转更多
        )
        px = math.copysign(allow_px, px)
        # 记实际发出的（用度记账，跨帧可比；像素会被取整吃掉）。
        sent_deg = abs(px) * cfg.mouse_deg_per_px
        budget = max(0.0, budget - sent_deg)
        if budget <= 0.0:
            # ★ 预算花完 -> **强制结束站定并走一帧**。
            # 不走这一帧，朝向就永远刷新不了，下一轮又按陈旧值转，
            # 于是原地振荡 —— 这正是 2026-10-14 闭环仿真抓到的死法。
            now_stand = False
            event = "done"

    seg_n = cfg.turn_stand_segments if now_stand else None
    segs, owed = split_turn(px, seg_n, cfg)

    # ★★ **边走边转**（原则 5）：只有**站定**才停步。
    #
    # ⚠️ 2026-10-14 修一个会让整条寻路死锁的错。
    #   原先这里写的是 ``walk = blind and now_stand``，等于
    #   「站定期间且窗口将死时才走」—— 也就是**平时转向一律不走路**。
    #   后果：角色每帧只转一点点、从不前进，于是朝向窗口（它唯一的
    #   数据来源）永远攒不起来 -> ``heading`` 恒为 ``None`` ->
    #   退化成盲走 -> 再也判不出真实夹角。闭环表现为**原地无限打转**。
    #   真机仿真（2026-10-14）：目标在北 30m，转了 1498 帧、一步没走。
    #   之所以 72 条单测全绿，是因为它们**逐个测函数**，从没跑过闭环。
    #
    # 正确语义：站定（|Δ|>60°）才停步；否则一律边走边转。
    # 两个例外出口，都**必须**走：窗口快见底（安全阀）、预算花完。
    # 否则站定逻辑会因判据失效而静默失效。
    exhausted = (budget <= 0.0) and not now_stand and event == "done"
    walk = (not now_stand) or blind or exhausted

    return SteerPlan(
        action="TURN", dist_m=dist, bearing=bear, delta_deg=delta,
        turn_px=px, segments=tuple(segs), owed_px=owed,
        walk=walk, sprint=False,
        stand_active=now_stand, event=event, window_blind=blind,
        stand_budget_deg=budget, budget_exhausted=exhausted)
