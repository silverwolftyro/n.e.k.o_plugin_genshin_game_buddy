"""人类化处理：让输入时序不像机器人。

对应架构文档 6.7。

**BetterGI 明确没有这些** `[实测]`：`Core/Simulator/` 全目录 grep
`Random|Bezier|Curve|jitter` → **0 命中**。它的转视角全是
`MoveMouseBy(dx, dy)` 直线跳，按键时长零抖动，所有 `Delay(xx)` 都是字面量。

3.0 为什么加：
    1. **未来-proof** —— 目标换成有反作弊的竞技游戏时这是必需品
    2. **减少极端时序导致的游戏内异常** —— 固定 16ms 间隔可能触发丢帧

**但不要过度投入**：v1 只做三项最有效的（下表），贝塞尔轨迹放 v2。

| 项 | 幅度 | 为什么 |
|---|---|---|
| 按键保持时长 | × U(0.85, 1.15) | 固定时长的按键间隔最容易被统计 |
| 动作间隔 | × U(0.8, 1.25) | 间隔比时长更明显 |
| 转视角抖动 | + U(-2°, 2°) | 抄 BetterGI 唯一接近的抖动（`±1/12 幅度`）的思路 |
| 点击位置抖动 | + U(-3, 3) px | 避免像素级重复点击同一点 |

**随机源可注入**（`rng` 参数），这样单测可以固定种子、可复现。
"""

from __future__ import annotations

import random
from dataclasses import dataclass



@dataclass(frozen=True)
class HumanizerConfig:
    """人类化参数。全部默认开启，幅度可调，设 0 即关闭。"""

    enabled: bool = True

    #: 按键保持时长抖动：hold_ms × U(1-jitter, 1+jitter)
    hold_jitter: float = 0.15
    #: 动作间隔抖动：delay_ms × U(1-jitter, 1+jitter)
    delay_jitter: float = 0.25
    delay_floor: float = 0.8
    delay_ceil: float = 1.25

    #: 转视角目标角度抖动（度）
    heading_jitter_deg: float = 2.0
    #: 鼠标点击位置抖动（像素）
    click_jitter_px: int = 3
    #: 鼠标转视角每步的抖动（度），拆成多步时逐步累加
    turn_step_jitter_deg: float = 0.8

    #: 兜底下限 / 上限（毫秒），避免抖动后变成 0 或离谱长
    min_hold_ms: int = 8
    max_hold_ms: int = 5000
    min_delay_ms: int = 10
    max_delay_ms: int = 10_000


class Humanizer:
    """给意图执行器提供"带抖动"的时序与坐标。

    用法::

        h = Humanizer(rng=random.Random(42))   # 固定种子便于测试
        h.hold_ms(100)          # -> 100 * U(0.85,1.15)
        h.delay_ms(500)         # -> 500 * U(0.8,1.25)
        h.jitter_heading(90.0)  # -> 90.0 ± 2
    """

    def __init__(
        self,
        config: HumanizerConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.cfg = config or HumanizerConfig()
        self._rng = rng or random.Random()

    # -- 时序 ------------------------------------------------------------

    def hold_ms(self, hold_ms: int) -> int:
        """按键保持时长加抖动。"""
        base = max(0, int(hold_ms))
        if not self.cfg.enabled or self.cfg.hold_jitter <= 0:
            return base
        lo = 1.0 - self.cfg.hold_jitter
        hi = 1.0 + self.cfg.hold_jitter
        v = int(round(base * self._rng.uniform(lo, hi)))
        return int(max(self.cfg.min_hold_ms, min(self.cfg.max_hold_ms, v)))

    def delay_ms(self, delay_ms: int) -> int:
        """动作间隔加抖动。"""
        base = max(0, int(delay_ms))
        if not self.cfg.enabled or self.cfg.delay_jitter <= 0:
            return base
        v = int(round(base * self._rng.uniform(self.cfg.delay_floor, self.cfg.delay_ceil)))
        return int(max(self.cfg.min_delay_ms, min(self.cfg.max_delay_ms, v)))

    def extra_delay_ms(self, base_ms: int = 0) -> int:
        """额外的随机停顿（0 基线上抖动）。用于两个动作之间的"喘口气"。

        ★★ 入参先钳到非负（2026-10-13 修）
          原来钳位写在 ``randint`` **之后**：负的 ``base_ms`` 会让
          ``randint(0, int(base * jitter))`` 拿到负上界，直接抛
          ``ValueError: empty range`` —— 一个「多点一下就崩」的边界。
          负停顿没有物理意义，夹到 0 才是对的答案。
        """
        base = max(0, int(base_ms))
        if not self.cfg.enabled or self.cfg.delay_jitter <= 0:
            return base
        v = base + self._rng.randint(0, int(base * self.cfg.delay_jitter) or 1)
        return int(max(self.cfg.min_delay_ms, min(self.cfg.max_delay_ms, v)))

    # -- 坐标 ------------------------------------------------------------

    def jitter_heading(self, heading_deg: float) -> float:
        """转视角目标角度抖动。返回 [-360, 360) 内的角度。"""
        h = float(heading_deg)
        if not self.cfg.enabled or self.cfg.heading_jitter_deg <= 0:
            return h % 360.0
        h += self._rng.uniform(-self.cfg.heading_jitter_deg,
                               self.cfg.heading_jitter_deg)
        return h % 360.0

    def jitter_click(self, x: int, y: int) -> tuple[int, int]:
        """点击位置抖动（像素）。"""
        if not self.cfg.enabled or self.cfg.click_jitter_px <= 0:
            return int(x), int(y)
        r = self.cfg.click_jitter_px
        return (int(x) + self._rng.randint(-r, r),
                int(y) + self._rng.randint(-r, r))

    def turn_steps(self, total_deg: float, *, gain: float = 1.0) -> list[int]:
        """把一次转视角拆成多步，每步带抖动。返回每步的相对位移（像素量级）。

        原则 5：转向必须在移动中完成，所以转视角是**边走边发**的，
        拆步能让"边走边转"看起来更像人在慢慢调整。
        """
        total = float(total_deg) * float(gain)
        if not self.cfg.enabled or self.cfg.turn_step_jitter_deg <= 0:
            return [int(round(total))]
        step = max(1.0, self.cfg.turn_step_jitter_deg * 2.0)
        n = max(1, int(abs(total) / step))
        out: list[int] = []
        per = total / n
        for _ in range(n):
            out.append(int(round(per + self._rng.uniform(
                -self.cfg.turn_step_jitter_deg, self.cfg.turn_step_jitter_deg
            ))))
        return out

    def sleep(self, ms: int) -> None:
        """按抖动后的时长真实睡眠。"""
        import time  # noqa: PLC0415

        d = self.delay_ms(ms)
        if d > 0:
            time.sleep(d / 1000.0)

    # -- 自检 ------------------------------------------------------------

    def describe(self) -> str:
        c = self.cfg
        if not c.enabled:
            return "人类化：关闭"
        return (
            f"人类化：开  时长±{c.hold_jitter:.0%}  "
            f"间隔×{c.delay_floor:.2f}~{c.delay_ceil:.2f}  "
            f"转向±{c.heading_jitter_deg:.1f}°  点击±{c.click_jitter_px}px"
        )


def angle_diff(target: float, current: float) -> float:
    """角度差，返回 (-180, 180]。current 需要转到 target 还差多少度。"""
    d = (float(target) - float(current) + 180.0) % 360.0 - 180.0
    return d


def apply_turn(
    simulator,
    heading_deg: float,
    current_heading: float,
    *,
    humanizer: "Humanizer | None" = None,
    dpi_gain: float = 1.0,
    move_key: int | None = None,
    sim: "object | None" = None,
) -> float:
    """边走边把视角转到 ``heading_deg``。返回转完后的近似朝向。

    **原则 5 的落地**：``move_key`` 非空时先按住 W（移动），再发鼠标相对移动。
    停下就转不动 —— 停下就没法转镜头 → 没法产生新观测 → 定位停滞 → 死锁。
    """
    s = sim if sim is not None else simulator
    h = humanizer or Humanizer(HumanizerConfig(enabled=False))
    target = h.jitter_heading(heading_deg)
    steps = h.turn_steps(angle_diff(target, current_heading), gain=dpi_gain)
    for px in steps:
        if px == 0:
            continue
        s.mouse_move_relative(int(px), 0)
    return (current_heading + sum(steps)) % 360.0


# ==================================================================
# 节奏门（2026-10-16 加，B1）
# ==================================================================

#: 默认受节奏门管辖的动作。
#:
#: ★ 为什么只管这一个
#:   B1 的现象是「连击像机器」—— 355 次普攻 / 76.8 秒，均值 0.217s、
#:   **标准差仅 0.089s**。那是普攻独有的问题：
#:     · 闪避/技能本来就带冷却（traits 里 1500ms 之类），天然不均匀；
#:     · 移动是持续按住，本来就没有"节奏"可言。
#:   把它们一起管只会平白增加延迟，不改任何观感。
#:
#: ⚠️ **刻意不含任何有冷却语义的动作**：节奏门只决定"这一 tick 要不要打"，
#:    不参与也不改变冷却。两者是独立的闸门，串在一起会出现
#:    "冷却刚结束又被节奏门挡一次"的双重等待。
DEFAULT_GATED_ACTIONS: frozenset[str] = frozenset({"NORMAL_ATTACK"})


@dataclass(frozen=True)
class CadenceConfig:
    """连击节奏门参数。

    ★ 分布形状（这是全部意义所在）
    ------------------------------
    真机实测的连击间隔标准差只有 0.089s —— 几乎是一台节拍器。
    要让它像人，**不需要**均值变，只需要**方差变大**：
    大部分间隔短而均匀，偶尔来一次长的（手酸、换目标、重新对准）。

    ★ 为什么 ``lo_ms`` 默认是 0 而不是 200（2026-10-16 实测后改的）
    -------------------------------------------------------
    主循环是 200ms 的**离散栅格**：门抽一个间隔，实际只能在
    「下一个 tick」(200ms) 或「下下个 tick」(400ms) 里挑。

    于是区间下界放在哪里，决定了「有多少比例走 1 个 tick」：

    ==============  ==================================  ==========
    区间           走 1 tick 的比例                        实际均值
    ==============  ==================================  ==========
    U(200, 350)    **≈0**（只有恰好 200 才走 1 tick）      ≈400ms
    U(0, 300)      2/3                                    ≈290ms
    ==============  ==================================  ==========

    最初按「200~350ms 随机分布」直接写成 ``U(200, 350)``，结果**攻击频率
    直接砍半**（实测 2.13 倍慢），被
    ``tests/unit/test_cadence_gate.py::test_human_cadence_does_not_collapse_the_dps``
    拦下来了。所以 ``lo_ms = 0`` 的含义是「不额外等待，走下一个 tick」。

    ⚠️ 栅格量化是**固有的**，不是实现瑕疵：想要方差就必须有「跨过一个 tick」
       的时刻，而那必然拉长均值。本门把均值拉长控制在 1.5 倍以内，
       攻击频率仍有 3 次/秒以上 —— 比真人连打快，但观感不再是节拍器。

    :param lo_ms: 间隔下界。**0 = 不额外等待**
    :param hi_ms: 间隔上界
    :param fatigue_p: 出现一次"喘口气"的概率
    :param fatigue_lo_ms: 喘口气的间隔下界
    :param fatigue_hi_ms: 喘口气的间隔上界
    """

    enabled: bool = True
    lo_ms: int = 0
    hi_ms: int = 300
    fatigue_p: float = 0.05
    fatigue_lo_ms: int = 450
    fatigue_hi_ms: int = 950

    def __post_init__(self) -> None:
        # 参数写反时宁可夹住也不要静默产生负区间
        if self.lo_ms > self.hi_ms:
            raise ValueError(
                f"CadenceConfig: lo_ms({self.lo_ms}) 不能大于 hi_ms({self.hi_ms})")
        if self.fatigue_lo_ms > self.fatigue_hi_ms:
            raise ValueError(
                f"CadenceConfig: fatigue_lo_ms({self.fatigue_lo_ms}) "
                f"不能大于 fatigue_hi_ms({self.fatigue_hi_ms})")
        if not 0.0 <= self.fatigue_p <= 1.0:
            raise ValueError(
                f"CadenceConfig: fatigue_p({self.fatigue_p}) 必须在 0~1 之间")


class CadenceGate:
    """按动作记录"下一次允许发招的时刻"。

    ★ 两段式 API，而且 ``allows`` 是**纯函数**
    --------------------------------------------
    :meth:`allows` 只回答"现在到点了没"，**绝不推进状态**；
    真正推进发生在 :meth:`note_fired`，由调用方在**动作真的发出之后**调用。

    为什么要拆开：规则的求值顺序是「先判条件、再决定发不发」。
    如果 ``allows`` 自己推进，那么"条件为假"的那一轮也会白白消耗掉一次
    机会 —— 敌人一消失，下次打出来就得多等一个完整间隔，手感会莫名其妙变慢。
    只在真发出时记账，才是"上一次出招到现在过了多久"的真实语义。

    ★ 主循环的 tick 是 200ms，所以实际间隔是 200ms 的倍数
    --------------------------------------------------
    抽出来的是连续值，落到 200ms 栅格上只有 200 / 400 / 600… 三档。
    **这不是缺陷**：目标从来不是"精确命中 350ms"，而是把
    「恒 200ms」变成「多数 200、偶尔 400、偶尔 600+」的混合分布。
    真正决定观感的是**标准差**，不是均值。量化带来的均值代价见
    :class:`CadenceConfig` 的类文档。

    ⚠️ 跨 run 不重置是**对的**：停了十分钟再启动，`now` 远大于
    ``next_due``，第一次调用直接放行，不需要额外状态。
    """

    def __init__(
        self,
        config: CadenceConfig | None = None,
        *,
        actions: frozenset[str] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.cfg = config or CadenceConfig()
        self.actions = actions if actions is not None else DEFAULT_GATED_ACTIONS
        self._rng = rng or random.Random()
        #: 动作 -> 下一次允许发招的绝对时刻（ms）
        self._next_due_ms: dict[str, int] = {}

    # ------------------------------------------------------------------
    def governs(self, action: str) -> bool:
        """这个动作是否受本门管辖。"""
        return self.cfg.enabled and action in self.actions

    def allows(self, action: str, now_ms: int) -> bool:
        """现在允许发这个动作吗？**纯判断，不改状态。**"""
        if not self.governs(action):
            return True
        due = self._next_due_ms.get(action)
        # 从没打过 -> 立刻放行（暖机）。第一下不能等。
        return due is None or int(now_ms) >= due

    def note_fired(self, action: str, now_ms: int) -> None:
        """动作**真的发出**了，记账并抽下一次的时刻。"""
        if not self.governs(action):
            return
        c = self.cfg
        if self._rng.random() < c.fatigue_p:
            gap = self._rng.randint(c.fatigue_lo_ms, c.fatigue_hi_ms)
        else:
            gap = self._rng.randint(c.lo_ms, c.hi_ms)
        self._next_due_ms[action] = int(now_ms) + gap

    # ------------------------------------------------------------------
    def next_due_ms(self, action: str) -> int | None:
        """下一次允许的时刻；``None`` = 从没打过。给诊断用。"""
        return self._next_due_ms.get(action)

    def describe(self) -> str:
        if not self.cfg.enabled:
            return "节奏门：关闭"
        c = self.cfg
        return (
            f"节奏门：开  常规 {c.lo_ms}~{c.hi_ms}ms  "
            f"喘气 {c.fatigue_p:.0%}→{c.fatigue_lo_ms}~{c.fatigue_hi_ms}ms  "
            f"管辖 {','.join(sorted(self.actions)) or '（无）'}"
        )


__all__ = [
    "CadenceConfig",
    "CadenceGate",
    "Humanizer",
    "HumanizerConfig",
    "angle_diff",
    "apply_turn",
]
