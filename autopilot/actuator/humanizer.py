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
    humanizer: Humanizer | None = None,
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


__all__ = [
    "Humanizer",
    "HumanizerConfig",
    "angle_diff",
    "apply_turn",
]
