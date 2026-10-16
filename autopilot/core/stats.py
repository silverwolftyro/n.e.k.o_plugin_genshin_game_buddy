"""运行统计。**只做内存聚合，绝不每 tick 写盘。**

写盘这件事有个很诱人的错误做法：每 tick 往 JSONL 追加一行，
"这样日志完整"。实际后果是：

* 5 Hz 跑一小时 = 18000 次文件打开写 flush，感知帧预算被吃掉一大截
* 进程被 kill -9 时缓冲区里的统计全丢 —— 恰恰是最需要它的时候
* 统计和事件日志混在一起，真正想看的"刚才发生了什么"被噪声淹没

所以：**内存聚合 + 环形缓冲 + 快照时再序列化**。
要落盘就在 stop() 时一次性 dump（见 :meth:`LoopStats.snapshot`）。

统计口径（每个数字都只记一次，早先 double-count 过）
------------------------------------------------
* ``ticks``               —— 执行过完整 tick 的次数（含空转）
* ``idle_ticks``          —— 决策层没给出意图（**正常**，不是故障）
* ``consecutive_idle``    —— 当前连续空转轮数（命中即归零）
* ``intents_executed``    —— 执行器返回 OK 的次数
* ``by_action``           —— 每个动作**执行过一次**记一次
* ``by_outcome``          —— 只有最终结果，不混中间态
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopStats:
    """运行统计。给 ``--stats`` 和排障用。"""

    # ── 循环 ──────────────────────────────────────────────────
    ticks: int = 0
    started_ms: int = 0
    stopped_ms: int = 0

    # ── 各层计数 ────────────────────────────────────────────────
    states_built: int = 0
    capture_failures: int = 0
    perception_failures: int = 0
    policy_errors: int = 0
    tick_crashes: int = 0

    # ── 意图 ────────────────────────────────────────────────────
    intents_executed: int = 0
    intents_rejected: int = 0
    intents_failed: int = 0
    intents_interrupted: int = 0
    intents_timed_out: int = 0

    #: 空转轮数（决策层没有动作 —— 这是**正常**状态，不是故障）
    idle_ticks: int = 0
    #: 当前连续空转轮数。命中一次归零。持续走高 = 规则没覆盖当前场景。
    consecutive_idle: int = 0
    #: **暂停期间仍然完成的感知次数**。
    #: PAUSED 不停感知（恢复时状态要是新鲜的），但那一段的产物不进入决策 ——
    #: 所以它既不是 tick 也不是 idle，单独记，否则会被误读成"在跑决策"。
    paused_perception_ticks: int = 0

    #: 连续 tick 崩溃次数。达到阈值主循环会自动暂停。
    consecutive_crashes: int = 0

    by_action: dict[str, int] = field(default_factory=dict)
    by_outcome: dict[str, int] = field(default_factory=dict)
    by_scene: dict[str, int] = field(default_factory=dict)

    #: ★ 存 **float** 毫秒。用 int 截断会让"每 tick 快过 1ms"的情况
    #: 全部记成 0，统计里就出现 avg=0.0 这种明显不对的数字。
    tick_ms_samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=300), repr=False, compare=False)

    # ── 聚合 ───────────────────────────────────────────────────

    def record_tick(self, ms: float) -> None:
        self.ticks += 1
        self.tick_ms_samples.append(float(ms))

    def record_idle(self) -> None:
        self.idle_ticks += 1
        self.consecutive_idle += 1

    def record_paused_perception(self) -> None:
        """PAUSED 期间完成了一次感知（未进入决策）。"""
        self.paused_perception_ticks += 1

    def record_crash(self) -> None:
        self.tick_crashes += 1
        self.consecutive_crashes += 1

    def clear_crashes(self) -> None:
        self.consecutive_crashes = 0

    def record_hit(self, scene: str = "") -> None:
        """决策层给出了动作 —— **空转计数归零**。"""
        self.consecutive_idle = 0
        if scene:
            self.by_scene[scene] = self.by_scene.get(scene, 0) + 1

    def record_intent(self, action: str, outcome: str) -> None:
        """记一次意图。**每个意图只调一次**（拿到最终 outcome 时）。

        早先在"产出时"和"执行完"各调一次，导致 by_action 翻倍 ——
        统计说 10 个 USE_SKILL，实际只执行了 5 个。排障时这种数字
        最容易把人带沟里。
        """
        self.by_action[action] = self.by_action.get(action, 0) + 1
        self.by_outcome[outcome] = self.by_outcome.get(outcome, 0) + 1

        bucket = {
            "OK": "intents_executed",
            "REJECTED": "intents_rejected",
            "FAILED": "intents_failed",
            "INTERRUPTED": "intents_interrupted",
            "TIMEOUT": "intents_timed_out",
        }.get(outcome)
        if bucket:
            setattr(self, bucket, getattr(self, bucket) + 1)

    # ── 派生 ───────────────────────────────────────────────────

    @property
    def uptime_s(self) -> float:
        if not self.started_ms:
            return 0.0
        end = self.stopped_ms or int(time.time() * 1000)
        return max(0.0, (end - self.started_ms) / 1000.0)

    @property
    def avg_tick_ms(self) -> float:
        if not self.tick_ms_samples:
            return 0.0
        return sum(self.tick_ms_samples) / len(self.tick_ms_samples)

    @property
    def p95_tick_ms(self) -> float:
        if not self.tick_ms_samples:
            return 0.0
        s = sorted(self.tick_ms_samples)
        return s[min(len(s) - 1, int(len(s) * 0.95))]

    @property
    def actual_hz(self) -> float:
        """实际 tick 频率。**用来验证「设 10Hz 真的跑到 10Hz 了吗」** ——
        设了不代表跑得到：截图 80ms + 识别 40ms，一帧就吃掉一半预算。"""
        if self.uptime_s <= 0:
            return 0.0
        return self.ticks / self.uptime_s

    @property
    def health(self) -> str:
        """一句话健康度。给 banner 和告警用。"""
        if not self.started_ms:
            return "未启动"
        if self.tick_crashes:
            return f"异常（{self.tick_crashes} 次 tick 崩溃）"
        if self.perception_failures:
            return f"感知异常（{self.perception_failures} 次）"
        if self.consecutive_idle > 30:
            return f"长期空转（连续 {self.consecutive_idle} 轮无动作）"
        if self.ticks == 0:
            return "未产生 tick"
        return "正常"

    def snapshot(self) -> dict[str, Any]:
        """当前统计快照。**这是唯一该被序列化的出口**。"""
        return {
            "health": self.health,
            "uptime_s": round(self.uptime_s, 1),
            "ticks": self.ticks,
            "actual_hz": round(self.actual_hz, 2),
            "avg_tick_ms": round(self.avg_tick_ms, 3),
            "p95_tick_ms": round(self.p95_tick_ms, 3),
            "states_built": self.states_built,
            "capture_failures": self.capture_failures,
            "perception_failures": self.perception_failures,
            "policy_errors": self.policy_errors,
            "tick_crashes": self.tick_crashes,
            "intents_executed": self.intents_executed,
            "intents_rejected": self.intents_rejected,
            "intents_failed": self.intents_failed,
            "intents_interrupted": self.intents_interrupted,
            "intents_timed_out": self.intents_timed_out,
            "idle_ticks": self.idle_ticks,
            "consecutive_idle": self.consecutive_idle,
            "paused_perception_ticks": self.paused_perception_ticks,
            "by_action": dict(sorted(self.by_action.items())),
            "by_outcome": dict(sorted(self.by_outcome.items())),
            "by_scene": dict(sorted(self.by_scene.items())),
        }

    def format_report(self, title: str = "运行统计") -> str:
        """人读的一行行报告。``--stats`` 直接打这个。"""
        s = self.snapshot()
        w = 34
        lines = [
            f"── {title} " + "─" * max(0, w - len(title) - 4),
            f"  健康度        {s['health']}",
            f"  运行时长      {s['uptime_s']}s",
            f"  tick 数       {s['ticks']}  "
            f"（实际 {s['actual_hz']} Hz，平均 {s['avg_tick_ms']}ms，"
            f"P95 {s['p95_tick_ms']}ms）",
            f"  状态包构建    {s['states_built']}",
            f"  意图执行      {s['intents_executed']}  "
            f"空转 {s['idle_ticks']}（连续 {s['consecutive_idle']}）",
        ]
        if s["intents_rejected"] or s["intents_failed"] \
                or s["intents_interrupted"] or s["intents_timed_out"]:
            lines.append(
                f"  异常意图      拒绝 {s['intents_rejected']}  "
                f"失败 {s['intents_failed']}  "
                f"中断 {s['intents_interrupted']}  "
                f"超时 {s['intents_timed_out']}")
        if s["paused_perception_ticks"]:
            lines.append(
                f"  暂停期感知    {s['paused_perception_ticks']} 次"
                f"（未进入决策）")
        if s["capture_failures"] or s["perception_failures"] \
                or s["policy_errors"] or s["tick_crashes"]:
            lines.append(
                f"  故障          截图 {s['capture_failures']}  "
                f"感知 {s['perception_failures']}  "
                f"决策 {s['policy_errors']}  "
                f"崩溃 {s['tick_crashes']}")
        if s["by_action"]:
            top = sorted(s["by_action"].items(), key=lambda x: -x[1])[:6]
            lines.append("  动作分布      " + "  ".join(
                f"{a}×{n}" for a, n in top))
        if s["by_scene"]:
            top = sorted(s["by_scene"].items(), key=lambda x: -x[1])[:4]
            lines.append("  场景分布      " + "  ".join(
                f"{a}×{n}" for a, n in top))
        return "\n".join(lines)


__all__ = ["LoopStats"]
