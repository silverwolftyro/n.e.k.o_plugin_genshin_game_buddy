"""事件：本地结构化事件流。

对应架构文档 5.3。事件是「响亮失败」的载体 —— 识别失败不能只返回 None，
必须产生一个**带诊断信息**的事件，UI 能显示、日志能追溯。

关键设计：``context`` 携带诊断数据。识别失败时用户看到的不是"识别失败"，
而是"最高连续命中 2 次，阈值要 3 次"（原则 9）。
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

_event_counter = itertools.count(1)


class Severity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class EventCategory(str, Enum):
    PERCEPTION = "PERCEPTION"
    DECISION = "DECISION"
    EXECUTION = "EXECUTION"
    CONNECTION = "CONNECTION"
    SYSTEM = "SYSTEM"


def _xy_ctx(x: float, y: float) -> dict[str, float]:
    """事件 context 的坐标部分：**``y`` 规范 + ``z`` 兼容镜像**。

    两个键放**同一个值**。这看着冗余，但事件流是落盘的对外契约 ——
    历史 ``.stream.jsonl`` 与外部读侧都按 ``z`` 取。少写一个 ``z``，
    旧消费者就会拿到 ``None`` 而不是报错，那才是真事故。
    """
    xr = round(float(x), 1)
    yr = round(float(y), 1)
    return {"x": xr, "y": yr, "z": yr}


@dataclass(frozen=True)
class Event:
    """结构化事件。"""

    severity: Severity
    category: EventCategory
    code: str
    message: str
    context: dict[str, Any]
    event_id: str = ""
    timestamp_ms: int = 0
    state_version: int = 0

    def __post_init__(self) -> None:
        # frozen dataclass 的字段补默认值走 object.__setattr__
        if not self.event_id:
            object.__setattr__(self, "event_id", f"evt_{next(_event_counter):06d}")
        if not self.timestamp_ms:
            object.__setattr__(self, "timestamp_ms", int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp_ms": self.timestamp_ms,
            "severity": self.severity.value,
            "category": self.category.value,
            "code": self.code,
            "message": self.message,
            "context": self.context,
            "state_version": self.state_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self) -> str:
        ctx = " ".join(f"{k}={v}" for k, v in self.context.items())
        return f"[{self.severity.value}] {self.category.value}/{self.code}: {self.message} {ctx}".rstrip()

    # -- 工厂：把常见的失败做成具名构造，调用点读起来像句子 ----------------

    @classmethod
    def template_score_low(
        cls,
        item: str,
        max_consecutive: int,
        required: int,
        last_score: float,
        threshold: float,
        state_version: int = 0,
    ) -> "Event":
        """模板匹配连续低分。

        携带 max_consecutive / required，让用户知道**差多少**、**怎么调**，
        而不是只看到"识别失败"（原则 9）。
        """
        return cls(
            severity=Severity.WARN,
            category=EventCategory.PERCEPTION,
            code="TEMPLATE_SCORE_LOW",
            message=f"{item} 连续 {max_consecutive} 帧低分",
            context={
                "item": item,
                "max_consecutive_count": max_consecutive,
                "required_count": required,
                "last_score": round(last_score, 4),
                "threshold": threshold,
            },
            state_version=state_version,
        )

    @classmethod
    def ocr_failed(cls, fail_streak: int, error: str, state_version: int = 0) -> "Event":
        """OCR 连续失败。

        注意：单次 OCR 失败返回空文本是**合法结果**；只有连续失败才升级为事件。
        （2.0 的 ocr_engine.py 在 except 里静默 return "" 是要修正的做法。）
        """
        return cls(
            severity=Severity.WARN if fail_streak < 5 else Severity.ERROR,
            category=EventCategory.PERCEPTION,
            code="OCR_FAILED",
            message=f"OCR 连续失败 {fail_streak} 次",
            context={"fail_streak": fail_streak, "error": error[:200]},
            state_version=state_version,
        )

    @classmethod
    def ocr_recovered(cls, fail_streak: int) -> "Event":
        return cls(
            severity=Severity.INFO,
            category=EventCategory.PERCEPTION,
            code="OCR_RECOVERED",
            message=f"OCR 已恢复（此前连续失败 {fail_streak} 次）",
            context={"fail_streak": fail_streak},
        )

    @classmethod
    def screenshot_failed(cls, error: str, consecutive: int) -> "Event":
        sev = Severity.ERROR if consecutive >= 3 else Severity.WARN
        return cls(
            severity=sev,
            category=EventCategory.PERCEPTION,
            code="SCREENSHOT_FAIL",
            message=f"截图连续失败 {consecutive} 次",
            context={"error": error[:200], "consecutive": consecutive},
        )

    @classmethod
    def black_frame(cls, luma: float) -> "Event":
        return cls(
            severity=Severity.WARN,
            category=EventCategory.PERCEPTION,
            code="BLACK_FRAME",
            message="画面接近全黑（游戏可能最小化或切屏）",
            context={"luma": round(luma, 2)},
        )

    @classmethod
    def yolo_unavailable(cls, error: str) -> "Event":
        return cls(
            severity=Severity.ERROR,
            category=EventCategory.PERCEPTION,
            code="YOLO_UNAVAILABLE",
            message="YOLO 推理不可用",
            context={"error": error[:200]},
        )

    @classmethod
    def yolo_timeout(cls, elapsed_ms: float, budget_ms: float) -> "Event":
        return cls(
            severity=Severity.WARN,
            category=EventCategory.PERCEPTION,
            code="YOLO_TIMEOUT",
            message=f"YOLO 推理超时（{elapsed_ms:.0f}ms > {budget_ms:.0f}ms），本帧丢弃",
            context={"elapsed_ms": round(elapsed_ms, 1), "budget_ms": budget_ms},
        )

    @classmethod
    def position_lost(cls, lost_frames: int) -> "Event":
        return cls(
            severity=Severity.WARN,
            category=EventCategory.PERCEPTION,
            code="POSITION_LOST",
            message=f"定位丢失 {lost_frames} 帧，导航类意图将被拒绝",
            context={"lost_frames": lost_frames},
        )

    @classmethod
    def config_error(cls, where: str, error: str) -> "Event":
        """配置/规则加载失败。

        **加载期就该炸**，不做降级（原则 1）。调用点应当直接抛异常，
        这个工厂用于把异常信息同时送进事件流。
        """
        return cls(
            severity=Severity.ERROR,
            category=EventCategory.SYSTEM,
            code="CONFIG_ERROR",
            message=f"配置加载失败：{where}",
            context={"where": where, "error": error[:500]},
        )

    @classmethod
    def intent_rejected(cls, intent_id: str, reason: str) -> "Event":
        return cls(
            severity=Severity.INFO,
            category=EventCategory.EXECUTION,
            code="INTENT_REJECTED",
            message=f"意图被拒绝：{reason}",
            context={"intent_id": intent_id, "reason": reason},
        )

    # -- 业务事件（2026-10-08）：给「事件流 -> YUI 说话」用 ------------
    #
    # ★ 为什么单独一组
    #   ----------------
    #   上面那些都是**系统/故障**事件（截图挂了、OCR 挂了、位置丢了），
    #   它们的听众是开发者和日志。而 YUI 要说的是**游戏里发生了什么**：
    #   「走到哪了」「看见敌人了」「到了」「又卡住了」。
    #
    #   v1 的 EventCode 是**开放集合**（读侧遇到表外 code 按 INFO 处理），
    #   所以这里不需要改 schema 版本 —— 这正是当初那个设计要换来的东西。
    #
    # ★ 平面轴名统一为 ``x`` / ``y``（2026-10-14）
    #   ---------------------------------------------
    #   规范键是 ``y``，但 context 里**同时**写一份 ``z`` 兼容镜像。
    #   理由：事件流是**落盘的对外契约**，历史 ``.stream.jsonl`` 里全是
    #   ``z``，外部读侧（YUI 适配器 / 旧脚本）按 ``z`` 取。直接改名会让
    #   旧消费者静默拿到 None —— 而「坐标缺失」与「键名写错」在日志里
    #   长得一模一样，排查成本极高。
    #   读侧请优先读 ``y``；``z`` 会在契约稳定后移除。

    @classmethod
    def position_changed(cls, x: float, y: float, dist: float,
                         state_version: int = 0) -> "Event":
        """位置发生了**可感知**的移动。

        ★ 不要每帧都发
        ------------
        5Hz 的主循环每帧都发的话，5 秒就有 25 条同义事件，YUI 会被
        淹掉（适配器那边有节流兜底，但源头就不该产生）。
        调用方应当只在位移**超过阈值**时调这里（见 run_loop 的判定）。
        """
        return cls(
            severity=Severity.INFO,
            category=EventCategory.PERCEPTION,
            code="POSITION_CHANGED",
            message=f"移动到 ({x:.1f}, {y:.1f})，离目标 {dist:.1f}m",
            context={
                **_xy_ctx(x, y),
                "dist_m": round(float(dist), 1),
            },
            state_version=state_version,
        )

    @classmethod
    def enemy_detected(cls, count: int, nearest_m: float = -1.0,
                       state_version: int = 0) -> "Event":
        """检到敌人。``nearest_m < 0`` 表示距离不可知。"""
        ctx: dict[str, Any] = {"count": int(count)}
        if nearest_m >= 0:
            ctx["nearest_m"] = round(float(nearest_m), 1)
        return cls(
            severity=Severity.WARN if int(count) > 0 else Severity.INFO,
            category=EventCategory.PERCEPTION,
            code="ENEMY_DETECTED",
            message=f"附近有 {count} 个敌人",
            context=ctx,
            state_version=state_version,
        )

    @classmethod
    def pathfind_start(cls, target_x: float, target_y: float,
                       dist: float, state_version: int = 0) -> "Event":
        return cls(
            severity=Severity.INFO,
            category=EventCategory.DECISION,
            code="PATHFIND_START",
            message=f"开始前往 ({target_x:.1f}, {target_y:.1f})，{dist:.1f}m",
            context={
                "target_x": round(float(target_x), 1),
                "target_y": round(float(target_y), 1),
                # 兼容镜像：老读侧按 target_z 取。见 _xy_ctx 的说明。
                "target_z": round(float(target_y), 1),
                "dist_m": round(float(dist), 1),
            },
            state_version=state_version,
        )

    @classmethod
    def pathfind_arrived(cls, x: float, y: float, spent_s: float = 0.0,
                         state_version: int = 0) -> "Event":
        ctx: dict[str, Any] = _xy_ctx(x, y)
        if spent_s > 0:
            ctx["spent_s"] = round(float(spent_s), 1)
        return cls(
            severity=Severity.INFO,
            category=EventCategory.EXECUTION,
            code="PATHFIND_ARRIVED",
            message=f"到达 ({x:.1f}, {y:.1f})",
            context=ctx,
            state_version=state_version,
        )

    @classmethod
    def pathfind_stuck(cls, level: int, moved_m: float = 0.0,
                        state_version: int = 0) -> "Event":
        return cls(
            severity=Severity.WARN,
            category=EventCategory.EXECUTION,
            code="PATHFIND_STUCK",
            message=f"卡住了（L{level} 脱困）",
            context={
                "level": int(level), "moved_m": round(float(moved_m), 2)},
            state_version=state_version,
        )

    @classmethod
    def escape_success(cls, level: int, moved_m: float = 0.0,
                       state_version: int = 0) -> "Event":
        """脱困**成功**了 —— 之前只有「失败/继续升级」的事件，
        「走出来了」从来没被当成一件事报过，而那恰恰是用户想听的。"""
        return cls(
            severity=Severity.INFO,
            category=EventCategory.EXECUTION,
            code="ESCAPE_SUCCESS",
            message=f"L{level} 脱困成功，走出 {moved_m:.1f}m",
            context={
                "level": int(level), "moved_m": round(float(moved_m), 2)},
            state_version=state_version,
        )


__all__ = ["Event", "EventCategory", "Severity"]
