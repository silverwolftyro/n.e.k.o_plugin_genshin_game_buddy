"""调试叠加层：把识别结果画在屏幕上。

对应架构文档原则 4「工具 = 运行时」：**画的就是本帧真实产出的框**，
不重新跑一遍识别。用户在屏幕上看框 + 看 JSONL 日志，能直接验证对不对。

用法::

    with DebugOverlay() as ov:
        while True:
            frame = cap.grab()
            state = builder.build(frame)
            if ov.render(frame, state, dets, matches) == "ESC":
                break
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from .world_entities import BASE_SIZE

logger = logging.getLogger(__name__)

#: 按标签前缀上色（BGR）
_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "main_menu": (80, 200, 80),
    "bag_inventory": (80, 160, 240),
    "bag_character": (60, 130, 220),
    "map_big": (200, 160, 60),
    "combat": (60, 60, 235),
    "dialog": (200, 120, 200),
    "loading": (140, 140, 140),
    "tpl:": (60, 220, 220),
    "char:": (100, 255, 255),
}
_DEFAULT_COLOR = (200, 200, 200)


class DebugOverlay:
    """OpenCV imshow 叠加层。"""

    def __init__(
        self,
        window_name: str = "N.E.K.O. autopilot",
        *,
        alpha: float = 0.35,
        show_labels: bool = True,
        scale: float = 1.0,
    ) -> None:
        self._win = window_name
        self._alpha = float(alpha)
        self._show_labels = show_labels
        self._scale = float(scale)
        self._opened = False

    def __enter__(self) -> "DebugOverlay":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._opened:
            try:
                cv2.destroyWindow(self._win)
            except Exception:  # noqa: BLE001
                pass
            self._opened = False

    def render(
        self,
        frame: Any,
        state: Any = None,
        detections: list | None = None,
        matches: list | None = None,
        entities: list | None = None,
    ) -> str:
        """画一帧。返回用户按键（"ESC" / "q" 表示退出，其余为空串）。

        ESC 与 q 都能退出 —— cv2.waitKey 对不同平台的键值不一致。

        :param entities: 世界实体（F2 方案，builder.last_world_entities）。
            它们的 ``bbox`` 是**归一化**的，这里按 canvas 尺寸换算回像素。
        """
        img = frame.image if hasattr(frame, "image") else frame
        if img is None or img.size == 0:
            return ""
        canvas = img.copy()
        if self._scale != 1.0:
            canvas = cv2.resize(
                canvas, None, fx=self._scale, fy=self._scale,
                interpolation=cv2.INTER_LINEAR,
            )
        k = self._scale
        H, W = canvas.shape[:2]

        for m in matches or []:
            self._draw_box(canvas, m.box, m.score, "tpl:" + m.name, k)
        for d in detections or []:
            self._draw_box(canvas, d.box, d.conf, d.label, k)

        # 世界实体（F2 方案：结果在 builder 上，不在 StatePacket 里）。
        # 这里画的是**本帧真实产出**，不重跑推理 —— 原则 4「工具 = 运行时」。
        for e in entities or []:
            # e.bbox 是归一化的，(x, y, w, h) -> 乘回画面尺寸
            # bbox 是**相对传入帧**的归一化值 -> 乘 canvas 尺寸
            ex, ey, ew, eh = (int(v * s) for v, s in
                              zip(e.bbox, (W, H, W, H)))
            self._draw_box(canvas, (ex, ey, ew, eh), e.confidence,
                           f"we:{e.kind}", k)
            # ⚠️ screen_pos 是 **720 基准**像素坐标，不是传入帧的坐标。
            #    要按基准 -> canvas 的比例换算，不能直接乘 k。
            sx = int(e.screen_pos[0] * W / BASE_SIZE[0])
            sy = int(e.screen_pos[1] * H / BASE_SIZE[1])
            cv2.circle(canvas, (sx, sy), 3, (0, 0, 255), -1)

        self._draw_hud(canvas, state, k)

        if not self._opened:
            cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
            self._opened = True
        cv2.imshow(self._win, canvas)
        key = cv2.waitKey(1) & 0xFF
        return {27: "ESC", ord("q"): "q", ord("Q"): "q"}.get(key, "")

    # -- 绘制 ------------------------------------------------------------

    @staticmethod
    def _draw_box(
        canvas: np.ndarray,
        box: tuple[int, int, int, int],
        score: float,
        label: str,
        k: float,
    ) -> None:
        x, y, w, h = (int(v * k) for v in box)
        color = _DEFAULT_COLOR
        for prefix, c in _LABEL_COLORS.items():
            if label.startswith(prefix):
                color = c
                break
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            canvas, f"{label} {score:.2f}", (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    def _draw_hud(self, canvas: np.ndarray, state: Any, k: float) -> None:
        if state is None:
            return
        sc = getattr(state, "scene", None)
        rs = getattr(state, "resource", None)
        lines = [
            f"scene: {getattr(sc, 'id', '?')}  conf={getattr(sc, 'confidence', 0):.2f}",
            f"sv: {getattr(state, 'state_version', 0)}   "
            f"fps: {getattr(rs, 'fps', 0):.1f}   tick: {getattr(rs, 'tick_ms', 0)}ms",
            f"ok: {getattr(rs, 'perception_ok', True)}   "
            f"ocr: {'Y' if getattr(rs, 'ocr_available', True) else 'N'}"
            f" (streak {getattr(rs, 'ocr_fail_streak', 0)})",
            f"yolo {getattr(rs, 'yolo_ms', 0):.0f}ms   "
            f"tpl {getattr(rs, 'template_ms', 0):.0f}ms",
        ]
        pos = getattr(state, "position", None)
        if pos is not None:
            lines.append(
                f"pos: {getattr(pos, 'quality', '?')} "
                f"({getattr(pos, 'x', 0):.0f}, {getattr(pos, 'y', 0):.0f})"
            )
        else:
            lines.append("pos: (未接定位)")

        y = int(18 * k)
        for ln in lines:
            # 半透明底条保证文字可读
            (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, 1)
            cv2.rectangle(
                canvas, (0, y - th - 4), (tw + 10, y + 4), (0, 0, 0), -1
            )
            cv2.putText(
                canvas, ln, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k,
                (0, 255, 255), 1, cv2.LINE_AA,
            )
            y += int(18 * k)


__all__ = ["DebugOverlay"]
