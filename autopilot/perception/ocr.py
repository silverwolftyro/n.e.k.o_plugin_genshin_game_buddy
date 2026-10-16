"""OCR 封装（复用 2.0 的 WinRT 引擎）。

对应架构文档 6.4。

**与 2.0 的关键差异**：2.0 的 `ocr_engine.py` 在异常时 `LogWarning` + `return ""`，
注释自陈是「临时诊断（2.0.18）：暴露被 except 吞掉的异常，便于定位 OCR 静默失败」。

3.0 把这个临时诊断**变成正式机制**：
  · 连续失败计数 ``fail_streak`` 进状态包
  · 达到阈值产生 ``Event.ocr_failed``（带 streak 与错误串）
  · 降级到第二引擎或标记 ``ocr_available=False``

另外，本模块**只处理文字**。血量/能量这类数值走几何投影，不走 OCR。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .screen_region import ImageRegion

logger = logging.getLogger(__name__)

#: Windows OCR 会在汉字之间插空格，去掉（保留中英之间的空格）。
#: 与 2.0 的 ocr_engine._HAN_SPACE_RE 一致。
_HAN_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])")

#: 连续失败多少次后判定引擎不可用
DEFAULT_FAIL_THRESHOLD = 5


class TriState(Enum):
    """三态。**None/UNKNOWN 不得压缩成 bool**（原则 2）。"""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class OcrResult:
    """一次 OCR 的结果。"""

    text: str
    matched: TriState = TriState.UNKNOWN
    confidence: float = 0.0
    fail_streak: int = 0
    engine_available: bool = True
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """是否是**确认的**成功。UNKNOWN 不算成功。"""
        return self.matched is TriState.TRUE

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:200],
            "matched": self.matched.value,
            "fail_streak": self.fail_streak,
            "engine_available": self.engine_available,
        }


class OcrEngine:
    """OCR 门面。

    优先用 2.0 已有的 WinRT 引擎（零安装成本）；不可用时可降级到 RapidOCR。
    本类**只做同步调用**，async 版本见 ``aread``。
    """

    def __init__(
        self,
        *,
        legacy_engine: Any = None,
        fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
        allow_rapid_fallback: bool = True,
    ) -> None:
        """``legacy_engine`` 传 2.0 的 ``OcrEngine`` 实例以复用其 WinRT 封装。"""
        self._legacy = legacy_engine
        self._rapid: Any = None
        self._allow_rapid = allow_rapid_fallback
        self._fail_streak = 0
        self._fail_threshold = int(fail_threshold)
        self._available = True
        self._last_error = ""

    # -- 引擎管理 --------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    @property
    def fail_streak(self) -> int:
        return self._fail_streak

    @property
    def last_error(self) -> str:
        return self._last_error

    def warmup(self) -> None:
        """预热引擎，避免首帧把 WinRT 初始化耗时算进 tick。"""
        try:
            if self._legacy is not None and hasattr(self._legacy, "warmup"):
                self._legacy.warmup()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR warmup 失败: %s", exc)

    def close(self) -> None:
        if self._legacy is not None and hasattr(self._legacy, "close"):
            try:
                self._legacy.close()
            except Exception:
                pass
        self._legacy = None
        self._rapid = None

    def _ensure_rapid(self) -> bool:
        if self._rapid is not None:
            return True
        if not self._allow_rapid:
            return False
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415

            self._rapid = RapidOCR()
            logger.info("RapidOCR 降级引擎就绪")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("RapidOCR 不可用: %s", exc)
            return False

    # -- 识别 ------------------------------------------------------------

    def read(
        self,
        region: ImageRegion | np.ndarray,
        *,
        contains_any: list[str] | None = None,
        contains_all: list[str] | None = None,
        regex: list[str] | None = None,
        replace: dict[str, list[str]] | None = None,
    ) -> OcrResult:
        """读 ROI 的文字并做匹配判定。

        三种匹配只能给一种。都没给时只返回原始文本（matched=UNKNOWN）。
        """
        img = region.image if isinstance(region, ImageRegion) else region
        t0 = time.perf_counter()
        text = ""
        ok = False
        err = ""

        try:
            text, ok = self._do_read(img)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"

        elapsed = (time.perf_counter() - t0) * 1000

        if not ok:
            self._fail_streak += 1
            self._last_error = err or "engine returned empty"
            if self._fail_streak >= self._fail_threshold:
                self._available = False
            return OcrResult(
                text="",
                matched=TriState.UNKNOWN,
                fail_streak=self._fail_streak,
                engine_available=self._available,
                elapsed_ms=elapsed,
            )

        if self._fail_streak:
            logger.info("OCR 恢复（此前连续失败 %d 次）", self._fail_streak)
        self._fail_streak = 0

        text = _HAN_SPACE_RE.sub("", text)
        if replace:
            text = _apply_replace(text, replace)

        matched = _match(text, contains_any, contains_all, regex)
        return OcrResult(
            text=text,
            matched=matched,
            confidence=1.0 if matched is TriState.TRUE else 0.0,
            fail_streak=0,
            engine_available=True,
            elapsed_ms=elapsed,
        )

    async def aread(self, region: ImageRegion | np.ndarray, **kwargs) -> OcrResult:
        """异步包装。2.0 的 WinRT 引擎本身是 async 的。"""
        img = region.image if isinstance(region, ImageRegion) else region
        if self._legacy is not None and hasattr(self._legacy, "extract_text_from_base64"):
            # 走 2.0 路径：PIL/BGR -> PNG -> base64 -> WinRT
            import base64  # noqa: PLC0415

            b64 = base64.b64encode(_encode_png(img)).decode()
            res = await self._legacy.extract_text_from_base64(b64)
            if not res:
                self._fail_streak += 1
                return OcrResult(text="", matched=TriState.UNKNOWN, fail_streak=self._fail_streak)
            self._fail_streak = 0
            text = _HAN_SPACE_RE.sub("", res)
            if kwargs.get("replace"):
                text = _apply_replace(text, kwargs["replace"])
            return OcrResult(
                text=text,
                matched=_match(text, kwargs.get("contains_any"),
                              kwargs.get("contains_all"), kwargs.get("regex")),
            )
        import asyncio  # noqa: PLC0415

        return await asyncio.to_thread(self.read, img, **kwargs)

    def _do_read(self, img: np.ndarray) -> tuple[str, bool]:
        """先试 WinRT（2.0 引擎），失败降级 RapidOCR，再失败返回空。"""
        if self._legacy is not None:
            if hasattr(self._legacy, "extract_text_from_base64"):
                # 同步路径：直接调 2.0 的底层（若存在）
                fn = getattr(self._legacy, "_read_sync", None)
                if callable(fn):
                    text = fn(img)
                    if text:
                        return text, True
            elif hasattr(self._legacy, "read"):
                text = self._legacy.read(img)
                if text:
                    return text, True

        if self._ensure_rapid():
            res, _ = self._rapid(img)
            if res:
                lines = [r[1] for r in res if len(r) > 1]
                return "\n".join(lines), True

        return "", False

    # -- 数值提取 --------------------------------------------------------

    def read_number(
        self, region: ImageRegion | np.ndarray, *, min_value: int = 0
    ) -> int | None:
        """从文本里抠数字。抠不到返回 **None（不是 0）**。

        0 和"读不到"是两回事，返回 0 会让上层以为数值真的是 0。
        """
        res = self.read(region)
        if res.matched is TriState.UNKNOWN and not res.text:
            return None
        m = re.search(r"\d+", res.text.replace(",", ""))
        if not m:
            return None
        v = int(m.group())
        return v if v >= min_value else None

    def health(self) -> dict[str, Any]:
        return {
            "available": self._available,
            "fail_streak": self._fail_streak,
            "threshold": self._fail_threshold,
            "last_error": self._last_error[:200],
            "engine": "winrt" if self._legacy is not None else
                      ("rapid" if self._rapid is not None else "none"),
        }


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _encode_png(img: np.ndarray) -> bytes:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def _apply_replace(text: str, table: dict[str, list[str]]) -> str:
    """误识别替换。抄 BetterGI 的 replace 表思路：
    {"播放": ["播故", "搰放"]} —— 把误识别的写法换回正确文本。"""
    for correct, wrongs in table.items():
        for wrong in wrongs:
            if wrong and wrong in text:
                text = text.replace(wrong, correct)
    return text


def _match(
    text: str,
    contains_any: list[str] | None,
    contains_all: list[str] | None,
    regex: list[str] | None,
) -> TriState:
    """匹配判定。没给任何条件时返回 UNKNOWN（调用方自己解释文本）。"""
    if not text:
        return TriState.FALSE  # 识别成功但没读到字 = 确认不存在
    if contains_any:
        return TriState.TRUE if any(c in text for c in contains_any) else TriState.FALSE
    if contains_all:
        return TriState.TRUE if all(c in text for c in contains_all) else TriState.FALSE
    if regex:
        try:
            return TriState.TRUE if all(re.search(p, text) for p in regex) else TriState.FALSE
        except re.error as exc:
            # 正则写错是配置错误，响亮失败
            raise ValueError(f"OCR 正则非法: {exc}") from exc
    return TriState.UNKNOWN


__all__ = [
    "DEFAULT_FAIL_THRESHOLD",
    "OcrEngine",
    "OcrResult",
    "TriState",
]
