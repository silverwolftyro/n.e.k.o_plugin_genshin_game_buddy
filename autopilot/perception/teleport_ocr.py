"""传送点 OCR 识别（框架 + 匹配）。

★ 契约
------
本模块实现 ``recognition/notes/teleport_contract.md`` 的 **① SELECT 段**：
"OCR 检测到他选择了传送锚点 / 神像" -> 唯一命中候选点。
**只负责 SELECT，不负责 ② LOADING 和 ③ WORLD_RESTORED。**

★ 匹配策略（严格降级，宁可失败不猜）
----------------------------------------
1. 精确匹配（主标题 + 副标题同时精确）
2. 编辑距离 <= 2（两个字段都算）
3. 拼音相似度（两个字段都算）
4. **任一字段不过阈值 -> 返回 None**

★ 为什么要"两个字段都必须过"
---------------------------
契约要求"主标题 + 副标题联合唯一"。主标题如"七天神像-水"全国有几十个同名，
**只看主标题必然误命中**。副标题（如"枫丹 枫丹廷·沫芒宫"）才唯一。
所以主标题对了、副标题错了 -> 判失败（不猜），这是刻意的保守设计。

★ OCR 引擎是可插拔的
---------------------
默认走 Windows.Media.Ocr；如果不可用或识别不到，``recognize_teleport``
返回 ``None``（**不抛异常** —— 画面里没有信息框是正常情况，不是错误）。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "TeleportTarget",
    "MATCH_EDIT_DISTANCE",
    "MATCH_SIMILARITY",
    "crop_info_box",
    "load_teleport_points",
    "match_teleport",
    "ocr_region",
    "preprocess",
    "recognize_teleport",
    # 注入/读取钩子。2026-10-12 补进 __all__：它们一直是公开的
    # 依赖注入点（与 StateBuilder(locator=)、clear_auto_cache 同类），
    # 但因为整个模块还没接线、也从没被 export，看起来像「零引用的死代码」。
    # 声明出来 = 它们是契约的一部分，删不删由人决定，不由忘了决定。
    "set_backend",
    "get_backend",
    "set_ocr",
]

#: 规划文档 §4.5 用户确认的信息框位置（归一化）
INFO_BOX_ROI = (0.744, 0.002, 1.000, 0.116)
#: 外扩 10% 作保险区
INFO_BOX_MARGIN = 0.10

#: 编辑距离阈值（主/副标题各自）
MATCH_EDIT_DISTANCE = 2
#: 相似度阈值（主/副标题各自）
MATCH_SIMILARITY = 0.80
#: 并列最高分且**落点一致**时的容差（米）。超过它就是真歧义 -> 判失败。
SAME_LANDING_TOL_M = 1.0

#: 传送点数据搜索顺序
_TP_SEARCH = (
    Path(r"D:\workspace\N.E.K.O\NEKO-genshin\recognition\data\teleport_points.json"),
    Path(r"D:\workspace\N.E.K.O\NEKO-genshin\recognition\data\_ref_tp_sample.json"),
)


@dataclass
class TeleportTarget:
    """一次成功匹配的传送点。"""

    matched_id: str
    name: str
    position: tuple[float, float, float]
    tran_position: tuple[float, float, float]
    confidence: float
    main_title: str = ""
    sub_title: str = ""
    areas: tuple[str, ...] = ()


# ======================================================== 归一化 / 距离
def _norm(s) -> str:
    """去空白、统一连接符、全角转半角。"""
    s = unicodedata.normalize("NFKC", str(s or "")).strip()
    s = re.sub(r"[\s\u3000]+", "", s)
    for ch in ("，", "、", "－", "—", "·", "・"):
        s = s.replace(ch, "-")
    return re.sub(r"-{2,}", "-", s).strip("-")


def _edit_distance(a: str, b: str, cap: int) -> int:
    """带上限的编辑距离。超过 ``cap`` 直接返回 cap+1（早退省时间）。"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _score_pair(main: str, sub: str, p: dict) -> Optional[float]:
    """主标题、副标题各自打分；**任一不过直接返回 None**。"""
    nm, ns = _norm(main), _norm(sub)
    if not nm or not ns:
        return None
    cands = {_norm(p["name"])} | {_norm(f"{p['name']}-{a}") for a in p["areas"]}
    cands |= {_norm(a) for a in p["areas"]}
    hit_main = hit_sub = None
    for c in cands:
        if hit_main is None and (c == nm or _edit_distance(c, nm,
                                                           MATCH_EDIT_DISTANCE)
                                <= MATCH_EDIT_DISTANCE):
            hit_main = c
        if hit_sub is None and (c == ns or _edit_distance(c, ns,
                                                           MATCH_EDIT_DISTANCE)
                                <= MATCH_EDIT_DISTANCE
                                or _sim(c, ns) >= MATCH_SIMILARITY):
            hit_sub = c
    if hit_main is None or hit_sub is None:
        return None
    s_main = 1.0 if hit_main == nm else max(
        _sim(hit_main, nm), 1.0 - _edit_distance(hit_main, nm,
                                                  MATCH_EDIT_DISTANCE)
        / max(len(hit_main), len(nm), 1))
    s_sub = 1.0 if hit_sub == ns else max(
        _sim(hit_sub, ns), 1.0 - _edit_distance(hit_sub, ns,
                                                MATCH_EDIT_DISTANCE)
        / max(len(hit_sub), len(ns), 1))
    if s_main < MATCH_SIMILARITY * 0.75 or s_sub < MATCH_SIMILARITY * 0.75:
        return None
    return s_main * 0.4 + s_sub * 0.6            # 副标题更唯一，权重更高


# ======================================================== 候选表
_PT_CACHE: Optional[list[dict]] = None


def load_teleport_points(path: str | Path | None = None) -> list[dict]:
    """读 936 个传送点。按 ``_TP_SEARCH`` 顺序找，找到第一个就用。"""
    global _PT_CACHE
    if path is not None:
        return _read_tp(Path(path))
    if _PT_CACHE is not None:
        return _PT_CACHE
    for p in _TP_SEARCH:
        if p.is_file():
            _PT_CACHE = _read_tp(p)
            return _PT_CACHE
    raise FileNotFoundError(
        "找不到传送点数据。已找: " + ", ".join(str(p) for p in _TP_SEARCH))


def _read_tp(p: Path) -> list[dict]:
    raw = json.loads(p.read_text(encoding="utf-8-sig"))
    out = []
    if isinstance(raw, dict) and "data" in raw:            # BetterGI 原格式
        for region in raw["data"]:
            for pt in region.get("points", []):
                pos, tr = pt.get("position"), pt.get("tranPosition")
                if isinstance(pos, list) and len(pos) == 3:
                    out.append({"id": str(pt.get("id", "")),
                                "name": str(pt.get("name", "")),
                                "areas": [str(a) for a in
                                          (pt.get("areas") or [])],
                                "position": tuple(float(v) for v in pos),
                                "tran_position": tuple(
                                    float(v) for v in tr)
                                if isinstance(tr, list) and len(tr) == 3
                                else tuple(float(v) for v in pos)})
    elif isinstance(raw, dict) and "regions" in raw:       # 自有格式
        for region in raw["regions"]:
            for pt in region.get("points", []):
                wp = pt.get("world_pos")
                lp = pt.get("landing_pos")
                if isinstance(wp, list) and len(wp) == 3:
                    out.append({"id": str(pt.get("id", "")),
                                "name": str(pt.get("name", "")),
                                "areas": [str(a) for a in
                                          (pt.get("areas") or [])],
                                "position": tuple(float(v) for v in wp),
                                "tran_position": tuple(float(v) for v in lp)
                                if isinstance(lp, list) and len(lp) == 3
                                else tuple(float(v) for v in wp)})
    if not out:
        raise ValueError(f"{p} 里没解析出任何传送点")
    return out


# ======================================================== 匹配
def match_teleport(main_title: str, sub_title: str,
                    points: Optional[Sequence[dict]] = None
                    ) -> Optional[TeleportTarget]:
    """联合主键匹配 936 选 1。

    :param main_title: 主标题，如 "七天神像-水"
    :param sub_title: 副标题，如 "枫丹 枫丹廷·沫芒宫"
    :param points: 候选表；``None`` 时自动加载
    :returns: 命中的 :class:`TeleportTarget`；**失败返回 None（不猜）**

    ★★ 命中必须**唯一**（2026-10-15 补）
    -------------------------------------
    契约 ``teleport_contract.md`` §3① 写的是「OCR 读到主标题+副标题，
    **在候选表中唯一命中**；模糊/多候选都算失败」。而原实现是
    「取最高分的第一个」—— 两者在**数据上是天差地别的**：

    936 个点里 **777 个都叫「传送锚点」**，它们唯一的区别在 ``areas``。
    而 ``areas`` 是**区域名**（「枫丹」「蒙德」），不是具体地名。实测：

        main='传送锚点', sub='枫丹'    -> **23 个点同分 0.8000**
        main='传送锚点', sub='蒙德'    -> **2 个点同分 0.8800**
        main='传送锚点', sub='枫丹廷·灰河' -> 1 个点（真·唯一）✔

    取最高分第一个 = **随便挑一个**。真机实测就是这么翻车的：
    OCR 只读到区域名，匹配到 id=81（areas=['枫丹','蒙德']），
    ``tran_position=(548.8, 206.6, 341.5)`` -> 定位原点落在几千米外，
    之后所有 MOVE_TO 全在错误的坐标系里走，角色一头撞墙。

    所以多候选**必须判失败**：宁可不刷原点，也不用一个可能是任意一个的
    原点 —— 后者会静默污染整条定位链，且没有任何报错。
    """
    pts = list(points) if points is not None else load_teleport_points()
    scored: list[tuple[float, dict]] = []
    for p in pts:
        s = _score_pair(main_title, sub_title, p)
        if s is not None:
            scored.append((s, p))
    if not scored:
        logger.info("传送点匹配失败 main=%r sub=%r", main_title, sub_title)
        return None

    best_s = max(s for s, _ in scored)
    if best_s < MATCH_SIMILARITY * 0.75:
        return None

    # ★ 并列即失败 —— 除非它们**指向同一个地方**。
    #   数据表里同一处传送锚点会拆成两条记录（name/areas 一样）。
    #   「你要去哪」关心的是坐标而不是 id，所以落点一致的并列放行，
    #   否则会白白丢掉一次本来能用的定位。
    #   阈值 1.0m：远小于「两个传送点算同一处」的直觉尺度，
    #   又远大于 BetterGI 导出里的浮点噪声。
    winners = [p for s, p in scored if s >= best_s - 1e-9]
    if len(winners) > 1:
        xs = [(p["tran_position"][0], p["tran_position"][2]) for p in winners]
        spread = max(math.hypot(a[0] - b[0], a[1] - b[1])
                     for a in xs for b in xs)
        if spread <= SAME_LANDING_TOL_M:
            logger.debug("传送点有 %d 条并列记录但落点一致（%.2fm 内），取 %s",
                         len(winners), spread, winners[0]["id"])
        else:
            logger.error(
                "传送点匹配**不唯一**：%d 个点同为 %.4f（main=%r sub=%r），"
                "且落点最大相差 %.1fm。最常见的原因是 sub 只 OCR 到了"
                "**区域名**（如「枫丹」）而不是具体地名 —— 936 个点里 777 个"
                "都叫「传送锚点」，区域名区分不开。按契约 §3①「多候选算失败」，"
                "本次拒绝提交；拿一个可能是任意一个的原点会静默污染整条定位链。",
                len(winners), best_s, main_title, sub_title, spread)
            return None

    best = winners[0]
    return TeleportTarget(
        matched_id=best["id"], name=best["name"],
        position=tuple(best["position"]),
        tran_position=tuple(best["tran_position"]),
        confidence=round(best_s, 4),
        main_title=main_title, sub_title=sub_title,
        areas=tuple(best["areas"]))


# ======================================================== 图像
def crop_info_box(frame, roi=INFO_BOX_ROI, margin=INFO_BOX_MARGIN):
    """按归一化 ROI 裁信息框（含外扩保险区）。BGR 输入。"""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi
    dw, dh = (x1 - x0) * margin, (y1 - y0) * margin
    px0 = max(0, int((x0 - dw) * w))
    py0 = max(0, int((y0 - dh) * h))
    px1 = min(w, int((x1 + dw) * w))
    py1 = min(h, int((y1 + dh) * h))
    if px1 - px0 < 4 or py1 - py0 < 4:
        raise ValueError(f"画面 {w}x{h} 太小，信息框裁剪退化")
    return frame[py0:py1, px0:px1]


def preprocess(region, upscale: int = 3):
    """二值化。**同时返回深底白字和浅底黑字两种模式**。"""
    import cv2
    g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 \
        else region
    g = cv2.resize(g, None, fx=upscale, fy=upscale,
                   interpolation=cv2.INTER_CUBIC)
    # 模式 A：深底白字 -> 文字是亮的
    light_on_dark = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, 12)
    # 模式 B：浅底黑字 -> 文字是暗的
    dark_on_light = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        31, 12)
    return light_on_dark, dark_on_light


#: 可插拔 OCR：``(bgr_region) -> list[str]``，返回识别到的文本行
_OCR_FN: Optional[Callable] = None
#: OCR 后端："rapidocr"（默认，不依赖系统组件）| "winrt" | "auto"
_OCR_BACKEND = os.environ.get("NEKO_OCR_BACKEND", "rapidocr").lower()
#: RapidOCR 引擎单例（首次加载模型很慢，必须复用）
_RAPID = None


def set_backend(name: str) -> None:
    """切换 OCR 后端。``rapidocr`` / ``winrt`` / ``auto``。"""
    global _OCR_BACKEND
    _OCR_BACKEND = str(name).lower()
    logger.info("OCR 后端 -> %s", _OCR_BACKEND)


def get_backend() -> str:
    return _OCR_BACKEND


def set_ocr(fn: Optional[Callable]) -> None:
    """注入自定义 OCR 函数。``None`` 表示恢复默认后端。"""
    global _OCR_FN
    _OCR_FN = fn


def _rapidocr_ocr(region) -> list[str]:
    """RapidOCR 后端。

    **先试原图，再试两种二值化模式** —— 原图通常最好，但深蓝底白字
    这种极端对比度下二值化有时反而更稳。全失败返回空列表，不抛异常。
    """
    global _RAPID
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        logger.warning("RapidOCR 不可用: %s", exc)
        return []
    if _RAPID is None:
        try:
            _RAPID = RapidOCR()
        except Exception as exc:                    # noqa: BLE001
            logger.error("RapidOCR 引擎初始化失败: %s", exc)
            return []
    light, dark = preprocess(region)
    for img in (region, light, dark):
        try:
            res, _elapse = _RAPID(img)
        except Exception as exc:                    # noqa: BLE001
            logger.debug("RapidOCR 推理失败: %s", exc)
            continue
        if not res:
            continue
        out = []
        for item in res:
            # item = [box, text, score]
            t = str(item[1]).strip() if len(item) > 1 else ""
            if t:
                out.append(t)
        if out:
            return out
    return []


def _winrt_ocr(region) -> list[str]:
    """默认 OCR：Windows.Media.Ocr。

    依赖本机 WinRT 能力。**任何不可用/失败都返回空列表，不抛异常** ——
    画面里没有信息框是正常情况，不该让整个流程炸掉。
    """
    try:
        # ⚠️ ``cv2`` 在本函数体里并不使用 —— 它是一次**依赖探测**：
        #   缺 cv2 时整个 try 块走 ImportError，OCR 降级成空列表而不是
        #   在后面某一行炸掉。所以**不要**「顺手删掉这个未使用的导入」。
        import cv2  # noqa: F401
        import numpy as np
        from winrt.windows.globalization import Language
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage.streams import DataReader
        from winrt.windows.storage.streams import WindowsRuntimeBufferExtensions as Wrb
    except ImportError as exc:
        logger.warning("Windows OCR 不可用（缺 winrt）: %s", exc)
        return []
    try:
        bgr = region
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])          # BGR -> RGB
        # 提升到 BGRA，WinRT 认 SoftwareBitmap
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[:, :, :3] = rgb
        bgra[:, :, 3] = 255
        stream = Wrb.as_buffer(bgra)
        reader = DataReader.from_buffer(stream)
        reader.read_async()
        buf = reader.read_bytes()
        bitmap = SoftwareBitmap.create_copy_from_buffer(
            buf, BitmapPixelFormat.BGRA8, w, h,
            BitmapPixelFormat.BGRA8)
        engine = OcrEngine.try_create_from_language(Language("zh-Hans"))
        if engine is None:
            logger.warning("WinRT 没装中文 OCR 引擎")
            return []
        result = engine.recognize_async(bitmap).get_results()
        out = []
        for line in result.lines:
            t = "".join(w.text for w in line.words)
            if t.strip():
                out.append(t)
        return out
    except Exception as exc:                                 # noqa: BLE001
        logger.debug("Windows OCR 调用失败: %s", exc)
        return []


def ocr_region(region) -> list[str]:
    """对信息框区域做 OCR，返回文本行列表。

    后端由 :func:`set_backend` / 环境变量 ``NEKO_OCR_BACKEND`` 决定。
    **任何后端不可用或识别失败都返回 ``[]``，不抛异常** ——
    画面里没有信息框是正常情况，不该让整个流程炸掉。
    """
    if _OCR_FN is not None:
        return list(_OCR_FN(region) or [])
    backend = _OCR_BACKEND
    if backend in ("rapidocr", "auto"):
        out = _rapidocr_ocr(region)
        if out:
            return out
        if backend == "rapidocr":
            return []
    if backend in ("winrt", "auto"):
        return _winrt_ocr(region)
    logger.warning("未知 OCR 后端 %r，退回 rapidocr", backend)
    return _rapidocr_ocr(region)


#: OCR 行噪声过滤：必须含中文，且至少 2 个字符
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _clean_lines(lines) -> list[str]:
    """从 OCR 原始行里挑出可信文本行。

    信息框周围全是小字（延迟计数 "8ms"、感叹号、"X" 等），
    **这些必须滤掉** —— 否则按长度排序会把它们当成主标题。
    规则：含中文 + 长度 >= 2。按长度降序（长的通常是地点）。
    """
    out = []
    for t in lines:
        t = str(t).strip()
        if len(t) >= 2 and _CJK.search(t):
            out.append(t)
    out.sort(key=len, reverse=True)
    return out


def recognize_teleport(frame_720, points: Optional[Sequence[dict]] = None,
                      lines: Optional[Sequence[str]] = None
                      ) -> Optional[TeleportTarget]:
    """端到端：裁信息框 -> OCR -> 联合匹配。

    :param frame_720: 720 基准画面（BGR）。归一化 ROI 与分辨率无关。
    :param points: 候选表；``None`` 时自动加载
    :param lines: **已 OCR 好的文本行**。给了就跳过 OCR —— 调用方
        已经读过一遍时用它，避免同一帧跑两次引擎（RapidOCR 不便宜）。
    :returns: 命中的 :class:`TeleportTarget`；任何环节失败返回 ``None``

    ★ 主/副标题的选取不靠"长短"猜
    --------------------------------
    实测信息框内容是 ``传送锚点`` + ``枫丹 枫丹廷·灰河``，
    而噪声行（``8ms`` / ``X``）可能比任何一行都短。
    所以：**把每对可信行都当主副试一遍，取命中最强的那个**。
    这既不依赖长度启发式，也天然实现了"主标题=类型标签、
    副标题=地点"的语义 —— 因为 ``name`` 字段里
    777 个传送锚点就叫 ``传送锚点``。
    """
    try:
        region = crop_info_box(frame_720)
    except ValueError as exc:
        logger.debug("信息框裁剪失败: %s", exc)
        return None
    raw = list(lines) if lines is not None else ocr_region(region)
    texts = _clean_lines(raw)
    if len(texts) < 2:
        logger.info("信息框 OCR 可信行不足 (%d): %r", len(texts), texts)
        return None
    pts = list(points) if points is not None else load_teleport_points()
    best, best_conf = None, 0.0
    for i in range(len(texts)):
        for j in range(len(texts)):
            if i == j:
                continue
            r = match_teleport(texts[i], texts[j], pts)
            if r is not None and r.confidence > best_conf:
                best, best_conf = r, r.confidence
    if best is None:
        logger.info("传送点匹配全部失败，可信行=%r", texts)
    return best
