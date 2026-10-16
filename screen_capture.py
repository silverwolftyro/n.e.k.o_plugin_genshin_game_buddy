"""主动抓取目标游戏窗口画面 → 裸 base64（喂 OcrEngine）。

降级链 PrintWindow -> mss -> 宿主抓屏桥。全程内存、不落盘；绝不记录任何图像或
OCR 内容（隐私红线）。第三方依赖(mss/PIL/win32gui/win32process/psutil)一律函数内
lazy import，顶层只用 stdlib —— 本模块不在红线扫描名单里，但保持 ocr_engine.py
的既有惯例。
"""
from __future__ import annotations

import base64
import io
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

BLACK_FRAME_LUMA_THRESHOLD = 12.0

#: 「死像素」占比阈值 —— 超过它就判黑，**不看**平均亮度（2026-10-16 加）。
#:
#: ★ 为什么只看平均亮度不够（残帧 = 静默通过的黑帧）
#:   某些通道（BitBlt 实锤）在窗口尺寸与 GDI DC 对不上时只 blit 出画面的一部分，
#:   其余是**未绘制的纯黑 padding**。这种"残帧"的全图平均亮度被亮的那部分拉高到
#:   18~25，**稳稳越过** :data:`BLACK_FRAME_LUMA_THRESHOLD`=12，被判成"不是黑帧"
#:   原样送进 OCR —— OCR 只看得到左上角那一块，识不出东西。
#:   用户侧表现就是「实际上有画面，但 OCR 说黑屏」。
#:
#: ★ 实测分布（``autopilot/_runtime/capmatrix_*.png``，2026-10-13 抓取矩阵）
#:   ------------------------------------------------------------------
#:   完整帧   capmatrix_normal_wgc    dead 0.0%
#:            capmatrix_normal_dxcam  dead 0.2%
#:            capmatrix_normal_mss    dead 2.8%
#:   残帧     capmatrix_normal_bitblt    dead 54.6%  (luma 21.26 —— 越过 12!)
#:            capmatrix_minimize_bitblt  dead 55.6%  (luma 18.66)
#:            capmatrix_offscreen_bitblt  dead 53.9%  (luma 20.19)
#:   ------------------------------------------------------------------
#:   两侧各有 **30 个百分点以上**余量（最坏完整帧 2.8% vs 最轻残帧 53.9%），
#:   所以 0.20 不是拍出来的，是夹在两个实测簇中间。
#:
#: ⚠️ 已知边界：真正很暗的场景（夜晚/洞里）dead% 也会偏高，**可能被误判成黑**。
#:    缓解：``is_black_frame(..., ocr_len=...)`` 那条严格判据（0 个字即判黑）
#:    在交互轮次优先生效，暗场景里有 UI 文字就不会被误杀；
#:    且误判方向是"这张不要、下一张"，不会污染状态。
BLACK_FRAME_DEAD_FRAC_THRESHOLD = 0.20

# 拍板 2.0.69：mss 全局锁——mss.mss() 官方文档明确非线程安全（grab 会改共享内部缓冲）。
# 真机 2.0.68 暴露：2 worker 并发抓屏时 mss 内部死锁，_ocr_in_flight_count 永远 = workers，
# 后续所有 tick 被 worker 闸静默挡掉 → OCR 完全停滞。所有 mss.mss() 调用都必须走 _MSS_LOCK。
# mss_lock 是模块级同步原语；只保护 mss.mss() 这一段，不影响 PIL/bridge。
# 拍板 2.0.73：改 RLock——非 reentrant Lock 在 capture_active_frame → _capture_mss 嵌套调用时
# 双重 acquire 导致 worker 永久挂起（真机 2.0.72 暴露：3 分钟 body_exit 冻结、watchdog 救不了）。
# watchdog 触发 cf_future.cancel() 对运行中任务返回 False 不抛、只标 cancelled——worker 仍卡死。
# 唯一修法：让 _MSS_LOCK 可重入。
_MSS_LOCK = threading.RLock()

# 本地游戏进程/标题（拍板 2.0.14：YuanShen/GenshinImpact/StarRail/ZenlessZoneZero）
_GAME_EXE = {
    "yuanshen.exe", "genshinimpact.exe", "starrail.exe", "zenlesszonezero.exe",
}
_GAME_TITLE_HINTS = (
    "原神", "genshin", "崩坏：星穹铁道", "崩坏星穹铁道", "starrail", "绝区零", "zenless",
)

#: 窗口枚举里"单窗口处理失败"的告警预算（按不同异常去重后计）。
#:
#: ★ 为什么必须有这个预算
#:   ``EnumWindows`` 一次回调几百个窗口，每个窗口都独立 ``except``。
#:   不节流的话，**一行 typo 就能刷出 418 条 WARNING**（真机实测值），
#:   把真正的日志淹掉，而且看起来像"系统在疯狂报警"。
_ENUM_WARN_BUDGET = 3
_ENUM_SEEN_ERRORS: set = set()


def _warn_window_skip(where: str, exc: BaseException) -> None:
    """单个窗口处理失败时记一条可见告警，按 (位置, 异常类型, 消息) 去重。

    ★ 这里曾经是 ``except Exception: return True`` —— 纯静默
      2026-10-10 真机事故：``win32gui.GetWindowThreadProcessId`` 写错了模块
      （真 pywin32 里它在 **win32process**），每个窗口都 AttributeError，
      然后被这个裸 except 全部吞掉。``find_genshin_window()`` 于是恒返回
      None，从外面看跟"游戏没开"一模一样 —— 而当时的单测替身还把这个错 API
      塞进了假 win32gui，把 bug 一起认证成正确的。
      静默兜底可以留，但**不能再不留痕迹**。
    """
    key = (where, type(exc).__name__, str(exc))
    if key in _ENUM_SEEN_ERRORS:
        return
    _ENUM_SEEN_ERRORS.add(key)
    n = len(_ENUM_SEEN_ERRORS)
    if n <= _ENUM_WARN_BUDGET:
        _LOGGER.warning(
            "窗口枚举[%s] 跳过一个窗口：%s: %s", where, type(exc).__name__, exc)
        return
    if n == _ENUM_WARN_BUDGET + 1:
        _LOGGER.warning(
            "窗口枚举[%s] 还有别类错误在发生，去重后不再逐条打印（已去重 %d 类）。"
            "上面的 %d 条之外若有**新类型**的错误仍会再报一条。",
            where, n, _ENUM_WARN_BUDGET)


@dataclass(frozen=True)
class CaptureResult:
    ok: bool
    image_base64: Optional[str]  # 裸 base64，绝不带 "data:" 前缀
    width: int
    height: int
    method: str  # "printwindow" | "mss" | "bridge" | ""
    error: str  # "" | "no_window" | "black_frame" | "capture_failed" | ...
    avg_luma: float  # 0-255 平均亮度，未知填 -1.0


def _luma_stats(img: Any) -> Tuple[float, float]:
    """PIL Image -> ``(平均亮度, 死像素占比)``。

    * 平均亮度 —— 0~255，与 :data:`BLACK_FRAME_LUMA_THRESHOLD` 同尺度。
    * 死像素占比 —— 0.0~1.0，与 :data:`BLACK_FRAME_DEAD_FRAC_THRESHOLD` 同尺度。

    「死像素」= 亮度 < :data:`BLACK_FRAME_LUMA_THRESHOLD` 的像素，
    与平均亮度**同源同一个阈值**，不另立标准。

    统计前先缩到 32x32：够判"是不是大面积黑"，又不会因为几万个像素
    把解码开销拖进抓屏热路径。

    失败返回 ``(-1.0, 1.0)`` —— 「解不出来」按「全黑」处理，
    宁可丢帧也不能把一张读不懂的图送进 OCR。
    """
    try:
        gray = img.convert("L").resize((32, 32))
        px = list(gray.getdata())
        n = max(1, len(px))
        luma = float(sum(px)) / float(n)
        dead = sum(1 for p in px if p < BLACK_FRAME_LUMA_THRESHOLD) / float(n)
        return luma, dead
    except Exception:
        return -1.0, 1.0


def _img_luma(img: Any) -> float:
    """PIL Image -> 平均亮度(0-255)。失败返回 -1.0。"""
    return _luma_stats(img)[0]


def _encode(img: Any, fmt: str = "JPEG") -> Tuple[str, int, int]:
    """PIL Image -> (裸 base64, width, height)。"""
    try:
        buf = io.BytesIO()
        if fmt == "JPEG" and img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format=fmt, quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii"), int(img.width), int(img.height)
    except Exception:
        return "", 0, 0


def _stats_from_b64(image_base64: str) -> Tuple[float, float]:
    """裸 base64 图 -> ``(平均亮度, 死像素占比)``。解码失败返回 ``(-1.0, 1.0)``。"""
    try:
        from PIL import Image  # lazy: 第三方

        raw = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(raw))
        return _luma_stats(img)
    except Exception:
        return -1.0, 1.0


def _avg_luma_from_b64(image_base64: str) -> float:
    """裸 base64 图 -> 平均亮度。解码失败返回 -1.0。"""
    return _stats_from_b64(image_base64)[0]


def is_black_frame(image_base64: str, ocr_len: Optional[int] = None) -> bool:
    """黑帧判定。两条判据，任一命中即判黑（2026-10-16 起）。

    * 给了 ``ocr_len``（严格 D6 判据）：0 个字即判黑，非 0 判非黑，**覆盖下面两条**。
    * 没给 ``ocr_len``：``luma < 12`` **或** ``dead% > 0.20``。

    ★ 为什么必须有第二条（残帧）
      ``luma`` 是**全图平均**。一张「左上角 45% 有画面、右下 55% 是未绘制黑区」
      的残帧，均值能被亮的部分拉到 21 —— 越过 12 这道门槛，被判成"不是黑帧"。
      实测三张 BitBlt 残帧 luma 只有 18.66~21.26，而死像素占比是 53.9%~55.6%。
      单看均值永远抓不住它，必须看**比例**。
      依据与实测数据见 :data:`BLACK_FRAME_DEAD_FRAC_THRESHOLD` 的注释。

    ⚠️ 已知边界：真正很暗的场景（夜晚/洞里）dead% 偏高，可能被误判。
      给了 ``ocr_len`` 的调用点走的是严格判据，不受影响。
    """
    if ocr_len is not None:
        return int(ocr_len) == 0
    luma, dead = _stats_from_b64(image_base64)
    return luma < BLACK_FRAME_LUMA_THRESHOLD or dead > BLACK_FRAME_DEAD_FRAC_THRESHOLD


def _process_name(pid: Any) -> str:
    """pid -> 小写进程名（如 'yuanshen.exe'）。失败返回 ''。"""
    try:
        import psutil  # lazy: 第三方

        return (psutil.Process(pid).name() or "").lower()
    except Exception:
        pass
    try:
        import win32process  # lazy: 第三方

        return (win32process.GetModuleFileNameEx(pid, 0) or "").lower()
    except Exception:
        return ""


_ENUM_CENSUS_DONE = False


def _log_window_census_once() -> None:
    """**每个进程只跑一次**的窗口普查，排障入口。

    ★ 为什么需要它
      2026-10-10 事故二段：把 ``win32gui`` -> ``win32process`` 的 typo 修好之后，
      同一台机器上**独立跑**这份模块能正确返回原神窗口，
      可插件宿主进程里 ``find_genshin_window()`` 依然返回 None。
      同一份字节码、同一个桌面，结论却相反 —— 那差异只可能在**进程环境**里
      （宿主是不是跑在别的 window station / 会话、能不能读到高完整性窗口标题…）。
      光看返回值永远分不出来，得把"这个进程到底看见了什么"打出来。

    只在**首次调用**时枚举一遍，之后立刻返回，不给 5s 一 tick 的探测添开销。
    故意放在真正的枚举**之前**：找不到窗口的那一次才是需要证据的那一次。
    """
    global _ENUM_CENSUS_DONE
    if _ENUM_CENSUS_DONE:
        return
    _ENUM_CENSUS_DONE = True

    try:
        import os
        import sys

        import win32gui
        import win32process

        total = visible = titled = 0
        samples: list = []

        def _peek(hwnd: Any, _extra: Any) -> bool:
            nonlocal total, visible, titled
            total += 1
            if not win32gui.IsWindowVisible(hwnd):
                return True
            visible += 1
            title = win32gui.GetWindowText(hwnd) or ""
            if not title:
                return True
            titled += 1
            if len(samples) < 8:
                samples.append(title[:32])
            return True

        win32gui.EnumWindows(_peek, None)
        _LOGGER.warning(
            "[窗口普查] 模块=%s pid=%s python=%s 回调=%d 可见=%d 有标题=%d "
            "win32gui有API=%s win32process有API=%s 进程名表=%d 标题样本=%r",
            _LOGGER.name, os.getpid(), sys.executable, total, visible, titled,
            hasattr(win32gui, "GetWindowThreadProcessId"),
            hasattr(win32process, "GetWindowThreadProcessId"),
            len(_GAME_EXE), samples,
        )
    except Exception as exc:  # noqa: BLE001 —— 普查自己不能把主流程带崩
        _LOGGER.warning("[窗口普查] 普查本身失败：%s: %s", type(exc).__name__, exc)


def find_genshin_window() -> Optional[dict]:
    """找本地游戏窗口（原神/星铁/绝区零；进程名优先，标题兜底）。返回 {hwnd,pid,title,rect} 或 None。"""
    try:
        import win32gui  # lazy: 第三方
        import win32process  # lazy: 第三方
    except Exception:
        return None

    _log_window_census_once()

    found: dict = {}

    def _cb(hwnd: Any, _extra: Any) -> bool:
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd) or ""
            if not title:
                return True
            # ★ 必须是 win32process，不是 win32gui。
            #   真 pywin32 里 GetWindowThreadProcessId 只存在于 win32process；
            #   写错模块 -> AttributeError -> 被下面 except 吞成"没找到窗口"。
            _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
            name = _process_name(pid)
            is_gen = (name in _GAME_EXE) or any(h in title.lower() for h in _GAME_TITLE_HINTS)
            if is_gen and not found:
                rect = win32gui.GetWindowRect(hwnd)
                found.update({"hwnd": hwnd, "pid": pid, "title": title, "rect": tuple(rect)})
            return True
        except Exception as exc:
            _warn_window_skip("find_genshin_window", exc)
            return True

    try:
        win32gui.EnumWindows(_cb, None)
    except Exception:
        return None
    return dict(found) if found else None


def _enum_large_visible_windows() -> list[dict]:
    """枚举所有可见顶层窗口（rect ≥ 500x400，过滤任务栏/小工具/桌面 helper），按面积降序。

    拍板 2.0.15 兜底用：GetForegroundWindow 在 worker 线程里可能返回 0（前台在别的
    session / 调用线程没附着到前台输入队列），这时退而求其次抓屏幕上最大的可见窗口
    —— 云游戏跑 Edge 全屏时 GetWindowText 可能为空但 rect 占满主屏。
    """
    try:
        import win32gui  # lazy: 第三方
        import win32process  # lazy: 第三方
    except Exception:
        return []
    out: list[dict] = []

    def _cb(hwnd: Any, _extra: Any) -> bool:
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            rect = win32gui.GetWindowRect(hwnd)
            if not rect:
                return True
            left, top_, right, bottom = rect
            w = int(right) - int(left)
            h_ = int(bottom) - int(top_)
            if w < 500 or h_ < 400:
                return True
            title = win32gui.GetWindowText(hwnd) or ""
            _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
            out.append({
                "hwnd": hwnd, "pid": pid, "title": title,
                "rect": tuple(rect), "area": w * h_,
            })
        except Exception as exc:
            _warn_window_skip("_enum_large_visible_windows", exc)
        return True

    try:
        win32gui.EnumWindows(_cb, None)
    except Exception:
        pass
    out.sort(key=lambda x: x["area"], reverse=True)
    return out


def find_foreground_window() -> Optional[dict]:
    """找前台窗口；GetForegroundWindow 拿不到 → 降级到最大可见窗口（云游戏跑 Edge）。

    返回 {hwnd, pid, title, rect} 或 None（连一个像样窗口都没有）。
    """
    try:
        import win32gui  # lazy: 第三方
        import win32process  # lazy: 第三方
    except Exception:
        return None

    # 路径 1：直接问 shell 谁是前台（worker 线程也可能拿到，但常常返回 0）
    try:
        hwnd = win32gui.GetForegroundWindow()
        if hwnd:
            try:
                title = win32gui.GetWindowText(hwnd) or ""
                _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                left, top_, right, bottom = (rect or (0, 0, 0, 0))
                if int(right) > int(left) and int(bottom) > int(top_):
                    return {"hwnd": hwnd, "pid": pid, "title": title, "rect": tuple(rect)}
            except Exception as exc:
                _warn_window_skip("find_foreground_window/path1", exc)
    except Exception as exc:
        _warn_window_skip("find_foreground_window/getfg", exc)

    # 路径 2：EnumWindows 取最大可见窗口（≥500x400，跳过任务栏/桌面 helper）
    candidates = _enum_large_visible_windows()
    if candidates:
        top = candidates[0]
        return {"hwnd": top["hwnd"], "pid": top["pid"], "title": top["title"], "rect": top["rect"]}
    return None


def _capture_printwindow(hwnd: Any, rect: Any) -> CaptureResult:
    """ctypes GDI PrintWindow 抓被遮挡窗口自身内容。"""
    try:
        import ctypes
        from ctypes import wintypes

        from PIL import Image  # lazy: 第三方

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        pw_renderfullcontent = 2
        left, top, right, bottom = (rect or (0, 0, 0, 0))
        w = max(1, int(right) - int(left))
        h = max(1, int(bottom) - int(top))

        hwnd_dc = user32.GetWindowDC(hwnd)
        mfc_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
        old = gdi32.SelectObject(mfc_dc, bitmap)
        ok = user32.PrintWindow(hwnd, mfc_dc, pw_renderfullcontent)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h  # top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0
        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(mfc_dc, bitmap, 0, h, buf, ctypes.byref(bmi), 0)
        img = Image.frombytes("RGBA", (w, h), bytes(buf), "raw", "BGRA").convert("RGB")

        gdi32.SelectObject(mfc_dc, old)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mfc_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)

        luma = _img_luma(img)
        b64, ww, hh = _encode(img)
        good = bool(ok) and ww > 0
        return CaptureResult(good, b64 if good else None, ww, hh, "printwindow",
                             "" if good else "capture_failed", luma if good else -1.0)
    except Exception as exc:
        return CaptureResult(False, None, 0, 0, "printwindow", f"error:{type(exc).__name__}", -1.0)


def _capture_mss(rect: Any) -> CaptureResult:
    """mss 抓屏幕矩形（顶层窗口内容，被遮挡时是遮挡者）。"""
    try:
        import mss  # lazy: 第三方
        from PIL import Image  # lazy: 第三方

        left, top, right, bottom = (rect or (0, 0, 0, 0))
        mon = {
            "left": int(left), "top": int(top),
            "width": max(1, int(right) - int(left)), "height": max(1, int(bottom) - int(top)),
        }
        # 拍板 2.0.69：mss.mss() 非线程安全——mss 全局锁内串行 grab
        with _MSS_LOCK, mss.mss() as sct:
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.rgb)
        luma = _img_luma(img)
        b64, ww, hh = _encode(img)
        good = ww > 0
        return CaptureResult(good, b64 if good else None, ww, hh, "mss",
                             "" if good else "capture_failed", luma if good else -1.0)
    except Exception as exc:
        return CaptureResult(False, None, 0, 0, "mss", f"error:{type(exc).__name__}", -1.0)


def grab_primary_for_dhash() -> str:
    """拍板 2.0.69：mss 全局锁内抓主屏 → 9x8 灰度 dHash。

    集中 mss.mss() 调用点（之前 _light_capture_dhash 在 __init__.py 里 importlib 调用
    mss 不走 screen_capture 模块——绕过了本模块新加的 _MSS_LOCK）。改用本函数：
    ① 走 _MSS_LOCK 防并发死锁
    ② 失败/缺依赖 → 返回空串（与旧 _light_capture_dhash 契约一致）
    ③ 不存像素，只算 64-bit dHash（16 hex chars）
    """
    import importlib
    try:
        mss = importlib.import_module("mss")
        Image = importlib.import_module("PIL.Image")
        with _MSS_LOCK, mss.mss() as sct:
            monitors = sct.monitors or []
            if len(monitors) < 2:
                return ""
            mon = monitors[1]
            shot = sct.grab({
                "left": int(mon["left"]),
                "top": int(mon["top"]),
                "width": int(mon["width"]),
                "height": int(mon["height"]),
            })
        img = Image.frombytes("RGB", shot.size, shot.rgb).convert("L").resize((9, 8))
        pixels = list(img.getdata())  # 72 个 0-255（9*8）
        bits = 0
        for y in range(8):
            for x in range(8):
                bits <<= 1
                # 横向差：左 > 右 → 1（dHash 标准约定；让亮→暗边界明显）
                if pixels[y * 9 + x] > pixels[y * 9 + x + 1]:
                    bits |= 1
        return f"{bits:016x}"
    except Exception:
        return ""


def _main_server_port() -> int:
    try:
        from config import MAIN_SERVER_PORT  # 动态读宿主端口

        return int(MAIN_SERVER_PORT)
    except Exception:
        return 48911


def _capture_bridge(pid: Any, title: Any) -> CaptureResult:
    """宿主抓屏桥兜底：POST /api/capture/screenshot（回环 HTTP，最慢）。"""
    try:
        import json
        import urllib.request

        url = f"http://127.0.0.1:{_main_server_port()}/api/capture/screenshot"
        body = json.dumps({"target_id": 0, "pid": int(pid or 0), "title": str(title or "")}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        if not data.get("success") or not data.get("image"):
            return CaptureResult(False, None, 0, 0, "bridge",
                                 str(data.get("error") or "bridge_failed"), -1.0)
        img_b64 = data["image"]
        if isinstance(img_b64, str) and img_b64.startswith("data:"):
            img_b64 = img_b64.split(",", 1)[-1]
        luma = _avg_luma_from_b64(img_b64)
        return CaptureResult(True, img_b64, int(data.get("width") or 0), int(data.get("height") or 0),
                             "bridge", "", luma)
    except Exception as exc:
        return CaptureResult(False, None, 0, 0, "bridge", f"error:{type(exc).__name__}", -1.0)


def capture_window_base64(target: Optional[dict] = None) -> CaptureResult:
    """降级链 PrintWindow -> mss -> bridge。target=None 时先 find_genshin_window()。

    每路 not ok 或黑帧则降级；返回第一个 ok 且非黑帧的结果；全失败返回最后一路的失败结果。
    """
    if target is None:
        target = find_genshin_window()
    if not target:
        return CaptureResult(False, None, 0, 0, "", "no_window", -1.0)

    hwnd = target.get("hwnd")
    rect = target.get("rect")
    pid = target.get("pid")
    title = target.get("title", "")

    routes = (
        ("printwindow", lambda: _capture_printwindow(hwnd, rect)),
        ("mss", lambda: _capture_mss(rect)),
        ("bridge", lambda: _capture_bridge(pid, title)),
    )
    last: Optional[CaptureResult] = None
    for name, fn in routes:
        try:
            res = fn()
        except Exception as exc:  # 单路炸掉也必须降级，不冒泡
            res = CaptureResult(False, None, 0, 0, name, f"error:{type(exc).__name__}", -1.0)
        last = res
        if res and res.ok and res.image_base64 and not is_black_frame(res.image_base64):
            return res

    if last is None:
        return CaptureResult(False, None, 0, 0, "", "no_window", -1.0)
    err = last.error
    if not err:
        err = "black_frame" if (last.ok and last.image_base64) else "capture_failed"
    return CaptureResult(False, None, last.width, last.height, last.method or "bridge", err, last.avg_luma)


def capture_active_frame() -> Tuple[CaptureResult, str]:
    """拍板 2.0.16：本地进程优先；前景路径走 mss 主屏 → bridge 兜底（不依赖 win32gui 枚举）。

    返回 (CaptureResult, capture_path ∈ {"local_process","foreground"})。
    本地：PrintWindow→mss→bridge 降级链。
    前景：mss 主屏（云游戏整屏即画面，不需要精确知道是哪个窗口）→ 宿主抓屏桥
    （HTTP 抓宿主自己的屏，宿主 UI 线程必有访问权，绕过宿主营养不良的 win32 上下文）。
    前景两条路**都要过 is_black_frame**（残帧守卫，2026-10-16 补的）——
    此前前景路径直接 ``if result.ok: return``，是全模块唯一不看黑帧的出口。
    前景错误统一归 capture_failed（附 mss/bridge 两路子错误），不再有 no_hwnd/no_rect。
    find_foreground_window / _enum_large_visible_windows 仍保留作可选 helper，
    本函数不再调用（宿主沙箱里 win32gui 拿不到桌面窗口，详见诊断报告）。
    """
    target = find_genshin_window()
    if target:
        return capture_window_base64(target), "local_process"

    def _strip(s: str) -> str:
        return s[len("error:"):] if s.startswith("error:") else s

    # 前景路径 1：mss 主屏（sct.monitors[1] = 主显示器；云游戏跑全屏就是它）
    mss_sub = ""
    try:
        import mss  # lazy: 第三方
        # 拍板 2.0.69：mss.mss() 走 _MSS_LOCK 内串行（防止 2 worker 并发抓屏死锁）
        with _MSS_LOCK, mss.mss() as sct:
            monitors = sct.monitors or []
            if len(monitors) < 2:
                raise RuntimeError("no_primary_monitor")
            mon = monitors[1]
        rect = (int(mon["left"]), int(mon["top"]),
                int(mon["left"]) + int(mon["width"]),
                int(mon["top"]) + int(mon["height"]))
        result = _capture_mss(rect)
        # ★ 残帧守卫（2026-10-16）。本地分支的 capture_window_base64 三路都过
        #   is_black_frame，唯独前景路径此前**一道检查都没有**——mss 抓到啥就是啥，
        #   全黑/残帧也照样 ok=True 直接返回，OCR 于是拿着一张废图去识别。
        #   这里与本地分支对齐：判黑（或压根没图）就继续往下走 bridge。
        if result.ok and result.image_base64:
            if not is_black_frame(result.image_base64):
                return result, "foreground"
            mss_sub = "black_frame"
        elif result.ok:
            mss_sub = "no_image"
        else:
            mss_sub = _strip(result.error or "unknown")
    except Exception as exc:
        mss_sub = type(exc).__name__

    # 前景路径 2：宿主抓屏桥（HTTP，宿主 UI 线程一定有访问权）
    bridge = _capture_bridge(0, "")
    if bridge.ok and bridge.image_base64:
        if not is_black_frame(bridge.image_base64):
            return bridge, "foreground"
        # ★ 最后一路也是黑的/残的。返回 ok=False 而不是那张废图 ——
        #   与本地分支 capture_window_base64「三路全黑 → ok=False」保持一致。
        #   若在此返回 ok=True，ocr_perceive 会拿它去 OCR，拿到 0 个字，
        #   而 __init__.py:3020 的兜底 is_black_frame 对残帧（luma≈21）不生效，
        #   空文本就被当成真帧推进状态机 —— 正是这次要修的那个洞。
        bridge_sub = "black_frame"
    elif bridge.ok:
        bridge_sub = "no_image"
    else:
        bridge_sub = _strip(bridge.error or "unknown")

    # 都失败 → capture_failed（附两路子错误）
    err = f"capture_failed:mss={mss_sub}|bridge={bridge_sub}"
    return CaptureResult(False, None, 0, 0, bridge.method or "bridge", err, -1.0), "foreground"
