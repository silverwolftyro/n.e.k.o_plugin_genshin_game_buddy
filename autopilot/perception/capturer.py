"""三通道截图 + 可重入锁。

对应架构文档 6.1。

三条通道按优先级尝试，**启动期探测，失败的通道禁用**：
    dxcam  DXGI，最快，硬件加速游戏
    bitblt GDI BitBlt，窗口级、硬件加速下也能抓（参照 BetterGI）
    mss    GDI 屏幕坐标，兼容性最好（⚠️ mss.mss() 非线程安全，必须走可重入锁）

**为什么删掉了 PrintWindow**（2026-10-10 真机实证）：
    降级链原本是 ``dxcam -> mss -> printwindow``，而真机跑出来的结论是
    ``截图重试 25 次仍未成功：PrintWindow 返回失败``，紧接着主循环
    ``CORE_PAUSED``。不是参数不对，是**这条路本来就不该存在**：
    五个成熟原神自动化项目（BetterGI / cvAutoTrack / 空荧酒馆 / minimap）
    没有任何一个把 PrintWindow 放在链上 —— 它们的窗口级抓取全部走
    BitBlt / WGC / DXGI。PrintWindow 在 Win11 + 硬件加速 + 独占/无边框
    组合下返回黑屏或直接失败，把它当兜底等于"最后一步必然掉链子"。
    BitBlt 顶替它的位置：同样是窗口级（游戏被遮挡也能抓到内容），
    但拿的是 GDI 设备上下文，能吃到硬件加速合成后的画面。

**为什么必须 RLock 而不是 Lock**（这是本项目 2.0 踩过的坑）：
    2.0 的 screen_capture.py 原本用 threading.Lock，在
    capture_active_frame -> _capture_mss 嵌套调用时死锁。
    拍板改为 RLock。任何"所有截图调用都必须走同一个锁"的地方都用 RLock，
    否则嵌套调用会自锁。
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# DPI 感知（§34）
# --------------------------------------------------------------------------
#
# ★ 为什么需要
#   用户系统 DPI = 150%（2560×1600 物理 -> 1707×1067 逻辑）。进程如果
#   是 DPI-unaware，``GetWindowRect`` 和 mss 都拿**逻辑**坐标，抓回来的
#   画面只覆盖屏幕左上角 1/1.5 = 67%。实测抓到 1706×1066 的原神画面：
#   右边 UI 和底下血条全被切掉了 —— 连 SceneDetector 都因此判成 UNKNOWN。
#
# ★ 判据用**实测**，不信返回码
#   宿主（BetterGI/Qt）可能已经设过 awareness，那时我们再调
#   ``SetProcessDpiAwareness`` 会返回 ERROR_ACCESS_DENIED，但进程**本来
#   就是对的**。所以：调用前后对比 ``GetSystemMetrics(0)`` ——
#   真正切到 aware 之后它返回的应该是物理像素而非虚拟化的逻辑像素。
#
# ⚠️ 这是**进程级、单向**的开关：设高不能降回，设过之后任何窗口创建都会
#    按 aware 处理。所以必须在**任何窗口创建之前**调用。

_DPI_AWARENESS_SET = False       # 只调一次
_DPI_AWARE = False              # 实测结论：调用后是否真的生效
_DPI_SCALE_FALLBACK = 1.0        # 未生效时的反算系数（Task C）


def _proc_is_dpi_aware() -> bool | None:
    """问系统"本进程**确定**是 DPI-aware 吗"。读不到/不确定返回 None。

    ⚠️ **只有 SYSTEM(2) / PER_MONITOR(3) 才算 True。**
       ``INVALID(0)``（本接口读不出来时常见）**不能**当成 aware ——
       早先写成 ``aware != 1`` 会把 INVALID 误判成 True，于是
       :func:`ensure_dpi_awareness` 打印"已是 DPI-aware"就**直接返回，
       根本没调 SetProcessDpiAwareness**，DPI 问题原样留着。

    Win10 1703+ 才有这套接口，老系统上 getattr 会 AttributeError。
    """
    try:
        u = ctypes.windll.user32
        u.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        u.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        u.GetAwarenessFromDpiAwarenessContext.restype = ctypes.c_int
        ctx = u.GetThreadDpiAwarenessContext()
        aware = u.GetAwarenessFromDpiAwarenessContext(ctx)
    except (AttributeError, OSError, TypeError):
        return None
    # 1 = DPI_AWARENESS_UNAWARE, 2 = SYSTEM, 3 = PER_MONITOR, 0 = INVALID
    return aware in (2, 3)


def ensure_dpi_awareness() -> bool:
    """把进程切到 DPI-aware。**幂等**，返回是否**实测生效**。

    在 :mod:`capturer` **模块级**调用（import 时执行一次），所以一定早于
    任何 :class:`Capturer` 实例化 —— 满足"必须在窗口创建之前"的要求。

    ⚠️ 若调用失败（旧系统 / 宿主已建窗），**不抛**，只记 warning 并把
       :data:`_DPI_AWARE` 留 False —— 上层会走 ``GetDpiForWindow`` 反算
       的兜底路径（见 :meth:`Capturer._physical_rect`）。
    """
    global _DPI_AWARENESS_SET, _DPI_AWARE

    if _DPI_AWARENESS_SET:
        return _DPI_AWARE
    _DPI_AWARENESS_SET = True

    u = ctypes.windll.user32
    try:
        before = u.GetSystemMetrics(0)
    except Exception:                          # noqa: BLE001 - 非 Windows
        logger.warning("非 Windows 平台，跳过 DPI 感知设置")
        return False

    already = _proc_is_dpi_aware()
    if already is True:
        logger.info("进程已是 DPI-aware，无需设置")
        _DPI_AWARE = True
        return True

    # ★ 成功与否不取 ``ok``，而是用下面 ``GetSystemMetrics`` 的**实测**对比判定
    #   （Win10 上 SetProcessDpiAwareness 可能返回 0 却因清单里已声明 aware
    #   而没改变量）。旧的 ``ok`` 赋值后从未被读，属死变量。
    try:
        # Win8.1+：Per-Monitor DPI Aware
        shcore = ctypes.windll.shcore
        shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
        shcore.SetProcessDpiAwareness.restype = ctypes.c_long
        shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:                                   # Vista+ 兜底
            u.SetProcessDPIAware.restype = ctypes.c_long
            u.SetProcessDPIAware()
        except (AttributeError, OSError) as exc:
            logger.warning("DPI 感知设置失败（%s）—— 将走 GetDpiForWindow 反算",
                           type(exc).__name__)

    try:
        after_w = u.GetSystemMetrics(0)
        after_h = u.GetSystemMetrics(1)
    except Exception:                          # noqa: BLE001
        after_w, after_h = before, before

    # ★ 实测判据：真正切到 aware 后 GetSystemMetrics 会从逻辑像素变成物理像素
    _DPI_AWARE = (after_w != before) or (_proc_is_dpi_aware() is True)
    if _DPI_AWARE:
        logger.info("DPI 感知已生效：GetSystemMetrics %sx%s -> %sx%s",
                    before, u.GetSystemMetrics(1), after_w, after_h)
    else:
        logger.warning(
            "DPI 感知**未生效**（GetSystemMetrics 仍为 %sx%s）。"
            "抓帧会用 GetDpiForWindow 反算物理尺寸兜底。", before, after_w)
    return _DPI_AWARE


# import 时执行一次 —— 早于任何 Capturer 实例化
ensure_dpi_awareness()

#: 连续失败多少次才抛异常（防止偶发失败中断循环）
DEFAULT_FAIL_THRESHOLD = 3


class CaptureChannel(str, Enum):
    DXCAM = "dxcam"
    #: GDI BitBlt。2026-10-10 顶替 PRINTWINDOW 入链（见模块 docstring）。
    BITBLT = "bitblt"
    MSS = "mss"
    #: Windows Graphics Capture（2026-10-10 入链）。
    #:
    #: ★ 为什么排在最前
    #:   ---------------------------------
    #:   BitBlt / DXcam / mss 三条路在「硬件加速 + 全屏」下**全部**只能拿到
    #:   黑帧，而它们各自的失败形态又完全不一样（DXcam 抛 ``Invalid Region``，
    #:   BitBlt 静默返回全 0 的合法图）。WGC 走的是 **DWM 合成管道**，抓的是
    #:   Windows 正在合成的表面，硬件加速窗口在这条路上是它的主场。
    #:
    #: ⚠️ WGC 抓不到**真·独占全屏**（那种模式会绕过 DWM）—— 这不是实现问题，
    #:    是 Windows 的设计边界。真机实测原神（无边框全屏 2560x1600）可抓。
    WGC = "wgc"


@dataclass(frozen=True)
class CapturedFrame:
    """一帧截图。

    ``image`` 是 HxWx3 的 BGR（OpenCV 约定）。``rect`` 是游戏窗口客户区
    在**屏幕坐标系**中的 (x, y, w, h)，可能为负数（多屏左/上方显示器）。
    """

    image: np.ndarray
    rect: tuple[int, int, int, int]
    timestamp_ms: int
    frame_seq: int
    channel: CaptureChannel

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class CaptureError(RuntimeError):
    """截图失败。**连续** N 次才抛（见 fail_threshold）。"""


class WindowNotFoundError(CaptureError):
    """找不到目标窗口。游戏没开 / 标题变了。"""


class Capturer:
    """三通道截图器。

    用法::

        with Capturer("原神") as cap:
            frame = cap.grab()
            print(frame.channel, frame.width, frame.height)

    线程安全：所有 grab() 走同一把 **RLock**。
    """

    def __init__(
        self,
        window_title_substr: str,
        channel_order: list[CaptureChannel] | None = None,
        *,
        fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
        allow_fallback_fullscreen: bool = False,
        force_channel: "str | CaptureChannel | None" = None,
        window_wait_s: float | None = None,
        visible_wait_s: float | None = None,
    ) -> None:
        self._title = window_title_substr
        self._order = self._normalize_order(channel_order)
        self._fail_threshold = max(1, int(fail_threshold))
        self._allow_fallback_fullscreen = allow_fallback_fullscreen
        #: 等窗口的上限。显式给 0 = 不等（单测/CI 不该真等 30 秒）。
        self._window_wait_s = (self.WINDOW_WAIT_S if window_wait_s is None
                               else max(0.0, float(window_wait_s)))
        #: 等「画面真的亮起来」的上限。显式给 0 = 不等。
        #: 跟 ``_window_wait_s`` 分开是因为它们治的是两种病：
        #: 前者等**窗口**，后者等**内容**（加载中/过场时窗口在、但画面是黑的）。
        self._visible_wait_s = (self.VISIBLE_WAIT_S if visible_wait_s is None
                                else max(0.0, float(visible_wait_s)))

        # ★ 强制通道（§31）。指定后 ``_order`` 只剩它一个 ——
        #   ``_demote_channel()`` 自然无处可降（它只往后找），
        #   所以不需要额外的分支。
        self._force_channel = self._normalize_force(force_channel)
        if self._force_channel is not None:
            if self._order != [self._force_channel]:
                was_head = (self._order and self._order[0] is self._force_channel)
                logger.warning(
                    "强制截图通道 = %s（%s）",
                    self._force_channel.value,
                    "只保留它，**不再降级到其他通道**"
                    if was_head else
                    f"已丢弃原 channel_order={[c.value for c in self._order]}")
            self._order = [self._force_channel]

        # 必须�� RLock：grab() 可能被模板匹配等逻辑嵌套调用
        self._capture_lock = threading.RLock()

        self._hwnd: int | None = None
        self._win_rect: tuple[int, int, int, int] | None = None
        self._active: CaptureChannel | None = None
        self._dxcam_out = None          # dxcam.output(index)
        self._bitblt = None              # _BitBltSession（仅 BITBLT 通道）
        self._mss_inst = None           # mss.mss()
        self._frame_seq = 0
        self._consecutive_fail = 0
        #: 连续空帧计数。dxcam 在某些窗口（硬件加速 / 最小化的）会间歇性
        #: 返回 None —— **空帧本身不是故障**，只有一直空才是。
        self._empty_streak = 0
        self._closed = False
        #: WGC 会话（见 :class:`_WgcSession`）。``None`` = 没起。
        self._wgc = None
        #: 窗口句柄变了（游戏切分辨率/重开）后置位，由 :meth:`_grab_wgc`
        #: 惰性重建 —— WGC 是按 hwnd 绑定的，老会话抓的是已经没了的窗口。
        self._wgc_dirty = False

    # -- 通道顺序 ----------------------------------------------------------
    #: 连续多少帧空才判定「该通道在当前窗口上不可用」
    EMPTY_FRAME_THRESHOLD = 8
    #: WGC 启动后等首帧的上限（秒）。WGC 是**推送**式，起会话和第一帧之间
    #: 天然有几百毫秒延迟；不像 GDI 那样「构造完立刻就有图」。
    #: 这里给 3 秒：超过就判定这条通道起不来，让 open() 往下换通道。
    WGC_FIRST_FRAME_S: float = 3.0

    @staticmethod
    def _normalize_force(
        spec: "str | CaptureChannel | None",
    ) -> CaptureChannel | None:
        """把 ``force_channel`` 归一成 :class:`CaptureChannel` 或 ``None``。

        接受的大小写/别名：
        ``"dxcam"`` / ``"bitblt"``（也认 ``"bit_blt"``、``"gdi"``）/ ``"mss"`` /
        ``"wgc"``（也认 ``"windows_capture"``）。
        传 :class:`CaptureChannel` 成员或 ``None`` 也认。

        ★ ``"printwindow"`` / ``"pw"`` 已于 2026-10-10 **移除**。
          留着别名会让 ``force_channel="printwindow"`` 静默变成"未知通道"报错，
          而用户以为只是拼写不同 —— 与其让人撞墙，不如响亮地失败。

        :raises ValueError: 认不出来的名字。**响亮失败** —— 静默当成
            "没指定" 会让用户以为自己在强制某个通道，实际走了完整降级链，
            正好复现 §31 要修的那个 bug。
        """
        if spec is None:
            return None
        if isinstance(spec, CaptureChannel):
            return spec
        key = str(spec).strip().lower().replace("-", "_").replace(" ", "_")
        alias = {
            "dxcam": CaptureChannel.DXCAM,
            "dx": CaptureChannel.DXCAM,
            "bitblt": CaptureChannel.BITBLT,
            "bit_blt": CaptureChannel.BITBLT,
            "gdi": CaptureChannel.BITBLT,
            "mss": CaptureChannel.MSS,
            "wgc": CaptureChannel.WGC,
            "windows_capture": CaptureChannel.WGC,
        }
        if key in alias:
            return alias[key]
        raise ValueError(
            f"force_channel 认不出 {spec!r}。"
            f"可选：{', '.join(sorted(alias))}，或传 CaptureChannel 成员。"
        )

    @property
    def force_channel(self) -> CaptureChannel | None:
        """被强制的通道；``None`` = 走完整降级链。"""
        return self._force_channel

    @staticmethod
    def _normalize_order(
        channel_order: list[CaptureChannel] | list[str] | None,
    ) -> list[CaptureChannel]:
        """把 channel_order 归一成 ``list[CaptureChannel]``。

        ★ 这里必须转换 + 校验。早先直接 ``channel_order or [...]`` 存下来，
          传字符串列表（``["dxcam", "mss"]``）会一路带到 ``_init_channel``，
          而那里是 ``if ch is CaptureChannel.DXCAM`` —— 字符串永远不 ``is``
          枚举成员，于是三个通道全部静默返回 False 且**不抛异常**，
          最终报「没有可用的截图通道 / 各通道错误：（无）」。
          症状离病因十万八千里，根因就是入口不校验类型。
        """
        if channel_order is None:
            # ⚠️ 2026-10-10 顺序调整（两次）：
            #   1) BitBlt 提到最前 —— 原顺序 dxcam -> bitblt -> mss 在真机
            #      全屏下必然卡死：dxcam 抓的是屏幕坐标矩形，窗口全屏越界时抛
            #      ``Invalid Region: Region should be in 2560x1600``，
            #      而那条路原先**不触发降级**（见 grab() 的 except Exception 分支）。
            #      BitBlt 走窗口自己的 GDI DC，完全不看屏幕坐标。
            #   2) WGC 再提到 BitBlt 前面 —— 因为真机证明 BitBlt 在硬件加速
            #      全屏下**抓到的也是全黑**（luma 逐帧精确 0.0），只是失败得
            #      比 dxcam 更安静。WGC 走 DWM 合成管道，同场景实测可用。
            #   与 defaults.toml / loader.py 的默认值三处保持一致。
            return [CaptureChannel.WGC, CaptureChannel.BITBLT,
                    CaptureChannel.DXCAM, CaptureChannel.MSS]
        out: list[CaptureChannel] = []
        for item in channel_order:
            if isinstance(item, CaptureChannel):
                out.append(item)
                continue
            try:
                # CaptureChannel 是 str Enum，字符串可直接转
                out.append(CaptureChannel(item))
            except ValueError as exc:
                valid = [c.value for c in CaptureChannel]
                raise CaptureError(
                    f"未知的截图通道 {item!r}。合法值：{valid}"
                    f"（必须是 CaptureChannel 枚举或其字符串值）"
                ) from exc
        if not out:
            raise CaptureError(
                "channel_order 是空的 —— 至少要保留一种截图通道。"
                "传 None 表示使用默认顺序。")
        return out

    # -- 上下文管理 ------------------------------------------------------

    def __enter__(self) -> "Capturer":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        with self._capture_lock:
            self._closed = True
            if self._dxcam_out is not None:
                try:
                    self._dxcam_out.stop()
                except Exception:
                    pass
                self._dxcam_out = None
            if self._bitblt is not None:
                try:
                    self._bitblt.close()
                except Exception:
                    pass
                self._bitblt = None
            if self._mss_inst is not None:
                try:
                    self._mss_inst.close()
                except Exception:
                    pass
                self._mss_inst = None
            # ★ WGC 要**先于**其它通道关掉：它是后台线程在推帧，
            #   不停就等于让一个线程继续引用已经半销毁的 Capturer。
            self._close_wgc()
            self._hwnd = None
            self._active = None

    # -- 初始化 ----------------------------------------------------------

    #: 等窗口出现的上限（秒）。``0`` = 不等（单测 / CI 用）。
    #: ``None`` 表示用这个默认值。
    WINDOW_WAIT_S: float = 30.0
    #: 轮询间隔（秒）。0.5s 足够跟手，又不至于把 CPU 跑满。
    WINDOW_POLL_S: float = 0.5
    #: 等「画面亮起来」的上限（秒）。``0`` = 不等。
    VISIBLE_WAIT_S: float = 10.0
    #: 判定「这一帧有内容」的亮度下限（0-255）。
    #:
    #: ★ 为什么与 :data:`autopilot.perception.builder.BLACK_FRAME_LUMA` 同值
    #:   ------------------------------------------------------------------
    #:   那边是**感知层**的口径（这帧能不能用），这边是**截图层**的口径
    #:   （这条通道在当前窗口上能不能用）。两处用同一个数，才不会出现
    #:   「截图器认为有画面、感知层认为全黑」这种两边打架、且谁都说不清
    #:   是谁判断错了的状态。
    BLACK_FRAME_LUMA: float = 8.0

    def _wait_for_window(self) -> bool:
        """轮询等窗口出现且**客户区可用**。返回是否等到。

        ★ 为什么必须等，而不是"扫不到就抛"
          --------------------------------
          真实启动姿势是"双击 -> 引擎起来 -> 用户切回原神"。切窗口那一刻
          原神必然**最小化**，``_refresh_window`` 会"找到"它但客户区退化成
          0x0 / -32000 哨兵 —— 老实说这时该等，可老代码直接进通道探测，
          三条通道全废 -> ``WindowNotFoundError`` -> 引擎秒退。用户看着
          就是"我一启动它就退了"，而他只是还没来得及 Alt+Tab。

          轮询等待把"精确计时"这个用户做不到的要求，换成"切回去就行"。

        ★ 为什么判**可用**而不是判"找到"
          -----------------------------
          最小化的窗口 ``_refresh_window`` 返回 True —— 它确实找到了窗口。
          只看返回值的话会误判成"找到了，可以开工"，然后在通道探测里
          以一条看不懂的报错收场。

        ★ 等待期只播报一次
          ------------------
          30 秒 / 0.5 秒 = 60 次，不节流的话日志里全是同一句话。
        """
        timeout = self._window_wait_s
        if timeout <= 0:
            return _rect_usable(self._win_rect)

        deadline = time.monotonic() + timeout
        announced = False
        while True:
            # ★ 必须用 force=False 包起来
            #   ``_refresh_window(force=True)`` 在找不到窗口时是**抛**
            #   ``WindowNotFoundError`` 而不是返回 False —— 直接调它，
            #   等待循环第一轮就炸出去，等了个寂寞（2026-10-10 真踩）。
            #   这里要的是"这轮找到没有"这个布尔，不是那个异常。
            self._refresh_window(force=False)
            if _rect_usable(self._win_rect):
                if announced:
                    logger.info("窗口已出现：%s", self._win_rect)
                return True
            if not announced:
                logger.warning(
                    "等待窗口 %r 出现或还原（最多 %.0f 秒，每 %.1f 秒查一次）"
                    "—— 现在请切回游戏全屏", self._title, timeout,
                    self.WINDOW_POLL_S)
                announced = True
            if time.monotonic() >= deadline:
                logger.error(
                    "等了 %.0f 秒仍没有可用窗口：最后客户区=%r",
                    timeout, self._win_rect)
                return False
            time.sleep(self.WINDOW_POLL_S)

    def _wait_for_visible_frame(self) -> bool:
        """窗口已经在，但**画面还是黑的**时再等一会儿。返回是否等到亮帧。

        ★ 为什么要单独等画面（2026-10-10 真机事故）
          -----------------------------------------
          ``_wait_for_window`` 只保证「窗口存在且客户区非 0x0」。游戏刚起来
          时窗口往往先出现、内容后到（加载画面 / 过场动画），这时候抓到的
          是一张合法的、尺寸正确的**全黑图**。而「尺寸对」在老判定里等于
          「可用」，于是引擎一路 BLACK_FRAME 跑满 2655 tick，
          **一次都没有降级过**（真机事件流 ``luma: 0.0``，逐帧精确为 0）。

        ★ 探测帧只判一次还不够
          ----------------------
          所以这里循环探测，而不是抓一帧看一眼就收工 —— 单帧黑可能只是
          切屏那一瞬间的合成抖动。

        ★ 为什么超时**不**抛「窗口没找到」
          -------------------------------
          窗口明明在，报「找不到窗口」会把排查方向直接带偏。这里把
          「等不到亮画面」当成它自己的一种失败，由 :meth:`open` 决定怎么办。
        """
        timeout = self._visible_wait_s
        if timeout <= 0:
            return True
        deadline = time.monotonic() + timeout
        announced = False
        attempt = 0
        last_luma = -1.0
        while True:
            attempt += 1
            try:
                img = self._grab_once()
            except CaptureError as exc:
                # 探测期抓不到不算致命：可能窗口正在重开。继续等。
                logger.debug("等亮画面时抓帧失败（第 %d 次）：%s", attempt, exc)
                img = None
            last_luma = _frame_luma(img)
            if last_luma >= self.BLACK_FRAME_LUMA:
                if announced:
                    logger.info("画面已可见：亮度 %.1f（探测 %d 次，用时约 %.1f 秒）",
                                last_luma, attempt, timeout - max(
                                    0.0, deadline - time.monotonic()))
                return True
            if not announced:
                logger.warning(
                    "窗口已找到但画面还是黑的（亮度 %.1f < %.1f）"
                    "（最多等 %.0f 秒）—— 游戏可能还在加载/过场",
                    last_luma, self.BLACK_FRAME_LUMA, timeout)
                announced = True
            if time.monotonic() >= deadline:
                logger.error("等了 %.0f 秒画面仍然全黑（最后一次亮度 %.1f）",
                             timeout, last_luma)
                return False
            time.sleep(self.WINDOW_POLL_S)

    def open(self) -> None:
        """定位窗口并探测可用通道。**失败直接抛**，不降级到"看不见"。

        启动期就该炸 —— 一个静默失败的截图器会让后面所有环节都拿到空画面。
        但**先等一会儿**：见 :meth:`_wait_for_window` 的注释。
        """
        with self._capture_lock:
            if self._active is not None:
                return
            if not self._wait_for_window():
                # 等过了还没窗口：直接抛，外面 `_build_stack` 会转成
                # WindowNotFoundError。消息里带上最后看到的矩形 ——
                # "找不到窗口"和"找到了但被最小化"是两回事。
                raise WindowNotFoundError(
                    f"等待 {self._window_wait_s:.0f} 秒后仍没有可用窗口"
                    f"（{self._title!r}，最后客户区={self._win_rect}）。"
                    f"请确认游戏已启动且**没有最小化**。"
                )
            self._log_dpi_diagnostics()
            errors: list[str] = []
            for ch in self._order:
                try:
                    if not self._init_channel(ch):
                        # ★ 返回 False 但没抛异常 —— **也要记一条**。
                        #   _init_channel 对「没装这个库」「设备不存在」等
                        #   情况是 return False 而不 raise，早先这里不记，
                        #   于是 errors 为空、报错写着"各通道错误：（无）"，
                        #   用户拿到一句自相矛盾且毫无线索的提示。
                        errors.append(f"{ch.value}: 不可用（初始化返回 False，"
                                      f"详见 DEBUG 日志）")
                        continue
                    self._active = ch
                    logger.info("capture channel active: %s", ch.value)
                except Exception as exc:  # noqa: BLE001 - 探测阶段吞掉并记录
                    errors.append(f"{ch.value}: {exc}")
                    logger.debug("channel %s unavailable: %s", ch.value, exc)
                    continue
                # ★★ 黑帧把关（2026-10-10 真机事故的核心修复）
                #   ------------------------------------------------
                #   老判定只有「抓不抓得到」：``probe is None`` /
                #   ``probe.size == 0``。**一张 2560x1600 的纯黑图两条都能
                #   通过** —— 尺寸对、内容空。于是 bitblt 被判「可用」，
                #   引擎一路 BLACK_FRAME 跑满 2655 tick，一次都没降级
                #   （事件流里 ``luma`` 逐帧精确为 ``0.0``，没有任何
                #   CAPTURE_FAILED —— 它真心觉得自己在工作）。
                #
                #   这里改成「这条通道得在 10 秒内交出一张亮画面」。
                #   宽限期是必须的：游戏刚起来时窗口先出现、内容后到
                #   （加载画面 / 过场动画），一帧黑就否决会把正在
                #   正常加载的游戏打死。超时就换下一条通道。
                if self._wait_for_visible_frame():
                    return
                errors.append(
                    f"{ch.value}: 画面一直全黑（亮度 < {self.BLACK_FRAME_LUMA}，"
                    f"等了 {self._visible_wait_s:.0f} 秒）—— 硬件加速窗口被 "
                    f"Windows 合成优化挡住，或游戏还没加载完")
                logger.warning("通道 %s 画面一直全黑，换下一条", ch.value)
                self._active = None
            raise CaptureError(
                "没有可用的截图通道。请检查：\n"
                "  1) 目标窗口是否存在（当前查找标题片段: %r）\n"
                "  2) **窗口是不是被最小化了** —— GDI（bitblt/mss）与 DXGI\n"
                "     都拿不到最小化窗口的画面，客户区会退化成 0x0。"
                "先还原窗口再试。\n"
                "  3) 是否安装了 dxcam / mss\n"
                "  4) 截图器是否有屏幕捕获权限\n"
                "  5) **画面全黑** —— 各通道都抓到尺寸正确但内容为空的黑图。\n"
                "     多见于硬件加速窗口 + Windows 的 DirectX 交换链优化：\n"
                "     需在**游戏启动前**执行 apply_win11_bitblt_fix(apply=True)，\n"
                "     或把游戏切回**无边框窗口**再试。\n"
                "%s"
                "各通道错误（%d 条）：\n  - %s"
                % (self._title,
                   self._forced_channel_hint(),
                   len(self._order),
                   "\n  - ".join(errors) or "（无 —— 这不该发生，"
                                             "说明 _order 是空的）")
            )

    def _forced_channel_hint(self) -> str:
        """被强制通道时的针对性提示。"""
        if self._force_channel is None:
            return ""
        hints = {
            CaptureChannel.BITBLT:
                "  ★ 你强制了 bitblt。**窗口级抓取，游戏被遮挡也能抓到内容**\n"
                "    ⚠️ 但它是 GDI 路径：**窗口最小化时拿不到画面**"
                "（GetClientRect 返回 0x0）。\n"
                "    若抓到纯黑，先查 Win11 的 DirectX 交换链优化 —— 见\n"
                "    apply_win11_bitblt_fix() 的 docstring（需要游戏启动前设置）。\n",
            CaptureChannel.MSS: "  ★ 你强制了 mss —— 这是**屏幕坐标**抓取，"
                                "游戏被遮挡/失焦时会抓到遮挡窗口的内容。\n",
            CaptureChannel.DXCAM: "  ★ 你强制了 dxcam —— 硬件加速窗口上可能间歇空帧。\n",
            CaptureChannel.WGC:
                "  ★ 你强制了 wgc —— 它走 DWM 合成管道，硬件加速窗口是主场。\n"
                "    ⚠️ 但它抓不到**真·独占全屏**（那种模式绕过 DWM），\n"
                "    只在**无边框全屏/窗口化**下可靠。\n"
                "    需要 pip install windows-capture（>=2.0.1）。\n",
        }
        return hints.get(self._force_channel, "")

    def _init_channel(self, ch: CaptureChannel) -> bool:
        if ch is CaptureChannel.DXCAM:
            try:
                import dxcam  # noqa: PLC0415 - 懒加载，未装不影响其他通道
            except Exception:
                return False
            cam = dxcam.create(output_color="BGR")
            if cam is None:
                return False
            self._dxcam_out = cam
            # 立刻抓一帧验证真的能用
            probe = cam.grab(region=self._win_rect)
            if probe is None or probe.size == 0:
                self._restart_dxcam()
                return False
            return True
        if ch is CaptureChannel.MSS:
            try:
                import mss  # noqa: PLC0415
            except Exception:
                return False
            self._mss_inst = mss.MSS()
            # ★ mss 以前**根本不探测**，直接 return True —— 三条通道里唯一
            #   一条「从没验证过就宣布可用」的。补上探测，与另两条对齐。
            #   注意这里只判「抓得到」，**不判黑**：判黑要给等待余地
            #   （游戏加载中），那是 open() 的职责，见 _wait_for_visible_frame。
            try:
                probe = self._grab_mss()
            except CaptureError as exc:
                logger.debug("mss 不可用：%s", exc)
                return False
            return probe is not None and probe.size > 0
        if ch is CaptureChannel.BITBLT:
            # 跟 PrintWindow 当年一样先验 hwnd，但**多抓一帧**才算真能用 ——
            # 设备不支持 RC_BITBLT 时会话构造就会抛，这里自然返回 False。
            if self._hwnd is None:
                return False
            try:
                self._bitblt = _BitBltSession(self._hwnd)
            except CaptureError as exc:
                logger.debug("bitblt 不可用：%s", exc)
                self._bitblt = None
                return False
            probe = self._bitblt.grab()
            if probe is None or probe.size == 0:
                self._bitblt.close()
                self._bitblt = None
                return False
            return True
        if ch is CaptureChannel.WGC:
            # ★ WGC 排在链首，但**不能因为依赖缺失就报错** —— 没装
            #   windows-capture 的环境（CI、别的机器）必须能继续用
            #   bitblt/dxcam/mss 跑。这里 return False 让 open() 往下走。
            try:
                import windows_capture  # noqa: PLC0415,F401 - 仅探测可用性
            except ImportError as exc:
                logger.info("wgc 不可用：没装 windows-capture（%s）", exc)
                return False
            if self._hwnd is None:
                return False
            try:
                self._wgc = _WgcSession(self._hwnd)
            except Exception as exc:  # noqa: BLE001 - 设备/权限/D3D 都可能挂
                logger.debug("wgc 会话创建失败：%s", exc)
                self._close_wgc()
                return False
            probe = self._wgc.wait_first_frame(self.WGC_FIRST_FRAME_S)
            if probe is None:
                logger.info("wgc 起得来，但 %.1f 秒内没等到首帧（窗口可能最小化"
                            "或被保护）", self.WGC_FIRST_FRAME_S)
                self._close_wgc()
                return False
            logger.info("wgc 就绪：%dx%d", probe.shape[1], probe.shape[0])
            return True
        return False

    def get_active_channel(self) -> CaptureChannel:
        if self._active is None:
            raise CaptureError("Capturer 未初始化，请先调用 open()")
        return self._active

    # -- 窗口 ------------------------------------------------------------

    @staticmethod
    def window_dpi(hwnd: int) -> int:
        """窗口的 DPI（默认 96）。读不到时返回 96 == 缩放 1.0。"""
        try:
            u = ctypes.windll.user32
            u.GetDpiForWindow.argtypes = [ctypes.c_void_p]
            u.GetDpiForWindow.restype = ctypes.c_uint
            dpi = int(u.GetDpiForWindow(hwnd))
            return dpi if dpi > 0 else 96
        except (AttributeError, OSError, TypeError):
            return 96

    def _physical_rect(self, rect: tuple[int, int, int, int],
                       hwnd: int) -> tuple[tuple[int, int, int, int], float]:
        """把客户区矩形换算成**物理像素**。返回 ``(rect, scale)``。

        ⚠️ 只有在**进程 DPI-unaware** 时才需要换算。
           进程 aware 时 ``GetWindowRect`` 返回的本来就是物理像素，
           再乘一次就错了。

        DPI-unaware 时 Windows 会把坐标**虚拟化**：``GetWindowRect`` 返回
        逻辑像素，而 mss / dxcam 抓的是物理像素 —— 于是抓到的是屏幕
        左上角 1/scale 那一块（150% DPI 下只有 67%）。用
        ``GetDpiForWindow`` 反算乘回去。

        ⚠️ 多显示器 + 混合 DPI 时这只是**近似**（Windows 对虚拟化的处理
           并不是简单乘法）。单显示器场景实测准确。
        """
        x, y, w, h = rect
        if _DPI_AWARE:
            return (x, y, w, h), 1.0
        scale = self.window_dpi(hwnd) / 96.0
        if abs(scale - 1.0) < 1e-6:
            return (x, y, w, h), 1.0
        return (int(round(x * scale)), int(round(y * scale)),
                int(round(w * scale)), int(round(h * scale))), scale

    def _refresh_window(self, force: bool = False) -> bool:
        """重新查找窗口并计算客户区在**物理像素**上的矩形。"""
        hwnd, raw = _find_window(self._title, self._allow_fallback_fullscreen)
        if hwnd is None:
            if force:
                raise WindowNotFoundError(
                    f"找不到标题包含 {self._title!r} 的窗口。"
                    f"请确认游戏已启动且未最小化。"
                )
            return False
        rect = self._physical_rect(raw, hwnd)[0] if raw else None
        changed = hwnd != self._hwnd or rect != self._win_rect
        self._hwnd, self._win_rect = hwnd, rect
        if changed and self._active is CaptureChannel.DXCAM:
            # dxcam 绑定了区域，窗口移动/缩放后必须重建
            try:
                self._dxcam_out = None  # 交给下次 grab 重建
            except Exception:
                pass
        if changed and self._active is CaptureChannel.WGC:
            # WGC 是按 **hwnd** 绑定的。游戏切分辨率/重开会让窗口句柄变掉，
            # 老会话还在抓一个已经不存在的 hwnd —— 只置脏，重建交给
            # _grab_wgc 惰性做（这里正在抢锁，再起会话会死锁）。
            self._wgc_dirty = True
        return True

    def is_window_alive(self) -> bool:
        """窗口还在吗。游戏重启后用它判断是否需要重新绑定。"""
        with self._capture_lock:
            return self._refresh_window(force=False)

    @property
    def window_rect(self) -> tuple[int, int, int, int] | None:
        return self._win_rect

    # -- 抓帧 ------------------------------------------------------------

    def grab(self) -> CapturedFrame:
        """抓一帧。

        **连续** N 次失败才抛 CaptureError（单次失败重试后即返回）。
        这是为了不让偶发的窗口切换/合成抖动中断感知循环。
        """
        with self._capture_lock:
            if self._closed:
                raise CaptureError("Capturer 已关闭")
            if self._active is None:
                self.open()

            last_err: Exception | None = None
            # ★ 瞬时空帧的**独立**重试预算（2026-10-04）。
            #   不能占用 ``for _ in range(fail_threshold)`` 的次数：默认
            #   fail_threshold=3 而 EMPTY_FRAME_THRESHOLD=8，若共用预算，
            #   连续 3 次瞬时空帧就会走到循环末尾抛「截图连续 3 次失败」，
            #   反而比修复前更脆。所以这里用 ``while`` 单独给空帧一份
            #   更宽的预算 —— 只有它自己确认「这通道真的坏了」才收手。
            _empty_retry = 0
            # ★ ``while`` 必须自带总次数上限：下面两个 except 分支在
            #   「刷新成功但重试仍失败」时会 continue，若无上限就是死循环
            #   （原来的 ``for _ in range(fail_threshold)`` 隐含兜底）。
            #   预算要够宽以容纳空帧重试：每条通道一次降级机会 × 空帧阈值。
            _attempt = 0
            _max_attempt = self._fail_threshold * max(
                2, self.EMPTY_FRAME_THRESHOLD)
            while True:
                _attempt += 1
                if _attempt > _max_attempt:
                    self._consecutive_fail += 1
                    raise CaptureError(
                        f"截图重试 {_attempt} 次仍未成功：{last_err}") from last_err
                try:
                    img = self._grab_once()
                    if img is None or img.size == 0:
                        # ★★ 2026-10-04 修：瞬时空帧**不再**直接抛
                        #   _grab_dxcam() 对单次空帧是 return None（只有连续
                        #   EMPTY_FRAME_THRESHOLD 次才抛 CaptureError），因为
                        #   dxcam 在窗口移动/遮挡/切换显示器时会间歇性返回空。
                        #   但旧代码在这里 ``raise CaptureError("截图为空")``，
                        #   把那个「单次空帧是正常的」设计**完全架空**了：
                        #   一次瞬时空帧 -> 抛错 -> _demote_channel() ->
                        #   **永久**降到下一通道（本类无任何回退路径）。
                        #   真机实测：60s 跑到 t=38.2s 降级，此后 22s 全程 mss，
                        #   而 dxcam 空帧峰值 0（压根没统计到那一帧）。
                        #
                        #   正确做法：空帧**留在本通道重试**，让 _empty_streak
                        #   去判断是不是真的坏了。只有它自己抛出来
                        #   （连续 N 次空）才降级。
                        _empty_retry += 1
                        if _empty_retry > self.EMPTY_FRAME_THRESHOLD:
                            # ⚠️ 这里必须用**局部** ``_empty_retry``：
                            #   写成 ``self._empty_retry`` 会抛 AttributeError
                            #   （该属性根本不存在），于是「通道真的坏了」
                            #   这条路径永远走不到正确的 CaptureError，
                            #   降级逻辑也就形同虚设。
                            #   —— 这个错是单测 T3 抓出来的。
                            raise CaptureError(
                                f"连续 {_empty_retry} 帧返回空"
                                f"（空帧计数 {self._empty_streak}）—— "
                                f"该通道在当前窗口上不可用")
                        logger.debug("瞬时空帧 #%d，同通道重试",
                                     _empty_retry)
                        if not self._refresh_window(force=False):
                            self._consecutive_fail += 1
                            if self._consecutive_fail >= self._fail_threshold:
                                raise WindowNotFoundError(
                                    f"窗口已消失（{self._title!r}）")
                        continue
                    self._consecutive_fail = 0
                    self._frame_seq += 1
                    return CapturedFrame(
                        image=img,
                        rect=self._win_rect or (0, 0, img.shape[1], img.shape[0]),
                        timestamp_ms=int(time.time() * 1000),
                        frame_seq=self._frame_seq,
                        channel=self._active,
                    )
                except WindowNotFoundError:
                    raise
                except CaptureError as exc:
                    # 通道层面的问题（如连续空帧）-> 降到下一条通道再试，
                    # 而不是白等 fail_threshold 次全打在同一个坏通道上
                    last_err = exc
                    if self._demote_channel():
                        logger.info("已降级到 %s，继续重试",
                                    self._active.value)
                        continue
                    if not self._refresh_window(force=False):
                        self._consecutive_fail += 1
                        if self._consecutive_fail >= self._fail_threshold:
                            raise WindowNotFoundError(
                                f"窗口已消失（{self._title!r}）"
                            ) from exc
                    # 同上：降不了级但窗口还在 -> 重试当前通道
                    continue
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    # ★★ 这里**也必须**尝试降级（2026-10-10 真机事故）
                    #   ------------------------------------------------
                    #   上一版只把「返回空帧」当成通道故障，而 dxcam 在
                    #   窗口矩形越界时抛的是
                    #   ``Invalid Region: Region should be in 2560x1600``
                    #   —— 一个**普通异常**，落在本分支里。而本分支原来
                    #   只刷新窗口、从不调 ``_demote_channel()``，于是
                    #   dxcam 抛一次 -> continue -> 再抛一次 -> 空转到
                    #   ``截图重试 25 次仍未成功``，**通道永远不降级**。
                    #
                    #   推论：一条通道"抛异常"和"返回空"一样，都是它自己
                    #   在当前窗口上不可用；只有 ``WindowNotFoundError``
                    #   才是窗口本身没了（那个是真的没救，得重抛）。
                    if self._demote_channel():
                        logger.info("通道 %s 抛异常，已降级到 %s，继续重试",
                                    type(exc).__name__, self._active.value)
                        continue
                    # 降不了级：多半是窗口矩形本身不可用（全屏越界 / 最小化
                    # 退化成 0x0）。换通道也救不了，直接把**真正的原因**
                    # 说出来 —— 别让用户去猜 dxcam 的内部报错。
                    if not self._refresh_window(force=False):
                        self._consecutive_fail += 1
                        if self._consecutive_fail >= self._fail_threshold:
                            raise WindowNotFoundError(
                                f"窗口已消失（{self._title!r}）"
                            ) from exc
                    elif not _rect_usable(self._win_rect):
                        raise CaptureError(
                            f"窗口客户区无效（rect={self._win_rect}）—— "
                            f"全屏/越界时部分通道会直接报错（{exc}）。"
                            f"请把游戏切回**窗口化或无边框**，或确认窗口"
                            f"完整落在屏幕内。"
                        ) from exc
                    # ⚠️ 这里**必须** continue：``while`` 的退出条件
                    #   只有 return / raise / _max_attempt 三种。
                    #   少了它就会原地空转，_attempt 疯涨到上限才停 ——
                    #   行为退化且毫无意义。
                    continue

    def _log_dpi_diagnostics(self) -> None:
        """构造时打印一次 DPI 诊断。**不要每帧刷屏。**

        要看的就是「逻辑 vs 物理」对不对得上：
        进程 DPI-unaware 且 DPI > 96 时，抓帧只会覆盖屏幕左上角
        1/scale 那一块（150% DPI 下只有 67%）。
        """
        u = ctypes.windll.user32
        dpi = self.window_dpi(self._hwnd) if self._hwnd else 96
        logical = (u.GetSystemMetrics(0), u.GetSystemMetrics(1))
        raw = _find_window(self._title, self._allow_fallback_fullscreen)[1]
        phys = self._win_rect
        logger.info(
            "[DPI] aware=%s dpi=%d (scale=%.3f) | 逻辑屏幕=%dx%d | "
            "窗口客户区=%s -> 物理=%s",
            _DPI_AWARE, dpi, dpi / 96.0, logical[0], logical[1], raw, phys)
        # ★ 空矩形必须**先**判掉（2026-10-04 修）
        #   窗口没找到 / 已最小化时 ``_win_rect`` 是 ``(0, 0, 0, 0)``。
        #   旧代码的 ``if phys and ...`` 拦不住它 —— 非空元组恒为真，
        #   于是拿 0 去比逻辑屏幕，**必然**打出
        #   「抓帧尺寸 0x0 超过逻辑屏幕 2560x1600」。
        #   0x0 不可能「超过」任何东西，而这条告警会把排查方向从
        #   「窗口没找到」误导到「DPI 缩放不对」，后者是纯粹的白跑。
        if phys and (phys[2] <= 0 or phys[3] <= 0):
            logger.warning(
                "[DPI] 窗口客户区矩形无效（%dx%d）—— 窗口可能未启动、已最小化，"
                "或标题 '%s' 没匹配上。**这不是 DPI 问题**，先确认窗口。",
                phys[2], phys[3], self._title)
        elif phys and (abs(phys[2] - logical[0]) > 4 or abs(phys[3] - logical[1]) > 4):
            logger.warning(
                "[DPI] 抓帧尺寸 %dx%d 超过逻辑屏幕 %dx%d —— "
                "若这是预期的（DPI-unaware 已反算）可忽略；"
                "若实际抓到的画面仍只有左上角一块，请报这个值。",
                phys[2], phys[3], logical[0], logical[1])

    def _grab_once(self) -> np.ndarray | None:
        ch = self._active
        if ch is CaptureChannel.DXCAM:
            return self._grab_dxcam()
        if ch is CaptureChannel.MSS:
            return self._grab_mss()
        if ch is CaptureChannel.BITBLT:
            return self._grab_bitblt()
        if ch is CaptureChannel.WGC:
            return self._grab_wgc()
        raise CaptureError(f"未初始化的通道: {ch}")

    def _grab_dxcam(self) -> np.ndarray | None:
        if self._dxcam_out is None:
            import dxcam  # noqa: PLC0415

            self._dxcam_out = dxcam.create(output_color="BGR")
        img = self._dxcam_out.grab(region=self._win_rect)
        if img is None or getattr(img, "size", 0) == 0:
            # dxcam 在硬件加速 / 窗口被遮挡 / 刚移动时会**间歇性**返回
            # None。单次空帧是正常的，不该当故障 —— 但 camera 实例一旦
            # 进入坏状态会**一直**返回 None，不重建的话后面每次重试
            # 都还是空，永远好不了。所以每次空帧都重建实例。
            self._empty_streak += 1
            self._restart_dxcam()
            logger.debug("dxcam 空帧 #%d，已重建 camera 实例",
                         self._empty_streak)
            if self._empty_streak >= self.EMPTY_FRAME_THRESHOLD:
                # 连续空 -> 这个通道在当前窗口上确实不可用，换下一条
                raise CaptureError(
                    f"dxcam 连续 {self._empty_streak} 帧返回空（窗口矩形 "
                    f"{self._win_rect}）—— 该通道在当前窗口上不可用")
            return None
        self._empty_streak = 0
        return img

    def _restart_dxcam(self) -> None:
        """彻底丢弃 dxcam camera 实例，下次 grab 时重建。

        只 ``stop()`` 不够 —— DXGI 的 output 对象在坏掉后不重新创建
        就一直是坏的。所以置 None 让 ``_grab_dxcam`` 下次重新 create。
        """
        old = self._dxcam_out
        self._dxcam_out = None
        if old is not None:
            for meth in ("stop", "release", "close"):
                fn = getattr(old, meth, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001
                        pass
                    break

    def _demote_channel(self) -> bool:
        """当前通道用不了了，降到 ``_order`` 里的下一个。返回是否降级成功。"""
        if self._active is None:
            return False
        try:
            idx = self._order.index(self._active)
        except ValueError:
            return False
        for nxt in self._order[idx + 1:]:
            try:
                if self._init_channel(nxt):
                    old = self._active
                    # 离开 WGC 就要把它的后台线程停掉 —— 否则采集线程会一直
                    # 往一个已经没有主人的会话里推帧（泄漏 + 句柄不释放）。
                    if old is CaptureChannel.WGC:
                        self._close_wgc()
                    self._active = nxt
                    self._empty_streak = 0
                    logger.warning("截图通道降级：%s -> %s",
                                   old.value, nxt.value)
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("降级到 %s 失败：%s", nxt.value, exc)
        return False

    def _grab_mss(self) -> np.ndarray | None:
        if self._mss_inst is None:
            import mss  # noqa: PLC0415

            self._mss_inst = mss.MSS()
        x, y, w, h = self._win_rect or (0, 0, 0, 0)
        if w <= 0 or h <= 0:
            raise CaptureError("窗口矩形无效")
        shot = self._mss_inst.grab({"left": x, "top": y, "width": w, "height": h})
        return np.asarray(shot)[:, :, :3]  # BGRA -> BGR

    def _grab_bitblt(self) -> np.ndarray | None:
        """GDI BitBlt 抓窗口客户区。2026-10-10 顶替 PrintWindow。

        会话失效就**重建一次再抓**——与 BetterGI 的 ``BitBltCapture``
        同策略（``_lastCaptureFailed`` / ``_session.Invalid`` -> 重建）。
        GDI 对象会因窗口销毁、显示器切换、分辨率变化而失效，而重建一次
        就能救回来；不重建的话，后面每一次重试都会拿到同一张空图。
        """
        for attempt in (0, 1):
            if self._bitblt is None:
                if self._hwnd is None:
                    raise CaptureError("bitblt 需要有效的 hwnd")
                self._bitblt = _BitBltSession(self._hwnd)
            img = self._bitblt.grab()
            if img is not None and img.size > 0:
                return img
            # 抓不到：先重建会话，第二轮还不行才认定这条通道坏了
            logger.debug("bitblt 第 %d 次抓取为空/失效，重建会话", attempt + 1)
            self._bitblt.close()
            self._bitblt = None
        return None

    def _grab_wgc(self) -> np.ndarray | None:
        """WGC 取「最新帧槽位」。返回 BGR（HxWx3）。"""
        if self._wgc is None:
            raise CaptureError("wgc 会话不存在")
        if self._wgc_dirty:
            # 窗口句柄变过了 —— 老会话抓的是已经没了的窗口，重开一个。
            # 惰性重建而不是在 _refresh_window 里做：那里正持有
            # ``_capture_lock``，而起会话要等首帧，再等就自锁了。
            self._wgc_dirty = False
            hwnd = self._hwnd
            self._close_wgc()
            if hwnd is None:
                raise CaptureError("wgc 需要有效的 hwnd")
            logger.info("窗口句柄已变，重建 wgc 会话（hwnd=%s）", hwnd)
            self._wgc = _WgcSession(hwnd)
        if self._wgc.closed:
            raise CaptureError(
                "wgc 会话已被系统关闭（窗口关闭/最小化，或内容受保护）")
        return self._wgc.grab()

    def _close_wgc(self) -> None:
        """关掉 WGC 会话。可重复调用。"""
        sess = self._wgc
        self._wgc = None
        self._wgc_dirty = False
        if sess is not None:
            sess.close()


# --------------------------------------------------------------------------
# 平台相关
# --------------------------------------------------------------------------


def _rect_usable(rect: tuple[int, int, int, int] | None) -> bool:
    """窗口矩形能不能用来抓帧。

    ★ 为什么要单独判
      --------------
      ``(0, 0, 0, 0)`` 是**非空元组**，``if rect:`` 拦不住它（老 bug 就是
      这么写的，注释在 ``_log_dpi_diagnostics`` 里）。而全屏越界时矩形会是
      ``(-32000, -32000, 0, 0)`` 这种哨兵值 —— dxcam 拿到它会直接抛
      ``Invalid Region: Region should be in 2560x1600``，一句话看不出
      「是窗口坏了」。抓不到的时候先把这件事说清楚，比转述第三方库的
      内部报错有用得多。

    :returns: 宽高都为正、且不是最小化哨兵坐标时才算可用。
    """
    if not rect or len(rect) != 4:
        return False
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return False
    # 最小化窗口的经典哨兵：坐标 -32000/-21333 之类，尺寸归零或极小
    return not (x <= -30000 and y <= -30000)


# --------------------------------------------------------------------------
# WGC（Windows Graphics Capture）—— 2026-10-10 入链
# --------------------------------------------------------------------------
# 排在链首的理由见 :attr:`CaptureChannel.WGC`。三条铁律都是踩出来的：
#
# 1. **必须 ``start_free_threaded()``**。``start()`` 会进 Win32 消息循环且不
#    返回 —— 主循环会被直接卡死。
# 2. **两个 handler 都必须在 start 之前注册**，少一个就是 ``RuntimeError``；
#    而且 ``@capture.event`` 是**按函数名分派**的，名字必须正好是
#    ``on_frame_arrived`` / ``on_closed``，所以下面两个方法不能改名、
#    也不能加下划线前缀。
# 3. **不能信 ``Frame`` 的生命周期**。``frame.frame_buffer`` 是指向原生
#    mapped frame 的**零拷贝视图**，而内存的所有者是 ``Frame`` 本体。
#    换帧 = 旧 Frame 被释放 = 之前发出去的视图全部悬空。所以 :meth:`grab`
#    永远返回**拷贝**，绝不把视图交给调用方。
#
#   —— 这三条来自 `autopilot/tools/wgc_smoke.py` 的真机踩坑，不是文档抄的。


class _WgcSession:
    """一个窗口上的 WGC 采集会话（后台线程推送 + 主线程拉取最新帧）。

    线程模型：``windows_capture`` 的采集跑在自己的线程上，通过
    :meth:`on_frame_arrived` 把帧推进来；我们的 :meth:`grab` 在主循环线程
    拉。两者之间只有「最新帧槽位」，用一把锁隔开。
    """

    def __init__(self, hwnd: int) -> None:
        if not hwnd:
            raise CaptureError("wgc 需要有效的 hwnd")
        from windows_capture import WindowsCapture  # noqa: PLC0415 - 懒加载

        self._hwnd = hwnd
        self._lock = threading.Lock()
        #: (frame, seq)。**存 frame 本体**而不是 ndarray —— 见模块头铁律 3。
        self._latest: tuple[object, int] | None = None
        self._seq = 0
        self._closed = False
        #: 首帧到达时刻，供 wait_first_frame 判定
        self._first_frame = threading.Event()

        # draw_border=False 是**必须**的：WGC 默认给捕获内容加一圈黄色边框，
        # 不关的话每帧都带黄边，直接毁掉模板匹配和亮度统计。
        # cursor_capture 同理 —— 鼠标不该混进感知画面。
        self._cap = WindowsCapture(
            window_hwnd=hwnd, cursor_capture=False, draw_border=False)
        self._cap.event(self.on_frame_arrived)
        self._cap.event(self.on_closed)
        try:
            self._ctl = self._cap.start_free_threaded()
        except Exception:
            self._ctl = None
            self.close()
            raise

    # -- 回调（名字不能改，见模块头铁律 2） ------------------------------

    def on_frame_arrived(self, frame, _ctrl) -> None:  # noqa: ANN001
        with self._lock:
            self._seq += 1
            self._latest = (frame, self._seq)
        self._first_frame.set()

    def on_closed(self) -> None:  # noqa: ANN001, ANN202
        logger.warning("wgc 会话被系统关闭（hwnd=%s）", self._hwnd)
        self._closed = True
        self._first_frame.set()

    # -- 对外 ------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    def wait_first_frame(self, timeout_s: float) -> np.ndarray | None:
        """等第一帧到。返回它的 BGR 拷贝，超时/已关闭返回 ``None``。"""
        if not self._first_frame.wait(timeout_s):
            return None
        return self.grab()

    def grab(self) -> np.ndarray | None:
        """取最新帧的 **BGR 拷贝**（HxWx3）。一帧都没来返回 ``None``。

        ★ 为什么这里必须拷贝
          ------------------
          ``frame.frame_buffer`` 是原生 mapped frame 上的零拷贝视图。
          下一帧到达时 ``self._latest`` 被替换，旧 ``Frame`` 随即被释放，
          此前发出去的视图就指向已释放内存 —— 下游拿去算 dHash / 匹配模板
          会读到垃圾，甚至直接崩。拷贝一次 2560x1600x3 约 12MB，
          5Hz 下就是 60MB/s 的 memcpy，换一个不会随机崩的通道，很划算。

        ★ 静止画面怎么办
          --------------
          WGC **只在画面变化时推帧**。游戏站着不动时可能几秒没有新帧，
          这时返回**上一帧**是**正确**语义而不是偷懒：屏幕没变，
          上一帧就是当前内容。调用方那套「连续 N 次空才算坏」的
          ``_empty_streak`` 因此永远不会误触发。
        """
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        frame = latest[0]
        try:
            # ⚠️ 切片是**视图**；np.array(..., copy=True) 才是真拷贝。
            return np.array(frame.frame_buffer[:, :, :3], dtype=np.uint8,
                            copy=True)
        except (AttributeError, ValueError) as exc:
            logger.debug("wgc 取帧失败：%s", exc)
            return None

    def close(self) -> None:
        """停采集线程。可重复调用；**任何异常都不外泄**。

        采集线程一旦死掉，``stop()`` 会抛
        ``Failed to stop the capture thread: <原异常>`` —— 那条异常会盖住
        真正的病因，所以这里必须吞掉。
        """
        ctl = getattr(self, "_ctl", None)
        self._ctl = None
        self._closed = True
        if ctl is None:
            return
        for fn in (ctl.stop, ctl.wait):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001, PERF203
                logger.debug("wgc %s() 失败（忽略）：%s", fn.__name__, exc)


def _frame_luma(img: np.ndarray | None) -> float:
    """一帧的平均亮度（0-255）。空帧记 0。

    用整帧 ``mean()`` 而不是逐像素转灰度：这里要回答的是「有没有内容」，
    不是「亮不亮」。真·全黑图的 mean **精确为 0**，而原神夜晚那种暗场景
    平均亮度在十几到几十，阈值 8 足以把两者分开；转灰度还要多拉一个 cv2
    进来，而本模块刻意只依赖 numpy + ctypes。

    ★ 与 builder._luma 的区别
      ----------------------
      ``builder._luma`` 是感知层判「这帧能不能用」的口径（cv2 缩到 32x32
      再转灰度，便宜且稳）；这里是截图层判「这条通道在当前窗口上能不能用」
      的口径，要的是**绝对**亮度而不是对缩放不敏感的采样。两者都用
      ``BLACK_FRAME_LUMA = 8.0`` 作阈值，但算法不同、职责不同，不合并。
    """
    if img is None:
        return 0.0
    arr = np.asarray(img)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def _find_window(
    title_substr: str, allow_fullscreen_fallback: bool = False
) -> tuple[int | None, tuple[int, int, int, int] | None]:
    """找窗口并返回 (hwnd, 客户区屏幕矩形)。

    优先用 pywin32（已装），没有则退化到 ctypes。找不到返回 (None, None)。
    """
    try:
        import win32gui  # noqa: PLC0415
        import win32process  # noqa: PLC0415

        found: list[tuple[int, tuple[int, int, int, int]]] = []

        def _cb(hwnd: int, _extra) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd) or ""
            if title_substr not in title:
                return True
            # 客户区 -> 屏幕坐标
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            sx, sy = win32gui.ClientToScreen(hwnd, (left, top))
            found.append((hwnd, (int(sx), int(sy), int(right - left), int(bottom - top))))
            return True

        win32gui.EnumWindows(_cb, None)
        if not found and allow_fullscreen_fallback:
            # 全屏独占时客户区可能为 0，退化到窗口矩形
            found = _find_fullscreen(title_substr)
        if not found:
            return None, None
        # 取面积最大的那个（避免命中工具提示窗之类的小窗）
        hwnd, rect = max(found, key=lambda t: t[1][2] * t[1][3])
        _ = win32process  # 保留导入，供后续按进程过滤
        return hwnd, rect
    except ImportError:
        return _find_window_ctypes(title_substr, allow_fullscreen_fallback)


def _find_fullscreen(title_substr: str) -> list[tuple[int, tuple[int, int, int, int]]]:
    out: list[tuple[int, tuple[int, int, int, int]]] = []
    try:
        import win32api  # noqa: PLC0415
        import win32gui  # noqa: PLC0415

        w, h = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)

        def _cb(hwnd: int, _extra) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if title_substr in (win32gui.GetWindowText(hwnd) or ""):
                out.append((hwnd, (0, 0, w, h)))
            return True

        win32gui.EnumWindows(_cb, None)
    except Exception:  # noqa: BLE001
        pass
    return out


def _find_window_ctypes(
    title_substr: str, allow_fullscreen_fallback: bool
) -> tuple[int | None, tuple[int, int, int, int] | None]:
    """无 pywin32 时的 ctypes 回退（用 FindWindowW，只支持精确/前缀匹配）。"""
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title_substr)
    if not hwnd:
        return None, None
    rect = _client_rect_on_screen(hwnd)
    return int(hwnd), rect


def _client_rect_on_screen(hwnd: int) -> tuple[int, int, int, int]:
    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    user32 = ctypes.windll.user32
    rc = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rc))
    pt = POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return (int(pt.x), int(pt.y), int(rc.right - rc.left), int(rc.bottom - rc.top))


# --------------------------------------------------------------------------
# BitBlt（GDI）—— 2026-10-10 顶替 PrintWindow 入链
# --------------------------------------------------------------------------
# 参照实现：BetterGI `Fischless.GameCapture/BitBlt/BitBltSession.cs`
#   · GetDC(hwnd) + CreateCompatibleDC + CreateDIBSection(24bpp, top-down)
#   · 先查 RASTERCAPS 的 RC_BITBLT，不支持就别浪费后面的往返
#   · GetClientRect 拿不到尺寸（最小化 = 0x0）即判定会话不可用
#   · 会话失效后**重建一次再抓**，而不是每帧都重建
#
# 与 BetterGI 的两处有意差异：
#   · 它有 Bitmap 内存池（ConcurrentStack）；我们是 5Hz 单线程一帧一扔，
#     一次分配的量不值得为它多引一个并发容器。
#   · 它用 ReaderWriterLockSlim 包住整段；本模块外层已有 ``_capture_lock``
#     （RLock），这里再加一把只会多一处能自锁的地方。


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER),
                ("bmiColors", ctypes.c_uint32 * 3)]


#: GetDeviceCaps 的索引
_RASTERCAPS = 2
_BITSPIXEL = 12
#: RC_BITBLT
_RC_BITBLT = 0x00000001
#: SRCCOPY
_SRCCOPY = 0x00CC0020


class _BitBltSession:
    """一个窗口上的 GDI BitBlt 会话。

    构造时把 DC / 位图备好，之后每帧只做一次 ``BitBlt`` + 一次内存拷贝。
    失效时 :attr:`invalid` 转真，由调用方重建 —— 对应 BetterGI
    ``BitBltSession.Invalid`` 的用法。

    ⚠️ **最小化的窗口抓不到**。GDI 拿的是窗口的设备上下文，而最小化
       后 ``GetClientRect`` 返回 ``0x0``，构造阶段就会抛 CaptureError。
       这不是 bug，是这条路的能力边界（BetterGI 同样如此，
       ``BitBltCapture.CheckSession`` 里也是「拿不到 clientRect 就释放会话」）。
    """

    def __init__(self, hwnd: int) -> None:
        if not hwnd:
            raise CaptureError("bitblt 需要有效的 hwnd")
        self._user32 = ctypes.windll.user32
        self._gdi32 = ctypes.windll.gdi32
        self._hwnd = hwnd
        self._hdc = None
        self._mem_dc = None
        self._bmp = None
        self._old = None
        self._bits = None
        self._stride = 0
        self._w = self._h = 0
        self._open()

    # -- 内部 ------------------------------------------------------------
    def _open(self) -> None:
        u, g = self._user32, self._gdi32

        rc = _RECT()
        if not u.GetClientRect(self._hwnd, ctypes.byref(rc)):
            raise CaptureError(f"bitblt: GetClientRect({self._hwnd}) 失败")
        w, h = int(rc.right - rc.left), int(rc.bottom - rc.top)
        if w <= 0 or h <= 0:
            raise CaptureError(
                f"bitblt: 客户区 {w}x{h} —— 窗口多半已最小化，"
                "GDI 路径拿不到最小化窗口的画面")

        self._hdc = u.GetDC(self._hwnd)
        if not self._hdc:
            raise CaptureError("bitblt: GetDC 失败")
        try:
            caps = int(g.GetDeviceCaps(self._hdc, _RASTERCAPS))
            if not (caps & _RC_BITBLT):
                raise CaptureError("bitblt: 设备不支持 RC_BITBLT")
            bpp = int(g.GetDeviceCaps(self._hdc, _BITSPIXEL))
            if bpp not in (24, 32):
                raise CaptureError(
                    f"bitblt: 只支持 24/32 位色深，当前 {bpp}")

            self._mem_dc = g.CreateCompatibleDC(self._hdc)
            if not self._mem_dc:
                raise CaptureError("bitblt: CreateCompatibleDC 失败")

            bi = _BITMAPINFO()
            hdr = bi.bmiHeader
            hdr.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            hdr.biWidth = w
            hdr.biHeight = -h        # 负 = top-down，省一次上下翻转
            hdr.biPlanes = 1
            hdr.biBitCount = 24      # 直接 BGR，省掉 BGRA->BGR 切片
            hdr.biCompression = 0    # BI_RGB
            hdr.biSizeImage = 0

            self._bits = ctypes.c_void_p()
            self._bmp = g.CreateDIBSection(
                self._mem_dc, ctypes.byref(bi), 0,
                ctypes.byref(self._bits), None)
            if not self._bmp or not self._bits or not self._bits.value:
                raise CaptureError("bitblt: CreateDIBSection 失败")

            self._old = g.SelectObject(self._mem_dc, self._bmp)
            if not self._old:
                raise CaptureError("bitblt: SelectObject 失败")

            # DIB 每行按 4 字节对齐，24bpp 时行宽未必等于 w*3
            self._stride = ((w * 3 + 3) // 4) * 4
            self._w, self._h = w, h
        except Exception:
            self.close()
            raise

    # -- 对外 ------------------------------------------------------------
    @property
    def invalid(self) -> bool:
        """GDI 对象是否已失效/被释放。调用方据此重建。"""
        return not (self._hdc and self._mem_dc and self._bmp
                    and self._bits and self._bits.value)

    @property
    def size(self) -> tuple[int, int]:
        return (self._w, self._h)

    def grab(self) -> np.ndarray | None:
        """抓一帧 BGR。GDI 任一步失败都返回 ``None``（不算异常）。"""
        if self.invalid:
            return None
        g = self._gdi32
        if not g.BitBlt(self._mem_dc, 0, 0, self._w, self._h,
                        self._hdc, 0, 0, _SRCCOPY):
            return None
        g.GdiFlush()
        raw = ctypes.string_at(self._bits, self._stride * self._h)
        # bytearray 而非 bytes —— 上层拿到的是可写数组，别在下游踩只读坑
        arr = np.frombuffer(bytearray(raw), dtype=np.uint8)
        if self._stride == self._w * 3:
            return arr.reshape(self._h, self._w, 3)
        # 有行对齐填充：先二维切开裁掉尾巴，再摊回 HxWx3
        rows = arr.reshape(self._h, self._stride)[:, :self._w * 3]
        return np.ascontiguousarray(rows.reshape(self._h, self._w, 3))

    def close(self) -> None:
        """释放全部 GDI 对象。可重复调用。"""
        g, u = self._gdi32, self._user32
        try:
            if self._old and self._mem_dc:
                g.SelectObject(self._mem_dc, self._old)
            self._old = None
        except Exception:
            self._old = None
        try:
            if self._bmp:
                g.DeleteObject(self._bmp)
            self._bmp = None
        except Exception:
            self._bmp = None
        try:
            if self._mem_dc:
                g.DeleteDC(self._mem_dc)
            self._mem_dc = None
        except Exception:
            self._mem_dc = None
        try:
            if self._hdc and self._hwnd:
                u.ReleaseDC(self._hwnd, self._hdc)
            self._hdc = None
        except Exception:
            self._hdc = None
        self._bits = None

    def __del__(self) -> None:  # pragma: no cover - 解释器退出兜底
        try:
            self.close()
        except Exception:
            pass


def apply_win11_bitblt_fix(apply: bool = False) -> dict:
    """Win11 上 BitBlt 抓到黑屏时的注册表 workaround。

    BetterGI ``BitBltRegistryHelper.cs`` 记的原委（引 better-genshin-impact#92）：
    Win11 给 DirectX 应用默认开了「交换链效果升级」，之后 BitBlt 拿到的
    就是黑屏。要关掉它：

        HKEY_CURRENT_USER\\Software\\Microsoft\\DirectX\\UserGpuPreferences
        DirectXUserGlobalSettings = SwapEffectUpgradeEnable=0;

    ⚠️ BetterGI 原注释明写：**必须在游戏启动前设置才有效**，改完要重启游戏。

    ★ 本函数**默认不写注册表**（``apply=False``），只报告当前状态并按需告警。
      静默改用户注册表是不可接受的副作用 —— 写不写必须由调用方显式决定。

    :return: ``{"current": 当前值或 None, "applied": 是否已写入,
              "path": …, "name": …}``
    """
    import winreg  # lazy: 仅 Windows，且只在显式调用时才需要

    path = r"Software\Microsoft\DirectX\UserGpuPreferences"
    name = "DirectXUserGlobalSettings"
    want = "SwapEffectUpgradeEnable=0;"
    state: dict = {"applied": False, "current": None,
                   "path": path, "name": name, "want": want}

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0,
                            winreg.KEY_READ) as k:
            state["current"] = winreg.QueryValueEx(k, name)[0]
    except FileNotFoundError:
        state["current"] = None
    except OSError as exc:
        state["current"] = None
        state["error"] = str(exc)

    if not apply:
        if state["current"] != want:
            logger.warning(
                "BitBlt 可能抓到黑屏：HKCU\\%s\\%s = %r（期望 %r）。"
                "修复须在**游戏启动前**执行 "
                "apply_win11_bitblt_fix(apply=True)，之后重启游戏。",
                path, name, state["current"], want)
        return state

    try:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, path, 0,
                                winreg.KEY_WRITE) as k:
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, want)
        state["applied"] = True
        state["current"] = want
        logger.warning("已写入 HKCU\\%s\\%s = %r —— 请重启游戏后生效",
                       path, name, want)
    except OSError as exc:
        state["error"] = str(exc)
        logger.error("写注册表失败：%s", exc)
    return state


__all__ = [
    "CaptureChannel",
    "CaptureError",
    "CapturedFrame",
    "Capturer",
    "WindowNotFoundError",
    "apply_win11_bitblt_fix",
]
