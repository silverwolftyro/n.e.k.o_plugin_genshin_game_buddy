"""overlay 主循环（tkinter）。

★ 省电是这个文件的核心约束
--------------------------
10Hz 轮询，但绝大多数时候**什么都不做**：

* 读快照前先 ``stat`` 大小，没变就不解析 pydantic（省 CPU 更明显）
* 渲染前比对 :meth:`PanelModel.content_key` 指纹，**没变就完全不碰 tk**
* 无动画、无 GIF、无渐变、无阴影 —— 每帧重画都是纯浪费
* 空转时的开销约等于「每 100ms 一次 stat」

早期版本每 100ms 无条件重绘全部文本，Task Manager 上稳定 4~6% CPU。
加上指纹比对后掉到 < 1%。

★ 崩溃不许影响 run_loop
----------------------
overlay 是**独立进程**，它崩了 run_loop 不受影响；run_loop 崩了 overlay
显示「离线」。两者互不 import 对方的运行时状态，只读文件。
"""

from __future__ import annotations

import json
import logging
import os
import time
import tkinter as tk
import uuid
from pathlib import Path

from . import console_launcher, layout, theme
from .interaction import Interaction

logger = logging.getLogger(__name__)

#: 实时缩放通道的轮询间隔（毫秒）。12~13 次/秒，肉眼看是连续的。
#:
#: ★ 为什么不是 20Hz
#:   每轮是一次 ``stat()``（Windows 上不便宜）。20Hz 只会把磁盘问得更勤，
#:   视觉上没有区别 —— 上限由"重排一次要销毁+重建 7 个控件"决定，
#:   真正的瓶颈在这里，不在轮询频率。
LIVE_SCALE_POLL_MS = 80

#: 临时通道文件名。与主控台 ``channels/overlay.py`` 的
#: ``LIVE_SCALE_FILE_NAME`` 必须一致。
LIVE_SCALE_FILE = "overlay_scale_live.json"

#: 新旧缩放差小于这个值就不重排。
#: 滑块一格是 5%，而浮点比较永远差一点，所以 0.005 足够滤掉
#: "值其实没变只是重新读了一遍文件"的抖动，同时不会漏掉任何真实变化。
LIVE_SCALE_EPS = 0.005


#: 主控台命令目录的**环境变量名**。
#:
#: overlay 由主控台 spawn 时会把主控台的 runtime 目录通过这个变量传进来，
#: 这样它不必知道主控台装在哪、也不必猜路径。
CONSOLE_CMD_DIR_ENV = "NEKO_CONSOLE_CMD_DIR"

#: 环境变量没给时的兜底路径。只在 overlay 被单独跑起来（验收场景 3）时用得上。
CONSOLE_CMD_DIR_FALLBACK = r"D:\NEKO\NEKO-Console\runtime"

#: 主控台命令文件。与主控台 bridge.py::COMMANDS_FILE_NAME 必须一致。
CONSOLE_COMMANDS_FILE = "console_commands.jsonl"


def console_cmd_dir() -> Path | None:
    """主控台的命令目录；目录不存在返回 None。

    ★ 为什么优先读环境变量
      overlay 可能跑在两种方式下：主控台 spawn 它（知道路径，传环境变量），
      或用户手动 ``python -m autopilot.tools.overlay``（什么都不知道）。
      前者准确，后者只能靠约定 —— 所以环境变量是主路径，硬编码是兜底。
    """
    env = os.environ.get(CONSOLE_CMD_DIR_ENV, "").strip()
    cand = Path(env) if env else Path(CONSOLE_CMD_DIR_FALLBACK)
    try:
        return cand if cand.is_dir() else None
    except OSError:
        return None


class OverlayApp:
    """独立 overlay 窗口。

    ★ 排版参数是**类属性**，不是实例属性
      -------------------------------------
      :class:`~autopilot.ui.overlay.dev_panel.DevOverlayApp` 要换行数与
      宽度。早期版本把它们放在 ``__init__`` 里赋值，结果踩了两个坑：
      ① 父类构造里第一件事就是 ``_compute_metrics()``，子类必须抢在
      ``super().__init__()`` **之前**赋值，顺序一错高度就按 5 行排；
      ② 父类 ``__init__`` 后半段又会 ``self.text_rows = 5`` 把子类的
      值**覆盖掉**。类属性由 ``self.`` 读取，子类直接声明覆盖，
      两个坑一起消失。
    """

    #: 正文行数。**行数恒定 => 面板高度不跳**，这是产品版刻意守住的。
    text_rows: int = 5
    #: 面板宽度的**设计基准像素**（未乘缩放）。实际像素见 :attr:`panel_width`。
    #: 写成基准值而不是直接存像素，是为了让 :func:`theme.apply_user_scale`
    #: 重算 ``SCALE`` 之后**自动生效** —— 存像素的话那份值会停在导入时刻。
    panel_width_units: float = 480

    def __init__(self, reader, *, sender=None, x: int | None = None,
                 y: int | None = None, alpha: float | None = None,
                 scale: float | None = None,
                 reset_pos: bool = False) -> None:
        from autopilot.ui.commands import CommandSender  # noqa: PLC0415
        from autopilot.ui.reader import ChannelReader  # noqa: PLC0415

        # 直接用协议层给的两个 class，不另造中间层
        self.reader: ChannelReader = (
            reader if isinstance(reader, ChannelReader) else ChannelReader(reader))
        self.sender: CommandSender = sender or CommandSender()
        # 只关心"从现在起"的事件，不刷启动前的历史
        self.reader.attach_stream_tail(-1)

        # ★ 用户缩放必须在 ``Tk()`` **之前**生效。
        #   apply_user_scale() 改的是字号；行高由 _compute_metrics() 在 Tk
        #   建好之后**实测**字体得到。顺序对了，尺寸常量、控件、按钮区
        #   全自动跟着走，一处都不用单独改。
        #   放到 Tk 之后会先按旧字号排一帧、再跳一下，用户看得见闪。
        #
        # ★ 命令行没给 --scale 时，回读上次 Ctrl+滚轮 存下的倍率。
        #   优先级：命令行 > 持久化 > 1.0。主控台每次启动都显式传
        #   --scale（schema.py: 不为 1.0 时才追加），所以持久化**不会**
        #   跟主控台打架；只有手动 ``python -m autopilot.tools.overlay``
        #   才会走这条回读。
        if scale is None:
            saved_pos = layout.load_pos()
            if saved_pos is not None and saved_pos.scale is not None:
                scale = saved_pos.scale
                logger.info("回读已保存的用户缩放倍率 %.3f", scale)
        if scale is not None:
            logger.info("应用用户缩放倍率 %.3f", scale)
            theme.apply_user_scale(scale)

        self.root = tk.Tk()
        self.root.title("N.E.K.O.")
        self.root.overrideredirect(True)          # 无标题栏
        self.root.attributes("-topmost", True)
        self.root.configure(bg=theme.BG)

        # ★ 先量字体再定尺寸 —— geometry() 用的是实测面板高度，
        #   顺序反了的话第一帧仍按旧的 px(20) 行距排版。
        self._compute_metrics()

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        pos = layout.resolve_pos(sw, sh, x=x, y=y, reset=reset_pos)
        if alpha is not None:
            pos.alpha = max(0.15, min(1.0, float(alpha)))
        self.root.geometry(
            f"{self.panel_width}x{self._panel_h}+{pos.x}+{pos.y}")
        self.root.attributes("-alpha", pos.alpha)

        # 事件提示（只在有事件时出现，3 秒后消失）
        self._toast: str | None = None
        self._toast_color = theme.DIM
        self._toast_until = 0.0
        self._panic_until = 0.0
        self._last_key: tuple | None = None
        self._last_channel = ""       # run_loop 停止后通道会空，记住最后一个

        self._build_widgets()
        self.interaction = Interaction(
            self.root,
            on_move=self._move,
            on_close=self.root.destroy,
            on_kill=self._confirm_kill,
            on_restart=self._restart_engine,
            on_scale=self._scale_by,
            on_alpha=self._alpha_by,
            get_pos=self._get_pos,
            on_persist=self._persist_pos,
            on_open_console=self._open_console,
            on_stop_task=self._stop_task,
            on_stop_all=self._stop_all,
        )
        self.interaction.layout_buttons(self.panel_width, self._panel_h)

        # 实时缩放通道：主控台拖滑块时写临时文件，这里轮询跟着变。
        # 必须放在最后 —— 它依赖 widgets / interaction 都已就位。
        self._setup_live_scale_channel()
        self.root.after(LIVE_SCALE_POLL_MS, self._poll_live_scale)

        # Alt+T 轮换颜色主题（2026-10-10）。只换**颜色**不换字号/尺寸 ——
        # 那两条轴归 --scale 管，混在一起会出现"两套缩放"的歧义。
        # 绑在 toplevel 而不是某个 Label 上：键盘焦点在面板任意位置都收得到。
        self.root.bind("<Alt-t>", self._cycle_theme)

    # ------------------------------------------------------------------
    # 颜色主题
# ------------------------------------------------------------------
    def _cycle_theme(self, _ev=None) -> str:
        """Alt+T：在 dark / light / high-contrast 之间轮换。"""
        name = theme.apply_theme(theme.next_theme(theme.CURRENT_THEME))
        self._toast = f"主题：{name}"
        self._toast_color = theme.ACCENT_IDLE
        self._toast_until = time.monotonic() + theme.EVENT_TOAST_MS
        self._last_key = None             # 强制下一帧重画全部控件颜色
        # 立刻重画一次：apply_theme 改了模块全局，但已创建的 Label 不会
        # 自己更新前景/背景色 —— 不重画的话用户得等下次内容变化才看到。
        self._force_theme_repaint()
        return "break"

    def _force_theme_repaint(self) -> None:
        """把新主题刷到已存在的控件上。**背景色必须逐控件改**。

        Label 是在构造时用当时的 theme.BG 建的，主题换了它不会自己变。
        ``root.configure(bg=...)`` 只改根窗口 —— 子控件不透明，会盖住。
        """
        bg = theme.BORDER if self._panic_until > time.monotonic() else theme.BG
        self.root.configure(bg=bg)
        for lbl in list(self._lines) + [self._toast_lbl]:
            if lbl is not None:
                lbl.configure(bg=bg)
        # 按钮也是同样问题：bg/fg 是创建时烤进去的。
        for lbl in getattr(self.interaction, "_widgets", {}).values():
            try:
                lbl.configure(bg=theme.BTN_BG, fg=theme.BTN_FG)
            except Exception:  # noqa: BLE001 - 换肤不该因为某个按钮异常而失败
                pass
        self._last_key = None

    # ------------------------------------------------------------------
    # 尺寸
    # ------------------------------------------------------------------
    @property
    def panel_width(self) -> int:
        """面板物理宽度。**每次都按当前 SCALE 换算** ——
        ``theme.apply_user_scale()`` 改了倍率之后这里自动跟上，
        而存死像素的话会停在导入时刻（这正是开发版第一版的 bug）。"""
        return theme.px(self.panel_width_units)

    def _compute_metrics(self) -> None:
        """**实测**字体行高，用它排版 —— 不再手算。

        ★ 为什么必须实测
        --------------
        ``theme.LINE_HEIGHT`` 是按 10pt 字体标定的（20px），而 10pt 在
        96 DPI 下的真实行高约 24px —— **这个失配一直存在**，只是以前
        Windows 替整个进程做 1.5 倍虚拟放大，字体和行距被同样放大、
        失配比例不变，所以看不出来。

        上一轮在入口声明了 DPI 感知之后，Windows 不再虚拟放大，代码里
        ``FONT_SIZE`` 又自己乘了 SCALE，于是 15pt 字体需要约 36px 行高，
        而行距仍是 px(20)=30px —— **行距 < 字高**，于是：
            · 相邻两行重叠
            · 最后一行 + 提示行溢出面板底边，被 Tk 裁掉
            · 表现为「意」的下半「心」消失 -> 看成「音图」

        靠猜（CJK 行高 ≈ 字号 × 1.2）会随字体、缩放、系统设置漂移。
        直接问 Tk 要 ``linespace`` 才是准的。
        """
        import tkinter.font as tkfont  # noqa: PLC0415

        font = tkfont.Font(root=self.root,
                           family=theme.FONT_FAMILY,
                           size=theme.FONT_SIZE)
        try:
            line_h = int(font.metrics("linespace"))
        finally:
            # ★ 必须显式 delete。tkfont.Font 会在 Tk 里注册一个具名字体，
            #   不删就一直留在 font database 里。以前只在启动时调一次，
            #   泄漏无所谓；现在拖动时每秒要调 12 次，攒下来就是几百个
            #   僵尸字体，刷新字体列表会越来越慢。
            try:
                font.delete()
            except Exception:  # noqa: BLE001 - 删不掉也不能挡住排版
                pass
        # 至少要比字号本身高（微软雅黑 CJK 行距宽松，但极端情况下
        # metrics 也可能返回偏小值）
        self._line_h = max(line_h, theme.FONT_SIZE + 2)
        self._panel_h = (theme.PANEL_PADDING * 2
                         + self._line_h * (self.text_rows + 1)
                         + theme.px(4))          # 提示行下方余量
        # 文字左边界。
        #
        # ★ 说实话：这一行**并没有**真的让开整个按钮区
        #   ------------------------------------------
        #   注释以前写的是「保证第一行绝不跑到按钮底下」，那是**错的**：
        #   加了第 4 个按钮（▶）后，最左按钮 x = PANEL_WIDTH-146（100% 屏实测 334），
        #   而下面算出来的 _text_left 只有 96 —— 文字区（96 起）与按钮行
        #   （334 起）在横向上**本来就重叠**，3 按钮时代就是这样（370 起），
        #   不是本轮引入的。
        #
        #   为什么一直没人看见：所有 5 行文本都**左对齐且够短**，短到
        #   根本够不到 334/370 那一列。靠的是「文本长度」而不是「留白」。
        #
        #   ★ 所以**不要**把这里改成 button_row_left()-px(4)：那会把
        #   可用文字宽度从 374px 压到 140px，5 行全部截断 —— 是把一个
        #   「目前不显形的重叠」换成一个「立刻显形的截断」。
        #   真要根治得连面板一起重排（按钮移出右上角，或文字换行），
        #   那是单独一轮的事。本轮只把注释改回**和实现一致**。
        self._text_left = (theme.PANEL_PADDING
                           + 2 * theme.BUTTON_HIT
                           + theme.BTN_MARGIN * 3
                           + theme.px(4))
        logger.debug(
            "浮窗排版：系统缩放=%.2f 用户倍率=%.2f 有效=%.2f "
            "font=%spt 行高=%d 面板=%dx%d 文字左边界=%d",
            theme.BASE_SCALE, theme.user_scale(), theme.SCALE,
            theme.FONT_SIZE, self._line_h,
            theme.PANEL_WIDTH, self._panel_h, self._text_left)

    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        self._lines: list[tk.Label] = []
        pad = theme.PANEL_PADDING
        lh, left = self._line_h, self._text_left
        row_w = self.panel_width - left - pad
        for i in range(self.text_rows):
            lbl = tk.Label(self.root, text="", bg=theme.BG, fg=theme.FG,
                           anchor="w", justify="left",
                           font=(theme.FONT_FAMILY, theme.FONT_SIZE))
            # ★ 显式给 height = 实测行高：行与行对齐，
            #   且**不会**小于字高（小于就会被裁）。
            lbl.place(x=left, y=pad + i * lh, width=row_w, height=lh)
            self._lines.append(lbl)
        self._toast_lbl = tk.Label(self.root, text="", bg=theme.BG,
                                   fg=theme.DIM, anchor="w", justify="left",
                                   font=(theme.FONT_FAMILY, theme.FONT_SIZE))
        self._toast_lbl.place(x=left, y=pad + self.text_rows * lh,
                              width=row_w, height=lh)

    # ------------------------------------------------------------------
    def _get_pos(self) -> tuple[int, int]:
        return self.root.winfo_x(), self.root.winfo_y()

    def _persist_pos(self, x: int | None = None,
                     y: int | None = None) -> None:
        """把**完整**的窗口状态写回 ``overlay_pos.json``。

        ★ 为什么必须由这里统一写
          ------------------------
          位置、透明度、缩放三样都只有这个类拿得到真值。以前
          ``Interaction`` 在拖动结束时自己 ``OverlayPos(x, y)``，
          alpha 落回默认 0.72 —— **拖一下窗口就把 ``--alpha`` 吃掉了**，
          而且没有任何报错。现在保存点只报「坐标变了」，具体写什么由
          这里决定，字段就再也丢不了。

        :param x: 新坐标；``None`` = 用当前窗口位置。
        """
        if x is None or y is None:
            cur_x, cur_y = self._get_pos()
            x = cur_x if x is None else x
            y = cur_y if y is None else y
        # alpha 直接问 Tk 要，别缓存 —— alpha 是 Tk 的属性，不是我们的状态。
        try:
            alpha = float(self.root.attributes("-alpha"))
        except (tk.TclError, ValueError):
            alpha = 0.72
        layout.save_pos(layout.OverlayPos(x=int(x), y=int(y),
                                          alpha=alpha,
                                          scale=theme.user_scale()))

    # ------------------------------------------------------------------
    # 实时缩放通道（拖动中跟随）
    # ------------------------------------------------------------------
    @staticmethod
    def live_scale_reader(base_dir: object | Path | None = None) -> Path:
        """临时缩放文件的路径。

        ★ 刻意做成**不依赖实例**的静态方法
          --------------------------------
          两个进程必须对"文件在哪"达成一致，而这条路径是唯一的共识点。
          把它做成纯函数，命令行就能直接验证（``python -c "print(...)"``），
          不用先起一个 Tk 窗口 —— 路径算错是这类跨进程协议最常见的故障。

          ★★ ``@staticmethod`` 不能漏
            漏了它就是个普通方法，``self.live_scale_reader()`` 会把
            **self 当成第一个位置参数 base_dir** 传进去，于是
            ``Path(OverlayApp实例)`` 抛
            ``TypeError: expected str, bytes or os.PathLike object`` ——
            overlay 一起手就崩。调用点不用改，加装饰器就够。

        ★ 路径来源
          用 ``layout.POS_FILE.parent``，而不是自己拼 ``_runtime``：
          这样"位置文件在哪，缩放文件就在哪"永远成立，将来改目录只改一处。
          ``base_dir`` 只给测试/临时目录用，生产路径恒走 ``None`` 分支。
        """
        if base_dir is not None:
            return Path(base_dir) / LIVE_SCALE_FILE
        return layout.POS_FILE.parent / LIVE_SCALE_FILE

    def _setup_live_scale_channel(self) -> None:
        """定位临时文件，并**把首帧的 stamp 记下来**。

        ★ 为什么要先记 stamp
          ------------------
          主控台崩溃或被强杀时可能来不及删临时文件。若启动时不记 stamp，
          第一次轮询就会看到"文件变了"（其实没变）并套用一个中间值 ——
          用户看到的是"启动时浮窗大小莫名跳了一下，然后也不对"。
          先记 stamp 就等于宣告"现状即基准，之后只认变化"。
        """
        self._live_scale_path = self.live_scale_reader()
        self._last_live_stamp = None
        try:
            st = self._live_scale_path.stat()
        except OSError:
            return                       # 正常情况：没人拖过
        self._last_live_stamp = (st.st_size, st.st_mtime_ns)
        logger.info("已存在实时缩放文件 %s，认作基准不套用（避免启动闪跳）",
                    self._live_scale_path)

    # ------------------------------------------------------------------
    # Ctrl+滚轮缩放
    # ------------------------------------------------------------------
    def _scale_by(self, multiplier: float) -> None:
        """Ctrl+滚轮：调用户缩放。**入参是相对量，不是绝对值。**

        ★ 为什么收相对量
          --------------
          调用方（滚轮）只知道「往上一格 / 往下一格」，不知道当前倍率是多少
          —— 让它去读 theme 再算加法，就把「当前值」这件事复制到了两个地方。
          传 +0.05，边界夹取、落库、报错全在这一处。

        ★ 夹在边界上也要给反馈
          --------------------
          用户滚到顶还继续滚，倍率纹丝不动。**什么都不说是最糟的** ——
          浮窗在任何地方都不显示倍率，用户会以为滚轮坏了。这里报一句
          「已到最大 / 已到最小」，让「没反应」变成「到头了」。

        ★ 缩放后必须落库，否则滚轮白滚
          ------------------------------
          主控台的滑块是**临时**通道（``overlay_scale_live.json``
          拖完就删），Ctrl+滚轮是用户对着浮窗的直接操作，预期是
          「下次启动还在」。所以走 :meth:`_persist_pos` 一起存。
        """
        before = theme.user_scale()
        theme.apply_user_scale(before + multiplier)
        after = theme.user_scale()      # ★ 夹取后的真实值，不是 apply 的返回值
        # apply_user_scale 返回的是「有效缩放 = BASE × 倍率」，不是倍率本身。

        if abs(after - before) < 1e-9:
            # 已经贴着边界了：没变，就别重排（重排是纯浪费，
            # 而且会让面板闪一下，等于用抖动告诉用户「没反应」）。
            if before <= theme.USER_SCALE_MIN + 1e-9:
                self._toast = "已到最小缩放"
            else:
                self._toast = "已到最大缩放"
            self._toast_color = theme.DIM
            self._toast_until = time.monotonic() + theme.EVENT_TOAST_MS
            self._last_key = None         # 强制下一帧重画 toast
            return

        logger.info("Ctrl+滚轮缩放 %.2f -> %.2f", before, after)
        self._reflow()                    # 重建控件 + 改 geometry，位置保持
        # ★ 存的是 _reflow 之后的位置：窗口尺寸变了，winfo 才是最终值。
        self._persist_pos()
        self._toast = f"缩放 {before:.2f} → {after:.2f}"
        self._toast_color = theme.ACCENT_RUN
        self._toast_until = time.monotonic() + theme.EVENT_TOAST_MS
        self._last_key = None             # 强制下一帧重画 toast

    # ------------------------------------------------------------------
    # 透明度（Alt+滚轮）
    # ------------------------------------------------------------------
    #: 透明度边界。下限 0.15 是有意卡的：再低文字就糊到读不出来。
    #: 想"看清游戏"时用户自己会用 --alpha 调，不需要 0.05。
    ALPHA_MIN = 0.15
    ALPHA_MAX = 1.0

    def _alpha_by(self, multiplier: float) -> None:
        """Alt+滚轮：调透明度。**入参是相对量**（与 :meth:`_scale_by` 同约定）。

        ★ 为什么调完要 toast
          ------------------
          透明度是「越滚越不像」的量：滚了三格才发现刚才那两格没生效，
          用户只能怀疑「Alt 好像没反应」。报一句当前值，把静默变成确定。

        ★ 为什么要 ``_persist_pos()``
          --------------------------
          透明度以前只有 ``--alpha`` 一个入口，改完关窗就没了。现在它跟着
          ``overlay_pos.json`` 走，下次启动自动恢复 —— ``_persist_pos``
          是**从 Tk 问真值**而不是读缓存，所以写回去的一定是刚改成的那���。

        ★ 到边界时也要写盘
          --------------
          值没变，但「用户第一次把透明度顶到最透」这件事本身值得记住，
          而 ``save_pos`` 只在值变化时才有意义 —— 所以这里无条件调用。
        """
        try:
            before = float(self.root.attributes("-alpha"))
        except (tk.TclError, ValueError):
            before = 0.72
        after = max(self.ALPHA_MIN, min(self.ALPHA_MAX, before + multiplier))
        self.root.attributes("-alpha", after)

        if abs(after - before) < 1e-6:
            edge = "最透" if after <= self.ALPHA_MIN + 1e-9 else "最实"
            self._toast = f"透明度已到{edge}（{after:.2f}）"
            self._toast_color = theme.DIM
        else:
            self._toast = f"透明度 {before:.2f} → {after:.2f}"
            self._toast_color = theme.ACCENT_IDLE
        self._toast_until = time.monotonic() + theme.EVENT_TOAST_MS
        self._last_key = None             # 强制下一帧重画 toast

        self._persist_pos()

    # ------------------------------------------------------------------
    # 实时缩放通道（拖动中跟随）
    # ------------------------------------------------------------------
    def _poll_live_scale(self) -> None:
        """轮询临时缩放文件。

        ★ 下一轮先排上，再处理本轮
          ----------------------------
          顺序反了的话，下面任何一次异常（读文件、JSON 解析、_reflow）
          都会把整条 after 链掐断，实时缩放**永久失效**，而且没有任何报错。
          先排 next 就没有这个风险。
        """
        self.root.after(LIVE_SCALE_POLL_MS, self._poll_live_scale)
        path = self._live_scale_path
        if path is None:
            return
        try:
            st = path.stat()
        except OSError:
            # 文件不存在 = 用户没在拖 / 已经松手删除。什么都不做，保持现状。
            return
        cur = (st.st_size, st.st_mtime_ns)
        if cur == self._last_live_stamp:
            return                       # 没变，连文件都不读
        self._last_live_stamp = cur
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            new_scale = float(data["scale"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # 半截 JSON（主控台正在 replace 的那一瞬间）等下一轮即可
            logger.debug("实时缩放文件暂时读不了，等下一轮：%s", exc)
            return
        if abs(new_scale - theme.user_scale()) < LIVE_SCALE_EPS:
            return                       # 差得可忽略，不值得重排
        logger.debug("套用实时缩放 %.3f -> %.3f",
                     theme.user_scale(), new_scale)
        theme.apply_user_scale(new_scale)
        self._reflow()

    def _reflow(self) -> None:
        """缩放变了之后重建全部子控件与窗口尺寸。

        ★ 为什么销毁重建，而不是原地改
          --------------------------------
          字号、行高、面板宽高、按钮触达区**全部**由 theme 常量推导。
          原地改就得手工同步：6 个 Label 的 font 和 place、2 个按钮的
          坐标与字形、窗口 geometry —— 漏改任何一处就是错位。
          「意」被裁成「音图」就是漏改了行高。

          销毁重建只有一条路径：重算 -> 全建 -> 设 geometry，走不歪。

        ★ 为什么不销毁 root / interaction
          它们绑着事件和回调，换掉等于重写整个 overlay。

        ★ 代价
          每次重排新建 7 个 Label + 2 个按钮 Label。拖动时每秒 12 次，
          量级很小；真正要注意的是别漏删旧的（见下）。
        """
        # ★ 位置必须**先**记下来。改窗口尺寸会让 Windows 重新摆放窗口，
        #   事后再问 winfo_x() 拿到的可能已经是新位置，用户拖好的浮窗就跑了。
        x, y = self._get_pos()

        # 销毁 root 的全部子控件：N 行正文 + 1 行提示 + 2 个按钮。
        # ★ 必须整体删干净 —— interaction._make_button() 每次都新建 Label
        #   塞进自己的字典，但**不删旧的**。只重建不清理的话，
        #   拖 3 秒就会在同一个位置叠 36 个半透明按钮。
        for child in self.root.winfo_children():
            child.destroy()
        self._lines = []
        self._toast_lbl = None

        self._compute_metrics()      # 新字号 -> 新行高 -> 新面板高
        self._build_widgets()        # 重建 N+1 个 Label
        self.interaction.layout_buttons(self.panel_width, self._panel_h)
        self.root.geometry(f"{self.panel_width}x{self._panel_h}+{x}+{y}")

        # 立刻重画一次，不等下一轮 tick —— 否则松手后会看到最多 100ms 的空白
        self._last_key = None
        self.tick()

    def _move(self, x: int, y: int) -> None:
        self.root.geometry(f"+{int(x)}+{int(y)}")

    def _send_console_cmd(self, cmd: str, tag: str) -> tuple[bool, str]:
        """往主控台命令通道追加一行。返回 ``(成功, 失败原因)``。

        ★ 为什么要抽出来
          --------------
          ↻（重启引擎）、单击（轻停）、双击/长按（全停）三条路径都要
          写同一个文件、同一种 JSON 形状。各写一遍的话，改了格式只改一处
          就���命令静默到不了主控台 —— 而主控台那边
          ``CommandPoller._parse`` 对未注册命令**直接拒收且只打 warning**，
          用户看到的是「点了没反应」。这跟 :mod:`interaction` 里按钮几何
          「两处各算各的」是同一类事故。

        ★ 命令名是**对外契约**
          --------------------
          ``neko_console/bridge.py`` 的 ``VALID_COMMANDS`` 只有
          ``START_ENGINE / STOP_ENGINE / RESTART_ENGINE / START_OVERLAY /
          STOP_OVERLAY / QUIT_CONSOLE``。写 ``"STOP"`` 会被静默拒掉 ——
          这里必须写 ``STOP_ENGINE``。

        :param cmd: 命令字，必须在 ``VALID_COMMANDS`` 里。
        :param tag: ``cmd_id`` 前缀，方便在日志里认出是谁发的。
        """
        d = console_cmd_dir()
        if d is None:
            return False, "主控台未启动或路径不可达"
        payload = {
            "schema_version": "1.0",
            # ★ uuid 而不是递增序号：overlay 每次重启都会重置任何模块级
            #   计数器，而主控台侧靠 cmd_id 去重。递增号会和主控台自己写的
            #   命令撞号，导致**新命令被当重复丢掉**。
            "cmd_id": f"{tag}_{uuid.uuid4().hex[:12]}",
            "ts_ms": int(time.time() * 1000),
            "cmd": cmd,
            "args": {},
        }
        try:
            f = d / CONSOLE_COMMANDS_FILE
            # 单次 open+write：主控台正按 offset 读这个文件，
            # 分两次写会让它读到半截 JSON。
            with open(f, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            return False, f"写入失败：{exc}"
        logger.info("已发送 %s 命令到 %s", cmd, d)
        return True, str(d)

    def _restart_engine(self) -> None:
        """↻ 按钮：请主控台重启引擎。

        ★ overlay **不自己启停引擎**
          引擎是主控台的 Popen 句柄，overlay 碰不到它；真去 fork 一个出来
          就是又一份孤儿进程来源 —— 正是这套东西要消灭的问题。
          这里只发一条命令，让**唯一的持有者**去做。

        成功/失败都给 toast：用户点了按钮却毫无反应，会以为按钮坏了。
        """
        ok, detail = self._send_console_cmd("RESTART_ENGINE", "restart")
        if ok:
            self._toast, self._toast_color = (
                "已发送重启引擎命令", theme.ACCENT_RUN)
        else:
            logger.warning("发送 RESTART_ENGINE 失败：%s", detail)
            self._toast, self._toast_color = (
                f"重启失败：{detail}", theme.ACCENT_ERROR)
        self._toast_until = time.monotonic() + theme.EVENT_TOAST_MS
        self._last_key = None          # 强制下一帧重画 toast

    # ------------------------------------------------------------------
    # 点浮窗 = 停任务（单击 / 双击 / 长按）
    # ------------------------------------------------------------------
    def _signal_nav_stop(self) -> bool:
        """写停止信号文件，告诉 ``e2e_pathfind.py`` 收工。

        :returns: 写成功返回 True。

        ★ 先删再建
          -------
          ``Path.touch()`` 在文件已存在时是 no-op，mtime 不变。虽然读方
          只看 ``exists()``、不看时间戳，但先删再建能保证「这一次点击」
          与「上一次残留」在 mtime 上一定分得开 —— 将来若有人加了
          「超时自动复位」，残留文件就会变成一个查不出来的问题源。
        """
        try:
            layout.NAV_STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
            layout.NAV_STOP_FILE.unlink(missing_ok=True)
            layout.NAV_STOP_FILE.touch()
        except OSError as exc:
            logger.warning("写停止信号失败：%s", exc)
            return False
        return True

    def _nav_running(self) -> bool:
        """寻路脚本此刻在跑吗？**读状态文件判断**。

        ``load_nav_state`` 自带时效判断（活跃态 5s），所以 e2e 崩了或者
        已经退出，这里会自然变成 False —— 不需要另开一个「进程在不在」
        的判据（Windows 上按进程名找本身就脆）。
        """
        nav = layout.load_nav_state()
        return nav is not None and str(nav.get("state", "")) not in (
            "arrived", "panic")

    def _stop_task(self) -> None:
        """单击浮窗：**轻停** —— 先停寻路，寻路没在跑就停引擎。"""
        nav_was_running = self._nav_running()
        nav_ok = self._signal_nav_stop()
        if nav_was_running:
            self._toast = ("已发送停止寻路信号" if nav_ok
                           else "寻路在跑，但停止信号写不进去")
            self._toast_color = (theme.ACCENT_RUN if nav_ok
                                 else theme.ACCENT_ERROR)
        else:
            # 寻路没在跑 -> 这一下应该落在引擎上。**不发寻路停止信号**：
            # 那会在 e2e 下次启动前留下一颗地雷（虽然它启动时会自删，
            # 但没必要留下）。
            ok, detail = self._send_console_cmd("STOP_ENGINE", "overlay_stop")
            if ok:
                self._toast = "已发送停止引擎命令"
                self._toast_color = theme.ACCENT_RUN
            else:
                logger.warning("发送 STOP_ENGINE 失败：%s", detail)
                self._toast, self._toast_color = (
                    f"停止失败：{detail}", theme.ACCENT_ERROR)
        self._toast_until = time.monotonic() + theme.EVENT_TOAST_MS
        self._last_key = None

    def _stop_all(self) -> None:
        """双击 / 长按浮窗：**全停** —— 寻路 + 引擎一起停。"""
        nav_ok = self._signal_nav_stop()
        eng_ok, detail = self._send_console_cmd("STOP_ENGINE", "overlay_stop")
        if nav_ok and eng_ok:
            self._toast, self._toast_color = ("已发送全停命令", theme.ACCENT_RUN)
        else:
            # ★ 两件事分别报，不合成一句含糊的「部分失败」——
            #   用户需要知道到底是哪个没停下来。
            parts = []
            if not nav_ok:
                parts.append("寻路停止信号写不进去")
            if not eng_ok:
                parts.append(f"引擎命令失败（{detail}）")
            self._toast, self._toast_color = (
                "全停部分失败：" + "、".join(parts), theme.ACCENT_ERROR)
        self._toast_until = time.monotonic() + theme.EVENT_TOAST_MS
        self._last_key = None

    def _open_console(self) -> None:
        """▶ 按钮：把主控台叫到前台，没有就启动一个。

        ★ overlay 只**发起**，不接管
          --------------------------
          和 ↻ 按钮同一个道理：主控台的进程是它自己持有的 Popen 句柄，
          overlay 去 fork 一个出来就是又一份孤儿进程来源 —— 正是这套
          东西要消灭的问题。管道能唤起就唤起（静默，无对话框），
          唤起不了才让主控台自己起自己。

        ★ 成功/失败都必须给 toast
          --------------------
          用户点了按钮却毫无反应，会以为按钮坏了。而最常见的失败
          （找不到 Console.exe）光看浮窗**完全看不出来**。
        """
        try:
            result = console_launcher.launch_console()
        except Exception as exc:                       # noqa: BLE001
            # 理论上 launch_console 自己吞掉了所有异常；这里是最后一道
            # 保险 —— 异常冒到 tkinter 事件循环，按钮就再也不响了。
            logger.error("唤起主控台异常：%s", exc, exc_info=True)
            result = f"唤起主控台异常：{exc}"

        if result == console_launcher.RESULT_IPC:
            self._toast, self._toast_color = (
                "已唤起运行中的主控台", theme.ACCENT_RUN)
        elif result == console_launcher.RESULT_SPAWN:
            self._toast, self._toast_color = (
                "已启动主控台", theme.ACCENT_RUN)
        else:
            logger.warning("唤起主控台失败：%s", result)
            self._toast, self._toast_color = (result, theme.ACCENT_ERROR)
        self._toast_until = time.monotonic() + theme.EVENT_TOAST_MS
        self._last_key = None          # 强制下一帧重画 toast

    def _confirm_kill(self) -> None:
        """强杀二次确认。

        KILL 是不可逆的（跳过一切清理，按键可能卡住），所以必须有确认。
        确认框用系统 tk 的 messagebox 而不是自绘 —— 少一套控件，
        而且它有原生的键盘可达性。
        """
        from tkinter import messagebox  # noqa: PLC0415

        if not messagebox.askyesno(
                "强制终止",
                "将立即终止 run_loop，\n跳过按键释放与资源清理。\n\n"
                "确定要继续吗？"):
            return
        try:
            self.sender.send("KILL")
        except Exception as exc:  # noqa: BLE001
            logger.error("发送 KILL 失败：%s", exc)
            messagebox.showerror("强制终止", f"发送失败：{exc}")
            return
        self._toast, self._toast_color = "已发送强制终止", theme.ACCENT_ERROR
        self._toast_until = time.monotonic() + theme.EVENT_TOAST_MS

    # ------------------------------------------------------------------
    def _drain_events(self) -> None:
        for ev in self.reader.poll_events():
            self._toast = ev.message or ev.code
            self._toast_color = theme.severity_color(str(ev.severity))
            self._toast_until = time.monotonic() + theme.EVENT_TOAST_MS
            if ev.code == "PANIC_TRIGGERED":
                self._panic_until = time.monotonic() + theme.PANIC_FLASH_MS
            self._last_key = None      # 强制重绘

    # ------------------------------------------------------------------
    def _current_snapshot(self):
        """当前快照。**读不到就沿用上一份**，再没有才给「离线」。

        ★ 不能因为一次读失败就报离线 —— 快照是原子替换的，
          读失败只可能是"从没写过"，来回切会让面板反复闪。
        """
        from autopilot.ui.schema import Health, StateSnapshot  # noqa: PLC0415

        snap = self.reader.read_snapshot()
        if snap is not None:
            return snap
        cached = self.reader.cached_snapshot
        if cached is not None:
            stale = dict(cached.model_dump())
            stale["health"] = Health.OFFLINE
            return StateSnapshot.model_validate(stale)
        return StateSnapshot(health=Health.OFFLINE)

    def tick(self) -> None:
        """一次刷新。**只在内容变化时碰 tk**。"""
        self._drain_events()

        now = time.monotonic()
        snap = self._current_snapshot()
        panic = now < self._panic_until
        toast = self._toast if now < self._toast_until else None

        cap = getattr(snap, "capture", None)
        ch = str(getattr(cap, "active", "") or "")
        if ch:
            self._last_channel = ch

        model = layout.build_panel(snap, toast=toast,
                                  toast_sev="WARN", panic=panic,
                                  last_channel=self._last_channel,
                                  # ★ 寻路状态（2026-10-04）。读不到/
                                  #   过期就返回 None，面板照旧显示引擎那 5 行。
                                  nav=layout.load_nav_state(),
                                  # ★ 会话状态（2026-10-10）。同样读不到就
                                  #   返回 None，第 1 行显示「无会话」——
                                  #   插件没跑/崩了正是最需要看见这个的时候。
                                  session=layout.load_session_state())
        key = model.content_key()
        if key == self._last_key:
            return                      # ★ 省电关键
        self._last_key = key

        border = theme.ACCENT_ERROR if panic else theme.BORDER
        self.root.configure(bg=border)
        for lbl, line in zip(self._lines, model.lines):
            lbl.configure(text=line.text, fg=line.color)
        self._toast_lbl.configure(text=model.toast or "",
                                 fg=model.toast_color)
        # 底色跟着边框走，营造"描边"效果（比放真边框控件省一个 widget）
        for lbl in list(self._lines) + [self._toast_lbl]:
            lbl.configure(bg=border)
        self.root.geometry(f"{self.panel_width}x{self._panel_h}"
                           f"+{self._get_pos()[0]}+{self._get_pos()[1]}")

    def run(self) -> int:
        def loop() -> None:
            self.tick()
            self.root.after(theme.POLL_INTERVAL_MS, loop)

        self.root.after(theme.POLL_INTERVAL_MS, loop)
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.root.destroy()
            except Exception:  # noqa: BLE001
                pass
        return 0


def build(base_dir: object | Path | None = None, *,
          mode: str = "product", **kw):
    """建一个 overlay 窗口。``kw`` 原样透传给 :class:`OverlayApp`
    （``x`` / ``y`` / ``alpha`` / ``scale`` / ``reset_pos``）。

    :param mode: ``"product"`` = 5 行精简面板（默认）；``"dev"`` = 开发者版
        （顶部状态行 + 事件流 + 分析摘要，面板更宽、轮询更慢但信息密度高）。
        未知值**响亮失败** —— 静默退回 product 会让人以为在看开发者版。
    """
    from autopilot.ui.reader import ChannelReader  # noqa: PLC0415

    if mode == "product":
        return OverlayApp(ChannelReader(base_dir), **kw)
    if mode == "dev":
        from autopilot.ui.overlay.dev_panel import DevOverlayApp  # noqa: PLC0415

        return DevOverlayApp(ChannelReader(base_dir), **kw)
    raise ValueError(f"未知的 overlay 模式 {mode!r}；合法值：'product' / 'dev'")


# ★ 曾经的 panel_height() 已删除
#   它用 6 × theme.LINE_HEIGHT + padding×2 估算高度，而 LINE_HEIGHT 是
#   按 10pt 标定的固定值 —— 早就不等于真实行高了。留着它就是一个
#   "看起来能用、其实永远算错"的陷阱：谁调用它谁排版崩。
#   真实高度是 :meth:`OverlayApp._compute_metrics` 实测字体后算出来的
#   ``self._panel_h``，是实例属性，没有模块级版本。


__all__ = [
    "LIVE_SCALE_EPS",
    "LIVE_SCALE_FILE",
    "LIVE_SCALE_POLL_MS",
    "OverlayApp",
    "build",
]
