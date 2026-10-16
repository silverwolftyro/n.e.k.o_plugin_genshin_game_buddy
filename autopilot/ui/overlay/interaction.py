"""overlay 交互：拖动 / 键盘微调 / 按钮。

★ 拖动阈值（3px）
-----------------
阈值设为 0 的话，**点一下按钮也会触发拖动** —— 窗口跟着跳一下，
用户想点「关闭」结果窗口飞走了。3px 能区分「手抖」和「真拖」。

★ 按钮触达区域比视觉大
---------------------
触达 32×32、字形 14pt。目标用户不该需要瞄准 —— 手抖 / 追踪不准是常态，
按钮做小一点等于同时给误触和点不中创造机会。

★ 按钮必须**画出来**（曾经的 bug）
--------------------------------
早期版本只造了 :class:`HitBox`（一个纯逻辑的热区）却没创建任何控件，
于是「右上角有个按钮」只存在于代码里 —— 用户看得见面板、看不见关闭键，
而 overlay 是 ``overrideredirect(True)`` 的无边框窗，Ctrl+C 在 Windows 上
也不吃，等于只能去终端里杀进程。

现在按钮是真正的 ``tk.Label``：有背景色、有字形、有悬停反馈。
:class:`HitBox` 退化为热区记录，与控件位置由 ``layout_buttons`` 一次算清。
"""

from __future__ import annotations

import logging
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Callable

from . import theme

logger = logging.getLogger(__name__)


@dataclass
class HitBox:
    """按钮热区。"""

    x: int
    y: int
    w: int
    h: int
    action: str

    def contains(self, px: int, py: int) -> bool:
        return (self.x <= px < self.x + self.w
                and self.y <= py < self.y + self.h)


#: 按钮个数。★ **加减按钮只改这一处**。
#:
#: 定义在类外面（而不是 :attr:`Interaction.N_BUTTONS`）是为了让
#: :func:`button_row_left` 能被 :mod:`app` 在**排版之前**调用 ——
#: ``_compute_metrics`` 要用最左按钮的 x 来定文字左边界，那时按钮
#: 控件还没建。
N_BUTTONS = 4

#: 从右往左的按钮名。与破坏性递增一一对应。
BUTTON_ORDER = ("close", "kill", "restart", "console")

# ------------------------------------------------------------------ 点击语义
# ★ 这三个手势是**破坏性**的（会停任务），所以识别必须比拖动更保守。
#: 双击判定窗口（毫秒）。两次「按下-抬起」间隔 ≤ 它 = 双击。
DOUBLE_CLICK_MS = 250

#: 长按阈值（毫秒）。按住**不动**超过这么久 = 长按。
LONG_PRESS_MS = 500

# ★ 为什么单击不在「按下时」就起 250ms 计时，而要等到**抬起**
#   ----------------------------------------------------------
#   需求是「单击 → 轻停，双击 → 全停，长按 >500ms → 全停」。
#   三个判定的**时序必须能共存**，而长按阈值(500ms) 比双击窗口(250ms) 长：
#   若按下 250ms 就判单击，那么「按住 600ms」会先在 250ms 触发一次轻停，
#   再在 500ms 触发一次全停 —— 用户想长按，结果先被轻停了一下。
#
#   所以改成经典做法：**抬起**才算一次点击候选，
#   抬起后再等一个双击窗口看有没有第二下。
#   这样长按路径（一直不抬起）根本不会进入单击判定。
DOUBLE_CLICK_S = DOUBLE_CLICK_MS / 1000.0


def button_xs(width: int) -> list[int]:
    """四个按钮的 x 坐标，**从右往左**（索引 0 = 最右的 ×）。

    ★ 为什么提成模块级函数
      --------------------
      以前这里是 :meth:`Interaction.layout_buttons` 里三行算式，而
      ``app._compute_metrics`` 又**另写了一遍** ``2 * BUTTON_HIT`` 去
      估按钮区宽度。两处各算各的，加按钮时改一处忘一处就是错位 ——
      「两处各算各的，算岔了」正是命令通道那次事故的成因。
      现在只有这一处真源。

    :returns: 长度恒为 :data:`N_BUTTONS` 的 list。
    """
    size, m = theme.BUTTON_HIT, theme.BTN_MARGIN
    gap = theme.px(4)               # 间隔走 theme.px，别写死
    return [width - (i + 1) * size - m - i * gap
            for i in range(N_BUTTONS)]


def button_row_left(width: int) -> int:
    """按钮行最左边那个按钮的 x（**不是**左边界，还差一个 size）。"""
    return button_xs(width)[-1]


class Interaction:
    """绑定到一个 tk 窗口上。

    只做"意图"：把鼠标/键盘事件翻译成 移动/点击/命令 三类动作，
    具体的窗口操作交给回调。
    """

    def __init__(self, root, *, on_move: Callable[[int, int], None],
                 on_close: Callable[[], None],
                 on_kill: Callable[[], None],
                 on_restart: Callable[[], None],
                 on_scale: Callable[[float], None],
                 on_alpha: Callable[[float], None] | None = None,
                 on_persist: Callable[..., None],
                 on_open_console: Callable[[], None],
                 on_stop_task: Callable[[], None],
                 on_stop_all: Callable[[], None],
                 get_pos: Callable[[], tuple[int, int]]) -> None:
        self.root = root
        self.on_move = on_move
        self.on_close = on_close
        self.on_kill = on_kill
        self.on_restart = on_restart
        #: Ctrl+滚轮 -> 相对倍率（+0.05 放大 / -0.05 缩小）。
        #: 传**相对量**而不是绝对值：夹边界是调用方的责任，
        #: 而 ``theme.apply_user_scale`` **已经**夹了（USER_SCALE_MIN/MAX）。
        self.on_scale = on_scale
        #: Alt+滚轮 -> 相对透明度（+0.05 更实 / -0.05 更透）。
        #: ``None`` = 这个调用方不接（如只想调缩放的旧路径），滚轮静默忽略。
        self.on_alpha = on_alpha
        #: 窗口状态该落库了 —— 由 app 用它手里**完整**的状态去写。
        #: ★ 这里**不自己调 layout.save_pos()**：本类只拿得到 x/y，
        #:   写下去会把 alpha（和 scale）换成默认值 ——
        #:   拖一下窗口就悄悄把用户的 --alpha 吃掉。
        #:   「要持久化」是意图，「写什么」是有真值的那个类的责任。
        self.on_persist = on_persist
        #: ▶ 按钮：把主控台叫到前台，没有就启动一个。
        self.on_open_console = on_open_console
        #: 点浮窗空白处 = 轻停（先停寻路，没寻路就停引擎）。
        self.on_stop_task = on_stop_task
        #: 双击 / 长按 = 全停（寻路 + 引擎）。
        #:
        #: ★ 这两个是**破坏性**回调
        #:   识别必须比拖动更保守（见 :data:`DOUBLE_CLICK_MS` /
        #:   :data:`LONG_PRESS_MS` 的说明）。用户一次误拖不能停掉任务 ——
        #:   而拖浮窗恰恰是他最常做的动作。
        self.on_stop_all = on_stop_all
        self.get_pos = get_pos

        self._press_at: tuple[int, int] | None = None
        self._press_root: tuple[int, int] | None = None
        self._dragging = False
        self.buttons: list[HitBox] = []
        self._widgets: dict[str, tk.Label] = {}

        # 点击判定的运行时状态
        self._click_timer = None          # 待决「是不是单击」的 after id
        self._long_timer = None           # 长按倒计时的 after id
        self._last_click_at: float | None = None   # 上一次抬起的时间戳
        self._long_armed = False          # 本次按下已按住超过长按阈值
        self._press_on_button = False    # 本次按下命中了按钮热区

        self._bind()

    # ------------------------------------------------------------------
    # 按钮几何见模块级 :func:`button_xs` —— 那是**唯一真源**
    # ------------------------------------------------------------------
    def layout_buttons(self, width: int, height: int) -> None:
        """在右上角摆出**真正可见**的 × 、⏹、↻ 、▶ 四个按钮。

        ★ 从右往左 = 破坏性递增
          ------------------------------
              ×  最右 —— Windows 标题栏关闭键的肌肉记忆位置
              ⏹        —— 危险操作（杀引擎）
              ↻        —— 重启引擎（幂等，且唯一点下去会变绿）
              ▶  最左 —— 打开主控台，纯导航，什么都不会被弄坏

          用户戳右下角触到的一直是最危险的那个，这是刻意的。

        ★ 四个按钮**不会越界**，不需要加宽面板
          ----------------------------------------------
          x 坐标每一项（size / m / gap）都来自 ``theme.px()``，即都乘了
          ``SCALE``；而 ``width`` = ``theme.PANEL_WIDTH`` 同样乘了 SCALE。
          **整个布局是缩放无关的**，任意缩放下最左按钮的左边距都是
          ``PANEL_WIDTH - 4*size - m - 3*gap``，恒为正。

          在 100% 屏（SCALE=1.0，实测）：
              PANEL_WIDTH=480, size=32, m=6, gap=4
              close_x=442  kill_x=406  restart_x=370  console_x=334
              （最左边界 334 >> 0，占宽度的 30%）
          所以 PANEL_WIDTH 维持 480 不变，5 行状态文字的排版也不受影响。
        """
        close_x, kill_x, restart_x, console_x = button_xs(width)
        y = theme.BTN_MARGIN
        size = theme.BUTTON_HIT

        self.buttons = [
            HitBox(restart_x, y, size, size, "restart"),
            HitBox(kill_x, y, size, size, "kill"),
            HitBox(close_x, y, size, size, "close"),
            HitBox(console_x, y, size, size, "console"),
        ]

        self._make_button("restart", restart_x, y, size,
                          theme.RESTART_GLYPH,
                          bg=theme.BTN_BG, fg=theme.BTN_FG_HI,
                          hover_bg=theme.RESTART_HOVER_BG,
                          hover_fg=theme.BTN_KILL_FG)
        self._make_button("kill", kill_x, y, size, theme.KILL_GLYPH,
                          bg=theme.BTN_KILL_BG, fg=theme.BTN_KILL_FG,
                          hover_bg=theme.BTN_KILL_BG,
                          hover_fg=theme.ACCENT_ERROR)
        self._make_button("close", close_x, y, size, theme.CLOSE_GLYPH,
                          bg=theme.BTN_BG, fg=theme.BTN_FG_HI,
                          hover_bg=theme.CLOSE_HOVER_BG,
                          hover_fg=theme.BTN_KILL_FG)
        # ★ 悬停保持中性灰，不跟 ↻ 抢绿色 —— 「会变绿」是 ↻ 的专属标识，
        #   两个按钮都用绿色就没有标识可言了。
        self._make_button("console", console_x, y, size,
                          theme.CONSOLE_GLYPH,
                          bg=theme.BTN_BG, fg=theme.BTN_FG_HI,
                          hover_bg=theme.BTN_HOVER_BG,
                          hover_fg=theme.FG)

    def _make_button(self, name: str, x: int, y: int, size: int,
                     glyph: str, *, bg: str, fg: str,
                     hover_bg: str, hover_fg: str) -> None:
        """建一个按钮控件。**tk.Label 而不是 Button** ——
        Button 自带 relief/border，样式难统一；Label 只是一个带背景的文本，
        正好是我们要的"色块 + 字形"。
        """
        lbl = tk.Label(
            self.root, text=glyph, bg=bg, fg=fg,
            font=(theme.FONT_FAMILY, theme.BTN_GLYPH_SIZE, "bold"),
            width=2, height=1, cursor="hand2",
        )
        lbl.place(x=x, y=y, width=size, height=size)

        # 悬停反馈用 Label 自己的 bind，**不用 root 的 _on_press** ——
        # 否则会同时触发 root 的拖动逻辑。
        def enter(_e=None, l=lbl, b=hover_bg, f=hover_fg):
            l.configure(bg=b, fg=f)

        def leave(_e=None, l=lbl, b=bg, f=fg):
            l.configure(bg=b, fg=f)

        lbl.bind("<Enter>", enter)
        lbl.bind("<Leave>", leave)
        # ★★ 必须 return "break" —— 这条不是洁癖，是正确性问题。
        #
        #   tkinter 的事件**不是**「不冒泡到 toplevel」的。Label 的
        #   bindtags 是 ``('Label', '.', 'all')``，第二项就是**顶层窗口**。
        #   不 break 的话，点一下按钮会依次触发：
        #     1. Label 自己的 <Button-1>  -> _activate(name)
        #     2. 顶层 '.' 的 <ButtonPress-1> -> root._on_press
        #   于是
        #     · _activate 被调用**两次**（⏹ 会弹两次确认框；↻ 会重启两次引擎）
        #     · root 那边还会记一次「准备拖动」，松手时可能真的挪窗口
        #     · 加上「点浮窗即停任务」之后，点 ⏹ 还会顺手把任务停了
        #   三条里任何一条单独出现都够糟。
        def _btn(_e, n=name):
            self._activate(n)
            return "break"

        lbl.bind("<Button-1>", _btn)
        self._widgets[name] = lbl

    def _activate(self, name: str) -> None:
        """点按分发。**关闭不确认，强杀才确认**（由调用方负责）。"""
        if name == "close":
            self.on_close()
        elif name == "kill":
            self.on_kill()
        elif name == "restart":
            # ★ 重启**不确认**：它是幂等的，而且要真等引擎停下来再起，
            #   弹个模态框只会挡住用户。失败有 toast 提示。
            self.on_restart()
        elif name == "console":
            # ★ 也不确认：它只会「叫出主控台」或「启动一个主控台」，
            #   不碰引擎、不杀进程、最坏情况是多开一个实例，而主控台
            #   自己的单例会兜住。成功/失败都有 toast。
            self.on_open_console()

    # ------------------------------------------------------------------
    def _bind(self) -> None:
        r = self.root
        r.bind("<ButtonPress-1>", self._on_press)
        r.bind("<B1-Motion>", self._on_drag)
        r.bind("<ButtonRelease-1>", self._on_release)
        r.bind("<KeyPress>", self._on_key)
        # 获得焦点，否则方向键不会触发
        r.bind("<FocusIn>", lambda _e: r.focus_set())
        # ★ Ctrl + 滚轮 = 缩放（2026-10-04）。
        #   必须绑两个序列：Windows 上 <MouseWheel> 的 delta 是 ±120，
        #   Ctrl 按住时会变成 <Control-MouseWheel>，只绑后者在某些
        #   焦点状态下收不到事件（面板是 overrideredirect 窗，
        #   焦点状态不稳）。
        r.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        r.bind("<MouseWheel>", self._on_wheel_plain)
        # ★ Alt+滚轮 = 透明度（2026-10-10）。Ctrl 已经占了缩放，不抢。
        #   Windows 上 <MouseWheel> 带修饰键会变成**不同**的序列
        #   （<Alt-MouseWheel> / <Control-Alt-MouseWheel>），
        #   所以这三者互不冲突，各绑各的。
        #   Ctrl+Alt 是第二档：某些环境（远程桌面、部分 IME）会吃掉裸 Alt，
        #   两个都绑上总有一个能用。
        if self.on_alpha is not None:
            r.bind("<Alt-MouseWheel>", self._on_alpha_wheel)
            r.bind("<Control-Alt-MouseWheel>", self._on_alpha_wheel)

    # -- 缩放 ------------------------------------------------------------
    #: 一格滚轮改多少倍率。滑块一格是 5%，滚轮用同一档，
    #: 免得用户要滚 20 格才看到一点变化。
    WHEEL_SCALE_STEP = 0.05

    def _on_wheel_plain(self, ev) -> None:
        """没按 Ctrl 的滚轮：**不缩放，也不做任何事**。

        ★ 为什么要显式吃掉它
        --------------------
        浮窗是 ``overrideredirect(True)`` 的无边框窗，滚轮事件落在它身上
        却没有绑定时，tk 会把它**冒泡**给父/兄弟控件；万一将来有人加了
        带滚轮语义的控件，这里就会误触发。显式吞掉 = 意图明确。
        """
        return "break"

    def _on_ctrl_wheel(self, ev) -> None:
        """Ctrl+滚轮 -> 调缩放。**步进方向和滚轮方向一致。"""
        if ev.delta > 0:
            self.on_scale(self.scale_step)          # 滚上 = 放大
        elif ev.delta < 0:
            self.on_scale(-self.scale_step)         # 滚下 = 缩小
        return "break"

    @property
    def scale_step(self) -> float:
        return self.WHEEL_SCALE_STEP

    # -- 透明度 ----------------------------------------------------------
    #: 一格滚轮改多少透明度。0.05 = 20 格从全透走到全实。
    WHEEL_ALPHA_STEP = 0.05

    def _on_alpha_wheel(self, ev) -> None:
        """Alt+滚轮 -> 调透明度。步进方向和滚轮方向一致（滚上更实）。"""
        cb = self.on_alpha
        if cb is None:
            return "break"
        if ev.delta > 0:
            cb(self.WHEEL_ALPHA_STEP)
        elif ev.delta < 0:
            cb(-self.WHEEL_ALPHA_STEP)
        return "break"

    # -- 鼠标 ------------------------------------------------------------
    def _on_press(self, ev) -> str:
        self.root.focus_set()
        # ★ 这是按钮分发的**第二道保险**。
        #   第一道是按钮 Label 自己的 <Button-1>，它已经 return "break"，
        #   所以正常情况下事件根本到不了这里。留着是因为 Label 万一没建起来
        #   （_reflow 销毁重建的间隙、或 tk 建控件失败），热区还在，
        #   按钮不至于彻底失灵。返回 "break" 同样是为了不进入点击判定。
        for b in self.buttons:
            if b.contains(ev.x, ev.y):
                self._activate(b.action)
                # ★ 记住「这次按的是按钮」。按钮 Label 那道 break 只拦住了
                #   <ButtonPress-1>，<ButtonRelease-1> 照样会冒泡到 root ——
                #   不记住的话 _on_release 会把这次抬手记成一次点击，
                #   抬手 250ms 后顺手把用户的任务停了。点 × 关窗口时
                #   窗口已经没了看不出来，点 ↻ / ▶ 时就是实打实的误杀。
                self._press_on_button = True
                return "break"

        self._press_on_button = False
        self._press_at = (ev.x_root, ev.y_root)
        self._press_root = self.get_pos()
        self._dragging = False
        self._long_armed = False
        # ★ 长按倒计时从**按下**这一刻起算，不是从「判定为点击」起算。
        #   到点只是布防（见 _on_long_press），抬手才结算。
        self._cancel_long()
        self._long_timer = self.root.after(LONG_PRESS_MS,
                                           self._on_long_press)
        return "break"

    def _on_drag(self, ev) -> None:
        if self._press_at is None or self._press_root is None:
            return
        dx = ev.x_root - self._press_at[0]
        dy = ev.y_root - self._press_at[1]
        # ★ 阈值：超过 theme.DRAG_THRESHOLD 才算拖。
        #   **一旦判定为拖，立刻作废本次的点击与长按判定** ——
        #   否则「按下 → 犹豫两秒 → 拖走」会在 500ms 时被判成长按，
        #   把用户的任务停了，而他只是想挪一下浮窗。这是本轮最危险的
        #   组合：拖浮窗恰恰是他最常做的动作。
        if not self._dragging:
            if abs(dx) < theme.DRAG_THRESHOLD and \
                    abs(dy) < theme.DRAG_THRESHOLD:
                return
            self._dragging = True
            self._cancel_click()
            self._cancel_long()
            self._long_armed = False
        self.on_move(self._press_root[0] + dx, self._press_root[1] + dy)

    def _on_release(self, _ev) -> None:
        self._cancel_long()
        was_drag = self._dragging
        was_long = self._long_armed
        was_button = self._press_on_button
        self._press_at = None
        self._press_root = None
        self._dragging = False
        self._long_armed = False
        self._press_on_button = False
        if was_button:
            # ★ 按的是按钮 -> 这次抬手**不是**一次浮窗点击。到此为止。
            #   连 _cancel_click() 都不必调：按钮点击根本没进过点击判定，
            #   待决的那个单击定时器属于上一次浮窗点击，不该被这次带偏。
            return
        if was_drag:
            # 拖过 -> 只落库位置。点击与长按**都已被 _on_drag 作废**，
            # 这里绝不能再补一发停止 —— 用户挪浮窗不该停掉任何东西。
            self.on_persist(*self.get_pos())
            return
        if was_long:
            # 长按到期后才抬的手 -> 全停。**同时作废单击**，
            # 否则待决的那个定时器稍后还会再补一发轻停，连打两次。
            self._cancel_click()
            self.on_stop_all()
            return
        self._register_click()

    # -- 点击判定 ---------------------------------------------------------
    def _register_click(self) -> None:
        """一次「按下-抬起」完成了。判定它属于单击还是双击。"""
        now = time.monotonic()
        # ★ 必须**先取**再取消。_cancel_click() 会把 _last_click_at 清成
        #   None，先调它就等于自己把自己的上一点擦掉了 —— 双击永远
        #   判不出来，每次都退化成两次单击。
        prev = self._last_click_at
        self._cancel_click()
        if prev is not None and (now - prev) <= DOUBLE_CLICK_S:
            # 第二下落在窗口内 -> 立刻判双击（全停）。这里**不能**再留一个
            # 待决的单击定时器：它稍后还会补一发 on_stop_task，用户双击
            # 一下就被「全停 + 轻停」连打两次。
            self.on_stop_all()
            return
        self._last_click_at = now
        self._click_timer = self.root.after(DOUBLE_CLICK_MS,
                                            self._resolve_single)

    def _resolve_single(self) -> None:
        """双击窗口到期，仍没有第二下 -> 单击成立（轻停）。"""
        self._click_timer = None
        self._last_click_at = None
        self.on_stop_task()

    def _on_long_press(self) -> None:
        """按住不动超过 :data:`LONG_PRESS_MS` -> **布防**，不下手。

        ★★ 为什么这里**不**直接调 ``on_stop_all``
          --------------------------------------
          「按下 → 心里犹豫两秒 → 才开始拖」是很正常的手势。如果到点就
          动手，用户会在**还没挪动窗口**的时候任务已经被停了 —— 而他
          以为自己只是换了个位置。这个误判的代价是破坏性的。

          所以长按到期只**布防**：记下「这次按下已经算长按」。等**抬手**
          时才结算 —— 那时如果他真的拖动了，:meth:`_on_drag` 早已把布防
          取消；没拖动才真正发全停。

          代价：一直按着不抬手就什么都不会发生。这正是想要的 —— 用户
          还没松手，就还没结束。
        """
        self._long_timer = None
        self._long_armed = True

    def _cancel_click(self) -> None:
        if self._click_timer is not None:
            self.root.after_cancel(self._click_timer)
            self._click_timer = None
        self._last_click_at = None

    def _cancel_long(self) -> None:
        if self._long_timer is not None:
            self.root.after_cancel(self._long_timer)
            self._long_timer = None

    # -- 键盘 ------------------------------------------------------------
    def _on_key(self, ev) -> None:
        """Esc = 关闭；方向键 = 微调位置。普通 1px，Shift 10px。

        ★ Esc 是**附加**的关闭路径，不是"更好的关闭方式"。目标用户不该
          被要求必须瞄准一个 32px 的按钮 —— 万一手抖、或根本不知道那个
          叉在哪儿，键盘永远是最快的出路。方向键行为一个字没改。
        """
        if ev.keysym == "Escape":
            self.on_close()
            return "break"

        step = theme.NUDGE_LARGE if (ev.state & 0x0001) else theme.NUDGE_SMALL
        dx = dy = 0
        if ev.keysym in ("Left",):
            dx = -step
        elif ev.keysym in ("Right",):
            dx = step
        elif ev.keysym in ("Up",):
            dy = -step
        elif ev.keysym in ("Down",):
            dy = step
        else:
            return
        x, y = self.get_pos()
        self.on_move(x + dx, y + dy)
        self.on_persist(x + dx, y + dy)
        return "break"


__all__ = ["BUTTON_ORDER", "DOUBLE_CLICK_MS", "LONG_PRESS_MS", "N_BUTTONS",
           "HitBox", "Interaction", "button_row_left", "button_xs"]
