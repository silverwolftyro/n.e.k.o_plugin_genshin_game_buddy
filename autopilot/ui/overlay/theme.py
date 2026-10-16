"""颜色 / 字体 / 尺寸常量。

Catppuccin Mocha 配色。**全部硬编码**，不读主题文件 —— 少一次 IO，
少一个可能坏掉的配置项。overlay 的视觉不该需要用户配置。

★ 高 DPI：本文件的所有尺寸都是**物理像素**
--------------------------------------------------
overlay 入口 (``tools/overlay.py``) 在 ``Tk()`` 创建**之前**声明了
进程 DPI 感知。声明之后 Windows 不再替我们做虚拟放大，tkinter 拿到的是
物理像素 —— 于是代码里写 ``FONT_SIZE = 10`` 就是真的物理 10pt，
在 150% 缩放的屏幕上看起来比声明前小 1.5 倍（标题挤成一团、文字被裁切）。

所以这里统一乘一次 :data:`SCALE`。**所有尺寸常量乘 SCALE，字号另外乘**
（两者不是一回事：字号是排版单位，尺寸是像素）。

★ 缩放 = 自动探测 × 用户倍率
--------------------------------------
    有效缩放 SCALE = BASE_SCALE × _USER_SCALE
                    └系统 DPI└  └用户 --scale┘

`BASE_SCALE` 是开机探测出来的（150% 屏 = 1.5）。但**探测值不一定合手意** ——
用户可能就想看小一点、少挡点游戏画面。所以留了 `_USER_SCALE` 这个倍率，
``1.0`` 表示"别管，就用系统的"。主控台的滑块和 ``--scale`` 命令行改的是它。

改缩放**只能**走 :func:`apply_user_scale`（它会重算所有派生常量）。
直接给 ``SCALE`` 赋值不会触发重算，会留下"字号变、行距没变"的错位状态。

新增任何尺寸常量时，记得同时加进 :func:`_recompute`。
"""

from __future__ import annotations

import logging
import math
import sys

logger = logging.getLogger(__name__)


def detect_scale() -> float:
    """探测系统缩放比（96 DPI = 1.0）。

    读 ``LOGPIXELSX``（索引 88）。**拿不到就返回 1.0** ——
    缩放探测失败不该让 overlay 起不来，只会让它偏小/偏大一点。

    ★★ 必须在**进程声明 DPI 感知之后**调用，否则结果偏小一半 ★★
    ----------------------------------------------------------
    桌面 DC 的 ``LOGPIXELSX`` 会随进程的 DPI 感知状态变化。150% 屏实测：

        声明前   LOGPIXELSX = 96   SM_CXSCREEN = 1707（虚拟化）
        声明后   LOGPIXELSX = 144  SM_CXSCREEN = 2560（物理像素）
                 -> detect_scale() 从 1.0 变成 1.5

    因为本模块在导入时就把结果算进模块级常量，**导入时机就是一切**。
    入口 ``tools/overlay.py`` 因此把 theme 的导入推迟到
    ``declare_dpi_aware()`` 之后（app 是函数内延迟导入的）。

    ⚠ 任何在本模块声明之前 import 它的代码，都会把 ``BASE_SCALE``
      永久钉死在 1.0 —— 表现为"在 150% 屏上浮窗只有应有尺寸的 2/3"。
      要单独测本模块（比如验证缩放数学），**必须先自己声明 DPI 感知**。
    """
    if sys.platform != "win32":
        return 1.0
    try:
        import ctypes  # noqa: PLC0415 - 只在 Windows 上需要

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        hdc = user32.GetDC(0)
        if not hdc:
            return 1.0
        try:
            dpi = gdi32.GetDeviceCaps(hdc, 88)   # LOGPIXELSX
        finally:
            user32.ReleaseDC(0, hdc)
        if not dpi or dpi <= 0:
            return 1.0
        return float(dpi) / 96.0
    except Exception:  # noqa: BLE001 - 探测失败必须静默回退
        return 1.0


#: 自动探测到的系统缩放比（1.0 / 1.25 / 1.5 / 2.0 …）。
#: **只读** —— 要改缩放请调 :func:`apply_user_scale`，不要直接赋值。
BASE_SCALE: float = detect_scale()

#: 用户缩放倍率的合法区间。下限不是 0：倍率 0 会让 :func:`px` 全返回 0，
#: 面板塌成一个点 —— 那种值只可能是填错了，不是"想要的效果"。
USER_SCALE_MIN = 0.5
USER_SCALE_MAX = 3.0

#: 用户缩放倍率。**1.0 = 不覆盖**（此时 SCALE == BASE_SCALE，与拆分前完全一致）
_USER_SCALE: float = 1.0

#: 有效缩放 = :data:`BASE_SCALE` × :data:`_USER_SCALE`。
#:
#: ★ 所有尺寸/字号常量都由它推导，**不要**直接赋值 SCALE ——
#:   直接赋值不会触发重算，会留下"字号变了、行距没变"的半拉子状态，
#:   表现为文字互相重叠、末行被面板底边裁掉（上一轮「意」看成「音图」
#:   就是这个状态）。要改就调 :func:`apply_user_scale`。
SCALE: float = BASE_SCALE


def user_scale() -> float:
    """当前用户缩放倍率（1.0 = 未覆盖）。给日志/调试用。"""
    return _USER_SCALE


def px(value: float) -> int:
    """把设计基准像素缩放到物理像素。**一律取整** ——
    tkinter 收到浮点尺寸会出现半像素抖动。"""
    return int(round(value * SCALE))

# ── 颜色 ────────────────────────────────────────────────────────────────
BG = "#1e1e2e"            # 面板底
FG = "#cdd6f4"            # 正文
BORDER = "#313244"        # 边框
DIM = "#6c7086"           # 次要文字

ACCENT_RUN = "#a6e3a1"    # 运行中（绿）
ACCENT_PAUSE = "#f9e2af"   # 暂停（黄）
ACCENT_STOP = "#f38ba8"    # 停止（红）
ACCENT_WARN = "#fab387"    # 告警（橙）
ACCENT_ERROR = "#f38ba8"   # 错误（红）
ACCENT_IDLE = "#89b4fa"    # 离线（蓝）

BTN_BG = "#313244"
BTN_FG = "#cdd6f4"
BTN_HOVER = "#45475a"
BTN_KILL_BG = "#f38ba8"
BTN_KILL_FG = "#1e1e2e"

# ── 颜色主题（2026-10-10）──────────────────────────────────────────────────
#
# 上面那批常量是**当前生效**的值；:func:`apply_theme` 会在运行时改写它们。
# 因为消费方一律写 ``theme.BG`` 而不是 ``from .theme import BG``，
# 改完立刻生效 —— 这是本模块能支持换肤的唯一前提，改引用方式会失效。
#
# ★ 为什么只换颜色、不换字号/尺寸
#   ----------------------------
#   字号尺寸走 :data:`SCALE`（用户缩放倍率），再叠一套主题会出现
#   "两套缩放叠在一起"的歧义。主题只管**看**，缩放只管**多大**，
#   两条轴分开，换肤不会动布局。
THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "BG": "#1e1e2e", "FG": "#cdd6f4", "BORDER": "#313244", "DIM": "#6c7086",
        "ACCENT_RUN": "#a6e3a1", "ACCENT_PAUSE": "#f9e2af",
        "ACCENT_STOP": "#f38ba8", "ACCENT_WARN": "#fab387",
        "ACCENT_ERROR": "#f38ba8", "ACCENT_IDLE": "#89b4fa",
        "BTN_BG": "#313244", "BTN_FG": "#cdd6f4", "BTN_HOVER": "#45475a",
        "BTN_KILL_BG": "#f38ba8", "BTN_KILL_FG": "#1e1e2e",
    },
    # 浅色：浮窗压在**浅色**游戏画面（雪地/沙漠/白天）上时深色底很扎眼。
    "light": {
        "BG": "#eff1f5", "FG": "#4c4f69", "BORDER": "#ccd0da", "DIM": "#8c8fa1",
        "ACCENT_RUN": "#40a02b", "ACCENT_PAUSE": "#df8e1d",
        "ACCENT_STOP": "#d20f39", "ACCENT_WARN": "#fe640b",
        "ACCENT_ERROR": "#d20f39", "ACCENT_IDLE": "#1e66f5",
        "BTN_BG": "#ccd0da", "BTN_FG": "#4c4f69", "BTN_HOVER": "#bcc0cc",
        "BTN_KILL_BG": "#d20f39", "BTN_KILL_FG": "#eff1f5",
    },
    # 高对比度：色弱/强光环境。对比度拉到 WCAG AAA 附近，且
    # 不再靠"红=错、绿=对"这种隐含约定 —— 错误同时用红色 + 更亮的底色区分。
    "high-contrast": {
        "BG": "#000000", "FG": "#ffffff", "BORDER": "#ffffff", "DIM": "#b0b0b0",
        "ACCENT_RUN": "#00ff00", "ACCENT_PAUSE": "#ffff00",
        "ACCENT_STOP": "#ff5555", "ACCENT_WARN": "#ffaa00",
        "ACCENT_ERROR": "#ff5555", "ACCENT_IDLE": "#55ffff",
        "BTN_BG": "#000000", "BTN_FG": "#ffffff", "BTN_HOVER": "#333333",
        "BTN_KILL_BG": "#ff5555", "BTN_KILL_FG": "#000000",
    },
}

#: 当前主题名。:func:`apply_theme` 维护它。
CURRENT_THEME = "dark"


def _theme_key(name: str) -> str:
    """主题名的归一化：``high-contrast`` / ``high contrast`` / ``high_contrast`` 等价。

    ★ 为什么要归一化
      --------------
      只 replace 一次是不够的 —— ``"high-contrast"`` 换成 ``"high_contrast"``
      而键名还是连字符，就**匹配不到自己那套主题**（真踩过：high-contrast
      一申请就退回 dark）。两边都走同一个函数才不会漏。
    """
    return str(name or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")


#: 归一化名 -> 真实键
_THEME_LOOKUP = {_theme_key(k): k for k in THEMES}


def apply_theme(name: str) -> str:
    """切到某套主题，直接改写本模块的颜色常量。返回最终生效的主题名。

    :returns: 实际生效的主题名；认不出来的会退到 ``dark`` 并告警 ——
        浮窗起不来比主题不对更糟。
    """
    global CURRENT_THEME
    key = _THEME_LOOKUP.get(_theme_key(name), "dark")
    if key == "dark" and _theme_key(name) != _theme_key("dark"):
        logger.warning("未知主题 %r，可选：%s；退回 dark",
                       name, ", ".join(sorted(THEMES)))
    palette = THEMES[key]
    # 直接写模块全局：消费方用的是 theme.BG 这种属性访问，改完立刻生效。
    for attr, value in palette.items():
        globals()[attr] = value
    CURRENT_THEME = key
    logger.info("已应用主题：%s", key)
    return key


def next_theme(current: str | None = None) -> str:
    """按固定顺序轮换主题。给快捷键用。"""
    order = sorted(THEMES)
    cur = _theme_key(current or CURRENT_THEME or "")
    real = _THEME_LOOKUP.get(cur)
    if real is None:
        return order[0]
    return order[(order.index(real) + 1) % len(order)]

# ── 字体 ────────────────────────────────────────────────────────────────
#: Microsoft YaHei UI 是 Windows 自带字体，**不下载**
FONT_FAMILY = "Microsoft YaHei UI"
#: 正文字号（pt）。字号是**排版单位**，不是像素，所以单独乘。
FONT_SIZE = max(1, int(round(10 * SCALE)))
FONT_SIZE_TITLE = max(1, int(round(11 * SCALE)))
FONT_MONO = ("Consolas", max(1, int(round(10 * SCALE))))

# ── 尺寸（全部物理像素）──────────────────────────────────────────────────
PANEL_WIDTH = px(480)
PANEL_PADDING = px(10)
LINE_HEIGHT = px(20)
BUTTON_SIZE = px(26)        # 视觉尺寸
BUTTON_HIT = px(32)         # 触达区域（视觉外扩，留 3px 余量）
CORNER = px(8)

# ── 按钮（overlay 自己画，不靠系统标题栏）───────────────────────────────
#: 按钮距面板上边 / 右边距
BTN_MARGIN = px(6)
#: 关闭按钮字形。**不用字母 x** —— 小字号下 x 和乘号几乎不像，
#: 而且用户要找的是"叉"，不是"字母 x"。
CLOSE_GLYPH = "×"        # U+00D7 乘号
#: 强杀按钮字形
KILL_GLYPH = "⏹"         # U+23F9 黑方块
#: 重启引擎按钮字形。**放在 ⏹ 左边、× 更左边** ——
#: 从右往左是"破坏性递增"：×（关窗口）→ ⏹（杀引擎）→ ↻（重启引擎）最安全，
#: 肌肉记忆往右下角戳到的始终是最危险的那个。
RESTART_GLYPH = "↻"      # U+21BB 顺时针箭头
#: 重启按钮悬停背景 —— 变绿。这是全套按钮里唯一"会变绿"的一个：
#: 绿色 = 成功/安全，用户一眼就知道"这个不会出事"。
RESTART_HOVER_BG = "#a6e3a1"
#: 打开主控台按钮字形。**放在最左** —— 从右往左是破坏性递增：
#: ×（关窗口）→ ⏹（杀引擎）→ ↻（重启引擎，最安全且会变绿）→ ▶（开主控台）。
#
#: ★ 为什么是 ▶ 而不是 "C"
#:   --------------------
#:   "C" 是字母，和 × 并排时小字号下读起来像同一类符号（都是「一个字符」）；
#:   ▶ 是唯一的实心向前图形，和 ↻ 的圆弧箭头在轮廓上差别最大。
#:   语义上确实有歧义（▶ 像「播放」、↻ 像「重启」），但那靠**颜色**兜底：
#:   ↻ 悬停变绿，▶ 悬停保持中性灰 BTN_HOVER_BG，两个一眼能分开。
#:   所以 ▶ 不需要新的 hover 颜色常量 —— 绿色留给 ↻ 一家。
CONSOLE_GLYPH = "▶"        # U+25B6 黑色右指向三角形
#: 按钮字形字号（比正文大一号，看得清）
BTN_GLYPH_SIZE = max(1, int(round(14 * SCALE)))
#: 按钮常态前景（浅色，与 #1e1e2e 底有明确对比）
BTN_FG_HI = "#e6edf3"
#: 按钮悬停背景（瞬间切换，不做渐变）
BTN_HOVER_BG = "#585b70"
#: 关闭按钮悬停背景 —— 用红的，"点它 = 关掉" 的语义直接写在颜色里
CLOSE_HOVER_BG = "#f38ba8"

# ── 位置（相对屏幕比例）────────────────────────────────────────────────
#: 默认贴在右半区、纵向对齐小地图中心。
#: 2560x1600 下 ≈ (1485, 136) —— 小地图在游戏右上区域。
#: 硬编码比例而不是自动检测小地图：v1 不做图像识别（见「v1 明确不做」）。
POS_RATIO_X = 0.58
POS_RATIO_Y = 0.085

#: 位置微调步长（像素）
NUDGE_SMALL = 1
NUDGE_LARGE = 10

#: 拖动判定阈值 —— 位移小于这个值算点击，不算拖动。
#:
#: ★ 为什么从 3 提到 5（2026-10-05）
#:   --------------------------------
#:   这个阈值原先只服务一件事：别让「点按钮」被当成拖动。
#:   现在它多了一件**后果严重得多**的事 —— 别让「拖浮窗」被当成点击，
#:   而点击会**停掉用户的任务**。两件事的失败代价完全不对称：
#:
#:     阈值太小（误判为拖）：窗口前几像素不跟手，用户再挪一下就好了。
#:     阈值太大（误判为点）：**任务被停了**。用户以为自己只是挪了浮窗。
#:
#:   同样的偏差，一边是「手感略涩」，另一边是「干活的东西被干掉」。
#:   所以该往**大**偏。
#:
#:   5px 在 100% 缩放下约 1.3mm：比拇指按住时的自然抖动大，
#:   又远小于任何一次有意识的拖拽距离。微调位置另有方向键
#:   （NUDGE_SMALL=1px），不需要靠鼠标拖那几像素。
#:
#: ★ 注意它**不参与** :func:`_recompute`
#:   它是「判定阈值」不是「尺寸」，按物理像素写死，不跟用户缩放倍率走 ——
#:   放大了也不该「更容易被拖误判」。
DRAG_THRESHOLD = 5

# ── 时序 ────────────────────────────────────────────────────────────────
POLL_INTERVAL_MS = 100        # 10Hz 读快照
EVENT_POLL_MS = 100
EVENT_TOAST_MS = 3000         # 事件提示停留 3 秒
PANIC_FLASH_MS = 200


def _recompute() -> None:
    """按当前 :data:`SCALE` 重算**全部**派生常量。

    ★ 单一真相源
    ------------
    这里列的每个名字都是「设计基准值 × SCALE」。模块顶层按同一公式算过
    一遍（那是**初值**，让静态读者/IDE 能看到真实数字），但改缩放的
    唯一入口是 :func:`apply_user_scale`，它会回来调用本函数。

    漏掉任何一个常量 = 留下一个不跟着缩放的尺寸 = 排版错位。
    以后新增尺寸常量时**必须**同时加进这个函数。
    """
    global FONT_SIZE, FONT_SIZE_TITLE, FONT_MONO, BTN_GLYPH_SIZE
    global PANEL_WIDTH, PANEL_PADDING, LINE_HEIGHT
    global BUTTON_SIZE, BUTTON_HIT, CORNER, BTN_MARGIN

    # 字号（pt，排版单位）与尺寸（px，像素）不是一回事，但吃同一个倍率
    FONT_SIZE = max(1, int(round(10 * SCALE)))
    FONT_SIZE_TITLE = max(1, int(round(11 * SCALE)))
    FONT_MONO = ("Consolas", max(1, int(round(10 * SCALE))))
    BTN_GLYPH_SIZE = max(1, int(round(14 * SCALE)))

    PANEL_WIDTH = px(480)
    PANEL_PADDING = px(10)
    LINE_HEIGHT = px(20)
    BUTTON_SIZE = px(26)      # 视觉尺寸
    BUTTON_HIT = px(32)       # 触达区域（视觉外扩，留余量）
    CORNER = px(8)
    BTN_MARGIN = px(6)


def apply_user_scale(multiplier: float) -> float:
    """设置用户缩放倍率并重算全部尺寸。返回**生效后的** :data:`SCALE`。

    :param multiplier: 相对「系统默认大小」的倍率。``1.0`` = 系统默认
        （不做覆盖）；``1.25`` = 大两成半；``0.75`` = 小两成半。
    :returns: ``BASE_SCALE × 用户倍率`` —— 调用方可以拿它确认是否被夹过。

    ★ 必须在 ``Tk()`` 创建**之前**调用
      ---------------------------------
      行高是 :meth:`OverlayApp._compute_metrics` 在 Tk 建好之后**实测**
      字体得到的，本函数改的是字号。在 Tk 之后调虽然也能生效，但那一帧
      已按旧字号排过版，会闪一下错位的布局。放在 Tk 之前最稳。

    ★ 越界**夹到边界**，不抛异常
      --------------------------
      调用方分两类：命令行（自己先校验，给出人话错误后拒绝启动）和
      主控台滑块（构造上就夹在区间内）。真到了这里还越界，说明上游漏了
      —— 夹住并让调用方从返回值看出实际生效值，比炸掉整个浮窗划算。
    """
    global _USER_SCALE, SCALE

    try:
        raw = float(multiplier)
    except (TypeError, ValueError):
        raw = 1.0                     # 不是数字 = 用户没给 = 不覆盖
    if not math.isfinite(raw):
        raw = 1.0                     # NaN / inf 同样按"没给"处理

    _USER_SCALE = max(USER_SCALE_MIN, min(USER_SCALE_MAX, raw))
    SCALE = BASE_SCALE * _USER_SCALE
    _recompute()
    return SCALE


def phase_color(state: str) -> str:
    """状态 -> 主色。未知状态走灰色而不是抛异常（UI 不能因脏数据崩）。"""
    return {
        "RUNNING": ACCENT_RUN,
        "PAUSED": ACCENT_PAUSE,
        "STOPPING": ACCENT_STOP,
        "STOPPED": ACCENT_STOP,
        "IDLE": ACCENT_IDLE,
        "UNKNOWN": DIM,
    }.get(str(state), DIM)


def health_color(health: str) -> str:
    return {
        "OK": ACCENT_RUN,
        "OFFLINE": ACCENT_IDLE,
        "IDLE_LONG": ACCENT_PAUSE,
        "DEGRADED": ACCENT_WARN,
        "CRASHED": ACCENT_ERROR,
    }.get(str(health), DIM)


def severity_color(sev: str) -> str:
    return {
        "INFO": DIM,
        "WARN": ACCENT_WARN,
        "ERROR": ACCENT_ERROR,
    }.get(str(sev), DIM)


__all__ = [
    "ACCENT_ERROR", "ACCENT_IDLE", "ACCENT_PAUSE", "ACCENT_RUN",
    "ACCENT_STOP", "ACCENT_WARN", "BASE_SCALE", "BG", "BORDER", "BTN_BG",
    "BTN_FG", "BTN_FG_HI", "BTN_GLYPH_SIZE", "BTN_HOVER", "BTN_HOVER_BG",
    "BTN_KILL_BG", "BTN_KILL_FG", "BTN_MARGIN", "BUTTON_HIT", "BUTTON_SIZE",
    "CLOSE_GLYPH", "CLOSE_HOVER_BG", "CONSOLE_GLYPH", "CORNER", "DIM",
    "DRAG_THRESHOLD",
    "EVENT_POLL_MS", "EVENT_TOAST_MS", "FG", "FONT_FAMILY", "FONT_MONO",
    "FONT_SIZE", "FONT_SIZE_TITLE", "KILL_GLYPH", "LINE_HEIGHT",
    "NUDGE_LARGE", "NUDGE_SMALL", "PANEL_PADDING", "PANEL_WIDTH",
    "PANIC_FLASH_MS", "POLL_INTERVAL_MS", "POS_RATIO_X", "POS_RATIO_Y",
    "RESTART_GLYPH", "RESTART_HOVER_BG", "SCALE", "USER_SCALE_MAX",
    "USER_SCALE_MIN", "apply_user_scale", "detect_scale", "health_color",
    "phase_color", "px", "severity_color", "user_scale",
]
