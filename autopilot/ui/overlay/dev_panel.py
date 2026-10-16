"""开发者版浮窗：**信息密度优先**，不做「好看的产品版」。

    python -m autopilot.tools.overlay --mode dev

产品版（:mod:`autopilot.ui.overlay.app`）只有 5 行，回答「现在能不能玩」。
开发者版回答的是另一个问题：**链路到底卡在哪一环**。所以它显示三段：

    ┌────────────────────────────────────────────┐
    │ ● 会话:原神 | RUNNING | 12.5Hz | 场景:WORLD │  ← 状态行：一眼看有没有会话
    ├────────────────────────────────────────────┤
    │ [01:22:06] CORE_STARTED    主循环已启动…     │  ← 事件流：最新 N 条
    │ [01:22:06] CAPTURE_FAILED  截图重试 25 次…   │     红/黄/灰按 severity
    │ [01:22:07] CORE_PAUSED     主循环已暂停…     │
    ├────────────────────────────────────────────┤
    │ 意图:3 通道:bitblt 寻路:— 事件:23 条        │  ← 分析摘要：卡在哪个环节
    └────────────────────────────────────────────┘

★ 为什么不改 ``build_panel`` 而是新开一个
  --------------------------------------
  产品版的 5 行高度是**刻意恒定**的（行数固定 => 面板不跳），连注释都
  逐条解释过哪一行不能动。开发者版要的恰恰相反 —— 行数随内容变、面��更宽。
  在同一套控件里塞两种排版，最后两边都别扭。子类化 :class:`OverlayApp`
  只覆写「排版 + tick」两处，窗口/位置/缩放/按钮/拖动全部复用。

★ 为什么轮询反而更慢（500ms vs 100ms）
  --------------------------------------
  产品版 10Hz 是为了「状态一变立刻变色」；开发者版的重头是事件流，
  而事件是**秒级**的 —— 10Hz 轮询只是把同一屏内容重画 10 次。
  500ms 对人眼已经完全跟得上，却把 CPU 占用降到 1/5。

★ 为什么启动时要回灌历史事件
  --------------------------
  :class:`OverlayApp` 构造时调 ``attach_stream_tail(-1)``，**只关心从现在
  起**的事件 —— 对 toast 来说是对的（不该拿一小时前的事弹提示）。
  但调试恰恰相反：开发者打开卡片就是要看「刚才发生了什么」，
  所以这里额外回灌当前流的**尾部 N 条**，之后才转为追尾。

排障：面板不更新 / 怎么认出僵尸 overlay
--------------------------------------
**先看摘要行**，三种状态各有各的处置（都是 :class:`layout.Line` 的首行前缀）：

  ``⚠N秒无新事件（引擎可能已停）``
      overlay 活着，是**引擎**停了。2026-10-10 13:42 真机就是这样：
      引擎 ``STOPPED`` 之后再没有新事件，面板永远停在最后一屏 ——
      而"停在最后一屏"和"overlay 卡死"在画面上**一模一样**，不报出来
      就只能靠猜。**处置**：把引擎拉起来，别去动 overlay。

  ``⚠reader异常(...)``
      这是**我们自己的读侧**炸了（文件轮转 / 句柄失效 / JSON 坏行）。
      详情同时 ``logger.exception`` 落到 stderr。**处置**：重启 overlay；
      反复出现说明事件流被换 run 的方式有问题，查 :class:`ChannelReader`。

  摘要行里的 ``run:xxx``
      事件流跟的是**快照里的** run（:class:`ChannelReader` 的 ``_run_id``
      来自 ``StateSnapshot.run_id``），**不是** ``run_current.txt`` ——
      那是 Console 侧写的另一套，两者不一致时**以快照为准**。

**僵尸 overlay 怎么认**
  --------------------
  重复启动会留下多个 overlay，它们**都显示同一块区域**，肉眼只看得见最上面
  那个（``overrideredirect`` + ``-topmost``），而底下那个可能早就死了 ——
  于是你会以为"改了没生效"。识别靠**命令行**，不是靠窗口：

      Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*autopilot.tools.overlay*--mode dev*' } |
        ForEach-Object { "pid=$($_.ProcessId) 启动=$($_.CreationDate)" }

  列出后按 pid 结束；要一次清空：

      Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*autopilot.tools.overlay*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

  ★ 注意：一条 ``.venv\\Scripts\\python.exe -m ...`` 会显示成**两个**进程
    （venv 启动器 + 真实解释器），启动时刻相同 —— 那是**同一个** overlay，
    不要当成两个僵尸去杀。产品版标题是 ``N.E.K.O.``，开发者版是
    ``N.E.K.O. [dev]``，按标题 ``MainWindowTitle`` 过滤也能区分。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from . import layout, theme
from .app import OverlayApp

logger = logging.getLogger(__name__)

#: 事件流占几行。命令行 ``--event-rows`` 可覆盖。
DEV_EVENT_ROWS = 10

#: 开发者版面板宽度的**设计基准像素**。产品版是 480（theme.PANEL_WIDTH
#: 的来源）—— 那是在「别太挡游戏画面」的前提下定的。调试时反过来：
#: 屏幕是玩家的，信息是开发者的，宁可宽一点。
#:
#: ★ 存**基准值**而不是 ``theme.px(760)`` 的结果
#:   ------------------------------------------------
#:   ``theme.apply_user_scale()`` 会在 Tk 之前重算 ``SCALE``。模块导入
#:   那一刻算出来的像素值会停在导入时的缩放上，后面 scale 怎么改都不动 ——
#:   第一版就是这么把 760 画成了 360 的（用户倍率 0.5 × 基准 1.5）。
DEV_PANEL_WIDTH_UNITS = 760

#: 事件流刷新间隔。**比产品版慢**（见模块 docstring）。
DEV_POLL_INTERVAL_MS = 500

#: 启动回灌时最多读多少字节的流尾部。事件行约 200 字节，
#: 128KB 足够几千条；而整文件读一遍在长跑之后可能上百 MB。
_TAIL_READ_BYTES = 128 * 1024


# --------------------------------------------------------------------------
# 文本裁剪
# --------------------------------------------------------------------------

def display_width(s: str) -> int:
    """按「CJK 算 2 列、其余算 1 列」估宽。

    ★ 为什么不直接 ``len(s)``
      ----------------------
      一行里时间戳/事件码是 ASCII（窄），message 多半是中文（宽）。
      用 ``len()`` 裁会把中文那一半砍掉一大截，而真机画面里截掉的
      恰恰是**最有信息量的那句话**。
    """
    return sum(2 if ord(ch) > 0x2000 else 1 for ch in s)


def fit(s: str, budget: int) -> str:
    """把 ``s`` 裁到 ``budget`` 列以内（超出加省略号）。

    ★ 预留位要按 ``display_width("…")`` 算，不能写死 1
      ---------------------------------------------------
      ``…``(U+2026) 落在 ``> 0x2000`` 那一侧，被 :func:`display_width`
      算成 **2 列**。写死预留 1 会让结果超预算 1 列 —— 而这一列
      恰恰是最容易撞到右边按钮/边界的地方。
    """
    ell = "…"
    ell_w = display_width(ell)
    if display_width(s) <= budget:
        return s
    limit = max(0, budget - ell_w)
    out: list[str] = []
    used = 0
    for ch in s:
        w = 2 if ord(ch) > 0x2000 else 1
        if used + w > limit:
            break
        out.append(ch)
        used += w
    return "".join(out) + ell


# --------------------------------------------------------------------------
# 事件缓冲
# --------------------------------------------------------------------------

def load_recent_events(limit: int, base_dir: Any = None,
                       run_id: str | None = None) -> list[Any]:
    """读**指定 run** 的流尾部 ``limit`` 条。读不到就返回空列表。

    :param run_id: 要读哪一轮。``None`` 才退回读 ``run_current.txt``；
        调用方一律传**快照里的** run_id —— 理由见函数体内那段注释。

    只 seek 末尾 :data:`_TAIL_READ_BYTES` 字节 —— 长跑之后整个流可能很大，
    而人眼能看的永远只有最后几十条。

    解析不了的行**整行丢掉**而不是中断：一条坏行不该让调试面板变瞎。
    """
    from pydantic import ValidationError  # noqa: PLC0415

    from autopilot.ui.protocol import RUN_ID_NAME, stream_path  # noqa: PLC0415
    from autopilot.ui.schema import StateEvent  # noqa: PLC0415

    run_dir = layout.POS_FILE.parent
    if base_dir is not None:
        from autopilot.ui.protocol import run_dir as _rd  # noqa: PLC0415

        run_dir = _rd(base_dir)
    # ★ run_id 由调用方给（快照里的那个），**不是** run_current.txt。
    #   两个是不同的真相来源：ChannelReader 追 StateSnapshot.run_id（引擎写的），
    #   run_current.txt 是 Console 侧另写的一套。真机 2026-10-10 出现过不一致
    #   （快照 20261010_125648 / 指针 20261010_131350）—— 那时启动回灌从指针读、
    #   之后追尾跟快照，**两轮事件混在同一屏**，切轮提示还报个错的 run。
    #   回灌和追尾必须同源，所以这里默认用快照的 run_id；显式传空才退回指针。
    if not run_id:
        try:
            run_id = (run_dir / RUN_ID_NAME).read_text(encoding="utf-8").strip()
        except OSError:
            return []
    if not run_id:
        return []

    path = stream_path(run_dir, run_id)
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > _TAIL_READ_BYTES:
                fh.seek(size - _TAIL_READ_BYTES)
                fh.readline()          # 丢掉可能被切断的半行
            raw = fh.read()
    except OSError:
        return []

    out: list[Any] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(StateEvent.model_validate_json(line))
        except ValidationError as exc:
            logger.debug("事件行非法（跳过）：%s", exc)

    # ★ 按 ts_ms **降序**返回（最新在前），不靠文件顺序。
    #   文件顺序 = 追加顺序 = 通常也是时间顺序，所以只取 tail 时两者一致；
    #   但只要发生过「补写」「跨 run」「时钟回拨」，文件顺序就未必等于
    #   时间顺序 —— 而用户看到的**就是时间**。排序只在这一处做一次。
    out.sort(key=lambda e: int(getattr(e, "ts_ms", 0) or 0), reverse=True)
    return out[:limit]


class EventFeed:
    """最近 N 条事件的环形缓冲，**最新在最前**。

    ★ 为什么倒序
      ----------
      正序（最旧在上）看着像日志，但在「盯实时」的场景下有两个问题：
      新事件排在最后，用户得扫到最后一行才知道刚才发生了什么；
      而「旧事件被顶掉」这件事在正序下也完全看不出来——新的一条悄悄
      出现在末尾，注意力不在那儿就永远错过。倒序让**顶掉**变成可见的。

    :meth:`push` 把新事件**插到队首**（而不是追加到队尾）。

    暂停由调用方决定：不调 :meth:`push` 即可，缓冲内容原封不动。
    """

    def __init__(self, capacity: int = DEV_EVENT_ROWS) -> None:
        self._dq: deque = deque(maxlen=max(1, int(capacity)))
        #: 最新一条进来的墙钟时刻（``time.time()``）。用来做「新事件高亮」——
        #: 高亮的是**收到**的时刻，不是事件自带的 ``ts_ms``（那可能因为
        #: 时钟回拨是负数差，算出来的倒计时会乱跳）。
        self._last_push_at = 0.0

    def push(self, events: list) -> None:
        if not events:
            return
        # ★ 入参顺序是「旧 -> 新」（ChannelReader 顺着文件读出来的），
        #   而队首要放最新的，所以**正序**遍历 + 每次 appendleft：
        #     push([a, b, c])  ->  [c, b, a]，c 最新，在队首。
        #
        #   ★ 这里曾经多写了一个 reversed()，于是 appendleft 又把它掰回
        #     原样，最终队首是**最旧**那条 —— 屏幕上"旧事件在顶上"，
        #     看起来就像时间倒流。reversed + appendleft 是两次反向 = 没做。
        for ev in events:
            self._dq.appendleft(ev)
        self._last_push_at = time.time()

    def seconds_since_last(self, now: float | None = None) -> float | None:
        """距离**收到**最近一批事件的秒数。没有事件返回 ``None``。

        ⚠️ 这个只用来判断「高亮期过了没有」，**不能拿去当新鲜度显示**：
        启动回灌那一批是磁盘上的历史事件，收到它们只是"刚刚读进来"，
        事件本身可能是一小时前的。拿它当新鲜度会显示「0秒前」——
        那是在骗人。显示新鲜度用 :meth:`newest_event_age`。
        """
        if not self._last_push_at:
            return None
        return max(0.0, (now if now is not None else time.time())
                   - self._last_push_at)

    def newest_event_age(self, now: float | None = None) -> float | None:
        """队首那条**事件自身**的年龄（秒）。空缓冲 / ts 缺失返回 ``None``。

        队首就是最新的（见类 docstring 的倒序约定），所以只看它一个。
        """
        if not self._dq:
            return None
        ts = int(getattr(self._dq[0], "ts_ms", 0) or 0)
        if ts <= 0:
            return None
        return max(0.0, (now if now is not None else time.time()) - ts / 1000.0)

    def items(self) -> list:
        return list(self._dq)

    def __len__(self) -> int:
        return len(self._dq)


# --------------------------------------------------------------------------
# 渲染模型（不碰 tkinter，可脱 GUI 单测 —— 与 layout 同一原则）
# --------------------------------------------------------------------------

#: 事件码 -> 该不该「重点标」。表外 code 一律用蓝色，天然暴露未知事件。
_SEV_COLOR = {
    "INFO": theme.FG,
    "WARN": theme.ACCENT_WARN,
    "ERROR": theme.ACCENT_ERROR,
}

#: 新到事件的临时高亮色。
#:
#: ★ 为什么单独给一个颜色，而不是复用 severity
#:   ------------------------------------
#:   severity 已经有三个了：INFO 白 / WARN 橙 / ERROR 红。再加一个「蓝」
#:   会跟「表外 code」的 :data:`theme.ACCENT_IDLE` 撞脸，两个不同的意思
#:   用同一个颜色，久了就分不出是"新来的"还是"不认识的事件码"。
#:   粉红跟这三个都不沾，一眼就知道是**时间维度**而不是**严重度维度**。
_FRESH_COLOR = "#f5c2e7"

#: 高亮停留时长（秒）。够扫一眼，又不至于让"新"变成常态。
FRESH_HIGHLIGHT_S = 1.5

#: 连续多少轮没等到新事件就报「可能断流」。
#:
#: ★ 为什么要这个阈值而不是"事件超过 N 秒没来"
#:   ------------------------------
#:   引擎在跑的时候也可能十几秒不写事件（跑长路线、切场景）。按时间判断
#:   会在正常运行时误报，反而会被习惯性忽略。按**连续空转的轮数**判断
#:   只在"真的什么都不来了"时触发，且阈值跟着轮询间隔走，改刷新频率不用
#:   同步改这里。5 轮 × 500ms ≈ 2.5 秒，引擎一停很快就报出来。
STALE_TICK_THRESHOLD = 5


def _clock(ts_ms: int) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(ts_ms / 1000.0))
    except (ValueError, OSError, OverflowError):
        return "--:--:--"


def _enum_text(v: Any, default: str = "?") -> str:
    """枚举取 ``.value``，普通值原样。

    ★ 为什么不能直接 ``str(v)``
      ------------------------
      ``schema.py`` 里是 ``class RuntimePhase(str, Enum)``。Python 3.11+
      对这种混合枚举的 ``str()`` 返回的是 ``"RuntimePhase.STOPPED"``
      而不是 ``"STOPPED"`` —— 第一版开发者版就是这样把状态行写成
      ``RuntimePhase.STOPPED/Health.OK`` 的，看着像变量名不像状态。
    """
    if v is None:
        return default
    val = getattr(v, "value", v)
    s = str(val).strip()
    return s or default


def age_text(seconds: float | None) -> str:
    """把「多少秒前」写成一句人话。超过 1 小时改用分钟/小时。"""
    if seconds is None:
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}秒前"
    if s < 3600:
        return f"{s // 60}分前"
    if s < 86400:
        return f"{s // 3600}小时前"
    return f"{s // 86400}天前"


def _status_line(snap: Any, session: dict | None, budget: int,
                 fresh_age: float | None = None) -> layout.Line:
    """第 1 行：会话 + 引擎 + 频率 + 场景 + **最新事件多久之前**。

    新鲜度放在这里而不是每条事件后面，是因为「**这一屏有多旧**」是一个
    全局判断，而屏幕宽度是稀缺的 —— 10 条行各挂一个「N 分钟前」会把真正
    的消息文本挤成省略号。

    ★ 字段顺序不是随便排的
      --------------------
      状态行右边被按钮吃掉一截（见 budget），小缩放下会被裁。所以按
      「**被裁掉时最不能丢**」排序：**会话**（今天第一个会断的地方）和
      **最新事件年龄**（用户真正在盯的东西）排最前，Hz 和场景垫后 ——
      先被裁掉的应该是信息密度最低的那一段。

    :param budget: 这一行专属的显示列预算。**比下面那些行小** ——
        右上角那排按钮就压在第 1 行上，产品版靠「5 行文本都短到够不着
        按钮列」侥幸不显形（app.py:214 有整段说明）。开发者版的状态行
        很长，不给它单独缩预算就会直接盖在按钮上。
    :param fresh_age: 最新事件的年龄（秒）。``None`` = 还没有任何事件。
    """
    if session and bool(session.get("active")):
        name = str(session.get("display_name")
                   or session.get("game_id") or "会话").strip()
        dot = theme.ACCENT_RUN
    else:
        name, dot = "无会话", theme.ACCENT_WARN

    state = _enum_text(getattr(snap, "state", None))
    health = _enum_text(getattr(snap, "health", None))
    hz = float(getattr(snap, "tick_hz", 0.0) or 0.0)
    scene = _enum_text(getattr(snap, "scene", None))

    # 新鲜度只出文字、不另上色：layout.Line 一个 Label 只有一种 fg，
    # 想把「最新:18分钟前」单独染成橙色就得再开一个 Label 拼同一行，
    # 而「18分钟前」四个字本身已经把意思说全了 —— 不值当。
    age = age_text(fresh_age)

    text = fit(f"● 会话:{name}  最新:{age}  {state}/{health}  {hz:.1f}Hz"
               f"  场景:{scene}", budget)
    return layout.Line(text=text, color=theme.FG, accent=dot)


def _event_line(ev: Any, budget: int, fresh: bool = False) -> layout.Line:
    """一行事件。``fresh=True`` 时用 :data:`_FRESH_COLOR` 高亮。"""
    stamp = _clock(int(getattr(ev, "ts_ms", 0) or 0))
    code = str(getattr(ev, "code", "") or "?")
    msg = str(getattr(ev, "message", "") or "")
    sev = str(getattr(ev, "severity", "INFO") or "INFO")
    known = bool(getattr(ev, "known", True))

    head = f"[{stamp}] {code}"
    line_budget = max(8, budget - display_width(head) - 2)
    text = f"{head}  {fit(msg, line_budget)}"

    if fresh:
        return layout.Line(text=text, color=_FRESH_COLOR)

    color = _SEV_COLOR.get(sev, theme.FG)
    # 表外 code 用蓝色 —— 一眼看出「这个事件 code 读侧还不认识」，
    # 比默默当 INFO 处理有用得多（schema.py 明确要求读侧不许崩）。
    if not known:
        color = theme.ACCENT_IDLE
    return layout.Line(text=text, color=color)


def _summary_line(snap: Any, nav: dict | None, feed: EventFeed,
                  run_id: str, budget: int, paused: bool = False,
                  stale_ticks: int = 0, reader_error: str = "",
                  session: dict | None = None,
                  now: float | None = None) -> layout.Line:
    """最后一行：**自诊断 + 分析摘要**。这条是「卡在哪了」的总答。

    ★ 为什么诊断信息放这一行而不是状态行
      ------------------------------------
      状态行右边被按钮吃掉一截（见 :func:`_status_line` 的 budget），
      小缩放下只剩 30 来列 —— 塞进 run/条数/异常会立刻被裁掉。
      这一行是全宽的，而且它本来就是「这一屏都怎么了」的总览。
    """
    intent = getattr(snap, "intent", None)
    cnt = int(getattr(intent, "count", 0) or 0)
    streak = int(getattr(intent, "idle_streak", 0) or 0)
    cap = getattr(snap, "capture", None)
    ch = str(getattr(cap, "active", "") or "—")
    deg = str(getattr(cap, "degraded_from", "") or "")

    if nav:
        nav_txt = str(nav.get("phase") or nav.get("state") or "寻路中")
        hint = str(nav.get("hint") or "")
        if hint:
            nav_txt = f"{nav_txt}/{hint}"
    else:
        nav_txt = "寻路:—"

    parts = [f"意图:{cnt}"]
    if streak:
        parts.append(f"空转:{streak}")
    parts.append(f"通道:{ch}" + (f"←{deg}" if deg else ""))
    parts.append(nav_txt)
    parts.append(f"事件:{len(feed)}")
    parts.append(f"run:{run_id or '—'}")
    # 探测反馈已经独占一行（见 build_dev_panel），这里**不再重复**。
    if paused:
        parts.insert(0, "⏸已暂停")

    age = feed.newest_event_age(now)
    if age is not None:
        parts.append(f"最新:{age_text(age)}")

    # 颜色优先级：reader 异常 > 断流 > 暂停 > 平时。
    # 越往上越"和我们自己的读侧有关"，越要显眼。
    color = theme.DIM
    if reader_error:
        parts.insert(0, f"⚠reader异常({fit(reader_error, 26)})")
        color = theme.ACCENT_ERROR
    elif stale_ticks >= STALE_TICK_THRESHOLD:
        secs = int(stale_ticks * DEV_POLL_INTERVAL_MS / 1000)
        parts.insert(0, f"⚠{secs}秒无新事件（引擎可能已停）")
        color = theme.ACCENT_WARN
    elif paused:
        color = theme.ACCENT_WARN
    return layout.Line(text=fit("  ".join(parts), budget), color=color)


def build_dev_panel(snap: Any, *, events: list, nav: dict | None = None,
                    session: dict | None = None, feed: EventFeed | None = None,
                    run_id: str = "", row_width: int = 60,
                    status_width: int | None = None,
                    fresh_count: int = 0,
                    paused: bool = False,
                    stale_ticks: int = 0,
                    reader_error: str = "") -> layout.PanelModel:
    """把快照 + 事件流变成开发者版的 N 行。

    :param row_width: 每行的显示列预算（:func:`display_width` 口径）。
        由 app 层按面板实际宽度算好传进来 —— 这里不碰 theme/tkinter，
        所以能脱 GUI 单测。
    :param status_width: 状态行**专属**预算，缺省与 ``row_width`` 相同。
        app 层会把它压到「不撞右上角按钮」的宽度。
    :param fresh_count: 队首有几条处于高亮期（刚收到）。只影响前 N 行的颜色。
    :param paused: 事件流是否冻结。冻结时摘要行前置「⏸已暂停」。
    :param stale_ticks: 连续多少轮没等到新事件。超阈值摘要行报「可能断流」。
    :param reader_error: 读侧异常文案。非空时摘要行前置「⚠reader异常」，
        并写 stderr 日志 —— 面板上看得见、日志里查得到。
    """
    feed = feed if feed is not None else EventFeed(len(events) or 1)
    lines: list[layout.Line] = [
        _status_line(snap, session,
                     row_width if status_width is None else status_width,
                     fresh_age=feed.newest_event_age())
    ]

    # 探测反馈**独占一行**，不挤进摘要行。
    # ★ 为什么不给它挤摘要行
    #   ------------------
    #   摘要行本来就满：断流告警 + 意图 + 通道 + 寻路 + 事件数 + run + 新鲜度。
    #   实测把「探测:第9次 · 注册失败 · 原因」再塞进去，行尾就被截成
    #   「… 通…」，最该看的那半句反而没了。探测是排查会话时**第一眼**
    #   要看的东西，给它整行、全宽。
    probe_txt, probe_color = layout.probe_status(session)
    lines.append(layout.Line(text=fit(probe_txt, row_width), color=probe_color))

    for idx, ev in enumerate(events):
        lines.append(_event_line(ev, row_width, fresh=(idx < fresh_count)))

    lines.append(_summary_line(snap, nav, feed, run_id, row_width,
                                paused=paused, stale_ticks=stale_ticks,
                                reader_error=reader_error, session=session))
    return layout.PanelModel(lines=lines, height=len(lines))


# --------------------------------------------------------------------------
# 窗口
# --------------------------------------------------------------------------

class DevOverlayApp(OverlayApp):
    """开发者版窗口。**复用父类的窗口/位置/缩放/按钮/拖动**，只换排版与 tick。"""

    #: 轮询比产品版慢 —— 事件是秒级的，10Hz 只是重画同一屏（见模块 docstring）
    poll_interval_ms = DEV_POLL_INTERVAL_MS

    #: 事件流占几行，由 :attr:`_event_rows` 在 __init__ 里定；
    #: 总正文行数 = 事件行 + 状态行 + 摘要行。
    panel_width_units = DEV_PANEL_WIDTH_UNITS

    def __init__(self, reader, *, event_rows: int = DEV_EVENT_ROWS,
                 panel_width_units: float = DEV_PANEL_WIDTH_UNITS, **kw) -> None:
        self._event_rows = max(1, int(event_rows))
        self._feed = EventFeed(self._event_rows)
        # ★ 先定 text_rows 再 super().__init__()：父类构造里第一件事就是
        #   _compute_metrics()，它按 text_rows 算面板高。
        #   （宽度不用这么做 —— panel_width_units 是类属性，每次 geometry
        #   都现算，theme.apply_user_scale() 改完 SCALE 自动跟上。）
        self.text_rows = self._event_rows + 3
        # 状态行 + **探测行** + 事件行 + 摘要行。
        # 探测行是 2026-10-10 加的：插件侧那个 5s 探针的状态与引擎无关，
        # 却是排查会话时第一眼要看的东西 —— 挤进摘要行会被「断流告警 +
        # 意图 + 通道 + …」顶到行尾截断，实测只剩省略号。独占一行才看得全。
        super().__init__(reader, **kw)
        self.root.title("N.E.K.O. [dev]")
        # 回灌尾部事件 —— 调试要看「刚才发生了什么」，而不是从打开这一刻
        # 起才开始记（父类 attach_stream_tail 正是后者）。
        # ★ run_id 传快照里那个：父类构造时已经 read_snapshot() 过，
        #   reader.run_id() 此时就是 StateSnapshot.run_id，与后面追尾同源。
        self._feed.push(load_recent_events(self._event_rows, self.reader.dir,
                                            run_id=self.reader.run_id()))
        # ★ 空格暂停。绑在 toplevel 上（而不是某个 Label），键盘焦点在
        #   面板任意位置都收得到 —— overrideredirect 窗口的焦点状态本来
        #   就不稳，只绑子控件等于让这个键时灵时不灵。
        self._paused = False
        self.root.bind("<KeyPress-space>", self._toggle_pause)
        self.root.bind("<KeyPress>", self._on_dev_key)

        # -- 自诊断状态（2026-10-10）----
        self._seen_run_id = str(self.reader.run_id() or "")
        self._stale_ticks = 0
        self._reader_error = ""

    # ------------------------------------------------------------------
    def _poll_safely(self) -> list:
        """读事件。**抛异常不外泄**，但一定留痕（返回空列表继续跑）。

        ★ 为什么要包一层
          ----------------
          ``poll_events()`` 内部要 stat + seek + 逐行 json 校验，任何一步
          出问题（文件被轮转、句柄失效、JSON 坏在半行）都可能抛。
          而它在 ``tick()`` 里被调用 —— 一抛就是整条 after 链断掉、
          面板变砖。所以这里把异常收成一条 ``_reader_error`` 打到面板上，
          **同时** logger.exception 落 stderr：面板上看得见，日志里查得到。

        ★ 为什么不给它重试/降级
          -------------------
          读失败最常见的原因是"引擎停了、文件不再变"，那本来就不该重试；
          真需要重新初始化时，下一 tick 自然会重来（reader 内部对
          ``run_id`` 变化有从头跟的逻辑）。
        """
        try:
            self._reader_error = ""
            return self.reader.poll_events()
        except Exception as exc:                     # noqa: BLE001
            self._reader_error = f"{type(exc).__name__}: {exc}"
            logger.exception("事件流读取失败（已跳过该帧）：%s", exc)
            return []

    # ------------------------------------------------------------------
    def _toggle_pause(self, _ev=None) -> str:
        """空格：冻结/恢复事件滚动。

        ★ 暂停的是**读**，不是显示
          ----------------------
          这里只拦掉 ``poll_events()`` —— 事件继续在磁盘上追加、引擎照跑，
          我们只是不再往缓冲里塞。所以恢复之后缓冲**仍然是完整的**，
          没有丢事件；而不是"暂停期间攒着，恢复时一次全灌进来"
          （那会一次性把用户刚看懂的那屏冲掉，正是要解决的问题）。
        """
        self._paused = not self._paused
        self._last_key = None            # 强制重画（状态行要变）
        return "break"

    def _on_dev_key(self, ev) -> str | None:
        """开发者版额外的快捷键。空格之外一律**放行**给父类
        （Esc 关闭 / 方向键微调），别把既有键位吃掉。"""
        if ev.keysym == "p" or ev.keysym == "P":
            return self._toggle_pause()
        return None                      # 不 break -> 事件继续传给 Interaction

    # ------------------------------------------------------------------
    def tick(self) -> None:
        """一次刷新。事件进环形缓冲，其余复用父类的快照/定位逻辑。"""
        # ★ 暂停时**照样 poll**，只是不往缓冲里放。
        #   poll_events() 是「读文件 + 推进游标」，游标必须一直往前走，
        #   否则恢复时会一次性重放暂停期间的全部事件，把用户刚看懂的那屏
        #   冲掉 —— 那正是加暂停要解决的问题。丢进黑洞的是"显示"，不是"读"。
        new_events = self._poll_safely()
        if not self._paused:
            self._feed.push(new_events)

        now = time.monotonic()
        snap = self._current_snapshot()
        panic = now < self._panic_until
        toast = self._toast if now < self._toast_until else None

        # ★ run_id 跟着**快照**走，不是 run_current.txt。
        #   ChannelReader._run_id 来自 StateSnapshot.run_id，而引擎换 run 时
        #   先写新的快照再写事件 —— 所以这里拿到的就是活着的那个 run。
        #   run_current.txt 是另一套（Console 侧）写的，两者不一致时**以快照为准**。
        run_id = str(self.reader.run_id() or "")
        if run_id != self._seen_run_id:
            if self._seen_run_id:
                logger.info("事件流 run 切换：%s -> %s", self._seen_run_id, run_id)
                self._toast = f"已切换到新一轮 {run_id}"
                self._toast_color = theme.ACCENT_RUN
                self._toast_until = now + theme.EVENT_TOAST_MS
            self._seen_run_id = run_id
            self._stale_ticks = 0          # 换 run 重新开始计"没新事件"
            self._reader_error = ""
            self._feed.push(load_recent_events(self._event_rows, self.reader.dir,
                                            run_id=run_id))
            self._last_key = None

        # 连续 N 轮没有新事件 -> 大概率引擎已经停了。
        # 引擎停了**不会**再有任何事件，面板就会永远停在最后一屏，
        # 而"停在最后一屏"和"overlay 卡死了"在画面上**一模一样**。
        # 不说的话用户只能靠猜 —— 2026-10-13:48 真机就是这么误判的。
        if new_events:
            self._stale_ticks = 0
        elif not self._paused:
            self._stale_ticks += 1

        cap = getattr(snap, "capture", None)
        ch = str(getattr(cap, "active", "") or "")
        if ch:
            self._last_channel = ch

        since = self._feed.seconds_since_last()
        fresh = 0
        if not self._paused and since is not None and since < FRESH_HIGHLIGHT_S:
            fresh = max(1, min(len(new_events), self._event_rows))

        model = build_dev_panel(
            snap,
            events=self._feed.items(),
            nav=layout.load_nav_state(),
            session=layout.load_session_state(),
            feed=self._feed,
            run_id=run_id,
            row_width=self._row_budget(),
            status_width=self._status_budget(),
            fresh_count=fresh,
            paused=self._paused,
            stale_ticks=self._stale_ticks,
            reader_error=self._reader_error,
        )
        model.toast = toast
        model.toast_color = theme.ACCENT_WARN
        model.panic = panic

        key = model.content_key()
        if key == self._last_key:
            return
        self._last_key = key

        border = theme.ACCENT_ERROR if panic else theme.BORDER
        self.root.configure(bg=border)
        # ★ 行数按需补齐而不是写死 —— parent 只建了 _text_rows 个，
        #   这里刚好也是 _text_rows，zip 不会截断；写死反而会在
        #   事件条数少于行数时留下空白 Label。
        for lbl, line in zip(self._lines, model.lines):
            lbl.configure(text=line.text, fg=line.color)
        for lbl in list(self._lines)[len(model.lines):]:
            lbl.configure(text="")            # 多余的行清空
        self._toast_lbl.configure(text=model.toast or "", fg=model.toast_color)
        for lbl in list(self._lines) + [self._toast_lbl]:
            lbl.configure(bg=border)
        self.root.geometry(f"{self.panel_width}x{self._panel_h}"
                           f"+{self._get_pos()[0]}+{self._get_pos()[1]}")

    def run(self) -> int:
        # ★ 下一轮**先排上**，再处理本轮。
        #   2026-10-10 实测踩过：``tick()`` 抛了一次
        #   （build_dev_panel 少接一个关键字参数），而原来是
        #   ``tick()`` 跑完才 ``after(...)`` —— 于是这一抛把整条链掐断，
        #   面板变成**一块永远不刷新的空白板**，没有任何报错、没有 traceback。
        #   同文件 :_poll_live_scale 的注释早写过这条（"顺序反了就会把
        #   整条 after 链掐断"），这里就是它说的那个坑。
        def loop() -> None:
            self.root.after(self.poll_interval_ms, loop)
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001
                # 一帧失败不该让面板变砖。吞掉但**留下痕迹** ——
                # 静默续跑等于把同样的错再犯几十次还没人知道。
                logger.exception("dev 面板刷新失败（已跳过该帧）: %s", exc)

        self.root.after(self.poll_interval_ms, loop)
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

    def _row_budget(self) -> int:
        """每行能放多少显示列。**按实测宽度算**，不写死。"""
        return self._cols_between(self._text_left, self.panel_width)

    def _status_budget(self) -> int:
        """状态行专属预算：右边界让开右上角那排按钮。

        ★ 为什么只有第 1 行要缩
          ----------------------
          按钮贴的是面板**右上角**，只压第 1 行；事件行从第 2 行起都在
          按钮下面，横向可以用满。app.py:214-229 有一整段说明产品版
          为什么一直没发现这个重叠 —— 它的 5 行文本都短到够不着按钮列。
          开发者版的状态行很长，「靠文本够短」这条侥幸就不成立了。
        """
        return self._cols_between(self._text_left,
                                  self.panel_width - self._button_column())

    @staticmethod
    def _button_column() -> int:
        """右上角按钮区占的物理像素宽。

        与 :mod:`interaction` 的按钮排布约定一致（最左按钮 x = 宽 - 146），
        再留 4px 余量。这里保守取大一点，宁可状态行短一点也不许压按钮。
        """
        return theme.px(150)

    def _cols_between(self, left: int, right: int) -> int:
        """一段物理宽度折算成显示列数。

        中文按 13px、ASCII 按 6.5px 估；留 2% 余量给 Tk 的内边距。
        这个估算不精确，但只在「会不会撞按钮」这件事上起保守作用 ——
        估多了顶多少显示几个字，估少了会压到按钮，所以取偏保守的一侧。
        """
        usable = max(0, right - left - theme.PANEL_PADDING)
        return max(20, int(usable / 7.5 * 0.98))


__all__ = [
    "DEV_EVENT_ROWS",
    "DEV_PANEL_WIDTH_UNITS",
    "DEV_POLL_INTERVAL_MS",
    "DevOverlayApp",
    "EventFeed",
    "build_dev_panel",
    "display_width",
    "fit",
    "load_recent_events",
]