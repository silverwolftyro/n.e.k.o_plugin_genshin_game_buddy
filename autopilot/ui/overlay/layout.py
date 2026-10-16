"""布局：窗口位置 / 尺寸，以及把 StateSnapshot 变成 5 行文本。

**渲染与计算分离**：这一层只产出"该画什么"，不碰 tkinter 控件。
所以它可以脱离 GUI 单测（overlay 的排版错位是最难在实机上看出来的）。

位置持久化写 ``config/overlay_pos.json``，**不写 defaults.toml** ——
后者是 schema 的一部分，往里塞 UI 私有状态会让配置校验和 UI 耦合。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import theme

logger = logging.getLogger(__name__)

#: 位置保存文件（独立于 defaults.toml）
POS_FILE = (Path(__file__).resolve().parents[2] / "_runtime" / "overlay_pos.json")

#: ★ 寻路状态文件（2026-10-04）。由 ``e2e_pathfind.py`` 写，overlay 只读。
#: 与 :data:`POS_FILE` 同目录 —— 「运行期状态都在 _runtime」这条不变。
NAV_STATE_FILE = (Path(__file__).resolve().parents[2]
                  / "_runtime" / "pathfind_state.json")

#: 停止信号文件。overlay 点浮窗就写它，``e2e_pathfind.py`` 主循环每帧看它。
#:
#: ★ 为什么是「文件」而不是信号/管道
#:   ------------------------------
#:   两边是**互不 import 的独立进程**，还可能跑在不同的 Python 环境里。
#:   文件是唯一不需要握手、进程死一个另一个照常工作的通道 —— 这也正是
#:   :mod:`console_launcher` 那套 ``QLocalServer`` 要解决的问题，而这里
#:   连主控台都不经过。零字节、单文件、无需回收。
#:
#: ★ 为什么是「存在即停」而不是读内容
#:   ----------------------------
#:   内容毫无信息量（写什么都一样），而**「文件存在」本身就是全部语义**。
#:   多写内容只会带来「读到了半截」这种没必要的问题。
#:
#: ★ 残留怎么办
#:   ----------
#:   写方（overlay）**先删再建**，保证 mtime 一定是新的；读方
#:   （e2e）**启动时先删一次**，所以上一次跑剩下来的文件不会让这一次
#:   一启动就秒停。两条规则缺一不可 —— 只做前一条会「上次停过的这次起不来」，
#:   只做后一条会「上次停过的这次一起来就停」。
NAV_STOP_FILE = (Path(__file__).resolve().parents[2]
                 / "_runtime" / "nav_stop.flag")

#: **活跃态**的最长有效期（秒）。超过就当没有。
#:
#: ★ 为什么必须有它
#:   ``e2e_pathfind.py`` 崩了 / 被 Ctrl+C / 被任务管理器杀掉时都来不及
#:   清理文件。没有时效判断，overlay 会**永远**显示几分钟前的
#:   「● 寻路中」—— 用户据此判断「我明明停了怎么还在跑」，比不显示更糟。
NAV_STATE_MAX_AGE_S = 5.0

#: **终态**的最长有效期（秒）。
#:
#: ★ 为什么终态要另开一档，而且给到 5 分钟
#:   -------------------------------------
#:   活跃态是「**状态**」—— 写进程死了它就该消失，否则是在骗人。
#:   终态是「**结果**」—— 脚本已经退出了，没有进程会再来刷新它。
#:   真机上跑出 arrive、用户把目光移回屏幕，也就两三秒的事；一个
#:   5 秒的档会让「● 已到达」**闪一下就消失**，用户根本没看清。
#:   结果已经确定了，它不会再变，凭什么 5 秒就过期。
#:
#:   代价只有一个：崩在寻路途中时，浮窗最多多显示 5 分钟的「● 寻路中」。
#:   拿一个「看得见的结果」换「多显示一会儿的已知假象」，这笔交易划算。
NAV_STATE_FINAL_MAX_AGE_S = 300.0

#: 哪些 state 算「终态」。**新增终态必须同步加进来** —— 漏加的后果是
#: 静默退回 5 秒档，不报错、只是「已到达」又变成一闪而过。
NAV_FINAL_STATES = frozenset({"arrived", "panic"})

#: 寻路状态文件 schema 版本。**对不上就不显示**（见 :func:`load_nav_state`）。
#: 写方是 ``e2e_pathfind.py``，它不 import 本模块 —— 改版本号时两边都要改。
#:
#: ★ v2 = 加了 ``phase`` / ``hint`` / ``eta_s``（提前展现意图）
NAV_SCHEMA_VERSION = 2


#: ★ 会话状态文件（2026-10-10）。由**插件**（N.E.K.O 侧 B 轨）写，overlay 只读。
#: 与 :data:`NAV_STATE_FILE` 同目录 ——「运行期状态都在 _runtime」这条不变。
#:
#: ★ 为什么**不**塞进 state_snapshot.json
#:   那个文件归**引擎**（run_loop 经 ChannelWriter）所有，而「有没有会话」
#:   是 N.E.K.O 插件侧的事：插件不跑引擎、引擎也不认识 N.E.K.O 的会话。
#:   混写会变成两边抢同一个文件，且 StateSnapshot 的字段是引擎形状的
#:   （tick_hz / capture / intent），塞进去既不属于它也会破坏 schema 契约。
#:   ——和 pathfind_state.json 是同一个道理：独立文件 + 各自拥有者。
SESSION_STATE_FILE = (Path(__file__).resolve().parents[2]
                      / "_runtime" / "session_state.json")

#: 会话状态 schema 版本。写方是插件 ``__init__.py``，它不 import 本模块
#: ——和 :data:`NAV_SCHEMA_VERSION` 一样，两边都要改，改名就靠版本号挡住。
SESSION_SCHEMA_VERSION = 1

#: 会话状态的最长有效期（秒）。
#:
#: ★ 与 :data:`NAV_STATE_MAX_AGE_S` 的理由不同
#:   寻路是**每帧**写，所以文件很新；会话是**变化时**写、之后靠心跳续命，
#:   一份「已建立」的文件可能要放几十分钟。所以这里取一个长 TTL，并让插件
#:   每 :data:`SESSION_HEARTBEAT_S` 秒续写一次 —— 插件崩了/被杀，
#:   overlay 最迟在这个 TTL 内自己把「已建立」收回成「未建立」，
#:   不会一直拿一份陈旧文件骗人。
SESSION_STATE_MAX_AGE_S = 90.0

#: 插件侧的续写间隔。与上面的 TTL 配套（3 倍余量，容忍偶发延迟）。
SESSION_HEARTBEAT_S = 30.0

#: 面板里要不要显示 Hz / P95 这两个性能数字（2026-10-16，B2）。
#:
#: ★ 为什么默认关
#:   它们是给开发者看的。观众看到「4.8 Hz · 223ms P95」只会得到一个
#:   模糊的印象：这数字是好还是坏？解释成本远大于信息量。
#:   而它们原本挤在**第一行**（最显眼的位置），把「会话 · 状态」压得
#:   只剩半行 —— 真正该一眼看到的东西被两个诊断数字挤掉了。
#:
#: ★ 为什么走环境变量而不是配置项
#:   这是**排障时临时打开**的开关，不是用户要配置的东西。
#:   放配置文件里等于承诺「这是个正经设置项」，而它其实不是。
#:   设 ``NEKO_OVERLAY_METRICS=1`` 即可临时打开。
SHOW_METRICS = os.environ.get("NEKO_OVERLAY_METRICS", "").strip() not in (
    "", "0", "false", "False", "no", "off")


def load_session_state(max_age_s: float = SESSION_STATE_MAX_AGE_S) -> dict | None:
    """读会话状态。**读不到 / 过期 / 损坏 / 版本不符一律返回 None。**

    None 的语义是「没有可用信息」——overlay 据此显示「无会话」，
    所以**过期必须返回 None 而不是返回陈旧内容**：插件崩了还显示
    「已建立」，用户会以为引擎在跑而它其实没跑。

    和 :func:`load_nav_state` 同样先 stat 后 parse，常态路径零解析。
    """
    try:
        age = time.time() - SESSION_STATE_FILE.stat().st_mtime
    except OSError:
        return None                        # 正常：插件还没写过
    if age > max_age_s:
        return None                        # 过期：写方（插件）没了
    try:
        doc = json.loads(SESSION_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("会话状态暂时读不了，等下一轮：%s", exc)
        return None
    if not isinstance(doc, dict):
        return None
    if int(doc.get("schema_version", 0) or 0) != SESSION_SCHEMA_VERSION:
        logger.debug("会话状态 schema 版本不符，忽略：%r",
                     doc.get("schema_version"))
        return None
    return doc


def load_nav_state(max_age_s: float = NAV_STATE_MAX_AGE_S,
                   final_max_age_s: float = NAV_STATE_FINAL_MAX_AGE_S,
                   ) -> dict | None:
    """读寻路状态。**读不到 / 过期 / 损坏 / 版本不符一律返回 None。**

    调用方据此**切换整个面板**（见 :func:`build_panel`），所以返回
    None 必须是「没有寻路状态」这一个明确语义，不能有第三种情况。

    ★ 有效期**按 state 分档**
      -------------------
      ``arrived`` / ``panic`` 用 :data:`NAV_STATE_FINAL_MAX_AGE_S`，
      其余用 :data:`NAV_STATE_MAX_AGE_S`。理由见那两个常量的说明。

    :param max_age_s: **活跃态**的最长有效期（秒）。
    :param final_max_age_s: **终态**的最长有效期（秒）。
    :returns: 解析好的 dict，或 None。
    """
    try:
        age = time.time() - NAV_STATE_FILE.stat().st_mtime
    except OSError:
        return None                        # 正常：没在寻路
    # ★ 先用**最宽**的那一档粗筛。
    #   比它还老的文件在两档下都必然过期，连 JSON 都不用读 ——
    #   「没在寻路」这条热路径因此仍然只有一次 stat，零解析。
    #   （代价：崩在寻路途中的话，那 5 分钟里每 tick 都会尝试解析一次
    #   几百字节的文件，可忽略；换来的是常态路径不引入任何开销。）
    if age > max(max_age_s, final_max_age_s):
        return None
    try:
        doc = json.loads(NAV_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # 半截 JSON（写方正好在 replace 前的瞬间）等下一轮即可。
        logger.debug("寻路状态暂时读不了，等下一轮：%s", exc)
        return None
    if not isinstance(doc, dict):
        return None
    # schema 版本对不上就**不显示**：宁可少显示，也不能拿旧字段名
    # 去渲染出一堆错的文字。
    if int(doc.get("schema_version", 0) or 0) != NAV_SCHEMA_VERSION:
        logger.debug("寻路状态 schema 版本不符，忽略：%r",
                     doc.get("schema_version"))
        return None
    # ★ 分档必须放在**解析之后**：得先知道 state 才知道该用哪一档。
    ttl = (final_max_age_s
           if str(doc.get("state", "") or "") in NAV_FINAL_STATES
           else max_age_s)
    if age > ttl:
        return None                        # 过期：写进程没了
    return doc


@dataclass
class OverlayPos:
    """overlay 窗口状态（位置 + 透明度 + 用户缩放）。

    ★ 为什么 alpha / scale 会丢
    --------------------------
    早期版本只有 ``x``/``y``，保存点直接构造 ``OverlayPos(x, y)`` ——
    ``alpha`` 落回默认值 0.72，于是**拖一下窗口就把 ``--alpha`` 吃掉了**。
    现在 alpha/scale 都必须由持有真值的 :class:`~.app.OverlayApp` 一起写出，
    保存点不许自己编造半个对象（``Interaction`` 已改走 ``on_persist`` 回调）。

    :param scale: 用户缩放倍率。``None`` = 不知道（老文件 / 没缩过），
        与 0 无关 —— 0 不是合法倍率。
    """

    x: int
    y: int
    alpha: float = 0.72
    scale: float | None = None


def default_pos(screen_w: int, screen_h: int) -> OverlayPos:
    """默认位置：屏幕宽 58%、高 8.5%（小地图中心附近）。"""
    return OverlayPos(x=int(screen_w * theme.POS_RATIO_X),
                      y=int(screen_h * theme.POS_RATIO_Y))


def load_pos() -> OverlayPos | None:
    """读保存的状态。**读不到返回 None**（首次启动 / 文件损坏）。"""
    try:
        data = json.loads(POS_FILE.read_text(encoding="utf-8"))
        # ★ scale 用 get + 类型转换，老文件没这个键就是 None（= 不知道），
        #   不是 0，也不是 1.0 —— 猜一个值会把用户的缩放悄悄重置。
        raw_scale = data.get("scale")
        return OverlayPos(x=int(data["x"]), y=int(data["y"]),
                          alpha=float(data.get("alpha", 0.72)),
                          scale=(None if raw_scale is None
                                 else float(raw_scale)))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_pos(pos: OverlayPos) -> bool:
    """保存窗口状态。**失败只告警不抛** —— 拖一下窗口不该让 overlay 崩。"""
    try:
        POS_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"x": int(pos.x), "y": int(pos.y),
                   "alpha": float(pos.alpha)}
        # scale 是 None 时**不写这个键**，而不是写 null —— 半截字段
        # 比缺字段更难让下游判断「这份数据可不可信」。
        if pos.scale is not None:
            payload["scale"] = float(pos.scale)
        POS_FILE.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8")
        return True
    except OSError as exc:
        logger.warning("保存 overlay 位置失败：%s", exc)
        return False


def clear_pos() -> bool:
    try:
        POS_FILE.unlink()
        return True
    except OSError:
        return False


def resolve_pos(screen_w: int, screen_h: int, *,
                x: int | None = None, y: int | None = None,
                reset: bool = False) -> OverlayPos:
    """决定最终位置：命令行 > 保存的 > 默认比例。"""
    if reset:
        clear_pos()
    saved = load_pos()
    base = saved if (saved and not reset) else default_pos(screen_w, screen_h)
    return OverlayPos(x=base.x if x is None else int(x),
                      y=base.y if y is None else int(y),
                      alpha=base.alpha)


# --------------------------------------------------------------------------
# 枚举取值
# --------------------------------------------------------------------------

#: RuntimePhase -> 中文标签。**只做显示**，判断逻辑一律用枚举值本身。
_PHASE_LABEL: dict[str, str] = {
    "IDLE": "空闲",
    "RUNNING": "运行中",
    "PAUSED": "已暂停",
    "STOPPING": "停止中",
    "STOPPED": "已停止",
}


def _enum_value(raw: Any, default: str = "UNKNOWN") -> str:
    """取枚举的**值**，不是全限定名。

    ★ 这是本文件最容易踩的坑
    ----------------------
    ``RuntimePhase`` 是 ``class X(str, Enum)``。Python 3.11 里 ``str()``
    对这种混型枚举走 ``Enum.__str__``，给出 ``"RuntimePhase.STOPPED"``
    而不是 ``"STOPPED"`` —— 于是：

    * ``_state_text()`` 查表落空，原样显示 "RuntimePhase.STOPPED"
    * ``theme.phase_color()`` 同样落空，颜色退化成灰色
    * ``health == "OFFLINE"`` 永远不成立，离线分支形同虚设

    三处都是同一个根因，在这里统一收口。枚举取 ``.value``、
    普通字符串原样返回、其它类型退回默认值 —— 读侧对脏数据必须宽容。
    """
    if raw is None:
        return default
    value = getattr(raw, "value", None)       # 枚举 -> "STOPPED"
    if isinstance(value, str):
        return value
    if isinstance(raw, str):
        return raw
    return default


# --------------------------------------------------------------------------
# 渲染模型（不碰 tkinter）
# --------------------------------------------------------------------------


@dataclass
class Line:
    """一行文本。``accent`` 非 None 时用主色画前缀圆点。"""

    text: str
    color: str = theme.FG
    accent: str | None = None


@dataclass
class PanelModel:
    """整个面板该显示什么。"""

    lines: list[Line] = field(default_factory=list)
    height: int = 0
    #: 边框是否闪红（急停）
    panic: bool = False
    #: 底部事件提示（None = 不显示）
    toast: str | None = None
    toast_color: str = theme.DIM

    def content_key(self) -> tuple:
        """内容指纹。**只在指纹变化时才重绘** —— 省电的关键。

        没有这个，空转时每 100ms 都会触发一次 tk 文本重画，
        白白消耗 CPU，还可能引起闪烁。
        """
        return (tuple((ln.text, ln.color, ln.accent) for ln in self.lines),
                self.panic, self.toast, self.toast_color)


#: 寻路 state -> (中文标签, 颜色)。未知值走 (原样, DIM) —— UI 不能因脏数据崩。
_NAV_STATE_STYLE: dict[str, tuple[str, str]] = {
    "navigating": ("寻路中", theme.ACCENT_RUN),
    "arrived":    ("已到达", theme.ACCENT_RUN),
    "escaping":   ("脱困中", theme.ACCENT_WARN),
    "stuck":      ("卡住",   theme.ACCENT_WARN),
    "panic":      ("已急停", theme.ACCENT_ERROR),
    "paused":     ("已暂停", theme.ACCENT_PAUSE),
    # 「用户点了浮窗主动停的」。刻意**不进** NAV_FINAL_STATES：
    # 已停止是一次性事件，不该在浮窗上挂 5 分钟（5s 档正好）。
    "stopped":    ("已停止", theme.ACCENT_PAUSE),
    "dry_run":    ("空跑",   theme.ACCENT_IDLE),
}


def _fmt_target(nav: dict) -> str:
    """目标显示：优先名字，没有名字就用坐标。

    ``--walk-forward`` 没有目标名（那是「当前朝往前 30m」），
    这种情况下坐标是唯一能告诉用户「要去哪」的东西。
    """
    name = str(nav.get("target_name") or "").strip()
    if name:
        return name
    pos = nav.get("target_pos") or ()
    try:
        if len(pos) >= 2:
            return f"({float(pos[0]):.0f}, {float(pos[1]):.0f})"
    except (TypeError, ValueError):
        pass
    return "未知"


def _fmt_dist(nav: dict) -> str:
    """距离取整到 10m。

    ★ 为什么取整到 10 而不是显示 20.5
    ---------------------------------
    ``content_key()`` 是**逐帧**比对的省电机制。距离每帧都在变，
    显示到小数点会让指纹每帧都变，10Hz 无条件重绘全部 5 个 Label ——
    正好把 :mod:`app` 那个「省电关键」废掉。
    10m 粒度在寻路尺度（几十~几百米）上足够读，跨档才变一次。
    """
    try:
        d = float(nav.get("dist"))
    except (TypeError, ValueError):
        return "—"
    if d != d:                                    # NaN
        return "—"
    if d < 5.0:
        return "已到位"
    # ★ 不能用 round()：Python 的 round 是**银行家舍入**
    #   （round(2.5)=2、round(3.5)=4、round(4.5)=4），
    #   于是 25.0m 会显示成 20m、35.0m 显示成 40m —— 距离档位
    #   系统性偏移半个档。改用「先加半档再取整」的普通四舍五入。
    return f"{int(d / 10.0 + 0.5) * 10}m"


def _nav_lines(nav: dict, panic_key: str) -> list[Line]:
    """寻路模式的 5 行。**行数与原面板严格相同**（控件数是写死的 5）。

    :param nav: 已通过 :func:`load_nav_state` 校验的 dict（不会是 None）。
    :param panic_key: 急停键名。寻路模式**也要显示** —— 这是唯一的刹车。

    ★ 第 3 行是「**下一步**」，不是「当前在干什么」
      ----------------------------------------------
      v1 那行显示 ``next_action``（右转/左转/直行），那是**执行态**：
      用户看到「右转」时人已经在转了，来不及准备 —— 那是复述日志，
      不是预告。v2 改成写方算好的 ``hint``（「右转中 · 约 20 秒后到达」），
      是**计划态**：人还没转就已经知道要转、还有多久到。
    """
    state = str(nav.get("state", "") or "")
    label, color = _NAV_STATE_STYLE.get(state, (state or "未知", theme.DIM))
    # ★ 整句话由**写方**生成，overlay 一个字都不改。
    #   取整、量化、滞回全在 e2e_pathfind.py 做完 —— 这边只要做字符串
    #   直出，指纹才天然稳定。这里任何一处加工都是每帧变动的来源。
    hint = str(nav.get("hint") or "").strip()
    if not hint:
        # ★ v2 里 hint 必填，但写方若崩在写它之前、或字段缺失，
        #   退回 next_action 仍然说得出一句人话，而不是空着。
        hint = str(nav.get("next_action", "") or "") or "—"
    return [
        Line(f"● {label}", color, color),
        Line(f"目标: {_fmt_target(nav)}", theme.FG),
        Line(f"下一步: {hint}", theme.FG),
        Line(f"距离: {_fmt_dist(nav)}", theme.FG),
        Line(f"急停: {panic_key}", theme.DIM),
    ]


def _session_prefix(session: dict | None) -> tuple[str, str]:
    """会话状态 -> ``(短标签, 颜色)``。

    刻意做得很短：它要塞进第 1 行那一串「状态 · Hz · P95」里，
    再长就把 P95 挤掉了。显示原文用**简称**，完整信息在日志里。

    ★ ``None``（读不到 / 过期）与 ``active=False`` 显示成同一个词，
      是有意的：两者对用户的行动建议完全一样 —— 「去把原神打开」。
      区分它们对 overlay 没有价值，对日志有价值，所以日志里写了。
    """
    if not session:
        return "无会话", theme.ACCENT_WARN
    if not bool(session.get("active")):
        return "无会话", theme.ACCENT_WARN
    name = str(session.get("display_name") or session.get("game_id") or "").strip()
    return (name or "会话"), theme.ACCENT_RUN


#: 探测结论 -> (标签, 颜色)。写侧是插件的 ``_probe_and_activate``，
#: 这里只做显示；**不认识的值原样透出**（不静默吞掉新结果码）。
_PROBE_RESULT_STYLE: dict[str, tuple[str, str]] = {
    "found":           ("已建立", theme.ACCENT_RUN),
    "no_window":       ("未找到窗口", theme.ACCENT_WARN),
    "no_match":        ("找到窗口但不匹配", theme.ACCENT_WARN),
    "register_failed": ("注册失败", theme.ACCENT_ERROR),
    "error":           ("探测异常", theme.ACCENT_ERROR),
}


def probe_status(session: dict | None) -> tuple[str, str]:
    """会话探测反馈 -> ``(短标签, 颜色)``。给开发者版卡片显示。

    ★ 回答的是「探针**在不在跑**、上次结论是什么」
      ------------------------------------------------
      产品版第 1 行只说「有无会话」，看不出后台那个 5s 一次的探针
      到底活着没有。用户在游戏画面上看到「第 12 次 · 未找到窗口」
      就知道「它一直在找，只是原神没开」——而不用切到 PowerShell 翻日志。

    ★ 颜色语义
      ----------
      绿 = 已建立；橙 = 还在等（没窗口 / 窗口不匹配）；红 = 出问题了
      （注册失败 / 探测异常）。「没找到窗口」用橙而不是红：那是**常态**，
      用户只是还没开游戏。

    ★ 缺字段时退回「探测:—」
      ----------------------
      老版本插件写的文件没有这几个键。此时不能假装探测失败（那会误导），
      只能说「不知道」—— 于是显示 ``探测:—``，颜色用最暗的 DIM。

    :param session: :func:`load_session_state` 的返回值（可能 ``None``）。
    """
    if not session:
        return "探测:—", theme.DIM
    result = str(session.get("probe_result") or "")
    if not result:
        # 文件存在但没有探测字段 = 写它的插件是旧版
        return "探测:—(旧版)", theme.DIM

    label, color = _PROBE_RESULT_STYLE.get(result, (result, theme.FG))
    count = int(session.get("probe_count") or 0)
    if result == "found":
        head = "探测: 已建立"
    else:
        head = "探测: 第{}次 · {}".format(count, label)

    detail = str(session.get("probe_detail") or "").strip()
    if detail and result != "found":
        # 「未找到窗口」没有 detail 可言；有 detail 的一定是**具体**原因，
        # 那才是用户要的（"注册失败（什么什么）"而不是"注册失败"）。
        head = "{} · {}".format(head, detail)

    # 版本号挂在**末尾**：探测结论在前（用户先看它），版本是核对信息。
    # 2026-10-10 加 —— 排查时反复卡在"我改了怎么没生效"，根因都是宿主
    # 还跑着上一次的构建；版本号摆在卡片上就不用再猜了。
    ver = str(session.get("plugin_version") or "").strip()
    if ver:
        head = "{}  v{}".format(head, ver)
    return head, color


def build_panel(snap: Any, *, toast: str | None = None,
                toast_sev: str = "INFO", panic: bool = False,
                last_channel: str = "",
                nav: dict | None = None,
                session: dict | None = None) -> PanelModel:
    """把 StateSnapshot 变成 5 行。

    行数固定 5 —— 高度固定就不会因为文本长短跳动。

    :param last_channel: 上一次见到的通道名，由调用方（app）持有。
        run_loop 停止时会释放 capturer，快照里的 ``capture.active``
        随之变空；那一行若显示 "—" 就把"刚才用的是哪个通道"这个
        最该被记住的信息丢了。本函数保持纯粹（无模块级可变状态），
        记忆交给调用方。
    :param nav: 寻路状态（:func:`load_nav_state` 的返回值）。
        **非 None 时整个面板被替换成寻路模式**（见 :func:`_nav_lines`），
        引擎那 5 行（场景/意图/通道/Hz）全部让位 —— 寻路时用户只关心
        「去哪」「在干嘛」，引擎内部状态是噪声。
    :param session: 会话状态（:func:`load_session_state` 的返回值）。
        **只占第 1 行的标签位**，不新增行 —— 见 :func:`_session_prefix`
        上面那段为什么非要挤进行 1：
        「急停: F12」在第 5 行是用户的肌肉记忆，往下挪一行就会让人
        慌乱时按错地方。行数不变 = 位置不变 = 不挪。
    """
    panic_key = str(getattr(snap, "panic_key", "F12"))
    sess_txt, sess_color = _session_prefix(session)

    # ★ 寻路优先：nav 有效时**不再看 snap**。
    #   寻路跑在 run_loop 之外（独立脚本 e2e_pathfind.py），此刻引擎
    #   很可能是 OFFLINE —— 若先判 OFFLINE 就会显示「○ 离线」，
    #   用户明明在寻路却看到离线。
    if nav is not None:
        return PanelModel(lines=_nav_lines(nav, panic_key),
                          height=5 * theme.LINE_HEIGHT,
                          toast=toast,
                          toast_color=theme.severity_color(toast_sev),
                          panic=panic or str(nav.get("state")) == "panic")

    state = str(getattr(snap, "state", "UNKNOWN"))
    health = str(getattr(snap, "health", "OFFLINE"))
    hz = float(getattr(snap, "tick_hz", 0.0))
    p95 = float(getattr(snap, "tick_p95_ms", 0.0))
    scene = str(getattr(snap, "scene", "UNKNOWN"))
    intent = getattr(snap, "intent", None)
    cap = getattr(snap, "capture", None)

    if health == "OFFLINE":
        lines = [
            Line(f"○ 离线 · {sess_txt}", theme.ACCENT_IDLE, theme.ACCENT_IDLE),
            Line("run_loop 未运行", theme.DIM),
            Line("", theme.DIM),
            Line("", theme.DIM),
            Line(f"急停: {panic_key}", theme.DIM),
        ]
        return PanelModel(lines=lines, height=5 * theme.LINE_HEIGHT,
                          toast=toast,
                          toast_color=theme.severity_color(toast_sev),
                          panic=panic)
    # 通道三级兜底：快照当前值 -> 上一次见到的值 -> 破折号。
    # run_loop 停止会释放 capturer，capture.active 随之变空，直接显示
    # "—" 会让用户以为从来没成功用过截图通道。
    active = str(getattr(cap, "active", "") or "") or str(last_channel or "")
    cap_txt = active or "—"
    deg = getattr(cap, "degraded_from", "") or ""
    if deg:
        cap_txt += f" ← {deg}（已降级）"

    idle = int(getattr(intent, "idle_streak", 0) or 0)
    cnt = int(getattr(intent, "count", 0) or 0)
    intent_txt = f"{cnt} 次" if not idle else f"空转（连续 {idle}）"

    # ★ 行首颜色：无会话时用橙色而不是相位色。
    #   引擎跑得好好的（绿）但会话没建 = 803 次 tick 全空转，
    #   那种状态用绿色显示会让人以为一切正常 —— 这正是今天踩的坑。
    head_color = theme.phase_color(state) if sess_color == theme.ACCENT_RUN else sess_color

    # ★★ A6（2026-10-16，demo 出厂前）：把「正常跑完」和「崩了」分开显示。
    #
    #   粗糙度审计的 A1 级问题：真机 2026-10-14 run 结束后，留给观众
    #   视网膜上的**最后一帧**是这样的：
    #
    #       ● 无会话 · 已停止   4.8 Hz   223ms P95     ← 橙色（无会话）
    #       场景: UNKNOWN
    #       意图: 355 次
    #       通道: —
    #
    #   橙字「已停止」+「UNKNOWN」+「通道 —」，观众读作「演示崩了」。
    #   而那一次的 health 其实是 **OK** —— 引擎是自己正常结束的。
    #
    #   判据用 health 而不是「用户有没有点停止」：health=OK 且无会话
    #   才算「跑完了」；崩溃时 health 会是别的值，仍然显示红色「已停止」。
    completed = state == "STOPPED" and health == "OK" and not session
    state_txt = "已完成" if completed else _state_text(state)
    if completed:
        head_color = theme.ACCENT_RUN

    # 场景枚举同理：UNKNOWN 是内部值，观众看到只会以为没做完
    scene_txt = "未知" if scene == "UNKNOWN" else scene

    lines = [
        Line(f"● {sess_txt} · {state_txt}", head_color, head_color),
        Line(f"场景: {scene_txt}", theme.FG),
        Line(f"意图: {intent_txt}", theme.FG),
    ]
    # ★ 通道只在**有话可说**时占一行（2026-10-16，B2）
    #   原来是常驻的 `通道: —`。而演示粗糙度审计把「通道: —」和
    #   橙字「已停止」并列为 A6 级问题：观众读到破折号只会理解成
    #   「这一项坏了」，而不是「这一项暂时没内容」。
    #   现在只在真的抓到过通道、或发生了降级时才显示。
    if cap_txt != "—":
        lines.append(Line(f"通道: {cap_txt}",
                          theme.ACCENT_WARN if deg else theme.FG))
    lines.append(Line(f"急停: {panic_key}", theme.DIM))

    # ★ Hz / P95 默认**不进面板**（2026-10-16，B2）
    #   这两个是给开发者看的性能数字，对观众零信息量，却占了整个
    #   headline 行的三分之二 —— 挤得「会话 · 状态」都读不全。
    #   需要时打开 SHOW_METRICS（走环境变量，不进 UI）。
    if SHOW_METRICS:
        lines.insert(1, Line(f"{hz:.1f} Hz · {p95:.0f}ms P95", theme.DIM))

    return PanelModel(lines=lines, height=5 * theme.LINE_HEIGHT,
                      toast=toast,
                      toast_color=theme.severity_color(toast_sev),
                      panic=panic)


def _state_text(state: str) -> str:
    return {"RUNNING": "运行中", "PAUSED": "已暂停", "STOPPING": "停止中",
            "STOPPED": "已停止", "IDLE": "空闲"}.get(state, state)


__all__ = [
    "Line",
    "NAV_FINAL_STATES",
    "NAV_SCHEMA_VERSION",
    "NAV_STATE_FINAL_MAX_AGE_S",
    "NAV_STATE_FILE",
    "NAV_STATE_MAX_AGE_S",
    "NAV_STOP_FILE",
    "SESSION_HEARTBEAT_S",
    "SESSION_SCHEMA_VERSION",
    "SESSION_STATE_FILE",
    "SESSION_STATE_MAX_AGE_S",
    "load_session_state",
    "OverlayPos",
    "POS_FILE",
    "PanelModel",
    "build_panel",
    "clear_pos",
    "default_pos",
    "load_nav_state",
    "load_pos",
    "resolve_pos",
    "save_pos",
]
