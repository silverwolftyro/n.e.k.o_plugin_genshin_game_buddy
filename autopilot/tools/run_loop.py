"""跑主循环。这是 N.E.K.O. 3.0 的「一键启动」入口。

    python -m autopilot.tools.run_loop                       # 干跑 5Hz
    python -m autopilot.tools.run_loop --live                # ⚠ 真发按键
    python -m autopilot.tools.run_loop --tick-rate 10 --duration 30
    python -m autopilot.tools.run_loop --title 原神 --stats-interval 5

**默认是干跑**：执行层用 DryRunBackend，一个键都不会发出去。
真发必须显式加 --live —— 这个开关控制的是「会不会动你的游戏」，
不该是默认值。

退出方式（三选一，效果相同，都是优雅停止：松键 -> 等 tick -> 关资源）：
    F12        急停（无焦点，游戏里也能按）
    Ctrl+C     SIGINT
    --duration 到点自动停

⚠ --live 的常见失败（报错会明确告诉你是哪一种）：
    · 权限不足 -> 以管理员身份运行
    · 被安全软件拦截 -> 关掉 360 / 火绒 的键鼠防护
    · UIPI -> 目标窗口权限高于本进程时无法注入
    先跑 loop_diagnose.py 确认执行层可用，再 --live。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import (  # noqa: E402
        C, bootstrap, line, setup_logging,
    )
else:
    from ._bootstrap import C, bootstrap, line, setup_logging

BAR = "═" * 54

logger = logging.getLogger("autopilot.run_loop")

#: ``--title`` 的 ASCII 别名 -> 真实窗口标题。
#:
#: ★ 为什么需要它（2026-10-10 真机事故）
#:   ---------------------------------
#:   批处理文件里放中文 -> 要么存 UTF-8（936 控制台下 cmd 读成 GBK，
#:   整份脚本乱码，`--title 原神` 变成 `--title 鍘熺`，实测直接
#:   「不是内部或外部命令」连启动都做不到），要么存 GBK（换台机器、
#:   换个控制台代码页又炸）。**纯 ASCII 的 .bat 才没有这个问题。**
#:
#: ★ 为什么别名必须翻译成中文，不能直接用英文
#:   --------------------------------------
#:   ``capturer._find_window()`` 只做 **窗口标题子串匹配**
#:   （``if title_substr in win32gui.GetWindowText(hwnd)``），
#:   **完全不看进程名** —— 原神窗口的标题就是「原神」，
#:   传 ``--title genshin`` 一个窗口都匹配不到。
#:   （B 轨的 ``screen_capture.find_genshin_window()`` 才兼看进程名，
#:   那是另一个模块、另一条路，别混。）
#:
#: 用法：``--title genshin`` 等价于 ``--title 原神``。
#: 认不出来的原样保留（用户可能传了完整标题），由 _find_window 去匹配。
TITLE_ALIASES: dict[str, str] = {
    "genshin": "原神",
    "genshinimpact": "原神",
    "genshin impact": "原神",
    "yuanshen": "原神",
    "原神": "原神",
}

#: A 轨快照写入间隔。200ms 是「人眼察觉不到差别」的上限，
#: 再快只是白费 IO；overlay 读得更勤也没意义（读的是同一份文件）。
UI_SNAPSHOT_INTERVAL_S = 0.2

#: 命令有效期（秒）。
#:
#: ★ 为什么需要它
#:   ``commands.jsonl`` 是"主控台 -> 引擎"的单向通道，**没人清空过**。
#:   用户在引擎没跑时点过一次「完全退出 / 强杀」，文件里就留下一条 KILL；
#:   下次引擎启动读到它，当场退出 —— 表现就是"点启动秒退"，而且必现。
#:   真正的修复是消费完就清空（``cmd_poller.clear()``），本阈值只是第二道：
#:   引擎启动前 30 秒内的命令视为"本次会话的"，更早的一律丢弃。
#:   30 秒足够覆盖"主控台刚发完 STOP，引擎紧接着启动"的场景。
COMMAND_TTL_S = 30


def _ui_health(st: dict) -> "Health":
    """把 LoopStats.health（中文长句）归并成协议里的粗粒度枚举。

    ★ 协议里只放稳定英文枚举，中文长句留在 raw_stats 做诊断 ——
      否则 overlay 的显示逻辑要跟着中文措辞走，改一次文案就坏一次。
    """
    from autopilot.ui.schema import Health

    h = str(st.get("health", ""))
    if "tick 崩溃" in h:
        return Health.CRASHED
    if "感知" in h or "故障" in h:
        return Health.DEGRADED
    if "长期空转" in h:
        return Health.IDLE_LONG
    return Health.OK


def _ui_kill() -> None:
    """KILL：立即退出，**跳过一切清理**。

    ★ 任何「先清理再退出」的实现在这里都会跟着一起卡住 ——
      卡住的正是清理里那步（按键释放 / 截图器关闭）。
      所以延迟一点让事件落盘，然后直接走。
    """
    time.sleep(0.2)
    logger.warning("KILL: os._exit(1) —— 跳过一切清理")
    os._exit(1)


def _ui_snapshot(ctx, loop, run_id: str, scene: str, last_action: str):
    """从 ctx / loop 组一份 StateSnapshot。

    ★ 每个来源都单独 try —— 截图器、stats、bus 任何一处出问题
      都不该让 UI 线程死掉（死掉的话 overlay 会一直显示"离线"，
      而 run_loop 明明还好好地跑着）。
    """
    from autopilot.ui.schema import (  # noqa: PLC0415
        CaptureStat, IntentStat, RuntimePhase, StateSnapshot,
    )

    try:
        st = ctx.stats.snapshot()
    except Exception as exc:  # noqa: BLE001
        logger.debug("读 stats 失败（UI 用空快照）：%s", exc)
        st = {}

    try:
        phase = RuntimePhase(loop.state.value)
    except Exception:  # noqa: BLE001
        phase = RuntimePhase.UNKNOWN

    try:
        gac = getattr(ctx.capturer, "get_active_channel", None)
        ch = gac() if callable(gac) else None
        active = str(getattr(ch, "value", "") or "")
    except Exception:  # noqa: BLE001
        active = ""

    return StateSnapshot(
        run_id=run_id,
        written_at_ms=int(time.time() * 1000),
        uptime_s=float(st.get("uptime_s", 0.0)),
        state=phase,
        health=_ui_health(st),
        tick_hz=float(st.get("actual_hz", 0.0)),
        tick_avg_ms=float(st.get("avg_tick_ms", 0.0)),
        tick_p95_ms=float(st.get("p95_tick_ms", 0.0)),
        tick_count=int(st.get("ticks", 0)),
        scene=scene,
        intent=IntentStat(count=int(st.get("intents_executed", 0)),
                          idle_streak=int(st.get("consecutive_idle", 0)),
                          last_action=last_action),
        capture=CaptureStat(active=active, degraded_from=""),
        raw_stats=st,
    )


def _ui_event_to_state(ev) -> tuple[str, str, str, dict]:
    """core 的 Event -> (code, severity, message, context)。

    只做字段搬运 + severity 归一，**不解释业务** —— 解释是 overlay 的事。
    """
    sev = str(getattr(ev, "severity", "") or "INFO")
    sev = sev.split(".")[-1].upper()
    if sev not in ("INFO", "WARN", "ERROR"):
        sev = "INFO"
    return (str(getattr(ev, "code", "") or ""),
            sev,
            str(getattr(ev, "message", "") or ""),
            dict(getattr(ev, "context", {}) or {}))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="运行 N.E.K.O. 主循环（默认 dry-run，不发按键）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--title", default="",
                    help="窗口匹配关键字（默认取配置里的 window_title_substr）")
    ap.add_argument("--rules-dir", default="", help="覆盖默认规则目录")
    ap.add_argument("--dry-run", action="store_true",
                    help="执行层用 DryRunBackend（默认）")
    ap.add_argument("--live", action="store_true",
                    help="⚠ 真的发送按键。会控制你的键鼠")
    ap.add_argument("--tick-rate", type=float, default=5.0,
                    help="tick 频率 Hz，0 = 不限速（默认 5）")
    ap.add_argument("--stats-interval", type=float, default=10.0,
                    help="每 N 秒打一行统计，0 = 不打（默认 10）")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="跑 N 秒后自动停，0 = 无限（默认 0）")
    # ★ 2026-10-10 新增。为「游戏只能全屏」的用户准备：双击脚本 -> 引擎
    #   还没起来 -> 此刻窗口在前台、游戏必然是最小化 -> 立刻 build_stack()
    #   就会因为抓不到窗口而失败。延迟给你时间 alt-tab 回游戏。
    ap.add_argument("--delay", type=float, default=0.0,
                    help="组装前先等 N 秒，期间打倒计时（默认 0 = 不等）。"
                         "★ 给「只能全屏/最小化」的游戏用：等这段时间里切回游戏")
    ap.add_argument("--force-channel", default="",
                    help="只用这一条截图通道，不再降级。合法值："
                         "bitblt / dxcam / mss（别名 bit_blt、gdi、dx 也认）。"
                         "排查某条通道在**全屏**下到底行不行时用它")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARN", "ERROR"])
    # ★ 让 stdout/stderr **编码失败不等于崩溃**（2026-10-10）。
    #   默认 GBK 控制台编不了 `⚠`(U+26A0)、`▶`、`↻` 这类字符，
    #   而上面 --live 的帮助文本里就有 `⚠`。
    #   argparse 的 ``--help`` 是在 **parse_args() 内部**就 print_help()
    #   然后 ``sys.exit()`` 的 —— 所以这一段**必须排在 parse_args() 之前**，
    #   放到 bootstrap() 后面（曾经放那儿）永远轮不到执行，
    #   ``--help`` 照样抛 UnicodeEncodeError 退出 1。
    #   只加 errors="replace"、**不改编码**：本来就编得动的中文照常显示，
    #   编不动的退化成 `?`，总比整个程序起不来强。
    #   （start_dev.bat 里有 ``chcp 65001``，那条路本来就不受影响。）
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")      # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass                                      # 重定向到不支持的流就跳过

    args = ap.parse_args()

    bootstrap()
    setup_logging(args.log_level == "DEBUG")

    if not sys.stdout.isatty():
        C.disable()

    # ---- 参数校验（响亮，不静默修正）----
    if args.live and args.dry_run:
        print(f"{C.ERR}✗ --live 与 --dry-run 不能同时给{C.OFF}", file=sys.stderr)
        print(f"{C.DIM}  想真发就只给 --live；只干跑就只给 --dry-run（或者什么都不给）。{C.OFF}")
        return 2
    if args.tick_rate < 0:
        print(f"{C.ERR}✗ --tick-rate 不能为负（{args.tick_rate}）{C.OFF}",
              file=sys.stderr)
        return 2
    if args.duration < 0:
        print(f"{C.ERR}✗ --duration 不能为负（{args.duration}）{C.OFF}",
              file=sys.stderr)
        return 2
    if args.delay < 0:
        print(f"{C.ERR}✗ --delay 不能为负（{args.delay}）{C.OFF}",
              file=sys.stderr)
        return 2

    live = args.live and not args.dry_run

    # ---- 延迟启动（--delay）----
    #
    # ★ 为什么必须放在 build_stack() **之前**，而不是主循环之前
    #   ------------------------------------------------------
    #   ``build_stack()`` 内部会 ``capturer.open()``，而全屏游戏一旦
    #   失去前台就最小化，客户区退化成 0x0 —— 那一步直接抛
    #   WindowNotFoundError。所以"等用户切回游戏"这件事必须发生在
    #   截图器打开**之前**，否则等多久都没用。
    #
    # ★ 为什么打倒计时而不是静默 sleep
    #   ------------------------------
    #   用户双击之后**看不到任何反馈**，会以为程序卡死或根本没启动，
    #   于是又双击一次 —— 第二个实例立刻跟第一个抢同一个游戏窗口。
    #   每秒一行让"它还活着，还有 N 秒"这件事是确定的。
    if args.delay > 0:
        total = int(round(args.delay))
        print(f"{C.WARN}▶ 等 {total} 秒后开始 —— 现在请切回游戏窗口！{C.OFF}")
        try:
            for left in range(total, 0, -1):
                print(f"  {C.DIM}{left:>3} 秒后开始组装（切回游戏）{C.OFF}",
                      flush=True)
                time.sleep(1.0)
        except KeyboardInterrupt:
            print(f"\n{C.ERR}✗ 等待期间被中断{C.OFF}", file=sys.stderr)
            return 130
        print(f"{C.OK}✓ 等待结束，开始组装{C.OFF}")

    # ---- 组装三层 ----
    from autopilot.config.loader import load_config  # noqa: PLC0415

    try:
        cfg = load_config()
        if args.title:
            # ASCII 别名 -> 真实标题（见 TITLE_ALIASES 的注释）。
            # 未命中的原样传下去：用户可能直接给了完整标题片段。
            raw = args.title
            title = TITLE_ALIASES.get(raw.strip().lower(), raw)
            if title != raw:
                logger.info("--title %r 是别名，已解析为 %r", raw, title)
            cfg.capture.window_title_substr = title

        # ★ --force-channel 用**收窄 channel_order** 实现，不去改 _stack.py
        #   的函数签名 —— 后者是本轮约束之外的另一个文件。
        #   单元素 channel_order 在 Capturer 里的语义与 force_channel 完全
        #   等价：``_demote_channel()`` 往后找不到下一条，只能如实报错。
        if args.force_channel:
            spec = args.force_channel.strip().lower()
            alias = {"bit_blt": "bitblt", "gdi": "bitblt",
                     "dx": "dxcam", "dxgi": "dxcam", "gdi_screen": "mss"}
            ch = alias.get(spec, spec)
            if ch not in ("bitblt", "dxcam", "mss"):
                print(f"{C.ERR}✗ --force-channel 只能是 "
                      f"bitblt / dxcam / mss（收到 {args.force_channel!r}）{C.OFF}",
                      file=sys.stderr)
                return 2
            cfg.capture.channel_order = [ch]
            logger.info("--force-channel=%s -> 只用它，不再降级", ch)
        # ★ 用**绝对**导入而不是 `from ._stack import`：直接
        #   `python tools/run_loop.py` 时 __package__ 为空，
        #   相对导入会 ImportError —— 只有 `python -m` 才走得通。
        from autopilot.tools._stack import build_stack  # noqa: PLC0415

        st = build_stack(cfg=cfg, rules_dir=args.rules_dir or None,
                         dry_run=not live)
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 组装失败：{type(exc).__name__}: {exc}{C.OFF}",
              file=sys.stderr)
        print(f"{C.DIM}  先跑 python -m autopilot.tools.loop_diagnose 看哪一环挂了{C.OFF}")
        return 2

    for w in st.warnings or []:
        print(f"{C.WARN}⚠ {w}{C.OFF}")

    # ---- 主循环 ----
    from autopilot.actuator.safety import PANIC_KEY  # noqa: PLC0415
    from autopilot.core import CoreLoop
    from autopilot.core.runtime import RuntimeContext  # noqa: PLC0415
    from autopilot.core.events import LogSubscriber  # noqa: PLC0415

    # ★ 不引用 cfg.loop.* —— 那些字段是否已进 schema 在本轮无法确认，
    #   引用它等于给 run_loop 埋一个 AttributeError。这里用 RuntimeContext
    #   / CoreLoop 自己的默认值（本来就一致：5Hz / 10 次 / 5 次 / 10s）。
    ctx = RuntimeContext(capturer=st.capturer, builder=st.builder,
                         policy=st.policy, executor=st.executor,
                         guard=st.guard, tick_rate_hz=args.tick_rate)
    # 急停直接读 SafetyGuard —— 它是唯一碰 GetAsyncKeyState 的地方，
    # 这里不再自己轮询一遍（两处轮询迟早会不一致）。
    guard = st.guard
    loop = CoreLoop(ctx, panic_check=((lambda: guard.panic_active) if guard else None))
    ctx.bus.subscribe(LogSubscriber(), name="log")

    # ---- A 轨 UI 通道 ----
    # ★ 只加一个 subscriber + 一条后台线程，**不改 CoreLoop / EventBus /
    #   RuntimeContext 的任何定义**。core 是已验收的核心，UI 的 IO 需求
    #   不该去改它。
    from autopilot.ui.commands import CommandPoller  # noqa: PLC0415
    from autopilot.ui.schema import EventCode, StateEvent  # noqa: PLC0415
    from autopilot.ui.writer import ChannelWriter  # noqa: PLC0415

    ui_state = {"stop": False, "kill": False,
                "scene": "UNKNOWN", "last_action": ""}
    ui_stop = threading.Event()
    ui_thread: threading.Thread | None = None
    ui_unsub = None
    writer = None
    try:
        writer = ChannelWriter()
        cmd_poller = CommandPoller()
        # 过期判定的基准时刻：UI 通道建立的瞬间（≈引擎启动时刻）
        _START_MS = int(time.time() * 1000)

        def _command_expired(c) -> bool:
            """这条命令是不是"上次会话留下的"。"""
            ts = getattr(c, "ts_ms", None)
            if not isinstance(ts, (int, float)):
                return False        # 没有时间戳的老格式：放行，别误杀
            return ts < _START_MS - COMMAND_TTL_S * 1000
    except Exception as exc:  # noqa: BLE001 - UI 建不起来不该挡住主循环
        logger.warning("UI 通道初始化失败，本次不带可视化：%s", exc)
        writer = None
        cmd_poller = None

    if writer is not None:

        def _ui_on_event(ev) -> None:
            """EventBus 订阅者：core 的 Event -> 协议的 StateEvent。

            跑在 EventBus 的**分发线程**里（不是主循环），所以这里的 IO
            不占 tick 预算。
            """
            try:
                code, sev, msg, ctxt = _ui_event_to_state(ev)
                if code == "SCENE_CHANGED":
                    ui_state["scene"] = str(ctxt.get("to", "UNKNOWN"))
                elif code == "INTENT_EXECUTED":
                    ui_state["last_action"] = str(ctxt.get("action", ""))
                writer.write_event(StateEvent(code=code, severity=sev,
                                              message=msg, context=ctxt))
            except Exception:  # noqa: BLE001
                pass      # 事件落盘失败绝不冒泡

        def _ui_pump() -> None:
            """UI 后台线程：快照 200ms + 命令每轮。

            ★ 用 ``Event.wait`` 而不是 ``time.sleep`` —— 退出时能立刻唤醒，
              不必等满一个周期。
            """
            while not ui_stop.is_set():
                try:
                    writer.write_snapshot(_ui_snapshot(
                        ctx, loop, writer.run_id,
                        ui_state["scene"], ui_state["last_action"]))
                except Exception as exc:  # noqa: BLE001
                    # ★ 从 debug 提到 warning（2026-10-13 修）。
                    #   「写快照失败」在默认 INFO 级别下**完全不可见** ——
                    #   症状是 overlay 上的数字几分钟不更新，看起来像「引擎
                    #   空闲」，而真相可能是磁盘满 / 目录被锁。
                    #   快照是给人看的诊断通道，它静默失效等于监控失灵。
                    #   频次上界是 5Hz，真出问题时会刷屏 —— 那正是
                    #   「确实坏了」时该有的样子。
                    logger.warning("写快照失败（忽略）：%s", exc)
                try:
                    got = cmd_poller.poll()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("轮询命令失败（忽略）：%s", exc)
                    got = []
                if got:
                    for c in got:
                        # ── 防线 1：过期命令直接丢弃，不执行 ──
                        if _command_expired(c):
                            logger.warning(
                                "丢弃过期命令 %s（ts_ms=%s，早于本次启动 "
                                "%d 秒）—— 它是上次会话的残留",
                                c.cmd, getattr(c, "ts_ms", "?"), COMMAND_TTL_S)
                            continue
                        try:
                            _ui_handle(c)
                        except Exception as exc:  # noqa: BLE001
                            logger.exception("处理命令 %s 失败（忽略）", c.cmd)
                    # ── 防线 2：消费完立刻清空（核心修复）────────────
                    #   不清就必现"上次残留的 KILL 把这次启动打死"。
                    #   CommandPoller.clear() 内部会把 offset 归零，
                    #   所以下一轮读到的是**清空之后**新写的命令。
                    try:
                        cmd_poller.clear()
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("清空命令文件失败（忽略）：%s", exc)
                ui_stop.wait(UI_SNAPSHOT_INTERVAL_S)

        def _ui_handle(c) -> None:
            """命令 -> 标志位。**只置标志，不直接调 loop.***。

            ``loop.stop()`` 会 join 工作线程；从后台线程直接调等于让线程
            去 join 自己。真正的停止由主循环执行（那里才是安全的地方）。
            """
            if c.cmd == "STOP":
                ui_state["stop"] = True
                note, sev = "收到停止命令", "INFO"
            elif c.cmd == "KILL":
                ui_state["kill"] = True
                note, sev = "收到强制终止命令，即将退出", "ERROR"
                threading.Thread(target=_ui_kill, name="ui-kill",
                                 daemon=True).start()
            elif c.cmd in ("PAUSE", "RESUME"):
                # v1 不实现：CoreLoop.pause() 要在主循环里调才安全，
                # 而命令是后台线程收的。宁可不响应，也不要做一半。
                logger.warning("命令 %s 暂未实现（v1），已忽略", c.cmd)
                note, sev = f"命令 {c.cmd} 暂未实现", "WARN"
            else:
                logger.warning("忽略不支持的命令：%r", c.cmd)
                note, sev = f"忽略命令 {c.cmd}", "WARN"
            try:
                writer.write_event(StateEvent(
                    code=EventCode.COMMAND_RECEIVED.value, severity=sev,
                    message=note,
                    context={"cmd_id": c.cmd_id, "cmd": c.cmd}))
            except Exception:  # noqa: BLE001
                pass

        ui_unsub = ctx.bus.subscribe(_ui_on_event, name="ui-channel")

    # ---- banner（一次性，不刷屏）----
    win_ok = bool(getattr(st.capturer, "window_rect", None))
    win_txt = (f"{cfg.capture.window_title_substr} {C.OK}(已找到){C.OFF}"
               if win_ok else
               f"{cfg.capture.window_title_substr} {C.ERR}(未找到){C.OFF}")
    chan = getattr(st.capturer, "get_active_channel", None)
    chan_txt = f"，通道 {chan().value}" if callable(chan) else ""
    interval_ms = ctx.target_interval_s * 1000
    mode_txt = (f"{C.ERR}LIVE{C.OFF} {C.WARN}⚠ 会真的控制键鼠{C.OFF}"
                if live else
                f"{C.OK}DRY-RUN{C.OFF} {C.DIM}(不发按键){C.OFF}")

    print(f"\n{BAR}")
    print(f"{C.BOLD}N.E.K.O. Autopilot — 主循环{C.OFF}")
    print(f"{BAR}")
    print(f"  模式      : {mode_txt}")
    print(f"  窗口      : {win_txt}{chan_txt}")
    print(f"  规则      : {st.rule_count} 条（目录: {st.rules_dir}）")
    print(f"  Tick 率   : {args.tick_rate:g} Hz"
          + (f"（预期每 {interval_ms:.0f}ms 一帧）" if interval_ms else "（不限速）"))
    print(f"  急停键    : {PANIC_KEY.name}")
    print(f"  统计      : 每 {args.stats_interval:g}s 一行"
          if args.stats_interval else "  统计      : 关闭")
    if args.duration:
        print(f"  自动停止  : {args.duration:g}s 后")
    print(f"  退出      : {C.BOLD}Ctrl+C 或 {PANIC_KEY.name}{C.OFF}")
    print(f"{BAR}\n")

    if not win_ok and live:
        print(f"{C.WARN}⚠ 没找到目标窗口 —— 现在跑 --live 只会对着别的窗口发按键{C.OFF}")
        print(f"{C.DIM}  确认游戏已启动，且标题包含"
              f"「{cfg.capture.window_title_substr}」{C.OFF}\n")

    # ---- 定时统计 ----
    stop_stats = threading.Event()

    def stats_pump() -> None:
        while not stop_stats.wait(args.stats_interval):
            s = ctx.stats.snapshot()
            print(f"{C.DIM}[{s['ticks']:>5} tick] {s['actual_hz']:>5.1f}Hz "
                  f"avg {s['avg_tick_ms']:>6.1f}ms | "
                  f"意图 {s['intents_executed']:>3} 空转 {s['idle_ticks']:>3} | "
                  f"故障 {s['capture_failures'] + s['perception_failures']} | "
                  f"{s['health']}{C.OFF}")

    if args.stats_interval > 0:
        threading.Thread(target=stats_pump, name="stats", daemon=True).start()

    # ---- Ctrl+C ----
    interrupted = threading.Event()

    def on_sigint(*_a):
        interrupted.set()
        print(f"\n{C.WARN}收到 Ctrl+C，正在优雅停止…{C.OFF}")

    try:
        signal.signal(signal.SIGINT, on_sigint)
    except ValueError:  # pragma: no cover - 非主线程
        pass

    # ---- 主流程 ----
    t0 = time.monotonic()
    reason = "duration 到点" if args.duration > 0 else "运行结束"
    try:
        loop.start()
        # ★ CoreLoop 启动**之后**才 spawn —— 此时 EventBus 已 start，
        #   订阅者才收得到启动期的事件。
        if writer is not None:
            ui_thread = threading.Thread(target=_ui_pump, name="ui-pump",
                                         daemon=True)
            ui_thread.start()
            try:
                writer.write_event(StateEvent(
                    code=EventCode.RUN_STARTED.value, severity="INFO",
                    message="run_loop 已启动", context={}))
            except Exception:  # noqa: BLE001
                pass
        while loop.state.alive:
            time.sleep(0.05)
            if ui_state["stop"] or ui_state["kill"]:
                reason = "收到 overlay 停止命令"
                break
            if interrupted.is_set():
                reason = "Ctrl+C"
                break
            if args.duration > 0 and (time.monotonic() - t0) >= args.duration:
                break
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    finally:
        # ★ 双保险：这里和 loop.stop() 里各 release_all 一次。
        #   任何一边提前抛异常，另一边仍然会松键。
        stop_stats.set()
        try:
            if loop.state.alive:
                loop.stop(reason)
        except Exception as exc:  # noqa: BLE001
            print(f"{C.ERR}停止时出错：{exc}{C.OFF}", file=sys.stderr)
        finally:
            try:
                loop.ctx.executor.sim.release_all()
            except Exception:  # noqa: BLE001
                pass
            # ---- UI 通道收尾（顺序有讲究）----
            # ① 先停线程：它还可能在写快照，先关 writer 会让它写进死通道
            ui_stop.set()
            if ui_thread is not None and ui_thread.is_alive():
                ui_thread.join(timeout=2.0)
                if ui_thread.is_alive():
                    print(f"{C.WARN}UI 后台线程 2s 内未退出（忽略）{C.OFF}",
                          file=sys.stderr)
            # ② 再退订：否则事件还在往一个准备关闭的通道里灌
            if ui_unsub is not None:
                try:
                    ui_unsub()
                except Exception:  # noqa: BLE001
                    pass
            # ③ 最后补一条 RUN_STOPPED 并关闭 —— 用户在 overlay 上
            #    能看到"它停了"，而不是突然变成离线
            if writer is not None:
                try:
                    writer.write_snapshot(_ui_snapshot(
                        ctx, loop, writer.run_id,
                        ui_state["scene"], ui_state["last_action"]))
                    writer.write_event(StateEvent(
                        code=EventCode.RUN_STOPPED.value, severity="INFO",
                        message=f"run_loop 已停止（{reason}）", context={}))
                    writer.close()
                except Exception as exc:  # noqa: BLE001
                    print(f"{C.WARN}关闭 UI 通道时出错：{exc}{C.OFF}",
                          file=sys.stderr)

    # ---- 最终统计 ----
    print(f"\n{line()}")
    print(f"{C.BOLD}最终统计{C.OFF}  {C.DIM}({reason}){C.OFF}")
    print(ctx.stats.format_report("运行结束"))
    bs = ctx.bus.stats()
    print(f"  事件        发布 {bs['published']} / 投递 {bs['delivered']} / "
          f"丢弃 {bs['dropped']} / 去重抑制 {bs['suppressed']}")
    if ctx.stats.health not in ("正常", "未启动"):
        print(f"{C.WARN}  健康度: {ctx.stats.health}{C.OFF}")
    print(f"{line()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
