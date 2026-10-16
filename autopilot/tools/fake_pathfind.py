"""假寻路脚本 —— 不开原神也能把 Console 的寻路链路跑通一遍。

> 这是**工具**，不是测试夹具。给你验证 Console 寻路 UI 用的：
> `python -m autopilot.tools.fake_pathfind --live --seconds 8` 起一个，
> Console 里照常点「启动寻路」就行 —— 它会把这个脚本当寻路脚本拉起来。

★ 为什么需要它
  --------------
  Console 的寻路链路（起进程 → 写状态 → 读状态 → 停止）平时只有真脚本能用，
  而真脚本要开原神、要管理员、会在窗口里真的按 Shift+W。想验 UI 就得开游戏 ——
  于是「寻路功能能不能用」这件事**没法离线回答**，只能靠感觉。

  这个脚本复刻真脚本对外可见的全部行为：同样的参数、同样的
  ``pathfind_state.json``（schema 2）、同样的 ``nav_stop.flag`` 停止协议、
  同样的原子写。它**不注入按键、不截图、不碰游戏**。

★ 参数刻意与真脚本一致
  -------------------
  Console 是把 ``--start-world`` / ``--target-world`` / ``--live`` / ``--delay``
  发给「寻路脚本」的。参数名不一样的话，Console 侧就测不到「参数拼装对不对」
  这件事 —— 那才是接线最容易错的地方。

  真脚本的参数表在 ``_SPEC``；这里逐条对齐，并**多**接受几个 ``--fake-*``
  （真脚本没有，Console 也不会发 —— 它们只给你手动调试用）。

★ 不用 argparse
  真脚本就是直接读 ``sys.argv``（未知参数静默跳过）。照做，
  这样参数解析的分歧不会成为「到底测的是谁」的干扰项。

★ ``--fake-mode``
  | 值 | 模拟的现象 |
  |---|---|
  | ``arrive``（默认） | 正常走完 ``--seconds`` 后 arrived |
  | ``stuck`` | 永远 navigating，不前进也不退出（角色卡住） |
  | ``crash`` | 走 3 个 tick 后直接 exit 9，状态文件**停在 navigating** |
  | ``ignore-stop`` | 无视停止 flag —— 用来验 Console 的 terminate 兜底 |

★ 停止协议必须一模一样
  ``nav_stop.flag``「存在即停」，本脚本每 tick 看一次；看到就**先写
  ``state="stopped"`` 再退出**。Console 侧据此显示「已停止（flag 生效）」；
  如果它没退、只能 terminate，Console 会如实提示「按键状态可能不明」。
  那个提示只有在真脚本**真的能响应 flag** 时才是对的 —— 所以这里必须复刻。

用法::

    # 前台跑 8 秒后自动到达（Debug 窗口能看打印）
    python -m autopilot.tools.fake_pathfind --live --seconds 8

    # 模拟卡住：让 Console 启动它，然后去点「停止寻路」
    python -m autopilot.tools.fake_pathfind --live --fake-mode stuck

    # 模拟崩溃：3 个 tick 后 exit 9，状态永远停在 navigating
    python -m autopilot.tools.fake_pathfind --live --fake-mode crash

    # 让 Console 用它：Console 的寻路页签不需要改配置，
    # 只要 engine.pathfind_script 指到本文件即可
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# 运行时定位 —— 与真脚本 e2e_pathfind.py 同一套约定
# --------------------------------------------------------------------------

#: 插件根。**环境变量优先**，与真脚本 ``NEKO_PLUGIN_ROOT`` 同名同义 ——
#: Console 拉起子进程时会继承父进程环境，所以测试里 monkeypatch 这个变量即可
#: 把整个链路指向临时目录，不需要改任何代码。
#:
#: 默认值优先从 ``__file__`` 推导（脚本位于 ``<root>/autopilot/tools/``），
#: 推导不出来才退回本机硬编码路径 —— 真脚本只有硬编码那份，
#: 拷到别的 checkout 就得改代码。
_DEFAULT_ROOT = None
try:
    _p = Path(__file__).resolve()
    for _cand in _p.parents:
        if _cand.name == "autopilot" and (_cand.parent / "plugin").exists():
            _DEFAULT_ROOT = _cand.parent
            break
except OSError:
    pass

PLUGIN_ROOT = Path(os.environ.get(
    "NEKO_PLUGIN_ROOT",
    str(_DEFAULT_ROOT) if _DEFAULT_ROOT else
    r"D:\NEKO\N.E.K.O-main\N.E.K.O-main\plugin\plugins\genshin_game_buddy",
))
RUNTIME_DIR = PLUGIN_ROOT / "autopilot" / "_runtime"

#: 与真脚本 layout.NAV_STATE_FILE / e2e_pathfind.NAV_STOP_FILE 同名同位置
STATE_FILE = RUNTIME_DIR / "pathfind_state.json"
STOP_FILE = RUNTIME_DIR / "nav_stop.flag"

#: 与真脚本 ``NAV_SCHEMA_VERSION`` 一致
NAV_SCHEMA_VERSION = 2

#: 真脚本用的运行标记文件（Console 的 ``read_recent_events`` 靠它找事件流）
RUN_CURRENT_FILE = RUNTIME_DIR / "run_current.txt"

VALID_MODES = ("arrive", "stuck", "crash", "ignore-stop")


# --------------------------------------------------------------------------
# 参数（不用 argparse，照真脚本直接读 argv）
# --------------------------------------------------------------------------

def _pair(s: str) -> tuple[float, float]:
    """与真脚本 ``_pair`` 同语义：``"X,Z"`` -> ``(x, z)``。"""
    return float(s.split(",")[0]), float(s.split(",")[1])


def _parse_args(argv: list[str]) -> dict:
    """``参数名 -> (目标键, 转换, 是否带值)``。结构照抄真脚本的 ``_SPEC``。"""
    spec = {
        "--start-world": ("start", _pair, True),
        "--target-world": ("target", _pair, True),
        "--seconds": ("seconds", float, True),
        "--delay": ("delay", float, True),
        "--live": ("live", True, False),
        "--walk": ("live", False, False),
        # 以下是本脚本独有，真脚本没有；Console 也不会发
        "--fake-mode": ("fake_mode", str, True),
        "--fake-tick-ms": ("tick_ms", float, True),
        "--fake-crash-after": ("crash_after", float, True),
        "--run-id": ("run_id", str, True),
        "--title": ("title", str, True),
        "--channel": ("channel", str, True),
    }
    a: dict = {"live": False, "seconds": 8.0, "delay": 0.0,
               "fake_mode": "arrive", "tick_ms": 500.0, "crash_after": 3.0,
               "run_id": None, "title": None, "channel": None,
               "start": None, "target": None}
    args, i = list(argv), 0
    while i < len(args):
        k = args[i]
        item = spec.get(k)
        if item is None:
            # 真脚本对未知参数是「静默跳过」，这里保持一致
            if not k.startswith("-"):
                a["title"] = k
            i += 1
            continue
        key, conv, takes = item
        if not takes:
            a[key] = conv
            i += 1
            continue
        v = args[i + 1] if i + 1 < len(args) else None
        try:
            a[key] = conv(v)
        except (ValueError, IndexError, TypeError):
            print(f"[fake] ✗ {k} 取值不合法: {v!r}", file=sys.stderr)
            raise SystemExit(2)
        i += 2
    if a["fake_mode"] not in VALID_MODES:
        print(f"[fake] ✗ --fake-mode 只能是 {VALID_MODES}，收到 {a['fake_mode']!r}",
              file=sys.stderr)
        raise SystemExit(2)
    if a["start"] is None or a["target"] is None:
        print("[fake] ✗ 必须给 --start-world X,Z 和 --target-world X,Z",
              file=sys.stderr)
        raise SystemExit(2)
    return a


# --------------------------------------------------------------------------
# 状态写入（原子，与真脚本一致）
# --------------------------------------------------------------------------

_seq = 0


def write_state(state: str, target, dist_m: float,
                phase: str, hint: str, eta_s: float | None) -> None:
    """写 ``pathfind_state.json``。**先写 tmp 再 os.replace** —— 与真脚本一致。

    ★ 为什么要原子
      Console 每 500ms 读一次，而本脚本也每 500ms 写一次。非原子写会让
      Console 撞上「只写了一半」，只能吞掉当帧读数。真脚本是原子的，
      假脚本也必须是 —— 否则这条测试测的是一个**比真实现更糟**的世界。
    """
    global _seq
    _seq += 1
    doc = {
        "schema_version": NAV_SCHEMA_VERSION,
        "written_at_ms": int(time.time() * 1000),
        "seq": _seq,
        "state": state,
        "target_name": None,
        "target_pos": [round(float(target[0]), 1), round(float(target[1]), 1)],
        "dist": round(float(dist_m), 1),
        "delta_deg": None,
        "next_action": "stop" if state != "navigating" else "walk_forward",
        "phase": phase,
        "hint": hint,
        "eta_s": eta_s,
        # ★ 本脚本独有：真机器上靠日志判断，这里给个明确标记，
        #   测试与人都能一眼知道「这是假脚本写的」。
        "fake": True,
    }
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, STATE_FILE)
    except OSError as exc:
        print(f"[fake] [warn] 状态写入失败（不影响模拟）: {exc}")


def emit_event(run_id: str, code: str, message: str) -> None:
    """往 ``<run_id>.stream.jsonl`` 追加一条事件。

    Console 的 ``read_recent_events`` 读的就是这个文件，取 ``code``/``message``。
    """
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        with (RUNTIME_DIR / f"{run_id}.stream.jsonl").open("a",
                                                           encoding="utf-8") as f:
            f.write(json.dumps({"code": code, "message": message,
                                "ts_ms": int(time.time() * 1000)},
                               ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[fake] [warn] 事件写入失败: {exc}")


# --------------------------------------------------------------------------
# 主循环
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    a = _parse_args(list(sys.argv[1:] if argv is None else argv))

    run_id = a["run_id"] or time.strftime("%Y%m%d_%H%M%S")
    print(f"[fake] 模式={a['fake_mode']}  live={a['live']}  "
          f"seconds={a['seconds']}  tick={a['tick_ms']}ms")
    print(f"[fake] runtime = {RUNTIME_DIR}")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        RUN_CURRENT_FILE.write_text(run_id, encoding="utf-8")
    except OSError:
        pass

    # ★ 启动前删掉残留 flag —— 与真脚本同一个理由：
    #   flag 的语义是「存在即停」，不清掉的话新实例一起来就秒停，
    #   而 Console 侧的日志只会显示「转一下就没了」。
    try:
        STOP_FILE.unlink(missing_ok=True)
    except OSError as exc:
        print(f"[fake] [warn] 残留 flag 删不掉: {exc}")

    if a["delay"] > 0:
        print(f"[fake] 等待 {a['delay']}s（模拟 Console 的启动延迟）")
        time.sleep(a["delay"])

    start, target = a["start"], a["target"]
    total = max(1e-6, float(((target[0] - start[0]) ** 2
                              + (target[1] - start[1]) ** 2) ** 0.5))
    tick_ms = max(20.0, float(a["tick_ms"]))
    seconds = float(a["seconds"])

    emit_event(run_id, "PATHFIND_START",
               f"假寻路启动 {start} -> {target}")
    print(f"[fake] 起跑 {start} -> {target}，全程 {total:.1f} 米")

    t0 = time.monotonic()
    tick = 0
    while True:
        elapsed = time.monotonic() - t0
        frac = min(1.0, elapsed / max(1e-6, seconds))

        # --- 停止协议（ignore-stop 模式故意无视）---
        if a["fake_mode"] != "ignore-stop" and STOP_FILE.exists():
            write_state("stopped", target, total * (1.0 - frac),
                        "stopped", "已按停止信号停下", 0.0)
            emit_event(run_id, "PATHFIND_STOPPED", "收到 nav_stop.flag，已松键")
            print(f"[fake] ★ 收到 {STOP_FILE.name} -> 写 state=stopped，退出")
            return 0

        if a["fake_mode"] == "crash" and tick >= int(a["crash_after"]):
            # ★ 崩溃时**不**改状态文件：留下的最后一帧是 navigating，
            #   这正是「进程死了但状态文件还在」那个最危险的场景。
            emit_event(run_id, "PATHFIND_CRASHED", "模拟崩溃（状态留在 navigating）")
            print(f"[fake] ✗ 模拟崩溃，exit 9；状态文件停在 navigating")
            return 9

        if a["fake_mode"] == "stuck":
            write_state("navigating", target, total, "stuck",
                        "卡住了（模拟）", None)
        elif frac >= 1.0:
            write_state("arrived", target, 0.0, "arrive", "已到达", 0.0)
            emit_event(run_id, "PATHFIND_ARRIVED", f"到达 {target}")
            print(f"[fake] ✓ arrived，用时 {elapsed:.1f}s")
            return 0
        else:
            write_state("navigating", target, total * (1.0 - frac),
                        "approach", "前往目标", max(0.0, seconds - elapsed))

        tick += 1
        time.sleep(tick_ms / 1000.0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
