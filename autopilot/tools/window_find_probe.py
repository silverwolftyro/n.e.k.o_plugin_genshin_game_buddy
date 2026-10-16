"""窗口查找诊断：把"独立进程看到什么"和"插件宿主看到什么"摆在一起。

    python -m autopilot.tools.window_find_probe

★ 为什么需要它
--------------
2026-10-10 真机事故：``screen_capture.find_genshin_window()`` 在独立进程里
能返回原神窗口，在N.E.K.O.exe 的插件子进程里恒返回 None，而外层只看到
「没找到游戏窗口」，与「游戏没开」完全无法区分。

本脚本输出三段，**并排对照**：

  A) 本进程的主路径结果（screen_capture.find_genshin_window）
  B) 本进程的标题兜底结果（window_probe.find_window_by_title）
  C) 插件日志里宿主自己打出的窗口普查（``[窗口普查`` ）

若 A 有、B 有、C 也说命中 -> 宿主没问题，看别处。
若 A 无、B 有、C 也说没枚举到 -> 宿主**看不见**游戏窗口。
若 A 无、B 有、C **枚举到了但命中 0** -> psutil 在宿主里抛了（主路径就死在这）。
若 C 一条都没有-> 宿主还没跑到普查那一步（先确认插件版本装对了）。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from ._bootstrap import C, bootstrap, line  # noqa: F401

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_LOG_DIR = Path(r"C:\Users\alpha\AppData\Local\N.E.K.O\logs\plugin")

#: 独立进程用哪个解释器跑，取决于调用者；这里只是打出来方便对照。
#: 插件宿主用的是 Electron 自带的那个，**和这里不是同一个** ——
#: 2026-10-10 排查时最容易被这条误导。


def _census_from_plugin_log(limit: int = 6) -> list[str]:
    """把插件日志里宿主打出的窗口普查捞出来。

    ★ 为什么捞得到：普查现在走插件**自己**的 logger
      （``plugin.genshin_game_buddy``）—— 宿主只给这一个名字装了 handler。
      凌晨那版写在 screen_capture 里、用子 logger 打，日志里一条都没有：
      记录往 root 冒泡时没人接，被日志配置吃掉了。
    """
    try:
        logs = sorted(PLUGIN_LOG_DIR.glob("*genshin*.log"),
                      key=lambda p: p.stat().st_mtime)
        if not logs:
            return ["（找不到插件日志目录 %s）" % PLUGIN_LOG_DIR]
        text = logs[-1].read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ["（读插件日志失败：%s）" % exc]
    return [ln for ln in text.splitlines() if "[窗口普查" in ln][-limit:] or [
        "（今天的插件日志里还没有 [窗口普查] —— 宿主可能还在跑旧版本，"
        "或者还没连续 miss 到 3 轮）"]


def main() -> int:
    bootstrap()
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass

    ap = argparse.ArgumentParser(description="窗口查找诊断（对照宿主）")
    ap.add_argument("--title", default="原神", help="窗口标题关键字")
    args = ap.parse_args()

    if str(PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(PLUGIN_ROOT))
    sys.path.insert(0, str(PLUGIN_ROOT / "plugin" / "plugins"))

    print()
    print("=" * 74)
    print(" 窗口查找诊断")
    print("=" * 74)
    print("  解释器 :", sys.executable)
    print("  插件根 :", PLUGIN_ROOT)

    # ---- A) 主路径 -------------------------------------------------------
    print()
    print("A) 主路径 screen_capture.find_genshin_window()")
    main_result = None
    try:
        from plugin.plugins.genshin_game_buddy.screen_capture import find_genshin_window  # noqa: PLC0415
        main_result = find_genshin_window()
        print("   ->", main_result)
        if main_result:
            print("   注意：它可能命中的是**别的**同名窗口，看 title 确认。")
    except Exception as exc:  # noqa: BLE001
        print("   !! %s: %s" % (type(exc).__name__, exc))

    # ---- B) 标题兜底 -----------------------------------------------------
    print()
    print("B) 兜底 window_probe.find_window_by_title(%r)" % args.title)
    fallback = None
    try:
        from plugin.plugins.genshin_game_buddy.window_probe import find_window_by_title  # noqa: PLC0415
        fallback = find_window_by_title(args.title)
        print("   ->", fallback)
    except Exception as exc:  # noqa: BLE001
        print("   !! %s: %s" % (type(exc).__name__, exc))

    # ---- 本进程普查 -----------------------------------------------------
    print()
    print("B2) 本进程的窗口普查（对照宿主的 C 段）")
    try:
        from plugin.plugins.genshin_game_buddy.window_probe import log_window_census  # noqa: PLC0415
        logging.basicConfig(level=logging.WARNING,
                            format="   %(levelname)s %(message)s")
        log_window_census(logging.getLogger("probe"), "本进程")
    except Exception as exc:  # noqa: BLE001
        print("   !! %s: %s" % (type(exc).__name__, exc))

    # ---- C) 宿主普查 ----------------------------------------------------
    print()
    print("C) 插件日志里**宿主自己**打出的窗口普查")
    for ln in _census_from_plugin_log():
        print("   ", ln[:300])

    # ---- 结论 -----------------------------------------------------------
    print()
    print("=" * 74)
    print(" 怎么读")
    print("=" * 74)
    print("  A 有 / B 有 / C 说命中      -> 宿主没问题，问题在别处")
    print("  A 无 / B 有 / C 也枚举不到  -> 宿主**看不见**游戏窗口")
    print("  A 无 / B 有 / C 枚举到但命中 0 -> psutil 在宿主里抛了（主路径死在这）")
    print("  C 一条都没有                 -> 宿主还没跑够 3 轮 miss，或插件版本旧")
    print()
    print("  提醒：游戏**最小化**时三条通道都拿不到（客户区退化成 0x0），")
    print("        这不是 bug —— 排查前先确认游戏在前台显示。")
    return 0


if __name__ == "__main__":
    time.sleep(0.1)
    raise SystemExit(main())