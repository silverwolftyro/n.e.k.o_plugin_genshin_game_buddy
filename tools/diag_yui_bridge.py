#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""diag_yui_bridge.py — 诊断「事件流 -> YUI」这条链路到底通没通。

只读脚本：不改任何插件文件、不改事件流、不碰 NEKO 进程。
所有结论都打印证据行，看不懂可以直接核对原文。

用法（在任意位置）：
    python tools/diag_yui_bridge.py
    python tools/diag_yui_bridge.py --tail 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:  # 控制台默认 GBK，强制 UTF-8，否则中文证据会炸成乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - 老解释器兜底
    pass


# ---------------------------------------------------------------- 常量区

PLUGIN_DIR = Path(__file__).resolve().parents[1]          # 源码插件目录
SOURCE_RUNTIME = PLUGIN_DIR / "autopilot" / "_runtime"   # 源码树里的事件流目录

NEKO_LOCAL = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\alpha\AppData\Local")) / "N.E.K.O"
NEKO_ROAMING = Path(os.environ.get("APPDATA", r"C:\Users\alpha\AppData\Roaming")) / "N.E.K.O"
INSTALL_ROOT = NEKO_LOCAL / ".neko-plugin-installations" / "plugins"
INSTALLED_PLUGIN = INSTALL_ROOT / "genshin_game_buddy"

ENV_SWITCH = "MGC_EVENT_TO_YUI"
ADAPTER_MODULE = "event_to_yui.py"

# 适配器在日志里会打的标记（任一命中 = 它跑起来过）
LOG_MARKERS = ["事件流", "event_to_yui", "YUI 播报", ENV_SWITCH]

PLUGIN_PROCESS_NAMES = ("N.E.K.O.exe", "projectneko_server.exe", "browser.exe")


# ---------------------------------------------------------------- 小工具

def hr(title: str = "") -> None:
    if title:
        print()
        print("=" * 78)
        print(title)
        print("=" * 78)
    else:
        print("-" * 78)


def mt(path: Path) -> str:
    try:
        return time.strftime("%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
    except OSError:
        return "?"


def sz(path: Path) -> str:
    try:
        return str(path.stat().st_size)
    except OSError:
        return "?"


# ---------------------------------------------------------------- 1. 进程

def check_processes() -> list[dict]:
    hr("1. NEKO 主程序进程")
    ps = (
        "Get-CimInstance Win32_Process | "
        "Select-Object Name,ProcessId,ExecutablePath,CreationDate,CommandLine | "
        "ConvertTo-Json -Compress -Depth 3"
    )
    try:
        raw = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, timeout=60,
        )
        data = json.loads(raw.stdout.decode("utf-8", errors="replace") or "[]")
    except Exception as exc:
        print("  [WARN] 进程查询失败：%s" % exc)
        return []
    if isinstance(data, dict):
        data = [data]

    rows = [
        p for p in data
        if str(p.get("Name", "")).lower() in {n.lower() for n in PLUGIN_PROCESS_NAMES}
    ]
    if not rows:
        print("  [FAIL] 没找到 N.E.K.O.exe / projectneko_server.exe —— NEKO 主程序没在跑")
        return []

    for p in sorted(rows, key=lambda x: str(x.get("CreationDate", ""))):
        print("  %-26s pid=%-7s 启动=%s" % (p.get("Name"), p.get("ProcessId"), p.get("CreationDate")))
        exe = str(p.get("ExecutablePath") or "")
        if exe:
            print("      exe: %s" % exe)
    main = [p for p in rows if str(p.get("Name", "")).lower() == "n.e.k.o.exe" and "--type=" not in str(p.get("CommandLine", ""))]
    if main:
        exe = str(main[0].get("ExecutablePath") or "")
        print()
        print("  主进程 exe = %s" % exe)
        if "steamapps" in exe.lower() or ".exe" in exe.lower():
            print("  -> 这是【打包后的 Electron 客户端】，不是源码树直跑。")
            print("     它读的是 .neko-plugin-installations，不读 D:\\NEKO 源码（见第 2 节）。")
    return rows


# ---------------------------------------------- 2. 源码 vs 安装副本（核心）

def check_code_deployment() -> bool:
    hr("2. 源码副本 vs 运行中安装副本（本次最关键的一节）")
    print("  源码副本  : %s" % PLUGIN_DIR)
    print("  安装副本  : %s" % INSTALLED_PLUGIN)
    print()

    src_adapter = PLUGIN_DIR / ADAPTER_MODULE
    inst_adapter = INSTALLED_PLUGIN / ADAPTER_MODULE
    print("  源码 %-18s exists=%-5s  mtime=%s  size=%s"
          % (ADAPTER_MODULE, src_adapter.exists(), mt(src_adapter), sz(src_adapter)))
    print("  安装 %-18s exists=%-5s  mtime=%s  size=%s"
          % (ADAPTER_MODULE, inst_adapter.exists(), mt(inst_adapter), sz(inst_adapter)))

    src_init = PLUGIN_DIR / "__init__.py"
    inst_init = INSTALLED_PLUGIN / "__init__.py"
    print()
    print("  __init__.py 源码  mtime=%s size=%s" % (mt(src_init), sz(src_init)))
    print("  __init__.py 安装  mtime=%s size=%s" % (mt(inst_init), sz(inst_init)))

    src_has = src_adapter.exists()
    inst_has = inst_adapter.exists()
    drifted = (src_has != inst_has) or (src_has and inst_has and mt(src_init) != mt(inst_init))
    print()
    if inst_has is False and src_has:
        print("  [根因] 适配器只存在于源码树，运行中的安装副本里根本没有这个文件。")
        print("         -> 跑着的 NEKO 从未加载过 event_to_yui，环境变量设了也没用。")
    elif mt(src_init) != mt(inst_init):
        print("  [WARN] 安装副本明显落后于源码（接线代码不在运行版本里）。")
    if drifted:
        print("  -> 结论：改动【没有部署】到运行中的 NEKO。必须 build + 重新导入 + 重启。")
    else:
        print("  -> 代码副本一致（若适配器仍无输出，问题在别处，见第 4/5 节）。")
    return drifted


# ------------------------------------------------- 3. 事件流目录归属

def check_runtime_dirs() -> None:
    hr("3. 事件流目录归属")
    inst_runtime = INSTALLED_PLUGIN / "autopilot" / "_runtime"
    for label, d in (("源码树", SOURCE_RUNTIME), ("安装副本", inst_runtime)):
        print("  [%s] %s" % (label, d))
        print("      exists=%s" % d.exists())
        if not d.exists():
            continue
        try:
            files = [f for f in d.iterdir() if f.is_file()]
        except OSError as exc:
            print("      读取失败：%s" % exc)
            continue
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[:6]:
            print("      %10s  %s  %s" % (sz(f), mt(f), f.name))
        if not files:
            print("      (空)")
    print()
    print("  适配器里的 runtime_dir 是从模块位置算的（DEFAULT_RUN_DIR）。")
    print("  -> 部署后它会盯【安装副本】的 _runtime，不是 D:\\NEKO 那个。")
    print("  -> 手动追加事件必须追加到安装副本的目录，否则永远读不到。")


# ------------------------------------------------------ 4. 事件流内容

def check_stream(tail: int) -> None:
    hr("4. 当前事件流内容")
    for label, runtime in (("源码树", SOURCE_RUNTIME),
                           ("安装副本", INSTALLED_PLUGIN / "autopilot" / "_runtime")):
        print("  [%s]" % label)
        cur = runtime / "run_current.txt"
        if not cur.exists():
            print("      没有 run_current.txt")
            continue
        run_id = cur.read_text("utf-8", errors="replace").strip()
        print("      run_current.txt -> %s" % run_id)
        stream = runtime / ("%s.stream.jsonl" % run_id)
        if not stream.exists():
            print("      找不到 %s" % stream.name)
            continue
        print("      事件流 %s  size=%s  mtime=%s"
              % (stream.name, sz(stream), mt(stream)))
        raw = stream.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        lines = [L for L in text.split("\n") if L.strip()]
        bad = 0
        for L in lines:
            try:
                json.loads(L)
            except Exception:
                bad += 1
        print("      行数=%d  非法JSON行=%d  以换行结尾=%s"
              % (len(lines), bad, raw.endswith(b"\n")))
        print("      ---- 最近 %d 条 ----" % tail)
        for L in lines[-tail:]:
            try:
                o = json.loads(L)
                ctx = o.get("context")
                extra = ""
                if isinstance(ctx, dict):
                    extra = " ctx=" + json.dumps(ctx, ensure_ascii=False)[:120]
                print("      seq=%-4s %-18s ts_ms=%-14s%s"
                      % (o.get("seq"), o.get("code"), o.get("ts_ms"), extra))
            except Exception as exc:
                print("      [非法] %s :: %s" % (exc, L[:160]))
    print()
    print("  注意：手写的事件 ts_ms 若是 0，且插件没在跑主循环，也仍然应该被")
    print("        适配器读到 —— 适配器是独立追尾线程，不依赖 core loop。")


# ------------------------------------------------------ 5. 日志搜索

def check_logs() -> None:
    hr("5. NEKO 日志搜索")
    log_dir = NEKO_LOCAL / "logs"
    print("  日志目录: %s  exists=%s" % (log_dir, log_dir.exists()))
    if not log_dir.exists():
        return
    logs = sorted(log_dir.rglob("*.log"))
    print("  日志文件数: %d" % len(logs))

    hit_any = False
    for f in logs:
        try:
            txt = f.read_text("utf-8", errors="replace")
        except OSError:
            continue
        found = {m: txt.count(m) for m in LOG_MARKERS if m in txt}
        if found:
            hit_any = True
            print("  [命中] %-58s %s" % (str(f.relative_to(log_dir)), found))

    plugin_log = log_dir / "plugin" / (
        "N.E.K.O_Plugin_genshin_game_buddy_%s.log" % time.strftime("%Y%m%d"))
    if plugin_log.exists():
        print()
        print("  插件日志: %s  size=%s  mtime=%s" % (plugin_log, sz(plugin_log), mt(plugin_log)))
        lines = plugin_log.read_text("utf-8", errors="replace").splitlines()
        print("  头部 3 行（看插件从哪儿加载）：")
        for L in lines[:3]:
            print("    " + L[:240])
        print("  尾部 3 行（看现在还活着没）：")
        for L in lines[-3:]:
            print("    " + L[:240])
    print()
    if not hit_any:
        print("  [结论] 全部日志里 %s 四个标记一个都没有。" % " / ".join(LOG_MARKERS))
        print("         连「未启用」那行都没打出来 -> 说明带这个分支的代码")
        print("         压根没被加载（印证第 2 节）。")


# ------------------------------------------------------ 6. 环境变量

def check_env() -> None:
    hr("6. %s 环境变量" % ENV_SWITCH)
    here = os.environ.get(ENV_SWITCH)
    print("  当前 shell: %s" % ("未设置" if here is None else "=%s" % here))
    try:
        import psutil  # type: ignore
    except Exception:
        print("  [跳过] 读其它进程环境需要 psutil，本机没有。")
        print("         改用日志证据判断（见第 5 节）。")
        return
    try:
        found = False
        for p in psutil.process_iter(["pid", "name", "environ"]):
            try:
                env = p.info.get("environ") or {}
            except Exception:
                continue
            if ENV_SWITCH in env:
                found = True
                print("  [命中] pid=%s %s -> %s=%s"
                      % (p.info["pid"], p.info.get("name"), ENV_SWITCH, env[ENV_SWITCH]))
        if not found:
            print("  没有任何进程带这个变量。")
    except Exception as exc:
        print("  [WARN] psutil 查询失败：%s" % exc)


# ------------------------------------------------------ 7. 结论

def verdict(drifted: bool) -> None:
    hr("7. 结论")
    if drifted:
        print("  适配器【没在跑】。根因不是环境变量、不是事件格式、不是 run_id ——")
        print("  是运行中的 NEKO 加载的是 09-29 打的安装包，里面没有 event_to_yui.py，")
        print("  __init__.py 也没有 MGC_EVENT_TO_YUI 接线。")
        print()
        print("  修复顺序：")
        print("    1) 在源码根目录打包：")
        print("       python -m plugin.neko_plugin_cli build genshin_game_buddy")
        print("    2) NEKO 插件中心导入 plugin/neko_plugin_cli/target/genshin_game_buddy.neko-plugin")
        print("    3) 设 %s=1 后重启 NEKO（插件代码只在启动时加载）" % ENV_SWITCH)
        print("    4) 事件要追加到【安装副本】的 autopilot/_runtime/<run_id>.stream.jsonl")
    else:
        print("  代码副本一致 —— 若仍无播报，请重点看第 4/5 节：")
        print("    a) 事件是否追加进了【安装副本】的 _runtime，而不是源码树那份")
        print("    b) 日志里有没有 %s=1 的启动标记" % ENV_SWITCH)


def main() -> int:
    ap = argparse.ArgumentParser(description="诊断事件流 -> YUI 链路")
    ap.add_argument("--tail", type=int, default=8, help="事件流显示最近多少条（默认 8）")
    args = ap.parse_args()

    print("诊断时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("插件源码: %s" % PLUGIN_DIR)

    check_processes()
    drifted = check_code_deployment()
    check_runtime_dirs()
    check_stream(args.tail)
    check_logs()
    check_env()
    verdict(drifted)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())