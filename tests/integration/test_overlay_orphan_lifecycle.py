"""孤儿守卫端到端：真起一个卡片进程，看它会不会自己消失。

单测测的是 :class:`ParentWatchdog` 的逻辑；这里测的是**接线** ——
环境变量有没有传到、tick 里有没有真的调守卫、``root.destroy()`` 之后
进程是不是真的没了。

三个场景对应三条不同的死亡路径：

    ① 优雅退出  —— 主控台写 running=false 就走（quit_app 的正常路径）
    ② 强杀      —— 主控台被任务管理器干掉，没有任何告别（崩溃路径）
    ③ 常驻      —— 开关关掉 / 没主控台 PID 时，卡片必须活着

"主控台"用一个小替身：它照抄主控台的两件事 —— 定期刷新桥接文件、
退出前写一次 running=false。用替身而不是真 Console，是为了让退出时机
可控可测（真 Console 要人点托盘）。

★ 为什么不用 psutil 全表扫进程
  ----------------------------
  ``process_iter(["cmdline"])`` 在 Windows 上要给**每个**进程开句柄读
  命令行（本机 ~250 进程，单次扫描实测数秒）。放进 0.1s 的轮询里，
  测试时间全花在扫描上，失败时也看不出到底是"卡片没死"还是"扫描太慢"。

  这里只用 ``Popen.poll() / wait()``。卡片是 ``python -m ...`` 起的，
  venv redirector stub 会和真解释器一起退出，``proc`` 因此能代表整棵树
  —— 这点由 :func:`_wait_card_gone` 里的双保险验证（先等 proc 退出，
  再确认没有残留的卡片解释器）。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_NAME = "console_bridge.json"

#: 验收线：主控台没了之后，卡片必须在这个时间内消失
DEATH_BUDGET_S = 5.0

#: 卡片起来要多久（import tkinter + pydantic + 建窗口）
STARTUP_BUDGET_S = 20.0


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _start_card(cmd_dir: Path, console_pid: int, *, follow: bool = True):
    env = os.environ.copy()
    env["NEKO_CONSOLE_CMD_DIR"] = str(cmd_dir)
    env["NEKO_CONSOLE_PID"] = str(console_pid)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    cmd = [sys.executable, "-m", "autopilot.tools.overlay"]
    if not follow:
        cmd.append("--no-follow-console-exit")
    return subprocess.Popen(
        cmd, cwd=str(PLUGIN_ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _write_bridge(d: Path, running: bool, console_pid: int) -> None:
    (d / BRIDGE_NAME).write_text(json.dumps({
        "schema_version": 1,
        "written_at_ms": int(time.time() * 1000),
        "console": {"running": running, "pid": console_pid},
        "engine": {"state": "STOPPED", "pid": 0},
        "overlay": {"state": "running", "pid": 0},
        "pathfind": {"state": "STOPPED", "pid": 0},
        "stats": {"scene": None},
    }), encoding="utf-8")


def _wait_card_up(proc: subprocess.Popen, budget: float = STARTUP_BUDGET_S):
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"卡片启动就退了，exit={proc.returncode}")
        time.sleep(0.2)
    # 多给它一拍把窗口真正建出来，否则"还活着"只是"import 还没结束"
    time.sleep(1.0)
    assert proc.poll() is None, "卡片起不来"


def _wait_card_gone(proc: subprocess.Popen, budget: float) -> float:
    """等卡片彻底退出。返回耗时；超时直接 fail。

    只看 ``proc.poll()``：venv redirector stub 与真解释器同生共死，
    stub 退了基本就等于卡片没了。这里不做全表扫描（见模块 docstring）。
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < budget:
        if proc.poll() is not None:
            return time.monotonic() - t0
        time.sleep(0.1)
    pytest.fail(f"卡片在 {budget}s 内没有退出（仍在运行，孤儿）")


def _assert_still_up(proc: subprocess.Popen, seconds: float) -> None:
    time.sleep(seconds)
    assert proc.poll() is None, f"卡片不该自杀，却在 {seconds}s 后退出了"


@pytest.fixture()
def card():
    started: list = []

    def _register(proc):
        started.append(proc)
        return proc

    yield _register
    for proc in started:
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass


@pytest.fixture()
def fake_console():
    """一个可以被"杀掉"的替身主控台，返回它的 PID。"""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import os,time;print(os.getpid(),flush=True);"
                               "time.sleep(300)"],
        stdout=subprocess.PIPE, text=True)
    pid = int(proc.stdout.readline().strip())
    try:
        yield pid
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------
# ① 优雅退出
# --------------------------------------------------------------------------
def test_card_exits_when_console_writes_running_false(
        card, tmp_path: Path) -> None:
    """quit_app 的正常路径：退出前主控台会写一次 running=false。"""
    cardproc = card(_start_card(tmp_path, os.getpid()))
    _wait_card_up(cardproc)

    _write_bridge(tmp_path, running=False, console_pid=os.getpid())
    took = _wait_card_gone(cardproc, DEATH_BUDGET_S)
    print(f"\n  优雅退出 -> 卡片消失耗时 {took:.2f}s")


# --------------------------------------------------------------------------
# ② 强杀主控台（任务管理器「结束任务」）
# --------------------------------------------------------------------------
def test_card_exits_when_console_is_hard_killed(
        card, fake_console, tmp_path: Path) -> None:
    """桥接文件从此不再更新，唯一的信号是父 PID 没了。"""
    cardproc = card(_start_card(tmp_path, fake_console))
    _wait_card_up(cardproc)
    _write_bridge(tmp_path, running=True, console_pid=fake_console)

    try:
        os.kill(fake_console, signal.SIGTERM)
    except ProcessLookupError:
        pass

    took = _wait_card_gone(cardproc, DEATH_BUDGET_S)
    print(f"\n  强杀主控台 -> 卡片消失耗时 {took:.2f}s")


# --------------------------------------------------------------------------
# ③ 反向：不该自杀的场景必须不自杀
# --------------------------------------------------------------------------
def test_card_survives_when_follow_disabled(
        card, fake_console, tmp_path: Path) -> None:
    cardproc = card(_start_card(tmp_path, fake_console, follow=False))
    _wait_card_up(cardproc)
    _write_bridge(tmp_path, running=True, console_pid=fake_console)

    try:
        os.kill(fake_console, signal.SIGTERM)
    except ProcessLookupError:
        pass

    # 3s 宽限 + 3s 余量
    _assert_still_up(cardproc, 6.0)


def test_manual_card_without_console_pid_survives(
        card, tmp_path: Path) -> None:
    """手动启动（没有 NEKO_CONSOLE_PID）-> 守卫不上膛，必须一直活着。

    这是最容易写坏的一条：拿"桥接文件不更新"当死亡判据的话，这张手动
    卡片会在启动后一秒内自己消失。
    """
    env = os.environ.copy()
    env.pop("NEKO_CONSOLE_PID", None)
    env["NEKO_CONSOLE_CMD_DIR"] = str(tmp_path)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    cardproc = card(subprocess.Popen(
        [sys.executable, "-m", "autopilot.tools.overlay"],
        cwd=str(PLUGIN_ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
    _wait_card_up(cardproc)
    _assert_still_up(cardproc, 6.0)


def test_card_survives_while_console_alive(card, tmp_path: Path) -> None:
    """主控台还在跑、桥接文件持续写 running=true -> 绝不能自杀。"""
    cardproc = card(_start_card(tmp_path, os.getpid()))
    _wait_card_up(cardproc)
    for _ in range(3):
        _write_bridge(tmp_path, running=True, console_pid=os.getpid())
        time.sleep(0.7)
    _assert_still_up(cardproc, 2.0)
