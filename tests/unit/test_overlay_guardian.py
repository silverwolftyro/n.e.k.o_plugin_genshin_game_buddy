"""孤儿守卫（autopilot/ui/overlay/guardian.py）的单测。

守卫生死存亡靠两条判据，都得测到边界：

    ① 桥接文件写 running=false -> 立即退，不等宽限
    ② 父 PID 消失              -> 累计超过宽限才退

以及三个**不能误杀**的场景：手动启动没上膛、查询被拒当成活着、
父进程"假死一下又活回来"要把计时清掉。
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

from autopilot.ui.overlay import guardian
from autopilot.ui.overlay.guardian import (
    REASON_GRACEFUL,
    REASON_PARENT_DEAD,
    ParentWatchdog,
    pid_alive,
)


def _kill_now(pid: int) -> None:
    """按 PID 硬杀，并等它真的从进程表里消失。

    ``os.kill(pid, SIGTERM)`` 在 Windows 上走 TerminateProcess，
    对任意 PID（包括子解释器）都成立 —— 这正是我们要模拟的
    「任务管理器里强杀」。
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return
        time.sleep(0.05)
    pytest.fail(f"进程 {pid} 在 5s 内没有消失")


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------
def _write_bridge(path: Path, running: bool | None) -> Path:
    payload: dict = {
        "schema_version": 1,
        "written_at_ms": 0,
        "console": {"running": running, "pid": 1234},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def bridge(tmp_path: Path) -> Path:
    return _write_bridge(tmp_path / "console_bridge.json", True)


@pytest.fixture()
def sleeper():
    """一个活着的长进程，测完必须收干净。

    ★ 要它**自己报 PID**，不能直接用 ``Popen.pid``
      --------------------------------------------
      ``.venv\\Scripts\\python.exe`` 是 47KB 的重定向 stub，真解释器是
      它的**子进程**（实测 91648 字节那个）。拿 stub 的 PID 测存活，
      测的是 stub 而不是"卡片真正所在的那个解释器"。
    """
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import os,time;print(os.getpid(),flush=True);time.sleep(300)"],
        stdout=subprocess.PIPE, text=True,
    )
    real_pid = int(proc.stdout.readline().strip())
    try:
        yield real_pid
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


# --------------------------------------------------------------------------
# pid_alive
# --------------------------------------------------------------------------
def test_pid_alive_true_for_self():
    assert pid_alive(os.getpid()) is True


def test_pid_alive_false_for_impossible_pid():
    # 0 和负数不是合法进程号
    assert pid_alive(0) is False
    assert pid_alive(-1) is False


def test_pid_alive_false_after_process_exits(sleeper):
    pid = sleeper
    assert pid_alive(pid) is True
    _kill_now(sleeper)
    assert pid_alive(pid) is False


# --------------------------------------------------------------------------
# 上膛条件
# --------------------------------------------------------------------------
def test_not_armed_without_console_pid():
    """手动启动的卡片没有主控台 PID —— 守卫必须完全不设防。"""
    w = ParentWatchdog(None, None, follow_exit=True)
    assert w.armed is False
    # 反复 check 也不该有任何反应
    assert [w.check(now=t) for t in (0, 1, 10, 1000)] == [None] * 4


def test_not_armed_when_follow_exit_disabled(sleeper):
    """用户显式关掉开关 -> 有 PID 也不盯。"""
    w = ParentWatchdog(sleeper, None, follow_exit=False)
    assert w.armed is False
    assert w.check(now=0) is None
    assert w.check(now=999) is None


def test_armed_with_pid_and_switch_on(sleeper):
    assert ParentWatchdog(sleeper, None, follow_exit=True).armed is True


# --------------------------------------------------------------------------
# ① 优雅退出：running=false 立即退，不等宽限
# --------------------------------------------------------------------------
def test_graceful_signal_exits_immediately(tmp_path: Path):
    """第一轮 check 就退 —— 宽限是给崩溃用的，优雅路径没有误报风险。"""
    b = _write_bridge(tmp_path / "console_bridge.json", False)
    w = ParentWatchdog(os.getpid(), b, grace_s=5.0)
    # 父进程明明活着（就是自己），但 running=false 优先级更高
    assert w.check(now=0.0) == REASON_GRACEFUL


def test_graceful_signal_ignores_grace_window(tmp_path: Path):
    b = _write_bridge(tmp_path / "console_bridge.json", False)
    w = ParentWatchdog(os.getpid(), b, grace_s=999.0)
    assert w.check(now=0.0) == REASON_GRACEFUL


def test_bridge_running_true_keeps_card_alive(sleeper, bridge):
    w = ParentWatchdog(sleeper, bridge, grace_s=5.0)
    assert w.check(now=0.0) is None
    assert w.check(now=100.0) is None


# --------------------------------------------------------------------------
# 读不到桥接文件时必须"别猜"
# --------------------------------------------------------------------------
def test_missing_bridge_file_is_not_death_signal(sleeper, tmp_path: Path):
    w = ParentWatchdog(sleeper, tmp_path / "nope.json", grace_s=5.0)
    assert w.check(now=0.0) is None


def test_corrupt_bridge_file_is_not_death_signal(sleeper, tmp_path: Path):
    p = tmp_path / "console_bridge.json"
    p.write_text("{ this is not json", encoding="utf-8")
    w = ParentWatchdog(sleeper, p, grace_s=5.0)
    assert w.check(now=0.0) is None


def test_bridge_without_console_key_is_not_death_signal(sleeper, tmp_path: Path):
    p = tmp_path / "console_bridge.json"
    p.write_text(json.dumps({"engine": {"state": "RUNNING"}}), encoding="utf-8")
    w = ParentWatchdog(sleeper, p, grace_s=5.0)
    assert w.check(now=0.0) is None


def test_bridge_non_bool_running_is_not_death_signal(sleeper, tmp_path: Path):
    """running 写成字符串 "false" 也**不能**当退出信号 —— 契约坏了就别猜。"""
    p = tmp_path / "console_bridge.json"
    p.write_text(json.dumps({"console": {"running": "false"}}),
                 encoding="utf-8")
    w = ParentWatchdog(sleeper, p, grace_s=5.0)
    assert w.check(now=0.0) is None


# --------------------------------------------------------------------------
# ② 崩溃：父进程没了，累计超过宽限才退
# --------------------------------------------------------------------------
def test_parent_dead_waits_grace_then_exits(sleeper, tmp_path: Path):
    p = sleeper
    _kill_now(sleeper)

    w = ParentWatchdog(p, None, grace_s=5.0)
    assert w.check(now=0.0) is None, "刚消失就该有宽限期，不能立刻自杀"
    assert w.check(now=1.0) is None
    assert w.check(now=4.99) is None, "还没到宽限"
    assert w.check(now=5.0) == REASON_PARENT_DEAD


def test_parent_dead_recovery_clears_timer(sleeper, tmp_path: Path):
    """假死一下又活回来 -> 计时必须清零，不能累积到自杀。"""
    p = sleeper
    w = ParentWatchdog(p, None, grace_s=5.0)

    # 先让它"死"两轮
    assert w.check(now=0.0) is None
    assert w.check(now=2.0) is None

    # 活着的每一轮都要清计时（这里 sleeper 还活着）
    assert w.check(now=3.0) is None

    # 现在真死。**计时从"第一次发现它死了"那一刻起算**，不是从 0.0 ——
    # 这才是"假死恢复后计时要清零"的真正含义：清掉的是上一段死区间的
    # 累计，不是让新死区间继承旧进度。
    _kill_now(sleeper)
    assert w.check(now=7.9) is None, "刚发现死，起计时"
    assert w.check(now=12.89) is None, "4.99s < 5s 宽限"
    assert w.check(now=12.9) == REASON_PARENT_DEAD


def test_parent_alive_forever_never_exits(sleeper, tmp_path: Path):
    w = ParentWatchdog(sleeper, None, grace_s=0.1)
    for t in (0, 1, 2, 3, 100, 1000):
        assert w.check(now=float(t)) is None


def test_zero_grace_exits_on_first_dead_poll(sleeper):
    """宽限设 0 = 立刻退（测试和极端场景用）。"""
    p = sleeper
    _kill_now(sleeper)
    w = ParentWatchdog(p, None, grace_s=0.0)
    assert w.check(now=0.0) == REASON_PARENT_DEAD


# --------------------------------------------------------------------------
# from_env
# --------------------------------------------------------------------------
def test_from_env_reads_pid(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(guardian.CONSOLE_PID_ENV, str(os.getpid()))
    w = ParentWatchdog.from_env(tmp_path)
    assert w.armed is True
    assert w.console_pid == os.getpid()
    assert w.bridge_path == tmp_path / guardian.BRIDGE_FILE_NAME


def test_from_env_without_pid_is_disarmed(monkeypatch, tmp_path: Path):
    monkeypatch.delenv(guardian.CONSOLE_PID_ENV, raising=False)
    w = ParentWatchdog.from_env(tmp_path)
    assert w.armed is False


def test_from_env_ignores_garbage_pid(monkeypatch, tmp_path: Path):
    """PID 写成 "abc" 不能让 os.getpid() 之类的东西误判成有效。"""
    monkeypatch.setenv(guardian.CONSOLE_PID_ENV, "not-a-pid")
    assert ParentWatchdog.from_env(tmp_path).armed is False
    monkeypatch.setenv(guardian.CONSOLE_PID_ENV, "-1")
    assert ParentWatchdog.from_env(tmp_path).armed is False


def test_from_env_without_cmd_dir(monkeypatch):
    monkeypatch.setenv(guardian.CONSOLE_PID_ENV, str(os.getpid()))
    w = ParentWatchdog.from_env(None)
    assert w.armed is True
    assert w.bridge_path is None


def test_env_var_name_matches_console_side():
    """两侧常量必须同名字符串 —— 对不上守卫会静默失效。"""
    console_src = Path(r"D:\workspace\N.E.K.O\NEKO-genshin\NEKO-Console\neko_console\channels\overlay.py")
    if not console_src.is_file():
        pytest.skip("主控台源码不在本机")
    assert 'CONSOLE_PID_ENV = "NEKO_CONSOLE_PID"' in console_src.read_text(
        encoding="utf-8")
    assert guardian.CONSOLE_PID_ENV == "NEKO_CONSOLE_PID"
