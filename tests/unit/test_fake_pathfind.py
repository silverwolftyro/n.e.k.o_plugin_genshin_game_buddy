"""``tools/fake_pathfind.py`` 单测（插件侧，2026-10-12）。

这个脚本是**工具**（给用户离线验 Console 寻路用的），但它的参数解析是
「与真脚本 e2e_pathfind.py 保持一致」这条约定的载体 —— 一旦漂移，
Console 的端到端测试测的就不是同一套契约了。所以把可测的部分钉住。

需要真子进程的行为（写状态、响应 flag、崩溃）由 Console 侧
``tests/test_pathfind_e2e_offline.py`` 用**同一个文件**覆盖，
本文件只测不需要起进程的部分，避免两边重复又容易过期。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autopilot.tools import fake_pathfind as FP


# ==================================================================
# 参数解析（照真脚本 _SPEC 的语义）
# ==================================================================

def test_parse__requires_start_and_target() -> None:
    with pytest.raises(SystemExit) as ei:
        FP._parse_args(["--start-world", "1,2"])
    assert ei.value.code == 2


def test_parse__reads_the_args_console_actually_sends() -> None:
    """★ 这四个就是 ``PathfindChannel.start`` 会拼出来的。"""
    a = FP._parse_args(["--start-world", "1,2", "--target-world", "3,4",
                        "--live", "--delay", "5"])
    assert a["start"] == (1.0, 2.0) and a["target"] == (3.0, 4.0)
    assert a["live"] is True and a["delay"] == 5.0


def test_parse__live_defaults_to_false() -> None:
    """★ 默认**不** live —— 与真脚本同一哲学：会发按键的事必须显式。"""
    a = FP._parse_args(["--start-world", "0,0", "--target-world", "1,1"])
    assert a["live"] is False


def test_parse__walk_turns_live_off() -> None:
    a = FP._parse_args(["--start-world", "0,0", "--target-world", "1,1",
                        "--live", "--walk"])
    assert a["live"] is False


def test_parse__unknown_flag_is_skipped_quietly() -> None:
    """★ 与真脚本一致：未知参数不报错。

    真脚本的解析循环对未知 flag 是 ``continue``。这里如果改成报错，
    Console 侧就会因为「多发了一个它不知道的 flag」而整条链路起不来。
    """
    a = FP._parse_args(["--start-world", "0,0", "--target-world", "1,1",
                        "--future-flag", "whatever"])
    assert a["start"] == (0.0, 0.0)


def test_parse__bare_word_becomes_title() -> None:
    a = FP._parse_args(["原神", "--start-world", "0,0",
                        "--target-world", "1,1"])
    assert a["title"] == "原神"


def test_parse__bad_pair_exits_2() -> None:
    with pytest.raises(SystemExit) as ei:
        FP._parse_args(["--start-world", "abc", "--target-world", "1,1"])
    assert ei.value.code == 2


def test_parse__rejects_unknown_fake_mode() -> None:
    """★ 模式名写错要立刻报错，不能默默退回 arrive。"""
    with pytest.raises(SystemExit) as ei:
        FP._parse_args(["--start-world", "0,0", "--target-world", "1,1",
                        "--fake-mode", "飞起来"])
    assert ei.value.code == 2


def test_parse__defaults() -> None:
    a = FP._parse_args(["--start-world", "0,0", "--target-world", "1,1"])
    assert a["fake_mode"] == "arrive"
    assert a["tick_ms"] == 500.0
    assert a["seconds"] == 8.0


# ==================================================================
# 运行时定位 —— 与真脚本同一套约定
# ==================================================================

def test_paths__derive_from_plugin_root_env(monkeypatch, tmp_path) -> None:
    """★ 改环境变量就能把整条链路指到别处 —— 与真脚本 ``NEKO_PLUGIN_ROOT`` 同名同义。"""
    root = tmp_path / "fake_root"
    (root / "autopilot" / "tools").mkdir(parents=True)
    monkeypatch.setenv("NEKO_PLUGIN_ROOT", str(root))
    import importlib

    mod = importlib.reload(FP)
    try:
        assert mod.RUNTIME_DIR == root / "autopilot" / "_runtime"
        assert mod.STATE_FILE.name == "pathfind_state.json"
        assert mod.STOP_FILE.name == "nav_stop.flag"
        assert mod.RUNTIME_DIR == mod.STATE_FILE.parent
    finally:
        monkeypatch.delenv("NEKO_PLUGIN_ROOT", raising=False)
        importlib.reload(FP)


def test_paths__state_and_stop_are_siblings_in_runtime() -> None:
    """★ Console 那边是 ``runtime_dir()/name`` 拼出来的，位置必须一致。"""
    assert FP.STATE_FILE.parent == FP.STOP_FILE.parent == FP.RUNTIME_DIR
    assert FP.RUN_CURRENT_FILE.parent == FP.RUNTIME_DIR


def test_paths__schema_version_matches_the_real_script() -> None:
    """★ 与真脚本 ``NAV_SCHEMA_VERSION`` 同为 2。

    overlay 靠它挑解析器；对不上时它**不显示**（宁可少显示也不拿错字段渲染）。
    """
    assert FP.NAV_SCHEMA_VERSION == 2


# ==================================================================
# 端到端（起真进程，但只测最关键的两条）
# ==================================================================

def _run(tmp_path: Path, argv: list[str], timeout: float = 30.0):
    env = {**os.environ, "NEKO_PLUGIN_ROOT": str(tmp_path),
           "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(Path(FP.__file__).resolve()), *argv],
        capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace", env=env)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "autopilot" / "_runtime").mkdir(parents=True)
    return tmp_path


def test_run__arrive_writes_state_then_exits_zero(root) -> None:
    r = _run(root, ["--start-world", "0,0", "--target-world", "10,10",
                    "--live", "--seconds", "1", "--fake-tick-ms", "200"])
    assert r.returncode == 0, r.stdout + r.stderr
    st = json.loads((root / "autopilot" / "_runtime"
                     / "pathfind_state.json").read_text(encoding="utf-8"))
    assert st["state"] == "arrived"
    assert st["fake"] is True


def test_run__crash_leaves_navigating_behind(root) -> None:
    """★ 崩溃**不改**状态文件 —— 留下的最后一帧是 navigating。

    这正是 Console 侧「陈旧比没有状态更危险」要覆盖的场景。
    """
    r = _run(root, ["--start-world", "0,0", "--target-world", "10,10",
                    "--live", "--fake-mode", "crash",
                    "--fake-crash-after", "2", "--fake-tick-ms", "200"])
    assert r.returncode == 9
    st = json.loads((root / "autopilot" / "_runtime"
                     / "pathfind_state.json").read_text(encoding="utf-8"))
    assert st["state"] == "navigating"


def test_run__stale_stop_flag_is_cleared_on_startup(root) -> None:
    """★ 上次的 flag 不清掉，一起来就秒停。

    ``stuck`` 模式本来就不自己退出，所以这里用 Popen + 轮询，
    不能用 ``subprocess.run``（它会直接抛 TimeoutExpired）。
    """
    import time

    flag = root / "autopilot" / "_runtime" / "nav_stop.flag"
    flag.write_text("上次残留", encoding="utf-8")

    env = {**os.environ, "NEKO_PLUGIN_ROOT": str(root),
           "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen(
        [sys.executable, str(Path(FP.__file__).resolve()),
         "--start-world", "0,0", "--target-world", "10,10", "--live",
         "--fake-mode", "stuck", "--fake-tick-ms", "200"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    state_f = root / "autopilot" / "_runtime" / "pathfind_state.json"
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if state_f.exists():
                break
            time.sleep(0.05)
        # 写出了 navigating 说明启动时没被残留 flag 秒停
        st = json.loads(state_f.read_text(encoding="utf-8"))
        assert st["state"] == "navigating"
        assert p.poll() is None, "进程还活着（stuck 模式本该一直跑）"
    finally:
        p.kill()
        p.wait(timeout=5)


def test_run__events_are_appended_to_the_stream(root) -> None:
    _run(root, ["--start-world", "0,0", "--target-world", "4,4",
                "--live", "--seconds", "1", "--fake-tick-ms", "200"])
    run_id = (root / "autopilot" / "_runtime"
              / "run_current.txt").read_text(encoding="utf-8").strip()
    lines = (root / "autopilot" / "_runtime"
             / f"{run_id}.stream.jsonl").read_text(encoding="utf-8").splitlines()
    codes = [json.loads(l)["code"] for l in lines if l.strip()]
    assert codes[0] == "PATHFIND_START"
    assert "PATHFIND_ARRIVED" in codes
