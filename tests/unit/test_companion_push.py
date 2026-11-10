"""``companion_push.py`` 的契约测试（2026-10-20）。

★ 这组用例守的是**合规红线**，不只是功能
  --------------------------------------
  合作方的要求是「项目必须杜绝任何『自动代打』的嫌疑」。脚本本身是
  合规工具 —— 它只往事件流里追加一句话，不按键、不截图、不连招。

  但「注释里写着不碰」不是保证。所以这里把红线写成**可执行的断言**：

    1. 脚本的 import 全是标准库（AST 扫源码，不是扫运行时）
    2. 运行时自检：进程里出现按键/截图类模块 -> 拒绝运行
    3. 脚本只写「事件流文件」这一种路径

  将来谁想在这个脚本里加一行 ``import pyautogui``，这里立刻红。

脚本本体在 ``_release_build/`` 下（发行辅助工具，不进插件运行时）。
找不到时整组 skip —— 它不在插件树里，不是插件的一部分。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

#: 脚本可能所在的目录。``_release_build`` 是发行辅助目录，不在插件树内。
_HERE = Path(__file__).resolve()
_CANDIDATES = [
    Path(r"D:\workspace\N.E.K.O\NEKO-genshin\_构建与发布\_构建脚本\companion_push.py"),
    _HERE.parents[7] / "_release_build" / "companion_push.py",
    _HERE.parents[6] / "_release_build" / "companion_push.py",
]


def _find_script() -> Path | None:
    for candidate in _CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


_SCRIPT_PATH = _find_script()

if _SCRIPT_PATH is None:                                  # pragma: no cover
    pytest.skip(
        "companion_push.py 不在 _release_build/ 下（发行辅助目录），跳过",
        allow_module_level=True)


def _load() -> types.ModuleType:
    """按路径加载脚本（它不在插件包里，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(
        "_companion_push_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cp() -> types.ModuleType:
    return _load()


# ==================================================================
# ★ 合规红线
# ==================================================================

class TestCompliance:
    """脚本**不许**有游戏控制能力。"""

    #: 允许出现在脚本 import 里的顶层模块。全部是标准库。
    ALLOWED_STDLIB = {
        "__future__", "argparse", "json", "os", "sys", "time",
        "pathlib", "typing",
    }

    def test_imports_are_stdlib_only(self) -> None:
        """脚本的 import **必须**全是标准库。

        ★ 用 AST 扫，不扫运行时
          扫运行时只能看到「这次执行恰好 import 了什么」——
          条件分支里的 import 就漏了。而 import 列表是**源码属性**，
          评审时看的就是源码。
        """
        tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:          # 相对 import：肯定不是标准库
                    imported.add(f".{node.module or ''}")
                elif node.module:
                    imported.add(node.module.split(".")[0])
        outside = imported - self.ALLOWED_STDLIB
        assert outside == set(), (
            f"companion_push.py 只允许用标准库，发现了 {sorted(outside)}。"
            "任何第三方库（尤其能操作游戏的）都会破坏「只发话」的合规性。")

    def test_no_control_module_names_are_imported(self) -> None:
        """点名检查：按键 / 截图 / 注入类库一个都不能被 import。"""
        source = _SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        banned = {"pyautogui", "pynput", "keyboard", "pygetwindow",
                  "pydirectinput", "cv2", "dxcam", "mss",
                  "win32api", "win32con", "win32gui"}
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported & banned == set(), (
            f"companion_push.py import 了 {sorted(imported & banned)} —— "
            "这个脚本只该发事件。")

    def test_runtime_guard_rejects_control_modules(
            self, cp: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        """进程里出现按键/截图类模块时，脚本**当场拒绝运行**。

        ★ 这是把红线从「注释里的承诺」变成「运行时断言」。
          注释可以骗人，断言不会。
        """
        monkeypatch.setitem(sys.modules, "pyautogui", types.ModuleType("pyautogui"))
        with pytest.raises(SystemExit) as exc:
            cp.assert_no_control_surface()
        assert "合规自检" in str(exc.value)
        assert "pyautogui" in str(exc.value)

    def test_runtime_guard_passes_for_clean_process(
            self, cp: types.ModuleType) -> None:
        """干净进程里自检必须放行（否则脚本自己都跑不了）。

        ★ 显式传模块集合，而不是查真实 ``sys.modules``
          pytest 是**同进程**跑几百条用例的，别处的用例早就 import 过
          ``cv2`` / ``mss`` 了 —— 拿真实 ``sys.modules`` 断言「一定干净」
          必然红，而红的原因跟这个脚本毫无关系。
        """
        cp.assert_no_control_surface(modules=["argparse", "json", "os",
                                              "pathlib", "sys", "time"])

    def test_runtime_guard_matches_reality(self, cp: types.ModuleType) -> None:
        """自检的判定必须**忠实于**真实进程状态。

        ★ 换一种说法：这条断言在「干净进程」和「pytest 全量进程」里
          都必须成立 —— 否则它只在其中一种环境里有意义。
        """
        expected_bad = [m for m in cp.FORBIDDEN_MODULES if m in sys.modules]
        if expected_bad:
            with pytest.raises(SystemExit):
                cp.assert_no_control_surface()
        else:
            cp.assert_no_control_surface()

    def test_runtime_guard_rejects_when_injected(
            self, cp: types.ModuleType) -> None:
        """显式给一个含禁止模块的集合 -> 必须拦下。"""
        with pytest.raises(SystemExit) as exc:
            cp.assert_no_control_surface(modules=["pyautogui", "argparse"])
        assert "pyautogui" in str(exc.value)

    def test_guard_covers_autopilot_execution_layers(self) -> None:
        """自检名单必须覆盖插件的执行层（actuator / core / perception）。"""
        cp = _load()
        for name in ("autopilot.actuator", "autopilot.core",
                     "autopilot.perception"):
            assert name in cp.FORBIDDEN_MODULES

    def test_script_only_writes_the_stream_file(self) -> None:
        """脚本里出现的写操作只有「追加事件流」这一处。

        ★ 用 AST 找所有「写」调用：``write_text`` / ``open(...,"w")`` /
          ``mkdir`` / ``unlink`` 等。``write_text`` 只允许出现在
          「自建 run」那条分支里（写 ``run_current.txt``）——
          而那条分支必须由 ``--new-run`` 显式触发。
        """
        source = _SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        writers: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
                if name in {"write_text", "write_bytes", "unlink", "rmdir",
                            "replace", "rename"}:
                    writers.append(name)
            elif isinstance(func, ast.Name) and func.id in {"mkdir"}:
                writers.append(func.id)
        # 允许的写操作：write_text（run_current.txt）+ mkdir（建目录）
        assert set(writers) <= {"write_text", "mkdir"}, (
            f"companion_push.py 出现了意料之外的写操作：{sorted(set(writers))}。"
            "它只该往事件流追加一行 JSON。")


# ==================================================================
# 运行目录探测
# ==================================================================

class TestRunDirPicking:
    """判定顺序必须与插件 ``_resolve_run_dir`` 一致。"""

    def test_env_is_hard_override(self, cp: types.ModuleType, tmp_path: Path) -> None:
        """环境变量**直接采用**，不参与 mtime 竞争。

        ★ 用户显式声明了目录还让它输给 mtime 竞争，是把明确意图当噪声。
        """
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "a.stream.jsonl").write_text("{}", encoding="utf-8")
        other = tmp_path / "other"
        other.mkdir()
        (other / "b.stream.jsonl").write_text("{}", encoding="utf-8")
        picked, why = cp.pick_run_dir(
            [("候选", other)], env_value=str(env_dir))
        assert picked == env_dir
        assert "环境变量" in why

    def test_sidecar_is_hard_override(
            self, cp: types.ModuleType, tmp_path: Path) -> None:
        """sidecar 同样硬覆盖（排在 env 之后、mtime 竞争之前）。"""
        sidecar_dir = tmp_path / "side"
        sidecar_dir.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        (other / "b.stream.jsonl").write_text("{}", encoding="utf-8")
        picked, why = cp.pick_run_dir(
            [("候选", other)], env_value="", sidecar=sidecar_dir)
        assert picked == sidecar_dir
        assert "sidecar" in why

    def test_newest_stream_wins(self, cp: types.ModuleType, tmp_path: Path) -> None:
        """没有显式声明时，按「stream 文件最新」决胜。"""
        old_dir = tmp_path / "old"
        new_dir = tmp_path / "new"
        old_dir.mkdir()
        new_dir.mkdir()
        (old_dir / "a.stream.jsonl").write_text("{}", encoding="utf-8")
        (new_dir / "b.stream.jsonl").write_text("{}", encoding="utf-8")
        import os
        os.utime(old_dir / "a.stream.jsonl", (1000, 1000))
        os.utime(new_dir / "b.stream.jsonl", (2000, 2000))
        picked, _why = cp.pick_run_dir([("旧的", old_dir), ("新的", new_dir)])
        assert picked == new_dir

    def test_directory_without_stream_is_skipped(
            self, cp: types.ModuleType, tmp_path: Path) -> None:
        """没有事件流的目录不算数（否则会选到空目录）。"""
        empty = tmp_path / "empty"
        empty.mkdir()
        withstream = tmp_path / "live"
        withstream.mkdir()
        (withstream / "b.stream.jsonl").write_text("{}", encoding="utf-8")
        picked, _why = cp.pick_run_dir([("空的", empty), ("活的", withstream)])
        assert picked == withstream

    def test_env_pointing_at_missing_dir_falls_back(
            self, cp: types.ModuleType, tmp_path: Path) -> None:
        """环境变量指向不存在的目录时退回自动探测，而不是崩。"""
        live = tmp_path / "live"
        live.mkdir()
        (live / "b.stream.jsonl").write_text("{}", encoding="utf-8")
        picked, _why = cp.pick_run_dir(
            [("活的", live)], env_value=str(tmp_path / "不存在"))
        assert picked == live


# ==================================================================
# 事件格式
# ==================================================================

class TestEventFormat:
    """事件必须与 ``schema.StateEvent`` 对得上。"""

    def test_event_validates_against_real_schema(
            self, cp: types.ModuleType) -> None:
        """★ 防漂移：脚本造的事件必须能被**真正的** ``StateEvent`` 校验。

        ★ 为什么这条最值钱
          脚本是「手抄协议」的典型候选 —— 字段名抄错一个，
          适配器那边读出来就是 ``code`` 为空、模板不匹配、话被吞掉。
          这里直接拿真 schema 校验，抄错就红。
        """
        from autopilot.ui.schema import StateEvent

        event = cp.build_event("20261020_120000", 7, "COMPANION_LINE", "你好")
        parsed = StateEvent(**event)          # pydantic 校验
        assert parsed.code == "COMPANION_LINE"
        assert parsed.run_id == "20261020_120000"
        assert parsed.seq == 7
        assert parsed.context["text"] == "你好"
        assert parsed.message == "你好"

    def test_text_is_in_both_message_and_context(
            self, cp: types.ModuleType) -> None:
        """台词同时放 ``message`` 和 ``context["text"]``。

        ★ 双保险
          模板 ``"{text}"`` 取的是 context；万一适配器那边模板没登记，
          兜底 ``FALLBACK_PREFIX + "{message}"`` 还能把话送到 ——
          宁可多一个前缀，也不能让这句话**整个丢掉**。
        """
        event = cp.build_event("r", 1, "COMPANION_LINE", "台词在这里")
        assert event["message"] == "台词在这里"
        assert event["context"]["text"] == "台词在这里"

    def test_next_seq_continues_engine_sequence(
            self, cp: types.ModuleType, tmp_path: Path) -> None:
        """接着引擎写过的 seq 往下排，不能从 1 重来。"""
        stream = tmp_path / "r.stream.jsonl"
        with stream.open("w", encoding="utf-8") as fh:
            for i in range(1, 6):
                fh.write(json.dumps({"seq": i, "code": "X"}) + "\n")
        assert cp.next_seq(stream) == 6

    def test_next_seq_on_empty_or_missing_file(
            self, cp: types.ModuleType, tmp_path: Path) -> None:
        """空文件 / 不存在的文件 -> seq 从 1 开始。"""
        empty = tmp_path / "e.stream.jsonl"
        empty.write_text("", encoding="utf-8")
        assert cp.next_seq(empty) == 1
        assert cp.next_seq(tmp_path / "nope.stream.jsonl") == 1

    def test_next_seq_ignores_broken_lines(
            self, cp: types.ModuleType, tmp_path: Path) -> None:
        """半行 JSON（引擎正在写）不能让 seq 算错。"""
        stream = tmp_path / "r.stream.jsonl"
        stream.write_text(
            json.dumps({"seq": 3}) + "\n" + '{"seq": 4, "brok',
            encoding="utf-8")
        assert cp.next_seq(stream) == 4

    def test_append_keeps_engine_lines(self, cp: types.ModuleType, tmp_path: Path) -> None:
        """追加的事件必须排在引擎写的行**后面**，不能覆盖。"""
        stream = tmp_path / "r.stream.jsonl"
        stream.write_text(
            json.dumps({"seq": 1, "code": "RUN_STARTED"}) + "\n", encoding="utf-8")
        cp.append_event(stream, cp.build_event("r", 2, "COMPANION_LINE", "在吗"))
        lines = [json.loads(x) for x in
                 stream.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(lines) == 2
        assert lines[0]["code"] == "RUN_STARTED"
        assert lines[1]["code"] == "COMPANION_LINE"
        assert lines[1]["context"]["text"] == "在吗"


# ==================================================================
# run 指针
# ==================================================================

class TestRunPointer:
    def test_reads_run_id(self, cp: types.ModuleType, tmp_path: Path) -> None:
        (tmp_path / "run_current.txt").write_text(
            "20261020_120000", encoding="utf-8")
        assert cp.read_current_run_id(tmp_path) == "20261020_120000"

    def test_takes_last_non_empty_line(
            self, cp: types.ModuleType, tmp_path: Path) -> None:
        """``append_text`` 是追加模式，可能有多行 —— 取最后一行非空。"""
        (tmp_path / "run_current.txt").write_text(
            "old\nnew\n", encoding="utf-8")
        assert cp.read_current_run_id(tmp_path) == "new"

    def test_missing_file_returns_empty(
            self, cp: types.ModuleType, tmp_path: Path) -> None:
        assert cp.read_current_run_id(tmp_path) == ""


# ==================================================================
# 台词来源
# ==================================================================

class TestLineCollection:
    def test_reads_file_skipping_comments_and_blanks(
            self, cp: types.ModuleType, tmp_path: Path) -> None:
        path = tmp_path / "lines.txt"
        path.write_text("# 注释\n第一句\n\n  第二句  \n", encoding="utf-8")
        assert cp.read_lines_file(path) == ["第一句", "第二句"]

    def test_dedupes_but_keeps_order(self, cp: types.ModuleType) -> None:
        import argparse
        args = argparse.Namespace(lines=["A", "B", "A"], file="")
        assert cp.collect_lines(args) == ["A", "B"]
