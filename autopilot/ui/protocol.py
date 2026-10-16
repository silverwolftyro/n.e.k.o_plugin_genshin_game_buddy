"""通道基础设施（v1 = 文件实现）。

★ 这里**只有工具函数，没有任何业务 class**。职责边界是硬约定：

    schema.py     三个 pydantic 模型（数据长什么样）
    protocol.py   ← 本文件：路径 / 原子写 / 追写 / 轮转（数据怎么落盘）
    writer.py     ChannelWriter   （写侧语义）
    reader.py     ChannelReader   （读侧语义）
    commands.py   CommandSender + CommandPoller（反向控制）

把 class 放在这里会让 protocol 同时背上"传输"和"语义"两件事，
v2 换 socket 时要重写的东西会多出一倍。

v2 换 socket / named pipe 时：**只重写本文件**，schema 一个字段不动。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 文件名与目录 ───────────────────────────────────────────────────────
#: 默认运行目录：<autopilot>/run/
DEFAULT_RUN_DIR = Path(__file__).resolve().parents[1] / "_runtime"

SNAPSHOT_NAME = "state_snapshot.json"
STREAM_NAME = "state_stream.jsonl"
COMMANDS_NAME = "commands.jsonl"
RUN_ID_NAME = "run_current.txt"

# ── 归档（2026-10-13 新增）───────────────────────────────────────────────────
#: 归档子目录。放在 _runtime 下面，和运行期文件同盘 —— 免得跨盘写。
#:
#: ★ 为什么要它
#:   ``state_snapshot.json`` 是**覆盖写**的（overlay 要读「当前态」），
#:   于是 13 天下来只活下来**一份**快照 —— 任何性能分析都会卡在 n=1，
#:   连「有没有退化」都判断不了（见 recognition/notes/performance_baseline.md）。
ARCHIVE_DIR_NAME = "archive"
#: 归档快照文件名模板
ARCHIVE_SNAPSHOT_FMT = "{run_id}.snapshot.json"
#: 性能采样文件名模板
ARCHIVE_PERF_FMT = "{run_id}.perf.jsonl"
#: 性能采样间隔（秒）。**不要调小**：快照本身 200ms 一次，采样只是顺带取数。
PERF_SAMPLE_INTERVAL_S = 10.0
#: 保留最近几个 run 的归档。**必须有上限** —— 归档目录会一直长，
#: 不设限就是往磁盘里倒垃圾。
ARCHIVE_KEEP_RUNS = 20

#: 事件流单文件上限，超过改名 .1 并新建（只保留一代，不做归档）
STREAM_MAX_BYTES = 5 * 1024 * 1024


def run_dir(base: Path | str | None = None) -> Path:
    """解析运行目录。"""
    return Path(base) if base else DEFAULT_RUN_DIR


def new_run_id() -> str:
    """生成 run_id：``20260928_170541``。一次启动一个，天然唯一到秒。"""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def stream_path(base: Path | str, run_id: str) -> Path:
    """事件流路径。**按 run_id 分文件** —— 不同 run 的 seq 不能混在一起，
    否则读侧分不清哪些事件属于本次运行。"""
    return run_dir(base) / f"{run_id}.stream.jsonl"


# ── 原子写 ─────────────────────────────────────────────────────────────


def atomic_write_text(path: Path | str, text: str) -> None:
    """临时文件 + ``os.replace``。

    ★ 为什么不能直接 ``open(path, "w")``
      那个模式先把文件截断成 0 字节再逐字节写。这个窗口里读到的是
      **半截 JSON**，读侧 pydantic 校验直接抛，overlay 闪一下就没了。
      同文件系统内的 ``os.replace`` 是原子的：读侧要么看到完整的旧版、
      要么看到完整的新版，**永远读不到半截**。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp_",
                               suffix=p.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── 追写 ───────────────────────────────────────────────────────────────


def append_jsonl(path: Path | str, obj: dict[str, Any]) -> None:
    """追加一行 JSON（UTF-8，``O_APPEND``）。

    ★ 必须用 ``"a"`` 模式：POSIX 下等价于 ``O_APPEND``，每次写都追加到
      尾部，且在本地文件系统上是**单次原子写**。这样 overlay 和插件面板
      （B 轨）并发追加同一份流也不会互相覆盖。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_text(path: Path | str, line: str) -> None:
    """追加一行纯文本（``run_current.txt`` 用）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line)


# ── 轮转 ───────────────────────────────────────────────────────────────


def rotate_if_needed(path: Path | str, max_bytes: int = STREAM_MAX_BYTES) -> bool:
    """超过上限就改名 ``.1`` 并让调用方新建。返回是否发生了轮转。

    ★ 只保留一代。v1 不做归档 —— 归档策略是可配置的运维问题，
      现在定死一个简单规则，好过定一个错但看起来很完备的规则。
    """
    p = Path(path)
    try:
        if p.stat().st_size <= max_bytes:
            return False
    except OSError:
        return False
    old = p.with_suffix(p.suffix + ".1")
    try:
        if old.exists():
            old.unlink()
        p.rename(old)
        return True
    except OSError as exc:
        logger.warning("轮转失败（继续写原文件）：%s", exc)
        return False


# ── stat 短路（省电核心）─────────────────────────────────────────────────


def stamp(path: Path | str) -> tuple[int, int] | None:
    """``(st_size, st_mtime_ns)``。文件不存在返回 None。

    ★ 省电的关键。空转时 overlay 10Hz 轮询，如果每次都 open+read+parse，
      单核轻松上 1%；只做一次 ``os.stat``（微秒级、无 IO 数据读取）
      就返回，能把空转开销压到接近 0。

    ★ 为什么 size **加** mtime：只比 size 会漏掉"同样长度但内容变了"
      （比如两个 100 字节的快照来回切换），那种情况下会永远读不到新内容。
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def stamp_changed(cur: tuple[int, int] | None,
                  prev: tuple[int, int] | None) -> bool:
    """和上次比有没有变。**没变就立刻 return，不读文件。**"""
    if cur is None:
        return prev is not None      # 文件消失了也算变化（要报离线）
    return cur != prev


__all__ = [
    "COMMANDS_NAME",
    "DEFAULT_RUN_DIR",
    "RUN_ID_NAME",
    "SNAPSHOT_NAME",
    "STREAM_MAX_BYTES",
    "STREAM_NAME",
    "append_jsonl",
    "append_text",
    "atomic_write_text",
    "new_run_id",
    "rotate_if_needed",
    "run_dir",
    "stamp",
    "stamp_changed",
    "stream_path",
]
