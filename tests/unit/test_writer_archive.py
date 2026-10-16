"""``ChannelWriter`` 的归档轮转（2026-10-13 新增功能）。

★ 为什么要它
  ``state_snapshot.json`` 是**覆盖写**的（overlay 要读「当前态」），于是
  13 天只活下来**一份**快照 —— 任何性能分析都卡在 n=1，连「有没有退化」
  都判断不了（见 ``recognition/notes/performance_baseline.md``）。

  这里验的是：主文件行为**完全不变**（overlay 依赖它），归档是**附加**的，
  失败时**只警告不中断**，而且归档目录**有上限**、不会一直长。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from autopilot.ui import protocol as P
from autopilot.ui.schema import CaptureStat, StateSnapshot
from autopilot.ui.writer import ChannelWriter

pytestmark = pytest.mark.unit


@pytest.fixture
def w(tmp_path):
    wr = ChannelWriter(base_dir=tmp_path, run_id="20261013_120000")
    try:
        yield wr
    finally:
        wr.close()


def _snap(n: int = 1, **kw) -> StateSnapshot:
    base = dict(run_id="20261013_120000", written_at_ms=1_700_000_000_000,
                uptime_s=n * 0.2, state="RUNNING", tick_hz=5.0,
                tick_avg_ms=43.06, tick_p95_ms=142.55, tick_count=n,
                scene="WORLD", capture=CaptureStat(active="dxcam"))
    base.update(kw)
    return StateSnapshot(**base)


# ==================================================================
# 主文件行为：一丝不变
# ==================================================================

def test_snapshot__main_file_is_still_overwritten(w) -> None:
    """★ overlay 读的就是这个文件 —— 路径与覆盖语义都不能变。"""
    w.write_snapshot(_snap(1))
    first = w.dir / P.SNAPSHOT_NAME
    assert first.is_file()
    w.write_snapshot(_snap(2))
    assert first.is_file()
    assert json.loads(first.read_text(encoding="utf-8"))["tick_count"] == 2


def test_snapshot__no_archive_snapshot_before_close(w) -> None:
    """★ 归档里的**终态快照**只能在 run 结束时才出现。

    （性能归档 ``.perf.jsonl`` 是例外 —— 它按设计就该边跑边写。）
    """
    w.write_snapshot(_snap(1))
    w.write_snapshot(_snap(2))
    snaps = list((w.dir / P.ARCHIVE_DIR_NAME).glob("*.snapshot.json")) \
        if (w.dir / P.ARCHIVE_DIR_NAME).exists() else []
    assert snaps == [], snaps


def test_snapshot__still_raises_after_close(w) -> None:
    w.close()
    with pytest.raises(RuntimeError):
        w.write_snapshot(_snap(1))


# ==================================================================
# 性能采样
# ==================================================================

def test_perf__samples_are_appended_to_the_archive(w) -> None:
    """★ 连续写快照 -> 归档里出现多行性能数据。"""
    # 直接把采样间隔调 0，省得真的等 10 秒
    old = P.PERF_SAMPLE_INTERVAL_S
    import autopilot.ui.writer as W

    W.PERF_SAMPLE_INTERVAL_S = 0.0
    try:
        for i in range(1, 4):
            w.write_snapshot(_snap(i))
    finally:
        W.PERF_SAMPLE_INTERVAL_S = old
    rows = [json.loads(l) for l in
            w.perf_path().read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 3, rows
    assert [r["tick_count"] for r in rows] == [1, 2, 3]


def test_perf__row_carries_the_measured_fields(w) -> None:
    old = P.PERF_SAMPLE_INTERVAL_S
    import autopilot.ui.writer as W

    W.PERF_SAMPLE_INTERVAL_S = 0.0
    try:
        w.write_snapshot(_snap(1))
        w.write_snapshot(_snap(2))
    finally:
        W.PERF_SAMPLE_INTERVAL_S = old
    rows = [json.loads(l) for l in
            w.perf_path().read_text(encoding="utf-8").splitlines() if l.strip()]
    r = rows[-1]
    for k in ("run_id", "ts_ms", "uptime_s", "tick_hz", "tick_avg_ms",
              "tick_p95_ms", "derived_tick_hz", "state", "scene", "channel"):
        assert k in r, k


def test_perf__derived_hz_is_null_on_the_first_sample(w) -> None:
    """★ 第一次采样没有「上一次」，反推不出来就是 null —— 不能编一个。"""
    import autopilot.ui.writer as W

    old = P.PERF_SAMPLE_INTERVAL_S
    W.PERF_SAMPLE_INTERVAL_S = 0.0
    try:
        w.write_snapshot(_snap(1))
    finally:
        W.PERF_SAMPLE_INTERVAL_S = old
    first = json.loads(w.perf_path().read_text(encoding="utf-8").splitlines()[0])
    assert first["derived_tick_hz"] is None


def test_perf__hot_path_is_throttled_by_default(w) -> None:
    """★ 默认 10 秒间隔下，连写 20 次快照只应产生 1 行。

    这是「归档不影响主循环性能」的直接证据：热路径上只有一次比较。
    """
    for i in range(1, 21):
        w.write_snapshot(_snap(i))
    assert not w.perf_path().is_file() or \
        len(w.perf_path().read_text(encoding="utf-8").splitlines()) == 1


def test_perf__write_failure_does_not_break_the_loop(w, monkeypatch) -> None:
    """★ 归档写不进去只能 warning —— 磁盘满 / 目录被锁都不该让引擎停。"""
    import autopilot.ui.writer as W

    def _boom(*a, **kw):
        raise OSError("磁盘满了")

    old = P.PERF_SAMPLE_INTERVAL_S
    W.PERF_SAMPLE_INTERVAL_S = 0.0
    monkeypatch.setattr(W, "append_jsonl", _boom)
    try:
        for i in range(1, 4):          # 抛三次，一次都不许冒到外面
            w.write_snapshot(_snap(i))
        # 主文件照样在写
        assert (w.dir / P.SNAPSHOT_NAME).is_file()
    finally:
        W.PERF_SAMPLE_INTERVAL_S = old


# ==================================================================
# 终态快照归档
# ==================================================================

def test_close__archives_the_final_snapshot(w) -> None:
    w.write_snapshot(_snap(7))
    w.close()
    arch = w.dir / P.ARCHIVE_DIR_NAME / "20261013_120000.snapshot.json"
    assert arch.is_file()
    assert json.loads(arch.read_text(encoding="utf-8"))["tick_count"] == 7


def test_close__archives_even_without_any_snapshot(w) -> None:
    """一个快照都没写就 close -> 不该崩，也不该造出空文件。"""
    w.close()
    arch = w.dir / P.ARCHIVE_DIR_NAME
    assert not arch.exists() or not list(arch.glob("*.snapshot.json"))


def test_close__is_idempotent(w) -> None:
    """★ 重复 close 不该炸，也不该重复归档。"""
    w.write_snapshot(_snap(1))
    w.close()
    w.close()
    snaps = list((w.dir / P.ARCHIVE_DIR_NAME).glob("*.snapshot.json"))
    assert len(snaps) == 1


def test_close__archive_failure_is_only_a_warning(w, monkeypatch) -> None:
    """★ 归档失败绝不冒泡 —— 引擎退出路径不能因为诊断功能炸掉。"""
    w.write_snapshot(_snap(1))

    import autopilot.ui.writer as W

    def _boom(self):
        raise OSError("目录被锁")

    monkeypatch.setattr(W.ChannelWriter, "_archive_final_snapshot", _boom)
    w.close()          # 不抛即可


# ==================================================================
# 归档清理：有上限
# ==================================================================

def test_prune__keeps_only_the_most_recent_runs(w, monkeypatch) -> None:
    """★ 归档目录只增不减就会吃满磁盘 —— 必须封顶。"""
    import autopilot.ui.writer as W

    monkeypatch.setattr(W, "ARCHIVE_KEEP_RUNS", 3)
    arch = w.dir / P.ARCHIVE_DIR_NAME
    arch.mkdir(parents=True)
    now = time.time()
    for i in range(10):                       # 造 10 个「历史 run」
        rid = f"old_{i}"
        for suffix in (".snapshot.json", ".perf.jsonl"):
            p = arch / f"{rid}{suffix}"
            p.write_text("{}", encoding="utf-8")
            import os
            os.utime(p, (now + i, now + i))   # 老的更早

    w.write_snapshot(_snap(1))
    w.close()

    left = {p.name.split(".")[0] for p in arch.iterdir() if p.is_file()}
    assert "old_9" in left and "old_8" in left, "最新的两个应保留"
    assert "old_0" not in left, "最老的应被清掉"
    assert len(left) <= 3, left


def test_prune__deletes_both_files_of_a_pruned_run(w, monkeypatch) -> None:
    """★ 同一个 run 的快照与性能文件要**一起**删干净，不能只删一半。"""
    import autopilot.ui.writer as W

    monkeypatch.setattr(W, "ARCHIVE_KEEP_RUNS", 1)
    arch = w.dir / P.ARCHIVE_DIR_NAME
    arch.mkdir(parents=True)
    (arch / "old.snapshot.json").write_text("{}", encoding="utf-8")
    (arch / "old.perf.jsonl").write_text("{}", encoding="utf-8")

    w.write_snapshot(_snap(1))
    w.close()
    assert not (arch / "old.snapshot.json").exists()
    assert not (arch / "old.perf.jsonl").exists()


def test_prune__below_the_limit_is_a_no_op(w) -> None:
    arch = w.dir / P.ARCHIVE_DIR_NAME
    arch.mkdir(parents=True)
    (arch / "only.snapshot.json").write_text("{}", encoding="utf-8")
    w.write_snapshot(_snap(1))
    w.close()
    assert (arch / "only.snapshot.json").exists()


# ==================================================================
# 契约
# ==================================================================

def test_contract__archive_dir_lives_under_runtime(w) -> None:
    """归档必须和运行期文件**同盘** —— 跨盘写会引入不必要的失败模式。"""
    assert Path(w._archive).parent == w.dir


def test_contract__keep_runs_is_finite() -> None:
    """把这个值改成无穷大等于没有上限 —— 守卫住。"""
    assert 0 < P.ARCHIVE_KEEP_RUNS <= 365
    assert P.PERF_SAMPLE_INTERVAL_S > 0
