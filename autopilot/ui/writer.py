"""写侧语义：**只放 ChannelWriter**。

具体 IO 全在 :mod:`autopilot.ui.protocol`，这里只管"什么时候写什么"。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .protocol import (
    ARCHIVE_DIR_NAME,
    ARCHIVE_KEEP_RUNS,
    ARCHIVE_PERF_FMT,
    ARCHIVE_SNAPSHOT_FMT,
    PERF_SAMPLE_INTERVAL_S,
    RUN_ID_NAME,
    SNAPSHOT_NAME,
    append_jsonl,
    atomic_write_text,
    new_run_id,
    rotate_if_needed,
    run_dir,
    stream_path,
)
from .schema import StateEvent, StateSnapshot

logger = logging.getLogger(__name__)


class ChannelWriter:
    """run_loop 侧写通道。

    ::

        w = ChannelWriter(run_dir, run_id)
        w.write_snapshot(snap)
        w.write_event(evt)
        w.close()

    事件流**按 run_id 分文件**、seq 从 1 单调递增、run 内唯一。
    """

    def __init__(self, base_dir: Path | str | None = None,
                 run_id: str | None = None) -> None:
        self.dir = run_dir(base_dir)
        self.run_id = run_id or new_run_id()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._stream = stream_path(self.dir, self.run_id)
        self._seq = 0
        self._closed = False
        #: 归档目录（run 结束时才真正创建 —— 不跑完就不该留空目录）
        self._archive = self.dir / ARCHIVE_DIR_NAME
        #: 上次性能采样的 monotonic 时刻。0 = 从没采过。
        self._last_perf_at = 0.0
        #: 上一次采样的真实时刻，用来**反推**实际 tick 频率
        self._last_perf_mono: float | None = None
        # 让读侧快速判断"当前是哪个 run"
        atomic_write_text(self.dir / RUN_ID_NAME, self.run_id)

    # -- 快照 ------------------------------------------------------------
    def write_snapshot(self, snap: StateSnapshot) -> None:
        """覆盖写当前态。**原子**（临时文件 + rename）。

        快照 200ms 一次就够 —— overlay 是给人看的，200ms 内人的眼睛
        察觉不到差别，而写频率翻 10 倍只是白费 IO。

        ★ 顺带做性能采样
          这里只多了一次 ``time.monotonic()`` 比较；真正写文件的动作
          每 :data:`PERF_SAMPLE_INTERVAL_S` 才发生一次（快照 200ms 一次，
          所以约 1/50）。热路径上的额外开销可以忽略，而归档这件事
          **只有挂在写快照上才不需要改 run_loop**。
        """
        if self._closed:
            raise RuntimeError("ChannelWriter 已关闭")
        atomic_write_text(self.dir / SNAPSHOT_NAME,
                          snap.model_dump_json())
        self._maybe_sample_perf(snap)

    # -- 性能归档 --------------------------------------------------------
    def perf_path(self) -> Path:
        return self._archive / ARCHIVE_PERF_FMT.format(run_id=self.run_id)

    def _maybe_sample_perf(self, snap: StateSnapshot) -> None:
        """按固定间隔追加一行性能数据。**任何异常都吞掉**，绝不拖垮主循环。

        ★ 为什么归档失败只 warning
          归档是**诊断**功能。磁盘满了 / 目录被杀毒软件锁住 / 权限不对 ——
          这些都不该让引擎停摆。写不进就写不进，下个周期再试。
        """
        now = time.monotonic()
        if now - self._last_perf_at < PERF_SAMPLE_INTERVAL_S:
            return
        prev = self._last_perf_mono
        self._last_perf_at = now
        self._last_perf_mono = now

        # ★ derived_tick_hz 是**实测值**：由相邻两次采样的真实间隔反推。
        #   快照里的 tick_hz 是 run_loop 的**配置目标**（事件流里 48 个 run
        #   全写 5.0、方差 0），拿它算性能等于拿配置当测量。
        derived = None
        if prev is not None and (now - prev) > 1e-6:
            derived = round(1.0 / (now - prev), 3)

        row = {
            "run_id": self.run_id,
            "ts_ms": int(time.time() * 1000),
            "uptime_s": snap.uptime_s,
            "tick_count": snap.tick_count,
            # 以下三个是 run_loop 自报的
            "tick_hz": snap.tick_hz,
            "tick_avg_ms": snap.tick_avg_ms,
            "tick_p95_ms": snap.tick_p95_ms,
            # 这个是**实测**的
            "derived_tick_hz": derived,
            "state": str(getattr(snap.state, "value", snap.state)),
            "scene": snap.scene,
            "channel": snap.capture.active,
        }
        try:
            append_jsonl(self.perf_path(), row)
        except OSError as exc:
            logger.warning("性能归档写入失败（本周期跳过）：%s", exc)

    def _archive_final_snapshot(self) -> None:
        """把终态快照存一份到归档目录。**失败只 warning**。"""
        try:
            src = self.dir / SNAPSHOT_NAME
            if not src.is_file():
                return                       # 一个快照都没写过，没什么可存的
            self._archive.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self._archive / ARCHIVE_SNAPSHOT_FMT.format(run_id=self.run_id),
                src.read_text(encoding="utf-8"),
            )
        except OSError as exc:
            logger.warning("终态快照归档失败：%s", exc)

    def _prune_archives(self) -> None:
        """只保留最近 :data:`ARCHIVE_KEEP_RUNS` 个 run 的归档。

        ★ 为什么必须有上限
          归档目录**只增不减**的话，跑得越久越大 —— 一个长期运行的实例
          迟早把磁盘吃满。宁可丢掉最老的历史，也不要冒这个风险。
        """
        try:
            runs: dict[str, float] = {}
            for p in self._archive.glob("*"):
                if not p.is_file():
                    continue
                rid = p.name.split(".")[0]
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                runs[rid] = max(runs.get(rid, 0.0), mtime)
            if len(runs) <= ARCHIVE_KEEP_RUNS:
                return
            # 最老的先走
            for rid, _m in sorted(runs.items(), key=lambda kv: kv[1])[
                    :len(runs) - ARCHIVE_KEEP_RUNS]:
                for p in self._archive.glob(f"{rid}.*"):
                    try:
                        p.unlink()
                    except OSError:
                        logger.debug("归档清理失败：%s", p)
        except OSError as exc:
            logger.warning("归档清理失败：%s", exc)

    # -- 事件 ------------------------------------------------------------
    def write_event(self, evt: StateEvent) -> None:
        """追加一条事件。自动补 seq / run_id / ts。"""
        if self._closed:
            raise RuntimeError("ChannelWriter 已关闭")
        payload = evt.model_dump(mode="json")
        payload["run_id"] = self.run_id
        payload["seq"] = self._seq = self._seq + 1
        if not payload.get("ts_ms"):
            payload["ts_ms"] = int(time.time() * 1000)
        rotate_if_needed(self._stream)
        append_jsonl(self._stream, payload)

    def close(self) -> None:
        """结束本次 run。**幂等** —— 归档失败也照样把关闭标记打上。

        ★ 关闭时的顺序
          1. 先标 ``_closed``：后续任何 ``write_snapshot`` 立刻抛，
             不会出现「归档写了一半又写了一次快照」的交错；
          2. 再归档终态快照；
          3. 最后清理旧归档 —— 这步放最后，万一它出问题，
             至少本 run 的归档已经落盘了。

        ★ 两步都再兜一层 try
          归档是**诊断**功能，而 ``close()`` 在引擎的退出路径上 ——
          它抛出去，退出码就变了，排查的人会先怀疑「是不是退出时挂了」。
          宁可什么都丢，也不能让诊断功能改写主流程的结局。
        """
        if self._closed:
            return
        self._closed = True
        for step in (self._archive_final_snapshot, self._prune_archives):
            try:
                step()
            except Exception as exc:                     # noqa: BLE001
                logger.warning("关闭时归档步骤 %s 失败（不影响退出）：%s",
                               step.__name__, exc)


__all__ = ["ChannelWriter"]
