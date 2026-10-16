"""读侧语义：**只放 ChannelReader**。

★ 省电是这个类的全部设计动机
----------------------------
overlay 10Hz 轮询，如果每次都 ``open + read + pydantic.parse``，
空转时单核轻松超过 1%。所以：

* **先 ``os.stat``**，比对 ``(st_size, st_mtime_ns)``（见 ``protocol.stamp``）
* 没变 → **立刻 return，一次 read 都不做**
* 变了才读 + 校验

只比 size 不够：两个等长快照来回切换时 size 不变，会永远读不到新内容。
所以必须带上 mtime_ns。

读侧对坏数据**宽容**：快照是原子替换的，坏数据只可能来自"从没写过"，
此时返回 None 让调用方沿用上一份，而不是抛异常让 overlay 闪退。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pydantic import ValidationError

from .protocol import (
    RUN_ID_NAME,
    SNAPSHOT_NAME,
    run_dir,
    stamp,
    stream_path,
)
from .schema import StateEvent, StateSnapshot

logger = logging.getLogger(__name__)

#: 快照超过这么久没更新就算陈旧（2026-10-16 补，集成审计 F3）。
#:
#: ★ 为什么 overlay / 面板侧也需要它
#:   本类的全部设计动机是「省电」：stat 没变就零 IO 直接返回缓存。于是
#:   「引擎已经死了、文件停在最后一次的内容」与「引擎活着、这一轮恰好没变化」
#:   在调用方眼里**完全一样**。
#:
#:   后果实测过（集成审计 F4）：overlay 可以在引擎**根本没启动**的情况下
#:   拉起卡片，把上次遗留的快照当成当前状态显示 —— 帧率、场景、意图数
#:   全是几分钟前的。观众看到的是一个「看起来在跑但其实没跑」的引擎。
#:
#: 取 5.0s：引擎 5Hz 写，5s 是 25 个周期的余量，正常绝不会误判。
STALE_AFTER_S = 5.0


class ChannelReader:
    """overlay / 插件面板侧读通道。

    ::

        r = ChannelReader()
        snap = r.read_snapshot()          # None = 还没写过/读失败
        for evt in r.poll_events():       # 只给新增
            ...
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.dir = run_dir(base_dir)
        self._snap_stamp: tuple[int, int] | None = None
        self._snap: StateSnapshot | None = None

        self._stream_stamp: tuple[int, int] | None = None
        self._stream_offset = 0
        self._stream_path: Path | None = None

        self._run_id = ""

    # ==================================================================
    # 快照
    # ==================================================================
    def read_snapshot(self, *, force: bool = False) -> StateSnapshot | None:
        """读当前态。**未变化时返回缓存的那一份，不碰文件。**

        ``force=True`` 用于"我明知道文件变了"的场合（比如刚收到命令）。
        """
        p = self.dir / SNAPSHOT_NAME
        cur = stamp(p)
        if not force and cur == self._snap_stamp:
            return self._snap

        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self._snap_stamp = cur      # 记住这次 stamp，别每次重试 stat
            return self._snap
        if not text.strip():
            self._snap_stamp = cur
            return self._snap
        try:
            snap = StateSnapshot.model_validate_json(text)
        except ValidationError as exc:
            logger.debug("快照非法（沿用上一份）：%s", exc)
            self._snap_stamp = cur
            return self._snap
        self._snap_stamp = cur
        self._snap = snap
        self._run_id = snap.run_id
        return snap

    @property
    def cached_snapshot(self) -> StateSnapshot | None:
        """最近一次成功读到的快照（不做任何 IO）。"""
        return self._snap

    # ==================================================================
    # 陈旧判定（2026-10-16 补，集成审计 F3/F4）
    # ==================================================================
    def snapshot_age_s(self) -> float:
        """快照文件**最后修改**距今多少秒。

        ★ 用文件的 mtime，不信快照里的时间戳
          引擎每 200ms 原子替换一次快照，mtime 就是「它还在动」的可靠证据；
          而 ``StateSnapshot`` 内部时间字段的语义随 producer 变化，
          出错时恰好是最需要它准的时候。

        读不到文件返回 ``inf``（= 最陈旧），让调用方统一走「不可信」分支。
        """
        try:
            return max(0.0, time.time() - (self.dir / SNAPSHOT_NAME).stat().st_mtime)
        except OSError:
            return float("inf")

    def is_snapshot_stale(self) -> bool:
        """手上这份快照是否已陈旧（超过 :data:`STALE_AFTER_S` 没更新）。

        ★ 与 ``cached_snapshot is None`` 是两件事
          那个只说明「从没读到过」。**读到过之后引擎死掉**仍然返回上一份 ——
          而那一份可能已经很旧。调用方要判断「这份还能不能信」必须问这里。

        ⚠️ 一条刻意的设计：这里**不做 stat 短路**，每次调用都真的 stat。
           它和 ``read_snapshot`` 是两条独立路径 —— 后者省电靠 stamp 比对，
           这个要的是「此刻」的真值，两者不能互相污染。
        """
        return self.snapshot_age_s() > STALE_AFTER_S

    # ==================================================================
    # 事件流
    # ==================================================================
    def poll_events(self) -> list[StateEvent]:
        """只给**新增**事件。空转时只 stat 一次，零 IO。"""
        run_id = self.run_id()
        if not run_id:
            return []
        path = stream_path(self.dir, run_id)

        # run 换了 -> 从头跟
        if self._stream_path != path:
            self._stream_path = path
            self._stream_offset = 0
            self._stream_stamp = None

        cur = stamp(path)
        if cur is None:
            return []
        if cur == self._stream_stamp:
            return []                     # ★ 没变 -> 不读
        # 文件被截断/重建（轮转或换 run）-> 重新从头
        if self._stream_stamp is not None and cur[0] < self._stream_offset:
            self._stream_offset = 0
        self._stream_stamp = cur

        out: list[StateEvent] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                fh.seek(self._stream_offset)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(StateEvent.model_validate_json(line))
                    except ValidationError as exc:
                        # 单行坏了不能拖垮整条流
                        logger.debug("事件行非法，跳过：%s", exc)
                self._stream_offset = fh.tell()
        except OSError:
            return out
        return out

    def attach_stream_tail(self, offset: int = -1) -> None:
        """游标挪到尾部（只关心"从现在起"）。

        v2 换 socket 时这就是接缝：socket 侧实现"从当前 tail 订阅"。
        """
        run_id = self.run_id()
        if not run_id:
            return
        path = stream_path(self.dir, run_id)
        self._stream_path = path
        cur = stamp(path)
        size = cur[0] if cur else 0
        self._stream_stamp = cur
        self._stream_offset = size if offset < 0 else max(0, min(offset, size))

    # ==================================================================
    def run_id(self) -> str | None:
        """当前 run_id。缓存住，避免每次读文件。"""
        if self._run_id:
            return self._run_id
        try:
            self._run_id = (self.dir / RUN_ID_NAME).read_text(
                encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
        return self._run_id or None


__all__ = ["STALE_AFTER_S", "ChannelReader"]
