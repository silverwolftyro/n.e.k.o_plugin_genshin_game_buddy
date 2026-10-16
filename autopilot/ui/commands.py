"""反向命令通道：**只放 CommandSender + CommandPoller**。

overlay / 插件面板 → run_loop。

★ 两个方向都只负责"送达与取出"，**不解释命令含义**
--------------------------------------------------
* :class:`CommandSender` 不校验 cmd 是否被支持 —— 协议层一拦，
  新增命令就要两边同时升级，协议层解耦的意义就没了。
* :class:`CommandPoller` 不执行命令 —— 执行必须在 run_loop 主循环里
  （那里才是能安全调 ``loop.stop()`` 的地方）。

★ 命令文件用「读完清空」而不是游标
------------------------------------
文件通道没有"已读游标"这个概念，命令本身也不该被重复执行
（STOP 执行两次、KILL 执行两次，后果都比执行一次糟）。
所以取完就清空，最省事也最不容易出错。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .protocol import COMMANDS_NAME, append_jsonl, run_dir, stamp
from .schema import Command, new_cmd_id

logger = logging.getLogger(__name__)

SUPPORTED = ("STOP", "KILL", "PAUSE", "RESUME")


class CommandSender:
    """overlay / 插件面板侧：写一条命令。"""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.dir = run_dir(base_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def send(self, cmd: str, args: dict[str, Any] | None = None) -> str:
        """写一条命令，返回 ``cmd_id``。

        ⚠ **不校验** cmd 是否被支持。写了协议层的价值就是解耦 ——
          写侧拦掉会让"加一个命令"变成两件事。
        """
        self._seq += 1
        cid = new_cmd_id(self._seq)
        append_jsonl(self.dir / COMMANDS_NAME, {
            "schema_version": "1.0",
            "cmd_id": cid,
            "ts_ms": int(time.time() * 1000),
            "cmd": cmd,
            "args": args or {},
        })
        logger.info("已发送命令 %s (cmd_id=%s)", cmd, cid)
        return cid


class CommandPoller:
    """run_loop 侧：取走尚未处理的命令。

    空闲时**只 stat 一次**，没变就返回空列表（零 IO）。
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.dir = run_dir(base_dir)
        self._offset = 0
        self._stamp: tuple[int, int] | None = None

    def poll(self) -> list[Command]:
        p = self.dir / COMMANDS_NAME
        cur = stamp(p)
        if cur is None:
            return []
        if cur == self._stamp:
            return []                     # ★ 没变 -> 不读
        if self._stamp is not None and cur[0] < self._offset:
            self._offset = 0              # 被清空过
        self._stamp = cur

        out: list[Command] = []
        try:
            with p.open("r", encoding="utf-8") as fh:
                fh.seek(self._offset)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(Command.model_validate_json(line))
                    except ValidationError as exc:
                        logger.warning("命令行非法（忽略）：%s", exc)
                self._offset = fh.tell()
        except OSError:
            return out
        return out

    def clear(self) -> None:
        """取完清空。run_loop 处理完一轮后调。"""
        self._offset = 0
        try:
            (self.dir / COMMANDS_NAME).write_text("", encoding="utf-8")
        except OSError as exc:
            logger.warning("清空命令文件失败：%s", exc)


__all__ = ["SUPPORTED", "CommandPoller", "CommandSender"]
