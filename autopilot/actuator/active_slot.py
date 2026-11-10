"""在场槽位记账 + 编队名册索引 —— **不靠看画面，靠「谁按了切人键」+ OCR**。

这个模块回答两个问题：

1. :class:`ActiveSlotTracker` —— **现在谁在场？**
   切人只有两种来源，两种都能被观测到：

   a. **引擎自己切的** —— 执行层跑完 ``SWITCH_CHARACTER``，按的哪个数字键
      它自己知道（:mod:`autopilot.actuator.executor` 里调
      :meth:`ActiveSlotTracker.note_switch`）。
   b. **用户自己切的** —— ``physical_input`` 监听到数字键
      （:meth:`ActiveSlotTracker.note_physical_key`）。

2. :class:`CharSlotIndex` —— **1/2/3/4 号位分别是谁？**
   由队伍栏 OCR 填（:meth:`CharSlotIndex.note_name`），给执行层把
   ``char_id`` 翻译成「该按哪个数字键」（:meth:`CharSlotIndex.slot_of`）。

   ★ 这条是 2026-10-18 加的，动机是**免配置**：之前要求用户手填
     ``keymap.toml`` 的 ``[char_slots]``，而队伍是每个玩家自己的信息，
     逼用户抄一遍角色名既烦又容易抄错（抄错不报错，只会切错人）。
     OCR 在大世界本来就在跑（队伍栏名字就在屏幕上），白拿即可。

🔴 :class:`ActiveSlotTracker` **唯一**的失效场景
---------------------------------------------
**角色阵亡时游戏会自动切人，而引擎不知道。** 中途换队伍同理。
症状是「规则以为 A 在场，于是反复按 A 的数字键」——
按的是**已经切过去**的那个键，**无破坏性**（不会按错技能、不会乱动镜头），
只是白白浪费几帧。想缓解就把 ``stale_after_ms`` 设成大于
「一次连招时长」的值，让它过期后回到 UNKNOWN（宁可规则不触发）。

★ 为什么不做时间衰减
--------------------
本可以「切人后 X 秒内有效」。但那样每 X 秒 ``active_member`` 就会
翻回 UNKNOWN，连招规则会**周期性地集体熄火** —— 一个会自己关掉的
系统比一个偶尔说错话的系统难查得多。所以默认
``stale_after_ms = 0``（**永不过期**），把失效场景写在上面而不是
藏在阈值里。真要设就设成明确的一段时间，代价自己权衡。

线程安全：写入方有两个（执行层线程、监听线程），读取方是主循环线程。
用一把锁 —— 临界区只有几次字典读写，不会有争用。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

#: VK 名 -> 槽位。**只有前四个数字键**是切人键，别把 NUMPAD 也算进来 ——
#: 那是「放技能」的重复键位（游戏里小键盘 1-4 同样是切人），
#: 但用户几乎不按，且与技能键语义重叠，误判成本更高。
DIGIT_SLOTS: dict[str, int] = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
}

VALID_SLOTS = (1, 2, 3, 4)


class ActiveSlotTracker:
    """记住「现在 1..4 号位里哪个在场」。

    :param stale_after_ms: 超过这个时长没更新就当 UNKNOWN。
        ``0`` = 永不过期（**默认**，理由见模块文档）。
    :param now_fn: 取当前时刻的函数（毫秒）。测试注入虚拟时钟。

    用法::

        tracker = ActiveSlotTracker()
        tracker.note_switch(2, source="engine")     # 执行层切完人
        tracker.note_physical_key("TWO")            # 监听到用户按了 2
        if tracker.current() == 2: ...
    """

    def __init__(
        self,
        *,
        stale_after_ms: int = 0,
        now_fn: Callable[[], int] | None = None,
    ) -> None:
        self._stale_ms = max(0, int(stale_after_ms))
        self._now = now_fn or (lambda: int(time.time() * 1000))
        self._slot: int | None = None
        self._at_ms: int = 0
        self._source: str = "none"
        self._lock = threading.Lock()

    # -- 写入 ------------------------------------------------------------
    def note_switch(self, slot: int, *, source: str = "engine") -> bool:
        """记录一次切人。返回是否被接受。

        槽位不在 1..4 里**直接拒绝并返回 False** —— 调用方（执行层）
        照样按了键，但不该把一个非法槽位写进状态：那样 ``current()``
        永远返回 None，规则全变 UNKNOWN，而日志上却看不出是谁写的。
        """
        s = int(slot)
        if s not in VALID_SLOTS:
            return False
        with self._lock:
            self._slot = s
            self._at_ms = self._now()
            self._source = source
        return True

    def note_physical_key(self, key_name: str) -> bool:
        """监听到一次物理按键。是数字键就记账，不是就什么都不做。"""
        slot = DIGIT_SLOTS.get((key_name or "").strip().upper())
        if slot is None:
            return False
        return self.note_switch(slot, source="physical")

    def reset(self) -> None:
        """忘掉在场角色（换队伍、传送、用户手动接管时用）。"""
        with self._lock:
            self._slot = None
            self._at_ms = 0
            self._source = "none"

    # -- 读取 ------------------------------------------------------------
    def current(self) -> int | None:
        """当前在场的槽位，**过期或没记过就返回 None**（= UNKNOWN）。

        ★ 返回 ``None`` 而不是猜一个 1：``None`` 会让 ``onfield()``
          走 :meth:`PolicyContext._unk`，于是规则的 ``unknowns`` 里有
          ``onfield``，排查时能看出「感知没读到谁在场」；
          猜一个 1 会让规则**看起来在正常工作**。
        """
        with self._lock:
            slot, at = self._slot, self._at_ms
            stale, now = self._stale_ms, self._now()
        if slot is None:
            return None
        if stale > 0 and (now - at) >= stale:
            return None
        return slot

    @property
    def last_source(self) -> str:
        """最近一次记账的来源（``"engine"`` / ``"physical"`` / ``"none"``）。"""
        with self._lock:
            return self._source

    def age_ms(self) -> int | None:
        """这次记账有多旧。``None`` = 还没记过。诊断用。"""
        with self._lock:
            if self._slot is None:
                return None
            return self._now() - self._at_ms

    def snapshot(self) -> dict[str, object]:
        """诊断快照。给 YUI / Console 看。"""
        return {
            "slot": self.current(),
            "last_source": self.last_source,
            "age_ms": self.age_ms(),
            "stale_after_ms": self._stale_ms,
        }


class CharSlotIndex:
    """``槽位 <-> 角色名`` 的运行时索引（**免配置切人**的关键）。

    写入方：``StateBuilder._read_character_names``（队伍栏 OCR，每
    ``interval_ticks`` 轮转 1 格）。
    读取方：``IntentExecutor._slot_of`` —— ``SWITCH_CHARACTER(char_id='夜兰')``
    靠它换成「按数字键 2」。

    ★ 为什么需要它
    --------------
    规则里写的是**角色名**（``onfield('夜兰')``、``char_id = "夜兰"``），
    但引擎要按的是**数字键**。中间这层翻译以前靠用户在
    ``keymap.toml`` 手填 ``[char_slots]``。而队伍是每个玩家自己的信息，
    逼用户抄一遍角色名既烦又容易抄错 —— **抄错不会报错，只会切错人**。

    OCR 反正在大世界每一轮都在跑（名字就印在队伍栏上），白拿即可。
    实测：队伍栏名字 OCR 在 296 帧上单帧误识 **0.00%**，
    时序滤波后稳态正确率 100%（见 ``tests/unit/test_character_id_wiring.py``）。

    ★ 名字进来得**比想象的晚**
    -------------------------
    单次 OCR 实测 **1.0~1.5 秒**，是 200ms tick 预算的 5~7 倍，
    所以做成轮转（每轮 1 格，4 格一圈）。按默认
    ``interval_ticks = 25`` 算，**第一次填满要 ~20 秒**。
    在此之前 :meth:`slot_of` 返回 ``None``，
    ``SWITCH_CHARACTER`` 会跳过并在 steps 里记账 ——
    **不是静默失败，但也确实切不动人**。启动时会明说这一点。

    🔴 一个已知的真实缺陷
    -------------------
    槽位和名字是**分别缓存**的（名字来自轮转 OCR，槽位来自切人记账）。
    中途换队伍时，某个槽位可能**短暂挂着上一个队伍的名字**，
    于是 ``slot_of('夜兰')`` 会返回错误槽位、切到错的人。
    概率窗口 = OCR 轮转周期（~20 秒）。**不修**是因为它要求
    「新旧名字冲突」这个信号才能判，而 OCR 稳态下不会主动报错 ——
    拿不到证据的判据会「看起来在工作」。发现症状就调小
    ``interval_ticks``，或直接填 ``[char_slots]`` 覆盖。
    """

    def __init__(self) -> None:
        self._by_slot: dict[int, str] = {}
        self._lock = threading.Lock()

    # -- 写入 ------------------------------------------------------------
    def note_name(self, slot: int, name: str) -> bool:
        """记下「这个槽位现在站的是这个人」。

        :returns: 是否改变了名册（名字变了 / 之前是空的）。
        """
        s, n = int(slot), (name or "").strip()
        if s not in VALID_SLOTS:
            return False
        if not n:
            # ★ 空名字**不写入、也不清空**已有记录。
            #   理由与 ``CharacterDetector.update`` 相同：OCR 在切屏/开菜单
            #   时整体读不到，此时清空会让名册从「已知队伍」闪成空，
            #   执行层会突然切不动人。读不到就沿用上次的结果。
            return False
        with self._lock:
            if self._by_slot.get(s) == n:
                return False
            self._by_slot[s] = n
            return True

    def clear(self) -> None:
        with self._lock:
            self._by_slot.clear()

    # -- 读取 ------------------------------------------------------------
    def slot_of(self, char_id: str) -> int | None:
        """角色名 -> 槽位。查不到返回 ``None``。"""
        key = (char_id or "").strip()
        if not key:
            return None
        with self._lock:
            items = tuple(self._by_slot.items())
        for slot, name in items:
            if name == key:
                return slot
        return None

    def name_of(self, slot: int) -> str | None:
        """槽位 -> 角色名。"""
        with self._lock:
            return self._by_slot.get(int(slot))

    def mapping(self) -> dict[str, int]:
        """给 ``IntentExecutor(char_slots=...)`` 用的快照（``名字 -> 槽位``）。"""
        with self._lock:
            return {name: slot for slot, name in self._by_slot.items()}

    @property
    def complete(self) -> bool:
        """四格都认出来了？

        ``False`` 时 :meth:`slot_of` 可能对某些角色返回 ``None``。
        启动警告与 Console 面板用这个决定要不要提示用户。
        """
        with self._lock:
            return len(self._by_slot) == len(VALID_SLOTS)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {"slots": dict(self._by_slot), "complete": self.complete}


__all__ = ["DIGIT_SLOTS", "VALID_SLOTS", "ActiveSlotTracker", "CharSlotIndex"]
