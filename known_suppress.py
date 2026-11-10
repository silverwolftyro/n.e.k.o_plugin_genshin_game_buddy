"""攻略场景 ``known_suppress`` 的主动抑制登记簿。

背景
----
``terms/genshin/scenes.toml`` 里每个攻略场景都带 ``known_suppress = true``：
它表示"这个场景的攻略内容一旦被玩家看到（同话题），就不该再主动提"。

上一轮（2026-11-09）只把这个字段做成**可读可查**（``scene_store.SceneEntry``），
这一轮把它接到主动提示层，形成闭环::

    记录侧（场景内容注入成功）──▶ KnownSuppressRegistry ──▶ 消费侧（主动提示决策）

模块边界
--------
* **纯内存**：只存 ``dict[str, tuple[float, bool]]``，**不落盘、不写文件**。
  抑制窗口本来就只有分钟级，跨会话保留只会带来"上次会话的旧记录"这种噪声。
* **零第三方依赖**：只用标准库 ``threading`` / ``time``。
* **零网络**：与 ``game_data_store`` 同规矩 —— 本模块不 import 任何网络库。

时钟
----
用 ``time.monotonic()``（单调钟）：不受系统时间跳变（NTP 校正、用户改时区、
夏令时）影响。记录与查询走同一个钟，"窗口内 / 窗口外"的判定不会因为墙上
时间被改动而错乱。测试可以显式传 ``now`` 覆盖。

场景 id 空间
------------
这里的 scene_id 指的是**攻略场景**（``scenes.toml`` 里的"深境螺旋"等），
与 ``chat_state.py`` / ``event_to_yui.py`` 用的**引擎场景**（WORLD / COMBAT /
IDLE / UNKNOWN）是两套不相交的 id 空间。本模块只服务攻略场景。
"""

from __future__ import annotations

import threading
import time

__all__ = ["DEFAULT_WINDOW_SECONDS", "KnownSuppressRegistry"]


#: 默认抑制窗口（秒）。600 秒 = 10 分钟。
#:
#: 为什么是 10 分钟：攻略内容注入之后，玩家真正"读完并用上"的时间窗大约就是
#: 一次战斗 + 一次复盘（3~8 分钟）。窗口太短（如 60s）会在玩家还没消化完时
#: 又被主动提起；太长（如 1h）会让玩家离开该场景后很久都听不到该场景的
#: 主动提示。默认取中间值，具体由 ``known_suppress_window_s`` 配置覆盖。
DEFAULT_WINDOW_SECONDS = 600.0


class KnownSuppressRegistry:
    """进程内的"攻略场景内容已注入"登记簿。

    ★ 线程安全：``record`` 由 OCR 工作线程调用，``should_suppress`` 由定时器
      线程调用，两边不是同一个线程。用 ``threading.Lock`` 保护 dict。
      临界区里只做 dict 读写，不 await、不做 IO。
    """

    def __init__(self, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> None:
        # 负窗口没有意义，归零（= 关闭抑制）。
        self._window = max(0.0, float(window_seconds))
        # scene_id -> (注入时刻, 该场景的 known_suppress 快照)
        self._records: dict[str, tuple[float, bool]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------

    @property
    def window_seconds(self) -> float:
        """当前生效的抑制窗口（秒）。0 = 关闭抑制。"""
        return self._window

    def set_window(self, window_seconds: float) -> None:
        """改窗口（配置热更新时用）。负数按 0 处理。"""
        self._window = max(0.0, float(window_seconds))

    # ------------------------------------------------------------------
    # 记录侧
    # ------------------------------------------------------------------

    def record(self, scene_id: str, *, known_suppress: bool, now: float | None = None) -> bool:
        """记下"``scene_id`` 的攻略内容刚刚被注入语境"。

        :param scene_id: 攻略场景 id（``scenes.toml`` 的表名）。
        :param known_suppress: 该场景的抑制标记快照（来自 ``SceneEntry``）。
        :param now: 覆盖时间戳（测试用）；默认取 ``time.monotonic()``。
        :return: 是否真的记下了（空 scene_id 返回 False）。
        """
        key = (scene_id or "").strip()
        if not key:
            return False
        stamp = time.monotonic() if now is None else float(now)
        with self._lock:
            self._records[key] = (stamp, bool(known_suppress))
            self._prune_locked(stamp)
        return True

    # ------------------------------------------------------------------
    # 查询侧
    # ------------------------------------------------------------------

    def should_suppress(self, scene_id: str, now: float | None = None) -> bool:
        """该场景是否应当被主动提示层抑制。

        判据（两条都满足才 True）：

        1. 该 scene_id 在窗口内被注入过语境；
        2. 该场景的 ``known_suppress`` 为 True。

        窗口关闭（``window_seconds <= 0``）时恒 False —— 回到改造前行为。
        """
        if self._window <= 0:
            return False
        key = (scene_id or "").strip()
        if not key:
            return False
        stamp = time.monotonic() if now is None else float(now)
        with self._lock:
            record = self._records.get(key)
        if record is None:
            return False
        injected_at, known_suppress = record
        if not known_suppress:
            return False
        # age 为负（时钟异常 / 传了更早的 now）时按"新鲜"处理：
        # 宁可少说一次，也不要误判成"窗口已过"又多嘴。
        return (stamp - injected_at) < self._window

    def last_injected_at(self, scene_id: str) -> float | None:
        """最近一次注入的时间戳；没有记录返回 None（排查日志用）。"""
        key = (scene_id or "").strip()
        if not key:
            return None
        with self._lock:
            record = self._records.get(key)
        return None if record is None else record[0]

    def known_suppress_flag(self, scene_id: str) -> bool:
        """该场景记录的 ``known_suppress`` 快照；没有记录返回 False。"""
        key = (scene_id or "").strip()
        if not key:
            return False
        with self._lock:
            record = self._records.get(key)
        return False if record is None else bool(record[1])

    def forget(self, scene_id: str) -> bool:
        """忘掉某场景的记录（场景退出 / 配置重置时用）。"""
        key = (scene_id or "").strip()
        if not key:
            return False
        with self._lock:
            return self._records.pop(key, None) is not None

    def clear(self) -> None:
        """清空全部记录（配置重载时用）。"""
        with self._lock:
            self._records.clear()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _prune_locked(self, now: float) -> None:
        """丢掉窗口外的旧记录，保持 dict 有界。

        ★ 调用方必须已持有 ``self._lock``。
        ★ 窗口关闭（<=0）时**不**清理：此时记录虽然不参与判定，但保留下来
          能让"关掉再打开"时立刻恢复原状态。
        """
        if self._window <= 0:
            return
        stale = [key for key, (stamp, _flag) in self._records.items() if (now - stamp) >= self._window]
        for key in stale:
            del self._records[key]
