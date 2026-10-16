"""子系统故障守卫：连续失败才判死，连续成功才复活。

★ 为什么需要它
--------------
上一版的容错是"单帧失败即判死"，在 5 Hz（200ms/帧）下后果很重：
``core/loop.py:525`` 看到 ``perception_ok=False`` 会**零阈值立即**
``_auto_pause()``。所以一次 YOLO 瞬时异常就会让整个 autopilot 自停。

:class:`FaultGuard` 把"抖一下"和"真坏了"分开：

    连续失败 N 帧 -> 判死；连续成功 M 帧 -> 复活

★ 为什么要双向迟滞
------------------
仓库里已有的五种容错模式（``capturer._consecutive_fail`` /
``ocr.fail_streak`` / ``StableValueAccumulator`` /
``_consecutive_capture_failures`` / ``consecutive_crashes``）
**全部是"一成功就归零"**，没有复活迟滞。那会在阈值附近震荡：
判死 -> 一帧成功就复活 -> 一帧失败又判死 -> ...

所以复活要**独立计数**。推荐配比是 ``recover_threshold`` **小于**
``fail_threshold``（难判死、易复活），从时间上看"放弃"比"恢复"更保守。

★ 阈值不是拍脑袋的
------------------
按 ``tick_rate_hz = 5.0``（200 ms/帧）换算，全部落在 600 ms ~ 2 s：

===========  ==================  ==================
子系统        fail（判死）        recover（复活）
===========  ==================  ==================
YOLO         3 帧 = 600 ms       2 帧 = 400 ms
OCR          5 帧 = 1.0 s        2 帧 = 400 ms
定位          3 帧 = 600 ms       2 帧 = 400 ms
===========  ==================  ==================

OCR 取 5 不是因为它更脆，是因为 **``ocr.DEFAULT_FAIL_THRESHOLD`` 本身就是 5** ——
builder 不能比它包装的模块更早判死，否则 OCR 引擎还在挣扎、
builder 已经报死。
"""

from __future__ import annotations

__all__ = ["FaultGuard"]


class FaultGuard:
    """连续失败 ``fail_threshold`` 次判死，连续成功 ``recover_threshold`` 次复活。

    :param fail_threshold: 连续失败多少次才判死（≥1）
    :param recover_threshold: 判死后连续成功多少次才复活（≥1）

    用法::

        g = FaultGuard(fail_threshold=3, recover_threshold=2)
        try:
            do_work()
        except Exception:
            if g.record_failure():
                log.error("子系统判死")
        else:
            if g.record_success():
                log.info("子系统复活")
    """

    def __init__(self, fail_threshold: int, recover_threshold: int) -> None:
        f, r = int(fail_threshold), int(recover_threshold)
        if f < 1:
            raise ValueError(f"fail_threshold 必须 >= 1，收到 {fail_threshold}")
        if r < 1:
            raise ValueError(f"recover_threshold 必须 >= 1，收到 {recover_threshold}")
        self._fail_threshold = f
        self._recover_threshold = r
        self.reset()

    # -- 参数 ------------------------------------------------------------
    @property
    def fail_threshold(self) -> int:
        return self._fail_threshold

    @property
    def recover_threshold(self) -> int:
        return self._recover_threshold

    # -- 状态 ------------------------------------------------------------
    @property
    def is_failed(self) -> bool:
        """当前是否处于"判死"状态。"""
        return self._failed

    @property
    def consecutive_failures(self) -> int:
        """当前连续失败数，**封顶在 ``fail_threshold``**。"""
        return self._fail_streak

    @property
    def consecutive_successes(self) -> int:
        """当前连续成功数，**封顶在 ``recover_threshold``**。"""
        return self._ok_streak

    def reset(self) -> None:
        """清空状态，回到"健康"。"""
        self._failed = False
        self._fail_streak = 0
        self._ok_streak = 0

    # -- 记录 ------------------------------------------------------------
    def record_failure(self) -> bool:
        """记一次失败。

        :returns: ``True`` = **这一次刚好判死**（仅状态翻转那一次为 True，
            之后继续失败都返回 False，不会重复触发）
        """
        self._ok_streak = 0
        self._fail_streak = min(self._fail_streak + 1, self._fail_threshold)
        if not self._failed and self._fail_streak >= self._fail_threshold:
            self._failed = True
            return True
        return False

    def record_success(self) -> bool:
        """记一次成功。

        :returns: ``True`` = **这一次刚好复活**（仅状态翻转那一次为 True）
        """
        self._fail_streak = 0
        self._ok_streak = min(self._ok_streak + 1, self._recover_threshold)
        if self._failed and self._ok_streak >= self._recover_threshold:
            self._failed = False
            return True
        return False

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        state = "FAILED" if self._failed else "ok"
        return (f"FaultGuard({state}, fail={self._fail_streak}/"
                f"{self._fail_threshold}, ok={self._ok_streak}/"
                f"{self._recover_threshold})")
