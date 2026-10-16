"""``event_to_yui`` 适配器单测（2026-10-08）

★ 测什么
-------
本轮新增的这条链路「引擎事件 -> ``.stream.jsonl`` -> 追尾读 -> 翻译台词 ->
``push_message``」之前**零测试覆盖**，只在临时脚本里手验过。
这里把五个最容易悄悄坏掉的行为钉死：

1. 六条业务台词映射（``EVENT_TO_SPEECH``）
2. 节流：同 code 在窗口内只说一次
3. **半行 JSON 不丢**（offset 逐行推进，不是循环后 ``fh.tell()``）
4. 表外 code 兜底（``EventCode`` 是开放集合，不许崩）
5. 空内容闭嘴（占位符全缺时不播报「游戏那边」这种废话）

★ 为什么不碰宿主
--------------
适配器只依赖一个 ``push_fn`` 回调，测试注入假函数即可 ——
不需要起宿主、不需要 ``NekoPluginBase``，也不碰 ``autopilot`` 引擎。
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))


# ---------------------------------------------------------------- 夹具
class _Rig:
    """一套「假事件流目录 + 假 push + 真适配器」。"""

    def __init__(self, throttle: float = 5.0, templates=None,
                 poll: float = 99.0):
        self.dir = Path(tempfile.mkdtemp(prefix="e2y_test_"))
        self.run_id = "20261008_120000"
        (self.dir / "run_current.txt").write_text(
            self.run_id + "\n", encoding="utf-8")
        self.stream = self.dir / f"{self.run_id}.stream.jsonl"
        self.sent: list[dict] = []
        from event_to_yui import EventToYui
        self.a = EventToYui(
            lambda **kw: (self.sent.append(kw) or {"submitted": True}),
            self.dir, poll_seconds=poll, throttle_seconds=throttle,
            templates=templates)
        self.a.log.addHandler(_Null())

    @property
    def texts(self) -> list[str]:
        return [s["parts"][0]["text"] for s in self.sent]

    def write(self, code: str, message: str = "", **ctx) -> None:
        with self.stream.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"schema_version": "1.0", "run_id": self.run_id, "seq": 0,
                 "ts_ms": 0, "code": code, "severity": "INFO",
                 "message": message, "context": ctx},
                ensure_ascii=False) + "\n")

    def raw(self, text: str) -> None:
        with self.stream.open("a", encoding="utf-8") as fh:
            fh.write(text)

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)


class _Null(logging.Handler):
    """吞掉适配器的 INFO/DEBUG/WARNING，避免测试输出被日志淹没。

    ★ 必须是**真的** logging.Handler
      ----------------------
      最早这只是个带 addHandler/setLevel 的空壳，结果适配器一打
      ``logger.warning``，logging 就去读 ``hdlr.level`` —— 空壳没这个属性，
      AttributeError 把整个测试进程带走了。logging 的协议比看起来严。
    """

    def __init__(self) -> None:
        super().__init__(level=logging.CRITICAL)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        pass

import time

import pytest

from event_to_yui import (BUSINESS_CODES, EVENT_TO_SPEECH,
                          FALLBACK_PREFIX, EventToYui)


@pytest.fixture()
def rig():
    r = _Rig()
    yield r
    r.cleanup()


# ================================================================== T1 台词映射
def test_t1_business_codes_all_mapped():
    assert len(BUSINESS_CODES) == 6
    assert [c for c in BUSINESS_CODES if c not in EVENT_TO_SPEECH] == []
    # 工程腔词汇说明是写给日志看的，不是说给人听的
    for t in EVENT_TO_SPEECH.values():
        for w in ("激活", "感知", "关键词", "检测", "意图"):
            assert w not in t, f"{w!r} 出现在台词 {t!r}"


def test_t1_six_lines_rendered_and_pushed(rig):
    rig.write("PATHFIND_START", target_x=200.0, target_z=-300.0, dist_m=195.0)
    rig.write("ENEMY_DETECTED", count=3, nearest_m=8.5)
    rig.write("POSITION_CHANGED", x=2050.0, z=-800.0, dist_m=45.0)
    rig.write("PATHFIND_STUCK", level=3, moved_m=0.3)
    rig.write("ESCAPE_SUCCESS", level=3, moved_m=4.8)
    rig.write("PATHFIND_ARRIVED", x=2094.7, z=-707.7, spent_s=58.0)
    assert rig.a.drain_once() == 6
    assert rig.texts == ["走，去那边看看", "有怪！3个，小心点", "跑了45.0米了",
                         "又卡住了……", "出来了，继续", "到了到了"]
    # 占位符真的被替换了，不是把 "{count}" 原样推出去
    assert "3个" in rig.texts[1] and "45.0" in rig.texts[2]
    # respond = 要 YUI 开口；read 只进上下文不触发回复
    assert all(s["ai_behavior"] == "respond" for s in rig.sent)
    # visibility=["chat"] = 台词**原样上屏**。留空的话用户只能看到 YUI 的复述，
    # 看不到我们写的台词 —— 这是 2026-10-09 真机"没反应"的根因，别退回 []。
    assert all(s["visibility"] == ["chat"] for s in rig.sent)


# ================================================================== T2 节流
# ★ 样本码用 ENEMY_DETECTED 而不是 INTENT_EXECUTED（2026-10-11）
#   INTENT_EXECUTED 已进 SKIP_CODES（真机实测单 run 触发 533 次，占九成
#   以上播报量，整类不播）。拿它当节流样本会让本组用例测不到节流 ——
#   它们会全被跳过，然后在 assert drain_once() == 1 处红成一个跟节流
#   毫无关系的失败。节流要测的是节流，就用一只**会播**的码。
def test_t2_same_code_throttled(rig):
    for _ in range(4):
        rig.write("ENEMY_DETECTED", count=1)
    assert rig.a.drain_once() == 1
    assert len(rig.texts) == 1


def test_t2_throttle_zero_disables_dedup():
    r = _Rig(throttle=0.0)
    try:
        r.write("ENEMY_DETECTED", count=1)
        r.write("ENEMY_DETECTED", count=2)
        assert r.a.drain_once() == 2
    finally:
        r.cleanup()


def test_t2_different_codes_not_throttled(rig):
    rig.write("ENEMY_DETECTED", count=1)
    rig.a.drain_once()
    rig.write("PATHFIND_STUCK", level=2)
    rig.a.drain_once()
    assert len(rig.texts) == 2


# ================================================== SKIP_CODES（2026-10-11）
def test_skip_codes__intent_executed_never_broadcasts(rig):
    """INTENT_EXECUTED 有模板，但**整类不播**。

    真机实测单次 run 触发 533 次，而全 run 只有 4 种其它码 ——
    不跳过的话九成以上的播报都是同一句「按计划走了」。
    """
    from event_to_yui import SKIP_CODES
    assert "INTENT_EXECUTED" in SKIP_CODES
    assert "NORMAL_ATTACK" not in SKIP_CODES, (
        "NORMAL_ATTACK 是 context['action'] 的值，不是事件 code；"
        "加进来这张表就形同虚设")

    for i in range(50):
        rig.write("INTENT_EXECUTED", action="NORMAL_ATTACK")
    assert rig.a.drain_once() == 0
    assert rig.texts == []


def test_skip_codes__skipped_code_does_not_light_up_throttle(rig):
    """跳过**不能**顺手写 _last_spoke。

    节流窗口是按 code 记时间的；跳过时若把它点亮，等于宣称「这个 code
    刚说过」，将来真要播它时反而要多空等一个窗口。
    """
    from event_to_yui import SKIP_CODES
    code = next(iter(SKIP_CODES))
    rig.write(code)
    rig.a.drain_once()
    assert code not in rig.a._last_spoke


def test_skip_codes__other_codes_still_broadcast(rig):
    """跳过是**逐类**的，不能连坐。

    ⚠️ 2026-10-16 起 ``CORE_STARTED`` **也在** SKIP_CODES 里了（A4），
    所以这条不能再拿它当「不连坐」的样本 —— 改用 ``RUN_STARTED``：
    它与 ``CORE_STARTED`` 是同一件事的两次上报，开场只该说一次，
    而它**必须**还能说。
    """
    rig.write("RUN_STARTED")
    rig.a.drain_once()
    assert "启动了" in rig.texts


def test_skip_codes__core_started_is_now_skipped(rig):
    """★★ ``CORE_STARTED`` 被显式跳过（2026-10-16，A4）。

    它与 ``RUN_STARTED`` 是同一件事的两次上报，真机在 t=0.00s 同时触发；
    「引擎起来了」+「启动了」连着说，观众只会听到一句，
    而且「引擎」是工程词。详见 :data:`event_to_yui.SKIP_CODES` 里的说明。
    """
    from event_to_yui import SKIP_CODES

    assert "CORE_STARTED" in SKIP_CODES
    rig.write("CORE_STARTED")
    rig.a.drain_once()
    assert rig.texts == []


# ================================================================== T3 半行 JSON
def test_t3_torn_json_line_not_lost(rig):
    rig.raw('{"code":"TORN')
    o = rig.a._offset
    assert rig.a.drain_once() == 0
    assert rig.a._offset == o, "半行被吞：offset 推进了，这行永久丢失"
    rig.raw("\n" + json.dumps(
        {"code": "RUN_STOPPED", "severity": "INFO",
         "message": "", "context": {}}, ensure_ascii=False) + "\n")
    rig.a.drain_once()
    assert "歇下了" in rig.texts
    assert not any("TORN" in t for t in rig.texts)


# ================================================================== T4 表外 code
def test_t4_unknown_code_falls_back(rig):
    rig.write("BRAND_NEW_CODE_X", message="出事了了")
    assert rig.a.drain_once() == 1
    assert FALLBACK_PREFIX in rig.texts[0] and "出事了了" in rig.texts[0]


def test_t4_empty_code_ignored(rig):
    rig.write("")
    assert rig.a.drain_once() == 0
    assert rig.texts == []


# ================================================================== T5 空内容闭嘴
def test_t5_empty_render_stays_silent():
    r = _Rig(templates={"X_EMPTY": "{nope_a}{nope_b}"})
    try:
        r.write("X_EMPTY")
        assert r.a.drain_once() == 0
        assert r.texts == []
        r.write("X_EMPTY")            # 已记节流，不该每帧重试
        assert r.a.drain_once() == 0
    finally:
        r.cleanup()


def test_t5_fallback_without_message_stays_silent(rig):
    rig.write("NEW_CODE")             # 兜底模板 = 前缀 + "{message}"
    assert rig.a.drain_once() == 0
    assert rig.texts == []


# ================================================================== T6 线程
def test_t6_thread_lifecycle_and_really_reads(rig):
    rig.a._poll = 0.3
    assert rig.a.start() is True
    try:
        assert rig.a.start() is False          # 幂等，不重复起线程
        assert rig.a.running is True
        rig.write("PATHFIND_ARRIVED", x=1.0, z=2.0, spent_s=10.0)
        t0 = time.monotonic()
        while not rig.texts and (time.monotonic() - t0) < 5.0:
            time.sleep(0.05)
        assert rig.texts == ["到了到了"], "后台线程没读到文件"
    finally:
        rig.a.stop()
    assert rig.a.running is False


# ================================================================== T7 push 失败
def test_t7_push_exception_does_not_kill_reader(rig):
    def _boom(**_kw):
        raise RuntimeError("宿主不可用")

    rig.a._push = _boom
    rig.write("PATHFIND_ARRIVED", x=1.0, z=2.0, spent_s=10.0)
    assert rig.a.drain_once() == 0


def test_t7_not_submitted_counts_as_failure(rig):
    rig.a._push = lambda **_k: {"submitted": False, "reason": "队列满"}
    rig.write("PATHFIND_ARRIVED", x=1.0, z=2.0, spent_s=10.0)
    assert rig.a.drain_once() == 0


def test_t7_construction_has_no_side_effect():
    """构造适配器不许起线程 / 读文件（KB R8 的同一精神）。"""
    d = Path(tempfile.mkdtemp(prefix="e2y_ct_"))
    try:
        a = EventToYui(lambda **_k: {"submitted": True}, d)
        assert a.running is False
        assert a._thread is None
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
