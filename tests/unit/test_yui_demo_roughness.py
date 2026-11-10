"""Demo 出厂前粗糙度修复的回归锁（2026-10-16）。

覆盖审计里的 A1 / A3 / A4：
    A1  同一句在一次 run 里最多说 N 次（真机实测曾 x39）
    A3  ``SCENE_CHANGED`` 的 ``to=UNKNOWN`` 不播（工程枚举）
    A4  ``CORE_STARTED`` / ``RUN_STARTED`` 同槽位互斥（开场不口吃）

全部用**真实事件流路径**（写临时 jsonl -> ``drain_once``）驱动，
不用直接调 ``render`` —— 节流、槽位、配额都挂在读流路径上。

依据：``recognition/notes/demo_roughness_audit.md``
"""

from __future__ import annotations

import json
from pathlib import Path

from event_to_yui import (
    EVENT_TO_SPEECH_POOLS,
    MAX_LINES_PER_RUN,
    REPEAT_PRONE_CODES,
    SKIP_CODES,
    SPEECH_SLOTS,
    EventToYui,
)


def _drive(tmp_path: Path, events: list[dict]) -> list[str]:
    """把 events 写成事件流，跑一轮 ``drain_once``，返回实际说出口的话。

    ★ ``throttle_seconds=0.0`` 是**故意关掉节流**的。
      一轮 ``drain_once`` 里的事件时间戳几乎相同，5 秒节流会把第 2 条
      就挡掉 —— 那样测到的永远是「1 句」，根本碰不到重复上限。
      关掉之后才测得到 A1 本身（配额），节流由 test_event_to_yui.py 负责。
    """
    said: list[str] = []
    rt = tmp_path / "rt"
    rt.mkdir(parents=True, exist_ok=True)
    (rt / "run_current.txt").write_text("t", encoding="utf-8")
    with (rt / "t.stream.jsonl").open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    EventToYui(
        push_fn=lambda **kw: (
            said.append("".join(x.get("text", "") for x in kw["parts"]))
            or {"submitted": True}),
        runtime_dir=rt,
        throttle_seconds=0.0,
        # ★ 场景组冷却**关掉**
        #   这份测试验的是「重复上限 / 场景枚举 / 槽位」三层，
        #   场景组冷却（Companion Mode）是第四层 —— 它若开着，
        #   同组内第二条非 tier1 台词会被它挡掉，
        #   测到的就不是被测的那一层了。
        #   场景组冷却由 ``test_yui_companion_mode.py`` 单独钉。
        scene_group_cooldown_s=0.0).drain_once()
    return said


def _ev(code: str, **ctx) -> dict:
    return {"code": code, "message": code, "context": ctx}


# ==================================================================
# A1 · 同句重复上限
# ==================================================================
def test_perception_down_is_capped(tmp_path: Path) -> None:
    """★ 真实机跑里这句话出现过 39 次（207s，每 5.1s 一遍）。

    现在最多说 :data:`MAX_LINES_PER_RUN` 次。
    """
    events = [_ev("PERCEPTION_DOWN")] * 40
    said = _drive(tmp_path, events)
    # ★ 判据用**池子成员**，不用固定片段 ——
    #   台词池会随文案变长（「看不清」只在第二句里，拿它当判据会时灵时不灵），
    #   而这条用例验的是「重复上限」，不是某一句台词。
    #   按池子成员计数 -> 池子以后加词 / 换词，这条用例都不会假失败。
    pool = EVENT_TO_SPEECH_POOLS["PERCEPTION_DOWN"]
    hits = [s for s in said if any(line in s for line in pool)]
    assert len(hits) == MAX_LINES_PER_RUN, said


def test_intent_rejected_is_capped(tmp_path: Path) -> None:
    """★ 另一条真机复读 31 次的（155s）。"""
    events = [_ev("INTENT_REJECTED")] * 40
    said = _drive(tmp_path, events)
    assert len([s for s in said if "没敢做" in s]) == MAX_LINES_PER_RUN, said


def test_repeat_cap_does_not_silence_normal_codes() -> None:
    """★ 上限只约束 REPEAT_PRONE_CODES。

    ``RUN_STARTED`` 低频且有语义，**不许**被这条规则误伤。
    """
    assert "RUN_STARTED" not in REPEAT_PRONE_CODES
    assert "ENEMY_DETECTED" not in REPEAT_PRONE_CODES
    assert "PERCEPTION_DOWN" in REPEAT_PRONE_CODES
    assert "INTENT_REJECTED" in REPEAT_PRONE_CODES


def test_single_instance_budget_resets_across_runs(tmp_path: Path) -> None:
    """★★ 单个适配器实例连跑两个 run，第二个 run 仍能开口。

    真实生产里 ``EventToYui`` 是**常驻单例**，跨 run 不重建。
    配额若不随 run 切换清空，第二个 run 全哑 ——
    表现为「第一次跑会说话，第二次跑一声不吭」。
    """
    rt = tmp_path / "rt"
    rt.mkdir(parents=True, exist_ok=True)
    said: list[str] = []
    obj = EventToYui(
        push_fn=lambda **kw: (
            said.append("".join(x.get("text", "") for x in kw["parts"]))
            or {"submitted": True}),
        runtime_dir=rt,
        throttle_seconds=0.0)

    def feed(run_id: str, n: int) -> None:
        (rt / "run_current.txt").write_text(run_id, encoding="utf-8")
        with (rt / f"{run_id}.stream.jsonl").open("w", encoding="utf-8") as fh:
            for _ in range(n):
                fh.write(json.dumps(_ev("PERCEPTION_DOWN"),
                                    ensure_ascii=False) + "\n")
        obj.drain_once()

    feed("runA", 10)
    after_a = len(said)
    feed("runB", 1)

    assert after_a == MAX_LINES_PER_RUN, said
    assert len(said) == after_a + 1, (
        f"第二个 run 应该还能说 1 句，实际总共 {len(said)} 句")


# ==================================================================
# A3 · SCENE_CHANGED 的 UNKNOWN 不播
# ==================================================================
def test_scene_changed_unknown_is_dropped(tmp_path: Path) -> None:
    got = _drive(tmp_path, [_ev("SCENE_CHANGED", to="UNKNOWN")])
    assert got == [], got


def test_scene_changed_known_scene_still_speaks(tmp_path: Path) -> None:
    """★ 只压 UNKNOWN，真实场景名照常播 —— 不能连功能一起关掉。"""
    got = _drive(tmp_path, [_ev("SCENE_CHANGED", to="WORLD")])
    assert len(got) == 1, got


def test_scene_changed_unknown_survives_lowercase(tmp_path: Path) -> None:
    """★ 大小写不敏感 —— 真机枚举可能写成 unknown。"""
    got = _drive(tmp_path, [_ev("SCENE_CHANGED", to="unknown")])
    assert got == [], got


# ==================================================================
# A4 · 开场 / 收尾槽位互斥
# ==================================================================
def test_core_started_and_run_started_do_not_both_speak(tmp_path: Path) -> None:
    """★★ 真机实测这两个在 t=0.00s 同时触发 -> 观众听到「引擎起来了」
    紧接着「开始咯~ 主人今天想干嘛？」。现在 ``CORE_STARTED`` 直接跳过，只剩一句。
    """
    got = _drive(tmp_path, [_ev("CORE_STARTED"), _ev("RUN_STARTED")])
    assert got == ["开始咯~ 主人今天想干嘛？"], got


def test_lifecycle_slot_prefers_the_human_word(tmp_path: Path) -> None:
    """★ 活下来的必须是「开始咯~ 主人今天想干嘛？」，不是「引擎起来了」——

    后者含「引擎」这种工程词，正是审计里「台词全是工程黑话」的来源。
    显式走 SKIP_CODES 而不是靠槽位时序：``CORE_STARTED`` 恒为 seq=1，
    先到先得的槽位必然让它抢赢。
    """
    assert "CORE_STARTED" in SKIP_CODES
    assert "CORE_STOPPED" in SKIP_CODES
    assert "RUN_STARTED" not in SKIP_CODES
    got = _drive(tmp_path, [_ev("RUN_STARTED"), _ev("CORE_STARTED")])
    assert got == ["开始咯~ 主人今天想干嘛？"], got


def test_stop_slot_also_single(tmp_path: Path) -> None:
    """★ 收尾同理：CORE_STOPPED 跳过，RUN_STOPPED 一句。"""
    got = _drive(tmp_path, [_ev("CORE_STOPPED"), _ev("RUN_STOPPED")])
    assert got == ["歇下了~ 主人辛苦啦。"], got


def test_stopping_and_stopped_do_not_both_speak(tmp_path: Path) -> None:
    """★ RUN_STOPPING / RUN_STOPPED 共享 stop 槽位，只放行先到的那个。"""
    got = _drive(tmp_path, [_ev("RUN_STOPPING"), _ev("RUN_STOPPED")])
    assert len(got) == 1, got


def test_slot_resets_across_runs(tmp_path: Path) -> None:
    """★★ 槽位必须随 run 清空，否则第二个 run 开场直接哑掉。"""
    rt = tmp_path / "rt"
    rt.mkdir(parents=True, exist_ok=True)
    said: list[str] = []
    obj = EventToYui(
        push_fn=lambda **kw: (
            said.append("".join(x.get("text", "") for x in kw["parts"]))
            or {"submitted": True}),
        runtime_dir=rt,
        throttle_seconds=0.0)

    for run_id in ("runA", "runB"):
        (rt / "run_current.txt").write_text(run_id, encoding="utf-8")
        with (rt / f"{run_id}.stream.jsonl").open("w", encoding="utf-8") as fh:
            for code in ("CORE_STARTED", "RUN_STARTED"):
                fh.write(json.dumps(_ev(code), ensure_ascii=False) + "\n")
        obj.drain_once()

    assert len(said) == 2, said
    assert all(t in EVENT_TO_SPEECH_POOLS["RUN_STARTED"] for t in said), said
    assert said[0] != said[1], f"连着两句不该是同一句：{said}"


# ==================================================================
# 回归：这几处修复不许误伤「正常播报」
# ==================================================================
def test_normal_demo_flow_speaks_the_world_part_only(tmp_path: Path) -> None:
    """★ 一次正常演示：**该说的说了，不该说的闭嘴了**。

    ★ 2026-10-29（Companion Mode）语义变了
      场景状态机接管了「现在能不能开口」，于是：

        · ``RUN_STARTED``        -> 世界态，tier2 -> 播
        · ``ENEMY_DETECTED`` ×2  -> 世界态，tier2 -> 都播
          （刷怪防抖在这里是**关**的：``throttle_seconds=0`` 时它一并关，
            那是既有语义，见 ``test_dry_run_and_speech_noise.py``）
        · ``SCENE_CHANGED to=COMBAT`` -> 进战斗，tier3 -> **闭嘴**
        · ``RUN_STOPPED``        -> 还在战斗态，tier2 -> **闭嘴**

      最后两条闭嘴，正是用户的第一条诉求：「战斗状态 YUI 闭嘴，
      仅保留关键异常播报」。
    """
    got = _drive(tmp_path, [
        _ev("CORE_STARTED"),
        _ev("RUN_STARTED"),
        _ev("ENEMY_DETECTED", count=2),
        _ev("ENEMY_DETECTED", count=1),
        _ev("SCENE_CHANGED", to="COMBAT"),
        _ev("RUN_STOPPED"),
    ])
    assert len(got) == 3, got
    assert got[0] == "开始咯~ 主人今天想干嘛？", got
    # ★ 比的是**渲染后**的台词（池子里存的是带 ``{count}`` 的模板）
    enemy_pool = {line.format(count=2)
                  for line in EVENT_TO_SPEECH_POOLS["ENEMY_DETECTED"]}
    assert got[1] in enemy_pool, got
    enemy_pool1 = {line.format(count=1)
                   for line in EVENT_TO_SPEECH_POOLS["ENEMY_DETECTED"]}
    assert got[2] in enemy_pool1, got
    assert not any("引擎起来了" in g for g in got), "开场仍被压制的那句不该出现"


def test_unknown_scene_does_not_eat_a_normal_scene(tmp_path: Path) -> None:
    """★ 被压掉的 UNKNOWN 不该影响后续真实场景的播报。"""
    got = _drive(tmp_path, [
        _ev("SCENE_CHANGED", to="UNKNOWN"),
        _ev("SCENE_CHANGED", to="WORLD"),
    ])
    # ★ 场景 id **不再念出来**（``_PROMPT_GUIDE.md`` §6.4）
    assert got == ["换个地方啦~"], got
