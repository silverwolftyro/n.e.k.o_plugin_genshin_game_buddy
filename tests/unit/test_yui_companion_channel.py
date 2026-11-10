"""YUI「陪伴台词」通道的契约（2026-10-20 起，2026-10-22 补「第一人称」）。

★ 这个通道是干什么的
  ``companion_push.py`` 和独立触发器（``_demo_standalone``）往事件流里写一条
  ``COMPANION_LINE``，适配器读到后让 YUI **按原话**说一句。
  它是「录视频时猫娘陪聊」的唯一入口。

★ 四条必须成立的契约
  ------------------------------------------------------------------
  1. **逐字、无前缀**
     表外 code 会走兜底模板 ``FALLBACK_PREFIX``（「游戏那边 …」）——
     陪伴台词带这个前缀就变成「转述播报」，不是「陪着说话」了。

  2. **不被 5 秒通用节流吞掉**
     通用节流是「同一个 code 多久能再说一次」，给状态类事件设计。
     用户按自己的节奏推台词（``--every 3``），5 秒窗口会**悄悄吞掉**
     其中一半 —— 症状是「我明明推了，它没说」。

  3. **静音档也听得见**
     用户主动要它说，档位必须是「关键」而不是默认的「最啰嗦」。
     忘了登记 = 落在 ``TIER_VERBOSE`` = 用户把滑块调到 0.5 就全被压掉。

  4. ★ **第一人称原样说**（2026-10-22 新增）
     用户实测症状：推过去的台词被宿主当成「插件发来的通知」，
     YUI 像复读机一样说「连插件都在夸你啦」，
     而不是自己说出「主人你随便莽」。
     ⇒ 推给 LLM 的文本 = **硬指令 + 【台词】**；
     并且 ``visibility=[]`` —— 硬指令一个字都不许漏到聊天框里
     （观众看到的应该是 YUI 说的话，不是我们的内部指令）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_to_yui import (
    CODE_TIER,
    COMPANION_LINE_PREFIX,
    COMPANION_LINE_VISIBILITY,
    DRY_RUN_MUTED_CODES,
    EVENT_TO_SPEECH,
    FALLBACK_PREFIX,
    REPEAT_PRONE_CODES,
    SKIP_CODES,
    SPEECH_SLOTS,
    THROTTLE_EXEMPT_CODES,
    TIER_CRITICAL,
    EventToYui,
    tier_of,
)

pytestmark = pytest.mark.unit

CODE = "COMPANION_LINE"


def _drain(runtime: Path, payloads: list[dict], **kwargs) -> list[str]:
    """把事件写进真格式的流，让真 ``EventToYui`` 追尾渲染。"""
    said: list[str] = []
    (runtime / "run_current.txt").write_text("t", encoding="utf-8")
    with (runtime / "t.stream.jsonl").open("w", encoding="utf-8") as fh:
        for p in payloads:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    EventToYui(
        push_fn=lambda **kw: (
            said.append("".join(x.get("text", "") for x in kw["parts"]))
            or {"submitted": True}),
        runtime_dir=runtime, **kwargs).drain_once()
    return said


def _event(text: str, seq: int = 1) -> dict:
    return {"run_id": "t", "seq": seq, "code": CODE,
            "severity": "INFO", "message": text,
            "context": {"text": text}}


def _assert_first_person(said: list[str], line: str) -> None:
    """★ 推给 LLM 的文本 = 硬指令 + 【台词】，台词一字不差地在里面。

    ★ 为什么断言的是「包含」而不是「等于」
      硬指令是**故意**加在台词外面的 ——
      宿主的插件 API 没有「原样念出来」的选项（``ai_behavior`` 只有
      ``respond|read|blind``），只能靠硬指令把模型钉在「第一人称原样说」。
      所以「整条推送文本 == 台词」这个断言已经不成立了；
      成立的是「台词**一字不差**地出现在里面」。
    """
    assert len(said) == 1, said
    text = said[0]
    assert text.startswith(COMPANION_LINE_PREFIX), text
    assert text.endswith(f"【{line}】"), text
    assert line in text, text


# ==================================================================
# 契约 1：逐字、无前缀
# ==================================================================

class TestVerbatim:
    def test_template_is_verbatim(self) -> None:
        assert EVENT_TO_SPEECH[CODE] == "{text}"

    def test_says_exactly_what_was_pushed(self, tmp_path: Path) -> None:
        said = _drain(tmp_path, [_event("主人，这边这边")])
        _assert_first_person(said, "主人，这边这边")

    def test_no_fallback_prefix(self, tmp_path: Path) -> None:
        """★ 绝不能带「游戏那边」前缀。"""
        said = _drain(tmp_path, [_event("我在这儿呢")])
        assert said and not said[0].startswith(FALLBACK_PREFIX)
        assert FALLBACK_PREFIX not in said[0]

    def test_message_is_kept_as_fallback(self, tmp_path: Path) -> None:
        """``message`` 也带同一句 —— 模板没登记时兜底还能把话送到。"""
        ev = _event("备用通道")
        assert ev["message"] == ev["context"]["text"]


# ==================================================================
# 契约 4：★ 第一人称原样说（2026-10-22）
# ==================================================================

class TestFirstPerson:
    def test_prefix_is_the_hard_instruction(self) -> None:
        """★ 硬指令的内容 —— 换句话说「你就是 YUI 本人」。"""
        assert "你现在就是 YUI 本人" in COMPANION_LINE_PREFIX
        assert "第一人称" in COMPANION_LINE_PREFIX
        assert "不要说这是插件说的" in COMPANION_LINE_PREFIX

    def test_push_carries_the_hard_instruction(self, tmp_path: Path) -> None:
        """★ 推给 LLM 的文本**以硬指令开头**。"""
        said = _drain(tmp_path, [_event("主人你随便莽")])
        assert said[0].startswith(COMPANION_LINE_PREFIX), said[0]

    def test_the_line_is_inside_the_brackets(self, tmp_path: Path) -> None:
        """★ 台词用【】包住 —— 给模型一个明确的「复述边界」。

        少了括号，模型经常把指令本身也复述出来
        （观众会看到「你现在就是 YUI 本人…：主人你随便莽」整句）。
        """
        said = _drain(tmp_path, [_event("主人你随便莽")])
        assert said[0].endswith("【主人你随便莽】"), said[0]

    def test_the_instruction_is_not_shown_on_screen(self, tmp_path: Path) -> None:
        """★ 观众只看 YUI 说的那一句 —— 聊天框里不许出现工程腔指令。"""
        pushed: list[dict] = []
        (tmp_path / "run_current.txt").write_text("t", encoding="utf-8")
        with (tmp_path / "t.stream.jsonl").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(_event("在吗"), ensure_ascii=False) + "\n")
        EventToYui(
            push_fn=lambda **kw: (pushed.append(kw) or {"submitted": True}),
            runtime_dir=tmp_path).drain_once()
        assert len(pushed) == 1
        assert pushed[0]["visibility"] == COMPANION_LINE_VISIBILITY
        assert pushed[0]["visibility"] == [], pushed[0]["visibility"]

    def test_there_is_still_a_voice(self, tmp_path: Path) -> None:
        """★ 必须是 ``respond`` —— 声音挂在 LLM 回复上。

        ``blind`` 才是结构上的零加工，但那样**视频里没有 YUI 的声音**
        （台词完全不进 LLM，没人开口）。所以这里保留 respond，
        用硬指令约束模型原样说。
        """
        pushed: list[dict] = []
        (tmp_path / "run_current.txt").write_text("t", encoding="utf-8")
        with (tmp_path / "t.stream.jsonl").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(_event("在吗"), ensure_ascii=False) + "\n")
        EventToYui(
            push_fn=lambda **kw: (pushed.append(kw) or {"submitted": True}),
            runtime_dir=tmp_path).drain_once()
        assert pushed[0]["ai_behavior"] == "respond", pushed[0]

    def test_other_codes_are_not_affected(self, tmp_path: Path) -> None:
        """★ 硬指令**只对陪伴台词生效** —— 状态事件仍旧逐字上屏。

        状态事件（「发现了 3 个敌人」）本来就该让模型用自己的话接一句，
        裹上「你就是 YUI 本人」反而会把它逼成复述。
        """
        pushed: list[dict] = []
        (tmp_path / "run_current.txt").write_text("t", encoding="utf-8")
        with (tmp_path / "t.stream.jsonl").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"run_id": "t", "seq": 1, "code": "ENEMY_DETECTED",
                 "message": "发现1个敌人", "context": {"count": 1}},
                ensure_ascii=False) + "\n")
        EventToYui(
            push_fn=lambda **kw: (pushed.append(kw) or {"submitted": True}),
            runtime_dir=tmp_path).drain_once()
        assert len(pushed) == 1
        assert pushed[0]["visibility"] == ["chat"], pushed[0]["visibility"]
        text = "".join(x.get("text", "") for x in pushed[0]["parts"])
        assert not text.startswith(COMPANION_LINE_PREFIX), text


# ==================================================================
# 契约 2：不被通用节流吞掉
# ==================================================================

class TestThrottleExemption:
    def test_code_is_throttle_exempt(self) -> None:
        assert CODE in THROTTLE_EXEMPT_CODES

    def test_three_lines_in_one_batch_all_speak(self, tmp_path: Path) -> None:
        """★★ 核心用例：5 秒窗口内连推 3 条，**三条都要说出来**。

        ★ 为什么会红
          通用节流按 code 记时间。第一条说完后 ``_last_spoke[CODE]``
          被点亮，接下来 5 秒内的同 code 事件全被丢掉 ——
          于是「推了 3 句只说了 1 句」。用户推的节奏由
          ``--every`` 决定，节流无权吞。
        """
        payloads = [_event("第一句", 1), _event("第二句", 2), _event("第三句", 3)]
        said = _drain(tmp_path, payloads, throttle_seconds=5.0)
        assert len(said) == 3, said
        for i, line in enumerate(("第一句", "第二句", "第三句")):
            _assert_first_person([said[i]], line)

    def test_other_codes_still_throttled(self, tmp_path: Path) -> None:
        """★ 豁免**只对陪伴台词生效**，别的 code 的节流不能被带坏。"""
        payloads = [
            {"run_id": "t", "seq": 1, "code": "ENEMY_DETECTED",
             "message": "发现1个敌人", "context": {"count": 1}},
            {"run_id": "t", "seq": 2, "code": "ENEMY_DETECTED",
             "message": "发现1个敌人", "context": {"count": 1}},
        ]
        said = _drain(tmp_path, payloads, throttle_seconds=5.0)
        assert len(said) == 1


# ==================================================================
# 契约 3：静音档也听得见
# ==================================================================

class TestTier:
    def test_registered_as_critical(self) -> None:
        assert tier_of(CODE) == TIER_CRITICAL
        assert CODE_TIER[CODE] == TIER_CRITICAL

    def test_speaks_at_minimum_verbosity(self, tmp_path: Path) -> None:
        """用户把滑块拉到 0（静音档）时，陪伴台词**照样播**。"""
        said = _drain(tmp_path, [_event("还在吗")], verbosity=0.0)
        _assert_first_person(said, "还在吗")

    def test_speaks_at_half_verbosity(self, tmp_path: Path) -> None:
        said = _drain(tmp_path, [_event("还在吗")], verbosity=0.5)
        _assert_first_person(said, "还在吗")


# ==================================================================
# 不该被别的闸门拦住
# ==================================================================

class TestNotBlocked:
    def test_not_in_skip_codes(self) -> None:
        assert CODE not in SKIP_CODES

    def test_not_in_dry_run_muted_codes(self) -> None:
        """干跑（引擎没在打）时陪伴台词**照样播**。

        ★ 用户录视频时引擎很可能只开着观察模式；这时候
          「猫娘说句话」正是想要的，不是「战斗 chatter」。
        """
        assert CODE not in DRY_RUN_MUTED_CODES

    def test_speaks_in_dry_run(self, tmp_path: Path) -> None:
        said = _drain(tmp_path, [_event("干跑也要说")],
                      run_mode_fn=lambda: "DRY_RUN")
        _assert_first_person(said, "干跑也要说")

    def test_not_in_speech_slots(self) -> None:
        """陪伴台词不占生命周期槽位（不跟「引擎起来了」抢）。"""
        assert CODE not in SPEECH_SLOTS

    def test_not_repeat_prone(self) -> None:
        """不受「一句最多说 2 次」的限制 —— 推几次说几次。"""
        assert CODE not in REPEAT_PRONE_CODES

    def test_repeats_are_not_capped(self, tmp_path: Path) -> None:
        payloads = [_event(f"第{i}句", i) for i in range(1, 6)]
        said = _drain(tmp_path, payloads)
        assert len(said) == 5
