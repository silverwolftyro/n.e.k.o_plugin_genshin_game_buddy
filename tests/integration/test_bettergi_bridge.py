# -*- coding: utf-8 -*-
"""genshin_game_buddy · BetterGI 桥接离线测试。

★ 2026-11-07：从独立插件 ``bettergi_companion`` 移植过来。
  移植时**目标变了**：原来打的是 ``BettergiCompanionPlugin``（插件类），
  现在打的是 ``BettergiBridge``（SDK 无关的桥接层）。

  为什么换目标
  ------------
  合并后桥接逻辑不再寄生在插件类上，而是下沉到 ``bettergi/bridge.py``：

  * 插件类只做**接线**（构造桥接、注入 ``push_message`` / ``logger``、转发
    ``@llm_tool`` 调用）；
  * 桥接层**不 import NEKO 插件 SDK**，所以离线测试不必构造 172 KB 的
    插件类，也就不必拉起 OCR / 会话探测 / 感知状态机那一整套。

  ⇒ 同样的用例覆盖同样的行为，但跑得快一个数量级，且不碰宿主。

不依赖 NEKO 宿主，也不依赖 BetterGI：
* 用 **mock ctx** 顶替宿主的 ``push_message``（v2 签名）。
* 用 ``port = 0`` 让操作系统分配临时端口，避免和真实插件抢占 17890。

运行方式::

    cd <插件目录>
    python -m pytest tests/integration/test_bettergi_bridge.py -q
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ── 定位插件目录 ────────────────────────────────────────────────────────
# __file__ = <插件>/tests/integration/test_bettergi_bridge.py
#   parents[0] = integration
#   parents[1] = tests
#   parents[2] = <插件>
PLUGIN_DIR = Path(__file__).resolve().parents[2]
BETTERGI_DIR = PLUGIN_DIR / "bettergi"

from plugin.plugins.genshin_game_buddy.bettergi.bridge import (  # noqa: E402
    CONTROL_SELF_CHECK_REMINDER,
    BettergiBridge,
)
from plugin.plugins.genshin_game_buddy.bettergi.control import (  # noqa: E402
    CommandQueue,
)
from plugin.plugins.genshin_game_buddy.bettergi.event_mapper import (  # noqa: E402
    DEFAULT_CONFIG,
    GROUP_COMBAT,
    GROUP_MENU,
    GROUP_TRANSITION,
    GROUP_WORLD,
    SCENE_GROUP,
    EventMapper,
    group_for_scene,
)

#: §6.1 规定的 ``domain.start`` 随机池。
DOMAIN_START_LINES = (
    "来了来了，主人尽管冲，我在旁边看着~",
    "打架了打架了！……主人小心点呀。",
    "开打！输了我可不哄你哦~",
    "自动打手上线啦，主人快找好位置看戏~",
)

#: §6.1 规定的 ``domain.end`` 随机池。
DOMAIN_END_LINES = (
    "打完啦~ 主人歇会儿吧，喘口气。",
    "呼——总算结束了，主人喝水去。",
    "辛苦了笨蛋主人，我给你记着战绩呢。",
    "自动打手完工啦，主人快来验收战绩~",
)


# ══════════════════════════════════════════════════════════════════════
# Mock 件
# ══════════════════════════════════════════════════════════════════════
class MockLogger:
    """收集日志的假 logger。"""

    def __init__(self) -> None:
        self.records: List[Tuple[str, str]] = []

    def _emit(self, level: str, message: str) -> None:
        self.records.append((level, message))

    def debug(self, message: str, *a: Any, **k: Any) -> None:
        self._emit("debug", message)

    def info(self, message: str, *a: Any, **k: Any) -> None:
        self._emit("info", message)

    def warning(self, message: str, *a: Any, **k: Any) -> None:
        self._emit("warning", message)

    def error(self, message: str, *a: Any, **k: Any) -> None:
        self._emit("error", message)

    def exception(self, message: str, *a: Any, **k: Any) -> None:
        self._emit("error", message)


class MockCtx:
    """模拟宿主的 ``ctx``：提供 ``logger`` 与 ``push_message``。"""

    def __init__(self) -> None:
        self.logger = MockLogger()
        self.pushed: List[Dict[str, Any]] = []
        self.push_should_fail = False

    def push_message(self, **kwargs: Any) -> Dict[str, Any]:
        """签名对齐 SDK 的 v2 ``push_message``。"""
        if self.push_should_fail:
            raise RuntimeError("模拟的推送失败")
        self.pushed.append(kwargs)
        return {"submitted": True}


# ── 工具 ────────────────────────────────────────────────────────────────
def post(url: str, payload: Any, timeout: float = 5.0) -> Tuple[int, Any]:
    """POST 一个 JSON 到本地服务器，返回 ``(状态码, 解析后的响应)``。"""
    if isinstance(payload, (bytes, str)):
        body = payload.encode("utf-8") if isinstance(payload, str) else payload
    else:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def get(url: str, timeout: float = 5.0) -> Tuple[int, Any]:
    """GET 一个本地 URL，返回 ``(状态码, 解析后的响应)``。"""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def make_bridge(
    port: int = 0,
    control: Optional[Dict[str, Any]] = None,
) -> Tuple[BettergiBridge, MockCtx]:
    """造一个用 mock ctx 驱动的桥接实例（走真实 ``start`` 路径）。

    ``port=0`` 让操作系统分配临时端口，避免和真实插件抢占 17890。
    ``control`` 用来模拟宿主配置面板里的 ``[genshin_game_buddy.bettergi.control]`` 段。
    """
    ctx = MockCtx()
    bridge = BettergiBridge(push_fn=ctx.push_message, logger=ctx.logger)
    section: Dict[str, Any] = {
        "webhook": {"host": "127.0.0.1", "port": port, "path": "/bettergi"}
    }
    if control is not None:
        section["control"] = dict(control)
    status = bridge.start(section)
    assert status.get("status") == "running", f"桥接没能起来: {status}"
    return bridge, ctx


def pushed_texts(ctx: MockCtx) -> List[str]:
    """从 mock 记录里取出所有文本片段。"""
    out: List[str] = []
    for call in ctx.pushed:
        for part in call.get("parts", []):
            if part.get("type") == "text":
                out.append(part.get("text", ""))
    return out


def inner_line(text: str) -> str:
    """从「硬指令前缀 + 【台词】」里取出干净的台词。

    台词推给 LLM 时会被包成 ``COMPANION_LINE_PREFIX + "【" + 台词 + "】"``。
    用【】是为了给模型一个明确的「要复述的边界」—— 少了括号模型会把指令本身
    也复述出来。
    """
    start = text.rfind("【")
    end = text.rfind("】")
    assert start != -1 and end > start, f"台词没有被【】包住: {text!r}"
    return text[start + 1 : end]


# ══════════════════════════════════════════════════════════════════════
# 正向链路：BetterGI 事件 → 台词 → 推送
# ══════════════════════════════════════════════════════════════════════
def test_four_events_produce_pushes_under_group_cooldown() -> None:
    """四个登记事件 → 两次 push_message（场景组冷却生效后的预期）。

    ★ 为什么不是四次（Companion Mode）
      ``domain.start`` / ``domain.end`` / ``task.error`` 都属于
      **COMBAT_GROUP** —— 一次战斗就是「开打 ->（可能失败）-> 打完了」，
      属于同一段场景。它们被**同一把 90s 锁**压住，
      所以这一串里只有第一句能出口。

      ``js.custom`` 是**用户主动推**的台词（tier 1），
      不受场景组冷却限制 —— 用户要听就得听到。
    """
    bridge, ctx = make_bridge()
    try:
        url = bridge.server.url
        status, body = post(url, {"event": "domain.start", "message": "开始"})
        assert status == 200, f"domain.start 状态码 {status}"
        assert body.get("handled") is True, f"domain.start 未处理: {body}"

        # ↓↓ 这两个被 COMBAT_GROUP 的 90s 锁压住
        status, body = post(url, {"event": "domain.end", "message": "结束"})
        assert status == 200, f"domain.end: {status}"
        assert body.get("handled") is False, f"domain.end 应被场景组冷却压住: {body}"

        status, body = post(url, {"event": "task.error", "message": "失败"})
        assert status == 200, f"task.error: {status}"
        assert body.get("handled") is False, f"task.error 应被场景组冷却压住: {body}"

        status, body = post(
            url, {"event": "js.custom", "message": "主人我捡到东西啦", "data": "主人我捡到东西啦"}
        )
        assert status == 200 and body.get("handled") is True, f"js.custom: {status} {body}"

        assert len(ctx.pushed) == 2, f"预期 2 次推送，实际 {len(ctx.pushed)}"
        assert bridge.mapper.stats["dropped_scene_group"] == 2, (
            f"场景组冷却计数不对: {bridge.mapper.stats['dropped_scene_group']}"
        )

        texts = pushed_texts(ctx)
        # ★ 推给 LLM 的文本 = 硬指令前缀 + 【台词】
        assert inner_line(texts[0]) in DOMAIN_START_LINES, f"domain.start 台词不对: {texts[0]}"
        assert inner_line(texts[1]) == "主人我捡到东西啦", f"js.custom 未原样透传: {texts[1]}"

        # 推送参数必须是 v2 形态
        call = ctx.pushed[0]
        # ★ visibility 必须是**空列表** —— 硬指令不该上屏，观众只看 YUI 说的那一句
        assert call.get("visibility") == [], f"visibility 应为空列表: {call.get('visibility')!r}"
        assert call.get("ai_behavior") == "respond", f"ai_behavior 不对: {call.get('ai_behavior')}"
        # ★ 单插件架构下来源标识是插件自己的 id（不再是 bettergi_companion）
        assert call.get("source") == "genshin_game_buddy", f"source 不对: {call.get('source')}"
        assert call.get("parts", [{}])[0].get("type") == "text", "parts 类型不对"
        # 弃用字段一个都不能带值。
        for legacy in ("message_type", "content", "delivery", "reply", "description",
                       "binary_data", "binary_url", "mime"):
            assert not call.get(legacy), f"用了已弃用字段 {legacy}={call.get(legacy)!r}"
        assert call.get("unsafe") is not True, "unsafe 不得为 True"
        assert call.get("fast_mode") is not True, "fast_mode 不得为 True"
    finally:
        bridge.stop()


def test_line_is_pushed_as_first_person_instruction() -> None:
    """台词必须以「硬指令前缀 + 【台词】」推给 LLM，且不直接上屏。

    契约：
      · parts 文本以第一人称硬指令开头（强制原样说）
      · 台词被【】包住（给模型明确的复述边界）
      · ``visibility == []``（内部指令不直接上屏）
      · ``ai_behavior == "respond"``（有声音，走 LLM 回合）
      · 日志里记的是**干净台词**（不含前缀），便于排查
    """
    bridge, ctx = make_bridge()
    try:
        status, body = post(bridge.server.url, {"event": "domain.end", "message": "结束"})
        assert status == 200 and body.get("handled") is True, f"{status} {body}"

        assert len(ctx.pushed) == 1, f"预期 1 次推送，实际 {len(ctx.pushed)}"
        call = ctx.pushed[0]
        text = call["parts"][0]["text"]

        # ① 硬指令前缀（强制第一人称原样说）
        assert text.startswith("（你现在就是 YUI 本人。"), f"缺少第一人称硬指令: {text!r}"
        assert "不要说这是插件说的" in text, f"硬指令不完整: {text!r}"
        # ② 【】边界
        assert inner_line(text) in DOMAIN_END_LINES, f"【】里的台词不对: {text!r}"
        # ③ 不直接上屏
        assert call.get("visibility") == [], f"visibility 应为空列表: {call.get('visibility')!r}"
        # ④ 有声音（不能是 blind，否则 TTS 没声音）
        assert call.get("ai_behavior") == "respond", f"ai_behavior 不对: {call.get('ai_behavior')}"
        # ⑤ 日志记干净台词，不带硬指令
        infos = [m for lvl, m in ctx.logger.records if lvl == "info"]
        assert any(line in m for line in DOMAIN_END_LINES for m in infos), f"日志里应有干净台词: {infos}"
        assert not any("你现在就是 YUI 本人" in m for m in infos), f"日志里不该有硬指令: {infos}"
    finally:
        bridge.stop()


def test_unknown_event_is_ignored() -> None:
    """未登记事件 → 被忽略，不推送。"""
    bridge, ctx = make_bridge()
    try:
        status, body = post(bridge.server.url, {"event": "totally.unknown", "message": "x"})
        assert status == 200, f"状态码 {status}"
        assert body.get("handled") is False, f"未登记事件不应被处理: {body}"
        assert len(ctx.pushed) == 0, f"未登记事件却推送了 {len(ctx.pushed)} 次"
        assert bridge.mapper.stats["dropped_unknown_event"] == 1, "统计没记 unknown"
    finally:
        bridge.stop()


def test_rate_limit_blocks_repeat_event() -> None:
    """同一事件快速连发 6 次 → 只有 1 次推送（限流）。"""
    bridge, ctx = make_bridge()
    try:
        for _ in range(6):
            status, _ = post(bridge.server.url, {"event": "domain.start", "message": "开始"})
            assert status == 200, f"状态码 {status}"
        assert len(ctx.pushed) == 1, f"预期 1 次推送（限流），实际 {len(ctx.pushed)}"
        assert bridge.mapper.stats["dropped_cooldown"] == 5, (
            f"限流计数不对: {bridge.mapper.stats['dropped_cooldown']}"
        )
    finally:
        bridge.stop()


def test_anti_repeat_blocks_same_line() -> None:
    """同一句台词在防复读窗口内不重复（用多条台词的事件把限流绕开）。"""
    mapper = EventMapper(anti_repeat_s=30.0)
    rule = mapper.config["events"]["domain.start"]
    rule["cooldown_s"] = 0  # 关掉限流，只留防复读
    # ★ 关掉场景组冷却
    #   否则它会先一步把第二句挡掉 —— 那样测到的就不是防复读了，
    #   而是场景组冷却（那是另一条用例的事）。
    mapper.config["yui"]["scene_group_cooldown_s"] = 0

    # ★ 池子大小会随文案变，所以这里按**实际池长**推，
    #   而不是硬编码「3 句」—— 否则加一句台词就会把这条用例变成假失败。
    pool_size = len(mapper.config["events"]["domain.start"]["lines"])
    picked = [mapper.map_event("domain.start", message=str(i)) for i in range(pool_size + 1)]

    assert picked[0] is not None, "第一次应该有台词"
    assert picked[1] is not None, "第二次应该换一句"
    assert picked[0] != picked[1], f"两次说了同一句: {picked[0]}"
    assert all(p is not None for p in picked[:pool_size]), (
        f"前 {pool_size} 句都该有（台词池有 {pool_size} 句）"
    )
    assert picked[pool_size] is None, (
        f"第 {pool_size + 1} 句应该被防复读挡掉（台词池只有 {pool_size} 句）"
    )


# ══════════════════════════════════════════════════════════════════════
# 场景组冷却（Companion Mode）
# ══════════════════════════════════════════════════════════════════════
def test_scene_group_table_covers_all_thirteen_ids() -> None:
    """★ 13 个真实场景 id 一个不漏地落进 4 个组。

    ★ 为什么要钉
      漏一个 id，那个场景的台词就**永远不受组冷却约束** ——
      退化成改造前的行为，而且**没人会发现**（因为它只是「多说了几句」）。
    """
    assert len(SCENE_GROUP) == 13, f"场景表应有 13 个 id，实际 {len(SCENE_GROUP)}"
    assert set(SCENE_GROUP.values()) == {
        GROUP_WORLD, GROUP_COMBAT, GROUP_MENU, GROUP_TRANSITION}
    assert SCENE_GROUP["WORLD"] == GROUP_WORLD
    assert SCENE_GROUP["COMBAT"] == GROUP_COMBAT
    assert SCENE_GROUP["COMBAT_RESULT"] == GROUP_COMBAT
    for menu_scene in ("MAIN_MENU", "DIALOG", "BAG_INVENTORY", "MAP_BIG",
                       "GAME_MENU", "NOTICE", "TELEPORT_MAP"):
        assert SCENE_GROUP[menu_scene] == GROUP_MENU, menu_scene
    for transition_scene in ("LOADING", "CUTSCENE", "UNKNOWN"):
        assert SCENE_GROUP[transition_scene] == GROUP_TRANSITION, transition_scene
    # 不认识的场景 id 兜底到 TRANSITION（防刷屏的最保守选择）
    assert group_for_scene("SOMETHING_NEW") == GROUP_TRANSITION


def test_scene_group_cooldown_blocks_the_next_line_in_same_group() -> None:
    """★ 同一场景组内，非关键台词 90s 只准一句。"""
    mapper = EventMapper()
    rule = mapper.config["events"]["domain.start"]
    rule["cooldown_s"] = 0        # 关掉按事件码的限流，只留组冷却
    mapper.config["events"]["domain.end"]["cooldown_s"] = 0

    assert mapper.map_event("domain.start") is not None, "第一句应该能出口"
    assert mapper.map_event("domain.end") is None, "同一场景组的第二句应被压住"
    assert mapper.map_event("task.error") is None, "第三个事件也该被压住"
    assert mapper.stats["dropped_scene_group"] == 2, mapper.stats


def test_scene_group_cooldown_expires_after_the_window() -> None:
    """★ 窗口过了就放行 —— 冷却不是永久静音。"""
    now = {"t": 1000.0}
    mapper = EventMapper(clock=lambda: now["t"])
    for name in ("domain.start", "domain.end"):
        mapper.config["events"][name]["cooldown_s"] = 0

    assert mapper.map_event("domain.start") is not None
    now["t"] += 89.0
    assert mapper.map_event("domain.end") is None, "89s 时还该被压着"
    now["t"] += 1.0
    assert mapper.map_event("domain.end") is not None, "90s 后应该放行"


def test_user_pushed_line_is_exempt_from_group_cooldown() -> None:
    """★ 用户主动推的台词（js.custom）不受场景组冷却限制。

    他要听就得听到 —— 被 90s 锁挡掉会让人以为通道坏了。
    """
    mapper = EventMapper()
    # 关掉按事件码的限流，只留场景组冷却 —— 否则是限流在挡它
    mapper.config["events"]["js.custom"]["cooldown_s"] = 0
    assert mapper.map_event("js.custom", data="主人我捡到东西啦") is not None
    assert mapper.map_event("js.custom", data="主人我捡到东西啦") is not None
    assert mapper.stats["dropped_scene_group"] == 0, mapper.stats


def test_scene_group_cooldown_can_be_disabled_by_config() -> None:
    """★ 调成 0 = 关掉这一层，退回改造前的行为。"""
    mapper = EventMapper()
    mapper.config["yui"]["scene_group_cooldown_s"] = 0
    mapper.config["events"]["domain.start"]["cooldown_s"] = 0
    mapper.config["events"]["domain.end"]["cooldown_s"] = 0

    assert mapper.map_event("domain.start") is not None
    assert mapper.map_event("domain.end") is not None, "关掉后不该再被组冷却压住"


def test_new_persona_lines_replaced_the_old_ones() -> None:
    """★ §6.1 的随机池两处都要生效：内置默认 + 插件目录的 config.toml。"""
    assert list(DEFAULT_CONFIG["events"]["domain.start"]["lines"]) == list(DOMAIN_START_LINES)
    assert list(DEFAULT_CONFIG["events"]["domain.end"]["lines"]) == list(DOMAIN_END_LINES)

    # 插件目录里的 config.toml（运行时生效值）也要是同一套
    live = EventMapper().config
    assert list(live["events"]["domain.start"]["lines"]) == list(DOMAIN_START_LINES)
    assert list(live["events"]["domain.end"]["lines"]) == list(DOMAIN_END_LINES)

    # ★ task.error 池：内置默认与 config.toml 必须是**同一套**。
    assert list(DEFAULT_CONFIG["events"]["task.error"]["lines"]) == list(
        live["events"]["task.error"]["lines"]
    )

    # ★ 旧台词不许还留着
    blob = repr(DEFAULT_CONFIG) + repr(live)
    for old in ("开打开打，主人掩护我！", "冲鸭！", "呼，打完了，主人歇口气。",
                "唔，这次没成，主人别急。"):
        assert old not in blob, f"旧台词还在: {old}"


def test_engineering_jargon_is_not_spoken() -> None:
    """★ §3 硬边界第 1 条：台词里不许出现工程黑话。"""
    banned = ("检测", "感知", "激活", "意图", "阈值", "provider", "fps",
              "场景换成了", "引擎", "主循环", "模块")
    mapper = EventMapper()
    lines: List[str] = []
    for name in ("domain.start", "domain.end", "task.error"):
        for _ in range(20):
            text = mapper.map_event(name)
            if text:
                lines.append(text)
            mapper.reset_cooldowns()
    assert lines, "应该抽到台词"
    for line in lines:
        for word in banned:
            assert word not in line, f"{word!r} 出现在台词 {line!r}"


def test_naive_local_timestamp_is_accepted() -> None:
    """BetterGI 发的是**不带时区**的 ISO-8601 本地时间 —— 不能被误判成重放。

    这是实测踩过的坑：若按 UTC 解释，每条通知都会凭空差一个时区偏移，
    结果就是每次都被 409 拒收，插件彻底不工作。
    """
    bridge, ctx = make_bridge()
    try:
        naive = datetime.now().isoformat()
        status, body = post(bridge.server.url, {"event": "domain.end", "timestamp": naive})
        assert status == 200, f"本地时间戳应被接受，实际 {status} {body}"
        assert len(ctx.pushed) == 1, "本地时间戳的请求应正常推送"
    finally:
        bridge.stop()


def test_screenshot_field_is_accepted_and_dropped() -> None:
    """带 screenshot（base64 JPEG）的通知要被接受，且不占内存。"""
    bridge, ctx = make_bridge()
    try:
        fake_jpeg = "A" * 4096
        status, body = post(
            bridge.server.url,
            {"event": "domain.end", "message": "结束", "screenshot": fake_jpeg},
        )
        assert status == 200, f"带 screenshot 的请求应被接受，实际 {status} {body}"
        assert len(ctx.pushed) == 1
    finally:
        bridge.stop()


def test_wrong_path_and_bad_json() -> None:
    """路径不对 → 404；非法 JSON → 400。"""
    bridge, ctx = make_bridge()
    try:
        base = f"http://127.0.0.1:{bridge.server.port}"
        status, _ = post(base + "/wrong", {"event": "domain.start"})
        assert status == 404, f"错误路径应 404，实际 {status}"

        status, _ = post(bridge.server.url, "{not json")
        assert status == 400, f"非法 JSON 应 400，实际 {status}"

        status, _ = post(bridge.server.url, {"message": "没有 event"})
        assert status == 400, f"缺 event 应 400，实际 {status}"
    finally:
        bridge.stop()


def test_stale_timestamp_is_rejected() -> None:
    """带超窗 timestamp 的请求 → 409（防重放）。"""
    bridge, ctx = make_bridge()
    try:
        status, body = post(
            bridge.server.url, {"event": "domain.start", "timestamp": 1000000000}
        )
        assert status == 409, f"过期时间戳应 409，实际 {status}"
        assert len(ctx.pushed) == 0, "过期请求不应推送"
    finally:
        bridge.stop()


def test_push_failure_does_not_kill_server() -> None:
    """push_message 抛异常 → 服务器继续活着，错误被记下来。

    契约：事件已被插件消费，推送失败属于「插件选择不说话」，
    所以 HTTP 仍回 200 + handled=false（BetterGI 不需要重试），
    但必须留下 ERROR 日志。
    """
    ctx = MockCtx()
    ctx.push_should_fail = True
    bridge = BettergiBridge(push_fn=ctx.push_message, logger=ctx.logger)
    status = bridge.start({"webhook": {"host": "127.0.0.1", "port": 0, "path": "/bettergi"}})
    assert status.get("status") == "running", status
    try:
        code, body = post(bridge.server.url, {"event": "domain.end", "message": "x"})
        assert code == 200, f"推送异常不应让 Webhook 报错，实际 {code}"
        assert body.get("handled") is False, f"handled 应为 false: {body}"
        assert bridge.server.is_running, "服务器应仍在运行"

        # 再推一次，服务器依然能收
        code, _ = post(bridge.server.url, {"event": "domain.end", "message": "y"})
        assert code == 200, "服务器已死"

        errors = [m for lvl, m in ctx.logger.records if lvl == "error"]
        assert errors, "推送失败必须留下 ERROR 日志"
        assert any("push_message" in m for m in errors), f"错误日志内容不对: {errors}"
    finally:
        bridge.stop()


def test_missing_config_falls_back_to_defaults() -> None:
    """配置文件缺失 → 回落到内置默认映射，不崩。"""
    mapper = EventMapper(Path(__file__).resolve().parent / "_不存在的配置.toml")
    assert mapper.webhook_config["port"] == 17890, "默认端口不对"
    assert "domain.start" in mapper.known_events(), "默认事件表不对"
    assert mapper.map_event("domain.start") is not None, "默认映射应能出台词"


# ══════════════════════════════════════════════════════════════════════
# 反向控制通道（YUI → BetterGI）
# ══════════════════════════════════════════════════════════════════════
def test_control_queue_roundtrip() -> None:
    """指令队列：入队 → 取走 → 回执 → 取结果。"""
    queue = CommandQueue()
    cmd = queue.enqueue("start_auto", {"note": "测试"})

    assert queue.pending_count == 1, "入队后应有 1 条待发指令"

    taken = queue.take_next(hold_s=0.5)
    assert taken is not None, "应该能取到指令"
    assert taken.command_id == cmd.command_id
    assert taken.action == "start_auto"
    assert taken.args == {"note": "测试"}
    assert queue.pending_count == 0, "取走后队列应为空"
    assert queue.take_next(hold_s=0.1) is None, "队列空了不该再取到东西"

    assert queue.take_result(cmd.command_id) is None, "还没回执时不该有结果"

    assert queue.ack(cmd.command_id, True, "已经开始自动战斗了。") is True
    result = queue.take_result(cmd.command_id)
    assert result is not None, "回执后应该能取到结果"
    assert result.ok is True
    assert "自动战斗" in result.message


def test_control_queue_rejects_unknown_action() -> None:
    """未知动作必须被拒绝（不能让 LLM 随便下指令）。"""
    queue = CommandQueue()
    try:
        queue.enqueue("drop_table")
    except ValueError as exc:
        assert "drop_table" in str(exc)
    else:
        raise AssertionError("未知动作应该抛 ValueError")


def test_control_endpoints_over_http() -> None:
    """HTTP 端点：``GET /command`` 取指令，``POST /ack`` 回执。"""
    bridge, ctx = make_bridge(
        control={"enabled": True, "ack_timeout_s": 5, "poll_hold_s": 0.3}
    )
    try:
        base = bridge.server.url
        assert base.endswith("/bettergi"), f"基础地址不对: {base}"
        queue = bridge.control_queue
        assert queue is not None, "control 开启后应该有指令队列"

        # 没有指令时：204
        status, body = get(base + "/command")
        assert status == 204, f"空队列应返回 204，实际 {status} / {body}"

        # 入队一条 → GET 拿到
        cmd = queue.enqueue("auto_fishing")
        status, body = get(base + "/command")
        assert status == 200, f"有指令时应返回 200，实际 {status}"
        assert body["command_id"] == cmd.command_id
        assert body["action"] == "auto_fishing"

        # 回执
        status, body = post(base + "/ack", {"command_id": cmd.command_id, "ok": True, "message": "开始钓鱼"})
        assert status == 200, f"回执应返回 200，实际 {status}"
        assert body["matched"] is True
        result = queue.take_result(cmd.command_id)
        assert result is not None and result.ok is True
    finally:
        bridge.stop()


def test_command_bettergi_reports_success() -> None:
    """工具调用：BetterGI 回执成功 → 返回 is_error=False。"""
    bridge, ctx = make_bridge(control={"enabled": True, "ack_timeout_s": 5})
    try:
        queue = bridge.control_queue
        assert queue is not None

        def responder() -> None:
            cmd = queue.take_next(hold_s=3.0)
            if cmd is not None:
                queue.ack(cmd.command_id, True, "已经开始自动战斗了。")

        worker = threading.Thread(target=responder, daemon=True)
        worker.start()
        result = asyncio.run(bridge.dispatch_command("start_auto"))
        worker.join(timeout=3)

        assert result["is_error"] is False, f"应该成功: {result}"
        assert "自动战斗" in result["output"], f"输出不对: {result}"
        assert result["action"] == "start_auto"
    finally:
        bridge.stop()


def test_command_bettergi_reports_failure() -> None:
    """工具调用：BetterGI 回执失败 → 返回 is_error=True，且**如实**带回原因。"""
    bridge, ctx = make_bridge(control={"enabled": True, "ack_timeout_s": 5})
    try:
        queue = bridge.control_queue
        assert queue is not None

        def responder() -> None:
            cmd = queue.take_next(hold_s=3.0)
            if cmd is not None:
                queue.ack(cmd.command_id, False, "角色已死亡，无法继续战斗")

        worker = threading.Thread(target=responder, daemon=True)
        worker.start()
        result = asyncio.run(bridge.dispatch_command("start_auto"))
        worker.join(timeout=3)

        assert result["is_error"] is True, f"失败必须标 is_error: {result}"
        assert "角色已死亡" in result["output"], f"失败原因必须如实带回: {result}"
        assert "角色已死亡" in result["error"], f"error 字段也要有原因: {result}"
    finally:
        bridge.stop()


def test_command_bettergi_times_out() -> None:
    """工具调用：没人回执 → 超时并返回 is_error=True（不能假装成功）。"""
    bridge, ctx = make_bridge(control={"enabled": True, "ack_timeout_s": 1})
    try:
        result = asyncio.run(bridge.dispatch_command("start_auto"))
        assert result["is_error"] is True, f"超时时必须报错: {result}"
        assert "回执" in result["output"], f"超时提示要说清楚: {result}"
        assert "JS HTTP 权限" in result["output"], f"超时提示要给出排查方向: {result}"
    finally:
        bridge.stop()


def test_command_bettergi_rejects_unknown_action() -> None:
    """工具调用：未知动作 → is_error=True，不能瞎发。"""
    bridge, ctx = make_bridge(control={"enabled": True, "ack_timeout_s": 5})
    try:
        result = asyncio.run(bridge.dispatch_command("rm_rf"))
        assert result["is_error"] is True, f"未知动作必须报错: {result}"
        assert "rm_rf" in result["output"], f"要说清是哪个动作不认: {result}"
    finally:
        bridge.stop()


def test_command_bettergi_disabled_when_control_off() -> None:
    """反向控制**显式**关掉时，工具必须明确报错，而不是静默成功。

    ★ 注意语义
      ``[control].enabled`` 的默认值从 ``false`` 改成了 **true**
      （反向控制已经是核心功能，不该让用户先翻配置）。
      所以「关掉」必须**显式**做 —— 这里用宿主配置段覆盖成 ``False``。
    """
    bridge, ctx = make_bridge(control={"enabled": False})
    try:
        assert bridge.control_queue is None, "显式关掉后不该建指令队列"
        result = asyncio.run(bridge.dispatch_command("start_auto"))
        assert result["is_error"] is True, f"没开启时必须报错: {result}"
        assert "没有启用" in result["output"], f"要说清是通道没开: {result}"
    finally:
        bridge.stop()


# ══════════════════════════════════════════════════════════════════════
# 启动自检（反向通道通没通）
# ══════════════════════════════════════════════════════════════════════
def test_control_section_is_actually_read_from_config_toml() -> None:
    """``[control]`` 段必须真的被读进 EventMapper 的配置里。

    ★ 这是一个真 bug 的回归测试
      以前 ``event_mapper.load_config()`` 只合并 ``[webhook]`` 和
      ``[events.*]``，**从来不读 ``[control]``** ——
      于是桥接里 ``mapper.config["control"]`` 永远是空，
      ``[control].enabled`` 写 true 也等于没写，反向通道根本开不起来。
    """
    mapper = EventMapper(BETTERGI_DIR / "config.toml")
    control = mapper.config.get("control")
    assert isinstance(control, dict), f"[control] 段没被读进来: {list(mapper.config.keys())}"
    assert control.get("enabled") is True, f"[control].enabled 默认应为 true: {control}"
    assert float(control.get("ack_timeout_s", 0)) > 0, "ack_timeout_s 没读到"
    assert float(control.get("poll_hold_s", 0)) > 0, "poll_hold_s 没读到"
    assert float(control.get("self_check_delay_s", 0)) > 0, "self_check_delay_s 没读到"
    assert int(control.get("self_check_max", 0)) >= 0, "self_check_max 没读到"


def test_control_is_enabled_by_default() -> None:
    """反向控制**默认就是开的** —— 不用用户先翻配置。"""
    bridge, ctx = make_bridge()
    try:
        assert bridge.control_queue is not None, "默认应该建指令队列（[control].enabled 默认 true）"
        assert bridge.server.command_path.endswith("/command"), "取指令端点应该在"
    finally:
        bridge.stop()


def test_self_check_reminds_when_channel_is_not_connected() -> None:
    """通道没接上 → 自检推一条提醒，让 YUI 开口叫用户去配置。"""
    bridge, ctx = make_bridge()
    try:
        assert bridge.server.control_poll_count == 0, "还没人取过指令"
        connected = bridge.do_self_check()
        assert connected is False, "没人取指令 = 没接上"

        assert len(ctx.pushed) == 1, f"应该提醒一次，实际 {len(ctx.pushed)} 次"
        text = pushed_texts(ctx)[0]
        assert inner_line(text) == CONTROL_SELF_CHECK_REMINDER, f"提醒文案不对: {text!r}"
        # 提醒本身也要走「硬指令前缀 + 不上屏」那一套
        assert ctx.pushed[0].get("visibility") == [], "提醒不该直接上屏"
        assert ctx.pushed[0].get("ai_behavior") == "respond", "提醒要有声音"
        assert ctx.pushed[0].get("metadata", {}).get("event") == "control.self_check"
    finally:
        bridge.stop()


def test_self_check_stays_quiet_once_connected() -> None:
    """BetterGI 的代理来取过一次指令 → 自检不再打扰用户。"""
    bridge, ctx = make_bridge(control={"enabled": True, "poll_hold_s": 0.3})
    try:
        # 假装自己是 BetterGI 的 JS 代理：取一次指令
        status, _ = get(bridge.server.url + "/command")
        assert status == 204, f"空队列应返回 204，实际 {status}"
        assert bridge.server.control_poll_count == 1, "取指令次数没记上"

        connected = bridge.do_self_check()
        assert connected is True, "有人取指令 = 通道已接通"
        assert len(ctx.pushed) == 0, "接通了就不该再提醒"
    finally:
        bridge.stop()


def test_self_check_respects_max_reminders() -> None:
    """提醒次数有上限 —— 不能一直烦用户。"""
    bridge, ctx = make_bridge()
    try:
        limit = bridge._control_self_check_max
        assert limit >= 1, "至少该提醒一次"
        for _ in range(limit + 3):
            bridge.do_self_check()
        assert len(ctx.pushed) == limit, f"应该只提醒 {limit} 次，实际 {len(ctx.pushed)} 次"
    finally:
        bridge.stop()


def test_self_check_is_scheduled_at_startup() -> None:
    """start 必须把自检排上 —— 否则提醒永远不会发生。"""
    bridge, ctx = make_bridge()
    try:
        assert bridge._self_check_timer is not None, "start 没安排自检定时器"
        assert bridge._control_self_check_delay_s > 0, "自检延迟不对"
    finally:
        bridge.stop()
        assert bridge._self_check_timer is None, "stop 应该把定时器收掉"


def test_self_check_not_scheduled_when_control_off() -> None:
    """反向控制关着 → 不做自检（没什么好提醒的）。"""
    bridge, ctx = make_bridge(control={"enabled": False})
    try:
        assert bridge._self_check_timer is None, "通道关着不该排自检"
    finally:
        bridge.stop()


# ══════════════════════════════════════════════════════════════════════
# 合并特有的回归：SDK 无关性 + 配置重载
# ══════════════════════════════════════════════════════════════════════
def test_bridge_modules_do_not_import_neko_sdk() -> None:
    """★ 合并的红线：``bettergi/`` 子包**不许** import NEKO 插件 SDK。

    这条约束是整个合并方案成立的前提：桥接层 SDK 无关 ⇒ 离线测试不必
    构造插件类；插件类只做接线。

    一旦有人图省事在 ``bettergi/`` 里 ``from plugin... import``，
    这个子包就再也脱不开宿主，测试会跟着一起变慢、一起变脆。
    """
    banned = ("plugin.sdk", "from plugin", "import plugin", "NekoPluginBase")
    for path in sorted(BETTERGI_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, (
                f"{path.name} 里出现了宿主耦合 {token!r} —— bettergi/ 必须 SDK 无关"
            )


def test_bridge_reload_reloads_config_without_raising() -> None:
    """★ ``reload()`` 必须真的能重载配置（合并时踩到过一个 AttributeError）。

    历史上写成 ``type(self._mapper).load_config(...)``，而 ``load_config``
    是**模块级函数**不是类方法 —— 一调 reload 就炸。
    """
    bridge, ctx = make_bridge()
    try:
        bridge.reload()  # 不抛异常即通过
        assert bridge.mapper is not None
        assert "domain.start" in bridge.mapper.known_events()
    finally:
        bridge.stop()


def test_bridge_start_is_idempotent_enough_to_stop_twice() -> None:
    """``stop()`` 可重复调用（插件 shutdown 可能被调两次）。"""
    bridge, ctx = make_bridge()
    bridge.stop()
    bridge.stop()
    assert bridge.server is None
    assert bridge.control_queue is None or bridge.control_queue.pending_count == 0


@pytest.mark.parametrize(
    "action",
    ["start_auto", "stop_auto", "status"],
)
def test_known_actions_are_accepted(action: str) -> None:
    """已登记的动作必须被接受（未知动作在另一条用例里被拒）。"""
    bridge, ctx = make_bridge(control={"enabled": True, "ack_timeout_s": 1})
    try:
        # 没人回执 → 超时是预期结果；关键是**不是**「未知动作」错误。
        result = asyncio.run(bridge.dispatch_command(action))
        assert "未知动作" not in result["output"], f"{action} 被当成了未知动作: {result}"
    finally:
        bridge.stop()
