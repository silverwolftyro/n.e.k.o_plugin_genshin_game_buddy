# -*- coding: utf-8 -*-
"""genshin_game_buddy · BetterGI 桥接**接线层**集成测试。

★ 2026-11-07 新增（任务 B 的验收面）。

与 ``test_bettergi_bridge.py`` 的分工
--------------------------------------
* ``test_bettergi_bridge.py`` 打的是 :class:`BettergiBridge` —— 桥接**本身**的
  行为（事件映射、限流、回执、自检）。
* 本文件打的是**插件类** —— 桥接有没有被正确接到宿主上：

  1. ``startup`` 真的把桥接起了吗？端口真的在监听吗？
  2. BetterGI 的事件到了 → 有没有变成 YUI 的台词（走的是插件的
     ``push_message``，不是桩件）？
  3. ``@llm_tool command_bettergi`` 注册上了吗？参数 schema 对吗？
  4. ``shutdown`` 真的把线程和端口收回来了吗？
  5. **桥接监听和感知栈会不会互相踩**？

★ 关于端口
  本文件用 ``port = 0``（系统分配临时端口），**不**占用 17890 ——
  否则 48 条会调 ``plugin.startup()`` 的既有用例会互相抢端口。
  「默认端口是 17890」这件事由 ``test_default_webhook_port_is_17890`` 钉住
  （直接查清单与 bettergi/config.toml）。
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from plugin.sdk.plugin.llm_tool import (
    LLM_TOOL_META_ATTR,
    collect_llm_tool_methods,
)

from conftest import PLUGIN_DIR, make_config

pytestmark = pytest.mark.integration


# ── 工具 ────────────────────────────────────────────────────────────────
def post_json(url: str, payload: Any, timeout: float = 5.0) -> Tuple[int, Any]:
    """POST 一个 JSON，返回 ``(状态码, 解析后的响应)``。"""
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


def port_is_listening(host: str, port: int) -> bool:
    """端口上真的有东西在听吗（用一次真实 connect 验证）。"""
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


def bridge_config(**extra: Any) -> Dict[str, Any]:
    """一份让桥接用**临时端口**的宿主配置段。"""
    webhook: Dict[str, Any] = {"host": "127.0.0.1", "port": 0, "path": "/bettergi"}
    webhook.update(extra.pop("webhook", {}) or {})
    section: Dict[str, Any] = {"webhook": webhook}
    section.update(extra)
    return section


# ══════════════════════════════════════════════════════════════════════
# 1) 接线：startup / shutdown
# ══════════════════════════════════════════════════════════════════════
async def test_startup_wires_the_bridge_and_it_listens(build_plugin) -> None:
    """★ startup 之后桥接必须**真的在监听**端口。"""
    plugin = build_plugin(config=make_config(options={"bettergi": bridge_config()}))
    await plugin.startup()
    try:
        bridge = getattr(plugin, "_bettergi", None)
        assert bridge is not None, "startup 没有把 BetterGI 桥接接上"
        server = bridge.server
        assert server is not None, "桥接没有起 HTTP 服务器"
        assert server.is_running, "HTTP 服务器没在跑"
        assert port_is_listening("127.0.0.1", server.port), (
            f"{server.port} 端口上没有监听（桥接只是嘴上说起来了）"
        )
    finally:
        await plugin.shutdown()


async def test_shutdown_releases_the_bridge_and_the_guard(build_plugin) -> None:
    """★ shutdown 之后端口和线程都要收回来（否则重启插件会撞端口）。"""
    plugin = build_plugin(config=make_config(options={"bettergi": bridge_config()}))
    await plugin.startup()
    bridge = plugin._bettergi
    assert bridge is not None and bridge.server is not None
    port = bridge.server.port
    guard = getattr(plugin, "_guard", None)
    assert guard is not None, "生命周期守卫没接上"
    assert guard.running is True, "守卫线程没起来"

    await plugin.shutdown()

    assert plugin._bettergi is None, "shutdown 后桥接引用没清掉"
    assert bridge.server is None, "shutdown 后服务器没停"
    assert not port_is_listening("127.0.0.1", port), f"{port} 端口还在监听（没释放）"
    assert plugin._guard is None, "shutdown 后守卫引用没清掉"
    assert guard.running is False, "守卫线程没停"


async def test_bridge_failure_does_not_break_plugin_startup(build_plugin) -> None:
    """★ 端口被占用时插件**照常起来** —— 桥接降级，不拖垮整个插件。

    这是合并时最现实的故障：用户电脑上已经有别的东西占了 17890。
    """
    # 先占一个端口，让插件的桥接必然绑不上
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    squatter.bind(("127.0.0.1", 0))
    squatter.listen(1)
    busy_port = squatter.getsockname()[1]
    try:
        plugin = build_plugin(
            config=make_config(
                options={
                    "bettergi": {
                        "webhook": {"host": "127.0.0.1", "port": busy_port, "path": "/bettergi"}
                    }
                }
            )
        )
        await plugin.startup()
        try:
            # 插件本身必须活着：能读配置、能拿游戏列表
            listing = await plugin.list_games()
            assert listing is not None, "插件在桥接降级后起不来"
        finally:
            await plugin.shutdown()
    finally:
        squatter.close()


# ══════════════════════════════════════════════════════════════════════
# 2) 正向链路：BetterGI 事件 → 插件的 push_message → YUI
# ══════════════════════════════════════════════════════════════════════
async def test_bettergi_event_reaches_the_plugin_push_message(build_plugin) -> None:
    """★ BetterGI 推事件 → 走的是**插件自己的** ``push_message``。

    这条钉住「桥接不是悬空的」：桥接拿到的 ``push_fn`` 必须是插件实例的
    ``push_message``，这样台词才会真的进宿主。
    """
    plugin = build_plugin(config=make_config(options={"bettergi": bridge_config()}))
    await plugin.startup()
    try:
        bridge = plugin._bettergi
        assert bridge is not None and bridge.server is not None
        before = len(plugin.ctx.pushed)

        status, body = post_json(
            bridge.server.url, {"event": "domain.start", "message": "开始"}
        )
        assert status == 200, f"Webhook 状态码 {status}: {body}"
        assert body.get("handled") is True, f"事件没被处理: {body}"

        assert len(plugin.ctx.pushed) == before + 1, "桥接没往宿主推台词"
        pushed = plugin.ctx.pushed[-1]
        assert pushed.get("source") == "genshin_game_buddy"
        assert pushed.get("visibility") == []
        assert pushed.get("ai_behavior") == "respond"
        text = pushed["parts"][0]["text"]
        assert "你现在就是 YUI 本人" in text, f"台词不是第一人称硬指令: {text!r}"
        assert "【" in text and "】" in text, f"台词没有【】边界: {text!r}"
    finally:
        await plugin.shutdown()


# ══════════════════════════════════════════════════════════════════════
# 3) 反向链路：@llm_tool 注册与路由
# ══════════════════════════════════════════════════════════════════════
async def test_command_bettergi_is_registered_as_llm_tool(build_plugin) -> None:
    """★ ``@llm_tool command_bettergi`` 真的被 SDK 收集到了。

    注册路径：``plugin/sdk/plugin/llm_tool.py`` 的 ``collect_llm_tool_methods``
    → ``NekoPluginBase`` 的 startup → ``LLM_TOOL_REGISTER`` IPC。
    装饰器扫的是**插件类的方法**，跟它在哪个插件里无关 ——
    所以从独立插件并进本插件后，这条链路不用改。
    """
    plugin = build_plugin(config=make_config(options={"bettergi": bridge_config()}))
    await plugin.startup()
    try:
        tools = collect_llm_tool_methods(plugin)
        names = [meta.name for meta, _fn in tools]
        assert "command_bettergi" in names, f"LLM 工具表里没有 command_bettergi: {names}"

        meta = next(m for m, _fn in tools if m.name == "command_bettergi")
        assert meta.description, "工具描述不能为空（LLM 靠它决定要不要调）"
        assert "明确要求" in meta.description, (
            f"工具描述必须写清「只有主人明确要求时才调」: {meta.description!r}"
        )
        props = meta.parameters.get("properties", {})
        assert "action" in props, f"参数 schema 缺 action: {meta.parameters}"
        assert "args" in props, f"参数 schema 缺 args: {meta.parameters}"
        assert meta.parameters.get("required") == ["action"]
        # action 的枚举必须是已登记的动作，不能是空壳
        enum = props["action"].get("enum") or []
        assert "start_auto" in enum, f"action 枚举不对: {enum}"
        assert meta.timeout_seconds > 0
    finally:
        await plugin.shutdown()


async def test_command_bettergi_method_is_tagged(build_plugin) -> None:
    """★ 方法本体上带着 ``@llm_tool`` 的元数据（装饰器没被吃掉）。"""
    plugin = build_plugin()
    meta = getattr(plugin.command_bettergi, LLM_TOOL_META_ATTR, None)
    assert meta is not None, "command_bettergi 上没有 @llm_tool 元数据"
    assert meta.name == "command_bettergi"


async def test_command_bettergi_routes_to_the_bridge(build_plugin) -> None:
    """★ 工具调用确实落到桥接上（不是自己瞎编一个返回值）。

    没人回执 → 超时 → 必须**如实**返回 ``is_error=True``。
    """
    plugin = build_plugin(
        config=make_config(
            options={
                "bettergi": bridge_config(
                    control={"enabled": True, "ack_timeout_s": 1}
                )
            }
        )
    )
    await plugin.startup()
    try:
        result = await plugin.command_bettergi(action="start_auto")
        assert isinstance(result, dict)
        assert result["is_error"] is True, f"超时必须报错: {result}"
        assert "回执" in result["output"], f"超时提示要说清楚: {result}"
        assert result.get("action") == "start_auto"
    finally:
        await plugin.shutdown()


async def test_command_bettergi_without_bridge_reports_error(build_plugin) -> None:
    """★ 桥接没起来时，工具必须**明确报错**，而不是静默成功。

    「假装成功」是这里最坏的失败模式 —— 用户以为指令发出去了。
    """
    plugin = build_plugin()
    plugin._bettergi = None
    result = await plugin.command_bettergi(action="start_auto")
    assert result["is_error"] is True, f"没桥接时必须报错: {result}"
    assert "17890" in result["output"], f"提示里要点明端口: {result}"


# ══════════════════════════════════════════════════════════════════════
# 4) 桥接 vs 感知栈：互不踩
# ══════════════════════════════════════════════════════════════════════
async def test_bridge_listener_survives_perception_suspension(build_plugin) -> None:
    """★ 感知栈挂起时，**Webhook 监听必须照旧活着**。

    这是任务 C 的核心约束：「保留极低功耗的 Webhook 监听，以响应 NEKO 的查询」。
    守卫判「游戏不在」→ 感知栈停摆 → 但 BetterGI 还能把最后一条事件送进来，
    用户才看得到收尾台词。
    """
    plugin = build_plugin(config=make_config(options={"bettergi": bridge_config()}))
    await plugin.startup()
    try:
        bridge = plugin._bettergi
        guard = plugin._guard
        assert bridge is not None and bridge.server is not None
        assert guard is not None

        # 假装「游戏进程没了」：直接把守卫判成挂起
        guard._state = "ABSENT"
        guard._suspended = True
        assert plugin._perception_suspended() is True, "守卫说挂起，插件却没挂起"

        # ★ 挂起状态下 Webhook 仍然收得到事件
        before = len(plugin.ctx.pushed)
        status, body = post_json(
            bridge.server.url, {"event": "domain.end", "message": "结束"}
        )
        assert status == 200, f"挂起后 Webhook 不该挂: {status} {body}"
        assert body.get("handled") is True, f"挂起后事件不该被丢: {body}"
        assert len(plugin.ctx.pushed) == before + 1, "挂起后台词没送出去"
    finally:
        await plugin.shutdown()


async def test_guard_disabled_by_config_leaves_perception_running(build_plugin) -> None:
    """★ ``lifecycle_guard_enabled = false`` → 守卫不建，感知栈永不挂起。

    这是给「用户就是想让它一直盯着」的逃生舱。
    """
    plugin = build_plugin(
        config=make_config(
            options={
                "lifecycle_guard_enabled": False,
                "bettergi": bridge_config(),
            }
        )
    )
    await plugin.startup()
    try:
        assert plugin._guard is None, "守卫被关掉了却还是建了"
        assert plugin._perception_suspended() is False
    finally:
        await plugin.shutdown()


async def test_allow_background_option_is_parsed(build_plugin) -> None:
    """★ 「允许后台运行」这个开关真的被解析成选项了。"""
    plugin = build_plugin(
        config=make_config(options={"lifecycle_guard_allow_background": True})
    )
    await plugin.startup()
    try:
        assert plugin._options.lifecycle_guard_allow_background is True
        assert plugin._guard is not None
        assert plugin._guard.allow_background is True
    finally:
        await plugin.shutdown()


async def test_poll_seconds_option_is_clamped(build_plugin) -> None:
    """★ 轮询周期越界要被夹住（清单写错值不该让插件疯掉）。"""
    plugin = build_plugin(
        config=make_config(options={"lifecycle_guard_poll_seconds": 9999})
    )
    await plugin.startup()
    try:
        assert plugin._options.lifecycle_guard_poll_seconds == 300, (
            f"轮询周期没被夹到上限: {plugin._options.lifecycle_guard_poll_seconds}"
        )
    finally:
        await plugin.shutdown()


# ══════════════════════════════════════════════════════════════════════
# 5) 清单口径：默认端口 17890
# ══════════════════════════════════════════════════════════════════════
def test_default_webhook_port_is_17890() -> None:
    """★ 默认端口必须是 17890（与原独立插件一致，BetterGI 侧配置不用改）。

    两处都得是 17890：
      · 宿主清单 ``plugin.toml`` 的 ``[genshin_game_buddy.bettergi.webhook]``
      · 插件目录里的 ``bettergi/config.toml`` 的 ``[webhook]``
    """
    import tomllib

    with (PLUGIN_DIR / "plugin.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    host_webhook = manifest["genshin_game_buddy"]["bettergi"]["webhook"]
    assert host_webhook["port"] == 17890, f"清单端口不对: {host_webhook}"
    assert host_webhook["host"] == "127.0.0.1", "监听地址必须是回环（别暴露到局域网）"

    with (PLUGIN_DIR / "bettergi" / "config.toml").open("rb") as stream:
        live = tomllib.load(stream)
    assert live["webhook"]["port"] == 17890, f"config.toml 端口不对: {live['webhook']}"
    assert live["webhook"]["path"] == "/bettergi"


def test_manifest_declares_lifecycle_guard_keys() -> None:
    """★ 清单里声明的三个守卫键必须与代码里的字段名**逐字一致**。

    清单与代码脱节 = 用户改了配置没反应，这是「死配置键」的老毛病。
    """
    import tomllib

    with (PLUGIN_DIR / "plugin.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    section = manifest["genshin_game_buddy"]
    for key in (
        "lifecycle_guard_enabled",
        "lifecycle_guard_poll_seconds",
        "lifecycle_guard_allow_background",
    ):
        assert key in section, f"清单里少了 {key}"
        assert key in PluginOptions_field_names(), f"代码里没有 {key} 这个选项字段"


def PluginOptions_field_names() -> set[str]:
    """``PluginOptions`` 的字段名集合。"""
    from plugin.plugins.genshin_game_buddy.game_registry import PluginOptions

    return {f.name for f in PluginOptions.__dataclass_fields__.values()}  # type: ignore[attr-defined]
