"""2026-11-10：场景级 avoid 与场景正文**同批**注入。

背景
----
``terms/genshin/scenes.toml`` 里 18 个场景每个都带 ``avoid``（防剧透清单），
``render_scene_avoid`` 上一轮也实现了 —— 但它此前**只在**
``_push_context``（语境块）里被渲染，而默认配置
``reinject_every_n_messages = 0`` → 语境块永不重推 →

    **avoid 写了没人读，防剧透约束形同虚设。**

端到端验证实证：场景命中后的**全部** push 里，"本场景避免"标签与三条
avoid 一个字都没出现；模型只拿到场景正文自带的【避免】（另一套、只覆盖
buff 那条）。于是"不要剧透当期深渊敌人的配招与后续版本改动"、
"不要把唯一配队名单当成标准答案"这两条防剧透条款**从未进入模型上下文**。

这是接线缺失（数据写了没人读），不是设计变更 —— 修它让已有的约束真正生效。

这一组钉住：

1. 场景命中推 scene prompt 时，avoid 与正文在**同一次 push** 里；
2. 场景没有 avoid 时不产生悬空标签；
3. 用户主动提问路径不受影响。
"""
from __future__ import annotations

import dataclasses
from typing import Any

import plugin.plugins.genshin_game_buddy as mgc_module
import pytest

PROBE_SCENE = "深境螺旋"
AVOID_LABEL = "本场景避免："
PROBE_AVOID_ITEMS = (
    "不要剧透当期深渊敌人的配招与后续版本改动",
    "不要照抄游戏内可读的 buff 数值",
    "不要把唯一配队名单当成标准答案",
)
OCR_TEXT = "深境螺旋 元素反应 蒸发"


class _FakeCapture:
    """抓屏桩——只走字段，不碰真实屏幕。"""

    ok = True
    image_base64 = "ZmFrZQ=="
    method = "test"
    avg_luma = 128.0


def _install_capture_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_capture(*_args: Any, **_kwargs: Any):
        return _FakeCapture(), "test"

    monkeypatch.setattr(mgc_module, "capture_active_frame", _fake_capture)


def _install_ocr_stub(plugin: Any, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    async def _fake_extract(_b64: str) -> str:
        return text

    monkeypatch.setattr(plugin._ocr, "extract_text_from_base64", _fake_extract)


async def _genshin(build_plugin: Any) -> Any:
    plugin = build_plugin()
    await plugin.set_game(game="原神")
    return plugin


def _scene_push_texts(plugin: Any) -> list[str]:
    """已 push 的、带「当前场景」标记的文本块。"""
    out: list[str] = []
    for item in plugin.ctx.pushed:
        text = "\n".join(str(part.get("text", "")) for part in item.get("parts") or [])
        if "【当前场景：" in text:
            out.append(text)
    return out


# =============================================================================
# 1. avoid 与场景正文同批进上下文
# =============================================================================


@pytest.mark.integration
async def test_scene_push__carries_scene_avoid(build_plugin: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """★ 接线断言：场景命中推出去的正文里**真的有**该场景的 avoid 约束。"""
    plugin = await _genshin(build_plugin)
    assert PROBE_SCENE in plugin._scene_store

    _install_capture_stub(monkeypatch)
    _install_ocr_stub(plugin, monkeypatch, OCR_TEXT)

    await plugin._ocr_perceive_body()

    scene_texts = _scene_push_texts(plugin)
    assert scene_texts, "场景命中却没有推 scene prompt"
    blob = "\n".join(scene_texts)
    assert AVOID_LABEL in blob
    for item in PROBE_AVOID_ITEMS:
        assert item in blob, f"avoid 没进上下文：{item}"
    # 场景正文本身没被挤掉
    assert "【当前场景：深境螺旋】" in blob
    assert "玩家正在挑战原神的深境螺旋" in blob


@pytest.mark.integration
async def test_scene_push__avoid_is_in_the_same_push_as_the_prompt(
    build_plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """avoid 与场景正文在**同一次 push** 里（不是另推一条）。"""
    plugin = await _genshin(build_plugin)
    _install_capture_stub(monkeypatch)
    _install_ocr_stub(plugin, monkeypatch, OCR_TEXT)

    await plugin._ocr_perceive_body()

    scene_pushes = [
        item
        for item in plugin.ctx.pushed
        if any("【当前场景：" in str(part.get("text", "")) for part in item.get("parts") or [])
    ]
    assert len(scene_pushes) == 1
    text = "\n".join(str(part.get("text", "")) for part in scene_pushes[0]["parts"] or [])
    assert AVOID_LABEL in text
    assert "玩家正在挑战原神的深境螺旋" in text


@pytest.mark.integration
async def test_scene_push__no_avoid__no_dangling_label(build_plugin: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """场景没有 avoid 时不产生悬空的"本场景避免："标签。"""
    plugin = await _genshin(build_plugin)
    scene = plugin._scene_store[PROBE_SCENE]
    plugin._scene_store = {
        PROBE_SCENE: dataclasses.replace(scene, avoid=()),
    }

    _install_capture_stub(monkeypatch)
    _install_ocr_stub(plugin, monkeypatch, OCR_TEXT)

    await plugin._ocr_perceive_body()

    scene_texts = _scene_push_texts(plugin)
    assert scene_texts
    assert AVOID_LABEL not in "\n".join(scene_texts)
    # 正文照常推
    assert "玩家正在挑战原神的深境螺旋" in "\n".join(scene_texts)


# =============================================================================
# 2. 用户主动提问路径不受影响
# =============================================================================


@pytest.mark.integration
async def test_user_query__unaffected_by_scene_avoid_push(
    build_plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """场景推送之后，用户主动提问照常应答（术语卡照推）。"""
    plugin = await _genshin(build_plugin)
    _install_capture_stub(monkeypatch)
    _install_ocr_stub(plugin, monkeypatch, OCR_TEXT)

    await plugin._ocr_perceive_body()
    before = len(plugin.ctx.pushed)

    await plugin.on_chat_message(text="胡桃怎么养？", sender="user")

    new_texts = [
        "\n".join(str(part.get("text", "")) for part in item.get("parts") or [])
        for item in plugin.ctx.pushed[before:]
    ]
    assert any("用户刚提到：胡桃" in text for text in new_texts), new_texts


@pytest.mark.integration
async def test_user_query__lookup_still_works_after_scene_push(
    build_plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """场景推送之后，查询工具照常返回术语（known_suppress 不拦用户提问）。"""
    plugin = await _genshin(build_plugin)
    _install_capture_stub(monkeypatch)
    _install_ocr_stub(plugin, monkeypatch, OCR_TEXT)

    await plugin._ocr_perceive_body()

    result = await plugin.lookup_game_term(query="深境螺旋怎么打？")
    assert result["output"]["found"] is True
    assert result["output"]["entries"][0]["key"] == "深境螺旋"
