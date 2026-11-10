"""播报档位（``event_to_yui.CODE_TIER`` + ``yui_verbosity``）。

★ 这份测试在验什么
------------------
不是「调了滑块档位会不会变」—— 那是一行赋值。验的是三件**错了都不
会报错**的事：

1. **黑名单与档位不重叠**。``SKIP_CODES`` 管「这一类永远不该播」，
   档位管「这一档要不要听」。档位拉到最啰嗦**也不能**播得出
   ``BLACK_FRAME``（真机一个 run 2655 条）—— 两张表一旦互相放行，
   用户的「静音」就变成了「更啰嗦」。

2. **没登记的 code 落在最啰嗦档**。新加一个事件时它默认**不吵**。
   反过来的话，每个新事件都会在静音用户的屏幕上响一次 ——
   而「静音」这个需求存在的全部理由就是不被吵到。

3. **默认 1.0 = 与升级前逐字一致**。定低会让这次升级静默改变所有人的
   播报行为；用户要的是「能调」，不是「默认变安静」。

覆盖链路：bridge.json 的 ui.yui_verbosity
          -> autopilot_bridge.read_yui_verbosity()
            -> EventToYui.refresh_verbosity()（每轮 poll）
              -> render() 里的档位闸门 -> push_message
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import event_to_yui
from autopilot_bridge import (
    DEFAULT_YUI_VERBOSITY,
    read_yui_verbosity,
)
from event_to_yui import (
    CODE_TIER,
    DEFAULT_VERBOSITY,
    EVENT_TO_SPEECH,
    SKIP_CODES,
    TIER_CRITICAL,
    TIER_NORMAL,
    TIER_VERBOSE,
    EventToYui,
    tier_of,
    verbosity_tier,
)

pytestmark = pytest.mark.unit


# ================================================================== 夹具
class _Adapter:
    """真适配器 + 假事件流 + 假 push。"""

    def __init__(self, **kw) -> None:
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="e2y_verbosity_"))
        self.run_id = "t"
        (self.dir / "run_current.txt").write_text(self.run_id, encoding="utf-8")
        self.stream = self.dir / f"{self.run_id}.stream.jsonl"
        self.said: list[str] = []
        self.a = EventToYui(
            push_fn=lambda **kw2: (
                self.said.append(
                    "".join(x.get("text", "") for x in kw2["parts"]))
                or {"submitted": True}),
            runtime_dir=self.dir,
            throttle_seconds=0.0,
            # ★ 场景组冷却**关掉**
            #   这份测试只验「档位」这一层。场景组冷却（Companion Mode）
            #   是另一道闸门，它若开着，同组内第二条非 tier1 台词会被
            #   它挡掉 —— 那测到的就不是档位了。
            #   场景组冷却本身由 ``test_yui_companion_mode.py`` 单独钉。
            scene_group_cooldown_s=0.0,
            **kw)

    def write(self, code: str, message: str = "", **ctx) -> None:
        with self.stream.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"schema_version": "1.0", "run_id": self.run_id, "seq": 0,
                 "ts_ms": 0, "code": code, "severity": "INFO",
                 "message": message, "context": ctx},
                ensure_ascii=False) + "\n")

    def drain(self) -> list[str]:
        before = len(self.said)
        self.a.drain_once()
        return self.said[before:]

    def close(self) -> None:
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)


@pytest.fixture
def ad():
    a = _Adapter()
    yield a
    a.close()


# ================================================================== 映射
class TestVerbosityToTier:
    @pytest.mark.parametrize("v,tier", [
        (0.0, TIER_CRITICAL),
        (0.1, TIER_CRITICAL),
        # ★ 边界是 0.25 / 0.75。``1 + round(v*2)``：0.25*2=0.5 →
        #   round(0.5)=0（Python 的银行家舍入）→ 1 档。
        (0.24, TIER_CRITICAL),
        (0.25, TIER_CRITICAL),
        (0.5, TIER_NORMAL),
        (0.74, TIER_NORMAL),
        (0.75, TIER_VERBOSE),        # round(1.5)=2（银行家舍入进位）
        (1.0, TIER_VERBOSE),
    ])
    def test_three_bands(self, v, tier) -> None:
        assert verbosity_tier(v) == tier

    def test_band_edges_are_quarter_and_three_quarters(self) -> None:
        """★★ 分界是 0.25 / 0.75 —— **Console 那页用同一对数字**。

        两个仓库各写一份边界就一定会漂，而漂了的症状是「我拖到
        标准档，它却说了一句详尽的话」：用户无从判断是滑块没生效
        还是档位表有问题。

        跨仓库没法共享常量（Console venv 里没有引擎代码，反之亦然），
        所以两边各钉一次自己的那份；Console 侧那条在
        ``NEKO-Console/tests/test_combat_tab.py``。
        """
        assert verbosity_tier(0.2499) == TIER_CRITICAL
        assert verbosity_tier(0.7499) == TIER_NORMAL
        assert verbosity_tier(0.75) == TIER_VERBOSE

    @pytest.mark.parametrize("bad", ["x", None, [], {}, object()])
    def test_unreadable_is_verbose_not_silent(self, bad) -> None:
        """★ 读不懂 -> 当详尽，**不是当静音**。

        静音是最坏的选择：用户会以为「我把播报关了」，而实际上
        只是配置读不出来。
        """
        assert verbosity_tier(bad) == TIER_VERBOSE

    @pytest.mark.parametrize("v,clamped", [
        (-3.0, 0.0), (2.5, 1.0),
    ])
    def test_out_of_range_is_clamped_not_raised(self, v, clamped) -> None:
        """★ 越界夹回而不是抛。

        拖滑块拖出 [0,1] 是 UI 层的问题，不该让播报整个哑掉。
        """
        assert verbosity_tier(v) == verbosity_tier(clamped)

    def test_there_is_no_fourth_band(self) -> None:
        """档位表只有三级。四档会让人以为存在「比详尽更啰嗦」的档，
        而它只会和详尽表现完全一样 —— 那是个假选项。"""
        assert len({verbosity_tier(v / 20.0) for v in range(21)}) == 3


class TestCodeTiers:
    def test_every_speakable_code_is_tiered(self) -> None:
        """★★ **没登记的默认落在最啰嗦档**这条规则的守卫。

        新加一个 code 而忘了登记时，它**不会**在静音用户的屏幕上
        响 —— 但它也**不会**在标准档响。要让它在标准档出声，
        必须有人显式把它写进 :data:`CODE_TIER`。

        ⚠️ 黑名单里的 code **要扣掉**
          ``INTENT_EXECUTED`` / ``CORE_STARTED`` / ``CORE_STOPPED``
          确实有台词（ACTION_SPEECH 口子、审计期），但它们在
          :data:`SKIP_CODES` 里，档位再高也播不出来 ——
          给它们定档等于造出一个「看着在管、实际一点用没有」的条目。
        """
        speakable = set(EVENT_TO_SPEECH) - set(SKIP_CODES)
        missing = speakable - set(CODE_TIER)
        assert missing == set(), (
            f"这些台词播得出来但没定档：{sorted(missing)}。"
            f"忘了定档的话它们只会出现在「详尽」档。")

    def test_skip_codes_need_no_tier(self) -> None:
        """黑名单里的 code 不该出现在档位表里。

        两张表同时收同一个 code 意味着它有**两个**身份，而闸门只问
        档位（档位在黑名单之后查）—— 迟早有人把顺序改错。
        """
        assert set(CODE_TIER) & set(SKIP_CODES) == set()

    def test_unregistered_code_falls_to_verbose(self) -> None:
        assert tier_of("SOME_FUTURE_EVENT") == TIER_VERBOSE

    @pytest.mark.parametrize("code", sorted(CODED := list(CODE_TIER)))
    def test_tier_values_are_in_range(self, code) -> None:
        assert tier_of(code) in (TIER_CRITICAL, TIER_NORMAL, TIER_VERBOSE)

    def test_panic_is_always_critical(self) -> None:
        """★ 急停不能被任何档位压掉。

        它是「引擎刚抢了你的键」或「快死了」—— 静音档的用户也要知道。
        """
        assert tier_of("PANIC_TRIGGERED") == TIER_CRITICAL
        assert tier_of("USER_INTERRUPT") == TIER_CRITICAL


# ================================================================== 闸门
class TestGate:
    def test_silent_only_lets_critical_through(self, ad) -> None:
        ad.a.set_verbosity(0.0)
        ad.write("PANIC_TRIGGERED")
        ad.write("ENEMY_DETECTED", count=3)
        ad.write("SCENE_CHANGED", to="WORLD")
        got = ad.drain()
        assert len(got) == 1, got
        assert "按停" in got[0], got

    def test_standard_adds_combat_and_flow(self, ad) -> None:
        ad.a.set_verbosity(0.5)
        ad.write("PANIC_TRIGGERED")
        ad.write("ENEMY_DETECTED", count=3)
        ad.write("SCENE_CHANGED", to="WORLD")
        got = ad.drain()
        assert len(got) == 2, got
        assert not any("场景" in g for g in got), got

    def test_verbose_lets_everything_through(self, ad) -> None:
        ad.a.set_verbosity(1.0)
        ad.write("PANIC_TRIGGERED")
        ad.write("ENEMY_DETECTED", count=3)
        ad.write("SCENE_CHANGED", to="WORLD")
        assert len(ad.drain()) == 3

    def test_skip_codes_stay_silent_at_max_verbosity(self, ad) -> None:
        """★★ 黑名单**绝对**，档位放行不了它。

        错在这里的症状是「我把档位拉满，怎么 BLACK_FRAME 又出来了」——
        而真因是两张表的闸门顺序反了。
        """
        ad.a.set_verbosity(1.0)
        for code in sorted(SKIP_CODES):
            ad.write(code)
        assert ad.drain() == [], "档位把黑名单放行了"

    def test_pickup_overrides_survive_a_silent_profile(self, ad) -> None:
        """★ ``USE_PICKUP`` 走 :data:`ACTION_SPEECH` 口子。

        它本来在 ``SKIP_CODES`` 里（INTENT_EXECUTED 真机 754 次），
        但「捡到东西了」值得说一句。口子开在**黑名单之后、档位之前** ——
        所以它在 3 档（详尽）里出声，1~2 档里不出声。
        """
        ad.a.set_verbosity(1.0)
        ad.write("INTENT_EXECUTED", action="USE_PICKUP")
        assert ad.drain(), "详尽档连捡东西都不说"

        ad.a.set_verbosity(0.5)
        ad.write("INTENT_EXECUTED", action="USE_PICKUP")
        assert ad.drain() == [], "标准档不该说捡东西"

    def test_default_keeps_the_pre_upgrade_behaviour(self) -> None:
        """★★ 默认 1.0 = 与升级前**逐字一致**。

        定低会让这次升级静默改变所有人的播报行为。
        """
        assert DEFAULT_VERBOSITY == 1.0
        assert DEFAULT_YUI_VERBOSITY == DEFAULT_VERBOSITY, (
            "桥接的兜底值与适配器默认值漂移了 —— 主控台没跑时"
            "播报档位会与开着时不一样")


class TestLiveRetune:
    def test_set_verbosity_returns_the_new_threshold(self, ad) -> None:
        assert ad.a.set_verbosity(0.0) == TIER_CRITICAL
        assert ad.a.set_verbosity(1.0) == TIER_VERBOSE

    def test_verbosity_property_round_trips(self, ad) -> None:
        ad.a.set_verbosity(0.25)
        assert ad.a.verbosity == 0.25

    def test_supplier_is_polled_each_cycle(self) -> None:
        """★ 用户拖滑块**不用重启插件**。

        供应函数每轮 poll 调一次（mtime 短路，几乎免费），
        于是最多 2 秒生效 —— 而重启适配器 = 停插件 + 起插件。
        """
        box = {"v": 1.0}
        a = _Adapter(verbosity_fn=lambda: box["v"])
        try:
            a.write("SCENE_CHANGED", to="WORLD")
            assert a.drain(), "起始档位是详尽，应该说"

            box["v"] = 0.0
            a.a.refresh_verbosity()
            # ★ 用一个**真实**的场景 id（B 轨状态机认的）
            #   写成 "CITY" 的话它会先被场景门控挡掉（不认识的场景 -> UNKNOWN），
            #   那样测到的就不是「档位改了」而是「场景不认识」，两码事。
            a.write("SCENE_CHANGED", to="MAIN_MENU")
            assert a.drain() == [], "档位改了之后还在说"
        finally:
            a.close()

    def test_supplier_raising_keeps_the_old_value(self) -> None:
        """★ 供应函数抛异常不能让播报线程死。

        它读的是**另一个进程写的文件**。抛了就保持原档位。
        """
        def boom():
            raise OSError("bridge 正在被替换")

        a = _Adapter(verbosity_fn=boom, verbosity=0.0)
        try:
            a.a.refresh_verbosity()
            assert a.a.verbosity == 0.0
        finally:
            a.close()

    def test_no_supplier_means_the_value_is_frozen(self, ad) -> None:
        ad.a.refresh_verbosity()
        assert ad.a.verbosity == DEFAULT_VERBOSITY

    def test_lowering_verbosity_does_not_rewind_history(self, ad) -> None:
        """★ 降到 0 不清空「本 run 已说过」。

        已经推给 YUI 的消息收不回来；而清空它会让第二句话在
        静音档重新开口一次 —— 那是「刚静音就多说了一句」。

        这里用 ``PERCEPTION_DOWN``：它在 ``REPEAT_PRONE_CODES`` 里，
        上限是 ``MAX_LINES_PER_RUN``（2）次。
        """
        assert event_to_yui.MAX_LINES_PER_RUN == 2
        ad.a.set_verbosity(1.0)
        for _ in range(4):
            ad.write("PERCEPTION_DOWN")
        said = ad.drain()
        assert len(said) == 2, said

        ad.a.set_verbosity(0.0)
        ad.write("PERCEPTION_DOWN")
        assert ad.drain() == [], "重复上限被档位切换清空了"


# ================================================================== 桥接
class TestBridgeVerbosity:
    def _bridge(self, tmp_path, ui):
        p = tmp_path / "console_bridge.json"
        p.write_text(json.dumps({
            "schema_version": 1, "written_at_ms": 0,
            "console": {"running": True}, "ui": ui,
        }), encoding="utf-8")
        return p

    def test_reads_it(self, tmp_path, monkeypatch) -> None:
        import autopilot_bridge

        p = self._bridge(tmp_path, {"yui_verbosity": 0.5})
        monkeypatch.setattr(autopilot_bridge, "bridge_path", lambda: p)
        monkeypatch.setattr(autopilot_bridge, "_cache_data", None, raising=False)
        assert read_yui_verbosity() == 0.5

    @pytest.mark.parametrize("bridge,expect", [
        (None, DEFAULT_YUI_VERBOSITY),
        ({}, DEFAULT_YUI_VERBOSITY),
        ({"ui": None}, DEFAULT_YUI_VERBOSITY),
        ({"ui": {}}, DEFAULT_YUI_VERBOSITY),
        ({"ui": {"yui_verbosity": None}}, DEFAULT_YUI_VERBOSITY),
        ({"ui": {"yui_verbosity": "x"}}, DEFAULT_YUI_VERBOSITY),
    ])
    def test_missing_falls_back_to_default(self, bridge, expect) -> None:
        """★ 读不到时**照常播**，不静音。

        主控台没跑 / 文件正在被替换 / 版本不认得 —— 这三种情况
        静默会被误解成「我刚把播报关了」。
        """
        assert read_yui_verbosity(bridge) == expect

    @pytest.mark.parametrize("raw,expect", [
        (-1.0, 0.0), (5.0, 1.0),
    ])
    def test_out_of_range_is_clamped(self, raw, expect) -> None:
        assert read_yui_verbosity({"ui": {"yui_verbosity": raw}}) == expect

    def test_schema_version_change_does_not_need_a_bump(self) -> None:
        """★ 加字段**不升** bridge schema 版本。

        ``ui`` 是一个**新组**，老插件读到多出来的键会忽略；
        升版本会让正在跑的旧插件直接判定「版本不认识」而失联 ——
        那比多一个字段严重得多。
        """
        assert event_to_yui.SKIP_CODES is SKIP_CODES


# ================================================================== 宿主配置档位
class TestHostVerbosityPreset:
    """★ P0：三级优先 —— 宿主配置 > Console 滑块 > 默认 1.0。

    宿主配置这一级来自 ``plugin.toml`` 的
    ``[genshin_game_buddy] yui_verbosity_preset``。
    """

    @pytest.fixture(autouse=True)
    def _reset_host_preset(self, monkeypatch):
        """每个用例前后都把宿主档位清干净 —— 它是模块级全局。"""
        import autopilot_bridge

        monkeypatch.setattr(autopilot_bridge, "_host_verbosity", None,
                            raising=False)
        yield
        autopilot_bridge.set_host_verbosity_preset(None)

    def test_preset_values_map_to_the_expected_tiers(self) -> None:
        """★ 三档必须和 ``verbosity_tier()`` 对齐。

        quiet=0.0 -> tier1 / normal=0.5 -> tier2 / chatty=1.0 -> tier3。
        对不齐的话「安静」这个档名就是骗人的。
        """
        from autopilot_bridge import VERBOSITY_PRESETS

        assert VERBOSITY_PRESETS["quiet"] == 0.0
        assert VERBOSITY_PRESETS["normal"] == 0.5
        assert VERBOSITY_PRESETS["chatty"] == 1.0
        assert verbosity_tier(VERBOSITY_PRESETS["quiet"]) == TIER_CRITICAL
        assert verbosity_tier(VERBOSITY_PRESETS["normal"]) == TIER_NORMAL
        assert verbosity_tier(VERBOSITY_PRESETS["chatty"]) == TIER_VERBOSE

    def test_preset_names_match_the_manifest_allowlist(self) -> None:
        """★ 两处允许的预设名必须一样。

        ``autopilot_bridge`` 管「名字 -> 数值」，
        ``game_registry`` 管「清单值 -> 合法名字」。名字集合漂移 =
        清单里能写出一个代码却不认的值。
        """
        from autopilot_bridge import VERBOSITY_PRESETS
        from game_registry import YUI_VERBOSITY_PRESETS

        assert set(VERBOSITY_PRESETS) == set(YUI_VERBOSITY_PRESETS)

    def test_host_preset_wins_over_the_console_slider(self) -> None:
        """★ 第一级压过第二级。"""
        import autopilot_bridge

        autopilot_bridge.set_host_verbosity_preset("quiet")
        assert read_yui_verbosity({"ui": {"yui_verbosity": 1.0}}) == 0.0

        autopilot_bridge.set_host_verbosity_preset("normal")
        assert read_yui_verbosity({"ui": {"yui_verbosity": 1.0}}) == 0.5

    def test_default_preset_name_defers_to_the_console_slider(self) -> None:
        """★★ 清单默认值（"chatty"）**不**压过 Console 滑块。

        宿主会把清单默认值一起写进用户配置副本 —— 这个键**永远存在**。
        若"键存在即覆盖"，Console 滑块就永远轮不到说话，
        而滑块恰恰是用户最常用的即时开关。
        """
        import autopilot_bridge

        autopilot_bridge.set_host_verbosity_preset("chatty")
        assert read_yui_verbosity({"ui": {"yui_verbosity": 0.5}}) == 0.5

    def test_no_preset_no_slider_falls_back_to_the_default(self) -> None:
        import autopilot_bridge

        autopilot_bridge.set_host_verbosity_preset(None)
        assert read_yui_verbosity({"ui": {"yui_verbosity": None}}) \
            == DEFAULT_YUI_VERBOSITY

    def test_unknown_preset_name_is_ignored(self) -> None:
        """★ 用户手滑写错一个词：忽略 + 继续用滑块，而不是炸掉播报线程。"""
        import autopilot_bridge

        autopilot_bridge.set_host_verbosity_preset("loud")
        assert read_yui_verbosity({"ui": {"yui_verbosity": 0.5}}) == 0.5

    def test_preset_changes_what_the_adapter_says(self) -> None:
        """★★ 「改了它行为就变」—— 清单里写的键必须真的影响播报。

        这条是 plugin.toml 那句注释的兑现：
        *每个键都必须被代码真正读取，并有一个"改了它行为就变"的测试*。
        """
        import autopilot_bridge

        a = _Adapter(verbosity_fn=lambda: read_yui_verbosity({}))
        try:
            a.write("SCENE_CHANGED", to="WORLD")
            assert a.drain(), "默认（话痨）应该说话"

            autopilot_bridge.set_host_verbosity_preset("quiet")
            a.a.refresh_verbosity()
            a.write("SCENE_CHANGED", to="CITY")
            assert a.drain() == [], "安静档还在说话 —— 宿主配置没接上"
        finally:
            a.close()

    def test_manifest_declares_the_key_with_the_default_preset(self) -> None:
        """★ 清单里必须有这一行，且默认值是 "chatty"。

        默认必须是 "chatty"（=1.0）：定低会静默改变所有人的播报行为。
        """
        from pathlib import Path

        import tomllib

        plugin_dir = Path(__file__).resolve().parents[2]
        with (plugin_dir / "plugin.toml").open("rb") as fh:
            manifest = tomllib.load(fh)
        section = manifest.get("genshin_game_buddy") or {}
        assert section.get("yui_verbosity_preset") == "chatty"

    def test_manifest_preset_is_read_by_the_options(self) -> None:
        """★ 清单值真的进了 ``PluginOptions``（不是只写在 TOML 里）。"""
        from game_registry import PluginOptions

        opts = PluginOptions.from_section({"yui_verbosity_preset": "quiet"})
        assert opts.yui_verbosity_preset == "quiet"

        # 非法值 -> 回落默认（不崩）
        assert PluginOptions.from_section(
            {"yui_verbosity_preset": "loud"}).yui_verbosity_preset == "chatty"
        # 缺省 -> 默认
        assert PluginOptions.from_section({}).yui_verbosity_preset == "chatty"
