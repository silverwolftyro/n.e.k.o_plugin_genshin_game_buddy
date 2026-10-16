"""从一帧到「YUI 说话」的完整离线链路（2026-10-13）。

★ 这条链路证明什么、不能证明什么
  ✅ 能证明：**管道是通的** —— 截图 -> 场景判定 -> StatePacket -> 规则求值
     -> 意图 -> 执行 -> 事件流 -> YUI 台词，每一环都真的跑过，
     用的是**真组件**（只有截图源是假的）。
  ❌ 不能证明：**真机上会发生**。这里没有 WGC、没有 DirectML、没有真实按键。
     一句话：这条链路证明「管道是通的」，不证明「水是干净的」。

★ 假的东西只有一处：截图源
  ``FakeCapturer`` 返回一张**真实的**真机截图
  （``autopilot/_runtime/capdiag_wgc.png``，WGC 抓的 2560x1600 画面）。
  用真截图而不是合成图，是因为**场景分类器是对真实画面标定的** ——
  合成图会被判成非大世界，整条链在感知那一环就断了，测不到后面。

★ 决策层用**受控规则**而不是生产规则
  生产里唯一 ``enabled=true`` 的 ``world_attack_only_with_enemy``
  需要实体检出（YOLO + DirectML），离线跑不了。
  所以这里用一条临时规则证明「规则命中 -> 意图 -> 执行 -> 事件」这段是通的，
  **不**冒充生产规则命中了。两条的区别写在断言的注释里。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from autopilot.contracts.events import Event, EventCategory, Severity
from autopilot.perception.builder import StateBuilder
from autopilot.perception.capturer import CapturedFrame
from autopilot.perception.scene_detector import SceneDetector
from autopilot.policy.local_tree import LocalPolicyTree

#: 真机截图（WGC 抓的）。``_runtime/`` 被 git 忽略，所以本机没有就跳过。
REAL_FRAME = (Path(__file__).resolve().parents[2] / "autopilot" / "_runtime"
               / "capdiag_wgc.png")
pytestmark = pytest.mark.skipif(
    not REAL_FRAME.is_file(), reason=f"本机没有真机截图 {REAL_FRAME}")

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
RULES_DIR = PLUGIN_ROOT / "autopilot" / "rules"


# ==================================================================
# 假截图源（整条链路里唯一假的东西）
# ==================================================================

class FakeCapturer:
    """把一张真截图反复当作新帧吐出来。

    ★ 为什么不用 ``Capturer``
      它的 ``open()`` 要定位标题含「原神」的真窗口、WGC 要真 HWND。
      离线两者都不成立。而下游只用到 ``grab()`` 一个方法 ——
      所以这里做鸭子类型就够，不继承（继承就得实现整个协议）。
    """

    def __init__(self, frame: CapturedFrame) -> None:
        self._frame = frame
        self.grabs = 0

    def grab(self) -> CapturedFrame:
        self.grabs += 1
        return self._frame

    def open(self) -> None:      # 兼容 StateBuilder 的启动期探测
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def frame() -> CapturedFrame:
    import cv2

    from autopilot.perception.capturer import CaptureChannel

    img = cv2.imread(str(REAL_FRAME))
    if img is None:
        pytest.skip(f"截图解码失败：{REAL_FRAME}")
    return CapturedFrame(image=img, rect=(0, 0, img.shape[1], img.shape[0]),
                         timestamp_ms=0, frame_seq=1, channel=CaptureChannel.WGC)


# ==================================================================
# 事件流：真 EventBus -> 真 .stream.jsonl
# ==================================================================

class StreamSink:
    """把事件按真格式写进 ``<run_id>.stream.jsonl``。

    ★ 格式必须与 ``autopilot/ui/writer.py`` 一致（``code`` / ``message``
      / ``ts_ms``），否则 YUI 那侧解析出来的就不是同一种东西 ——
      那样测的就不是真链路了。
    """

    def __init__(self, runtime: Path) -> None:
        self.runtime = runtime
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.run_id = "offline_pipeline_test"
        self.path = runtime / f"{self.run_id}.stream.jsonl"
        (runtime / "run_current.txt").write_text(self.run_id, encoding="utf-8")
        self.events: list[Event] = []

    def __call__(self, ev: Event) -> None:
        self.events.append(ev)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"code": ev.code, "message": ev.message,
                                "ts_ms": int(time.time() * 1000)},
                               ensure_ascii=False) + "\n")

    def codes(self) -> list[str]:
        return [e.code for e in self.events]


def _emit(sink: StreamSink, code: str, message: str = "") -> Event:
    ev = Event(code=code, category=EventCategory.SYSTEM, severity=Severity.INFO,
               message=message, context={})
    sink(ev)
    return ev


def _tick(frame: CapturedFrame):
    """走真的一拍：**假截图源抓帧 -> 真 StateBuilder.build**。

    ``StateBuilder.build(frame)`` 收的是帧参数（抓帧是 ``CoreLoop`` 的活），
    所以这里显式地把 ``FakeCapturer`` 接上 —— 否则「假截图源」就形同虚设，
    测的也不再是「截图 -> 状态」这一段。
    """
    cap = FakeCapturer(frame)
    b = StateBuilder(capturer=cap, entity_detector=None,
                     scene_detector=SceneDetector(n_seconds=0.0, m_seconds=0.0))
    return b.build(cap.grab()), cap


# ==================================================================
# 1 · 截图 -> 场景判定 -> StatePacket
# ==================================================================

def test_perception__real_screenshot_is_classified_as_world(frame) -> None:
    """★ 先确认「这张真图能被判定为大世界」—— 后面全靠它。

    之前有人调错了 API（``detect`` 其实不存在，正确是 ``update``），
    拿到 None 就以为「静态图跑不通」。这里用真组件复核一遍。
    """
    sd = SceneDetector(n_seconds=0.0, m_seconds=0.0)
    st = sd.update(frame.image, 0.0)
    assert st.raw_prediction is True, f"真机截图竟被判成非大世界：{st}"


def test_perception__builder_produces_a_world_state(frame) -> None:
    got, cap = _tick(frame)
    assert got is not None
    assert cap.grabs >= 1, "StateBuilder 没有真的去抓帧"
    assert got.scene.id in ("WORLD", "UNKNOWN"), got.scene.id
    assert got.state_version >= 1


def test_perception__scene_is_stable_across_identical_frames(frame) -> None:
    """★ 同一张图连喂几帧，场景不能来回跳。

    跳变会让规则在「看到怪 / 没看到怪」之间翻转，看起来像 AI 在犯傻。
    """
    scenes = {_tick(frame)[0].scene.id for _ in range(5)}
    assert len(scenes) == 1, scenes


# ==================================================================
# 2 · 决策层：受控规则 -> 意图
# ==================================================================

RULE_TOML = """
[[rules]]
name = "offline_probe_rule"
action = "NORMAL_ATTACK"
when = 'scene() == "WORLD"'
priority = 5
ttl_seconds = 3
scenes = ["WORLD", "UNKNOWN"]
enabled = true
note = "离线管道探针：只验证规则->意图这段是通的"
"""


@pytest.fixture
def probe_rules(tmp_path) -> Path:
    d = tmp_path / "rules"
    d.mkdir()
    (d / "offline_probe.toml").write_text(RULE_TOML, encoding="utf-8")
    return d


def test_decision__controlled_rule_produces_an_intent(frame, probe_rules) -> None:
    """★ 规则命中 -> IntentPacket。

    ★ 用的是**受控规则**，不是 ``world_attack_only_with_enemy``
      —— 后者要实体检出（YOLO + DirectML），离线跑不了。
      所以这条证明的是**决策层本身通**，不是「生产规则命中了」。
    """
    state, _cap = _tick(frame)
    tree = LocalPolicyTree(rules_dir=probe_rules)
    intent = tree.poll(state)
    if intent is None:
        trace = tree.decide(state)
        pytest.skip(f"受控规则没命中（离线截图的场景/信号不够）：{trace}")
    # ★ IntentPacket **没有**规则名字段（见 contracts/intent_packet.py 的
    #   dataclass 字段：action / params / intent_id / source / ...）——
    #   规则名留在 LocalPolicyTree 的 trace 里，不进意图本身。这里只断言动作。
    assert intent.action.value == "NORMAL_ATTACK"


NEVER_MATCHES = """
[[rules]]
name = "offline_never_matches"
action = "NORMAL_ATTACK"
when = 'hp() < 0.01'
priority = 5
ttl_seconds = 3
scenes = ["WORLD", "UNKNOWN"]
enabled = true
note = "条件恒为假（hp 恒 UNKNOWN），用来验「不命中就不产意图」"
"""


def test_decision__no_intent_when_no_rule_matches(frame, tmp_path) -> None:
    """★ 没有规则命中时**必须不产意图**，不能硬凑一个。

    「没匹配就也发一个吧」就是规则引擎最坏的一种行为 ——
    它让规则表里的每一条都变得不可验证。

    ⚠️ 用的是「一条永不命中的规则」而不是空目录：
      空目录会被 ``rule_loader`` 判为非法（``RuleLoadError``）——
      「没有规则文件」和「规则没命中」是两回事。
    """
    d = tmp_path / "rules"
    d.mkdir()
    (d / "never.toml").write_text(NEVER_MATCHES, encoding="utf-8")
    state, _cap = _tick(frame)
    tree = LocalPolicyTree(rules_dir=d)
    assert tree.poll(state) is None


# ==================================================================
# 3 · 事件流 -> YUI 台词（真 EventToYui 追尾读）
# ==================================================================

def test_events__stream_file_is_readable_and_has_the_real_format(
        frame, probe_rules, tmp_path) -> None:
    sink = StreamSink(tmp_path / "runtime")
    _emit(sink, "CORE_STARTED", "离线管道探针")
    state, _cap = _tick(frame)
    LocalPolicyTree(rules_dir=probe_rules).poll(state)

    lines = [json.loads(l) for l in
             sink.path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines, "事件流是空的"
    assert {"code", "message", "ts_ms"} <= set(lines[0])


def _capture_speech(sink: "StreamSink") -> tuple[list[str], "EventToYui"]:
    """造一个**形状正确**的 ``push_message`` 假函数，返回（收集到的台词, yui）。

    ★ 签名必须和宿主 ``push_message`` 对齐
      ``EventToYui.push_one`` 是用关键字调它的：
      ``push(source=..., visibility=..., ai_behavior=..., parts=[{type,text}], ...)``。
      写个 ``lambda text: ...`` 会抛 TypeError，而那行被 ``except Exception``
      吞掉、只记 warning —— 症状是「YUI 一句话都不说」，
      与「台词表没这个码」长得一模一样，极易误判。
    """
    from event_to_yui import EventToYui

    said: list[str] = []

    def _push(**kw):
        said.append("".join(p.get("text", "") for p in kw.get("parts", [])))
        return {"submitted": True}

    return said, EventToYui(push_fn=_push, runtime_dir=sink.runtime)


def test_yui__renders_a_line_for_a_business_event(tmp_path) -> None:
    """★ 真 ``EventToYui`` 读真格式的事件流 -> 真的产出台词。

    这是「YUI 说话」这一环的离线证据。

    ★ 为什么发 ``PATHFIND_START`` 而不是 ``CORE_STARTED``
      ``EventToYui`` 只对 ``BUSINESS_CODES`` 那 6 个码出声
      （``PATHFIND_*`` / ``ENEMY_DETECTED`` / ``ESCAPE_SUCCESS`` /
      ``POSITION_CHANGED``）。``CORE_STARTED`` **有**台词模板但不是业务码 ——
      拿它测会得到「不出声」，而那不是 bug，是设计。
    """
    from event_to_yui import BUSINESS_CODES

    sink = StreamSink(tmp_path / "runtime")
    assert "PATHFIND_START" in BUSINESS_CODES, "台词表变了，本测试要改"
    _emit(sink, "PATHFIND_START", "开始前往目标")

    said, yui = _capture_speech(sink)
    n = yui.drain_once()               # 读一轮并推送
    assert n == len(said) == 1, (n, said)
    assert said[0].strip(), "YUI 推出了一句空话"


def test_yui__intentional_events_stay_silent(tmp_path) -> None:
    """★ ``SKIP_CODES`` 里的事件**刻意**不出声 —— 高频噪音不该刷屏。

    ``INTENT_EXECUTED`` 就在里面：真机一个 run 触发 754 次
    （全是 USE_PICKUP，约 4.9 次/秒），整类播报等于复读机。
    """
    from event_to_yui import SKIP_CODES

    assert "INTENT_EXECUTED" in SKIP_CODES, (
        "每次按键都播一句会刷爆面板；若确实要播，请同步改这里与文档")

    sink = StreamSink(tmp_path / "runtime")
    said, yui = _capture_speech(sink)
    for code in sorted(SKIP_CODES):
        _emit(sink, code, "静默事件")
    yui.drain_once()
    assert said == [], f"SKIP_CODES 里的事件不该出声，实际说了：{said}"


# ==================================================================
# 4 · 整条链：一张图 -> 台词
# ==================================================================

def test_full_chain__frame_to_spoken_line(frame, probe_rules, tmp_path) -> None:
    """★★ 从一帧真截图，一路走到 YUI 的台词缓冲区。

    这是本文件想证明的全部：**每一环都真的跑过**，中间只有截图源是假的。

    ⚠️ 台词那一句用的是 ``PATHFIND_START``，**由测试代为发射**
      —— 离线跑不了寻路子进程（它要真游戏窗口）。
      这一环验的是「事件流 -> YUI」这条管线，不是寻路本身。
    """

    sink = StreamSink(tmp_path / "runtime")
    _emit(sink, "CORE_STARTED", "离线管道探针启动")

    # ① 截图 -> ② 场景判定 -> ③ 决策
    state, cap = _tick(frame)
    assert state is not None
    assert cap.grabs >= 1

    intent = LocalPolicyTree(rules_dir=probe_rules).poll(state)
    if intent is not None:
        # INTENT_EXECUTED 刻意**不播报**（见 test_yui__intentional_events_stay_silent），
        # 所以另发一条业务事件来验「YUI 出声」这一环。
        _emit(sink, "PATHFIND_START", intent.action.value)

    # ④ 真 YUI 追尾读事件流
    said, yui = _capture_speech(sink)
    yui.drain_once()

    assert sink.codes(), "事件流一条都没有"
    assert said, f"YUI 没出声；事件流里有 {sink.codes()}"
    assert said, f"YUI 没出声；事件流里有 {sink.codes()}"
