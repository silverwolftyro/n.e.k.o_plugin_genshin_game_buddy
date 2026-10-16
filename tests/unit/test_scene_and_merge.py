"""场景去抖 + 策略合并的单测（2026-10-11 补测）。

这两个模块此前零覆盖，而它们各自藏着一个「看起来能跑、实际悄悄错」
的失败模式：

* ``scene_detector`` 的去抖：时间戳倒退、时间不前进、首次调用 ——
  任何一个处理错了，``is_world`` 就会在错误的时刻翻转，
  而下游只看到「场景变了」，看不到「判定器自己坏了」。
* ``merger`` 的职责边界：它**只仲裁不校验本地意图**。把「本地场景不匹配
  也该拒」这种误解写进测试，会逼着后人往一个不做这件事的函数里塞校验。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.contracts.intent_packet import (  # noqa: E402
    ActionType,
    IntentPacket,
)
from autopilot.contracts.state_packet import StatePacket  # noqa: E402
from autopilot.perception.scene_detector import SceneDetector  # noqa: E402
from autopilot.policy.merger import (  # noqa: E402
    MergeResult,
    NullSignatureVerifier,
    merge_intents,
)

# ==================================================================
# scene_detector
# ==================================================================

def _detector(pred, n=0.2, m=0.2):
    """``pred`` 是可变开关（单元素 list），classifier 读它的当前值。"""
    return SceneDetector(n_seconds=n, m_seconds=m,
                         classifier=lambda _frame: pred[0])


@pytest.mark.unit
def test_scene__first_call_adopts_prediction_immediately() -> None:
    """首次调用**不该**等去抖时间 —— 否则引擎启动后 1 秒内场景是 UNKNOWN，
    而 WORLD 场景下唯三能触发的规则全哑火。"""
    pred = [True]
    st = _detector(pred).update(object(), 0.0)
    assert st.is_first is True
    assert st.raw_prediction is True
    assert st.is_world is True, "首次调用就应采用判定结果，不等去抖"
    # changed 的语义是「相对**上一个稳定态**发生了变化」。首次没有上一个，
    # 所以它是 False —— 这是对的：把首次当成「变了」会让上层误以为刚切过场景。
    assert st.changed is False


@pytest.mark.unit
def test_scene__debounces_entering_non_world() -> None:
    """连续 N 秒非大世界才切走 —— 单帧抖动不能触发场景翻转。"""
    pred = [True]
    d = _detector(pred, n=0.2)
    d.update(object(), 0.0)                     # 稳态 = WORLD

    pred[0] = False
    st1 = d.update(object(), 0.05)              # 只持续 50ms < 200ms
    assert st1.raw_prediction is False
    assert st1.is_world is True, "去抖期内不该翻转"
    assert st1.pending_seconds == pytest.approx(0.05, abs=1e-6)

    st2 = d.update(object(), 0.30)              # 累计 250ms > 200ms
    assert st2.is_world is False, "超过 N 秒就该切走"
    assert st2.changed is True


@pytest.mark.unit
def test_scene__pending_resets_when_prediction_flips_back() -> None:
    """预测又变回大世界时，**累计的待切换秒数必须清零**。

    不清零的话：三次各 100ms 的小抖动会被当成连续 300ms，于是一个
    从来没真正离开过 WORLD 的画面被判成非大世界。
    """
    pred = [True]
    d = _detector(pred, n=0.2)
    d.update(object(), 0.0)

    pred[0] = False
    d.update(object(), 0.10)                    # 累计 100ms
    pred[0] = True
    st = d.update(object(), 0.20)
    assert st.is_world is True
    assert st.pending_seconds == 0.0, "预测翻回时累计量必须清零"

    pred[0] = False
    d.update(object(), 0.30)
    st = d.update(object(), 0.55)
    assert st.is_world is False, "确实从 0 重新攒够了才该切"


@pytest.mark.unit
def test_scene__backwards_timestamp_is_clamped_to_zero_step() -> None:
    """时间戳倒退必须钳成 0 步长，不能把 pending 算成负数。"""
    pred = [True]
    d = _detector(pred, n=0.2)
    d.update(object(), 10.0)
    pred[0] = False
    st = d.update(object(), 5.0)                # 倒退 5 秒
    assert st.pending_seconds >= 0.0, "倒退的时间戳不能让 pending 变负"
    assert st.is_world is True, "0 步长不该触发翻转"


@pytest.mark.unit
def test_scene__same_timestamp_yields_zero_step() -> None:
    pred = [True]
    d = _detector(pred, n=0.2)
    d.update(object(), 1.0)
    pred[0] = False
    st = d.update(object(), 1.0)                # 完全相同的时间戳
    assert st.pending_seconds == 0.0


@pytest.mark.unit
def test_scene__reset_returns_to_uninitialized() -> None:
    pred = [True]
    d = _detector(pred)
    d.update(object(), 0.0)
    d.reset()
    assert d.update(object(), 1.0).is_first is True


@pytest.mark.unit
def test_scene__rejects_negative_debounce() -> None:
    with pytest.raises(ValueError):
        SceneDetector(n_seconds=-1.0)
    with pytest.raises(ValueError):
        SceneDetector(m_seconds=-1.0)


@pytest.mark.unit
def test_scene__zero_debounce_switches_immediately() -> None:
    """n=0 是合法配置（「不要去抖」），且应该立刻生效。"""
    pred = [True]
    d = _detector(pred, n=0.0)
    d.update(object(), 0.0)
    pred[0] = False
    assert d.update(object(), 0.001).is_world is False


# ==================================================================
# merger —— 只测它**真正做**的事
# ==================================================================

class _AlwaysOk:
    """假装签名有效的验证器。"""

    def verify(self, intent):
        return True, "OK"


def _intent(action="USE_SKILL", version=1, signature=""):
    """造一个意图。

    ⚠️ ``IntentPacket.make()`` **不接受** ``signature=``（它是 dataclass 字段
    但不在工厂签名里），所以只能造完再 ``object.__setattr__`` 补 ——
    dataclass 是 frozen 的，正常赋值会抛 FrozenInstanceError。
    另外 ``is_valid_for`` 只在 ``source == "CLOUD"`` 时才查签名
    （intent_packet.py:249），所以本地意图带不带签名都无所谓。
    """
    it = IntentPacket.make(
        ActionType(action), params={}, state_version=version,
        ttl_seconds=2, priority=50, scene_required=(), source="LOCAL")
    if signature:
        object.__setattr__(it, "signature", signature)
        object.__setattr__(it, "source", "CLOUD")
    return it


def _state():
    """一个**健康**的 WORLD 场景状态包。

    ⚠️ 直接用 ``StatePacket.unknown()`` 会让 merge 用例**假通过**：它的
       ``perception_ok=False``，于是 ``is_valid_for`` 一律返回
       ``PERCEPTION_DOWN``，云端意图根本走不到真正要测的那条校验 ——
       测试会绿，但绿的原因和它宣称的不一样。
       （同一条坑在 ``test_pick_mark.py`` 已经踩过一次。）
    """
    from autopilot.contracts.state_packet import ResourceInfo, SceneInfo

    st = StatePacket.unknown(game_id="genshin", state_version=1,
                             reason="test")
    object.__setattr__(st, "scene",
                       SceneInfo(id="WORLD", confidence=0.9, evidence=("t",)))
    object.__setattr__(st, "resource", ResourceInfo(perception_ok=True))
    return st


@pytest.mark.unit
def test_merge__local_wins_and_cloud_is_never_consulted() -> None:
    """本地优先：只要本地有意图，云端**连验签都不做**。"""
    calls = []

    class _Spy(_AlwaysOk):
        def verify(self, intent):
            calls.append(1)
            return True, "OK"

    res = merge_intents(_intent(), _intent(), _state(), now_ms=None,
                        signature_verifier=_Spy())
    assert res.source == "LOCAL"
    assert calls == [], "本地赢了就不该再去验云端的签"


@pytest.mark.unit
def test_merge__local_is_NOT_validated_here() -> None:
    """★ 本模块**不校验本地意图** —— 这是真实契约，不是漏洞。

    ``merge_intents`` 对 local 只做「有就返回」（merger.py:189-193），
    验签与 ``is_valid_for`` 都只作用于 cloud。本地意图来自我们自己的规则树，
    再验一遍是浪费；而场景 / 频率 / 急停的校验在**下游**
    ``executor.execute() -> validator.validate()`` —— 那里才是它们的归属地
    （SafetyGuard 的拒绝理由对本地和云端一视同仁，见 merger.py:174-184）。

    所以「merger 应该拒绝场景不匹配的本地意图」这种测试是**错的**：它会逼着
    后人往一个不做这件事的函数里塞校验，把职责搅乱。这条固化的是边界。
    """
    from autopilot.contracts.state_packet import SceneInfo

    st = _state()
    object.__setattr__(st, "scene",
                       SceneInfo(id="WORLD", confidence=0.9, evidence=("t",)))
    need_map = IntentPacket.make(
        ActionType.OPEN_MAP, params={}, state_version=1, ttl_seconds=2,
        priority=50, scene_required=("MAP_BIG",), source="LOCAL")
    assert merge_intents(need_map, None, st, now_ms=None,
                         signature_verifier=_AlwaysOk()).ok is True


@pytest.mark.unit
def test_merge__missing_signature_is_reported_before_verifier() -> None:
    """★ 验签的**顺序**有讲究：先看有没有签名，再谈验签。

    没有签名时直接报 ``SIGNATURE_MISSING``，**根本不调用验证器** ——
    少一次调用，也少一个「验证器抛异常」的分支。
    """
    calls = []

    class _Spy(_AlwaysOk):
        def verify(self, intent):
            calls.append(1)
            return True, "OK"

    res = merge_intents(None, _intent(), _state(), now_ms=None,
                        signature_verifier=_Spy())
    assert res.ok is False
    assert res.cloud_rejected == "SIGNATURE_MISSING"
    assert calls == [], "没有签名就不该去问验证器"


@pytest.mark.unit
def test_merge__signed_cloud_without_verifier_is_rejected_loudly() -> None:
    """有签名但**没配验证器** —— fail-closed，且原因要直白。"""
    res = merge_intents(None, _intent(signature="sig"), _state(), now_ms=None)
    assert res.ok is False
    assert "SIGNATURE_UNVERIFIABLE" in (res.cloud_rejected or "")


@pytest.mark.unit
def test_merge__signed_cloud_is_accepted() -> None:
    res = merge_intents(None, _intent(signature="sig"), _state(), now_ms=None,
                        signature_verifier=_AlwaysOk())
    assert res.ok is True
    assert res.source == "CLOUD"


@pytest.mark.unit
def test_merge__cloud_with_stale_state_version_is_rejected() -> None:
    """签名过了还要过 ``is_valid_for``：签名只证明「没被伪造」，
    不证明「这条意图还对应当前世界」。"""
    res = merge_intents(None, _intent(version=999, signature="sig"),
                        _state(), now_ms=None,
                        signature_verifier=_AlwaysOk())
    assert res.ok is False


@pytest.mark.unit
def test_merge__nothing_to_do_returns_none() -> None:
    """两个意图都没有时 ok=False，但**不抛** —— 空转是正常状态。"""
    res = merge_intents(None, None, _state(), now_ms=None)
    assert res.ok is False
    assert res.intent is None
    assert res.source == "NONE"


@pytest.mark.unit
def test_merge__null_verifier_is_the_default() -> None:
    assert isinstance(NullSignatureVerifier(), NullSignatureVerifier)


@pytest.mark.unit
def test_mergeresult__ok_tracks_intent_presence() -> None:
    # source / reason 是**必填位置参数** —— MergeResult 刻意强制每条结果
    # 都带原因（原则 1），不能省。
    assert MergeResult(None, "NONE", "空转").ok is False
    assert MergeResult(_intent(), "LOCAL", "本地").ok is True


@pytest.mark.unit
def test_merge__to_dict_is_jsonable() -> None:
    """MergeResult 要能进事件流，所以 to_dict 必须可序列化。"""
    import json

    d = merge_intents(_intent(), None, _state(), now_ms=None).to_dict()
    json.dumps(d, ensure_ascii=False)          # 不抛即为可序列化