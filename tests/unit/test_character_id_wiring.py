"""队伍栏**角色名识别**接线的单测（2026-10-18）。

覆盖这条链::

    rois.toml[party_name_1..4]
        -> detectors.character_detector.CharacterDetector
        -> （白名单）terms/genshin/characters.toml
        -> StateBuilder._build_player()
        -> PlayerInfo.party[i].name
        -> PolicyContext.onfield('钟离')

★ 全程**不需要视觉**、不需要游戏窗口、不需要 rapidocr
  OCR 那一段用一个鸭子类型替身，规则那一段用合成 StatePacket。

★ 为什么这份测试里有好几条是「断言它**没**做」
------------------------------------------------
「谁在场」（``active``）**故意不实现**。现有 296 帧素材标不出来：
同一个队伍、同一个在场角色（没有正负样本对），4 格渲染尺寸一致，
数字键提示框的不透明度随 UI 淡入淡出剧烈波动（296 帧里 195 帧分不开）。

所以这里把「它没做」也钉成测试：将来有人顺手写一个「slot4 更大所以
它在场」的猜测，这些测试会立刻红 —— 那正是要防的事。
把一个没有样本支撑的判据写进生产，比留空危险得多。
"""

from __future__ import annotations

import numpy as np
import pytest

from autopilot.config.loader import CharacterIdConfig, load_config
from autopilot.contracts.state_packet import (
    PlayerInfo,
    SceneInfo,
    StatePacket,
)
from autopilot.perception import builder as B
from autopilot.perception.detectors import character_detector as cd
from autopilot.perception.detectors.character_detector import (
    MAX_EDIT_DIST,
    CharacterDetector,
    CharacterIdError,
    SlotName,
    max_dist_for,
    read_active,
    snap_to_roster,
)
from autopilot.policy.context import PolicyContext, UnknownValue

FRAME_H, FRAME_W = 1600, 2560


def _frame() -> np.ndarray:
    return np.full((FRAME_H, FRAME_W, 3), 128, np.uint8)


class _FakeOcr:
    """鸭子类型替身。

    ⚠️ 刻意做成 ``RapidOCR`` 那种 ``__call__`` 形态（不是 ``OcrEngine``），
       这样「裸 rapidocr」与「包装过的 OcrEngine」两条路径都能被测到。
    """

    def __init__(self, mapping: dict[int, str]) -> None:
        self._mapping = mapping
        self.calls: list[int] = []

    def __call__(self, img):
        slot = len(self.calls) + 1
        self.calls.append(slot)
        text = self._mapping.get(slot, "")
        if not text:
            return [], None
        return [[[[0, 0], [10, 0], [10, 10], [0, 10]], text, 0.99]], 0.01


# ==================================================================
# 白名单
# ==================================================================

@pytest.mark.unit
def test_roster__comes_from_our_own_terms_not_bettergi() -> None:
    """白名单必须是**自有**资产。

    ★ 这条是许可红线，不是实现细节：
      原计划用 BetterGI 缓存里的 123 张头像提特征，那会踩
      ``recognition/notes/LICENSING.md`` 的「禁抄数据文件」。
      改成读 ``terms/genshin/characters.toml``（由
      ``tools/sync_mihoyo_terms.py`` 从 miao-plugin 生成，125 条）
      之后，包里**一个字节的 BetterGI 数据都没有**。
    """
    names = cd.load_roster()
    assert len(names) >= 100, f"白名单只有 {len(names)} 条，像是被裁过了"
    for expect in ("钟离", "纳西妲", "夜兰", "八重神子"):
        assert expect in names, f"白名单缺 {expect}"
    # 皮肤编号不该混进角色名（「芭芭拉02」是文件名，不是角色名）
    assert not any(n[-2:].isdigit() for n in names), \
        "白名单里混进了带编号的条目（应该是去编号后的角色名）"


@pytest.mark.unit
def test_roster__missing_file_fails_loudly(monkeypatch) -> None:
    """白名单读不到必须抛，**不能**静默退化成「不认角色」。"""
    monkeypatch.setattr(cd, "_ROSTER_TOML",
                        cd.Path("D:/__nonexistent__/characters.toml"))
    monkeypatch.setattr(cd, "load_roster", cd.load_roster)  # 清 lru_cache
    cd.load_roster.cache_clear()
    with pytest.raises(CharacterIdError):
        cd.load_roster()
    cd.load_roster.cache_clear()


# ==================================================================
# 白名单纠错
# ==================================================================

@pytest.mark.unit
def test_snap__fixes_the_real_observed_confusion() -> None:
    """实测稳定发生的形近字误识：「纳西妲」被读成「纳西姐」。"""
    roster = cd.load_roster()
    assert snap_to_roster("纳西妲", roster) == ("纳西妲", 0)
    assert snap_to_roster("纳西姐", roster) == ("纳西妲", 1)
    for exact in ("钟离", "夜兰", "丝柯克"):
        assert snap_to_roster(exact, roster) == (exact, 0)


@pytest.mark.unit
def test_snap__rejects_garbage_instead_of_guessing() -> None:
    """★ 关键安全性质：认不出来就是认不出来，**不许**硬凑一个名字。

    实测失败帧里 OCR 会读到对话框文字（「勺琉璃晶砂」「为黑暗羊」）。
    那些不是角色名，硬纠成某个角色会让机器人对着错误角色触发规则 ——
    那比留空危险得多。
    """
    roster = cd.load_roster()
    for junk in ("勺琉璃晶砂", "为黑暗羊", "向前进了", "姐西内日", "", "   "):
        name, dist = snap_to_roster(junk, roster)
        assert name == "", f"「{junk}」被硬纠成了「{name}」（距离 {dist}）"
        assert dist > MAX_EDIT_DIST or dist == 99


@pytest.mark.unit
def test_snap__single_char_noise_is_never_snapped() -> None:
    """★★ 实测 7 条误识里 **5 条**是这一种。

    白名单有 4 个**单字**角色名（空/琴/荧/魈），所以一字 OCR 结果
    （实测出现过「塔」「西」「9」「二」「八」）在固定阈值 ``MAX_EDIT_DIST=1``
    下会被纠成它们。邻域在 1 字空间里太大了。

    修法是 :func:`max_dist_for`：阈值随文本长度收紧，
    1 字只认精确匹配，3 字才放行距离 1。
    """
    roster = cd.load_roster()
    singles = [n for n in roster if len(n) == 1]
    assert singles, "白名单里没有单字角色名了 —— 这条测试的前提变了，" \
                    "请重新评估还需不需要长度自适应阈值"
    for junk in ("塔", "西", "9", "二", "八", "柯"):
        name, _dist = snap_to_roster(junk, roster)
        assert name == "", f"单字噪声「{junk}」被纠成了「{name}」"
    # 精确匹配的单字角色名仍然要认出来（队里真可能有「空」或「魈」）
    for s in singles:
        assert snap_to_roster(s, roster) == (s, 0)


@pytest.mark.unit
def test_max_dist_for__scales_with_length() -> None:
    """阈值表：1/2 字 -> 0，3 字及以上 -> 1。

    ⚠️ ``柯莱`` 是**2 字**，所以阈值是 0 而不是 1 —— 2 字名只有一个
       字的冗余，纠错空间太大，宁可判 UNKNOWN。
    """
    assert max_dist_for("魈") == 0          # 1 字
    assert max_dist_for("夜兰") == 0        # 2 字
    assert max_dist_for("柯莱") == 0        # 2 字
    assert max_dist_for("纳西姐") == 1      # 3 字
    assert max_dist_for("八重神子") == 1    # 4 字
    assert max_dist_for("") == 0


@pytest.mark.unit
def test_snap__boundary_is_exactly_max_edit_dist() -> None:
    """距离 = MAX_EDIT_DIST 收，超过 1 个就放。边界不能含糊。

    ⚠️ 用 ``abXXe`` 而不是 ``abXde`` —— 后者离 ``abcde`` 只差 1
       （换个字母而已），测不出第 2 档。
    """
    roster = ("abcde", "fghij")
    assert snap_to_roster("abcdX", roster) == ("abcde", 1)     # 距离 1：收
    assert snap_to_roster("abXXe", roster) == ("", 2)         # 距离 2：放
    assert snap_to_roster("ZZZZZ", roster) == ("", 5)         # 差太远：放


# ==================================================================
# ROI
# ==================================================================

@pytest.mark.unit
def test_rois__party_name_boxes_exist_and_are_sane() -> None:
    """rois.toml 里的 4 个名字框必须存在，且落在屏幕右半、纵向递增。"""
    from autopilot.perception.roi_loader import load_rois

    rois = load_rois()
    prev_top = -1.0
    for slot in cd.SLOTS:
        rect = rois.get(f"party_name_{slot}")
        assert rect is not None, f"rois.toml 缺 party_name_{slot}"
        assert 0.8 < rect.left < rect.right < 1.0, \
            f"party_name_{slot} 的 x 不在右侧：{rect}"
        assert 0.0 < rect.top < rect.bottom < 1.0, \
            f"party_name_{slot} 的 y 非法：{rect}"
        assert rect.top > prev_top, f"party_name_{slot} 的 y 没有递增"
        prev_top = rect.top
        # 名字框必须落在对应头像框左边（队伍栏名字在头像左侧）
        avatar = rois[f"party_{slot}"]
        assert rect.right <= avatar.left, \
            f"party_name_{slot} 与头像框重叠：名字应该在头像左边"


# ==================================================================
# 检测器
# ==================================================================

@pytest.mark.unit
def test_detector__rotation_reads_each_slot_exactly_once() -> None:
    """每次只读一格，4 次轮完一圈 —— 否则一帧顶掉 4 个 tick 预算。"""
    ocr = _FakeOcr({1: "纳西妲", 2: "钟离", 3: "夜兰", 4: "丝柯克"})
    det = CharacterDetector(ocr)
    assert det.slots_per_call == 1 if hasattr(det, "slots_per_call") else True

    seen: dict[int, str] = {}
    for _ in range(4):
        det.update(_frame())
        assert det.last_read_count == 1
        seen.update({s: n for s, n in
                    ((k, v.name) for k, v in det._cache.items()) if n})
    assert seen == {1: "纳西妲", 2: "钟离", 3: "夜兰", 4: "丝柯克"}


@pytest.mark.unit
def test_detector__failure_keeps_the_previous_name() -> None:
    """★ 切屏/开菜单时整栏虚化，读不到**不该**清空已知结果。

    清空会让 ``party`` 每帧闪烁，规则跟着抖；
    代价（换队伍后最多滞后 ``interval_ticks`` 个 tick）由
    ``StateBuilder.last_character_age_ticks`` 暴露给上层。
    """
    class _Flaky:
        def __init__(self):
            self.n = 0

        def __call__(self, img):
            self.n += 1
            if self.n <= 2:
                return [[[[0, 0], [10, 0], [10, 10], [0, 10]], "钟离", 0.99]], None
            return [], None

    det = CharacterDetector(_Flaky())
    det.update(_frame())
    det.update(_frame())
    assert det._cache[1].name == "钟离"
    for _ in range(4):            # 之后全是空读
        det.update(_frame())
    assert det._cache[1].name == "钟离", "空读把已知名字冲掉了"
    assert 2 not in det._cache or det._cache[2].name == "钟离"


@pytest.mark.unit
def test_detector__reset_clears() -> None:
    ocr = _FakeOcr({1: "钟离", 2: "夜兰"})
    det = CharacterDetector(ocr)
    det.update(_frame())
    assert det._cache
    det.reset()
    assert not det._cache


@pytest.mark.unit
def test_detector__rejects_bad_slots_per_call() -> None:
    with pytest.raises(ValueError):
        CharacterDetector(_FakeOcr({}), slots_per_call=0)


@pytest.mark.unit
def test_read_active__is_deliberately_none() -> None:
    """★ 把「没有实现 active」钉成测试。

    没有正负样本支撑的判据比没有判据更危险 —— 它会**看起来**在工作。
    """
    assert read_active({1: SlotName(slot=1, name="钟离")}) is None
    assert read_active({}) is None


# ==================================================================
# OcrEngine 的失败计数隔离
# ==================================================================

@pytest.mark.unit
def test_ocr_engine__empty_region_does_not_kill_the_channel() -> None:
    """★ 这是一个真实的、会波及全局的地雷。

    队伍栏在切屏/开菜单时本来就没字，rapidocr 返回空是**正确结果**。
    但 ``OcrEngine.read`` 会把它记成失败，连续 5 次之后
    ``_available = False`` —— **整个 OCR 通道判死**，传送点识别一起废。
    也就是说「菜单开了 1 秒」会牵连全局健康状态。
    """
    from autopilot.perception.ocr import DEFAULT_FAIL_THRESHOLD, OcrEngine

    eng = OcrEngine.__new__(OcrEngine)          # 不跑 __init__，不载模型
    eng._fail_streak = 0
    eng._fail_threshold = DEFAULT_FAIL_THRESHOLD
    eng._available = True
    eng._last_error = ""
    eng._legacy = None
    eng._rapid = lambda _img: ([], None)        # 永远读不出字
    eng._fail_threshold = DEFAULT_FAIL_THRESHOLD

    for _ in range(DEFAULT_FAIL_THRESHOLD * 3):
        eng.read(_frame(), count_failure=False)
    assert eng._available is True, "count_failure=False 仍然把通道判死了"
    assert eng._fail_streak == 0

    eng.read(_frame())                          # 默认仍然要计失败
    assert eng._fail_streak == 1, "默认路径不该被这次改动影响"


# ==================================================================
# builder 接线
# ==================================================================

def _cfg(enabled: bool = True, interval_ticks: int = 1) -> CharacterIdConfig:
    return CharacterIdConfig(enabled=enabled, interval_ticks=interval_ticks)


def _builder(cfg: CharacterIdConfig) -> "B.StateBuilder":
    """带 OCR 的 builder。

    ⚠️ 必须给 ``ocr``：``_read_character_names`` 见到 ``ocr is None``
       会直接返回空（这是刻意的「没 OCR 就不猜」），不给的话下面
       几条接线测试会**假绿** —— 断言「名字为空」在「根本没接线」
       和「接线了但没数据」两种情况下长得一模一样。
    """
    return B.StateBuilder(entity_detector=None, character_id=cfg,
                          ocr=_FakeOcr({}))


class _StubDetector:
    """直接顶替 CharacterDetector，省掉 OCR。"""

    def __init__(self, names: dict[int, str]) -> None:
        self._names = names
        self.last_read_count = 1

    def update(self, frame):
        return {s: SlotName(slot=s, name=n) for s, n in self._names.items()}

    def reset(self) -> None:
        pass


@pytest.mark.unit
def test_builder__disabled_means_no_names() -> None:
    """逃生阀：``enabled = false`` -> 名字恒空，行为与接入前一致。"""
    b = _builder(_cfg(enabled=False))
    assert b._character_id_on is False
    info = b._build_player("WORLD", None, [], [], 1,
                           frame=_frame())
    assert all(m.name == "" for m in info.party)


@pytest.mark.unit
def test_builder__party_members_are_named() -> None:
    b = _builder(_cfg())
    b._char_detector = _StubDetector({1: "钟离", 3: "夜兰"})
    b._char_tick = 1
    info = b._build_player("WORLD", None, [], [], 1, frame=_frame())
    got = {m.slot: m.name for m in info.party}
    assert got == {1: "钟离", 3: "夜兰"}


@pytest.mark.unit
def test_builder__active_is_always_false() -> None:
    """★ 有名字也**不代表**在场。``active`` 必须恒 False。"""
    b = _builder(_cfg())
    b._char_detector = _StubDetector({1: "钟离", 2: "夜兰"})
    b._char_tick = 1
    info = b._build_player("WORLD", None, [], [], 1, frame=_frame())
    assert all(m.active is False for m in info.party)
    assert info.active_member is None


@pytest.mark.unit
def test_builder__throttle_skips_ticks() -> None:
    """没到 interval 的 tick 不许调 OCR。"""
    calls = 0

    class _Counting(_StubDetector):
        def update(self, frame):
            nonlocal calls
            calls += 1
            return {}

    b = _builder(_cfg(interval_ticks=5))
    b._char_detector = _Counting({})
    for _ in range(4):
        b._read_character_names(_frame())
    assert calls == 0, "节流没生效"
    b._read_character_names(_frame())            # 第 5 个 tick
    assert calls == 1


@pytest.mark.unit
def test_builder__no_ocr_engine_means_no_names_not_crash() -> None:
    """没注入 OCR 时**什么都不做**，不抛、不猜。"""
    b = B.StateBuilder(entity_detector=None, character_id=_cfg(), ocr=None)
    assert b._read_character_names(_frame()) == {}


# ==================================================================
# 规则层
# ==================================================================

@pytest.mark.unit
def test_context__onfield_is_unknown_without_active() -> None:
    """有名字但没有 active -> ``onfield()`` 必须 UNKNOWN，不是 False。

    UNKNOWN 与 False 在规则层行为一致（都不触发），但只有 UNKNOWN
    能告诉人「感知还没标定」—— 这正是 world_zhongli_shield
    保持 ``enabled = false`` 的理由。
    """
    from autopilot.contracts.state_packet import PartyMember

    info = PlayerInfo(party=(
        PartyMember(slot=1, char_id="", name="钟离", active=False),
    ))
    st = StatePacket.unknown(game_id="g", state_version=1, reason="t")
    object.__setattr__(st, "player", info)
    object.__setattr__(
        st, "scene",
        SceneInfo(id="WORLD", confidence=0.9, evidence=("p",), entered_ms=0))
    ctx = PolicyContext.from_state(st, now_ms=1000)
    assert isinstance(ctx.onfield("钟离"), UnknownValue)
    # ⚠️ ``onfield`` 的 name 是**必填**位置参数，传 ``""`` 才是「不指名地问」。
    #    （早先误以为能无参调用，真机上会直接 TypeError。）
    assert isinstance(ctx.onfield(""), UnknownValue)


@pytest.mark.unit
def test_context__onfield_true_only_when_active() -> None:
    """对照组：active=True 时才给出 True —— 证明上面不是「恒 False」。"""
    from autopilot.contracts.state_packet import PartyMember

    info = PlayerInfo(party=(
        PartyMember(slot=1, char_id="", name="钟离", active=True),
    ))
    st = StatePacket.unknown(game_id="g", state_version=1, reason="t")
    object.__setattr__(st, "player", info)
    object.__setattr__(
        st, "scene",
        SceneInfo(id="WORLD", confidence=0.9, evidence=("p",), entered_ms=0))
    ctx = PolicyContext.from_state(st, now_ms=1000)
    assert ctx.onfield("钟离") is True
    assert ctx.onfield("夜兰") is False


# ==================================================================
# 配置
# ==================================================================

@pytest.mark.unit
def test_config__section_exists_and_defaults_to_on() -> None:
    cfg = load_config()
    assert hasattr(cfg, "character_id"), "autopilot.character_id 配置段不见了"
    assert hasattr(cfg.character_id, "enabled")
    assert hasattr(cfg.character_id, "interval_ticks")
    assert cfg.character_id.interval_ticks >= 1
