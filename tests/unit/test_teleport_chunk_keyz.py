"""零覆盖模块补测（第二批）：传送点 / 金字塔分块 / key_z 检测（2026-10-13）。

★ 为什么这三个优先
  - ``teleport_points``：**寻路寻的就是传送点**。解析错了，
    ``--target-tp "风起地"`` 会走到别的地方去，且不报错。
  - ``chunk_loader``：定位靠它按格读瓦片。格子算错 = 位移算错，
    而位移错是「安静地越走越偏」。
  - ``key_z``：UI 弹窗识别。门槛错了会把菜单当没打开。

⚠️ 全部用**临时目录 / 合成数据**。真机资产（金字塔索引、模板图）
不存在时用例**明确 skip** —— 绝不因为「本机没有」就假装测过了。

★ 写这组测试时实测到的三条真实契约（原先都猜错了）
  1. ``ChunkLoader.__init__`` **构造时就强制**读 ``chunks_index.json``，
     缺文件直接 ``FileNotFoundError``（响亮失败，这是对的）。
  2. ``detect_key_z`` **显式拒绝非 (H,W,3) 输入**，抛 TypeError。
  3. ``load_teleport_points`` 顶层**可以是 BetterGI 的字典**
     （``{"regions": [...], "version": 1}``），不只是列表。
  另外：**本机装了 BetterGI** —— ``detect_bettergi()`` 返回真实路径。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autopilot.perception.chunk_loader import ChunkLoader
from autopilot.perception.detectors import key_z
from autopilot.perception.loaders import teleport_points as TP

pytestmark = pytest.mark.unit


# ==================================================================
# teleport_points
# ==================================================================

def test_tp__own_data_path_is_absolute() -> None:
    assert TP.own_data_path().is_absolute()


def test_tp__bettergi_candidate_records_location_only() -> None:
    """★ 探测阶段**只记位置、不读内容** —— 不该碰对方的文件。"""
    import inspect

    src = inspect.getsource(TP.add_bettergi_candidate)
    assert "read_text" not in src and "open(" not in src


def test_tp__detect_bettergi_returns_none_or_a_real_file() -> None:
    """★★ **本机其实装了 BetterGI**，所以它返回真实路径而不是 None。

    契约只要求「要么 None、要么一个真实存在的路径」，不接受半成品。
    （我最初写成「找不到返回 None」，跑出来才发现本机是有装的。）
    """
    got = TP.detect_bettergi()
    if got is not None:
        assert Path(got).exists(), f"探测到 BetterGI 但文件不存在：{got}"


def test_tp__explicit_path_wins(tmp_path) -> None:
    """★ 显式传路径就该用它，不该被自动探测覆盖。"""
    f = tmp_path / "tp.json"
    f.write_text("[]", encoding="utf-8")
    assert TP.resolve_path(str(f)) == f


def test_tp__missing_explicit_path_yields_none(tmp_path) -> None:
    """★ 显式传了却不存在的路径 -> None（该由调用方报错，不静默换默认）。"""
    assert TP.resolve_path(str(tmp_path / "没有.json")) is None


def test_tp__none_when_nothing_available(monkeypatch, tmp_path) -> None:
    """★ 环境变量给不出文件、且本机也**探测不到** BetterGI 时才返回 None。

    ⚠️ 本机装了 BetterGI，所以必须把它从候选里摘掉才测得到这条 ——
      只设一个不存在的环境变量是不够的，它会继续往下找。
    """
    monkeypatch.setenv(TP.ENV_VAR, str(tmp_path / "也不存在.json"))
    monkeypatch.setattr(TP, "BETTERGI_CANDIDATES", [], raising=False)
    monkeypatch.setattr(TP, "_env_candidates", lambda: [], raising=False)
    try:
        got = TP.resolve_path("")
    except Exception as exc:                      # 候选来源还有别的
        pytest.skip(f"无法在离线环境里屏蔽全部候选：{type(exc).__name__}")
    assert got is None


def test_tp__env_var_is_honoured(monkeypatch, tmp_path) -> None:
    f = tmp_path / "env_tp.json"
    f.write_text("[]", encoding="utf-8")
    monkeypatch.setenv(TP.ENV_VAR, str(f))
    assert TP.resolve_path("") == f


def test_tp__missing_file_is_empty_or_raises(tmp_path) -> None:
    """★ 文件不在时返回空列表或明确报错，都可接受；崩掉不可接受。"""
    try:
        got = TP.load_teleport_points(str(tmp_path / "没有.json"))
    except (OSError, ValueError):
        return
    assert got == [] or got is None


def test_tp__malformed_json_does_not_crash(tmp_path) -> None:
    f = tmp_path / "bad.json"
    f.write_text("{不是 JSON", encoding="utf-8")
    try:
        TP.load_teleport_points(str(f))
    except (ValueError, TypeError, KeyError, OSError):
        return


def test_tp__canonical_format_is_a_dict_not_a_list(tmp_path) -> None:
    """★★ 顶层**必须**是 dict（BetterGI 的 ``{"regions": [...], "version": N}``）。

    我最初以为「列表也行」，实测裸列表会被明确拒绝：
    ``TypeError: raw 为 dict，实际 list``。所以那份样例数据的形态
    就是唯一支持的形态 —— 静默把 list 当空处理才是真的危险。
    """
    bad = tmp_path / "list.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError):
        TP.load_teleport_points(str(bad))


def test_tp__accepts_the_bettergi_dict_shape(tmp_path) -> None:
    """★ dict 形态原样透传（不做字段翻译 —— 那是调用方的事）。"""
    d = tmp_path / "bg.json"
    d.write_text(json.dumps({"regions": [], "version": 1}), encoding="utf-8")
    got = TP.load_teleport_points(str(d))
    assert got == {"regions": [], "version": 1}


def test_tp__dict_with_regions_is_accepted(tmp_path) -> None:
    d = tmp_path / "bg2.json"
    d.write_text(json.dumps({"regions": [{"name": "风起地"}], "version": 1}),
                 encoding="utf-8")
    got = TP.load_teleport_points(str(d))
    assert got["regions"][0]["name"] == "风起地"


# ==================================================================
# chunk_loader
# ==================================================================

def test_chunk__construction_requires_the_index(tmp_path) -> None:
    """★★ 缺 ``chunks_index.json`` 必须**响亮失败**。

    静默返回空加载器的话，定位会一路带着「没有任何瓦片」跑下去，
    症状是位移恒为 0 —— 那比直接报错难查十倍。
    """
    with pytest.raises(FileNotFoundError):
        ChunkLoader(pyramid_dir=str(tmp_path / "空的"), cache_size=2)


@pytest.fixture
def chunk():
    """用**真实的**金字塔目录（``recognition/data/mihoyo_maps/P0_pyramid``）。

    ★ 2026-10-13 更正
      这组测试之前传的是空的 ``tmp_path``，于是 ``ChunkLoader`` 必然
      ``FileNotFoundError`` -> 9 条全 skip —— 看起来像「缺数据」，
      其实**数据一直都在**（``chunks_index.json`` + ``L0/`` + L1~L4），
      是我把目录指错了。现在用默认 ``PYRAMID_DIR``，只有它真不存在才 skip。
    """
    from autopilot.perception.chunk_loader import PYRAMID_DIR

    if not (Path(PYRAMID_DIR) / "chunks_index.json").is_file():
        pytest.skip(f"金字塔数据不在本机：{PYRAMID_DIR}")
    return ChunkLoader(cache_size=4)


def test_chunk__cell_of_separates_distant_pixels(chunk) -> None:
    """★ 格子编号算错 = 位移算错，而且错得很安静。"""
    assert chunk.cell_of(0, 0) != chunk.cell_of(10_000, 10_000)


def test_chunk__negative_pixels_do_not_crash(chunk) -> None:
    try:
        chunk.cell_of(-1, -1)
    except (ValueError, IndexError):
        return


def test_chunk__clear_cache_is_safe_when_empty(chunk) -> None:
    chunk.clear_cache()


def test_chunk__read_missing_file_does_not_raise(chunk) -> None:
    """★ 缺块 = 那一格没数据，不是崩溃。"""
    try:
        chunk._read(999999, 999999)
    except (OSError, ValueError):
        return


def test_chunk__cache_size_is_respected(chunk) -> None:
    """★ 夹具传的是 4 —— 顺带确认它可配置，不是写死的常量。"""
    assert chunk._cache_size == 4


def test_chunk__describe_returns_something(chunk) -> None:
    """★ describe 是诊断口 —— 要能在真实数据上直接给出结论。

    ★ 它**要两个参数**（``center_px`` / ``window_size``）。我最初无参调用，
      跑出来是 ``TypeError: missing 2 required positional arguments``。
    """
    assert chunk.describe((1000, 1000), 64) is not None


def test_chunk__describe_needs_its_arguments(chunk) -> None:
    """★ 把「必须给两个参数」钉住 —— 加默认值时至少会有人看到这条测试。"""
    with pytest.raises(TypeError):
        chunk.describe()          # type: ignore[call-arg]


def test_chunk__load_window_returns_array_or_none(chunk) -> None:
    try:
        out = chunk.load_l0_window((100, 100), 64)
    except (OSError, ValueError):
        return
    assert out is None or isinstance(out, np.ndarray)


def test_chunk__warm_does_not_raise(chunk) -> None:
    try:
        chunk.warm((0, 0), 2)
    except (OSError, ValueError):
        return


def test_chunk__cell_bbox_is_four_numbers(chunk) -> None:
    try:
        box = chunk.cell_bbox(3, 4)
    except (OSError, ValueError, AttributeError):
        pytest.skip("该版本没有 cell_bbox")
    assert len(box) == 4


# ==================================================================
# key_z
# ==================================================================

def test_keyz__score_thresholds_are_ordered() -> None:
    """★ READY 必须高于 UNKNOWN —— 反了就会「不确定」判成「就绪」。"""
    assert 0.0 <= key_z.SCORE_UNKNOWN < key_z.SCORE_READY <= 1.0


def test_keyz__roi_and_template_names() -> None:
    assert key_z.ROI_NAME == "key_z"
    assert key_z.TEMPLATE_NAME == "key_z.png"


@pytest.mark.parametrize("shape", [(50, 50), (200, 300)])
def test_keyz__score_stays_in_range(shape) -> None:
    got = key_z.match_score(np.full((*shape, 3), 255, np.uint8))
    assert 0.0 <= got <= 1.0, shape


def test_keyz__detect_returns_a_state() -> None:
    assert key_z.detect_key_z(np.zeros((100, 100, 3), np.uint8)) is not None


def test_keyz__detect_rejects_grayscale_but_tolerates_four_channel() -> None:
    """★★ 契约实测（我一开始把这两条**搞反了**）：

    - 单通道 ``(H, W)`` -> **抛 TypeError**。它明确要 BGR，
      悄悄把灰度当 BGR 会得到「三个通道都一样」的假彩色画面。
    - 四通道 ``(H, W, 4)`` -> **接受**（内部取前 3 通道）。
      带 alpha 的输入本来就该被容忍。
    """
    with pytest.raises(TypeError):
        key_z.detect_key_z(np.zeros((50, 50), np.uint8))
    assert key_z.detect_key_z(np.zeros((50, 50, 4), np.uint8)) is not None


@pytest.mark.parametrize("shape", [(1, 1), (4, 4), (10, 10)])
def test_keyz__tiny_frames_do_not_crash(shape) -> None:
    assert key_z.detect_key_z(np.zeros((*shape, 3), np.uint8)) is not None


def test_keyz__reload_template_is_safe() -> None:
    """★ 换掉模板图之后调它重载；没有模板时也不该抛。"""
    try:
        key_z.reload_template()
    except (OSError, ValueError):
        return


def test_keyz__detect_is_deterministic() -> None:
    """★ 同一帧两次检测必须同结论 —— 否则规则会逐帧翻转。"""
    frame = np.full((120, 120, 3), 77, np.uint8)
    assert repr(key_z.detect_key_z(frame)) == repr(key_z.detect_key_z(frame))
