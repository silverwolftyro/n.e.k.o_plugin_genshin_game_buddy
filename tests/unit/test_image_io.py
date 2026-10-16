"""``perception/image_io.py`` 的单测（2026-10-12 任务 C）。

本文件的存在理由是**一条被证实的 Windows 坑**：

    cv2.imwrite(r"D:\\正常显示.png", img)
    -> 文件实际存成 D:\\姝ｅ父鏄剧ず.png，p.exists() == False
       而 cv2.imread 同一个路径「成功」了 —— 因为两者都走 ANSI 码页，
       互相自洽，**全程不报任何错**。

静默写坏文件名比直接报错难查得多：资源管理器里找不到、glob 匹配不到、
备份和清理脚本看不见。所以必须有测试钉住。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.perception.image_io import (  # noqa: E402
    imread_unicode,
    imwrite_unicode,
)

CN_NAME = "正常显示_脱困_测试"


def _img(w=60, h=40, v=128) -> np.ndarray:
    return np.full((h, w, 3), v, np.uint8)


# ==================================================================
# imwrite
# ==================================================================

@pytest.mark.unit
def test_imwrite__creates_a_file_with_a_chinese_name(tmp_path) -> None:
    """★ 核心回归：落盘文件名必须是**真中文**，不是乱码。"""
    p = tmp_path / f"{CN_NAME}.png"
    assert imwrite_unicode(p, _img()) is True
    assert p.exists(), (
        "文件没落到预期路径 —— imwrite_unicode 退回 ANSI 码页了？")
    assert p.stat().st_size > 0


@pytest.mark.unit
def test_imwrite__the_name_on_disk_is_exactly_what_we_asked_for(tmp_path) -> None:
    """反证：落盘名字就是中文本身。

    上一次写的是「GBK->UTF-8 还原后 != 原名」，但正确的中文名
    ``正常显示`` 恰好能被那个变换映射到别的串（甚至抛
    ``UnicodeDecodeError``）—— 判据本身不可靠。直接断言
    「落盘名 == 期望名」才是硬证据。
    """
    p = tmp_path / f"{CN_NAME}.png"
    assert imwrite_unicode(p, _img()) is True
    assert [q.name for q in tmp_path.iterdir()] == [f"{CN_NAME}.png"]


@pytest.mark.unit
def test_imwrite__adds_the_extension_when_missing(tmp_path) -> None:
    p = tmp_path / "no_ext"
    assert imwrite_unicode(p, _img()) is True
    assert p.with_suffix(".png").exists()


@pytest.mark.unit
def test_imwrite__creates_parent_dirs(tmp_path) -> None:
    p = tmp_path / "深" / "层" / "目录" / "图.png"
    assert imwrite_unicode(p, _img()) is True
    assert p.exists()


@pytest.mark.unit
def test_imwrite__unknown_extension_returns_false_not_raises(tmp_path) -> None:
    """★ ``cv2.imencode`` 对未知扩展名是**抛 cv2.error**，不是返回 False。

    只判返回值的实现会在这里整个崩掉 —— 这条钉的是「不抛」。
    """
    assert imwrite_unicode(tmp_path / "x.unknownext", _img()) is False


@pytest.mark.unit
def test_imwrite__a_directory_in_the_way_returns_false_not_raises(tmp_path) -> None:
    """目标是个已存在的目录 -> 写不进去 -> False，不抛。

    注：不能用 ``chmod(0o400)`` 制造「只读目录」—— Windows 忽略 POSIX
    权限位，那样造不出失败条件，测试会假绿。
    """
    d = tmp_path / "同名目录.png"
    d.mkdir()
    assert imwrite_unicode(d, _img()) is False


# ==================================================================
# imread
# ==================================================================

@pytest.mark.unit
def test_imread__reads_back_a_chinese_named_file(tmp_path) -> None:
    p = tmp_path / f"{CN_NAME}.png"
    imwrite_unicode(p, _img())
    got = imread_unicode(p)
    assert got is not None and got.shape == (40, 60, 3)


@pytest.mark.unit
def test_imread__honours_the_grayscale_flag(tmp_path) -> None:
    p = tmp_path / f"{CN_NAME}.png"
    imwrite_unicode(p, _img())
    import cv2
    got = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
    assert got is not None and got.ndim == 2


@pytest.mark.unit
def test_imread__missing_file_returns_none_not_raises(tmp_path) -> None:
    assert imread_unicode(tmp_path / "不存在.png") is None


@pytest.mark.unit
def test_imread__empty_file_returns_none(tmp_path) -> None:
    p = tmp_path / "空的.png"
    p.write_bytes(b"")
    assert imread_unicode(p) is None


@pytest.mark.unit
def test_imread__corrupt_file_returns_none(tmp_path) -> None:
    """有内容但不是图 -> None，不是异常。"""
    p = tmp_path / "假的.png"
    p.write_bytes(b"not an image at all")
    assert imread_unicode(p) is None


@pytest.mark.unit
def test_imread__directory_path_returns_none(tmp_path) -> None:
    assert imread_unicode(tmp_path) is None


@pytest.mark.unit
def test_imread__accepts_a_plain_string_path(tmp_path) -> None:
    """签名与 cv2.imread 一致，允许传 str。"""
    p = tmp_path / f"{CN_NAME}.png"
    imwrite_unicode(p, _img())
    assert imread_unicode(str(p)) is not None


# ==================================================================
# 与 cv2 原生 API 的对照（把坑写进测试）
# ==================================================================

@pytest.mark.unit
def test_cv2_native__really_does_corrupt_a_chinese_filename(tmp_path) -> None:
    """★ 把「原生 cv2 会写坏名字」钉成事实。

    这条是上面所有修复的**理由**。哪天 OpenCV 修好了（它确实有可能
    改用 W APIs），这条会红 —— 那时可以去删掉 image_io 这个绕路。
    在那之前它保证我们不会忘了为什么绕。
    """
    import cv2

    p = tmp_path / f"{CN_NAME}.png"
    cv2.imwrite(str(p), _img())

    if p.exists():
        pytest.skip("这个 OpenCV 版本已经走 Unicode 路径了 —— "
                    "image_io 可以退休了")
    # 文件确实没落在预期名字上
    landed = [q for q in tmp_path.iterdir() if q.name != p.name]
    assert landed, "既没有写在预期路径、也没有任何文件落盘？"


@pytest.mark.unit
def test_round_trip__preserves_pixel_values(tmp_path) -> None:
    """不只是能读能写，内容还得一模一样。"""
    src = np.zeros((12, 20, 3), np.uint8)
    src[2:5, 3:7] = (10, 200, 30)
    p = tmp_path / f"{CN_NAME}.png"
    assert imwrite_unicode(p, src) is True
    got = imread_unicode(p)
    assert got is not None
    assert np.array_equal(got, src)
