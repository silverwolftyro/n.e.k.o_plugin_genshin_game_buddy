"""WGC 通道单测（2026-10-10）。

测的是**契约**，不是实现细节。假 `windows_capture` 模块刻意复刻了真包
两条最容易踩的约定，否则测出来的东西没意义：

1. ``event(handler)`` **按函数名分派** —— 名字不对就 ``ValueError``。
2. ``start()`` 会被我们永不调用；测试里记录调用的是哪个，防止有人改回
   阻塞版（本工具链上那会直接把主循环卡死）。

真机上 ``Frame.frame_buffer`` 是原生 mapped frame 的**零拷贝视图**，换帧
即释放旧内存。这里用普通 ndarray 替身，但「grab 必须返回拷贝」那条断言
照样成立且有效。
"""

from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

from autopilot.perception.capturer import (
    CaptureChannel,
    CaptureError,
    Capturer,
    _WgcSession,
)

# --------------------------------------------------------------------------
# 假 windows_capture
# --------------------------------------------------------------------------


class _FakeFrame:
    """替身。真包的 frame_buffer 是 (h, w, 4) BGRA 的零拷贝视图。"""

    def __init__(self, buf: np.ndarray) -> None:
        self.frame_buffer = buf
        self.height = int(buf.shape[0])
        self.width = int(buf.shape[1])
        self.timespan = 0


class _FakeCtl:
    def __init__(self) -> None:
        self.stopped = False
        self.waited = False
        self.stop_raises = False

    def stop(self) -> None:
        if self.stop_raises:
            raise RuntimeError("Failed to stop the capture thread: boom")
        self.stopped = True

    def wait(self) -> None:
        self.waited = True

    def is_finished(self) -> bool:
        return self.stopped


class _FakeWindowsCapture:
    """复刻真包的关键契约（按名字分派 / start_free_threaded）。"""

    instances: list["_FakeWindowsCapture"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.frame_handler = None
        self.closed_handler = None
        self.start_calls: list[str] = []
        self.ctl = _FakeCtl()
        _FakeWindowsCapture.instances.append(self)

    def event(self, handler):
        if handler.__name__ == "on_frame_arrived":
            self.frame_handler = handler
        elif handler.__name__ == "on_closed":
            self.closed_handler = handler
        else:
            raise ValueError(
                f"Invalid Event Handler {handler.__name__}")
        return handler

    def start(self):  # pragma: no cover - 我们不该走到这
        self.start_calls.append("start")

    def start_free_threaded(self):
        self.start_calls.append("start_free_threaded")
        return self.ctl

    # -- 测试用 --
    def push(self, buf: np.ndarray) -> None:
        assert self.frame_handler is not None, "on_frame_arrived 没注册"
        self.frame_handler(_FakeFrame(buf), _FakeCtl())

    def fire_closed(self) -> None:
        assert self.closed_handler is not None, "on_closed 没注册"
        self.closed_handler()


@pytest.fixture
def fake_wgc(monkeypatch):
    """把假模块塞进 sys.modules，替掉真包。"""
    _FakeWindowsCapture.instances.clear()
    mod = types.ModuleType("windows_capture")
    mod.WindowsCapture = _FakeWindowsCapture
    monkeypatch.setitem(sys.modules, "windows_capture", mod)
    return mod


def _bgra(h=8, w=8, value=200) -> np.ndarray:
    buf = np.zeros((h, w, 4), np.uint8)
    buf[:, :, 0] = value      # B
    buf[:, :, 1] = value      # G
    buf[:, :, 2] = value      # R
    return buf


def _sess(hwnd=1234) -> tuple[_WgcSession, _FakeWindowsCapture]:
    sess = _WgcSession(hwnd)
    return sess, _FakeWindowsCapture.instances[-1]


# --------------------------------------------------------------------------
# 枚举 / 顺序 / 别名
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_wgc__is_first_in_default_chain() -> None:
    order = [c.value for c in Capturer._normalize_order(None)]
    assert order == ["wgc", "bitblt", "dxcam", "mss"]


@pytest.mark.unit
@pytest.mark.parametrize("alias", ["wgc", "windows_capture", "WGC"])
def test_wgc__force_channel_alias(alias: str) -> None:
    cap = Capturer("x", force_channel=alias)
    assert cap._order == [CaptureChannel.WGC]


# --------------------------------------------------------------------------
# 初始化
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_wgc__registers_both_handlers_before_start(fake_wgc) -> None:
    sess, cap = _sess()
    assert cap.frame_handler is not None
    assert cap.closed_handler is not None
    # ★ 绝不能是阻塞版 start()
    assert cap.start_calls == ["start_free_threaded"]
    sess.close()


@pytest.mark.unit
def test_wgc__constructor_disables_border_and_cursor(fake_wgc) -> None:
    sess, cap = _sess(hwnd=4242)
    assert cap.kwargs["window_hwnd"] == 4242
    assert cap.kwargs["draw_border"] is False, "黄框不关会毁掉模板匹配"
    assert cap.kwargs["cursor_capture"] is False
    sess.close()


@pytest.mark.unit
def test_wgc__rejects_zero_hwnd(fake_wgc) -> None:
    with pytest.raises(CaptureError):
        _WgcSession(0)


@pytest.mark.unit
def test_init_channel__wgc_returns_false_when_dep_missing(monkeypatch) -> None:
    """没装 windows-capture 的机器上必须安静跳过，不能炸。"""
    import builtins

    real_import = builtins.__import__

    def _no_wgc(name, *a, **kw):
        if name == "windows_capture":
            raise ImportError("No module named 'windows_capture'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_wgc)
    cap = Capturer("x")
    cap._hwnd = 1
    assert cap._init_channel(CaptureChannel.WGC) is False
    assert cap._wgc is None


# --------------------------------------------------------------------------
# 取帧
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_wgc__grab_before_first_frame_is_none(fake_wgc) -> None:
    sess, _ = _sess()
    assert sess.grab() is None
    sess.close()


@pytest.mark.unit
def test_wgc__grab_returns_bgr_copy(fake_wgc) -> None:
    sess, cap = _sess()
    src = _bgra(value=200)
    cap.push(src)

    out = sess.grab()
    assert out is not None
    assert out.shape == (8, 8, 3), "必须是 HxWx3 BGR"
    # ★ 拷贝，不是视图：改返回值不能污染帧池的原始缓冲
    out[0, 0, 0] = 1
    assert src[0, 0, 0] == 200, "grab 泄漏了原生缓冲的视图"
    sess.close()


@pytest.mark.unit
def test_wgc__static_scene_returns_last_frame(fake_wgc) -> None:
    """WGC 只在画面变化时推帧；静止时必须返回上一帧而不是 None。

    返回 None 会让上游那套「连续 N 次空才算坏」的 _empty_streak 误判，
    把一条好端端的通道降级掉。
    """
    sess, cap = _sess()
    cap.push(_bgra(value=111))
    first = sess.grab()
    # 之后再推别的帧，验证旧帧不是被就地改写
    cap.push(_bgra(value=222))
    second = sess.grab()
    assert first is not None and second is not None
    assert float(first.mean()) == pytest.approx(111.0)
    assert float(second.mean()) == pytest.approx(222.0)
    sess.close()


@pytest.mark.unit
def test_wgc__on_closed_marks_session_closed(fake_wgc) -> None:
    sess, cap = _sess()
    cap.push(_bgra())
    cap.fire_closed()
    assert sess.closed is True


@pytest.mark.unit
def test_wgc__wait_first_frame_timeout_returns_none(fake_wgc) -> None:
    sess, _ = _sess()
    assert sess.wait_first_frame(0.2) is None
    sess.close()


@pytest.mark.unit
def test_wgc__wait_first_frame_returns_bgr(fake_wgc) -> None:
    sess, cap = _sess()

    def _later():
        # 从别的线程推，模拟真实的采集线程
        threading.Event().wait(0.05)
        cap.push(_bgra(value=99))

    threading.Thread(target=_later, daemon=True).start()
    out = sess.wait_first_frame(3.0)
    assert out is not None
    assert float(out.mean()) == pytest.approx(99.0)
    sess.close()


# --------------------------------------------------------------------------
# 关闭 / 重建
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_wgc__close_swallows_stop_exception(fake_wgc) -> None:
    """采集线程死掉后 stop() 会抛，异常会盖住真正的病因，必须吞掉。"""
    sess, cap = _sess()
    cap.ctl.stop_raises = True
    sess.close()          # 不抛就算过
    assert sess.closed is True


@pytest.mark.unit
def test_wgc__close_is_idempotent(fake_wgc) -> None:
    sess, _ = _sess()
    sess.close()
    sess.close()


@pytest.mark.unit
def test_capturer__grab_wgc_rebuilds_on_new_hwnd(fake_wgc) -> None:
    """窗口句柄变了必须重开会话，否则一直抓一个已经不存在的窗口。"""
    cap = Capturer("x")
    cap._active = CaptureChannel.WGC
    cap._hwnd = 1
    cap._wgc, first = _sess(hwnd=1)
    cap._wgc_dirty = True

    cap._hwnd = 2
    out = cap._grab_wgc()
    assert out is None, "新会话还没收到帧"
    assert cap._wgc is not first, "脏标记没有触发重建"
    assert _FakeWindowsCapture.instances[-1].kwargs["window_hwnd"] == 2
    assert cap._wgc_dirty is False
    cap._close_wgc()


@pytest.mark.unit
def test_capturer__grab_wgc_raises_when_closed(fake_wgc) -> None:
    cap = Capturer("x")
    cap._active = CaptureChannel.WGC
    cap._hwnd = 1
    cap._wgc, ctl_cap = _sess(hwnd=1)
    ctl_cap.fire_closed()
    with pytest.raises(CaptureError, match="已被系统关闭"):
        cap._grab_wgc()
    cap._close_wgc()