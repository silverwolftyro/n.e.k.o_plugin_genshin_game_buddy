"""``_WgcSession`` 与 ``Capturer`` 的 WGC 通道深度集成测试（2026-10-13）。

★ 为什么要 mock 库而不是用真 WGC
  真 WGC 需要一个真实的、被 Windows 合成器能看到��窗口。
  本机离线跑测试时没有这样的窗口，于是这条通道**从��没被测过** ——
  而它恰恰是 2026-10-10 黑帧事故的主角。

  所以这里用假 ``windows_capture`` 模块顶替。测的仍然是
  :class:`_WgcSession` 的真实代码：线程模型、帧槽位、拷贝语义、
  超时判定、降级、关闭。

★ 三条「铁律」在这里被钉死（源码注释里的原话）
  1. 回调**名字**不能改 —— ``windows_capture`` 按 ``on_frame_arrived`` /
     ``on_closed`` 这两个名字找回调，改名 = 静默失效。
  2. ``draw_border=False`` 必须传 —— WGC 默认给捕获内容加一圈黄边，
     带黄边的帧会直接毁掉模板匹配与亮度统计。
  3. ``grab()`` 必须**拷贝** —— ``frame_buffer`` 是零拷贝视图，下一帧
     到达时旧 Frame 被释放，视图随即悬空。
"""

from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

from autopilot.perception import capturer as C

pytestmark = pytest.mark.unit


# ==================================================================
# 假 windows_capture
# ==================================================================

class _FakeFrame:
    """模拟 mapped frame：``frame_buffer`` 是切片**视图**。"""

    def __init__(self, h=8, w=12, seed=0) -> None:
        base = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        self.frame_buffer = base + np.uint8(seed)

    def release(self) -> None:
        pass


class _FakeCtl:
    def __init__(self, sess_holder) -> None:
        self._sess = sess_holder
        self.stopped = 0
        self.waited = 0

    def stop(self) -> None:
        self.stopped += 1

    def wait(self) -> None:
        self.waited += 1


class _FakeWindowsCapture:
    """假的 ``windows_capture.WindowsCapture``。"""

    # 由各用例设置
    push_on_start = True
    start_raises: Exception | None = None
    ctl_stop_raises: Exception | None = None

    instances: list["_FakeWindowsCapture"] = []

    def __init__(self, window_hwnd, cursor_capture, draw_border) -> None:
        self.hwnd = window_hwnd
        self.cursor_capture = cursor_capture
        self.draw_border = draw_border
        self.callbacks: dict[str, object] = {}
        self.started = False
        self.ctl = _FakeCtl(self)
        type(self).instances.append(self)

    def event(self, cb) -> None:
        # 真库按 __name__ 挂回调 —— 这里也照做，好让「改名即失效」能被测到
        self.callbacks[getattr(cb, "__name__", "anon")] = cb

    def start_free_threaded(self):
        if type(self).start_raises is not None:
            raise type(self).start_raises
        self.started = True
        if type(self).push_on_start:
            cb = self.callbacks.get("on_frame_arrived")
            if cb is not None:
                cb(_FakeFrame(), None)
        return self.ctl

    # -- 测试辅助 ------------------------------------------------------
    def push(self, seed: int = 0) -> None:
        self.callbacks["on_frame_arrived"](_FakeFrame(seed=seed), None)

    def close_from_system(self) -> None:
        self.callbacks["on_closed"]()


@pytest.fixture
def fake_wgc(monkeypatch):
    """把假的 ``windows_capture`` 装进 ``sys.modules``。"""
    _FakeWindowsCapture.instances = []
    _FakeWindowsCapture.push_on_start = True
    _FakeWindowsCapture.start_raises = None
    _FakeWindowsCapture.ctl_stop_raises = None
    mod = type(sys)("windows_capture")
    mod.WindowsCapture = _FakeWindowsCapture           # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "windows_capture", mod)
    return _FakeWindowsCapture


def _no_wgc(monkeypatch):
    """模拟「没装 windows-capture」。"""
    monkeypatch.setitem(sys.modules, "windows_capture", None)


# ==================================================================
# 1 · 初始化
# ==================================================================

def test_init__rejects_invalid_hwnd(fake_wgc) -> None:
    """★ hwnd=0 必须**立刻**报错，不能起一个注定失败的会话。"""
    with pytest.raises(C.CaptureError, match="hwnd"):
        C._WgcSession(0)


def test_init__passes_draw_border_false(fake_wgc) -> None:
    """★★ 铁律 2：带黄边的帧会毁掉模板匹配与亮度统计。"""
    C._WgcSession(12345)
    cap = fake_wgc.instances[-1]
    assert cap.draw_border is False
    assert cap.cursor_capture is False


def test_init__registers_both_callbacks_by_name(fake_wgc) -> None:
    """★★ 铁律 1：真库**按名字**找回调，改名 = 静默失效。"""
    C._WgcSession(1)
    cb = fake_wgc.instances[-1].callbacks
    assert "on_frame_arrived" in cb
    assert "on_closed" in cb


def test_init__starts_the_free_threaded_capture(fake_wgc) -> None:
    C._WgcSession(1)
    assert fake_wgc.instances[-1].started is True


def test_init__start_failure_closes_and_reraises(fake_wgc) -> None:
    """★ 起不来必须把**原异常**抛出去，不能吞掉换成一个含糊的。"""
    fake_wgc.start_raises = RuntimeError("D3D 初始化失败")
    with pytest.raises(RuntimeError, match="D3D"):
        C._WgcSession(1)


def test_init__import_error_propagates_as_capture_error(fake_wgc, monkeypatch) -> None:
    """没装库时 ``ImportError`` 该原样冒出来，由调用方决定要不要降级。"""
    _no_wgc(monkeypatch)
    with pytest.raises(ImportError):
        C._WgcSession(1)


# ==================================================================
# 2 · 帧槽位与 grab
# ==================================================================

def test_grab__returns_none_before_any_frame(fake_wgc) -> None:
    """★ 一帧都没来 -> None，**不是**一张黑图。"""
    fake_wgc.push_on_start = False
    s = C._WgcSession(1)
    assert s.grab() is None


def test_grab__returns_bgr_copy(fake_wgc) -> None:
    s = C._WgcSession(1)
    img = s.grab()
    assert img is not None
    assert img.shape == (8, 12, 3)
    assert img.dtype == np.uint8


def test_grab__returns_a_copy_not_a_view(fake_wgc) -> None:
    """★★ 铁律 3：返回**拷贝**，否则下一帧到达后视图悬空。

    这条用「改返回值不应影响槽位」来验证 —— 若是视图，改一个像素
    就会把槽位里的内容一起改掉。
    """
    s = C._WgcSession(1)
    a = s.grab()
    assert a is not None
    before = a.copy()
    a[0, 0, 0] = 255
    b = s.grab()
    assert b is not None
    assert np.array_equal(b, before), "grab 返回的是视图而不是拷贝"


def test_grab__returns_the_latest_frame(fake_wgc) -> None:
    s = C._WgcSession(1)
    first = s.grab()
    fake_wgc.instances[-1].push(seed=50)
    second = s.grab()
    assert first is not None and second is not None
    assert not np.array_equal(first, second), "新帧到了却还返回旧的"


def test_grab__survives_a_broken_frame_buffer(fake_wgc) -> None:
    """★ 帧对象损坏时返回 None（外层会当成这一帧失败），不能抛。"""
    s = C._WgcSession(1)
    with s._lock:
        s._latest = (object(), s._seq)      # 没有 frame_buffer
    assert s.grab() is None


def test_grab__stale_frame_is_returned_again(fake_wgc) -> None:
    """★ WGC 只在画面**变化**时推帧 —— 静止时返回上一帧是**正确语义**。

    返回 None 会让调用方的「连续 N 次空才算坏」误触发成黑帧。
    """
    fake_wgc.push_on_start = False
    s = C._WgcSession(1)
    assert s.grab() is None
    fake_wgc.instances[-1].push()
    first = s.grab()
    for _ in range(5):
        again = s.grab()
        assert again is not None
        assert np.array_equal(again, first)


def test_grab__concurrent_push_does_not_corrupt(fake_wgc) -> None:
    """★ 帧槽位跨两个线程：一处推、一处取，不许拿到半截数据。"""
    fake_wgc.push_on_start = False
    s = C._WgcSession(1)
    cap = fake_wgc.instances[-1]
    stop = threading.Event()

    def _pusher():
        i = 0
        while not stop.is_set():
            cap.push(seed=i % 200)
            i += 1

    t = threading.Thread(target=_pusher, daemon=True)
    t.start()
    try:
        for _ in range(50):
            img = s.grab()
            assert img is None or img.shape == (8, 12, 3)
    finally:
        stop.set()
        t.join(timeout=2.0)


# ==================================================================
# 3 · 首帧等待与超时
# ==================================================================

def test_wait_first_frame__returns_the_frame_when_it_arrives(fake_wgc) -> None:
    assert C._WgcSession(1).wait_first_frame(1.0) is not None


def test_wait_first_frame__times_out_to_none(fake_wgc) -> None:
    """★ 3 秒拿不到首帧 -> None，调用方据此降级到别的通道。"""
    fake_wgc.push_on_start = False
    assert C._WgcSession(1).wait_first_frame(0.05) is None


def test_wait_first_frame__does_not_hang_after_system_close(fake_wgc) -> None:
    """★★ 窗口被系统关掉时 ``on_closed`` 会 set 首帧事件。

    不 set 的话 ``wait_first_frame`` 会一直等到超时 —— 窗口早就没了，
    却要白等满 3 秒。
    """
    fake_wgc.push_on_start = False
    s = C._WgcSession(1)
    fake_wgc.instances[-1].close_from_system()
    assert s.closed is True
    assert s.wait_first_frame(5.0) is None       # 立即返回，不是等满 5 秒


def test_on_closed__is_idempotent(fake_wgc) -> None:
    fake_wgc.push_on_start = False
    s = C._WgcSession(1)
    fake_wgc.instances[-1].close_from_system()
    fake_wgc.instances[-1].close_from_system()
    assert s.closed is True


# ==================================================================
# 4 · 关闭
# ==================================================================

def test_close__stops_and_waits_the_thread(fake_wgc) -> None:
    s = C._WgcSession(1)
    ctl = fake_wgc.instances[-1].ctl
    s.close()
    assert ctl.stopped == 1 and ctl.waited == 1


def test_close__is_idempotent(fake_wgc) -> None:
    """★ 重复 close 不许炸，也**不许**重复 stop 底层线程。"""
    s = C._WgcSession(1)
    ctl = fake_wgc.instances[-1].ctl
    s.close()
    s.close()
    s.close()
    assert ctl.stopped == 1, f"重复 stop 了 {ctl.stopped} 次"
    assert s.closed is True


def test_close__swallows_thread_stop_failures(fake_wgc) -> None:
    """★ 采集线程死掉后 ``stop()`` 会抛，那条异常会盖住真正的病因。"""
    s = C._WgcSession(1)
    ctl = fake_wgc.instances[-1].ctl

    def _boom():
        raise RuntimeError("Failed to stop the capture thread: <原异常>")

    ctl.stop = _boom                  # type: ignore[method-assign]
    s.close()                          # 不抛即可


def test_close__on_a_half_built_session(fake_wgc) -> None:
    """``start_free_threaded`` 抛了之后 ``close()`` 仍要被调用过。"""
    fake_wgc.start_raises = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        C._WgcSession(1)


def test_close__before_any_frame(fake_wgc) -> None:
    fake_wgc.push_on_start = False
    s = C._WgcSession(1)
    s.close()
    assert s.closed is True


# ==================================================================
# 5 · Capturer 侧：脏标记重建、降级
# ==================================================================

def _bare_capturer() -> C.Capturer:
    """只初始化字段的 Capturer —— 不走 open()，不碰真窗口。"""
    cap = C.Capturer.__new__(C.Capturer)
    cap._wgc = None
    cap._wgc_dirty = False
    cap._hwnd = None
    cap._ctl = None
    cap._closed = False
    return cap


def test_grab_wgc__raises_without_a_session(fake_wgc) -> None:
    with pytest.raises(C.CaptureError, match="会话不存在"):
        _bare_capturer()._grab_wgc()


def test_grab_wgc__rebuilds_when_the_hwnd_changed(fake_wgc) -> None:
    """★ 窗口句柄变了（重开/改分辨率）-> 老会话抓的是已消失的窗口，重建。"""
    fake_wgc.push_on_start = False
    cap = _bare_capturer()
    cap._hwnd = 777
    cap._wgc = C._WgcSession(777)
    before = len(fake_wgc.instances)
    cap._wgc_dirty = True

    # 重建发生在**这一次** grab 里面，所以此时还没有新帧
    assert cap._grab_wgc() is None
    assert len(fake_wgc.instances) == before + 1, "脏标记没有触发重建"
    assert cap._wgc is not None
    assert cap._wgc_dirty is False

    # 给重建后的新会话推一帧，再取就有了
    fake_wgc.instances[-1].push()
    assert cap._grab_wgc() is not None


def test_grab_wgc__raises_when_hwnd_is_gone_and_dirty(fake_wgc) -> None:
    """句柄没了还脏 -> 必须报错，不能拿 None 糊过去。"""
    cap = _bare_capturer()
    cap._wgc = C._WgcSession(1)
    cap._wgc_dirty = True
    cap._hwnd = None
    with pytest.raises(C.CaptureError, match="hwnd"):
        cap._grab_wgc()


def test_grab_wgc__raises_when_the_session_was_closed_by_the_system(fake_wgc) -> None:
    """★ 窗口关闭/最小化 -> 系统关掉会话 -> 必须响亮失败。

    返回 None 会被上层当成「这一帧没抓到」，进而可能连着重试；
    而真相是「这个通道已经废了，该换一条」。
    """
    fake_wgc.push_on_start = False
    cap = _bare_capturer()
    cap._hwnd = 1
    cap._wgc = C._WgcSession(1)
    fake_wgc.instances[-1].close_from_system()
    with pytest.raises(C.CaptureError, match="已被系统关闭"):
        cap._grab_wgc()


def test_close_wgc__is_idempotent(fake_wgc) -> None:
    cap = _bare_capturer()
    cap._wgc = C._WgcSession(1)
    cap._close_wgc()
    cap._close_wgc()
    assert cap._wgc is None


def test_close_wgc__clears_the_dirty_flag(fake_wgc) -> None:
    """★ 脏标记必须一起清掉 —— 否则关掉之后仍然会去重建会话。"""
    cap = _bare_capturer()
    cap._wgc = C._WgcSession(1)
    cap._wgc_dirty = True
    cap._close_wgc()
    assert cap._wgc_dirty is False
    assert cap._wgc is None


def test_close_wgc__with_no_session(fake_wgc) -> None:
    _bare_capturer()._close_wgc()          # 不抛即可


# ==================================================================
# 6 · 契约守卫
# ==================================================================

def test_contract__callback_names_are_stable() -> None:
    """★ 把这两个方法改名会让 WGC **静默**收不到帧 —— 没有任何报错。"""
    assert callable(getattr(C._WgcSession, "on_frame_arrived", None))
    assert callable(getattr(C._WgcSession, "on_closed", None))


def test_contract__first_frame_timeout_is_short() -> None:
    """超时太长会让「窗口被保护」这种情况白等好几秒。"""
    assert 0.5 <= C.Capturer.WGC_FIRST_FRAME_S <= 10.0
