"""Windows 输入模拟：ctypes SendInput 薄封装。

对应架构文档 3.5 与 6.7。

🔴 **为什么不用 pyautogui / pynput**：
    两者都没有 **SendInput 返回值校验**。`SendInput` 的返回值是「实际注入的
    事件数」——**不等就说明失败**（权限不足 / 被安全软件拦截 / UIPI 拦下），
    而这种失败是完全静默的。BetterGI 专门为此自己实现了 `InputSimulator`
    并在返回值不足时抛异常 `[实测]`，这是正确的做法，这里照抄这个思路。

    pynput 还有个问题：它的抽象是「事件监听 + 发送」，不适合做
    **组合键的原子提交**（Shift+W 分两次注入，游戏可能只收到 W）。

**测试友好**：`_backend` 可注入，用 `DryRunBackend` 可以在不发任何真实输入的
情况下完整跑通 executor / validator / 熔断器的逻辑。

    with InputSimulator(dry_run=True) as sim:
        sim.key_combo([VK.SHIFT], VK.W)   # 只记录，不真的发
"""

from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Win32 常量
# --------------------------------------------------------------------------

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

# MapVirtualKey / MapVirtualKeyEx 的 uMapType
MAPVK_VK_TO_VSC = 0
MAPVK_SCANCODE_TO_VK = 1
MAPVK_VK_TO_EXSCANCODE = 4

#: 键盘注入模式（`_mk_keys` 的 `mode` 参数）。
#: 纯 VK（wScan=0，不带 KEYEVENTF_SCANCODE）是 Windows 标记的
#: 「程序合成的虚拟键码事件」；真实硬件走的是扫描码路径
#: （物理键 set-1 扫描码 → i8042/PS2 → 系统翻译成 VK）。
#: 带有反作弊/自定义输入层的游戏可以区分两者，于是出现
#: 「手动 Shift+W 跑步、注入 Shift+W 只走路」。
KEY_MODE_VK = "vk"
KEY_MODE_SCANCODE = "scancode"
KEY_MODE_BOTH = "both"
KEY_MODES: frozenset[str] = frozenset({KEY_MODE_VK, KEY_MODE_SCANCODE, KEY_MODE_BOTH})

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

WHEEL_DELTA = 120

#: 需要 wVk=0 标记的扩展键（Shift/Ctrl/Alt/Win 及小键盘、方向键等）。
#: 少了这个标记，某些键位「按下没反应」—— BetterGI 的注释原话：
#: 「解决 shift 之类的键位没法正常使用的问题」
EXTENDED_KEYS: frozenset[int] = frozenset({
    0x10, 0x11,                       # VK_SHIFT / VK_CONTROL
    0x12,                             # VK_MENU
    0x5B, 0x5C, 0x5D,                 # VK_LWIN / VK_RWIN / VK_APPS
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E,
    0x2F, 0x5E,                       # 方向键 / NUMLOCK / SCROLLLOCK
    0x6A, 0x6B, 0x6C, 0x6D,           # 小键盘 * / + / - / .
    0x90, 0x91,                       # NUMLOCK / SCROLLLOCK
})


class VK(IntEnum):
    """Windows 虚拟键码。只列执行层会用到的。"""

    BACK = 0x08
    TAB = 0x09
    ENTER = 0x0D
    SHIFT = 0x10
    CONTROL = 0x11
    MENU = 0x12
    PAUSE = 0x13
    CAPITAL = 0x14
    ESCAPE = 0x1B
    SPACE = 0x20
    PRIOR = 0x21
    NEXT = 0x22
    END = 0x23
    HOME = 0x24
    LEFT = 0x25
    UP = 0x26
    RIGHT = 0x27
    DOWN = 0x28
    SELECT = 0x29
    PRINT = 0x2A
    EXECUTE = 0x2B
    SNAPSHOT = 0x2C
    INSERT = 0x2D
    DELETE = 0x2E
    HELP = 0x2F

    ZERO = 0x30
    ONE = 0x31
    TWO = 0x32
    THREE = 0x33
    FOUR = 0x34
    FIVE = 0x35
    SIX = 0x36
    SEVEN = 0x37
    EIGHT = 0x38
    NINE = 0x39
    A = 0x41
    B = 0x42
    C = 0x43
    D = 0x44
    E = 0x45
    F = 0x46
    G = 0x47
    H = 0x48
    I = 0x49
    J = 0x4A
    K = 0x4B
    L = 0x4C
    M = 0x4D
    N = 0x4E
    O = 0x4F
    P = 0x50
    Q = 0x51
    R = 0x52
    S = 0x53
    T = 0x54
    U = 0x55
    V = 0x56
    W = 0x57
    X = 0x58
    Y = 0x59
    Z = 0x5A
    F1 = 0x70
    F2 = 0x71
    F3 = 0x72
    F4 = 0x73
    F5 = 0x74
    F6 = 0x75
    F7 = 0x76
    F8 = 0x77
    F9 = 0x78
    F10 = 0x79
    F11 = 0x7A
    F12 = 0x7B
    NUMLOCK = 0x90
    SCROLLLOCK = 0x91
    OEM_1 = 0xBA

    # 小键盘
    NUMPAD0 = 0x60
    NUMPAD1 = 0x61
    NUMPAD2 = 0x62
    NUMPAD3 = 0x63
    NUMPAD4 = 0x64
    NUMPAD5 = 0x65
    NUMPAD6 = 0x66
    NUMPAD7 = 0x67
    NUMPAD8 = 0x68
    NUMPAD9 = 0x69
    NUMPADMULTIPLY = 0x6A
    NUMPADADD = 0x6B
    NUMPADSUBTRACT = 0x6D
    NUMPADDECIMAL = 0x6E
    NUMPADENTER = 0x0D  # 小键盘回车与主回车同码

    # 鼠标
    LBUTTON = 0x01
    RBUTTON = 0x02
    MBUTTON = 0x04


#: 鼠标左/右/中键 —— release_all 统一抬起
MOUSE_BUTTON_VKS = (VK.LBUTTON, VK.RBUTTON, VK.MBUTTON)


class KeyId(int):
    """与 VK 解耦的中间键位标识（**三级解耦的中间层**：Action → KeyId → VK）。

    它是 ``int`` 的子类，所以可以当数字用；但**不重复** VK 的全部成员 ——
    只保留两个哨兵（NONE / UNKNOWN）。其余键位统一用 VK。

    哨兵的意义：「用户没配这个键」可以表达为 ``KeyId.NONE``，
    而不是假装是某个真实键位 —— 这是原则 2 的三态思想在键位上的应用。
    """

    __slots__ = ()

    #: 显式「不发按键」
    NONE = 0x0000
    #: 「配了但认不出」
    UNKNOWN = 0x00FF

    @classmethod
    def from_name(cls, name: str) -> "KeyId":
        """从配置里的字符串解析。**非法名字直接抛**（响亮失败）。

        支持 ``"W"`` / ``"VK_W"`` / ``"LBUTTON"`` / ``"0x57"`` / ``"NONE"``。
        """
        raw = (name or "").strip()
        if not raw:
            raise KeyIdError("键位名为空")
        up = raw.upper().replace("-", "_").replace("VK_", "")

        if up in ("NONE", "NULL"):
            return cls(cls.NONE)
        if up in ("UNKNOWN", "?"):
            return cls(cls.UNKNOWN)
        if up.startswith("0X"):
            try:
                return cls(int(up, 16))
            except ValueError as exc:
                raise KeyIdError(f"非法的十六进制键位: {raw!r}") from exc

        # 查 VK 表（含 LBUTTON 等鼠标键名）
        if hasattr(VK, up):
            return cls(int(getattr(VK, up)))
        # 容忍手误：大小写/前缀差异
        for name_, member in VK.__members__.items():
            if name_.upper() == up:
                return cls(int(member))

        valid = ", ".join(sorted(VK.__members__))
        raise KeyIdError(
            f"未知键位名 {raw!r}。\n"
            f"支持 VK 名（W / E / F12 / LBUTTON / SHIFT…）、0x 前缀十六进制、或 NONE。\n"
            f"全部可用：{valid}"
        )

    @property
    def is_sentinel(self) -> bool:
        return int(self) in (KeyId.NONE, KeyId.UNKNOWN)

    def to_vk(self) -> int:
        """转成真实 VK 码。哨兵抛异常 —— 不能拿它去发按键。"""
        if self.is_sentinel:
            raise KeyIdError(f"键位 {self.display_name} 不是真实按键，无法注入")
        return int(self)

    @property
    def display_name(self) -> str:
        if int(self) == KeyId.NONE:
            return "NONE"
        if int(self) == KeyId.UNKNOWN:
            return "UNKNOWN"
        return _vk_name(int(self))

    def __repr__(self) -> str:
        return f"KeyId({self.display_name})"


class InputInjectionError(RuntimeError):
    """注入失败。最常见原因是权限不足（SendInput 受 UIPI 限制）。"""


class InvalidStateError(RuntimeError):
    """执行层处于**不可用状态**：backend 缺失 / 非 Windows / DLL 加载失败。

    与 `InputInjectionError` 分开：前者是「发这一次没发出去」，
    后者是「这台机器上根本没法发」。后者必须在**构造期**就炸 ——
    留到调用 `.send()` 时才炸，栈里全是下游业务代码，
    排查成本高一个数量级（原则 1）。
    """


class KeyIdError(ValueError):
    """键位名/键位码非法。**加载期就该炸**，不是运行期。"""


# --------------------------------------------------------------------------
# Win32 结构体
# --------------------------------------------------------------------------

_ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]


def _user32() -> ctypes.WinDLL:
    if not hasattr(_user32, "_lib"):
        _user32._lib = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
    return _user32._lib  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# 后端（可注入，便于测试）
# --------------------------------------------------------------------------


@dataclass
class SentEvent:
    """一次注入的记录（dry_run 与真实后端都会记，便于断言与排错）。"""

    kind: str            # "key" | "mouse"
    name: str            # 键名("E"/"SHIFT") 或鼠标动作名
    down: bool
    dx: int = 0
    dy: int = 0
    #: 键盘事件的数字 VK 码。鼠标事件为 0。
    #: `name` 给人看，`vk` 给机器 —— 判定"还按着哪些键"必须用数字。
    vk: int = 0


#: 鼠标事件标志 -> VK 码。**鼠标键必须折算成 VK**：``release_all()`` 是
#: 遍历 ``VK`` 逐个问 ``is_key_down`` 的，鼠标事件不折算就永远问不到 ——
#: ``NORMAL_ATTACK`` 按住的左键在干跑时"卡住了"却查不出来。
#: （真实后端走 ``GetAsyncKeyState(VK_LBUTTON)`` 本来就答得上，
#:   折算是为了让 DryRun 与真实后端口径一致。）
_MOUSE_FLAG_VK: dict[int, int] = {
    MOUSEEVENTF_LEFTDOWN: VK.LBUTTON, MOUSEEVENTF_LEFTUP: VK.LBUTTON,
    MOUSEEVENTF_RIGHTDOWN: VK.RBUTTON, MOUSEEVENTF_RIGHTUP: VK.RBUTTON,
    MOUSEEVENTF_MIDDLEDOWN: VK.MBUTTON, MOUSEEVENTF_MIDDLEUP: VK.MBUTTON,
}
_MOUSE_BUTTON_MASK: int = (MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP
                           | MOUSEEVENTF_RIGHTDOWN | MOUSEEVENTF_RIGHTUP
                           | MOUSEEVENTF_MIDDLEDOWN | MOUSEEVENTF_MIDDLEUP)


def decode_events(inputs: list[INPUT]) -> list[SentEvent]:
    """把 INPUT 结构解码成可读事件。

    DryRunBackend 与 InputSimulator 的注入历史共用这一份解码，
    避免"干跑看到的"和"真实记录的"是两种口径。
    """
    out: list[SentEvent] = []
    for ev in inputs:
        u = ev.union
        if ev.type == INPUT_KEYBOARD:
            vk = int(u.ki.wVk)
            # ★ 扫描码模式下 wVk 按规范必须是 0（有效字段是 wScan）。
            #   不还原的话下面记出来的 vk=0 会被 DryRunBackend 当成
            #   「move/wheel，不是按键」跳过 —— 干跑就完全看不到
            #   「Shift 按住了」，而那正是这套诊断要证明的事。
            if not vk and (u.ki.dwFlags & KEYEVENTF_SCANCODE):
                vk = _scancode_to_vk(int(u.ki.wScan))
            is_down = not (u.ki.dwFlags & KEYEVENTF_KEYUP)
            out.append(SentEvent("key", _vk_name(vk), is_down, vk=vk))
        else:
            f = u.mi.dwFlags
            down = bool(f & (MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_RIGHTDOWN |
                             MOUSEEVENTF_MIDDLEDOWN))
            out.append(
                SentEvent("mouse", _mouse_name(f), down, int(u.mi.dx),
                          int(u.mi.dy),
                          _MOUSE_FLAG_VK.get(f & _MOUSE_BUTTON_MASK, 0))
            )
    return out


class InputBackend(Protocol):
    """注入后端。真实实现走 SendInput；测试用 DryRun。"""

    def send(self, inputs: list[INPUT]) -> int: ...
    def is_key_down(self, vk: int) -> bool: ...


class Win32Backend:
    """真实后端：user32.SendInput。"""

    def send(self, inputs: list[INPUT]) -> int:
        if not inputs:
            return 0
        arr = (INPUT * len(inputs))(*inputs)
        return int(_user32().SendInput(len(inputs), arr, ctypes.sizeof(INPUT)))

    def is_key_down(self, vk: int) -> bool:
        # GetAsyncKeyState 的高位(0x8000)为 1 表示当前按下
        return bool(_user32().GetAsyncKeyState(vk) & 0x8000)


class DryRunBackend:
    """空跑后端：只记录不发送。用于单测、CI、以及首次干跑。"""

    def __init__(self) -> None:
        self.events: list[SentEvent] = []
        self._down: set[int] = set()

    def send(self, inputs: list[INPUT]) -> int:
        for ev in decode_events(inputs):
            # ★ 键盘**和鼠标**都要跟踪。release_all() 遍历 VK 逐个问
            #   is_key_down，鼠标事件折算成 VK 之后才答得上来 ——
            #   只跟键盘的话，NORMAL_ATTACK 按住的左键永远"查不到卡住"，
            #   干跑就看不出松键逻辑到底有没有生效。
            #   （vk == 0 的是 move / wheel，不是按键，跳过。）
            if ev.vk:
                if ev.down:
                    self._down.add(ev.vk)
                else:
                    self._down.discard(ev.vk)
            self.events.append(ev)
        return len(inputs)

    def is_key_down(self, vk: int) -> bool:
        return vk in self._down

    @property
    def pressed_keys(self) -> set[int]:
        return set(self._down)


# --------------------------------------------------------------------------
# 输入模拟器
# --------------------------------------------------------------------------


@dataclass
class InputSimulator:
    """SendInput 薄封装。

    传 ``DryRunBackend()`` 时只记录不发送 —— 单测与首次干跑用。

    ★ **不要写 ``InputSimulator(backend=None)``**：dataclass 的
    ``field(default_factory=...)`` 只在**不传该参数**时生效，显式传 None
    会覆盖默认工厂，拿到一个字面量 None，然后在下游 ``.send()`` 处炸出
    ``'NoneType' object has no attribute 'send'``。构造期就会拦下。
    """

    backend: InputBackend | None = field(default_factory=Win32Backend)
    raise_on_mismatch: bool = True
    #: 记录最近多少条事件（防止无限增长）
    history_size: int = 500
    #: 键盘注入模式（见 :data:`KEY_MODES`）。默认 :data:`KEY_MODE_VK`
    #: —— 保持历史行为不变；怀疑游戏按「真实/合成」区分输入时切到
    #: :data:`KEY_MODE_SCANCODE`。
    key_mode: str = KEY_MODE_VK

    _log: list[SentEvent] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._log = []
        # ★ 响亮失败：模式写错要在构造期就炸，不能等到第一次注入
        if self.key_mode not in KEY_MODES:
            raise InvalidStateError(
                f"未知键盘注入模式 {self.key_mode!r}，可选：{sorted(KEY_MODES)}"
            )
        # ★ 响亮失败：backend 不可用立刻抛，绝不拖到第一次 send() 才炸
        self.require_backend()

    def require_backend(self) -> InputBackend:
        """校验 backend 可用，不可用则抛 `InvalidStateError`。

        在三处调用：`__post_init__`（构造期）、`_send`、`release_all`。
        构造期是主防线；后两处防的是「构造完再手工赋 `sim.backend = None`」
        这类绕过 —— 那种情况下栈里若只有 `.send()`，排查成本高得多。
        """
        rep = probe_backend(self.backend)
        if not rep.ok:
            raise InvalidStateError(rep.explain())
        return self.backend  # type: ignore[return-value]

    def diagnose(self) -> BackendReport:
        """返回当前 backend 的诊断报告（不发送任何输入）。"""
        return probe_backend(self.backend)

    # -- 上下文 ----------------------------------------------------------

    def __enter__(self) -> "InputSimulator":
        return self

    def __exit__(self, *exc) -> None:
        self.release_all()

    @property
    def history(self) -> list[SentEvent]:
        return list(self._log)

    def clear_history(self) -> None:
        self._log.clear()

    # -- 键盘 ------------------------------------------------------------

    def key_down(self, vk: int | KeyId) -> None:
        self._send_key(vk, up=False)

    def key_up(self, vk: int | KeyId) -> None:
        self._send_key(vk, up=True)

    def key_press(self, vk: int | KeyId, hold_ms: int = 30) -> None:
        """按下 → 保持 → 抬起。hold_ms 让游戏能收到按下沿。"""
        self._send_key(vk, up=False)
        time.sleep(max(0, hold_ms) / 1000.0)
        self._send_key(vk, up=True)

    def key_combo(self, modifiers: list[int | KeyId], key: int | KeyId) -> None:
        """**原子**提交组合键：mod1↓ mod2↓ key↓ key↑ mod2↑ mod1↑。

        一次 SendInput 发完全部 2×(n+1) 条事件。分两次注入的话，
        游戏可能只收到主键（modifier 还没生效就松开了）。

        ★ ``key_mode=both`` 时每条逻辑事件会展开成 2 条物理事件
          （VK + 扫描码），所以实际注入数是 ``4×(n+1)``。
        """
        mods = [_vk_of(m) for m in modifiers]
        main = _vk_of(key)
        if not mods:
            self.key_press(main)
            return
        ins: list[INPUT] = []
        for m in mods:
            ins.extend(_mk_keys(m, up=False, mode=self.key_mode))
        ins.extend(_mk_keys(main, up=False, mode=self.key_mode))
        ins.extend(_mk_keys(main, up=True, mode=self.key_mode))
        for m in reversed(mods):          # 逆序抬起，与按下配对
            ins.extend(_mk_keys(m, up=True, mode=self.key_mode))
        self._send(ins, desc=f"combo({[m for m in mods]}+{main})")

    def _send_key(self, vk: int | KeyId, *, up: bool) -> None:
        v = _vk_of(vk)
        self._send(_mk_keys(v, up=up, mode=self.key_mode),
                   desc=f"key {v} {'up' if up else 'down'}")

    # -- 鼠标 ------------------------------------------------------------

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        """相对移动。转视角的主力。"""
        if dx == 0 and dy == 0:
            return
        self._send([_mk_mouse(dx=int(dx), dy=int(dy), flags=MOUSEEVENTF_MOVE)],
                   desc=f"mouse_rel({dx},{dy})")

    def mouse_move_to(self, x: int, y: int, *, virtual_desk: bool = False) -> None:
        """绝对移动到**虚拟桌面**坐标。归一化到 0..65535。

        传的是**物理像素**，这里负责换算 —— 换算错是最常见的点击偏移来源。
        """
        vw, vh = _virtual_screen_size() if virtual_desk else _primary_screen_size()
        nx = int(x * 65535 / max(1, vw - 1))
        ny = int(y * 65535 / max(1, vh - 1))
        flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
        if virtual_desk:
            flags |= MOUSEEVENTF_VIRTUALDESK
        self._send([_mk_mouse(dx=nx, dy=ny, flags=flags)],
                   desc=f"mouse_abs({x},{y})")

    def mouse_click(
        self, button: int = VK.LBUTTON, hold_ms: int = 25
    ) -> None:
        """一次 SendInput 提交 down+up（BetterGI 的做法：零延迟）。"""
        down, up = _mouse_button_flags(button)
        self._send(
            [_mk_mouse(flags=down), _mk_mouse(flags=up)],
            desc=f"click({_vk_name(int(button))})",
        )
        if hold_ms:
            time.sleep(hold_ms / 1000.0)

    def mouse_tap(self, button: int = VK.LBUTTON, hold_ms: int = 120) -> None:
        """**真实保持**的点按：按下 -> 真的按住 ``hold_ms`` -> 抬起。

        ★ 和 :meth:`mouse_click` 的区别，**别混用**
          ``mouse_click`` 把 down+up 放进**同一次** SendInput 原子提交，
          游戏/内核看到的是**零时长**点按；它的 ``hold_ms`` 是**发完
          之后**的等待，不构成按键保持时长。BetterGI 用它做原神普攻
          是可行的，但对「游戏轮询刚好错过 down 沿」这类边界没有余量。

          ``mouse_tap`` 的 down 与 up 之间有**真实间隔**，对任何轮询式
          输入都成立 —— 原神 PC 正是轮询式（两个来源都把「按住左键」
          明确描述为进入蓄力重击，见 traits.NORMAL_ATTACK 的注释）。

        ⚠️ ``hold_ms`` 是**真的 sleep**，会占用 tick 预算。调用方
           （executor）必须把它算进 ``max_duration_ms``。
        """
        down, up = _mouse_button_flags(button)
        self._send([_mk_mouse(flags=down)],
                   desc=f"mtap-down({_vk_name(int(button))})")
        if hold_ms > 0:
            time.sleep(hold_ms / 1000.0)
        self._send([_mk_mouse(flags=up)],
                   desc=f"mtap-up({_vk_name(int(button))})")

    def mouse_down(self, button: int = VK.LBUTTON) -> None:
        down, _ = _mouse_button_flags(button)
        self._send([_mk_mouse(flags=down)], desc=f"mdown({_vk_name(int(button))})")

    def mouse_up(self, button: int = VK.LBUTTON) -> None:
        _, up = _mouse_button_flags(button)
        self._send([_mk_mouse(flags=up)], desc=f"mup({_vk_name(int(button))})")

    def mouse_scroll(self, clicks: int) -> None:
        """滚轮。clicks 为正向上滚。"""
        if clicks == 0:
            return
        self._send(
            [_mk_mouse(mouse_data=int(clicks) * WHEEL_DELTA, flags=MOUSEEVENTF_WHEEL)],
            desc=f"scroll({clicks})",
        )

    # -- 兜底 ------------------------------------------------------------

    def release_all(self) -> list[int]:
        """**无条件**释放所有按下的键与鼠标键。返回被释放的键码。

        为什么必须做：任何异常/中断都可能留下"卡住的按键"——
        W 被按住不放 = 角色一直往前跑。BetterGI 在 `TaskRunner.End()`、
        `AutoFightTask.finally`、`AutoBossTask.finally` 都调这个
        `[实测]`，这是异常安全的关键。

        本方法**自身不抛异常**：它出现在 finally 里，再抛会盖掉真正的错误。
        但**绝不静默**：失败一律 `logger.error(..., exc_info=True)` 带完整
        栈打出来。曾经这里写的是「已忽略」，把 backend 为 None 这种
        构造期就能发现的问题拖到了排障末尾。
        """
        released: list[int] = []
        try:
            backend = self.require_backend()
        except InvalidStateError as exc:
            logger.error("release_all 跳过：backend 不可用，没有键可释放。%s", exc,
                         exc_info=True)
            return []

        try:
            for vk in list(VK):
                if not backend.is_key_down(int(vk)):
                    continue
                # ★ 鼠标键必须发**鼠标抬起**，不能发键盘抬起（§37）。
                #   原实现对所有 VK 一律 ``_mk_key(up=True)`` —— 对
                #   VK_LBUTTON(0x01) 来说那是「键盘 A 键松开」，
                #   左键**根本没被松开**。表现是 release_all 返回值里
                #   明明有 1，游戏的左键却还按着；而 NORMAL_ATTACK 正是
                #   靠 release_all 兜底的（长按语义），于是角色一直平A。
                #   MOUSE_BUTTON_VKS 早就定义了却没人用。
                if int(vk) in MOUSE_BUTTON_VKS:
                    _down_flag, up_flag = _mouse_button_flags(int(vk))
                    self._send([_mk_mouse(flags=up_flag)],
                               desc=f"release_all {vk.name}", check=False)
                else:
                    # ★ 走 _mk_keys 而不是 _mk_key —— release_all 必须
                    #   和按下时用**同一个模式**。SCANCODE 模式按下的键
                    #   用 VK 模式抬起，系统层不一定认（wVk 与 wScan 是
                    #   两条不同路径），键会**卡在游戏里**。
                    self._send(_mk_keys(int(vk), up=True, mode=self.key_mode),
                               desc=f"release_all {vk.name}", check=False)
                released.append(int(vk))
        except Exception:  # noqa: BLE001 - finally 里绝不外抛，但必须留痕
            logger.exception(
                "release_all 中途失败（已释放 %d 个键，其余可能仍卡住，需手动松开）",
                len(released),
            )
            return released

        if released:
            logger.info("release_all 释放了: %s", ", ".join(_vk_name(v) for v in released))
        return released

    def is_key_down(self, vk: int | KeyId) -> bool:
        return self.backend.is_key_down(_vk_of(vk))

    # -- 底层 ------------------------------------------------------------

    def _send(self, inputs: list[INPUT], *, desc: str, check: bool = True) -> None:
        backend = self.require_backend()      # ★ 不让 None 流到 backend.send()
        want = len(inputs)
        try:
            sent = backend.send(inputs)
        except OSError as exc:
            raise InputInjectionError(
                f"SendInput 调用失败（{desc}）: {exc}\n"
                f"常见原因：权限不足，或被安全软件/反作弊拦截。"
            ) from exc
        # ★ 无论成功失败都要记账 —— 注入记录是排障的唯一依据，
        #   只在成功时记会让「按了但游戏没反应」变成无证可查。
        self._record(inputs)
        if check and self.raise_on_mismatch and sent != want:
            # ★ 这就是不用 pyautogui 的核心理由
            raise InputInjectionError(
                f"SendInput 只注入了 {sent}/{want} 个事件（{desc}）。\n"
                f"几乎总是权限问题：\n"
                f"  · 以管理员身份运行\n"
                f"  · 关掉 360 / 火绒 等拦截输入的安全软件\n"
                f"  · 检查 UIPI：目标窗口若以更高权限运行，本进程无法向其注入"
            )
        if sent < want:
            logger.warning("SendInput 部分失败 %d/%d（%s）", sent, want, desc)

    def _record(self, inputs: list[INPUT]) -> None:
        """把注入事件写进环形历史，并按 ``history_size`` 截断。"""
        self._log.extend(decode_events(inputs))
        if self.history_size > 0 and len(self._log) > self.history_size:
            del self._log[: len(self._log) - self.history_size]


# --------------------------------------------------------------------------
# 构造辅助
# --------------------------------------------------------------------------


def _vk_of(vk: int | KeyId) -> int:
    if isinstance(vk, KeyId):
        return vk.to_vk()
    return int(vk)


# --------------------------------------------------------------------------
# VK <-> 扫描码
# --------------------------------------------------------------------------
#: 扫描码 -> VK 的**兜底表**（PS/2 set-1），仅在 ``MapVirtualKey`` 不可用时用
#: （非 Windows 平台 / 单测假后端）。表里没有的键一律当 0 处理。
#:
#: ★ 为什么只列这一小撮
#: ----------------------
#: 手抄一整张 set-1 全表极容易出错（同一个扫描码在不同键位表里含义不同、
#: 扩展键要 E0 前缀、还有 ``0x2A``/``0x36`` 双 Shift 这类特例），
#: 抄错一个键 = 静默按错键。所以这里**只列执行层真正用到的键**
#: （对照 ``executor.py`` 的 ACTION_KEYS：W/A/S/D/Shift/Space/E/Q/One~Four/
#: B/M/Escape/Enter/F12），其余全交给 API。
#: 另有单测拿真实 ``MapVirtualKey`` 逐个核对这张表，防止它悄悄过期。
_SCAN_TO_VK_FALLBACK: dict[int, int] = {
    0x11: 0x57,  # W
    0x1E: 0x41,  # A
    0x1F: 0x53,  # S
    0x20: 0x44,  # D
    0x2A: 0x10,  # LSHIFT -> 泛用 VK_SHIFT(0x10)
    0x39: 0x20,  # SPACE
    0x12: 0x45,  # E
    0x10: 0x51,  # Q
    0x02: 0x31,  # ONE
    0x03: 0x32,  # TWO
    0x04: 0x33,  # THREE
    0x05: 0x34,  # FOUR
    0x30: 0x42,  # B
    0x32: 0x4D,  # M
    0x01: 0x1B,  # ESCAPE
    0x1C: 0x0D,  # ENTER
    0x58: 0x7B,  # F12
}
#: VK -> 扫描码的兜底（由上面反向生成，不手抄第二份 —— 两份表必然发散）
_VK_TO_SCAN_FALLBACK: dict[int, int] = {}
for _sc, _v in _SCAN_TO_VK_FALLBACK.items():
    _VK_TO_SCAN_FALLBACK.setdefault(int(_v), _sc)

_scan_cache: dict[int, tuple[int, bool]] = {}
_scan_rev_cache: dict[int, int] = {}


def _vk_to_scancode(vk: int) -> tuple[int, bool]:
    """返回 ``(扫描码, 是否扩展键)``。

    ★ 扩展键标志**不能**复用 :data:`EXTENDED_KEYS`
    --------------------------------------
    那个集合是 **VK 侧**的启发式（BetterGI 留下的），描述的是
    「哪些 VK 要配 ``KEYEVENTF_EXTENDEDKEY`` 才能正常工作」。
    扫描码侧的 E0 前缀是**另一套**判定，且本机实测：

    ``MapVirtualKeyEx(VK_SHIFT=0x10, MAPVK_VK_TO_EXSCANCODE) = 0x0000002A``

    高字节是 ``0x00`` **不是** ``0xE0`` —— LSHIFT 确实没有 E0 前缀；
    方向键/Insert/Delete/小键盘加减同样实测为 ``0x00``（现代键盘这些键
    走独立的非 E0 扫描码）。若在这里误加 ``KEYEVENTF_EXTENDEDKEY``，
    等于告诉系统「这个键带 E0 前缀」，Shift 会直接**按不动**。

    所以这里一律以 ``MAPVK_VK_TO_EXSCANCODE`` 的高字节为准，
    API 不可用时退回「非扩展」。
    """
    v = int(vk)
    hit = _scan_cache.get(v)
    if hit is not None:
        return hit
    scan, ext = 0, False
    try:
        if IS_WINDOWS:
            # 一次调用同时拿到扫描码和 E0 标记（高字节 0xE0 = 扩展）
            raw = int(_user32().MapVirtualKeyExW(v, MAPVK_VK_TO_EXSCANCODE, 0))
            scan = raw & 0xFF
            ext = ((raw >> 24) & 0xFF) == 0xE0
            if scan == 0:
                scan = _VK_TO_SCAN_FALLBACK.get(v, 0)
    except Exception:  # noqa: BLE001 - 平台差异不该炸掉整条注入链
        scan = _VK_TO_SCAN_FALLBACK.get(v, 0)
    _scan_cache[v] = (scan, ext)
    return scan, ext


def _scancode_to_vk(scan: int) -> int:
    """扫描码 -> VK（反向）。用于 SCANCODE 模式下把 ``wVk=0`` 的事件
    **还原**成 VK —— 不还原的话 :func:`decode_events` 会把所有扫描码事件
    记成 ``vk=0``，而 :class:`DryRunBackend` 明确跳过 ``vk == 0``
    （那是 move/wheel 的约定），于是干跑**完全看不到按键被按下**，
    ``release_all`` 也永远问不到「还有哪个键卡着」。"""
    s = int(scan) & 0xFF
    hit = _scan_rev_cache.get(s)
    if hit is not None:
        return hit
    vk = 0
    try:
        if IS_WINDOWS:
            vk = int(_user32().MapVirtualKeyW(s, MAPVK_SCANCODE_TO_VK))
            if vk == 0:
                vk = _SCAN_TO_VK_FALLBACK.get(s, 0)
    except Exception:  # noqa: BLE001
        vk = _SCAN_TO_VK_FALLBACK.get(s, 0)
    _scan_rev_cache[s] = vk
    return vk


def _mk_key(vk: int, *, up: bool) -> INPUT:
    """**VK 模式**（历史行为，保持字节级不变）。"""
    flags = KEYEVENTF_KEYUP if up else 0
    if vk in EXTENDED_KEYS:
        flags |= KEYEVENTF_EXTENDEDKEY
    ev = INPUT(type=INPUT_KEYBOARD)
    ev.union.ki = _KEYBDINPUT(
        wVk=vk & 0xFFFF, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0
    )
    return ev


def _mk_scan_key(vk: int, *, up: bool) -> INPUT:
    """**扫描码模式**。

    ★ ``wVk`` **必须为 0**：带 ``KEYEVENTF_SCANCODE`` 时 ``wScan`` 才是
    有效字段，``wVk`` 只在**没有** SCANCODE 标志时才有意义。
    """
    scan, ext = _vk_to_scancode(vk)
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    if ext:
        flags |= KEYEVENTF_EXTENDEDKEY
    ev = INPUT(type=INPUT_KEYBOARD)
    ev.union.ki = _KEYBDINPUT(
        wVk=0, wScan=scan & 0xFF, dwFlags=flags, time=0, dwExtraInfo=0
    )
    return ev


def _mk_keys(vk: int, *, up: bool, mode: str = KEY_MODE_VK) -> list[INPUT]:
    """按 ``mode`` 构造一次按键事件（1 条，``both`` 模式下 2 条）。"""
    if mode == KEY_MODE_VK:
        return [_mk_key(vk, up=up)]
    if mode == KEY_MODE_SCANCODE:
        return [_mk_scan_key(vk, up=up)]
    if mode == KEY_MODE_BOTH:
        # 先 VK 后扫描码：两条都发，让游戏两条路径都收到
        return [_mk_key(vk, up=up), _mk_scan_key(vk, up=up)]
    raise InputInjectionError(
        f"未知键盘注入模式 {mode!r}，可选：{sorted(KEY_MODES)}"
    )



def _mk_mouse(
    dx: int = 0, dy: int = 0, mouse_data: int = 0, flags: int = MOUSEEVENTF_MOVE
) -> INPUT:
    ev = INPUT(type=INPUT_MOUSE)
    ev.union.mi = _MOUSEINPUT(
        dx=dx, dy=dy, mouseData=mouse_data, dwFlags=flags, time=0, dwExtraInfo=0
    )
    return ev


def _mouse_button_flags(button: int) -> tuple[int, int]:
    b = int(button)
    if b == VK.LBUTTON:
        return MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
    if b == VK.RBUTTON:
        return MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP
    if b == VK.MBUTTON:
        return MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP
    raise InputInjectionError(f"未知鼠标键: {button}")


def _mouse_name(flags: int) -> str:
    for f, n in (
        (MOUSEEVENTF_LEFTDOWN, "Ldown"), (MOUSEEVENTF_LEFTUP, "Lup"),
        (MOUSEEVENTF_RIGHTDOWN, "Rdown"), (MOUSEEVENTF_RIGHTUP, "Rup"),
        (MOUSEEVENTF_MIDDLEDOWN, "Mdown"), (MOUSEEVENTF_MIDDLEUP, "Mup"),
        (MOUSEEVENTF_WHEEL, "wheel"), (MOUSEEVENTF_MOVE, "move"),
    ):
        if flags & f:
            return n
    return "mouse"


_VK_LOOKUP: dict[int, str] = {int(v): v.name for v in VK}


def _vk_name(vk: int) -> str:
    return _VK_LOOKUP.get(int(vk), f"VK_0x{int(vk):02X}")


def _primary_screen_size() -> tuple[int, int]:
    try:
        return int(_user32().GetSystemMetrics(0)), int(_user32().GetSystemMetrics(1))
    except Exception:  # noqa: BLE001
        return 1920, 1080


def _virtual_screen_size() -> tuple[int, int]:
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
    try:
        w = int(_user32().GetSystemMetrics(SM_XVIRTUALSCREEN))
        h = int(_user32().GetSystemMetrics(SM_YVIRTUALSCREEN))
        if w > 0 and h > 0:
            return w, h
    except Exception:  # noqa: BLE001
        pass
    return _primary_screen_size()


#: Windows 平台守卫 —— 非 Windows 直接拒绝，避免 import 期就崩得莫名其妙
IS_WINDOWS = hasattr(ctypes, "windll")


def require_windows() -> None:
    if not IS_WINDOWS:
        raise InvalidStateError(
            "输入模拟只在 Windows 上可用（SendInput 是 Win32 API）。"
            "其他平台请实现自己的 InputBackend。"
        )


# --------------------------------------------------------------------------
# 诊断
# --------------------------------------------------------------------------

#: Windows 完整性级别 RID -> 名字
_INTEGRITY_LABELS = {
    0x0000: "UNPROTECTED",
    0x1000: "LOW",
    0x2000: "MEDIUM",
    0x3000: "HIGH",
    0x4000: "SYSTEM",
}


def process_integrity() -> tuple[int | None, str]:
    """探测**当前进程**的完整性级别。

    返回 ``(rid, 标签)``。这是判断 UIPI 的关键：SendInput 受 UIPI 限制，
    目标窗口权限高于本进程时注入会被静默丢弃（返回 0）。本进程是
    MEDIUM 而游戏以管理员运行时，必失败 —— 这时候你会看到
    ``rid=0x3000 (MEDIUM)``，答案就很明确了。
    """
    if not IS_WINDOWS:
        return None, "非 Windows，无法探测"
    try:
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        TOKEN_QUERY = 0x0008
        TOKEN_INTEGRITY_LEVEL = 25

        # ★ restype 必须显式声明。不声明时 ctypes 默认按 c_int 解释返回值，
        #   64 位进程下 GetCurrentProcess 的伪句柄（-1）会被截断成
        #   0xFFFFFFFF 之外的垃圾，OpenProcessToken 直接报
        #   winerror=6 (ERROR_INVALID_HANDLE)。
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
        ]
        advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ]

        token = wintypes.HANDLE()
        proc = kernel32.GetCurrentProcess()
        if not advapi32.OpenProcessToken(proc, TOKEN_QUERY, ctypes.byref(token)):
            err = ctypes.get_last_error()
            return None, f"OpenProcessToken 失败 (winerror={err})"

        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, None, 0,
                                     ctypes.byref(needed))
        if not needed.value:
            return None, "GetTokenInformation 未返回所需长度"

        buf = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, buf,
                                            needed.value,
                                            ctypes.byref(needed)):
            err = ctypes.get_last_error()
            return None, f"GetTokenInformation 失败 (winerror={err})"

        # ⚠ GetTokenInformation 返回的**不是裸 SID**，而是 TOKEN_MANDATORY_LABEL：
        #     struct { PSID_LABEL Label; DWORD LabelSize; /* 4B padding on x64 */ }
        #     SID 从「sizeof(void*) + sizeof(DWORD)」对齐后的偏移开始：
        #     x64 = 16，x86 = 8。直接把整个 buffer 当 SID 解析会读出
        #     revision=192 / subauth_count=156 这种垃圾 → 静默报 UNPROTECTED。
        raw = bytes(buf)
        sid_off = 16 if ctypes.sizeof(ctypes.c_void_p) == 8 else 8
        if len(raw) < sid_off + 8:
            return None, f"TOKEN_MANDATORY_LABEL 过短（{len(raw)} 字节）"

        # SID 布局：revision(1) subauth_count(1) identifier_authority(6) subauths(4*n)
        sid = raw[sid_off:]
        revision, count = sid[0], sid[1]
        if revision != 1 or count < 1:
            # 结构对不上就说对不上，绝不返回猜测值（原则 2：不乐观）
            return None, (
                f"SID 结构异常（revision={revision}, count={count}）—— "
                f"完整性级别读取不可信，不予猜测"
            )
        # RID 是**最后一个** SubAuthority
        offset = 8 + (count - 1) * 4
        rid = int.from_bytes(sid[offset:offset + 4], "little")
        return rid, _INTEGRITY_LABELS.get(rid, f"0x{rid:04X}")
    except Exception as exc:  # noqa: BLE001 - 诊断本身不该炸
        return None, f"探测异常: {type(exc).__name__}: {exc}"


@dataclass
class BackendReport:
    """backend 可用性诊断报告。给 `--diagnose` 和异常消息共用。"""

    ok: bool
    backend_type: str
    is_windows: bool = False
    user32_loaded: bool = False
    sendinput_callable: bool = False
    vk_count: int = 0
    integrity_rid: int | None = None
    integrity_label: str = ""
    problems: list[str] = field(default_factory=list)

    def explain(self) -> str:
        """人类可读的失败原因。**逐条给出怎么修**，不只说"失败了"。"""
        if self.ok:
            return "backend 可用"
        lines = ["SendInput 初始化失败："]
        for i, p in enumerate(self.problems, 1):
            lines.append(f"  {i}. {p}")
        lines.append(f"  --- 环境 ---")
        lines.append(f"    平台        : {'Windows' if self.is_windows else '非 Windows'}")
        lines.append(f"    backend     : {self.backend_type}")
        lines.append(f"    user32.dll  : {'已加载' if self.user32_loaded else '加载失败'}")
        lines.append(f"    SendInput   : {'可调用' if self.sendinput_callable else '不可调用'}")
        lines.append(f"    进程完整性  : {self.integrity_label or '未知'}")
        lines.append(f"    VK 表条目   : {self.vk_count}")
        lines.append("  --- 怎么办 ---")
        if not self.is_windows:
            lines.append("    · SendInput 是 Win32 API，换平台需自行实现 InputBackend")
        if not self.user32_loaded or not self.sendinput_callable:
            lines.append("    · user32.dll 不可用，确认系统未被精简/沙箱限制")
        if self.integrity_label in ("MEDIUM", "LOW"):
            lines.append("    · 以管理员身份重跑本脚本（UIPI：低权限进程无法向高权限窗口注入）")
        return "\n".join(lines)


def probe_backend(backend: object | None) -> BackendReport:
    """探测 backend 是否可用。**不发送任何输入**，只做只读检查。

    对 `DryRunBackend` 直接返回 ok —— 干跑本来就不需要 Win32。
    """
    rep = BackendReport(ok=False, backend_type=type(backend).__name__,
                        is_windows=IS_WINDOWS, vk_count=len(list(VK)))

    if backend is None:
        rep.problems.append(
            "backend 是 None。\n"
            "     ⚠ dataclass 的 field(default_factory=...) 只在**不传该参数**时生效；\n"
            "       显式写 backend=None 会直接覆盖掉默认工厂，拿到字面量 None。\n"
            "       正确写法：InputSimulator() 或 InputSimulator(backend=DryRunBackend())"
        )
        rid, label = process_integrity()
        rep.integrity_rid, rep.integrity_label = rid, label
        return rep

    if isinstance(backend, DryRunBackend):
        rep.ok = True
        return rep

    if not hasattr(backend, "send") or not hasattr(backend, "is_key_down"):
        rep.problems.append(
            f"backend 缺少 send/is_key_down 方法，不符合 InputBackend 协议："
            f"{backend!r}"
        )
        return rep

    rid, label = process_integrity()
    rep.integrity_rid, rep.integrity_label = rid, label

    try:
        lib = _user32()
        rep.user32_loaded = True
        rep.sendinput_callable = callable(getattr(lib, "SendInput", None))
    except Exception as exc:  # noqa: BLE001
        rep.problems.append(
            f"加载 user32.dll 失败：{type(exc).__name__}: {exc}"
        )

    if not rep.sendinput_callable and rep.user32_loaded:
        rep.problems.append("user32.dll 已加载但没有 SendInput 导出（系统异常）")
    if not rep.is_windows:
        rep.problems.append("当前不是 Windows，SendInput 不可用")

    rep.ok = not rep.problems
    return rep


__all__ = [
    "EXTENDED_KEYS",
    "IS_WINDOWS",
    "INPUT",
    "BackendReport",
    "DryRunBackend",
    "InputBackend",
    "InputInjectionError",
    "InputSimulator",
    "InvalidStateError",
    "KeyId",
    "KeyIdError",
    "MOUSEEVENTF_ABSOLUTE",
    "MOUSEEVENTF_MOVE",
    "MOUSEEVENTF_VIRTUALDESK",
    "SentEvent",
    "VK",
    "Win32Backend",
    "decode_events",
    "probe_backend",
    "process_integrity",
    "require_windows",
]
