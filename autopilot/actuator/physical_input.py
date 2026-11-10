"""物理输入监听：**用户的手永远优先于引擎的虚拟按键**。

★ 为什么要这个（2026-10-17 实机教训）
----------------------------------------
实机测试时用户手动按 E 放技能，被引擎的按键**冲突卡住** ——
引擎正在按 E（或者按住某个键没松），用户的物理 E 键按下去没反应，
或者两个 E 叠在一起，技能放不出来。

根因不是「引擎不该按键」，而是**没有人在听**：
引擎只知道「我要按 E」，不知道「用户正在按 E」。

物理输入的优先级必须**绝对高于**虚拟输入 ——
这是工具的底线：人随时能接管，工具随时能松手。

★ 机制
--------
一个**独立后台线程**以 50Hz 轮询 ``GetAsyncKeyState``：

* 独立线程 = 不占主循环预算（主循环 5Hz，一个 tick 要做感知+决策，
  塞进按键轮询只会拖慢它）
* 50Hz = 20ms 间隔，能抓住 100ms 量级的轻点（用户按 E 的典型时长）。
  再快也没意义了 —— 比人反应还快的轮询只是烧 CPU。
* ``GetAsyncKeyState`` 一次 ~1µs，10 个键合计 ~0.5ms/s，占用可忽略。

★ 边沿检测（只报「新按下」）
----------------------------
按住不放的键**只报一次**。否则长按 W 会在 20ms 一次的重触发下
把让权窗口无限续期，引擎永远不恢复。

★ 让权窗口
----------
检测到物理输入后，引擎**在用户还在操作时**一直不发按键。
不是暂停整个循环 —— 感知继续跑（停了眼睛，恢复时看到的是陈旧画面），
只是不动手。

🔴 让权是**动态**的，不是固定倒计时（2026-10-19 实机反馈）
--------------------------------------------------------
旧实现是「一按就让权 3 秒」。实测两个方向都不对：

* 用户点一下走位 -> 干等 3 秒才恢复接管，**割裂感极强**；
* 用户按住 W 走 5 秒 -> 3 秒窗口一过引擎就来抢方向，**抢不动**。

现在的规则是**跟着用户的手指走**（见 :class:`TakeoverState`）：

* 用户还**按着**键         -> 一直让权（他还在操作）；
* 用户**松手**了           -> 再宽限 :data:`TAKEOVER_IDLE_S` 秒（防手抖连击）；
* 宽限也过了               -> 立刻恢复接管。

也就是「用户最后一次物理动作之后 1 秒」引擎才接手 ——
点一下走位，1 秒后就回来；按住不放，引擎一直不抢。

🔴 它**分不出「用户按的」和「引擎注入的」（2026-10-18 实测）
------------------------------------------------------------
``GetAsyncKeyState`` 对 ``SendInput`` 注入的按键**同样返回按下**
（实测：注入后 30 ms 读高位 = 1）。所以本模块回调的每一个键，
**默认都可能是引擎自己刚按的**。

过滤在调用方做：``InputSimulator.is_own_echo(key_name)``，
由 ``core/loop.py::_on_physical_input`` 在做任何事**之前**调用。
**直接用本模块而不加那层过滤，等于让引擎把自己掐掉** ——
实测 ``sim.key_press(VK.HELP, hold_ms=30)``（USE_SKILL 走的就是它）
在加过滤前 100% 触发让权。

为什么不换成「能区分注入」的 Win32 API（低级键盘钩子的
``LLKHF_INJECTED``、或注册 raw input 时带 ``RIDEV_NOINJECTED``）：
那要把一个真机验证过的安全特性整体重写。注册表侧的做法
（知道**自己**什么时候按了什么）只加十几行，出错面小得多。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .simulator import VK, _user32

logger = logging.getLogger(__name__)

#: 被监听的物理键。
#:
#: ★ 选的是「用户操作游戏时最常按的那几个」：
#:   WASD（移动）Q/E（技能）SHIFT（冲刺）SPACE（跳跃）
#:   + 鼠标左右键（瞄准 / 交互）
#:   + 数字键 1-4（切人，2026-10-18 加）。
#:
#: ⚠️ **不监听 F12** —— 那是急停键，safety.py 自己管。
#:   两个监听器都看 F12 的话，急停会被当成「用户接管」，
#:   语义完全错位（急停是**停机**，让权只是**让 3 秒**）。
#:
#: ★ 为什么只加 ONE..FOUR、不加 NUMPAD0..3
#:   漏一次切人只是少一条规则命中；误记一次会让规则以为别人在场，
#:   代价不对称 —— 所以只监听游戏默认绑定的**主键盘**数字键。
#:
#: 🔴 监听器**分不出「用户按的」和「引擎注入的」** —— ``GetAsyncKeyState``
#:   两边都看得见（实测 SendInput 注入后 30 ms 读高位 = 1）。
#:   过滤在 :meth:`autopilot.actuator.simulator.InputSimulator.is_own_echo`
#:   那一侧做，由 ``core/loop.py::_on_physical_input`` 调用。
#:   本模块**保持 dumb**：只负责如实上报，不做判断。
WATCHED_KEYS: tuple[VK, ...] = (
    VK.W, VK.A, VK.S, VK.D,
    VK.Q, VK.E,
    VK.SHIFT, VK.SPACE,
    VK.ONE, VK.TWO, VK.THREE, VK.FOUR,
    VK.LBUTTON, VK.RBUTTON,
)

#: 轮询间隔（秒）。50Hz。
POLL_INTERVAL_S: float = 0.02

#: 让权**空闲宽限**（秒）：距用户最后一次物理动作不到这么久，引擎继续让权。
#:
#: ★ 为什么不再有「固定 3 秒」（2026-10-19 实机反馈）
#:   固定窗口的问题不是 3 秒太长，是**它不看用户在干什么**：
#:   点一下走位就干等 3 秒；而按住 W 走 5 秒时窗口一过引擎就来抢方向。
#:
#:   现在的规则跟着手指走：按着就一直让权；松手后再宽限这么久。
TAKEOVER_IDLE_S: float = 1.0


class TakeoverState:
    """让权状态：**动态**跟随用户的实际操作。

    ★ 它是 :mod:`autopilot.actuator.safety` 与
      :mod:`autopilot.core.loop` 共用的**唯一**让权判据 ——
      两边各写一套「窗口还剩多少」的逻辑，迟早会在边界上打架
      （一个说在让权、另一个说不在，然后引擎在用户按键时插了一脚）。

    ★ 判定规则（两条，满足其一即让权）
      1. 用户**还按着**某个被监听的键 —— 他还在操作，引擎不能抢。
      2. 距用户**最后一次物理动作**（按下或松开）不到 :attr:`idle_s` 秒。

    ⚠️ :meth:`note_press` **不**把键加进「按着」集合。
      「按下的那一瞬间键确实是按着的」是**调用方的知识** ——
      调用方（:mod:`autopilot.core.loop`）已经先过滤过引擎自己的回声，
      只有真人按的键才算数。这里分开是为了让
      「记一次输入」和「这个键现在被按着」两件事**不互相冒充**。
    """

    def __init__(self, idle_s: float = TAKEOVER_IDLE_S) -> None:
        #: 空闲宽限（秒）。
        self.idle_s: float = max(0.0, float(idle_s))
        self._active: bool = False
        #: 最后一次物理动作（按下或松开）的 monotonic 时刻。
        self._last_input: float = 0.0
        #: 用户**此刻**还按着的键名。
        self._held: set[str] = set()
        self._last_key: str = ""

    # -- 事件 ------------------------------------------------------------
    def note_press(self, key_name: str) -> None:
        """用户按下了某个键。"""
        self._active = True
        self._last_input = time.monotonic()
        self._last_key = key_name

    def note_hold(self, key_name: str) -> None:
        """这个键**现在被用户按着**。"""
        self._held.add(key_name)

    def note_release(self, key_name: str) -> None:
        """用户松开了某个键。

        ★ 松开**也刷新**空闲时钟：手指刚离开键盘的那一刻是最不该抢的
          —— 引擎如果在这一瞬间把鼠标/方向键夺走，用户会觉得「被抢了」。
        """
        self._held.discard(key_name)
        self._last_input = time.monotonic()

    def clear(self) -> None:
        """手动清掉让权（用户主动恢复时用）。"""
        self._active = False
        self._last_input = 0.0
        self._held.clear()
        self._last_key = ""

    # -- 读侧 ------------------------------------------------------------
    @property
    def active(self) -> bool:
        """现在是不是在让权。**这是唯一判据**。"""
        if not self._active:
            return False
        if self._held:
            # 用户还按着键 —— 他还在操作，多久都不抢。
            return True
        if time.monotonic() - self._last_input >= self.idle_s:
            # 空闲宽限用完 -> 恢复接管（顺手把状态清干净）。
            self._active = False
            self._held.clear()
            return False
        return True

    @property
    def held(self) -> frozenset[str]:
        """用户此刻按着的键。**只读**。"""
        return frozenset(self._held)

    @property
    def last_key(self) -> str:
        """最后一次物理动作涉及的键。"""
        return self._last_key

    @property
    def idle_remaining_s(self) -> float:
        """空闲宽限还剩多少秒（0 = 不在让权）。"""
        if not self._active or self._held:
            return 0.0
        return max(0.0, self.idle_s - (time.monotonic() - self._last_input))


class PhysicalInputWatcher:
    """后台线程轮询物理按键，检测到**新按下**就回调。

    用法::

        watcher = PhysicalInputWatcher(on_input=lambda key: ...)
        watcher.start()
        ...
        watcher.stop()

    ★ 回调在**监听线程**里被调用 —— 它必须快、且不能抛。
      真正要做的重活（松键、记事件）交给调用方在自己的线程里做。

    ★ ``on_release``（可选）回调的是**松开**的边沿。
      没有它就只知道「用户按过」，不知道「用户还在按」——
      而「还在按」正是动态让权的第一条规则。
    """

    def __init__(
        self,
        on_input: Callable[[str], None],
        *,
        on_release: Callable[[str], None] | None = None,
        keys: tuple[VK, ...] = WATCHED_KEYS,
        poll_s: float = POLL_INTERVAL_S,
    ) -> None:
        self._on_input = on_input
        self._on_release = on_release
        self._keys = tuple(keys)
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: 当前处于「按下」状态的键。用于**边沿检测**。
        self._down: set[VK] = set()
        #: ★ 用户**此刻正按着**的物理键（VK 码），2026-10-19 加。
        #:
        #: ★ 为什么不能直接用 ``_down``
        #:   ``_down`` 是**边沿检测**用的状态（配合「只报新按下」规则），
        #:   而「用户正按着的键」是**另一件事**：它要回答
        #:   「现在松手会不会打断用户」。
        #:
        #:   ★ 它为什么是**修 Bug 的关键**（2026-10-19）
        #:     ``release_all()`` 遍历所有按下的键无条件发 KeyUp，
        #:     分不出「引擎按的」和「用户按的」—— 于是用户按住 E 长按技能时，
        #:     一次让权就把 E 弹起来，用户的长按**当场断掉**
        #:     （表现：「长按出不来」「按 A/W 走不动」）。
        #:     有了这个集合，执行层就能**跳过用户正按着的键**。
        #:
        #: ⚠️ 它**分不出**「用户按的」和「引擎注入的」——
        #:   ``GetAsyncKeyState`` 两边都看得见（实测 SendInput 注入后
        #:   30 ms 读高位 = 1）。所以它**不能单独**当「用户的键」用：
        #:   ``InputSimulator`` 会把它减去「引擎自己按着的键」
        #:   （``_engine_down``）之后才拿来保护。
        self._held_physical_keys: set[VK] = set()
        self._lock = threading.Lock()

    # -- 生命周期 --------------------------------------------------------
    def start(self) -> None:
        """启动监听线程。**幂等**。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="physical-input", daemon=True)
        self._thread.start()
        logger.info("物理输入监听已启动（%d 个键，%.0fHz）",
                    len(self._keys), 1.0 / self._poll_s if self._poll_s else 0)

    def stop(self) -> None:
        """停止监听线程。**幂等**，不抛。"""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- 轮询 ------------------------------------------------------------
    def _run(self) -> None:
        try:
            get_async_key_state = _user32().GetAsyncKeyState
        except Exception:  # noqa: BLE001 - 非 Windows / 拿不到 user32
            logger.warning("拿不到 user32，物理输入监听不可用", exc_info=True)
            return

        # ★ 启动时先记下当前按住的状态
        #   否则「引擎刚起，用户正好按着 W」会被当成一次新按下，
        #   白白触发一次让权。
        self._down = {vk for vk in self._keys
                      if self._pressed(get_async_key_state, vk)}
        # ★ 「用户正按着的键」同样要按启动时的实际状态初始化 ——
        #   引擎起来时用户已经按着 W，那 W 就**现在**正被按着，
        #   后续的松键必须绕开它。
        self._held_physical_keys = set(self._down)

        while not self._stop.is_set():
            try:
                self._poll_once(get_async_key_state)
            except Exception:  # noqa: BLE001 - 监听线程绝不能死
                logger.debug("物理输入轮询异常", exc_info=True)
            self._stop.wait(self._poll_s)

    def _poll_once(self, get_async_key_state) -> None:
        for vk in self._keys:
            pressed = self._pressed(get_async_key_state, vk)
            with self._lock:
                was_down = vk in self._down
                if pressed:
                    self._down.add(vk)
                else:
                    self._down.discard(vk)
                # ★ 「用户正按着的键」：**按下时加入，松开时移除**。
                #   这是「让权时不碰用户的键」的唯一数据来源。
                if pressed:
                    self._held_physical_keys.add(vk)
                else:
                    self._held_physical_keys.discard(vk)
            # ★ 双向边沿：按下和松开都要报。
            #   松开这一条是动态让权的第一条规则（「用户还按着就别抢」）
            #   的数据来源 —— 只报按下的话，引擎永远不知道用户松手了。
            if pressed and not was_down:
                self._fire(vk)
            elif was_down and not pressed:
                self._fire_release(vk)

    @staticmethod
    def _pressed(get_async_key_state, vk: VK) -> bool:
        """这个键现在按着吗。拿不到就当没按（不误触发）。"""
        try:
            return bool(get_async_key_state(int(vk)) & 0x8000)
        except Exception:  # noqa: BLE001
            return False

    def _fire(self, vk: VK) -> None:
        name = vk.name
        logger.info("检测到物理输入：%s", name)
        try:
            self._on_input(name)
        except Exception:  # noqa: BLE001 - 回调抛了也不能让监听线程死
            logger.exception("物理输入回调抛异常（已忽略）")

    def _fire_release(self, vk: VK) -> None:
        """报一次**松开**。没注册 ``on_release`` 就什么都不做。"""
        if self._on_release is None:
            return
        name = vk.name
        logger.debug("检测到物理松开：%s", name)
        try:
            self._on_release(name)
        except Exception:  # noqa: BLE001 - 回调抛了也不能让监听线程死
            logger.exception("物理松开回调抛异常（已忽略）")

    # -- 测试用 ----------------------------------------------------------
    def snapshot_down(self) -> set[VK]:
        """当前按住（按下边缘之外）的键集合。**只读**，给测试用。"""
        with self._lock:
            return set(self._down)

    def held_physical_keys(self) -> frozenset[VK]:
        """用户**此刻正按着**的物理键（VK 码）。**只读**。

        ★ 这是「让权时**不能**松开用户的键」的数据来源
          （2026-10-19）。接法::

              sim.set_physical_hold_provider(watcher.held_physical_keys)

          之后 ``sim.release_all(exclude_physical=True)`` 就会
          跳过这些键 —— 用户按住 E 长按技能时，不再被合成 KeyUp 掐断。

        ⚠️ 它**分不出**「用户按的」和「引擎注入的」
          （``GetAsyncKeyState`` 两边都看得见）。调用方
          （:class:`autopilot.actuator.simulator.InputSimulator`）
          负责把「引擎自己按着的键」减掉。
        """
        with self._lock:
            return frozenset(self._held_physical_keys)


__all__ = [
    "WATCHED_KEYS", "POLL_INTERVAL_S", "TAKEOVER_IDLE_S",
    "PhysicalInputWatcher", "TakeoverState",
]
