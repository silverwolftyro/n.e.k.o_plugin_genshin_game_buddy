"""轴名统一 / min_gap=300 / MOVE_TO 意图（2026-10-14）。

★★ 这一轮的三件事互相关联，测试也按这个关系组织
--------------------------------------------------
1. **轴名统一 ``x`` / ``y``**：``PositionInfo`` 一直就是 ``x`` / ``y``
   （其中 ``y`` 存的是**世界 z**），但寻路脚本、事件 context、部分测试
   按 ``x`` / ``z`` 写。现已内部统一，**外部文件格式不变**
   （Console 路点文件与 ``e2e_pathfind.py --start-world X,Z`` 仍是 z）。
2. **``NORMAL_ATTACK.min_gap_ms = 300``**：从 0 改成 300。真机实测过
   4.4 次/秒是因为 0 形同虚设（唯一在拦的 ``rate_limit_ms=50``
   本来就小于 200ms/tick）。
3. **``MOVE_TO``**：全枚举里第一个带目标的导航意图。

⚠️ 值得单独记一笔的坑：``PositionInfo.heading_deg`` **恒为 None**
（朝向提取已被证伪，见 ``perception/position_mapping.py``）。所以
``MOVE_TO`` 只能靠执行器自己攒位移窗口估朝向 —— 这正是下面
``TestNavWindow`` 存在的原因，也是 ``MOVE_TO`` 最大的实现成本所在。
"""

from __future__ import annotations

import time

import pytest

from autopilot.actuator.executor import IntentExecutor, Outcome, _NavWindow
from autopilot.actuator.safety import SafetyConfig, SafetyGuard
from autopilot.actuator.simulator import VK
from autopilot.actuator.traits import TRAITS, traits_of
from autopilot.nav.steer import DEFAULT_CONFIG as nav_steer_cfg
from autopilot.contracts.events import Event
from autopilot.contracts.intent_packet import (
    ActionType,
    IntentPacket,
    IntentTarget,
)
from autopilot.contracts.state_packet import (
    SCENE_UNKNOWN,
    SCENE_WORLD,
    PositionInfo,
    ResourceInfo,
    SceneInfo,
    StatePacket,
)

pytestmark = pytest.mark.unit

#: 用**真实墙钟**，不用固定常量 —— ``IntentPacket.make`` 的
#: ``issued_at_ms`` 取的就是墙钟，拿一个 2023 年的常量当 now 会让
#: 每条意图都 ``INTENT_EXPIRED``（而那条日志长得像引擎坏了）。
NOW = int(time.time() * 1000)

#: TTL 给足：本组测试要跨多个 tick 推进同一条 MOVE_TO，
#: 默认的 10s 会让第 N 帧的意图被判过期。
#: ⚠️ 300 是校验器允许的**上限**（超了报 TTL_TOO_LONG），别再往上加。
TTL = 300

#: 仿真里「游戏」对鼠标位移的响应系数，与 ``nav.steer`` 默认配置同源。
#: 从 steer 模块取，避免两处各写一个数字悄悄漂掉。
_MOUSE_DEG_PER_PX = nav_steer_cfg.mouse_deg_per_px


# ==================================================================
# 测试脚手架
# ==================================================================

class _Recorder:
    """记下所有注入的输入。**不发任何真实键鼠**。

    ⚠️ 同时维护 ``held``（**当前仍按着**的键）。只数 ``down`` 事件是
    不够的：MOVE_TO 靠 ``_held`` 把 W 跨 tick 按住，第二 tick 起就
    不再发新的 ``key_down`` —— 仿真若只看 down 事件，会以为角色停了。
    """

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.held: set[int] = set()

    def require_backend(self) -> None:
        return None

    def key_down(self, vk) -> None:
        self.events.append(("down", int(vk)))
        self.held.add(int(vk))

    def key_up(self, vk) -> None:
        self.events.append(("up", int(vk)))
        self.held.discard(int(vk))

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        self.events.append(("rel", int(dx), int(dy)))

    def release_all(self) -> list[int]:
        self.events.append(("release_all",))
        n = sorted(self.held)
        self.held.clear()
        return n

    def key_press(self, vk, hold_ms: int = 30) -> None:
        self.events.append(("press", int(vk)))

    def mouse_tap(self, btn, hold_ms: int = 120) -> None:
        self.events.append(("tap", int(btn)))

    def mouse_move_to(self, x, y, **kw) -> None:
        self.events.append(("to", int(x), int(y)))

    def downs(self, vk) -> int:
        return sum(1 for e in self.events if e[0] == "down" and e[1] == int(vk))

    def ups(self, vk) -> int:
        return sum(1 for e in self.events if e[0] == "up" and e[1] == int(vk))

    def is_held(self, vk) -> bool:
        return int(vk) in self.held


def _executor() -> tuple[IntentExecutor, _Recorder]:
    """一个**永不碰真实键鼠**的执行器（DryRun 起手，再换成记录器）。"""
    guard = SafetyGuard(SafetyConfig(
        require_foreground=False,
        panic_key_enabled=False,      # 不去问 GetAsyncKeyState
        max_actions=10_000,
    ))
    ex = IntentExecutor(dry_run=True, guard=guard)
    rec = _Recorder()
    ex.sim = rec                 # type: ignore[assignment]
    return ex, rec


def _state(x: float | None = 0.0, y: float | None = 0.0,
           sv: int = 1) -> StatePacket:
    """一份**正规构造**的 StatePacket。

    ``x`` / ``y`` 即平面两轴（``y`` 存的是世界 z）。
    ``x`` 或 ``y`` 为 None -> 定位丢失（``quality="LOST"``）。

    ⚠️ 不要用 ``StatePacket.__new__`` 造 —— 安全闸与校验器会读
    ``scene`` / ``resource``，缺字段会 AttributeError，
    那看起来像被测代码坏了，其实只是脚手架坏了。
    """
    pos = (PositionInfo(quality="GOOD", x=x, y=y, source="test")
           if x is not None and y is not None else PositionInfo.lost())
    return StatePacket(
        scene=SceneInfo(id=SCENE_WORLD, confidence=0.9,
                        entered_ms=NOW - 5000),
        position=pos,
        resource=ResourceInfo(perception_ok=True),
        timestamp_ms=NOW,
        state_version=sv,
    )


def _attack(sv: int = 1) -> IntentPacket:
    return IntentPacket.make(ActionType.NORMAL_ATTACK, state_version=sv,
                             ttl_seconds=TTL)


def _move_to(x: float, y: float, radius: float = 1.5,
             sv: int = 1) -> IntentPacket:
    """一条 MOVE_TO。**每次调用都是新的 intent_id** ——
    ``SeenIntentStore`` 会把重复 intent_id 当网络重投拒掉，
    而同一条 MOVE_TO 跨 tick 重复下发本来就是正常形态。"""
    return IntentPacket.make(
        ActionType.MOVE_TO, state_version=sv, ttl_seconds=TTL,
        target=IntentTarget(world_position={"x": x, "y": y,
                                            "radius": radius}),
    )


# ==================================================================
# 任务 A · 轴名统一
# ==================================================================

class TestAxisUnification:
    def test_xy_is_the_canonical_key(self) -> None:
        assert IntentTarget(world_position={"x": 1.0, "y": 2.0}).world_xy() \
            == (1.0, 2.0)

    def test_legacy_z_is_still_accepted(self) -> None:
        """★ 兼容层：历史事件流与手写 JSON 里是 ``z``。

        读到 None 会把「键名写错」变成「坐标缺失」，两者在日志里
        长得一模一样，排查成本极高。
        """
        assert IntentTarget(world_position={"x": 1.0, "z": 2.0}).world_xy() \
            == (1.0, 2.0)

    def test_y_wins_when_both_present(self) -> None:
        assert IntentTarget(
            world_position={"x": 1.0, "y": 20.0, "z": 99.0}
        ).world_xy() == (1.0, 20.0)

    def test_missing_component_is_none_not_a_crash(self) -> None:
        assert IntentTarget(world_position={"x": 1.0}).world_xy() is None
        assert IntentTarget(world_position={"y": 2.0}).world_xy() is None
        assert IntentTarget(world_position={}).world_xy() is None
        assert IntentTarget(world_position=None).world_xy() is None

    def test_non_numeric_is_none(self) -> None:
        assert IntentTarget(
            world_position={"x": "abc", "y": 2.0}).world_xy() is None

    def test_non_finite_is_none(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            got = IntentTarget(
                world_position={"x": bad, "y": 1.0}).world_xy()
            assert got is None, bad

    def test_arrive_radius_reads_and_defaults(self) -> None:
        assert IntentTarget(
            world_position={"x": 1.0, "y": 2.0, "radius": 3.5}
        ).arrive_radius(9.0) == 3.5
        assert IntentTarget(world_position={"x": 1.0, "y": 2.0}
                            ).arrive_radius(9.0) == 9.0

    def test_arrive_radius_rejects_garbage(self) -> None:
        for bad in ("x", None, -1.0, float("nan")):
            t = IntentTarget(world_position={"x": 1.0, "y": 2.0,
                                             "radius": bad})
            assert t.arrive_radius(9.0) == 9.0, bad


class TestEventAxisCompat:
    """事件 context 是**落盘的对外契约**，所以 ``z`` 要留镜像。"""

    def test_position_changed_carries_both_keys(self) -> None:
        c = Event.position_changed(1.0, 2.0, 3.0).context
        assert c["x"] == 1.0
        assert c["y"] == 2.0
        assert c["z"] == 2.0            # 兼容镜像，与 y 同值

    def test_pathfind_start_mirrors_target(self) -> None:
        c = Event.pathfind_start(10.0, 20.0, 5.0).context
        assert c["target_x"] == 10.0
        assert c["target_y"] == 20.0
        assert c["target_z"] == 20.0

    def test_pathfind_arrived_carries_both_keys(self) -> None:
        c = Event.pathfind_arrived(4.0, 5.0, 12.0).context
        assert c["x"] == 4.0
        assert c["y"] == 5.0
        assert c["z"] == 5.0
        assert c["spent_s"] == 12.0

    def test_mirror_is_never_inconsistent(self) -> None:
        """镜像写错值比没有镜像更糟 —— 读侧会信它。"""
        for ev in (Event.position_changed(7.0, 8.0, 1.0),
                   Event.pathfind_arrived(7.0, 8.0)):
            assert ev.context["y"] == ev.context["z"]


# ==================================================================
# 任务 B · NORMAL_ATTACK.min_gap_ms = 300
# ==================================================================

class TestNormalAttackMinGap:
    def test_trait_is_300(self) -> None:
        assert traits_of("NORMAL_ATTACK").min_gap_ms == 300

    def test_second_attack_inside_300ms_is_rejected(self) -> None:
        g = SafetyGuard(SafetyConfig(require_foreground=False,
                                     panic_key_enabled=False))
        st = _state()
        ok1, _ = g.check(_attack(), st, NOW)
        assert ok1 is True
        g.mark_executed("NORMAL_ATTACK", NOW)

        ok2, reason = g.check(_attack(), st, NOW + 200)
        assert ok2 is False
        assert reason.startswith("MIN_GAP:"), reason

    def test_attack_after_300ms_is_allowed(self) -> None:
        g = SafetyGuard(SafetyConfig(require_foreground=False,
                                     panic_key_enabled=False))
        st = _state()
        g.check(_attack(), st, NOW)
        g.mark_executed("NORMAL_ATTACK", NOW)
        ok, reason = g.check(_attack(), st, NOW + 301)
        assert ok is True, reason

    def test_main_loop_tick_rate_is_now_blocked(self) -> None:
        """★★ 这就是本次改动的**核心意图**。

        5Hz 主循环 = 200ms/tick。原来 ``min_gap_ms=0`` 时每 tick 都打
        一次（真机实测 4.4 次/秒），因为唯一在拦的 ``rate_limit_ms=50``
        比 200ms 还小，形同虚设。现在 200ms < 300ms 必然被拒。
        """
        g = SafetyGuard(SafetyConfig(require_foreground=False,
                                     panic_key_enabled=False))
        st = _state()
        allowed = 0
        for i in range(10):                       # 10 tick = 2 秒
            ok, _ = g.check(_attack(), st, NOW + i * 200)
            if ok:
                allowed += 1
                g.mark_executed("NORMAL_ATTACK", NOW + i * 200)
                g.mark_success()
        # 逐 tick 推演（200ms 步长，min_gap 300ms）：
        #   t=0 放行 -> 200 拒 -> 400 放行 -> 600 拒 -> 800 放行
        #   -> 1000 拒 -> 1200 放行 -> 1400 拒 -> 1600 放行 -> 1800 拒
        assert allowed == 5, f"2 秒内放行了 {allowed} 次"


# ==================================================================
# 任务 C · MOVE_TO
# ==================================================================

class TestMoveToRegistration:
    def test_enum_and_kind(self) -> None:
        assert ActionType.MOVE_TO.value == "MOVE_TO"
        assert ActionType.MOVE_TO.kind == "NAV"

    def test_traits_are_nav_shaped(self) -> None:
        t = traits_of("MOVE_TO")
        assert t.min_gap_ms == 0, "每 tick 推进一次是正常的，限频会让走到天亮"
        assert t.max_duration_ms >= 60_000
        assert t.holds_keys is True
        assert t.abort_on_target_lost is True

    def test_allowed_scenes_is_not_world_only(self) -> None:
        """★ 只写 SCENE_WORLD 会让这条意图**一条都放不出来**。

        ``traits.py`` 上方记载：场景判定降级为二值
        （WORLD / UNKNOWN），真机户外行走常常是 UNKNOWN。
        症状是「规则命中了但没有动作」，极难联想。

        这里断言的是**通配**而不是枚举：``SCENE_ANY``（``"*"``）本身就
        覆盖 UNKNOWN，所以断言 ``SCENE_UNKNOWN in scenes`` 是错的 ——
        那是「逐个列举」才成立的形式。
        """
        from autopilot.actuator.traits import SCENE_ANY
        scenes = TRAITS["MOVE_TO"].allowed_scenes
        assert scenes == (SCENE_ANY,), scenes


class TestMoveToExecution:
    def test_arrives_when_inside_radius(self) -> None:
        ex, rec = _executor()
        r = ex.execute(_move_to(0.5, 0.0, radius=1.5), _state(0.0, 0.0), NOW)
        assert r.outcome is Outcome.OK
        assert any("到达" in s for s in r.steps), r.steps
        assert rec.downs(VK.W) == 0, "已到达就不该再按 W"

    def test_missing_world_position_fails_loudly(self) -> None:
        """★ 响亮失败，不猜一个"大概是那边"的点走下去。"""
        ex, rec = _executor()
        it = IntentPacket.make(ActionType.MOVE_TO, state_version=1,
                               ttl_seconds=TTL)      # 故意不给坐标
        r = ex.execute(it, _state(0.0, 0.0), NOW)
        assert r.outcome is Outcome.OK          # 没抛异常
        assert any("缺少可用的 x/y" in s for s in r.steps), r.steps
        assert rec.downs(VK.W) == 0, "缺坐标时不该按任何键"
        assert not [e for e in rec.events if e[0] == "rel"], \
            "缺坐标时不该转视角"

    def test_lost_position_is_blocked_before_the_executor(self) -> None:
        """★ 定位丢失时，**校验器**就先拦下了 —— 执行器根本轮不到。

        这是既有 ``validator._check_position`` 的功劳（``IntentPacket``
        的 validate 第 5 步也查同一处）。执行器里的那份检查是第二道
        防线：正常路径走不到，但它保证「定位在中途消失」时同样不会
        带着脏状态走下去。
        """
        ex, rec = _executor()
        r = ex.execute(_move_to(50.0, 50.0), _state(None, None), NOW)
        assert r.outcome is Outcome.REJECTED
        assert r.reason.startswith("POSITION_UNAVAILABLE"), r.reason
        # ⚠️ 不能断言 ``rec.events == []``：非 OK 的收尾本来就会调
        #   ``release_all()``（那是对的）。要断言的是**没有按下任何键**。
        assert rec.downs(VK.W) == 0, "定位丢了却还在按 W"
        assert not [e for e in rec.events if e[0] == "rel"], \
            "定位丢了却还在转视角"

    def test_blind_walk_when_heading_unknown(self) -> None:
        """窗口还没有足够位移时**照走**，但如实标记为盲走。

        单点位置给不出朝向；这时候「不动」等于卡死。
        """
        ex, rec = _executor()
        r = ex.execute(_move_to(0.0, 50.0), _state(0.0, 0.0), NOW)
        assert r.outcome is Outcome.OK
        assert rec.downs(VK.W) == 1

    def test_turns_once_heading_is_known(self) -> None:
        ex, rec = _executor()
        for i, x in enumerate((0.0, 1.0, 2.0)):
            ex.execute(_move_to(0.0, 50.0), _state(x, 0.0), NOW + i * 200)
        rel = [e for e in rec.events if e[0] == "rel"]
        assert rel, "目标在正左方却一个都没转"
        assert sum(e[1] for e in rel) < 0, "Δ>0（目标在左）必须下发负像素"

    def test_releases_w_when_plan_says_stop_walking(self) -> None:
        """★★ 「明明该停下却一直往前冲」的回归。

        上一 tick 判「盲走」按了 W，本 tick 判「站着转」——
        只「不再按下新键」是不够的，``_held`` 会把 W 无限结转下去。
        """
        ex, rec = _executor()
        ex.execute(_move_to(0.0, 50.0), _state(0.0, 0.0), NOW)   # 盲走 -> 按下 W
        assert rec.downs(VK.W) == 1
        ex.execute(_move_to(0.0, 50.0), _state(1.0, 0.0), NOW + 200)
        ex.execute(_move_to(0.0, 50.0), _state(2.0, 0.0), NOW + 400)
        assert rec.ups(VK.W) == 1, "该停却没松 W"
        assert rec.downs(VK.W) == 1, "不应该重复按 W"

    def test_does_not_spin_forever_on_arrival(self) -> None:
        """到达后必须清空窗口，否则下一段路开头会带着旧站定状态。"""
        ex, _rec = _executor()
        ex.execute(_move_to(0.2, 0.0, radius=1.5), _state(0.0, 0.0), NOW)
        assert ex._nav.target_key is None

    def test_changing_target_resets_the_window(self) -> None:
        """旧目标的位移方向对新目标毫无意义。

        ⚠️ 两个目标都必须是**远处**的 —— 目标就在脚下会直接判 ARRIVED，
        而 ARRIVED 会 ``_nav.reset()``，那样测的就不是「换目标」了。
        """
        ex, _rec = _executor()
        ex.execute(_move_to(0.0, 50.0), _state(0.0, 0.0), NOW)
        assert len(ex._nav._hist) == 1
        ex.execute(_move_to(1.0, 60.0), _state(1.0, 0.0), NOW + 200)
        assert len(ex._nav._hist) == 1, "换目标后窗口应该只剩当前样本"

    def test_repeated_ticks_do_not_multiply_mouse_input(self) -> None:
        """每 tick 只注入**本 tick 算出来的**那点像素，不重复累加。"""
        ex, rec = _executor()
        for i, x in enumerate((0.0, 1.0, 2.0)):
            ex.execute(_move_to(0.0, 50.0), _state(x, 0.0), NOW + i * 200)
        rel = [e for e in rec.events if e[0] == "rel"]
        assert rel
        # 每一段都必须落在单段饱和上限内（否则就是没限幅硬塞）
        assert all(abs(e[1]) <= 40 for e in rel)


class TestClosedLoopConvergence:
    """★★ 闭环：让「转向 -> 走路 -> 转向」自己跑起来，看能不能走到。

    ★ 为什么必须有这一组
    -------------------
    逐个测函数的单测**全绿也可能是错的**。2026-10-14 就真踩到过：
    ``nav.steer`` 里 ``walk = blind and now_stand`` 写成「只有站定且
    窗口将死时才走路」，于是平时转向**一步都不走**；角色不前进 ->
    朝向窗口攒不起来 -> ``heading`` 恒为 ``None`` -> 退化成盲走 ->
    **原地无限打转**。72 条单测全过，只有闭环能抓到它。

    所以这里用一个极简游戏模型（朝向随鼠标位移变、按住 W 才前进）
    把整套链跑一遍。这是本轮最有价值的一条测试。

    ⚠️ 这里把 ``NAV_TURN_SEG_SLEEP_S`` 置零：那个 sleep 的唯一作用是
    「等游戏渲染一帧」，仿真里没有游戏要渲染。它只影响挂钟耗时，
    不改变任何控制逻辑 —— 置零后 5 个场景从 3 分钟降到 1 秒。
    """

    @pytest.fixture(autouse=True)
    def _no_render_sleep(self, monkeypatch) -> None:
        import autopilot.actuator.executor as ex_mod
        monkeypatch.setattr(ex_mod, "NAV_TURN_SEG_SLEEP_S", 0.0)

    def _simulate(self, target: tuple[float, float],
                  sprint: bool = False,
                  tick_s: float = 0.2,
                  max_ticks: int = 400) -> dict:
        """闭环跑一遍，返回收敛统计。

        ⚠️ **必须**给 ``_nav`` 注入虚拟时钟并随 tick 推进：窗口老化完全靠
        它。用真实 ``monotonic`` 而不 sleep 的话「窗口永不老化」，旧样本
        会一直把朝向均值拖偏 —— 那是**仿真的失真**而不是产品的毛病
        （2026-10-14 在这儿栽过一次，差点照着一个假振荡去调参数）。
        """
        import math

        ex, rec = _executor()
        vclock = [1000.0]
        ex._nav = type(ex._nav)(clock=lambda: vclock[0])

        px, pz, head = 0.0, 0.0, 0.0
        walk_mps, run_mps = 3.0, 11.0
        path = 0.0
        stand_ticks = 0
        arrived_tick = None
        heads: list[float] = []

        for tick in range(max_ticks):
            vclock[0] += tick_s
            st = _state(px, pz, sv=tick + 1)
            if sprint:
                it = IntentPacket.make(
                    ActionType.MOVE_TO, state_version=tick + 1,
                    ttl_seconds=TTL, params={"sprint": True},
                    target=IntentTarget(world_position={
                        "x": target[0], "y": target[1], "radius": 1.5}))
            else:
                it = _move_to(target[0], target[1], sv=tick + 1)

            before = len(rec.events)
            r = ex.execute(it, st, NOW + tick * int(tick_s * 1000))
            assert r.outcome is Outcome.OK, (tick, r.reason)

            # 游戏对鼠标位移的响应：转视角
            dx = sum(e[1] for e in rec.events[before:] if e[0] == "rel")
            head += -dx * _MOUSE_DEG_PER_PX
            head = (head + 180.0) % 360.0 - 180.0
            heads.append(head)

            # 按住 W 才前进；按住 Shift 是冲刺。判「是否按住」要看
            # **当前持有状态**，不是本 tick 有没有新的 down 事件。
            if rec.is_held(VK.W):
                v = run_mps if (sprint and rec.is_held(VK.SHIFT)) else walk_mps
                nx = px + math.cos(math.radians(head)) * v * tick_s
                nz = pz + math.sin(math.radians(head)) * v * tick_s
                path += math.hypot(nx - px, nz - pz)
                px, pz = nx, nz
            elif dx:
                stand_ticks += 1

            if any("到达" in s for s in r.steps):
                arrived_tick = tick
                break

        straight = math.hypot(*target)
        return {
            "arrived_tick": arrived_tick,
            "seconds": (arrived_tick * tick_s if arrived_tick is not None
                        else None),
            "leftover": math.hypot(target[0] - px, target[1] - pz),
            "detour": path / straight if straight else 0.0,
            "stand_ticks": stand_ticks,
            "pos": (px, pz),
            "heads": heads,
        }

    #: 五个目标方向，**等距 30m**。0° 是「开局就对准」的基线；
    #: 站定阈值是 60°，所以 90°/135°/180° 都会进站定，45° 不会。
    DIRECTIONS = [0, 45, 90, 135, 180]

    @staticmethod
    def _at(deg: float, dist: float = 30.0) -> tuple[float, float]:
        import math
        r = math.radians(deg)
        return dist * math.cos(r), dist * math.sin(r)

    @pytest.mark.parametrize("deg", DIRECTIONS)
    def test_reaches_target_in_every_direction(self, deg: int) -> None:
        r = self._simulate(self._at(deg))
        assert r["arrived_tick"] is not None, (
            f"目标方位 {deg}° 没走到：最后位置 {r['pos']}"
            f"（剩余 {r['leftover']:.1f}m）")
        assert r["leftover"] <= 1.5
        # 30m 直线 @3m/s 至少 10s；给 3 倍余量当合理上限。
        assert r["seconds"] <= 30.0, \
            f"目标方位 {deg}° 走了 {r['seconds']:.1f}s，太慢"

    @pytest.mark.parametrize("deg", DIRECTIONS)
    def test_heading_does_not_oscillate(self, deg: int) -> None:
        """★ 振荡的判据不是「没到」，是**朝向来回甩**。

        2026-10-14 的 bug：站定期间 heading 冻结，每帧都按陈旧 Δ 发满
        22.4°，角色在 -158°~+157° 之间反复，位置几乎不动。
        这里数「尾段朝向换向次数」：真收敛时只朝一个方向走。
        """
        r = self._simulate(self._at(deg))
        heads = r["heads"]
        assert len(heads) >= 10, "样本太少，判不了"
        tail = heads[len(heads) // 3:]
        flips, prev_sign = 0, 0
        for a, b in zip(tail, tail[1:]):
            d = b - a
            if abs(d) < 1.0:            # 忽略亚度抖动
                continue
            s = 1 if d > 0 else -1
            if prev_sign and s != prev_sign:
                flips += 1
            prev_sign = s
        assert flips <= 6, \
            f"目标方位 {deg}° 的朝向在尾段换了 {flips} 次向 —— 这是振荡"

    def test_sprint_is_faster_than_walking(self) -> None:
        walk = self._simulate(self._at(90.0))
        run = self._simulate(self._at(90.0), sprint=True)
        assert walk["seconds"] is not None and run["seconds"] is not None
        assert run["seconds"] < walk["seconds"], \
            "冲刺不比走路快，说明 Shift 没被按住"

    def test_detour_stays_bounded(self) -> None:
        """绕路系数 = 实走 / 直线。真机历史值约 1.3~1.8。

        远大于 2.0 说明它在绕弧而不是直奔目标 —— 大转向的代价本应被
        站定压住。
        """
        r = self._simulate(self._at(90.0))
        assert r["detour"] < 2.0, \
            f"绕路系数 {r['detour']:.2f}，像在画圈"



class TestNavWindow:
    """``_NavWindow`` 的窗口语义。

    ⚠️ 全部用**注入的虚拟时钟**（2026-10-14）。真实 ``monotonic`` 下，
    不 sleep 的一串 ``push`` 全部落在同一时刻、窗口永不老化 ——
    那种失真会让人去调一个根本不存在的问题。
    """

    @staticmethod
    def _win():
        c = [1000.0]
        return _NavWindow(clock=lambda: c[0]), c

    def test_needs_three_samples_to_get_a_heading(self) -> None:
        w, c = self._win()
        w.push(0.0, 0.0)
        assert w.heading() is None
        c[0] += 0.2
        w.push(1.0, 0.0)
        assert w.heading() is None              # 只有 1 段
        c[0] += 0.2
        w.push(2.0, 0.0)
        assert w.heading() == pytest.approx(0.0)

    def test_heading_tracks_displacement_direction(self) -> None:
        w, c = self._win()
        for x, y in ((0.0, 0.0), (0.0, 1.0), (0.0, 2.0)):
            w.push(x, y)
            c[0] += 0.2
        assert w.heading() == pytest.approx(90.0)

    def test_standing_still_yields_no_heading(self) -> None:
        """★ 停在原地时位移是噪声，不能当成方向。"""
        w, c = self._win()
        for _ in range(3):
            w.push(1.0, 1.0)
            c[0] += 0.2
        assert w.heading() is None

    def test_repeated_same_point_does_not_stuff_the_window(self) -> None:
        """定位没动时重复采样只会稀释窗口 —— 应该顶掉上一条。"""
        w, c = self._win()
        for _ in range(4):
            w.push(0.0, 0.0)
            c[0] += 0.2
        assert len(w._hist) == 1

    def test_window_move_is_net_displacement(self) -> None:
        w, c = self._win()
        w.push(0.0, 0.0)
        assert w.window_move_m() is None
        c[0] += 0.2
        w.push(3.0, 4.0)
        assert w.window_move_m() == pytest.approx(5.0)

    def test_window_is_bounded(self) -> None:
        w, c = self._win()
        for i in range(_NavWindow.MAX_SAMPLES * 3):
            w.push(float(i), 0.0)
            c[0] += 0.2
        assert len(w._hist) <= _NavWindow.MAX_SAMPLES

    def test_old_samples_age_out(self) -> None:
        """★ 窗口真的会老：超过 ``NAV_WINDOW_S`` 的样本必须被丢掉。

        这条正是闭环仿真能不能代表真机的分界 —— 窗口不老化，
        朝向均值就会被历史样本长期拖偏。
        """
        w, c = self._win()
        for x in (0.0, 1.0, 2.0):
            w.push(x, 0.0)
            c[0] += 0.2
        assert w.heading() == pytest.approx(0.0)
        # 空窗 2.5s（> NAV_WINDOW_S）后沿 +y 走三步
        c[0] += 2.5
        for y in (1.0, 2.0, 3.0):
            w.push(2.0, y)
            c[0] += 0.2
        assert w.heading() == pytest.approx(90.0), \
            "旧的 +x 样本没被淘汰，朝向被历史拖偏"

    def test_reset_clears_stand_state(self) -> None:
        w, _c = self._win()
        w.stand_active = True
        w.stand_budget_deg = 42.0
        w.push(0.0, 0.0)
        w.reset()
        assert w.stand_active is False
        assert w.stand_budget_deg == 0.0
        assert w._hist == []
