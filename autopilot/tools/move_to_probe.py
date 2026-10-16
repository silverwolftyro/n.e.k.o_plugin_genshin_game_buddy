"""真机探针：驱动 **MOVE_TO** 走到一个世界坐标。

★ 为什么单独一个工具
------------------
``run_loop`` 的决策来自规则文件，而**没有任何规则会产出 MOVE_TO**
（坐标级导航还没写进 DSL）。所以现在没法用 ``run_loop`` 测这条路 ——
这个工具绕过决策层，直接每 tick 发一条 MOVE_TO 意图，把
「感知 -> 朝向窗口 -> 转向 -> 行走」整条链接起来。

它**只做一件事**：验证 MOVE_TO 在真机上能不能走到。碰到异常就停下并
说明是哪一环出的问题，不做自动脱困、不重试。

用法::

    # 干跑（默认，不发任何按键）——先确认感知与坐标链路通
    python -m autopilot.tools.move_to_probe --target 4634,4687

    # 真发按键（会控制键鼠！）
    python -m autopilot.tools.move_to_probe --target 4634,4687 --live

⚠️ **前置条件**
--------------
1. ``autopilot.locator.enabled = true``（否则 ``position`` 恒为 None，
   每条 MOVE_TO 都会被校验器以 ``POSITION_UNAVAILABLE`` 拒掉）。
2. **先在游戏里传送一次**建立定位原点。``LocalizationService`` 初始是
   ``UNINITIALIZED`` —— 它没有起点，契约见
   ``recognition/notes/teleport_contract.md``。
3. 紧急停：按住 **F12**。它走的是 ``SafetyGuard.panic_active``，
   在动作之间检查，5Hz 主循环下最坏延迟 200ms。
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time

from autopilot.contracts.intent_packet import (
    ActionType,
    IntentPacket,
    IntentTarget,
)
from autopilot.tools._stack import build_stack


def _parse_xy(text: str) -> tuple[float, float]:
    """"X,Y" -> (x, y)。⚠️ 第二个分量是**平面北南向**，不是高度。"""
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--target 要写成 'X,Y'（收到 {text!r}）")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--target 的两个分量都必须是数字（收到 {text!r}）：{exc}") from exc


def _set_origin(locator, key: str) -> int:
    """**手动**喂契约 ① SELECT —— OCR 认不出来时的兜底。

    ★ 它只替换契约的**第①段**（SELECT）
    ------------------------------------
    ② LOADING（连续 3s 非大世界）与 ③ WORLD_RESTORED 仍由状态机按
    契约自动判定 —— 所以你**仍然必须真的传送一次**。这条命令不绕过
    契约，只是把「OCR 读信息框」换成「用户直接点名」。

    ⚠️ 别把它当成硬设原点：契约的全部意义就是「没观察到传送就不刷新
    坐标」，绕过去就没有这个保护了。
    """
    from autopilot.perception import teleport_ocr as tp

    try:
        points = tp.load_teleport_points()
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 读不到传送点数据：{type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    want = tp._norm(key)                      # noqa: SLF001 - 同包内工具
    exact = [p for p in points if tp._norm(p["name"]) == want]   # noqa: SLF001
    fuzzy = ([p for p in points if want in tp._norm(p["name"])]   # noqa: SLF001
             if not exact else [])
    hits = exact or fuzzy
    if not hits:
        print(f"✗ 没有叫 {key!r} 的传送点（共 {len(points)} 个候选）。\n"
              f"  试一个名字里的片段，比如「枫丹」「蒙德」。", file=sys.stderr)
        return 2
    if len(hits) > 1:
        print(f"⚠ {key!r} 命中 {len(hits)} 个，取第一个：")
        for h in hits[:6]:
            print(f"    {h['name']}")
    p = hits[0]
    target = tp.TeleportTarget(
        matched_id=p["id"], name=p["name"],
        position=tuple(p["position"]), tran_position=tuple(p["tran_position"]),
        confidence=1.0, main_title="", sub_title=p["name"],
        areas=tuple(p.get("areas", ())),
    )
    try:
        locator.on_teleport_ocr(target)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ on_teleport_ocr 失败：{type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    wx, _, wz = target.tran_position
    print(f"✓ 契约① SELECT 已挂起：{target.name}")
    print(f"  落点 (x={wx:.1f}, z={wz:.1f})")
    print("  ★ 现在去游戏里**真的传送一次**到该点 —— "
          "契约②③（3s 非大世界 + 场景切回）会自动完成刷新。")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="真机探针：驱动 MOVE_TO 走到一个世界坐标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--target", required=True, type=_parse_xy,
                    metavar="X,Y", help="目标世界坐标，例：4634,4687")
    ap.add_argument("--radius", type=float, default=1.5,
                    help="到达半径（米，默认 1.5）")
    ap.add_argument("--sprint", action="store_true", help="全程冲刺")
    ap.add_argument("--live", action="store_true",
                    help="⚠ 真的发送按键，会控制你的键鼠（默认干跑）")
    ap.add_argument("--tick-rate", type=float, default=5.0,
                    help="tick 频率 Hz（默认 5，与主循环一致）")
    ap.add_argument("--duration", type=float, default=120.0,
                    help="最多跑 N 秒（默认 120）")
    ap.add_argument("--rules-dir", default="", help="覆盖默认规则目录")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARN", "ERROR"])
    ap.add_argument("--set-origin", default="", metavar="NAME",
                    help="★ OCR 认不出传送点时的**手动兜底**：直接点名一个"
                         "传送点完成契约① SELECT（名字或名字片段）。"
                         "只替换①段，②③仍要求你真的传送一次。")
    args_list = list(sys.argv[1:] if argv is None else argv)

    # ★ 让 stdout/stderr **编码失败不等于崩溃**（照抄 tools/run_loop.py
    #   2026-10-10 的处理）。默认 GBK 控制台编不了 `⚠`(U+26A0)，而
    #   ``--help`` 是在 **parse_args() 内部**就 print_help() 的 —— 所以
    #   这一段**必须排在 parse_args() 之前**，否则 `--help` 照样抛
    #   UnicodeEncodeError 退出 1。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")      # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass

    args = ap.parse_args(args_list)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from autopilot.config.loader import load_config

    try:
        cfg = load_config()
        st = build_stack(cfg=cfg, rules_dir=args.rules_dir or None,
                         dry_run=not args.live)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 组装失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for w in st.warnings:
        print(f"  ⚠ {w}")

    if st.locator is None:
        print("✗ 定位服务没启用（autopilot.locator.enabled = false）。\n"
              "  没有它 position 恒为 None，每条 MOVE_TO 都会被 "
              "POSITION_UNAVAILABLE 拒掉。", file=sys.stderr)
        return 2

    if args.set_origin:
        rc = _set_origin(st.locator, args.set_origin)
        if rc != 0:
            return rc
        print()

    ex = st.executor
    guard = st.guard
    tx, ty = args.target
    period = 1.0 / max(0.5, args.tick_rate)
    print(f"目标 ({tx:.1f}, {ty:.1f})  半径 {args.radius}m  "
          f"{'★ LIVE 会控制键鼠' if args.live else '干跑（不发按键）'}")
    print("急停：按住 F12\n")

    # ★ 启动自检（2026-10-15）。「传送十几次 position 还是 LOST」的
    #   根因曾经是 on_teleport_ocr 零调用方 —— 而那在日志里**一行痕迹
    #   都留不下**。所以这里直接把三件事摆出来。
    b = st.builder
    has_hook = hasattr(st.locator, "on_teleport_ocr")
    wired = bool(getattr(b, "_teleport_ocr", False)) and has_hook
    print("定位链路自检：")
    print(f"  LocalizationService     : {st.locator!r}")
    print(f"  当前状态                : {getattr(st.locator, 'state', '?')}"
          f"  (UNINITIALIZED = 没有原点)")
    print(f"  on_teleport_ocr 存在    : {has_hook}")
    print(f"  传送 OCR 已接线         : {wired}")
    if not wired:
        print("  ★ 没接线 -> 契约①永远不触发 -> position 恒为 LOST。")
        print("    排查：autopilot.locator.enabled 与 teleport_ocr_enabled")
        print("    兜底：用 --set-origin <传送点名> 手动完成①段")
    print()

    t0 = time.monotonic()
    deadline = t0 + args.duration
    tick = 0
    arrived_at = None
    reason = "跑到时长上限"
    next_print = 0.0

    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_print:
                time.sleep(period)
                continue
            next_print = now + period

            if guard.panic_active:
                reason = "F12 急停"
                break

            tick += 1
            try:
                frame = st.capturer.grab()
                state = st.builder.build(frame)
            except Exception as exc:  # noqa: BLE001
                reason = f"感知失败：{type(exc).__name__}: {exc}"
                break

            pos = state.position
            if pos is None or pos.x is None or pos.y is None:
                q = pos.quality if pos is not None else "NONE"
                if tick <= 3 or tick % 10 == 0:
                    print(f"  [{tick:4d}] 定位不可用（quality={q}）"
                          f" —— 传送一次建立原点？")
                time.sleep(period)
                continue

            intent = IntentPacket.make(
                ActionType.MOVE_TO, state_version=state.state_version,
                ttl_seconds=300, params={"sprint": bool(args.sprint)},
                target=IntentTarget(world_position={
                    "x": tx, "y": ty, "radius": args.radius}),
            )
            res = ex.execute(intent, state, int(time.time() * 1000))

            dist = math.hypot(tx - pos.x, ty - pos.y)
            nav = ex._nav
            nh = nav.heading()
            print(f"  [{tick:4d}] {res.outcome.value:9s} "
                  f"pos=({pos.x:8.1f},{pos.y:8.1f}) "
                  f"d={dist:6.2f}m "
                  f"heading={'-' if nh is None else f'{nh:6.1f}'} "
                  f"stand={int(nav.stand_active)} "
                  f"bud={nav.stand_budget_deg:5.1f} "
                  f"| {'; '.join(res.steps[-2:])}")

            if any("到达" in s for s in res.steps):
                arrived_at = time.monotonic() - t0
                reason = "到达"
                break
            if res.outcome.value not in ("OK",):
                reason = f"{res.outcome.value}: {res.reason}"
                # 拒绝多半是配置问题，继续跑没有意义
                if res.outcome.value == "REJECTED":
                    break
            time.sleep(period)
    finally:
        released = ex.release_held()
        print(f"\n收尾：松开 {len(released)} 个键")

    print("=" * 60)
    if arrived_at is not None:
        print(f"★ 到达：耗时 {arrived_at:.1f}s（{tick} tick）")
        return 0
    print(f"✗ 未到达：{reason}（{tick} tick / "
          f"{time.monotonic() - t0:.1f}s）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
