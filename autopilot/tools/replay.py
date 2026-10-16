"""决策回放：喂一段状态序列，输出对应的意图序列。

    python -m autopilot.tools.replay --input capture.jsonl
    python -m autopilot.tools.replay --input capture.jsonl --summary
    python -m autopilot.tools.replay --input capture.jsonl --output intents.jsonl

这是「工具 = 运行时」原则最直接的落地：决策层是**纯函数**（状态进、
意图出，无副作用），所以能在真机上录一段，之后离线反复回放调规则，
不必每次都开游戏重打一遍。

**纯离线**：不截图、不连窗口、不发任何输入。

JSONL 格式：每行一个 ``StatePacket.to_dict()`` 的 JSON。
空行与 ``#`` 开头的行会被跳过（方便手写测试样例）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, line, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, line, setup_logging


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line_ = raw.strip()
        if not line_ or line_.startswith("#"):
            continue
        try:
            rows.append(json.loads(line_))
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {i} 行不是合法 JSON：{exc}") from exc
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="决策回放（纯离线，不连实机）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="状态序列 JSONL")
    ap.add_argument("--output", default="", help="意图序列 JSONL 输出路径")
    ap.add_argument("--rules-dir", default="", help="规则目录（默认内置）")
    ap.add_argument("--summary", action="store_true",
                    help="只打印统计（帧数/命中率/动作分布/空转原因）")
    ap.add_argument("--quiet", action="store_true",
                    help="不逐帧打印，只写 --output")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    from autopilot.contracts.state_packet import StatePacket  # noqa: PLC0415
    from autopilot.policy.local_tree import LocalPolicyTree  # noqa: PLC0415
    from autopilot.policy.rule_loader import RuleLoadError  # noqa: PLC0415

    src = Path(args.input)
    if not src.exists():
        print(f"{C.ERR}✗ 输入文件不存在：{src}{C.OFF}", file=sys.stderr)
        return 2
    try:
        rows = read_jsonl(src)
    except ValueError as exc:
        print(f"{C.ERR}✗ {src} 解析失败：{exc}{C.OFF}", file=sys.stderr)
        return 2
    if not rows:
        print(f"{C.ERR}✗ {src} 里没有任何数据行{C.OFF}", file=sys.stderr)
        return 2

    from autopilot.contracts.events import Event  # noqa: PLC0415

    events: list = []

    def sink(ev: Event) -> None:
        events.append(ev)

    kw = {"rules_dir": Path(args.rules_dir)} if args.rules_dir else {}
    try:
        tree = LocalPolicyTree(emit_event=sink, **kw)
    except RuleLoadError as exc:
        print(f"{C.ERR}✗ 规则加载失败{C.OFF}\n\n{exc}", file=sys.stderr)
        return 2

    results: list[dict] = []
    action_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    idle_reasons: dict[str, int] = {}
    hits = 0

    for i, row in enumerate(rows, 1):
        try:
            st = StatePacket.from_dict(row)
        except Exception as exc:  # noqa: BLE001
            print(f"{C.ERR}✗ 第 {i} 行状态包解析失败：{exc}{C.OFF}",
                  file=sys.stderr)
            return 2

        trace = tree.decide(st)
        rule = trace.matched_rule
        if rule is not None:
            hits += 1
            action_counts[rule.action.value] = action_counts.get(rule.action.value, 0) + 1
            rule_counts[rule.name] = rule_counts.get(rule.name, 0) + 1
        else:
            # 空转原因归类：第一个「因未知而不触发」的理由
            why = "条件全为假"
            for e in trace.entries:
                if e.unknowns:
                    why = e.unknowns[0].split(":")[0]
                    break
            idle_reasons[why] = idle_reasons.get(why, 0) + 1

        results.append({
            "frame": i,
            "state_version": st.state_version,
            "timestamp_ms": st.timestamp_ms,
            "scene": st.scene.id,
            "hit": trace.hit,
            "rule": rule.name if rule else None,
            "action": rule.action.value if rule else None,
            "priority": rule.priority if rule else None,
        })

        if not args.quiet and not args.summary:
            if rule is not None:
                print(f"  {C.OK}#{i:<4}{C.OFF} {st.scene.id:<14} "
                      f"{C.OK}{rule.action.value:<18}{C.OFF} "
                      f"{C.DIM}<- {rule.name} (p{rule.priority}){C.OFF}")
            else:
                print(f"  {C.DIM}#{i:<4}{C.OFF} {st.scene.id:<14} "
                      f"{C.DIM}空转{C.OFF}")

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n{C.OK}✓ 已写出 {len(results)} 条意图 -> {outp}{C.OFF}")

    if args.summary:
        total = len(results)
        print(f"\n{C.BOLD}回放统计{C.OFF}  {C.DIM}{src.name}{C.OFF}")
        print(f"\n  总帧数     {total}")
        print(f"  命中       {hits}")
        print(f"  空转       {total - hits}")
        rate = (hits / total * 100) if total else 0.0
        col = C.OK if rate >= 50 else (C.WARN if rate > 0 else C.ERR)
        print(f"  命中率     {col}{rate:.1f}%{C.OFF}")
        print(f"  规则总数   {len(tree.rules)}（enabled={tree.stats()['enabled']}）")

        if action_counts:
            print(f"\n{C.BOLD}动作分布{C.OFF}")
            for a, n in sorted(action_counts.items(), key=lambda x: -x[1]):
                bar = "█" * max(1, round(n / total * 30))
                print(f"    {a:<20} {n:>4}  {C.DIM}{bar}{C.OFF}")
        if rule_counts:
            print(f"\n{C.BOLD}命中规则分布{C.OFF}")
            for r, n in sorted(rule_counts.items(), key=lambda x: -x[1]):
                print(f"    {r:<30} {n:>4}")
        if idle_reasons:
            print(f"\n{C.BOLD}空转原因{C.OFF}")
            for w, n in sorted(idle_reasons.items(), key=lambda x: -x[1]):
                print(f"    {w:<30} {n:>4}")
            if len(idle_reasons) == 1 and "has_enemy" in next(iter(idle_reasons)):
                print(f"{C.WARN}    ⚠ 全部空转都卡在 has_enemy —— "
                      f"战斗规则现在靠 scene()=='COMBAT' 近似，"
                      f"这条录像里场景可能根本不是 COMBAT{C.OFF}")
        errs = [e for e in events if getattr(e, "severity", None)
                and e.severity.value == "ERROR"]
        if errs:
            print(f"\n{C.ERR}求值错误 {len(errs)} 次{C.OFF}")
            for e in errs[:5]:
                print(f"    {e.message[:76]}")
    else:
        print(f"\n{line()}")
        print(f"{C.BOLD}共 {len(results)} 帧{C.OFF}  命中 {hits}  "
              f"空转 {len(results) - hits}")
        if idle_reasons:
            top = max(idle_reasons.items(), key=lambda x: x[1])
            print(f"{C.DIM}  主要空转原因: {top[0]}（{top[1]} 帧）{C.OFF}")
        print(f"{C.DIM}  加 --summary 看完整分布{C.OFF}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
