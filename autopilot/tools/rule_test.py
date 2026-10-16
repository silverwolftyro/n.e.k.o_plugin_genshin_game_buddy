"""规则离线测试：喂一个状态包，看**哪条规则命中、为什么是它**。

    python -m autopilot.tools.rule_test --state combat.json
    python -m autopilot.tools.rule_test --state bag.json --repeat 5
    python -m autopilot.tools.rule_test --state x.json --rules-dir my_rules/

**纯离线**：不截图、不连窗口、不发任何输入。只读 JSON，喂给决策层。

它的价值在于：规则为什么没触发通常是**最难查**的问题 —— 规则语法没
报错、策略没崩溃，就是"什么都不做"。这个工具把每条规则的判定过程
全打出来，让"没触发"变成"因为 hp() 未知"这种可回答的问题。

退出码
------
    0 = 命中规则并产出意图
    1 = 空转（一轮下来没有任何规则命中）
    2 = 用法 / 加载 / 文件错误
这样可以直接进 CI 或脚本：``rule_test.py ... || echo "策略没反应"``。
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="规则离线测试（不连实机、不发输入）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--state", required=True,
                    help="状态包 JSON 文件（StatePacket.to_json() 的产物）")
    ap.add_argument("--rules-dir", default="",
                    help="规则目录（默认 autopilot/rules/）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="同一状态重复评估几轮（验证冷却/幂等，默认 1）")
    ap.add_argument("--json", action="store_true",
                    help="输出机器可读 JSON（给脚本用）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    from autopilot.contracts.state_packet import StatePacket  # noqa: PLC0415
    from autopilot.policy.local_tree import LocalPolicyTree  # noqa: PLC0415
    from autopilot.policy.rule_loader import RuleLoadError  # noqa: PLC0415

    # ---- 状态包 ----
    sp = Path(args.state)
    if not sp.exists():
        print(f"{C.ERR}✗ 状态包不存在：{sp}{C.OFF}", file=sys.stderr)
        return 2
    try:
        state = StatePacket.from_json(sp.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 状态包解析失败：{sp}\n  {exc}{C.OFF}", file=sys.stderr)
        return 2

    # ---- 规则树（加载期失败会直接抛，带文件名+行号）----
    kw = {"rules_dir": Path(args.rules_dir)} if args.rules_dir else {}
    try:
        tree = LocalPolicyTree(**kw)
    except RuleLoadError as exc:
        print(f"{C.ERR}✗ 规则加载失败{C.OFF}\n\n{exc}", file=sys.stderr)
        return 2

    out: dict = {"state_file": str(sp), "rounds": []}
    for rnd in range(max(1, args.repeat)):
        trace = tree.decide(state)
        matched = trace.matched_rule
        # ★ 只调 decide()，不调 poll()：decide 是纯判定、无副作用，
        #   poll 会写冷却时间戳（重复调用就不是同一个观测了）。
        out["rounds"].append({
            "round": rnd + 1,
            "hit": trace.hit,
            "matched": matched.name if matched else None,
            "action": matched.action.value if matched else None,
            "trace": trace.to_dict(),
        })

    final = out["rounds"][-1]
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if final["hit"] else 1

    # ---- 人读输出 ----
    print(f"\n{C.BOLD}状态包{C.OFF}  {sp}")
    print(f"  场景={state.scene.id}({state.scene.confidence:.2f})  "
          f"hp={state.player.hp_ratio}  energy={state.player.energy_ratio}  "
          f"sv={state.state_version}")
    pos = state.position.quality if state.position else None
    print(f"  定位={pos}  perception_ok={state.resource.perception_ok}")
    print(f"{C.DIM}  规则目录: {tree.rules_dir}  共 {len(tree.rules)} 条{C.OFF}")

    print(f"\n{line()}")
    for rnd in out["rounds"]:
        tag = f"第 {rnd['round']} 轮"
        if len(out["rounds"]) > 1:
            print(f"\n{C.BOLD}{tag}{C.OFF}")
        tr = rnd["trace"]
        for e in tr["entries"]:
            color = {True: C.OK, False: C.DIM}.get(
                e["verdict"] == "MATCHED", C.ERR)
            vlabel = {"MATCHED": "命中", "SKIPPED": "未中",
                      "ERRORED": "出错"}.get(e["verdict"], e["verdict"])
            print(f"  [{e['priority']:>3}] {e['rule']:<30} {color}{vlabel}{C.OFF}"
                  f"  {e['reason'][:60]}")
            for u in e["unknowns"]:
                print(f"        {C.WARN}⚠ 未知字段: {u[:70]}{C.OFF}")
        if tr["skipped_by_shortcircuit"]:
            print(f"  {C.DIM}（命中即停，后续 {tr['skipped_by_shortcircuit']} "
                  f"条未评估）{C.OFF}")

    print(f"\n{line()}")
    if final["hit"]:
        print(f"{C.OK}✓ 命中规则 {final['matched']} -> 动作 {final['action']}{C.OFF}")
        print(f"{C.DIM}  本工具不执行动作（离线）。要实际发按键见 autopilot_demo。{C.OFF}")
    else:
        print(f"{C.ERR}✗ 空转：没有任何规则命中{C.OFF}")
        print(f"{C.DIM}  往上翻未命中原因，多数是「未知字段」—— "
              f"那是感知层没产出该信号，不是规则写错。{C.OFF}")
    return 0 if final["hit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
