"""策略诊断：当前用的是哪个实现、加载了多少规则、分布如何。

    python -m autopilot.tools.policy_diagnose
    python -m autopilot.tools.policy_diagnose --rules-dir my_rules/
    python -m autopilot.tools.policy_diagnose --json

**纯离线**：不截图、不连窗口、不发任何输入。

什么时候跑它
------------
* 改了规则文件之后 —— 确认能加载、条数对、优先级分布符合预期
* 机器人"什么都不干"的时候 —— 先看这里是不是一条规则都没加载上
* 想知道某条规则为什么永远不触发 —— 看它挂在哪个场景下

退出码：0 = 规则全部加载成功；2 = 有加载错误。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, line, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, line, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(
        description="策略诊断（纯离线）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rules-dir", default="",
                    help="规则目录（默认 autopilot/rules/）")
    ap.add_argument("--kind", default="LOCAL",
                    help="策略类型 LOCAL/CLOUD/DISABLED（默认 LOCAL）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--list-rules", action="store_true",
                    help="逐条列出规则及其条件")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    # ⚠ 本工具**刻意不 import actuator / perception**。它要能在只装了
    #   policy 的环境里跑（比如别人的机器上排查规则为什么没生效），
    #   而 actuator 会把 ctypes/SendInput 一路拖进来。
    #   「规则场景声明 vs traits.allowed_scenes」的相容性检查改由
    #   autopilot/tools 之外的开发期验证脚本承担（见 _verify_policy4.py）。
    from autopilot.policy import (  # noqa: PLC0415
        create_policy_source, registered_kinds,
    )
    from autopilot.policy.base import describe_source  # noqa: PLC0415
    from autopilot.policy.local_tree import (  # noqa: PLC0415
        LocalPolicyTree,
    )

    out: dict = {"kind": args.kind, "ok": False}

    # ---- 策略源（走工厂，不绕过包边界）----
    try:
        cfg: dict = {"kind": args.kind}
        if args.rules_dir:
            cfg["rules_dir"] = args.rules_dir
        src = create_policy_source(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"{C.ERR}✗ 策略源创建失败{C.OFF}\n\n{exc}", file=sys.stderr)
        return 2
    out["source"] = describe_source(src)
    out["registered_kinds"] = list(registered_kinds())

    # ---- 规则加载 ----
    load_err = ""
    rules = []
    if isinstance(src, LocalPolicyTree):
        rules = list(src.rules)
        out["rules_dir"] = str(src.rules_dir)
    else:
        out["rules_dir"] = "(非本地策略，无规则)"

    if not isinstance(src, LocalPolicyTree):
        print(f"{C.WARN}策略类型 {args.kind} 不是本地规则树，"
              f"没有规则可诊断{C.OFF}")
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # 统计
    by_scene = Counter()
    by_action = Counter()
    by_priority = Counter()
    disabled = []
    for r in rules:
        for s in (r.scenes or ("(不限)",)):
            by_scene[s] += 1
        by_action[r.action.value] += 1
        by_priority[r.priority] += 1
        if not r.enabled:
            disabled.append(r)

    # 覆盖检查：规则声明的场景必须被该动作的 traits 允许。
    # 这里**读不到 traits**（刻意不依赖 actuator），所以只做两类自查：
    #   ① 动作名在 ActionType 白名单里（rule_loader 加载期已保证，这里复核）
    #   ② 没有任何规则声明了「不限场景」的危险动作（见下）
    from autopilot.contracts.intent_packet import ActionType  # noqa: PLC0415

    conflicts: list[str] = []
    UNSCOPED_RISKY = {"DODGE", "NORMAL_ATTACK", "SWITCH_CHARACTER"}
    for r in rules:
        try:
            ActionType(r.action.value)
        except ValueError:
            conflicts.append(f"{r.name}: 动作 {r.action.value} 不在 ActionType 白名单")
        if not r.scenes and r.action.value in UNSCOPED_RISKY:
            conflicts.append(
                f"{r.name}: 动作 {r.action.value} 是场景相关动作，"
                f"却没声明 scenes（= 任意场景都能触发，风险高）"
            )

    out["total"] = len(rules)
    out["enabled"] = len(rules) - len(disabled)
    out["disabled"] = [r.name for r in disabled]
    out["by_scene"] = dict(sorted(by_scene.items()))
    out["by_action"] = dict(sorted(by_action.items()))
    out["by_priority"] = dict(sorted(by_priority.items()))
    out["conflicts"] = conflicts
    out["stats"] = src.stats()
    out["ok"] = bool(rules) and not conflicts

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out["ok"] else 2

    # ---- 人读输出 ----
    print(f"\n{C.BOLD}策略诊断{C.OFF}")
    print(f"\n{C.BOLD}实现{C.OFF}")
    print(f"  类型          {args.kind}")
    print(f"  类            {out['source']['type']}")
    print(f"  name          {out['source']['name']}")
    print(f"  latency_class {out['source']['latency_class']}")
    print(f"  available     {out['source']['available']}")
    print(f"  规则目录      {out['rules_dir']}")
    print(f"  已注册类型    {', '.join(out['registered_kinds'])}")

    print(f"\n{C.BOLD}规则{C.OFF}")
    total = len(rules)
    col = C.OK if total else C.ERR
    print(f"  总数          {col}{total}{C.OFF}")
    print(f"  启用          {out['enabled']}")
    if disabled:
        print(f"  {C.WARN}禁用      {len(disabled)}"
              f"（{', '.join(r.name for r in disabled)}）{C.OFF}")

    if by_priority:
        print(f"\n{C.BOLD}优先级分布{C.OFF}  {C.DIM}(越小越优先){C.OFF}")
        for p in sorted(by_priority):
            n = by_priority[p]
            bar = "▇" * n
            print(f"    {p:>3}  {bar} {n}")

    print(f"\n{C.BOLD}场景分布{C.OFF}")
    for s, n in sorted(by_scene.items(), key=lambda x: -x[1]):
        print(f"    {s:<24} {n:>3}")

    print(f"\n{C.BOLD}动作分布{C.OFF}")
    for a, n in sorted(by_action.items(), key=lambda x: -x[1]):
        print(f"    {a:<24} {n:>3}")

    if conflicts:
        print(f"\n{C.ERR}✗ 自查发现 {len(conflicts)} 处问题{C.OFF}")
        for c in conflicts:
            print(f"    {c}")
    else:
        print(f"\n{C.OK}✓ 动作全在 ActionType 白名单内；"
              f"场景相关动作均已声明 scenes{C.OFF}")

    if args.list_rules:
        print(f"\n{line()}")
        print(f"{C.BOLD}规则清单{C.OFF}（按 priority 升序，命中即停）")
        print(f"{C.DIM}{'优先级':<6}{'规则':<32}{'动作':<18}{'场景':<22}条件{C.OFF}")
        for r in rules:
            flag = f"{C.WARN}禁{C.OFF}" if not r.enabled else "  "
            sc = ",".join(r.scenes) if r.scenes else "*"
            print(f"  {r.priority:<6}{r.name:<32}{r.action.value:<18}{sc:<22}"
                  f"{r.when}")

    print()
    if not rules:
        print(f"{C.ERR}✗ 一条规则都没加载 —— 策略必然永远空转{C.OFF}")
        print(f"{C.DIM}  检查规则目录：{out['rules_dir']}{C.OFF}")
    elif conflicts:
        print(f"{C.ERR}✗ 有 {len(conflicts)} 处冲突，规则可能被 SafetyGuard 拒绝{C.OFF}")
    else:
        print(f"{C.OK}✓ 策略可用：{total} 条规则已加载{C.OFF}")
        if not src.available():
            print(f"{C.WARN}  ⚠ available()=False（所有规则都被禁用了）{C.OFF}")
    print()
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
