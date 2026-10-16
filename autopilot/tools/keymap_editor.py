"""交互式键位编辑器。

    python -m autopilot.tools.keymap_editor            # 交互编辑
    python -m autopilot.tools.keymap_editor --show     # 只看当前映射
    python -m autopilot.tools.keymap_editor --reset    # 恢复内置默认
    python -m autopilot.tools.keymap_editor --check    # 只校验配置，不进交互
    python -m autopilot.tools.keymap_editor --set USE_SKILL=R USE_BURST=Q

**三级解耦**（这就是为什么改键不会破坏协议层）::

    ActionType.USE_SKILL          协议层的动作名，跨云端稳定
        ↓  玩家配的是这一层
    KeyId("E")                   int 子类 + NONE / UNKNOWN 哨兵
        ↓  to_vk() 时校验哨兵
    VK.E = 0x45                  Win32 私有码，真正发给 SendInput

编辑的是中间那层。改键帽、改输入法布局都不影响 `ActionType`。

⚠ **方向键与队伍槽位键不在这里配** —— 它们被硬编码在 executor 里
（`_MOVE_KEYS` / `_SLOT_KEYS`）。改了会破坏导航与切人逻辑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, setup_logging

#: 不可编辑的动作 —— 键位被 executor 硬编码，改了会坏
LOCKED = {
    "MOVE_FORWARD": "W", "MOVE_BACK": "S",
    "MOVE_LEFT": "A", "MOVE_RIGHT": "D",
}

GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("COMBAT 战斗", ("NORMAL_ATTACK", "USE_SKILL", "USE_BURST", "DODGE",
                     "SWITCH_CHARACTER")),
    ("NAV 移动", ("SPRINT", "JUMP")),
    ("UI 界面", ("OPEN_INVENTORY", "OPEN_MAP", "OPEN_PAIMON_MENU",
                 "CLICK_ELEMENT", "CONFIRM", "CANCEL", "CLOSE")),
    ("SYSTEM 系统", ("WAIT", "SCREENSHOT", "NOOP")),
]


def _show(km: dict, path: Path) -> None:
    print(f"\n{C.BOLD}当前键位{C.OFF}  {C.DIM}{path}{C.OFF}")
    for title, actions in GROUPS:
        print(f"\n  {C.BOLD}{title}{C.OFF}")
        for a in actions:
            if a not in km:
                continue
            kid = km[a]
            name = f"{C.DIM}NONE{C.OFF}" if int(kid) == 0 else kid.display_name
            code = f"{C.DIM}0x{int(kid):02X}{C.OFF}"
            print(f"    {a:<20} {name:<12} {code}")
    print(f"\n  {C.DIM}硬编码不可改（改会破坏导航/切人）："
          f"{'、'.join(f'{k}={v}' for k, v in LOCKED.items())}{C.OFF}")


def _parse_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{C.ERR}✗ 参数格式应为 ACTION=KEY，收到 {it!r}{C.OFF}")
        k, _, v = it.partition("=")
        out[k.strip().upper()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="键位编辑器", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--show", action="store_true", help="只显示当前映射")
    ap.add_argument("--check", action="store_true",
                    help="只校验配置合法性（加载期报错验证）")
    ap.add_argument("--reset", action="store_true", help="恢复内置默认并写盘")
    ap.add_argument("--set", nargs="+", metavar="ACTION=KEY",
                    help="非交互直接设置，如 --set USE_SKILL=R JUMP=SPACE")
    ap.add_argument("--out", default="", help="写到指定路径（默认见下方提示）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    from autopilot.actuator import (  # noqa: PLC0415
        DEFAULT_KEYMAP, KeyId, KeyIdError, KeymapError,
        keymap_path, load_keymap, save_keymap,
    )
    from autopilot.actuator.keymap import _as_keyid  # noqa: PLC0415

    path = Path(args.out) if args.out else keymap_path()

    # ---- --reset ----
    if args.reset:
        p = save_keymap(dict(DEFAULT_KEYMAP), path)
        print(f"{C.OK}✓ 已恢复内置默认{C.OFF} -> {p}")
        return 0

    # ---- 加载（非法配置在这里就炸，绝不带病进入交互）----
    try:
        km = load_keymap(path)
    except KeymapError as exc:
        print(f"{C.ERR}✗ 键位配置非法，已中止{C.OFF}\n\n{exc}")
        print(f"\n{C.DIM}文件：{path}{C.OFF}")
        print(f"{C.DIM}修好，或先跑 --reset 恢复默认。{C.OFF}")
        return 1

    # ---- --check ----
    if args.check:
        print(f"{C.OK}✓ 配置合法{C.OFF}  {path}")
        print(f"  {C.DIM}{len(km)} 个动作{C.OFF}")
        return 0

    # ---- --show ----
    if args.show:
        _show(km, path)
        return 0

    # ---- --set（非交互）----
    if args.set:
        try:
            for action, value in _parse_kv(args.set).items():
                if action in LOCKED:
                    raise KeymapError(
                        f"{action} 是硬编码的（{LOCKED[action]}），不能改 —— "
                        f"改了会破坏导航/切人逻辑"
                    )
                km[action] = _as_keyid(KeyId.from_name(value))
            p = save_keymap(km, path)
        except KeymapError as exc:
            print(f"{C.ERR}✗ {exc}{C.OFF}")
            return 1
        print(f"{C.OK}✓ 已更新{C.OFF} -> {p}")
        _show(km, p)
        return 0

    # ---- 交互模式 ----
    print(f"{C.BOLD}键位编辑器{C.OFF}  {C.DIM}{path}{C.OFF}")
    _show(km, path)
    print(f"\n{C.DIM}输入动作名改键（如 USE_SKILL），"
          f"n=下一项，p=上一项，w=保存退出，q=放弃退出{C.OFF}")
    print(f"{C.DIM}键位可写：VK 名（W / E / SHIFT / LBUTTON / F12…）、"
          f"0x57、或 NONE（该动作不发按键）{C.OFF}\n")

    editable = [a for _, acts in GROUPS for a in acts]
    cursor = 0
    dirty = False

    while True:
        a = editable[cursor]
        kid = km.get(a)
        cur = "NONE" if int(kid) == 0 else kid.display_name
        print(f"{C.BOLD}[{cursor+1}/{len(editable)}] {a}{C.OFF}  当前 = {cur}")
        try:
            raw = input(f"  新键位（回车跳过）> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.WARN}已取消{C.OFF}")
            return 130

        if not raw:
            cursor = (cursor + 1) % len(editable)
            if cursor == 0 and dirty:
                break
            continue

        cmd = raw.lower()
        if cmd == "q":
            print(f"{C.WARN}已放弃，未写入{C.OFF}")
            return 0
        if cmd in ("n", "p"):
            cursor = (cursor + (1 if cmd == "n" else -1)) % len(editable)
            continue
        if cmd == "w":
            break

        if raw.upper() in LOCKED and a in LOCKED:
            print(f"  {C.ERR}✗ {a} 是硬编码的（{LOCKED[a]}），不能改{C.OFF}")
            continue
        try:
            km[a] = KeyId.from_name(raw)
        except KeyIdError as exc:
            # 响亮失败：把键名错误原样打回，不静默跳过
            print(f"  {C.ERR}✗ {str(exc).splitlines()[0]}{C.OFF}")
            continue
        dirty = True
        print(f"  {C.OK}✓ {a} -> {raw}{C.OFF}")
        cursor = (cursor + 1) % len(editable)

    if not dirty:
        print(f"{C.DIM}没有改动。{C.OFF}")
        return 0
    try:
        p = save_keymap(km, path)
    except KeymapError as exc:
        print(f"{C.ERR}✗ 保存失败，未写入{C.OFF}\n{exc}")
        return 1
    print(f"\n{C.OK}✓ 已保存{C.OFF} -> {p}")
    _show(km, p)
    print(f"\n  {C.DIM}建议接着跑 python -m autopilot.tools.input_demo --live 验证{C.OFF}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
