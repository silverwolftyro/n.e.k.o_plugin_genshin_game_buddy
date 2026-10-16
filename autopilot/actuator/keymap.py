"""键位配置：三级解耦 ``Action -> KeyId -> VK``。

    config/keymap.toml
          │  动作名（字符串，来自 contracts.intent_packet.ActionType）
          ▼
      KeyId          ← int 子类，带 NONE / UNKNOWN 哨兵
          │  to_vk()
          ▼
        VK           ← 真正发给 SendInput 的 wVk

为什么要中间这一层：``ActionType`` 是协议（跨云端要稳定），``VK`` 是平台
（Win32 私有）。玩家改键、改键帽、改输入法，都不该让协议层跟着动。

**加载期严格失败**：配置里出现未知动作 / 未知键名 / 非法十六进制，
一律抛 ``KeymapError`` 并指名道姓告诉你是哪一行哪个键 —— 不静默跳过、
不回落默认值（原则 1：一个打错的键位如果被悄悄忽略，等于角色永远按不出技能）。
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

from .executor import DEFAULT_KEYMAP
from .simulator import KeyId

logger = logging.getLogger(__name__)

#: 默认配置文件位置（插件内，首次运行时由 keymap_editor 生成）
DEFAULT_KEYMAP_PATH = Path(__file__).resolve().parents[1] / "config" / "keymap.toml"


class KeymapError(Exception):
    """键位配置非法。**只在加载期抛**，运行期绝不。"""


def _as_keyid(value: Any) -> KeyId:
    """把 ``VK`` 成员 / 裸 int / KeyId 统一成 ``KeyId``。

    ``DEFAULT_KEYMAP`` 里的值是 ``VK``（IntEnum），而配置读出来的是
    ``KeyId``（int 子类）。不归一化的话，下游 ``to_vk()`` / ``display_name``
    会在 ``VK`` 上 AttributeError。
    """
    return value if isinstance(value, KeyId) else KeyId(int(value))


def keymap_path() -> Path:
    """用户配置优先，回落插件内默认值。

    查找顺序：
      1. 环境变量 ``NEKO_KEYMAP``
      2. ``<repo>/config/keymap.toml``（用户改配置放这儿，不进版本库）
      3. 插件内 ``autopilot/config/keymap.toml``
    """
    import os

    env = os.environ.get("NEKO_KEYMAP")
    if env:
        return Path(env)
    user = Path(__file__).resolve().parents[3] / "config" / "keymap.toml"
    if user.exists():
        return user
    return DEFAULT_KEYMAP_PATH


def load_keymap(path: Path | str | None = None) -> dict[str, KeyId]:
    """从 TOML 加载键位表。文件不存在时返回 ``DEFAULT_KEYMAP`` 的副本。

    :raises KeymapError: TOML 语法错 / 未知动作 / 未知键名 / 值不是字符串。
    """
    p = Path(path) if path else keymap_path()
    if not p.exists():
        logger.info("未找到 %s，使用内置默认键位", p)
        # ⚠️ 也要过一遍 ``_as_keyid``：``DEFAULT_KEYMAP`` 的值是 ``VK``
        #   成员，而正常路径返回的是 ``KeyId``。两条路径类型不一致的话，
        #   下游 ``to_vk()`` / ``display_name`` 只在「配了文件」时能用 ——
        #   而恰恰是「没配文件」的新用户最需要它能用。
        return {a: _as_keyid(k) for a, k in DEFAULT_KEYMAP.items()}

    try:
        with p.open("rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise KeymapError(f"{p} TOML 语法错误：\n{exc}") from exc
    except OSError as exc:
        raise KeymapError(f"读取 {p} 失败：{exc}") from exc

    table = data.get("keymap")
    if not isinstance(table, dict):
        raise KeymapError(
            f"{p} 缺少 [keymap] 表。文件应形如：\n\n"
            f'  [keymap]\n  USE_SKILL = "E"\n'
        )

    # 默认项是 VK 成员，配置项是 KeyId —— 统一成 KeyId 再返回，
    # 否则下游拿到的 dict 里混着两种类型，KeyId 专属方法时有时无。
    out: dict[str, KeyId] = {
        a: _as_keyid(k) for a, k in DEFAULT_KEYMAP.items()
    }
    unknown_actions: list[str] = []
    for action, raw in table.items():
        if action not in DEFAULT_KEYMAP:
            # 拼错的动作名 —— 很可能是想写 MOVE_FORWARD 写了 MOVE_FROWARD
            unknown_actions.append(action)
            continue
        if not isinstance(raw, str):
            raise KeymapError(
                f"{p}: [keymap].{action} 必须是字符串，"
                f"实际是 {type(raw).__name__}（{raw!r}）"
            )
        try:
            out[action] = KeyId.from_name(raw)
        except Exception as exc:  # KeyIdError
            raise KeymapError(
                f"{p}: [keymap].{action} = {raw!r} 不是合法键位。\n{exc}"
            ) from exc

    if unknown_actions:
        hint = _suggest(unknown_actions)
        raise KeymapError(
            f"{p}: [keymap] 里有未知动作 {unknown_actions}。\n"
            f"合法动作名：\n  " + "\n  ".join(sorted(DEFAULT_KEYMAP)) + f"\n{hint}"
        )

    # 哨兵拦一道：UNKNOWN 意味着「这里没配」，不能进运行时
    for action, kid in out.items():
        if int(kid) == KeyId.UNKNOWN:
            raise KeymapError(
                f"{p}: [keymap].{action} 被设成 UNKNOWN —— "
                f"UNKNOWN 是「未配置」的哨兵，不能写进配置。"
                f"不想要这个动作请写 \"NONE\"。"
            )

    logger.info("已加载键位配置 %s（覆盖 %d 项）", p, len(table))
    return out


def load_char_slots(path: Path | str | None = None) -> dict[str, int]:
    """从键位配置的 ``[char_slots]`` 表读「角色 id -> 槽位 1..4」。

    :returns: 没配 / 文件不存在时返回 ``{}``（**不是**报错）——
        编队是用户才有的信息，没有就等于「不知道切谁」，而
        ``_slot_of`` 对空表返回 None、由调用方跳过并记账。
    :raises KeymapError: 配了但值不是 1..4 的整数 —— 那一定是写错了，
        响亮失败比静默跳过来得好。

    ★ 为什么单独一个函数而不是并进 :func:`load_keymap`
      ``load_keymap`` 的契约是「返回 ``dict[str, KeyId]``」，且**故意**
      对未知动作名报错（``MOVE_FROWARD`` 这种拼错要当场发现）。
      ``[char_slots]`` 是另一张表、另一套语义，塞进去只会打破那个契约。
    """
    p = Path(path) if path else keymap_path()
    if not p.exists():
        return {}
    try:
        with p.open("rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError):
        # 键位表本身已经由 load_keymap 报过错了，这里不再重复刷屏
        return {}
    table = data.get("char_slots")
    if not isinstance(table, dict) or not table:
        return {}

    out: dict[str, int] = {}
    for char_id, raw in table.items():
        try:
            v = int(raw)
        except (TypeError, ValueError):
            raise KeymapError(
                f"{p}: [char_slots].{char_id} 必须是 1~4 的整数，"
                f"实际是 {raw!r}"        ) from None
        if not 1 <= v <= 4:
            raise KeymapError(
                f"{p}: [char_slots].{char_id} = {v} 超出槽位范围 1~4。"
                f"队伍只有 4 个位置。"
            )
        out[str(char_id)] = v
    logger.info("已加载编队槽位 %s（%d 个角色）", p, len(out))
    return out


def _suggest(names: list[str]) -> str:
    """给拼错的动作名找最接近的合法名。用 difflib，不引第三方依赖。"""
    import difflib  # noqa: PLC0415

    valid = sorted(DEFAULT_KEYMAP)
    lines = ["可能想写："]
    for n in names:
        close = difflib.get_close_matches(n, valid, n=1, cutoff=0.6)
        if close:
            lines.append(f"  {n} -> {close[0]}")
    return "\n".join(lines) if len(lines) > 1 else ""


def dump_keymap(keymap: dict[str, KeyId]) -> str:
    """序列化成 TOML 文本。键名排序，保证同一配置每次生成的文件一致。"""
    groups: list[tuple[str, tuple[str, ...]]] = [
        ("COMBAT", ("NORMAL_ATTACK", "USE_SKILL", "USE_BURST", "DODGE",
                    "SWITCH_CHARACTER")),
        ("NAV", ("MOVE_FORWARD", "MOVE_BACK", "MOVE_LEFT", "MOVE_RIGHT",
                 "SPRINT", "JUMP", "MOVE_TO")),
        ("UI", ("OPEN_INVENTORY", "OPEN_MAP", "OPEN_PAIMON_MENU",
                "CLICK_ELEMENT", "CONFIRM", "CANCEL", "CLOSE")),
        ("SYSTEM", ("WAIT", "SCREENSHOT", "NOOP")),
    ]
    lines = [
        "# N.E.K.O. 键位配置",
        "#",
        "# 改完直接生效。值写 VK 名（W / E / SHIFT / LBUTTON / F12…）、",
        "# 0x 前缀十六进制，或 \"NONE\"（该动作不发按键）。",
        "#",
        "# ⚠ 方向键与队伍槽位键**不在这里配** —— 它们是硬编码的，",
        "#   改了会破坏导航与切人逻辑。",
        "#",
        "# 用 keymap_editor 交互式编辑：",
        "#   python -m autopilot.tools.keymap_editor",
        "",
        "[keymap]",
    ]
    for title, actions in groups:
        lines.append("")
        lines.append(f"# ---- {title} ----")
        for action in actions:
            if action not in keymap:
                continue
            kid = _as_keyid(keymap[action])
            name = "NONE" if int(kid) == KeyId.NONE else kid.display_name
            lines.append(f'{action} = "{name}"')
    lines.append("")
    return "\n".join(lines)


def save_keymap(keymap: dict[str, KeyId], path: Path | str | None = None) -> Path:
    """写回 TOML。**先校验再落盘** —— 不让非法配置污染磁盘。"""
    probe = {a: _as_keyid(k) for a, k in keymap.items()}
    for action, kid in probe.items():
        if action not in DEFAULT_KEYMAP:
            raise KeymapError(f"未知动作 {action!r}，不能写入配置")
        if int(kid) == KeyId.UNKNOWN:
            raise KeymapError(
                f"[keymap].{action} 是 UNKNOWN 哨兵，不能写入配置（用 \"NONE\"）"
            )
        if int(kid) != KeyId.NONE and int(kid) <= 0:
            raise KeymapError(f"[keymap].{action} 的 VK 码非法：{int(kid)}")

    p = Path(path) if path else keymap_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_keymap(probe), encoding="utf-8")
    logger.info("键位配置已写入 %s", p)
    return p


__all__ = [
    "DEFAULT_KEYMAP_PATH",
    "KeymapError",
    "dump_keymap",
    "keymap_path",
    "load_keymap",
    "save_keymap",
]
