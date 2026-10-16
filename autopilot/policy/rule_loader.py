"""从 TOML 加载规则，**加载期响亮失败**。

一条规则里任何一个字段不对，都要在启动时就报出来，指名道姓：
哪个文件、第几行、哪条规则、哪个字段、期望什么。

    RuleLoadError:
      rules/genshin_combat.toml  第 23 行  规则 burst_when_full
        动作 'USE_ULT' 不在白名单。合法动作：CONFIRM, CANCEL, ...

**不静默跳过任何规则。** 跳过一条坏规则 = 一场战斗少一个决策，
而你不会知道少了什么；更糟的是它可能恰好是保命那条（原则 1）。

依赖边界
--------
只依赖 ``contracts``（ActionType / VALID_SCENE_IDS）与 ``policy.dsl``。
**不 import actuator** —— 动作名白名单在 ``contracts.intent_packet.ActionType``
（那是「有哪些动作」），traits（在 actuator）只是「每个动作怎么执行」，
校验合法性不需要它。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..contracts.intent_packet import ActionType
from ..contracts.state_packet import VALID_SCENE_IDS
from .dsl import CompiledExpr, DslEvaluator, RuleEvalError

#: 规则的合法字段。出现表以外的字段 = 大概率是拼错了。
RULE_FIELDS: frozenset[str] = frozenset({
    "name", "action", "when", "priority", "ttl_seconds", "cooldown_ms",
    "scenes", "params", "max_attempts", "enabled", "note",
})

#: 顶层允许的表
TOP_LEVEL_TABLES: frozenset[str] = frozenset({"meta", "rules"})

PRIORITY_RANGE = (0, 100)
TTL_RANGE = (1, 300)
MAX_ATTEMPTS_RANGE = (1, 10)

#: 可选字段的默认值。**刻意与 contracts 的默认值一致** ——
#: IntentPacket.priority=50 / ttl_seconds=10、IntentConstraints.max_attempts=1。
#: 规则文件只写决策逻辑，重复的协议数字不必抄一遍；写了但写错照样响亮失败。
DEFAULT_PRIORITY = 50
DEFAULT_TTL_SECONDS = 5
DEFAULT_MAX_ATTEMPTS = 1


class RuleLoadError(Exception):
    """规则文件非法。**一定带 文件 / 行号 / 规则名**。"""

    def __init__(
        self,
        message: str,
        *,
        source_file: str = "",
        line_no: int = 0,
        rule_name: str = "",
        field_name: str = "",
        cause: BaseException | None = None,
    ) -> None:
        self.message = message
        self.source_file = source_file
        self.line_no = line_no
        self.rule_name = rule_name
        self.field_name = field_name
        self.cause = cause
        super().__init__(self._render())

    def _render(self) -> str:
        loc = self.source_file or "<未指定文件>"
        if self.line_no:
            loc += f" 第 {self.line_no} 行"
        head = f"{loc}"
        if self.rule_name:
            head += f"  规则 {self.rule_name!r}"
        if self.field_name:
            head += f"  字段 {self.field_name}"
        body = f"\n  {self.message}"
        if self.cause is not None:
            body += f"\n  起因: {type(self.cause).__name__}: {self.cause}"
        return head + body

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "source_file": self.source_file,
            "line_no": self.line_no,
            "rule_name": self.rule_name,
            "field_name": self.field_name,
        }


@dataclass(frozen=True)
class Rule:
    """一条规则。加载后**只读**。"""

    name: str
    action: ActionType
    when: str
    priority: int = 50
    ttl_seconds: int = 5
    cooldown_ms: int = 0
    #: 场景过滤。空 = 不限（再由 traits 的 allowed_scenes 兜一层）。
    scenes: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 1
    enabled: bool = True
    note: str = ""

    #: 定位信息 —— 报错和诊断都要用
    source_file: str = ""
    line_no: int = 0
    #: 加载时编译好的表达式（避免每帧重解析）
    compiled: CompiledExpr | None = field(default=None, compare=False, repr=False)

    def matches_scene(self, scene_id: str) -> bool:
        return not self.scenes or scene_id in self.scenes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action": self.action.value,
            "when": self.when,
            "priority": self.priority,
            "ttl_seconds": self.ttl_seconds,
            "cooldown_ms": self.cooldown_ms,
            "scenes": list(self.scenes),
            "params": self.params,
            "max_attempts": self.max_attempts,
            "enabled": self.enabled,
            "note": self.note,
            "source_file": self.source_file,
            "line_no": self.line_no,
        }

    def describe(self) -> str:
        sc = ",".join(self.scenes) if self.scenes else "*"
        flag = "" if self.enabled else "  (已禁用)"
        return (f"[{self.priority:>3}] {self.name:<24} {self.action.value:<18} "
                f"scene={sc:<12} {self.when}{flag}")


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------


def load_rules(path: Path | str, *, evaluator: DslEvaluator | None = None) -> list[Rule]:
    """加载单个 TOML 文件里的全部规则。"""
    return load_rules_from_files([Path(path)], evaluator=evaluator)


def load_rules_from_dir(
    rules_dir: Path | str,
    *,
    only: Iterable[str] | None = None,
    evaluator: DslEvaluator | None = None,
) -> list[Rule]:
    """加载目录下全部 ``*.toml``。

    文件按**名字排序**加载，保证优先级相同时跨文件的顺序可复现。
    """
    d = Path(rules_dir)
    if not d.is_dir():
        raise RuleLoadError(
            f"规则目录不存在：{d}", source_file=str(d)
        )
    files = sorted(d.glob("*.toml"))
    if only:
        want = {str(x) for x in only}
        files = [f for f in files if f.name in want or str(f) in want]
        missing = want - {f.name for f in files}
        if missing:
            raise RuleLoadError(
                f"规则目录 {d} 里找不到 {sorted(missing)}",
                source_file=str(d),
            )
    if not files:
        raise RuleLoadError(f"规则目录 {d} 里没有任何 *.toml", source_file=str(d))
    return load_rules_from_files(files, evaluator=evaluator)


def load_rules_from_files(
    files: Iterable[Path | str],
    *,
    evaluator: DslEvaluator | None = None,
) -> list[Rule]:
    """加载多个文件并**稳定排序**（priority 升序，同优先级保持声明顺序）。"""
    ev = evaluator or DslEvaluator()
    rules: list[Rule] = []
    seen_names: dict[str, str] = {}          # 规则名 -> 首次出现的文件
    seen_files: dict[str, Path] = {}         # 绝对路径 -> 原始入参

    for raw in files:
        p = Path(raw)
        # 同一文件被加载两次会让优先级排序的「声明顺序」失去意义，
        # 且规则名唯一性检查会误报成「重复规则」。早失败早说清。
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen_files:
            raise RuleLoadError(
                f"同一个文件被重复加载：{seen_files[key]} 与 {p}",
                source_file=str(p),
            )
        seen_files[key] = p

        data, _text = _read_toml(p)
        _check_top_level(p, data, _text)
        table = data.get("rules")
        if not isinstance(table, list):
            raise RuleLoadError(
                "缺少 [[rules]] 数组。规则应写成：\n"
                "  [[rules]]\n"
                '  name = "xxx"\n'
                '  action = "USE_BURST"\n'
                '  when = "q_ready()"',
                source_file=str(p), field_name="rules",
            )

        for idx, raw_rule in enumerate(table):
            rules.append(
                _parse_rule(p, raw_rule, idx, ev, seen_names)
            )

    if not rules:
        raise RuleLoadError("加载到 0 条规则", source_file=",".join(
            str(f) for f in files))

    # ★ 稳定排序：sorted 对同 key 保持原顺序，而原顺序 = 文件名序 + 文件内声明序
    rules.sort(key=lambda r: r.priority)
    return rules


# --------------------------------------------------------------------------
# 内部
# --------------------------------------------------------------------------


def _read_toml(p: Path) -> tuple[dict[str, Any], str]:
    if not p.exists():
        raise RuleLoadError(f"规则文件不存在：{p}", source_file=str(p))
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise RuleLoadError(f"读取失败：{exc}", source_file=str(p), cause=exc) from exc
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RuleLoadError(
            f"文件不是 UTF-8：{exc}", source_file=str(p), cause=exc) from exc
    except tomllib.TOMLDecodeError as exc:
        raise RuleLoadError(
            f"TOML 语法错误：{exc}", source_file=str(p), cause=exc) from exc
    return data, raw.decode("utf-8", errors="replace")


def _line_of(text: str, key: str, start_hint: int = 0) -> int:
    """尽力找到 ``key =`` 所在行号。

    纯文本扫描，不是真解析器 —— 定位信息是为了让人快速跳过去看，
    差一两行不影响排障。找不到就返回 0（调用方据此省略行号）。
    """
    for i, ln in enumerate(text.splitlines()[start_hint:], start_hint + 1):
        if ln.strip().startswith(f"{key}") or ln.strip().startswith(f'"{key}"'):
            return i
    for i, ln in enumerate(text.splitlines(), 1):
        if key in ln:
            return i
    return 0


def _check_top_level(p: Path, data: dict[str, Any], text: str = "") -> None:
    """顶层表白名单。

    多半是把 ``[[rules]]`` 写成了 ``[[rule]]``，或者误放了别的配置段 ——
    这种错不指出来就会被静默忽略（整个文件等于没加载）。
    """
    unknown = set(data) - TOP_LEVEL_TABLES
    if not unknown:
        return
    names = sorted(unknown)
    line = 0
    if text:
        line = next(
            (i for i, ln in enumerate(text.splitlines(), 1)
             if ln.strip().strip("[]").strip() in unknown),
            0,
        )
    raise RuleLoadError(
        f"顶层有未知表 {names}。只允许 {sorted(TOP_LEVEL_TABLES)}。"
        f"（拼错的表会被整个忽略掉 —— 比如 [[rule]] 不会被当成 [[rules]]）",
        source_file=str(p), line_no=line, field_name=names[0],
    )


def _parse_rule(
    p: Path, raw: Any, idx: int, ev: DslEvaluator, seen_names: dict[str, str]
) -> Rule:
    where = f"{idx}"
    if not isinstance(raw, dict):
        raise RuleLoadError(
            f"第 {where} 条规则不是表（每条规则应写成 [[rules]]）",
            source_file=str(p), line_no=0,
        )

    text_cache: dict[str, int] = {}
    try:
        file_text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        file_text = ""

    def loc(key: str) -> int:
        if key not in text_cache:
            text_cache[key] = _line_of(file_text, key) if file_text else 0
        return text_cache[key]

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RuleLoadError(
            "缺少 name，或不是非空字符串", source_file=str(p), line_no=0,
        )
    name = name.strip()
    line_no = loc("name")

    if name in seen_names:
        raise RuleLoadError(
            f"规则名 {name!r} 重复（已出现在 {seen_names[name]}）。"
            f"规则名必须全局唯一，否则日志里分不清是哪条规则触发的。",
            source_file=str(p), line_no=line_no, rule_name=name,
        )
    seen_names[name] = str(p)

    unknown_fields = set(raw) - RULE_FIELDS
    if unknown_fields:
        near = _suggest_fields(unknown_fields)
        raise RuleLoadError(
            f"有未知字段 {sorted(unknown_fields)}。"
            + (f" 是否想写 {near}？" if near else "")
            + f"\n  合法字段：{', '.join(sorted(RULE_FIELDS))}",
            source_file=str(p), line_no=line_no, rule_name=name,
        )

    # -- action --
    raw_action = raw.get("action")
    if not isinstance(raw_action, str):
        raise RuleLoadError(
            f"缺少 action（当前值 {raw_action!r}）",
            source_file=str(p), line_no=line_no, rule_name=name, field_name="action",
        )
    try:
        action = ActionType(raw_action.strip())
    except ValueError as exc:
        raise RuleLoadError(
            f"动作 {raw_action!r} 不在白名单。\n"
            f"  合法动作：{', '.join(a.value for a in ActionType)}",
            # ★ 指到 action 那一行，不是 name 那一行 ——
            #   报错要让人一眼看到写错的那个字段。
            source_file=str(p), line_no=loc("action"), rule_name=name,
            field_name="action", cause=exc,
        ) from exc

    # -- when --
    raw_when = raw.get("when")
    if not isinstance(raw_when, str) or not raw_when.strip():
        raise RuleLoadError(
            f"缺少 when（条件表达式），当前值 {raw_when!r}",
            source_file=str(p), line_no=line_no, rule_name=name, field_name="when",
        )
    try:
        compiled = ev.compile(raw_when.strip(), rule_name=name,
                              source_file=str(p), line_no=loc("when"))
    except RuleEvalError as exc:
        raise RuleLoadError(
            f"条件表达式不合法：{exc.message}",
            source_file=str(p), line_no=loc("when"), rule_name=name,
            field_name="when", cause=exc,
        ) from exc

    # -- 数值区间 --
    # 默认值与 contracts.IntentPacket / IntentConstraints 的默认值一致，
    # 这样规则文件可以只写决策逻辑，不必重复协议层已经定好的数字。
    # 「写了但写错」照样响亮失败 —— 省的是重复，不是校验。
    priority = _int_field(raw, "priority", PRIORITY_RANGE, p, name,
                          loc("priority"), default=DEFAULT_PRIORITY)
    ttl = _int_field(raw, "ttl_seconds", TTL_RANGE, p, name,
                     loc("ttl_seconds"), default=DEFAULT_TTL_SECONDS)
    attempts = _int_field(raw, "max_attempts", MAX_ATTEMPTS_RANGE, p, name,
                          loc("max_attempts"), default=DEFAULT_MAX_ATTEMPTS)
    cooldown = _int_field(raw, "cooldown_ms", (0, 3_600_000), p, name,
                          loc("cooldown_ms"), default=0)

    # -- scenes --
    scenes = _scenes_field(raw.get("scenes"), p, name, loc("scenes"))

    # -- params --
    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise RuleLoadError(
            f"params 必须是表，当前是 {type(params).__name__}",
            source_file=str(p), line_no=loc("params"), rule_name=name,
            field_name="params",
        )

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise RuleLoadError(
            f"enabled 必须是布尔，当前是 {type(enabled).__name__}",
            source_file=str(p), line_no=loc("enabled"), rule_name=name,
            field_name="enabled",
        )
    note = raw.get("note", "")
    if not isinstance(note, str):
        raise RuleLoadError(
            f"note 必须是字符串，当前是 {type(note).__name__}",
            source_file=str(p), line_no=loc("note"), rule_name=name,
            field_name="note",
        )

    return Rule(
        name=name, action=action, when=raw_when.strip(),
        priority=priority, ttl_seconds=ttl, cooldown_ms=cooldown,
        scenes=scenes, params=dict(params), max_attempts=attempts,
        enabled=enabled, note=note,
        source_file=str(p), line_no=line_no, compiled=compiled,
    )


def _int_field(
    raw: dict[str, Any], key: str, rng: tuple[int, int],
    p: Path, rule: str, line: int, *, default: int | None = None,
) -> int:
    if key not in raw:
        if default is None:
            raise RuleLoadError(
                f"缺少 {key}", source_file=str(p), line_no=line,
                rule_name=rule, field_name=key,
            )
        return default
    v = raw[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise RuleLoadError(
            f"{key} 必须是整数，当前是 {type(v).__name__}（{v!r}）",
            source_file=str(p), line_no=line, rule_name=rule, field_name=key,
        )
    lo, hi = rng
    if not (lo <= v <= hi):
        raise RuleLoadError(
            f"{key} = {v} 超出范围 [{lo}, {hi}]",
            source_file=str(p), line_no=line, rule_name=rule, field_name=key,
        )
    return v


def _scenes_field(
    v: Any, p: Path, rule: str, line: int
) -> tuple[str, ...]:
    if v is None:
        return ()
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        raise RuleLoadError(
            f"scenes 必须是字符串数组，当前是 {type(v).__name__}",
            source_file=str(p), line_no=line, rule_name=rule, field_name="scenes",
        )
    out: list[str] = []
    for s in v:
        if not isinstance(s, str):
            raise RuleLoadError(
                f"scenes 里必须是字符串，当前有 {type(s).__name__}（{s!r}）",
                source_file=str(p), line_no=line, rule_name=rule,
                field_name="scenes",
            )
        sid = s.strip()
        if sid not in VALID_SCENE_IDS:
            import difflib  # noqa: PLC0415

            near = difflib.get_close_matches(sid, sorted(VALID_SCENE_IDS), 1, 0.6)
            raise RuleLoadError(
                f"场景 {sid!r} 不存在。"
                + (f" 是否想写 {near[0]!r}？" if near else "")
                + f"\n  合法场景：{', '.join(sorted(VALID_SCENE_IDS))}",
                source_file=str(p), line_no=line, rule_name=rule,
                field_name="scenes",
            )
        if sid not in out:
            out.append(sid)
    return tuple(out)


def _suggest_fields(unknown: set[str]) -> str:
    import difflib  # noqa: PLC0415

    hits: list[str] = []
    for u in unknown:
        m = difflib.get_close_matches(u, sorted(RULE_FIELDS), n=1, cutoff=0.6)
        if m:
            hits.append(f"{u} -> {m[0]}")
    return "; ".join(hits)


__all__ = [
    "RULE_FIELDS",
    "Rule",
    "RuleLoadError",
    "load_rules",
    "load_rules_from_dir",
    "load_rules_from_files",
]
