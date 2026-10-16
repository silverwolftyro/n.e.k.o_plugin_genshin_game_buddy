"""规则条件 DSL：用 ``ast`` 求值，**不手写 tokenizer**。

安全模型
--------
规则来自 TOML 文件，而 TOML 可能是用户手写、也可能是以后从外部来的。
所以求值器是**白名单沙箱**：只允许常量、变量名、算术/比较/逻辑运算和
白名单函数调用。任何属性访问、下标、lambda、推导式、f-string 一律拒绝。

    ✅ "has_enemy() and q_ready()"
    ✅ "scene() == 'COMBAT' and hp() < 0.35"
    ✅ "min(hp(), 0.5) < 0.3"
    ❌ "__import__('os').system('calc')"
    ❌ "hp.__class__"          （Attribute 直接在节点层面拒绝）

白名单**在加载期**就检查（``rule_loader`` 会调 ``compile``），
不靠运行时才炸 —— 语法错必须在启动时看到，不是跑到一半才崩。

求值异常不吞
------------
任何求值期异常都转成 :class:`RuleEvalError`，携带 **表达式 + 规则名 +
state_version + 来源文件行号**。上层据此发事件并继续下一条规则，
但绝不静默（原则 1）。

三态
----
条件为 ``UnknownValue`` 时返回 ``False``（不动作），
并由 :class:`EvalOutcome` 带上原因供诊断。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from .context import FUNCTION_NAMES, PolicyContext, UnknownValue

# --------------------------------------------------------------------------
# 异常
# --------------------------------------------------------------------------


class RuleEvalError(RuntimeError):
    """DSL 编译或求值失败。**一律携带定位信息**，不许只给一句 str(e)。"""

    def __init__(
        self,
        message: str,
        *,
        expression: str = "",
        rule_name: str = "",
        state_version: int = -1,
        source_file: str = "",
        line_no: int = 0,
        cause: BaseException | None = None,
    ) -> None:
        self.message = message
        self.expression = expression
        self.rule_name = rule_name
        self.state_version = state_version
        self.source_file = source_file
        self.line_no = line_no
        self.cause = cause
        super().__init__(self._render())

    def _render(self) -> str:
        head = self.message
        loc = []
        if self.source_file:
            loc.append(f"文件={self.source_file}")
        if self.line_no:
            loc.append(f"行={self.line_no}")
        if self.rule_name:
            loc.append(f"规则={self.rule_name}")
        if self.expression:
            loc.append(f"表达式={self.expression!r}")
        if self.state_version >= 0:
            loc.append(f"state_version={self.state_version}")
        if loc:
            head += "\n  " + "  ".join(loc)
        if self.cause is not None:
            head += f"\n  起因: {type(self.cause).__name__}: {self.cause}"
        return head

    def to_dict(self) -> dict[str, Any]:
        """转成事件 context。"""
        return {
            "message": self.message,
            "expression": self.expression,
            "rule_name": self.rule_name,
            "state_version": self.state_version,
            "source_file": self.source_file,
            "line_no": self.line_no,
            "cause": f"{type(self.cause).__name__}: {self.cause}"
            if self.cause else "",
        }


# --------------------------------------------------------------------------
# 白名单
# --------------------------------------------------------------------------

#: 允许的 AST 节点类型。**刻意保守** —— 多加一种节点等于开一类新能力。
#:
#: 没有 IfExp（``a if b else c``）：规则应该短小可读，分支逻辑应该
#: 拆成多条 priority 不同的规则，那样"哪条赢了"是可解释的。
ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Constant,      # 字面量
    ast.Name,          # 函数名（不会是变量——没有变量）
    ast.Call,
    ast.Load,
    ast.Tuple,         # 供 scene() in ('COMBAT', 'MAP') 使用；元素仍是常量
    ast.UnaryOp, ast.UAdd, ast.USub, ast.Not,
    ast.BinOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow,
    ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp,
    ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
)

#: 白名单里额外提供的函数（不是 context 的，是 DSL 自己的）。
#:
#: ``min`` / ``max`` 必须自己实现：内置版遇到 UnknownValue 会抛 TypeError，
#: 而 ``min(UNKNOWN, 0.5)`` 的正确答案是 UNKNOWN。
_DSL_FUNCS: frozenset[str] = frozenset({"min", "max"})

#: TOML 风格的布尔字面量。
#:
#: 规则写在 TOML 里，作者写 ``when = "false"`` 是完全自然的（TOML 用小写）。
#: 但 Python 的字面量是 ``False``，``false`` 会被解析成**变量名**。
#: 不认它的话，每个写规则的人都会踩一次坑，所以这里显式认下。
_TOML_BOOLS: dict[str, bool] = {"true": True, "false": False}

#: 完整函数白名单 = context 函数 + DSL 内建
ALLOWED_FUNCS: frozenset[str] = FUNCTION_NAMES | _DSL_FUNCS


# --------------------------------------------------------------------------
# min / max（Unknown-aware）
# --------------------------------------------------------------------------


def _dsl_min(*args: Any) -> Any:
    if any(isinstance(a, UnknownValue) for a in args):
        return args[0] if isinstance(args[0], UnknownValue) else UnknownValue(
            "min() 收到未知值"
        )
    if not args:
        raise ValueError("min() 至少要一个参数")
    return min(args)


def _dsl_max(*args: Any) -> Any:
    if any(isinstance(a, UnknownValue) for a in args):
        return args[0] if isinstance(args[0], UnknownValue) else UnknownValue(
            "max() 收到未知值"
        )
    if not args:
        raise ValueError("max() 至少要一个参数")
    return max(args)


_EXTRA_FUNCS = {"min": _dsl_min, "max": _dsl_max}


# --------------------------------------------------------------------------
# 编译结果
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledExpr:
    """编译好的条件表达式。**不可变**，可跨帧复用。"""

    source: str
    tree: ast.Expression
    func_names: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class EvalOutcome:
    """一次求值的结果 + 为什么。"""

    value: bool
    #: 命中 / 未命中 / 因未知而未命中的说明
    reason: str = ""
    #: 求值过程中遇到的「未知」字段（来自 PolicyContext.unknown_hits 的增量）
    unknowns: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.value


# --------------------------------------------------------------------------
# 求值器
# --------------------------------------------------------------------------


class DslEvaluator:
    """条件求值器。无状态（除了编译缓存），可长期复用。"""

    def __init__(self, *, allowed_funcs: frozenset[str] | None = None) -> None:
        self.allowed_funcs = allowed_funcs or ALLOWED_FUNCS
        self._cache: dict[str, CompiledExpr] = {}

    # -- 编译 ------------------------------------------------------------
    def compile(
        self,
        expr: str,
        *,
        rule_name: str = "",
        source_file: str = "",
        line_no: int = 0,
    ) -> CompiledExpr:
        """解析 + 白名单校验。**任何不合法的部分都在这里抛**。"""
        src = (expr or "").strip()
        if not src:
            raise RuleEvalError(
                "条件表达式为空", expression=expr, rule_name=rule_name,
                source_file=source_file, line_no=line_no,
            )
        if len(src) > 500:
            raise RuleEvalError(
                f"条件表达式过长（{len(src)} 字符，上限 500）。"
                f"复杂逻辑请拆成多条规则。",
                expression=src[:80] + "...", rule_name=rule_name,
                source_file=source_file, line_no=line_no,
            )

        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError as exc:
            raise RuleEvalError(
                f"语法错误：{exc.msg}",
                expression=src, rule_name=rule_name,
                source_file=source_file, line_no=line_no, cause=exc,
            ) from exc

        used = self._validate(tree, src, rule_name, source_file, line_no)
        compiled = CompiledExpr(source=src, tree=tree, func_names=used)
        self._cache[src] = compiled
        return compiled

    def _validate(
        self, tree: ast.AST, src: str, rule_name: str,
        source_file: str, line_no: int,
    ) -> frozenset[str]:
        """白名单校验。返回用到的函数名。"""
        used: set[str] = set()
        # ★ 先收集「被当作函数调用的 Name 节点」。ast.walk 会**独立访问**到
        #   Call.func 本身，它就是个 Name 节点 —— 不先标出来的话，
        #   下面那条「裸 Name 就是非法变量」的检查会把 q_ready() 也毙掉。
        call_names = {
            id(n.func) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }

        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                raise RuleEvalError(
                    f"不允许的语法 {type(node).__name__}。"
                    f"规则里只能用：常量、函数调用、算术/比较/逻辑运算。",
                    expression=src, rule_name=rule_name,
                    source_file=source_file, line_no=line_no,
                )
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise RuleEvalError(
                        f"不支持的调用形式 {type(node.func).__name__}。"
                        f"只能直接调用白名单函数（如 hp()），"
                        f"不能取方法、属性或下标。",
                        expression=src, rule_name=rule_name,
                        source_file=source_file, line_no=line_no,
                    )
                fn = node.func.id
                if fn not in self.allowed_funcs:
                    near = _closest(fn, self.allowed_funcs)
                    raise RuleEvalError(
                        f"未知函数 {fn}()。"
                        + (f"是否想写 {near}()？" if near else "")
                        + f"\n  可用函数：{', '.join(sorted(self.allowed_funcs))}",
                        expression=src, rule_name=rule_name,
                        source_file=source_file, line_no=line_no,
                    )
                used.add(fn)
            elif isinstance(node, ast.Name):
                # 走到这里的裸名字一定不是被调用的（ast.Call.func 已处理），
                # 即试图引用一个"变量"——DSL 没有变量概念。
                if id(node) in call_names:
                    continue
                if node.id in _TOML_BOOLS:
                    continue
                raise RuleEvalError(
                    f"未知的名字 {node.id!r}。"
                    f"DSL 里没有变量，只能调用函数。",
                    expression=src, rule_name=rule_name,
                    source_file=source_file, line_no=line_no,
                )
            if isinstance(node, ast.Constant) and not isinstance(
                node.value, (int, float, str, bool, type(None))
            ):
                raise RuleEvalError(
                    f"不允许的常量类型 {type(node.value).__name__}"
                    f"（只支持数字/字符串/布尔）",
                    expression=src, rule_name=rule_name,
                    source_file=source_file, line_no=line_no,
                )
        return frozenset(used)

    # -- 求值 ------------------------------------------------------------
    def evaluate(
        self,
        compiled: CompiledExpr | str,
        ctx: PolicyContext,
        *,
        rule_name: str = "",
        source_file: str = "",
        line_no: int = 0,
    ) -> EvalOutcome:
        """求值。**抛异常而不是吞**；调用方决定要不要转成事件继续。"""
        ce = self.compile(compiled, rule_name=rule_name,
                          source_file=source_file, line_no=line_no) \
            if isinstance(compiled, str) else compiled

        before = len(ctx.unknown_hits)
        ns = ctx.namespace()
        ns.update(_EXTRA_FUNCS)
        ns.update(_TOML_BOOLS)          # true / false -> Python True / False

        try:
            raw = eval(  # noqa: S307 - 已做 AST 白名单校验，无属性/下标/推导式
                compile(ce.tree, filename="<dsl>", mode="eval"), ns
            )
        except RuleEvalError:
            raise
        except Exception as exc:
            raise RuleEvalError(
                f"求值失败：{type(exc).__name__}: {exc}",
                expression=ce.source,
                rule_name=rule_name,
                state_version=ctx.state.state_version,
                source_file=source_file, line_no=line_no, cause=exc,
            ) from exc

        unknowns = tuple(ctx.unknown_hits[before:])

        if isinstance(raw, UnknownValue):
            # 方向性选择：未知 -> 不动作，但记下原因
            return EvalOutcome(
                False,
                f"因未知而不触发（{unknowns[-1] if unknowns else raw.reason}）",
                unknowns,
            )
        if not isinstance(raw, bool):
            raise RuleEvalError(
                f"条件结果必须是布尔，实际是 {type(raw).__name__} = {raw!r}。"
                f"写 when 时要得到 True/False（比如 'hp() < 0.35'），"
                f"不能直接返回数值。",
                expression=ce.source, rule_name=rule_name,
                state_version=ctx.state.state_version,
                source_file=source_file, line_no=line_no,
            )
        return EvalOutcome(raw, "" if raw else "条件为假", unknowns)


def _closest(name: str, candidates: frozenset[str]) -> str:
    import difflib  # noqa: PLC0415

    m = difflib.get_close_matches(name, sorted(candidates), n=1, cutoff=0.6)
    return m[0] if m else ""


#: 模块级默认求值器（无状态，可直接用）
DEFAULT_EVALUATOR = DslEvaluator()

__all__ = [
    "ALLOWED_FUNCS",
    "ALLOWED_NODES",
    "DEFAULT_EVALUATOR",
    "CompiledExpr",
    "DslEvaluator",
    "EvalOutcome",
    "RuleEvalError",
]
