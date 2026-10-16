"""ROI 表达式求值（Python ast + 节点白名单）。

对应架构文档原则 7「ROI 表达式自适应分辨率」。

**为什么不手写 tokenizer**：
    BetterGI 为了支持 `last-exec`、`q-ready` 这类带连字符的函数名，
    手写了一个 32KB 的 tokenizer，还得做「连字符合并」（扫描到最长的、
    能在已知标识符表里命中的前缀才吞掉 -）。Python 的 ast 直接给出
    完整语法树，**更短、更安全、更好维护**。

**白名单是安全边界**：
    只允许 Constant / Name / BinOp / UnaryOp / Compare / BoolOp / Call / Load。
    Attribute（下标访问/方法调用）、Subscript、Lambda、Comprehension 一律拒绝。
    这不是"防止表达式太复杂"，是**防止配置文件执行任意代码**。

编译错误在**加载期**抛出，不留到运行期（原则 1：响亮失败）。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

#: 允许的 AST 节点类型
ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

#: 允许调用的函数名 -> 签名 (x, y, w, h) -> (x, y, w, h)
ALLOWED_CALLS: frozenset[str] = frozenset({
    "rect",
    "cutLeft", "cutRight", "cutTop", "cutBottom",
    "cutLeftTop", "cutRightTop", "cutLeftBottom", "cutRightBottom",
})

#: 允许出现在表达式里的变量名（不含自定义 vars）
BUILTIN_VARS: frozenset[str] = frozenset({"cw", "ch", "cx", "cy", "s"})


class RoiExprError(ValueError):
    """ROI 表达式错误。

    继承 ValueError，这样 pydantic / 加载器能统一捕获。
    消息里带**行号和原文**，让用户能直接定位（原则 1）。
    """


@dataclass(frozen=True)
class CompiledRoi:
    """编译后的 ROI 表达式。"""

    source: str
    tree: ast.Expression
    refs: frozenset[str]          # 引用到的变量名
    calls: frozenset[str]         # 调用到的函数名


class RoiExprEvaluator:
    """ROI 表达式求值器。

    用法::

        ev = RoiExprEvaluator(ref_size=(1920, 1080))
        c = ev.compile("rect(cw*0.72, ch*0.55, cw*0.28, ch*0.45)")
        x, y, w, h = ev.eval(c, cw=2560, ch=1600)
    """

    def __init__(self, ref_size: tuple[int, int] = (1920, 1080)) -> None:
        self.ref_w = int(ref_size[0])
        self.ref_h = int(ref_size[1])

    # -- 编译（加载期） --------------------------------------------------

    def compile(self, expr: str, *, where: str = "") -> CompiledRoi:
        """解析并校验表达式。**任何非法写法在这里抛**，不拖到运行期。"""
        text = (expr or "").strip()
        if not text:
            raise RoiExprError(f"ROI 表达式为空{_at(where)}")

        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise RoiExprError(
                f"ROI 表达式语法错误{_at(where)}: {text!r} —— {exc.msg} (行 {exc.lineno})"
            ) from exc

        refs: set[str] = set()
        calls: set[str] = set()
        # Call.func 本身也是 ast.Name，但它不是"变量引用"，要排除掉
        func_names: set[int] = set()

        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                raise RoiExprError(
                    f"ROI 表达式包含不允许的语法{_at(where)}: "
                    f"{type(node).__name__} —— {text!r}"
                )
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise RoiExprError(
                        f"只允许直接调用白名单函数{_at(where)}: {text!r}"
                    )
                name = node.func.id
                if name not in ALLOWED_CALLS:
                    raise RoiExprError(
                        f"未知函数 {name!r}{_at(where)}；"
                        f"可用函数: {', '.join(sorted(ALLOWED_CALLS))}"
                    )
                calls.add(name)
                func_names.add(id(node.func))
            if isinstance(node, ast.Name) and id(node) not in func_names:
                if node.id not in BUILTIN_VARS:
                    refs.add(node.id)
            if isinstance(node, ast.Constant) and not isinstance(
                node.value, (int, float)
            ):
                raise RoiExprError(
                    f"ROI 表达式只允许数字常量{_at(where)}: {text!r}"
                )

        return CompiledRoi(source=text, tree=tree, refs=frozenset(refs), calls=frozenset(calls))

    # -- 求值（运行期） --------------------------------------------------

    def eval(
        self,
        compiled: CompiledRoi,
        cw: int,
        ch: int,
        cx: int = 0,
        cy: int = 0,
        vars: dict[str, float] | None = None,
    ) -> tuple[int, int, int, int]:
        """求值并返回 (x, y, w, h)。结果会被裁剪到画面内且保证 w/h ≥ 1。"""
        if compiled.refs and not vars:
            missing = ", ".join(sorted(compiled.refs))
            raise RoiExprError(f"ROI 表达式引用了未提供的变量: {missing} —— {compiled.source!r}")

        env = self._env(cw, ch, cx, cy, vars, compiled)
        try:
            value = self._eval_node(compiled.tree.body, env, cw, ch)
        except RoiExprError:
            raise
        except ZeroDivisionError as exc:
            raise RoiExprError(f"ROI 表达式除零: {compiled.source!r}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RoiExprError(f"ROI 表达式求值失败: {compiled.source!r} —— {exc}") from exc

        if not isinstance(value, tuple) or len(value) != 4:
            raise RoiExprError(
                f"ROI 表达式必须返回 rect 四元组，实际得到 {type(value).__name__}: "
                f"{compiled.source!r}"
            )
        return self._clamp(value, cw, ch, compiled.source)

    def eval_scalar(
        self,
        compiled: CompiledRoi,
        cw: int,
        ch: int,
        cx: int = 0,
        cy: int = 0,
        vars: dict[str, float] | None = None,
    ) -> float:
        """求值成**标量**。用于 ``vars``（如 topBarHeight = 100 * s）。

        标量表达式不允许调用 rect/cut* 这类返回矩形的函数 —— 那是配置错误。
        """
        if compiled.refs and not vars:
            missing = ", ".join(sorted(compiled.refs))
            raise RoiExprError(f"标量表达式引用了未提供的变量: {missing}")
        if compiled.calls:
            raise RoiExprError(
                f"标量表达式不能调用函数 {sorted(compiled.calls)}: {compiled.source!r}"
            )
        env = self._env(cw, ch, cx, cy, vars, compiled)
        try:
            value = self._eval_node(compiled.tree.body, env, cw, ch)
        except ZeroDivisionError as exc:
            raise RoiExprError(f"标量表达式除零: {compiled.source!r}") from exc
        except RoiExprError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RoiExprError(f"标量表达式求值失败: {compiled.source!r} —— {exc}") from exc
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RoiExprError(
                f"标量表达式必须返回数字，实际得到 {type(value).__name__}: {compiled.source!r}"
            )
        return float(value)

    def _env(
        self,
        cw: int,
        ch: int,
        cx: int,
        cy: int,
        vars: dict[str, float] | None,
        compiled: CompiledRoi,
    ) -> dict[str, Any]:
        env: dict[str, Any] = {
            "cw": float(cw),
            "ch": float(ch),
            "cx": float(cx),
            "cy": float(cy),
            "s": (cw / self.ref_w) if cw < self.ref_w else 1.0,
        }
        if vars:
            for k, v in vars.items():
                if k in BUILTIN_VARS:
                    raise RoiExprError(f"变量名 {k!r} 与内置变量冲突")
                env[k] = float(v)
        return env

    # -- 内部 ------------------------------------------------------------

    def _eval_node(self, node: ast.AST, env: dict[str, Any], cw: int, ch: int) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise RoiExprError(f"未定义的变量: {node.id}")
            return env[node.id]
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, env, cw, ch)
            right = self._eval_node(node.right, env, cw, ch)
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, env, cw, ch)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            raise RoiExprError(f"不支持的一元运算: {type(node.op).__name__}")
        if isinstance(node, ast.BoolOp):
            vals = [self._eval_node(v, env, cw, ch) for v in node.values]
            result = vals[0]
            for v in vals[1:]:
                result = (result and v) if isinstance(node.op, ast.And) else (result or v)
            return result
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, env, cw, ch)
            for op, comp in zip(node.ops, node.comparators):
                right = self._eval_node(comp, env, cw, ch)
                if not _CMP_OPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_CALLS:
                raise RoiExprError("只允许调用白名单函数")
            if node.keywords:
                raise RoiExprError("ROI 函数不支持关键字参数")
            args = [self._eval_node(a, env, cw, ch) for a in node.args]
            return _make_rect(node.func.id, args, cw, ch)
        raise RoiExprError(f"不支持的表达式节点: {type(node).__name__}")

    @staticmethod
    def _clamp(
        rect: tuple[float, float, float, float],
        cw: int,
        ch: int,
        source: str,
    ) -> tuple[int, int, int, int]:
        x, y, w, h = (float(v) for v in rect)
        if w <= 0 or h <= 0:
            raise RoiExprError(
                f"ROI 宽高必须为正，得到 w={w:.1f} h={h:.1f}: {source!r}"
            )
        # 裁剪到画面内（UI 元素可能部分在屏外，这不是错误）
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(cw), x + w), min(float(ch), y + h)
        if x2 <= x1 or y2 <= y1:
            raise RoiExprError(
                f"ROI 完全落在画面外: ({x:.0f},{y:.0f},{w:.0f},{h:.0f}) "
                f"vs {cw}x{ch} —— {source!r}"
            )
        return (int(round(x1)), int(round(y1)),
                max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1))))


# --------------------------------------------------------------------------
# 函数实现
# --------------------------------------------------------------------------


def _make_rect(name: str, args: list[Any], cw: int, ch: int) -> tuple[float, float, float, float]:
    if name == "rect":
        if len(args) != 4:
            raise RoiExprError("rect(x, y, w, h) 需要 4 个参数")
        return (float(args[0]), float(args[1]), float(args[2]), float(args[3]))

    pct = _pct(args, 0, name)
    if name == "cutLeft":
        return (0.0, 0.0, cw * pct, float(ch))
    if name == "cutRight":
        w = cw * pct
        return (cw - w, 0.0, w, float(ch))
    if name == "cutTop":
        return (0.0, 0.0, float(cw), ch * pct)
    if name == "cutBottom":
        h = ch * pct
        return (0.0, ch - h, float(cw), h)

    wp = _pct(args, 0, name)
    hp = _pct(args, 1, name)
    w, h = cw * wp, ch * hp
    if name == "cutLeftTop":
        return (0.0, 0.0, w, h)
    if name == "cutRightTop":
        return (cw - w, 0.0, w, h)
    if name == "cutLeftBottom":
        return (0.0, ch - h, w, h)
    if name == "cutRightBottom":
        return (cw - w, ch - h, w, h)
    raise RoiExprError(f"未知函数: {name}")


def _pct(args: list[Any], idx: int, name: str) -> float:
    if len(args) <= idx:
        raise RoiExprError(f"{name}() 缺少第 {idx + 1} 个参数")
    return float(args[idx])


def _at(where: str) -> str:
    return f"（{where}）" if where else ""


_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_CMP_OPS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


# --------------------------------------------------------------------------
# 便捷入口
# --------------------------------------------------------------------------


def eval_roi_expression(
    expr: str,
    cw: int,
    ch: int,
    *,
    ref_size: tuple[int, int] = (1920, 1080),
    vars: dict[str, float] | None = None,
    where: str = "",
) -> tuple[int, int, int, int]:
    """一步到位：编译 + 求值。适合工具脚本，不适合热路径。"""
    ev = RoiExprEvaluator(ref_size=ref_size)
    return ev.eval(ev.compile(expr, where=where), cw, ch, vars=vars)


__all__ = [
    "ALLOWED_CALLS",
    "ALLOWED_NODES",
    "BUILTIN_VARS",
    "CompiledRoi",
    "RoiExprError",
    "RoiExprEvaluator",
    "eval_roi_expression",
]
