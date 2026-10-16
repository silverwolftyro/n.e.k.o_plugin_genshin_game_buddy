"""环境自检 —— **你的第一条命令**。

    python -m autopilot.tools.check_env --title 原神

检查项：
    1. Python 版本
    2. 必需依赖（onnxruntime / opencv / numpy / mss / pydantic / pywin32）
    3. onnxruntime providers（CUDA 是否可用）
    4. 🔴 ultralytics 许可证提醒（AGPL，运行时禁止 import）
    5. 目标窗口是否存在
    6. 截图通道逐个实测（dxcam / bitblt / mss）+ 延迟基线
    7. autopilot 配置校验

输出绿/黄/红三色报告。**红色项不修就跑不起来**，黄色项影响性能。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import platform
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # 直接 python xxx.py 运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, line, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, line, setup_logging

REQUIRED = {
    "numpy": "数组运算",
    "cv2": "模板匹配（opencv-python）",
    "onnxruntime": "YOLO 推理（AGPL 避雷，只用 ONNX）",
    "mss": "截图备选通道",
    "pydantic": "配置校验",
}
OPTIONAL = {
    "dxcam": "截图主通道（建议装，延迟最低）",
    "win32gui": "窗口查找（pywin32）",
    "rapidocr_onnxruntime": "OCR 降级引擎",
    "ultralytics": "⚠️ 仅训练用，运行时禁止 import（AGPL）",
    "matplotlib": "confusion matrix 绘图（eval_yolo 用）",
}

#: 许可证风险项：装了就提醒，但不算错
LICENSE_SENSITIVE = {"ultralytics"}


def main() -> int:
    ap = argparse.ArgumentParser(description="autopilot 环境自检")
    ap.add_argument("--title", default="原神", help="目标窗口标题片段")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--skip-capture", action="store_true", help="跳过截图实测")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    red = yellow = green = 0
    print(f"\n{C.BOLD}N.E.K.O. Autopilot 环境自检{C.OFF}")
    print(line("═"))

    # ① Python
    print(f"\n{C.BOLD}[1] Python{C.OFF}")
    py_ok = sys.version_info >= (3, 11)
    _mark(py_ok, f"版本 {platform.python_version()}（需要 >= 3.11）")
    green, red = (green + 1, red) if py_ok else (green, red + 1)
    print(f"      {C.DIM}解释器: {sys.executable}{C.OFF}")
    print(f"      {C.DIM}平台: {platform.platform()}{C.OFF}")

    # ② 依赖
    print(f"\n{C.BOLD}[2] 必需依赖{C.OFF}")
    for mod, why in REQUIRED.items():
        ok, ver = _try_import(mod)
        _mark(ok, f"{mod:16s} {ver or '缺失':12s} {C.DIM}{why}{C.OFF}")
        if ok:
            green += 1
        else:
            red += 1

    # ③ 可选依赖
    print(f"\n{C.BOLD}[3] 可选依赖{C.OFF}")
    for mod, why in OPTIONAL.items():
        ok, ver = _try_import(mod)
        if mod in LICENSE_SENSITIVE and ok:
            print(f"  {C.WARN}!{C.OFF}     {mod:16s} {ver or '':12s} {C.WARN}{why}{C.OFF}")
            yellow += 1
        elif ok:
            _mark(True, f"{mod:16s} {ver or '':12s} {C.DIM}{why}{C.OFF}")
            green += 1
        else:
            _mark(False, f"{mod:16s} {'缺失':12s} {C.WARN}{why}{C.OFF}", warn=True)
            yellow += 1

    # ④ onnxruntime provider
    print(f"\n{C.BOLD}[4] ONNX Runtime 推理后端{C.OFF}")
    try:
        import onnxruntime as ort  # noqa: PLC0415

        provs = ort.get_available_providers()
        print(f"      可用: {', '.join(provs)}")
        if any("CUDA" in p for p in provs):
            _mark(True, "CUDA 可用 —— YOLO 推理会走 GPU")
            green += 1
        else:
            _mark(False, "只有 CPU —— YOLO 会明显变慢（建议 pip install onnxruntime-gpu）", warn=True)
            yellow += 1
    except Exception as exc:  # noqa: BLE001
        _mark(False, f"onnxruntime 不可用: {exc}")
        red += 1

    # ⑤ 许可证提醒
    print(f"\n{C.BOLD}[5] 🔴 许可证检查（最重要）{C.OFF}")
    print(f"      {C.OK}N.E.K.O.{C.OFF}     Apache-2.0")
    print(f"      {C.ERR}ultralytics{C.OFF} AGPL-3.0  → 运行时 import 会传染")
    print(f"      {C.DIM}本包已确保 perception/yolo.py 不 import ultralytics{C.OFF}")
    try:
        import autopilot.perception.yolo as y  # noqa: PLC0415

        # 用 AST 精确判断有没有真正的 import 语句 —— 文本 grep 会把
        # 文档字符串里"禁止 import ultralytics"这句话误判成违规。
        src = Path(y.__file__).read_text(encoding="utf-8")
        bad = _imports_module(src, "ultralytics")
        if bad:
            _mark(False, f"❌ perception/yolo.py 导入了 {bad}")
            red += 1
        else:
            _mark(True, "perception/yolo.py 无 ultralytics 导入 ✓（AST 校验）")
            green += 1

        # 顺带扫整个运行时包，确认没有别处偷渡
        runtime_dir = Path(y.__file__).resolve().parents[1]
        leaks: list[str] = []
        for f in runtime_dir.rglob("*.py"):
            if f.name == "__init__.py" and f.parent.name == "tools":
                continue  # tools 是开发工具，允许 ultralytics
            if f.parent.name == "tools":
                continue
            m = _imports_module(f.read_text(encoding="utf-8", errors="replace"), "ultralytics")
            if m:
                leaks.append(f"{f.relative_to(runtime_dir)}")
        if leaks:
            _mark(False, f"❌ 运行时包里有 ultralytics 导入: {', '.join(leaks)}")
            red += 1
        else:
            print(f"      {C.OK}✓{C.OFF} 整个运行时包（不含 tools/）无 ultralytics 导入")
    except Exception as exc:  # noqa: BLE001
        _mark(False, f"无法校验 yolo.py: {exc}")
        red += 1

    # ⑥ 窗口 + 截图
    if not args.skip_capture:
        print(f"\n{C.BOLD}[6] 目标窗口与截图通道{C.OFF}")
        red_c, green_c, yellow_c = _check_capture(args.title)
        red += red_c
        green += green_c
        yellow += yellow_c
    else:
        print(f"\n{C.BOLD}[6] 截图实测{C.OFF}\n      {C.DIM}已跳过{C.OFF}")

    # ⑦ 配置
    print(f"\n{C.BOLD}[7] autopilot 配置{C.OFF}")
    try:
        from autopilot.config.loader import AUTOPILOT_ROOT, load_config  # noqa: PLC0415

        cfg_path = AUTOPILOT_ROOT / "config" / "defaults.toml"
        print(f"      {C.DIM}路径: {cfg_path}{C.OFF}")
        cfg = load_config(cfg_path)
        _mark(True, f"配置校验通过（enabled={cfg.enabled}）")
        green += 1
        print(f"      {C.DIM}模板 {len(cfg.templates)} 个 | 场景路由 {len(cfg.scene_routes)} 个"
              f" | YOLO {'开' if cfg.perception.yolo.enabled else '关'}{C.OFF}")
        if not cfg.perception.yolo.enabled:
            print(f"      {C.WARN}!{C.OFF} YOLO 未启用 —— 场景识别会退化为单模板探测")
            yellow += 1
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        # 「模板文件不存在」是首次运行的正常状态 —— 模板要用户自己裁。
        # 这里降级为警告并给步骤；其余配置错误仍然是硬失败（响亮失败）。
        if "模板文件不存在" in msg:
            missing = [ln.strip()[2:] for ln in msg.splitlines() if "模板文件不存在" in ln]
            _mark(False, f"还缺 {len(missing)} 个模板图（正常，裁完就能跑）", warn=True)
            for m in missing[:5]:
                print(f"      {C.DIM}{m}{C.OFF}")
            print(f"      {C.DIM}裁图：python -m autopilot.tools.capture_demo --title 原神 "
                  f"--save work/shot.png{C.OFF}")
            print(f"      {C.DIM}然后裁出目标，存到 autopilot/perception/templates/{C.OFF}")
            yellow += 1
            if all("模板文件不存在" in ln for ln in msg.splitlines() if ln.strip().startswith("-")):
                print(f"      {C.OK}✓{C.OFF} 除模板外配置无其他问题")
            else:
                print(f"      {C.ERR}✗ 还有其它配置问题：{C.OFF}")
                for ln in msg.splitlines():
                    if ln.strip().startswith("-") and "模板文件不存在" not in ln:
                        print(f"        {ln.strip()}")
                red += 1
        else:
            _mark(False, f"配置校验失败:\n      {msg}")
            red += 1

    # 汇总
    print(line("═"))
    print(f"\n  {C.OK}通过 {green}{C.OFF}   {C.WARN}警告 {yellow}{C.OFF}   {C.ERR}失败 {red}{C.OFF}\n")
    if red:
        print(f"{C.ERR}有 {red} 项失败，先修好再往下走。{C.OFF}")
        print("  常用修复：")
        print("    pip install -r autopilot/../requirements-autopilot.txt")
        print("    pip install onnxruntime-gpu   # CUDA 推理")
        print("    pip install dxcam             # 截图主通道")
        return 1
    if yellow:
        print(f"{C.WARN}可以跑，但有 {yellow} 项建议优化（看上面黄色项）。{C.OFF}\n")
    else:
        print(f"{C.OK}环境就绪，可以开始下一步。{C.OFF}")
        print(f"  下一步：python -m autopilot.tools.capture_demo --title {args.title}\n")
    return 0


# --------------------------------------------------------------------------


def _imports_module(src: str, module: str) -> str | None:
    """用 AST 判断源码里是否真的 import 了某个模块（顶层或包内任意层级）。

    文本 grep 会把文档字符串 / 注释里出现的模块名误判成导入，所以必须走 AST。
    返回模块名（发现违规时）或 None。
    """
    import ast  # noqa: PLC0415

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module or alias.name.startswith(module + "."):
                    return alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base == module or base.startswith(module + "."):
                return base
    return None


def _mark(ok: bool, msg: str, warn: bool = False) -> None:
    if ok:
        print(f"  {C.OK}✓{C.OFF} {msg}")
    elif warn:
        print(f"  {C.WARN}!{C.OFF} {msg}")
    else:
        print(f"  {C.ERR}✗{C.OFF} {msg}")


def _try_import(mod: str) -> tuple[bool, str]:
    name = {"cv2": "opencv-python", "win32gui": "pywin32"}.get(mod, mod)
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "")
        if not ver:
            try:
                ver = md.version(name)
            except Exception:  # noqa: BLE001
                ver = ""
        return True, str(ver)[:12]
    except Exception:  # noqa: BLE001
        return False, ""


def _check_capture(title: str) -> tuple[int, int, int]:
    red = green = yellow = 0
    from autopilot.perception.capturer import (  # noqa: PLC0415
        CaptureChannel,
        CaptureError,
        Capturer,
        WindowNotFoundError,
    )

    try:
        cap = Capturer(title)
        cap.open()
    except WindowNotFoundError:
        print(f"  {C.ERR}✗{C.OFF} 找不到标题含 {title!r} 的窗口")
        print(f"      {C.DIM}请确认游戏已启动、未最小化；用 --title 换关键词{C.OFF}")
        return red + 1, green, yellow
    except CaptureError as exc:
        print(f"  {C.ERR}✗{C.OFF} 截图器初始化失败:\n      {exc}")
        return red + 1, green, yellow

    try:
        rx, ry, rw, rh = cap.window_rect or (0, 0, 0, 0)
        print(f"      窗口位置: ({rx}, {ry})  尺寸: {rw}x{rh}")
        if rw <= 0 or rh <= 0:
            print(f"  {C.ERR}✗{C.OFF} 窗口尺寸为 0（可能最小化了）")
            return red + 1, green, yellow
    finally:
        cap.close()

    # 逐通道实测
    for ch in CaptureChannel:
        c = Capturer(title, channel_order=[ch])
        try:
            c.open()
            # 预热后测 5 帧
            c.grab()
            samples: list[float] = []
            for _ in range(5):
                t0 = time.perf_counter()
                f = c.grab()
                samples.append((time.perf_counter() - t0) * 1000)
            samples.sort()
            p50, p95 = samples[len(samples) // 2], samples[-1]
            verdict = "✓" if p95 < 30 else "!"
            color = C.OK if p95 < 30 else C.WARN
            print(f"  {color}{verdict}{C.OFF} {ch.value:12s} "
                  f"{f.width}x{f.height}  P50={p50:.1f}ms  P95={p95:.1f}ms")
            if p95 < 30:
                green += 1
            else:
                yellow += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  {C.WARN}!{C.OFF} {ch.value:12s} 不可用: {str(exc)[:80]}")
            yellow += 1
        finally:
            c.close()

    print(f"      {C.DIM}截图延迟是感知层的硬预算：tick_ms 超过 100ms 就该优化了{C.OFF}")
    return red, green, yellow


if __name__ == "__main__":
    raise SystemExit(main())
