"""主循环依赖体检。**不启动循环**，只逐环检查 + 试抓一帧。

    python -m autopilot.tools.loop_diagnose
    python -m autopilot.tools.loop_diagnose --title 原神
    python -m autopilot.tools.loop_diagnose --json

什么时候跑它
------------
* ``run_loop`` 跑不起来时，先跑这个 —— 它会告诉你**哪一环**挂了
* 改完配置 / 换机器 / 升依赖之后，跑一次确认没坏
* ``--live`` 之前跑一次：执行层那条「进程完整性 MEDIUM + 游戏以管理员
  运行」是最常见的失败原因，这里会提前指出来

九项检查：Python 环境 / 依赖 / 配置 / 窗口 / 截图器 / 感知层 /
决策层 / 执行层 / 急停键 / 主循环配置 / 总判定。

⚠ 本工具**会真的抓一帧**（截图器那一项），抓完立刻释放。**不发任何按键**。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import (  # noqa: E402
        C, bootstrap, line, setup_logging,
    )
else:
    from ._bootstrap import C, bootstrap, line, setup_logging

#: autopilot 包根（YOLO 模型路径相对它解析）
AUTOPILOT_ROOT = Path(__file__).resolve().parents[1]

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    hint: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "detail": self.detail, "hint": self.hint}


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", hint: str = "") -> None:
        self.checks.append(Check(name, status, detail, hint))

    @property
    def verdict(self) -> str:
        if any(c.status == FAIL for c in self.checks):
            return "不可运行"
        if any(c.status == WARN for c in self.checks):
            return "部分可用"
        return "可运行"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "counts": {s: sum(1 for c in self.checks if c.status == s)
                       for s in (OK, WARN, FAIL, SKIP)},
            "checks": [c.to_dict() for c in self.checks],
        }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="主循环依赖体检（不启动循环，不发按键）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--title", default="", help="窗口匹配关键字")
    ap.add_argument("--rules-dir", default="", help="规则目录")
    ap.add_argument("--skip-capture", action="store_true",
                    help="跳过截图测试（无窗口 / 远程桌面时用）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    r = Report()
    cfg = _check_env(r, args)

    cap = None
    if cfg is not None:
        cap = _check_capture(r, cfg, args)
        _check_perception(r, cfg)
        _check_policy(r, args)
    _check_actuator(r)
    _check_panic(r)
    _check_loop_cfg(r)

    if args.json:
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
        return 0 if r.verdict == "可运行" else 1
    _print_report(r)
    return 0 if r.verdict == "可运行" else 1


# --------------------------------------------------------------------------
def _describe_config_error(exc: BaseException, verbose: bool) -> tuple[str, str]:
    """把配置加载异常拆成「摘要 + 完整明细」。

    ★ 早先这里写的是 ``str(exc).splitlines()[0]`` —— 只留第一行，
      而 loader 的错误形如
      ``配置校验失败（3 项）：\\n  - 项1\\n  - 项2\\n  - 项3``，
      于是 3 条明细全被丢掉，用户只看到「3 项」却不知道是哪 3 项。
      **报错工具把最关键的信息截断，比不报错更糟**（原则 1）。

    :returns: (detail, hint)
    """
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    head = lines[0] if lines else f"{type(exc).__name__}: (空消息)"

    # ① pydantic ValidationError —— 逐字段展开 loc / type / msg / 实际值
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            items = errors()
        except Exception:  # noqa: BLE001
            items = []
        if items:
            detail = head
            for e in items[:20]:
                loc = ".".join(str(x) for x in e.get("loc", ())) or "<根>"
                detail += (f"\n      · {loc}"
                           f"\n        类型  {e.get('type', '?')}"
                           f"\n        原因  {e.get('msg', '?')}"
                           f"\n        收到  {e.get('input', '?')!r}")
            return detail, "字段路径对不上 defaults.toml —— 改 TOML，别改 schema"
        if type(exc).__name__ == "ValidationError":
            return "\n      ".join(lines), "pydantic 校验失败"

    # ② loader 的自定义校验（ValueError，明细在第二行起，每行以 "- " 开头）
    items = [ln.lstrip("- ").strip() for ln in lines[1:]
             if ln.lstrip().startswith("- ")]
    if items:
        detail = head
        for i, it in enumerate(items, 1):
            detail += f"\n      {i}. {it}"
        return detail, ("上面每一条都是独立的配置问题，逐个修完再跑。"
                        "模板图缺失的话，按 defaults.toml 里的说明裁："
                        "capture_demo --save shot.png → 裁剪 → 存到 "
                        "perception/templates/。")

    # ③ 兜底：全文给出，绝不只留第一行
    return "  ".join(lines) if lines else head, ""


def _check_env(r: Report, args) -> object | None:
    """Python / 依赖 / 配置。返回 cfg（失败时为 None）。"""
    v = sys.version_info
    py_ok = (3, 11) <= (v.major, v.minor) < (3, 14)
    r.add("Python 环境", OK if py_ok else WARN,
          f"{v.major}.{v.minor}.{v.micro}  {platform.system()} {platform.machine()}",
          "" if py_ok else "本项目在 3.11 上开发，3.14 未验证")

    deps = []
    for mod, why in [("numpy", "数值"), ("cv2", "图像"),
                     ("mss", "截图兜底通道"), ("onnxruntime", "YOLO 推理"),
                     ("pydantic", "配置校验"), ("win32gui", "窗口查找")]:
        try:
            __import__(mod)
            deps.append((mod, True, why))
        except ImportError:
            deps.append((mod, False, why))
    missing = [m for m, g, _ in deps if not g]
    r.add("依赖", OK if not missing else WARN,
          "  ".join(f"{m}{'✓' if g else '✗'}" for m, g, _ in deps),
          "" if not missing else
          f"缺 {missing} —— pip install -r requirements-autopilot.txt")

    try:
        from autopilot.config.loader import load_config  # noqa: PLC0415

        cfg = load_config()
        if args.title:
            cfg.capture.window_title_substr = args.title
        n_tpl = len(cfg.templates)
        # tick 率取自 RuntimeContext 的默认值，不读 cfg.loop ——
        # 后者是否存在取决于 schema 状态，引用它会让体检工具自己崩掉。
        from autopilot.core.runtime import RuntimeContext as _RC

        r.add("配置", OK,
              f"{cfg.game_id}，enabled={cfg.enabled}，"
              f"窗口「{cfg.capture.window_title_substr}」，"
              f"模板 {n_tpl} 个，loop {_RC().tick_rate_hz:g}Hz")
        if n_tpl == 0:
            r.add("配置 · 场景识别", WARN, "没有启用任何模板",
                  "场景识别基本认不出东西，主循环会大量空转（预期行为）。"
                  "要真跑起来先裁模板图：capture_demo --save shot.png → "
                  "裁剪 → 存到 perception/templates/，再到 defaults.toml "
                  "里取消注释对应的段。")
        return cfg
    except Exception as exc:  # noqa: BLE001
        detail, hint = _describe_config_error(exc, args.verbose)
        r.add("配置", FAIL, detail, hint)
        return None


def _check_capture(r: Report, cfg, args):
    """窗口 + 截图器 + 试抓一帧。抓完立刻释放。"""
    title = cfg.capture.window_title_substr
    cap = None
    try:
        from autopilot.perception.capturer import Capturer  # noqa: PLC0415

        cap = Capturer(title, channel_order=list(cfg.capture.channel_order))
        cap.open()
        rect = getattr(cap, "window_rect", None)
        r.add("窗口", OK if rect else FAIL,
              f"「{title}」{'已找到' if rect else '未找到'}"
              + (f"，矩形 {rect}" if rect else ""),
              "" if rect else
              "启动游戏后再跑；或用 --title 换关键字（任务管理器→详细信息里看）")
        r.add("截图器", OK, f"通道 {cap.get_active_channel().value}")
    except Exception as exc:  # noqa: BLE001
        r.add("窗口", FAIL, f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
              "找不到窗口 / 三条通道都不可用。先跑 check_env.py 看细节")
        r.add("截图器", FAIL, "未能初始化")
        return None

    if args.skip_capture:
        r.add("抓帧测试", SKIP, "--skip-capture")
    else:
        t0 = time.perf_counter()
        try:
            frame = cap.grab()
            ms = (time.perf_counter() - t0) * 1000
            good = frame.height > 0 and frame.width > 0
            r.add("抓帧测试", OK if good else FAIL,
                  f"{frame.width}x{frame.height}，{ms:.1f}ms，序号 {frame.frame_seq}",
                  "" if ms < 100 else
                  f"{ms:.0f}ms 偏慢 —— 5Hz 每帧只有 200ms 预算")
        except Exception as exc:  # noqa: BLE001
            r.add("抓帧测试", FAIL, f"{type(exc).__name__}: {exc}",
                  "试试 --skip-capture，或跑 check_env.py 看通道详情")
    try:
        cap.close()
    except Exception:  # noqa: BLE001
        pass
    return cap


def _check_perception(r: Report, cfg) -> None:
    """感知层：模板 + YOLO。**不 import ultralytics**。"""
    try:
        specs = cfg.template_specs()
        if specs:
            from autopilot.perception.template import TemplateMatcher  # noqa: PLC0415

            TemplateMatcher(specs, roi_expressions=cfg.roi_expressions(),
                            roi_vars=cfg.roi_var_values())
            r.add("感知层 · 模板", OK, f"{len(specs)} 个模板")
        else:
            r.add("感知层 · 模板", WARN, "配置里没有 templates",
                  "场景识别会退化成只用 YOLO。用 template_demo.py 裁几张图")
    except Exception as exc:  # noqa: BLE001
        r.add("感知层 · 模板", FAIL, f"{type(exc).__name__}: {exc}")

    y = cfg.perception.yolo
    if not y.enabled:
        r.add("感知层 · YOLO", WARN, "未启用（enabled=false）",
              "没模型时场景识别只能靠模板/OCR。"
              "训好后跑 python -m autopilot.tools.train_yolo")
        return
    try:
        p = Path(y.onnx_path)
        if not p.is_absolute():
            p = AUTOPILOT_ROOT / y.onnx_path
        r.add("感知层 · YOLO", OK if p.exists() else FAIL,
              f"启用，模型 {p.name}" + ("" if p.exists() else "（文件不存在）"),
              "" if p.exists() else
              f"先训练并导出：python -m autopilot.tools.train_yolo")
    except Exception as exc:  # noqa: BLE001
        r.add("感知层 · YOLO", FAIL, f"{type(exc).__name__}: {exc}")


def _check_policy(r: Report, args) -> None:
    """决策层：规则加载。**走工厂**，不直接 import LocalPolicyTree。"""
    try:
        from autopilot.tools._stack import build_policy  # noqa: PLC0415

        pol, rdir, n = build_policy(args.rules_dir or None)
        by_scene: dict[str, int] = {}
        for rule in getattr(pol, "rules", ()):
            for s in (rule.scenes or ("(不限)",)):
                by_scene[s] = by_scene.get(s, 0) + 1
        dist = "，".join(f"{k}×{v}" for k, v in sorted(by_scene.items()))
        r.add("决策层", OK,
              f"{type(pol).__name__}，{n} 条规则（{rdir}）"
              + (f"  分布 {dist}" if dist else ""))
    except Exception as exc:  # noqa: BLE001
        r.add("决策层", FAIL, f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
              "规则加载失败。报错里有文件名 + 行号 + 规则名")


def _check_actuator(r: Report) -> None:
    """执行层：干跑后端 + 真实 SendInput 探测 + 进程完整性。"""
    try:
        from autopilot.actuator import DryRunBackend, InputSimulator  # noqa: PLC0415

        InputSimulator(DryRunBackend())
        r.add("执行层 · 干跑", OK, "DryRunBackend 可用（不发按键）")
    except Exception as exc:  # noqa: BLE001
        r.add("执行层 · 干跑", FAIL, f"{type(exc).__name__}: {exc}")
        return

    try:
        from autopilot.actuator.simulator import (  # noqa: PLC0415
            Win32Backend, probe_backend, process_integrity,
        )

        rep = probe_backend(Win32Backend())
        rid, label = process_integrity()
        where = f"（rid=0x{rid:04X}）" if rid is not None else ""
        if not rep.ok:
            r.add("执行层 · SendInput", FAIL,
                  f"不可用：{rep.problems[0] if rep.problems else '未知'}",
                  "以管理员身份运行；关掉安全软件的键鼠防护")
        elif label in ("MEDIUM", "LOW"):
            r.add("执行层 · SendInput", WARN,
                  f"可用，但进程完整性 {label} {where}",
                  f"若游戏以管理员运行，SendInput 会被 UIPI 拦截。"
                  f"以管理员身份重跑本工具再 --live。")
        else:
            r.add("执行层 · SendInput", OK,
                  f"可用，进程完整性 {label} {where}")
    except Exception as exc:  # noqa: BLE001
        r.add("执行层 · SendInput", FAIL, f"{type(exc).__name__}: {exc}")


def _check_panic(r: Report) -> None:
    """急停键：确认监听器装得上。"""
    try:
        from autopilot.actuator import PANIC_KEY  # noqa: PLC0415
        from autopilot.actuator.safety import SafetyGuard  # noqa: PLC0415

        pressed = SafetyGuard().panic_active      # 读一次，确认 API 通
        r.add("急停键", OK,
              f"{PANIC_KEY.name}（VK=0x{int(PANIC_KEY):02X}），监听器可安装，"
              f"当前按下={pressed}")
    except Exception as exc:  # noqa: BLE001
        r.add("急停键", FAIL, f"{type(exc).__name__}: {exc}",
              "没有急停键就没有最后一道保险 —— 别用 --live")


def _check_loop_cfg(r: Report) -> None:
    try:
        from autopilot.core.runtime import RuntimeContext  # noqa: PLC0415

        c = RuntimeContext()
        r.add("主循环配置", OK,
              f"默认 {c.tick_rate_hz:g} Hz"
              f"（{c.target_interval_s * 1000:.0f}ms/帧），"
              f"连续截图失败上限 {c.max_consecutive_capture_failures}")
    except Exception as exc:  # noqa: BLE001
        r.add("主循环配置", FAIL, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
def _print_report(r: Report) -> None:
    icon = {OK: (C.OK, "✓"), WARN: (C.WARN, "⚠"),
            FAIL: (C.ERR, "✗"), SKIP: (C.DIM, "–")}
    print(f"\n{C.BOLD}主循环体检{C.OFF}")
    print(f"{line()}")
    for c in r.checks:
        col, mark = icon[c.status]
        print(f"  {col}{mark}{C.OFF} {c.name:<16} {c.detail}")
        if c.hint:
            print(f"    {C.DIM}{c.hint}{C.OFF}")
    print(f"{line()}")
    col, mark, text = {
        "可运行": (C.OK, "✓", "可运行"),
        "部分可用": (C.WARN, "⚠", "部分可用（有告警）"),
        "不可运行": (C.ERR, "✗", "不可运行"),
    }[r.verdict]
    print(f"  总体：{col}{mark} {text}{C.OFF}")
    if r.verdict != "可运行":
        print(f"  {C.DIM}先修 ✗ 的项；✗ 全清掉再跑 run_loop --live。{C.OFF}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
