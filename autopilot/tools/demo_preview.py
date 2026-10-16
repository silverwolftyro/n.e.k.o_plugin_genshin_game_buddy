"""离线 demo 预演 —— **不开原神**走一遍完整决策链。

> ``python -m autopilot.tools.demo_preview --image <截图>``

★ 这个脚本输出的是**模拟**日志，不是真实运行记录
  ------------------------------------------------
  它**真的**跑了：感知链路（``StateBuilder``）、规则引擎（``LocalPolicyTree``）、
  事件流写入、YUI 台词渲染（``EventToYui``）。
  它**没有**跑：截图抓取（WGC 抓的是你给的静态图）、键鼠注入、真实游戏状态。

  所以它能证明「**管道是通的**」，证明不了「**水是干净的**」。
  演示当天可以拿它当暖场，**不能**拿它当证据 —— 见 ``--verbose`` 里
  每个阶段标注的「真机 / 模拟」。

★ 为什么需要它
  演示之前想看一眼「流程长什么样、会不会卡在某一步」，就得开原神 + 管理员
  + 游戏前台。预演把这部分成本降到零：任何一张截图都能跑。

用法::

    # 用仓库自带的那张真机截图 + 默认规则
    python -m autopilot.tools.demo_preview

    # 指定截图
    python -m autopilot.tools.demo_preview --image D:\\shots\\frame.png

    # 连「未来规则」一起评估（会显示它们为什么没命中）
    python -m autopilot.tools.demo_preview --rule-set all

    # 自定义规则目录
    python -m autopilot.tools.demo_preview --rule-set custom:D:\\my_rules

    # 打印每一帧的 StatePacket 全字段
    python -m autopilot.tools.demo_preview --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 输出格式

_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "mag": "\033[35m", "cyan": "\033[36m",
}


def _c(text: str, *styles: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(_C[s] for s in styles) + text + _C["reset"]


OK = lambda s: _c(s, "green")          # noqa: E731
WARN = lambda s: _c(s, "yellow")       # noqa: E731
ERR = lambda s: _c(s, "red")           # noqa: E731
DIM = lambda s: _c(s, "dim")           # noqa: E731
HEAD = lambda s: _c(s, "bold", "cyan")  # noqa: E731

SEP = "─" * 68


def _step(n: int, title: str, kind: str) -> None:
    """``kind`` 说明这一步是「真组件」还是「模拟」。"""
    tag = OK("[真组件]") if kind == "real" else WARN("[模拟]")
    print(f"\n{HEAD(f'[{n}] {title}')}  {tag}")


# ---------------------------------------------------------------- 规则集

def _resolve_rules(rule_set: str) -> tuple[Path, str]:
    """返回 ``(规则目录, 说明)``。

    ★ ``default`` 与 ``all`` 的区别
      ``_future/`` 里的规则**永不加载**（``rule_loader`` 用 ``glob`` 非递归），
      而它们引用的信号今天恒 UNKNOWN —— 全导进来只会让「为什么没命中」变得
      更难读。``all`` 把它们也加载上，好让预演能展示「差什么信号」。
    """
    base = Path(__file__).resolve().parents[1] / "rules"
    if rule_set == "default":
        return base, "默认（仅顶层，装机实际加载的那批）"
    if rule_set == "all":
        return base, "全部（顶层 + _future/，后者今天恒不命中）"
    if rule_set.startswith("custom:"):
        p = Path(rule_set[len("custom:"):].strip())
        if not p.is_dir():
            raise SystemExit(f"[X] 规则目录不存在：{p}")
        return p, f"自定义（{p}）"
    raise SystemExit(f"[X] 未知 rule-set：{rule_set!r}（default / all / custom:<路径>）")


# ---------------------------------------------------------------- 阶段

def _stage_sense(image: Path, verbose: bool) -> "object":
    import cv2

    from autopilot.perception.builder import StateBuilder
    from autopilot.perception.capturer import CapturedFrame
    from autopilot.perception.scene_detector import SceneDetector

    bgr = cv2.imread(str(image))
    if bgr is None:
        raise SystemExit(f"[X] 图片解码失败（不是图片？）：{image}")
    h, w = bgr.shape[:2]
    print(f"  截图 {w}×{h}  {DIM(str(image))}")

    frame = CapturedFrame(image=bgr, rect=(0, 0, w, h),
                          timestamp_ms=int(time.time() * 1000),
                          frame_seq=1, channel=None)
    builder = StateBuilder(entity_detector=None,
                           scene_detector=SceneDetector(n_seconds=0.0,
                                                       m_seconds=0.0))
    t0 = time.perf_counter()
    state = builder.build(frame)
    ms = (time.perf_counter() - t0) * 1000

    print(f"  感知耗时 {ms:.1f} ms")
    print(f"  场景     {OK(state.scene.id)}"
          f"   (置信度 {state.scene.confidence:.2f})")
    print(f"  资源     perception_ok={OK('True') if state.resource.perception_ok else ERR('False')}")
    if verbose:
        print(DIM("  ── StatePacket 全字段 ──"))
        raw = state.model_dump(mode="json")
        for line in json.dumps(raw, ensure_ascii=False,
                               indent=2).splitlines():
            print(DIM("  " + line))
    return state


def _stage_decide(state, rules_dir: Path, note: str) -> object | None:
    from autopilot.policy.local_tree import LocalPolicyTree

    tree = LocalPolicyTree(rules_dir=rules_dir)
    intent = tree.poll(state)
    total = len(tree.rules)
    print(f"  规则集   {note}")
    print(f"  规则数   {total}")

    trace = tree.decide(state)
    if intent is None:
        print(f"  结果     {WARN('无规则命中 -> 不产生意图')}")
        print(DIM("  唯一 enabled 的规则是 world_attack_only_with_enemy，"))
        print(DIM("  条件 has_nearby_enemy() 来自实体检出（YOLO + DirectML），"))
        print(DIM("  离线拿不到 —— 所以这里必然不命中。**这是预演的已知边界**，"))
        print(DIM("  不是 bug。真机上这一步会命中（历史记录 1352 次真实按键）。"))
        print(DIM(f"  逐条判定：{len(getattr(trace, 'rules', []) or [])} 条已评估"
                 if hasattr(trace, "rules") else f"  逐条判定：{trace}"))
    else:
        print(f"  命中     {OK(intent.action.value)}"
              f"   priority={intent.priority}")
        print(f"          params={intent.params}")
    return intent


def _stage_execute(intent) -> str | None:
    """模拟执行：**只打印会按什么键**，真的不发。"""
    from autopilot.actuator.keymap import DEFAULT_KEYMAP

    if intent is None:
        print("  没有意图 -> 不会按任何键")
        return None
    key = DEFAULT_KEYMAP.get(intent.action.value)
    label = key.name if key is not None else "（该动作未配键位）"
    print(f"  将会执行 {OK(label)}   {DIM('← 只是打印，没有真的按键')}")
    return str(label)


def _stage_speak(runtime: Path, codes: list[tuple[str, str]]) -> list[str]:
    from event_to_yui import EventToYui

    runtime.mkdir(parents=True, exist_ok=True)
    run_id = "demo_preview"
    (runtime / "run_current.txt").write_text(run_id, encoding="utf-8")

    said: list[str] = []

    def _push(**kw):
        said.append("".join(p.get("text", "") for p in kw.get("parts", [])))
        return {"submitted": True}

    for code, msg in codes:
        with (runtime / f"{run_id}.stream.jsonl").open(
                "a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"code": code, "message": msg,
                 "ts_ms": int(time.time() * 1000)},
                ensure_ascii=False) + "\n")

    EventToYui(push_fn=_push, runtime_dir=runtime).drain_once()
    if not said:
        print("  这些事件码不在台词表里，YUI 不会出声")
        print(DIM("  （只有 6 个业务码会播报：PATHFIND_* / ENEMY_DETECTED / "
                   "ESCAPE_SUCCESS / POSITION_CHANGED）"))
    for line in said:
        print(f"  YUI 说   {OK('“' + line + '”')}")
    return said


# ---------------------------------------------------------------- 主流程

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="离线 demo 预演（不开原神，纯模拟）")
    ap.add_argument("--image", default="", help="截图路径（默认用自带的真机截图）")
    ap.add_argument("--rule-set", default="default",
                    help="default / all / custom:<路径>")
    ap.add_argument("--verbose", action="store_true", help="打印 StatePacket 全字段")
    args = ap.parse_args(argv)

    default_img = (Path(__file__).resolve().parents[1] / "_runtime"
                   / "capdiag_wgc.png")
    image = Path(args.image) if args.image else default_img
    if not image.is_file():
        print(f"[X] 找不到截图：{image}")
        print("    用 --image 指定，或跑一次真机截图生成它")
        return 1

    rules_dir, note = _resolve_rules(args.rule_set)
    if not rules_dir.is_dir():
        print(f"[X] 找不到规则目录：{rules_dir}")
        return 1

    print(SEP)
    print(HEAD("N.E.K.O 离线 demo 预演"))
    print(WARN("以下全是模拟：没有开游戏、没有抓实时画面、没有注入按键"))
    print(DIM("能证明管道通，证明不了功能好 —— 别拿它当证据"))
    print(SEP)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="demo_preview_") as tmp:
        runtime = Path(tmp)

        _step(1, "截图 → 感知 → StatePacket", "real")
        state = _stage_sense(image, args.verbose)

        _step(2, "StatePacket → 规则求值", "real")
        intent = _stage_decide(state, rules_dir, note)

        _step(3, "意图 → 执行", "模拟")
        _stage_execute(intent)

        _step(4, "事件流 → YUI 台词", "real")
        codes = [("CORE_STARTED", "离线预演启动"),
                 ("SCENE_CHANGED",
                  f"场景切到 {state.scene.id}")]
        if intent is not None:
            codes.append(("INTENT_EXECUTED", intent.action.value))
        codes.append(("RUN_STOPPED", "离线预演结束"))
        _stage_speak(runtime, codes)

    print(f"\n{SEP}")
    print(HEAD("预演结束"))
    print(DIM(f"事件流在临时目录里，已随临时目录一起清理：{runtime}"))
    print(DIM("要留住证据请改用真机跑一次，并把 _runtime/ 里的 .stream.jsonl 留下"))
    print(SEP)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
