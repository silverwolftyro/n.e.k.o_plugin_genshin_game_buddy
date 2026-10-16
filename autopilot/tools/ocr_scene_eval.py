"""用真实 OCR 采样评估场景分类器（2026-10-12 任务 B）。

    python -m autopilot.tools.ocr_scene_eval --log-dir "<插件日志目录>"

⚠️ 先说清楚这个工具**算不出准确率**，以及为什么
------------------------------------------------
真实采样来自插件运行日志的 ``ocr_perceive sample ... text_head='...'`` 行。
1486 条真实样本（2026-10-10 / 10-11 两份日志）。

但**每条样本旁边没有任何场景标签**。原因不是日志坏了，而是那两天跑的
还是**二值**场景检测：

    ocr_perceive state=IN_GAME|UNKNOWN|AWAY    <- 这是「游戏是否在前台」
    ocr_perceive ... scenes=0                   <- 这是一个计数
    BAG_INVENTORY / MAP_BIG / DIALOG / COMBAT  <- 全日志出现 **0 次**

所以**没有 ground truth，准确率无法计算**。硬凑一个数字（比如拿
``state=IN_GAME`` 当标签）只会得到一个看起来精确、实际无意义的值。

于是本工具只报**能诚实报的指标**：

  1. **触发率** —— 分类器判出非 fallback 的比例。这是**误报面**：
     判错成 BAG_INVENTORY 会让引擎在野外疯狂按 B 键，是最贵的错。
  2. **触发样本的命中关键词** —— 人工可复核。
  3. **按「游戏是否在前台」分层** —— 非前台的样本几乎必然不该判出 UI 场景。
  4. **文本长度/内容异常** —— 日志里混着**桌面截图**（能读到「回收站」
     「控制面板」「新建文件夹」），那是抓屏抓到了 Windows 桌面。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, line, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, line, setup_logging

#: 采样行：``ocr_perceive sample (scenes=0) text_head='...``
SAMPLE_RE = re.compile(r"ocr_perceive sample \(scenes=(\d+)\) text_head='(.*)$")
#: 紧随其后的状态行：``ocr_perceive state=UNKNOWN hit=True scenes=0 active=25``
STATE_RE = re.compile(
    r"ocr_perceive state=(\S+?) hit=(\S+?) scenes=(\d+) active=(\d+)")
#: 时间戳前缀 ``2026-10-11 00:00:23``
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

#: Windows 桌面上会出现的字。日志里确实抓到了这些 —— 说明**抓屏抓到了桌面**，
#: 这类样本不该判出任何游戏 UI 场景。
DESKTOP_MARKERS = (
    "回收站", "控制面板", "新建文件夹", "此电脑", "网络", "文档",
    "下载", "回收", "Microsoft", "Recycle", "Control Panel",
)


@dataclass
class Sample:
    """一条真实 OCR 采样。"""

    ts: str
    log: str
    text: str
    state: str = ""          # IN_GAME / UNKNOWN / AWAY
    scenes: int = -1

    @property
    def looks_like_desktop(self) -> bool:
        return any(m in self.text for m in DESKTOP_MARKERS)


def parse_log(path: Path) -> list[Sample]:
    """从一份日志里抽出所有 OCR 采样，并把状态行配对上。

    配对方式：采样行的**下一条非空行**里找 ``state=``。日志是严格串行写的
    （一个 tick 一段），所以顺序配对是可靠的。
    """
    out: list[Sample] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    pending: Sample | None = None
    for raw in lines:
        m = SAMPLE_RE.search(raw)
        if m:
            ts = TS_RE.match(raw)
            pending = Sample(
                ts=ts.group(1) if ts else "",
                log=path.name,
                text=m.group(2).rstrip("'"),
                scenes=int(m.group(1)),
            )
            out.append(pending)
            continue
        if pending is not None:
            sm = STATE_RE.search(raw)
            if sm:
                pending.state = sm.group(1)
                pending.scenes = int(sm.group(3))
                pending = None
    return out


#: 默认只收**本插件**的日志。目录里还有 multi_game_companion 等别的插件，
#: 它们也打 ``ocr_perceive sample``，混进来会把样本量撑大好几倍，
#: 而那些采样的游戏/UI 与原神无关 —— 关键词表完全对不上。
DEFAULT_PATTERN = "N.E.K.O_Plugin_genshin_game_buddy_*.log"


def collect(log_dir: Path, pattern: str = DEFAULT_PATTERN) -> list[Sample]:
    """扫日志目录，收齐所有 OCR 采样。"""
    out: list[Sample] = []
    for p in sorted(log_dir.glob(pattern)):
        if "error" in p.name:
            continue
        got = parse_log(p)
        if got:
            print(f"  {p.name:<46} {len(got):>5} 条")
        out.extend(got)
    return out


def evaluate(samples: list[Sample]) -> dict:
    """跑分类器，汇总能诚实报的指标。"""
    from autopilot.perception.scene_from_ocr import (
        SCENE_KEYWORDS,
        classify_scene_from_text,
    )

    labels: Counter = Counter()
    keywords: Counter = Counter()
    fired = []
    by_state: dict[str, Counter] = {}

    for s in samples:
        # 非前台的样本强制回落 —— 游戏不在前台时不该判 UI 场景
        fallback = "UNKNOWN" if s.state in ("AWAY", "UNKNOWN", "") else "WORLD"
        g = classify_scene_from_text(s.text, fallback=fallback)
        labels[g.scene_id] += 1
        by_state.setdefault(s.state or "(无)", Counter())[g.scene_id] += 1
        if g.scene_id not in ("WORLD", "UNKNOWN"):
            fired.append((s, g))
            keywords[g.matched] += 1

    desktop = [s for s in samples if s.looks_like_desktop]
    desktop_fired = [s for s, _ in fired if s.looks_like_desktop]

    # 逐关键词出现次数 —— 判断「这批样本到底有没有 UI 场景」的唯一硬证据。
    # 实战教训：日志里 state=IN_GAME 的样本，绝大多数抓到的是 **Windows 桌面 /
    # 文件管理器**（capture_path=foreground + mss 抓的是前台窗口，而前台是资源
    # 管理器），文本是 "N.E.KO-main > plugin > plugins > ..."、新建文件夹、迅雷。
    # 那种样本里出现「设置」「选项」全是误命中，跟游戏 UI 无关。
    kw_hits: dict[str, Counter] = {}
    for s in samples:
        for scene, words in SCENE_KEYWORDS.items():
            for w in words:
                if w in s.text:
                    kw_hits.setdefault(scene, Counter())[w] += 1

    return {
        "total": len(samples),
        "labels": labels,
        "keywords": keywords,
        "fired": fired,
        "by_state": by_state,
        "desktop_total": len(desktop),
        "desktop_fired": desktop_fired,
        "scene_keywords": SCENE_KEYWORDS,
        "kw_hits": kw_hits,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="用真实 OCR 采样评估场景分类器（只报告，不调阈值）")
    ap.add_argument("--log-dir", default="",
                    help="插件日志目录（含 N.E.K.O_Plugin_*.log）")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help=f"日志文件名 glob（默认只收本插件：{DEFAULT_PATTERN}）")
    ap.add_argument("--show", type=int, default=15,
                    help="打印多少条触发样本明细（默认 15）")
    ap.add_argument("--json", default="", help="把汇总写成 JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    log_dir = Path(args.log_dir) if args.log_dir else Path.home() / ".neko"
    if not log_dir.is_dir():
        print(f"{C.ERR}✗ 日志目录不存在: {log_dir}{C.OFF}")
        return 1

    print(f"{C.BOLD}扫描日志{C.OFF}  {log_dir}")
    samples = collect(log_dir, args.pattern)
    if not samples:
        print(f"{C.WARN}一条 OCR 采样都没找到{C.OFF}")
        return 2

    r = evaluate(samples)
    n = r["total"]

    stamps = sorted(s.ts for s in samples if s.ts)

    print(line("═"))
    print(f"{C.BOLD}样本{C.OFF}  共 {n} 条"
          f"   （{stamps[0]} ~ {stamps[-1]}）")
    print(line("═"))

    # ── 1. 能不能算准确率 ──────────────────────────────────
    print(f"\n{C.WARN}准确率{C.OFF}: {C.BOLD}无法计算{C.OFF}")
    print("  日志里每条采样旁边**没有场景标签**。那两天跑的还是二值检测：")
    print("    state= 是「游戏是否在前台」（IN_GAME/UNKNOWN/AWAY），不是场景")
    print("    scenes=N 是计数，不是标签")
    print("    BAG_INVENTORY / MAP_BIG / DIALOG / COMBAT 在全日志出现 0 次")
    print("  没有 ground truth，硬凑的数字没有意义。所以下面只报能报的东西。")

    # ── 2. 标签分布 ────────────────────────────────────────
    print(line("═"))
    print(f"\n{C.BOLD}标签分布{C.OFF}")
    for lab, c in r["labels"].most_common():
        pct = 100.0 * c / n
        tag = "" if lab in ("WORLD", "UNKNOWN") else f"  {C.WARN}<- 触发{C.OFF}"
        print(f"  {lab:<16} {c:>5}  {pct:6.2f}%{tag}")

    fired = r["fired"]
    print(f"\n  触发（非回落）: {len(fired)} / {n} = "
          f"{100.0 * len(fired) / max(1, n):.2f}%")

    # ── 3. 命中关键词 ──────────────────────────────────────
    if r["keywords"]:
        print(line("═"))
        print(f"\n{C.BOLD}触发样本的命中关键词{C.OFF}")
        for kw, c in r["keywords"].most_common():
            print(f"  {kw:<12} {c:>5}")

    # ── 4. 分层 ────────────────────────────────────────────
    print(line("═"))
    print(f"\n{C.BOLD}按「游戏是否在前台」分层{C.OFF}")
    for st, c in sorted(r["by_state"].items()):
        tot = sum(c.values())
        nonfb = tot - c.get("WORLD", 0) - c.get("UNKNOWN", 0)
        print(f"  state={st:<12} n={tot:>5}  触发={nonfb:>4} "
              f"({100.0 * nonfb / max(1, tot):5.2f}%)  {dict(c)}")

    # ── 5. 桌面样本 ────────────────────────────────────────
    print(line("═"))
    print(f"\n{C.BOLD}桌面污染样本{C.OFF}")
    print(f"  含「回收站/控制面板/新建文件夹」等 Windows 桌面文字: "
          f"{r['desktop_total']} 条")
    if r["desktop_fired"]:
        print(f"  {C.ERR}其中 {len(r['desktop_fired'])} 条被判成了 UI 场景 —— "
              f"这是确定的误报{C.OFF}")
        for s in r["desktop_fired"][:5]:
            print(f"    · {s.ts} state={s.state}  {s.text[:80]!r}")
    else:
        print(f"  {C.OK}没有被判成 UI 场景 —— 关键词表对桌面文字是干净的{C.OFF}")

    # ── 6. 样本代表性：这批数据到底能不能验证分类器 ────────────
    print(line("═"))
    print(f"\n{C.BOLD}样本代表性{C.OFF}  ← 最该看的一节")
    empty_scenes = []
    for scene, words in r["scene_keywords"].items():
        hits = dict(r["kw_hits"].get(scene, Counter()))
        if hits:
            print(f"  {scene:<15} {hits}")
        else:
            empty_scenes.append(scene)
    if empty_scenes:
        print(f"\n  {C.WARN}完全没出现的场景: {', '.join(empty_scenes)}{C.OFF}")

    desktop_frac = r["desktop_total"] / max(1, n)
    print(f"\n  桌面/文件管理器污染: {r['desktop_total']} 条 = "
          f"{100 * desktop_frac:.1f}%")
    if empty_scenes and desktop_frac > 0.2:
        print(f"\n  {C.ERR}结论：这批样本无法验证分类器。{C.OFF}")
        print("    原因不是分类器差，而是**数据里没有对应的正样本**：")
        print("      · state=IN_GAME 的样本里，抓到的常常是 Windows 桌面 /")
        print("        文件管理器（capture_path=foreground 抓的是**前台窗口**，")
        print("        而前台是资源管理器，不是游戏）")
        print("      · 真正含 UI 关键词的样本极少，且多数是文件路径里的误命中")
        print(f"    {C.BOLD}下一步{C.OFF}：重采一批 —— 进游戏后依次打开")
        print("    背包 / 地图 / 对话 / 商店，每个界面停 10 秒以上，")
        print("    让 ocr_perceive 跑到采样，再用本工具复跑。")

    # ── 7. 触发明细 ────────────────────────────────────────
    if fired:
        print(line("═"))
        print(f"\n{C.BOLD}触发样本明细{C.OFF}（最多 {args.show} 条）")
        for s, g in fired[:args.show]:
            print(f"  [{s.ts}] {g.scene_id:<14} conf={g.confidence:.2f} "
                  f"命中={g.matched!r} state={s.state}")
            print(f"      {s.text[:110]!r}")

    print(line("═"))
    print(f"\n{C.DIM}阈值（min_confidence={0.55}）**未做任何调整** —— "
          f"按要求只报告。{C.OFF}")

    if args.json:
        import json
        payload = {
            "total": n,
            "labels": dict(r["labels"]),
            "keywords": dict(r["keywords"]),
            "fired_count": len(fired),
            "desktop_total": r["desktop_total"],
            "desktop_fired": len(r["desktop_fired"]),
            "by_state": {k: dict(v) for k, v in r["by_state"].items()},
            "accuracy_computable": False,
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n汇总 -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
