"""E 技能（``key_e``）就绪阈值标定探针。

    python -m autopilot.tools.calibrate_skill_cd                  # 实时读游戏
    python -m autopilot.tools.calibrate_skill_cd --frames 600      # 只读 600 帧
    python -m autopilot.tools.calibrate_skill_cd --from-dir D:\\workspace\\N.E.K.O\\NEKO-genshin\\shots
    python -m autopilot.tools.calibrate_skill_cd --interval 0.1    # 10Hz，别太密

★ 这个脚本要解决的问题
----------------------
``detectors/key_e.py`` 里的 ``SAT_READY = 30.0`` 是**未标定值**。
模块文档记着当时的实测：就绪/冷却的饱和度分布是
``[10,20)`` 93 帧、``[40,50)`` 35 帧、``[110,120)`` 7 帧 ——
**连续的一团，没有天然谷底**。30 只是「大致对半」的位置，不是分界线。

后果很具体：阈值偏低 -> 冷却中也判就绪 -> 对着 CD 按 E；
阈值偏高 -> 就绪也判冷却 -> 永远不放技能。**两个方向的错都很贵。**

★ 怎么标定（这是本脚本存在的理由）
----------------------------------
1. 起游戏，进大世界，**把队伍栏和技能图标都露出来**（别开大地图/活动页，
   那两种画面会被判成 UNKNOWN，采到也白采）。
2. 开这个脚本，按提示打标记::

       F8  —— 此刻 E 是亮的（就绪）
       F9  —— 此刻 E 是灰的（冷却中）
       Esc —— 结束并出报告

   每个标记会记下**当前帧的饱和度 + 直方图**。
3. 报告会给三样东西：
   * 两簇各自的取值范围（min/max/均值/分位数）
   * **建议阈值**（取「就绪簇最低值」与「冷却簇最高值」的中点，
     两簇有重叠时会明确告诉你重叠有多严重）
   * 用建议阈值回测这两簇，算出准确率

4. 把建议值填进 ``autopilot/perception/detectors/key_e.py`` 的
   ``SAT_READY``，**不要填进 rois.toml** —— rois.toml 放的是**位置**，
   阈值是**判定**，混在一起会让「框歪了」和「阈值不对」两种问题互相掩盖。

★ 为什么必须人工打标记，不能自动聚类
------------------------------------
就绪/冷却两簇在饱和度上是**重叠的**（实测分布连续）。
任何无监督聚类都会在重叠区里随机切一刀，切出来的阈值
在「训练数据」上好看，换个角色/换个时间就崩。
唯一可靠的标签来源是**人眼** —— 你按 F8/F9 的那一刻就是标注。

⚠️ 至少采两类场景
   站着不动按一次 E（只拿到冷却这一段）+ 满 CD 时按一次 E（拿到就绪），
   每个状态各 30~50 个标记。少于这个数，报告里的建议阈值不可信。
   **不同角色的 E 颜色不一样**（金/雷/冰…），主 C 和后台都要各采一轮。

★ 离线模式
----------
``--from-dir`` 跑一批已存好的 PNG。没有标注时它只出分布直方图，
不会给阈值 —— 因为它不知道哪张是就绪、哪张是冷却。
配合 ``--label-ready`` / ``--label-cooldown`` 的文件名前缀可以离线标定::

    ready_001.png cooldown_001.png ...

★ 这个脚本**只读**，不改任何配置、不碰键鼠
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._bootstrap import C, bootstrap, line, setup_logging

#: 建议阈值落在这个区间之外就直接说「别用我的建议值」。
#: 低于它意味着就绪态本身也不够艳（多半是采样时机不对），
#: 高于它意味着冷却态里混进了彩色像素（多半是 ROI 框歪了）。
MIN_USABLE_SAT = 12.0
MAX_USABLE_SAT = 140.0


# ----------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------
def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))
    return s[k]


def _describe(name: str, vals: list[float], color: str) -> str:
    if not vals:
        return f"  {color}{name}{C.OFF}：（一个标记都没有）"
    return (f"  {color}{name:<10}{C.OFF} n={len(vals):<4} "
            f"min={min(vals):6.1f}  p10={_pct(vals, .10):6.1f}  "
            f"中位={_pct(vals, .50):6.1f}  p90={_pct(vals, .90):6.1f}  "
            f"max={max(vals):6.1f}  均值={sum(vals)/len(vals):6.1f}")


def _suggest(ready: list[float], cool: list[float]) -> tuple[float | None, str]:
    """给建议阈值 + 一句人话。

    取两簇**中位数**的中点而不是 min/max 的中点：实测分布长尾，
    用 min/max 会被一两个离群帧拽偏（曾经 7 帧落在 [110,120)）。
    """
    if len(ready) < 5 or len(cool) < 5:
        return None, (f"{C.WARN}标记太少（就绪 {len(ready)} / 冷却 {len(cool)}）。"
                      f"每个状态至少 30 个再谈标定，现在这个数没有统计意义。{C.OFF}")
    r_mid = _pct(ready, .50)
    c_mid = _pct(cool, .50)
    thr = (r_mid + c_mid) / 2.0

    overlap_hi = max(cool)
    overlap_lo = min(ready)
    msg = ""
    if overlap_lo < overlap_hi:
        span = overlap_hi - overlap_lo
        msg = (f"{C.WARN}两簇有重叠{C.OFF}：就绪最低 {overlap_lo:.1f} < "
               f"冷却最高 {overlap_hi:.1f}（重叠 {span:.1f}）。"
               f"任何单一阈值都在这段里必然犯错 —— "
               f"考虑「取中位数 + 连续 N 帧一致才认」的时间滤波。")
    if thr < MIN_USABLE_SAT or thr > MAX_USABLE_SAT:
        return None, (f"{C.ERR}建议值 {thr:.1f} 落在可用区间 "
                      f"[{MIN_USABLE_SAT:.0f}, {MAX_USABLE_SAT:.0f}] 之外，不建议采用。{C.OFF}"
                      f"多半不是阈值的问题，是 ROI 框歪了或者采样时机不对。{msg}")
    return thr, msg


def _backtest(ready: list[float], cool: list[float],
              thr: float) -> tuple[float, float, float]:
    """用 thr 回测：就绪召回 / 冷却召回 / 总体准确率。"""
    if not ready or not cool:
        return (float("nan"),) * 3
    hit_r = sum(1 for v in ready if v >= thr) / len(ready)
    hit_c = sum(1 for v in cool if v < thr) / len(cool)
    n = len(ready) + len(cool)
    acc = (hit_r * len(ready) + hit_c * len(cool)) / n
    return hit_r, hit_c, acc


def _histogram(all_vals: list[float], ready: list[float],
               cool: list[float], width: int = 48) -> str:
    """ASCII 直方图。``R``=就绪标记 ``C``=冷却标记 ``#``=其它采样。

    谷底就是阈值该放的地方 —— 看一眼比读三行数字快。
    """
    if not all_vals:
        return "（没有样本）"
    lo, hi = min(all_vals), max(all_vals)
    if hi <= lo:
        return f"所有样本都是 {lo:.1f} —— 没有分布可看"
    bins = 20
    step = (hi - lo) / bins
    r_bins, c_bins, n_bins = [0] * bins, [0] * bins, [0] * bins
    for v in all_vals:
        n_bins[min(bins - 1, int((v - lo) / step))] += 1
    for v in ready:
        r_bins[min(bins - 1, int((v - lo) / step))] += 1
    for v in cool:
        c_bins[min(bins - 1, int((v - lo) / step))] += 1

    peak = max(n_bins) or 1
    out = [f"  饱和度分布  [{lo:.0f} .. {hi:.0f}]  每格 {step:.1f}"]
    for i in range(bins):
        bar = "#" * max(1, round(n_bins[i] / peak * width))
        mark = ""
        if r_bins[i]:
            mark += f"  <-就绪 x{r_bins[i]}"
        if c_bins[i]:
            mark += f"  <-冷却 x{c_bins[i]}"
        if not n_bins[i]:
            bar = ""
        out.append(f"  {lo + i * step:6.1f} |{bar}{mark}")
    return "\n".join(out)


# ----------------------------------------------------------------------
# 采样
# ----------------------------------------------------------------------
def _sample(frame) -> dict:
    """一帧 -> key_e 的三个统计量。**任何异常都降级成 ok=False。**"""
    from autopilot.perception.detectors.key_e import center_stats

    try:
        st = center_stats(frame)
    except Exception as exc:                     # noqa: BLE001
        return {"ok": False, "sat": 0.0, "sat_ratio": 0.0, "val": 0.0,
                "err": f"{type(exc).__name__}: {exc}"}
    return {"ok": bool(st.get("ok")), "sat": float(st["center_sat"]),
            "sat_ratio": float(st["center_sat_ratio"]),
            "val": float(st["center_val"]), "err": ""}


def _verdict(s: dict, thr: float) -> tuple[str, str]:
    """当前阈值下这一帧会怎么判。"""
    if not s["ok"]:
        return "DIM", "读不到"
    if s["sat_ratio"] >= 0.30:
        return "WARN", "疑似地图/面板 -> UNKNOWN"
    if s["val"] >= 218.0 and s["sat"] < 27.0:
        return "WARN", "疑似活动页 -> UNKNOWN"
    return ("OK", "READY") if s["sat"] >= thr else ("DIM", "COOLDOWN")


def _bar(v: float, hi: float, width: int = 20) -> str:
    n = 0 if hi <= 0 else max(0, min(width, round(v / hi * width)))
    return "█" * n + "·" * (width - n)


def _poll_marks() -> tuple[bool, bool, bool]:
    """读 F8 / F9 / Esc 的**按下沿**。

    复用 ``actuator.physical_input`` 用的那套 ``GetAsyncKeyState``。
    这里自己轮询是因为标定时要的是「我看到这一帧，按了 F8」，
    不是「这一帧发生了什么」—— 顺序正好相反。
    """
    import ctypes

    user32 = ctypes.windll.user32
    prev = {"F8": False, "F9": False, "ESC": False}
    edges: dict[str, bool] = {}
    for name, vk in (("F8", 0x77), ("F9", 0x78), ("ESC", 0x1B)):
        down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
        edges[name] = down and not prev[name]
        prev[name] = down
    return edges["F8"], edges["F9"], edges["ESC"]


# ----------------------------------------------------------------------
# 实时模式
# ----------------------------------------------------------------------
def _run_live(args) -> int:
    from autopilot.config.loader import load_config
    from autopilot.perception.capturer import CaptureChannel, Capturer
    from autopilot.perception.detectors.key_e import SAT_READY

    cfg = load_config()
    order = [CaptureChannel(c) for c in cfg.capture.channel_order]
    cap = Capturer(cfg.capture.window_title_substr, channel_order=order)

    print(f"{C.BOLD}E 技能阈值标定{C.OFF}")
    print(line())
    print(f"  目标窗口 : {cfg.capture.window_title_substr}")
    print(f"  当前阈值 : SAT_READY = {SAT_READY}  "
          f"{C.DIM}(key_e.py 里那个未标定值){C.OFF}")
    print(f"  采集速率 : {1 / max(args.interval, 1e-3):.1f} Hz")
    print()
    print(f"  {C.BOLD}F8{C.OFF} = 此刻 E 是亮的（就绪）    "
          f"{C.BOLD}F9{C.OFF} = 此刻 E 是灰的（冷却中）    "
          f"{C.BOLD}Esc{C.OFF} = 结束")
    print(f"  {C.DIM}★ 先把游戏切出来。脚本起来后按 F8 标记前它会等你。{C.OFF}")
    print(line())

    if args.delay:
        for i in range(int(args.delay), 0, -1):
            print(f"  {i} 秒后开始采集 —— 现在切回游戏", end="\r", flush=True)
            time.sleep(1)
        print(" " * 50, end="\r")

    ready: list[float] = []
    cool: list[float] = []
    seen: list[float] = []
    unknown = 0
    n = 0
    bar_hi = 160.0

    try:
        cap.open()
    except Exception as exc:
        print(f"  {C.ERR}打开捕获器失败：{exc}{C.OFF}")
        return 2

    try:
        first = cap.wait_first_frame(timeout_s=args.timeout)
        if first is None:
            print(f"  {C.ERR}{args.timeout:.0f} 秒内没等到第一帧。{C.OFF}\n"
                  f"  {C.DIM}游戏开了吗？窗口最小化了吗？{C.OFF}")
            return 2

        while n < args.frames:
            try:
                frame = cap.grab()
            except Exception as exc:             # noqa: BLE001
                print(f"  {C.ERR}抓帧失败：{exc}{C.OFF}")
                return 2
            if frame is None:
                print(f"  {C.ERR}抓到 None{C.OFF}")
                return 2

            s = _sample(frame)
            n += 1
            tag, why = _verdict(s, SAT_READY)
            if s["ok"]:
                seen.append(s["sat"])
            else:
                unknown += 1

            f8, f9, esc = _poll_marks()
            note = ""
            if f8 and s["ok"]:
                ready.append(s["sat"])
                note = f"  {C.OK}<< F8 记为就绪 sat={s['sat']:.1f}{C.OFF}"
            elif f9 and s["ok"]:
                cool.append(s["sat"])
                note = f"  {C.WARN}<< F9 记为冷却 sat={s['sat']:.1f}{C.OFF}"

            line_txt = (f"[{n:5d}] sat={s['sat']:6.1f} "
                        f"ratio={s['sat_ratio']*100:5.1f}% "
                        f"val={s['val']:6.1f} "
                        f"{getattr(C, tag)}{why:<26}{C.OFF} "
                        f"{_bar(s['sat'], bar_hi)}{note}")
            print(line_txt, flush=True)

            if esc:
                print(f"\n  Esc 收到，停止采集。")
                break
            time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        print("\n  Ctrl-C 收到。")
    finally:
        try:
            cap.close()
        except Exception:
            pass

    return _report(seen, ready, cool, unknown, n, args)


# ----------------------------------------------------------------------
# 离线模式
# ----------------------------------------------------------------------
def _run_dir(args) -> int:
    from autopilot.perception.detectors.key_e import SAT_READY
    from autopilot.perception.image_io import imread_unicode

    d = Path(args.from_dir)
    if not d.is_dir():
        print(f"  {C.ERR}{d} 不是目录{C.OFF}")
        return 2
    files = sorted([p for p in d.rglob("*")
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
    if not files:
        print(f"  {C.ERR}{d} 里没有图片{C.OFF}")
        return 2

    print(f"{C.BOLD}E 技能阈值离线体检{C.OFF}")
    print(line())
    print(f"  目录 : {d}")
    print(f"  张数 : {len(files)}")
    print(f"  当前 : SAT_READY = {SAT_READY}")
    print(line())

    seen: list[float] = []
    ready: list[float] = []
    cool: list[float] = []
    unknown = 0

    for p in files:
        # ⚠️ 不用 cv2.imread：--from-dir 路径含中文时会静默读不到
        img = imread_unicode(p)
        if img is None:
            print(f"  {C.WARN}{p.name} 读不了，跳过{C.OFF}")
            continue
        s = _sample(img)
        name = p.stem.lower()
        if not s["ok"]:
            unknown += 1
            print(f"  {p.name:<34} {C.DIM}读不到（ROI 出框？）{C.OFF}")
            continue
        seen.append(s["sat"])
        # 文件名前缀即标签：ready_* / cooldown_*
        if name.startswith("ready"):
            ready.append(s["sat"])
            tag = f"{C.OK}就绪{C.OFF}"
        elif name.startswith(("cooldown", "cd_")):
            cool.append(s["sat"])
            tag = f"{C.WARN}冷却{C.OFF}"
        else:
            tag = f"{C.DIM}未标注{C.OFF}"
        print(f"  {p.name:<34} sat={s['sat']:6.1f}  "
              f"ratio={s['sat_ratio']*100:5.1f}%  val={s['val']:6.1f}  {tag}")

    return _report(seen, ready, cool, unknown, len(files), args)


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def _report(seen, ready, cool, unknown, n, args) -> int:
    print()
    print(line("="))
    print(f"{C.BOLD}标定报告{C.OFF}")
    print(line("="))
    print(f"  总采样 {n} 帧，其中读不到 {unknown} 帧"
          f"{'（' + C.DIM + 'ROI 没框住画面，先查 rois.toml 的 key_e' + C.OFF + '）' if unknown else ''}")

    if seen:
        print()
        print(_histogram(seen, ready, cool))

    print()
    print(f"{C.BOLD}标注统计{C.OFF}")
    print(_describe("就绪(F8)", ready, C.OK))
    print(_describe("冷却(F9)", cool, C.WARN))

    thr, msg = _suggest(ready, cool)
    print()
    if thr is None:
        print(f"{C.WARN}给不出可信的建议阈值。{C.OFF}")
        print(f"  {msg}")
        return 1

    hit_r, hit_c, acc = _backtest(ready, cool, thr)
    print(f"{C.BOLD}建议 SAT_READY = {thr:.1f}{C.OFF}")
    print(f"  回测：就绪召回 {hit_r*100:5.1f}%   冷却召回 {hit_c*100:5.1f}%   "
          f"总体 {acc*100:5.1f}%")
    if msg:
        print(f"  {msg}")
    if acc < 0.90:
        print(f"  {C.WARN}总体不足 90%。{C.OFF}别急着上线 —— "
              f"先确认 ROI 框对了（`--roi-preview`），再考虑时间滤波。")

    print()
    print(f"{C.BOLD}怎么用{C.OFF}")
    print(f"  1. 打开 autopilot/perception/detectors/key_e.py")
    print(f"     把  SAT_READY = 30.0  改成  SAT_READY = {thr:.1f}")
    print(f"     {C.DIM}（是阈值，不是位置 —— 不要填进 rois.toml）{C.OFF}")
    print(f"  2. 换角色重跑一遍。E 图标颜色随角色元素变，"
          f"阈值不一定通用。")
    print(f"  3. 到此为止就够放技能了。CD 数字、能量条是后面的活。")
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m autopilot.tools.calibrate_skill_cd",
        description="采集 key_e 饱和度并给出 SAT_READY 建议值（只读，不改配置）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("★ 这个脚本**只读**")[0],
    )
    p.add_argument("--frames", type=int, default=10 ** 9,
                   help="最多采多少帧（默认不限，按 Esc 停）")
    p.add_argument("--interval", type=float, default=0.05,
                   help="采样间隔秒。默认 0.05（20Hz）。"
                        "太快会让同一帧被重复采样，稀释统计")
    p.add_argument("--delay", type=float, default=0.0,
                   help="启动后等几秒再开始采（给 alt-tab 回游戏用）")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="等第一帧的秒数。默认 30")
    p.add_argument("--from-dir", metavar="DIR",
                   help="离线模式：跑一批已存好的图片。"
                        "文件名前缀 ready_/cooldown_ 会被当成标签")
    p.add_argument("--no-color", action="store_true", help="关掉 ANSI 颜色")
    p.add_argument("--verbose", action="store_true", help="开 DEBUG 日志")
    args = p.parse_args(argv)

    bootstrap()
    setup_logging(args.verbose)
    if args.no_color:
        C.disable()

    if args.from_dir:
        return _run_dir(args)
    return _run_live(args)


if __name__ == "__main__":
    sys.exit(main())
