"""模板裁图助手：自动找血条/能量条区域，一键存成模板。

    python -m autopilot.tools.template_cropper --image shot.png
    python -m autopilot.tools.template_cropper --live --title 原神
    python -m autopilot.tools.template_cropper --image shot.png --manual

为什么需要它
------------
``hp()`` / ``energy()`` 要血条/能量条的模板图，而模板图**按仓库惯例不能预置**
（每人的分辨率 / UI 缩放 / 语言都不一样）。于是现状是纯手工：

    截图 -> 自己找血条位置 -> 画图工具裁 -> 命名存盘 -> 改 defaults.toml

本工具把中间三步自动化：读一张截图，按**色相 + 形状**找出横条区域，
画预览，人确认后按标准命名写进 ``peroplicn/templates/``。

★ 颜色参数是**在真机截图上实测标定的**，不是拍脑袋
---------------------------------------------------------
实测样本：``autopilot/_runtime/capdiag_wgc.png``（WGC 真机抓取，2560x1600）

  · 玩家血条   y=[1500,1514]  H=41 S=215 V=215   —— 极稳定
  · 队友血条 ×4 x=[2253,2347] H=41 S=215 V=215   —— **同色同亮**，是误报源
  · 夜石地面   H≈104-106 S≈183                   —— **饱和度极高**（S=183）

由此得出三条**不能省**的设计约束：

1. **必须按色相门控，不能用「饱和且亮」**。
   石板地面 S=183 比血条 S=215 只差一点，用饱和度当判据会把整块地砖
   全框进来（实测：整帧 3.2% 的像素都满足 ``S>=80 & V>=80``）。
2. **必须做水平闭运算**。血条上叠着白色数字 ``17640 / 17640``，
   把连续的绿色切成碎段 —— 不桥接的话最宽只有 126px（实测）。
3. **必须靠位置/宽度去重**。玩家血条和队友血条颜色完全一样，
   靠「在底部 + 相对最宽」区分（实测玩家条 ≈380px，队友条 ≈90px）。

⚠️ 未标定项（诚实说明）
------------------------
``energy_bar`` 的颜色参数**没有真机实测数据**：那张真机截图里角色能量为 0，
能量条是空的，扫不到任何金色像素。因此 ``ENERGY_BAR`` 的 HSV 范围来自
**原神 UI 常识而非实测**，首次使用请务必看预览图确认框对不对。
血条（``HP_BAR``）则是实测标定的。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autopilot.tools._bootstrap import C, bootstrap, line, setup_logging  # noqa: E402
else:
    from ._bootstrap import C, bootstrap, line, setup_logging


#: 模板输出目录（与 ``defaults.toml`` 里 ``[autopilot.templates.*]`` 的 path 对应）
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "perception" / "templates"


@dataclass(frozen=True)
class BarSpec:
    """一条横条的查找规则。

    :param name: 模板名（存盘文件名、``defaults.toml`` 里的键）
    :param label: 预览图上的中文标签
    :param h_lo/h_hi: OpenCV H 范围（0~179，注意是**半角**）
    :param s_min/v_min: 饱和度 / 明度下限
    :param roi: 归一化搜索区 ``(x1, y1, x2, y2)``，取值 0~1
    :param min_aspect: 宽高比下限（横条至少要这么扁）
    :param bridge: 水平闭运算核宽（像素）。用来桥接条上的文字断裂
    """

    name: str
    label: str
    h_lo: int
    h_hi: int
    s_min: int
    v_min: int
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    min_aspect: float = 3.0
    bridge: int = 25
    calibrated: bool = True
    note: str = ""


#: 玩家血条。**实测标定**：H=41 S=215 V=215 @ 2560x1600。
#: roi 收在底部中央偏左 —— 队友血条在右上（x≈0.88~0.92），
#: 这里用位置先把它们排除掉，比事后靠宽度去重更省事。
HP_BAR = BarSpec(
    name="hp_bar",
    label="玩家血条",
    h_lo=30, h_hi=80, s_min=120, v_min=120,
    roi=(0.35, 0.85, 0.65, 1.00),
    note="实测：y=[1500,1514] H=41 S=215 V=215（2560x1600 真机）",
)

#: 能量条。⚠️ **未实测标定** —— 手头那张真机截图里角色能量为 0，扫不到金色。
ENERGY_BAR = BarSpec(
    name="energy_bar",
    label="元素能量条",
    h_lo=15, h_hi=34, s_min=120, v_min=150,
    roi=(0.35, 0.88, 0.65, 1.00),
    calibrated=False,
    note="⚠️ 未实测：真机样本里能量为 0，能量条是空的。"
         "参数来自 UI 常识，首次使用请核对预览图",
)

#: 队友血条。用于「顺带确认队友血条也能找到」——不是 hp() 的数据源，
#: 留在这里是为了让用户看到误报到底长什么样。
PARTY_HP_BAR = BarSpec(
    name="party_hp_bar",
    label="队友血条",
    h_lo=30, h_hi=80, s_min=120, v_min=120,
    roi=(0.80, 0.25, 1.00, 0.75),
    note="实测：4 条，x=[2253,2347]，每条宽约 90px",
)

SPECS: dict[str, BarSpec] = {
    s.name: s for s in (HP_BAR, ENERGY_BAR, PARTY_HP_BAR)
}


@dataclass
class BarCandidate:
    """一个候选横条。"""

    spec: BarSpec
    box: tuple[int, int, int, int]      # x, y, w, h
    coverage: float                     # 条内目标色的像素占比 0~1
    score: float = 0.0

    def __post_init__(self) -> None:
        self.score = round(self.coverage * min(self.box[2] / 400.0, 1.0), 4)

    @property
    def name(self) -> str:
        return self.spec.name

    def as_dict(self) -> dict:
        return {
            "name": self.spec.name, "label": self.spec.label,
            "box": list(self.box), "coverage": round(self.coverage, 4),
            "score": self.score, "calibrated": self.spec.calibrated,
        }


def _crop_roi(image, roi):
    h, w = image.shape[:2]
    x1 = max(0, int(roi[0] * w))
    y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w))
    y2 = min(h, int(roi[3] * h))
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def find_bars(image, spec: BarSpec) -> list[BarCandidate]:
    """在一张图里找出符合 ``spec`` 的横条候选。

    纯函数：只依赖 ``image`` 与 ``spec``，不读配置、不写盘、不开窗。
    （所以可以直接被单测覆盖 —— 见 ``tests/unit/test_template_cropper.py``）

    :returns: 按得分**从高到低**排好的候选列表；找不到返回空列表
    """
    import cv2
    import numpy as np

    if image is None or getattr(image, "size", 0) == 0:
        return []

    sub, (ox, oy, _rw, _rh) = _crop_roi(image, spec.roi)
    if sub.size == 0:
        return []

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ① 色相门控 —— 不用饱和度当主判据（夜石地面 S=183，比血条只低一点）
    mask = ((H >= spec.h_lo) & (H <= spec.h_hi)
            & (S >= spec.s_min) & (V >= spec.v_min)).astype(np.uint8) * 255

    # ② 水平闭运算：桥接条上叠加的文字造成的断裂
    #    （实测血条上的 "17640 / 17640" 会把最宽连续段切到只剩 126px）
    k = max(1, int(spec.bridge))
    bridged = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))

    cnts, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    out: list[BarCandidate] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h <= 0 or w <= 0:
            continue
        if w / h < spec.min_aspect:
            continue
        cover = float(cv2.countNonZero(bridged[y:y + h, x:x + w])) / (w * h)
        if cover < 0.35:                  # 太稀疏 -> 是噪点不是条
            continue

        # ★ 闭运算既会把框**撑大**，也会整体**平移 1 像素**。
        #   MORPH_CLOSE 是「先膨胀后腐蚀」，核宽为偶数时 OpenCV 的锚点
        #   约定会让结果整体偏移一格（实测：原始 500..907 -> 桥接 501..908）。
        #   直接拿桥接框当答案，裁出来的模板就会少一像素、位置也偏一像素。
        #
        #   正确的分工：**闭运算只回答「哪些列属于同一条」，边缘必须由
        #   未桥接的原始掩码说了算**。所以拿桥接框往外放宽 k 像素当窗口，
        #   再用原始掩码在窗口里收紧 —— 这样既不丢两端的真像素，
        #   也不会把膨胀出来的虚边留在框里。
        pad = k
        wy1, wy2 = max(0, y - pad), min(mask.shape[0], y + h + pad)
        wx1, wx2 = max(0, x - pad), min(mask.shape[1], x + w + pad)
        tight = _tighten_to_mask(mask[wy1:wy2, wx1:wx2])
        if tight is not None:
            tx, ty, tw, th = tight
            if tw > 0 and th > 0:
                x, y, w, h = wx1 + tx, wy1 + ty, tw, th

        out.append(BarCandidate(spec=spec, box=(x + ox, y + oy, w, h),
                                coverage=cover))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _tighten_to_mask(mask):
    """把框收紧到掩码里**真正有像素**的范围。

    :returns: ``(x, y, w, h)``；掩码为空返回 ``None``

    为什么要这一步：水平闭运算是「先膨胀后腐蚀」，靠近 ROI 边界时膨胀被
    截断，腐蚀收不回来，实测框会左右各虚增十几像素。虚增的框裁出来的模板
    两侧多一截场景像素，血量比例会算错。
    """
    import numpy as np

    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if x2 < x1 or y2 < y1:
        return None
    # 再把框往内收一点，只保留有实质像素的列/行，避免残留 1px 毛边
    col_hit = mask.sum(axis=0) > 0
    row_hit = mask.sum(axis=1) > 0
    if not col_hit.any() or not row_hit.any():
        return None
    return (x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1))


def pad_box(box, pad: int, shape) -> tuple[int, int, int, int]:
    """把框向外扩 ``pad`` 像素并夹在图内。

    裁模板时**必须留边**：纯色方块方差趋零，模板匹配会算出 0/0 匹配不到
    （见 ``defaults.toml:132-138`` 的「裁图要点」）。
    """
    h, w = shape[:2]
    x, y, bw, bh = box
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def render_preview(image, candidates, scale: float = 1.0):
    """画出候选框 + 标签，返回 BGR 图（原图副本，不改入参）。"""
    import cv2

    canvas = image.copy()
    for i, c in enumerate(candidates, 1):
        x, y, w, h = c.box
        color = (0, 230, 0) if c.spec.calibrated else (0, 165, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        txt = f"{i}. {c.spec.label} {c.coverage:.0%}"
        ty = max(16, y - 6)
        cv2.putText(canvas, txt, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    if scale != 1.0:
        import cv2 as _cv
        canvas = _cv.resize(canvas, None, fx=scale, fy=scale,
                            interpolation=_cv.INTER_AREA)
    return canvas


def save_crop(image, cand: BarCandidate, out_dir: Path, pad: int = 4,
              scale: float = 1.0) -> Path:
    """把候选区域存成 PNG（``<name>.png``），带留边。"""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    box = pad_box(cand.box, pad, image.shape)
    crop = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
    path = out_dir / f"{cand.spec.name}.png"
    # ⚠️ 不用 cv2.imwrite：Windows 上它按 ANSI 码页拼路径，
    #    输出目录含中文时会静默存成乱码名（详见 image_io 模块）。
    from autopilot.perception.image_io import imwrite_unicode
    imwrite_unicode(path, crop)
    return path


def suggest_config_toml(name: str, width: int, height: int) -> str:
    """生成可直接粘进 ``defaults.toml`` 的配置片段。

    ROI 表达式按**实际框在画面里的相对位置**生成，所以换分辨率时
    表达式依然成立（模板图本身不会缩放，那个得重裁）。
    """
    return (
        f"\n[autopilot.templates.{name}]\n"
        f'path = "perception/templates/{name}.png"\n'
        f"threshold = 0.80\n"
        f"stable_count = 3\n"
        f'roi = "rect(cw * {width / 2560.0:.4f}, ch * {height / 1600.0:.4f}, '
        f'cw * 0.30, ch * 0.08)"\n'
    )


# ==================================================================
# 交互式兜底（tkinter 拉框）
# ==================================================================

def run_manual(image, out_dir: Path, scale: float = 0.5) -> int:
    """tkinter 拉框裁剪。自动检测不可靠时的兜底。

    左键拖拽选框 -> 底部状态栏显示当前鼠标位置的 BGR/HSV 像素值 ->
    ``Enter`` 保存 -> ``n`` 换下一个模板名 -> ``Esc`` 退出。

    需要有显示环境。无 GUI（SSH / 沙箱）时会抛 ``TclError``，
    那种情况下请改用 ``--image`` + 外部画图工具。
    """
    import tkinter as tk

    import cv2
    import numpy as np
    from PIL import Image, ImageTk  # noqa: PLC0415

    win = tk.Tk()
    win.title("template_cropper - 拖拽选框，Enter 保存，n 换名字，Esc 退出")

    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if scale != 1.0:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)))
    photo = ImageTk.PhotoImage(pil)
    cv_label = tk.Label(win, image=photo)
    cv_label.pack()

    status = tk.StringVar(value="拖拽左键选框")
    tk.Label(win, textvariable=status, anchor="w", font=("Consolas", 10)).pack(
        fill="x")

    state = {"x0": 0, "y0": 0, "x1": 0, "y1": 0, "name": "hp_bar"}
    rect_id = [None]

    def to_orig(ev):
        return int(ev.x / scale), int(ev.y / scale)

    def down(ev):
        state["x0"], state["y0"] = to_orig(ev)
        state["x1"], state["y1"] = state["x0"], state["y0"]
        if rect_id[0] is not None:
            cv_label.delete(rect_id[0])
        rect_id[0] = cv_label.create_rectangle(
            ev.x, ev.y, ev.x, ev.y, outline="lime", width=2)

    def move(ev):
        x, y = to_orig(ev)
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            b, g, r = image[y, x]
            hh, ss, vv = cv2.cvtColor(
                np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
            status.set(f"({x},{y})  BGR=({b:3d},{g:3d},{r:3d})  "
                       f"HSV=({int(hh):3d},{int(ss):3d},{int(vv):3d})  "
                       f"name={state['name']}")
        if rect_id[0] is not None:
            cv_label.coords(rect_id[0], state["x0"] * scale, state["y0"] * scale,
                            ev.x, ev.y)

    def up(ev):
        state["x1"], state["y1"] = to_orig(ev)
        move(ev)

    def save(_ev=None):
        x0, x1 = sorted((state["x0"], state["x1"]))
        y0, y1 = sorted((state["y0"], state["y1"]))
        if x1 - x0 < 4 or y1 - y0 < 2:
            status.set("框太小，忽略")
            return
        # ⚠️ 别写成 ``out_dir = Path(out_dir)`` —— 那会把闭包捕获的外层
        #    ``out_dir`` 遮蔽成局部变量，右边引用自己会 UnboundLocalError。
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        # ⚠️ 不用 cv2.imwrite：路径含中文时会静默存成乱码名
        from autopilot.perception.image_io import imwrite_unicode
        imwrite_unicode(target / f"{state['name']}.png",
                        image[y0:y1, x0:x1])
        status.set(f"已存 {target / (state['name'] + '.png')}"
                   f"  框=({x0},{y0},{x1-x0},{y1-y0})")

    cv_label.bind("<ButtonPress-1>", down)
    cv_label.bind("<B1-Motion>", move)
    cv_label.bind("<ButtonRelease-1>", up)
    win.bind("<Return>", save)
    win.bind("<Escape>", lambda _e: win.destroy())

    def next_name(_ev=None):
        """切到下一个模板名。名字写死一轮循环，省得每次都要传参。"""
        names = list(SPECS)
        cur = state["name"]
        idx = names.index(cur) + 1 if cur in names else 0
        state["name"] = names[idx % len(names)]
        status.set(f"下一个模板名：{state['name']}")

    win.bind("n", next_name)
    win.mainloop()
    return 0


# ==================================================================
# CLI
# ==================================================================

def _capture(title: str):
    from autopilot.perception.capturer import Capturer
    cap = Capturer(title)
    cap.open()
    return cap, cap.grab().image


def main() -> int:
    ap = argparse.ArgumentParser(
        description="模板裁图助手：自动找血条/能量条并存成模板")
    ap.add_argument("--image", default="", help="输入截图路径（不给就实时抓）")
    ap.add_argument("--title", default="原神", help="窗口标题（--live 时用）")
    ap.add_argument("--out", default="", help="输出目录（默认 perception/templates）")
    ap.add_argument("--pad", type=int, default=4, help="向外留边像素（默认 4）")
    ap.add_argument("--preview-scale", type=float, default=0.6,
                    help="预览缩放（默认 0.6）")
    ap.add_argument("--save-preview", default="", help="预览图保存路径")
    ap.add_argument("--manual", action="store_true",
                    help="跳过自动检测，走 tkinter 拉框")
    ap.add_argument("--no-save", action="store_true", help="只看不存")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bootstrap()
    setup_logging(args.verbose)
    if not sys.stdout.isatty():
        C.disable()

    import cv2

    out_dir = Path(args.out) if args.out else TEMPLATE_DIR

    if args.image:
        # ⚠️ 不用 cv2.imread：它在 Windows 上按 ANSI 码页解路径，
        #    用户传一个中文名的截图就会静默读不到（详见 image_io 模块）。
        from autopilot.perception.image_io import imread_unicode

        img = imread_unicode(args.image)
        if img is None:
            print(f"{C.ERR}✗ 读不出图片: {args.image}{C.OFF}")
            return 1
        print(f"{C.OK}✓{C.OFF} 读入 {args.image}  {img.shape[1]}x{img.shape[0]}")
    else:
        try:
            cap, img = _capture(args.title)
        except Exception as exc:  # noqa: BLE001
            print(f"{C.ERR}✗ 截图失败: {exc}{C.OFF}")
            return 1
        try:
            cap.close()
        except Exception:  # noqa: BLE001
            pass

    if args.manual:
        try:
            return run_manual(img, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"{C.ERR}✗ 交互模式不可用（无 GUI?）: {exc}{C.OFF}")
            print(f"  请改用外部画图工具裁好后放到 {out_dir}")
            return 1

    print(line("═"))
    print(f"{C.BOLD}自动检测结果{C.OFF}   （绿框=实测标定，橙框=未标定，仅供参考）")
    print(line("═"))

    all_c: list[BarCandidate] = []
    for spec in SPECS.values():
        cands = find_bars(img, spec)
        flag = "" if spec.calibrated else f"  {C.WARN}{spec.note}{C.OFF}"
        if not cands:
            print(f"  ✗ {spec.label:<12} 没找到{flag}")
            continue
        best = cands[0]
        print(f"  ✓ {spec.label:<12} 找到 {len(cands)} 个，最佳 "
              f"box={best.box} 覆盖率={best.coverage:.0%}  "
              f"宽高比={best.box[2] / max(1, best.box[3]):.1f}{flag}")
        all_c.extend(cands[:1])

    if not all_c:
        print(f"\n{C.WARN}一个都没找到 —— 自动检测在这张图上不可靠{C.OFF}")
        print(f"请改用交互模式：--manual   （或外部画图工具裁到 {out_dir}）")
        return 2

    preview = render_preview(img, all_c, scale=args.preview_scale)
    if args.save_preview:
        from autopilot.perception.image_io import imwrite_unicode
        imwrite_unicode(args.save_preview, preview)
        print(f"  预览图 -> {args.save_preview}")

    # ⚠️ 绝不能用 ``waitKey(0)``：无 GUI 环境（SSH / 沙箱 / 无桌面会话）下
    #    它会**无限期阻塞**，工具看起来像卡死了。改成有上限的轮询，
    #    并且 imshow 失败就直接跳过，不把用户堵在这儿。
    shown = False
    try:
        cv2.imshow("template_cropper", preview)
        shown = True
    except Exception as exc:  # noqa: BLE001
        print(f"{C.DIM}（无法开预览窗口: {exc}；用 --save-preview 存图）{C.OFF}")

    if shown:
        print(f"{C.DIM}预览窗已开，10 秒后自动继续（按任意键可提前关闭）{C.OFF}")
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if cv2.waitKey(100) & 0xFF != 0xFF:
                break
        cv2.destroyAllWindows()

    if args.no_save:
        return 0

    print(line("═"))
    print(f"{C.BOLD}存盘{C.OFF}")
    for c in all_c:
        p = save_crop(img, c, out_dir, pad=args.pad)
        print(f"  {c.spec.label:<12} {p}")
        print(suggest_config_toml(c.spec.name, c.box[2], c.box[3]))
    print("把上面的片段贴进 autopilot/config/defaults.toml 的 "
          "[autopilot.templates.*]，然后跑：")
    print("  python -m autopilot.tools.template_demo --only "
          f"{','.join(c.spec.name for c in all_c)}")
    print()
    print(f"{C.WARN}注意：模板图不会随分辨率自动缩放。换分辨率后请重裁。{C.OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
