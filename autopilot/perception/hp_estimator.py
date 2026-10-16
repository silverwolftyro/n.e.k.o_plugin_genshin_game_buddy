"""血条 / 能量条**填充比例**估计器（从图像算出 0.0-1.0）。

★ 它解决什么问题
----------------
DSL 里 ``hp()`` / ``energy()`` / ``e_ready()`` 三个信号需要回答「这个条还剩多少」。
本模块提供那个信号的**纯逻辑内核**：给它一条已经裁好的**条形区域**图像，
它返回这条被填了百分之多少（``0.0`` = 空，``1.0`` = 满）。

刻意做成**纯函数**：只吃图和参数，不读配置、不开窗、不写盘、不依赖全局状态。
所以能用**合成图**精确构造满条/半条/空条/带描边/带噪声等场景做单测。

★ 为什么现在**没有**接进 ``builder.py``
----------------------------------------
``builder.py`` 里目前**不存在** ``hp()`` / ``energy() / ``e_ready()``（恒为 UNKNOWN，
有 5 条规则在等它们）。接线的前提是**每个分辨率各自的条形模板图**：

1. 条在画面里的**像素坐标随分辨率变化**，必须由用户提供；
2. 按仓库惯例这些裁图**不能预置进仓库**（见路线图 §4），必须用户自己用
   ``capture_demo --save`` 裁，且不同分辨率各一份；
3. 没有真实截图时硬接，只能造出一个"永远返回 None 的假信号"，
   那比现在的 UNKNOWN 更糟 —— 会让规则以为信号存在。

所以本模块**只交付逻辑，不接线**。等用户裁好模板图，再在 builder 里按
``rois.toml`` 里的框把图裁出来喂进来即可，那时补的是几行胶水，不是算法。

★ 颜色 / 坐标约定
-----------------
* 输入是 ``numpy.ndarray``，形状 ``(H, W, 3)``，``uint8``（也接受其它数值 dtype）。
* **颜色序按 BGR**（对齐 WGC 抓屏与仓库其余图像代码，如 ``image_io.py``）。
  但本模块只做**欧氏距离**比较，不做通道语义运算，所以喂 RGB 算出来的比例**完全相同**。
  ``BarParams.color_order`` 只用于文档与调试标注，不影响数值。
* ``orientation="horizontal"`` 时比例沿**列**方向算（沿宽度变化，左→右）；
  ``orientation="vertical"`` 时沿**行**方向算（沿高度变化，上→下）。
* ``align="start"`` 表示填充从条的一端开始（横条从左、竖条从上），
  ``align="end"`` 表示从另一端开始（横条从右、竖条从下）。

★ 什么情况下返回 None（**这是本模块最重要的一条**）
---------------------------------------------------
解析不出来就明确说"不知道"，**绝不返回 0 或 1**。因为下游会把它当决策依据
（``hp() < 0.3`` 就去吃药），猜错的代价远大于"不知道"——服错药、误判精英、
乱开大招都是真实事故。全仓库（``TriState`` / UNKNOWN）都是这个哲学。

返回 ``None`` 的情形：

1. **输入不合法**：``None``、非 ndarray、维度不是 ``(H,W,3)``、宽高为 0、
   含 NaN/Inf。
2. **尺寸过小**：任一边 < ``BarParams.min_side``；或 ``border_inset`` 内缩后区域为空。
3. **参数非法**（``orientation`` / ``align`` 拼错）。
4. **内部整块同色**：一块 60x12 的纯深灰矩形，既可能是"空条"，也可能是"满条"
   （条的填充色恰好等于底色时），还可能只是随手裁到的纯色块 ——
   **三者像素级完全一样，没有任何信息能区分** → ``None``。

   特别注意：这条**不能**靠描边色化解。满条和空条的内部颜色都不同于框色，
   光看框色区分不出"整块是填充"还是"整块是空白"。

   要判出 0.0 / 1.0，调用方必须**显式**给 ``background_color``（可以来自
   用户自己裁的"空条模板图"）。这恰好也是接线需要真实模板图的原因之一。
5. **形状不像条**：填充在轴上分成多段（超过 ``max_segments``），或者填充压根没贴在
   起始端 —— 这更像噪声或花纹，不像血条。
6. **填充没贴在对齐端**：从对齐端数过去一个切片都不算填充 → ``None``。
   这说明要么这不是条，要么方向反了（该用 ``align="end"``），不能硬报 0.0。

★ 真实数据校准（``autopilot/_runtime/capdiag_wgc.png``，WGC 实拍 2560x1600）
--------------------------------------------------------------------------
拿仓库里那张真实截图离线核过一遍（**只读截图，没有开游戏**）：

* 玩家血条 ``box = (x=1076, y=1500, w=408, h=15)``，条身绿色 ``BGR≈(34,215,150)``，
  **确实是满血**：408 列里 261 列整根全绿，其余 65 列被条上叠加的
  ``17640 / 17640`` 数字切断（剩 343 列过半仍是绿）。
* **直接喂这张裁图 → 返回 ``None``**。这是**正确行为**，不是 bug：
  满条内部整块同色，和"随手裁到的纯色块"像素级一样，**没有任何信息能分开**。
* 显式给 ``background_color`` 后稳定返回 ``1.0``，且对取值不敏感 ——
  (30,30,30) / (20,20,20) / (45,45,45) / (60,60,60) 四个值结果一致，
  ``border_inset`` 取 0 / 1 / 2 / 3 也一致。
* 这张图里**找不到能量条**：底栏区域扫过绿 / 蓝 / 金三个色域，蓝色命中的
  101547 px 全是石板地面，不是条。所以能量条**没有**真实数据验证过。

⚠️ 由此得出一条接线硬约束：**builder 必须把 ``background_color`` 传进来**
（来源只能是用户按自己分辨率裁的"空条模板图"），否则血量一满就永远是 UNKNOWN，
``hp() < 0.3`` 之类的规则永远等不到信号。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np

__all__ = [
    "BarParams",
    "classify",
    "estimate_fill_ratio",
    "make_params",
]

#: 合法取值，避免把字符串散落在各处。
_ORIENTATIONS = ("horizontal", "vertical")
_ALIGNS = ("start", "end")


# ==================================================================
# 参数
# ==================================================================

@dataclass(frozen=True)
class BarParams:
    """血条 / 能量条的识别参数。全部有默认值，**没有硬编码**。

    默认值按"典型横条 + 1px 描边 + 暗底彩填充"给，覆盖原神玩家血条这种常规情况；
    竖条 / 无描边 / 显式背景色等变体通过参数或 :meth:`from_params` 传。
    """

    #: 条的方向。``horizontal`` 沿列（宽）变化，``vertical`` 沿行（高）变化。
    orientation: str = "horizontal"
    #: 填充从哪一端起算：``start`` = 左/上，``end`` = 右/下。
    align: str = "start"
    #: 内缩几像素，把**描边**剥掉。0 表示不剥（条本身没框）。
    border_inset: int = 1
    #: 显式背景色（``(B,G,R)`` 或 ``(R,G,B)``）。给了它就能解开"纯色块"的歧义：
    #: 内部整块同色时，拿它当判据，0.0 / 1.0 才判得出来。``None`` = 自动推断。
    background_color: Optional[tuple] = None
    #: 像素与背景色的欧氏距离超过它就算"填充"。0-255 三通道尺度，参考值 0-441。
    diff_threshold: float = 40.0
    #: 某一条切片（列/行）里，填充像素占比达到它，这一整条才算"填充"。
    #: 这是**抗噪 + 抗渐变**的关键：不做逐像素硬比较，取截面均值。
    slice_threshold: float = 0.5
    #: 整条都算填充（≥）时直接返回 1.0 的门槛。
    full_threshold: float = 0.97
    #: 整条都算空白（≤）时直接返回 0.0 的门槛。
    empty_threshold: float = 0.03
    #: 宽 / 高任一边小于它直接返回 None（太窄的条，噪声就能占满截面）。
    min_side: int = 4
    #: 沿轴最多允许填充分成几段。血条**恒为 1 段**（从一端连续填到头）；
    #: 分成多段说明这是花纹/噪声，不像条 → None。
    max_segments: int = 1
    #: 空白端的"平度"容忍倍数：空白端像素到其中位色的最大距离，
    #: 不得超过 ``diff_threshold × bg_coherence_factor``。真血条的底色是平的；
    #: 散得太开说明压根不是条（随机噪声会栽在这里）→ None。
    bg_coherence_factor: float = 2.0
    #: 颜色序，仅作文档约定（算法只做欧氏距离，不受其影响）。
    color_order: str = "bgr"

    @classmethod
    def from_params(cls, **overrides: Any) -> "BarParams":
        """工厂：``BarParams.from_params(orientation="vertical")``。

        刻意不接受未知字段 —— 拼错参数名要立刻炸，而不是被静默忽略。
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(f"未知参数: {sorted(unknown)}；可用: {sorted(known)}")
        return cls(**overrides)


def make_params(**overrides: Any) -> BarParams:
    """模块级工厂，等价于 :meth:`BarParams.from_params`。"""
    return BarParams.from_params(**overrides)


# ==================================================================
# 内部工具
# ==================================================================

def _as_rgb_array(image: Any) -> Optional[np.ndarray]:
    """校验并转成 ``(h, w, 3)`` 的 float32 数组；不合法返回 None。

    这里就是"拿不准就说不知道"的**第一道闸**：宁可返回 None，
    也不让一个 (0, 0, 3) 的空数组往下走变成 ratio = 0.0。
    """
    if image is None or not isinstance(image, np.ndarray):
        return None
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    if image.shape[0] < 1 or image.shape[1] < 1:
        return None
    if not np.issubdtype(image.dtype, np.number):
        return None

    arr = image[:, :, :3].astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        return None
    return arr


def _dist(a: Any, b: Any) -> float:
    """两个颜色的欧氏距离（0-255 尺度下最大约 441）。"""
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64)))


def _median_color(arr: np.ndarray) -> np.ndarray:
    """一组像素的中位颜色。用中位而不是均值 —— 均值会被填充色把背景带跑。"""
    flat = arr.reshape(-1, arr.shape[-1])
    return np.median(flat.astype(np.float64), axis=0)


def _dist_per_pixel(arr: np.ndarray, color: Any) -> np.ndarray:
    """每个像素到参考色的欧氏距离，返回 ``(h, w)`` 数组。"""
    delta = arr.astype(np.float64) - np.asarray(color, np.float64)
    return np.sqrt(np.sum(delta * delta, axis=2))


def _end_band(inner: np.ndarray, orientation: str, at_end: bool, band: int) -> np.ndarray:
    """取轴向末端 ``band`` 根切片构成的条带（横条=列带，竖条=行带）。"""
    if orientation == "horizontal":
        return inner[:, -band:, :] if at_end else inner[:, :band, :]
    return inner[-band:, :, :] if at_end else inner[:band, :, :]


def _count_segments(filled: np.ndarray) -> int:
    """布尔序列里连续 True 段的数量（段与段之间至少隔一个 False）。"""
    if filled.size == 0:
        return 0
    padded = np.concatenate(([False], filled, [False]))
    return int(np.count_nonzero(padded[1:] & ~padded[:-1]))


def _run_from_end(filled: np.ndarray, at_end: bool) -> int:
    """``filled`` 是布尔切片序列；返回从指定端起连续为真的长度。

    两种方向统一成"先取好方向的序列，再数开头连续 True"：
    ``argmin`` 作用在 bool 数组上返回**第一个 False** 的下标，
    正好等于开头连续 True 的长度。
    """
    n = int(filled.size)
    if n == 0:
        return 0
    seq = filled[::-1] if at_end else filled
    if not bool(seq[0]):
        return 0
    if bool(seq.all()):
        return n
    return int(np.argmin(seq))


# ==================================================================
# 主函数
# ==================================================================

def estimate_fill_ratio(
    image: Optional[np.ndarray],
    params: Optional[BarParams] = None,
    **overrides: Any,
) -> Optional[float]:
    """估计一条形区域的**填充比例**，范围 ``[0.0, 1.0]``。

    参数
    ----
    image:
        ``(H, W, 3)`` 图像，BGR ``uint8``（见模块 docstring 的颜色约定）。
        必须是**裁好的条形区域**，不要带太多条外面的画面。
    params:
        :class:`BarParams`，``None`` 用默认。默认对应"横条 + 1px 描边 + 左对齐"。
    **overrides:
        逐项覆盖参数，如 ``orientation="vertical"``、``background_color=(25,25,25)``。

    返回
    ----
    ``float`` in ``[0.0, 1.0]``，或 ``None``（拿不准 —— 详见模块 docstring 第 ★ 节，
    共 6 类情形，**None 绝不等于 0 或 1**）。

    算法一句话
    ----------
    剥掉描边 → 以**空白端**的截面中位色为背景 → 逐像素按欧氏距离二值化 →
    沿填充轴取截面均值（抗噪、抗渐变）→ 数从填充端起连续为真的切片数。
    """
    p = params if params is not None else BarParams()
    if overrides:
        # dataclasses.replace 走 __init__，未知字段会直接 TypeError，
        # 和 from_params 一样"拼错就炸"，不会静默吞掉。
        p = replace(p, **overrides)

    # ---- 参数合法性（拼错就返回 None，而不是默默用默认值）----
    if p.orientation not in _ORIENTATIONS or p.align not in _ALIGNS:
        return None

    # ---- 输入合法性 ----
    arr = _as_rgb_array(image)
    if arr is None:
        return None
    h, w = arr.shape[0], arr.shape[1]
    if h < p.min_side or w < p.min_side:
        return None

    # ---- 剥描边 ----
    t = int(p.border_inset)
    if t < 0:
        return None
    inner_h, inner_w = h - 2 * t, w - 2 * t
    if inner_h < 1 or inner_w < 1:
        return None
    inner = arr[t: h - t, t: w - t]

    # ---- 背景色：调用方给了就用给的，否则取**空白端**截面的中位色 ----
    # align="start" → 空白在末端；align="end" → 空白在起始端。
    at_empty_end = p.align != "end"
    explicit = None
    if p.background_color is not None:
        explicit = np.asarray(p.background_color, np.float64)

    if p.orientation == "horizontal":
        n_slices = inner_w
    else:
        n_slices = inner_h

    if explicit is not None:
        bg = explicit
    else:
        # 空白端切片**必须沿着填充轴取**：横条切片是列（取最后一列带），
        # 竖条切片是行（取最后一行带）。取错轴会把填充色当成背景色。
        band = max(1, n_slices // 4)
        while True:
            strip = _end_band(inner, p.orientation, at_empty_end, band)
            bg = _median_color(strip)
            # 空白端本身必须是**一整片同色**。真血条的底色是平的；
            # 散得太开说明压根不是"条的底"（随机噪声就是这样）。
            if float(_dist_per_pixel(strip, bg).max()) <= (
                float(p.diff_threshold) * float(p.bg_coherence_factor)
            ):
                break
            # 血很满时，固定宽度的带子会探进填充区。逐级收窄重试，
            # 收到只剩 1 根切片还散得开 → 这就不是条。
            if band == 1:
                return None
            band = max(1, band // 2)

    mask = _dist_per_pixel(inner, bg) > float(p.diff_threshold)

    # ---- 内部整块同色：没有显式背景色就"不知道" ----
    # 注意这里**不能用描边色当判据**：满条和空条的内部颜色都不同于框色，
    # 光靠框色区分不了"整块是填充"还是"整块是空白"。曾经在这里按框色反推，
    # 结果空条被判成 1.0 —— 正是本模块明令禁止的那种"看着合理但错的数字"。
    if bool(mask.all()) or not bool(mask.any()):
        if explicit is None:
            # 空条 / 满条 / 纯色块，三者在像素上完全一样 → None。
            return None
        inner_color = _median_color(inner)
        return 0.0 if _dist(inner_color, explicit) <= float(p.diff_threshold) else 1.0

    # ---- 沿填充轴做截面均值（抗噪 + 抗渐变）----
    profile = mask.mean(axis=0) if p.orientation == "horizontal" else mask.mean(axis=1)
    filled = profile >= float(p.slice_threshold)

    ratio = float(_run_from_end(filled, at_end=not at_empty_end)) / float(n_slices)

    # 贴着对齐端一个切片都没填 → 这不像条（或者说方向反了），"不知道"。
    if ratio <= 0.0:
        return None

    # 条只有一段填充：数出来的"连续段"必须覆盖全部 True 切片。
    # 随机噪声的 filled 序列几乎不可能是"前缀 True + 后缀 False"，
    # 这条检查就是用来把噪声挡在外面，不让它报出一个看似合理的比例。
    if _count_segments(filled) > int(p.max_segments):
        return None

    return min(1.0, max(0.0, float(ratio)))


# ==================================================================
# 便捷判定
# ==================================================================

def classify(
    ratio: Optional[float],
    *,
    empty_threshold: float = 0.03,
    full_threshold: float = 0.97,
) -> Optional[str]:
    """把比例归成 ``"empty"`` / ``"partial"`` / ``"full"``；``ratio`` 为 None 则原样返回 None。

    给 ``e_ready()`` 这类只需要"满 / 没满"的信号用。**None 一路透传**，
    不会被误判成 empty —— 没测到和测到是 0，两回事。
    """
    if ratio is None:
        return None
    if ratio <= empty_threshold:
        return "empty"
    if ratio >= full_threshold:
        return "full"
    return "partial"
