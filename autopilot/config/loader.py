"""配置加载：TOML → pydantic 校验 → 配置对象。

对应架构文档 3.1（配置选型）与原则 1（响亮失败）。

**加载期校验不可省**：
    模板路径不存在、ROI 表达式语法错、阈值超范围 —— 任何一项都应该在
    启动时炸掉并指出具体位置，而不是在感知循环里每帧失败一次。
    pydantic 的 ValidationError 直接向上抛，不吞。
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..perception.roi_expr import RoiExprError, RoiExprEvaluator
from ..perception.template import TemplateSpec

logger = logging.getLogger(__name__)

#: 插件根目录（autopilot/config/loader.py -> 上溯三级）
PLUGIN_ROOT = Path(__file__).resolve().parents[2]
#: autopilot 包根
AUTOPILOT_ROOT = Path(__file__).resolve().parents[1]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 未知键直接报错，不静默忽略


#: 已废弃的截图通道名 -> 现行名字。
#:
#: ★ 背景（2026-10-10）
#:   PrintWindow 从降级链里出链，顶替它的是 BitBlt —— 同样是窗口级
#:   （游戏被遮挡也能抓到内容），但 PrintWindow 在 Win11 + 硬件加速下
#:   恒失败（真机证据：``截图重试 25 次仍未成功：PrintWindow 返回失败``
#:   → ``CORE_PAUSED``），而 BitBlt 在同样的窗口上抓到了完整画面。
#:
#: ★ 为什么迁移放在**配置层**而不是让调用方各自认
#:   ----------------------------------------------
#:   ``tools/_stack.py:94`` 是**裸调** ``CaptureChannel(c)``，不经过
#:   ``Capturer._normalize_order`` 那层带说明的包装。配置里的字符串
#:   一到那儿就抛 Python 原生的
#:   ``ValueError: 'printwindow' is not a valid CaptureChannel``，
#:   而这句话既没提"去改哪"也没提"改成什么" —— 用户只会盯着
#:   "is not a valid" 猜。在配置层迁移一次，
#:   ``_stack`` / ``loop_diagnose`` / ``autopilot_demo`` 三个消费方
#:   一次性全好，谁都不用动。
CAPTURE_CHANNEL_ALIASES: dict[str, str] = {
    "printwindow": "bitblt",
    "print_window": "bitblt",
    "pw": "bitblt",
}


class CaptureConfig(_Base):
    window_title_substr: str = "原神"
    #: ★ 顺序与 ``defaults.toml`` 的 ``channel_order`` 必须一致 ——
    #:   wgc 优先（走 DWM 合成管道，硬件加速窗口是它的主场；真机实测原神
    #:   全屏下 bitblt/dxcam/mss 三个全部只拿到黑帧），
    #:   bitblt 次之（窗口 GDI DC，不看屏幕坐标），
    #:   dxcam 再次（DXGI，硬件加速下最快，但窗口越界会直接抛），
    #:   mss 兜底（纯屏幕坐标，最后 resort）。
    #:
    #: ⚠️ ``wgc`` 依赖 ``windows-capture``；没装的机器上它会被
    #:   ``_init_channel`` 判为不可用并自动跳过，不会炸。
    channel_order: list[Literal["wgc", "bitblt", "dxcam", "mss"]] = Field(
        default_factory=lambda: ["wgc", "bitblt", "dxcam", "mss"]
    )
    fail_threshold: int = Field(default=3, ge=1, le=10)
    allow_fullscreen_fallback: bool = False

    @field_validator("channel_order", mode="before")
    @classmethod
    def _migrate_and_check(cls, v: object) -> object:
        """废弃通道名迁移 + 非空校验。**必须在 Literal 校验之前跑**
        （``mode="before"``），否则老配置会在类型检查那一步就被拒，
        根本走不到迁移。

        迁移只**告警**不静默：用户应该知道自己的配置被改写过了，
        下次手动改掉新名字。静默替换会让"我明明写的是 printwindow"
        和"它怎么跑成 bitblt 了"变成一桩无头案。
        """
        if not isinstance(v, (list, tuple)):
            return v                      # 交给 Literal 去报类型错
        out: list[str] = []
        for item in v:
            key = str(item).strip().lower()
            new = CAPTURE_CHANNEL_ALIASES.get(key)
            if new:
                logger.warning(
                    "截图通道 %r 已废弃 —— PrintWindow 在 Win11 上不可靠，"
                    "2026-10-10 出链，顶替它的是 bitblt；已自动按 %r 处理。"
                    "请把配置里的名字改成 %r。",
                    item, new, new)
                out.append(new)
                continue
            out.append(key)
        if not out:
            raise ValueError("channel_order 不能为空，至少要有一种截图通道")
        return out


class YoloConfig(_Base):
    enabled: bool = False
    onnx_path: str = "perception/models/yolo11n.onnx"
    class_names: list[str] = Field(default_factory=list)
    conf: float = Field(default=0.35, ge=0.0, le=1.0)
    iou: float = Field(default=0.5, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, ge=320, le=1280)
    timeout_ms: float = Field(default=500.0, gt=0)
    #: 场景路由：只在这些场景跑 YOLO；空 = 全场景
    scene_whitelist: list[str] = Field(default_factory=list)


class TemplateEntry(_Base):
    path: str
    threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    use_3_channels: bool = False
    use_green_mask: bool = False
    use_binary: bool = False
    binary_threshold: int = Field(default=128, ge=0, le=255)
    stable_count: int = Field(default=0, ge=0, le=10)
    roi: str | None = None


class OcrConfig(_Base):
    enabled: bool = True
    fail_threshold: int = Field(default=5, ge=1, le=50)
    allow_rapid_fallback: bool = True


class PerceptionConfig(_Base):
    yolo: YoloConfig = Field(default_factory=YoloConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    ref_size: tuple[int, int] = (1920, 1080)
    black_frame_luma: float = Field(default=8.0, ge=0.0, le=255.0)


class LoopConfig(_Base):
    """主循环配置。

    独立成 section 而不是往 [autopilot] 里塞字段：``AutopilotConfig``
    是 ``extra="forbid"``，未声明的键会直接报
    ``Extra inputs are not permitted`` —— 往已有 section 补字段是
    从「3 项错误」变成「4 项错误」，不是修复。

    这里对应 ``autopilot.core.runtime.RuntimeContext`` 的同名参数。
    ⚠ RuntimeContext 本身是**普通 dataclass**，不过 pydantic ——
      所以它有两个来源（配置文件 + 命令行），命令行显式给了就以命令行为准。
    """

    #: tick 频率 Hz。0 = 不限速
    tick_rate_hz: float = Field(default=5.0, ge=0.0, le=120.0)
    #: 感知层连续失败多少次后自动暂停
    max_consecutive_capture_failures: int = Field(default=10, ge=1, le=1000)
    #: 连续多少次 tick 崩溃后主动暂停
    max_consecutive_tick_crashes: int = Field(default=5, ge=1, le=1000)
    #: 事件总线容量（满了丢最旧）
    event_queue_size: int = Field(default=1000, ge=10, le=100000)
    #: 停止时等待工作线程退出的上限（秒）
    stop_timeout_s: float = Field(default=10.0, gt=0.0, le=120.0)


class LocatorConfig(_Base):
    """定位服务（小地图 + 相位相关）接线配置。

    ⚠️ **默认关闭，而且开着也不一定有用。**

    ``LocalizationService`` 初始状态是 ``UNINITIALIZED`` —— 它**没有起点**，
    必须先在游戏里**传送一次**建立原点（走 ``teleport_contract.md`` 的三段
    契约），之后才能做帧间位移。在那之前 ``position`` 恒为 ``quality="LOST"``。

    所以这个开关存在的意义是：**让接线这件事可以被测试**，而不是让用户
    一开就以为定位好了。开之前请先读 ``recognition/notes/teleport_contract.md``。
    """

    #: 总开关。``false``（默认）-> ``StateBuilder(locator=None)``，
    #: 与接入定位之前的行为**完全一致**（``position`` 恒为 None）。
    enabled: bool = False

    #: 相位相关峰值响应下限（归一化相关，0~1）。真机实测大世界段 p25 = 0.89；
    #: 静止基线 579 帧的 p99 |d| = 0.109 px，所以 0.10 是「几乎没动」的分界。
    #: ⚠️ 与 ``perception.localization_service.PHASE_MIN_PEAK_RESPONSE`` 同源，
    #:    由 ``tests/unit/test_locator_wiring.py`` 断言两者一致 ——
    #:    配置层不能 import 感知层（那会把依赖方向搞反），只能镜像 + 加守卫。
    min_peak_response: float = Field(default=0.10, gt=0.0, le=1.0)
    #: 圆内灰度标准差下限（0~255）。低于它说明圆内是纯色（黑屏/加载/纯水），
    #: 相位相关结果没有意义。真机实测大世界段 p25 = 39.1。
    min_circle_std: float = Field(default=20.0, gt=0.0, le=255.0)

    # ---- 传送契约 ① SELECT：自动喂 OCR（2026-10-15 接上）------------
    #
    # ★ 为什么必须有这个开关
    #   ``LocalizationService`` 只认 :meth:`on_teleport_ocr`，而它**从不猜**
    #   「这是不是传送」（§16.3 实测：帧间位移里混着大量 >3px 跳变，
    #   靠阈值区分「传送」与「玩家跑得快」不可靠）。契约把判据交给 OCR。
    #   在 2026-10-15 接线之前，``on_teleport_ocr`` 的调用方是 **0 个**
    #   —— 于是 ``enabled = true`` 也永远停在 ``UNINITIALIZED``，
    #   ``position`` 恒为 ``LOST``。这就是「传送十几次也没用」的原因。

    #: 是否在感知层自动跑传送 OCR 并喂给 ``on_teleport_ocr``。
    #: 默认 **True** —— 开了 ``enabled`` 就该是能用的；关掉它等于手动
    #: 契约（用 ``move_to_probe --set-origin``）。
    teleport_ocr_enabled: bool = True

    #: 传送 OCR 的**节流间隔**（秒）。只在「非大世界」帧尝试 ——
    #: 那正是传送地图打开、信息框可见的时候。地图可能开着几十秒，
    #: 每帧都跑一次 OCR（WinRT/RapidOCR）在 5Hz 下太贵。
    #: 0.5s 的响应延迟人察觉不到。
    teleport_ocr_interval_s: float = Field(default=0.5, ge=0.0, le=30.0)


class HpEstConfig(_Base):
    """血量几何投影接线配置。

    ⚠️ **默认关闭。** ``enabled = false`` 时 ``StateBuilder`` 的
    ``hp_estimation=None``，``PlayerInfo.hp_ratio`` 恒为 ``None`` = UNKNOWN，
    **与接入之前的行为完全一致**。

    打开后从本帧血条 ROI 做几何投影（长度比例），不经 OCR。
    拿不准就返回 ``None``，绝不填一个猜出来的数 —— 规则里
    ``hp() < 0.35`` 会据此决定要不要吃药，猜错比不知道糟得多。

    ★ ROI 用**相对坐标**而不是像素
      血条位置依赖分辨率。2560x1600 实测是 ``(1076, 1500, 408, 15)``，
      1920x1080 上那是错的框。所以配置里存**比例**，落地时乘以本帧尺寸。
      ``roi_frac = None`` 时用 ``templates/hp_bar_reference.json`` 里那份。
    """

    #: 总开关。``false``（默认）-> 不做血量估算，``hp_ratio`` 恒 None。
    enabled: bool = False

    #: 血条 ROI 比例 ``(x, y, w, h)``，全部相对帧宽/高，取值 0.0~1.0。
    #: ``None`` = 用 ``perception/templates/hp_bar_reference.json`` 的实测值。
    roi_frac: Optional[tuple[float, float, float, float]] = None

    #: 空条底色 BGR。**None = 不提供**。
    #:
    #: ★ 为什么它可以是 None
    #:   没有它时，**部分**血量的条仍能测准（内部有填充/空白分界）。
    #:   但恰好**满血**或**空血**时内部是一整片纯色，与「一块纯色补丁」在
    #:   像素上无法区分 —— 此时函数返回 ``None`` 而不是猜。
    #:   规则「hp() < 0.35」在满血时拿不到值只会跳过，不会误触发；
    #:   但如果你**要**让满血也出值，就得把底色填进来。
    background_bgr: Optional[tuple[int, int, int]] = None

    #: 剥掉几像素描边。默认 1，与 ``perception/templates`` 里实测的 1px 边框对齐。
    border_inset: int = Field(default=1, ge=0, le=8)

    @field_validator("roi_frac", "background_bgr", mode="before")
    @classmethod
    def _empty_list_is_none(cls, v: Any) -> Any:
        """TOML 里没填的值读出来是 ``[]``，不是 ``None``。

        ``Optional[tuple[...]]`` 不接受空 list（它会被当成「长度不对的元组」，
        报 ``Field required``）。空 = 没配，统一归一成 ``None``，
        这样「留空」在 TOML 和在代码里是同一个意思。
        """
        if isinstance(v, (list, tuple)) and len(v) == 0:
            return None
        return v


class AutopilotConfig(_Base):
    """autopilot 顶层配置。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False          # ★ 3.0 默认关闭，不影响 2.0 用户
    game_id: str = "genshin_impact"
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    #: 主循环（tick 循环）配置
    loop: LoopConfig = Field(default_factory=LoopConfig)
    #: 定位服务接线（默认关；开了也要先传送建立原点，见类文档）
    locator: LocatorConfig = Field(default_factory=LocatorConfig)
    #: 血量几何投影接线（默认关；见 :class:`HpEstConfig`）
    hp_estimation: HpEstConfig = Field(default_factory=HpEstConfig)
    #: 公共 ROI 变量（表达式里的自定义名）
    roi_vars: dict[str, str] = Field(default_factory=dict)
    #: 公共区域别名 -> ROI 表达式
    regions: dict[str, str] = Field(default_factory=dict)
    #: 模板名 -> 定义
    templates: dict[str, TemplateEntry] = Field(default_factory=dict)
    #: 场景路由：场景 ID -> 该场景要跑的模板名
    scene_routes: dict[str, list[str]] = Field(default_factory=dict)
    #: YOLO 标签 -> 场景 ID
    scene_map: dict[str, str] = Field(default_factory=dict)

    # -- 派生视图 --------------------------------------------------------

    def template_specs(self) -> dict[str, TemplateSpec]:
        """转成运行时用的 TemplateSpec。路径相对 autopilot 包根解析。"""
        out: dict[str, TemplateSpec] = {}
        for name, e in self.templates.items():
            p = Path(e.path)
            if not p.is_absolute():
                p = (AUTOPILOT_ROOT / p).resolve()
            out[name] = TemplateSpec(
                name=name,
                path=p,
                threshold=e.threshold,
                use_3_channels=e.use_3_channels,
                use_green_mask=e.use_green_mask,
                use_binary=e.use_binary,
                binary_threshold=e.binary_threshold,
                stable_count=e.stable_count,
            )
        return out

    def roi_expressions(self) -> dict[str, str]:
        """模板名 -> ROI 表达式（已展开 @区域别名）。"""
        out: dict[str, str] = {}
        for name, e in self.templates.items():
            if e.roi:
                out[name] = e.roi
        return out

    def roi_var_values(self) -> dict[str, float]:
        """把 roi_vars 里的表达式求值成标量。"""
        ev = RoiExprEvaluator(self.perception.ref_size)
        out: dict[str, float] = {}
        for k, expr in self.roi_vars.items():
            c = ev.compile(expr, where=f"roi_vars.{k}")
            out[k] = ev.eval_scalar(c, self.perception.ref_size[0], self.perception.ref_size[1])
        return out

    def resolve_region(self, alias: str) -> str:
        """展开 @别名（支持多层引用，检测循环）。"""
        seen: set[str] = set()
        cur = alias.strip()
        while cur.startswith("@"):
            key = cur[1:].strip()
            if key in seen:
                raise ValueError(f"regions 存在循环引用: {key}")
            seen.add(key)
            if key not in self.regions:
                raise ValueError(f"未定义的区域别名: @{key}")
            cur = self.regions[key].strip()
        return cur


# --------------------------------------------------------------------------
# 加载与校验
# --------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> AutopilotConfig:
    """加载配置。**文件不存在返回默认配置**，解析/校验失败**响亮抛错**。

    为什么"不存在返回默认"而不报错：用户可能还没配 autopilot，
    插件加载不该因为这个崩（对齐 2.0 的渐进式体验）。
    """
    p = Path(path) if path else (AUTOPILOT_ROOT / "config" / "defaults.toml")
    if not p.exists():
        logger.info("未找到 autopilot 配置，使用默认值: %s", p)
        return AutopilotConfig()

    try:
        with p.open("rb") as f:
            data: dict[str, Any] = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"配置文件 TOML 语法错误: {p}\n  {exc}") from exc

    section = data.get("autopilot", {})
    if not isinstance(section, dict):
        raise ValueError(f"配置文件的 [autopilot] 段必须是表: {p}")

    cfg = AutopilotConfig(**section)
    validate_config(cfg)
    logger.info("autopilot 配置已加载: %s（enabled=%s）", p, cfg.enabled)
    return cfg


def validate_config(cfg: AutopilotConfig) -> None:
    """加载期校验。**任何一项不通过都抛**，不降级。"""
    problems: list[str] = []

    # ① ROI 表达式
    ev = RoiExprEvaluator(cfg.perception.ref_size)
    var_vals: dict[str, float] = {}
    for k, expr in cfg.roi_vars.items():
        try:
            c = ev.compile(expr, where=f"roi_vars.{k}")
            var_vals[k] = ev.eval_scalar(
                c, cfg.perception.ref_size[0], cfg.perception.ref_size[1]
            )
        except RoiExprError as exc:
            problems.append(f"roi_vars.{k}: {exc}")
    for alias, expr in cfg.regions.items():
        try:
            ev.compile(expr, where=f"regions.{alias}")
        except RoiExprError as exc:
            problems.append(f"regions.{alias}: {exc}")
    for name, entry in cfg.templates.items():
        if entry.roi:
            try:
                src = entry.roi
                if src.strip().startswith("@"):
                    src = cfg.resolve_region(src)
                c = ev.compile(src, where=f"templates.{name}.roi")
                ev.eval(c, 1920, 1080, vars=var_vals)
            except (RoiExprError, ValueError) as exc:
                problems.append(f"templates.{name}.roi: {exc}")

    # ② 模板文件
    for name, spec in cfg.template_specs().items():
        if not spec.path.exists():
            problems.append(f"模板文件不存在: {name} -> {spec.path}")

    # ③ 场景路由引用的模板必须已定义
    for scene, names in cfg.scene_routes.items():
        for n in names:
            if n not in cfg.templates:
                problems.append(f"scene_routes.{scene} 引用了未定义的模板: {n}")

    # ④ YOLO
    if cfg.perception.yolo.enabled:
        if not cfg.perception.yolo.class_names:
            problems.append("perception.yolo.enabled=true 但 class_names 为空")
        ypath = Path(cfg.perception.yolo.onnx_path)
        if not ypath.is_absolute():
            ypath = (AUTOPILOT_ROOT / ypath).resolve()
        if not ypath.exists():
            problems.append(
                f"YOLO ONNX 不存在: {ypath}\n"
                f"    请先跑 tools/train_yolo.py 或 tools/convert_to_onnx.py"
            )

    if problems:
        raise ValueError(
            "autopilot 配置校验失败（%d 项）：\n  - %s"
            % (len(problems), "\n  - ".join(problems))
        )


__all__ = [
    "AUTOPILOT_ROOT",
    "PLUGIN_ROOT",
    "AutopilotConfig",
    "CaptureConfig",
    "LoopConfig",
    "OcrConfig",
    "PerceptionConfig",
    "TemplateEntry",
    "YoloConfig",
    "load_config",
    "validate_config",
]
