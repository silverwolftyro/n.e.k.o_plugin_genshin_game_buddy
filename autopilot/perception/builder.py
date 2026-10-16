"""状态包构建：融合识别结果 → StatePacket。

对应架构文档 4.1。

三条关键设计：
  1. **一轮只截一次图**：frame 由外部传入，builder 内部绝不重截。
  2. **场景路由**：先判 scene，再按 scene 决定跑哪些模板/OCR。
     对全屏识别 + 数十个模板匹配，不做路由的话每帧都要跑全部。
  3. **每个失败产生 Event**，不静默返回 None 了事（原则 1）。

★ §28 场景判定降级为二值
----------------------
原先的 7 类场景 YOLO **在这个工作区里根本没有模型**
（``models/trained/`` 是空的，``models/pretrained/`` 只有 BetterGI 的
bgi_*.onnx），所以那条路径一直在跑 ``_scene_from_templates_only`` 的退化分支。
现改用**已冻结的** :class:`SceneDetector`（session_a 1512 帧实测 98.25%）::

    scene_state = self._scene_detector.update(frame.image, ts_sec)
    scene.id    = "WORLD" if scene_state.is_world else "UNKNOWN"

⚠️ **三个必须知道的副作用**（详见规划文档 §28）：

* **"WORLD" 不在 ``contracts.SCENE_IDS`` 里**。而
  ``policy/rule_loader.py:482`` 会拿 ``VALID_SCENE_IDS`` 校验规则的
  ``scenes`` 字段 —— 所以**目前没有任何规则能引用 WORLD**。
* **模板匹配现在完全不跑**。``_scene_routes`` 里 WORLD / UNKNOWN 都是空表
  （PaimonMenu / BagGrid / EnemyHealthBar 是按 7 类场景设计的）。
* **`perception_ok` 恒为 True**。§21 把它收窄成"只由 YOLO 决定"，
  而 YOLO 现在不参与场景判定了。见 :meth:`_compute_perception_ok`。

🔴 **跨仓库硬依赖**：:class:`SceneDetector` 默认判定器要动态加载
   ``coord_scene_tree.py``（``D:\\NEKO\\recognition\\...\\scripts\\misc\\``），
   可用环境变量 ``NEKO_SCENE_TREE`` 覆盖。**缺文件时
   :meth:`build` 抛 FileNotFoundError** —— 故意不吞。静默判成"非大世界"
   会让定位永远不启动，且看不出原因。

★ §21 容错：瞬时故障 vs 持续故障
-------------------------------
``core/loop.py:525`` 对 ``perception_ok=False`` 是**零阈值立即**
``_auto_pause()`` —— 一帧就能让整个 autopilot 自停。而
``loop.py:383`` 的 ``_note_crash`` 里**无条件** ``_release_all()`` ——
一次异常就把按住的键全松开。

所以三个子系统各自挂一个 :class:`fault_guard.FaultGuard`（连续失败 N 帧
判死、连续成功 M 帧复活，**双向迟滞**）：

=============  ======  =======  ==============================
子系统          fail    recover  健康状态表达在
=============  ======  =======  ==============================
YOLO           3 帧    2 帧     ``resource.perception_ok``
OCR            5 帧    2 帧     ``ocr_available`` + ``ocr_fail_streak``
定位            3 帧    2 帧     ``position.quality = "LOST"``
=============  ======  =======  ==============================

**``perception_ok`` 收窄到只剩 YOLO**（用户 2026-10-02 拍板）——
OCR 和定位各有专属通道，且 ``PolicyContext`` 已经在读它们，
让它们去翻总开关是倒退。

⚠️ 唯一会抛的是**时间戳倒退**：那是调用方的契约错误，不是感知失败。
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..contracts.events import Event, EventCategory, Severity
from ..contracts.state_packet import (
    SCENE_WORLD,
    PlayerInfo,
    PositionInfo,
    ResourceInfo,
    SceneInfo,
    StatePacket,
)
from .capturer import CapturedFrame
from .fault_guard import FaultGuard
from .hp_estimator import BarParams, estimate_fill_ratio
from .ocr import OcrEngine
from .position_mapping import position_from_state
from .roi_expr import CompiledRoi, RoiExprEvaluator, RoiExprError
from .scene_detector import SceneDetector
from .scene_id_debouncer import SceneIdDebouncer
from .screen_region import ImageRegion, region_of_frame
from .template import TemplateMatch, TemplateMatcher
from .yolo import Detection, YoloDetector

logger = logging.getLogger(__name__)

#: 画面亮度低于此值判为黑帧（游戏最小化/切屏）。
#: 2.0 的 screen_capture.is_black_frame 已有同类判断，这里取同量级。
BLACK_FRAME_LUMA = 8.0

# --------------------------------------------------------------------------
# 故障守卫阈值（帧数）
#
# ⚠️ 这些不是拍脑袋的。``tick_rate_hz`` 默认 5.0（200 ms/帧），
#    全部落在 600 ms ~ 2 s；换算与依据见 fault_guard.py 模块文档。
#
# ★ 为什么只有 YOLO 能影响 ``perception_ok``
#   ``core/loop.py:525`` 看到 perception_ok=False 会**零阈值立即
#   _auto_pause()**（没有任何连续计数）。所以必须有帧级阈值先扛住
#   瞬时抖动。OCR 和定位各有专属健康通道，不需要也不应该翻这个总开关。
# --------------------------------------------------------------------------

#: YOLO 判死帧数。对齐 ``capturer.DEFAULT_FAIL_THRESHOLD = 3``（600 ms）。
YOLO_FAIL_THRESHOLD: int = 3
YOLO_RECOVER_THRESHOLD: int = 2

#: OCR 判死帧数。**必须 >= ``ocr.DEFAULT_FAIL_THRESHOLD = 5``** ——
#: builder 不能比它包装的模块更早判死（1.0 s）。
OCR_FAIL_THRESHOLD: int = 5
OCR_RECOVER_THRESHOLD: int = 2

#: 定位判死帧数。后果轻（只影响 position），可以与 YOLO 同档（600 ms）。
LOCATOR_FAIL_THRESHOLD: int = 3
LOCATOR_RECOVER_THRESHOLD: int = 2

#: 世界实体检测的**默认节流间隔**（tick）。转发自
#: :data:`world_entities.DEFAULT_ENTITY_INTERVAL_TICKS`（单一事实来源）。
DEFAULT_ENTITY_INTERVAL_TICKS: int = 5


def _pick_entity_interval(explicit: int | None, from_detector) -> int:
    """定节流间隔。优先级：显式参数 > 检测器自带的建议 > 默认 5。

    ⚠️ 间隔必须 >= 1。传 0 或负数会让 ``% n`` 直接抛 ZeroDivisionError /
       永不触发 —— 响亮失败，不静默纠正。
    """
    n = explicit if explicit is not None else from_detector
    if n is None:
        return DEFAULT_ENTITY_INTERVAL_TICKS
    n = int(n)
    if n < 1:
        raise ValueError(f"entity_interval_ticks 必须 >= 1，收到 {n}")
    return n


#: ``"auto"`` 模式的进程级缓存。**只探一次**。
#: 实测一次 ``StateBuilder(entity_detector="auto")`` 要 **1243 ms**
#: （加载 10 MB 模型 + 建 DirectML 会话 + 3 帧探测推理；随机器负载在
#: 1.2~1.4 s 之间浮动）。缓存之后后续构造接近 **0 ms** ——
#: 测试与工具里到处 new StateBuilder 才付得起。
#:
#: ⚠️ 缓存的三个已知副作用（:func:`clear_auto_cache` 是清理出口）：
#:   1. **测试污染** —— 多个 StateBuilder 共享同一个检测器实例
#:   2. **配置漂移** —— 运行期改 ``paths.BGI_WORLD_ONNX`` 后缓存不失效
#:   3. **多实例** —— 多个 builder 共用一个 ONNX 会话（省显存，但
#:      "谁先构造谁说了算"）
#: 生产环境这三者都可接受（进程内配置本来就不该中途改）；
#: **测试里必须 :func:`clear_auto_cache`**，否则断言互相干扰。
_AUTO_CACHE: dict[str, Any] = {}


def clear_auto_cache() -> bool:
    """清空 ``"auto"`` 模式的进程级缓存。**返回是否原本有缓存。**

    什么时候需要调：

    * **测试** —— 断言"两次构造拿到同一实例"之后，或想走
      "缺模型降级"分支时（缓存里可能已经躺着上一轮成功的结果）。
    * **运行期改了** :data:`perception.paths.BGI_WORLD_ONNX` /
      ``BGI_WORLD_CLASSES`` 的指向 —— 缓存不会自动失效。

    :meth:`StateBuilder.__init__` 里**不会**自动调它
    （那等于每次都付 1.2 s，生产上不可接受）。
    """
    had = bool(_AUTO_CACHE)
    _AUTO_CACHE.clear()
    return had


def _model_file_missing() -> bool:
    """模型**文件本身**在不在。用来给降级事件打 ``expected`` 标记。

    * 文件不存在 -> ``True``  —— **预期**（GPL 资产不随包分发）
    * 文件在但读不出来 -> ``False`` —— **该查**（损坏/权限/格式不对）

    ⚠️ 这里**刻意不吞宽谱异常**：上一次实现漏了 ``from pathlib import
    Path``，``Path(...)`` 抛 ``NameError`` 被 ``except Exception``
    吞掉，于是永远返回 ``False``，``expected`` 标记全是错的。
    真出意外就让它冒出去 —— 这个函数返回错值比抛异常危险得多。
    """
    from .world_entities import BGI_WORLD_ONNX  # noqa: PLC0415
    return not Path(BGI_WORLD_ONNX).is_file()


def _auto_entity_detector() -> tuple[Any, str]:
    """``"auto"`` 模式：探一次速，按结果决定开不开 / 每几个 tick 跑。

    降级时把一条 ``ENTITY_DETECTOR_UNAVAILABLE`` 事件**暂存**在缓存的
    ``pending_event`` 里，由 :meth:`StateBuilder._emit_auto_event` 取走
    并发出。**每个进程只发一次**（``pending_event`` 取走即清）。
    """
    if "result" in _AUTO_CACHE:
        return _AUTO_CACHE["result"]
    try:
        from .world_entities import (  # noqa: PLC0415
            ENTITY_ENABLE_THRESHOLD_MS, WorldEntityDetector, probe_speed,
            recommended_interval_ticks,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"auto 失败：模块不可用（{type(exc).__name__}）"
        _AUTO_CACHE["result"] = (None, msg)
        _AUTO_CACHE["pending_event"] = (
            Severity.WARN, msg,
            {"error": f"{type(exc).__name__}: {exc}", "reason": "import_failed",
             "expected": False})
        return _AUTO_CACHE["result"]
    try:
        det = WorldEntityDetector()
        det.load()
        ms = probe_speed(det)
        provider = det.load().current_provider()
        n = recommended_interval_ticks(ms)
        if n <= 0:
            msg = f"auto：{ms:.0f}ms 超阈值，保持关闭（{provider}）"
            result = (None, msg)
            _AUTO_CACHE["pending_event"] = (
                Severity.WARN, msg,
                {"measured_ms": round(ms), "provider": provider,
                 "threshold_ms": ENTITY_ENABLE_THRESHOLD_MS,
                 "reason": "over_budget", "expected": True})
        else:
            det.entity_interval_ticks = n
            result = (det, f"auto：{ms:.0f}ms，每 {n} tick（{provider}）")
    except Exception as exc:  # noqa: BLE001 - 缺模型等，不该炸构造
        msg = f"auto 失败：{exc}"
        result = (None, msg)
        # ⚠️ 用 WARN 不用 ERROR：bgi_* 是 GPL 资产、**不随包分发**，
        #    所以"模型不在"是**预期状态**（本仓库绝大多数机器都这样）。
        #    用 ERROR 会在每次正常启动时报警 —— 那是告警疲劳。
        #    真要报错的是"模型在但读不出来"，那要看 context 里的 error。
        #
        # ★ ``expected`` 进一步区分"文件根本不在"与"在但坏了"：
        #    前者是预期（大多数机器），后者才是真正该查的问题。
        _AUTO_CACHE["pending_event"] = (
            Severity.WARN, msg,
            {"error": f"{type(exc).__name__}: {exc}", "reason": "load_failed",
             "expected": _model_file_missing(),
             "expected_when": "bgi_* 是 GPL 资产，不随包分发"})
    _AUTO_CACHE["result"] = result
    return result


#: ``entity_detector`` 参数的**字符串关键字**。除此之外还接受
#: ``None``（关闭）和检测器实例（外部注入，鸭子类型）。
ENTITY_SPEC_AUTO: str = "auto"
ENTITY_SPEC_FORCE_ON: str = "force_on"
ENTITY_SPEC_FORCE_OFF: str = "force_off"
ENTITY_SPECS: frozenset[str] = frozenset(
    {ENTITY_SPEC_AUTO, ENTITY_SPEC_FORCE_ON, ENTITY_SPEC_FORCE_OFF})


def _forced_entity_detector() -> tuple[Any, str]:
    """``"force_on"``：跳过探测直接启用。**不碰 auto 缓存**。

    ★ 为什么要有它（§37）
    ------------------
    ``auto`` 的判据是「实测 p50 < 80 ms」，而 DirectML 的探测值在这台
    机器上实测漂移 **53~116 ms**（见
    :func:`perception.world_entities.probe_speed` 的记录）。后果不是
    「慢一点」，而是**同一台机器、不同负载给出相反结论** ——
    端到端验证因此**不可复现**：跑一次有敌人、跑一次没敌人，
    而系统本身没有任何问题。

    探测值适合回答「这台机器**够快**，值不值得开」，
    不适合回答「**能不能**开」。后者必须是显式决定。

    ⚠️ 节流间隔取 :data:`world_entities.DEFAULT_ENTITY_INTERVAL_TICKS`（5）
       —— 没探测就没有实测值可用，而 5 正是 30~80 ms 档的建议间隔，
       也恰好是 auto 探测通过时会给出的那个数。
    """
    try:
        from .world_entities import (  # noqa: PLC0415
            DEFAULT_ENTITY_INTERVAL_TICKS, WorldEntityDetector,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"force_on 失败：模块不可用（{type(exc).__name__}: {exc}）"
        _AUTO_CACHE["pending_event"] = (
            Severity.WARN, msg,
            {"error": f"{type(exc).__name__}: {exc}", "reason": "import_failed",
             "expected": False, "spec": ENTITY_SPEC_FORCE_ON})
        return None, msg

    try:
        det = WorldEntityDetector()
        # ⚠️ 必须真的 load —— 显式要求开启却开不了，是最需要被看见的
        #   一种失败。静默返回 None 的话，用户会以为在跑、其实
        #   world_entities 恒为 None，has_nearby_enemy() 恒为 UNKNOWN。
        provider = det.load().current_provider()
        det.entity_interval_ticks = DEFAULT_ENTITY_INTERVAL_TICKS
        return (det, f"force_on：已启用，每 {DEFAULT_ENTITY_INTERVAL_TICKS} "
                     f"tick（{provider}，**未探测**）")
    except Exception as exc:  # noqa: BLE001 - 缺模型要响亮，不能炸构造
        msg = f"force_on 失败：{exc}"
        _AUTO_CACHE["pending_event"] = (
            Severity.WARN, msg,
            {"error": f"{type(exc).__name__}: {exc}", "reason": "load_failed",
             "expected": _model_file_missing(),
             "expected_when": "bgi_* 是 GPL 资产，不随包分发",
             "spec": ENTITY_SPEC_FORCE_ON})
        return None, msg


def _resolve_entity_detector(spec: Any) -> tuple[Any, str]:
    """把 ``entity_detector`` 参数解成 ``(检测器 | None, 说明)``。

    * ``None``          -> 关闭，**不加载 GPL 模型**
    * ``"force_off"``   -> 显式关闭（工具/脚本用它排除干扰），同上
    * ``"force_on"``    -> 建检测器、**不探测**、直接启用（§37）
    * ``"auto"``        -> 建检测器、探一次速、按实测决定（**进程级只探一次**）
    * 其他对象          -> 直接用（鸭子类型）

    ⚠️ ``"auto"`` 第一次会真的加载模型并推理几帧（约 1.4 秒）。
    之后走缓存，不再付。

    ⚠️ 未知**字符串**直接抛 ``ValueError``（§37，响亮失败）。原先的
    实现把任何不匹配的值都当"外部注入的对象"放过去，于是
    ``"force-on"`` 这种拼写错误会一路流到 ``det.detect()`` 才炸出
    ``AttributeError: 'str' object has no attribute 'detect'`` ——
    报错点离错误点隔了四层，根本看不出是哪个参数写错了。
    """
    if spec is None:
        return None, "未配置（默认关闭）"
    # ⚠️ 先判 isinstance 再比字符串：检测器实例的 __eq__ 未必安全，
    #   而且下面要靠 isinstance 区分"关键字"与"外部注入的对象"。
    if isinstance(spec, str):
        if spec == ENTITY_SPEC_FORCE_OFF:
            return None, "force_off：显式关闭（不加载 GPL 模型）"
        if spec == ENTITY_SPEC_FORCE_ON:
            return _forced_entity_detector()
        if spec == ENTITY_SPEC_AUTO:
            return _auto_entity_detector()
        raise ValueError(
            f"未知的 entity_detector 取值 {spec!r}。合法字符串："
            f"{sorted(ENTITY_SPECS)}；也可以直接传 None（关闭）"
            f"或检测器实例（外部注入）。")
    return spec, "外部注入"


#: 血条参考资产（``autopilot/perception/templates/``）。缺失时 HP 估算自动降级。
HP_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_HP_REFERENCE_FILE = "hp_bar_reference.json"


def load_hp_reference() -> dict[str, Any] | None:
    """读实测的血条参考。**读不到返回 None**，由调用方降级成 UNKNOWN。

    ⚠️ 这个文件里只有**实测事实**（ROI 比例、填充色、外框色）。
      ``empty_background_bgr`` 是合成色，``empty_template_is_synthetic``
      标记为 True —— 引用它时别当成真机采集。
    """
    p = HP_TEMPLATE_DIR / _HP_REFERENCE_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _hp_roi_rect(ref: dict[str, Any], width: int,
                 height: int) -> tuple[int, int, int, int]:
    """把参考里的**比例** ROI 换算成本帧的像素框。

    ★ 为什么必须换算
      血条位置依赖分辨率。2560x1600 实测 ``(1076,1500,408,15)``，
      直接当像素用会在 1920x1080 上完全错位 —— 那时候读出来的是
      别的 UI 元素，得到的血量是**编出来的**。
    """
    fx, fy, fw, fh = ref["roi_frac"]
    x = max(0, min(int(round(fx * width)), max(0, width - 1)))
    y = max(0, min(int(round(fy * height)), max(0, height - 1)))
    w = max(0, min(int(round(fw * width)), width - x))
    h = max(0, min(int(round(fh * height)), height - y))
    return x, y, w, h


class StateBuilder:
    """把一帧变成 StatePacket。"""

    def __init__(
        self,
        *,
        capturer: Any = None,
        yolo: YoloDetector | None = None,
        template: TemplateMatcher | None = None,
        ocr: OcrEngine | None = None,
        roi: RoiExprEvaluator | None = None,
        locator: Any = None,
hp_estimation: Any = None,
        entity_detector: Any = "auto",
        entity_interval_ticks: int | None = None,
        scene_debouncer: SceneIdDebouncer | None = None,
        scene_detector: Any = None,
        game_id: str = "genshin_impact",
        scene_map: dict[str, str] | None = None,
        scene_routes: dict[str, list[str]] | None = None,
        teleport_ocr: bool = True,
        teleport_ocr_interval_s: float = 0.5,
    ) -> None:
        self._capturer = capturer
        if yolo is not None:
            # §28：YOLO 已退出场景判定（改用 SceneDetector 的二值结果）。
            # 参数保留只为签名兼容 —— 将来 7 类分类器回来会重新启用它。
            warnings.warn(
                "StateBuilder(yolo=...) 自 §28 起不再用于场景判定；"
                "场景改由 scene_detector（SceneDetector 二值）给出。"
                "该参数仅为签名兼容保留，将在 7 类分类器落地后重新启用。",
                DeprecationWarning, stacklevel=2)
        self._yolo = yolo
        self._template = template
        self._ocr = ocr
        self._roi = roi or RoiExprEvaluator()
        #: 定位服务（``LocalizationService`` 或任何有
        #: ``on_frame(frame_ndarray, ts_sec)`` 的对象）。``None`` = 不定位，
        #: ``position`` 恒为 None —— **与接入前行为完全一致**。
        self._locator = locator
        # ---- 传送契约 ① SELECT（2026-10-15 接上）------------------
        #: ★ 这段补的是一个**从来没接线**的洞。
        #:   ``LocalizationService`` 只认 ``on_teleport_ocr()``，而它的
        #:   调用方在全仓是 **0 个** —— 于是状态机永远停在 UNINITIALIZED,
        #:   ``position`` 恒为 ``LOST``，传送多少次都没用。
        #:
        #: 默认 **True**：开了 locator 就该是能用的。真正决定要不要跑的
        #: 是「locator 有没有 ``on_teleport_ocr``」——鸭子类型的老
        #: locator 没有这个方法，自动跳过，行为与之前完全一致。
        self._teleport_ocr = bool(teleport_ocr)
        self._tp_ocr_interval_s = max(0.0, float(teleport_ocr_interval_s))
        self._tp_ocr_next_ts = 0.0
        #: 上一次成功命中的目标名。用来抑制「每 0.5s 重复报同一条」。
        self._tp_last_name = ""
        #: 血量几何投影配置（``config.HpEstConfig``）。``None`` = 不做估算，
        #: ``PlayerInfo.hp_ratio`` 恒为 None —— **与接入之前行为完全一致**。
        self._hp_estimation = hp_estimation
        #: 世界实体检测器（``WorldEntityDetector`` 或任何有
        #: ``should_run_for_scene(str)`` + ``detect(frame)`` 的对象）。
        #:
        #: 四种取值：
        #:   * ``"auto"``（**默认**）—— 探一次速（进程级缓存），按实测决定开不开
        #:   * ``"force_on"``      —— **不探测**，直接启用（§37：可复现的验证用）
        #:   * ``"force_off"``     —— 显式关闭，**不加载 GPL 模型**
        #:   * ``None`` / 实例      —— 关闭 / 外部注入（鸭子类型）
        #:
        #: ⚠️ 未知字符串**抛 ValueError**，不当成外部注入放过去。
        #:
        #: ⚠️ **要"确定能开"就用 ``"force_on"``，别用默认的 "auto"**：
        #:    auto 的判据是实测 p50 < 80 ms，而 DirectML 探测值实测漂移
        #:    53~116 ms —— 骑在阈值上，**同一台机器会随机给出相反结论**，
        #:    端到端验证因此不可复现。
        #: ⚠️ **默认 "auto"** 是 2026-10-02 实测决定的：DirectML 下
        #:    bgi_world p50 = **66 ms**（batch_a/b 共 418 帧），落在
        #:    30~100 ms 档 -> 启用，间隔 5 tick（5 Hz 下 = 每秒 1 次）。
        #:    同一台机器纯 CPU 是 126 ms -> 当初只能保持关闭。
        #:
        #: ⚠️ **auto 缺模型会静默降级为关闭**（记在 ``_entity_note``）。
        #:    bgi_* 是 GPL 资产、不随包分发（见 notes/LICENSING.md），
        #:    所以没模型的机器上这一项就是关的 —— 这是预期行为，不是故障。
        self._entity_detector, self._entity_note = _resolve_entity_detector(
            entity_detector)
        #: 每几个 tick 跑一次实体检测（1 = 每 tick）。注入检测器时按
        #: ``entity_interval_ticks`` 参数；``"auto"`` 模式由实测耗时决定。
        self._entity_interval = _pick_entity_interval(
            entity_interval_ticks, getattr(self._entity_detector, "entity_interval_ticks", None))
        #: 自构造以来累计的 build 次数（节流计数器）。
        self._tick_count = 0
        #: :attr:`last_world_entities` 距离上次**真正**检测过了几个 tick。
        #: 0 = 本 tick 刚跑的。**上层必须看它** —— 节流期间那份数据是旧的。
        self.last_entity_age_ticks = 0
        self._game_id = game_id

        #: YOLO 标签 -> 内部场景 ID。
        #:
        #: ⚠️ **当前不用于场景判定**（§28 起场景走 :class:`SceneDetector` 的
        #:    二值结果）。保留是为了将来接回 7 类分类器时不必重写。
        self._scene_map = scene_map or {
            "main_menu": "MAIN_MENU",
            "bag_inventory": "BAG_INVENTORY",
            "bag_character": "BAG_CHARACTER",
            "map_big": "MAP_BIG",
            "combat": "COMBAT",
            "dialog": "DIALOG",
            "loading": "LOADING",
        }
        #: 场景 ID -> 该场景下要跑的模板名。
        #:
        #: ⚠️ **WORLD / UNKNOWN 目前都是空路由** —— 现有模板（PaimonMenu /
        #:    BagGrid / EnemyHealthBar）是按 BetterGI 的 7 类场景设计的，
        #:    二值化之后没有对应的场景 id 可挂。**这意味着模板匹配目前
        #:    完全不跑**（见 §28.4），等 7 类分类器回来才恢复。
        self._scene_routes = scene_routes or {
            "MAIN_MENU": ["PaimonMenu", "PartySlot1"],
            "BAG_INVENTORY": ["BagGrid"],
            "COMBAT": ["EnemyHealthBar"],
            "WORLD": [],
            "UNKNOWN": [],
        }
        self._events: list[Event] = []
        self._scene_entered_ms: dict[str, int] = {}
        self._state_version = 0
        self._specs: dict[str, Any] = {}
        self._roi_exprs: dict[str, str] = {}
        self._roi_var_values: dict[str, float] = {}
        self._last_yolo_ms = 0.0
        self._last_template_ms = 0.0
        #: 场景 ID 的第二层去抖。**默认 None = 不串**（见
        #: :meth:`_resolve_scene_binary` 的延迟实测）。传实例才启用。
        self._scene_debouncer = scene_debouncer
        #: 场景判定器（:class:`SceneDetector` 或任何有
        #: ``update(image, ts_sec)`` 且返回 ``is_world`` / ``changed`` 的对象）。
        #:
        #: 🔴 **跨仓库硬依赖**（§28.5）：``SceneDetector`` 的默认判定器要动态
        #:    加载 ``coord_scene_tree.py``，默认在
        #:    ``D:\\NEKO\\recognition\\exports\\benchmark\\scripts\\misc\\``，
        #:    可用环境变量 ``NEKO_SCENE_TREE`` 覆盖（也可调
        #:    :func:`scene_detector.set_classifier_module_path`）。
        #:    缺文件时 :meth:`SceneDetector.update` **抛 FileNotFoundError**，
        #:    本模块**不吞** —— 宁可启动失败，也不要静默把所有帧判成
        #:    "非大世界"（那会让定位永远不启动，且看不出原因）。
        self._scene_detector = (scene_detector if scene_detector is not None
                                else SceneDetector())
        #: 上一帧的秒级时间戳。**倒退即抛 ValueError**（不钳零）。
        self._last_ts: float | None = None
        # ---- 三个子系统各自的故障守卫（见 §21）
        #: **只有 YOLO 能让 perception_ok 变 False** —— 因为
        #: ``core/loop.py:525`` 对它是零阈值立即自停，必须先有阈值。
        self._yolo_guard = FaultGuard(
            fail_threshold=YOLO_FAIL_THRESHOLD,
            recover_threshold=YOLO_RECOVER_THRESHOLD)
        #: OCR 有专属健康通道（``ocr_available`` + ``ocr_fail_streak``），
        #: 且 ``PolicyContext.ocr_ok`` 已经在读它们 —— **不碰 perception_ok**。
        self._ocr_guard = FaultGuard(
            fail_threshold=OCR_FAIL_THRESHOLD,
            recover_threshold=OCR_RECOVER_THRESHOLD)
        #: 定位失败只影响 ``position.quality``，**不碰 perception_ok**。
        #: 定位挂了 YOLO/OCR 还能干活，决策层不至于完全停摆。
        self._locator_guard = FaultGuard(
            fail_threshold=LOCATOR_FAIL_THRESHOLD,
            recover_threshold=LOCATOR_RECOVER_THRESHOLD)
        #: 上一帧的**成功**产出。子系统瞬时失败时继续用它们，
        #: 而不是返回一个 UNKNOWN 空壳把下游全部污染。
        self._last_good_scene: Optional[dict[str, Any]] = None
        self._last_good_position: Any = None
        #: 本帧真实产出的检测框与匹配。**调试叠加层直接读这里**，
        #: 不重新跑一遍识别 —— 这就是原则 4「工具 = 运行时」。
        self.last_detections: list[Detection] = []
        self.last_matches: list[TemplateMatch] = []
        #: 本帧检出的世界实体（**不进 StatePacket**，F2 方案）。
        #: ``None`` = 没跑（未注入检测器 / 场景不允许 / 节流未到点）；
        #: ``[]`` = 跑了但确实没有。对应 PolicyContext 的 UNKNOWN / False 三态。
        self.last_world_entities: list | None = None
        #: 上一次世界实体检测的耗时（毫秒）。
        self.last_entity_ms: float = 0.0

        # ★ 必须放在 `_events` 建好之后 —— 降级事件要往里塞。
        self._emit_auto_event()

    # -- 事件出口 --------------------------------------------------------

    def drain_events(self) -> list[Event]:
        """取出并清空本轮产生的事件。"""
        evs, self._events = self._events, []
        return evs

    def _emit(self, ev: Event) -> None:
        self._events.append(ev)

    def _emit_auto_event(self) -> None:
        """把 ``"auto"`` 降级事件发出去（**每进程只发一次**）。

        为什么要发事件：``entity_detector="auto"`` 缺模型时会**静默降级**
        为关闭，只写在 ``_entity_note`` 里。属性没人读 = 等于没发生 ——
        这正是本仓库「静默失败禁令」要治的病。

        一次性的实现：事件暂存在 ``_AUTO_CACHE["pending_event"]``，
        :meth:`_emit` 取走时立刻从缓存里 ``pop`` 掉。

        ⚠️ 事件挂在**触发降级的那一个 builder** 上。若那个 builder 在
        ``drain_events()`` 之前就被丢弃，事件会丢。正常流程不会
        （loop 是构造完就一直用），但要知道这个边界。
        """
        pending = _AUTO_CACHE.pop("pending_event", None)
        if pending is None:
            return
        sev, msg, context = pending
        self._emit(Event(
            severity=sev, category=EventCategory.PERCEPTION,
            code="ENTITY_DETECTOR_UNAVAILABLE",
            message=msg, context=context,
        ))

    # -- 主流程 ----------------------------------------------------------

    def build(self, frame: CapturedFrame, *, forced_scene: str | None = None) -> StatePacket:
        """构建状态包。

        任何一步的失败都体现在 resource.perception_ok 和事件里。

        ⚠️ 会抛的两种情况：
           1. **时间戳倒退**（``ValueError``）—— 调用方的契约错误，不是
              感知失败；钳零会让去抖计时错乱并一路污染 ``entered_ms``。
           2. **SceneDetector 的底层模块缺失**（``FileNotFoundError``，
              §28.5）—— **故意不吞**。静默判成"非大世界"会让定位永远
              不启动，且看不出原因。
        """
        t_start = time.perf_counter()
        self._state_version += 1
        sv = self._state_version

        if frame is None or frame.image is None or frame.image.size == 0:
            self._emit(Event.screenshot_failed("empty frame", consecutive=1))
            return StatePacket.unknown(
                game_id=self._game_id, state_version=sv, reason="empty_frame"
            )

        region = region_of_frame(frame.image, frame.rect[0], frame.rect[1])

        # ① 黑帧检测（廉价且能救一大类"游戏没开/切屏"的问题）
        luma = _luma(frame.image)
        if luma < BLACK_FRAME_LUMA:
            self._emit(Event.black_frame(luma))
            return StatePacket.unknown(
                game_id=self._game_id, state_version=sv, reason="black_frame"
            )

        # ② 统一时间轴。**去抖器和定位服务共用这一个秒级时钟**，
        #    所以单调性检查也只做一次。
        ts_sec = frame.timestamp_ms / 1000.0
        if self._last_ts is not None and ts_sec < self._last_ts:
            raise ValueError(
                f"时间戳倒退: {ts_sec:.6f}s < 上一次 {self._last_ts:.6f}s"
                f"（frame.timestamp_ms 减小了）")
        self._last_ts = ts_sec

        # ③ 场景判定：SceneDetector 二值（WORLD / UNKNOWN）
        detections: list[Detection] = []
        scene = self._resolve_scene_binary(frame.image, ts_sec)
        scene_changed = bool(scene.pop("changed", False))
        if forced_scene:
            # 显式覆盖优先：跳过检测器，但**不**沿用二值结果的 confidence
            scene["id"] = forced_scene
            scene["raw_label"] = forced_scene
            scene["evidence"] = ("forced",)
            scene["confidence"] = 1.0
            scene_changed = True
        scene["entered_ms"] = self._track_scene_enter(scene["id"], scene_changed)

        # ④ 按场景路由跑模板
        template_ms = 0.0
        matches = self._run_scene_templates(scene["id"], region, sv)
        detections = detections if isinstance(detections, list) else []
        for m in matches:
            detections.append(Detection(f"tpl:{m.name}", m.box, m.score))
        # 供调试叠加层复用（原则 4：画的就是本帧真实产出）
        self.last_detections = list(detections)
        self.last_matches = list(matches)

        # ⑤ OCR。**只填健康字段，不碰 perception_ok**（见 §21）。
        ocr_available, ocr_streak = self._ocr_health(sv)

        # ⑥ 玩家信息（v1 最小集：只填能从模板/几何直接拿到的）
        #    血量几何投影默认关闭（hp_estimation=None）；关着时 hp_ratio 恒 None。
        hp = self._estimate_hp(frame.image)
        player = self._build_player(scene["id"], region, matches, detections,
                                    sv, hp=hp)

        # ⑦ 传送契约 ① SELECT（2026-10-15 接上）。
        #    **必须在 _locate 之前**：on_teleport_ocr 把状态机切进
        #    RESETTING，而 on_frame 在 RESETTING 下会走另一条分支。
        #    顺序反了的话，本帧仍然按 UNINITIALIZED 处理，延迟一帧才生效。
        self._feed_teleport_ocr(frame.image, scene["id"], ts_sec, sv)

        # ⑧ 定位。locator 未注入时 position 恒为 None —— 与接入前一致。
        position, locator_ms = self._locate(frame.image, ts_sec)

        # ⑨ 世界实体。**默认不跑**（entity_detector 未注入）。
        #    F2 方案：结果**不进 StatePacket**，只落在 builder 上供
        #    debug_overlay 与 PolicyContext 派生属性消费。
        entity_ms = self._detect_entities(frame.image, scene["id"], sv)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        yolo_ms, template_ms = self._last_yolo_ms, self._last_template_ms

        return StatePacket(
            scene=SceneInfo(
                id=scene["id"],
                confidence=scene["confidence"],
                evidence=tuple(scene["evidence"]),
                entered_ms=scene["entered_ms"],
                raw_label=scene["raw_label"],
            ),
            player=player,
            position=position,      # 无定位时为 None；**绝不用预测值填补**（原则 1）
            resource=ResourceInfo(
                fps=round(fps, 2),
                tick_ms=int(round(elapsed_ms)),
                perception_ok=self._compute_perception_ok(),
                ocr_available=ocr_available,
                ocr_fail_streak=ocr_streak,
                yolo_ms=round(yolo_ms, 2),
                template_ms=round(template_ms, 2),
                locator_ms=round(locator_ms, 2),
            ),
            game_id=self._game_id,
            timestamp_ms=frame.timestamp_ms,
            state_version=sv,
        )

    # -- 容错（§21）------------------------------------------------------

    def _compute_perception_ok(self) -> bool:
        """``perception_ok``。⚠️ **§28 之后这个信号实际上恒为 True。**

        历史：§21 把它收窄成"只由 YOLO 决定"。但 §28 把 YOLO 从场景判定
        里拿掉了（改用 :class:`SceneDetector` 的二值结果），于是
        ``_yolo_guard`` **再也不会被记失败** -> 这里恒返回 True。

        **这是 §28 引入的可观测性回退**，不是设计意图：
        现在"感知层不健康"只剩两条会产出包而非抛异常的路径 ——
        空帧 / 黑帧，而它们走的是 :meth:`StatePacket.unknown` 的
        early-return（``perception_ok=False``）。

        :class:`SceneDetector` 本身**没有健康信号**：它要么给出 is_world，
        要么直接抛（缺 ``coord_scene_tree.py``）。所以这里没有可接的口子。

        保留这个 guard 是为了 7 类分类器回来时有地方接。
        """
        return not self._yolo_guard.is_failed

    def _ocr_health(self, sv: int) -> tuple[bool, int]:
        """取 OCR 健康字段。返回 ``(ocr_available, ocr_fail_streak)``。

        ★ **不碰 ``perception_ok``** —— 用户 2026-10-02 拍板：
          OCR 走自己的专属通道，不再翻总开关。
        """
        if self._ocr is None:
            self._ocr_guard.record_success()
            return True, 0

        available = bool(getattr(self._ocr, "available", True))
        if available:
            if self._ocr_guard.record_success():
                logger.info("OCR 连续 %d 帧可用 -> 复活",
                            self._ocr_guard.recover_threshold)
                self._emit_guard_event(
                    "OCR_RECOVERED", Severity.INFO,
                    "OCR 已复活", {}, sv)
            streak = 0
        else:
            died = self._ocr_guard.record_failure()
            if died:
                logger.error("OCR 连续 %d 帧不可用 -> 判死 "
                             "（perception_ok 不受影响）",
                             self._ocr_guard.fail_threshold)
                self._emit_guard_event(
                    "OCR_DEAD", Severity.ERROR,
                    f"OCR 连续 {self._ocr_guard.fail_threshold} 帧不可用，判死",
                    {"streak": self._ocr_guard.consecutive_failures}, sv)
            else:
                logger.warning("OCR 不可用 %d/%d 帧，沿用上一帧数值",
                               self._ocr_guard.consecutive_failures,
                               self._ocr_guard.fail_threshold)
            self._emit(Event.ocr_failed(
                self._ocr_guard.consecutive_failures,
                getattr(self._ocr, "last_error", ""), sv))
            streak = self._ocr_guard.consecutive_failures

        return (not self._ocr_guard.is_failed), streak

    def _emit_guard_event(self, code: str, sev: Severity, msg: str,
                          context: dict[str, Any], sv: int) -> None:
        """守卫状态翻转时发一条事件。

        ⚠️ ``StatePacket`` **没有 note 字段**（B5：契约里没有就不改契约），
        所以降级信息走**日志 + Event** 两条既有通道。
        """
        self._emit(Event(
            severity=sev, category=EventCategory.PERCEPTION, code=code,
            message=msg, context=context, state_version=sv,
        ))

    def _resolve_scene_binary(self, image, ts_sec: int | float) -> dict[str, Any]:
        """**二值场景判定**（§28）：``SceneDetector`` -> ``WORLD`` / ``UNKNOWN``。

        :returns: ``{"id", "changed", "evidence", "entered_ms", "raw_label", "confidence"}``

        ⚠️ **debouncer 默认不串**（``scene_debouncer=None``）。
           实测（§28.3）：``SceneDetector`` 的 N=3.0/M=1.0 已经在去抖，
           再串一层同参数的 :class:`SceneIdDebouncer` 会让延迟
           **+2.9 s（双向都是）**，几乎翻倍 —— 玩家关个菜单要等 8.8 s
           才反映。传 ``scene_debouncer=`` 可以打开（留给未来的
           7 类分类器）。

        :raises FileNotFoundError: 底层 ``coord_scene_tree.py`` 缺失。
            **故意不吞** —— 静默判成"非大世界"会让定位永远不启动。
        """
        st = self._scene_detector.update(image, ts_sec)
        is_world = bool(st.is_world)
        changed = bool(st.changed)
        scene_id = "WORLD" if is_world else "UNKNOWN"

        if self._scene_debouncer is not None:
            d = self._scene_debouncer.update(scene_id, ts_sec)
            scene_id, changed = d.scene_id, d.changed

        return {
            "id": scene_id,
            "confidence": 1.0 if is_world else 0.0,
            "evidence": (f"scene_detector:is_world={int(is_world)}",),
            "entered_ms": None,
            "raw_label": "world" if is_world else "not_world",
            "changed": changed,
        }

    # -- 场景 ------------------------------------------------------------

    def _track_scene_enter(self, scene_id: str, changed: bool) -> int:
        """记录"进入该场景"的时刻。

        触发源是场景判定器的 ``changed`` **去抖沿**（§28 起是
        :class:`SceneDetector.changed`，N=3.0/M=1.0），不是每帧比对
        原始 id —— 后者会被单帧抖动带跑。
        """
        now = int(time.time() * 1000)
        if changed or scene_id not in self._scene_entered_ms:
            self._scene_entered_ms[scene_id] = now
        return self._scene_entered_ms.get(scene_id, now)

    def _detect_entities(self, image, scene_id: str, sv: int) -> float:
        """跑世界实体检测（带节流）。返回**本 tick**耗时（毫秒）。

        ★ **不进 StatePacket**（F2 方案）。结果落在
          :attr:`StateBuilder.last_world_entities`，供 ``debug_overlay``
          与 ``PolicyContext`` 的派生属性消费。

        ★ **节流**：默认每 :data:`DEFAULT_ENTITY_INTERVAL_TICKS`（5）个 tick
          才真跑一次。被跳过的 tick **保留上一次的结果**，并把
          :attr:`last_entity_age_ticks` 加一 —— **上层必须看这个字段**，
          否则会拿到"以为很新、其实很旧"的数据。

        ⚠️ **性能**：bgi_world 在纯 CPU 上实测 p50 126~147 ms
          （batch_a/b 共 418 帧），是 200 ms tick 预算的 63%~73%。
          节流只是把平均占用摊薄，**单次仍然超半帧**。要真解决得靠 GPU
          （见 notes §23）或降低 tick 频率。

        ⚠️ **None 与 [] 语义不同**（对应 PolicyContext 的三态）：

            * ``None`` —— **没跑**（未注入检测器 / 场景不允许 / 节流首帧前）
              → ``has_nearby_enemy()`` 返回 UNKNOWN，规则保守不触发
            * ``[]``  —— 跑了，确实没有 → 返回 False
        """
        self._tick_count += 1
        self.last_entity_age_ticks += 1
        det = self._entity_detector
        if det is None:
            self.last_world_entities = None
            self.last_entity_ms = 0.0
            return 0.0
        if not det.should_run_for_scene(scene_id):
            self.last_world_entities = None
            self.last_entity_ms = 0.0
            logger.debug("场景 %s 不跑世界实体检测", scene_id)
            return 0.0
        if self._tick_count % self._entity_interval != 0:
            # 节流命中：**保留上一次结果**，只把 age 加一
            return 0.0

        t0 = time.perf_counter()
        ents = det.detect(image)
        self.last_world_entities = list(ents)
        ms = (time.perf_counter() - t0) * 1000
        self.last_entity_ms = ms
        self.last_entity_age_ticks = 0          # 刚跑完，归零
        if ms > 200.0:
            logger.warning(
                "世界实体检测 %.0f ms，超过 200 ms 的 tick 预算（间隔 %d tick）",
                ms, self._entity_interval)
        logger.debug("世界实体 %d 个 (%.0f ms) scene=%s", len(ents), ms, scene_id)
        return ms

    # ------------------------------------------------------- 传送契约 ①
    def _feed_teleport_ocr(self, image, scene_id: str, ts_sec: float,
                            sv: int) -> None:
        """契约 ① SELECT：在传送地图里 OCR 出目标，喂给定位服务。

        ★ 为什么这里必须有它（2026-10-15）
        -----------------------------------
        ``LocalizationService.on_teleport_ocr()`` 在接线之前的调用方是
        **0 个**。而它自己明确「**不猜**这是不是传送」（§16.3 实测：
        帧间位移里混着大量 >3px 跳变，靠阈值区分传送与跑得快不可靠），
        契约把判据交给了 OCR。于是链条是这样断的::

            locator.enabled = true
              -> LocalizationService(...)          # 状态 = UNINITIALIZED
              -> builder._locate -> on_frame()      # 每帧都在跑
              -> on_frame: UNINITIALIZED 原地不动    # localization_service.py:327
              -> world_pos = None
              -> PositionInfo.lost()               # quality="LOST"

        传送多少次都没用，而日志里连一行 OCR 都没有 —— 因为
        ``teleport_ocr.recognize_teleport`` 同样没人调。

        触发条件：**非大世界帧**。那正是传送地图打开、信息框可见的时候；
        大世界里信息框根本不存在，跑 OCR 纯属浪费。

        失败一律吞掉并降级 —— 传送识别不到是**常态**（加载中、背包、
        任何非传送的弹窗），不能让感知跟着炸。
        """
        if not self._teleport_ocr or self._locator is None:
            return
        if not hasattr(self._locator, "on_teleport_ocr"):
            return                       # 鸭子类型 locator：与之前完全一致
        if scene_id == SCENE_WORLD:
            return                       # 大世界：信息框不在
        if ts_sec < self._tp_ocr_next_ts:
            return
        self._tp_ocr_next_ts = ts_sec + self._tp_ocr_interval_s

        try:
            from autopilot.perception import teleport_ocr as _tp
            target = _tp.recognize_teleport(image)
        except Exception as exc:  # noqa: BLE001
            # 最常见的是「找不到传送点数据文件」(FileNotFoundError)。
            # 那是一次性的环境问题，**每 0.5s 报一次会把日志淹掉**，
            # 所以只记 debug，并发一条 WARN 事件。
            logger.debug("传送 OCR 异常：%s: %s", type(exc).__name__, exc)
            self._emit(Event(
                severity=Severity.WARN, category=EventCategory.PERCEPTION,
                code="TELEPORT_OCR_FAILED",
                message=f"传送 OCR 异常：{type(exc).__name__}: {exc}",
                context={"streak": None, "max": None},
            ))
            return

        if target is None:
            return

        try:
            self._locator.on_teleport_ocr(target)
        except ValueError as exc:
            # 目标没有可用的三轴落点坐标 —— 响亮失败，但要带得动排查。
            logger.warning("传送目标落点非法，已跳过：%s", exc)
            self._emit(Event(
                severity=Severity.WARN, category=EventCategory.PERCEPTION,
                code="TELEPORT_TARGET_INVALID",
                message=str(exc), context={"name": getattr(target, "name", "")},
            ))
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("on_teleport_ocr 异常，已忽略：%s: %s",
                           type(exc).__name__, exc)
            return

        name = getattr(target, "name", "")
        first = name != self._tp_last_name
        self._tp_last_name = name
        if first:
            logger.info("传送契约① SELECT：识别到 %s（置信 %.2f）-> 进入 RESETTING",
                        name, float(getattr(target, "confidence", 0.0)))
            self._emit(Event(
                severity=Severity.INFO, category=EventCategory.PERCEPTION,
                code="TELEPORT_OCR_HIT",
                message=f"识别到传送目标 {name}",
                context={"name": name,
                         "matched_id": getattr(target, "matched_id", ""),
                         "confidence": round(
                             float(getattr(target, "confidence", 0.0)), 3)},
            ))

    def _locate(self, image, ts_sec: float):
        """跑定位服务。返回 ``(PositionInfo | None, 毫秒)``。

        ``self._locator`` 为 None 时**直接返回 ``(None, 0.0)``**，
        行为与接入定位前完全一致 —— 这是向后兼容的唯一保证。

        ★ 定位异常**不向上抛**（§21 收窄）。理由：``core/loop.py:513`` 收到
        异常会走 ``_note_crash``，而 ``_note_crash`` 里**无条件**
        ``_release_all()``（loop.py:383）—— 一次定位抖动就会把按住的键全松开。
        而且定位挂了 YOLO/OCR 还能干活，不该拖着整个 tick 一起崩。

        降级行为：
            * 单帧失败（未判死）-> 沿用 ``self._last_good_position``
            * 判死 -> ``position=None``（表达成 ``quality="LOST"``）
        """
        if self._locator is None:
            return None, 0.0

        t0 = time.perf_counter()
        try:
            loc_state = self._locator.on_frame(image, ts_sec)
        except Exception as exc:  # noqa: BLE001 - 见上方说明，不外抛
            self._emit(Event(
                severity=Severity.WARN, category=EventCategory.PERCEPTION,
                code="LOCATOR_FAILED",
                message=f"定位失败：{type(exc).__name__}: {exc}",
                context={"streak": self._locator_guard.consecutive_failures + 1,
                         "max": self._locator_guard.fail_threshold},
            ))
            died = self._locator_guard.record_failure()
            if died:
                logger.error("定位连续 %d 帧失败 -> 判死，position=None",
                             self._locator_guard.fail_threshold)
                self._emit_guard_event(
                    "LOCATOR_DEAD", Severity.ERROR,
                    f"定位连续 {self._locator_guard.fail_threshold} 帧失败，"
                    f"position 置 None（perception_ok 不受影响）",
                    {"streak": self._locator_guard.consecutive_failures}, 0)
            else:
                logger.warning("定位失败 %d/%d 帧，沿用上一帧位置",
                               self._locator_guard.consecutive_failures,
                               self._locator_guard.fail_threshold)
            if self._locator_guard.is_failed:
                # ⚠️ 这里给**显式 LOST 包**，不给 None。
                #    PolicyContext.pos_quality / pos_usable
                #    （policy/context.py:297, 304）在 position=None 时返回的是
                #    **UNKNOWN**，不是 False —— 规则拿到"不知道"而不是"不能用"。
                #    显式 LOST 才是"确定不能用"的语义。
                self._last_good_position = None
                return (PositionInfo(quality="LOST", source="locator_failed"),
                        (time.perf_counter() - t0) * 1000)
            if self._last_good_position is not None:
                return self._last_good_position, (time.perf_counter() - t0) * 1000
            # 连一帧都没成功过，无从沿用 -> 同样是"确定不可用"，不是"没配定位"
            return (PositionInfo(quality="LOST", source="locator_never_ok"),
                    (time.perf_counter() - t0) * 1000)

        revived = self._locator_guard.record_success()
        if revived:
            logger.info("定位连续 %d 帧成功 -> 复活", self._locator_guard.recover_threshold)
            self._emit_guard_event(
                "LOCATOR_RECOVERED", Severity.INFO, "定位已复活", {}, 0)
        if self._locator_guard.is_failed:
            # ⚠️ 迟滞未满：这帧虽然拿到了值，**也不采信**。否则会出现
            #    「已判死 -> 恢复第 1 帧就冒出 GOOD」的抖动，
            #    迟滞就白设了。保持 LOST 直到连续成功满 recover_threshold。
            return (PositionInfo(quality="LOST", source="locator_recovering"),
                    (time.perf_counter() - t0) * 1000)
        pos = position_from_state(loc_state)
        self._last_good_position = pos
        return pos, (time.perf_counter() - t0) * 1000

    # -- 模板 ------------------------------------------------------------

    def _run_scene_templates(
        self, scene_id: str, region: ImageRegion, sv: int
    ) -> list[TemplateMatch]:
        if self._template is None:
            return []
        t0 = time.perf_counter()
        out: list[TemplateMatch] = []
        for name in self._scene_routes.get(scene_id, []):
            spec = self._template_spec(name)
            if spec is None:
                self._emit(
                    Event(
                        severity=Severity.WARN, category=EventCategory.PERCEPTION,
                        code="TEMPLATE_NOT_CONFIGURED",
                        message=f"场景 {scene_id} 引用了未配置的模板 {name}",
                        context={"scene": scene_id, "template": name}, state_version=sv,
                    )
                )
                continue
            box = self._roi_box_for(name, region)
            try:
                m = self._template.find(region, spec, search_box=box)
            except Exception as exc:  # noqa: BLE001
                self._emit(
                    Event(
                        severity=Severity.WARN, category=EventCategory.PERCEPTION,
                        code="TEMPLATE_ERROR",
                        message=f"模板 {name} 匹配异常: {exc}",
                        context={"template": name, "error": str(exc)[:200]}, state_version=sv,
                    )
                )
                continue
            if m is None:
                acc = self._template.accumulator(name)
                if acc is not None and acc.max_consecutive_count > 0 and not acc.ever_settled:
                    self._emit(
                        Event.template_score_low(
                            name, acc.max_consecutive_count, acc.required_count,
                            acc.last_score, spec.threshold, sv,
                        )
                    )
                continue
            out.append(m)
        self._last_template_ms = (time.perf_counter() - t0) * 1000
        return out

    def _template_spec(self, name: str):
        """从配置取模板定义。未配置返回 None。"""
        specs: dict[str, Any] = getattr(self, "_specs", {}) or {}
        return specs.get(name)

    def set_templates(self, specs: dict[str, Any]) -> None:
        """注入模板定义（由 config/loader 调）。"""
        self._specs = dict(specs)

    def _roi_box_for(self, name: str, region: ImageRegion) -> tuple[int, int, int, int] | None:
        """取该模板的 ROI 表达式并求值。"""
        exprs: dict[str, str] = getattr(self, "_roi_exprs", {}) or {}
        src = exprs.get(name)
        if not src:
            return None
        try:
            compiled: CompiledRoi = self._roi.compile(src, where=f"template:{name}")
            return self._roi.eval(
                compiled, region.width, region.height,
                vars=self._roi_vars(region),
            )
        except RoiExprError as exc:
            # 表达式是加载期该校验的；运行期失败说明配置有问题 —— 响亮失败
            raise ValueError(f"模板 {name} 的 ROI 表达式无效: {exc}") from exc

    def _roi_vars(self, region: ImageRegion) -> dict[str, float]:
        v = getattr(self, "_roi_var_values", None) or {}
        return dict(v)

    def set_roi(self, exprs: dict[str, str], vars_: dict[str, float] | None = None) -> None:
        """注入 ROI 表达式与变量（由 config/loader 调）。"""
        self._roi_exprs = dict(exprs)
        self._roi_var_values = dict(vars_ or {})

    # -- 玩家 ------------------------------------------------------------

    def _estimate_hp(self, frame: np.ndarray) -> tuple[float | None, str]:
        """从本帧血条几何投影出 ``hp_ratio``。**拿不准返回 (None, "none")**。

        :returns: ``(hp_ratio, hp_source)``。``hp_source`` 取值
            ``"none"``（没启用/读不到/判不出来）、``"geometry"``（真读到了）。

        ★ 为什么这里绝不返回 0 或 1
          规则里 ``hp() < 0.35`` 会据此决定要不要吃药。猜一个「差不多空了」
          会让角色在满血时去吃药；猜一个「还满」则在濒死时不喝药 ——
          两种猜错的代价都远大于「这一帧不知道」。``hp_estimator`` 本身
          在满血/空血两档会返回 None，这里原样透传，不做任何补值。
        """
        cfg = self._hp_estimation
        if cfg is None or not getattr(cfg, "enabled", False):
            return None, "none"
        if frame is None or getattr(frame, "size", 0) == 0:
            return None, "none"

        ref = load_hp_reference()
        if ref is None:
            return None, "none"

        frac = getattr(cfg, "roi_frac", None)
        if frac is not None:
            x, y, w, h = (int(round(v * d)) for v, d in
                          zip(frac, (frame.shape[1], frame.shape[0],
                                     frame.shape[1], frame.shape[0])))
        else:
            x, y, w, h = _hp_roi_rect(ref, frame.shape[1], frame.shape[0])
        if w <= 0 or h <= 0:
            return None, "none"

        bg = getattr(cfg, "background_bgr", None)
        params = BarParams(
            border_inset=int(getattr(cfg, "border_inset", 1)),
            background_color=tuple(bg) if bg else None,
        )
        try:
            ratio = estimate_fill_ratio(frame[y:y + h, x:x + w], params)
        except Exception:                      # noqa: BLE001
            # 感知层任何异常都不该拖垮整条 tick —— 降级成 UNKNOWN 继续跑
            logger.exception("血量几何投影失败，本帧降级为 UNKNOWN")
            return None, "none"
        if ratio is None:
            return None, "none"
        return float(ratio), "geometry"

    def _build_player(
        self,
        scene_id: str,
        region: ImageRegion,
        matches: list[TemplateMatch],
        detections: list[Detection],
        sv: int,
        hp: tuple[float | None, str] | None = None,
    ) -> PlayerInfo:
        """v1 最小实现。

        血量：``hp`` 非空且 ``enabled`` 时由 :meth:`_estimate_hp` 几何投影给出；
        否则留空 = None = UNKNOWN，决策层会保守跳过（原则 2）——
        这比填一个猜出来的数值安全得多。

        能量：**仍留空**。它要在角色**有能量**的帧里标定，
        现有参考帧（满血、无队友）里压根没有能量条可测。
        """
        active: tuple[str, str, int] | None = None
        for d in detections:
            if d.label.startswith("char:"):
                active = (d.label[5:], d.label[5:], d.box[0])
                break
        party = ()
        if active:
            from ..contracts.state_packet import PartyMember

            party = (PartyMember(slot=1, char_id=active[0], name=active[1], active=True),)
        hp_ratio, hp_source = hp if hp else (None, "none")
        return PlayerInfo(hp_ratio=hp_ratio, hp_source=hp_source, party=party)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _luma(img: np.ndarray) -> float:
    import cv2  # noqa: PLC0415

    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def _scale_box(box: tuple[int, int, int, int], k: float) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return (int(x * k), int(y * k), int(w * k), int(h * k))


__all__ = [
    "BLACK_FRAME_LUMA",
    "DEFAULT_ENTITY_INTERVAL_TICKS",
    "ENTITY_SPEC_AUTO", "ENTITY_SPEC_FORCE_ON", "ENTITY_SPEC_FORCE_OFF",
    "ENTITY_SPECS",
    "StateBuilder",
    "YOLO_FAIL_THRESHOLD", "YOLO_RECOVER_THRESHOLD",
    "OCR_FAIL_THRESHOLD", "OCR_RECOVER_THRESHOLD",
    "LOCATOR_FAIL_THRESHOLD", "LOCATOR_RECOVER_THRESHOLD",
    "clear_auto_cache",
]
