"""世界实体检测（bgi_world.onnx）。

🔴 **许可证硬约束**（详见 ``recognition/notes/LICENSING.md``）
------------------------------------------------------------
模型是 **BetterGI（GPL-3.0）** 的产物，只在本地保留、**不随包分发**。
推理走 :mod:`perception.yolo` 的 **ONNX Runtime**（MIT），
**本模块禁止 import ultralytics**（AGPL-3.0，运行时链接即传染）。
缺模型时**抛** :class:`YoloUnavailableError`（响亮失败），
不静默返回空列表 —— 静默会让上层以为"世界里确实没东西"。

★ 实测能跑通什么（2026-10-02，全量扫描）
------------------------------------------
============  ====  ==========================================================
批次           帧数  各类检出数
============  ====  ==========================================================
batch_a        122   **7 类全覆盖**（health_bar 15 / enemy_identify 2 /
                     drops 5 / ore 3 / unmined_ore 13 / chest 5 / locked_chest 1）
batch_b        296   5 类（chest 32 / unmined_ore 13 / health_bar 6 /
                     drops 3 / locked_chest 2），**enemy_identify / ore 为 0**
============  ====  ==========================================================

CPU 单帧 **126 ms**（418 帧 p50），占 200 ms tick 预算的 63%。
**DirectML 下 66 ms（33%）** —— 2026-10-02 切换，见 §25。
所以**仍然需要节流**（``builder.entity_interval_ticks``，默认 5 tick），
见 :data:`WORLD_SCENES` 的场景门控。

★ 为什么 bbox 要归一化
----------------------
输入分辨率会变（720 基准 / 2560×1600 原生），下游规则不该关心。
``bbox`` 一律归一化到 ``[0, 1]``；``screen_pos`` 另给**720 基准的像素中心**，
方便直接和屏幕坐标 / ROI 体系对齐。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import BGI_WORLD_CLASSES, BGI_WORLD_ONNX
from .yolo import YoloDetector, YoloUnavailableError, available_providers

logger = logging.getLogger(__name__)

__all__ = [
    "BASE_SIZE", "DEFAULT_ENTITY_INTERVAL_TICKS",
    "ENTITY_EVERY_TICK_MAX_MS", "ENTITY_ENABLE_THRESHOLD_MS",
    "ENTITY_SLOWDOWN_FACTOR", "ENTITY_SLOW_INTERVAL_TICKS",
    "WORLD_ENTITY_CONF", "WORLD_ENTITY_TIMEOUT_MS",
    "WORLD_SCENES", "KIND_GROUPS", "WorldEntity", "WorldEntityDetector",
    "current_provider", "detect_world_entities", "default_detector",
    "load_class_names", "probe_speed", "recommended_interval_ticks",
]

#: 读一个正浮点环境变量，非法/缺失/非正都退回 ``default``。
#:
#: 为什么要环境变量而不是写死（2026-10-11）
#:   --------------------------------------------
#:   这个阈值是**按机器定的**：同一张卡在空闲时探到 63~71 ms、负载高时探到
#:   93 ms，写死一个数对每台机器都不对。而它又直接决定「实体检测开不开」，
#:   而 WORLD 场景下**唯一三条能触发的规则全靠实体检测**（见 rules/
#:   genshin_combat.toml 的 world_* 三条）—— 阈值高了，整条 WORLD 决策链
#:   就整条哑掉，而现象只是「卡片上没动作」，排查起来极难联想到它。
#:   既然它按机器而异，就不该编译进代码。
def _env_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r 不是数字，用默认值 %.0f", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s=%r 必须为正数，用默认值 %.0f", name, raw, default)
        return default
    if value != default:
        logger.info("%s=%s 生效（默认 %.0f）", name, value, default)
    return value


#: **自动启用的耗时上限**（毫秒）。p50 低于它才按常规频率跑。
#:
#: ⚠️ **可用环境变量 ``NEKO_ENTITY_THRESHOLD_MS`` 覆盖**。
#:   历史：100 → 80（2026-10-02 拍板）→ 120（2026-10-11）→ **200**（2026-10-14）。
#:
#:   **为什么一路调高**：本机 DirectML 实测 p50 在 **93 ~ 128 ms** 之间漂
#:   （跟负载、别的窗口、前后台都有关）。80 和 120 都被真实值骑过 ——
#:   症状是「实体检测整个关掉」，而 WORLD 场景下唯三能触发的规则
#:   （``world_attack_only_with_enemy`` / ``world_emergency_dodge`` /
#:   ``world_burst_only_with_enemy``）全都依赖 ``has_nearby_enemy()``，
#:   于是整条 WORLD 决策链哑掉，而现象只是「卡片上没动作」，
#:   排查时极难联想到它。**两次 demo 就是卡在这里。**
#:
#:   ★ 关键教训：**写死阈值必然被实测值骑过去**。现在默认抬到 200 覆盖
#:     已观测到的全部波动（128），并且超阈值**不再等于关闭**（见
#:     :func:`recommended_interval_ticks` 的降级档）。
#:     **换机器请用环境变量调**，别改这里。
ENTITY_ENABLE_THRESHOLD_MS: float = _env_positive_float(
    "NEKO_ENTITY_THRESHOLD_MS", 200.0)

#: 实测 p50 超过阈值时，**最多容忍到阈值的多少倍**。
#:
#: ★ 为什么要有这一档（2026-10-14）
#:   原来的三档是「超阈值就关」。但「关」意味着 WORLD 场景下**一条规则
#:   都触发不了**，而现象只是卡片没动 —— 这是最坏的一种失败：
#:   **宁可慢，也不要没有。**
#:   1.5 倍是「明显不对劲」与「只是慢一点」的分界：本机 128ms / 200ms = 0.64，
#:   属于「只是慢一点」，应当降频继续跑。
ENTITY_SLOWDOWN_FACTOR: float = 1.5

#: 降级档的间隔（每 N tick 跑一次）。5 Hz 下 = 每 2 秒一次。
ENTITY_SLOW_INTERVAL_TICKS: int = 10

#: 只有这一档真的划算（快到可以每 tick 跑）。
ENTITY_EVERY_TICK_MAX_MS: float = 30.0

#: 中间档（30~100 ms）建议的节流间隔：5 Hz 下 = **每秒 1 次**。
#: 实体位置以「秒」计，不需要 5 Hz 刷新。
DEFAULT_ENTITY_INTERVAL_TICKS: int = 5

#: 720 基准画面尺寸。``screen_pos`` 统一折算到这个坐标系。
BASE_SIZE: tuple[int, int] = (1280, 720)

#: 采信阈值。
#: ⚠️ **未标定**。取值依据：2026-10-02 对 batch_a 122 帧 + batch_b 296 帧
#:    全量扫描时，0.35 下两类零误报（``a:f00005`` 在 conf=0.10 都是 0 检出），
#:    且 ``chest`` 这类小目标在 0.5+ 会漏。**没有人工标注集做过 PR 曲线**，
#:    这个值是"看着能用"而不是"测出来的"。
WORLD_ENTITY_CONF: float = 0.35

#: **挂死守卫**（毫秒）。超过则丢弃本帧结果。
#:
#: ⚠️ **这不是性能预算。** 曾经按 300 ms 设过，结果是**随机丢帧**：
#:    同一批帧反复跑，检出数在 1~4 之间跳（实测 41/46 ~ 46/46 之间波动），
#:    因为耗时本身就在 250~430 ms 之间抖。一个 10%~50% 会触发的「超时」
#:    不是守卫，是掷骰子 —— 比慢更糟。
#:
#:    所以这里只保留一个**防挂死**上限。真正的时间预算要靠调度解决
#:    （见下方三条出路）。
#:
#: 实测（2026-10-02，batch_a + batch_b 共 418 帧，同一台 RTX 5070）：
#:
#:        provider            p50    p90    max    占 200ms tick
#:        ------------------  -----  -----  -----  --------------
#:        CPUExecutionProvider  126    181    338    63%
#:        DmlExecutionProvider   66     80    142    33%
#:
#: ⚠️ **上面这些数字只在"已降到 1280 宽"的前提下成立** —— 直接喂
#:    2304×1440 在 CPU 上是 p50 242 ms。降采样省掉约一半（见 :meth:`detect`）。
#:
#: ⚠️ **这个阈值不用改** —— 它是防挂死上限，不是性能预算。
#:    要看真实性能跑 ``scripts/misc/bench_world_entities.py``。
#:
#: **性能路线（§24~§25 的结论）**：
#:   1. ~~CUDA 13.x~~ ❌ 官方未证实支持 Blackwell(sm_120)，且 NVIDIA 说
#:      CUDA EP **不含 PTX**（不能 JIT 兜底）
#:   2. ✅ **DirectML** —— 只需 R570+ 驱动，p50 126 -> 66 ms（1.9×）
#:   3. ✅ **每 N 个 tick 跑一次** —— ``builder.entity_interval_ticks``，
#:      默认 5（5 Hz 下 = 每秒 1 次）
WORLD_ENTITY_TIMEOUT_MS: float = 1000.0

#: **只在这些场景跑**世界实体检测。其余场景（背包/地图/菜单）跑了纯浪费。
#: §28 起 builder 的场景判定是二值的（``SceneDetector`` -> WORLD/UNKNOWN），
#: 所以门控也改成对应的两值。``COMBAT`` 已移除 —— 没有 7 类分类器时它
#: **永远不会被命中**（原值是照着旧的 YOLO 场景表写的）。
#: 等分类器回来，把 MAIN_MENU / BAG_INVENTORY / COMBAT 等加回来。
WORLD_SCENES: frozenset[str] = frozenset({"WORLD", "UNKNOWN"})

#: 模型原始类名 -> 业务分组。给 ``PolicyContext`` 的派生属性用。
#: ⚠️ 分组是**业务约定**，不是模型语义。比如 ``health_bar`` 可能是敌人血条
#:    也可能是自己的，**本模块不做这个区分**（那是决策层的事）。
KIND_GROUPS: dict[str, str] = {
    "enemy_identify": "enemy",
    "health_bar": "combat",
    "drops": "drops",
    "ore": "ore",
    "unmined_ore": "ore",
    "chest": "chest",
    "locked_chest": "chest",
}


def load_class_names(path: str | Path | None = None) -> list[str]:
    """读类名表。一行一个，**顺序必须与模型的输出通道一致**。"""
    p = Path(path) if path else BGI_WORLD_CLASSES
    if not p.is_file():
        raise YoloUnavailableError(
            f"找不到类名表: {p}\n"
            f"（bgi_* 系列是 BetterGI GPL-3.0 资产，只在本地保留、"
            f"不随包分发。详见 recognition/notes/LICENSING.md）")
    names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    if not names:
        raise YoloUnavailableError(f"类名表是空的: {p}")
    return names


@dataclass(frozen=True)
class WorldEntity:
    """一个世界实体。

    :param kind: 模型原始类名（``enemy_identify`` / ``chest`` / ``ore`` / ...）
    :param bbox: ``(x, y, w, h)``，**归一化到 [0, 1]**，相对传入帧
    :param confidence: 模型置信度（已过 :data:`WORLD_ENTITY_CONF`）
    :param screen_pos: ``(cx, cy)`` 中心点，折算到 **720 基准**像素坐标
    """

    kind: str
    bbox: tuple[float, float, float, float]
    confidence: float
    screen_pos: tuple[int, int]

    @property
    def group(self) -> str:
        """业务分组（见 :data:`KIND_GROUPS`）。未知类名原样返回。"""
        return KIND_GROUPS.get(self.kind, "other")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "group": self.group,
                "bbox": [round(v, 4) for v in self.bbox],
                "confidence": round(self.confidence, 4),
                "screen_pos": list(self.screen_pos)}


class WorldEntityDetector:
    """bgi_world 封装。**懒加载模型**，构造本身很便宜。"""

    def __init__(self,
                 model_path: str | Path | None = None,
                 classes_path: str | Path | None = None,
                 conf: float = WORLD_ENTITY_CONF,
                 timeout_ms: float = WORLD_ENTITY_TIMEOUT_MS) -> None:
        self._model_path = Path(model_path) if model_path else BGI_WORLD_ONNX
        self._classes_path = Path(classes_path) if classes_path else BGI_WORLD_CLASSES
        self._conf = float(conf)
        self._timeout_ms = float(timeout_ms)
        #: 推理宽度上限。模型输入固定 1280×1280，宽于此的原图先等比降采样
        #: （实测省 71 ms 中位耗时，见 :meth:`detect` 的实测表）。
        self._infer_width = 1280
        self._det: Optional[YoloDetector] = None
        self._last_ms = 0.0
        self._last_count = 0

    # -- 生命周期 --------------------------------------------------------
    def load(self) -> YoloDetector:
        """加载模型。**已加载就直接返回**（重复调用无副作用）。"""
        if self._det is None:
            names = load_class_names(self._classes_path)
            self._det = YoloDetector(self._model_path, names)
            logger.info("bgi_world 已加载: %s providers=%s imgsz=%d",
                        self._model_path.name, self._det.providers,
                        self._det._imgsz)
        return self._det

    def warmup(self) -> None:
        """跑一张假图。避免第一帧把初始化耗时算进 tick。"""
        try:
            import numpy as np  # noqa: PLC0415
            self.detect(np.zeros((BASE_SIZE[1], BASE_SIZE[0], 3), dtype=np.uint8))
        except Exception as exc:  # noqa: BLE001
            logger.warning("bgi_world warmup 失败: %s", exc)

    @property
    def is_loaded(self) -> bool:
        return self._det is not None

    @property
    def last_ms(self) -> float:
        """上一次推理耗时（毫秒）。"""
        return self._last_ms

    @property
    def last_count(self) -> int:
        """上一次检出的实体数。"""
        return self._last_count

    # -- 场景门控 --------------------------------------------------------
    @staticmethod
    def should_run_for_scene(scene_id: str | None) -> bool:
        """该场景值不值得跑。背包/地图/菜单里跑纯属浪费 180 ms。"""
        return scene_id in WORLD_SCENES

    # -- 主入口 ----------------------------------------------------------
    def detect(self, frame, base_size: tuple[int, int] = BASE_SIZE):
        """检测一帧的世界实体。

        :param frame: BGR ndarray，任意分辨率
        :param base_size: ``screen_pos`` 折算到的基准尺寸，默认 720
        :returns: :class:`WorldEntity` 列表；按置信度降序
        :raises YoloUnavailableError: 模型或类名表不存在

        ⚠️ **宽于 1280 的输入会先降采样**。模型输入固定 1280×1280，
           letterbox 之前那一次大图 resize 是纯浪费。2026-10-02 实测
           （batch_a 20 帧，2304×1440 原图）：

           ==================  =====  =====  =====
           喂法                p50    p90    max
           ==================  =====  =====  =====
           原始 2304×1440      242    311    369
           先降到 1280 宽       170    249    255
           ==================  =====  =====  =====

           **省 71 ms 中位耗时**。但即便如此 p50 仍是 200ms tick 预算的 85%。
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            self._last_count = 0
            return []

        h0, w0 = frame.shape[:2]
        src = frame
        if w0 > self._infer_width:
            # 等比缩到 _infer_width 宽（模型输入就是 1280，更宽没有意义）
            import cv2  # noqa: PLC0415
            nh = max(1, int(round(h0 * self._infer_width / w0)))
            src = cv2.resize(frame, (self._infer_width, nh),
                             interpolation=cv2.INTER_LINEAR)

        det = self.load()
        t0 = time.perf_counter()
        hits = det.detect(src, conf=self._conf, timeout_ms=self._timeout_ms)
        self._last_ms = (time.perf_counter() - t0) * 1000

        # 坐标还原到**原始帧**尺度，再折算到 base_size
        h, w = h0, w0
        sx_src = w / max(1, src.shape[1])
        sy_src = h / max(1, src.shape[0])
        bw, bh = base_size
        fx, fy = bw / max(1, w), bh / max(1, h)

        out: list[WorldEntity] = []
        for d in hits:
            x, y, bw_px, bh_px = d.box
            x0, y0 = x * sx_src, y * sy_src
            w0_, h0_ = bw_px * sx_src, bh_px * sy_src
            out.append(WorldEntity(
                kind=d.label,
                bbox=(x0 / w, y0 / h, w0_ / w, h0_ / h),
                confidence=d.conf,
                screen_pos=(int(round((x0 + w0_ / 2) * fx)),
                            int(round((y0 + h0_ / 2) * fy))),
            ))
        out.sort(key=lambda e: -e.confidence)
        self._last_count = len(out)
        return out


# --------------------------------------------------------------------------
# 模块级便捷入口
# --------------------------------------------------------------------------

_DEFAULT: Optional[WorldEntityDetector] = None


def default_detector() -> WorldEntityDetector:
    """进程级默认检测器（懒建）。"""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = WorldEntityDetector()
    return _DEFAULT


def detect_world_entities(frame, base_size: tuple[int, int] = BASE_SIZE):
    """用 bgi_world 检测世界实体。便捷入口，内部用进程级默认检测器。

    适合脚本/测试。**主循环请显式注入 :class:`WorldEntityDetector`**，
    这样才能读 ``last_ms`` 做耗时监控。
    """
    return default_detector().detect(frame, base_size)


# --------------------------------------------------------------------------
# provider 自适应（§23）
# --------------------------------------------------------------------------


def current_provider() -> str:
    """当前**真正**在用的 provider（建个空 session 探一下）。

    ⚠️ 不能用 ``ort.get_available_providers()`` —— 那是编译期清单，
    装了 ``onnxruntime-gpu`` 但缺 CUDA 运行时 DLL 时它照样列 CUDA，
    而实际会静默回退 CPU。本函数读的 ``session.get_providers()`` 才是真值。
    """
    try:
        import onnxruntime as ort  # noqa: PLC0415
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(
            str(BGI_WORLD_ONNX), sess_options=so,
            providers=list(available_providers()))
        ps = list(sess.get_providers())
        return ps[0] if ps else "unknown"
    except Exception as exc:  # noqa: BLE001
        logger.warning("探测 provider 失败: %s", exc)
        return "unknown"


def probe_speed(det: "WorldEntityDetector", frames: int = 5,
                warmup: int = 3) -> float:
    """在**合成帧**上量一次 p50 耗时（毫秒）。用于启动期决定要不要启用。

    ⚠️ 合成帧（纯噪声）不代表真实场景内容，但**计算量只取决于张量形状**，
       所以耗时本身是可比的。

    ★ **预热必须给够**（:func:`warmup` 默认 3 次）。
       2026-10-02 实测：只预热 1 次时探测值在 **63~107 ms** 之间乱跳，
       正好跨在 100 ms 阈值上 —— 直接后果是
       ``entity_detector="auto"`` 这个**默认值会随机开/关**
       （同一台机器、不同负载，给出相反结论）。
       原因是 DirectML/GPU 还在升频，前几次推理系统性偏慢。
       真实帧实测 p50 是 66 ms，探测偏高就是这个预热不足造成的。
    """
    import numpy as np  # noqa: PLC0415
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (BASE_SIZE[1], BASE_SIZE[0], 3), dtype=np.uint8)
    for _ in range(max(1, warmup)):
        det.detect(img)                            # 预热：不计入
    ts: list[float] = []
    for _ in range(max(1, frames)):
        # ⚠️ 不能写 ``det.detect(img) and det.last_ms`` —— detect() 返回
        #    **列表**（空列表假、非空列表真），``and`` 会返回那个列表。
        det.detect(img)
        ts.append(det.last_ms)
    ts.sort()
    return float(ts[len(ts) // 2])                 # 中位数，抗单帧抖动


def recommended_interval_ticks(measured_ms: float) -> int:
    """给定实测 p50，返回**建议每几个 tick 跑一次**。

    ============  ==================  ==========================
    实测 p50      建议间隔             理由
    ============  ==================  ==========================
    < 30 ms       1（每 tick）          快到可以实时跟
    30 ~ 阈值      **5**                中等，5 Hz 下 = 每秒 1 次
    阈值 ~ 1.5×    **10**               慢，但**开着**（见下）
    > 1.5 × 阈值   **0 = 别开**         确实慢到不能用
    ============  ==================  ==========================

    ★★ 第三档（降频但继续跑）是 2026-10-14 加的
      --------------------------------------------------
      原来只有三档，「超阈值」直接等于「关」。而关掉的代价是：
      WORLD 场景下**唯三可触发的规则全都依赖** ``has_nearby_enemy()``，
      于是整条 WORLD 决策链哑掉 —— 而现场现象只是「卡片没动作」，
      极难联想到是耗时阈值。

      **宁可慢，也不要没有。** 本机实测 128ms / 200ms = 0.64，
      属于「只是慢一点」，应当降频继续跑；只有明显不对劲（>1.5 倍）
      才彻底放弃。

    ⚠️ 中间档直接取 **5** 而不是按耗时反推 —— 实体位置以「秒」计，
       40 ms 和 70 ms 不需要区别对待。分档是**工程取舍**，不是算出来的。
    """
    if measured_ms < ENTITY_EVERY_TICK_MAX_MS:
        return 1
    if measured_ms < ENTITY_ENABLE_THRESHOLD_MS:
        return DEFAULT_ENTITY_INTERVAL_TICKS      # 5
    # ★ 超过阈值**不等于关闭**：先看是不是只是慢一点
    if measured_ms < ENTITY_ENABLE_THRESHOLD_MS * ENTITY_SLOWDOWN_FACTOR:
        return ENTITY_SLOW_INTERVAL_TICKS       # 10
    return 0            # 0 = 别开
