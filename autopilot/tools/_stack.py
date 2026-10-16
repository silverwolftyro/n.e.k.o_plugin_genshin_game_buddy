"""三层组装：把配置变成可运行的 capturer / builder / policy / executor。

run_loop 与 loop_diagnose 都用它。**单独成文件而不是复制两份** ——
两个工具对「什么算组装成功」的定义一旦漂移，diagnose 说绿灯、
run 却跑不起来，那是最难查的一类不一致。

组装原则：**任一层组装失败就抛**，绝不返回半成品。
「能装就装，缺了就留 None」的做法会让主循环跑到一半才发现 policy 是 None。

用法范本见 tools/autopilot_demo.py（TemplateMatcher 无构造参数，
模板定义与 ROI 靠 StateBuilder.set_templates / set_roi 注入）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Stack:
    """组装好的三层 + 配置摘要。"""

    capturer: Any = None
    builder: Any = None
    policy: Any = None
    executor: Any = None
    guard: Any = None
    #: 定位服务。``None`` = 未启用（默认），此时 ``StatePacket.position`` 恒为 None
    locator: Any = None
    cfg: Any = None
    rules_dir: Path | None = None
    rule_count: int = 0
    warnings: list[str] = field(default_factory=list)


def build_policy(rules_dir: str | Path | None = None) -> tuple[Any, Path, int]:
    """决策层。走工厂，不直接 import LocalPolicyTree。"""
    from autopilot.policy.factory import PolicyConfig, create_policy_source

    cfg = PolicyConfig(kind="LOCAL",
                       rules_dir=Path(rules_dir) if rules_dir else None)
    src = create_policy_source(cfg)
    rd = Path(getattr(src, "rules_dir", Path("?")))
    n = len(getattr(src, "rules", ()))
    return src, rd, n


def build_actuator(*, dry_run: bool = True,
                   keymap_path: str | Path | None = None) -> tuple[Any, Any]:
    """执行层（+ 安全闸）。"""
    from autopilot.actuator import DryRunBackend, InputSimulator, Win32Backend
    from autopilot.actuator.executor import IntentExecutor
    from autopilot.actuator.keymap import KeymapError, load_char_slots, load_keymap
    from autopilot.actuator.safety import SafetyConfig, SafetyGuard

    backend = DryRunBackend() if dry_run else Win32Backend()
    sim = InputSimulator(backend=backend)

    keymap = None
    if keymap_path:
        try:
            keymap = load_keymap(keymap_path)
        except KeymapError as exc:
            logger.warning("键位配置非法，改用内置默认：%s",
                           str(exc).splitlines()[0])
            build_actuator.last_warning = f"键位配置非法（已退回内置默认）：{str(exc).splitlines()[0]}"

    # 编队槽位（char_id -> 1..4）。**读不到就是空表**而不是报错 ——
    #   编队是用户才有的信息；缺了就等于「不知道切谁」，
    #   SWITCH_CHARACTER 会跳过并在 steps 里记账（2026-10-13 修：
    #   此前 _slot_of 读一个从未被赋值的属性，等于永远跳过且不解释）。
    char_slots = {}
    if keymap_path:
        try:
            char_slots = load_char_slots(keymap_path)
        except KeymapError as exc:
            logger.warning("编队槽位配置非法，已忽略：%s",
                           str(exc).splitlines()[0])
            build_actuator.last_warning = (
                f"编队槽位配置非法（已忽略，SWITCH_CHARACTER 会跳过）："
                f"{str(exc).splitlines()[0]}")

    guard = SafetyGuard(SafetyConfig())
    ex = IntentExecutor(simulator=sim, guard=guard, keymap=keymap,
                        char_slots=char_slots)
    return ex, guard


build_actuator.last_warning = ""


def build_locator(cfg: Any) -> Any:
    """按配置造一个定位服务。**关着就返回 None**，不造半成品。

    ★ 为什么默认关
    --------------
    ``LocalizationService`` 初始是 ``UNINITIALIZED`` —— **它没有起点**。
    必须先在游戏里传送一次建立原点（见 ``teleport_contract.md``），
    在那之前 ``position`` 恒为 ``quality="LOST"``。所以这个开关的意义是
    **让接线这件事可被测试**，不是「一开就能定位」。

    :returns: ``LocalizationService`` 实例，或 ``None``（未启用）
    :raises ImportError: 启用但模块缺失 —— **响亮失败**，不静默降级成 None
    """
    loc_cfg = getattr(cfg, "locator", None)
    if loc_cfg is None or not getattr(loc_cfg, "enabled", False):
        return None

    # 故意不在 try 里吞：配置说要用定位，模块却不在，那是环境坏了，
    # 静默返回 None 会让 position 恒 None，用户以为「配置没生效」。
    from autopilot.perception.localization_service import LocalizationService

    return LocalizationService(
        min_peak_response=loc_cfg.min_peak_response,
        min_circle_std=loc_cfg.min_circle_std,
    )


def build_perception(cfg: Any, *, open_capturer: bool = True) -> Stack:
    """感知层（截图器 + 状态构建器）。"""
    from autopilot.perception.builder import StateBuilder
    from autopilot.perception.capturer import CaptureChannel, Capturer
    from autopilot.perception.template import TemplateMatcher

    st = Stack(cfg=cfg)

    # ★ 2026-10-15 修一个只在**运行时**才炸的 NameError。
    #   下面构造 StateBuilder 时要读传送 OCR 的两个配置项，之前直接
    #   写了 ``loc_cfg`` —— 那是 :func:`build_locator` 里的**局部变量**，
    #   在本函数里根本没定义。症状是工具一启动就
    #   ``NameError: name 'loc_cfg' is not defined``，而全量单测**照样
    #   全绿**（没有任何测试真的调 build_perception）。所以这里取本函数
    #   自己的一份。
    loc_cfg = getattr(cfg, "locator", None)

    # ★ channel_order **必须转成 CaptureChannel 枚举**。
    #   配置里存的是字符串（"dxcam"），而 Capturer._init_channel 用的是
    #   `if ch is CaptureChannel.DXCAM` —— 字符串永远不 is 枚举成员，
    #   三个通道会全部静默返回 False 且不抛异常，最终报
    #   「没有可用的截图通道 / 各通道错误：（无）」。
    #   capture_demo 一直正常，是因为它不传这个参数（用默认枚举列表）。
    #   Capturer 侧现在也加了兜底校验，但这里显式转更清楚。
    order = [CaptureChannel(c) for c in cfg.capture.channel_order]
    st.capturer = Capturer(cfg.capture.window_title_substr,
                           channel_order=order)
    if open_capturer:
        st.capturer.open()

    # ★ 定位服务接线（2026-10-12）。默认 ``None`` —— 与接入之前行为完全一致，
    #   ``StatePacket.position`` 恒为 None。这是向后兼容的唯一保证。
    st.locator = build_locator(cfg)
    if st.locator is not None:
        st.warnings.append(
            "定位服务已启用，但它初始是 UNINITIALIZED（没有起点）—— "
            "必须先在游戏里**传送一次**建立原点，否则 position 恒为 LOST。"
            "契约见 recognition/notes/teleport_contract.md")

    # ★ 血量几何投影接线（2026-10-12）。默认 ``enabled = false`` —— 传 None，
    #   ``PlayerInfo.hp_ratio`` 恒为 None = UNKNOWN，与接入之前完全一致。
    #
    #   ⚠️ 开着也不等于「满血能读出值」：满血/空血两档内部是纯色，
    #      没有 background_bgr 时会返回 UNKNOWN（不猜）。详见 HpEstConfig。
    hp_cfg = getattr(cfg, "hp_estimation", None)
    st.hp_estimation = hp_cfg if (hp_cfg is not None and
                                  getattr(hp_cfg, "enabled", False)) else None
    if st.hp_estimation is None:
        st.warnings.append(
            "血量几何投影已关闭（autopilot.hp_estimation.enabled = false）—— "
            "hp() 信号恒为 UNKNOWN，依赖它的规则不会触发。")
    elif st.hp_estimation.background_bgr is None:
        st.warnings.append(
            "血量几何投影已启用，但没填 background_bgr —— "
            "恰好满血/空血时读不出值（内部纯色，与纯色补丁无法区分），"
            "会返回 UNKNOWN 而不是猜。规则「hp() < 0.35」在濒死时仍会命中。")

    st.builder = StateBuilder(
        capturer=st.capturer,
        template=TemplateMatcher(),
        locator=st.locator,
        hp_estimation=st.hp_estimation,
        game_id=cfg.game_id,
        scene_map=dict(cfg.scene_map) or None,
        scene_routes=dict(cfg.scene_routes) or None,
        # ★ 传送契约 ① SELECT（2026-10-15 接上）。没有它，
        #   ``on_teleport_ocr`` 的调用方永远是 0 —— 状态机卡在
        #   UNINITIALIZED，传送多少次 position 都是 LOST。
        teleport_ocr=(st.locator is not None
                      and bool(getattr(loc_cfg, "teleport_ocr_enabled", True))),
        teleport_ocr_interval_s=float(
            getattr(loc_cfg, "teleport_ocr_interval_s", 0.5)),
    )

    try:
        specs = cfg.template_specs()
        st.builder.set_templates(specs)
        st.builder.set_roi(cfg.roi_expressions(), cfg.roi_var_values())
        if not specs:
            st.warnings.append(
                "配置里没有启用任何模板 —— 场景识别会退化成只用 YOLO。"
                "模板图需要按你自己的分辨率裁（capture_demo --save → 裁剪 "
                "→ 存到 perception/templates/），仓库里不能预置。"
                "在裁好之前，主循环会大量空转 —— 这是预期行为。")
    except Exception as exc:
        logger.warning("模板/ROI 配置失败：%s", exc)
        st.warnings.append(f"模板/ROI 配置失败（模板匹配将被跳过）：{exc}")

    return st


def build_stack(
    *,
    cfg: Any = None,
    rules_dir: str | Path | None = None,
    dry_run: bool = True,
    keymap_path: str | Path | None = None,
    open_capturer: bool = True,
) -> Stack:
    """组装完整三层。任一层失败直接抛。"""
    from autopilot.config.loader import load_config

    cfg = cfg or load_config()
    st = build_perception(cfg, open_capturer=open_capturer)
    st.policy, st.rules_dir, st.rule_count = build_policy(rules_dir)
    st.executor, st.guard = build_actuator(dry_run=dry_run,
                                          keymap_path=keymap_path)
    warn = getattr(build_actuator, "last_warning", "")
    if warn:
        st.warnings.append(warn)
    return st


__all__ = ["Stack", "build_actuator", "build_locator", "build_perception",
           "build_policy", "build_stack"]