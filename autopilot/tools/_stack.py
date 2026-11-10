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
from collections.abc import Mapping
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
    #: ★ 在场槽位记账器（``actuator.active_slot.ActiveSlotTracker``）。
    #:   同一个对象喂给 executor（切完人记一笔）、loop（用户按数字键记一笔）
    #:   和 builder（据此把某个 ``PartyMember.active`` 置 True）。
    #:   ``None`` = 不记 -> ``active_member`` 恒 None -> ``onfield()`` 恒 UNKNOWN。
    active_slots: Any = None
    #: ★ 编队名册索引（``actuator.active_slot.CharSlotIndex``）。
    #:   感知层写（队伍栏 OCR）、执行层读（``char_id`` -> 数字键）。
    #:   这是**免配置切人**的关键：不用手填 ``keymap.toml`` 的 ``[char_slots]``。
    char_slot_index: Any = None
    #: 队伍栏头像分类器（``perception.detectors.avatar_classifier.AvatarClassifier``）。
    #: ``None`` = 不走这条路（默认，且实测准确率不过线时**应当如此**）。
    avatar_classifier: Any = None
    #: ★ 连招状态机（``policy.combo_manager.ComboRunner``）。
    #: ``None`` = 没找到连招配置 = **纯反射模式**，决策层只剩规则。
    #: 和 :attr:`policy` / :attr:`active_slots` 共用同一套对象 ——
    #: runner 要靠它们确认「切完人没有」，各建各的等于永远切不动人。
    combo_runner: Any = None
    #: 实际读到的连招配置文件。``None`` = 没读（纯反射模式）。
    combos_path: Path | None = None
    cfg: Any = None
    rules_dir: Path | None = None
    rule_count: int = 0
    warnings: list[str] = field(default_factory=list)


def build_policy(rules_dir: str | Path | None = None,
                 *, combo_runner: Any = None) -> tuple[Any, Path, int]:
    """决策层。走工厂，不直接 import LocalPolicyTree。"""
    from autopilot.policy.factory import PolicyConfig, create_policy_source

    cfg = PolicyConfig(kind="LOCAL",
                       rules_dir=Path(rules_dir) if rules_dir else None,
                       combo_runner=combo_runner)
    src = create_policy_source(cfg)
    rd = Path(getattr(src, "rules_dir", Path("?")))
    n = len(getattr(src, "rules", ()))
    return src, rd, n


def load_static_char_slots(keymap_path: str | Path | None = None) -> dict[str, int]:
    """读 ``keymap.toml`` 的 ``[char_slots]``。**读不到就是空 dict**，不报错。

    🔴 为什么单独读一遍而不复用 :func:`build_actuator` 里那一份
      -------------------------------------------------------------------
      连招执行器要的是「这个角色几号位」，执行器要的是同一件事 ——
      两边**必须是同一个答案**（静态优先，再退 OCR 名册）。
      而连招要造的时机比 ``build_actuator`` 早（决策层在建执行层之前），
      拿不到已经解析好的执行器。

      重复读一次 keymap 是几百字节的纯解析，代价可以忽略；反过来让两边
      各算各的，会出现「规则切对了人、连招切错了人」—— 那才是要命的。

    ``[char_slots]`` 缺失 / 写坏时返回空 dict 并记一条 warning：
    免配置切人的默认路径就是空的，不该因此报错。
    """
    from autopilot.actuator.keymap import KeymapError, load_char_slots

    if not keymap_path:
        return {}
    try:
        return dict(load_char_slots(keymap_path))
    except (KeymapError, OSError, ValueError) as exc:
        logger.warning("读 keymap 的 [char_slots] 失败（按没配处理）：%s", exc)
        return {}


def build_combo_runner(combos_path: str | Path | None = None,
                       *, active_slots: Any = None,
                       slot_index: Any = None,
                       static_slots: Mapping[str, int] | None = None) -> tuple[Any, Path | None]:
    """造连招状态机。**没配置文件就返回 ``(None, None)``**。

    这是「没配 ``combos.toml`` 就退回纯反射模式」的那条通道：
    文件不存在是**正常配置**，返回 None 让上层照常跑规则；
    文件存在但写坏了则抛 :class:`ComboLoadError`（响亮失败）——
    那两种情况必须分开，否则玩家会看到「引擎不认我的连招」
    而屏幕和日志上什么都没有。

    ⚠️ ``active_slots`` / ``slot_index`` **必须**是 ``build_stack`` 里
      那两个对象（同 :attr:`Stack.active_slots` 的要求）。晚建一步，
      连招就会一直「不知道谁在场」而逐条跳过，且只在事件里留一句。

    ⚠️ ``static_slots`` 是 ``keymap.toml`` 的 ``[char_slots]``。它
      **优先于** OCR 名册 —— 和 ``executor._slot_of`` 同口径（见
      :meth:`ComboRunner._slot_of`）。漏传的后果是玩家手填的
      ``[char_slots]`` 对连招无效，而文档正让他们填它来绕开 20 秒空窗。
    """
    from autopilot.policy.combo_manager import ComboRunner, load_optional_combos

    book = load_optional_combos(combos_path)
    if book is None:
        return None, None
    runner = ComboRunner(book, active_slots=active_slots, slot_index=slot_index,
                         static_slots=static_slots)
    return runner, (Path(combos_path) if combos_path
                   else runner.book.source_file or None)


def build_actuator(*, dry_run: bool = True,
                   keymap_path: str | Path | None = None,
                   active_slots: Any = None,
                   slot_index: Any = None) -> tuple[Any, Any]:
    """执行层（+ 安全闸）。"""
    from autopilot.actuator import DryRunBackend, InputSimulator, Win32Backend
    from autopilot.actuator.executor import IntentExecutor
    from autopilot.actuator.keymap import KeymapError, load_char_slots, load_keymap
    from autopilot.actuator.safety import SafetyConfig, SafetyGuard

    backend = DryRunBackend() if dry_run else Win32Backend()
    sim = InputSimulator(backend=backend)

    def _warn(msg: str) -> None:
        """追加一条启动警告。**累加而不是覆盖** ——
        键位表坏了**和**编队槽位空了是两件独立的事，
        后写的那条不该把前一条顶掉。"""
        prev = getattr(build_actuator, "last_warning", "")
        build_actuator.last_warning = f"{prev}；{msg}" if prev else msg

    keymap = None
    if keymap_path:
        try:
            keymap = load_keymap(keymap_path)
        except KeymapError as exc:
            logger.warning("键位配置非法，改用内置默认：%s",
                           str(exc).splitlines()[0])
            _warn(f"键位配置非法（已退回内置默认）：{str(exc).splitlines()[0]}")

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
            _warn(f"编队槽位配置非法（已忽略，SWITCH_CHARACTER 会跳过）："
                  f"{str(exc).splitlines()[0]}")

    # ★ 空表也要说一声（2026-10-18）。
    #   `[char_slots]` 是空的**不会报错**：load_char_slots 返回 {}，
    #   _slot_of 返回 None，SWITCH_CHARACTER 在 steps 里留一句就跳过。
    #   后果是「连招规则全开着，但一个都切不动人」—— 而屏幕上什么异常都没有。
    #   这正是 warnings 存在的意义：让「没生效」在启动时就看得见。
    if not char_slots:
        _warn("[char_slots] 是空的：切人将**完全依赖大世界 OCR 自动识别队伍**"
              "（免配置，autopilot.character_id 默认开着）。"
              "代价是启动后约 ~20 秒内名册还没填满（单次 OCR 实测 1.0~1.5 秒，"
              "每 25 tick 轮转 1 格，4 格一圈），这段时间 SWITCH_CHARACTER 会"
              "跳过并在 steps 里记账 —— 不是静默失败，但确实切不动人。"
              "想立刻可用就自己填，例如：[char_slots]  \"夜兰\" = 2")

    guard = SafetyGuard(SafetyConfig())
    ex = IntentExecutor(simulator=sim, guard=guard, keymap=keymap,
                        char_slots=char_slots,
                        active_slot_tracker=active_slots,
                        slot_index=slot_index)
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


def build_perception(cfg: Any, *, open_capturer: bool = True,
                     active_slots: Any = None,
                     char_slot_index: Any = None) -> Stack:
    """感知层（截图器 + 状态构建器）。

    :param active_slots: 在场槽位记账器（``actuator.active_slot.ActiveSlotTracker``）。
        ``None`` = 不记 -> ``PlayerInfo.active_member`` 恒 None ->
        ``onfield()`` 恒 UNKNOWN。**由 :func:`build_stack` 传入**，
        不要在这里自己造 —— 必须和执行层/监听层共用**同一个对象**。
    :param char_slot_index: 编队名册索引（``actuator.active_slot.CharSlotIndex``）。
        builder 是**唯一写入方**（队伍栏 OCR）。``None`` = 不发布。
    """
    from autopilot.perception.builder import StateBuilder
    from autopilot.perception.capturer import CaptureChannel, Capturer
    from autopilot.perception.template import TemplateMatcher

    st = Stack(cfg=cfg)
    st.active_slots = active_slots
    st.char_slot_index = char_slot_index

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

    # ★ E 技能就绪感知接线（2026-10-18）。默认 ``enabled = false`` —— 传 None，
    #   ``PlayerInfo.party`` 恒为 ``()``，``e_ready()`` 恒 UNKNOWN，
    #   与接入之前完全一致（引擎只放普攻）。
    sc_cfg = getattr(cfg, "skill_cd", None)
    st.skill_cd = sc_cfg if (sc_cfg is not None and
                             getattr(sc_cfg, "enabled", False)) else None
    if st.skill_cd is None:
        st.warnings.append(
            "E 技能就绪感知已关闭（autopilot.skill_cd.enabled = false）—— "
            "e_ready() 信号恒为 UNKNOWN，USE_SKILL 类规则不会触发。")
    else:
        st.warnings.append(
            "E 技能就绪感知已启用，但它的**特征被背景主导**："
            "E 徽章是半透明的，中心方块饱和度与同帧背景的相关性 r≈0.88"
            "（实测 296 帧，见 detectors/key_e.py 的模块文档）。"
            "所以当前判出来的是「场景亮不亮」，不是「E 好不好」—— "
            "沙漠等亮场景会倾向判就绪，夜里等暗场景会倾向判冷却。"
            "调 SAT_READY 修不好这个（只会换一个场景出错）。"
            "要修得换特征：对比量，或 ONNX 分类器。")

    # ★ Q 大招就绪感知接线（2026-10-18）。默认 ``enabled = true``。
    #   开关与 E **分开**，因为两个信号可信度不同（见下）。
    bc_cfg = getattr(cfg, "burst_cd", None)
    st.burst_cd = bc_cfg if (bc_cfg is not None and
                             getattr(bc_cfg, "enabled", False)) else None
    if st.burst_cd is None:
        st.warnings.append(
            "Q 大招就绪感知已关闭（autopilot.burst_cd.enabled = false）—— "
            "cd_ready 里没有 burst 这一把，q_ready() 恒 UNKNOWN，"
            "USE_BURST 类规则不会触发。")
    else:
        st.warnings.append(
            "Q 大招就绪感知已启用：阈值**在 296 帧上量出来了**"
            "（就绪 cyan_ratio 0.822~0.824 / 不就绪 <=0.091，间隔 0.731，"
            "15 命中 0 误报 0 漏报）。但它有两条已知缺口："
            "① 判不出「在冷却」——296 帧里零个冷却样本，不就绪一律报 "
            "UNKNOWN（对规则等价，但 unknowns 里会出现 cd）；"
            "② 只在 1 个角色的爆发按钮上标定过，换角色若不准，"
            "先查 detectors/key_q.py 的模块文档，别调 CYAN_RATIO_READY。")

    # ★ 队伍栏头像分类器接线（2026-10-18）。**默认关**，而且是有依据的关：
    #   实测真机 697 张头像上 top1 ≈ 0.4%（124 类任务，随机 0.81%），
    #   比一个「什么都不学的最近邻」（6.74%）还差。
    #   开着只会制造**错误的角色名**，比读不到更糟 —— onfield() 会拿
    #   错名字去判连招。详见 config.AvatarClassifierConfig 的类文档。
    ac_cfg = getattr(cfg, "avatar_classifier", None)
    if ac_cfg is not None and getattr(ac_cfg, "enabled", False):
        from autopilot.perception.detectors.avatar_classifier import (  # noqa: PLC0415
            AvatarClassifier, resolve_model_path,
        )

        mpath = resolve_model_path(getattr(ac_cfg, "path", ""))
        st.avatar_classifier = AvatarClassifier.load(mpath)
        if st.avatar_classifier is None:
            st.warnings.append(
                f"队伍栏头像分类器已开启但**加载失败**（{mpath}）—— "
                f"该通道保持关闭，角色名仍走 OCR。")
        else:
            st.warnings.append(
                "队伍栏头像分类器已开启，作为角色名 OCR 的**降级**路径。"
                "⚠️ 它和 OCR **不是互补**：OCR 读不到名字时多半是切屏/开菜单/"
                "队伍栏虚化，那���头像也跟着糊。")

    # ★ 队伍栏角色名识别接线（2026-10-18）。关着时每个 PartyMember 的
    #   name 恒为 ""，onfield('钟离') 与任何按名字判的规则恒 UNKNOWN。
    ch_cfg = getattr(cfg, "character_id", None)
    st.character_id = ch_cfg if (ch_cfg is not None and
                                 getattr(ch_cfg, "enabled", False)) else None
    if st.character_id is None:
        st.warnings.append(
            "队伍栏角色名识别已关闭（autopilot.character_id.enabled = false）"
            "—— PartyMember.name 恒为空，于是 onfield('钟离') 之类**按名字判**的"
            "规则恒为 False（注意是 False 不是 UNKNOWN：active 由切人记账给出，"
            "只是名字对不上）。")
    else:
        st.warnings.append(
            "队伍栏角色名识别已启用。角色名靠 OCR 轮转（每 interval_ticks 读一格，"
            "单次实测 ~1 秒，是 200ms tick 预算的 5 倍），所以**刚切过去的那一帧**"
            "名字可能还是空的 —— 那时 onfield('X') 暂时为 False，一两轮之后才准。"
            "「谁在场」本身不靠看画面，由切人记账给出（见下面那条警告）。")

    st.builder = StateBuilder(
        capturer=st.capturer,
        template=TemplateMatcher(),
        locator=st.locator,
        hp_estimation=st.hp_estimation,
        skill_cd=st.skill_cd,
        burst_cd=st.burst_cd,
        active_slots=st.active_slots,
        char_slot_index=st.char_slot_index,
        avatar_classifier=st.avatar_classifier,
        avatar_classifier_cfg=ac_cfg,
        character_id=st.character_id,
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
    combos_path: str | Path | None = None,
) -> Stack:
    """组装完整三层。任一层失败直接抛。"""
    from autopilot.config.loader import load_config

    cfg = cfg or load_config()

    # ★ 在场槽位记账器（2026-10-18）。
    #   🔴 顺序是硬要求：**必须在 build_perception 之前**建并传进去。
    #     StateBuilder 拿的是**引用**，晚一步就传进去一个 None ——
    #     而症状是「切人切了但 onfield() 还是 UNKNOWN」，界面上完全看不出
    #     是这里顺序错了。（这个坑真的踩过一次。）
    #   默认永不过期（stale_after_ms=0）：过期会让连招规则每过一会儿
    #   集体熄火一次，那比偶尔说错话难查得多。失效场景写在
    #   actuator/active_slot.py 的模块文档里。
    from autopilot.actuator.active_slot import ActiveSlotTracker, CharSlotIndex

    tracker = ActiveSlotTracker(stale_after_ms=0)
    #: ★ 编队名册索引（免配置切人）。感知层写、执行层读。
    #:   和 tracker 一样**必须共用同一个对象** —— 晚建一步就静默失效。
    slot_index = CharSlotIndex()

    st = build_perception(cfg, open_capturer=open_capturer,
                          active_slots=tracker,
                          char_slot_index=slot_index)
    st.warnings.append(
        "在场角色靠**切人记账**得出（引擎按了哪个数字键 / 用户按了哪个数字键），"
        "不是看画面认出来的。已知失效场景：**角色阵亡自动切人**、中途换队伍 —— "
        "引擎无从发现，症状是「反复按同一个数字键」，无破坏性。")
    st.warnings.append(
        "编队名册（1/2/3/4 号位分别是谁）由**大世界 OCR 自动建立**，"
        "不需要手填 keymap.toml 的 [char_slots]。名册填满约需 ~20 秒"
        "（单次 OCR 1.0~1.5 秒，每 25 tick 转 1 格，4 格一圈）。"
        "手填了 [char_slots] 的话以你为准 —— 用户配置优先于 OCR。")

    # ★ 连招状态机（2026-10-18）。
    #   🔴 顺序同样是硬要求：**必须在 build_perception 之后、build_policy
    #     之前**建，且要传入**同一对** tracker / slot_index ——
    #     连招执行器靠它们回答两个问题：「切完人没有」（tracker）
    #     和「这个人在几号位」（slot_index）。给错了对象的后果是
    #     「连招每一步都被跳过」，只有事件里留一句 COMBO_STEP_UNKNOWN。
    #
    #   没找到 combos.toml -> 返回 None -> 决策层退回**纯反射模式**
    #   （只有规则），与接入连招之前完全一致。
    st.combo_runner, st.combos_path = build_combo_runner(
        combos_path, active_slots=tracker, slot_index=slot_index,
        static_slots=load_static_char_slots(keymap_path))
    if st.combo_runner is None:
        logger.info("未找到连招配置，决策层按纯反射模式运行")
    else:
        _book = st.combo_runner.book
        st.warnings.append(
            f"已加载连招配置 {st.combos_path}：{len(_book)} 条，"
            f"其中**开着 {len(_book.enabled)} 条**。")
        if not _book.enabled:
            st.warnings.append(
                "连招一条都没开（全是 enabled = false）—— 这是默认值："
                "队伍顺序和 wait 时间是每个玩家自己的事，引擎不替你决定。")
        else:
            st.warnings.append(
                "连招依赖编队名册：名册填满约需 ~20 秒，在那之前连招的每一步"
                "都会因为「不知道这个人在几号位」被跳过（会发 COMBO_STEP_UNKNOWN"
                "事件，不是静默失败）。想立刻可用就手填 keymap.toml 的 "
                "[char_slots]。连招进行中 priority < 20 的规则"
                "（恢复类 / 闪避）仍会打断它。")

    st.policy, st.rules_dir, st.rule_count = build_policy(
        rules_dir, combo_runner=st.combo_runner)
    st.executor, st.guard = build_actuator(dry_run=dry_run,
                                          keymap_path=keymap_path,
                                          active_slots=st.active_slots,
                                          slot_index=st.char_slot_index)
    warn = getattr(build_actuator, "last_warning", "")
    if warn:
        st.warnings.append(warn)
    return st


__all__ = ["Stack", "build_actuator", "build_combo_runner", "build_locator",
           "build_perception", "build_policy", "build_stack"]