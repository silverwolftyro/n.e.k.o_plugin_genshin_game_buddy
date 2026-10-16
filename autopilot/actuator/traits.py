"""ActionTraits：每个动作的差异化行为表。

对应架构文档**原则 8**（抄自 MaaEnd 的 `ActionTraits` 静态策略表思路 `[实测]`）。

**为什么要这张表而不是 if-else**：
    BetterGI 的 `move_mode` 是硬编码 if-else 分支处理 fly/jump/run/dash/else，
    加新模式要改多处；MaaEnd 把「到点要不要精确停 / 要不要先走路 / 是不是
    路线边界」抽成 7 个 traits 字段的静态表 `[实测]`。加新动作 = 加一行，
    改任何分支逻辑都不需要动。

**收益**：
    · 安全策略集中化：`allowed_scenes` 统一管「什么场景能做什么」
    · 可测试：traits 是纯数据，可以直接单测
    · 隐式互斥变显式：BetterGI 的 `SkillCdTrigger` 用静态 Suspend/Resume 与
      AutoCombo 互斥，注释警告「调用方保证成对」；用本表 + 仲裁可以消掉这类隐患

**一个 traits 都没有的默认行为是保守的** —— 新增动作忘了配 traits 时会走
`ActionTraits()` 默认值，**默认拒绝高风险行为**（严格到点、场景受限）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..contracts.state_packet import SCENE_IDS as _SCENE

if TYPE_CHECKING:  # 避免运行期循环导入
    pass

#: 常用场景 ID。**从 contracts.state_packet.SCENE_IDS 派生**，不另写一份 ——
#: 权威源在 contracts，policy 校验规则时读那份，两边不会漂移。
#: （早先这里自带一份，policy 为了校验 scene 不得不反向依赖 actuator。）
SCENE_ANY = _SCENE["SCENE_ANY"]
SCENE_UNKNOWN = _SCENE["SCENE_UNKNOWN"]
SCENE_WORLD = _SCENE["SCENE_WORLD"]
SCENE_MAIN_MENU = _SCENE["SCENE_MAIN_MENU"]
SCENE_COMBAT = _SCENE["SCENE_COMBAT"]
SCENE_LOADING = _SCENE["SCENE_LOADING"]
SCENE_CUTSCENE = _SCENE["SCENE_CUTSCENE"]
SCENE_DIALOG = _SCENE["SCENE_DIALOG"]
SCENE_BAG = _SCENE["SCENE_BAG"]
SCENE_MAP = _SCENE["SCENE_MAP"]
SCENE_BATTLE_RESULT = _SCENE["SCENE_BATTLE_RESULT"]
SCENE_GAME_MENU = _SCENE["SCENE_GAME_MENU"]
SCENE_NOTICE = _SCENE["SCENE_NOTICE"]
SCENE_TELEPORT = _SCENE["SCENE_TELEPORT"]


@dataclass(frozen=True)
class ActionTraits:
    """一个动作的差异化行为。全部字段有保守默认值。"""

    #: 到点必须精确停住（到点后再做小碎步纠正）。适合交互/跳台/过图。
    strict_arrival: bool = False
    #: 临近目标的一段路允许切走路，避免冲过头。
    approach_by_walking: bool = True
    #: 执行途中目标消失/场景变了就中止，而不是硬走完。
    abort_on_target_lost: bool = False
    #: 允许执行的场景。`("*",)` = 不限。空元组 = **任何场景都不许**。
    #:
    #: ★ §29：战斗类动作（普攻/技能/大招/闪避/切人）的 allowed_scenes
    #:   **加了 SCENE_WORLD**。原因：7 类场景分类器没有模型，场景判定降级为
    #:   SceneDetector 的**二值**结果（WORLD / UNKNOWN），**永不产出 COMBAT**
    #:   —— 不加的话这五个动作一条也放不出来。
    #:   SCENE_COMBAT 保留，等 7 类分类器回来。
    #:
    #: ⚠️ **allowed_scenes 是护栏，不是触发器。**
    #:   它只回答"这个场景**允不允许**做这个动作"，**不回答"现在该不该做"**。
    #:   真正的战斗决策由 policy 层根据世界实体检测
    #:   （``PolicyContext.has_nearby_enemy()``）判断 —— 见 §29.4。
    #:   也就是说：放宽护栏**不等于**"在 WORLD 里见人就打"，
    #:   只是让 policy 有资格产出战斗意图。
    allowed_scenes: tuple[str, ...] = (SCENE_ANY,)
    #: 与上一个动作的最小间隔（毫秒），防连点。
    min_gap_ms: int = 0
    #: 单个动作的最长执行时长（毫秒），超时判失败。
    max_duration_ms: int = 5000
    #: 执行期间允许同时按住 W（移动中转向，原则 5）。
    allow_move_while_turning: bool = False
    #: 需要目标窗口在前台才执行。
    requires_foreground: bool = True
    #: 该动作会修改键鼠状态，结束必须 release_all。
    #:
    #: ⚠️ **纯声明字段 —— executor 从不读它**（§40 全仓库核实：
    #:    grep 只在 traits.py 的定义与文档里命中，无任何消费方）。
    #:    动作到底会不会长按，**只由 executor 的分支决定**。
    #:
    #:    这个字段曾经两次说谎：NORMAL_ATTACK 标 True 且实现也长按
    #:    （但那个长按语义本身是错的，§39）；DODGE 标 False（默认值，
    #:    压根没写这个字段）而实现却长按 —— **声明与实现正好相反**，
    #:    照着 traits 读代码会得出完全错误的结论。
    #:
    #:    保留它是因为「动作会留下按住的键」是有价值的文档信息，
    #:    但**改动它不会改变任何行为**。要改行为请改 executor 分支。
    holds_keys: bool = False
    #: 「点按」类动作的**真实按键保持时长**（毫秒）。down 与 up 之间
    #: 真的间隔这么久。只对点按类动作有意义（普攻/技能/大招/闪避）。
    #:
    #: ⚠️ **未标定**：这批值来自社区实测区间，不是本项目的实测。
    #:    改之前先看 :data:`NORMAL_ATTACK_TAP_HOLD_MS` 的说明。
    tap_hold_ms: int = 0

    def allows_scene(self, scene_id: str) -> bool:
        return SCENE_ANY in self.allowed_scenes or scene_id in self.allowed_scenes


#: 面向所有动作的默认 traits
DEFAULT_TRAITS = ActionTraits()

#: 普攻的**真实按键保持时长**（毫秒）——§39，从长按改点按。
#:
#: ⚠️ **未标定 —— 这是本项目里少数几个纯经验值之一。**
#:    依据：社区实测「短按 tap 左键做普攻」区间 100~200 ms，取中值 120。
#:    我做检索时**没有找到**任何给出精确毫秒阈值的权威来源 ——
#:    能确认的只有定性机制（按住 = 蓄力重击、点按 = 普攻），两个独立
#:    来源一致；「100~200ms」这个数字属于社区口口相传，未经核实。
#:
#:    **怎么标定**（唯一可靠的办法是真机跑）：
#:      1. ``e2e_dryrun.py --live --frames 10``
#:      2. 游戏里出**连续普攻**且没有蓄力光效 -> 偏小，OK
#:         出**重击**（蓄力条）          -> 调小到 60~80
#:         偶尔**没反应**（点击丢了）     -> 调大到 160~200
#:
#:    安全边界：原神蓄力重击是**秒级**（那维莱特满蓄约 4 s），120 ms
#:    离蓄力阈值有两个数量级的余量，不存在「刚好误触发蓄力」的风险。
NORMAL_ATTACK_TAP_HOLD_MS: int = 120

#: 闪避的**真实按键保持时长**（毫秒）——§40。
#:
#: ★ 为什么要点按：原神里 Shift 是**同一个键**，
#:   **点按 = 闪避，长按 = 冲刺**。原实现对 Shift 用了 ``hold_key()``，
#:   于是 ``world_emergency_dodge`` 触发时角色**冲刺**而不是闪避 ——
#:   而且冲刺移速更快，看起来更像「闪避成功了」，极难自查。
#:
#: ⚠️ **未标定**，与 :data:`NORMAL_ATTACK_TAP_HOLD_MS` 同源同依据
#:    （社区实测区间 100~200 ms 取中值）。取 120 是为了与普攻一致，
#:    这样「一次 tick 最多睡多久」有个统一上界（见 §40 的预算分析）。
#:
#:    **怎么标定**：``e2e_dryrun.py --live`` 触发 DODGE，看游戏里出的是
#:    闪避位移还是持续冲刺。偏冲刺就调小，偶发无效就调大。
DODGE_TAP_HOLD_MS: int = 120

#: ⭐ 动作白名单 -> traits。**新增动作请在这里加一行**，不要在别处写 if-else。
TRAITS: dict[str, ActionTraits] = {
    # ---------------- COMBAT ----------------
    # 普攻 = **点按左键**（短 tap），**不是长按**（§39）。
    #
    # ★ 为什么从长按改成点按（真机实测）
    #   --live 首测：6 帧 NORMAL_ATTACK 意图 -> 游戏里只触发 **1 次重击**。
    #   根因是实现走 ``mouse_down(LBUTTON)`` 一直不松，而原神的机制
    #   就是「按住 = 蓄力」：两个独立来源都写明
    #   「**长按普攻键(PC端为按住鼠标左键)将进入蓄力准备阶段**」，
    #   移动端对照表同样是「普通攻击 Tap / 蓄力重击 Hold」。
    #   所以长按不是「攻击得更久」，而是**换了一个动作**。
    #
    # ⚠️ :data:`NORMAL_ATTACK_TAP_HOLD_MS` 是**未标定值** —— 来自社区
    #    实测区间 100~200 ms 的中值，**本项目没有真机标定**。
    #    真机验证方法：跑 ``e2e_dryrun.py --live``，看游戏里出的是
    #    连续普攻还是重击；偏重击就调小，偏漏点击就调大。
    "NORMAL_ATTACK": ActionTraits(
        strict_arrival=False,
        approach_by_walking=True,
        abort_on_target_lost=False,
        allowed_scenes=(SCENE_COMBAT, SCENE_WORLD),
        # ★ 2026-10-14 用户拍板：0 -> 300。
        #   原来写 0 的后果是真机实测出来的：5Hz 主循环下每 tick 都会打
        #   一次，实测 80.5 秒 355 次（4.4 次/秒）。唯一在拦的只有
        #   ``SafetyConfig.rate_limit_ms = 50``（20 次/秒），而 200ms/tick
        #   本来就大于它，所以形同虚设。
        #   300ms -> 上限 3.3 次/秒，接近真机连点的合理上限。
        min_gap_ms=300,
        # 从 30_000 降到 2_000：现在是「按一下就完事」，30 秒的
        # max_duration 是长按语义留下的垃圾值。
        max_duration_ms=2_000,
        holds_keys=False,         # ★ 点按 = 按下即抬起，不再是长按
        tap_hold_ms=NORMAL_ATTACK_TAP_HOLD_MS,
    ),
    "USE_SKILL": ActionTraits(
        strict_arrival=True,      # 技能要等到 E 转好，必须精确
        allowed_scenes=(SCENE_COMBAT, SCENE_WORLD),
        min_gap_ms=150,
        max_duration_ms=2000,
    ),
    "USE_BURST": ActionTraits(
        strict_arrival=True,      # 大招要等充能满 + 角色在场
        abort_on_target_lost=True,
        allowed_scenes=(SCENE_COMBAT, SCENE_WORLD),
        min_gap_ms=800,           # 大招后摇，给足冷却
        max_duration_ms=4000,
    ),
    "DODGE": ActionTraits(
        strict_arrival=True,      # 闪避窗口很短，晚一点就空
        abort_on_target_lost=True,
        allowed_scenes=(SCENE_COMBAT, SCENE_WORLD),
        min_gap_ms=1500,          # 闪避内置 CD（闪避与冲刺**共享**这个 CD，
                                  # 所以改点按不影响这里的口径）
        # 从 800 收紧到 500：现在是「按一下就完事」，
        # 800ms 是长按语义留下的宽裕值。
        max_duration_ms=500,
        # ★ 这个字段**本来就没写**（走默认 False），而 executor 却一直
        #   长按 Shift —— 声明与实现正好相反（§40）。现在两边终于一致，
        #   但要记住：改它不改变行为，行为由 executor 分支决定。
        holds_keys=False,
        tap_hold_ms=DODGE_TAP_HOLD_MS,
    ),
    "SWITCH_CHARACTER": ActionTraits(
        strict_arrival=True,      # 要切到指定的人场上
        abort_on_target_lost=True,
        allowed_scenes=(SCENE_COMBAT, SCENE_WORLD),
        min_gap_ms=200,
        max_duration_ms=3000,
    ),

    # ---------------- NAV ----------------
    # 移动类：**允许移动中转向**（原则 5：转向必须在移动中完成）
    "MOVE_FORWARD": ActionTraits(
        strict_arrival=True,
        approach_by_walking=True,
        allowed_scenes=(SCENE_ANY,),
        max_duration_ms=60_000,
        allow_move_while_turning=True,
        holds_keys=True,
        requires_foreground=False,  # 走路不需要严格前台
    ),
    "MOVE_BACK": ActionTraits(
        strict_arrival=True, allowed_scenes=(SCENE_ANY,),
        max_duration_ms=10_000, allow_move_while_turning=True,
        holds_keys=True, requires_foreground=False,
    ),
    "MOVE_LEFT": ActionTraits(
        strict_arrival=True, allowed_scenes=(SCENE_ANY,),
        max_duration_ms=5_000, allow_move_while_turning=True,
        holds_keys=True, requires_foreground=False,
    ),
    "MOVE_RIGHT": ActionTraits(
        strict_arrival=True, allowed_scenes=(SCENE_ANY,),
        max_duration_ms=5_000, allow_move_while_turning=True,
        holds_keys=True, requires_foreground=False,
    ),
    "SPRINT": ActionTraits(
        strict_arrival=False,
        allowed_scenes=(SCENE_ANY,),
        min_gap_ms=1500,           # 与 MaaEnd 的 kAutoSprintCooldownMs 一致
        max_duration_ms=5_000,
        holds_keys=True,
        requires_foreground=False,
    ),
    "JUMP": ActionTraits(
        strict_arrival=False,
        allowed_scenes=(SCENE_ANY,),
        min_gap_ms=200,
        max_duration_ms=1_500,
        requires_foreground=False,
    ),

    # ---------------- MOVE_TO（2026-10-14）----------------
    # 「走到世界坐标 (x, y)」—— **唯一带目标的导航意图**。
    #
    # ★ 它是**目标**而不是**动作**：执行器每次只推进**一步**（转一段 +
    #   决定走不走），到达与否由下一 tick 的同一意图继续推进。
    #   所以 min_gap_ms 必须是 0 —— 同一条 MOVE_TO 每 tick 命中一次是
    #   **正常**的，限频会让「推进」变成「每 300ms 走一步」，走到天亮。
    #   （真正会刷屏的是 INTENT_EXECUTED 统计，那是另一回事，见下。）
    "MOVE_TO": ActionTraits(
        strict_arrival=True,
        approach_by_walking=True,
        # ★ 定位一丢就停。定位丢时**禁止**用运动预测填补 —— 那是
        #   ``contracts/state_packet.PositionInfo`` 的明文契约。
        abort_on_target_lost=True,
        # ⚠️ 这里**故意用 SCENE_ANY 而不是 SCENE_WORLD**。
        #   本文件上方（allowed_scenes 注释）记载：7 类场景分类器没有模型，
        #   场景判定降级为 SceneDetector 的**二值**结果（WORLD / UNKNOWN），
        #   **永不产出 COMBAT**。而真机在户外行走时场景常常是 UNKNOWN ——
        #   只写 (SCENE_WORLD,) 会让这条意图**一条都放不出来**，
        #   症状是「规则命中了但没有动作」，极难联想。
        #   与同组的 MOVE_FORWARD / MOVE_BACK 保持一致。
        allowed_scenes=(SCENE_ANY,),
        min_gap_ms=0,
        # ★ 导航是**长程**动作。走到目标可能要几十秒；用默认的 5_000ms
        #   会在半路被判 TIMEOUT。
        max_duration_ms=180_000,
        allow_move_while_turning=True,
        holds_keys=True,             # 转向与行走都要按住 W
        requires_foreground=False,   # 与其它移动类一致
    ),

    # ---------------- INTERACT（2026-10-11）----------------
    # F 键：拾取 / 采集 / 开宝箱 / 对话。**点按**，不是长按。
    # 允许 WORLD 与 UNKNOWN —— 二值场景判定下「不确定但画面里有个
    # 可交互物」时按一次 F 的代价极低（最坏是没反应），而漏掉的代价是
    # 资源白扔。min_gap 给 800ms 是为了防止对着同一个箱子连按。
    "USE_PICKUP": ActionTraits(
        strict_arrival=False,
        allowed_scenes=(SCENE_WORLD, SCENE_UNKNOWN),
        min_gap_ms=800,
        max_duration_ms=1_000,
        holds_keys=False,
        requires_foreground=True,
    ),

    # ---------------- UI ----------------
    # 菜单类：允许 UNKNOWN 场景（识别不准时也能点开/关掉）
    "OPEN_INVENTORY": ActionTraits(
        strict_arrival=True, abort_on_target_lost=True,
        allowed_scenes=(SCENE_MAIN_MENU, SCENE_COMBAT, SCENE_BAG, SCENE_MAP,
                        SCENE_UNKNOWN),
        min_gap_ms=300, max_duration_ms=3_000,
    ),
    "OPEN_MAP": ActionTraits(
        strict_arrival=True, abort_on_target_lost=True,
        allowed_scenes=(SCENE_MAIN_MENU, SCENE_COMBAT, SCENE_MAP, SCENE_UNKNOWN),
        min_gap_ms=300, max_duration_ms=3_000,
    ),
    "OPEN_PAIMON_MENU": ActionTraits(
        strict_arrival=True, abort_on_target_lost=True,
        allowed_scenes=(SCENE_MAIN_MENU, SCENE_BAG, SCENE_MAP, SCENE_UNKNOWN),
        min_gap_ms=300, max_duration_ms=3_000,
    ),
    "CLICK_ELEMENT": ActionTraits(
        strict_arrival=True, abort_on_target_lost=True,
        min_gap_ms=100, max_duration_ms=2_000,
    ),
    "CONFIRM": ActionTraits(
        strict_arrival=True, abort_on_target_lost=True,
        min_gap_ms=200, max_duration_ms=2_000,
    ),
    "CANCEL": ActionTraits(
        strict_arrival=True, abort_on_target_lost=True,
        min_gap_ms=200, max_duration_ms=2_000,
    ),
    "CLOSE": ActionTraits(
        strict_arrival=True, abort_on_target_lost=True,
        # 战斗中关 UI 是危险操作（可能退出战斗）—— 这里显式禁止
        allowed_scenes=(SCENE_BAG, SCENE_MAP, SCENE_GAME_MENU,
                        SCENE_NOTICE, SCENE_DIALOG, SCENE_UNKNOWN),
        min_gap_ms=200, max_duration_ms=2_000,
    ),

    # ---------------- SYSTEM ----------------
    "WAIT": ActionTraits(
        strict_arrival=False,
        allowed_scenes=(SCENE_ANY,),
        max_duration_ms=30_000,
        requires_foreground=False,
    ),
    "SCREENSHOT": ActionTraits(
        strict_arrival=False,
        allowed_scenes=(SCENE_ANY,),
        max_duration_ms=5_000,
        requires_foreground=False,
    ),
    "NOOP": ActionTraits(
        strict_arrival=False,
        allowed_scenes=(SCENE_ANY,),
        max_duration_ms=100,
        requires_foreground=False,
    ),
}


def traits_of(action: str) -> ActionTraits:
    """取动作的 traits。**未配置返回保守默认**并告警 —— 静默走默认很危险。

    返回的是 `ActionTraits()`：`allowed_scenes=("*",)` 看似宽松，但
    `strict_arrival=True` + `min_gap_ms` 默认，配合 validator 的
    白名单校验，未知动作会因为**不在 TRAITS 里**被 validator 拦下。
    """
    if isinstance(action, str) and action in TRAITS:
        return TRAITS[action]
    from ..contracts.intent_packet import ActionType

    try:
        ActionType(action)
    except ValueError:
        # 连白名单都不在 —— 这是硬错误，让 validator 去报
        return ActionTraits(allowed_scenes=(), strict_arrival=True)
    import logging

    logging.getLogger(__name__).warning(
        "动作 %s 没有配置 ActionTraits，使用保守默认值（strict_arrival=True）", action
    )
    return ActionTraits(strict_arrival=True)


def missing_traits() -> list[str]:
    """列出「在白名单里但没配 traits」的动作。供 check_env / 单测使用。"""
    from ..contracts.intent_packet import ActionType

    return sorted(a.value for a in ActionType if a.value not in TRAITS)


def extra_traits() -> list[str]:
    """列出「配了 traits 但不在白名单」的动作。多半是拼错了。"""
    from ..contracts.intent_packet import ActionType

    valid = {a.value for a in ActionType}
    return sorted(set(TRAITS) - valid)


def validate_table() -> list[str]:
    """校验 traits 表自身的一致性。**返回问题列表**，空 = 正常。

    这一条对应原则 1：加新动作忘了配 traits / 拼错名字，应该在**加载期**
    就被看到，而不是运行到一半行为诡异。
    """
    problems: list[str] = []
    for name in missing_traits():
        problems.append(f"动作 {name} 在白名单里但没有 ActionTraits")
    for name in extra_traits():
        problems.append(f"TRAITS 里的 {name} 不在 ActionType 白名单中（拼错了？）")
    for name, t in TRAITS.items():
        if t.max_duration_ms < 0:
            problems.append(f"{name}.max_duration_ms 为负")
        if t.min_gap_ms < 0:
            problems.append(f"{name}.min_gap_ms 为负")
        if not t.allowed_scenes:
            problems.append(f"{name}.allowed_scenes 为空 —— 任何场景都执行不了")
    return problems


__all__ = [
    "DEFAULT_TRAITS",
    "DODGE_TAP_HOLD_MS",
    "NORMAL_ATTACK_TAP_HOLD_MS",
    "SCENE_ANY",
    "SCENE_BAG",
    "SCENE_BATTLE_RESULT",
    "SCENE_COMBAT",
    "SCENE_CUTSCENE",
    "SCENE_DIALOG",
    "SCENE_GAME_MENU",
    "SCENE_LOADING",
    "SCENE_MAIN_MENU",
    "SCENE_MAP",
    "SCENE_NOTICE",
    "SCENE_TELEPORT",
    "SCENE_UNKNOWN",
    "SCENE_WORLD",
    "TRAITS",
    "ActionTraits",
    "extra_traits",
    "missing_traits",
    "traits_of",
    "validate_table",
]
