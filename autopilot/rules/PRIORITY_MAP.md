# 规则优先级地图

> 2026-10-11 · 对应 `autopilot/rules/*.toml` 的**实际**状态，不是设计意图。
> 生成方式：跑 `build_policy()` 加载全部规则后按 `(scenes, priority)` 分组。

## ★ 2026-10-11 更新：36 条拆成「15 活跃 / 21 待接入」

原先 21 条里有 **21 条**（扩到 36 条后是 21 条）依赖今天恒 UNKNOWN 的信号，
已整类挪进 `rules/_future/`。加载器是 `d.glob("*.toml")`（**非递归**），
所以那个子目录**不会被加载**——它们只是躺着，等信号接上。

**判死标准**（`tests/unit/test_rule_expansion.py::test_active_dir__has_no_unreachable_rules`）：

- `when` 引用了 `hp / energy / resin / pos_quality / pos_usable / since /
  q_ready / e_ready / has_enemy / target_locked / onfield / slot`
- `when` 拿 `scene()` 比一个二值检测器**永不产出**的值
- `when` 是字面量 `false`（2 条占位规则）

| 目录 | 文件 | 条数 |
|---|---|---|
| **活跃** `rules/` | `genshin_combat.toml` | 1 |
| | `genshin_combat_advanced.toml` | 3 |
| | `genshin_interaction.toml` | 4 |
| | `genshin_recovery.toml` | 4 |
| | `genshin_world.toml` | 3 |
| | `genshin_team.toml` | 0（带诊断的空文件） |
| **待接入** `rules/_future/` | `genshin_bag.toml` | 7 |
| | `genshin_bag_advanced.toml` | 1 |
| | `genshin_nav.toml` | 4 |
| | `genshin_combat_signals_pending.toml` | 9 |
| | `genshin_world_signals_pending.toml` | 0（拆分时空的，全部落在 combat 里） |

解锁路线见 `recognition/notes/signal_integration_roadmap.md`。

> ⚠️ **地基守卫**：`rule_loader` 现在用非递归 `glob`。哪天有人改成 `rglob`，
> 21 条死规则会**原地复活且不报任何错**。这一点由
> `test_future_dir__exists_and_is_not_loaded` 盯着。

---

## 排序规则回顾

`LocalPolicyTree.decide()` **按 priority 升序遍历，命中即停**（`local_tree.py:290`）。
所以 priority 越小越优先，同 priority 时**列表顺序**决定谁先被评估 ——
而这个顺序取决于文件加载顺序，**不是显式声明的**。

★ 这就是同优先级冲突的真正危害：它看起来是稳定的，实际是文件名的偶然。

---

## 1 · WORLD 场景下的优先级分布（唯一的活场景）

场景识别是**二值**的（`WORLD` / `UNKNOWN`），所以下面这张表**就是全部**。

| priority | 规则 | 条件依赖 | 状态 |
|---|---|---|---|
| 5 | `recovery_hold_when_perception_down` | `perception_ok()` | 可触发 |
| 8 | `recovery_hold_when_ocr_fails` | `ocr_ok()` | 可触发 |
| 10 | `combat_group_spread_when_swarmed` | `enemy_count()` | 可触发 |
| 12 | `recovery_cautious_when_scene_unsure` | `scene_confidence()` | 可触发 |
| 15 | `combat_skip_locked_chest` | `has_locked_chest()` | 可触发 |
| 20 | `world_open_chest` | `has_nearby_chest()` | 可触发（`enabled=false`） |
| 25 | `interact_avoid_chest_storm` | `nearby_count()` | 可触发 |
| 30 | `world_pickup_ore` | `has_nearby_ore()` | 可触发（`enabled=false`） |
| 32 | `interact_pick_ore_when_close` | `nearest_group_px()` | 可触发 |
| 40 | `world_pickup_drops` | `has_nearby_drops()` | 可触发（`enabled=false`） |
| 42 | `interact_pick_drops_when_close` | `nearest_group_px()` | 可触发 |
| 45 | `interact_walk_to_nearest_ore_first` | `nearest_group_px()` | 可触发 |
| 50 | `world_attack_only_with_enemy` | `has_nearby_enemy()` | 可触发（**唯一 enabled=true**） |
| 50 | `recovery_stand_down_when_cleared` | `enemy_count()` | 可触发 |
| 60 | `combat_single_focus_attack` | `enemy_count() == 1` | 可触发 |

**这 15 条全部 `enabled = false`，只有 `world_attack_only_with_enemy` 是开着的。**
换句话说：引擎现在**只会做一件事**——看见敌人就平A。

---

## 2 · 同优先级冲突（必须处理）

### ⚠ 冲突 A：p5 完全重复 —— **已修**
`recovery_hold_when_perception_down` 与 `combat_hold_when_perception_bad`
条件一模一样。已删除后者（保留 recovery 的那条，语义归属更准）。

### ⚠ 冲突 B：p30 三条挤在一起 —— **已修**
原状：`world_pickup_ore` / `interact_pick_ore_when_close` /
`world_burst_only_with_enemy` 同在 p30。已把 `interact_pick_ore_when_close`
错开到 **p32**，p30 只剩两条。

> 注意这三条是**互补**的（一个管远、一个管近），靠 `nearest_group_px()`
> 的互斥避免双触发 —— 很脆：距离阈值一旦重叠两条会同时命中，而「命中即停」
> 意味着只有靠前的那条生效。开的时候**两条必须一起开**。

### ⚠ 冲突 C：p50 两条语义相反 —— **未修，待决策**
`world_attack_only_with_enemy`（开打）与 `recovery_stand_down_when_cleared`
（收手）条件**可能同时为真**（有敌人但血条数为 0）。谁赢取决于文件顺序，
而那**是巧合不是设计**。
> **建议**：显式错开（收手 p48 < 开打 p50）。没做是因为两条现在都关着，
> 改了也验证不了 —— 留给你开规则时顺手定。

### ⚠ 冲突 D：p20 两条 —— 虚占，不是真冲突
`world_emergency_dodge`（恒 UNKNOWN，永不触发）与 `world_open_chest`。
它占着 p30/p20 这类位置**让人误以为那里有竞争**，实际上它从不参与。
这条是 `test_rule_expansion.py::test_rules__world_scene_collision_is_known`
**自动抓出来的**（我没手动记录），已登记。

---

## 3 · 建议的优先级分层

| 段 | 用途 | 规则 |
|---|---|---|
| 1–9 | **保命** —— 感知不可信、OCR 不可用 | 停下 |
| 10–19 | **群体生存** —— 被围、识别没把握 | 后退 / 警惕 |
| 20–29 | **高价值目标** —— 宝箱、上锁箱 | 交互 |
| 30–39 | **资源采集** —— 矿 | 先走近再采集 |
| 40–49 | **拾取** —— 掉落物 | 捡 |
| 50–59 | **开打 / 收手** | 战斗 |
| 60–89 | **单体战斗** | 平A |

**当前分层的问题**：p5 上有两条重复（冲突 A），p30 三条（冲突 B），
p50 两条（冲突 C）。**打开任何一条新规则之前，这三处必须先错开。**

---

## 4 · 一个更根本的建议

这张表暴露的真问题不是优先级，是**21 条老规则里有 14 条条件恒 UNKNOWN**。
它们不参与竞争，却**占着 priority 位置**，让「谁会先命中」这件事
看起来比实际更复杂。

> **先做减法再做加法**：把那 14 条恒 UNKNOWN 的规则删掉（或挪到
> `_disabled_pending_signals.toml` 之类的地方），WORLD 场景下的规则表
> 会从 18 条缩到 4~5 条 —— 那才是**真正在运行**的决策树。
> 在那之前，往里加规则等于往一个大部分成员都在睡觉的表里加成员。