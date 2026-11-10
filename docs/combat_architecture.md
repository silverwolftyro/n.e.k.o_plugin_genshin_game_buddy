# 战斗架构：规则 与 连招

> 面向**玩家**。你要改的大概率只有两处：
> `autopilot/rules/*.toml`（单步决策）和 `autopilot/combos/combos.toml`（连招顺序）。
>
> 面向维护者：`autopilot/policy/combo_manager.py`（状态机）、
> `autopilot/policy/local_tree.py`（拦截点）、
> `tests/unit/test_combo_sequence.py`（行为钉死）。

---

## 1 · 两层，各管一件事

```
        ┌─ 一帧 StatePacket ─────────────────────────────────┐
        │                                                     │
        ▼                                                     │
  LocalPolicyTree.poll(state)          ← 每 tick 调一次
        │                                                     │
        ├─ decide(state)  ──► 规则命中了谁（+ priority）        │
        │        （纯判定，不发意图、不记账；离线工具也用它）      │
        │                                                     │
        └─ ComboRunner.next_intent(ctx) ──► 连招该出哪一个      │
                 （有状态；知道自己在第几步）                    │
        │                                                     │
        ▼                                                     │
   一个 IntentPacket （**一轮只有一个**）                        │
        │                                                     │
        ▼                                                     │
   IntentExecutor.execute()  ──► validator → safety → 按键       │
```

| | 规则（`rules/*.toml`） | 连招（`combos/combos.toml`） |
|---|---|---|
| 回答的问题 | **这一帧**该做什么 | **接下来这一串**该做什么 |
| 有状态吗 | 没有（只记冷却） | 有（当前第几步） |
| 优先级机制 | `priority = 0..100`，小的先来，命中即停 | 无。**声明顺序**就是执行顺序 |
| 谁能让路 | 小的 priority 赢 | `break_below_priority` 以下的规则打断它 |
| 每步产出 | 一个 `IntentPacket` | 同样是一个 `IntentPacket` |

**关键：连招没有特权。** 它产出的意图和规则产出的是同一个类型，走同一个
`validator` / `safety` / `traits` 校验。没有旁路 —— 想让它更安全做不到，
想让它更放肆也做不到。

---

## 2 · 「一轮一个意图」是硬契约

不管规则还是连招，**一个 tick 最多发一个 `IntentPacket`**，绝不一次发一串。

为什么：执行层的限频（`min_gap_ms`）、安全闸（用户让权窗口）、长按键释放
全挂在这个契约上。一次发一串会同时绕过这三样 —— 而它们恰好是「机器人不会
在你抢鼠标时还乱按」的全部保证。

所以一条 4 步连招**至少**要 4 个 tick（实际通常更多，因为切人要占一 tick）。

---

## 3 · 写连招

编辑 `autopilot/combos/combos.toml`。

### 3.1 最小可用

```toml
[[combo]]
name = "钟夜"
enabled = true
trigger = "has_nearby_enemy() and scene() == 'WORLD'"
sequence = [
    { char = "钟离", action = "USE_SKILL", when = "e_ready()" },
    { char = "夜兰", action = "USE_BURST",  when = "q_ready()"  },
]
```

**`name` 必须全局唯一**（连招之间、连招与规则之间都算）。

### 3.2 字段表

#### `[[combo]]` 顶层

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | 字符串 | 必填 | 全局唯一 |
| `trigger` | DSL | 必填 | 什么时候开始这整条连招 |
| `sequence` | 表数组 | 必填 | 按什么顺序打，不能为空 |
| `enabled` | 布尔 | `true` | 关着的**必须**写 `note` |
| `note` | 字符串 | `""` | 中文说明。`enabled=false` 时必填 |
| `scenes` | 字符串数组 | `["WORLD"]` | 跑一半场景变了会**中断** |
| `cooldown_ms` | 整数 | `0` | 走完/被打断后多久不重启 |
| `break_below_priority` | 整数 | `20` | 命中 priority **小于**它的规则会打断它。`0` = 谁都打不断 |
| `switch_timeout_ms` | 整数 | `3000` | 等一次切人完成的上限，超了中断 |
| `timeout_ms` | 整数 | `20000` | 整条连招的上限，超了中断 |
| `ttl_seconds` | 整数 | `2` | 每步意图的有效期 |

#### `sequence` 里的一步（内联表）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `action` | 动作名 | 必填 | `USE_SKILL` / `USE_BURST` / `NORMAL_ATTACK` / `SWITCH_CHARACTER` / … |
| `char` | 角色名 | `""` | 对谁做。见第 4 节 |
| `repeat` | 整数 | `1` | 连续几个 tick 各做一次（1..64） |
| `when` | DSL | `""` | 这步的守卫。**为假/UNKNOWN 就跳过这一步** |
| `gap_ms` | 整数 | 不传 | 按完键后在**执行层里**多睡多久（0..10000） |
| `hold_ms` | 整数 | 不传 | 按键按住时长 |
| `settle_ms` | 整数 | `0` | 执行器返回后再等这么久才出下一步（0..10000） |

### 3.3 `gap_ms` 和 `settle_ms` 有什么区别

都在「等」，等的地方不一样：

- **`gap_ms`** 在**执行层**：按完键就地 `sleep`，**这一 tick 会超出 200ms 预算**
  （故意的）。用来给施法后摇留时间（钟离开盾后 700ms 再切人）。
- **`settle_ms`** 在**策略层**：执行器已经返回了，runner 自己记一个到期时刻，
  下一 tick 才开始出下一步。**不占用执行线程**。

能用 `settle_ms` 就别用 `gap_ms` —— 后者会拖慢整个循环。

---

## 4 · 切人是怎么发生的（**最容易踩坑的一节**）

你只写了 `action`，**从来没写过切人**。引擎会自己插进去。

引擎每一步都会问两个问题：

1. 「这个角色在几号位？」← **两级查找，静态优先**：
   先查 `keymap.toml` 的 `[char_slots]`（你手填的），
   查不到再查大世界 OCR 建出来的名册。
2. 「现在站着的是几号位？」← `ActiveSlotTracker`（切人记账）

| 名册说 | 记账说 | 引擎做什么 |
|---|---|---|
| 夜兰 = 2 号位 | 现在是 2 号 | 直接放大招 |
| 夜兰 = 2 号位 | 现在是 1 号 | **先发 `SWITCH_CHARACTER(夜兰)`**，等记账变成 2，再放大招 |
| **不知道** | 任意 | **跳过这一步**，发一条 `COMBO_STEP_UNKNOWN` 事件 |
| 任意 | **不知道** | **跳过这一步**，同上 |

「不知道就跳过」不是偷懒：猜错的后果是**夜兰的大招打在钟离身上**。
不做只是这一轮没输出，猜错是把增益和伤害喂给错的人。

⚠️ **静态优先这个顺序不能反。** 反过来的话，你手填 `[char_slots]` 去纠正
OCR 读错的名字时，规则会切对人、连招会切错人 —— 同一个错名字在两套机制里
表现不一致，是最难查的一类问题。连招与执行层用的是**同一条**查找逻辑
（`ComboRunner._slot_of` 与 `IntentExecutor._slot_of`）。

### 🔴 启动后约 20 秒内连招大概率不出手

编队名册靠大世界 OCR 轮转建立（单次 OCR 实测 1.0~1.5 秒，每 25 tick 转 1 格，
4 格一圈 → **~20 秒**）。在那之前「谁在场」「某人几号位」都是未知，
于是上面的「跳过」会**逐条跳过整个序列**。

不是静默失败（会有事件和日志），但确实打不出去。**想立刻可用就手填**
`autopilot/config/keymap.toml`：

```toml
[char_slots]
"钟离" = 1
"夜兰" = 2
"纳西妲" = 3
"神里绫华" = 4
```

左边必须和游戏里显示的名字**一字不差**（OCR 读的就是屏幕上的字）。
手填了 `[char_slots]` 就**以你为准**，OCR 名册只做兜底。

> 有测试钉着这条（`test_combo_sequence.py::TestStaticCharSlots`）：
> 名册一格都没填 + 手填了 `[char_slots]`，连招必须照常跑完。
> 否则这份文档就是在骗玩家。

### 显式切人

想让切人在序列里占一整步（比如「切过去之后先等两拍」）：

```toml
{ action = "SWITCH_CHARACTER", char = "夜兰", settle_ms = 400 }
```

`SWITCH_CHARACTER` **必须**写 `char` —— 切人靠角色名换数字键。

---

## 5 · 连招什么时候会让路

| 情况 | 行为 | 配置 |
|---|---|---|
| 命中 priority < `break_below_priority` 的规则 | **立刻中断**，这一轮归那条规则 | 默认 20 |
| 你按了任何游戏键 | **立刻中断** | 无（`loop.py::_on_physical_input`） |
| 按了 F12 急停 | **立刻中断** + 停机 | 无（`loop.py::_on_panic`） |
| 感知层不健康 | **立刻中断**，这一轮不发任何意图 | 无 |
| 场景掉出 `scenes` | **立刻中断** | `scenes` |
| 切人超过 `switch_timeout_ms` 没生效 | **立刻中断** | 默认 3000ms |
| 整条超过 `timeout_ms` 没走完 | **立刻中断** | 默认 20000ms |

中断之后走冷却，**不会原地复活**。冷却从**中断那一刻**起算，不是从启动起 ——
否则一条「刚开始就被应急规则打断」的连招会在几帧后立刻重开，变成抖动。

### 默认阈值 20 的来历

- **比 20 小** → 5/8/12 恢复类（感知不健康 / OCR 失败 / 场景没把握）、
  10 被围后退、15 开锁宝箱。这些是「不打会出事」，连招必须让路。
- **比 20 大** → 20 开宝箱、25 躲宝箱风暴、30~45 采集拾取、50 收手、
  60 集火、80 通用放 E、85 普攻兜底。战斗中捡矿石不该打断连招。
- 连招**压过 80/85 那两条通用规则** —— 不然你打开连招后会看到「什么也没变」，
  因为 80 号规则先赢了。

---

## 6 · 怎么让连招循环起来

不需要代码里的 `while`。**走完 + 冷却到期 + trigger 还成立 = 再来一遍**，
每次都从第 1 步开始。

想要「打完回到钟离补盾」，就把**最后一步的 `char` 写成钟离**：

```toml
[[combo]]
name = "钟夜循环"
enabled = true
trigger = "has_nearby_enemy() and scene() == 'WORLD'"
cooldown_ms = 8000
sequence = [
    { char = "钟离", action = "USE_SKILL", when = "e_ready()", gap_ms = 700 },
    { char = "夜兰", action = "USE_BURST",  when = "q_ready()" },
    { char = "钟离", action = "NORMAL_ATTACK", repeat = 3 },
]
```

---

## 7 · 条件表达式（DSL）

`trigger` 和每步的 `when` 用的是**和规则完全同一套** DSL
（`autopilot/policy/dsl.py`），同一份白名单。写了不认识的函数会**在启动时报错**，
不是运行时。

### 可用函数（全部）

```
count()  since()  last_exec()  t()                    # 时间 / 次数
scene()  scene_confidence()  perception_ok()  ocr_ok()  fps()
hp()  energy()  resin()                               # 资源
onfield('夜兰')  slot()  e_ready()  q_ready()          # 队伍 / 技能 CD
has_nearby_enemy()  has_nearby_chest()  has_nearby_ore()  has_nearby_drops()
has_locked_chest()  nearby_count()  nearest_group_px()  enemy_bar_count()
enemy_count()  nearby_entity_count()  has_enemy()  target_locked()
pos_quality()  pos_usable()
```

还支持 `min()` / `max()` 和 TOML 布尔 `true` / `false`。

### 🔴 三态：UNKNOWN 不是 False

信号读不到时返回「未知」，结果是 **UNKNOWN**。规则的反应是**不动作**，
并且会在诊断的 `unknowns` 里记一笔原因。

```
写 when = "hp() < 0.35"        # UNKNOWN 时不触发（保守）
```

三个「读不到就会变 UNKNOWN」的典型：没人切过人的时候 `onfield()` 恒 UNKNOWN；
`cd_ready` 没填时 `e_ready()` / `q_ready()` 恒 UNKNOWN。

### 🔴 `onfield()` 返回 bool，不是字符串

```
✅ onfield('钟离') == True
❌ onfield() == '钟离'          # 恒 False
```

### 🔴 `has_nearby_enemy()` 在真机上可能不准

它读的是世界实体检测器（`bgi_world`）的**实际检出**。**真机实测它检出的是
紫色电气水晶，不是 `enemy_identify`**。也就是说依赖它的 trigger **可能压根不触发**。

这是**感知层的缺口**，不是配置写错了。症状：连招日志里一条
`COMBO_STARTED` 都没有，但也没有任何报错。

---

## 8 · 没写 `combos.toml` 会怎样

**什么都不发生，引擎退回纯反射模式。**

- 文件不存在 = 正常配置，不是故障。决策层只剩规则。
- 发行的那份 `autopilot/combos/combos.toml` 里**每条都是 `enabled = false`** ——
  连招要玩家自己认账（队伍顺序、wait 时间），引擎不替你决定。
- **文件存在但写坏了** = 故障，**启动失败并指名道姓报错**（文件 + 行号 +
  连招名 + 字段名）。不会偷偷跳过你的连招。

这两条必须分开：如果「文件坏了」也被当成「没文件」，你会看到「引擎不认我的
连招」而屏幕和日志上什么都没有 —— 最难查的一类静默失效。

---

## 9 · 诊断

### 事件（EventBus / YUI）

| code | 含义 |
|---|---|
| `COMBO_STARTED` | 一条连招启动了 |
| `COMBO_ABORTED` | 连招被中断（`context` 里有原因） |
| `COMBO_STEP_UNKNOWN` | 某一步因为「不知道谁在场 / 几号位」被跳过 |
| `COMBO_EVAL_ERROR` | 某条 trigger 或某步的 `when` 求值失败 |

### 运行时统计

```python
policy.stats()["combos"]
# {'configured': 1, 'enabled': 0, 'started': 0, 'finished': 0,
#  'aborted': 0, 'last_reason': '', 'progress': {...}}

policy.combo_runner.progress()
# {'active': True, 'combo': '钟夜纳绫', 'step': 3, 'steps': 4,
#  'phase': 'WAIT_SWITCH', 'repeat_left': 0, 'switch_target': '纳西妲',
#  'elapsed_ms': 1400}
```

`phase` 的取值：`STEP`（准备出队）、`WAIT_SWITCH`（已发切人，等记账器更新）、
`SETTLE`（等 `settle_ms`）。

### 规则命中过程

`decide()` 返回完整的 `DecisionTrace`（每条规则的 MATCHED / SKIPPED / ERRORED
与原因），供 `rule_test` / `replay` / `policy_diagnose` 用。

⚠️ **`decide()` 完全不知道连招存在** —— 连招只在 `poll()` 里被问。
这样离线工具「拿一帧状态跑一次 decide」这个契约才成立。

---

## 10 · 已知的真机缺口（写连招前先看一眼）

| 缺口 | 影响 |
|---|---|
| `has_nearby_enemy()` 检出的是紫色电气水晶 | 依赖它的 trigger 可能不触发 |
| 编队名册要 ~20 秒才填满 | 那段时间连招的每一步都被跳过 |
| `e_ready()` 的特征被背景主导（r ≈ 0.88） | 判出来的是「场景亮不亮」，不是「E 好不好」 |
| 角色**阵亡自动切人**引擎不知道 | 切人记账会停在旧值；症状是反复按同一个数字键，**无破坏性** |
| 中途换队伍 | 同上，名册可能短暂挂着旧队的名字 |
| 执行层对 `NORMAL_ATTACK` 有 `min_gap_ms = 300` | 200ms 一 tick 下 `repeat = 5` 大约有一半会被执行层跳过（**这是对的**） |

---

## 11 · 一句话速查

```
写在 autopilot/combos/combos.toml
├── enabled = false 时必须写 note
├── name 全局唯一
├── SWITCH_CHARACTER 必须写 char
├── onfield('X') 返回 bool，写 == True 不写 == 'X'
├── 名册没填满时那一步会被跳过（不是猜）
├── 想立刻可用就手填 keymap.toml 的 [char_slots]
├── priority < break_below_priority 的规则能打断你
├── 用户一按键 / F12 / 感知不健康 -> 立刻中断
└── 循环不用写 while：走完 + 冷却到期 + trigger 还成立 = 再来一遍
```
