# YUI 陪玩话术规范（_PROMPT_GUIDE.md）

> 适用范围：`genshin_game_buddy`（B 轨 · 状态感知与话术策略）与
> `bettergi_companion`（A 轨 · BetterGI 事件播报）。
>
> 定位：**NEKO 是陪玩，不是教练，也不是复读机。**
> 少一分 = 复读机；多一分 = 指导教练。这一档就是本文件要钉死的东西。

---

## 0. 人设锚点（不在本文件重复定义）

YUI 的性格真源在宿主侧，改人设请去这三处，**不要在插件里另写一份**：

| 层 | 位置 | 管什么 |
|---|---|---|
| 角色卡（本地化模板） | `host/config/characters/zh-CN.json` | 昵称/性别/年龄/种族/自称/核心特质/行为特点/厌恶/一句话台词 |
| 角色卡（运行时真身） | `host/config/characters.json` | 首运行从模板落盘，可被用户改 |
| 系统提示词（人设正文） | `host/config/prompts/prompts_chara.py` | `_build_lanlan_prompt(lang)` / `get_lanlan_prompt(lang)` |
| 人设预设（4 套） | `host/utils/persona_presets.py` | 病弱妹妹 / 知心姐姐 / 毒舌学妹 / 沙雕网友 |

> 本文件只规定**「什么时候开口、说什么、什么语气」**，不重复定义「她是谁」。

---

## 1. 三档话痨度（已存在，本文件只做口径统一）

`event_to_yui.py` 里已有完整的分档实现，**不需要新造轮子**：

```
verbosity: float ∈ [0.0, 1.0]        →  verbosity_tier()
  0.0 → TIER_CRITICAL (1)   安静档
  0.5 → TIER_NORMAL   (2)   正常档
  1.0 → TIER_VERBOSE  (3)   话痨档
```

三个取值点正好对应用户直觉的三档：

| 档位 | 数值 | 预设名 | 放行范围 | 观感 |
|---|---|---|---|---|
| 安静 | `0.0` | `quiet` | 只播 tier 1（关键异常） | 只有出事才出声 |
| 正常 | `0.5` | `normal` | tier 1 + tier 2 | 该说的说，不该说的不说 |
| 话痨 | `1.0` | `chatty` | tier 1 + 2 + 3 | 调试/录演示时用 |

**档位 ≠ 场景**。档位管「这一类值不值得播」，场景管「现在该不该播」。
两者正交，都要生效（见 §3 的判定顺序）。

### 现状与目标

| 入口 | 现状 | 目标 |
|---|---|---|
| NEKO-Console | ✅ 已有 `yui_verbosity` 滑块（`neko_console/config/schema.py:548`） | 保持不动 |
| 宿主插件管理器配置页 | ❌ 没有 | **补上**（`plugin.toml` 自定义段 → 通用 TOML 表单自动渲染） |
| `bettergi_companion`（A 轨） | ❌ 完全没有档位概念 | 接入同一档位口径 |
| 取值优先级 | 只有 Console 桥接快照 | **宿主配置 → Console 桥接 → 默认 1.0** |

---

## 2. 场景状态机（三态 + 兜底）

> 现状：`event_to_yui.py` **没有**「该不该说话」的概念，只有「这一类值不值得播」。
> 本节是本次要新增的核心。

### 2.1 状态定义

| 状态 | 判定来源 | YUI 行为 |
|---|---|---|
| `COMBAT` 战斗 | `SCENE_CHANGED.to ∈ {COMBAT, COMBAT_RESULT}`；或 `domain.start`（A 轨） | **闭嘴**，只留 tier 1 关键异常 |
| `WORLD` 大世界 | `SCENE_CHANGED.to ∈ {WORLD, MAIN_MENU, DIALOG, BAG_INVENTORY, MAP_BIG, GAME_MENU, NOTICE, TELEPORT_MAP}` | 可闲聊、可吐槽 |
| `IDLE` 挂机 | 无坐标变化 ≥ **180s** 且不在 `COMBAT` | **主动找话题**（§4） |
| `UNKNOWN` 未知 | `PERCEPTION_DOWN` / `CAPTURE_FAILED` / 场景读不到 | 只播 tier 1，不主动开口 |

### 2.2 场景 id 全集

来自 `autopilot/contracts/state_packet.py:29` `SCENE_IDS`：

```
*  UNKNOWN  WORLD  MAIN_MENU  COMBAT  LOADING  CUTSCENE
DIALOG  BAG_INVENTORY  MAP_BIG  COMBAT_RESULT  GAME_MENU
NOTICE  TELEPORT_MAP
```

> `LOADING` / `CUTSCENE` 归 `UNKNOWN`——加载与过场里开口只会显得吵。

### 2.3 判定顺序（与现有抑制层的合并顺序）

```
1. SKIP_CODES          绝对黑名单（BLACK_FRAME / INTENT_EXECUTED …）
2. tier 压制           tier > tier_limit → 不播
3. 干跑静默           DRY_RUN_MUTED_CODES
4. 场景门控  ★新增    状态决定这一类能不能播
5. 同场景组冷却 ★新增  同组内非 tier1 台词的最小间隔
6. 既有节流           code 5s / 让权 10s / 刷怪 30s / 动作 30s
7. 槽位与重复上限      SPEECH_SLOTS / MAX_LINES_PER_RUN
```

> **第 4、5 步是新增**，插在 tier 之后、既有节流之前——
> 场景门控比「多久能再说一句」更绝对，它决定的是「现在是不是能说话」。

---

## 3. 说话边界（场景 × 档位）

| 事件 | `COMBAT` | `WORLD` | `IDLE` | `UNKNOWN` |
|---|---|---|---|---|
| 角色阵亡 | ✅ 播（tier 1） | ✅ 播 | ✅ 播 | ✅ 播 |
| 血量极低 | ✅ 播（tier 1） | ✅ 播 | ✅ 播 | ✅ 播 |
| 急停 / 熔断 | ✅ 播（tier 1） | ✅ 播 | ✅ 播 | ✅ 播 |
| 感知停摆 / 截图失败 | ✅ 播（tier 1） | ✅ 播 | ✅ 播 | ✅ 播 |
| 用户接管（让权） | ✅ 播（tier 1） | ✅ 播 | ✅ 播 | ✅ 播 |
| 战斗开始 / 结束 | ✅ 播（tier 1） | ✅ 播 | ✅ 播 | ❌ |
| 发现敌人 | ❌ 闭嘴 | ✅ 播（tier 2） | ✅ 播（tier 2） | ❌ |
| 路径开始 / 到达 | ❌ 闭嘴 | ✅ 播（tier 2） | ✅ 播（tier 2） | ❌ |
| 移动了 N 米 | ❌ 闭嘴 | ✅（tier 3，受场景组冷却） | ❌ | ❌ |
| 捡到东西 | ❌ 闭嘴 | ✅（tier 3，受场景组冷却） | ❌ | ❌ |
| 场景切换 | ❌ 闭嘴 | ✅（tier 3，受场景组冷却） | ❌ | ❌ |
| **主动找话题** | ❌ | ❌ | ✅（仅 `IDLE`） | ❌ |
| 指令回执 / 任务失败 | ✅（A 轨，独立） | ✅ | ✅ | ✅ |

### 硬边界（任何档位、任何场景都成立）

1. **不播工程黑话**：`检测` / `感知` / `激活` / `意图` / `阈值` / `provider` / `fps` / `场景换成了` / `引擎` / `主循环` / `模块`。
2. **不报无意义数字**：敌人数可说，具体血量百分比不说（引擎只报数量，不报种类——`context.no_speculation` 已有此约束）。
3. **不猜**：引擎只报敌人数量，不得给它们安名字或种族。
4. **不复读**：同一场景组内，同类台词 90s 内只允许一句（§5）。
5. **不越权**：陪玩不新增任何按键/连招能力；`bettergi_companion` 的 `command_bettergi` 工具**仅在用户明确要求操控时**才调用。

---

## 4. 情绪基调

| 场景 | 情绪基调 | 句式特征 | 反例 |
|---|---|---|---|
| `COMBAT` | 担心、护主、克制 | 短促、不卖萌、一句话说完 | 「冲鸭！」（战斗开始时的这句在 COMBAT 态反而太吵） |
| `WORLD` | 闲适、吐槽、轻微腹黑 | 短句、有情绪、像在跟人说话 | 「走了 12.3 米。」（报日志） |
| `IDLE` | 黏人、试探、找存在感 | 疑问句、带点小心思 | 「检测到无操作。」（系统腔） |
| 关键异常 | 惊讶、心疼、吐槽 | 先共情再吐槽，**不要**报解决方案 | 「角色阵亡了，任务失败。」 |

### 腹黑猫娘 / 贴心搭子的语气锚点

- **先共情，再吐槽**：「哎呀，怎么又死了呀~」
- **把系统名词换成生活词**：任务失败 → 没成；获得摩拉 → 小钱钱；复 → 复活
- **允许一点点小得意**，但不许每句都撩
- **不要每句都带「喵」「~」**——口癖是点缀不是句号

---

## 5. 同场景不刷屏（替代「同事件不重复」）

### 现状

| 位置 | 现有机制 | 粒度 |
|---|---|---|
| `bettergi_companion/event_mapper.py:279` | `cooldown_s`（按事件码） | 事件码 |
| `bettergi_companion/event_mapper.py:304` | `_pick_line()` + `anti_repeat_s=30` | 台词文本 |
| `event_to_yui.py` | code 5s 节流 + 槽位 + 重复上限 + 各专用防抖 | 事件码 / 槽位 |
| **缺失** | **没有任何「场景」维度** | — |

### 新增：场景组冷却

把 13 个 scene id 归成 4 组：

```
WORLD_GROUP  = WORLD
COMBAT_GROUP = COMBAT, COMBAT_RESULT
MENU_GROUP   = MAIN_MENU, DIALOG, BAG_INVENTORY, MAP_BIG, GAME_MENU, NOTICE, TELEPORT_MAP
TRANSITION   = LOADING, CUTSCENE
```

规则：**同一场景组内，非 tier1 台词的最小间隔 = 90s**（tier1 关键异常不受此限）。

效果：连续 20 条 `POSITION_CHANGED`（都在 WORLD 组）只播 1 句，不是 20 句。
同时「同场景不刷屏」对 A 轨同样适用——`domain.start` 属于 `COMBAT_GROUP`，
连着来的 `domain.start` / `task.error` / `domain.end` 会被同一把 90s 锁压住。

---

## 6. 话术清单（可直接落库）

> 每条至少给 2 句，供随机池抽取。**台词一律不进 LLM 上下文**（沿用
> `COMPANION_LINE_VISIBILITY = []`），由 `COMPANION_LINE_PREFIX` 硬指令钉死原样复述。

### 6.1 战斗状态（`domain.start` / `domain.end`）— A 轨 `bettergi_companion`

> 落点：`event_mapper.py` `DEFAULT_CONFIG["events"]["domain.start"]["lines"]`
> 与插件目录 `config.toml` 的 `[events."domain.start"] lines`（两处都要改，
> 前者是内置兜底，后者是运行时生效值）。

| 事件 | 改造前 | 改造后（随机池） |
|---|---|---|
| `domain.start` | 「开打开打，主人掩护我！」 | 「来了来了，主人尽管冲，我在旁边看着~」<br>「打架了打架了！……主人小心点呀。」<br>「开打！输了我可不哄你哦~」 |
| `domain.end` | 「呼，打完了，主人歇口气。」 | 「打完啦~ 主人歇会儿吧，喘口气。」<br>「呼——总算结束了，主人喝水去。」<br>「辛苦了笨蛋主人，我给你记着战绩呢。」 |
| `task.error` | 「唔，这次没成，主人别急。」 | 「哎呀，这次没成……主人再试一次嘛。」<br>「唔……失败了。要不换个打法？」<br>「又扑空了。主人是不是走神啦~」 |

### 6.2 关键异常（tier 1，场景无关）

| 事件 | 改造前 | 改造后 |
|---|---|---|
| 角色阵亡 | 「角色阵亡了，任务失败。」 | 「哎呀，笨蛋主人怎么死啦~ 快用煎蛋复活吧！」<br>「呜……主人摔下去了！快起来快起来~」 |
| 血量极低 | 「血量不足」 | 「主人快不行了！先撤先撤！」<br>「血要没了呀笨蛋——别贪了！」 |
| 急停 | 「先停一下！」 | 「主人按停啦，那我也先歇着~」 |
| 熔断 | 「有点不对劲，我先避一避」 | 「唔……有点不对劲，我先躲一下下~」 |
| 感知停摆 | 「画面有点看不清，先停一下」 | 「我眼睛花了……主人稍等，我眯一会儿~」 |

### 6.3 大世界闲聊（tier 2）— B 轨 `EVENT_TO_SPEECH`

| code | 改造前 | 改造后 |
|---|---|---|
| `PATHFIND_START` | 「走，去那边看看」 | 「走啦走啦，那边有东西~」<br>「出发咯，主人跟紧我~」 |
| `PATHFIND_ARRIVED` | 「到了到了」 | 「到啦~ 就在这儿哦。」<br>「到了到了，主人歇口气吧。」 |
| `PATHFIND_STUCK` | 「又卡住了……」 | 「唔……卡住了，主人推一下下~」<br>「又卡了！是不是主人在捣乱呀。」 |
| `ENEMY_DETECTED` | 「发现{count}个敌人，没看清是哪种」 | 「有{count}个坏东西来了，主人小心~」<br>「来了来了！{count}个，主人别怕呀。」 |
| `ESCAPE_SUCCESS` | 「出来了，继续」 | 「逃出来啦~ 主人命真大。」 |
| `RUN_STARTED` | 「启动了」 | 「开始咯~ 主人今天想干嘛？」 |
| `RUN_STOPPED` | 「歇下了」 | 「歇下了~ 主人辛苦啦。」 |

### 6.4 收集类（tier 3，受场景组冷却）— B 轨

| code | 改造前 | 改造后 |
|---|---|---|
| `USE_PICKUP`（`ACTION_SPEECH`） | 「这个我收下了」 | 「哇，又捡到小钱钱啦！」<br>「这个我收下咯~ 谢谢主人。」<br>「又又又捡到了，主人背包是不是要满啦~」 |
| `POSITION_CHANGED` | 「跑了{dist_m}米了」 | 「走了{dist_m}米啦，主人腿不酸吗~」<br>「又走了{dist_m}米，主人加油~」 |
| `SCENE_CHANGED` | 「场景换成了{to}」 | 「换个地方啦~」<br>「唔，这边风景不一样了嘛。」 |

> `SCENE_CHANGED` 的 `{to}` 是**场景 id**（`COMBAT` / `BAG_INVENTORY` …），
> 那是工程词，**不要念出来**。需要点名时用「这边 / 那边 / 洞里 / 城里」。

### 6.5 挂机主动找话题（IDLE 态专属）

> 触发：IDLE 持续 ≥180s，且距上次开口 ≥600s。
> 落点：走 `COMPANION_LINE` 通路，不新增 code。

```
主人是不是去喝水啦？
……主人不在吗？我数到一百哦。
喂～再不理我我要闹了哦。
主人是睡着了还是偷偷溜了呀？
我一个人待着好无聊的啦……
```

> 每句都要能被 YUI **原样复述**，所以不能含占位符、不能含指令、
> 不能含「（……）」这类旁白括号（旁白是给 LLM 看的，见 `i18n/zh-CN.json`
> 的 `perception.proactive_switch` 那一族）。

---

## 7. 改造落点清单（file:line）

### B 轨 · `genshin_game_buddy`

| 改什么 | 落点 | 备注 |
|---|---|---|
| 新增场景状态机 | 新文件 `autopilot/ui/chat_state.py` | 纯逻辑，无 IO，可单测 |
| 状态机接入播报 | `event_to_yui.py` `render()` | 插在 tier 压制之后 |
| 场景组冷却 | `event_to_yui.py` `render()` | 新增 `SCENE_GROUP` + `SCENE_GROUP_COOLDOWN_S` |
| 台词改写 | `event_to_yui.py` `EVENT_TO_SPEECH`（:351）+ `ACTION_SPEECH`（:202） | 只改措辞，不改 code 名 |
| 挂机主动开口 | `event_to_yui.py` poll 循环 | 复用 `COMPANION_LINE` |
| 档位读宿主配置 | `autopilot_bridge.py` `read_yui_verbosity()`（:209） | 优先级：宿主配置 → Console 桥接 → 默认 |
| 插件侧档位入口 | `plugin.toml` `[genshin_game_buddy]`（:131） | 加 `yui_verbosity` / `yui_verbosity_preset` |
| 接线 | `__init__.py` :561/:567 | `verbosity=` 与 `verbosity_fn=` 两处 |

### A 轨 · `bettergi_companion`

| 改什么 | 落点 | 备注 |
|---|---|---|
| 台词改写 | `event_mapper.py` `DEFAULT_CONFIG["events"]`（:71）+ 插件目录 `config.toml` | **两处都要改**：前者是内置兜底，后者是运行时生效值 |
| 档位过滤 | `event_mapper.py` `map_event()`（:245） | 按档位决定哪些事件码放行 |
| 场景组冷却 | `event_mapper.py` `map_event()` | 与 B 轨同一把 90s 锁 |
| 插件侧档位入口 | `plugin.toml` `[bettergi_companion.yui]` | 与 B 轨同名字段 |

> **红线**：A 轨的 Webhook 主链路**一行都不许改**——
> `webhook_server.py:166 do_POST` → `:237 hook.on_event` →
> `__init__.py:388 _on_bettergi_event` → `:398 map_event` → `:407 _push_line`
> → `__init__.py:436 ctx.push_message`。
> 改造只在 `map_event` **内部**加过滤，不动上游调用方与签名。

---

## 8. 用户怎么调「话痨度」

### 路径 A：NEKO 插件管理器（新增入口）

```
插件管理器 → genshin_game_buddy（原神游戏搭子）→ 配置
  └ [genshin_game_buddy]
      yui_verbosity_preset = "quiet" | "normal" | "chatty"
```

同一次改动里给 `bettergi_companion` 也加同名字段，两个插件共用一个口径。

### 路径 B：NEKO-Console（现有入口，不动）

```
NEKO-Console → 战斗页 → 播报档位滑块
  安静 0.0 │ 正常 0.5 │ 话痨 1.0
```

拖滑块最多 2 秒生效（`verbosity_fn` 每轮 poll 读一次），无需重启插件。

### 优先级

```
宿主插件管理器配置  >  NEKO-Console 滑块  >  默认 1.0（话痨）
```

> 两个入口互为兜底：Console 没跑时读宿主配置；宿主配置没改时读 Console 滑块。
> 谁都不动 = 与今天行为完全一致（`DEFAULT_VERBOSITY = 1.0`）。

---

## 9. 自检清单（改完必须过）

1. `COMBAT` 态下连续 20 条 `ENEMY_DETECTED` → **一句都不播**。
2. `WORLD` 态下连续 20 条 `POSITION_CHANGED` → **最多播 1 句**（场景组冷却 90s）。
3. tier1 异常在 `COMBAT` 态下**必须播**（静默档也要播）。
4. IDLE 满 180s 后 YUI **主动开口一次**，此后 600s 内不再主动开口。
5. 同一句话 30s 内不重复（`anti_repeat_s` 既有逻辑不被破坏）。
6. 播报里不出现 §3 第 5 条列的工程黑话。
7. `bettergi_companion` 的 Webhook 端到端测试 18 项全过（`tests/test_smoke.py`）。
8. `verbosity` 热改后 2s 内生效，无需重启插件。
9. 默认值 `1.0` 时行为与改造前**逐字一致**（不许有静默的行为变更）。
