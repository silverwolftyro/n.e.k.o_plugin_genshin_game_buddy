# autopilot 逻辑地图

> 生成方式：只读盘点（glob / grep / AST 检索），未执行任何模块。
> 路径基准：`D:\workspace\N.E.K.O\NEKO-genshin\N.E.K.O-main\N.E.K.O-main\plugin\plugins\genshin_game_buddy\autopilot\`
> 日期：2026-09-28

---

## 一、目录总览

```
autopilot/
├── contracts/          三层共用的数据契约（frozen dataclass + JSON Schema）。谁都不依赖谁，只被依赖。
├── perception/         截图 → StatePacket。YOLO 走 ONNX + onnxruntime（不 import ultralytics，AGPL 规避）。
├── policy/             StatePacket → IntentPacket。规则树 + DSL 求值。只依赖 contracts + state。
├── actuator/           IntentPacket → 真实按键。ctypes SendInput + traits + 校验 + 人类化 + 安全闸。
├── core/               tick 状态机 + 事件总线 + 统计 + WorldState 存储。只调三层，不做业务。
├── ui/                 双轨数据通道。协议层（schema/protocol）+ 读写两侧 + overlay（tkinter）。
├── config/             pydantic 配置模型 + defaults.toml / keymap.toml。加载期响亮失败。
├── rules/              决策规则 TOML（战斗 / 背包 / 导航）。
├── tools/              16 个可执行 CLI。离线工具不依赖实机资源。
├── state.py            WorldState 不可变快照 + WorldStateStore。core 与 policy 的共同底座。
├── run/                ★ 运行时产物（非源码）：快照 JSONL / 事件流。
└── README.md
```

### 逐文件清单

| 文件 | 行数 | 期数 | 职责 |
|---|---|---|---|
| `__init__.py` | — | 基础设施 | 包标记 |
| `state.py` | 264 | 1 | WorldState 快照 / WorldStateStore / ConnectionState |
| **contracts/** | | | |
| `contracts/__init__.py` | — | 1 | 包导出 |
| `contracts/state_packet.py` | 358 | 1 | StatePacket + 7 个子结构 + SCENE_IDS 权威场景枚举 |
| `contracts/intent_packet.py` | 290 | 1 | IntentPacket + ActionType（动作白名单）+ IntentTarget/Constraints |
| `contracts/events.py` | 213 | 1 | Event + Severity + EventCategory + 若干工厂方法 |
| `contracts/schemas/*.json` | — | 1 | 两份 JSON Schema（声明式，非 Python） |
| **perception/** | | | |
| `perception/capturer.py` | 300+ | 1 | 三通道截图（dxcam → bitblt → mss）+ RLock + 窗口查找 |
| `perception/builder.py` | 380+ | 1 | 场景路由：融合 YOLO/模板/OCR → StatePacket |
| `perception/template.py` | 300+ | 1 | 模板匹配 + NMS + StableValueAccumulator 抗抖 |
| `perception/roi_expr.py` | 150+ | 1 | ROI 表达式（ast 白名单，非手写 tokenizer） |
| `perception/screen_region.py` | 90 | 1 | ImageRegion 链式裁剪 + 坐标回推 |
| `perception/yolo.py` | 200+ | 1 | ONNX Runtime 推理（letterbox + NMS + 稳定性自检） |
| `perception/ocr.py` | 180+ | 1 | OCR 封装 + TriState 三态 |
| `perception/debug_overlay.py` | 200+ | 1 | 截图上画框 + 分数（仅被 autopilot_demo 使用） |
| `perception/__init__.py` | — | 1 | 包导出（含 DebugOverlay） |
| `perception/models/.gitkeep` | 0 | 1 | 放 ONNX 的空目录 |
| `perception/templates/.gitkeep` | 0 | 1 | 放模板 PNG 的空目录 |
| **policy/** | | | |
| `policy/base.py` | 120 | 3 | PolicySource Protocol（唯一对外契约）+ IntentSourceKind |
| `policy/factory.py` | 180 | 3 | 唯一决定用哪个实现的入口（注册表 + 惰性导入） |
| `policy/context.py` | 440 | 3 | DSL 函数集（21 个）+ UnknownValue 三态 + PolicySignals |
| `policy/dsl.py` | 340 | 3 | ast 白名单条件求值器 + RuleEvalError |
| `policy/rule_loader.py` | 520 | 3 | TOML 规则加载 + 17 类加载期校验 |
| `policy/local_tree.py` | 300+ | 3 | 规则树：priority 升序、命中即停（核心 3） |
| `policy/merger.py` | 300+ | 3 | 本地/云端仲裁 + 签名先验 + 不降级契约 |
| `policy/__init__.py` | 55 | 3 | 只导出 4 项契约，实现零泄漏 |
| **actuator/** | | | |
| `actuator/simulator.py` | 900+ | 2 | ctypes SendInput + 返回值校验 + VK/KeyId + release_all |
| `actuator/executor.py` | 470 | 2 | 编排：校验→安全闸→键位→人类化→执行→finally 松键 |
| `actuator/traits.py` | 300+ | 2 | ActionTraits 表（21 个动作全覆盖）+ 场景常量 |
| `actuator/validator.py` | 230 | 2 | IntentValidator：去重/TTL/漂移/场景/白名单 |
| `actuator/humanizer.py` | 220 | 2 | hold/delay/heading/click 抖动（默认开） |
| `actuator/safety.py` | 300+ | 2 | F12 急停 / 焦点 / 黑名单 / 频率 / 熔断 / 上限 |
| `actuator/keymap.py` | 230 | 2 | 键位配置加载/保存（三级解耦 + 加载期严格校验） |
| `actuator/__init__.py` | 100+ | 2 | 包导出 |
| **core/** | | | |
| `core/loop.py` | 700+ | 4 | CoreLoop：tick 状态机 + F12 独立监视线程 + 优雅退出 |
| `core/runtime.py` | 230 | 4 | RuntimeState 五态 + RuntimeContext + close_resources |
| `core/events.py` | 300+ | 4 | EventBus（有界队列 + 后台分发 + 去重）+ LogSubscriber |
| `core/stats.py` | 250 | 4 | LoopStats 内存聚合 + snapshot + health |
| `core/__init__.py` | 40 | 4 | 只导出 CoreLoop / RuntimeState |
| **ui/** | | | |
| `ui/schema.py` | 265 | 5·A 轨 | 三个 pydantic 模型 + 枚举（**不含传输**） |
| `ui/protocol.py` | 371 | 5·A 轨 | 纯基础设施：路径/原子写/追写/轮转/stat 短路。**零 class** |
| `ui/writer.py` | 265 | 5·A 轨 | ChannelWriter |
| `ui/reader.py` | 170 | 5·A 轨 | ChannelReader |
| `ui/commands.py` | 155 | 5·A 轨 | CommandSender + CommandPoller |
| `ui/__init__.py` | 20 | 5·A 轨 | 空（无 IO） |
| `ui/overlay/__init__.py` | 11 | 5·A 轨 | 空 |
| `ui/overlay/app.py` | 211 | 5·A 轨 | tkinter 主循环（10Hz + 内容指纹） |
| `ui/overlay/layout.py` | 198 | 5·A 轨 | 位置计算 + PanelModel（**不 import tkinter**） |
| `ui/overlay/interaction.py` | 139 | 5·A 轨 | 拖动（3px 阈值）/ 方向键微调 / 按钮 |
| `ui/overlay/theme.py` | 106 | 5·A 轨 | 颜色/字体/尺寸常量（全硬编码） |
| **config/** | | | |
| `config/loader.py` | 330 | 1 | AutopilotConfig / LoopConfig 等 pydantic 模型 + validate_config |
| `config/defaults.toml` | 175 | 1 | 默认配置（模板声明已注释） |
| `config/keymap.toml` | 43 | 2 | 21 个动作的默认键位 |
| `config/overlay_pos.json` | — | 5·A 轨 | ★ overlay 位置持久化（运行时产物） |
| `config/__init__.py` | — | 1 | 包导出 |
| **rules/** | | | |
| `rules/genshin_combat.toml` | 172 | 3 | 8 条战斗规则（含注释掉的 has_enemy 占位） |
| `rules/genshin_bag.toml` | 118 | 3 | 7 条背包规则（4 条 enabled=false 占位） |
| `rules/genshin_nav.toml` | 100 | 3 | 4 条导航规则 |
| **tools/**（见第三节） | | | |
| `run/` | — | 5·A 轨 | ★ 运行时产物 |

---

## 二、分层图

```
                    ┌─────────────────────────────────┐
                    │  contracts  （零内部依赖）        │
                    │  StatePacket / IntentPacket /    │
                    │  Event / ActionType / SCENE_IDS  │
                    └─────────────────────────────────┘
                       ▲          ▲          ▲
          ┌────────────┘          │          └────────────┐
          │                       │                       │
  ┌───────┴────────┐      ┌───────┴────────┐     ┌────────┴───────┐
  │   perception   │      │     policy     │     │    actuator    │
  │  截图/识别      │─────>│  规则/DSL      │────>│  按键/安全     │
  │  (无出边依赖)  │State │  (无出边依赖)  │Intent│  (无出边依赖)  │
  └────────────────┘Packet└────────────────┘Packet└────────────────┘
          ▲                       ▲                        ▲
          │                       │                        │
          └───────────────┬───────┴────────────────────────┘
                          │
                  ┌───────┴────────┐         ┌──────────────────┐
                  │      core      │         │   policy.merger  │
                  │  tick 状态机    │────────>│  本地/云端仲裁     │
                  │  事件总线/统计  │         └──────────────────┘
                  └───────┬────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
      ┌───────┴────────┐     ┌────────┴───────┐
      │  core.state /  │     │      ui       │
      │  core.events   │     │  A 轨数据通道   │
      │  core.stats    │     │  (无出边依赖)  │
      └───────┬────────┘     └────────┬───────┘
              │                       │
              └───────────┬───────────┘
                          │
                  ┌───────┴────────┐
                  │  config / rules │  （被上层引用，非层）
                  └───────┬────────┘
                          │
                  ┌───────┴────────┐
                  │      tools     │  ← 唯一可以横跨所有层的
                  └────────────────┘
```

### 依赖规则（已用 AST 静态扫描验证）

| 规则 | 状态 |
|---|---|
| `contracts` 不依赖任何 autopilot 内部模块 | ✅ |
| `policy` 不 import `perception` / `actuator` / `tools` | ✅ 全包零命中 |
| `actuator` 不 import `perception` / `policy` | ✅ |
| `perception` 不 import `policy` / `actuator` | ✅ |
| `ui` 不 import 任何 runtime 模块（只依赖 `ui.schema`） | ✅ |
| `core.loop` → `policy.merger`（唯一跨层调用，**已注释说明**） | ⚠ 见第五节 |
| 全包无 `ultralytics` import | ✅ 零命中 |

---

## 三、入口清单

所有含 `if __name__ == "__main__"` 的模块（16 个）：

| 模块 | 用途 | 参数摘要 | 依赖三层 |
|---|---|---|---|
| `tools/run_loop.py` | **主入口**，跑完整 tick 循环 | `--title --rules-dir --dry-run --live --tick-rate --stats-interval --duration --log-level` | 全部四层 |
| `tools/overlay.py` | A 轨独立浮窗（不发按键） | `--x --y --alpha --reset-pos --log-level` | ui |
| `tools/loop_diagnose.py` | 依赖体检（不启动循环） | `--title --rules-dir --skip-capture --json -v` | 全部（只构造不启动） |
| `tools/autopilot_demo.py` | 感知循环预览（叠加层） | `--title --interval --save --no-window` | perception |
| `tools/capture_demo.py` | 只截图 | `--title --interval --save --no-window` | perception |
| `tools/template_demo.py` | 模板匹配演示 | `--title --template` | perception |
| `tools/input_demo.py` | 输入模拟演示（默认干跑） | `--title --dry-run --live --tick-rate --skip-release --countdown` | actuator |
| `tools/keymap_editor.py` | 交互式改键位 | `--show --check --reset --set ACTION=KEY --out` | actuator |
| `tools/rule_test.py` | 单状态包喂规则，出命中原因 | `--state --rules-dir --repeat --json` | policy |
| `tools/replay.py` | JSONL 状态序列回放 | `--input --output --summary --quiet` | policy |
| `tools/policy_diagnose.py` | 策略实现/规则数/分布诊断 | `--rules-dir --kind --json --list-rules` | policy |
| `tools/check_env.py` | 环境自检（CUDA/依赖/窗口/通道） | `--title -v` | perception |
| `tools/check_dataset.py` | YOLO 数据集质量检查 | `--data` | 训练工具链 |
| `tools/train_yolo.py` | 训练 + 导出 ONNX（**唯一允许 import ultralytics**） | `--data --epochs --batch --device` | 工具链 |
| `tools/eval_yolo.py` | ONNX 评估（mAP + 混淆矩阵） | `--model --data` | 工具链 |
| `tools/convert_to_onnx.py` | 单独导出 ONNX | `--weights --opset` | 工具链 |

非入口但被 import 的私有模块：
`tools/_bootstrap.py`（sys.path 引导 + 终端着色）、`tools/_stack.py`（三层组装）。

---

## 四、状态 / 数据流

### 4.1 一次 tick 的对象与文件

```
run_loop.main()
  │
  ├─ ChannelWriter.write_snapshot()   ← 后台线程，每 200ms
  │    └─ _ui_snapshot(ctx, loop, run_id, ...)
  │         ├─ ctx.stats.snapshot()          [内存]
  │         ├─ loop.state.value              [内存]
  │         └─ ctx.capturer.get_active_channel()
  │    └─ atomic_write_text(run/state_snapshot.json)   ★覆盖写（tmp+rename）
  │
  ├─ loop.start() ─────────────────► CoreLoop 工作线程（daemon）
  │    │
  │    └─ 每 tick：_tick()
  │         │
  │         ├─ ① [PANIC/状态检查 ×4]
  │         │     _can_act("tick 开头" / "感知后" / "决策后")
  │         │
  │         ├─ ② 感知  _perceive()
  │         │     ├─ ctx.capturer.grab() ──► CapturedFrame
  │         │     │    └─ dxcam / bitblt / mss（RLock 保护）
  │         │     ├─ ctx.builder.build(frame) ──► StatePacket
  │         │     │    ├─ yolo.detect()        （ONNX Runtime）
  │         │     │    ├─ template.find_multi()（CV2 + NMS）
  │         │     │    ├─ ocr.recognize()
  │         │     │    └─ builder.drain_events() ──► [Event] ─┐
  │         │     ├─ builder.set_templates / set_roi（启动期一次）│
  │         │     └─ ctx.store.update(state) ──► WorldState   │
  │         │          （history 30 帧，state_version++）      │
  │         │                                                    │
  │         ├─ ③ 决策  ctx.policy.poll(world.current_state)      │
  │         │     └─ LocalPolicyTree.poll(state)               │
  │         │          ├─ ctx.bus? 否 → 内部 decide()           │
  │         │          ├─ 遍历 rules（priority 升序）          │
  │         │          │   ├─ matches_scene?                   │
  │         │          │   ├─ DslEvaluator.evaluate(when, PolicyContext)│
  │         │          │   │    └─ PolicyContext.hp()/q_ready()/… │
  │         │          │   └─ 命中即 break（核心 3）             │
  │         │          └─ IntentPacket.make(...)               │
  │         │     → IntentPacket | None（None = 空转，不执行）  │
  │         │                                                    │
  │         ├─ ④ 合并  merger.merge_intents(local, None, state)│
  │         │                                                    │
  │         └─ ⑤ 执行  ctx.executor.execute(intent, state)     │
  │               ├─ validator.validate(commit=True) ──► SeenIntentStore
  │               ├─ guard.check()      F12/焦点/黑名单/频率/熔断
  │               ├─ resolve_key() → KeyId → VK
  │               ├─ humanizer.hold_ms / delay_ms（抖动）
  │               ├─ sim.key_press() ──► ctypes SendInput
  │               │    └─ 校验 sent == count，否则 InputInjectionError
  │               ├─ sim.history.append()  [内存环形，500 条]
  │               └─ finally: sim.release_all()   ★无条件松键
  │                                                           │
  │    事件出口 ────────────────────────────────────────────────┘
  │      ctx.bus.publish(Event)
  │        ├─ EventBus 分发线程 ──► LogSubscriber（→ stderr）
  │        └─ EventBus 分发线程 ──► _ui_on_event（run_loop 订阅）
  │             └─ writer.write_event(StateEvent)
  │                  └─ append_jsonl(run/<run_id>.stream.jsonl)  ★追加写
  │
  └─ 命令通道  _ui_pump 每 200ms
       └─ cmd_poller.poll() ──► run/commands.jsonl   ★只 stat，无变化不读
            └─ STOP/KILL/PAUSE/RESUME → ui_state 标志 → 主循环消费
```

### 4.2 文件读写一览

| 文件 | 模式 | 写入方 | 读取方 | 频率 |
|---|---|---|---|---|
| `run/run_current.txt` | 原子覆盖 | ChannelWriter.__init__ | ChannelReader.run_id() | 每次启动一次 |
| `run/state_snapshot.json` | **原子覆盖**（tmp+rename） | ChannelWriter.write_snapshot | ChannelReader.read_snapshot | 200ms |
| `run/<run_id>.stream.jsonl` | **追加**（O_APPEND），>5MB 轮转 `.1` | ChannelWriter.write_event | ChannelReader.poll_events | 事件驱动 |
| `run/commands.jsonl` | 追加，读完清空 | CommandSender.send | CommandPoller.poll | 命令时 |
| `config/overlay_pos.json` | 直接覆盖 | layout.save_pos | layout.load_pos | 拖动/微调时 |
| `config/keymap.toml` | 直接覆盖 | keymap.save_keymap | keymap.load_keymap | 手动 |

### 4.3 三个省电闸

```
protocol.stamp()          → (st_size, st_mtime_ns) 比对
  └─ 未变 → ChannelReader.read_snapshot / poll_events 立刻返回，0 次 read
       └─ 未变 → app.tick() 的 content_key 相同，不碰任何 tk 控件
```

只比 `st_size` 不够 —— 两个等长快照来回切换时 size 不变会永远读不到新内容，所以必须带 `mtime_ns`。

---

## 五、可疑文件 / 存疑点

> 只列事实（文件 + 行号），不给处理建议。

### 5.1 疑似死代码

| 位置 | 事实 |
|---|---|
| `policy/merger.py` | `validate_cloud_intent()` 定义完整（含签名先验四道校验），全仓库唯一调用点是同文件的 `merge_intents()`。无外部调用者、无测试引用。 |
| `ui/schema.py:155` `EventCode` | 定义了 12 个 code。`BREAKER_OPENED` / `BREAKER_CLOSED` / `CHANNEL_DEGRADED` / `CHANNEL_RECOVERED` / `PANIC_CLEARED` 在 run_loop 的事件订阅里没有专门分支（统一走 `_ui_event_to_state` 透传）。 |
| `ui/schema.py:213` `CommandName.NUDGE` | `Command.supported` 属性显式把 NUDGE 判为不支持。`NUDGE` 枚举成员本身无产生方。 |
| `perception/models/`、`perception/templates/` | 仅 `.gitkeep`。`defaults.toml` 里 3 个模板声明已全部注释，`yolo.enabled=false`。即整条识别链路当前是空转的。 |

### 5.2 疑似遗留 / 旧版

| 位置 | 事实 |
|---|---|
| `ui/overlay/app.py:206` `build()` | 薄封装，供 `tools/overlay.py` 调用。当前 `tools/overlay.py` 也是直接调 `build()` —— 两层包装，功能上可合并。 |
| `ui/overlay/layout.py:194` `_state_text()` | 已改为 `_PHASE_LABEL.get()` 的薄封装，但 `build_panel()` 不再调用它（改为内联 `_PHASE_LABEL.get(state, state)`）。保留仅为外部兼容，文件内无调用点。 |
| `tools/check_env.py` vs `tools/loop_diagnose.py` | 两者都检查：Python 版本 / 依赖 / 窗口存在性 / 截图通道。`loop_diagnose`（第 4 期）覆盖了 `check_env`（第 1 期）的大部分职责并额外加了 policy / actuator / 急停三层。功能重叠。 |
| `tools/autopilot_demo.py` vs `tools/run_loop.py` | 两者都做「Capturer + StateBuilder 组装 + 主循环」。`autopilot_demo` 是第 1 期的感知预览（带叠加层、不接 policy/executor），`run_loop` 是第 4 期的全链路。组装代码（`tools/_stack.py`）是后来抽出来给 `run_loop` 用的，`autopilot_demo.py` 没有复用它。 |

### 5.3 疑似重复

| 位置 | 事实 |
|---|---|
| `tools/capture_demo.py` / `tools/autopilot_demo.py` / `tools/template_demo.py` | 三者都有「建 Capturer → open → grab → close」的前 20 行，各自独立复制，未共用 `tools/_stack.py`。 |
| `run_loop.py:255` vs `loop_diagnose.py:~130` | 两处各自实现「LoopStats.health 中文长句 → 协议枚举」的归并逻辑。`run_loop.py` 里叫 `_ui_health()`，`loop_diagnose.py` 里是内联的 `_check_loop_cfg()`。 |

### 5.4 顶层 IO

| 检查项 | 结果 |
|---|---|
| 模块顶层 `mkdir` / `write_text` / `open` / `stat` | **零命中**。全部在函数/方法体内。 |
| 模块顶层 `Path(...)` 常量（纯计算，无 IO） | `config/loader.py:26,28`、`ui/protocol.py:31`、`policy/local_tree.py:53`、`actuator/keymap.py:32`、`tools/_bootstrap.py:14`、`tools/loop_diagnose.py:39` |
| 模块顶层网络请求 | 零命中 |
| 模块顶层 `logging.basicConfig` | 零命中（在 `setup_logging()` 内） |

### 5.5 跨层调用（唯一一处）

| 位置 | 事实 |
|---|---|
| `core/loop.py` `_merge()` | `from ..policy.merger import merge_intents`，且被 `try/except ImportError` 包住，失败时 `return local`。core（本应只调三层的公开方法）直接引用了 policy 的具体模块。 |

### 5.6 其它事实

| 位置 | 事实 |
|---|---|
| `run/` | 含 `state_snapshot.json` / `run_current.txt` / `20260928_190226.stream.jsonl`，是 `run_loop` 一次真实运行的产物，落在源码目录内。 |
| `config/overlay_pos.json` | overlay 位置持久化文件，同样是运行时产物，落在源码目录内。 |
| `rules/genshin_combat.toml:145-172` | 3 段被整体注释的 `[[rules]]` 占位（has_enemy 版本）。因未知函数会在加载期报错，当前用 `when = "false"` 注释形态保存。 |
| `actuator/traits.py` | `SCENE_*` 常量从 `contracts.state_packet.SCENE_IDS` 派生（不是各写一份）。 |

---

## 六、周边文件（autopilot 目录之外）

### 6.1 `D:\workspace\N.E.K.O\NEKO-genshin\` 根目录下的 `.py`

全部是本次开发过程中我写的**离线验证脚本**，不属于交付物：

| 文件 | 行数 | 是什么 |
|---|---|---|
| `_verify_actuator.py` | 452 | 第 2 期 actuator 13 段干跑验证（假三层） |
| `_verify_backend.py` | 234 | backend=None / SendInput 返回值 / 诊断器验证 |
| `_verify_core.py` | 526 | 第 4 期初版 loop 验证（已被 core1/core2 取代） |
| `_verify_core1.py` | 364 | core/stats.py + core/events.py 验证 |
| `_verify_core2.py` | 565 | 重写后 loop.py 验证（11 段） |
| `_verify_keymap.py` | 147 | 键位加载器 17 类校验验证 |
| `_verify_policy1.py` | 317 | base/context/factory/__init__ 8 段验证 |
| `_verify_policy2.py` | 376 | dsl + rule_loader 验证 |
| `_verify_policy3.py` | 539 | local_tree + merger 11 段验证 |
| `_verify_policy4.py` | 274 | 三个 rules/*.toml 加载与行为验证 |
| `_verify_policy5.py` | 360 | 三个离线工具 CLI 端到端验证 |
| `_verify_tools.py` | 257 | 第 1 期 8 个工具 CLI 验证 |
| `_import_scan.py` | 34 | 递归 import 全 autopilot 模块，统计可导入数 |
| `_make_mock_states.py` | 107 | 生成 200 帧 mock 状态序列 |
| `_debug_integrity.py` | 61 | 一次性 ctypes 完整性级别排查脚本 |
| `_debug_integrity.py` 之外 | — | `_debug_integrity.py` 已在表内 |

（另有 `_verify_policy*.py` 等 15 个，总计约 5376 行。）

### 6.2 `mock_states.jsonl`

| 项 | 值 |
|---|---|
| 路径 | `D:\workspace\N.E.K.O\NEKO-genshin\N.E.K-O-main\mock_states.jsonl` |
| ⚠ 位置 | **在误建目录 `N.E.K-O-main\` 里，不在真仓库** |
| 大小 | 约 160 KB / 200 行 |
| 内容 | 200 帧 `StatePacket.to_dict()` 的 JSONL，覆盖战斗（满血/掉血/危急）、背包、地图、定位 GOOD/DEGRADED/LOST、场景 UNKNOWN、`perception_ok=False`。由 `_make_mock_states.py` 生成，供 `replay.py` 离线回放决策。 |

### 6.3 `.build-tmp/`

| 项 | 值 |
|---|---|
| 路径 | `D:\workspace\N.E.K.O\NEKO-genshin\N.E.K.O-main\.build-tmp\` **与** `D:\workspace\N.E.K.O\NEKO-genshin\N.E.K-O-main\.build-tmp\` |
| 状态 | **两个路径下都不存在**（glob 无匹配） |

### 6.4 误建目录 `D:\workspace\N.E.K.O\NEKO-genshin\N.E.K-O-main\`

⚠ **该目录不是工作仓库**，但里面有 autopilot 的**重复副本**（连字符 vs 圆点）：

| 文件 | 行数 | 与真仓库的差异 |
|---|---|---|
| `N.E.K-O-main/N.E.K.O-main/.../ui/protocol.py` | 182 | 真仓库 371 行。旧版，含 4 个业务 class |
| `N.E.K-O-main/N.E.K.O-main/.../ui/writer.py` | 82 | 真仓库 265 行。旧版，含 `RunChannel` |
| `N.E.K-O-main/N.E.K.O-main/.../ui/reader.py` | 83 | 真仓库 170 行。旧版，含 `UiReader` |
| `N.E.K-O-main/N.E.K-O-main/.../ui/commands.py` | 111 | 真仓库 155 行。旧版，含 `CommandDispatcher` |
| `N.E.K-O-main/N.E.K.O-main/.../tools/_stack.py` | 138 | 与真仓库同（`_stack.py` 一致） |
| `N.E.K-O-main/N.E.K.O-main/.../tools/overlay.py` | 72 | 与真仓库同 |

只有 `ui/` 4 个文件是真仓库的**旧版本**，`tools/` 2 个一致。真仓库
`D:\workspace\N.E.K.O\NEKO-genshin\N.E.K.O-main\`（圆点）才是有效工作目录。

---

## 七、盘点方法的局限

1. **未执行任何模块** —— 全部结论来自 glob / grep / AST 静态检索。「死代码」的判断依据是「全仓库无引用」，不能排除字符串反射式的动态引用。
2. **文件系统在本次盘点期间多次给出不一致结果** —— 同一 glob 前后返回不同的文件列表，`layout.py` / `app.py` / `theme.py` 中途一度「消失」又出现。上表采用最终一致的快照，但**个别文件的行数可能已变化**。
3. **`perception/` `actuator/` `policy/` 部分文件只读了 grep 索引，未逐行通读**，行数为索引回报值（以 `300+` 标注表示未精确核实）。
4. `run/` 下的文件是上一次真实运行的产物，可反推 run_loop 至少成功跑过一次，但**该次运行的完整日志不在仓库内**。
