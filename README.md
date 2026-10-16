# 原神游戏搭子

让猫娘知道你现在在玩哪款游戏，然后换用那款游戏的语境与术语陪你聊。

A companion plugin that switches game context and terminology on demand.

## 它做什么 / What it does

**术语陪聊（L1，这是一直以来的主线）**

- 登记当前在玩的游戏（原神、鸣潮，可自行扩展）
- 加载该游戏的术语库（`terms/<game>/`，四个维度：core / slang / characters / systems）
- 把一段小语境注入宿主 LLM 上下文（`visibility=[]` + `ai_behavior="read"`，不打断对话）
- 模型按需查术语（`lookup_game_term`），且严格限定在当前游戏内
- 切换游戏时卸载旧索引、加载新索引

**游戏陪玩（L2/L3，3.0 起）**

⚠️ **2026-10-11 更正**：本节以前写的是「不截图、不操作游戏、不做决策」，
那在 3.0 之后已经完全不成立了。以下是实际有的能力：

- **会话自动建立**：后台探针按窗口标题找到游戏窗口（`window_probe.py`，
  纯 ctypes，零第三方依赖），登记会话 —— 不需要用户手动 `set_game`
- **实时感知**：抓游戏窗口画面，四条通道依次降级
  `WGC → BitBlt → DXcam → mss`（见下）
- **世界实体检测**：YOLO 检出附近的敌人 / 宝箱 / 矿 / 掉落物（DirectML）
- **规则决策**：`autopilot/rules/*.toml` 里的 24 条规则按优先级命中即停
- **按键执行**：注入键盘/鼠标（战斗、闪避、拾取、导航）
- **事件播报**：引擎事件流经 `event_to_yui.py` 翻译成台词推给 YUI，
  让它主动开口（「引擎起来了」「有怪！3个，小心点」「这个我收下了」）

## 架构 / Architecture

```
N.E.K.O 宿主 (Electron)
  └─ 插件进程
       ├─ __init__.py            会话探针、术语索引、event_to_yui 追尾线程
       └─ 后台线程读 <plugin>/autopilot/_runtime/*.stream.jsonl
                ↑ 由引擎写入
引擎 run_loop（独立 Python 进程，venv）
  core/loop.py       5 Hz 主循环：感知 → 决策 → 执行
  perception/        capturer(WGC/BitBlt/DXcam/mss) + yolo + 模板匹配 + 定位
  policy/            rule_loader → dsl → local_tree（规则命中即停）
  actuator/          traits → safety（频率/熔断/焦点）→ executor（SendInput）
```

### 截图通道（2026-10-11 定序）

| 顺序 | 通道 | 说明 |
|---|---|---|
| 1 | **WGC** | Windows Graphics Capture，走 DWM 合成管道。硬件加速全屏下**唯一**能用的，零注册表改动。依赖 `windows-capture` |
| 2 | BitBlt | 窗口 GDI DC，不看屏幕坐标。全屏/窗口化都能用，但**硬件加速下会静默返回全 0 的合法图** |
| 3 | DXcam | DXGI，最快。但窗口越界时抛 `Invalid Region`，且全屏独占下不可用 |
| 4 | mss | 纯屏幕坐标抓取，最后兜底 |

可用 `--force-channel wgc` 强制单通道；阈值可用环境变量
`NEKO_ENTITY_THRESHOLD_MS` 调（默认 120ms，本机实测 p50≈93ms）。

### 它仍然不做什么 / What it still does not do

- **不读宿主的 `bus`**（conversations / frames 属隐私面）
- **不自行调用 LLM**（除用户显式开启的规划路径）
- **不抓真·独占全屏** —— WGC 的设计边界；原神默认的无边框全屏可以
- **不做路径规划** —— 感知层目前拿不到 `position`（定位服务未接入引擎），
  所以「走到某坐标」这条路还不存在；`MOVE_FORWARD` 只能盲走 60 秒

## 入口与工具 / Entries and tools

| 类型 | id | 说明 |
|---|---|---|
| entry | `set_game` | 登记 / 切换当前游戏 |
| entry | `get_current_game` | 查看当前登记的游戏 |
| entry | `list_games` | 列出可用游戏 |
| entry | `refresh_game_context` | 重新注入语境（并重新读一遍术语覆盖层） |
| entry | `activate_game` | 面板按钮：探测并建立会话 |
| llm tool | `set_current_game` | 对话里说「我在玩 XX」时由模型调用 |
| llm tool | `lookup_game_term` | 按需查术语，只查当前游戏 |
| tool | `autopilot.tools.run_loop` | 启动引擎主循环 |
| tool | `autopilot.tools.capture_diagnose` | 四通道抓图体检 |
| tool | `autopilot.tools.wgc_smoke` | WGC 冒烟（一次性） |
| tool | `autopilot.tools.loop_diagnose` | 定位某一环挂了 |

## 配置 / Configuration

插件清单里的 `[genshin_game_buddy]` 与 `[games.*]` 都可以被用户运行配置覆盖：

```text
<用户数据根目录>/plugins/genshin_game_buddy/config/plugin.toml   # 登记 / 调整游戏
<用户数据根目录>/plugins/genshin_game_buddy/data/terms/<游戏>/   # 覆盖术语（按 key 逐字段合并）
```

环境变量（都是**演示/实验开关**，用环境变量而不是配置文件，避免进用户面板）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MGC_EVENT_TO_YUI` | `0` | 事件流 → YUI 台词的适配器总开关 |
| `MGC_STRESS_MODE` | `0` | 内置压力模式（`fast` / `long`） |
| `NEKO_ENTITY_THRESHOLD_MS` | `120` | 世界实体检测的启用阈值（按机器而异） |
| `NEKO_PLUGIN_ROOT` | 本机路径 | `e2e_pathfind.py` 的插件根 |

> ⚠️ `MGC_*` 会被 `tests/conftest.py` 的 autouse fixture 在测试期间清空。
> 真机上设过它们会让**每条集成用例**都拉起后台追尾线程并往 `ctx` 推消息，
> 把断言打脏 —— 2026-10-11 为此踩过。

面向用户的使用说明见 `docs/quickstart.md`（会显示在插件管理器的指南页）。

## Development

The plugin source and its Git repository live at:

```text
N.E.K.O/plugin/plugins/genshin_game_buddy
```

插件源码及其 Git 仓库直接位于：

```text
N.E.K.O/plugin/plugins/genshin_game_buddy
```

プラグインのソースと Git リポジトリは次の場所にあります：

```text
N.E.K.O/plugin/plugins/genshin_game_buddy
```

When publishing to the plugin market, use this GitHub repository name:

发布到插件市场时，请使用以下 GitHub 仓库名：

プラグインマーケットへ公開する際は、次の GitHub リポジトリ名を使用してください：

```text
n.e.k.o_plugin_genshin_game_buddy
```

From this plugin repository root:

```bash
uvx ruff==0.12.4 check --ignore-noqa --config ruff.toml .
```

From the N.E.K.O repository root / 在 N.E.K.O 仓库根目录中 / N.E.K.O リポジトリのルートで：

```bash
uv run --with pip neko-plugin sync genshin_game_buddy --clean
uv run neko-plugin check genshin_game_buddy
uv run neko-plugin check -r genshin_game_buddy
```

Python runtime dependencies are declared in `pyproject.toml` and synced into
`vendor/` for packaging. The generated `vendor/` directory is not committed;
local builds and CI recreate it before release checks.

Python 运行时依赖声明在 `pyproject.toml` 中，并在打包时同步到 `vendor/`。
生成的 `vendor/` 不提交；本地构建和 CI 会在发布检查前重新生成它。

Python ランタイム依存関係は `pyproject.toml` に宣言し、パッケージ化時に
`vendor/` へ同期します。生成された `vendor/` はコミットせず、ローカルビルドと
CI が公開前チェックで再生成します。

## Market release / Market 发布 / Market 公開

Publish the version declared in `plugin.toml`. By default this pushes the Git
tag, waits for the standard GitHub Release, and notifies the plugin market.

发布 `plugin.toml` 中声明的版本。默认会推送 Git tag、等待标准 GitHub
Release，然后通知插件市场。

`plugin.toml` で宣言されたバージョンを公開します。既定では Git tag を
push し、標準 GitHub Release を待ってからプラグインマーケットへ通知します。

```bash
uv run neko-plugin publish genshin_game_buddy
```

To run only one half explicitly / 如需仅执行一部分 / 一方のみを実行する場合:

```bash
uv run neko-plugin publish github genshin_game_buddy
uv run neko-plugin publish market https://github.com/owner/repo/releases/tag/v0.1.0
```

The generated `.github/workflows/release.yml` builds and uploads
`genshin_game_buddy.neko-plugin`. The market independently verifies that Release
before publishing it.

生成的 `.github/workflows/release.yml` 会构建并上传插件包；Market 会独立验证
该 Release 后再发布。

生成された `.github/workflows/release.yml` がプラグインパッケージをビルドして
アップロードし、Market はその Release を独立検証してから公開します。

## Entry

```toml
entry = "plugin.plugins.genshin_game_buddy:GenshinGameBuddyPlugin"
```
