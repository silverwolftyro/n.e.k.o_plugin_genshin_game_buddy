# autopilot/tools/diag/ — 诊断工具索引

> 这个目录**不搬文件**，只当索引。理由见文末「为什么不搬」。

插件侧每一个「回答一个具体问题」的可执行工具，都在这里列一遍：
**什么问题 / 什么时候跑 / 输出怎么读**。

## 插件侧（`autopilot/tools/`）

| 工具 | 回答什么问题 | 什么时候跑 |
|---|---|---|
| `capture_diagnose.py` | **这一台机器、这个窗口，四条截图通道各自能抓到什么？** | 换了显示模式 / 换了游戏 / 截图突然黑帧 |
| `wgc_smoke.py` | **WGC 这条路通不通？**（一次性冒烟，20 行逻辑） | 怀疑 WGC 环境坏了；或给 `capture_diagnose` 做阳性对照 |
| `loop_diagnose.py` | 引擎哪一环挂了（感知 / 决策 / 执行） | `run_loop` 启动失败或跑不起来 |
| `policy_diagnose.py` | 某条规则为什么没命中 | 规则「明明写了却不触发」 |
| `window_find_probe.py` | 窗口查找主路径 vs 兜底路径对照 | 会话探针找不到窗口时 |
| `rule_test.py` | 规则 DSL 求值过程（**不接执行层**） | 离线调规则 |
| `replay.py` | 用录下的 state 回放决策 | 复现某个 tick 的判定 |

## Benchmark / e2e 工具链（`recognition/exports/benchmark/scripts/misc/`）

这些**不在插件仓库内**，是一套互相关联的独立工具链：

| 脚本 | 用途 |
|---|---|
| `bench_world_entities.py` | 实体检测的检出率 + 耗时基准 |
| `scan_bgi_world_dml.py` | DML provider 的类分布扫描 |
| `e2e_pathfind.py` | 寻路端到端（**唯一持有 `--delay` / `PLUGIN_ROOT` 环境变量**） |
| `e2e_dryrun.py` | 只读输入、不发按键的干跑 |
| `e2e_teleport.py` | 传送链路 |
| `diag_f12.py` / `diag_f12_watcher.py` | F12 急停是否真的生效 |
| `diag_sprint.py` / `diag_turn.py` | 按键注入（Shift 冲刺 / 转视角）是否真的到达游戏 |
| `diag_env.py` | 环境体检 |
| `uipi_check.py` | UIPI 权限门禁（`--live` 的前置） |

## 为什么不搬

把它们从 `recognition/.../misc/` 挪到 `autopilot/tools/diag/` 会：

1. **打断相互 import** —— `e2e_pathfind.py` 依赖同目录的 `uipi_check.py`、
   `path_planner.py`（后者还是懒加载）
2. **打断硬编码路径** —— `e2e_pathfind.py` 的 `PLUGIN_ROOT`、
   `ROUTES_DIR` 都按 `recognition/` 的布局写死
3. **跨越仓库边界** —— `recognition/` 是独立的知识/基准区，插件包不该吞掉它

所以正确的「归拢」是**这张索引表**，不是文件搬家。等哪天两边的边界真的
需要合并（或者 misc/ 整体废弃），再做一次性搬迁 + 改路径，那时才有意义。

## 一条通用建议

**判断「某个闸门/限流有没有生效」要看 `state_snapshot.json` 的
`raw_stats` 计数器，不要数事件流。** 事件总线带 `dedupe_key` 和默认 5 秒
去重窗口，数出来的一定偏小 —— 2026-10-11 为此误判过一次（详见
`recognition/notes/session_20261012_signals.md` §4.1）。