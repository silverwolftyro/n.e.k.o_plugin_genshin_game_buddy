# terms/genshin —— 原神攻略术语目录

> 最后更新：2026-11-09
> 本目录随插件包分发，只读；用户侧改写请走覆盖层（见文末「覆盖层机制」）。

## 一、本目录文件

| 文件 | 维度 | 内容 | 条目数 | 数据来源 |
|---|---|---|---|---|
| `core.toml` | core | 核心术语 + `[game].tone`（口吻） | 4 条机制 | 人工撰写 |
| `characters.toml` | characters | 角色术语 | 125 条 | **冻结快照**（`source = "miao-plugin"`） |
| `systems.toml` | systems | 机制与系统术语 | 15 条 | 人工撰写 |
| `slang.toml` | slang | 玩家俗语黑话 | 若干 | 人工撰写 |
| `scenes.toml` | scenes | 18 个游戏界面场景定义 | 18 个 | 人工撰写 |
| `README.md` | — | 本说明 | — | — |

`characters.toml` 之外的四个维度文件**没有** `schema_version` / `generated_at`
顶层字段——只有角色库是外部同步产物，需要版本锚点。

## 二、当前状态：角色库是**冻结快照**

`characters.toml` 的 125 条角色条目，每条都带 `source = "miao-plugin"`，
文件头带 `schema_version = 1` 与 `generated_at = "2026-11-09"`。

**为什么是冻结快照：**

- 生成脚本 `tools/sync_mihoyo_terms.py` **不在本工作区**（已全盘检索确认无命中）。
- 同步源 `miao-plugin` **也不在本工作区**（已全盘检索确认无命中）。
- 两者都不在 → **无法重跑生成**，也无法用脚本增量刷新。
- 因此文件内容按现状冻结：条目不重排、不删改、不自动覆盖。

原文件头里那句「AUTO-GENERATED，不要手改」记录的是**当初的生成方式**，
不是现在的维护方式。原文一律保留在文件头，仅供溯源。

## 三、后续更新路径

### 路径 A：人工增量（当前唯一可执行）

1. 手工编辑 `characters.toml`，新增 / 修正条目；**保持 `source` 字段**：
   - 来自历史 miao-plugin 快照的条目 → `source = "miao-plugin"`
   - 人工新增的条目 → `source = "manual"`
2. 新增或修改条目后，**手工把文件头的 `generated_at` 更新为本次修改日期**；
   `schema_version` 在结构不兼容变更时才 +1（例如新增条目级字段）。
3. 改动前先备份；改完用现有加载器校验（见第五节）。

### 路径 B：恢复同步脚本（需先找回两样东西）

要恢复自动同步，必须先找回：

1. `miao-plugin` 源数据（提供角色表 / 别名 / 缩写 / 天赋命座机制词高亮）；
2. `tools/sync_mihoyo_terms.py`（生成脚本）。

**当前这两样都不在工作区。** 在找回之前，路径 B **不可用**——不要假设它存在，
也不要新建一个「看起来像」的脚本名。找回后需注意：

- 脚本会**整体覆盖** `characters.toml`（幂等，不累积），
  所以人工增量必须先合并回源数据，否则会被覆盖丢失；
- 覆盖之后仍需人工补 `schema_version` / `generated_at`（脚本不写这两个字段）。

## 四、为什么不把 miao-plugin 一起分发

`recognition/notes/LICENSING.md` 已就第三方抓取数据给出边界：
不随包分发。角色库是**唯一**从第三方索引落进包里的内容，
条目级 `source` 字段就是为「这条数据从哪来」留的追溯点。

## 五、改完怎么校验

用现有加载器校验（不需要起插件、不需要联网）：

```python
import tomllib, pathlib
p = pathlib.Path("terms/genshin/characters.toml")
data = tomllib.loads(p.read_text(encoding="utf-8"))
assert data["schema_version"] == 1
assert len(data["terms"]) == 125
```

或直接跑插件自带的离线测试：

```
pytest tests
```

## 六、覆盖层机制（用户侧改写，不动包内文件）

术语库是**两层**结构：

- **默认层**（只读）：`<插件目录>/terms/genshin/<维度>.toml`，随包分发；
- **覆盖层**（可写）：`<data_path()>/terms/genshin/<维度>.toml`，用户自己的数据目录。

同名条目以**覆盖层为准**；同名 key 的多个字段按字段合并。
`scenes.toml` 同样是两层，但**整段替换**（不是逐字段合并）。

注意：包内只认 `core` / `characters` / `slang` / `systems` 四个术语维度，
`scenes.toml` 走独立的场景加载器。
