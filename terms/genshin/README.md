# terms/genshin —— 原神攻略术语目录

> 最后更新：2026-11-10
> 本目录随插件包分发，只读；用户侧改写请走覆盖层（见文末「覆盖层机制」）。

## 一、本目录文件

| 文件 | 维度 | 内容 | 条目数 | 数据来源 |
|---|---|---|---|---|
| `core.toml` | core | 核心术语 + `[game].tone`（口吻） | 4 条机制 | 人工撰写 |
| `characters.toml` | characters | 角色术语 | 125 条 | **冻结快照**（`source = "miao-plugin"`） |
| `systems.toml` | systems | 机制与系统术语 | 15 条 | 人工撰写 |
| `slang.toml` | slang | 玩家俗语黑话 | 若干 | 人工撰写 |
| `scenes.toml` | scenes | 18 个游戏界面场景定义 | 18 个 | 人工撰写 |
| `weapons.toml` | —（离线数据） | 武器：名称/类型/稀有度/基础攻击/主属性/特性/突破材料 | 10 条 | **外部快照**（`source = "genshin-db"`） |
| `artifacts.toml` | —（离线数据） | 圣遗物：套装名/可用稀有度/2 件套/4 件套/部位名 | 5 套 | **外部快照**（`source = "genshin-db"`） |
| `materials.toml` | —（离线数据） | 材料：名称/稀有度/类别/用途说明/获取途径 | 20 条 | **外部快照**（`source = "genshin-db"`） |
| `README.md` | — | 本说明 | — | — |

术语维度文件里**只有** `characters.toml` 有 `schema_version` / `generated_at` 顶层字段
（`core.toml` / `systems.toml` / `slang.toml` / `scenes.toml` 都没有）；
三个离线数据文件（`weapons.toml` / `artifacts.toml` / `materials.toml`）也有——
它们和角色库一样是外部同步产物，需要版本锚点。

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

## 三、原神离线数据（武器 / 圣遗物 / 材料）

`weapons.toml` / `artifacts.toml` / `materials.toml` 是**原神攻略离线数据集**，
与术语库无关：`term_store` 的维度白名单只有 `core` / `characters` / `slang` /
`systems` 四个，**不读**这三个文件，因此它们不会污染术语检索。

| 文件 | 条目 | 关键字段 |
|---|---|---|
| `weapons.toml` | 10 把武器 | 名称 / 类型 / 稀有度 / 基础攻击 / 主属性 / 武器特性 / 突破材料 |
| `artifacts.toml` | 5 套圣遗物 | 套装名 / 可用稀有度 / 2 件套 / 4 件套 / 五件部位名 |
| `materials.toml` | 20 种材料 | 名称 / 稀有度 / 类别 / 用途说明 / 获取途径 |

**数据源与可复现性：**

- 上游是社区数据源 `genshin-db`（仓库 `theBowja/genshin-db`）；
- 三个文件头都记着 `source` / `source_repo` / **`source_commit`**
  （固定到某个 commit），任何人重跑取数脚本都得到同一份数据；
- 取数脚本 `tools/fetch_genshin_db.py` **仅开发期使用，不随包运行**。

**运行时零网络：** 插件运行时只读这三个**本地文件**，不发任何请求。
查询入口是插件根目录的 `game_data_store.py`
（`load_game_data()` → `data.weapon("风鹰剑")` / `data.material("原石")`），
它**不 import** 任何网络库，也**不在**运行时 import 链上。

**维护路径：** 改条目 → 重跑 `python tools/fetch_genshin_db.py` → 提交新的 TOML。
脚本里的选品清单是显式白名单，增删条目要同时改脚本里的清单。

> ⚠️ 上游 `genshin-db` 的武器条目**没有副属性（affix）列表**，
> 因此 `weapons.toml` 里也没有该字段——不造上游没有的数据。

## 四、后续更新路径

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

## 五、为什么不把 miao-plugin 一起分发

`recognition/notes/LICENSING.md` 已就第三方抓取数据给出边界：
不随包分发。角色库是**唯一**从第三方索引落进包里的内容，
条目级 `source` 字段就是为「这条数据从哪来」留的追溯点。

## 六、改完怎么校验

用现有加载器校验（不需要起插件、不需要联网）：

```python
import tomllib, pathlib
p = pathlib.Path("terms/genshin/characters.toml")
data = tomllib.loads(p.read_text(encoding="utf-8"))
assert data["schema_version"] == 1
assert len(data["terms"]) == 125
```

离线数据同理（三个文件的条目数是固定的）：

```python
import tomllib, pathlib
p = pathlib.Path("terms/genshin/weapons.toml")
data = tomllib.loads(p.read_text(encoding="utf-8"))
assert data["schema_version"] == 1
assert len(data["weapons"]) == 10
```

或直接跑插件自带的离线测试：

```
pytest tests
```

## 七、覆盖层机制（用户侧改写，不动包内文件）

术语库是**两层**结构：

- **默认层**（只读）：`<插件目录>/terms/genshin/<维度>.toml`，随包分发；
- **覆盖层**（可写）：`<data_path()>/terms/genshin/<维度>.toml`，用户自己的数据目录。

同名条目以**覆盖层为准**；同名 key 的多个字段按字段合并。
`scenes.toml` 同样是两层，但**整段替换**（不是逐字段合并）。

注意：包内只认 `core` / `characters` / `slang` / `systems` 四个术语维度，
`scenes.toml` 走独立的场景加载器。

**三个离线数据文件（`weapons.toml` / `artifacts.toml` / `materials.toml`）
目前不参与覆盖层机制** —— 它们是包内只读快照，用户侧改写要另开一套约定。
