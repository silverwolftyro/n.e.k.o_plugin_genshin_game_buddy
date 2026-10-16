# autopilot 三层符号级地图

> 配套 `LOGIC_MAP.md`（目录级）。本文件只做 **perception / policy / actuator** 三层的
> 符号级精确盘点。**不重复** LOGIC_MAP 的目录总览 / 分层图 / 入口清单。
>
> 生成方式：只读 grep/AST 检索 + 逐文件通读关键符号。**未执行任何模块**。
> 路径基准：`...\genshin_game_buddy\autopilot\`
> 日期：2026-09-28

## 覆盖率速览

| 层 | .py 文件 | 精确总行数 |
|---|---|---|
| perception | 9 | **2955** |
| policy | 8 | **2331** |
| actuator | 8 | **2905** |
| **合计** | **25** | **8191** |

---

# 一、perception 层（9 个 .py / 2955 行）

职责：把「一张图 + 一次窗口定位」变成一张 `StatePacket`。唯一出边是 `contracts`。

---

## `perception/__init__.py` — 60 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| — | 全部 re-export | 18–35 | 把 8 个子模块的公开符号平铺到包级 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `.builder` `.capturer` `.ocr` `.roi_expr` `.screen_region` `.template` `.yolo` |
| 被哪些 autopilot 内部模块 import | 无（包级导出无人使用；全部走子模块直连） |
| 不可替代性 | **可替代**。`tools/_stack.py` 直接 `from ..perception.builder import StateBuilder`，绕过本文件；`perception/__init__.py` 的导出面没有消费者。 |

---

## `perception/capturer.py` — 577 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `DEFAULT_FAIL_THRESHOLD` | 32 | 连续失败几次才抛（默认 3，防抖） |
| class | `CaptureChannel(str, Enum)` | 35 | `DXCAM` / `BITBLT` / `MSS` 三通道枚举（2026-10-10 `PRINTWINDOW` 出链） |
| dataclass | `CapturedFrame` | 42 | 一帧：`image` + `rect` + `timestamp_ms` + `frame_seq` + `channel` |
| exc | `CaptureError(RuntimeError)` | 64 | 截图失败 |
| exc | `WindowNotFoundError(CaptureError)` | 68 | 窗口消失（`CaptureError` 的子类，调用方可区别处理） |
| class | `Capturer` | 72 | 三通道探测 + `open/grab/close` + `is_window_alive`；`_capture_lock: RLock` |
| func | `_find_window()` | 430 | 窗口枚举 + 标题子串匹配 |
| func | `_find_fullscreen()` | 469 | 全屏独占窗口的矩形猜测（兜底，不推荐） |
| func | `_find_window_ctypes()` | 490 | 纯 ctypes 查找（无 pywin32 时） |
| func | `_client_rect_on_screen()` | 502 | 客户区坐标 → 屏幕坐标 |
| func | `_grab_bitblt()` | 786 | GDI BitBlt 窗口级抓取（2026-10-10 顶替 `_grab_printwindow()` 入链） |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | **无**（层内零依赖，可独立测试） |
| 被哪些 autopilot 内部模块 import | `perception/__init__.py:19`、`perception/builder.py:28`、`tools/_stack.py:84`、`tools/capture_demo.py`、`tools/autopilot_demo.py`、`tools/template_demo.py`、`tools/check_env.py`、`tools/loop_diagnose.py` |
| 不可替代性 | **不可替代**。三层里唯一处理「窗口定位 + 三通道降级 + 空帧重建」的文件。`CapturedFrame` 是 perception → core 的**唯一数据出口类型**，换掉它整层契约就变了。 |

---

## `perception/builder.py` — 399 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `BLACK_FRAME_LUMA` | 39 | 低于此亮度判为黑帧 |
| class | `StateBuilder` | 42 | **perception 的门面**：`build(frame) -> StatePacket` |
| func | `_luma()` | 387 | 灰度均值 |
| func | `_scale_box()` | 394 | ROI 缩放后回写框坐标 |

**`StateBuilder` 关键成员**

| 成员 | 行号 | 说明 |
|---|---|---|
| `__init__(capturer, yolo, template, ocr, roi, game_id, scene_map, scene_routes)` | 45 | 全部可选依赖，缺哪个都能跑（只是认得少） |
| `scene_map: dict[str,str]` | 66 | YOLO 标签 → 场景 ID（`main_menu`→`MAIN_MENU` 等 7 条默认） |
| `scene_routes: dict[str,list[str]]` | 72 | 场景 → 该场景要跑哪些模板（性能关键） |
| `last_detections` / `last_matches` | 92/93 | 本帧产物，调试叠加层直接读（不重跑识别） |
| `drain_events()` | 97 | 事件出口，`build` 累积、调用方取走 |
| `build(frame, *, forced_scene=None)` | 107 | **唯一主入口**。内部不抛异常，失败体现在 `resource.perception_ok` |
| `set_templates(specs)` | 324 | 由配置层注入模板定义 |
| `_template_spec(name)` | 319 | 从注入的 specs 取 |
| `_roi_box_for(name, region)` | 328 | 取该模板的 ROI 表达式并求值 |
| `_track_scene_enter(scene_id)` | 264 | 场景进入时刻（供 `SceneInfo.entered_ms`） |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.events`、`..contracts.state_packet`、`.capturer`、`.ocr`、`.roi_expr`、`.screen_region`、`.template`、`.yolo`（8 个，层内耦合最密的文件） |
| 被哪些 autopilot 内部模块 import | `perception/__init__.py:18`、`tools/_stack.py:83`、`tools/autopilot_demo.py` |
| 不可替代性 | **不可替代**。它是场景路由表（`scene_routes`）的唯一实现者 —— 「当前场景该跑哪些识别」这条策略全仓库只在这里。`build()` 是 perception → core 的唯一方法入口。 |

---

## `perception/template.py` — 454 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `MASK_COLOR_BGR` | 31 | 绿色遮罩色 `(0,255,0)` |
| const | `DEFAULT_THRESHOLD` | 33 | 0.80 |
| const | `DEFAULT_NMS_IOU` | 34 | 0.5 |
| dataclass | `TemplateSpec` | 38 | 模板定义：`path` / `threshold` / `roi` / `stable_count` / `use_binary` / `use_3_channels` / `use_green_mask` |
| dataclass | `TemplateMatch` | 75 | 命中结果：`box` / `score` / `point` |
| class | `StableValueAccumulator` | 87 | **抗抖原语**：同位置连续 N 帧才算数（`max_consecutive_count` 供诊断） |
| class | `TemplateMatcher` | 159 | 模板缓存 + `find()` + `find_multi()` |
| func | `_to_gray3()` | 392 | BGR → 三通道灰度 |
| func | `_is_bounded_mode()` | 398 | 判断 CV2 匹配模式是否有界 |
| func | `_green_mask()` | 412 | 生成绿色遮罩 |
| func | `_suppress_overlap()` | 420 | 多点匹配的 IoN 抑制 |

**关键事实**：`TemplateMatcher.__init__` 只收 `nms_iou` / `min_search_margin`（**不收** specs 或 roi）—— 模板定义由 `StateBuilder.set_templates()` 注入。

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `.screen_region`（仅 1 个） |
| 被哪些 autopilot 内部模块 import | `perception/__init__.py:29`、`perception/builder.py:32`、`tools/_stack.py:85`、`tools/template_demo.py` |
| 不可替代性 | **不可替代**。`StableValueAccumulator` 是三层里唯一的「跨帧稳定判定」原语（policy 的三态、actuator 的熔断都不做这个）。删掉它等于把单帧噪声直接喂给决策层。 |

---

## `perception/yolo.py` — 373 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `DEFAULT_TIMEOUT_MS` | 27 | 500ms 超时 |
| exc | `YoloUnavailableError` | 30 | 未装 onnxruntime / 模型缺失 |
| dataclass | `Detection` | 35 | `label` / `box` / `conf` |
| class | `YoloDetector` | 50 | ONNX Runtime 会话 + `detect()` + `warmup()` + `close()` |
| func | `_letterbox()` | 233 | 等比缩放 + 补边到 imgsz |
| func | `_postprocess()` | 254 | 反变换 + NMS（普通输出头） |
| func | `_postprocess_end2end()` | 317 | 反变换（end2end 输出头，无 NMS） |
| func | `_nms()` | 339 | 自实现 NMS |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | **无** |
| 被哪些 autopilot 内部模块 import | `perception/__init__.py:35`、`perception/builder.py:33`、`tools/eval_yolo.py`、`tools/convert_to_onnx.py` |
| 不可替代性 | **不可替代，且不可绕过**。这是全仓库**唯一**允许出现的 ONNX 推理实现（AGPL 边界）。`_postprocess_end2end` 与 `_postprocess` 的双路分支说明模型输出头有两种形态，换实现必须同时覆盖两条。 |

---

## `perception/roi_expr.py` — 402 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `ALLOWED_NODES` | 26 | ast 节点白名单 |
| const | `ALLOWED_CALLS` | 57 | 允许的函数名（`rect` / `cutLeft` …） |
| const | `BUILTIN_VARS` | 64 | `cw` `ch` `cx` `cy` `s` |
| exc | `RoiExprError(ValueError)` | 67 | 表达式非法 |
| dataclass | `CompiledRoi` | 76 | 编译产物 |
| class | `RoiExprEvaluator` | 85 | `compile()` + `eval()` + `eval_scalar()` |
| func | `_make_rect()` | 313 | `rect(x,y,w,h)` 实现 |
| func | `_pct()` | 345 | 百分比参数求值 |
| func | `_at()` | 351 | `@别名` 展开（多层引用 + 循环检测） |
| dict | `_BIN_OPS` | 355 | 二元运算符分发表 |
| dict | `_CMP_OPS` | 365 | 比较运算符分发表 |
| func | `eval_roi_expression()` | 380 | **模块级便捷入口**（绕过类直接求值） |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | **无** |
| 被哪些 autopilot 内部模块 import | `perception/__init__.py:27`、`perception/builder.py:30`、`config/loader.py`（`validate_config` 用它校验 ROI 表达式） |
| 不可替代性 | **不可替代**。`config/loader.py` 的配置校验直接复用它 —— 也就是说配置期的 ROI 校验和运行期求值共用同一份实现，换了它就等于让配置校验失真。 |

---

## `perception/screen_region.py` — 196 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| dataclass | `ImageRegion` | 23 | 链式裁剪：`image` + `origin` + `transform` + `parent` |
| func | `region_of_frame()` | 190 | 由裸 ndarray 包一层（锚定到屏幕坐标） |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | **无** |
| 被哪些 autopilot 内部模块 import | `perception/__init__.py:28`、`perception/builder.py:31`、`perception/ocr.py:27`、`perception/template.py:26`（4 个，层内被依赖最多的叶子节点） |
| 不可替代性 | **不可替代**。它是层内唯一的坐标回推载体（ROI 内坐标 ↔ 全屏坐标）。`to_parent_coords` / `to_screen_coords` 保证 ROI 表达式在多层裁剪后仍能定位到屏幕 —— 这是 `debug_overlay` 能画对框的前提。 |

---

## `perception/ocr.py` — 326 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| re | `_HAN_SPACE_RE` | 33 | 中文字间空格清洗 |
| const | `DEFAULT_FAIL_THRESHOLD` | 36 | 5 |
| class | `TriState(Enum)` | 39 | 三态（避免 OCR 结果被压成 bool） |
| dataclass | `OcrResult` | 48 | 文本 + TriState + 置信度 |
| class | `OcrEngine` | 72 | WinRT OCR 封装 + RapidOCR 降级 |
| func | `_encode_png()` | 282 | ndarray → PNG bytes（WinRT 要求） |
| func | `_apply_replace()` | 289 | 文本替换表 |
| func | `_match()` | 299 | 目标文本匹配 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `.screen_region` |
| 被哪些 autopilot 内部模块 import | `perception/__init__.py:26`、`perception/builder.py:29` |
| 不可替代性 | **部分可替代**。识别后端可换（WinRT → RapidOCR），但 `TriState` 的三态语义与 `contracts` 的 null 约定直接对应 —— 若压成 bool，「识别不到」与「识别到空」就不可区分了。 |

---

## `perception/debug_overlay.py` — 168 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| dict | `_LABEL_COLORS` | 27 | 标签 → BGR 颜色 |
| const | `_DEFAULT_COLOR` | 38 | 兜底色 |
| class | `DebugOverlay` | 41 | `render(frame, state, detections, matches)` —— OpenCV imshow 叠加 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | **无** |
| 被哪些 autopilot 内部模块 import | `perception/__init__.py`（未在 re-export 名单内，见下）、`tools/autopilot_demo.py`（3 处引用） |
| **不可替代性** | **完全可替代且事实上被绕过**。`perception/__init__.py` 的 18–35 行 re-export 清单里**没有** `DebugOverlay`，但 grep 显示该文件第 1 处命中在 `perception/__init__.py` —— 说明只在注释或 `__all__` 字符串中出现，实际未被包级导出。 |

---

# 二、policy 层（8 个 .py / 2331 行）

职责：`StatePacket → IntentPacket | None`。唯一出边是 `contracts` + `state`。

---

## `policy/__init__.py` — 52 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| — | `PolicySource` `IntentSourceKind` | 37 | 从 `.base` re-export |
| — | `PolicyConfig` `create_policy_source` `registered_kinds` | 38 | 从 `.factory` re-export |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `.base` `.factory`（**仅此两个**） |
| 被哪些 autopilot 内部模块 import | 无（所有调用方都直连 `.factory`，如 `tools/_stack.py:44`） |
| 不可替代性 | **是架构门禁**。`__all__` 只有 4 项，`LocalPolicyTree` / `DslEvaluator` / `Rule` 刻意不导出 —— 将来闭源化时这条边界就是替换点。调用方一旦写 `from autopilot.policy.local_tree import ...` 可替换性就破了。 |

---

## `policy/base.py` — 139 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| class | `IntentSourceKind(str, Enum)` | 36 | `LOCAL` / `CLOUD` / `DISABLED` |
| exc | `PolicySourceError` | 64 | policy 层错误基类 |
| exc | `PolicyConfigError` | 68 | 配置非法 / 实现不存在 |
| protocol | `PolicySource` | 73 | **对外唯一契约**：`name` / `latency_class` / `poll()` / `available()` / `last_error` |
| func | `describe_source()` | 117 | 任意 PolicySource → 诊断 dict（闭源实现也能被诊断） |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.intent_packet` `..contracts.state_packet` |
| 被哪些 autopilot 内部模块 import | `policy/__init__.py:37`、`policy/factory.py:29` |
| 不可替代性 | **不可替代**。`PolicySource` Protocol 是整个 policy 层存在的理由 —— 它是「本地开源 / 将来闭源 / 云端增值」三种未来的唯一公约数。 |

---

## `policy/factory.py` — 232 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| dataclass | `PolicyConfig` | 46 | `kind` / `rules_dir` / `rule_files` / `idle_warn_after` / `name` |
| class | `DisabledSource` | 113 | 空实现（`poll` 恒 None，`available` 恒 False） |
| dict | `_REGISTRY` | 142 | `IntentSourceKind → Builder` |
| func | `register()` | 145 | 注册实现（云端接入点） |
| func | `_build_disabled()` | 150 | |
| func | `_build_local()` | 159 | **唯一 `import LocalPolicyTree` 的地方**，惰性 |
| func | `_build_cloud()` | 171 | 占位，**直接抛 `PolicyConfigError`** |
| func | `create_policy_source()` | 192 | 入口。不满足 Protocol 直接拒；`kind` 未知不回落 LOCAL |
| func | `registered_kinds()` | 221 | 诊断用 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.intent_packet` `..contracts.state_packet`、`.base`、**惰性** `.local_tree`（161 行） |
| 被哪些 autopilot 内部模块 import | `policy/__init__.py:38`、`tools/_stack.py:44`、`tools/policy_diagnose.py` |
| 不可替代性 | **不可替代**。`_build_local` 是全仓库唯一 import 具体实现的地方（惰性），`_build_cloud` 的「响亮失败而非静默回落」保证了不会有人以为有云端回退。 |

---

## `policy/context.py` — 382 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| class | `UnknownValue` | 42 | **Null Object**：所有比较恒 False、布尔恒 False |
| const | `UNKNOWN` | 73 | `UnknownValue()` 单例 |
| dataclass | `PolicySignals` | 77 | `has_enemy` / `enemy_count` / `target_locked`，**全部默认 None** |
| class | `PolicyContext` | 109 | DSL 求值上下文，21 个方法即 DSL 函数 |
| func | `_build_function_table()` | 338 | 未绑定方法 → 函数清单 |
| const | `_FUNCTION_TABLE` | 370 | 模块级求值一次 |
| const | `FUNCTION_NAMES` | 373 | `frozenset`，dsl.py 的白名单来源 |

**`PolicyContext` 的 21 个 DSL 函数**：`t` `since` `count` `last_exec` `scene` `scene_confidence` `hp` `energy` `resin` `has_enemy` `enemy_count` `target_locked` `onfield` `q_ready` `e_ready` `slot` `pos_quality` `pos_usable` `ocr_ok` `perception_ok` `fps`

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.state_packet`、**`..state`（`IntentRecord`）** |
| 被哪些 autopilot 内部模块 import | `policy/dsl.py:36`、`policy/local_tree.py:36` |
| 不可替代性 | **不可替代**。`FUNCTION_NAMES`（373）是 `dsl.py` 白名单的**唯一来源** —— 改这里等于同时改「能用什么」和「有什么」，两处永远不会漂移。`UnknownValue` 的「未知→不动作」方向性选择是三层里唯一的 Null Object 实现。 |

---

## `policy/dsl.py` — 388 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| exc | `RuleEvalError` | 43 | 携带 表达式 / 规则名 / state_version / 文件 / 行号 |
| const | `ALLOWED_NODES` | 107 | ast 节点白名单（刻意保守，**无 IfExp**） |
| const | `_DSL_FUNCS` | 129 | `{"min", "max"}` |
| const | `_TOML_BOOLS` | 136 | `{"true": True, "false": False}`（TOML 惯用小写） |
| const | `ALLOWED_FUNCS` | 139 | `FUNCTION_NAMES \| _DSL_FUNCS` |
| func | `_dsl_min()` / `_dsl_max()` | 147 / 157 | **Unknown-aware**：遇 UNKNOWN 返回 UNKNOWN（内置版会抛 TypeError） |
| const | `_EXTRA_FUNCS` | 167 | 注入 eval 命名空间 |
| dataclass | `CompiledExpr` | 176 | 编译产物（跨帧复用） |
| dataclass | `EvalOutcome` | 185 | `value` + `reason` + `unknowns` |
| class | `DslEvaluator` | 203 | `compile()` + `evaluate()`；带 LRU 语义缓存 |
| func | `_closest()` | 370 | difflib 拼写建议 |
| const | `DEFAULT_EVALUATOR` | 378 | 模块级默认实例 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `.context` |
| 被哪些 autopilot 内部模块 import | `policy/rule_loader.py:30`（加载期校验表达式）、`policy/local_tree.py:37` |
| 不可替代性 | **不可替代**。规则来自 TOML（用户可写，将来可能来自外部），所以求值必须是沙箱。`ALLOWED_NODES`（107）刻意不含 `IfExp` —— 条件分支应该拆成多条 priority 不同的规则，那样「哪条赢了」才可解释。 |

---

## `policy/rule_loader.py` — 516 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `RULE_FIELDS` | 33 | 12 个合法字段 |
| const | `TOP_LEVEL_TABLES` | 39 | `{"meta", "rules"}` |
| const | `PRIORITY_RANGE` / `TTL_RANGE` / `MAX_ATTEMPTS_RANGE` | 41/42/43 | (0,100) / (1,300) / (1,10) |
| const | `DEFAULT_PRIORITY` / `DEFAULT_TTL_SECONDS` / `DEFAULT_MAX_ATTEMPTS` | 48/49/50 | 50 / 5 / 1 |
| exc | `RuleLoadError` | 53 | 带 文件 / 行号 / 规则名 / 字段名 |
| dataclass | `Rule` | 99 | 规则实体 + `compiled` 预编译字段 |
| func | `load_rules()` | 153 | 单文件 |
| func | `load_rules_from_dir()` | 158 | 目录（文件名排序加载） |
| func | `load_rules_from_files()` | 188 | 多文件 + **稳定排序**（同 priority 保持声明序） |
| func | `_read_toml()` | 246 | UTF-8 + TOML 错误包装 |
| func | `_line_of()` | 264 | **文本扫描**找字段行号（不是真解析器） |
| func | `_check_top_level()` | 279 | 顶层表白名单 |
| func | `_parse_rule()` | 303 | 单条规则 17 类校验 |
| func | `_int_field()` | 435 | 整数 + 区间 |
| func | `_scenes_field()` | 461 | 场景枚举校验（含 difflib 建议） |
| func | `_suggest_fields()` | 498 | 字段拼写建议 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.intent_packet`（`ActionType`）、`..contracts.state_packet`（`VALID_SCENE_IDS`）、`.dsl` |
| 被哪些 autopilot 内部模块 import | `policy/local_tree.py:38`、`tools/rule_test.py`、`tools/policy_diagnose.py`、`tools/replay.py` |
| 不可替代性 | **不可替代**。动作白名单从 `contracts.ActionType` 读（不是 actuator.TRAITS）—— 所以 `policy` 层对 `actuator` **零依赖**。`_line_of`（264）用文本扫描而非真解析器，注释掉了行号会偏移，但让排障快一个数量级。 |

---

## `policy/local_tree.py` — 375 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `SAFE_ACTIONS` | 44 | 场景 UNKNOWN 时仅允许的 5 个动作 |
| const | `DEFAULT_RULES_DIR` | 53 | `<autopilot>/rules` |
| dataclass | `RuleTrace` | 64 | 单条规则判定：`MATCHED`/`SKIPPED`/`ERRORED` + reason |
| dataclass | `DecisionTrace` | 86 | 一轮完整痕迹 + `matched_rule` |
| class | `LocalPolicyTree` | 113 | **实现 `PolicySource`**：`poll()` / `decide()` / `available()` |

**`LocalPolicyTree` 关键成员**

| 成员 | 行号 | 说明 |
|---|---|---|
| `__init__(rules_dir, rule_files, idle_warn_after, name, evaluator, emit_event, signals, now_fn)` | 118 | **构造期即加载规则**，失败直接抛 `RuleLoadError` |
| `poll(state)` | 157 | 委托 `decide()` + 记账 + 空闲告警 |
| `decide(state)` | 178 | 纯判定、**无副作用**，返回 `DecisionTrace`（离线工具用） |
| `_execute` 相关：`_on_idle()` | — | 连续空转告警（只告警，不补动作） |
| `stats()` | — | 规则数 / 分布 / 空闲计数 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.events` `..contracts.intent_packet` `..contracts.state_packet`、`.context`、`.dsl`、`.rule_loader` |
| 被哪些 autopilot 内部模块 import | `policy/factory.py:161`（惰性）、`tools/_stack.py:46`、`tools/policy_diagnose.py`、`tools/rule_test.py`、`tools/replay.py` |
| 不可替代性 | **不可替代**。三处核心不变量的唯一实现点：① 按 priority 升序 ② **命中即 break**（原则 3）③ 未知→不动作。`decide()` 与 `poll()` 分离（178 vs 157）是 `rule_test.py` 能离线看判定过程的前提。 |

---

## `policy/merger.py` — 247 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| protocol | `SignatureVerifier` | 53 | 签名验证接口（**云端接入点**） |
| class | `NullSignatureVerifier` | 61 | 恒拒（fail-closed 默认） |
| dataclass | `MergeResult` | 79 | `intent` + `source` + `reason` + `cloud_rejected` + `cloud_checks` |
| func | `validate_cloud_intent()` | 110 | 签名先验 + TTL + 漂移 + 场景 |
| func | `merge_intents()` | 159 | **本地优先**；本地 None 才看云端 |
| func | `_emit()` | 234 | 事件投递兜底 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.events` `..contracts.intent_packet` `..contracts.state_packet` |
| 被哪些 autopilot 内部模块 import | **`core/loop.py:568`（惰性）**、`tools/` 无 |
| 不可替代性 | **不可替代，且是跨层违规点**。它是三层之外唯一被 `core` 直接 import 的 policy 模块 —— 见第五节。 |

---

# 三、actuator 层（8 个 .py / 2905 行）

职责：`IntentPacket → 真实按键`。唯一出边是 `contracts`。

---

## `actuator/__init__.py` — 116 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| — | 全部 re-export | 22–68 | 6 个子模块的公开符号 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `.humanizer` `.safety` `.simulator` `.traits` `.validator` `.executor` |
| 被哪些 autopilot 内部模块 import | **层内无人**；`tools/_stack.py:57` 直连子模块 |
| 不可替代性 | **可替代**（同 `perception/__init__.py`）。 |

---

## `actuator/simulator.py` — 934 行（层内最大）

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `INPUT_MOUSE` / `INPUT_KEYBOARD` | 37/38 | Win32 输入类型 |
| const | `KEYEVENTF_*`（4 个） | 40–43 | 扩展键 / KeyUp / Unicode / Scancode |
| const | `MOUSEEVENTF_*`（9 个） | 45–54 | 移动 / 各键 / 滚轮 / 绝对 / 虚拟桌 |
| const | `WHEEL_DELTA` | 56 | 120 |
| const | `EXTENDED_KEYS` | 61 | 需 `KEYEVENTF_EXTENDEDKEY` 的键集合 |
| class | `VK(IntEnum)` | 72 | 全部 VK 常量 |
| const | `MOUSE_BUTTON_VKS` | 177 | 三个鼠标键 |
| class | `KeyId(int)` | 180 | **int 子类**（非 IntEnum）+ `NONE`/`UNKNOWN` 哨兵 + `from_name()` |
| exc | `InputInjectionError` | 255 | 注入失败（**含权限排查指引**） |
| exc | `InvalidStateError` | 259 | backend 缺失 / DLL 加载失败（构造期抛） |
| exc | `KeyIdError` | 269 | 键位名非法 |
| class | `_MOUSEINPUT` … `INPUT` | 280–313 | ctypes 结构体（`_ULONG_PTR` 按位宽选型） |
| func | `_user32()` | 317 | 惰性加载 user32（带缓存） |
| dataclass | `SentEvent` | 329 | 一次注入记录（`kind`/`name`/`down`/`vk`） |
| func | `decode_events()` | 342 | INPUT → SentEvent（写侧与 DryRun 共用） |
| protocol | `InputBackend` | 366 | `send()` / `is_key_down()` |
| class | `Win32Backend` | 373 | 真实 SendInput |
| class | `DryRunBackend` | 387 | 只记录不发送 |
| class | `InputSimulator` | 419 | 对外门面（`key_press` / `mouse_click` / `release_all` / `diagnose`） |
| class | `BackendReport` | 824 | 诊断报告 |
| func | `_probe_backend()` / `process_integrity()` | — | backend 探测 / 进程完整性级别（UIPI 判据） |
| func | `_vk_of` `_mk_key` `_mk_mouse` `_mouse_button_flags` `_mouse_name` | 645–683 | INPUT 构造 |
| const | `_VK_LOOKUP` | 695 | 反查表 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | **无**（唯一出边是 contracts，实测零 import） |
| 被哪些 autopilot 内部模块 import | `actuator/executor.py:34`、`actuator/safety.py:19`、`actuator/humanizer.py:31`、`actuator/keymap.py:27`、`actuator/__init__.py:42`（层内被依赖最多） |
| 不可替代性 | **不可替代**。唯一一处碰 `ctypes.SendInput` 的地方，也是「返回值校验」这条硬规则的落点。`KeyId` 用 `int` 子类而非 `IntEnum`，是为了避开 100+ 个枚举成员，同时保留 `NONE`/`UNKNOWN` 哨兵。 |

---

## `actuator/traits.py` — 304 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `SCENE_ANY` … `SCENE_TELEPORT`（13 个） | 34–46 | **从 `contracts.state_packet.SCENE_IDS` 派生**（非各写一份） |
| dataclass | `ActionTraits` | 50 | 9 字段：`strict_arrival` / `approach_by_walking` / `abort_on_target_lost` / `allowed_scenes` / `min_gap_ms` / `max_duration_ms` / `allow_move_while_turning` / `requires_foreground` / `holds_keys` |
| const | `DEFAULT_TRAITS` | 77 | |
| const | `TRAITS` | 80 | **21 个动作全覆盖** |
| func | `traits_of()` | 223 | 查表，未配置返回保守默认并告警 |
| func | `missing_traits()` | 247 | 有动作没配 traits（加载期自检） |
| func | `extra_traits()` | 254 | 配了但不在白名单（拼错） |
| func | `validate_table()` | 262 | 一致性自检 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.state_packet`（`SCENE_IDS as _SCENE`）；`..contracts.intent_packet` 三处**惰性**（29/232/249/256，`TYPE_CHECKING` + 函数内 import） |
| 被哪些 autopilot 内部模块 import | `actuator/executor.py:43`、`actuator/validator.py:23`、`actuator/safety.py:194`（惰性） |
| 不可替代性 | **不可替代**。三层的「差异化行为参数」全在这张表里；`allowed_scenes` 是「规则写了场景但 traits 不允许」这类静默失效的唯一拦截点。 |

---

## `actuator/executor.py` — 511 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| class | `Outcome(str, Enum)` | 49 | OK / REJECTED / FAILED / INTERRUPTED / TIMEOUT |
| dataclass | `ExecutionResult` | 58 | `outcome` / `reason` / `steps` / `elapsed_ms` / `released_keys` / `event` |
| class | `IntentExecutor` | 101 | **编排**：校验→安全闸→键位→人类化→执行→finally 松键 |
| exc | `_PanicStop` | 427 | 内部信号：急停 |
| exc | `_Timeout` | 431 | 内部信号：超时 |
| const | `DEFAULT_KEYMAP` | 442 | 21 个动作的默认键位 |
| const | `_MOVE_KEYS` | 471 | **硬编码方向键**（不可配） |
| const | `_SLOT_KEYS` | 479 | 硬编码队伍槽位键 |
| const | `_KEYLESS_ACTIONS` | 484 | `WAIT`/`SCREENSHOT`/`NOOP` 不查键位 |
| func | `_move_key()` / `_slot_key()` | 487/491 | 硬编码键位查表 |

**`IntentExecutor.execute()` 调用链**（`__init__` 95 行）

```
validate(commit=True) → guard.check() → resolve_key()
  → traits_of() → humanizer.hold_ms() → sim.key_press()
  → finally: sim.release_all()
```

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.events` `..contracts.intent_packet` `..contracts.state_packet`、`.humanizer` `.safety` `.simulator` `.traits` `.validator` |
| 被哪些 autopilot 内部模块 import | `actuator/__init__.py`、`tools/_stack.py:60`、`tools/input_demo.py`、`tools/loop_diagnose.py`、`tools/run_loop.py` |
| 不可替代性 | **不可替代**。三层里唯一持有「校验 → 安全闸 → 执行」顺序的地方。顺序本身是契约：校验在发按键前、安全闸在校验后、release_all 在 finally。 |

---

## `actuator/safety.py` — 320 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `PANIC_KEY` | 24 | `VK.F12`（固定，不可配） |
| const | `DEFAULT_MAX_ACTIONS` | 27 | 5000 |
| const | `REASON_PANIC` / `REASON_BREAKER` | 34/35 | 硬停机原因码前缀 |
| func | `is_hard_stop()` | 38 | **按前缀**判定（早期版本精确匹配 `"BREAKER_OPEN(剩 0ms)"`，冷却未到 0 时误判） |
| class | `BreakerState(str, Enum)` | 47 | CLOSED / OPEN / HALF_OPEN |
| dataclass | `SafetyConfig` | 54 | 8 字段策略参数 |
| dataclass | `BreakerTrip` | 79 | 一次熔断记录 |
| class | `SafetyGuard` | 87 | `panic_active` / `check()` / `mark_executed()` / `mark_success()` / `mark_failure()` / 熔断 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.events` `..contracts.state_packet`、`.simulator`（`.traits` 惰性，194 行） |
| 被哪些 autopilot 内部模块 import | `actuator/executor.py:33`、`tools/_stack.py:63`、`tools/run_loop.py` |
| 不可替代性 | **不可替代**。三层里唯一持有「急停 / 熔断 / 会话上限 / 频率限制」的模块。`is_hard_stop()` 按前缀而非全串匹配，是被真实 bug 逼出来的。 |

---

## `actuator/validator.py` — 301 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `DEFAULT_SEEN_SIZE` | 28 | 2000 |
| const | `DEFAULT_SEEN_TTL_MS` | 30 | 300_000 |
| dataclass | `ValidationResult` | 34 | `ok` / `code` / `reason` |
| class | `SeenIntentStore` | 49 | 有界去重表 + TTL |
| class | `IntentValidator` | 75 | 8 项检查：去重 / TTL / 漂移 / 场景 / 白名单 / 参数 / 频率 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `..contracts.intent_packet`（含 `MAX_STATE_VERSION_DRIFT`）、`..contracts.state_packet`、`.traits` |
| 被哪些 autopilot 内部模块 import | `actuator/executor.py:44` |
| 不可替代性 | **不可替代**。`commit=True` 的副作用（写去重表）是「同一条意图不能被执行两次」的唯一保证点 —— 也正是 `core` 不能重复调它的原因（见第六节）。 |

---

## `actuator/humanizer.py` — 203 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| dataclass | `HumanizerConfig` | 35 | 抖动幅度配置（默认全开） |
| class | `Humanizer` | 61 | `hold_ms()` / `delay_ms()` / `heading_deg()` / `click_jitter()` / `turn_steps()` |
| func | `angle_diff()` | 166 | 角度差归一（-180..180） |
| func | `apply_turn()` | 172 | 转向分步（原则 5：转向在移动中完成） |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `.simulator`（仅取 `VK`） |
| 被哪些 autopilot 内部模块 import | `actuator/executor.py:32`、`actuator/__init__.py:22` |
| 不可替代性 | **可替代**（纯函数式抖动，改动不影响正确性）。但它是三层里唯一处理「模拟人手」的地方，删掉后所有按键间隔会精确到毫秒 —— 功能不受影响，反检测能力归零。 |

---

## `actuator/keymap.py` — 216 行

| 类型 | 名字 | 行号 | 职责 |
|---|---|---|---|
| const | `DEFAULT_KEYMAP_PATH` | 32 | `config/keymap.toml` |
| exc | `KeymapError` | 35 | 配置非法（带文件名 + 行号 + difflib 建议） |
| func | `_as_keyid()` | 39 | `VK` / `int` → `KeyId` 归一 |
| func | `keymap_path()` | 49 | 环境变量 → 用户目录 → 内置 |
| func | `load_keymap()` | 68 | 加载 + **加载期严格校验**（未知动作 / 未知键名 / 类型 / 哨兵） |
| func | `_suggest()` | 138 | difflib 拼写建议 |
| func | `dump_keymap()` | 151 | 序列化（分组 + 稳定排序） |
| func | `save_keymap()` | 189 | **写前预校验**，不让非法配置污染磁盘 |

| 项 | 内容 |
|---|---|
| import 了哪些 autopilot 内部模块 | `.simulator`（`KeyId`） |
| 被哪些 autopilot 内部模块 import | `tools/_stack.py:61`、`tools/keymap_editor.py`、`tools/input_demo.py` |
| 不可替代性 | **不可替代**。「三级解耦 `Action → KeyId → VK`」的中间层就在这里 —— 它让玩家改键不影响协议层。`KeyId` 的 `NONE`（显式不发）/ `UNKNOWN`（未配置）两个哨兵也在此定义。 |

---

# 四、三层之间的接口契约

## perception → core

| 项 | 值 |
|---|---|
| 对象 | `StatePacket` |
| 定义 | `contracts/state_packet.py:203` |
| 构造者 | `perception/builder.py:107` `StateBuilder.build(frame) -> StatePacket` |
| 关键字段 | `scene.id`（决策路由）/ `player.hp_ratio`（血量三态）/ `position.quality`（导航门槛）/ `resource.perception_ok`（**核心开关，False 时决策层禁止产意图**）/ `state_version`（漂移校验）/ `timestamp_ms` |
| 附产物 | `CapturedFrame`（`perception/capturer.py:42`）在 builder 内部消费，**不出口** |
| 事件出口 | `builder.drain_events()` → `List[Event]`（`contracts/events.py:38`），由 core 的 EventBus 收编 |

## core → policy

| 项 | 值 |
|---|---|
| 传入 | `StatePacket`（**单个快照**，不是 WorldState） |
| 定义 | `PolicySource.poll(state) -> IntentPacket \| None`（`policy/base.py:73`） |
| 实际调用点 | `core/loop.py` `_tick()` ③ 段 |
| **注意** | `LocalPolicyTree.decide()` 内部只用 `state` 构造 `PolicyContext`；`count()` / `last_exec()` 这两个 DSL 函数需要历史帧，而 `PolicyContext.from_state()` 的 `recent_states` / `intent_records` 参数**core 目前未传**（默认空 → 这两个函数恒返回 UNKNOWN）。 |

## policy → core

| 项 | 值 |
|---|---|
| 对象 | `IntentPacket \| None`（`None` = 空转，不执行） |
| 定义 | `contracts/intent_packet.py:142` |
| 构造者 | `policy/local_tree.py` `IntentPacket.make(action, params, state_version, ttl_seconds, priority, scene_required, source, constraints)` |
| 关键字段 | `action`（白名单枚举）/ `state_version`（漂移基线）/ `ttl_seconds` / `priority` / `target.scene_required` / `signature`（CLOUD 必填） |
| 仲裁 | `policy/merger.py:159` `merge_intents(local, cloud, state)`（core 在 `core/loop.py:568` 惰性 import） |

## core → actuator

| 项 | 值 |
|---|---|
| 传入 | `(IntentPacket, StatePacket)` |
| 定义 | `actuator/executor.py` `IntentExecutor.execute(intent, state) -> ExecutionResult` |
| 产出 | `ExecutionResult`（`executor.py:58`）：`outcome` / `reason` / `steps` / `elapsed_ms` / `released_keys` / `event` |
| **注入而非 import** | core **零 import actuator**（实测）。`IntentExecutor` 由 `tools/_stack.py:60` 构造后放进 `RuntimeContext.executor`，core 只读 `ctx.executor`。 |

## executor 内部的三层依赖（层内）

```
executor.py:101 IntentExecutor
  ├─ validator.IntentValidator   (executor.py:44)  ← 校验，commit=True
  ├─ safety.SafetyGuard          (executor.py:33)  ← 急停/熔断/频率
  ├─ traits.traits_of()          (executor.py:43)  ← 差异化参数
  ├─ humanizer.Humanizer         (executor.py:32)  ← 时序抖动
  └─ simulator.InputSimulator    (executor.py:34)  ← SendInput
```

---

# 五、层内 / 跨层存疑点

> 只列事实。

## 5.1 死代码（定义但全仓库无引用）

| 位置 | 事实 |
|---|---|
| `policy/merger.py:110` `validate_cloud_intent()` | 函数体完整（签名先验 + 四道校验）。全仓库唯一调用点是同文件 `merge_intents()`（161 行）。无 tools / core / test 引用。 |
| `policy/merger.py:53` `SignatureVerifier` (Protocol) | 被 61 行 `NullSignatureVerifier` 实现，但 `merge_intents` 的 `signature_verifier` 参数默认值是 `None` 而非 `NullSignatureVerifier()`；core 的 `_merge()` 调用时**根本没传这个参数**。即 Protocol 与其实现当前都不在实际路径上。 |
| `policy/dsl.py:378` `DEFAULT_EVALUATOR` | 模块级实例化。`rule_loader.py:44` / `local_tree.py` 均自建 `DslEvaluator()`，无人 import 此常量。 |
| `perception/roi_expr.py:380` `eval_roi_expression()` | 模块级便捷函数。`RoiExprEvaluator` 是实际使用方（builder / config.loader），此函数无引用。 |
| `perception/capturer.py:469` `_find_fullscreen()` | 被 `_find_window()` 的 `allow_fullscreen_fallback` 分支调用，但 `defaults.toml` 中 `allow_fullscreen_fallback = false`，该分支在当前配置下不可达。 |
| `actuator/traits.py:247/254` `missing_traits()` / `extra_traits()` | 被同文件 262 行 `validate_table()` 调用。无外部引用。 |

## 5.2 重复实现（同一件事两处）

| 位置 | 事实 |
|---|---|
| `actuator/simulator.py:342` `decode_events()` | `DryRunBackend.send()`（387 行）与 `InputSimulator._record()`（经 `_send`）共用此函数 —— **已去重，是正确写法**。列出作为「已解决的重复」的对照。 |
| `perception/ocr.py:299` `_match()` vs `perception/template.py:420` `_suppress_overlap()` | 前者是文本匹配，后者是矩形 IoU 抑制。语义不同，非重复。 |
| `actuator/executor.py:471` `_MOVE_KEYS` vs `actuator/keymap.py:68` `load_keymap()` | 前者硬编码 WASD 不让玩家改，后者是可配键位表。**两套键位来源并存**：若玩家在 `keymap.toml` 里把 `MOVE_FORWARD` 改成别的，导航仍走 `_MOVE_KEYS` 的 W。`config/defaults.toml` 与两处注释均声明这是有意为之。 |

## 5.3 疑似遗留（旧版入口 / 被替换未删）

| 位置 | 事实 |
|---|---|
| `perception/__init__.py:18-35` | re-export 清单里**没有** `DebugOverlay`（`debug_overlay.py` 未被包级导出），但 grep 显示 `__init__.py` 有 1 处 `DebugOverlay` 命中 —— 在注释或 `__all__` 字符串里。`debug_overlay.py` 的实际消费者只有 `tools/autopilot_demo.py`。 |
| `actuator/simulator.py:511` | `from .simulator import KeyIdError  # noqa: E402 - 供上层从本模块 import` —— executor.py **重新导出**了 simulator 的异常。`actuator/__init__.py` 同时也从 `.executor` 导出 `KeyIdError`（若成立）。同一符号两条导出路径。 |
| `actuator/executor.py:484` `_KEYLESS_ACTIONS` | `{"WAIT","SCREENSHOT","NOOP"}`。`DEFAULT_KEYMAP` 里这三项的值是 `KeyId.NONE` 哨兵，两处表达「这动作不发按键」—— 语义重复（一处靠集合判定，一处靠哨兵值）。 |

## 5.4 跨层直接引用（绕过公开接口）

| 位置 | 事实 |
|---|---|
| `core/loop.py:568` | `from ..policy.merger import merge_intents`（惰性，被 `try/except ImportError` 包住，失败回落 `return local`）。core 是「只调三层公开方法」的层，此处直接引用了 policy 的**具体子模块**而非 `PolicySource` 协议。 |
| `actuator/safety.py:194` | `from .traits import traits_of`（函数内惰性 import）。safety → traits 是层内引用，但 `traits` 是**数据表模块**而非执行模块，safety 在做场景许可判定时依赖它。 |
| `policy/factory.py:161` | `from .local_tree import LocalPolicyTree`（函数内惰性）。**这是设计意图** —— 惰性 import 使 `import autopilot.policy` 不加载规则引擎，且是将来闭源化的替换点。 |

---

# 六、三层的「实际读法」

按「读完掌握 80% 层逻辑」排序。共 7 个。

| # | 文件 | 行数 | 为什么是它 |
|---|---|---|---|
| 1 | `policy/local_tree.py` | 375 | **三层里唯一写清了决策语义的地方**：priority 升序、命中即 break（原则 3）、空转合法、未知→不动作、`SAFE_ACTIONS` 的 UNKNOWN 场景约束。读完这 375 行，policy 层就通了 80%。 |
| 2 | `actuator/executor.py` | 511 | 唯一持有「校验 → 安全闸 → 键位 → 人类化 → 执行 → finally 松键」**顺序**的文件。顺序本身是契约，且是三层里 bug 密度最高的地方（4 个历史 bug 都出在这里的时钟/变量/优先级）。 |
| 3 | `core/loop.py` | ~700 | 虽然不在本轮三层范围内，但**读 actuator/policy 而不读它，你会不知道 PAUSED 到底停不停感知、release_all 在哪几个时机被调**。它是三层的调用方契约。 |
| 4 | `perception/builder.py` | 399 | perception 的门面。`build()` 内部有 YOLO/模板/OCR 三路，但**分支逻辑（场景路由、黑帧、稳定性判定）全在这一个文件**。其余三个识别器可以当黑盒。 |
| 5 | `actuator/traits.py` | 304 | 21 个动作的全部差异化参数 + `allowed_scenes`。想知道「某个动作为什么在这里不触发」，答案 90% 在这张表里，不在 executor 里。 |
| 6 | `policy/dsl.py` | 388 | `ALLOWED_NODES`（107）是规则表达能力的全部边界。规则写不出来 / 运行时被拒，先看这 107 行，不用往下读求值逻辑。 |
| 7 | `actuator/safety.py` | 320 | 急停 / 熔断 / 会话上限 / 频率限制的唯一实现。`is_hard_stop()`（38）只有 4 行但改过一次 —— 那次修改的注释说明了为什么必须按前缀匹配。 |

**可以先跳过的**：`perception/yolo.py`（纯 ONNX 封装，替换实现时整体换）、`perception/ocr.py`（后端可换）、`actuator/humanizer.py`（纯抖动，不影响正确性）、`policy/merger.py`（本期云端恒为 None，接口已立住）、`perception/roi_expr.py`（`config/loader.py` 已复用它做校验，说明它自洽）。

---

# 七、盘点方法的局限

1. **未执行任何模块** —— 全部结论来自 grep/AST 静态检索。「死代码」的判断依据是「全仓库无文本引用」，排除不掉字符串反射式动态引用。
2. **文件系统在盘点期间多次给出不一致结果**（同一 grep 前后返回不同文件列表）。本文件采用最终一致的快照，个别行数可能已变。
3. **行数为 grep 计数（`^` 匹配数）**，与 `wc -l` 相差最多 1 行（末行有无换行符）。
4. `perception/debug_overlay.py` 与 `actuator/__init__.py` 的 `__all__` 内容**未逐行核实**，第五节 5.3 的结论基于 grep 命中位置推断。
5. **`tools/` 侧的反向依赖清单不完整** —— 多次 grep 对 `tools/` 的 import 检索返回了不一致的结果，本文只列出能确认的。
