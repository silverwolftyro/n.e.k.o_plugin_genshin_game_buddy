# N.E.K.O. Autopilot 3.0 —— 第 1 期 MVP 骨架

> 读屏 + 决策 + 模拟输入。本目录是**第 1 期（感知层）**的可运行骨架 + 第 2~4 期的数据契约预留。
> 设计依据：《N.E.K.O. 3.0 架构设计文档 v1.0》第 4 章（分层）、第 5 章（数据契约）、第 6 章（模块设计）、第 7 章（设计原则）。

---

## 🔴 先读这一条：许可证红线

| 组件 | 许可证 | 能否进运行时 |
|---|---|---|
| N.E.K.O. 本体 | Apache-2.0 | — |
| **ultralytics** | **AGPL-3.0** | ❌ **绝对禁止 import** |
| onnxruntime | MIT | ✅ |
| opencv / numpy / mss / pydantic | Apache-2.0 / BSD / MIT | ✅ |

**运行时推理只走 ONNX + onnxruntime。** 训练用 ultralytics 属于开发工具、不分发，
且**不要**把它写进 `requirements-autopilot.txt`。

`check_env.py` 会用 **AST 解析**（不是文本 grep）扫描整个运行时包，
一旦发现真的 `import ultralytics` 就报红。

---

## 快速开始

```bash
cd plugin/plugins/genshin_game_buddy

# 1) 装依赖（不含 ultralytics）
pip install -r requirements-autopilot.txt

# 2) 环境自检 ← 第一条命令
python -m autopilot.tools.check_env --title 原神

# 3) 只截图（最小可跑）
python -m autopilot.tools.capture_demo --title 原神

# 4) 裁模板图
python -m autopilot.tools.capture_demo --title 原神 --save work/shot.png
#    用 X-AnyLabeling 裁出目标，存到 autopilot/perception/templates/paimon_menu.png
#    在 config/defaults.toml 配好 [autopilot.templates.PaimonMenu]

# 5) 验证模板匹配
python -m autopilot.tools.template_demo --title 原神

# 6) 完整感知循环（截图→YOLO→模板→状态包→叠加层），不执行任何输入
python -m autopilot.tools.autopilot_demo --title 原神 --out work/states.jsonl
```

训练与评估（等有数据后）：

```bash
python -m autopilot.tools.check_dataset --data D:/NEKO/yolo_data   # 先查脏数据
pip install ultralytics                                          # 仅训练
python -m autopilot.tools.train_yolo  --data D:/NEKO/yolo_data    # 训完自动导出 ONNX
python -m autopilot.tools.eval_yolo    --data D:/NEKO/yolo_data    # 用 onnxruntime 评估
python -m autopilot.tools.convert_to_onnx --weights runs/detect/train/weights/best.pt
```

所有工具都支持 `python autopilot/tools/xxx.py` 直接运行（会自动 bootstrap 路径）。

---

## 目录结构

```
autopilot/
├── contracts/               数据契约（frozen dataclass + JSON Schema）
│   ├── state_packet.py      StatePacket 及子结构
│   ├── intent_packet.py     ActionType 白名单 + IntentPacket + is_valid_for
│   ├── events.py            Event + 工厂方法（带诊断信息）
│   └── schemas/*.json       JSON Schema
│
├── perception/              感知层（截图 → 识别）
│   ├── capturer.py          三通道截图 + RLock
│   ├── screen_region.py     链式裁剪，坐标双向可推
│   ├── roi_expr.py          ROI 表达式（ast 白名单）
│   ├── template.py          模板匹配 + NMS + StableValueAccumulator
│   ├── yolo.py              ONNX 推理（禁止 ultralytics）
│   ├── ocr.py               文字识别（复用 2.0 WinRT 引擎）
│   ├── builder.py           融合 → StatePacket（场景路由）
│   └── debug_overlay.py     画框（复用本帧真实产出）
│
├── state.py                 WorldState 不可变快照 + WorldStateStore
│
├── config/
│   ├── loader.py            pydantic 校验（加载期响亮失败）
│   └── defaults.toml        默认配置（示例合一）
│
├── tools/                   8 个独立可执行工具
│   ├── check_env.py         环境自检（含许可证 AST 扫描）
│   ├── capture_demo.py      只截图
│   ├── template_demo.py     只模板匹配
│   ├── autopilot_demo.py    完整感知循环
│   ├── check_dataset.py     数据集质量检查
│   ├── train_yolo.py        训练 + 导出 ONNX
│   ├── eval_yolo.py         评估（onnxruntime）
│   └── convert_to_onnx.py   单独导出
│
└── perception/models/       ← YOLO ONNX 放这里（train_yolo 自动复制）
    perception/templates/    ← 模板 PNG 放这里
```

---

## 已实现 vs 未实现

| 模块 | 状态 |
|---|---|
| 三通道截图（dxcam / bitblt / mss） | ✅ |
| ROI 表达式 + 锚点自适应 | ✅ 表达式部分；锚点换算在 v2 |
| 模板匹配 + NMS + 抗抖稳定计数 | ✅ |
| YOLO ONNX 推理 | ✅（需自行训练模型） |
| OCR（复用 2.0 WinRT） | ✅ 骨架 |
| 状态包构建 + 场景路由 | ✅ 骨架 |
| 调试叠加层 | ✅ |
| 配置校验 | ✅ |
| WorldState 不可变快照 | ✅ |
| 意图包契约 + 校验 | ✅（决策层未实现） |
| **定位（小地图两阶段匹配）** | ❌ 第 1 期不做，`position` 恒为 `null` |
| **决策层（if/elif + 条件 DSL）** | ❌ 第 3 期 |
| **执行层（SendInput）** | ❌ 第 2 期 |
| **通信层（WebSocket）** | ❌ 第 4 期 |

> v1 的 `position` 恒为 `null` 是**故意的**——不填比填一个猜的值安全
> （原则 1：响亮失败优于静默走错；原则 2：Unknown 不乐观）。

---

## 设计原则在代码里的落点

| 原则 | 落点 |
|---|---|
| **响亮失败** | 配置加载期 `validate_config()` 抛错；`StatePacket.unknown()` 显式标记 `perception_ok=False`；`Event` 工厂带诊断 |
| **三态 + Unknown** | `PartyMember.cd_ready: dict[str, bool \| None]`；`OcrResult.matched: TriState`；`PositionInfo.quality` |
| **一轮只截一次图** | `StateBuilder.build(frame)` 接收外部帧，内部不重截 |
| **工具 = 运行时** | `DebugOverlay` 读 `builder.last_detections/last_matches`，不重跑识别 |
| **ROI 表达式** | `roi_expr.py`，`ast` 白名单（拒绝 Attribute/Subscript/Lambda） |
| **抗抖 + 诊断** | `StableValueAccumulator` 携带 `max_consecutive_count` / `required_count` |
| **静默失败禁令** | OCR 失败进 `fail_streak` 并产生事件；YOLO 超时丢帧 + 事件；模板连续低分带诊断 |

---

## 已知限制

1. **`position` 未实现** —— `v1` 不做通用定位，只在第 3 期考虑有限场景。
2. **模板不随分辨率自动缩放** —— ROI 表达式负责 ROI 自适应，模板本身建议按当前分辨率裁。
3. **纯色模板匹配不到** —— 方差为零时 CCOEFF 算出 0/0。`load_template` 会告警，
   请重新裁一块有结构的区域。
4. **`derive_by_scale` 不跟踪缩放因子** —— 坐标回推在缩放层以下是近似的，
   缩放只用于喂模型。
5. **YOLO 未训练** —— 默认 `enabled=false`，场景识别退化为单模板探测。

---

## 相关文档

- 架构设计：`D:\NEKO\N.E.K.O.3.0-架构设计文档-v1.0.md`
- 参考项目逆向：`D:\NEKO\BetterGI-技术路线逆向报告.md`、`D:\NEKO\MaaEnd-技术路线逆向报告.md`
