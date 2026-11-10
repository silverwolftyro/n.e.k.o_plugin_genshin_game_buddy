# -*- coding: utf-8 -*-
"""打包卫生回归断言（2026-11-09 新增）。

为什么单独立一个文件
--------------------
``tests/conftest.py`` 里 ``SCRATCH_ROOT`` 的注释过去写的是「打包前必须删掉
``.pytest-tmp``」「即便目录是空的也一样会触发」。那句话描述的是**更早的
builder** 行为（暂存树检查不认点前缀）。现在的 builder 流程是：

1. 先按 ``[tool.neko.build]`` 的规则把插件目录复制成**暂存树**；
2. metadata 探测跑在**暂存副本**上（见
   ``plugin/neko_plugin_cli/core/metadata_probe.py`` 的 ``target_dir``）；
3. 暂存树里若有「装不进包」的空目录，就**跳过 metadata 推导**。

``.pytest-tmp`` 已列在 ``exclude_dirs`` 里，所以它**不进暂存树**，空目录也
不会因此出现 —— 打包前**不需要**再删它。

真正的风险不是「目录是不是空的」，而是**排除规则本身**
--------------------------------------------------------
一旦有人把 ``exclude_dirs`` 改掉（哪怕只是删掉一项），``.pytest-tmp`` 就会
重新进暂存树 → 暂存树出现空目录 → metadata 推导被**跳过** → 产物里
**静默**丢掉 ``plugin.meta.json`` → 而 ``check --release`` **仍然报绿**
（只多一条 WARN）。这比「打包直接红」更难发现：包能装上、能跑，只是少了
入口参数 schema，要等宿主运行时重扫才补得上。

所以这里把「规则不许被改」钉成红线：

* ``exclude_dirs`` 必须**至少**含 ``.pytest-tmp``、
  ``recognition/data/mihoyo_maps``、``recognition/data/mihoyo_tiles``；
* 用**宿主真函数**（``PluginBuilder.copy_plugin_runtime_files`` +
  ``should_skip_path``）复刻真实 staging，断言暂存树里没有空目录；
* 再用一份**被篡改的规则副本**（只改内存，不碰文件）证明上面这些断言
  确实会因为「排除项被删」而变红 —— 断言自己也得有牙齿。

本文件**不依赖**任何一次性实验脚本（例如 ``target/_batch1fix_stage_probe.py``），
可单独运行：``pytest tests/unit/test_packaging_hygiene.py``。
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

#: 插件目录：<引擎根>/plugin/plugins/genshin_game_buddy
PLUGIN_DIR = Path(__file__).resolve().parents[2]
#: 引擎根：<引擎根>/plugin/plugins/genshin_game_buddy -> parents[2]
ENGINE_ROOT = PLUGIN_DIR.parents[2]

# ---------------------------------------------------------------------------
# 宿主真函数
# ---------------------------------------------------------------------------
# 走 editable 安装的 .pth 时引擎根本来就在 sys.path 上；找不到时再显式补一次
# （不硬编码绝对路径，全部由本文件位置推导）。
try:  # pragma: no cover - 取决于安装形态
    from plugin.neko_plugin_cli.core.build import PluginBuilder
    from plugin.neko_plugin_cli.core.build_rules import load_build_rules, should_skip_path
    from plugin.server.infrastructure.packaged_metadata import empty_source_directories
except ModuleNotFoundError:  # pragma: no cover - 非 editable 安装时的兜底
    if str(ENGINE_ROOT) not in sys.path:
        sys.path.insert(0, str(ENGINE_ROOT))
    from plugin.neko_plugin_cli.core.build import PluginBuilder
    from plugin.neko_plugin_cli.core.build_rules import load_build_rules, should_skip_path
    from plugin.server.infrastructure.packaged_metadata import empty_source_directories


#: 必须留在 ``[tool.neko.build].exclude_dirs`` 里的目录。
#:
#: * ``.pytest-tmp`` —— 测试暂存根（tests/conftest.py 的 ``SCRATCH_ROOT``）。
#: * ``recognition/data/mihoyo_maps`` / ``recognition/data/mihoyo_tiles``
#:   —— 米游社地图与瓦片缓存（合计约 455 MB），禁止随包分发。
REQUIRED_EXCLUDE_DIRS: tuple[str, ...] = (
    ".pytest-tmp",
    "recognition/data/mihoyo_maps",
    "recognition/data/mihoyo_tiles",
)

#: 所有断言共用的一句人话：排除规则被改动意味着什么。
#:
#: 故意写成完整句子而不是短标签 —— 后来人 grep 到这条失败信息时，
#: 必须立刻明白「为什么一个目录规则会关系到 plugin.meta.json」。
RULE_TAMPER_HINT = (
    "排除规则被改动可能导致 plugin.meta.json 静默丢失："
    "`.pytest-tmp` 会重新进暂存树 → 暂存树出现空目录 → metadata 推导被跳过 → "
    "产物里没有 plugin.meta.json（入口参数 schema 要等宿主运行时重扫才补上），"
    "而 check --release 只多一条 WARN，门禁仍然是绿的。"
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _load_pyproject() -> dict[str, Any]:
    """读插件的 ``pyproject.toml``（原始 TOML，不做任何加工）。"""
    with (PLUGIN_DIR / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _build_rules() -> Any:
    """按宿主真函数解析出的打包规则集。"""
    return load_build_rules(_load_pyproject())


def _stage(source_dir: Path, destination_dir: Path, rules: Any) -> int:
    """用宿主**真** staging 实现复刻暂存树，返回复制进去的文件数。

    刻意调用 ``PluginBuilder.copy_plugin_runtime_files`` 而不是自己写一遍
    拷贝循环：它会**连目录一起复制**（空目录因此会出现在暂存树里），
    这正是 ``empty_source_directories`` 要检查的对象。自己写循环容易漏掉
    目录分支，漏掉之后断言就变成「平凡成立」，等于没有断言。
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = PluginBuilder().copy_plugin_runtime_files(
        source_dir,
        destination_dir,
        rules=rules,
    )
    return len(copied)


def _make_synthetic_source(root: Path, name: str = "src") -> Path:
    """造一棵最小合成源码树，覆盖三种危险形态。

    * ``.pytest-tmp/`` —— **空目录**（最坏形态：空目录照样打断打包）
    * ``recognition/data/mihoyo_maps/``、``.../mihoyo_tiles/`` —— 带文件的目录
    * ``pkg/`` —— 普通代码，必须**留在**暂存树里（防「一刀切全排」式误改）

    ``name`` 用来在同一根下区分多棵树（一个用例里往往要比「篡改前/篡改后」）。
    """
    source = root / name
    (source / ".pytest-tmp").mkdir(parents=True, exist_ok=True)
    (source / "recognition" / "data" / "mihoyo_maps").mkdir(parents=True, exist_ok=True)
    (source / "recognition" / "data" / "mihoyo_tiles").mkdir(parents=True, exist_ok=True)
    # recognition/data 下还有**不被排除**的正经文件（模型说明、许可清单等）。
    # 没有它，recognition/ 与 recognition/data/ 这两层会因为两个子目录都被
    # 排除而变成空目录 —— 那正是下面 test_recognition_inside_plugin_dir_would
    # 钉住的潜在缺口，不该混进「规则没问题」的对照组里。
    (source / "recognition" / "data" / "notes.md").write_text("notes\n", encoding="utf-8")
    (source / "pkg").mkdir(parents=True, exist_ok=True)
    (source / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (source / "pkg" / "data.bin").write_bytes(b"\x00")
    (source / "recognition" / "data" / "mihoyo_maps" / "P0.png").write_bytes(b"\x89PNG")
    (source / "recognition" / "data" / "mihoyo_tiles" / "t.bin").write_bytes(b"\x00")
    return source


# ---------------------------------------------------------------------------
# 断言
# ---------------------------------------------------------------------------
def test_exclude_dirs_contains_required_entries() -> None:
    """``exclude_dirs`` 必须至少含那三项，少一项就红。"""
    rules = _build_rules()
    missing = [item for item in REQUIRED_EXCLUDE_DIRS if item not in rules.exclude_dirs]
    assert not missing, (
        f"[tool.neko.build].exclude_dirs 缺少 {missing}；"
        f"当前 exclude_dirs={rules.exclude_dirs}。"
        f"{RULE_TAMPER_HINT}"
    )


def test_required_excluded_dirs_are_skipped_by_real_rules() -> None:
    """用宿主真规则引擎判定：那三个目录**必须**被判为要排除。

    这条断言直接顶在 ``should_skip_path`` 上，因此「规则项被删」与
    「规则项写错（大小写、斜杠、多余空格）」都会让它变红。
    """
    rules = _build_rules()
    leaked = [
        item
        for item in REQUIRED_EXCLUDE_DIRS
        if not should_skip_path(Path(item), is_dir=True, rules=rules)
    ]
    assert not leaked, (
        f"宿主真规则引擎没有排除 {leaked}；"
        f"当前 exclude_dirs={rules.exclude_dirs}。"
        f"{RULE_TAMPER_HINT}"
    )


def test_staged_tree_has_no_empty_directories(scratch: Path) -> None:
    """真实 staging 之后，暂存树里不得出现空目录。

    暂存树出现空目录 → metadata 推导被跳过 → 产物静默丢掉
    ``plugin.meta.json``。这条断言是整条链路的守门点。
    """
    rules = _build_rules()
    staged = scratch / "staged-real"
    _stage(PLUGIN_DIR, staged, rules)
    empties = empty_source_directories(staged)
    assert not empties, (
        f"暂存树出现装不进包的空目录 {empties}；metadata 推导会被跳过。"
        f"{RULE_TAMPER_HINT}"
    )


def test_excluded_dirs_never_reach_staged_tree(scratch: Path) -> None:
    """那三个目录连同其内容都不得出现在暂存树里。

    比「空目录为空」更直接：它同时守住**体积**（地图/瓦片约 455 MB）
    和**分发边界**（第三方站点抓取物不进包）。
    """
    rules = _build_rules()
    staged = scratch / "staged-synthetic"
    _stage(_make_synthetic_source(scratch, "src-clean"), staged, rules)

    assert not (staged / ".pytest-tmp").exists(), (
        f".pytest-tmp 进了暂存树；当前 exclude_dirs={rules.exclude_dirs}。"
        f"{RULE_TAMPER_HINT}"
    )
    assert not (staged / "recognition" / "data" / "mihoyo_maps").exists(), (
        "recognition/data/mihoyo_maps 进了暂存树（米游社地图，禁止随包分发）。"
    )
    assert not (staged / "recognition" / "data" / "mihoyo_tiles").exists(), (
        "recognition/data/mihoyo_tiles 进了暂存树（米游社瓦片，禁止随包分发）。"
    )
    # 反向保护：正常代码必须还在暂存树里，避免「一刀切全排」式误改。
    assert (staged / "pkg" / "__init__.py").is_file(), "正常代码没进暂存树，排除规则过头了。"


def test_assertions_go_red_when_exclusion_entry_is_removed(scratch: Path) -> None:
    """**断言自己也要有牙齿**：删掉一项排除规则，上面那些断言必须变红。

    这里只改**内存里的规则副本**（``model_copy``），不碰 ``pyproject.toml``，
    因此不存在「验证完忘了恢复」的副作用 —— 这正是把「规则被改则红」变成
    可持续回归的前提。
    """
    rules = _build_rules()
    tampered = rules.model_copy(
        update={
            "exclude_dirs": [item for item in rules.exclude_dirs if item != ".pytest-tmp"]
        }
    )
    assert ".pytest-tmp" not in tampered.exclude_dirs, "篡改副本没生效，测试本身有问题。"

    staged = scratch / "staged-tampered"
    _stage(_make_synthetic_source(scratch, "src-tampered"), staged, tampered)

    # 篡改后：.pytest-tmp 重新进暂存树，且（源里是空目录）触发空目录检查。
    assert (staged / ".pytest-tmp").exists(), "规则被删之后 .pytest-tmp 竟然没进暂存树？"
    assert ".pytest-tmp" in empty_source_directories(staged), (
        "规则被删之后暂存树竟然没有空目录 —— 那么上面两条断言就没有牙齿，"
        "整套回归是假的。"
    )

    # 对照：真规则下同样的输入必须是干净的。
    clean = scratch / "staged-clean"
    _stage(_make_synthetic_source(scratch, "src-clean"), clean, rules)
    assert not (clean / ".pytest-tmp").exists()
    assert not empty_source_directories(clean)


def test_recognition_inside_plugin_dir_would_leave_empty_parent_dirs(scratch: Path) -> None:
    """表征测试：把 ``recognition/`` 搬进插件目录，会留下两个空目录。

    这是本轮盘点发现的一个**已知缺口**，此处刻意把它钉住：

    ``exclude_dirs`` 只排了 ``recognition/data/mihoyo_maps`` 与
    ``recognition/data/mihoyo_tiles``。若日后 ``recognition/`` 真被搬进插件
    目录，这两条规则会让 ``recognition/data/`` 变成空目录，``recognition/``
    跟着也变空 —— 于是暂存树出现空目录 → metadata 推导被跳过 →
    ``plugin.meta.json`` 静默丢失，而门禁仍绿。

    今天它**不会**触发，因为 ``recognition/`` 是插件目录的**同级兄弟**，
    运行时经 ``NEKO_RECOGNITION_ROOT`` 引用，根本不在打包范围内。

    修法是给 ``exclude_dirs`` 再加 ``recognition``（整棵）—— 但那属于
    「修改排除清单」，超出本轮授权范围，故这里只做表征、并在决策清单里提出。
    本测试让缺口保持**可见**：谁要动排除规则，就必须先更新这里的期望值。
    """
    rules = _build_rules()
    source = scratch / "src-recognition-inside"
    (source / "recognition" / "data" / "mihoyo_maps").mkdir(parents=True)
    (source / "recognition" / "data" / "mihoyo_tiles").mkdir(parents=True)
    (source / "recognition" / "data" / "mihoyo_maps" / "P0.png").write_bytes(b"\x89PNG")
    (source / "recognition" / "data" / "mihoyo_tiles" / "t.bin").write_bytes(b"\x00")
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "__init__.py").write_text("", encoding="utf-8")

    staged = scratch / "staged-recognition-inside"
    _stage(source, staged, rules)

    empties = empty_source_directories(staged)
    assert empties == ["recognition", "recognition/data"], (
        f"recognition/ 搬进插件目录时的空目录集合变成了 {empties}（期望 "
        f"['recognition', 'recognition/data']）。"
        "若有人给 exclude_dirs 补了 'recognition'，请同步更新本条表征测试与决策清单。"
    )


@pytest.mark.parametrize("removed", list(REQUIRED_EXCLUDE_DIRS))
def test_each_required_exclusion_is_individually_load_bearing(
    removed: str,
    scratch: Path,
) -> None:
    """三项排除**各自**都承重：单独删任何一项，暂存树都会被污染。

    逐项验一遍，是因为三项的失效后果不同（``.pytest-tmp`` 伤 metadata，
    两项 recognition 伤体积与分发边界），不能只测其中一项。
    """
    rules = _build_rules()
    tampered = rules.model_copy(
        update={"exclude_dirs": [item for item in rules.exclude_dirs if item != removed]}
    )
    staged = scratch / f"staged-without-{removed.replace('/', '_')}"
    _stage(_make_synthetic_source(scratch, "src-tampered"), staged, tampered)

    leaked = (staged / Path(removed)).exists()
    assert leaked, (
        f"删掉 exclude_dirs 里的 {removed} 之后，暂存树居然没被污染 —— "
        f"这条排除项可能是装饰品。"
    )
