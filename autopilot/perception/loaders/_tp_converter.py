"""BetterGI ``tp.json`` 格式 -> 本项目 schema 的**字段映射**。

★ 这个文件是"怎么转换"，不是"转换什么"
--------------------------------------
里面**只有字段名映射，没有任何坐标、名称、区域的具体内容**。
所有数据都来自调用方传入的 ``raw``。

★ 为什么字段名要故意不一样
-------------------------
直接沿用 ``sceneId`` / ``tranPosition`` 这种源格式的名字，
等于把源数据的 schema 缝进我们的代码里 —— 将来换数据源就得改一遍。
换一套自有命名，映射就集中在这一个文件里，换源只改这里。

★ 合规
------
BetterGI 是 GPL-3.0。**读取**其数据做互操作研究，与**复制/内嵌**其内容
是两件事。本模块只描述结构转换，不携带任何数据。
分发口径见 ``recognition/notes/LICENSING.md``。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: 本项目 schema 的版本号。写进输出，供消费方判断是否需要迁移。
SCHEMA_VERSION = 1

#: 数据来源标识。**这是元信息，不是数据** —— 说明这份东西从哪来，
#: 便于日后审计和替换。
SOURCE_BETTERGI = "BetterGI:GameTask/AutoTrackPath/Assets/tp.json"

#: 源格式 -> 本项目格式的键名映射。一一对应，转换只认这张表。
_REGION_KEYS = {
    "sceneId": "region_id",
    "description": "display_name",     # 缺则回退 mapName
}
_REGION_FALLBACK_NAME_KEY = "mapName"
_POINTS_KEY = "points"
_REGION_KEY_OUT = "points"            # 源里叫 points，本项目里也叫 points

_POINT_KEYS = {
    "id": "id",
    "type": "category",
    "name": "name",
    "country": "country",
    "areas": "areas",
    "position": "world_pos",          # 传送点本体坐标
    "tranPosition": "landing_pos",     # 传送**落点**坐标（与本体不同）
}

#: 三个坐标分量的标准命名（3D：x / y=高度 / z）。**当前仅作约定说明**，
#: 转换逻辑不依赖它 —— 保持列序即 x,y,z。
#:
#: ⚠️⚠️ **这里的 ``y`` 是「高度」，与 `PositionInfo.y` 的「地面平面第二轴」
#: 不是一回事**（2026-10-14 轴名统一后必须记一笔）。
#:
#: 轴名统一决定的是**平面（ground plane）**用 ``x`` / ``y``：
#: ``PositionInfo`` / ``IntentTarget.world_position`` / 事件 context 里
#: 的 ``y`` 存的都是平面第二坐标（它的值来自世界 z）。
#: 而本文件的传送点是**三维**数据，三个分量是 x / 高度 / z —— 这里的
#: ``y`` 是海拔。
#:
#: 所以本文件**不在**轴名统一的范围内，**不要**把它改成 ``x`` / ``y``：
#: 那会把「海拔」和「地面北南向」在同一个仓库里绑成同一个名字，
#: 而两者在物理上毫无关系。
_COORD_NAMES = ("x", "y", "z")


def _vec3(value: Any) -> list[float] | None:
    """规整成 3 元浮点列表；形状不对返回 None（**不猜、不补零**）。"""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value]


def _convert_point(src: dict) -> dict:
    """单个传送点：映射已知键，其余一律收进 ``extra``。

    ★ "消费"的判定标准是**转换成功**，不是"键在表里"
    ------------------------------------------------
    比如 ``position`` 存在但不是 3 元数组 —— 这时既不写 ``world_pos``，
    也不该让原始值凭空消失，它会原样落进 ``extra``。
    映射失败和"没这个键"必须能被下游区分开。
    """
    out: dict[str, Any] = {}
    consumed: set[str] = set()

    for src_key, dst_key in _POINT_KEYS.items():
        if src_key not in src:
            continue
        val = src[src_key]

        if dst_key in ("world_pos", "landing_pos"):
            v = _vec3(val)
            if v is None:
                continue                    # 形状不对 -> 留给 extra
            out[dst_key] = v
        elif dst_key == "areas":
            if not isinstance(val, list):
                continue                    # 不是数组 -> 留给 extra，
            out[dst_key] = _str_list(val)   # 不伪造一个空列表
        elif dst_key == "id":
            # 源里 id 有时是 int 有时是 str，统一成 str 便于做主键
            out[dst_key] = str(val)
        else:
            out[dst_key] = val
        consumed.add(src_key)

    # ★ 差集收容
    #   源里有、且没被**成功**映射的键，一个不落地进 extra。
    #   不再靠硬编码字段白名单 —— 源格式加字段时数据不会静默消失，
    #   这是"收编外部数据"必须守的底线。
    out["extra"] = {k: v for k, v in src.items() if k not in consumed}
    return out


def _convert_region(src: dict) -> dict:
    out: dict[str, Any] = {}
    for src_key, dst_key in _REGION_KEYS.items():
        if src_key in src:
            out[dst_key] = src[src_key]
    if "display_name" not in out:
        # 源里 description 缺失时回退到地图代号（如 "Teyvat"）
        if _REGION_FALLBACK_NAME_KEY in src:
            out["display_name"] = str(src[_REGION_FALLBACK_NAME_KEY])
    if "region_id" in out:
        out["region_id"] = str(out["region_id"])
    pts = src.get(_POINTS_KEY)
    out[_REGION_KEY_OUT] = ([_convert_point(p) for p in pts
                             if isinstance(p, dict)] if isinstance(pts, list)
                            else [])
    return out


def convert_bettergi_format(raw: dict) -> dict:
    """把 BetterGI ``tp.json`` 的 dict 转成本项目 schema 的 dict。

    :param raw: 已解析的 JSON 对象
    :returns: ``{"version", "source", "regions": [...]}``

    ★ 未被成功映射的键一律进 ``extra``，不丢弃
      （以 ``src`` 与 ``consumed`` 的差集为准，不是硬编码清单。）
    """
    if not isinstance(raw, dict):
        raise TypeError(f"raw 应为 dict，实际 {type(raw).__name__}")
    data = raw.get("data")
    regions = ([_convert_region(r) for r in data if isinstance(r, dict)]
               if isinstance(data, list) else [])
    if not isinstance(data, list):
        logger.warning("源数据缺少 data 数组，转换出 0 个区域")
    return {
        "version": SCHEMA_VERSION,
        "source": SOURCE_BETTERGI,
        "regions": regions,
    }


__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_BETTERGI",
    "convert_bettergi_format",
]
