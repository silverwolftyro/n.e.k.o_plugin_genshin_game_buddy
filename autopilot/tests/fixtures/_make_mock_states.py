"""生成 200 帧 mock 状态序列（不截图、不连实机）。

覆盖多种场景切换 + 边界值，供 replay.py 验证决策逻辑：
  * 战斗全血 -> 低血（触发闪避/切治疗/大招/技能）
  * 背包打开/关闭
  * 地图打开
  * 定位 GOOD / DEGRADED / LOST
  * 感知不健康的帧
  * 场景 UNKNOWN（只允许安全意图）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(r"D:\workspace\N.E.K.O\NEKO-genshin\N.E.K.O-main\N.E.K.O-main")
PLUGIN_ROOT = REPO / "plugin" / "plugins" / "genshin_game_buddy"
sys.path.insert(0, str(PLUGIN_ROOT))

from autopilot.contracts.state_packet import (  # noqa: E402
    PartyMember, PlayerInfo, PositionInfo, ResourceInfo, SceneInfo, StatePacket,
)

NOW = 1_700_000_000_000
FRAME_MS = 200                      # 5 Hz
FRAMES = 200


def party(name, skill, burst):
    return (PartyMember(1, name.lower(), name, True,
                        {"skill": skill, "burst": burst}),)


def frame(i: int) -> StatePacket:
    """第 i 帧（0-based）的状态。分段脚本。"""
    t = NOW + i * FRAME_MS
    scene = "COMBAT"
    hp, energy, cd = 0.8, 0.0, party("香菱", False, False)
    pos = PositionInfo(quality="GOOD", x=100.0, y=200.0)
    perception_ok = True
    conf = 0.95

    # ── 分段脚本 ───────────────────────────────────────────────
    if i < 30:                       # 战斗：满血无资源
        pass
    elif i < 45:                     # ★ 满血 + 满能量 + 大招好
        # 这一段专门覆盖 combat_burst_full_energy。早先这几帧血量是 0.28，
        # heal_switch(priority 20) 永远压在大招(30)前面 —— 优先级是对的，
        # 但数据里就永远测不到大招分支了。
        hp, energy = 0.95, 1.0
        cd = party("香菱", True, True)
    elif i < 75:                     # 战斗中血量下滑
        hp = 0.45 - (i - 45) * 0.01
    elif i < 90:                     # 低血量 + 大招好了
        hp, energy = 0.28, 1.0
        cd = party("香菱", True, True)
    elif i < 100:                    # 极低血
        hp, energy = 0.15, 0.5
        cd = party("香菱", True, False)
    elif i < 105:                    # 背包
        scene, hp, energy = "BAG_INVENTORY", 0.9, 0.0
    elif i < 115:                    # 地图
        scene, hp = "MAP_BIG", 0.9
    elif i < 135:                    # 大世界探索，定位 GOOD
        scene, hp = "MAIN_MENU", 0.9
    elif i < 150:                    # 定位降级
        scene = "MAIN_MENU"
        pos = PositionInfo(quality="DEGRADED", x=100.0, y=200.0)
    elif i < 165:                    # 定位丢失（不该移动）
        scene = "MAIN_MENU"
        pos = PositionInfo(quality="LOST", x=None, y=None)
    elif i < 175:                    # 场景识别不出来（只允许安全意图）
        scene, conf = "UNKNOWN", 0.3
    elif i < 185:                    # 感知不健康 -> 不该产出任何意图
        scene, perception_ok = "COMBAT", False
    elif i < 195:                    # ★ 战斗结算
        scene, hp, energy = "COMBAT_RESULT", 0.6, 0.8
    else:                            # 回到战斗收尾
        scene, hp, energy = "COMBAT", 0.6, 0.8
        cd = party("香菱", True, False)

    return StatePacket(
        scene=SceneInfo(id=scene, confidence=conf, entered_ms=t - 5000),
        player=PlayerInfo(hp_ratio=hp, hp_source="geometry",
                          energy=int(energy * 100), energy_max=100,
                          party=cd, resin=140),
        position=pos,
        resource=ResourceInfo(fps=59.5, tick_ms=16, perception_ok=perception_ok,
                              ocr_available=True),
        timestamp_ms=t, state_version=i + 1,
    )


def main() -> int:
    out = REPO.parent / "mock_states.jsonl"
    rows = [json.dumps(frame(i).to_dict(), ensure_ascii=False)
            for i in range(FRAMES)]
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"已写 {FRAMES} 帧 -> {out}")
    print(f"文件大小 {out.stat().st_size} 字节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
