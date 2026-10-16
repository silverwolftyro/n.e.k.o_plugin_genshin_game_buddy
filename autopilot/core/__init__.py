"""N.E.K.O. Autopilot 主循环。

这是**三层之上**的唯一一层，只做四件事：

    1. 调三个方法：perception.build() / policy.poll() / executor.execute()
    2. 按状态机流转（IDLE/RUNNING/PAUSED/STOPPING/STOPPED）
    3. 把各层的事件汇到事件总线
    4. 聚合运行统计

它**不做**识别、不做规则求值、不做按键组装（见 loop.py 的详细说明）。
主循环是唯一三层都要碰的文件，一旦它长出业务逻辑，每次调整都会
牵动全部三层。

    from autopilot.core import CoreLoop, RuntimeContext, RuntimeState

    ctx = RuntimeContext(capturer=cap, builder=b, policy=p,
                         executor=ex, guard=guard, tick_rate_hz=5)
    with CoreLoop(ctx):
        ...          # 在这里跑
    print(ctx.stats.snapshot())

⚠ **导出面刻意只有两项**。``RuntimeContext`` / ``LoopStats`` /
``EventBus`` 都需要 ``CoreLoop`` 才能用得上，从包里再导一遍只是
多点一次。而 ``autopilot.core.stats`` / ``.events`` 这些子模块始终
可用 —— 需要时直接 import，不靠包根转发。

⚠ 本包**只有运行期代码会 import 三层**。离线工具（replay / rule_test /
policy_diagnose）一律不碰这里 —— 它们直接用 policy 层，不需要循环。

**不 import ultralytics**（AGPL）—— 感知层的推理走 ONNX。
"""

from __future__ import annotations

from .loop import CoreLoop
from .runtime import RuntimeState

__all__ = [
    "CoreLoop",
    "RuntimeState",
]
