"""队伍栏头像分类器（ONNX）—— ``character_detector`` 的**降级**路径。

★ 它排在什么位置
-----------------
队伍栏每个槽位有**两条**可读的信息：

1. 头像左边那行**角色名文字** —— 现有 :mod:`character_detector` 用 OCR 读它。
   实测 296 帧：单帧误识 **0.00%**，时序滤波后稳态 **100%**。
2. **头像本身** —— 这个模块读它。

OCR 那条路**已经跑通且够好**，所以本模块是**降级**：只在 OCR 读不出名字时
才启用。理由很直接：OCR 失败通常意味着「切屏 / 开菜单 / 队伍栏虚化」，
那时候头像也跟着一起糊 —— **两条路会同时失效**，不是互补关系。
真正互补的场景只有一个：**开了角色详情面板、名字被遮挡**。

🔴 所以本模块默认**关着**（``autopilot.avatar_classifier.enabled = false``）
--------------------------------------------------------------------------
除了上面那条理由，还有更硬的一条：这套权重的真实准确率见
``character_classifier.onnx.provenance.json`` 的 ``real_top1``。
在它低于可用线之前，开着只会**制造错误的名字**（比读不到更糟：
``onfield()`` 会拿一个错名字去判连招）。
``min_confidence`` 就是那道闸 —— 低于它就返回 UNKNOWN。

★ 模块顶层不碰 onnxruntime
-------------------------
引擎的 onnxruntime 是**函数内懒加载**（见 :mod:`world_detector`）。
照做，否则 import 这个包就会拉起 DirectML provider，
而那在没装 DirectML 的机器上会直接炸。

★ 没有模型文件时的行为
---------------------
返回 ``None``，**不抛**。调用方据此跳过。
这是可选功能，缺它不该让感知层少一条信息。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: 默认模型路径（相对插件根）。发布包里的完整路径见 build_engine.py。
DEFAULT_REL_PATH = Path("recognition") / "models" / "character_classifier.onnx"
#: 训练脚本写出的清单（含类别表与归一化参数）。
PROVENANCE_SUFFIX = ".provenance.json"


@dataclass(frozen=True)
class AvatarPrediction:
    """一个头像的分类结果。"""

    name: str = ""           # 角色名；``""`` = 读不出来 / 置信度不够
    confidence: float = 0.0  # softmax 最大值
    top5: tuple[tuple[str, float], ...] = ()

    @property
    def known(self) -> bool:
        return bool(self.name)


class AvatarClassifier:
    """ONNX 头像分类器。**一次加载，全程复用**。"""

    def __init__(self, session: Any, classes: list[str],
                 img_size: int, mean, std) -> None:
        self._sess = session
        self._classes = classes
        self._size = int(img_size)
        self._mean = np.asarray(mean, np.float32).reshape(1, 1, 3)
        self._std = np.asarray(std, np.float32).reshape(1, 1, 3)
        self._input = session.get_inputs()[0].name
        self._output = session.get_outputs()[0].name

    # -- 构造 ---------------------------------------------------------
    @classmethod
    def load(cls, onnx_path: str | Path) -> "AvatarClassifier | None":
        """加载模型。**任何问题都返回 None 并记日志**，不抛。

        ★ 为什么这么宽松
          这是可选功能。加载失败（文件没随包发出来、onnxruntime 版本对不上、
          模型被换过）都不该让感知层整条 tick 崩 —— 崩了就退回 OCR 那条路，
          那条路本来就够用。
        """
        p = Path(onnx_path)
        if not p.is_file():
            logger.info("没有头像分类器模型（%s），该通道保持关闭", p)
            return None
        prov_path = Path(str(p) + PROVENANCE_SUFFIX)
        if not prov_path.is_file():
            # ⚠️ 没有清单就没法把 logit 映射回角色名，模型等于废的。
            logger.warning("模型 %s 缺少 %s 清单，无法使用", p, PROVENANCE_SUFFIX)
            return None
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
            classes = list(prov["classes"])
            img_size = int(prov["img_size"])
            mean = prov["norm_mean"]
            std = prov["norm_std"]
        except Exception:  # noqa: BLE001
            logger.warning("头像分类器清单读取失败：%s", prov_path, exc_info=True)
            return None

        try:
            import onnxruntime as ort        # noqa: PLC0415 - 故意函数内导入
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        except Exception:  # noqa: BLE001
            logger.warning("onnxruntime 不可用或模型加载失败，头像分类器关闭",
                           exc_info=True)
            return None

        logger.info("头像分类器已加载：%d 类 / %d px（真实 top1=%.1f%%）",
                    len(classes), img_size,
                    float(prov.get("real_top1", -1)) * 100)
        return cls(sess, classes, img_size, mean, std)

    # -- 推理 ---------------------------------------------------------
    def classify(self, crop: np.ndarray, *, top_k: int = 5) -> AvatarPrediction:
        """分类一个头像裁切。**永远不抛**。"""
        if crop is None or getattr(crop, "size", 0) == 0:
            return AvatarPrediction()
        try:
            img = cv2.resize(crop, (self._size, self._size),
                             interpolation=cv2.INTER_AREA).astype(np.float32)
            img = (img / 255.0 - self._mean) / self._std
            x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
            logits = self._sess.run([self._output], {self._input: x})[0][0]
            e = np.exp(logits - logits.max())
            prob = e / max(e.sum(), 1e-9)
            k = min(top_k, len(self._classes))
            idx = np.argsort(-prob)[:k]
            pairs = tuple((self._classes[int(i)], float(prob[int(i)]))
                          for i in idx)
        except Exception:  # noqa: BLE001 - 逐帧推理失败不该掀翻整轮 tick
            logger.warning("头像分类推理异常，本槽位按 UNKNOWN 处理",
                           exc_info=True)
            return AvatarPrediction()
        return AvatarPrediction(name=pairs[0][0] if pairs else "",
                                confidence=pairs[0][1] if pairs else 0.0,
                                top5=pairs)


def resolve_model_path(explicit: str = "", plugin_root: Path | None = None
                       ) -> Path:
    """定位模型文件。

    :param explicit: 配置里写的绝对/相对路径（优先）
    :param plugin_root: 插件根目录。``None`` 时用本文件向上四级。
    """
    if explicit:
        q = Path(explicit)
        if q.is_absolute():
            return q
        root = plugin_root or Path(__file__).resolve().parents[3]
        return root / q
    root = plugin_root or Path(__file__).resolve().parents[3]
    # 发行包里 models/ 在插件根下面；源码树里在 <repo>/recognition/models。
    for cand in (root / DEFAULT_REL_PATH,
                 root.parent / DEFAULT_REL_PATH,
                 root / "models" / "character_classifier.onnx"):
        if cand.is_file():
            return cand
    return root / DEFAULT_REL_PATH


__all__ = [
    "DEFAULT_REL_PATH",
    "PROVENANCE_SUFFIX",
    "AvatarClassifier",
    "AvatarPrediction",
    "resolve_model_path",
]
