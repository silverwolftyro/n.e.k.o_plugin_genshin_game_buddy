"""Unicode 安全的图片读写（Windows）。

★ 为什么需要它
--------------
``cv2.imread`` / ``cv2.imwrite`` 在 Windows 上**不走 Unicode 路径 API**，
而是按系统 ANSI 码页（简体中文机器 = GBK）拼路径。结果是：

    cv2.imwrite(r"D:\\正常.png", img)   -> 文件**存成了乱码名**
                                          D:\\姝ｅ父.png
                                          p.exists() 是 False，glob 也匹配不到

而且 imread / imwrite **互相自洽**（都用 ANSI），所以「写得进去也读得回来」，
**不报任何错** —— 静默把文件名写坏。资源管理器里按原名找不到，
备份 / git / 清理脚本全部看不见它们。

本仓库至少有三个地方会踩到：

* ``capture_demo.py --save`` —— 用户给的路径
* ``template_cropper.py --image`` —— 用户给的截图；``--save-preview`` 同理
* ``template.py`` —— **模板图由用户命名**，起个中文名就中招

实测（2026-10-12，OpenCV 4.11.0 / Python 3.11 / Windows）::

    cv2.imwrite(".../正常显示_测试.png")  -> exists() == False
    cv2.imread(".../正常显示_测试.png")   -> 居然「成功」了（读到了同一个乱码文件）

所以正确做法是绕开 cv2 的文件 IO，只让它处理内存里的 buffer::

    imread : cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    imwrite: cv2.imencode(ext, img)[1].tofile(path)

这两个走的是 Python 的 Unicode 路径，与 ANSI 无关。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["imread_unicode", "imwrite_unicode"]


def imread_unicode(path, flags=None):
    """读图，**支持中文/任意 Unicode 路径**。读不出来返回 ``None``。

    与 ``cv2.imread`` 同签名语义（含失败返回 ``None`` 而不是抛异常），
    但路径不会被 ANSI 码页破坏。

    :param path: ``str`` 或 ``Path``
    :param flags: cv2 的读标志；``None`` 时用 ``cv2.IMREAD_COLOR``
    """
    import cv2
    import numpy as np

    p = Path(path)
    flags = cv2.IMREAD_COLOR if flags is None else flags
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_unicode(path, img, ext: str = ".png") -> bool:
    """写图，**支持中文/任意 Unicode 路径**。

    :param path: 目标路径；扩展名缺失时自动补 ``ext``
    :returns: 是否写成功（与 ``cv2.imwrite`` 一致，不抛异常）

    ⚠️ ``cv2.imencode`` 对**不认识的扩展名是抛 ``cv2.error``**，
       不是返回 ``(False, None)`` —— 只判返回值会漏掉，必须 try。
    """
    import cv2

    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix(ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok, buf = cv2.imencode(p.suffix or ext, img)
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        buf.tofile(str(p))
    except OSError:
        return False
    return True
