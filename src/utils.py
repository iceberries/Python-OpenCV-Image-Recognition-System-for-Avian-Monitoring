"""
工具函数模块
"""

import numpy as np
import cv2


def draw_bounding_box(
    image: np.ndarray,
    bbox: tuple,
    label: str = "",
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制边界框 (使用 OpenCV)

    Args:
        image: 输入图像 (RGB)
        bbox: (x, y, width, height)
        label: 标签文本
        color: 边框颜色 (BGR)
        thickness: 边框粗细

    Returns:
        标注后的图像
    """
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    result = image.copy()

    # RGB -> BGR for OpenCV drawing
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, label, (x, y - 10), font, 0.6, color, 2)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
