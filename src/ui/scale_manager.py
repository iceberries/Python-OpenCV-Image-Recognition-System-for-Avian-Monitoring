"""
缩放管理器 - 根据窗口大小自动按比例缩放 UI 元素

以最大化窗口为基准 (scale=1.0)，窗口缩放时按比例调整：
  - QSS 中的 font-size、padding、border-radius、min-width 等
  - 各组件的 setFixedWidth/setFixedHeight/setMinimumHeight 等
  - 字体大小
"""
import math
from typing import Optional

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QFont


class ScaleManager:
    """
    全局缩放管理器（单例）

    使用方式:
      1. 启动时调用 init() 传入参考窗口尺寸
      2. 窗口 resize 时调用 on_resize(new_size)
      3. 组件中用 scale(value) 获取缩放后的值
      4. 用 scale_font(base_size) 获取缩放后字体大小
    """

    _instance: Optional["ScaleManager"] = None

    def __init__(self):
        self._base_size = QSize(1920, 1080)  # 默认基准（会被 init 覆盖）
        self._current_scale = 1.0
        self._min_scale = 0.5
        self._max_scale = 2.0

    @classmethod
    def get(cls) -> "ScaleManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(self, base_size: QSize):
        """
        初始化基准尺寸（通常是屏幕可用区域大小）

        Args:
            base_size: 最大化窗口时的尺寸
        """
        self._base_size = base_size
        self._current_scale = 1.0

    def on_resize(self, new_size: QSize):
        """
        窗口尺寸变化时调用，计算新的 scale factor

        Returns:
            float: 新的 scale factor
        """
        # 取宽高缩放比的较小值，保持整体比例
        scale_w = new_size.width() / self._base_size.width()
        scale_h = new_size.height() / self._base_size.height()
        new_scale = min(scale_w, scale_h)

        # 限制范围
        new_scale = max(self._min_scale, min(self._max_scale, new_scale))

        # 四舍五入到 0.01 精度，避免微小抖动
        new_scale = round(new_scale, 2)

        old_scale = self._current_scale
        self._current_scale = new_scale

        return new_scale, old_scale

    @property
    def scale_factor(self) -> float:
        return self._current_scale

    def scale(self, value: float) -> float:
        """缩放一个数值"""
        return value * self._current_scale

    def scale_int(self, value: int) -> int:
        """缩放一个整数值"""
        return max(1, round(value * self._current_scale))

    def scale_font(self, base_size: int) -> int:
        """缩放字体大小，最小保证 8px"""
        return max(8, round(base_size * self._current_scale))

    def apply_font_to_widget(self, widget: QWidget, base_size: int, bold: bool = False):
        """对 widget 应用缩放后的字体"""
        font = widget.font()
        font.setPointSize(self.scale_font(base_size))
        font.setBold(bold)
        widget.setFont(font)

    def scale_widget_recursive(self, widget: QWidget):
        """
        递归遍历子组件，缩放字体大小

        对每个有字体的子组件，按 scale factor 缩放其字体
        """
        self._scale_fonts(widget)

    def _scale_fonts(self, widget: QWidget):
        """递归缩放字体"""
        # 跳过隐藏的 widget
        if not widget.isVisible() and not widget.isWidgetType():
            return

        font = widget.font()
        if font.pointSize() > 0:
            # 记录原始大小（通过 property），避免多次缩放
            original_size = widget.property("_base_font_size")
            if original_size is None:
                original_size = font.pointSize()
                widget.setProperty("_base_font_size", original_size)

            new_size = self.scale_font(original_size)
            if font.pointSize() != new_size:
                font.setPointSize(new_size)
                widget.setFont(font)

        # 递归子组件
        for child in widget.findChildren(QWidget):
            self._scale_fonts(child)
