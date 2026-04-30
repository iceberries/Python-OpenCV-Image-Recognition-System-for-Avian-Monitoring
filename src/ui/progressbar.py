"""
自定义圆角进度条
低进度时保持左直右圆，避免 Qt 默认样式表渲染问题
"""
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QPainterPath


class RoundedProgressBar(QProgressBar):
    """自定义圆角进度条，低进度时保持左直右圆"""

    def __init__(self, bar_color: str, bg_color: str, radius: int, parent=None):
        super().__init__(parent)
        self._bar_color = bar_color
        self._bg_color = bg_color
        self._radius = radius
        self.setTextVisible(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        h = self.height()
        w = self.width()
        r = self._radius

        # 背景
        painter.setBrush(QColor(self._bg_color))
        painter.drawRoundedRect(0, 0, w, h, r, r)

        # 进度
        if self.maximum() > self.minimum():
            progress = (self.value() - self.minimum()) / (self.maximum() - self.minimum())
            bar_w = int(w * progress)  # 转换为 int

            if bar_w > 0:
                painter.setBrush(QColor(self._bar_color))
                painter.drawRoundedRect(0, 0, bar_w, h, r, r)
        painter.end()