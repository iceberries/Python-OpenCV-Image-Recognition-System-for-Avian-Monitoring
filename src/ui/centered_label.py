"""
居中显示 QPixmap 的 QLabel 自定义实现
"""
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter


class CenteredPixmapLabel(QLabel):
    """
    支持 QPixmap 真正居中对齐的 QLabel 子类
    
    特性：
    - 图片按比例缩放至控件尺寸内
    - 缩放后的图片在控件中心绘制
    - 支持动态 resize 自动重缩放
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap: QPixmap = None  # 原始图片
        self._scaled_pixmap: QPixmap = None  # 缩放后的图片缓存
        self.setScaledContents(False)
    
    def setPixmap(self, pixmap: QPixmap):
        """设置图片源，会自动居中缩放"""
        self._source_pixmap = pixmap
        self._update_scaled()
        self.update()  # 触发 paintEvent 重绘
    
    def clear(self):
        """清空图片"""
        self._source_pixmap = None
        self._scaled_pixmap = None
        self.update()
    
    def _update_scaled(self):
        """根据当前控件尺寸重新计算缩放后的图片"""
        if self._source_pixmap is None or self._source_pixmap.isNull():
            self._scaled_pixmap = None
            return
        
        # 可用尺寸（减去边框等内边距）
        available = self.contentsRect().size()
        if available.width() <= 0 or available.height() <= 0:
            return
        
        self._scaled_pixmap = self._source_pixmap.scaled(
            available,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
    def resizeEvent(self, event):
        """窗口大小变化时重新缩放"""
        super().resizeEvent(event)
        self._update_scaled()
        self.update()
    
    def paintEvent(self, event):
        """自定义绘制：将缩放后的图片居中绘制"""
        if self._scaled_pixmap is None or self._scaled_pixmap.isNull():
            # 没有图片时，调用父类默认绘制（可能显示文本）
            super().paintEvent(event)
            return
        
        # 计算居中坐标
        cr = self.contentsRect()
        x = cr.left() + (cr.width() - self._scaled_pixmap.width()) // 2
        y = cr.top() + (cr.height() - self._scaled_pixmap.height()) // 2
        
        # 绘制图片
        painter = QPainter(self)
        painter.drawPixmap(x, y, self._scaled_pixmap)
        painter.end()