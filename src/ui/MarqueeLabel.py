"""
滚动文字标签 - 文字过长时水平滚动播放
"""
from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QFontMetrics, QPainter, QColor, QFont

class MarqueeLabel(QLabel):
    """
    跑马灯标签：文字超过显示宽度时自动水平滚动
    
    特性：
    - 文字短于显示宽度时正常居中/左对齐显示
    - 文字过长时自动开始水平滚动
    - 支持设置滚动速度、方向、间距
    - 鼠标悬停可暂停滚动
    """
    
    # 信号
    scrollingChanged = pyqtSignal(bool)  # 滚动状态变化
    
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        
        # 配置
        self._full_text = text          # 完整原始文字
        self._scroll_text = ""          # 滚动显示的文字（含间隔）
        self._offset = 0                # 当前滚动偏移（像素）
        self._is_scrolling = False      # 是否正在滚动
        self._is_paused = False         # 是否暂停（鼠标悬停）
        
        # 滚动参数
        self._speed = 1                 # 滚动速度（像素/帧）
        self._interval = 30             # 定时器间隔（毫秒）
        self._gap = "    "              # 文字间隔（用于循环衔接）
        self._direction = Qt.LeftToRight  # 滚动方向
        
        # 样式
        self.setWordWrap(False)         # 禁止换行
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # 定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        
        # 初始化显示
        self.setText(text)
        
    # ========== 公共接口 ==========
    
    def setText(self, text: str):
        """设置文字，自动判断是否启动滚动"""
        self._full_text = text
        self._offset = 0
        self._timer.stop()
        
        # 计算文字宽度
        fm = QFontMetrics(self.font())
        text_width = fm.horizontalAdvance(text)
        available_width = self.width() - 4  # 留点边距
        
        if text_width > available_width and available_width > 0:
            # 需要滚动：构造循环文字
            self._scroll_text = text + self._gap + text
            self._is_scrolling = True
            self._timer.start(self._interval)
            self.scrollingChanged.emit(True)
        else:
            # 不需要滚动：正常显示
            self._scroll_text = text
            self._is_scrolling = False
            super().setText(text)
            self.scrollingChanged.emit(False)
            
    def setSpeed(self, pixels_per_frame: int):
        """设置滚动速度（像素/帧）"""
        self._speed = max(1, pixels_per_frame)
        
    def setInterval(self, ms: int):
        """设置定时器间隔（毫秒）"""
        self._interval = max(10, ms)
        if self._timer.isActive():
            self._timer.setInterval(self._interval)
            
    def setGap(self, gap: str):
        """设置循环间隔字符串"""
        self._gap = gap
        # 重新计算
        self.setText(self._full_text)
        
    def isScrolling(self) -> bool:
        """返回是否正在滚动"""
        return self._is_scrolling
        
    # ========== 事件处理 ==========
    
    def resizeEvent(self, event):
        """尺寸变化时重新判断是否滚动"""
        super().resizeEvent(event)
        # 延迟重新计算，避免频繁调用
        QTimer.singleShot(0, lambda: self.setText(self._full_text))
        
    def enterEvent(self, event):
        """鼠标进入：暂停滚动"""
        if self._is_scrolling:
            self._is_paused = True
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """鼠标离开：恢复滚动"""
        self._is_paused = False
        super().leaveEvent(event)
        
    # ========== 滚动逻辑 ==========
    
    def _on_timer(self):
        """定时器回调：更新滚动偏移"""
        if self._is_paused or not self._is_scrolling:
            return
            
        fm = QFontMetrics(self.font())
        full_width = fm.horizontalAdvance(self._full_text + self._gap)
        
        # 更新偏移
        self._offset += self._speed
        
        # 循环：超过一个完整周期时归零
        if self._offset >= full_width:
            self._offset = 0
            
        self.update()  # 触发重绘
        
    def paintEvent(self, event):
        """自定义绘制：实现滚动效果"""
        if not self._is_scrolling:
            super().paintEvent(event)
            return
            
        # 使用 QPainter 手动绘制偏移文字
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setPen(self.palette().text().color())
        painter.setFont(self.font())
        
        # 计算绘制位置
        y = (self.height() + QFontMetrics(self.font()).ascent() - 
             QFontMetrics(self.font()).descent()) // 2
        
        # 绘制滚动文字（从偏移位置开始）
        x = -self._offset
        
        # 如果偏移导致右侧空白，在前面补绘一段实现无缝循环
        fm = QFontMetrics(self.font())
        text_width = fm.horizontalAdvance(self._scroll_text)
        
        # 绘制主文字
        painter.drawText(int(x), int(y), self._scroll_text)
        
        # 如果需要，绘制衔接段（实现无缝循环）
        if x + text_width < self.width():
            painter.drawText(int(x + text_width), int(y), self._scroll_text)
            
        painter.end()
        
    def sizeHint(self) -> QSize:
        """建议尺寸"""
        fm = QFontMetrics(self.font())
        return QSize(fm.horizontalAdvance(self._full_text) + 10, fm.height() + 10)