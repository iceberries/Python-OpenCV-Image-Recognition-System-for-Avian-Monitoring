"""
历史记录页面 - 查看本次会话识别历史
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage

from ui.styles import (
    PRIMARY_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR, BG_COLOR, CARD_BG,
)
from ui.event_bus import EventBus


class HistoryCard(QFrame):
    """单条历史记录卡片"""

    def __init__(self, index: int, image=None, result: dict = None, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")

        layout = QHBoxLayout(self)
        layout.setSpacing(16)

        # 缩略图
        thumb_frame = QFrame()
        thumb_frame.setFixedSize(80, 80)
        thumb_frame.setStyleSheet(f"background: {BG_COLOR}; border-radius: 6px;")
        thumb_layout = QVBoxLayout(thumb_frame)
        thumb_layout.setContentsMargins(0, 0, 0, 0)

        if image is not None:
            h, w = image.shape[:2]
            qimg = QImage(image.data, w, h, image.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                76, 76, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("border: none;")
            thumb_layout.addWidget(img_label)
        else:
            placeholder = QLabel("📷")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("border: none; font-size: 24px;")
            thumb_layout.addWidget(placeholder)

        layout.addWidget(thumb_frame)

        # 信息
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        idx_label = QLabel(f"记录 #{index}")
        idx_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; border: none;")
        info_layout.addWidget(idx_label)

        if result:
            class_name = result.get("class_name", "Unknown")
            confidence = result.get("confidence", 0)

            name_label = QLabel(f"🏷️ {class_name}")
            name_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; border: none;")
            info_layout.addWidget(name_label)

            # 置信度颜色
            if confidence > 0.9:
                conf_color = SUCCESS_COLOR
            elif confidence > 0.7:
                conf_color = WARNING_COLOR
            else:
                conf_color = DANGER_COLOR

            conf_label = QLabel(f"📊 置信度: {confidence:.1%}")
            conf_label.setStyleSheet(f"color: {conf_color}; font-weight: bold; border: none;")
            info_layout.addWidget(conf_label)

        info_layout.addStretch()
        layout.addLayout(info_layout, 1)


class HistoryPage(QWidget):
    """历史记录页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history = []  # [(image, result), ...]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        # 标题行
        header = QHBoxLayout()
        title = QLabel("📋 历史记录")
        title.setObjectName("PageTitle")
        header.addWidget(title)
        header.addStretch()

        self.count_label = QLabel("本次会话: 0 条记录")
        self.count_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        header.addWidget(self.count_label)

        self.btn_clear = QPushButton("🗑️ 清空记录")
        self.btn_clear.setObjectName("SecondaryButton")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self._clear_history)
        header.addWidget(self.btn_clear)

        layout.addLayout(header)

        # 分割线
        divider = QFrame()
        divider.setObjectName("Divider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setSpacing(8)
        self.list_layout.setAlignment(Qt.AlignTop)

        scroll.setWidget(self.list_widget)
        layout.addWidget(scroll, 1)

        # 空状态
        self.empty_label = QLabel("📭 暂无识别记录\n进行图片识别后，记录将自动保存于此")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 16px;")
        self.list_layout.addWidget(self.empty_label)

        # 监听事件
        EventBus.get().history_updated.connect(self._on_history_updated)
        EventBus.get().recognition_completed.connect(self._on_recognition)

    def _on_recognition(self, result: dict):
        self._history.append((None, result))
        self._refresh_list()

    def _on_history_updated(self):
        self._refresh_list()

    def _refresh_list(self):
        # 清除旧卡片
        while self.list_layout.count():
            child = self.list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not self._history:
            self.empty_label = QLabel("📭 暂无识别记录\n进行图片识别后，记录将自动保存于此")
            self.empty_label.setAlignment(Qt.AlignCenter)
            self.empty_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 16px;")
            self.list_layout.addWidget(self.empty_label)
            self.count_label.setText("本次会话: 0 条记录")
            return

        self.count_label.setText(f"本次会话: {len(self._history)} 条记录")

        for i, (image, result) in enumerate(reversed(self._history)):
            idx = len(self._history) - i
            card = HistoryCard(idx, image, result)
            self.list_layout.addWidget(card)

    def _clear_history(self):
        self._history = []
        self._refresh_list()
