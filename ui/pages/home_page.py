"""
首页 - 系统概览与快速入口
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QGridLayout, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from ui.styles import PRIMARY_COLOR, PRIMARY_DARK, BG_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR
from ui.event_bus import EventBus
from src.config import NUM_CLASSES, INPUT_SIZE, NUM_EPOCHS


class FeatureCard(QFrame):
    """功能卡片"""

    clicked_go = pyqtSignal(str)  # 发出目标页面名

    def __init__(self, icon: str, title: str, desc: str, target_page: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.target_page = target_page
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(220)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 白色背景，蓝色边框
        self.setStyleSheet(f"""
            QFrame#Card {{
                background-color: white;
                border: 2px solid {PRIMARY_COLOR};
                border-radius: 12px;
                padding: 24px;
            }}
            QFrame#Card:hover {{
                border-color: {PRIMARY_DARK};
                background-color: #f0f8fc;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)

        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFont(QFont("", 32))
        icon_label.setStyleSheet(f"color: {PRIMARY_COLOR}; border: none;")
        layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        title_label.setStyleSheet(f"color: {TEXT_PRIMARY}; border: none;")
        layout.addWidget(title_label)

        desc_label = QLabel(desc)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; border: none;")
        layout.addWidget(desc_label)

        layout.addStretch()

        btn = QPushButton("进入 →")
        btn.setObjectName("PrimaryButton")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedWidth(130)
        btn.clicked.connect(lambda: self.clicked_go.emit(self.target_page))
        layout.addWidget(btn, alignment=Qt.AlignCenter)

        layout.addSpacing(4)


class InfoCard(QFrame):
    """信息卡片"""

    def __init__(self, title: str, items: list, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")

        layout = QVBoxLayout(self)

        header = QLabel(title)
        header.setObjectName("SectionTitle")
        layout.addWidget(header)

        for item in items:
            label = QLabel(item)
            label.setStyleSheet(f"color: {TEXT_PRIMARY}; padding: 2px 0;")
            layout.addWidget(label)


class HomePage(QWidget):
    """首页"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(20)

        # 标题
        title = QLabel("🏠 首页")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        # 欢迎横幅
        banner = QFrame()
        banner.setStyleSheet(f"""
            QFrame {{
                background-color: {PRIMARY_COLOR};
                border-radius: 12px; padding: 32px;
            }}
        """)
        banner_layout = QVBoxLayout(banner)
        banner_text = QLabel("🐦 鸟类图像识别系统\n基于 OpenCV 与深度学习的智能鸟类监测平台")
        banner_text.setAlignment(Qt.AlignCenter)
        banner_text.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        banner_text.setStyleSheet("""
            color: white; border: none;
            line-height: 1.6;
        """)
        banner_layout.addWidget(banner_text)
        layout.addWidget(banner)

        # 功能卡片
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        card1 = FeatureCard("🔍", "单图识别", "上传单张鸟类图片\n即时获取识别结果", "单图识别")
        card2 = FeatureCard("🖼️", "批量识别", "批量上传图片\n高效处理大量数据", "批量识别")
        card3 = FeatureCard("📋", "历史记录", "查看本次会话的\n所有识别记录", "历史记录")

        for card in [card1, card2, card3]:
            card.clicked_go.connect(self._navigate)
            cards_layout.addWidget(card)

        layout.addLayout(cards_layout)

        # 系统信息
        info_layout = QHBoxLayout()
        info_layout.setSpacing(16)

        sys_info = InfoCard("📊 系统信息", [
            f"• 识别类别数: {NUM_CLASSES} 种鸟类",
            f"• 输入尺寸: {INPUT_SIZE}×{INPUT_SIZE}",
            f"• 训练轮次: {NUM_EPOCHS}",
        ])
        quick_start = InfoCard("🚀 快速开始", [
            "1. 在侧边栏选择「单图识别」",
            "2. 上传一张鸟类图片",
            "3. 点击「开始识别」查看结果",
        ])

        info_layout.addWidget(sys_info)
        info_layout.addWidget(quick_start)
        layout.addLayout(info_layout)

        layout.addStretch()

    def _navigate(self, page_name: str):
        EventBus.get().nav_requested.emit(page_name)
