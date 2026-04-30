"""
首页 - 系统概览与快速入口
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QGridLayout, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from src.ui.styles import PRIMARY_COLOR, PRIMARY_DARK, BG_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR
from src.ui.event_bus import EventBus
from src.ui.scale_manager import ScaleManager
from src.config import NUM_CLASSES, INPUT_SIZE, NUM_EPOCHS


class FeatureCard(QFrame):
    """功能卡片"""

    clicked_go = pyqtSignal(str)  # 发出目标页面名

    # 基准尺寸
    BASE_MIN_H = 220
    BASE_MIN_W = 200
    BASE_ICON_SIZE = 32
    BASE_TITLE_SIZE = 15
    BASE_DESC_SIZE = 13
    BASE_BTN_W = 130
    BASE_PADDING = 24

    def __init__(self, icon: str, title: str, desc: str, target_page: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.target_page = target_page
        self.setCursor(Qt.PointingHandCursor)
        sm = ScaleManager.get()
        self.setMinimumHeight(sm.scale_int(self.BASE_MIN_H))
        self.setMinimumWidth(sm.scale_int(self.BASE_MIN_W))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        pad = sm.scale_int(self.BASE_PADDING)
        # 白色背景，蓝色边框
        self.setStyleSheet(f"""
            QFrame#Card {{
                background-color: white;
                border: 2px solid {PRIMARY_COLOR};
                border-radius: 12px;
                padding: {pad}px;
            }}
            QFrame#Card:hover {{
                border-color: {PRIMARY_DARK};
                background-color: #f0f8fc;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)

        self.icon_label = QLabel(icon)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFont(QFont("", self.BASE_ICON_SIZE))
        self.icon_label.setStyleSheet(f"color: {PRIMARY_COLOR}; border: none;")
        self.icon_label.setProperty("_base_font_size", self.BASE_ICON_SIZE)
        layout.addWidget(self.icon_label)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Microsoft YaHei", self.BASE_TITLE_SIZE, QFont.Bold))
        self.title_label.setStyleSheet(f"color: {TEXT_PRIMARY}; border: none;")
        self.title_label.setProperty("_base_font_size", self.BASE_TITLE_SIZE)
        layout.addWidget(self.title_label)

        self.desc_label = QLabel(desc)
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {self.BASE_DESC_SIZE}px; border: none;")
        self.desc_label.setProperty("_base_font_size", self.BASE_DESC_SIZE)
        layout.addWidget(self.desc_label)

        layout.addStretch()

        self.btn = QPushButton("进入→")
        self.btn.setObjectName("PrimaryButton")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setFixedWidth(sm.scale_int(self.BASE_BTN_W))
        self.btn.clicked.connect(lambda: self.clicked_go.emit(self.target_page))
        layout.addWidget(self.btn, alignment=Qt.AlignCenter)

        layout.addSpacing(4)

    def apply_scale(self, scale: float):
        sm = ScaleManager.get()
        self.setMinimumHeight(sm.scale_int(self.BASE_MIN_H))
        self.setMinimumWidth(sm.scale_int(self.BASE_MIN_W))
        self.btn.setFixedWidth(sm.scale_int(self.BASE_BTN_W))
        pad = sm.scale_int(self.BASE_PADDING)
        self.setStyleSheet(f"""
            QFrame#Card {{
                background-color: white;
                border: 2px solid {PRIMARY_COLOR};
                border-radius: 12px;
                padding: {pad}px;
            }}
            QFrame#Card:hover {{
                border-color: {PRIMARY_DARK};
                background-color: #f0f8fc;
            }}
        """)


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

    # 基准尺寸
    BASE_BANNER_FONT = 22
    BASE_MARGIN = 32
    BASE_MARGIN_V = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        sm = ScaleManager.get()
        layout.setContentsMargins(
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
        )
        layout.setSpacing(20)

        # 标题
        title = QLabel("🏠 首页")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        # 欢迎横幅
        self.banner = QFrame()
        self.banner.setStyleSheet(f"""
            QFrame {{
                background-color: {PRIMARY_COLOR};
                border-radius: 12px; padding: 32px;
            }}
        """)
        banner_layout = QVBoxLayout(self.banner)
        self.banner_text = QLabel("🐦 鸟类图像识别系统\n基于 OpenCV 与深度学习的智能鸟类监测平台")
        self.banner_text.setAlignment(Qt.AlignCenter)
        self.banner_text.setFont(QFont("Microsoft YaHei", self.BASE_BANNER_FONT, QFont.Bold))
        self.banner_text.setProperty("_base_font_size", self.BASE_BANNER_FONT)
        self.banner_text.setStyleSheet("""
            color: white; border: none;
            line-height: 1.6;
        """)
        banner_layout.addWidget(self.banner_text)
        layout.addWidget(self.banner)

        # 功能卡片
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        self.cards = []
        card_defs = [
            ("🔍", "单图识别", "上传单张鸟类图片\n即时获取识别结果", "单图识别"),
            ("🖼️", "批量识别", "批量上传图片\n高效处理大量数据", "批量识别"),
            ("📋", "历史记录", "查看本次会话的\n所有识别记录", "历史记录"),
        ]
        for icon, title, desc, page in card_defs:
            card = FeatureCard(icon, title, desc, page)
            card.clicked_go.connect(self._navigate)
            cards_layout.addWidget(card)
            self.cards.append(card)

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

    def apply_scale(self, scale: float):
        sm = ScaleManager.get()
        layout = self.layout()
        if layout:
            m = sm.scale_int(self.BASE_MARGIN)
            v = sm.scale_int(self.BASE_MARGIN_V)
            layout.setContentsMargins(m, v, m, v)
        for card in self.cards:
            card.apply_scale(scale)

    def _navigate(self, page_name: str):
        EventBus.get().nav_requested.emit(page_name)
