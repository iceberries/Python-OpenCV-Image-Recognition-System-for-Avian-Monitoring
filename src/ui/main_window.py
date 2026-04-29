"""
主窗口 - 侧边栏导航 + 内容区页面切换
集成 ScaleManager 实现窗口缩放自适应
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QStackedWidget, QStatusBar,
    QFrame, QSizePolicy, QApplication,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont

from src.ui.styles import build_scaled_qss, NAV_ICONS, PRIMARY_COLOR
from src.ui.event_bus import EventBus
from src.ui.scale_manager import ScaleManager
from src.core.app_state import AppState
from src.core.model_manager import ModelManager


class NavButton(QPushButton):
    """侧边栏导航按钮"""

    clicked_nav = pyqtSignal(str)  # 发出页面名称

    def __init__(self, page_name: str, icon: str, parent=None):
        super().__init__(f"  {icon}  {page_name}", parent)
        self.page_name = page_name
        self.setObjectName("NavButton")
        self.setCursor(Qt.PointingHandCursor)
        self._selected = False
        self.clicked.connect(lambda: self.clicked_nav.emit(self.page_name))

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value: bool):
        self._selected = value
        self.setProperty("selected", "true" if value else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class Sidebar(QFrame):
    """侧边栏：Logo + 导航菜单 + 系统状态"""

    nav_changed = pyqtSignal(str)

    # 基准宽度（scale=1.0 时的值）
    BASE_WIDTH = 200

    def __init__(self, page_names: list, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self._base_width = self.BASE_WIDTH

        sm = ScaleManager.get()
        scaled_w = sm.scale_int(self._base_width)
        self.setFixedWidth(scaled_w)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo
        self.logo = QLabel("🐦 鸟类识别")
        self.logo.setObjectName("SidebarLogo")
        self.logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logo)

        self.subtitle = QLabel("Avian Monitoring")
        self.subtitle.setObjectName("SidebarSubtitle")
        self.subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.subtitle)

        # 分割线
        divider = QFrame()
        divider.setObjectName("Divider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        layout.addSpacing(8)

        # 导航按钮
        self.nav_buttons = {}
        for name in page_names:
            icon = NAV_ICONS.get(name, "📄")
            btn = NavButton(name, icon)
            btn.clicked_nav.connect(self._on_nav_clicked)
            layout.addWidget(btn)
            self.nav_buttons[name] = btn

        layout.addStretch()

        # 系统状态
        divider2 = QFrame()
        divider2.setObjectName("Divider")
        divider2.setFixedHeight(1)
        layout.addWidget(divider2)
        layout.addSpacing(8)

        self.status_label = QLabel("🖥️ 设备: 检测中...")
        self.status_label.setObjectName("SidebarSubtitle")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.model_label = QLabel("❌ 模型未加载")
        self.model_label.setObjectName("SidebarSubtitle")
        self.model_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.model_label)
        layout.addSpacing(8)

        # 默认选中第一个
        if page_names:
            self.nav_buttons[page_names[0]].selected = True

    def _on_nav_clicked(self, page_name: str):
        self._update_selection(page_name)
        self.nav_changed.emit(page_name)

    def _update_selection(self, page_name: str):
        """仅更新按钮选中状态，不发射信号"""
        for name, btn in self.nav_buttons.items():
            btn.selected = (name == page_name)

    def set_model_status(self, loaded: bool):
        self.model_label.setText("✅ 模型已加载" if loaded else "❌ 模型未加载")

    def set_device_info(self, info: str):
        self.status_label.setText(f"🖥️ {info}")

    def apply_scale(self, scale: float):
        """缩放侧边栏宽度"""
        sm = ScaleManager.get()
        self.setFixedWidth(sm.scale_int(self._base_width))


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self, pages: dict, parent=None):
        """
        Args:
            pages: {页面名称: QWidget 实例} 字典
        """
        super().__init__(parent)
        self.setWindowTitle("鸟类图像识别系统")
        self.setMinimumSize(900, 600)

        # 初始化 ScaleManager（使用屏幕尺寸作为基准）
        sm = ScaleManager.get()
        screen = QApplication.primaryScreen()
        if screen:
            available = screen.availableGeometry()
            sm.init(QSize(available.width(), available.height()))

        # 应用初始样式
        self.setStyleSheet(build_scaled_qss(sm.scale_factor))

        # 中心部件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 侧边栏
        page_names = list(pages.keys())
        self.sidebar = Sidebar(page_names)
        self.sidebar.nav_changed.connect(self._switch_page)
        main_layout.addWidget(self.sidebar)

        # 内容区
        self.stack = QStackedWidget()
        self.stack.setObjectName("ContentArea")
        main_layout.addWidget(self.stack)

        # 注册页面
        self._page_map = {}
        for name, widget in pages.items():
            self.stack.addWidget(widget)
            self._page_map[name] = self.stack.indexOf(widget)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # 初始化设备信息
        self._init_device_info()

        # 连接事件总线
        bus = EventBus.get()
        bus.nav_requested.connect(self._switch_page)
        bus.model_loaded.connect(self.sidebar.set_model_status)

        # 连接全局状态 - 模型状态变更更新状态栏
        AppState.get().modelStatusChanged.connect(self._on_model_status_changed)
        AppState.get().inferencingChanged.connect(self._on_inferencing_changed)

    def _switch_page(self, page_name: str):
        idx = self._page_map.get(page_name)
        if idx is not None:
            self.stack.setCurrentIndex(idx)
            self.status_bar.showMessage(f"当前页面: {page_name}")
            # 同步侧边栏选中状态（不触发信号）
            self.sidebar._update_selection(page_name)

    def _init_device_info(self):
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.sidebar.set_device_info(f"CUDA: {device_name[:20]}")
            else:
                self.sidebar.set_device_info("CPU")
        except ImportError:
            self.sidebar.set_device_info("CPU (PyTorch 未安装)")

    def _on_model_status_changed(self, ready: bool, model_name: str):
        """全局模型状态变更 - 更新状态栏"""
        if ready:
            self.status_bar.showMessage(f"模型就绪: {model_name}")
        else:
            self.status_bar.showMessage("模型未加载")

    def _on_inferencing_changed(self, inferencing: bool):
        """推理状态变更 - 更新状态栏"""
        if inferencing:
            self.status_bar.showMessage("推理中...")
        else:
            manager = ModelManager.get()
            if manager.is_ready:
                self.status_bar.showMessage("就绪")
            else:
                self.status_bar.showMessage("模型未加载")

    def resizeEvent(self, event):
        """窗口缩放时重新计算 scale 并更新 UI"""
        super().resizeEvent(event)

        sm = ScaleManager.get()
        new_scale, old_scale = sm.on_resize(self.size())

        if new_scale != old_scale:
            # 1. 重新生成并应用 QSS
            self.setStyleSheet(build_scaled_qss(new_scale))

            # 2. 缩放侧边栏宽度
            self.sidebar.apply_scale(new_scale)

            # 3. 递归缩放内容区字体
            sm.scale_widget_recursive(self.stack)

            # 4. 通知各页面进行自定义缩放
            for i in range(self.stack.count()):
                page = self.stack.widget(i)
                if hasattr(page, 'apply_scale'):
                    page.apply_scale(new_scale)
