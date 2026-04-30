"""
鸟类图像识别系统 - PyQt5 桌面端入口
"""
import sys
import os

# 项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer

from src.ui.main_window import MainWindow
from src.ui.pages.home_page import HomePage
from src.ui.pages.single_recognition_page import SingleRecognitionPage
from src.ui.pages.batch_recognition_page import BatchRecognitionPage
from src.ui.pages.history_page import HistoryPage
from src.ui.pages.settings_page import SettingsPage
from src.ui.event_bus import EventBus
from src.core.model_manager import ModelManager

def main():
    # 高 DPI 支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # 全局默认字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)



    # 注册页面
    pages = {
        "首页": HomePage(),
        "单图识别": SingleRecognitionPage(),
        "批量识别": BatchRecognitionPage(),
        "历史记录": HistoryPage(),
        "设置": SettingsPage(),
    }

    window = MainWindow(pages)
    window.showMaximized()

    # 异步加载模型（延迟 500ms，等 UI 完全渲染后再加载）
    bus = EventBus.get()
    manager = ModelManager.get()

    # 连接模型加载信号到事件总线
    manager.modelLoaded.connect(lambda ok, msg: bus.model_loaded.emit(ok))

    def _auto_load_model():
        """自动尝试加载默认模型"""
        from src.config import OUTPUT_DIR
        default_path = os.path.join(OUTPUT_DIR, "best_model.pth")
        if os.path.exists(default_path):
            manager.load_model_async(checkpoint_path=default_path)
        # 如果模型文件不存在，不做任何提示，用户可在设置页手动加载

    QTimer.singleShot(500, _auto_load_model)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
