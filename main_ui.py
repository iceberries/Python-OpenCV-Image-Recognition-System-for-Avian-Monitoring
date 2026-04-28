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

from ui.main_window import MainWindow
from ui.pages.home_page import HomePage
from ui.pages.single_recognition_page import SingleRecognitionPage
from ui.pages.batch_recognition_page import BatchRecognitionPage
from ui.pages.history_page import HistoryPage
from ui.pages.settings_page import SettingsPage


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
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    from PyQt5.QtCore import Qt
    main()
