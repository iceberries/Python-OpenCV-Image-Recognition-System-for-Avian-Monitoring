"""
全局事件总线 - 跨页面信号通信（单例模式）
"""
from PyQt5.QtCore import QObject, pyqtSignal


class EventBus(QObject):
    """全局事件总线，用于跨页面通信"""

    # 信号定义
    image_uploaded = pyqtSignal(object)           # 图片上传 (np.ndarray)
    recognition_completed = pyqtSignal(dict)       # 识别完成 (result dict)
    model_loaded = pyqtSignal(bool)               # 模型加载状态 (success)
    nav_requested = pyqtSignal(str)               # 页面导航请求 (page name)
    history_updated = pyqtSignal()                # 历史记录更新

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get(cls) -> "EventBus":
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
