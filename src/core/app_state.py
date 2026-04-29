"""
全局应用状态管理单例

管理跨页面共享状态，状态变更通过信号通知所有订阅者。
"""
from PyQt5.QtCore import QObject, pyqtSignal


class AppState(QObject):
    """
    全局应用状态单例

    Attributes:
        current_model_name: 当前加载的模型名称
        is_model_ready: 模型是否已加载就绪
        is_inferencing: 是否正在推理中
        current_results: 最近一次识别结果
    """

    # 状态变更信号
    modelStatusChanged = pyqtSignal(bool, str)       # (ready, model_name)
    inferencingChanged = pyqtSignal(bool)             # (is_inferencing)
    resultsUpdated = pyqtSignal(dict)                 # (latest_result)

    _instance = None

    @classmethod
    def get(cls) -> "AppState":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            super(AppState, cls._instance).__init__()
            cls._instance._current_model_name = ""
            cls._instance._is_model_ready = False
            cls._instance._is_inferencing = False
            cls._instance._current_results = {}
        return cls._instance

    @property
    def current_model_name(self) -> str:
        return self._current_model_name

    @current_model_name.setter
    def current_model_name(self, value: str):
        self._current_model_name = value

    @property
    def is_model_ready(self) -> bool:
        return self._is_model_ready

    @is_model_ready.setter
    def is_model_ready(self, value: bool):
        old = self._is_model_ready
        self._is_model_ready = value
        if old != value:
            self.modelStatusChanged.emit(value, self._current_model_name)

    @property
    def is_inferencing(self) -> bool:
        return self._is_inferencing

    @is_inferencing.setter
    def is_inferencing(self, value: bool):
        old = self._is_inferencing
        self._is_inferencing = value
        if old != value:
            self.inferencingChanged.emit(value)

    @property
    def current_results(self) -> dict:
        return self._current_results

    @current_results.setter
    def current_results(self, value: dict):
        self._current_results = value
        self.resultsUpdated.emit(value)
