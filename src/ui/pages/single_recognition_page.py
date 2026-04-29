"""
单图识别页面 - 上传图片 + 识别 + 结果展示
集成 UploadComponent + SingleResultPanel + InferenceWorker
"""
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QProgressBar, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.ui.styles import (
    PRIMARY_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR, BG_COLOR,
)
from src.ui.event_bus import EventBus
from src.ui.upload_component import UploadComponent
from src.ui.result_panel import SingleResultPanel, RecognitionResult
from src.ui.scale_manager import ScaleManager
from src.core.model_manager import ModelManager
from src.core.inference_worker import InferenceWorker
from src.core.app_state import AppState


class SingleRecognitionPage(QWidget):
    """单图识别页面"""

    # 基准尺寸
    BASE_MARGIN = 32
    BASE_MARGIN_V = 24
    BASE_CARD_MIN_H = 460

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[InferenceWorker] = None
        self._current_image: Optional[np.ndarray] = None

        sm = ScaleManager.get()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
        )
        layout.setSpacing(16)

        # 标题
        title = QLabel("🔍 单图识别")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        # 主体两栏
        body = QHBoxLayout()
        body.setSpacing(24)

        # ===== 左栏 - 上传 =====
        left_card = QFrame()
        left_card.setObjectName("Card")
        self._left_card = left_card
        left_card.setMinimumHeight(sm.scale_int(self.BASE_CARD_MIN_H))
        left_layout = QVBoxLayout(left_card)
        left_layout.setSpacing(12)

        upload_label = QLabel("📤 上传图片")
        upload_label.setObjectName("SectionTitle")
        upload_label.setStyleSheet("border: none;")
        left_layout.addWidget(upload_label)

        # 上传组件
        self.upload_component = UploadComponent(mode="single")
        self.upload_component.files_changed.connect(self._on_files_changed)
        self.upload_component.images_ready.connect(self._on_images_ready)
        left_layout.addWidget(self.upload_component, 1)

        # 识别按钮
        self.btn_recognize = QPushButton("🚀 开始识别")
        self.btn_recognize.setObjectName("PrimaryButton")
        self.btn_recognize.setCursor(Qt.PointingHandCursor)
        self.btn_recognize.setEnabled(False)
        self.btn_recognize.clicked.connect(self._run_recognition)
        left_layout.addWidget(self.btn_recognize)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)

        body.addWidget(left_card, 1)

        # ===== 右栏 - 结果面板 =====
        self.result_panel = SingleResultPanel(show_heatmap=True)
        body.addWidget(self.result_panel, 1)

        layout.addLayout(body)

        # 监听模型状态
        AppState.get().modelStatusChanged.connect(self._on_model_status_changed)

    # ===== 上传组件回调 =====
    def _on_files_changed(self, paths: list):
        self._update_recognize_button()

    def _on_images_ready(self, images: list):
        if images:
            self._current_image = images[0]
            EventBus.get().image_uploaded.emit(images[0])

    # ===== 模型状态 =====
    def _on_model_status_changed(self, ready: bool, model_name: str):
        self._update_recognize_button()

    def _update_recognize_button(self):
        """根据模型状态和图片状态更新识别按钮"""
        has_image = len(self.upload_component.get_file_paths()) > 0
        model_ready = ModelManager.get().is_ready
        self.btn_recognize.setEnabled(has_image and model_ready)

    # ===== 识别 =====
    def _run_recognition(self):
        images = self.upload_component.get_images()
        if not images:
            return

        # 检查模型
        manager = ModelManager.get()
        if not manager.is_ready:
            QMessageBox.warning(self, "模型未加载", "请先在设置页加载模型，或等待模型加载完成。")
            return

        # 检查是否已有推理在运行
        if self._worker and self._worker.isRunning():
            return

        self._current_image = images[0]
        self.btn_recognize.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.result_panel.show_loading()

        # 设置 AppState
        AppState.get().is_inferencing = True

        # 创建并启动推理线程
        self._worker = InferenceWorker()
        self._worker.setup_single(
            image=self._current_image,
            top_k=5,
            use_clahe=True,
            return_heatmap=True,
        )
        self._worker.singleResultReady.connect(self._on_result)
        self._worker.inferenceError.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: dict, original_image: np.ndarray):
        self.progress_bar.hide()
        self._update_recognize_button()
        AppState.get().is_inferencing = False

        # 构建 RecognitionResult
        rec_result = RecognitionResult(
            class_name=result.get("class_name", ""),
            confidence=result.get("confidence", 0.0),
            top_k=result.get("top_k", []),
            image=original_image,
            overlay_image=result.get("overlay_image"),
            heatmap=result.get("heatmap"),
            latency=result.get("latency"),
        )

        self.result_panel.set_result(rec_result)

        # 通知事件总线
        EventBus.get().recognition_completed.emit(result)
        EventBus.get().history_updated.emit()

        # 更新 AppState
        AppState.get().current_results = result

    def _on_error(self, error_msg: str):
        self.progress_bar.hide()
        self._update_recognize_button()
        AppState.get().is_inferencing = False

        QMessageBox.critical(self, "识别失败", error_msg)

    def apply_scale(self, scale: float):
        sm = ScaleManager.get()
        layout = self.layout()
        if layout:
            m = sm.scale_int(self.BASE_MARGIN)
            v = sm.scale_int(self.BASE_MARGIN_V)
            layout.setContentsMargins(m, v, m, v)
        self._left_card.setMinimumHeight(sm.scale_int(self.BASE_CARD_MIN_H))
        if hasattr(self.upload_component, 'apply_scale'):
            self.upload_component.apply_scale(scale)
        if hasattr(self.result_panel, 'apply_scale'):
            self.result_panel.apply_scale(scale)
