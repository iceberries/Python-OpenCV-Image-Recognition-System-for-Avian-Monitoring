"""
批量识别页面 - 批量上传 + 进度 + 统计
集成 UploadComponent + BatchResultPanel + InferenceWorker
"""
import time
from typing import Optional, List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QProgressBar, QSizePolicy,
    QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.ui.styles import (
    PRIMARY_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR, BG_COLOR,
)
from src.ui.event_bus import EventBus
from src.ui.upload_component import UploadComponent
from src.ui.result_panel import BatchResultPanel, RecognitionResult
from src.ui.scale_manager import ScaleManager
from src.core.model_manager import ModelManager
from src.core.inference_worker import InferenceWorker
from src.core.app_state import AppState


class BatchRecognitionPage(QWidget):
    """批量识别页面"""

    # 基准尺寸
    BASE_MARGIN = 32
    BASE_MARGIN_V = 24
    BASE_STAT_MIN_H = 100
    BASE_BTN_MIN_H = 40
    BASE_PROGRESS_W = 120

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list = []
        self._worker: Optional[InferenceWorker] = None
        self._batch_start_time: float = 0
        self._image_map: Dict[str, np.ndarray] = {}

        sm = ScaleManager.get()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
        )
        layout.setSpacing(16)

        # 标题
        title = QLabel("🖼️ 批量识别")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        # ===== 主体卡片容器 =====
        main_card = QFrame()
        main_card.setObjectName("Card")
        main_card_layout = QVBoxLayout(main_card)
        main_card_layout.setSpacing(16)
        main_card_layout.setContentsMargins(20, 20, 20, 20)

        # 上传组件（批量模式）- 调整高度策略使其更紧凑
        self.upload_component = UploadComponent(mode="batch")
        self.upload_component.files_changed.connect(self._on_files_changed)
        self.upload_component.images_ready.connect(self._on_images_ready)
        # 设置上传组件在布局中的拉伸因子，使其不会过度占用空间
        main_card_layout.addWidget(self.upload_component, 1)

        # 操作行 - 四个按钮按指定比例排列（选择图片、清空、批量识别、取消）
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        # 选择图片按钮 - stretch=2 (2/8)
        self.upload_component.btn_upload.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        action_row.addWidget(self.upload_component.btn_upload)
        action_row.setStretch(0, 2)

        # 清空按钮 - stretch=2 (2/8)
        self.upload_component.btn_clear.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        action_row.addWidget(self.upload_component.btn_clear)
        action_row.setStretch(1, 2)

        # 批量识别按钮 - stretch=3 (3/8)
        self.btn_run = QPushButton("🚀 批量识别")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setEnabled(False)
        self.btn_run.setMinimumHeight(sm.scale_int(self.BASE_BTN_MIN_H))
        self.btn_run.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_run.clicked.connect(self._run_batch)
        action_row.addWidget(self.btn_run)
        action_row.setStretch(2, 5)

        # 取消按钮 - stretch=1 (1/8)
        self.btn_cancel = QPushButton("⏹ 取消")
        self.btn_cancel.setObjectName("SecondaryButton")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setMinimumHeight(sm.scale_int(self.BASE_BTN_MIN_H))
        self.btn_cancel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_cancel.clicked.connect(self._cancel_batch)
        action_row.addWidget(self.btn_cancel)
        action_row.setStretch(3, 1)

        # 进度条和标签
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        action_row.addWidget(self.progress_bar, 1)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        self.progress_label.setFixedWidth(sm.scale_int(self.BASE_PROGRESS_W))
        self.progress_label.hide()
        action_row.addWidget(self.progress_label)
        main_card_layout.addLayout(action_row)

        # 统计卡片
        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)

        self.stat_total = self._make_stat_card("总数", "0")
        self.stat_avg = self._make_stat_card("平均置信度", "-")
        self.stat_high = self._make_stat_card("高置信度(>90%)", "0")
        for card in [self.stat_total, self.stat_avg, self.stat_high]:
            stats_row.addWidget(card)
        main_card_layout.addLayout(stats_row)

        # 结果面板（表格/卡片双视图）
        self.result_panel = BatchResultPanel()
        main_card_layout.addWidget(self.result_panel, 1)

        layout.addWidget(main_card, 1)

        # 监听模型状态
        AppState.get().modelStatusChanged.connect(self._on_model_status_changed)

    def _make_stat_card(self, title: str, value: str) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        sm = ScaleManager.get()
        card.setMinimumHeight(sm.scale_int(self.BASE_STAT_MIN_H))
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(6)
        card_layout.setContentsMargins(16, 12, 16, 12)

        t = QLabel(title)
        t.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; border: none;")
        t.setAlignment(Qt.AlignCenter)
        t.setWordWrap(True)
        card_layout.addWidget(t)

        v = QLabel(value)
        v.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        v.setStyleSheet(f"color: {PRIMARY_COLOR}; border: none;")
        v.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(v)

        card._value_label = v
        return card

    # ===== 上传组件回调 =====
    def _on_files_changed(self, paths: list):
        self._update_run_button()

    def _on_images_ready(self, images: list):
        for img in images:
            EventBus.get().image_uploaded.emit(img)

    # ===== 模型状态 =====
    def _on_model_status_changed(self, ready: bool, model_name: str):
        self._update_run_button()

    def _update_run_button(self):
        has_files = len(self.upload_component.get_file_paths()) > 0
        model_ready = ModelManager.get().is_ready
        self.btn_run.setEnabled(has_files and model_ready)

    # ===== 批量识别 =====
    def _run_batch(self):
        paths = self.upload_component.get_file_paths()
        if not paths:
            return
        images = self.upload_component.get_images()
        for path, img in zip(paths, images):
            self._image_map[path] = img  # 保存映射

        # 检查模型
        manager = ModelManager.get()
        if not manager.is_ready:
            QMessageBox.warning(self, "模型未加载", "请先在设置页加载模型，或等待模型加载完成。")
            return

        # 检查是否已有推理在运行
        if self._worker and self._worker.isRunning():
            return

        # 获取图片
        images = self.upload_component.get_images()
        if not images:
            QMessageBox.warning(self, "提示", "无法读取图片，请重新上传。")
            return

        filenames = [p.replace("\\", "/").split("/")[-1] for p in paths]

        self._results = []
        self._batch_start_time = time.time()
        self.result_panel.clear_results()
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_label.setText("0/" + str(len(paths)))

        AppState.get().is_inferencing = True

        # 创建并启动推理线程
        self._worker = InferenceWorker()
        self._worker.setup_batch(
            images=images,
            filenames=filenames,
            top_k=5,
            use_clahe=True,
            return_heatmap=True,  # 热力图
        )
        self._worker.batchProgress.connect(self._on_progress)
        self._worker.batchFinished.connect(self._on_finished)
        self._worker.inferenceError.connect(self._on_error)
        self._worker.start()

    def _cancel_batch(self):
        """取消批量推理"""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.progress_label.setText("取消中...")
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, current: int, total: int, result: dict):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

        elapsed = time.time() - self._batch_start_time
        if current > 0 and current < total:
            eta = elapsed / current * (total - current)
            self.progress_label.setText(f"{current}/{total} ETA:{eta:.0f}s")
        else:
            self.progress_label.setText(f"{current}/{total}")

        # 查找对应的原始图片
        filename = result.get("filename", "")
        original_image = None
        for path, img in self._image_map.items():
            if path.endswith(filename) or filename in path:
                original_image = img
                break

        # 构建完整结果
        rec_result = RecognitionResult(
            filename=filename,
            class_name=result.get("class_name", ""),
            confidence=result.get("confidence", 0.0),
            latency=result.get("latency"),
            image=original_image,
            overlay_image=result.get("overlay_image"),  # 来自 manager.predict
            heatmap=result.get("heatmap"),              # 来自 manager.predict
            top_k=result.get("top_k", []),
        )
        self._results.append(rec_result)
        self.result_panel.add_result(rec_result)

        # 更新统计
        self.stat_total._value_label.setText(str(len(self._results)))
        confs = [r.confidence for r in self._results]
        self.stat_avg._value_label.setText(f"{sum(confs)/len(confs):.1%}")
        self.stat_high._value_label.setText(str(sum(1 for c in confs if c > 0.9)))

    def _on_finished(self, results: list):
        self.btn_run.setEnabled(ModelManager.get().is_ready)
        self.btn_cancel.setEnabled(False)
        elapsed = time.time() - self._batch_start_time
        self.progress_label.setText(f"完成 ({elapsed:.1f}s)")

        AppState.get().is_inferencing = False

        # 通知事件总线
        EventBus.get().history_updated.emit()

        # 弹窗统计摘要
        if results:
            confs = [r.get("confidence", 0) for r in results if r.get("error") is None]
            errors = sum(1 for r in results if r.get("error"))
            msg = f"批量识别完成!\n\n总计: {len(results)} 张\n成功: {len(confs)} 张\n失败: {errors} 张\n耗时: {elapsed:.1f}s"
            if confs:
                msg += f"\n平均置信度: {sum(confs)/len(confs):.1%}"
            QMessageBox.information(self, "批量识别完成", msg)

    def _on_error(self, error_msg: str):
        self.btn_run.setEnabled(ModelManager.get().is_ready)
        self.btn_cancel.setEnabled(False)
        self.progress_label.setText("出错")
        AppState.get().is_inferencing = False

        QMessageBox.critical(self, "批量识别失败", error_msg)

    def apply_scale(self, scale: float):
        sm = ScaleManager.get()
        layout = self.layout()
        if layout:
            m = sm.scale_int(self.BASE_MARGIN)
            v = sm.scale_int(self.BASE_MARGIN_V)
            layout.setContentsMargins(m, v, m, v)
        self.btn_run.setMinimumHeight(sm.scale_int(self.BASE_BTN_MIN_H))
        self.btn_cancel.setMinimumHeight(sm.scale_int(self.BASE_BTN_MIN_H))
        self.progress_label.setFixedWidth(sm.scale_int(self.BASE_PROGRESS_W))
        for card in [self.stat_total, self.stat_avg, self.stat_high]:
            card.setMinimumHeight(sm.scale_int(self.BASE_STAT_MIN_H))
        if hasattr(self.upload_component, 'apply_scale'):
            self.upload_component.apply_scale(scale)
        if hasattr(self.result_panel, 'apply_scale'):
            self.result_panel.apply_scale(scale)
