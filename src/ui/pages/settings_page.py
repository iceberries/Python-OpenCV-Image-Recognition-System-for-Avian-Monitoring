"""
设置页面 - 模型配置、信息展示与系统参数
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QComboBox, QSpinBox,
    QDoubleSpinBox, QGroupBox, QFormLayout, QMessageBox,
    QFileDialog, QProgressBar,
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QFont

from src.ui.styles import TEXT_PRIMARY, TEXT_SECONDARY, PRIMARY_COLOR, BORDER_COLOR, DANGER_COLOR
from src.ui.scale_manager import ScaleManager
from src.core.model_manager import ModelManager
from src.core.app_state import AppState
from src.utils.model_utils import format_parameter_count, get_gpu_memory_info
from src.config import (
    OUTPUT_DIR, PROJECT_ROOT, NUM_CLASSES, INPUT_SIZE,
    BATCH_SIZE, LEARNING_RATE, LABEL_SMOOTHING,
    USE_TTA, USE_SE_ATTENTION, USE_COSINE_LR,
)


class SettingsPage(QWidget):
    """设置页面"""

    BASE_MARGIN = 32
    BASE_MARGIN_V = 24

    def __init__(self, parent=None):
        super().__init__(parent)

        self._settings = QSettings("AvianMonitor", "BirdRecognition")

        sm = ScaleManager.get()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
        )
        layout.setSpacing(16)

        # 标题
        title = QLabel("⚙️ 设置")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        # ===== 模型信息卡片 =====
        model_info_group = QGroupBox("🧠 模型信息")
        model_info_group.setStyleSheet(self._group_style())
        info_layout = QVBoxLayout(model_info_group)

        # 状态行
        self._model_status_label = QLabel("❌ 模型未加载")
        self._model_status_label.setStyleSheet(f"color: {DANGER_COLOR}; font-weight: bold; font-size: 15px; border: none;")
        info_layout.addWidget(self._model_status_label)

        # 信息表
        self._info_form = QFormLayout()
        self._info_form.addRow("模型架构:", QLabel("ResNet50 + SE-Attention"))
        self._info_form.addRow("类别数:", QLabel(str(NUM_CLASSES)))
        self._info_form.addRow("输入尺寸:", QLabel(f"{INPUT_SIZE}×{INPUT_SIZE}"))
        self._param_label = QLabel("-")
        self._info_form.addRow("参数量:", self._param_label)
        self._device_label = QLabel("-")
        self._info_form.addRow("推理设备:", self._device_label)
        self._acc_label = QLabel("-")
        self._info_form.addRow("训练精度:", self._acc_label)
        self._class_count_label = QLabel("-")
        self._info_form.addRow("类别文件:", self._class_count_label)
        info_layout.addLayout(self._info_form)

        # 加载进度条
        self._load_progress = QProgressBar()
        self._load_progress.setValue(0)
        self._load_progress.hide()
        info_layout.addWidget(self._load_progress)

        layout.addWidget(model_info_group)

        # ===== 模型加载操作 =====
        load_group = QGroupBox("🔄 模型加载")
        load_group.setStyleSheet(self._group_style())
        load_layout = QVBoxLayout(load_group)

        # 模型路径
        path_row = QHBoxLayout()
        self._model_path_label = QLabel(self._get_saved_model_path())
        self._model_path_label.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        self._model_path_label.setWordWrap(True)
        path_row.addWidget(self._model_path_label, 1)

        btn_browse = QPushButton("📁 选择权重")
        btn_browse.setObjectName("SecondaryButton")
        btn_browse.setCursor(Qt.PointingHandCursor)
        btn_browse.clicked.connect(self._browse_model)
        path_row.addWidget(btn_browse)
        load_layout.addLayout(path_row)

        # 加载/卸载按钮
        btn_row = QHBoxLayout()
        self._btn_load = QPushButton("🔄 加载模型")
        self._btn_load.setObjectName("PrimaryButton")
        self._btn_load.setCursor(Qt.PointingHandCursor)
        self._btn_load.clicked.connect(self._load_model)
        btn_row.addWidget(self._btn_load)

        self._btn_unload = QPushButton("🗑️ 卸载模型")
        self._btn_unload.setObjectName("SecondaryButton")
        self._btn_unload.setCursor(Qt.PointingHandCursor)
        self._btn_unload.setEnabled(False)
        self._btn_unload.clicked.connect(self._unload_model)
        btn_row.addWidget(self._btn_unload)
        btn_row.addStretch()
        load_layout.addLayout(btn_row)

        layout.addWidget(load_group)

        # ===== 环境检查 =====
        env_group = QGroupBox("🔍 环境检查")
        env_group.setStyleSheet(self._group_style())
        env_layout = QVBoxLayout(env_group)

        model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
        dataset_dir = os.path.join(PROJECT_ROOT, "CUB_200_2011", "CUB_200_2011")

        model_status = "✅ 已找到" if os.path.exists(model_path) else "❌ 未找到"
        dataset_status = "✅ 已找到" if os.path.exists(dataset_dir) else "❌ 未找到"

        env_layout.addWidget(QLabel(f"模型文件 ({model_path}): {model_status}"))
        env_layout.addWidget(QLabel(f"数据集目录: {dataset_status}"))

        # GPU 信息
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            env_layout.addWidget(QLabel(
                f"GPU: {gpu_info['total_mb']:.0f}MB 总显存, "
                f"{gpu_info['free_mb']:.0f}MB 空闲"
            ))
        else:
            env_layout.addWidget(QLabel("GPU: 不可用, 使用 CPU"))

        layout.addWidget(env_group)

        # ===== 模型参数（只读） =====
        model_group = QGroupBox("📊 训练配置（只读）")
        model_group.setStyleSheet(self._group_style())
        model_form = QFormLayout(model_group)

        model_form.addRow("类别数:", QLabel(str(NUM_CLASSES)))
        model_form.addRow("输入尺寸:", QLabel(f"{INPUT_SIZE}×{INPUT_SIZE}"))
        model_form.addRow("批次大小:", QLabel(str(BATCH_SIZE)))
        model_form.addRow("学习率:", QLabel(str(LEARNING_RATE)))
        model_form.addRow("Label Smoothing:", QLabel(str(LABEL_SMOOTHING)))
        model_form.addRow("TTA:", QLabel("开启" if USE_TTA else "关闭"))
        model_form.addRow("SE-Attention:", QLabel("开启" if USE_SE_ATTENTION else "关闭"))
        model_form.addRow("Cosine LR:", QLabel("开启" if USE_COSINE_LR else "关闭"))
        layout.addWidget(model_group)

        # ===== 运行时设置 =====
        runtime_group = QGroupBox("⚡ 运行时设置")
        runtime_group.setStyleSheet(self._group_style())
        runtime_form = QFormLayout(runtime_group)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self._settings.value("confidence_threshold", 0.5, float))
        self.conf_spin.setDecimals(2)
        runtime_form.addRow("置信度阈值:", self.conf_spin)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 20)
        self.topk_spin.setValue(self._settings.value("top_k", 5, int))
        runtime_form.addRow("Top-K 数量:", self.topk_spin)

        self.clahe_check = QComboBox()
        self.clahe_check.addItems(["开启", "关闭"])
        self.clahe_check.setCurrentIndex(0 if self._settings.value("use_clahe", True, bool) else 1)
        runtime_form.addRow("CLAHE 预处理:", self.clahe_check)

        layout.addWidget(runtime_group)

        layout.addStretch()

        # 保存按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_save = QPushButton("💾 保存设置")
        self.btn_save.setObjectName("PrimaryButton")
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.clicked.connect(self._save_settings)
        btn_row.addWidget(self.btn_save)
        layout.addLayout(btn_row)

        # 连接信号
        AppState.get().modelStatusChanged.connect(self._on_model_status_changed)
        ModelManager.get().modelLoaded.connect(self._on_model_loaded)
        ModelManager.get().modelLoadingProgress.connect(self._on_loading_progress)

        # 初始刷新
        self._refresh_model_info()

    # ===== 样式 =====
    def _group_style(self) -> str:
        return f"""
            QGroupBox {{
                font-weight: bold; font-size: 15px;
                color: {TEXT_PRIMARY}; border: 1px solid {BORDER_COLOR};
                border-radius: 8px; margin-top: 12px; padding-top: 20px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 16px; padding: 0 8px;
            }}
        """

    # ===== 模型路径 =====
    def _get_saved_model_path(self) -> str:
        return self._settings.value(
            "model_path",
            os.path.join(OUTPUT_DIR, "best_model.pth"),
        )

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重文件", OUTPUT_DIR,
            "PyTorch 模型 (*.pth *.pt);;所有文件 (*)",
        )
        if path:
            self._model_path_label.setText(path)
            self._settings.setValue("model_path", path)

    # ===== 模型加载/卸载 =====
    def _load_model(self):
        path = self._model_path_label.text()
        if not os.path.exists(path):
            QMessageBox.warning(self, "文件不存在", f"模型文件不存在:\n{path}")
            return

        self._btn_load.setEnabled(False)
        self._load_progress.setValue(0)
        self._load_progress.show()
        self._model_status_label.setText("⏳ 模型加载中...")
        self._model_status_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: bold; font-size: 15px; border: none;")

        # 异步加载
        manager = ModelManager.get()
        device = "cuda" if _torch_cuda_available() else "cpu"
        manager.load_model_async(checkpoint_path=path, device=device)

    def _unload_model(self):
        reply = QMessageBox.question(
            self, "确认卸载", "确定要卸载当前模型吗？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            ModelManager.get().release()
            self._refresh_model_info()

    def _on_model_loaded(self, success: bool, message: str):
        self._load_progress.hide()
        self._btn_load.setEnabled(True)
        self._refresh_model_info()
        if success:
            QMessageBox.information(self, "模型加载成功", message)
        else:
            QMessageBox.critical(self, "模型加载失败", message)

    def _on_loading_progress(self, percent: int):
        self._load_progress.setValue(percent)

    def _on_model_status_changed(self, ready: bool, model_name: str):
        self._refresh_model_info()

    def _refresh_model_info(self):
        """刷新模型信息显示"""
        manager = ModelManager.get()
        ready = manager.is_ready

        if ready:
            self._model_status_label.setText("✅ 模型已加载")
            self._model_status_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-weight: bold; font-size: 15px; border: none;")
            self._btn_unload.setEnabled(True)

            info = manager.get_model_info()
            self._param_label.setText(format_parameter_count(info.get("param_total", 0)))
            self._device_label.setText(info.get("device", "-"))
            meta = info.get("checkpoint_meta", {})
            acc = meta.get("best_accuracy", "N/A")
            self._acc_label.setText(f"{acc}%" if isinstance(acc, (int, float)) else str(acc))
            self._class_count_label.setText(f"{info.get('class_count', 0)} 个类别")
        else:
            self._model_status_label.setText("❌ 模型未加载")
            self._model_status_label.setStyleSheet(f"color: {DANGER_COLOR}; font-weight: bold; font-size: 15px; border: none;")
            self._btn_unload.setEnabled(False)
            self._param_label.setText("-")
            self._device_label.setText("-")
            self._acc_label.setText("-")
            self._class_count_label.setText("-")

    # ===== 保存设置 =====
    def _save_settings(self):
        self._settings.setValue("confidence_threshold", self.conf_spin.value())
        self._settings.setValue("top_k", self.topk_spin.value())
        self._settings.setValue("use_clahe", self.clahe_check.currentIndex() == 0)
        self._settings.setValue("model_path", self._model_path_label.text())
        QMessageBox.information(self, "提示", "设置已保存")

    def apply_scale(self, scale: float):
        sm = ScaleManager.get()
        layout = self.layout()
        if layout:
            m = sm.scale_int(self.BASE_MARGIN)
            v = sm.scale_int(self.BASE_MARGIN_V)
            layout.setContentsMargins(m, v, m, v)


def _torch_cuda_available() -> bool:
    """安全检测 CUDA"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
