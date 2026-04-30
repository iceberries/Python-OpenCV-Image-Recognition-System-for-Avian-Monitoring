"""
设置页面 - 模型配置、信息展示与系统参数
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QComboBox, QSpinBox,
    QDoubleSpinBox, QGroupBox, QFormLayout, QMessageBox,
    QFileDialog, QProgressBar, QScrollArea, QSizePolicy,
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

# 基准字体尺寸
BASE_TITLE_FONT = 15
BASE_GROUP_TITLE_FONT = 14
BASE_LABEL_FONT = 13
BASE_FORM_V_SPACING = 10
BASE_FORM_H_SPACING = 16
BASE_GROUP_PADDING_TOP = 24
BASE_GROUP_MARGIN_TOP = 14


class SettingsPage(QWidget):
    """设置页面"""

    BASE_MARGIN = 32
    BASE_MARGIN_V = 24

    def __init__(self, parent=None):
        super().__init__(parent)

        self._settings = QSettings("AvianMonitor", "BirdRecognition")

        # 需要随缩放更新的组件
        self._group_boxes = []
        self._form_layouts = []
        self._form_labels = []     # QFormLayout 中的所有 QLabel
        self._status_labels = []   # 状态标签（带颜色）

        sm = ScaleManager.get()
        scale = sm.scale

        # 外层滚动区域，防止窗口过小时内容被截断
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
        )
        layout.setSpacing(sm.scale_int(16))

        # 标题
        title = QLabel("⚙️ 设置")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        # ===== 模型信息卡片 =====
        model_info_group, info_layout = self._create_group("🧠 模型信息", layout)

        # 状态行
        self._model_status_label = QLabel("❌ 模型未加载")
        self._apply_status_style(self._model_status_label, DANGER_COLOR)
        info_layout.addWidget(self._model_status_label)
        self._status_labels.append(self._model_status_label)

        # 信息表 - 使用足够行间距
        info_form = QFormLayout()
        info_form.setVerticalSpacing(sm.scale_int(BASE_FORM_V_SPACING))
        info_form.setHorizontalSpacing(sm.scale_int(BASE_FORM_H_SPACING))
        info_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        self._info_form = info_form
        self._form_layouts.append(info_form)

        for label_text, default_val in [
            ("模型架构:", "ResNet50 + SE-Attention"),
            ("类别数:", str(NUM_CLASSES)),
            ("输入尺寸:", f"{INPUT_SIZE}×{INPUT_SIZE}"),
        ]:
            lbl = QLabel(label_text)
            val = QLabel(default_val)
            self._set_scaled_font(lbl, BASE_LABEL_FONT)
            self._set_scaled_font(val, BASE_LABEL_FONT)
            self._form_labels.extend([lbl, val])
            info_form.addRow(lbl, val)

        self._param_label = QLabel("-")
        self._add_form_row(info_form, "参数量:", self._param_label)
        self._device_label = QLabel("-")
        self._add_form_row(info_form, "推理设备:", self._device_label)
        self._acc_label = QLabel("-")
        self._add_form_row(info_form, "训练精度:", self._acc_label)
        self._class_count_label = QLabel("-")
        self._add_form_row(info_form, "类别文件:", self._class_count_label)

        info_layout.addLayout(info_form)

        # 加载进度条
        self._load_progress = QProgressBar()
        self._load_progress.setValue(0)
        self._load_progress.hide()
        info_layout.addWidget(self._load_progress)

        # ===== 模型加载操作 =====
        load_group, load_layout = self._create_group("🔄 模型加载", layout)

        # 模型路径
        path_row = QHBoxLayout()
        self._model_path_label = QLabel(self._get_saved_model_path())
        self._model_path_label.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        self._model_path_label.setWordWrap(True)
        self._set_scaled_font(self._model_path_label, BASE_LABEL_FONT)
        self._form_labels.append(self._model_path_label)
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

        # ===== 环境检查 =====
        env_group, env_layout = self._create_group("🔍 环境检查", layout)

        model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
        dataset_dir = os.path.join(PROJECT_ROOT, "CUB_200_2011", "CUB_200_2011")

        model_status = "✅ 已找到" if os.path.exists(model_path) else "❌ 未找到"
        dataset_status = "✅ 已找到" if os.path.exists(dataset_dir) else "❌ 未找到"

        for text in [
            f"模型文件 ({model_path}): {model_status}",
            f"数据集目录: {dataset_status}",
        ]:
            lbl = QLabel(text)
            self._set_scaled_font(lbl, BASE_LABEL_FONT)
            self._form_labels.append(lbl)
            lbl.setWordWrap(True)
            env_layout.addWidget(lbl)

        # GPU 信息
        gpu_info = get_gpu_memory_info()
        gpu_text = (
            f"GPU: {gpu_info['total_mb']:.0f}MB 总显存, {gpu_info['free_mb']:.0f}MB 空闲"
            if gpu_info else "GPU: 不可用, 使用 CPU"
        )
        gpu_lbl = QLabel(gpu_text)
        self._set_scaled_font(gpu_lbl, BASE_LABEL_FONT)
        self._form_labels.append(gpu_lbl)
        env_layout.addWidget(gpu_lbl)

        # ===== 训练配置（只读） =====
        train_group, train_inner = self._create_group("📊 训练配置（只读）", layout)
        train_form = QFormLayout()
        train_form.setVerticalSpacing(sm.scale_int(BASE_FORM_V_SPACING))
        train_form.setHorizontalSpacing(sm.scale_int(BASE_FORM_H_SPACING))
        train_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        train_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self._form_layouts.append(train_form)

        for label_text, val_text in [
            ("类别数:", str(NUM_CLASSES)),
            ("输入尺寸:", f"{INPUT_SIZE}×{INPUT_SIZE}"),
            ("批次大小:", str(BATCH_SIZE)),
            ("学习率:", str(LEARNING_RATE)),
            ("Label Smoothing:", str(LABEL_SMOOTHING)),
            ("TTA:", "开启" if USE_TTA else "关闭"),
            ("SE-Attention:", "开启" if USE_SE_ATTENTION else "关闭"),
            ("Cosine LR:", "开启" if USE_COSINE_LR else "关闭"),
        ]:
            lbl = QLabel(label_text)
            val = QLabel(val_text)
            self._set_scaled_font(lbl, BASE_LABEL_FONT)
            self._set_scaled_font(val, BASE_LABEL_FONT)
            self._form_labels.extend([lbl, val])
            train_form.addRow(lbl, val)

        train_inner.addLayout(train_form)

        # ===== 运行时设置 =====
        runtime_group, runtime_inner = self._create_group("⚡ 运行时设置", layout)
        runtime_form = QFormLayout()
        runtime_form.setVerticalSpacing(sm.scale_int(BASE_FORM_V_SPACING))
        runtime_form.setHorizontalSpacing(sm.scale_int(BASE_FORM_H_SPACING))
        runtime_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._form_layouts.append(runtime_form)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self._settings.value("confidence_threshold", 0.5, float))
        self.conf_spin.setDecimals(2)
        conf_lbl = QLabel("置信度阈值:")
        self._set_scaled_font(conf_lbl, BASE_LABEL_FONT)
        self._form_labels.append(conf_lbl)
        runtime_form.addRow(conf_lbl, self.conf_spin)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 20)
        self.topk_spin.setValue(self._settings.value("top_k", 5, int))
        topk_lbl = QLabel("Top-K 数量:")
        self._set_scaled_font(topk_lbl, BASE_LABEL_FONT)
        self._form_labels.append(topk_lbl)
        runtime_form.addRow(topk_lbl, self.topk_spin)

        self.clahe_check = QComboBox()
        self.clahe_check.addItems(["开启", "关闭"])
        self.clahe_check.setCurrentIndex(0 if self._settings.value("use_clahe", True, bool) else 1)
        clahe_lbl = QLabel("CLAHE 预处理:")
        self._set_scaled_font(clahe_lbl, BASE_LABEL_FONT)
        self._form_labels.append(clahe_lbl)
        runtime_form.addRow(clahe_lbl, self.clahe_check)

        runtime_inner.addLayout(runtime_form)

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

        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # 连接信号
        AppState.get().modelStatusChanged.connect(self._on_model_status_changed)
        ModelManager.get().modelLoaded.connect(self._on_model_loaded)
        ModelManager.get().modelLoadingProgress.connect(self._on_loading_progress)

        # 初始刷新
        self._refresh_model_info()

    # ===== 辅助方法 =====

    def _create_group(self, title: str, parent_layout: QVBoxLayout) -> tuple:
        """创建 QGroupBox 并返回 (group, inner_layout)"""
        sm = ScaleManager.get()
        group = QGroupBox(title)
        group.setStyleSheet(self._group_style())
        inner = QVBoxLayout(group)
        inner.setSpacing(sm.scale_int(8))
        parent_layout.addWidget(group)
        self._group_boxes.append(group)
        return group, inner

    def _add_form_row(self, form: QFormLayout, label_text: str, value_widget: QLabel):
        """向 QFormLayout 添加一行，同时注册标签以便缩放"""
        lbl = QLabel(label_text)
        self._set_scaled_font(lbl, BASE_LABEL_FONT)
        self._set_scaled_font(value_widget, BASE_LABEL_FONT)
        self._form_labels.extend([lbl, value_widget])
        form.addRow(lbl, value_widget)

    def _set_scaled_font(self, widget: QWidget, base_size: int):
        """设置缩放后的字体"""
        sm = ScaleManager.get()
        font = widget.font()
        font.setPixelSize(sm.scale_int(base_size))
        widget.setFont(font)

    def _apply_status_style(self, label: QLabel, color: str):
        """应用状态标签样式（缩放感知）"""
        sm = ScaleManager.get()
        px = sm.scale_int(BASE_LABEL_FONT)
        label.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: {px}px; "
            f"padding: 4px 0; border: none;"
        )

    # ===== 样式 =====

    def _group_style(self) -> str:
        sm = ScaleManager.get()
        title_px = sm.scale_int(BASE_GROUP_TITLE_FONT)
        pad_top = sm.scale_int(BASE_GROUP_PADDING_TOP)
        margin_top = sm.scale_int(BASE_GROUP_MARGIN_TOP)
        return f"""
            QGroupBox {{
                font-weight: bold; font-size: {title_px}px;
                color: {TEXT_PRIMARY}; border: 1px solid {BORDER_COLOR};
                border-radius: 8px; margin-top: {margin_top}px; padding-top: {pad_top}px;
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
        self._apply_status_style(self._model_status_label, TEXT_SECONDARY)

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
            self._apply_status_style(self._model_status_label, PRIMARY_COLOR)
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
            self._apply_status_style(self._model_status_label, DANGER_COLOR)
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
        """窗口缩放时更新所有尺寸"""
        sm = ScaleManager.get()

        # 更新外层 margin
        layout = self.layout()
        if layout:
            m = sm.scale_int(self.BASE_MARGIN)
            v = sm.scale_int(self.BASE_MARGIN_V)
            layout.setContentsMargins(m, v, m, v)

        # 更新 QGroupBox 样式（字体和间距）
        group_style = self._group_style()
        for group in self._group_boxes:
            group.setStyleSheet(group_style)

        # 更新 QFormLayout 间距
        v_sp = sm.scale_int(BASE_FORM_V_SPACING)
        h_sp = sm.scale_int(BASE_FORM_H_SPACING)
        for form in self._form_layouts:
            form.setVerticalSpacing(v_sp)
            form.setHorizontalSpacing(h_sp)

        # 更新所有标签字体
        for lbl in self._form_labels:
            font = lbl.font()
            font.setPixelSize(sm.scale_int(BASE_LABEL_FONT))
            lbl.setFont(font)

        # 更新状态标签样式（含字体大小）
        manager = ModelManager.get()
        color = PRIMARY_COLOR if manager.is_ready else DANGER_COLOR
        self._apply_status_style(self._model_status_label, color)


def _torch_cuda_available() -> bool:
    """安全检测 CUDA"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
