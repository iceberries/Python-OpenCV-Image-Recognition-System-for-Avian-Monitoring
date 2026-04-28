"""
设置页面 - 模型配置与系统参数
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QComboBox, QSpinBox,
    QDoubleSpinBox, QGroupBox, QFormLayout, QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.styles import TEXT_PRIMARY, TEXT_SECONDARY, PRIMARY_COLOR, BORDER_COLOR
from src.config import (
    OUTPUT_DIR, PROJECT_ROOT, NUM_CLASSES, INPUT_SIZE,
    BATCH_SIZE, LEARNING_RATE, LABEL_SMOOTHING,
    USE_TTA, USE_SE_ATTENTION, USE_COSINE_LR,
)


class SettingsPage(QWidget):
    """设置页面"""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        # 标题
        title = QLabel("⚙️ 设置")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        # 环境检查
        env_group = QGroupBox("🔍 环境检查")
        env_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold; font-size: 15px;
                color: {TEXT_PRIMARY}; border: 1px solid {BORDER_COLOR};
                border-radius: 8px; margin-top: 12px; padding-top: 20px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 16px; padding: 0 8px;
            }}
        """)
        env_layout = QVBoxLayout(env_group)

        model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
        dataset_dir = os.path.join(PROJECT_ROOT, "CUB_200_2011", "CUB_200_2011")

        model_status = "✅ 已找到" if os.path.exists(model_path) else "❌ 未找到"
        dataset_status = "✅ 已找到" if os.path.exists(dataset_dir) else "❌ 未找到"

        env_layout.addWidget(QLabel(f"模型文件 ({model_path}): {model_status}"))
        env_layout.addWidget(QLabel(f"数据集目录: {dataset_status}"))
        layout.addWidget(env_group)

        # 模型参数
        model_group = QGroupBox("🧠 模型参数（只读）")
        model_group.setStyleSheet(env_group.styleSheet())
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

        # 运行时设置
        runtime_group = QGroupBox("⚡ 运行时设置")
        runtime_group.setStyleSheet(env_group.styleSheet())
        runtime_form = QFormLayout(runtime_group)

        self.device_combo = QComboBox()
        import torch
        if torch.cuda.is_available():
            self.device_combo.addItem(f"CUDA ({torch.cuda.get_device_name(0)})")
        self.device_combo.addItem("CPU")
        runtime_form.addRow("推理设备:", self.device_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setDecimals(2)
        runtime_form.addRow("置信度阈值:", self.conf_spin)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 20)
        self.topk_spin.setValue(5)
        runtime_form.addRow("Top-K 数量:", self.topk_spin)
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

    def _save_settings(self):
        # TODO: 持久化设置
        QMessageBox.information(self, "提示", "设置已保存（当前仅会话内有效）")
