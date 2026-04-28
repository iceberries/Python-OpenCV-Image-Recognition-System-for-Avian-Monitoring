"""
单图识别页面 - 上传图片 + 识别 + 结果展示
"""
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QFileDialog, QProgressBar, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont

from ui.styles import (
    PRIMARY_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR, BG_COLOR,
)
from ui.event_bus import EventBus


class RecognitionWorker(QThread):
    """识别工作线程（占位，后续接入模型）"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.image = image

    def run(self):
        # TODO: 接入实际模型推理
        import time
        for i in range(1, 101, 20):
            time.sleep(0.2)
            self.progress.emit(i)

        # 模拟结果
        result = {
            "class_name": "Blue Jay",
            "confidence": 0.92,
            "top_k": [
                {"class_name": "Blue Jay", "confidence": 0.92},
                {"class_name": "Stellar's Jay", "confidence": 0.05},
                {"class_name": "Florida Jay", "confidence": 0.02},
            ],
        }
        self.finished.emit(result)


class SingleRecognitionPage(QWidget):
    """单图识别页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image = None
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        # 标题
        title = QLabel("🔍 单图识别")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        # 主体两栏
        body = QHBoxLayout()
        body.setSpacing(24)

        # ===== 左栏 - 上传（统一卡片） =====
        left_card = QFrame()
        left_card.setObjectName("Card")
        left_card.setMinimumHeight(460)
        left_layout = QVBoxLayout(left_card)
        left_layout.setSpacing(12)

        upload_label = QLabel("📤 上传图片")
        upload_label.setObjectName("SectionTitle")
        upload_label.setStyleSheet("border: none;")
        left_layout.addWidget(upload_label)

        # 上传区域
        self.upload_area = QFrame()
        self.upload_area.setObjectName("UploadArea")
        upload_layout = QVBoxLayout(self.upload_area)
        upload_layout.setAlignment(Qt.AlignCenter)

        self.upload_hint = QLabel("点击下方按钮选择图片\n支持 JPG / PNG / BMP")
        self.upload_hint.setAlignment(Qt.AlignCenter)
        self.upload_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px; border: none;")
        upload_layout.addWidget(self.upload_hint)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setMaximumSize(500, 400)
        self.image_label.hide()
        upload_layout.addWidget(self.image_label)

        left_layout.addWidget(self.upload_area, 1)

        # 按钮行
        btn_row = QHBoxLayout()
        self.btn_select = QPushButton("📁 选择图片")
        self.btn_select.setObjectName("SecondaryButton")
        self.btn_select.setCursor(Qt.PointingHandCursor)
        self.btn_select.clicked.connect(self._select_image)
        btn_row.addWidget(self.btn_select)

        self.btn_recognize = QPushButton("🚀 开始识别")
        self.btn_recognize.setObjectName("PrimaryButton")
        self.btn_recognize.setCursor(Qt.PointingHandCursor)
        self.btn_recognize.setEnabled(False)
        self.btn_recognize.clicked.connect(self._run_recognition)
        btn_row.addWidget(self.btn_recognize)
        left_layout.addLayout(btn_row)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)

        body.addWidget(left_card, 1)

        # ===== 右栏 - 结果（统一卡片） =====
        right_card = QFrame()
        right_card.setObjectName("Card")
        right_card.setMinimumHeight(460)
        right_layout = QVBoxLayout(right_card)
        right_layout.setSpacing(12)

        result_label = QLabel("📋 识别结果")
        result_label.setObjectName("SectionTitle")
        result_label.setStyleSheet("border: none;")
        right_layout.addWidget(result_label)

        self.result_class_label = QLabel("等待识别...")
        self.result_class_label.setAlignment(Qt.AlignCenter)
        self.result_class_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.result_class_label.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        right_layout.addWidget(self.result_class_label)

        self.result_conf_label = QLabel("")
        self.result_conf_label.setAlignment(Qt.AlignCenter)
        self.result_conf_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        self.result_conf_label.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        right_layout.addWidget(self.result_conf_label)

        # Top-K 区域
        self.top_k_frame = QFrame()
        self.top_k_frame.setStyleSheet("border: none;")
        self.top_k_layout = QVBoxLayout(self.top_k_frame)
        self.top_k_layout.setContentsMargins(0, 8, 0, 0)
        right_layout.addWidget(self.top_k_frame)

        right_layout.addStretch()

        body.addWidget(right_card, 1)

        layout.addLayout(body)

    def _select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择鸟类图片", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return

        self._current_image_path = path
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.show()
        self.upload_hint.hide()
        self.btn_recognize.setEnabled(True)

        # 存入 EventBus 通知
        img = QImage(path)
        ptr = img.bits()
        ptr.setsize(img.height() * img.bytesPerLine())
        arr = np.array(img).copy()
        EventBus.get().image_uploaded.emit(arr)

    def _run_recognition(self):
        if self._current_image_path is None:
            return

        self.btn_recognize.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self._worker = RecognitionWorker(np.zeros((224, 224, 3)))
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_result)
        self._worker.start()

    def _on_result(self, result: dict):
        self.progress_bar.hide()
        self.btn_recognize.setEnabled(True)

        class_name = result["class_name"]
        confidence = result["confidence"]

        # 置信度颜色
        if confidence > 0.9:
            conf_color = SUCCESS_COLOR
            conf_id = "ConfHigh"
        elif confidence > 0.7:
            conf_color = WARNING_COLOR
            conf_id = "ConfMedium"
        else:
            conf_color = DANGER_COLOR
            conf_id = "ConfLow"

        self.result_class_label.setText(class_name)
        self.result_class_label.setStyleSheet(f"color: {TEXT_PRIMARY}; border: none; font-size: 18px; font-weight: bold;")
        self.result_conf_label.setText(f"{confidence:.1%}")
        self.result_conf_label.setObjectName(conf_id)
        self.result_conf_label.setStyleSheet(f"color: {conf_color}; border: none; font-size: 24px; font-weight: bold;")

        # Top-K
        # 清除旧的
        while self.top_k_layout.count():
            child = self.top_k_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        top_k = result.get("top_k", [])
        for i, item in enumerate(top_k):
            bar_color = PRIMARY_COLOR if i == 0 else "#adb5bd"
            conf = item["confidence"]

            row = QHBoxLayout()
            name_l = QLabel(item["class_name"])
            name_l.setFixedWidth(120)
            name_l.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; border: none;")
            row.addWidget(name_l)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(conf * 100))
            bar.setFixedHeight(8)
            bar.setFormat("")
            bar.setStyleSheet(f"""
                QProgressBar {{ border: none; border-radius: 4px; background: {BORDER_COLOR}; }}
                QProgressBar::chunk {{ background: {bar_color}; border-radius: 4px; }}
            """)
            row.addWidget(bar)

            val_l = QLabel(f"{conf:.1%}")
            val_l.setFixedWidth(50)
            val_l.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; border: none;")
            row.addWidget(val_l)

            self.top_k_layout.addLayout(row)

        # 通知事件总线
        EventBus.get().recognition_completed.emit(result)
        EventBus.get().history_updated.emit()
