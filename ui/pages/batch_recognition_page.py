"""
批量识别页面 - 批量上传 + 进度 + 统计
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QFileDialog, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from ui.styles import (
    PRIMARY_COLOR, SUCCESS_COLOR, TEXT_PRIMARY,
    TEXT_SECONDARY, BORDER_COLOR, BG_COLOR,
)
from ui.event_bus import EventBus


class BatchWorker(QThread):
    """批量识别工作线程（占位）"""
    progress = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(dict)
    finished_all = pyqtSignal()

    def __init__(self, file_paths: list):
        super().__init__()
        self.file_paths = file_paths

    def run(self):
        import time, random
        for i, path in enumerate(self.file_paths):
            time.sleep(0.3)  # 模拟推理
            self.progress.emit(i + 1, len(self.file_paths))

            result = {
                "filename": path.split("/")[-1].split("\\")[-1],
                "class_name": f"Species_{i+1}",
                "confidence": round(0.7 + random.random() * 0.28, 4),
            }
            self.result_ready.emit(result)

        self.finished_all.emit()


class BatchRecognitionPage(QWidget):
    """批量识别页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_paths = []
        self._results = []
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
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

        # 上传行
        upload_row = QHBoxLayout()
        self.btn_select = QPushButton("📁 选择图片（多选）")
        self.btn_select.setObjectName("SecondaryButton")
        self.btn_select.setCursor(Qt.PointingHandCursor)
        self.btn_select.clicked.connect(self._select_files)
        upload_row.addWidget(self.btn_select)

        self.file_count_label = QLabel("未选择文件")
        self.file_count_label.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        upload_row.addWidget(self.file_count_label)
        upload_row.addStretch()
        main_card_layout.addLayout(upload_row)

        # 操作行
        action_row = QHBoxLayout()
        self.btn_run = QPushButton("🚀 批量识别")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setEnabled(False)
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_run.clicked.connect(self._run_batch)
        action_row.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        action_row.addWidget(self.progress_bar, 1)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"color: {TEXT_SECONDARY}; border: none;")
        self.progress_label.setFixedWidth(120)
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

        # 结果表格
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["文件名", "识别类别", "置信度"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background: white;
                gridline-color: {BORDER_COLOR};
            }}
            QTableWidget::item:alternate {{
                background-color: {BG_COLOR};
            }}
            QHeaderView::section {{
                background-color: {BG_COLOR};
                border: none;
                padding: 8px;
                font-weight: bold;
                color: {TEXT_PRIMARY};
                border-bottom: 1px solid {BORDER_COLOR};
            }}
        """)
        main_card_layout.addWidget(self.table, 1)

        layout.addWidget(main_card, 1)

    def _make_stat_card(self, title: str, value: str) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumHeight(100)
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

    def _select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择鸟类图片", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
        )
        if paths:
            self._file_paths = paths
            self.file_count_label.setText(f"已选择 {len(paths)} 张图片")
            self.btn_run.setEnabled(True)

    def _run_batch(self):
        if not self._file_paths:
            return

        self._results = []
        self.table.setRowCount(0)
        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.progress_label.show()

        self._worker = BatchWorker(self._file_paths)
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_result)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total}")

    def _on_result(self, result: dict):
        self._results.append(result)

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(result["filename"]))
        self.table.setItem(row, 1, QTableWidgetItem(result["class_name"]))

        conf = result["confidence"]
        conf_item = QTableWidgetItem(f"{conf:.1%}")
        conf_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 2, conf_item)

        # 更新统计
        self.stat_total._value_label.setText(str(len(self._results)))
        confs = [r["confidence"] for r in self._results]
        self.stat_avg._value_label.setText(f"{sum(confs)/len(confs):.1%}")
        self.stat_high._value_label.setText(str(sum(1 for c in confs if c > 0.9)))

    def _on_finished(self):
        self.btn_run.setEnabled(True)
        self.progress_label.setText("完成 ✅")
