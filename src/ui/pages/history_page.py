"""
历史记录页面 - 持久化存储的识别历史，支持筛选/分页/排序/导出/批量操作

布局:
  顶部: 筛选栏（日期/类别/置信度/搜索）
  中部: 分页数据表格（排序+详情展开）
  底部: 操作栏（批量删除/导出/统计概览）
"""
import os
from typing import Dict

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QLineEdit,
    QDateEdit, QSlider, QCheckBox,
    QSizePolicy, QMessageBox, QDialog,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QColor, QPixmap

from src.ui.styles import (
    PRIMARY_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR, BG_COLOR, CARD_BG,
)
from src.ui.event_bus import EventBus
from src.ui.scale_manager import ScaleManager
from src.ui.history_manager import HistoryRecord, get_history_db


# ==================== 统计概览卡片 ====================

class StatCard(QFrame):
    """统计数字卡片"""

    def __init__(self, title: str, value: str = "0", color: str = PRIMARY_COLOR, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._base_min_h = 90
        self._base_min_w = 140

        sm = ScaleManager.get()
        self.setMinimumHeight(sm.scale_int(self._base_min_h))

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; border: none;")
        self.title_label.setProperty("_base_font_size", 12)
        layout.addWidget(self.title_label)

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold; border: none;")
        self.value_label.setProperty("_base_font_size", 24)
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        self.value_label.setText(value)

    def apply_scale(self, scale: float):
        sm = ScaleManager.get()
        self.setMinimumHeight(sm.scale_int(self._base_min_h))


# ==================== 详情弹窗 ====================

class RecordDetailDialog(QDialog):
    """记录详情弹窗：原图预览 + 完整 Top-K"""

    def __init__(self, record: HistoryRecord, parent=None):
        super().__init__(parent)
        self.setWindowTitle("识别详情")
        self.setMinimumSize(520, 480)
        self.setStyleSheet(f"background-color: {CARD_BG};")

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title = QLabel(f"🏷️ {record.predicted_class}")
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {TEXT_PRIMARY}; border: none;")
        layout.addWidget(title)

        # 基本信息行
        info_layout = QHBoxLayout()
        info_layout.setSpacing(16)

        ts = record.timestamp[:19].replace("T", " ") if record.timestamp else ""
        info_layout.addWidget(QLabel(f"⏰ {ts}"))
        conf_color = SUCCESS_COLOR if record.confidence > 0.9 else (WARNING_COLOR if record.confidence > 0.7 else DANGER_COLOR)
        conf_label = QLabel(f"📊 置信度: {record.confidence:.1%}")
        conf_label.setStyleSheet(f"color: {conf_color}; font-weight: bold; border: none;")
        info_layout.addWidget(conf_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)

        # 图片预览
        if record.image_path and os.path.isfile(record.image_path):
            img_label = QLabel()
            pixmap = QPixmap(record.image_path).scaled(
                400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet(f"border: 1px solid {BORDER_COLOR}; border-radius: 6px;")
            layout.addWidget(img_label)
        elif record.thumbnail_path and os.path.isfile(record.thumbnail_path):
            img_label = QLabel()
            pixmap = QPixmap(record.thumbnail_path).scaled(
                400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet(f"border: 1px solid {BORDER_COLOR}; border-radius: 6px;")
            layout.addWidget(img_label)

        # Top-K 结果
        top_k = record.get_top_k()
        if top_k:
            top_k_label = QLabel("Top-K 预测结果")
            top_k_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TEXT_PRIMARY}; border: none; margin-top: 8px;")
            layout.addWidget(top_k_label)

            for i, item in enumerate(top_k[:10]):
                cls = item.get("class_name", item.get("class", "Unknown"))
                conf = item.get("confidence", 0)
                bar_w = int(conf * 100)
                color = SUCCESS_COLOR if conf > 0.9 else (WARNING_COLOR if conf > 0.7 else DANGER_COLOR)

                row = QHBoxLayout()
                row.setSpacing(8)

                rank = QLabel(f"#{i+1}")
                rank.setFixedWidth(28)
                rank.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; border: none;")
                row.addWidget(rank)

                name = QLabel(cls)
                name.setFixedWidth(160)
                name.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; border: none;")
                row.addWidget(name)

                bar_frame = QFrame()
                bar_frame.setFixedHeight(16)
                bar_frame.setStyleSheet(f"background: {BG_COLOR}; border-radius: 4px;")
                bar_layout = QHBoxLayout(bar_frame)
                bar_layout.setContentsMargins(0, 0, 0, 0)
                bar_layout.setSpacing(0)
                fill = QFrame()
                fill.setFixedWidth(bar_w)
                fill.setStyleSheet(f"background: {color}; border-radius: 4px;")
                bar_layout.addWidget(fill)
                bar_layout.addStretch()
                row.addWidget(bar_frame, 1)

                pct = QLabel(f"{conf:.1%}")
                pct.setFixedWidth(52)
                pct.setAlignment(Qt.AlignRight)
                pct.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold; border: none;")
                row.addWidget(pct)

                layout.addLayout(row)

        # 备注
        if record.notes:
            notes_label = QLabel(f"📝 {record.notes}")
            notes_label.setWordWrap(True)
            notes_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; border: none; margin-top: 8px;")
            layout.addWidget(notes_label)

        # 关闭按钮
        btn_close = QPushButton("关闭")
        btn_close.setObjectName("SecondaryButton")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close, alignment=Qt.AlignRight)


# ==================== 历史记录页面 ====================

class HistoryPage(QWidget):
    """历史记录页面 - 持久化存储"""

    # 基准尺寸常量
    BASE_MARGIN = 32
    BASE_MARGIN_V = 24
    BASE_ROW_H = 40
    BASE_FILTER_H = 36
    BASE_BTN_H = 36
    PAGE_SIZE = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = get_history_db()
        self._current_page = 1
        self._total_records = 0
        self._selected_ids: set = set()

        sm = ScaleManager.get()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
            sm.scale_int(self.BASE_MARGIN), sm.scale_int(self.BASE_MARGIN_V),
        )
        layout.setSpacing(12)

        # ========== 标题行 ==========
        header = QHBoxLayout()
        title = QLabel("📋 历史记录")
        title.setObjectName("PageTitle")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # ========== 统计概览 ==========
        self.stats_row = QHBoxLayout()
        self.stats_row.setSpacing(12)

        self.stat_total = StatCard("总记录数", "0", PRIMARY_COLOR)
        self.stat_today = StatCard("今日识别", "0", SUCCESS_COLOR)
        self.stat_avg_conf = StatCard("平均置信度", "0%", WARNING_COLOR)
        self.stat_classes = StatCard("识别类别", "0", DANGER_COLOR)

        self.stats_row.addWidget(self.stat_total)
        self.stats_row.addWidget(self.stat_today)
        self.stats_row.addWidget(self.stat_avg_conf)
        self.stats_row.addWidget(self.stat_classes)
        layout.addLayout(self.stats_row)

        # ========== 筛选栏 ==========
        filter_frame = QFrame()
        filter_frame.setObjectName("Card")
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setSpacing(10)
        filter_layout.setContentsMargins(12, 8, 12, 8)

        # 日期起
        filter_layout.addWidget(QLabel("起始日期:"))
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDate(QDate.currentDate().addMonths(-1))
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        self.date_start.setFixedHeight(sm.scale_int(self.BASE_FILTER_H))
        filter_layout.addWidget(self.date_start)

        # 日期止
        filter_layout.addWidget(QLabel("结束日期:"))
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDate(QDate.currentDate())
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        self.date_end.setFixedHeight(sm.scale_int(self.BASE_FILTER_H))
        filter_layout.addWidget(self.date_end)

        # 类别下拉
        filter_layout.addWidget(QLabel("类别:"))
        self.combo_class = QComboBox()
        self.combo_class.addItem("全部类别", "")
        self.combo_class.setFixedHeight(sm.scale_int(self.BASE_FILTER_H))
        self.combo_class.setMinimumWidth(sm.scale_int(140))
        filter_layout.addWidget(self.combo_class)

        # 置信度滑块
        filter_layout.addWidget(QLabel("最低置信度:"))
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(0)
        self.slider_conf.setFixedWidth(sm.scale_int(120))
        self.slider_conf.setToolTip("0%")
        self.label_conf_val = QLabel("0%")
        self.label_conf_val.setFixedWidth(sm.scale_int(36))
        self.slider_conf.valueChanged.connect(
            lambda v: self.label_conf_val.setText(f"{v}%")
        )
        filter_layout.addWidget(self.slider_conf)
        filter_layout.addWidget(self.label_conf_val)

        # 搜索框
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 搜索文件名...")
        self.search_input.setFixedHeight(sm.scale_int(self.BASE_FILTER_H))
        self.search_input.setMinimumWidth(sm.scale_int(160))
        filter_layout.addWidget(self.search_input)

        # 查询按钮
        self.btn_search = QPushButton("查询")
        self.btn_search.setObjectName("PrimaryButton")
        self.btn_search.setCursor(Qt.PointingHandCursor)
        self.btn_search.setFixedHeight(sm.scale_int(self.BASE_FILTER_H))
        self.btn_search.clicked.connect(self._on_search)
        filter_layout.addWidget(self.btn_search)

        # 重置按钮
        self.btn_reset = QPushButton("重置")
        self.btn_reset.setObjectName("SecondaryButton")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setFixedHeight(sm.scale_int(self.BASE_FILTER_H))
        self.btn_reset.clicked.connect(self._on_reset)
        filter_layout.addWidget(self.btn_reset)

        layout.addWidget(filter_frame)

        # ========== 数据表格 ==========
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "☑", "ID", "时间", "文件名", "预测类别", "置信度", "Top-1", "备注",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                gridline-color: {BORDER_COLOR};
                alternate-background-color: {BG_COLOR};
            }}
            QTableWidget::item {{
                padding: 4px 8px;
                border-bottom: 1px solid {BORDER_COLOR};
            }}
            QTableWidget::item:selected {{
                background-color: rgba(46,134,171,0.12);
                color: {TEXT_PRIMARY};
            }}
            QHeaderView::section {{
                background-color: {BG_COLOR};
                color: {TEXT_PRIMARY};
                font-weight: bold;
                padding: 6px 8px;
                border: none;
                border-bottom: 2px solid {PRIMARY_COLOR};
            }}
        """)

        # 列宽设置
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.Fixed)      # 复选框
        header_view.setSectionResizeMode(1, QHeaderView.Fixed)      # ID
        header_view.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 时间
        header_view.setSectionResizeMode(3, QHeaderView.Stretch)    # 文件名
        header_view.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 类别
        header_view.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # 置信度
        header_view.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Top-1
        header_view.setSectionResizeMode(7, QHeaderView.Stretch)    # 备注
        self.table.setColumnWidth(0, 36)
        self.table.setColumnWidth(1, 50)

        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table, 1)

        # ========== 分页 + 操作栏 ==========
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(12)

        # 批量选择
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.setObjectName("SecondaryButton")
        self.btn_select_all.setCursor(Qt.PointingHandCursor)
        self.btn_select_all.clicked.connect(self._on_select_all)
        bottom_bar.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton("取消全选")
        self.btn_deselect_all.setObjectName("SecondaryButton")
        self.btn_deselect_all.setCursor(Qt.PointingHandCursor)
        self.btn_deselect_all.clicked.connect(self._on_deselect_all)
        bottom_bar.addWidget(self.btn_deselect_all)

        # 批量删除
        self.btn_batch_delete = QPushButton("🗑️ 批量删除")
        self.btn_batch_delete.setObjectName("DangerButton")
        self.btn_batch_delete.setCursor(Qt.PointingHandCursor)
        self.btn_batch_delete.setEnabled(False)
        self.btn_batch_delete.clicked.connect(self._on_batch_delete)
        self._update_btn_batch_delete_style()
        bottom_bar.addWidget(self.btn_batch_delete)

        bottom_bar.addStretch()

        # 导出
        self.btn_export_csv = QPushButton("📄 导出 CSV")
        self.btn_export_csv.setObjectName("SecondaryButton")
        self.btn_export_csv.setCursor(Qt.PointingHandCursor)
        self.btn_export_csv.clicked.connect(lambda: self._on_export("csv"))
        bottom_bar.addWidget(self.btn_export_csv)

        self.btn_export_json = QPushButton("📄 导出 JSON")
        self.btn_export_json.setObjectName("SecondaryButton")
        self.btn_export_json.setCursor(Qt.PointingHandCursor)
        self.btn_export_json.clicked.connect(lambda: self._on_export("json"))
        bottom_bar.addWidget(self.btn_export_json)

        # 清理
        self.btn_cleanup = QPushButton("🧹 清理旧记录")
        self.btn_cleanup.setObjectName("SecondaryButton")
        self.btn_cleanup.setCursor(Qt.PointingHandCursor)
        self.btn_cleanup.clicked.connect(self._on_cleanup)
        bottom_bar.addWidget(self.btn_cleanup)

        bottom_bar.addStretch()

        # 分页控件
        self.btn_prev = QPushButton("◀ 上一页")
        self.btn_prev.setObjectName("SecondaryButton")
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        self.btn_prev.clicked.connect(self._on_prev_page)
        bottom_bar.addWidget(self.btn_prev)

        self.page_label = QLabel("1 / 1")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; border: none;")
        self.page_label.setProperty("_base_font_size", 13)
        bottom_bar.addWidget(self.page_label)

        self.btn_next = QPushButton("下一页 ▶")
        self.btn_next.setObjectName("SecondaryButton")
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.clicked.connect(self._on_next_page)
        bottom_bar.addWidget(self.btn_next)

        layout.addLayout(bottom_bar)

        # 监听识别完成事件，自动保存
        EventBus.get().recognition_completed.connect(self._on_recognition_completed)

        # 首次加载数据
        self._refresh_classes()
        self._load_data()

    # ==================== 数据加载 ====================

    def _get_filters(self) -> Dict:
        """从筛选栏构建查询条件"""
        filters = {}
        ds = self.date_start.date()
        de = self.date_end.date()
        if ds.isValid():
            filters["date_start"] = ds.toString("yyyy-MM-dd")
        if de.isValid():
            filters["date_end"] = de.toString("yyyy-MM-dd")

        cls = self.combo_class.currentData()
        if cls:
            filters["predicted_class"] = cls

        conf_min = self.slider_conf.value() / 100.0
        if conf_min > 0:
            filters["confidence_min"] = conf_min

        keyword = self.search_input.text().strip()
        if keyword:
            filters["filename"] = keyword

        return filters

    def _load_data(self):
        """加载当前页数据"""
        filters = self._get_filters()
        records, total = self._db.query(
            filters=filters,
            limit=self.PAGE_SIZE,
            offset=(self._current_page - 1) * self.PAGE_SIZE,
        )
        self._total_records = total
        self._selected_ids.clear()
        self._update_btn_batch_delete_style()

        # 填充表格
        self.table.setRowCount(len(records))
        row_h = ScaleManager.get().scale_int(self.BASE_ROW_H)

        for row, rec in enumerate(records):
            self.table.setRowHeight(row, row_h)

            # 复选框
            chk = QCheckBox()
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.addWidget(chk)
            chk_layout.setAlignment(Qt.AlignCenter)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            chk.stateChanged.connect(lambda state, rid=rec.id: self._on_check_changed(rid, state))
            self.table.setCellWidget(row, 0, chk_widget)

            # ID
            id_item = QTableWidgetItem(str(rec.id))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, id_item)

            # 时间
            ts = rec.timestamp[:19].replace("T", " ") if rec.timestamp else ""
            self.table.setItem(row, 2, QTableWidgetItem(ts))

            # 文件名
            self.table.setItem(row, 3, QTableWidgetItem(rec.filename))

            # 预测类别
            class_item = QTableWidgetItem(f"🏷️ {rec.predicted_class}")
            class_item.setForeground(QColor(PRIMARY_COLOR))
            self.table.setItem(row, 4, class_item)

            # 置信度
            conf = rec.confidence
            conf_color = SUCCESS_COLOR if conf > 0.9 else (WARNING_COLOR if conf > 0.7 else DANGER_COLOR)
            conf_item = QTableWidgetItem(f"{conf:.1%}")
            conf_item.setForeground(QColor(conf_color))
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 5, conf_item)

            # Top-1（从 top_k_results 提取）
            top_k = rec.get_top_k()
            top1_text = top_k[0].get("class_name", top_k[0].get("class", "")) if top_k else ""
            self.table.setItem(row, 6, QTableWidgetItem(top1_text))

            # 备注
            self.table.setItem(row, 7, QTableWidgetItem(rec.notes))

        self._update_pagination()
        self._update_statistics()

    def _refresh_classes(self):
        """刷新类别下拉框"""
        current_data = self.combo_class.currentData()
        self.combo_class.clear()
        self.combo_class.addItem("全部类别", "")
        for cls in self._db.get_all_classes():
            self.combo_class.addItem(cls, cls)
        # 恢复之前选择
        idx = self.combo_class.findData(current_data)
        if idx >= 0:
            self.combo_class.setCurrentIndex(idx)

    def _update_pagination(self):
        """更新分页控件"""
        total_pages = max(1, (self._total_records + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self.page_label.setText(f"{self._current_page} / {total_pages}")
        self.btn_prev.setEnabled(self._current_page > 1)
        self.btn_next.setEnabled(self._current_page < total_pages)

    def _update_statistics(self):
        """更新统计卡片"""
        stats = self._db.get_statistics()
        self.stat_total.set_value(str(stats["total"]))
        self.stat_today.set_value(str(stats["today_count"]))
        self.stat_avg_conf.set_value(f"{stats['avg_confidence']:.1%}")
        self.stat_classes.set_value(str(len(stats["class_distribution"])))

    # ==================== 事件处理 ====================

    def _on_search(self):
        self._current_page = 1
        self._refresh_classes()
        self._load_data()

    def _on_reset(self):
        self.date_start.setDate(QDate.currentDate().addMonths(-1))
        self.date_end.setDate(QDate.currentDate())
        self.combo_class.setCurrentIndex(0)
        self.slider_conf.setValue(0)
        self.search_input.clear()
        self._current_page = 1
        self._load_data()

    def _on_prev_page(self):
        if self._current_page > 1:
            self._current_page -= 1
            self._load_data()

    def _on_next_page(self):
        total_pages = max(1, (self._total_records + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        if self._current_page < total_pages:
            self._current_page += 1
            self._load_data()

    def _on_check_changed(self, record_id: int, state: int):
        if state == Qt.Checked:
            self._selected_ids.add(record_id)
        else:
            self._selected_ids.discard(record_id)
        self.btn_batch_delete.setEnabled(len(self._selected_ids) > 0)

    def _on_select_all(self):
        self._selected_ids.clear()
        for row in range(self.table.rowCount()):
            chk_widget = self.table.cellWidget(row, 0)
            if chk_widget:
                chk = chk_widget.findChild(QCheckBox)
                if chk:
                    chk.setChecked(True)
        self.btn_batch_delete.setEnabled(len(self._selected_ids) > 0)

    def _on_deselect_all(self):
        self._selected_ids.clear()
        for row in range(self.table.rowCount()):
            chk_widget = self.table.cellWidget(row, 0)
            if chk_widget:
                chk = chk_widget.findChild(QCheckBox)
                if chk:
                    chk.setChecked(False)
        self.btn_batch_delete.setEnabled(False)

    def _on_batch_delete(self):
        if not self._selected_ids:
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除选中的 {len(self._selected_ids)} 条记录吗？\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._db.delete_batch(list(self._selected_ids))
            self._selected_ids.clear()
            self._load_data()
            self._refresh_classes()

    def _on_row_double_clicked(self, row: int, _col: int):
        """双击行打开详情弹窗"""
        id_item = self.table.item(row, 1)
        if id_item is None:
            return
        record_id = int(id_item.text())
        record = self._db.get_by_id(record_id)
        if record:
            dialog = RecordDetailDialog(record, self)
            dialog.exec_()

    def _on_export(self, fmt: str):
        """导出"""
        filters = self._get_filters()
        try:
            if fmt == "csv":
                path = self._db.export_csv(filters=filters)
            else:
                path = self._db.export_json(filters=filters)
            QMessageBox.information(self, "导出成功", f"文件已保存至:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _on_cleanup(self):
        """清理旧记录"""
        reply = QMessageBox.question(
            self, "清理旧记录",
            "将删除超过 90 天的识别记录及其缩略图。\n确定继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            deleted = self._db.cleanup(90)
            QMessageBox.information(self, "清理完成", f"已删除 {deleted} 条过期记录。")
            self._load_data()
            self._refresh_classes()

    def _on_recognition_completed(self, result: dict):
        """识别完成时自动保存到数据库"""
        try:
            record = HistoryRecord(
                filename=result.get("filename", ""),
                image_path=result.get("image_path", ""),
                thumbnail_path=result.get("thumbnail_path", ""),
                predicted_class=result.get("class_name", result.get("predicted_class", "Unknown")),
                confidence=result.get("confidence", 0.0),
            )
            # 设置 top_k
            top_k = result.get("top_k", [])
            if top_k:
                record.set_top_k(top_k)

            self._db.save(record)
            self._refresh_classes()
            self._load_data()
        except Exception:
            pass  # 静默失败，不阻塞识别流程

    # ==================== 批量删除按钮样式 ====================

    def _update_btn_batch_delete_style(self):
        sm = ScaleManager.get()
        r = sm.scale_int(6)
        p = f"{sm.scale_int(8)}px {sm.scale_int(20)}px"
        fs = sm.scale_int(13)
        self.btn_batch_delete.setStyleSheet(f"""
            QPushButton#DangerButton {{
                background-color: transparent;
                color: {DANGER_COLOR};
                border: 1px solid {DANGER_COLOR};
                border-radius: {r}px;
                padding: {p};
                font-size: {fs}px;
            }}
            QPushButton#DangerButton:hover {{
                background-color: rgba(220,53,69,0.08);
            }}
            QPushButton#DangerButton:disabled {{
                color: {BORDER_COLOR};
                border-color: {BORDER_COLOR};
            }}
        """)

    # ==================== 缩放适配 ====================

    def apply_scale(self, scale: float):
        sm = ScaleManager.get()

        # 页面边距
        layout = self.layout()
        if layout:
            m = sm.scale_int(self.BASE_MARGIN)
            v = sm.scale_int(self.BASE_MARGIN_V)
            layout.setContentsMargins(m, v, m, v)

        # 筛选栏控件高度
        filter_h = sm.scale_int(self.BASE_FILTER_H)
        self.date_start.setFixedHeight(filter_h)
        self.date_end.setFixedHeight(filter_h)
        self.combo_class.setFixedHeight(filter_h)
        self.combo_class.setMinimumWidth(sm.scale_int(140))
        self.search_input.setFixedHeight(filter_h)
        self.search_input.setMinimumWidth(sm.scale_int(160))
        self.btn_search.setFixedHeight(filter_h)
        self.btn_reset.setFixedHeight(filter_h)
        self.slider_conf.setFixedWidth(sm.scale_int(120))
        self.label_conf_val.setFixedWidth(sm.scale_int(36))

        # 表格行高
        row_h = sm.scale_int(self.BASE_ROW_H)
        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, row_h)

        # 统计卡片
        for card in [self.stat_total, self.stat_today, self.stat_avg_conf, self.stat_classes]:
            card.apply_scale(scale)

        # 批量删除按钮样式
        self._update_btn_batch_delete_style()
