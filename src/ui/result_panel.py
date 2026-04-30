"""
识别结果动态展示面板

组件列表:
  - RingProgressWidget: 环形置信度进度条
  - TopKBarChart: Top-K 横向条形图
  - LatencyDisplay: 推理耗时面板
  - SingleResultPanel: 单图结果展示（左原图+右详情）
  - BatchResultTable: 批量结果表格（排序/筛选/详情按钮）
  - BatchResultCards: 批量结果卡片网格
  - ImagePreviewDialog: 图片放大预览模态窗口
  - ResultExporter: CSV/JSON 导出工具
"""
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.ui.MarqueeLabel import MarqueeLabel
from src.ui.progressbar import RoundedProgressBar
from src.ui.centered_label import CenteredPixmapLabel

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QProgressBar, QScrollArea,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QDialog, QSizePolicy, QFileDialog,
    QMessageBox, QTabWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRectF, QTimer
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QFont, QConicalGradient, QPalette,
    QPixmap, QImage, QBrush,QPainterPath,
)

from src.ui.styles import (
    PRIMARY_COLOR, PRIMARY_LIGHT, PRIMARY_DARK, SUCCESS_COLOR,
    WARNING_COLOR, DANGER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER_COLOR, BG_COLOR, CARD_BG,
)
from src.ui.scale_manager import ScaleManager

# 基准尺寸常量
BASE_RING_SIZE = 100
BASE_RING_PEN = 10
BASE_RING_FONT = 18
BASE_RANK_FONT = 12
BASE_NAME_FONT = 13
BASE_VAL_FONT = 12
BASE_BAR_H = 12
BASE_RANK_W = 24
BASE_NAME_W = 110
BASE_VAL_W = 56
BASE_LATENCY_TITLE_FONT = 13
BASE_LATENCY_LABEL_FONT = 12
BASE_PRED_HINT_FONT = 14
BASE_PRED_NAME_FONT = 22
BASE_TOPK_TITLE_FONT = 13
BASE_EMPTY_FONT = 16
BASE_CARD_W = 170
BASE_CARD_H = 200
BASE_THUMB_W = 150
BASE_THUMB_H = 120
BASE_CARD_NAME_FONT = 13
BASE_CARD_BADGE_FONT = 12


# ============================================================
# 数据模型
# ============================================================
@dataclass
class RecognitionResult:
    """识别结果数据模型"""
    class_name: str = ""
    confidence: float = 0.0
    top_k: List[Dict[str, object]] = field(default_factory=list)
    image: Optional[np.ndarray] = None          # 原图 BGR
    overlay_image: Optional[np.ndarray] = None   # 可视化叠加图 BGR
    heatmap: Optional[np.ndarray] = None         # 热力图 [0,1]
    latency: Optional[Dict[str, float]] = None   # {"preprocess": ms, "inference": ms, "postprocess": ms}
    filename: str = ""


# ============================================================
# 环形进度条
# ============================================================
class RingProgressWidget(QWidget):
    """环形置信度进度条，颜色按阈值自动变化"""

    def __init__(self, value: float = 0.0, size: int = BASE_RING_SIZE, parent=None):
        super().__init__(parent)
        self._value = value  # 0.0 ~ 1.0
        self._base_size = size
        sm = ScaleManager.get()
        actual = sm.scale_int(size)
        self.setFixedSize(actual, actual)

    def set_value(self, value: float):
        self._value = max(0.0, min(1.0, value))
        self.update()

    def _get_color(self) -> QColor:
        if self._value > 0.9:
            return QColor(SUCCESS_COLOR)
        elif self._value > 0.7:
            return QColor(WARNING_COLOR)
        else:
            return QColor(DANGER_COLOR)

    def paintEvent(self, event):
        sm = ScaleManager.get()
        pen_width = sm.scale_int(BASE_RING_PEN)
        font_px = sm.scale_int(BASE_RING_FONT)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        margin = pen_width // 2 + 2
        w = self.width()
        h = self.height()
        rect = QRectF(margin, margin, w - 2 * margin, h - 2 * margin)

        # 背景环
        bg_pen = QPen(QColor(BORDER_COLOR), pen_width, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(bg_pen)
        painter.drawArc(rect, 0, 360 * 16)

        # 前景环
        color = self._get_color()
        fg_pen = QPen(color, pen_width, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(fg_pen)
        span = int(self._value * 360 * 16)
        start_angle = 90 * 16  # 从顶部开始
        painter.drawArc(rect, start_angle, -span)

        # 中心文字 - 保留一位小数
        painter.setPen(QColor(TEXT_PRIMARY))
        font = QFont("Microsoft YaHei", font_px, QFont.Bold)
        font.setPixelSize(font_px)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, f"{self._value:.1%}")

        painter.end()

    def apply_scale(self, scale: float):
        sm = ScaleManager.get()
        actual = sm.scale_int(self._base_size)
        self.setFixedSize(actual, actual)
        self.update()


# ============================================================
# Top-K 条形图
# ============================================================
class TopKBarChart(QWidget):
    """Top-K 横向条形图"""

    def __init__(self, max_bars: int = 5, parent=None):
        super().__init__(parent)
        self._max_bars = max_bars
        self._items: List[Dict] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(ScaleManager.get().scale_int(6))

    def set_data(self, top_k: List[Dict]):
        sm = ScaleManager.get()

        # 清除旧条目
        while self._layout.count():
            child = self._layout.takeAt(0)
            if child is None:
                continue
            if child.layout():
                while child.layout().count():
                    sub = child.layout().takeAt(0)
                    if sub and sub.widget():
                        sub.widget().deleteLater()
                child.layout().deleteLater()
            if child.widget():
                child.widget().deleteLater()
            child.deleteLater()

        self._items = top_k[:self._max_bars]

        for i, item in enumerate(self._items):
            name = item.get("class_name", "")
            conf = item.get("confidence", 0.0)
            bar_color = PRIMARY_COLOR if i == 0 else "#adb5bd"

            row = QHBoxLayout()
            row.setSpacing(sm.scale_int(8))

            # 排名
            rank_label = QLabel(f"#{i + 1}")
            rank_label.setFixedWidth(sm.scale_int(BASE_RANK_W))
            rank_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {sm.scale_int(BASE_RANK_FONT)}px; border: none;")
            rank_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(rank_label)

            # 类别名 —— 修复：添加 row.addWidget
            name_label = MarqueeLabel(str(name))
            name_label.setFixedWidth(sm.scale_int(BASE_NAME_W))
            name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {sm.scale_int(BASE_NAME_FONT)}px; border: none;")
            name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            name_label.setSpeed(1)
            name_label.setInterval(30)
            row.addWidget(name_label)  # ✅ 修复：添加这行

            # 进度条
            bar = RoundedProgressBar(
                bar_color=bar_color,
                bg_color=BORDER_COLOR,
                radius=sm.scale_int(6)
            )
            bar.setRange(0, 100)
            bar.setValue(int(conf * 100))
            bar.setFixedHeight(sm.scale_int(BASE_BAR_H))
            row.addWidget(bar, 1)

            # 百分比
            val_label = QLabel(f"{conf:.1%}")
            val_label.setFixedWidth(sm.scale_int(BASE_VAL_W))
            val_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {sm.scale_int(BASE_VAL_FONT)}px; border: none;")
            val_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            row.addWidget(val_label)

            self._layout.addLayout(row)

    def apply_scale(self, scale: float):
        sm = ScaleManager.get()
        self._layout.setSpacing(sm.scale_int(6))


# ============================================================
# 推理耗时面板
# ============================================================
class LatencyDisplay(QFrame):
    """推理各阶段耗时显示"""

    def __init__(self, parent=None):
        super().__init__(parent)
        sm = ScaleManager.get()
        self.setObjectName("Card")

        title_px = sm.scale_int(BASE_LATENCY_TITLE_FONT)
        label_px = sm.scale_int(BASE_LATENCY_LABEL_FONT)

        self.setStyleSheet(f"""
            #Card {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: {sm.scale_int(8)}px {sm.scale_int(12)}px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(sm.scale_int(6))
        layout.setContentsMargins(sm.scale_int(10), sm.scale_int(8), sm.scale_int(10), sm.scale_int(8))

        title = QLabel("⏱️ 推理耗时")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {title_px}px; font-weight: bold; border: none;")
        layout.addWidget(title)

        self._rows = {}
        self._stage_labels = []
        self._ms_labels = []
        for stage, label in [("preprocess", "预处理"), ("inference", "模型推理"), ("postprocess", "后处理")]:
            row = QHBoxLayout()
            stage_label = QLabel(label)
            stage_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {label_px}px; border: none;")
            self._stage_labels.append(stage_label)
            row.addWidget(stage_label)
            row.addStretch()
            ms_label = QLabel("-")
            ms_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {label_px}px; border: none;")
            ms_label.setAlignment(Qt.AlignRight)
            self._ms_labels.append(ms_label)
            row.addWidget(ms_label)
            layout.addLayout(row)
            self._rows[stage] = ms_label

        # 总计行
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {BORDER_COLOR};")
        layout.addWidget(sep)

        total_row = QHBoxLayout()
        self._total_label_text = QLabel("总计")
        self._total_label_text.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {label_px}px; font-weight: bold; border: none;")
        total_row.addWidget(self._total_label_text)
        total_row.addStretch()
        self._total_label = QLabel("-")
        self._total_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: {title_px}px; font-weight: bold; border: none;")
        self._total_label.setAlignment(Qt.AlignRight)
        total_row.addWidget(self._total_label)
        layout.addLayout(total_row)

    def set_latency(self, latency: Optional[Dict[str, float]]):
        if latency is None:
            for label in self._rows.values():
                label.setText("-")
            self._total_label.setText("-")
            return

        total = 0.0
        for stage, ms_label in self._rows.items():
            val = latency.get(stage, 0.0)
            ms_label.setText(f"{val:.1f} ms")
            total += val
        self._total_label.setText(f"{total:.1f} ms")

    def apply_scale(self, scale: float):
        sm = ScaleManager.get()
        title_px = sm.scale_int(BASE_LATENCY_TITLE_FONT)
        label_px = sm.scale_int(BASE_LATENCY_LABEL_FONT)
        for lbl in self._stage_labels:
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {label_px}px; border: none;")
        for lbl in self._ms_labels:
            lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {label_px}px; border: none;")
        self._total_label_text.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {label_px}px; font-weight: bold; border: none;")
        self._total_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: {title_px}px; font-weight: bold; border: none;")


# ============================================================
# 图片放大预览模态窗口
# ============================================================
class ImagePreviewDialog(QDialog):
    """图片放大预览模态窗口"""

    def __init__(self, pixmap: QPixmap, title: str = "图片预览", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(500, 400)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # 图片标签
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        # 缩放到窗口大小
        screen_size = parent.screen().size() if parent and parent.screen() else QSize(1200, 900)
        max_w = min(pixmap.width(), screen_size.width() - 80)
        max_h = min(pixmap.height(), screen_size.height() - 120)
        scaled = pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label.setPixmap(scaled)
        layout.addWidget(img_label, 1)

        # 关闭按钮
        btn_close = QPushButton("关闭")
        btn_close.setObjectName("SecondaryButton")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setFixedWidth(120)
        btn_close.clicked.connect(self.close)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        btn_row.addStretch()
        layout.addLayout(btn_row)


# ============================================================
# 单图结果面板
# ============================================================
class SingleResultPanel(QWidget):
    """
    单图识别结果面板

    左栏: 原图 + 可视化叠加图（可切换热力图）
    右栏: 主预测 + 环形进度条 + Top-K 条形图 + 推理耗时
    """

    detail_requested = pyqtSignal()  # 查看详情

    def __init__(self, show_heatmap: bool = True, parent=None):
        super().__init__(parent)
        self._result: Optional[RecognitionResult] = None
        self._show_heatmap = show_heatmap
        self._showing_overlay = False
        sm = ScaleManager.get()

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(sm.scale_int(16))
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ===== 左栏: 图片区域 =====
        left_card = QFrame()
        left_card.setObjectName("Card")
        self._left_card = left_card
        left_card.setMinimumHeight(sm.scale_int(400))
        left_layout = QVBoxLayout(left_card)
        left_layout.setSpacing(sm.scale_int(8))

        # 图片标题行
        img_title_row = QHBoxLayout()
        img_title = QLabel("📷 图片")
        img_title.setObjectName("SectionTitle")
        img_title.setStyleSheet("border: none;")
        img_title_row.addWidget(img_title)
        img_title_row.addStretch()

        # 热力图切换按钮
        self.btn_toggle_overlay = QPushButton("🔥 热力图")
        self.btn_toggle_overlay.setCheckable(True)
        self.btn_toggle_overlay.setObjectName("SecondaryButton")
        self.btn_toggle_overlay.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_overlay.hide()
        self.btn_toggle_overlay.clicked.connect(self._toggle_overlay)
        img_title_row.addWidget(self.btn_toggle_overlay)

        # 放大按钮
        self.btn_zoom = QPushButton("🔍 放大")
        self.btn_zoom.setObjectName("SecondaryButton")
        self.btn_zoom.setCursor(Qt.PointingHandCursor)
        self.btn_zoom.hide()
        self.btn_zoom.clicked.connect(self._zoom_image)
        img_title_row.addWidget(self.btn_zoom)

        left_layout.addLayout(img_title_row)

        # 图片显示
        self.image_label = CenteredPixmapLabel()
        self.image_label.setMinimumSize(sm.scale_int(280), sm.scale_int(280))
        self.image_label.setMaximumSize(sm.scale_int(500), sm.scale_int(500))
        self.image_label.setStyleSheet(f"border-radius: 6px; background: {BG_COLOR};")
        left_layout.addWidget(self.image_label, 1)

        # 保存按钮
        self.btn_save = QPushButton("💾 保存可视化图片")
        self.btn_save.setObjectName("SecondaryButton")
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.hide()
        self.btn_save.clicked.connect(self._save_image)
        left_layout.addWidget(self.btn_save)

        main_layout.addWidget(left_card, 1)

        # ===== 右栏: 结果详情 =====
        right_card = QFrame()
        right_card.setObjectName("Card")
        self._right_card = right_card
        right_card.setMinimumHeight(sm.scale_int(400))
        right_layout = QVBoxLayout(right_card)
        right_layout.setSpacing(sm.scale_int(10))

        result_title = QLabel("📋 结果详情")
        result_title.setObjectName("SectionTitle")
        result_title.setStyleSheet("border: none;")
        right_layout.addWidget(result_title)

        # 主预测: 类别名 + 环形进度条
        pred_row = QHBoxLayout()
        pred_text_layout = QVBoxLayout()
        pred_text_layout.setSpacing(sm.scale_int(2))

        hint_px = sm.scale_int(BASE_PRED_HINT_FONT)
        name_px = sm.scale_int(BASE_PRED_NAME_FONT)
        self.class_label = QLabel("等待识别...")
        self.class_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {hint_px}px; border: none;")
        pred_text_layout.addWidget(self.class_label)

        self.class_name_label = QLabel("")
        self.class_name_label.setFont(QFont("Microsoft YaHei", name_px, QFont.Bold))
        self.class_name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {name_px}px; border: none;")
        pred_text_layout.addWidget(self.class_name_label)
        pred_row.addLayout(pred_text_layout, 1)

        self.ring_progress = RingProgressWidget(size=BASE_RING_SIZE)
        pred_row.addWidget(self.ring_progress)
        right_layout.addLayout(pred_row)

        # 分割线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {BORDER_COLOR};")
        right_layout.addWidget(sep)

        # Top-K 条形图
        topk_px = sm.scale_int(BASE_TOPK_TITLE_FONT)
        top_k_title = QLabel("Top-5 排行")
        top_k_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {topk_px}px; font-weight: bold; border: none;")
        right_layout.addWidget(top_k_title)
        self._topk_title = top_k_title

        self.top_k_chart = TopKBarChart(max_bars=5)
        right_layout.addWidget(self.top_k_chart)

        # 推理耗时
        right_layout.addStretch()
        self.latency_display = LatencyDisplay()
        right_layout.addWidget(self.latency_display)

        main_layout.addWidget(right_card, 1)

        # 空状态
        empty_px = sm.scale_int(BASE_EMPTY_FONT)
        self._empty_hint = QLabel("🔎 上传图片并点击识别查看结果")
        self._empty_hint.setAlignment(Qt.AlignCenter)
        self._empty_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {empty_px}px; border: none;")
        main_layout.addWidget(self._empty_hint)

        # 初始隐藏内容
        left_card.hide()
        right_card.hide()

    def set_result(self, result: RecognitionResult):
        """设置识别结果并更新 UI"""
        self._result = result
        self._showing_overlay = False
        sm = ScaleManager.get()

        # 显示内容，隐藏空状态
        self._left_card.show()
        self._right_card.show()
        self._empty_hint.hide()

        # 显示原图
        self._display_image(result.image)

        # 热力图按钮
        has_overlay = result.overlay_image is not None or result.heatmap is not None
        self.btn_toggle_overlay.setVisible(has_overlay and self._show_heatmap)
        self.btn_toggle_overlay.setChecked(False)
        self.btn_zoom.show()
        self.btn_save.show()

        # 右栏数据
        conf_color = SUCCESS_COLOR if result.confidence > 0.9 else (WARNING_COLOR if result.confidence > 0.7 else DANGER_COLOR)
        hint_px = sm.scale_int(BASE_PRED_HINT_FONT)
        # 置信度保留一位小数
        self.class_label.setText(f"预测类别（置信度 {result.confidence:.1%}）")
        self.class_label.setStyleSheet(f"color: {conf_color}; font-size: {hint_px}px; border: none;")
        self.class_name_label.setText(result.class_name)
        self.ring_progress.set_value(result.confidence)

        # Top-K
        self.top_k_chart.set_data(result.top_k)

        # 延迟
        self.latency_display.set_latency(result.latency)

    def show_empty(self):
        """显示空状态"""
        self._left_card.hide()
        self._right_card.hide()
        self._empty_hint.show()
        self._result = None

    def show_loading(self):
        """显示加载态"""
        self._left_card.hide()
        self._right_card.hide()
        self._empty_hint.setText("⏳ 识别中，请稍候...")
        self._empty_hint.show()

    # ----- 内部方法 -----
    def _ndarray_to_pixmap(self, img: np.ndarray) -> QPixmap:
        """BGR ndarray → QPixmap"""
        if img is None:
            return QPixmap()
        h, w = img.shape[:2]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        # BGR → RGB
        rgb = img[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _display_image(self, img: Optional[np.ndarray]):
        if img is None:
            self.image_label.setText("无法显示图片")
            return
        pixmap = self._ndarray_to_pixmap(img)
        # CenteredPixmapLabel 内部处理缩放居中
        self.image_label.setPixmap(pixmap)

    def _toggle_overlay(self, checked: bool):
        if self._result is None:
            return
        if checked:
            # 优先 overlay_image，其次 heatmap 叠加
            if self._result.overlay_image is not None:
                self._display_image(self._result.overlay_image)
            elif self._result.heatmap is not None and self._result.image is not None:
                try:
                    from src.visualizer import Visualizer
                    viz = Visualizer()
                    overlay = viz.draw_heatmap(self._result.image, self._result.heatmap, alpha=0.5)
                    self._display_image(overlay)
                except Exception:
                    self._display_image(self._result.image)
        else:
            self._display_image(self._result.image)

    def _zoom_image(self):
        if self._result is None:
            return
        img = self._result.overlay_image if (self._showing_overlay and self._result.overlay_image is not None) else self._result.image
        if img is None:
            return
        pixmap = self._ndarray_to_pixmap(img)
        dialog = ImagePreviewDialog(pixmap, "图片预览", self.window())
        dialog.exec_()

    def _save_image(self):
        if self._result is None:
            return
        img = self._result.overlay_image or self._result.image
        if img is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存可视化图片", "result.png",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)",
        )
        if path:
            try:
                from src.visualizer import Visualizer
                Visualizer.save(img, path)
                QMessageBox.information(self, "保存成功", f"图片已保存至:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", str(e))

    def apply_scale(self, scale: float):
        """缩放面板尺寸与字体"""
        sm = ScaleManager.get()
        self._left_card.setMinimumHeight(sm.scale_int(400))
        self._right_card.setMinimumHeight(sm.scale_int(400))
        self.image_label.setMinimumSize(sm.scale_int(280), sm.scale_int(280))
        self.image_label.setMaximumSize(sm.scale_int(500), sm.scale_int(500))
        self.ring_progress.apply_scale(scale)
        self.top_k_chart.apply_scale(scale)
        self.latency_display.apply_scale(scale)

        # 更新右栏字体
        hint_px = sm.scale_int(BASE_PRED_HINT_FONT)
        name_px = sm.scale_int(BASE_PRED_NAME_FONT)
        topk_px = sm.scale_int(BASE_TOPK_TITLE_FONT)
        empty_px = sm.scale_int(BASE_EMPTY_FONT)

        # 重新设置 class_label 样式（保留颜色）
        if self._result:
            conf = self._result.confidence
            conf_color = SUCCESS_COLOR if conf > 0.9 else (WARNING_COLOR if conf > 0.7 else DANGER_COLOR)
        else:
            conf_color = TEXT_SECONDARY
        self.class_label.setStyleSheet(f"color: {conf_color}; font-size: {hint_px}px; border: none;")
        self.class_name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {name_px}px; border: none;")
        font = self.class_name_label.font()
        font.setPixelSize(name_px)
        self.class_name_label.setFont(font)
        self._topk_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {topk_px}px; font-weight: bold; border: none;")
        self._empty_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {empty_px}px; border: none;")


# ============================================================
# 批量结果表格
# ============================================================
class BatchResultTable(QWidget):
    """
    批量识别结果表格

    功能: 排序、按类别筛选、查看详情、导出
    """

    export_requested = pyqtSignal(str)  # "csv" or "json"
    detail_requested = pyqtSignal(int)  # row index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: List[RecognitionResult] = []
        self._filtered_indices: List[int] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 工具栏
        toolbar = QHBoxLayout()

        # 类别筛选
        toolbar.addWidget(QLabel("筛选类别:"))
        self.filter_combo = QComboBox()
        self.filter_combo.setMinimumWidth(160)
        self.filter_combo.addItem("全部")
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self.filter_combo)

        toolbar.addStretch()

        # 排序按钮
        self.btn_sort_conf = QPushButton("按置信度排序")
        self.btn_sort_conf.setObjectName("SecondaryButton")
        self.btn_sort_conf.setCursor(Qt.PointingHandCursor)
        self.btn_sort_conf.clicked.connect(self._sort_by_confidence)
        toolbar.addWidget(self.btn_sort_conf)

        self.btn_sort_name = QPushButton("按文件名排序")
        self.btn_sort_name.setObjectName("SecondaryButton")
        self.btn_sort_name.setCursor(Qt.PointingHandCursor)
        self.btn_sort_name.clicked.connect(self._sort_by_name)
        toolbar.addWidget(self.btn_sort_name)

        # 导出按钮
        self.btn_export_csv = QPushButton("📄 导出 CSV")
        self.btn_export_csv.setObjectName("SecondaryButton")
        self.btn_export_csv.setCursor(Qt.PointingHandCursor)
        self.btn_export_csv.clicked.connect(lambda: self._export("csv"))
        toolbar.addWidget(self.btn_export_csv)

        self.btn_export_json = QPushButton("📄 导出 JSON")
        self.btn_export_json.setObjectName("SecondaryButton")
        self.btn_export_json.setCursor(Qt.PointingHandCursor)
        self.btn_export_json.clicked.connect(lambda: self._export("json"))
        toolbar.addWidget(self.btn_export_json)

        layout.addLayout(toolbar)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["文件名", "预测类别", "置信度", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.table.setColumnWidth(3, 90)
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
        layout.addWidget(self.table, 1)

    def set_results(self, results: List[RecognitionResult]):
        self._results = results
        self._update_filter_combo()
        self._refresh_table()

    def add_result(self, result: RecognitionResult):
        self._results.append(result)
        self._update_filter_combo()
        self._refresh_table()

    def clear_results(self):
        self._results.clear()
        self._filtered_indices.clear()
        self.table.setRowCount(0)
        self.filter_combo.clear()
        self.filter_combo.addItem("全部")

    def _update_filter_combo(self):
        current = self.filter_combo.currentText()
        classes = sorted(set(r.class_name for r in self._results))
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("全部")
        self.filter_combo.addItems(classes)
        idx = self.filter_combo.findText(current)
        if idx >= 0:
            self.filter_combo.setCurrentIndex(idx)
        self.filter_combo.blockSignals(False)

    def _on_filter_changed(self, text: str):
        self._refresh_table()

    def _refresh_table(self):
        filter_class = self.filter_combo.currentText()
        self._filtered_indices = []

        for i, r in enumerate(self._results):
            if filter_class == "全部" or r.class_name == filter_class:
                self._filtered_indices.append(i)

        self.table.setRowCount(len(self._filtered_indices))
        for row_idx, data_idx in enumerate(self._filtered_indices):
            r = self._results[data_idx]

            # 文件名
            self.table.setItem(row_idx, 0, QTableWidgetItem(r.filename))

            # 类别
            self.table.setItem(row_idx, 1, QTableWidgetItem(r.class_name))

            # 置信度 - 保留一位小数
            conf = r.confidence
            conf_item = QTableWidgetItem(f"{conf:.1%}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            # 颜色
            if conf > 0.9:
                conf_item.setForeground(QColor(SUCCESS_COLOR))
            elif conf > 0.7:
                conf_item.setForeground(QColor(WARNING_COLOR))
            else:
                conf_item.setForeground(QColor(DANGER_COLOR))
            self.table.setItem(row_idx, 2, conf_item)

            # 操作按钮
            btn_detail = QPushButton("详情")
            btn_detail.setCursor(Qt.PointingHandCursor)
            btn_detail.setStyleSheet(f"""
                QPushButton {{
                    background-color: {PRIMARY_COLOR};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 12px;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: {PRIMARY_DARK};
                }}
            """)
            btn_detail.clicked.connect(lambda checked, idx=data_idx: self.detail_requested.emit(idx))
            self.table.setCellWidget(row_idx, 3, btn_detail)

    def _sort_by_confidence(self):
        self._results.sort(key=lambda r: r.confidence, reverse=True)
        self._refresh_table()

    def _sort_by_name(self):
        self._results.sort(key=lambda r: r.filename.lower())
        self._refresh_table()

    def _export(self, fmt: str):
        if not self._results:
            QMessageBox.information(self, "导出", "暂无结果可导出")
            return

        if fmt == "csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "导出 CSV", "batch_results.csv",
                "CSV (*.csv);;All Files (*)",
            )
            if not path:
                return
            try:
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["文件名", "预测类别", "置信度"])
                    for r in self._results:
                        writer.writerow([r.filename, r.class_name, f"{r.confidence:.4f}"])
                QMessageBox.information(self, "导出成功", f"已导出至:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", str(e))

        elif fmt == "json":
            path, _ = QFileDialog.getSaveFileName(
                self, "导出 JSON", "batch_results.json",
                "JSON (*.json);;All Files (*)",
            )
            if not path:
                return
            try:
                data = []
                for r in self._results:
                    item = {
                        "filename": r.filename,
                        "class_name": r.class_name,
                        "confidence": round(r.confidence, 4),
                    }
                    if r.top_k:
                        item["top_k"] = [
                            {"class_name": t.get("class_name", ""), "confidence": round(t.get("confidence", 0), 4)}
                            for t in r.top_k
                        ]
                    if r.latency:
                        item["latency_ms"] = {k: round(v, 2) for k, v in r.latency.items()}
                    data.append(item)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "导出成功", f"已导出至:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", str(e))

    def get_results(self) -> List[RecognitionResult]:
        return self._results.copy()


# ============================================================
# 批量结果卡片网格
# ============================================================
class ResultCardWidget(QFrame):
    """单张结果卡片"""

    zoom_requested = pyqtSignal(int)  # data index

    def __init__(self, result: RecognitionResult, index: int, parent=None):
        super().__init__(parent)
        sm = ScaleManager.get()
        self._index = index
        self.setObjectName("ResultCard")
        self.setFixedSize(sm.scale_int(BASE_CARD_W), sm.scale_int(BASE_CARD_H))
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            #ResultCard {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
            #ResultCard:hover {{
                border-color: {PRIMARY_LIGHT};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(sm.scale_int(8), sm.scale_int(8), sm.scale_int(8), sm.scale_int(6))
        layout.setSpacing(sm.scale_int(4))

        # 缩略图
        self.thumb = QLabel()
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setFixedSize(sm.scale_int(BASE_THUMB_W), sm.scale_int(BASE_THUMB_H))
        self.thumb.setStyleSheet(f"border-radius: 4px; background: {BG_COLOR};")
        if result.image is not None:
            self._set_thumbnail(result.image)
        layout.addWidget(self.thumb)

        # 类别
        name = result.class_name
        if len(name) > 14:
            name = name[:12] + "..."
        name_px = sm.scale_int(BASE_CARD_NAME_FONT)
        class_label = QLabel(name)
        class_label.setAlignment(Qt.AlignCenter)
        class_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {name_px}px; font-weight: bold; border: none;")
        layout.addWidget(class_label)

        # 置信度徽章 - 保留一位小数
        conf = result.confidence
        if conf > 0.9:
            badge_color = SUCCESS_COLOR
        elif conf > 0.7:
            badge_color = WARNING_COLOR
        else:
            badge_color = DANGER_COLOR

        badge_px = sm.scale_int(BASE_CARD_BADGE_FONT)
        badge = QLabel(f"{conf:.1%}")
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(f"""
            background-color: {badge_color};
            color: white;
            border-radius: 10px;
            padding: 2px 10px;
            font-size: {badge_px}px;
            font-weight: bold;
        """)
        layout.addWidget(badge)

    def _set_thumbnail(self, img: np.ndarray):
        h, w = img.shape[:2]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        rgb = img[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        sm = ScaleManager.get()
        scaled = pixmap.scaled(
            sm.scale_int(BASE_THUMB_W), sm.scale_int(BASE_THUMB_H),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.thumb.setPixmap(scaled)


class BatchResultCards(QWidget):
    """批量结果卡片网格视图"""

    zoom_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: List[RecognitionResult] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background: {CARD_BG};
            }}
        """)

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(12)
        self.grid_layout.setContentsMargins(12, 12, 12, 12)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_container)

        layout.addWidget(self.scroll, 1)

    def set_results(self, results: List[RecognitionResult]):
        self._results = results
        self._refresh()

    def add_result(self, result: RecognitionResult):
        self._results.append(result)
        self._refresh()

    def clear_results(self):
        self._results.clear()
        self._refresh()

    def _refresh(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = 4
        for i, r in enumerate(self._results):
            card = ResultCardWidget(r, i)
            card.zoom_requested.connect(self.zoom_requested.emit)
            row, col = divmod(i, cols)
            self.grid_layout.addWidget(card, row, col)

    def get_results(self) -> List[RecognitionResult]:
        return self._results.copy()


# ============================================================
# 批量结果面板（表格 + 卡片 双视图）
# ============================================================
class BatchResultPanel(QWidget):
    """批量识别结果面板，含表格/卡片双视图切换"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: List[RecognitionResult] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 视图切换工具栏
        toolbar = QHBoxLayout()

        view_label = QLabel("视图:")
        view_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: bold; border: none;")
        toolbar.addWidget(view_label)

        self.btn_table_view = QPushButton("📊 表格")
        self.btn_table_view.setCheckable(True)
        self.btn_table_view.setChecked(True)
        self.btn_table_view.setObjectName("SecondaryButton")
        self.btn_table_view.setCursor(Qt.PointingHandCursor)
        self.btn_table_view.clicked.connect(lambda: self._switch_view("table"))
        toolbar.addWidget(self.btn_table_view)

        self.btn_card_view = QPushButton("🖼️ 卡片")
        self.btn_card_view.setCheckable(True)
        self.btn_card_view.setObjectName("SecondaryButton")
        self.btn_card_view.setCursor(Qt.PointingHandCursor)
        self.btn_card_view.clicked.connect(lambda: self._switch_view("card"))
        toolbar.addWidget(self.btn_card_view)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # 标签页容器
        self.view_stack = QTabWidget()
        self.view_stack.tabBar().hide()

        # 表格视图
        self.table_view = BatchResultTable()
        self.table_view.detail_requested.connect(self._show_detail)
        self.view_stack.addTab(self.table_view, "表格")

        # 卡片视图
        self.card_view = BatchResultCards()
        self.card_view.zoom_requested.connect(self._show_detail)
        self.view_stack.addTab(self.card_view, "卡片")

        layout.addWidget(self.view_stack, 1)

        # 详情面板（用于显示单张详情的弹窗）
        self._detail_dialog: Optional[QDialog] = None

    def _switch_view(self, view: str):
        if view == "table":
            self.view_stack.setCurrentIndex(0)
            self.btn_table_view.setChecked(True)
            self.btn_card_view.setChecked(False)
        else:
            self.view_stack.setCurrentIndex(1)
            self.btn_card_view.setChecked(True)
            self.btn_table_view.setChecked(False)

    def set_results(self, results: List[RecognitionResult]):
        self._results = results
        self.table_view.set_results(results)
        self.card_view.set_results(results)

    def add_result(self, result: RecognitionResult):
        self._results.append(result)
        self.table_view.add_result(result)
        self.card_view.add_result(result)

    def clear_results(self):
        self._results.clear()
        self.table_view.clear_results()
        self.card_view.clear_results()

    def get_results(self) -> List[RecognitionResult]:
        return self._results.copy()

    def _show_detail(self, index: int):
        if index < 0 or index >= len(self._results):
            return
        r = self._results[index]
        panel = SingleResultPanel(show_heatmap=True)
        panel.set_result(r)

        dialog = QDialog(self.window())
        dialog.setWindowTitle(f"识别详情 - {r.filename}")
        dialog.setMinimumSize(800, 500)
        dialog.setModal(True)

        d_layout = QVBoxLayout(dialog)
        d_layout.addWidget(panel)

        btn_close = QPushButton("关闭")
        btn_close.setObjectName("SecondaryButton")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setFixedWidth(120)
        btn_close.clicked.connect(dialog.close)
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(btn_close)
        close_row.addStretch()
        d_layout.addLayout(close_row)

        dialog.exec_()

    def apply_scale(self, scale: float):
        """缩放面板尺寸"""
        pass
