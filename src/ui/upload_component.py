"""
图像上传与预览组件 - 支持单击/拖拽上传、缩略图网格、删除、空状态
"""
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QFileDialog, QGridLayout,
    QScrollArea, QSizePolicy, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QDragEnterEvent, QDropEvent

from src.ui.styles import (
    PRIMARY_COLOR, PRIMARY_LIGHT, DANGER_COLOR,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR, BG_COLOR, CARD_BG,
)
from src.ui.scale_manager import ScaleManager

# 常量
MAX_FILE_SIZE_MB = 20
MAX_BATCH_COUNT = 50
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
THUMB_MAX_WIDTH = 400
GRID_COLUMNS = 4
THUMB_GRID_SIZE = 150


class ThumbnailWidget(QFrame):
    """单张缩略图卡片，含删除按钮和悬停信息"""

    delete_requested = pyqtSignal(str)  # 发出文件路径

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self._info_text = ""
        self.setObjectName("ThumbCard")
        self.setFixedSize(THUMB_GRID_SIZE + 16, THUMB_GRID_SIZE + 50)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(4)

        # 缩略图区域（含删除按钮）
        img_container = QFrame()
        img_container.setStyleSheet("border: none;")
        img_layout = QVBoxLayout(img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(0)

        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setFixedSize(THUMB_GRID_SIZE, THUMB_GRID_SIZE)
        self.thumb_label.setStyleSheet(f"border-radius: 4px; background: {BG_COLOR};")
        img_layout.addWidget(self.thumb_label)

        # 删除按钮（右上角覆盖）
        self.btn_delete = QPushButton("✕")
        self.btn_delete.setFixedSize(24, 24)
        self.btn_delete.setCursor(Qt.PointingHandCursor)
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self.file_path))
        self.btn_delete.setObjectName("DeleteButton")
        self.btn_delete.setStyleSheet(f"""
            QPushButton#DeleteButton {{
                background-color: rgba(220,53,69,0.85);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton#DeleteButton:hover {{
                background-color: {DANGER_COLOR};
            }}
        """)
        # 绝对定位到右上角
        self.btn_delete.setParent(self.thumb_label)
        self.btn_delete.move(THUMB_GRID_SIZE - 28, 4)

        layout.addWidget(img_container)

        # 文件名标签
        filename = os.path.basename(file_path)
        if len(filename) > 18:
            filename = filename[:15] + "..."
        name_label = QLabel(filename)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; border: none;")
        name_label.setProperty("_base_font_size", 12)
        name_label.setWordWrap(False)
        self.name_label = name_label
        layout.addWidget(name_label)

        # 加载图片
        self._load_thumbnail(file_path)

        # 悬停事件
        self.thumb_label.installEventFilter(self)

    def _load_thumbnail(self, file_path: str):
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.thumb_label.setText("无法加载")
            return

        # 获取文件信息用于 tooltip
        file_size = os.path.getsize(file_path)
        size_mb = file_size / (1024 * 1024)
        ext = os.path.splitext(file_path)[1].upper().lstrip(".")
        self._info_text = (
            f"尺寸: {pixmap.width()} × {pixmap.height()}\n"
            f"大小: {size_mb:.2f} MB\n"
            f"格式: {ext}"
        )
        self.setToolTip(self._info_text)

        scaled = pixmap.scaled(
            THUMB_GRID_SIZE, THUMB_GRID_SIZE,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.thumb_label.setPixmap(scaled)

    def eventFilter(self, obj, event):
        if obj is self.thumb_label and event.type() == event.ToolTip:
            return False  # 让 QToolTip 正常工作
        return super().eventFilter(obj, event)

    def apply_scale(self, scale: float):
        """根据缩放更新缩略图尺寸"""
        sm = ScaleManager.get()
        thumb_size = sm.scale_int(THUMB_GRID_SIZE)
        self.setFixedSize(thumb_size + sm.scale_int(16), thumb_size + sm.scale_int(50))
        self.thumb_label.setFixedSize(thumb_size, thumb_size)
        self.btn_delete.setFixedSize(sm.scale_int(24), sm.scale_int(24))
        self.btn_delete.move(thumb_size - sm.scale_int(28), sm.scale_int(4))
        self.btn_delete.setStyleSheet(f"""
            QPushButton#DeleteButton {{
                background-color: rgba(220,53,69,0.85);
                color: white;
                border: none;
                border-radius: {sm.scale_int(12)}px;
                font-size: {sm.scale_int(12)}px;
                font-weight: bold;
            }}
            QPushButton#DeleteButton:hover {{
                background-color: {DANGER_COLOR};
            }}
        """)
        self.name_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {sm.scale_int(12)}px; border: none;"
        )
        # 重新缩放图片
        pixmap = QPixmap(self.file_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                thumb_size, thumb_size,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.thumb_label.setPixmap(scaled)


class UploadComponent(QWidget):
    """
    通用图像上传与预览组件

    模式:
      - "single": 单图上传，显示大预览
      - "batch":  批量上传，网格缩略图

    信号:
      - files_changed(list[str]): 文件列表变化时发出（文件路径列表）
      - images_ready(list[np.ndarray]): 图片 numpy 数据就绪
    """

    files_changed = pyqtSignal(list)   # 文件路径列表
    images_ready = pyqtSignal(list)    # np.ndarray 列表

    def __init__(self, mode: str = "single", parent=None):
        super().__init__(parent)
        self._mode = mode
        self._file_paths: list[str] = []
        self._images: list[np.ndarray] = []
        self.setAcceptDrops(True)

        self._build_ui()

    def _build_ui(self):
        sm = ScaleManager.get()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 空状态 / 拖拽区域
        self.drop_zone = QFrame()
        self.drop_zone.setObjectName("UploadArea")
        self.drop_zone.setMinimumHeight(sm.scale_int(160))
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setAlignment(Qt.AlignCenter)
        drop_layout.setSpacing(8)

        # 空状态插画（用 emoji 代替）
        self.empty_icon = QLabel("🖼️")
        self.empty_icon.setAlignment(Qt.AlignCenter)
        self.empty_icon.setStyleSheet("font-size: 48px; border: none;")
        self.empty_icon.setProperty("_base_font_size", 36)  # emoji 用较小基准
        drop_layout.addWidget(self.empty_icon)

        self.empty_hint = QLabel("拖拽图片到此处或点击下方按钮上传\n支持 JPG / PNG / BMP，单文件 ≤ 20MB")
        self.empty_hint.setAlignment(Qt.AlignCenter)
        self.empty_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px; border: none;")
        self.empty_hint.setProperty("_base_font_size", 14)
        drop_layout.addWidget(self.empty_hint)

        if self._mode == "batch":
            self.empty_hint.setText(
                "拖拽图片到此处或点击下方按钮上传\n"
                "支持 JPG / PNG / BMP，单文件 ≤ 20MB，最多 50 张"
            )

        layout.addWidget(self.drop_zone)

        # 单图大预览
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(sm.scale_int(280), sm.scale_int(280))
        self.preview_label.setMaximumWidth(sm.scale_int(THUMB_MAX_WIDTH))
        self.preview_label.setStyleSheet(f"border-radius: 6px; background: {BG_COLOR};")
        self.preview_label.hide()
        if self._mode == "single":
            layout.addWidget(self.preview_label)

        # 批量缩略图网格（滚动区域）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background: {CARD_BG};
            }}
        """)
        self.scroll_area.hide()
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(12)
        self.grid_layout.setContentsMargins(12, 12, 12, 12)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.grid_container)
        if self._mode == "batch":
            layout.addWidget(self.scroll_area, 1)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_upload = QPushButton("📁 选择图片")
        if self._mode == "batch":
            self.btn_upload.setText("📁 选择图片（多选）")
        self.btn_upload.setObjectName("SecondaryButton")
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.clicked.connect(self._on_click_upload)
        btn_row.addWidget(self.btn_upload)

        self.btn_clear = QPushButton("🗑️ 清空")
        self.btn_clear.setObjectName("DangerButton")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(self.clear_all)
        self._update_btn_clear_style()
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 文件计数标签（批量模式）
        self.count_label = QLabel("")
        self.count_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; border: none;")
        if self._mode == "batch":
            layout.addWidget(self.count_label)

    # ========== 拖拽 ==========
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_zone.setStyleSheet(f"""
                #UploadArea {{
                    background-color: {BG_COLOR};
                    border: 2px dashed {PRIMARY_COLOR};
                    border-radius: 8px;
                    padding: 40px;
                }}
            """)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_zone.setStyleSheet("")  # 恢复 QSS 中的默认样式

    def dropEvent(self, event: QDropEvent):
        self.drop_zone.setStyleSheet("")  # 恢复默认
        urls = event.mimeData().urls()
        paths = []
        for url in urls:
            path = url.toLocalFile()
            if path and self._is_valid_file(path):
                paths.append(path)
        if paths:
            self._add_files(paths)
        event.acceptProposedAction()

    # ========== 点击上传 ==========
    def _on_click_upload(self):
        if self._mode == "single":
            paths, _ = QFileDialog.getOpenFileName(
                self, "选择鸟类图片", "",
                "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
            )
            if paths:
                self._add_files([paths])
        else:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "选择鸟类图片", "",
                "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
            )
            if paths:
                self._add_files(paths)

    # ========== 文件验证与添加 ==========
    def _is_valid_file(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in ALLOWED_EXTENSIONS

    def _add_files(self, paths: list[str]):
        valid_paths = []
        oversize_files = []

        for p in paths:
            # 文件类型
            if not self._is_valid_file(p):
                continue

            # 文件大小
            file_size = os.path.getsize(p)
            size_mb = file_size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                oversize_files.append(os.path.basename(p))
                continue

            # 去重
            if p not in self._file_paths:
                valid_paths.append(p)

        # 超限提示
        if oversize_files:
            QMessageBox.warning(
                self, "文件过大",
                f"以下文件超过 {MAX_FILE_SIZE_MB}MB 限制，已跳过：\n"
                + "\n".join(oversize_files[:10]),
            )

        # 批量数量限制
        if self._mode == "batch":
            remaining = MAX_BATCH_COUNT - len(self._file_paths)
            if remaining <= 0:
                QMessageBox.warning(self, "数量超限", f"最多同时上传 {MAX_BATCH_COUNT} 张图片")
                return
            valid_paths = valid_paths[:remaining]

        if not valid_paths:
            return

        if self._mode == "single":
            # 单图模式：替换
            self._file_paths = valid_paths[:1]
        else:
            self._file_paths.extend(valid_paths)

        self._load_images()
        self._refresh_view()
        self._emit_signals()

    def _load_images(self):
        """将文件路径加载为 np.ndarray"""
        self._images = []
        for path in self._file_paths:
            img = QImage(path)
            if img.isNull():
                self._images.append(None)
                continue
            # 转为 RGB ndarray
            img = img.convertToFormat(QImage.Format_RGB888)
            ptr = img.bits()
            ptr.setsize(img.height() * img.bytesPerLine())
            arr = np.array(ptr).reshape(img.height(), img.bytesPerLine(), 3)
            # 去除 stride 对齐多余列
            arr = arr[:, :img.width() * 3].reshape(img.height(), img.width(), 3)
            self._images.append(arr.copy())

    def _refresh_view(self):
        """根据当前文件列表刷新视图"""
        has_files = len(self._file_paths) > 0

        # 空状态
        self.drop_zone.setVisible(not has_files)
        self.btn_clear.setEnabled(has_files)

        if self._mode == "single":
            self._refresh_single(has_files)
        else:
            self._refresh_batch(has_files)

    def _refresh_single(self, has_files: bool):
        if has_files:
            path = self._file_paths[0]
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.preview_label.size(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                self.preview_label.setPixmap(scaled)

                file_size = os.path.getsize(path)
                size_mb = file_size / (1024 * 1024)
                ext = os.path.splitext(path)[1].upper().lstrip(".")
                self.preview_label.setToolTip(
                    f"尺寸: {pixmap.width()} × {pixmap.height()}\n"
                    f"大小: {size_mb:.2f} MB\n"
                    f"格式: {ext}"
                )
            self.preview_label.show()
        else:
            self.preview_label.hide()
            self.preview_label.clear()

    def _refresh_batch(self, has_files: bool):
        # 清除旧缩略图
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if has_files:
            self.scroll_area.show()
            for i, path in enumerate(self._file_paths):
                thumb = ThumbnailWidget(path)
                thumb.delete_requested.connect(self._on_delete_thumb)
                row, col = divmod(i, GRID_COLUMNS)
                self.grid_layout.addWidget(thumb, row, col)

            self.count_label.setText(f"已选择 {len(self._file_paths)} 张图片")
        else:
            self.scroll_area.hide()
            self.count_label.setText("")

    def _on_delete_thumb(self, file_path: str):
        """缩略图删除按钮回调"""
        if file_path in self._file_paths:
            idx = self._file_paths.index(file_path)
            self._file_paths.remove(file_path)
            if idx < len(self._images):
                self._images.pop(idx)
            self._refresh_view()
            self._emit_signals()

    def _emit_signals(self):
        self.files_changed.emit(self._file_paths.copy())
        valid_images = [img for img in self._images if img is not None]
        self.images_ready.emit(valid_images)

    # ========== 公共 API ==========
    def clear_all(self):
        """清空所有已上传文件"""
        self._file_paths.clear()
        self._images.clear()
        self._refresh_view()
        self._emit_signals()

    def get_file_paths(self) -> list[str]:
        return self._file_paths.copy()

    def get_images(self) -> list[np.ndarray]:
        return [img for img in self._images if img is not None]

    def has_files(self) -> bool:
        return len(self._file_paths) > 0

    def file_count(self) -> int:
        return len(self._file_paths)

    def _update_btn_clear_style(self):
        """根据当前缩放更新清空按钮样式"""
        sm = ScaleManager.get()
        r = sm.scale_int(6)
        p = f"{sm.scale_int(10)}px {sm.scale_int(24)}px"
        fs = sm.scale_int(14)
        self.btn_clear.setStyleSheet(f"""
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

    def apply_scale(self, scale: float):
        """缩放组件尺寸"""
        sm = ScaleManager.get()
        self.drop_zone.setMinimumHeight(sm.scale_int(160))
        if self._mode == "single":
            self.preview_label.setMinimumSize(sm.scale_int(280), sm.scale_int(280))
            self.preview_label.setMaximumWidth(sm.scale_int(THUMB_MAX_WIDTH))

        # 更新清空按钮样式
        self._update_btn_clear_style()

        # 更新缩略图网格间距与边距
        sp = sm.scale_int(12)
        self.grid_layout.setSpacing(sp)
        self.grid_layout.setContentsMargins(sp, sp, sp, sp)

        # 更新空状态图标与提示文字
        self.empty_icon.setStyleSheet(f"font-size: {sm.scale_int(48)}px; border: none;")
        self.empty_hint.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {sm.scale_int(14)}px; border: none;"
        )

        # 更新计数标签
        self.count_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {sm.scale_int(12)}px; border: none;"
        )

        # 更新滚动区圆角
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: {sm.scale_int(6)}px;
                background: {CARD_BG};
            }}
        """)

        # 重新缩放已有缩略图
        thumb_size = sm.scale_int(THUMB_GRID_SIZE)
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), ThumbnailWidget):
                item.widget().apply_scale(scale)
