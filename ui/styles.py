"""
全局 QSS 样式定义 - 现代扁平化风格
"""

# 主色
PRIMARY_COLOR = "#2E86AB"
PRIMARY_DARK = "#1a5f7a"
PRIMARY_LIGHT = "#4ba3c9"
BG_COLOR = "#F8F9FA"
CARD_BG = "#FFFFFF"
TEXT_PRIMARY = "#2C3E50"
TEXT_SECONDARY = "#7F8C8D"
BORDER_COLOR = "#E0E0E0"
SUCCESS_COLOR = "#28a745"
WARNING_COLOR = "#ffc107"
DANGER_COLOR = "#dc3545"

GLOBAL_QSS = f"""
/* ===== 全局 ===== */
QWidget {{
    font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", sans-serif;
    font-size: 14px;
    color: {TEXT_PRIMARY};
}}

QMainWindow {{
    background-color: {BG_COLOR};
}}

/* ===== 侧边栏 ===== */
#Sidebar {{
    background-color: {PRIMARY_COLOR};
    border: none;
    min-width: 200px;
    max-width: 200px;
}}

#SidebarLogo {{
    color: white;
    font-size: 22px;
    font-weight: bold;
    padding: 20px 16px 8px 16px;
}}

#SidebarSubtitle {{
    color: rgba(255,255,255,0.7);
    font-size: 12px;
    padding: 0px 16px 16px 16px;
}}

#NavButton {{
    background-color: transparent;
    color: rgba(255,255,255,0.85);
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: left;
    font-size: 15px;
    margin: 2px 8px;
}}

#NavButton:hover {{
    background-color: rgba(255,255,255,0.15);
    color: white;
}}

#NavButton[selected="true"] {{
    background-color: rgba(255,255,255,0.2);
    color: white;
    font-weight: bold;
}}

/* ===== 卡片 ===== */
#Card {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 20px;
}}

#Card:hover {{
    border-color: {PRIMARY_LIGHT};
}}

/* ===== 内容区标题 ===== */
#PageTitle {{
    font-size: 24px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
    padding: 0px 0px 8px 0px;
}}

#SectionTitle {{
    font-size: 16px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
    padding: 8px 0px;
}}

/* ===== 按钮 ===== */
QPushButton#PrimaryButton {{
    background-color: {PRIMARY_COLOR};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: bold;
}}

QPushButton#PrimaryButton:hover {{
    background-color: {PRIMARY_DARK};
}}

QPushButton#PrimaryButton:pressed {{
    background-color: {PRIMARY_DARK};
    padding-top: 12px;
}}

QPushButton#SecondaryButton {{
    background-color: transparent;
    color: {PRIMARY_COLOR};
    border: 1px solid {PRIMARY_COLOR};
    border-radius: 6px;
    padding: 10px 24px;
    font-size: 14px;
}}

QPushButton#SecondaryButton:hover {{
    background-color: rgba(46,134,171,0.08);
}}

/* ===== 输入框 ===== */
QLineEdit, QTextEdit, QPlainTextEdit {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 8px 12px;
    background-color: {CARD_BG};
    selection-background-color: {PRIMARY_LIGHT};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {PRIMARY_COLOR};
}}

/* ===== 状态栏 ===== */
QStatusBar {{
    background-color: {CARD_BG};
    border-top: 1px solid {BORDER_COLOR};
    color: {TEXT_SECONDARY};
    font-size: 12px;
    padding: 4px 8px;
}}

/* ===== 标签页 ===== */
QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    background: {CARD_BG};
}}

QTabBar::tab {{
    padding: 8px 16px;
    border: none;
    border-bottom: 2px solid transparent;
}}

QTabBar::tab:selected {{
    color: {PRIMARY_COLOR};
    border-bottom: 2px solid {PRIMARY_COLOR};
    font-weight: bold;
}}

/* ===== 滚动条 ===== */
QScrollBar:vertical {{
    border: none;
    background: {BG_COLOR};
    width: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background: #c0c0c0;
    border-radius: 4px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: #a0a0a0;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ===== 进度条 ===== */
QProgressBar {{
    border: none;
    border-radius: 4px;
    background-color: {BORDER_COLOR};
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {PRIMARY_COLOR};
    border-radius: 4px;
}}

/* ===== 分割线 ===== */
#Divider {{
    background-color: {BORDER_COLOR};
    max-height: 1px;
}}

/* ===== 上传区域 ===== */
#UploadArea {{
    background-color: {BG_COLOR};
    border: 2px dashed {BORDER_COLOR};
    border-radius: 8px;
    padding: 40px;
}}

#UploadArea:hover {{
    border-color: {PRIMARY_COLOR};
}}

/* ===== 置信度标签 ===== */
#ConfHigh {{
    color: {SUCCESS_COLOR};
    font-weight: bold;
    font-size: 20px;
}}

#ConfMedium {{
    color: {WARNING_COLOR};
    font-weight: bold;
    font-size: 20px;
}}

#ConfLow {{
    color: {DANGER_COLOR};
    font-weight: bold;
    font-size: 20px;
}}
"""

NAV_ICONS = {
    "首页": "🏠",
    "单图识别": "🔍",
    "批量识别": "🖼️",
    "历史记录": "📋",
    "设置": "⚙️",
}
