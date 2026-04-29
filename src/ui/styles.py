"""
全局 QSS 样式定义 - 现代扁平化风格
支持缩放：build_scaled_qss(scale) 生成适配当前窗口的样式表
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

NAV_ICONS = {
    "首页": "🏠",
    "单图识别": "🔍",
    "批量识别": "🖼️",
    "历史记录": "📋",
    "设置": "⚙️",
}


def _s(value: float, scale: float) -> int:
    """缩放 px 值，最小 1"""
    return max(1, round(value * scale))


def build_scaled_qss(scale: float = 1.0) -> str:
    """
    根据 scale factor 生成缩放后的全局 QSS

    Args:
        scale: 缩放因子，1.0 为最大化基准

    Returns:
        完整的 QSS 样式表字符串
    """
    # 缩放后的数值
    fs_base = _s(14, scale)       # 基础字体
    fs_sidebar_logo = _s(22, scale)
    fs_sidebar_sub = _s(12, scale)
    fs_nav_btn = _s(15, scale)
    fs_page_title = _s(24, scale)
    fs_section_title = _s(16, scale)
    fs_primary_btn = _s(14, scale)
    fs_secondary_btn = _s(14, scale)
    fs_status_bar = _s(12, scale)
    fs_conf_label = _s(20, scale)

    pad_sidebar_logo = _s(20, scale)
    pad_sidebar_logo_lr = _s(16, scale)
    pad_sidebar_sub_lr = _s(16, scale)
    pad_nav_btn = _s(12, scale)
    pad_nav_btn_lr = _s(16, scale)
    margin_nav = _s(2, scale)
    margin_nav_lr = _s(8, scale)
    pad_card = _s(20, scale)
    radius_card = _s(8, scale)
    radius_btn = _s(6, scale)
    pad_btn = _s(10, scale)
    pad_btn_lr = _s(24, scale)
    pad_input = _s(8, scale)
    pad_input_lr = _s(12, scale)
    pad_status = _s(4, scale)
    pad_status_lr = _s(8, scale)
    pad_tab = _s(8, scale)
    pad_tab_lr = _s(16, scale)
    sb_width = _s(8, scale)
    sb_min_h = _s(30, scale)
    pb_height = _s(8, scale)
    radius_pb = _s(4, scale)
    upload_pad = _s(40, scale)
    upload_border = max(1, _s(2, scale))
    upload_radius = _s(8, scale)

    sidebar_min_w = _s(200, scale)
    sidebar_max_w = _s(200, scale)
    nav_radius = _s(8, scale)
    input_radius = _s(6, scale)
    tab_radius = _s(6, scale)
    sb_radius = _s(4, scale)
    page_title_pad_b = _s(8, scale)
    section_pad = _s(8, scale)

    return f"""
/* ===== 全局 ===== */
QWidget {{
    font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", sans-serif;
    font-size: {fs_base}px;
    color: {TEXT_PRIMARY};
}}

QMainWindow {{
    background-color: {BG_COLOR};
}}

/* ===== 侧边栏 ===== */
#Sidebar {{
    background-color: {PRIMARY_COLOR};
    border: none;
    min-width: {sidebar_min_w}px;
    max-width: {sidebar_max_w}px;
}}

#SidebarLogo {{
    color: white;
    font-size: {fs_sidebar_logo}px;
    font-weight: bold;
    padding: {pad_sidebar_logo}px {pad_sidebar_logo_lr}px {_s(8, scale)}px {pad_sidebar_logo_lr}px;
}}

#SidebarSubtitle {{
    color: rgba(255,255,255,0.7);
    font-size: {fs_sidebar_sub}px;
    padding: 0px {pad_sidebar_sub_lr}px {_s(16, scale)}px {pad_sidebar_sub_lr}px;
}}

#NavButton {{
    background-color: transparent;
    color: rgba(255,255,255,0.85);
    border: none;
    border-radius: {nav_radius}px;
    padding: {pad_nav_btn}px {pad_nav_btn_lr}px;
    text-align: left;
    font-size: {fs_nav_btn}px;
    margin: {margin_nav}px {margin_nav_lr}px;
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
    border-radius: {radius_card}px;
    padding: {pad_card}px;
}}

#Card:hover {{
    border-color: {PRIMARY_LIGHT};
}}

/* ===== 内容区标题 ===== */
#PageTitle {{
    font-size: {fs_page_title}px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
    padding: 0px 0px {page_title_pad_b}px 0px;
}}

#SectionTitle {{
    font-size: {fs_section_title}px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
    padding: {section_pad}px 0px;
}}

/* ===== 按钮 ===== */
QPushButton#PrimaryButton {{
    background-color: {PRIMARY_COLOR};
    color: white;
    border: none;
    border-radius: {radius_btn}px;
    padding: {pad_btn}px {pad_btn_lr}px;
    font-size: {fs_primary_btn}px;
    font-weight: bold;
}}

QPushButton#PrimaryButton:hover {{
    background-color: {PRIMARY_DARK};
}}

QPushButton#PrimaryButton:pressed {{
    background-color: {PRIMARY_DARK};
    padding-top: {_s(12, scale)}px;
}}

QPushButton#SecondaryButton {{
    background-color: transparent;
    color: {PRIMARY_COLOR};
    border: 1px solid {PRIMARY_COLOR};
    border-radius: {radius_btn}px;
    padding: {pad_btn}px {pad_btn_lr}px;
    font-size: {fs_secondary_btn}px;
}}

QPushButton#SecondaryButton:hover {{
    background-color: rgba(46,134,171,0.08);
}}

/* ===== 输入框 ===== */
QLineEdit, QTextEdit, QPlainTextEdit {{
    border: 1px solid {BORDER_COLOR};
    border-radius: {input_radius}px;
    padding: {pad_input}px {pad_input_lr}px;
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
    font-size: {fs_status_bar}px;
    padding: {pad_status}px {pad_status_lr}px;
}}

/* ===== 标签页 ===== */
QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR};
    border-radius: {tab_radius}px;
    background: {CARD_BG};
}}

QTabBar::tab {{
    padding: {pad_tab}px {pad_tab_lr}px;
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
    width: {sb_width}px;
    border-radius: {sb_radius}px;
}}

QScrollBar::handle:vertical {{
    background: #c0c0c0;
    border-radius: {sb_radius}px;
    min-height: {sb_min_h}px;
}}

QScrollBar::handle:vertical:hover {{
    background: #a0a0a0;
}}

QScrollBar::add-line:vertical, QScrollBar:sub-line:vertical {{
    height: 0px;
}}

/* ===== 进度条 ===== */
QProgressBar {{
    border: none;
    border-radius: {radius_pb}px;
    background-color: {BORDER_COLOR};
    height: {pb_height}px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {PRIMARY_COLOR};
    border-radius: {radius_pb}px;
}}

/* ===== 分割线 ===== */
#Divider {{
    background-color: {BORDER_COLOR};
    max-height: 1px;
}}

/* ===== 上传区域 ===== */
#UploadArea {{
    background-color: {BG_COLOR};
    border: {upload_border}px dashed {BORDER_COLOR};
    border-radius: {upload_radius}px;
    padding: {upload_pad}px;
}}

#UploadArea:hover {{
    border-color: {PRIMARY_COLOR};
}}

/* ===== 置信度标签 ===== */
#ConfHigh {{
    color: {SUCCESS_COLOR};
    font-weight: bold;
    font-size: {fs_conf_label}px;
}}

#ConfMedium {{
    color: {WARNING_COLOR};
    font-weight: bold;
    font-size: {fs_conf_label}px;
}}

#ConfLow {{
    color: {DANGER_COLOR};
    font-weight: bold;
    font-size: {fs_conf_label}px;
}}
"""


# 保留旧的 GLOBAL_QSS 变量用于向后兼容（scale=1.0）
GLOBAL_QSS = build_scaled_qss(1.0)
