"""
鸟类图像识别系统 - Streamlit 主入口
"""
import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import OUTPUT_DIR, PROJECT_ROOT

# ==================== 页面模块导入 ====================
from pages.home import page_home
from pages.single_recognition import page_single_recognition
from pages.batch_recognition import page_batch_recognition
from pages.history import page_history
from pages.about import page_about


# ==================== 全局配置 ====================
st.set_page_config(
    page_title="鸟类图像识别系统",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义主题色
st.markdown("""
<style>
    :root {
        --primary-color: #2E86AB;
        --background-color: #F8F9FA;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2E86AB 0%, #1a5f7a 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .stApp {
        background-color: #F8F9FA;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 状态管理 ====================
def init_session_state():
    """初始化全局 session_state"""
    defaults = {
        "current_image": None,
        "recognition_result": None,
        "history": [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ==================== 模型缓存 ====================
@st.cache_resource
def load_model():
    """缓存模型加载，避免重复初始化"""
    from src.model import create_model
    import torch

    model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(num_classes=200, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ==================== 初始化检查 ====================
def check_environment():
    """页面加载时的初始化检查"""
    issues = []

    # 检查模型文件
    model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        issues.append("⚠️ 模型文件未找到：`output/best_model.pth`，识别功能暂不可用")

    # 检查数据集
    dataset_dir = os.path.join(PROJECT_ROOT, "CUB_200_2011", "CUB_200_2011")
    if not os.path.exists(dataset_dir):
        issues.append("⚠️ 数据集目录未找到：`CUB_200_2011/`，部分功能受限")

    return issues


# ==================== 侧边栏 ====================
def render_sidebar():
    """渲染侧边栏：Logo + 导航菜单"""
    with st.sidebar:
        # Logo 区域
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 28px; margin: 0;">🐦 鸟类识别</h1>
            <p style="font-size: 13px; opacity: 0.8; margin: 5px 0 0;">Avian Monitoring System</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # 导航菜单
        selected = option_menu(
            menu_title=None,
            options=["首页", "单图识别", "批量识别", "历史记录", "关于"],
            icons=["house", "search", "images", "clock-history", "info-circle"],
            menu_icon=None,
            default_index=0,
            styles={
                "container": {"padding": "0 !important", "background-color": "transparent"},
                "icon": {"color": "rgba(255,255,255,0.7)", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "4px 0",
                    "padding": "10px 16px",
                    "border-radius": "8px",
                    "color": "white",
                },
                "nav-link-selected": {
                    "background-color": "rgba(255,255,255,0.2)",
                    "font-weight": "bold",
                },
            },
        )

        st.divider()

        # 环境状态
        st.markdown("**系统状态**")
        model = load_model()
        if model is not None:
            st.success("✅ 模型已加载")
        else:
            st.warning("❌ 模型未加载")

        import torch
        st.info(f"🖥️ 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    return selected


# ==================== 主流程 ====================
def main():
    init_session_state()

    # 环境检查
    issues = check_environment()
    if issues:
        with st.sidebar:
            with st.expander("⚠️ 环境提示", expanded=False):
                for issue in issues:
                    st.markdown(issue)

    # 渲染侧边栏并获取选中页
    selected = render_sidebar()

    # 页面路由
    page_map = {
        "首页": page_home,
        "单图识别": page_single_recognition,
        "批量识别": page_batch_recognition,
        "历史记录": page_history,
        "关于": page_about,
    }

    page_func = page_map.get(selected, page_home)
    page_func()


if __name__ == "__main__":
    main()
