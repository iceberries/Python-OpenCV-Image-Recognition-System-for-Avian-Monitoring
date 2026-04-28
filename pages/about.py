"""
关于页面 - 系统介绍与技术信息
"""
import streamlit as st
import sys


def page_about():
    st.title("ℹ️ 关于")
    st.markdown("---")

    # 项目介绍
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #2E86AB 0%, #1a5f7a 100%);
        padding: 32px; border-radius: 16px; color: white; margin-bottom: 24px;
    ">
        <h2 style="margin: 0;">🐦 鸟类图像识别系统</h2>
        <p style="opacity: 0.9; margin-top: 8px;">
            Python OpenCV Image Recognition System for Avian Monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 功能介绍
    st.subheader("🎯 核心功能")
    features = [
        ("🔍 单图识别", "上传单张鸟类图片，基于深度学习模型进行分类识别"),
        ("🖼️ 批量识别", "支持批量上传图片，高效处理大量数据"),
        ("📋 历史记录", "自动保存识别结果，方便回溯查看"),
        ("🔥 热力图可视化", "Grad-CAM 可视化，展示模型关注区域"),
        ("⚡ 数据增强", "Mixup、CutMix、TTA 等多种增强策略"),
    ]
    for title, desc in features:
        st.markdown(f"**{title}** — {desc}")

    st.markdown("---")

    # 技术栈
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛠️ 技术栈")
        st.markdown("""
        - **框架**: PyTorch, Streamlit
        - **视觉**: OpenCV, PIL, Matplotlib
        - **模型**: ResNet-50 + SE-Attention
        - **数据集**: CUB-200-2011 (200 种鸟类)
        """)

    with col2:
        st.subheader("📊 模型配置")
        from src.config import (
            NUM_CLASSES, INPUT_SIZE, BATCH_SIZE,
            LABEL_SMOOTHING, USE_TTA, USE_SE_ATTENTION,
        )
        st.markdown(f"""
        - **类别数**: {NUM_CLASSES}
        - **输入尺寸**: {INPUT_SIZE}×{INPUT_SIZE}
        - **批次大小**: {BATCH_SIZE}
        - **Label Smoothing**: {LABEL_SMOOTHING}
        - **TTA**: {'开启' if USE_TTA else '关闭'}
        - **SE-Attention**: {'开启' if USE_SE_ATTENTION else '关闭'}
        """)

    st.markdown("---")

    # 运行环境
    st.subheader("🖥️ 运行环境")
    import torch
    import platform

    st.markdown(f"""
    - **Python**: {sys.version.split()[0]}
    - **操作系统**: {platform.system()} {platform.release()}
    - **PyTorch**: {torch.__version__}
    - **CUDA 可用**: {'是' if torch.cuda.is_available() else '否'}
    """ + (f"- **GPU**: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
