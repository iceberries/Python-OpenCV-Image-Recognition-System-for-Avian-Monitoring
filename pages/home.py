"""
首页 - 系统概览与快速入口
"""
import streamlit as st


def page_home():
    st.title("🏠 首页")
    st.markdown("---")

    # 欢迎横幅
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #2E86AB 0%, #1a5f7a 100%);
        padding: 40px;
        border-radius: 16px;
        color: black;
        text-align: center;
        margin-bottom: 30px;
    ">
        <h1 style="margin: 0; font-size: 36px; color: black;">🐦 鸟类图像识别系统</h1>
        <p style="font-size: 18px; color: black; margin-top: 12px;">
            基于 OpenCV 与深度学习的智能鸟类监测平台
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 功能卡片
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: white; padding: 24px; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;">
            <h2 style="color: #2E86AB;">🔍</h2>
            <h3 style="color: black;">单图识别</h3>
            <p style="color: black;">上传单张鸟类图片，<br>即时获取识别结果</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入单图识别 →", key="goto_single", use_container_width=True):
            st.session_state["_nav_select"] = "单图识别"
            st.rerun()

    with col2:
        st.markdown("""
        <div style="background: white; padding: 24px; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;">
            <h2 style="color: #2E86AB;">🖼️</h2>
            <h3 style="color: black;">批量识别</h3>
            <p style="color: black;">批量上传图片，<br>高效处理大量数据</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入批量识别 →", key="goto_batch", use_container_width=True):
            st.session_state["_nav_select"] = "批量识别"
            st.rerun()

    with col3:
        st.markdown("""
        <div style="background: white; padding: 24px; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;">
            <h2 style="color: #2E86AB;">📋</h2>
            <h3 style="color: black;">历史记录</h3>
            <p style="color: black;">查看本次会话的<br>所有识别记录</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("查看历史记录 →", key="goto_history", use_container_width=True):
            st.session_state["_nav_select"] = "历史记录"
            st.rerun()

    st.markdown("---")

    # 系统信息
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📊 系统信息")
        from src.config import NUM_CLASSES, INPUT_SIZE, NUM_EPOCHS
        st.markdown(f"""
        - **识别类别数**: {NUM_CLASSES} 种鸟类
        - **输入尺寸**: {INPUT_SIZE}×{INPUT_SIZE}
        - **训练轮次**: {NUM_EPOCHS}
        """)
    with col_b:
        st.subheader("🚀 快速开始")
        st.markdown("""
        1. 在侧边栏选择 **单图识别**
        2. 上传一张鸟类图片
        3. 点击 **开始识别** 查看结果
        """)
