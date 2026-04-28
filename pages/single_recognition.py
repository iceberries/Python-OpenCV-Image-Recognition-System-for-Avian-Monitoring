"""
单图识别页面 - 上传单张图片进行鸟类识别
"""
import streamlit as st
import numpy as np
from PIL import Image


def page_single_recognition():
    st.title("🔍 单图识别")
    st.markdown("---")

    # 图片上传区域
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("📤 上传图片")
        uploaded_file = st.file_uploader(
            "选择一张鸟类图片",
            type=["jpg", "jpeg", "png", "bmp"],
            key="single_uploader",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="上传的图片", use_container_width=True)

            # 存入 session_state
            st.session_state["current_image"] = np.array(image)

            # 识别按钮
            if st.button("🚀 开始识别", type="primary", use_container_width=True):
                _run_recognition(np.array(image))
        else:
            st.info("👆 请上传一张鸟类图片")

    with col_result:
        st.subheader("📋 识别结果")

        if st.session_state.get("recognition_result") is not None:
            _display_result(st.session_state["recognition_result"])
        else:
            st.empty().markdown("""
            <div style="text-align: center; padding: 80px 0; color: #999;">
                <h2>🎯</h2>
                <p>上传图片并点击识别后，结果将在此处显示</p>
            </div>
            """, unsafe_allow_html=True)


def _run_recognition(image: np.ndarray):
    """执行识别（占位逻辑，后续接入实际模型）"""
    with st.spinner("正在识别中..."):
        # TODO: 接入实际模型推理
        # model = load_model()
        # result = model.predict(image)

        # 模拟识别结果
        import time
        time.sleep(1)

        result = {
            "class_name": "Blue Jay",
            "confidence": 0.92,
            "top_k": [
                {"class_name": "Blue Jay", "confidence": 0.92},
                {"class_name": "Stellar's Jay", "confidence": 0.05},
                {"class_name": "Florida Jay", "confidence": 0.02},
            ],
        }

        st.session_state["recognition_result"] = result

        # 追加到历史记录
        st.session_state["history"].append({
            "image": image,
            "result": result,
        })

    st.success("✅ 识别完成！")


def _display_result(result: dict):
    """显示识别结果"""
    # 主分类
    class_name = result["class_name"]
    confidence = result["confidence"]

    # 置信度颜色
    if confidence > 0.9:
        color = "#28a745"
    elif confidence > 0.7:
        color = "#ffc107"
    else:
        color = "#dc3545"

    st.markdown(f"""
    <div style="
        background: white; padding: 24px; border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 16px;
    ">
        <h3 style="margin: 0; color: #2E86AB;">{class_name}</h3>
        <p style="font-size: 24px; color: {color}; font-weight: bold; margin: 8px 0 0;">
            {confidence:.1%}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top-K 排行
    top_k = result.get("top_k")
    if top_k:
        st.markdown("**📊 Top-K 排行**")
        for i, item in enumerate(top_k):
            conf = item["confidence"]
            bar_color = "#2E86AB" if i == 0 else "#adb5bd"
            st.markdown(f"""
            <div style="margin-bottom: 8px;">
                <span style="font-size: 14px;">{item['class_name']}</span>
                <div style="background: #e9ecef; border-radius: 4px; height: 8px; margin-top: 4px;">
                    <div style="background: {bar_color}; border-radius: 4px; height: 8px;
                                width: {conf * 100}%;"></div>
                </div>
                <span style="font-size: 12px; color: #666;">{conf:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
