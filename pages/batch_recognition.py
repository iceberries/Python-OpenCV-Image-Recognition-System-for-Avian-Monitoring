"""
批量识别页面 - 批量上传图片进行鸟类识别
"""
import streamlit as st
import numpy as np
from PIL import Image
import time


def page_batch_recognition():
    st.title("🖼️ 批量识别")
    st.markdown("---")

    # 上传区域
    uploaded_files = st.file_uploader(
        "选择多张鸟类图片",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if uploaded_files:
        st.info(f"已选择 {len(uploaded_files)} 张图片")

        # 预览
        with st.expander("📷 图片预览", expanded=False):
            cols = st.columns(min(len(uploaded_files), 5))
            for idx, file in enumerate(uploaded_files):
                col = cols[idx % len(cols)]
                with col:
                    image = Image.open(file).convert("RGB")
                    st.image(image, caption=file.name, use_container_width=True)

        # 批量识别按钮
        if st.button("🚀 批量识别", type="primary", use_container_width=True):
            _run_batch_recognition(uploaded_files)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 80px 0; color: #999;">
            <h2>📁</h2>
            <p>请上传多张鸟类图片进行批量识别</p>
        </div>
        """, unsafe_allow_html=True)

    # 批量结果区域
    if st.session_state.get("batch_results"):
        st.markdown("---")
        st.subheader("📋 识别结果")
        _display_batch_results(st.session_state["batch_results"])


def _run_batch_recognition(uploaded_files):
    """批量识别（占位逻辑）"""
    results = []
    progress = st.progress(0, text="识别中...")

    for i, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")
        image_array = np.array(image)

        # TODO: 接入实际模型推理
        time.sleep(0.3)  # 模拟推理

        result = {
            "filename": file.name,
            "class_name": f"Species_{i+1}",
            "confidence": round(0.7 + np.random.random() * 0.28, 4),
        }
        results.append(result)

        # 追加到历史记录
        st.session_state["history"].append({
            "image": image_array,
            "result": result,
        })

        progress.progress((i + 1) / len(uploaded_files),
                          text=f"已完成 {i + 1}/{len(uploaded_files)}")

    st.session_state["batch_results"] = results
    st.success(f"✅ 批量识别完成！共处理 {len(results)} 张图片")


def _display_batch_results(results):
    """显示批量识别结果"""
    # 统计卡片
    col1, col2, col3 = st.columns(3)
    confidences = [r["confidence"] for r in results]
    with col1:
        st.metric("总数", len(results))
    with col2:
        st.metric("平均置信度", f"{np.mean(confidences):.1%}")
    with col3:
        high_conf = sum(1 for c in confidences if c > 0.9)
        st.metric("高置信度(>90%)", high_conf)

    # 结果表格
    st.dataframe(
        data={
            "文件名": [r["filename"] for r in results],
            "识别类别": [r["class_name"] for r in results],
            "置信度": [f"{r['confidence']:.1%}" for r in results],
        },
        use_container_width=True,
        hide_index=True,
    )
