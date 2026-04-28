"""
历史记录页面 - 查看本次会话的识别历史
"""
import streamlit as st


def page_history():
    st.title("📋 历史记录")
    st.markdown("---")

    history = st.session_state.get("history", [])

    if not history:
        st.markdown("""
        <div style="text-align: center; padding: 80px 0; color: #999;">
            <h2>📭</h2>
            <p>暂无识别记录</p>
            <p>进行图片识别后，记录将自动保存于此</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # 统计信息
    col1, col2 = st.columns(2)
    with col1:
        st.metric("本次会话识别次数", len(history))
    with col2:
        if st.button("🗑️ 清空历史记录", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    st.markdown("---")

    # 历史记录列表
    for i, record in enumerate(reversed(history)):
        idx = len(history) - i
        result = record["result"]
        image = record.get("image")

        with st.container():
            cols = st.columns([1, 2])

            with cols[0]:
                if image is not None:
                    from PIL import Image as PILImage
                    st.image(PILImage.fromarray(image), use_container_width=True)

            with cols[1]:
                class_name = result.get("class_name", "Unknown")
                confidence = result.get("confidence", 0)

                # 置信度颜色
                if confidence > 0.9:
                    conf_color = "#28a745"
                elif confidence > 0.7:
                    conf_color = "#ffc107"
                else:
                    conf_color = "#dc3545"

                st.markdown(f"""
                **记录 #{idx}**

                - 🏷️ 类别: **{class_name}**
                - 📊 置信度: <span style="color: {conf_color}; font-weight: bold;">{confidence:.1%}</span>
                """, unsafe_allow_html=True)

            st.divider()
