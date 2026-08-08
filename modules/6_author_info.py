# -*- coding: utf-8 -*-
import streamlit as st

def render():
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h2 style="margin:0; font-size:1.8rem; color:var(--agoda-primary); font-weight:800;">👨‍💻 BƯỚC 6: INFO TÁC GIẢ</h2>
        <p style="margin:5px 0 0; font-size:0.9rem; color:var(--text-color); opacity:0.8;">Thông tin về nhóm phát triển đồ án tốt nghiệp K314</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### 👨‍💻 Thông tin Đồ án Tốt nghiệp & Nhóm Thực Hiện")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            *   **Tên đồ án**: Đồ Án Tốt Nghiệp - Agoda Hotel Recommender System (K314)
            *   **Đơn vị đào tạo**: Trung tâm Tin học - Trường Đại học Khoa học Tự nhiên TP.HCM
            """)
        with col_info2:
            st.markdown("""
            *   **Thành viên thực hiện**: **NGUYỄN VĂN NAM & LÊ VĂN LƯU**
            *   **Lớp học**: K314 — Máy Học & Khoa Học Dữ Liệu
            """)
