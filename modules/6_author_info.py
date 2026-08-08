# -*- coding: utf-8 -*-
import streamlit as st

def render():


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
