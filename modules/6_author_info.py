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

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("### 📋 Bảng Phân Công Nhiệm Vụ (Task Allocation)")
        
        st.markdown("""
<table style="width:100%; border-collapse: collapse; text-align: left; font-size: 0.95rem;">
    <thead>
        <tr style="background-color: rgba(59, 130, 246, 0.1); border-bottom: 2px solid rgba(255,255,255,0.1);">
            <th style="padding: 10px; width: 5%;">TT</th>
            <th style="padding: 10px; width: 55%;">Giai đoạn / Hạng mục công việc</th>
            <th style="padding: 10px; width: 40%;">Người thực hiện</th>
        </tr>
    </thead>
    <tbody>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 10px;">1</td>
            <td style="padding: 10px;"><b>Phần 1: Business Understanding</b></td>
            <td style="padding: 10px;">Nguyễn Văn Nam & Lê Văn Lưu</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 10px;">2</td>
            <td style="padding: 10px;"><b>Phần 2: Data Understanding</b></td>
            <td style="padding: 10px;">Lê Văn Lưu</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 10px;">3</td>
            <td style="padding: 10px;"><b>Phần 3: Data Preparation</b></td>
            <td style="padding: 10px;">Lê Văn Lưu</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 10px;">4</td>
            <td style="padding: 10px;"><b>Phần 4: Modeling</b> <i>(Xây dựng Hybrid Recommender System)</i></td>
            <td style="padding: 10px;">Nguyễn Văn Nam</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 10px;">5</td>
            <td style="padding: 10px;"><b>Phần 5: Evaluation</b> <i>(Đánh giá mô hình)</i></td>
            <td style="padding: 10px;">Nguyễn Văn Nam</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 10px;">6</td>
            <td style="padding: 10px;"><b>Phần 6: Xây dựng Giao diện (App UI / Insights)</b></td>
            <td style="padding: 10px;">Lê Văn Lưu</td>
        </tr>
        <tr>
            <td style="padding: 10px;">7</td>
            <td style="padding: 10px;"><b>Quản trị dự án: Triển khai GitHub & Cấu trúc lại mã nguồn</b></td>
            <td style="padding: 10px;">Nguyễn Văn Nam</td>
        </tr>
    </tbody>
</table>
        """, unsafe_allow_html=True)
