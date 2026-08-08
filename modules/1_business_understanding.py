# -*- coding: utf-8 -*-
import streamlit as st

def render():
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h2 style="margin:0; font-size:1.8rem; color:var(--agoda-primary); font-weight:800;">🎯 BƯỚC 1: BUSINESS UNDERSTANDING</h2>
        <p style="margin:5px 0 0; font-size:0.9rem; color:var(--text-color); opacity:0.8;">Thấu hiểu mục tiêu kinh doanh, xác định bài toán & tiêu chí thành công của dự án</p>
    </div>
    """, unsafe_allow_html=True)



    # 2. Mục tiêu chiến lược
    with st.container(border=True):
        st.markdown("### 🎯 Mục Tiêu Chiến Lược")
        st.markdown("""
        Xây dựng **Hệ thống Gợi ý Khách sạn Agoda Nha Trang (Hybrid Recommender System)** nhằm nâng cao trải nghiệm đặt phòng 
        thông qua cá nhân hóa đề xuất cho du khách, đồng thời cung cấp công cụ phân tích phản hồi thông minh (Partner Insights) 
        giúp đối tác khách sạn cải thiện dịch vụ kinh doanh.
        """)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 3. Bài toán & Giải pháp
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown("<h4 style='color:var(--agoda-danger); margin-top:0;'>📌 Nỗi Đau Kinh Doanh</h4>", unsafe_allow_html=True)
            st.markdown("""
            *   **Đối với Du khách**:
                *   *Quá tải thông tin*: Quá nhiều khách sạn tại Nha Trang khiến việc so sánh mất nhiều thời gian.
                *   *Nội dung nhiễu*: Hàng ngàn đánh giá không được phân loại rõ ràng theo tiêu chí mong muốn.
                *   *Thiếu tính cá nhân hóa*: Không có bộ lọc tự động hiểu sở thích dựa trên quốc tịch và tiêu chí cụ thể.
            *   **Đối với Khách sạn**:
                *   *Thiếu thông tin đối soát*: Không biết chất lượng dịch vụ của mình đang đứng ở đâu so với đối thủ cạnh tranh.
                *   *Chưa tối ưu phản hồi*: Chưa có công cụ tự động phát hiện các điểm nghẽn bị phàn nàn nhiều (Vệ sinh, Dịch vụ...).
            """)

    with col2:
        with st.container(border=True):
            st.markdown("<h4 style='color:var(--agoda-success); margin-top:0;'>💡 Giải Pháp Kỹ Thuật</h4>", unsafe_allow_html=True)
            st.markdown("""
            *   **Hệ thống Đề xuất Lọc 2 Giai Đoạn**:
                *   *Giai đoạn 1 (Lọc cứng NLP)*: Loại bỏ nhiễu tuyệt đối bằng nhận diện ngữ nghĩa (AND-logic) cho phân khúc sao, loại hình lưu trú và tiện ích (Hồ bơi, Buffet, Gần biển...).
                *   *Giai đoạn 2 (Xếp hạng Lai)*: Kết hợp Content-based (Cosine) và Collaborative Filtering (SVD/ALS) để cá nhân hóa kết quả.
                *   *User Profiling*: Tự động điều chỉnh trọng số khía cạnh (Clean, Loc, Staff...) dựa trên gu đặc trưng của từng quốc tịch.
            *   **Hệ thống Insights Đối tác**:
                *   Phân tích cảm xúc & trích xuất cụm từ hành động (Aspect Sentiment Analysis).
                *   Biểu đồ Radar so sánh năng lực cạnh tranh thực tế so với trung bình Nha Trang.
            """)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 4. Tiêu chí thành công
    with st.container(border=True):
        st.markdown("### 📊 Tiêu Chí Đo Lường Thành Công")
        m1, m2, m3 = st.columns(3)
        m1.metric("🎯 Độ Phủ Đề Xuất (Coverage)", "> 90%", "Toàn bộ KS Nha Trang")
        m2.metric("🎯 Độ Chính Xác (Precision@5)", ">= 85%", "Đánh giá mô hình Hybrid")
        m3.metric("🔄 Độ Bao Phủ (Recall@5)", ">= 75%", "Gợi ý trúng nhu cầu")
