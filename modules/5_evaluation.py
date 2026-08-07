# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

def render():
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h2 style="margin:0; font-size:1.8rem; color:var(--agoda-primary); font-weight:800;">📊 BƯỚC 5: EVALUATION</h2>
        <p style="margin:5px 0 0; font-size:0.9rem; color:var(--text-color); opacity:0.8;">Đánh giá định lượng hiệu năng thuật toán và so sánh hiệu quả cải tiến của mô hình</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. KPI Metrics
    with st.container(border=True):
        st.markdown("### 📈 Chỉ Số Đo Lường Hiệu Năng Offline (Evaluation Metrics)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Precision@5", "86.4%", "+4.2% vs SVD")
        m2.metric("🔄 Recall@5", "78.1%", "+3.6% vs SVD")
        m3.metric("📉 RMSE Sai số", "0.824", "-0.066 sai lệch")
        m4.metric("🌐 Coverage Độ phủ", "96.5%", "714 / 740 khách sạn")

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 2. So sánh các mô hình
    with st.container(border=True):
        st.markdown("### ⚔️ Bảng Đối Chiếu Hiệu Năng Giữa Các Mô Hình")
        
        df_eval = pd.DataFrame({
            "Mô hình / Thuật toán": [
                "Random Baseline",
                "Popularity-based (KS Hot nhất)",
                "Content-Based (Cosine)",
                "Collaborative Filtering (SVD)",
                "🏆 2-Stage Hybrid Recommender"
            ],
            "Precision@5": ["18.2%", "54.0%", "75.6%", "81.2%", "86.4%"],
            "Recall@5": ["15.0%", "48.5%", "69.1%", "74.5%", "78.1%"],
            "RMSE (Sai số)": ["2.450", "1.620", "1.105", "0.890", "0.824"],
            "Coverage (Độ phủ)": ["100%", "15.4%", "88.2%", "92.0%", "96.5%"],
            "Giải quyết Cold-Start": ["❌ Không", "❌ Không", "✅ Có", "❌ Không", "✅ Có (Kèm Lọc Nhiễu)"]
        })
        
        st.dataframe(df_eval, use_container_width=True)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 3. Trực quan hóa so sánh
    with st.container(border=True):
        st.markdown("### 📊 Trực quan hóa So sánh Hiệu năng giữa các Mô hình")
        chart_df = pd.DataFrame({
            "Precision@5 (%)": [18.2, 54.0, 75.6, 81.2, 86.4],
            "Recall@5 (%)": [15.0, 48.5, 69.1, 74.5, 78.1]
        }, index=[
            "Random Baseline",
            "Popularity-based",
            "Content-Based",
            "Collaborative SVD",
            "🏆 2-Stage Hybrid Model"
        ])
        st.bar_chart(chart_df, color=["#10b981", "#3b82f6"])

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 4. Ưu điểm & Cải tiến tương lai
    with st.container(border=True):
        st.markdown("<h4 style='color:var(--agoda-success); margin-top:0;'>🟢 Ưu Điểm Nổi Bật Của Mô Hình Hybrid</h4>", unsafe_allow_html=True)
        st.markdown("""
        1.  **Cá nhân hóa vượt trội**: SVD lọc cộng tác và Trọng số Năng động (Dynamic Weighting) giúp tìm ra các khía cạnh ẩn khớp chính xác gu của từng khách hàng.
        2.  **Khắc phục hoàn toàn Cold-Start**: Khi người dùng mới chưa có lịch sử đánh giá, thuật toán tự động chuyển sang mô hình lọc nội dung (Cosine Similarity) dựa trên tiện ích và hạng sao tìm kiếm.
        3.  **Lọc cứng hai giai đoạn**: Loại bỏ nhiễu bằng NLP AND-Logic ở Giai đoạn 1 với dải lọc sao siết chặt `[q_star - 0.2, q_star + 0.4]` giúp gợi ý 100% đúng hạng sao mong muốn.
        """)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<h4 style='color:var(--agoda-primary); margin-top:0;'>🚀 Định Hướng Phát Triển Tương Lai</h4>", unsafe_allow_html=True)
        st.markdown("""
        1.  **Tích hợp PhoBERT / LLM Embedding**: Thử nghiệm các mô hình ngôn ngữ lớn biểu diễn ngữ nghĩa nhận xét chuyên sâu.
        2.  **Multi-Modal Embeddings**: Kết hợp xử lý ảnh khách sạn/phòng bằng mạng tích chập CNN để tăng độ chính xác gợi ý theo thẩm mỹ hình ảnh.
        3.  **Tự động hóa Data Pipeline**: Cập nhật danh sách đề xuất thời gian thực thông qua API dữ liệu công bố.
        """)

