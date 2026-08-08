# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style matching the notebook
sns.set_theme(style="whitegrid")

def render():
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h2 style="margin:0; font-size:1.8rem; color:var(--agoda-primary); font-weight:800;">🤖 BƯỚC 4: MODELING</h2>
        <p style="margin:5px 0 0; font-size:0.9rem; color:var(--text-color); opacity:0.8;">Xây dựng và kết hợp các kiến trúc thuật toán đề xuất tiên tiến (Hybrid Recommender Architecture)</p>
    </div>
    """, unsafe_allow_html=True)

    # 4.1 & 4.2 Comparison Table
    with st.container(border=True):
        st.markdown("### 🔍 4.1 & 4.2. So Sánh Thực Nghiệm Giữa Scikit-Learn Cosine Và Gensim")
        st.markdown("""
        Mặc dù cả hai phương pháp đều dựa trên độ đo tương đồng **Cosine Similarity** và đặc trưng tần suất từ **TF-IDF**, nhưng do sự khác biệt trong cơ chế xử lý văn bản (Tokenization), chúng ta tiến hành đánh giá thực nghiệm trên **100 khách sạn mẫu** để so sánh hiệu năng.
        """)
        
        # Table of metrics
        metrics_data = {
            "Chỉ số đánh giá": [
                "Độ chính xác phân khúc (star_diff) ↓",
                "Độ bao phủ sản phẩm (coverage) ↑",
                "Độ khớp phân khúc (same_type) ↑"
            ],
            "Scikit-Learn Cosine": [
                "0.8710 sao (Tốt hơn)",
                "37.16%",
                "59.60% (Tốt hơn)"
            ],
            "Gensim (Word2Vec/TF-IDF)": [
                "0.9783 sao",
                "39.05% (Tốt hơn)",
                "56.20%"
            ]
        }
        st.table(pd.DataFrame(metrics_data).set_index("Chỉ số đánh giá"))
        
        st.markdown("""
        **Nhận định so sánh chuyên sâu (Business Interpretation):**
        1. **Về độ chính xác phân khúc (star_diff):** Scikit-Learn đạt độ lệch sao nhỏ hơn (`0.8710` so với `0.9783` của Gensim), điều này cho thấy các khách sạn được gợi ý bởi Scikit-Learn có xu hướng sát về hạng sao với khách sạn gốc hơn.
        2. **Về độ bao phủ sản phẩm (coverage):** Gensim vượt trội hơn với độ bao phủ đạt `39.05%` (so với `37.16%` của Scikit-Learn). Điều này có nghĩa là Gensim phân phối lượt gợi ý đều hơn, tránh hiện tượng chỉ tập trung vào một vài khách sạn phổ biến, rất tốt cho mục tiêu đa dạng hóa sản phẩm của Agoda.
        3. **Về độ khớp phân khúc (same_type):** Cosine (Scikit-Learn) đạt `59.60%` so với `56.20%` của Gensim, giữ đúng phân khúc Premium và Budget tốt hơn.
        4. **Lý do kỹ thuật:** Thư viện **Gensim** xử lý văn bản theo chuỗi từ được tokenize có bảo lưu các cụm từ ghép tiếng Việt (ví dụ: `sát_biển`, `phục_vụ_nhiệt_tình`), giúp các đặc trưng ngữ nghĩa mang đậm văn phong tiếng Việt hơn so với bộ Tokenizer mặc định.
        """)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 4.3 & 4.4 Collaborative Filtering
    # 4.3 Collaborative Filtering SVD
    with st.container(border=True):
        st.markdown("<h4 style='color:var(--agoda-primary); margin-top:0;'>🤖 4.3. Lọc Cộng Tác Surprise SVD (Single Node)</h4>", unsafe_allow_html=True)
        st.markdown("""
        Bên cạnh giải pháp xử lý song song trên dữ liệu lớn PySpark ALS, chúng ta xây dựng mô hình **Surprise SVD** (Local SVD) làm mô hình gợi ý chính cho môi trường máy đơn. Mô hình này học các nhân tố ẩn (Latent Factors) từ ma trận tương tác của người dùng và khách sạn.
        """)
        st.latex(r"\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u")
        st.markdown("""
        *   **μ**: Điểm trung bình toàn cục.
        *   **b_u**: Sai số của người dùng u.
        *   **b_i**: Sai số của khách sạn i.
        *   **q_i, p_u**: Vector đặc trưng ẩn của khách sạn và người dùng.
        """)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 4.4 PySpark ALS
    with st.container(border=True):
        st.markdown("<h4 style='color:var(--agoda-primary); margin-top:0;'>⚡ 4.4. Lọc Cộng Tác PySpark ALS (Big Data)</h4>", unsafe_allow_html=True)
        st.markdown("""
        Mô hình **Collaborative Filtering** phân tán sử dụng thuật toán **ALS (Alternating Least Squares)** trong hệ sinh thái **Apache Spark (PySpark)** được thiết kế để xử lý song song trên các tập dữ liệu có quy mô cực lớn (Big Data).
        """)
        st.markdown("""
        **Quy trình thực thi gồm các bước:**
        1. **Mã hóa ID (`StringIndexer`):** Chuyển đổi mã định danh dạng chuỗi chữ (`Reviewer ID`, `Hotel ID`) thành chỉ số số nguyên (`user_idx`, `item_idx`) để đưa vào thuật toán.
        2. **Huấn luyện mô hình:** Sử dụng thuật toán ALS tối ưu hóa luân phiên với tham số `coldStartStrategy="drop"` để tự động bỏ qua các lỗi khởi đầu lạnh trong quá trình đánh giá chất lượng mô hình.
        3. **Giải mã ngược kết quả gợi ý:** Chuyển đổi các ID dạng số sau khi dự đoán ngược lại thành Tên khách sạn và Địa chỉ thực tế để hiển thị kết quả trực quan.
        """)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 4.5 Hybrid Dynamic Weighting
    with st.container(border=True):
        st.markdown("### 🏗️ 4.5. 2-Stage Hybrid Recommender (Mô Hình Lọc 2 Giai Đoạn & Dynamic Weighting)")
        st.markdown("""
        Mô hình Gợi ý Cải tiến chia làm 2 giai đoạn (2-Stage Pipeline) để đảm bảo độ chính xác tuyệt đối về logic tập hợp, đồng thời cá nhân hóa tối đa.
        """)
        
        st.markdown("""
        **Giai đoạn 1: Lọc cứng bằng NLP (Hard Constraint Filter - AND Logic)**
        - **Số sao**: Ví dụ `khách sạn 3 sao` ➡️ Lọc cứng khoảng sao chặt `[q_star - 0.2, q_star + 0.4]` (Chỉ giữ đúng khách sạn 3.0 sao).
        - **Loại hình**: Nhận diện `biệt thự`, `resort`, `căn hộ`, `homestay`.
        - **Tiện ích**: Bắt buộc giao tập hợp `AND`. (Ví dụ: phải có cả `hồ bơi` VÀ `bữa sáng`).
        """)

        st.markdown("""
        **Giai đoạn 2: Tính điểm Hybrid Soft Ranking & Trọng Số Năng Động (Dynamic Weighting)**
        - Tự động nhận diện kiểu truy vấn để chuyển dịch bộ trọng số tối ưu:
        """)

        st.markdown("1️⃣ **Truy vấn thuần số sao (Pure Star Query - ví dụ: *khách sạn 3 sao*):**")
        st.latex(r"\text{Score}_{\text{Hybrid}} = 0.00 \cdot S_{\text{Cosine}} + 0.30 \cdot S_{\text{SVD}} + 0.50 \cdot S_{\text{Aspect}} + 0.20 \cdot S_{\text{Star}}")
        st.caption("💡 *Tắt Cosine (0%) để tránh nhiễu từ khóa, đẩy trọng số Aspect (50%) và SVD (30%) giúp phân tách điểm số hoàn hảo, không bị đồng điểm.*")

        st.markdown("2️⃣ **Truy vấn hỗn hợp kèm từ khóa mô tả (Standard Query):**")
        st.latex(r"\text{Score}_{\text{Hybrid}} = 0.25 \cdot S_{\text{Cosine}} + 0.15 \cdot S_{\text{SVD}} + 0.30 \cdot S_{\text{Aspect}} + 0.30 \cdot S_{\text{Star}}")

        # Draw a beautiful Seaborn bar chart for hybrid weights
        st.markdown("#### 📊 Phân bổ trọng số của mô hình Hybrid (Truy Vấn Chuẩn vs Thuần Số Sao)")
        fig, ax = plt.subplots(figsize=(10, 3.8))
        df_weights = pd.DataFrame({
            "Thành phần": ["Cosine Text", "SVD / ALS", "Aspect Match", "Star Match"],
            "Truy Vấn Chuẩn (%)": [25, 15, 30, 30],
            "Truy Vấn Thuần Sao (%)": [0, 30, 50, 20]
        }).melt(id_vars="Thành phần", var_name="Kịch Bản", value_name="Trọng Số (%)")

        sns.barplot(data=df_weights, x="Trọng Số (%)", y="Thành phần", hue="Kịch Bản", palette="coolwarm", ax=ax)
        ax.set_title('Bộ Trọng Số Năng Động (Dynamic Weighting) Trong Mô Hình Hybrid', fontsize=11, fontweight='bold')
        ax.set_xlabel('Trọng số (%)', fontsize=9)
        ax.set_xlim(0, 60)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 3. Code snippet
    with st.container(border=True):
        st.markdown("### 💻 Triển khai Mã Nguồn Huấn Luyện (Python Hybrid Engine)")
        st.code("""
# src/pipeline/hybrid_engine.py

# [GIAI ĐOẠN 1] Lọc Cứng (Hard Constraints)
def parse_nlp_query_constraints(df, query):
    # Lọc số sao, loại hình, và giao tập hợp AND tiện ích
    return filtered_df

# [GIAI ĐOẠN 2] Tính điểm Hybrid
def calculate_hybrid_scores(filtered_df, df_aspects, svd_model, reviewer_profile, search_query="", cosine_sim=None):
    res_df['Hybrid_Score'] = (
        0.25 * res_df['Cosine_Score'] +
        0.05 * res_df['SVD_Norm'] +
        0.15 * res_df['Aspect_Match_Score'] +
        0.55 * res_df['Star_Match_Score']
    ).round(4)
    
    return res_df.sort_values(by='Hybrid_Score', ascending=False)
        """, language="python")

    st.success("✅ Mô hình 2-Stage Hybrid Recommender đã giải quyết triệt để vấn đề nhiễu logic tập hợp và tối ưu hiệu suất gợi ý.")
