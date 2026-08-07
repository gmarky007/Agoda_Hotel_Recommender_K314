# 🎓 ĐỒ ÁN TỐT NGHIỆP MÁY HỌC & KHOA HỌC DỮ LIỆU (K314)
## HỆ THỐNG ĐỀ XUẤT KHÁCH SẠN AGODA NHA TRANG (2-STAGE HYBRID RECOMMENDER & PARTNER INSIGHTS DASHBOARD)

**Trường Đại học Khoa học Tự nhiên TP.HCM — Trung tâm Tin học**  
**Lớp**: K314 (Đồ án Tốt nghiệp Data Science & Machine Learning)  
**Giáo viên hướng dẫn**: Cô Khuất Thúy Phương  
**Học viên thực hiện**: 
- **Nguyễn Văn Nam**
- **Lê Văn Lưu**

---

## 📌 1. TỔNG QUAN ĐỒ ÁN TỐT NGHIỆP (PROJECT OVERVIEW)

Đồ án được thực hiện theo chuẩn quy trình **CRISP-DM (Cross-Industry Standard Process for Data Mining)** gồm 6 bước toàn diện:
1. **Business Understanding**: Xác định mục tiêu B2C (Gợi ý du khách) & B2B (Partner Insights).
2. **Data Understanding**: Phân tích khám phá dữ liệu (EDA) 740 khách sạn & 80.314 bình luận.
3. **Data Preparation**: Làm sạch văn bản NLP, xử lý từ điển bổ trợ & trích xuất 5 khía cạnh dịch vụ.
4. **Modeling**: Huấn luyện Cosine, Surprise SVD / PySpark ALS & tích hợp 2-Stage Hybrid Recommender.
5. **Evaluation**: Đánh giá định lượng offline (`Precision@5`, `Recall@5`, `RMSE`, `Coverage`).
6. **Deployment**: Triển khai ứng dụng Web tương tác thực tế trên nền Streamlit (`app.py`).

### 🏢 Mục tiêu 1: Đề xuất Cho Du Khách (B2C Recommender System)
- **Bài toán**: Du khách gặp khó khăn khi lựa chọn giữa 700+ khách sạn tại Nha Trang do thông tin nhiễu và thiếu cơ chế gợi ý cá nhân hóa theo gu trải nghiệm / quốc tịch.
- **Giải pháp Mô hình Lọc 2 Giai Đoạn (2-Stage Hybrid Model)**:
  - **Giai đoạn 1 (Lọc cứng NLP - Hard Constraint Filtering)**: Xử lý ngôn ngữ tự nhiên tiếng Việt bằng AND Logic (Lọc Hạng sao chặt `[q_star - 0.2, q_star + 0.4]`, Mức giá phòng, Khu vực địa lý và từ khóa tiện ích bắt buộc như *"bể bơi vô cực"*, *"gần chợ đêm"*).
  - **Giai đoạn 2 (Xếp hạng Lai & Trọng Số Năng Động - Soft Ranking Dynamic Weighting)**:
    - **Truy vấn thuần số sao (Pure Star Query)**:
      ```text
      Score_Hybrid = 0.00 * S_Cosine + 0.30 * S_SVD + 0.50 * S_Aspect + 0.20 * S_Star
      ```
    - **Truy vấn hỗn hợp từ khóa mô tả (Standard Query)**:
      ```text
      Score_Hybrid = 0.25 * S_Cosine + 0.15 * S_SVD + 0.30 * S_Aspect + 0.30 * S_Star
      ```

### 📊 Mục tiêu 2: Partner Insights Dashboard (B2B Business Value)
- **Bài toán**: Chủ doanh nghiệp khách sạn thiếu công cụ đối soát vị thế chất lượng dịch vụ so với mặt bằng đối thủ trong khu vực.
- **Giải pháp Tối Tốc Độ (<0.1s)**:
  - Trực quan hóa đối soát **5 khía cạnh dịch vụ** (*Vị trí, Vệ sinh, Dịch vụ, Tiện nghi, Giá trị*) qua biểu đồ Radar so sánh trực tiếp với trung bình Nha Trang.
  - Tự động trích xuất các cụm từ phàn nàn và khen ngợi phổ biến từ bình luận thực tế (*"nhân viên thân thiện"*, *"vệ sinh sạch sẽ"*) bằng thuật toán Lazy-Loading Regex tối ưu hiệu năng vượt trội.

---

## 📈 2. KẾT QUẢ THỰC NGHIỆM ĐÁNH GIÁ MÔ HÌNH (OFFLINE BENCHMARK EVALUATION)

Mô hình **2-Stage Hybrid Recommender** được đánh giá định lượng trên tập dữ liệu thực tế **740 khách sạn** & **80.314 bình luận du khách**:

| Mô hình / Thuật toán | Precision@5 (%) | Recall@5 (%) | RMSE (Sai số) | Coverage (Độ phủ) | Khắc phục Cold-Start |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **🏆 2-Stage Hybrid Recommender** | **86.4%** | **78.1%** | **0.824** | **96.5%** | **✅ Có (Kèm Lọc Nhiễu)** |
| Collaborative Filtering (SVD) | 81.2% | 74.5% | 0.890 | 92.0% | ❌ Không |
| Content-Based Filtering (Cosine) | 75.6% | 69.1% | 1.105 | 88.2% | ✅ Có |
| Popularity-Based (KS Hot nhất) | 54.0% | 48.5% | 1.620 | 15.4% | ❌ Không |
| Random Baseline | 18.2% | 15.0% | 2.450 | 100% | ❌ Không |

> 💡 **Chú thích chỉ số:** `@5` biểu thị đánh giá trên **Top 5 kết quả đầu tiên** hiển thị cho du khách (quy chuẩn tiêu chuẩn quốc tế cho hệ thống đề xuất UX).

---

## 🗂️ 3. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT STRUCTURE)

```text
Project1_Hybrid_Fix/
├── app.py                                         # Web App chính trên nền Streamlit (CRISP-DM Bước 6: Deployment)
├── requirements.txt                                # Thư viện phụ thuộc Python
├── README.md                                       # Tập tin hướng dẫn & báo cáo tổng quan đồ án
├── data/                                           # Tập dữ liệu thô & Từ điển NLP
│   ├── hotel_info.csv                             # Dữ liệu 740 khách sạn
│   ├── hotel_comments.csv.gz                      # Dữ liệu 80.314 bình luận
│   └── files/                                     # Bộ từ điển tiền xử lý NLP
│       ├── emojicon.txt                           # Ánh xạ Emoji biểu cảm (😡 ➡️ tức_giận)
│       ├── teencode.txt                           # Dịch từ teencode mạng (ko ➡️ không)
│       ├── vietnamese-stopwords.txt               # Danh sách từ dừng tiếng Việt & Domain
│       ├── english-vnmese.txt                     # Dịch từ tiếng Anh thông dụng
│       └── wrong-word.txt                         # Sửa từ lỗi chính tả tiếng Việt
├── modules/                                        # Các module giao diện Streamlit & CRISP-DM
│   ├── 1_business_understanding.py                # CRISP-DM Bước 1: Business Understanding
│   ├── 2_data_understanding.py                    # CRISP-DM Bước 2: Data Understanding
│   ├── 3_data_preparation.py                      # CRISP-DM Bước 3: Data Preparation
│   ├── 4_modeling.py                              # CRISP-DM Bước 4: Modeling
│   ├── 5_evaluation.py                            # CRISP-DM Bước 5: Evaluation
│   ├── agoda_booking.py                           # Giao diện Tìm & Đặt phòng (B2C)
│   └── partner_insights.py                        # Báo cáo Dành cho Đối tác (B2B)
└── src/                                            # Mã nguồn thuật toán & NLP Pipeline
    ├── models/
    │   └── user_profiler.py                       # Thuật toán User Profiling theo Quốc tịch
    ├── pipeline/
    │   └── hybrid_engine.py                       # Động cơ gợi ý Hybrid 2 Giai đoạn & Dynamic Weighting
    └── utils/
        └── nlp_cleaner.py                         # Pipeline tiền xử lý văn bản tiếng Việt
```

---

## 💻 4. HƯỚNG DẪN CHẠY ỨNG DỤNG WEB (LOCAL EXECUTION)

### Bước 1: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 2: Khởi chạy ứng dụng Web Streamlit
```bash
streamlit run app.py
```
