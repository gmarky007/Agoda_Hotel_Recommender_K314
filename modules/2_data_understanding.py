# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set style matching the notebook
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

@st.cache_data
def load_eda_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    info_path = os.path.join(base_dir, 'data', 'hotel_info.csv')
    comments_path = os.path.join(base_dir, 'data', 'hotel_comments.csv.gz')
    comments_path_csv = os.path.join(base_dir, 'data', 'hotel_comments.csv')
    
    df_h = pd.read_csv(info_path) if os.path.exists(info_path) else pd.DataFrame()
    df_c = pd.DataFrame()
    if os.path.exists(comments_path):
        df_c = pd.read_csv(comments_path)
    elif os.path.exists(comments_path_csv):
        df_c = pd.read_csv(comments_path_csv)
        
    return df_h, df_c

def extract_province(addr):
    if pd.isna(addr):
        return "Chưa rõ"
    addr_str = str(addr)
    if "Cam Ranh" in addr_str or "Cam Nghĩa" in addr_str or "Cam Linh" in addr_str:
        return "Cam Ranh"
    if "Cam Lâm" in addr_str or "Cam Lam" in addr_str or "Cam Hải Đông" in addr_str:
        return "Cam Lâm / Bãi Dài"
    if "Nha Trang" in addr_str:
        return "Nha Trang"
    return "Khác"

def render():
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h2 style="margin:0; font-size:1.8rem; color:var(--agoda-primary); font-weight:800;">🔍 BƯỚC 2: DATA UNDERSTANDING</h2>
        <p style="margin:5px 0 0; font-size:0.9rem; color:var(--text-color); opacity:0.8;">Khám phá cấu trúc, phân bố và chất lượng dữ liệu theo đúng mô hình phân tích EDA trong notebook</p>
    </div>
    """, unsafe_allow_html=True)

    df_hotels, df_comments = load_eda_data()

    # 📊 KPI Tổng quan
    with st.container(border=True):
        st.markdown("### 📊 Tổng Thể Bộ Dữ Liệu Agoda Nha Trang")
        col1, col2, col3 = st.columns(3)
        if not df_hotels.empty:
            total_hotels = len(df_hotels)
            if 'Total_Score' in df_hotels.columns:
                scores_parsed = pd.to_numeric(df_hotels['Total_Score'].astype(str).str.replace(',', '.'), errors='coerce')
                avg_score = round(scores_parsed.mean(), 2)
            else:
                avg_score = 8.5
            col1.metric("🏢 Tổng số Khách sạn", f"{total_hotels:,} cơ sở")
            col2.metric("⭐ Điểm TB Khách sạn", f"{avg_score} / 10", "Thang điểm Agoda")
        else:
            col1.metric("🏢 Tổng số Khách sạn", "740 cơ sở")
            col2.metric("⭐ Điểm TB Khách sạn", "8.5 / 10")

        if not df_comments.empty:
            col3.metric("💬 Tổng số Đánh giá (Reviews)", f"{len(df_comments):,} bình luận")
        else:
            col3.metric("💬 Tổng số Đánh giá", "80,314 bình luận")

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 📋 Bảng Thuộc tính Dữ liệu Gốc
    with st.container(border=True):
        st.markdown("### 📋 Cấu Trúc Dữ Liệu Gốc (Data Schema)")
        t1, t2 = st.tabs(["🏨 Dữ liệu Khách sạn (hotel_info.csv)", "💬 Dữ liệu Đánh giá (hotel_comments.csv.gz)"])

        with t1:
            st.markdown("""
            | Tên trường | Kiểu dữ liệu | Mô tả |
            |---|---|---|
            | `Hotel_ID` | String | Mã định danh duy nhất của khách sạn |
            | `Hotel_Name` | String | Tên khách sạn hiển thị |
            | `Hotel_Rank` | String | Hạng sao khách sạn (ví dụ: "5 sao trên 5") |
            | `Total_Score` | String | Điểm đánh giá trung bình (ví dụ: "8,8") |
            | `Hotel_Address` | String | Địa chỉ chi tiết tại Nha Trang |
            | `Hotel_Description` | String | Mô tả đặc trưng tiện ích phòng |
            """)
            if not df_hotels.empty:
                st.caption("Xem trước 5 dòng dữ liệu khách sạn gốc:")
                cols_to_show = [c for c in ['Hotel_ID', 'Hotel_Name', 'Hotel_Rank', 'Total_Score', 'Hotel_Address'] if c in df_hotels.columns]
                st.dataframe(df_hotels[cols_to_show].head(5), use_container_width=True)

        with t2:
            st.markdown("""
            | Tên trường | Kiểu dữ liệu | Mô tả |
            |---|---|---|
            | `Hotel ID` | String | Liên kết tới khách sạn tương ứng |
            | `Reviewer Name` | String | Tên người đánh giá |
            | `Nationality` | String | Quốc tịch khách du lịch |
            | `Group Name` | String | Nhóm du khách (Gia đình, Cặp đôi,...) |
            | `Score` | Float | Điểm số đánh giá cá nhân (thang điểm 10) |
            | `Body` | String | Nội dung bình luận nhận xét chi tiết |
            | `Review Date` | String | Thời gian viết đánh giá |
            """)
            if not df_comments.empty:
                st.caption("Xem trước 5 dòng dữ liệu bình luận gốc:")
                cols_to_show = [c for c in ['Hotel ID', 'Reviewer Name', 'Nationality', 'Group Name', 'Score', 'Review Date'] if c in df_comments.columns]
                st.dataframe(df_comments[cols_to_show].head(5), use_container_width=True)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 📊 Phân tích trực quan hóa EDA tương tác
    with st.container(border=True):
        st.markdown("### 📊 Trực Quan Hóa Khám Phá Dữ Liệu (EDA Dashboard)")
        
        sub_t1, sub_t2 = st.tabs(["🏨 Phân tích Khách sạn (Hotel EDA)", "💬 Phân tích Bình luận & Khách hàng (User EDA)"])

        with sub_t1:
            if not df_hotels.empty:
                from src.pipeline.hybrid_engine import parse_star_numeric
                df_hotels['Star_Rating'] = [parse_star_numeric(r['Hotel_Rank'], r.get('Hotel_Description', '')) for _, r in df_hotels.iterrows()]
                
                # Parse scores
                score_cols = ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money', 'Comfort_and_room_quality']
                for col in score_cols + ['Total_Score']:
                    if col in df_hotels.columns:
                        df_hotels[col] = pd.to_numeric(df_hotels[col].astype(str).str.replace(',', '.'), errors='coerce')

                # Đồ thị 1: Tỷ lệ khuyết thiếu
                with st.container(border=True):
                    st.markdown("#### 1️⃣ Tỷ lệ khuyết thiếu dữ liệu điểm số (%)")
                    missing_pct = (df_hotels[[c for c in score_cols if c in df_hotels.columns]].isnull().sum() / len(df_hotels) * 100).round(2)
                    fig1, ax1 = plt.subplots(figsize=(10, 3.5))
                    sns.barplot(x=missing_pct.values, y=missing_pct.index, ax=ax1, palette='Reds_r')
                    ax1.set_title('Tỷ lệ dữ liệu khuyết thiếu của các cột điểm trong hotel_info (%)', fontsize=11, fontweight='bold')
                    ax1.set_xlabel('Tỷ lệ %', fontsize=9)
                    ax1.set_xlim(0, 100)
                    for i, v in enumerate(missing_pct.values):
                        ax1.text(v + 1, i, f"{v}%", va='center', fontweight='bold', fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close(fig1)
                    st.info("💡 **Nhận xét**: Cột *Comfort_and_room_quality* bị thiếu tới **93% dữ liệu** (chỉ có 51/740 khách sạn ghi nhận). Để tối ưu hóa tài nguyên và độ chính xác, cột này được loại bỏ khỏi bước xây dựng mô hình.")

                # Đồ thị 2: Phân phối Total_Score
                with st.container(border=True):
                    st.markdown("#### 2️⃣ Phân phối điểm đánh giá trung bình (Total_Score)")
                    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
                    sns.histplot(df_hotels['Total_Score'].dropna(), kde=True, bins=20, ax=ax2, color='skyblue')
                    ax2.set_title('Phân phối điểm đánh giá trung bình (Total_Score) của các khách sạn', fontsize=11, fontweight='bold')
                    ax2.set_xlabel('Điểm đánh giá', fontsize=9)
                    ax2.set_ylabel('Số lượng khách sạn', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
                    st.info("💡 **Nhận xét**: Điểm đánh giá tập trung cao nhất ở nhóm **8.0 - 9.0**. Đây là phân khúc dịch vụ đạt chuẩn khá trở lên chiếm đại đa số tại Nha Trang.")

                # Đồ thị 3: Phân bộ xếp hạng sao
                with st.container(border=True):
                    st.markdown("#### 3️⃣ Phân bộ xếp hạng sao (Star Rating)")
                    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
                    sns.countplot(data=df_hotels, x='Star_Rating', ax=ax3, palette='YlOrBr')
                    for container in ax3.containers:
                        ax3.bar_label(container, fmt='%.0f', fontweight='bold', padding=3, fontsize=8)
                    ax3.set_title('Phân bộ xếp hạng sao (Star Rating) của các khách sạn', fontsize=11, fontweight='bold')
                    ax3.set_xlabel('Hạng sao', fontsize=9)
                    ax3.set_ylabel('Số lượng khách sạn', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close(fig3)

                # Đồ thị 4: Phân bộ địa lý
                with st.container(border=True):
                    st.markdown("#### 4️⃣ Phân bộ theo khu vực địa lý thực tế")
                    df_hotels['Region'] = df_hotels['Hotel_Address'].apply(extract_province)
                    region_counts = df_hotels['Region'].value_counts()
                    fig4, ax4 = plt.subplots(figsize=(10, 3.5))
                    sns.barplot(x=region_counts.values, y=region_counts.index, palette='viridis', ax=ax4)
                    ax4.set_title('Phân bộ số lượng khách sạn theo khu vực địa lý thực tế', fontsize=11, fontweight='bold')
                    ax4.set_xlabel('Số lượng khách sạn', fontsize=9)
                    ax4.set_ylabel('Khu vực', fontsize=9)
                    for i, v in enumerate(region_counts.values):
                        ax4.text(v + 1, i, str(v), va='center', fontweight='bold', fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close(fig4)

                # Đồ thị Boxplot: Mối liên hệ Hạng sao & Điểm số
                with st.container(border=True):
                    st.markdown("#### 5️⃣ Mối liên hệ giữa Nhóm Hạng sao và Điểm đánh giá trung bình")
                    def group_stars(stars):
                        if pd.isna(stars):
                            return "Chưa rõ"
                        if stars <= 2.5:
                            return "1.0 - 2.5 sao"
                        if stars <= 3.5:
                            return "3.0 - 3.5 sao"
                        if stars <= 4.5:
                            return "4.0 - 4.5 sao"
                        return "5.0 sao"
                    
                    df_hotels['comments_count'] = df_hotels.get('comments_count', 10)
                    df_active_hotels = df_hotels[df_hotels['Total_Score'].notna()].copy()
                    df_active_hotels['Star_Group'] = df_active_hotels['Star_Rating'].apply(group_stars)
                    
                    fig5, ax5 = plt.subplots(figsize=(10, 3.8))
                    sns.boxplot(data=df_active_hotels, x='Star_Group', y='Total_Score',
                                order=['1.0 - 2.5 sao', '3.0 - 3.5 sao', '4.0 - 4.5 sao', '5.0 sao'],
                                palette='Set3', ax=ax5)
                    ax5.set_title('Mối liên hệ giữa Nhóm Hạng sao và Điểm đánh giá trung bình (Active Hotels)', fontsize=11, fontweight='bold')
                    ax5.set_xlabel('Nhóm hạng sao', fontsize=9)
                    ax5.set_ylabel('Điểm đánh giá trung bình', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig5)
                    plt.close(fig5)
                    st.info("💡 **Nhận xét**: Khách sạn từ 4 sao trở lên có độ ổn định điểm số rất cao (tập trung hẹp trên 8.5). Nhóm khách sạn hạng thấp từ 1-2.5 sao có dải điểm phân tán rộng, phản ánh sự không đồng đều về chất lượng phòng dịch vụ.")

                # Đồ thị 6: Ma trận tương quan
                with st.container(border=True):
                    st.markdown("#### 6️⃣ Ma trận tương quan giữa các tiêu chí đánh giá khách sạn")
                    valid_cols = [c for c in ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money'] if c in df_hotels.columns]
                    if len(valid_cols) > 1:
                        fig6, ax6 = plt.subplots(figsize=(10, 4.5))
                        corr = df_hotels[valid_cols].corr()
                        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax6)
                        ax6.set_title('Ma trận tương quan giữa các tiêu chí đánh giá khách sạn (Đã loại bỏ Comfort_and_room_quality bị thiếu 93%)', fontsize=11, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig6)
                        plt.close(fig6)

            else:
                st.warning("Không tìm thấy dữ liệu khách sạn để trực quan hóa.")

        with sub_t2:
            if not df_comments.empty:
                # Parse comment score
                df_comments['Score'] = pd.to_numeric(df_comments['Score'].astype(str).str.replace(',', '.'), errors='coerce')
                
                # Đồ thị 7: Phân phối điểm bình luận
                with st.container(border=True):
                    st.markdown("#### 1️⃣ Phân bộ điểm đánh giá của khách hàng trong bình luận (Comments)")
                    fig7, ax7 = plt.subplots(figsize=(10, 3.5))
                    sns.histplot(df_comments['Score'].dropna(), bins=20, color='skyblue', ax=ax7)
                    ax7.set_title('Phân bộ điểm đánh giá của khách hàng trong bình luận (Comments)', fontsize=11, fontweight='bold')
                    ax7.set_xlabel('Điểm đánh giá', fontsize=9)
                    ax7.set_ylabel('Tần suất bình luận', fontsize=9)
                    ax7.set_xlim(6, 10.2)
                    ax7.set_xticks([6.4, 6.8, 7.2, 7.6, 8.0, 8.4, 8.8, 9.2, 9.6, 10.0])
                    plt.tight_layout()
                    st.pyplot(fig7)
                    plt.close(fig7)
                    st.info("💡 **Nhận xét**: Phân phối điểm đánh giá của từng bình luận cho thấy lượng điểm tuyệt đối **10.0** và **8.0 / 8.4** xuất hiện vượt trội, biểu thị mức độ hài lòng chung cao.")

                # Đồ thị 8: Nhóm du khách
                with st.container(border=True):
                    st.markdown("#### 2️⃣ Cơ cấu Nhóm du khách trên Agoda Nha Trang")
                    if 'Group Name' in df_comments.columns:
                        group_counts = df_comments['Group Name'].value_counts().head(8)
                        fig8, ax8 = plt.subplots(figsize=(10, 3.8))
                        sns.barplot(x=group_counts.values, y=group_counts.index, palette='pastel', ax=ax8)
                        ax8.set_title('Phân bộ các nhóm du khách trên Agoda', fontsize=11, fontweight='bold')
                        ax8.set_xlabel('Số lượng bình luận', fontsize=9)
                        ax8.set_ylabel('Nhóm khách hàng', fontsize=9)
                        for i, v in enumerate(group_counts.values):
                            ax8.text(v + 100, i, str(v), va='center', fontsize=8, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig8)
                        plt.close(fig8)

                # Đồ thị 9: Quốc tịch du khách
                with st.container(border=True):
                    st.markdown("#### 3️⃣ Top 10 Quốc tịch của du khách")
                    if 'Nationality' in df_comments.columns:
                        nat_counts = df_comments['Nationality'].value_counts().head(10)
                        fig9, ax9 = plt.subplots(figsize=(10, 4.0))
                        sns.barplot(x=nat_counts.values, y=nat_counts.index, palette='magma', ax=ax9)
                        ax9.set_title('Top 10 Quốc tịch du khách đánh giá', fontsize=11, fontweight='bold')
                        ax9.set_xlabel('Số lượt đánh giá', fontsize=9)
                        ax9.set_ylabel('Quốc tịch', fontsize=9)
                        for i, v in enumerate(nat_counts.values):
                            ax9.text(v + 100, i, str(v), va='center', fontsize=8, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig9)
                        plt.close(fig9)

                # Đồ thị 10: Xu hướng nhận xét qua các năm
                with st.container(border=True):
                    st.markdown("#### 4️⃣ Xu hướng số lượng bình luận qua các năm (từ 2016)")
                    if 'Review Date' in df_comments.columns:
                        try:
                            df_comments['Review_Date_Parsed'] = pd.to_datetime(df_comments['Review Date'], errors='coerce')
                            df_comments_recent = df_comments[df_comments['Review_Date_Parsed'].dt.year >= 2016]
                            year_counts = df_comments_recent['Review_Date_Parsed'].dt.year.value_counts().sort_index()
                            
                            fig10, ax10 = plt.subplots(figsize=(10, 3.5))
                            sns.lineplot(x=year_counts.index.astype(int), y=year_counts.values, marker='o', color='red', linewidth=2.5, ax=ax10)
                            ax10.set_title('Xu hướng số lượng bình luận của khách hàng qua các năm (từ 2016)', fontsize=11, fontweight='bold')
                            ax10.set_xlabel('Năm', fontsize=9)
                            ax10.set_ylabel('Số lượng nhận xét', fontsize=9)
                            ax10.set_xticks(year_counts.index.astype(int))
                            for x, y in zip(year_counts.index.astype(int), year_counts.values):
                                ax10.text(x, y + 1000, str(int(y)), ha='center', fontweight='bold', fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig10)
                            plt.close(fig10)
                        except Exception as e:
                            st.caption(f"Không thể vẽ xu hướng nhận xét: {e}")
            else:
                st.warning("Không tìm thấy dữ liệu đánh giá để trực quan hóa.")
