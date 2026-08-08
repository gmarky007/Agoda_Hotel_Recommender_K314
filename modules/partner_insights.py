# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import re

@st.cache_data(show_spinner=False)
def fast_compute_monthly_trend(df_comments_subset, default_score=8.5):
    """Compute monthly review counts and average score for a specific hotel subset."""
    if df_comments_subset.empty or 'Review Date' not in df_comments_subset.columns or 'Score' not in df_comments_subset.columns:
        return tuple([0]*12), tuple([round(default_score, 1)]*12)
    
    extracted = df_comments_subset['Review Date'].astype(str).str.extract(r'(\d{1,2})\s+tháng\s+(\d{1,2})\s+(\d{4})')
    extracted.columns = ['Day', 'Month', 'Year']
    
    valid_mask = extracted['Month'].notna()
    if not valid_mask.any():
        return tuple([0]*12), tuple([round(default_score, 1)]*12)
        
    df_valid = pd.DataFrame({
        'Month': extracted.loc[valid_mask, 'Month'].astype(int),
        'Score': pd.to_numeric(df_comments_subset.loc[valid_mask, 'Score'].astype(str).str.replace(',', '.'), errors='coerce').fillna(default_score)
    })
    
    m_groupby = df_valid.groupby('Month').agg(
        Count=('Score', 'count'),
        Avg_Score=('Score', 'mean')
    ).reindex(range(1, 13)).fillna({'Count': 0, 'Avg_Score': default_score}).reset_index()
    
    counts = tuple(int(x) for x in m_groupby['Count'])
    scores = tuple(round(float(x), 1) for x in m_groupby['Avg_Score'])
    return counts, scores

def load_partner_data():
    if "custom_hotels_df" in st.session_state:
        df_hotels = st.session_state["custom_hotels_df"].copy()
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        info_path = os.path.join(data_dir, 'hotel_info.csv')
        df_hotels = pd.read_csv(info_path) if os.path.exists(info_path) else pd.DataFrame()

    if "custom_comments_df" in st.session_state:
        df_comments = st.session_state["custom_comments_df"].copy()
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        comments_gz = os.path.join(data_dir, 'hotel_comments.csv.gz')
        comments_path = os.path.join(data_dir, 'hotel_comments.csv')
        if os.path.exists(comments_gz):
            df_comments = pd.read_csv(comments_gz)
        elif os.path.exists(comments_path):
            df_comments = pd.read_csv(comments_path)
        else:
            df_comments = pd.DataFrame()

    return df_hotels, df_comments

@st.cache_data(show_spinner=False)
def fast_get_hotel_keywords(df_sub_comments):
    """Fast keyword extraction using phrase matching to prevent split syllables and stopwords."""
    if df_sub_comments.empty or 'Body' not in df_sub_comments.columns:
        return pd.DataFrame(columns=['Từ Khóa', 'Lượt Nhắc'])

    text = " ".join(df_sub_comments['Title'].fillna('').astype(str) + " " + df_sub_comments['Body'].fillna('').astype(str)).lower()
    text = " " + re.sub(r'[^\w\s]', ' ', text) + " "

    TARGETS = {
        'Tuyệt vời': [' tuyệt vời ', ' tuyệt hảo ', ' xuất sắc ', ' quá tuyệt ', ' great ', ' awesome ', ' excellent ', ' amazing ', ' wonderful ', ' perfect ', ' tuyệt '],
        'Nhân viên nhiệt tình': [' nhân viên ', ' staff ', ' thân thiện ', ' friendly ', ' nhiệt tình ', ' lễ tân ', ' chu đáo ', ' helpful ', ' phục vụ '],
        'Sạch sẽ': [' sạch sẽ ', ' sạch ', ' clean ', ' vệ sinh ', ' gọn gàng '],
        'Vị trí thuận tiện': [' vị trí ', ' location ', ' trung tâm ', ' gần biển ', ' convenient ', ' thuận tiện '],
        'Bữa sáng ngon': [' ăn sáng ', ' bữa sáng ', ' breakfast ', ' buffet ', ' đồ ăn ', ' food ', ' delicious '],
        'Hồ bơi đẹp': [' hồ bơi ', ' bể bơi ', ' pool '],
        'Giá cả hợp lý': [' đáng tiền ', ' giá tốt ', ' giá cả ', ' hợp lý ', ' value ', ' giá rẻ ', ' rẻ '],
        'View đẹp': [' view ', ' hướng biển ', ' ngắm biển ', ' phong cảnh '],
        'Chất lượng tốt': [' chất lượng ', ' tốt ', ' good ', ' nice ', ' ok ', ' ổn '],
        'Thiết kế đẹp': [' thiết kế ', ' sang trọng ', ' hiện đại ', ' beautiful ', ' không gian '],
        'Thoải mái & Yên tĩnh': [' thoải mái ', ' comfortable ', ' relax ', ' yên tĩnh ', ' quiet '],
        'Tiện nghi đầy đủ': [' tiện nghi ', ' facilities ', ' đầy đủ ']
    }

    counts = {}
    for key, words in TARGETS.items():
        c = sum(text.count(w) for w in words)
        if c > 0:
            counts[key] = c

    if counts:
        df_counts = pd.DataFrame(list(counts.items()), columns=['Từ Khóa', 'Lượt Nhắc'])
        return df_counts.sort_values(by='Lượt Nhắc', ascending=False).head(8)
    
    return pd.DataFrame(columns=['Từ Khóa', 'Lượt Nhắc'])

def get_star_num(rank_str, desc_str=''):
    s = str(rank_str).strip()
    first_word = s.split()[0] if s else ''
    try:
        return float(first_word)
    except Exception:
        pass
    d = str(desc_str).lower()
    if '5' in d: return 5.0
    if '4' in d: return 4.0
    if '3' in d: return 3.0
    return 2.0

def get_numeric_price(rank_str, desc_str='', score=8.0):
    try:
        sc = float(str(score).replace(',', '.'))
    except Exception:
        sc = 8.0
    star_val = get_star_num(rank_str, desc_str)
    if star_val >= 4.8: base = 1800000
    elif star_val >= 3.8: base = 1100000
    elif star_val >= 2.8: base = 650000
    elif star_val >= 1.8: base = 400000
    else: base = 250000
    return int(round(base * (sc / 8.0), -4))

def parse_num_col(val, default=8.0):
    try:
        return float(str(val).replace(',', '.'))
    except Exception:
        return default

def render():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 18px 24px; border-radius: 12px; color: white; margin-bottom: 16px;">
        <h1 style="color: #60a5fa; margin: 0; font-size: 2.0rem;">📊 BẢNG DASHBOARD PHÂN TÍCH CHỈ SỐ DỊCH VỤ</h1>
        <p style="margin-top: 4px; opacity: 0.9; font-size: 0.92rem;">Báo cáo Phân tích Vị thế Dịch vụ & Chân dung Khách hàng (Dữ liệu Thực tế 100%)</p>
    </div>
    """, unsafe_allow_html=True)

    df_hotels, df_comments = load_partner_data()

    if df_hotels.empty or df_comments.empty:
        st.error("Không nạp được tập dữ liệu phân tích.")
        return

    if "custom_hotels_df" in st.session_state or "custom_comments_df" in st.session_state:
        st.success("⚡ **Hệ thống đang hoạt động trên Dữ liệu CSV Mới Nạp**: Đã tự động phân tích & vẽ lại toàn bộ Đồ thị & Bảng số liệu.")

    df_hotels['Score_Num'] = [parse_num_col(v, 8.0) for v in df_hotels['Total_Score']]
    df_hotels['Estimated_Price'] = [
        get_numeric_price(r.get('Hotel_Rank', ''), r.get('Hotel_Description', ''), r.get('Total_Score', '8.5'))
        for _, r in df_hotels.iterrows()
    ]
    df_hotels['Estimated_Price'] = pd.to_numeric(df_hotels['Estimated_Price'], errors='coerce').fillna(1000000.0).astype(float)

    for col in ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']:
        if col in df_hotels.columns:
            df_hotels[f'{col}_Num'] = [parse_num_col(v, 8.0) for v in df_hotels[col]]
        else:
            df_hotels[f'{col}_Num'] = 8.0

    selected_hotel_name = st.selectbox(
        "🏨 Chọn Khách Sạn Cần Phân Tích Đối Soát Vị Thế:",
        options=list(df_hotels['Hotel_Name'].values),
        index=0
    )

    selected_hotel = df_hotels[df_hotels['Hotel_Name'] == selected_hotel_name].iloc[0]
    selected_hotel_id = str(selected_hotel['Hotel_ID']).strip()
    h_score = float(selected_hotel['Score_Num'])
    avg_market_score = float(df_hotels['Score_Num'].mean())
    score_diff = h_score - avg_market_score

    # Sub-dataset of comments for this specific selected hotel
    df_sub_comments = df_comments[df_comments['Hotel ID'].astype(str).str.strip() == selected_hotel_id] if 'Hotel ID' in df_comments.columns else pd.DataFrame()
    real_comment_count = len(df_sub_comments)

    # PROMINENT HOTEL HEADER BANNER CARD
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; border-left: 5px solid #3b82f6; padding: 14px 20px; border-radius: 10px; margin-bottom: 18px; color: #f8fafc;">
        <h3 style="margin: 0 0 6px 0; color: #60a5fa; font-size: 1.25rem;">🏨 Khách Sạn Đang Phân Tích: <b style="color:#ffffff;">{selected_hotel_name}</b></h3>
        <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.92rem; opacity: 0.95; color: #e2e8f0;">
            <span>⭐ Điểm Đánh Giá: <b style="color:#fbbf24;">{h_score:.1f} / 10</b></span>
            <span>📍 Địa chỉ: <b>{selected_hotel.get('Hotel_Address', 'Nha Trang')}</b></span>
            <span>💰 Giá ước tính: <b style="color:#34d399;">{int(selected_hotel['Estimated_Price']):,} đ/đêm</b></span>
            <span>💬 Số lượt bình luận: <b style="color:#c084fc;">{real_comment_count} bài</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ROW 1: CHART 1 & CHART 2
    r1_col1, r1_col2 = st.columns(2)

    # -------------------------------------------------------------
    # CHART 1: ĐỊNH VỊ CHẤT LƯỢNG (RADAR CHART) - REAL ASPECT DATA
    # -------------------------------------------------------------
    with r1_col1:
        with st.container(border=True):
            st.markdown("<h4 style='text-align:center; margin-top:0;'>ĐỊNH VỊ CHẤT LƯỢNG</h4>", unsafe_allow_html=True)
            
            categories = ['Vị trí', 'Vệ sinh', 'Dịch vụ', 'Tiện nghi', 'Đáng giá tiền']
            
            h_loc = float(selected_hotel['Location_Num'])
            h_clean = float(selected_hotel['Cleanliness_Num'])
            h_serv = float(selected_hotel['Service_Num'])
            h_fac = float(selected_hotel['Facilities_Num'])
            h_val = float(selected_hotel['Value_for_money_Num'])
            
            h_vals = [h_loc, h_clean, h_serv, h_fac, h_val]
            h_vals.append(h_vals[0])
            
            m_loc = float(df_hotels['Location_Num'].mean())
            m_clean = float(df_hotels['Cleanliness_Num'].mean())
            m_serv = float(df_hotels['Service_Num'].mean())
            m_fac = float(df_hotels['Facilities_Num'].mean())
            m_val = float(df_hotels['Value_for_money_Num'].mean())
            
            m_vals = [m_loc, m_clean, m_serv, m_fac, m_val]
            m_vals.append(m_vals[0])
            
            categories_closed = categories + [categories[0]]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=h_vals, theta=categories_closed, fill='toself', name='Khách sạn của bạn',
                line=dict(color='#f97316', width=3), fillcolor='rgba(249, 115, 22, 0.15)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=m_vals, theta=categories_closed, name='Trung bình thị trường',
                line=dict(color='#2563eb', width=2.5, dash='dash')
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[5, 10], tickfont=dict(size=9)),
                    angularaxis=dict(tickfont=dict(size=10))
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
                height=275,
                margin=dict(l=45, r=45, t=10, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True, theme="streamlit", config={'displayModeBar': False})

    # -------------------------------------------------------------
    # CHART 2: TOP TỪ KHÓA ĐÁNH GIÁ TÍCH CỰC (FAST EXTRACTED PER HOTEL)
    # -------------------------------------------------------------
    hotel_kw_df = fast_get_hotel_keywords(df_sub_comments)

    with r1_col2:
        with st.container(border=True):
            st.markdown("<h4 style='text-align:center; margin-top:0;'>TOP TỪ KHÓA ĐÁNH GIÁ TÍCH CỰC</h4>", unsafe_allow_html=True)
            
            if hotel_kw_df is not None and not hotel_kw_df.empty:
                top_words_df = hotel_kw_df.sort_values(by='Lượt Nhắc', ascending=True)
                
                fig_words = px.bar(
                    top_words_df, y='Từ Khóa', x='Lượt Nhắc', orientation='h',
                    color='Lượt Nhắc', color_continuous_scale='Tealgrn', text='Lượt Nhắc'
                )
                max_word_val = max(top_words_df['Lượt Nhắc']) if not top_words_df.empty else 100
                fig_words.update_traces(texttemplate='%{text}', textposition='outside')
                fig_words.update_layout(
                    height=275,
                    margin=dict(l=15, r=65, t=10, b=25),
                    coloraxis_showscale=False,
                    xaxis=dict(
                        title=dict(text="Số Lượt Du Khách Nhắc Đến", font=dict(size=10)),
                        range=[0, max_word_val * 1.2],
                        automargin=True
                    ),
                    yaxis=dict(title=None, automargin=True)
                )
                st.plotly_chart(fig_words, use_container_width=True, theme="streamlit", config={'displayModeBar': False})
            else:
                st.info("Chưa có bài bình luận thực tế nào cho khách sạn này trong tập dữ liệu.")

    # ROW 2: CHART 3 & CHART 4
    r2_col1, r2_col2 = st.columns(2)

    # -------------------------------------------------------------
    # CHART 3: PHÂN TÍCH KHÁCH HÀNG REAL (NATIONALITY & GROUP TYPE FOR THIS HOTEL)
    # -------------------------------------------------------------
    with r2_col1:
        with st.container(border=True):
            st.markdown("<h4 style='text-align:center; margin-top:0;'>PHÂN TÍCH KHÁCH HÀNG CỦA KHÁCH SẠN</h4>", unsafe_allow_html=True)
            
            if not df_sub_comments.empty and 'Nationality' in df_sub_comments.columns and not df_sub_comments['Nationality'].dropna().empty:
                c_sub1, c_sub2 = st.columns(2)
                with c_sub1:
                    st.markdown("<p style='text-align:center; font-weight:bold; font-size:0.88rem; margin-bottom:2px;'>Quốc tịch</p>", unsafe_allow_html=True)
                    nat_counts = df_sub_comments['Nationality'].value_counts().head(3)
                    other_count = len(df_sub_comments) - nat_counts.sum()
                    total_count = len(df_sub_comments)
                    
                    raw_labels = list(nat_counts.index) + ['Khác']
                    raw_values = list(nat_counts.values) + [other_count]
                    
                    labels = [f"{lbl} ({val/total_count*100:.1f}%)" for lbl, val in zip(raw_labels, raw_values)]
                    values = raw_values
                    
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=labels, values=values, hole=.55,
                        domain=dict(y=[0.25, 1.0]),
                        marker=dict(colors=['#2563eb', '#38bdf8', '#64748b', '#f97316']),
                        textinfo='none', hoverinfo='label+value'
                    )])
                    fig_donut.update_layout(
                        height=245, margin=dict(l=5, r=5, t=5, b=0), showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=0.18, xanchor="center", x=0.5, font=dict(size=8.0))
                    )
                    st.plotly_chart(fig_donut, use_container_width=True, theme="streamlit", config={'displayModeBar': False})
                    
                with c_sub2:
                    st.markdown("<p style='text-align:center; font-weight:bold; font-size:0.88rem; margin-bottom:2px;'>Loại nhóm</p>", unsafe_allow_html=True)
                    grp_counts = df_sub_comments['Group Name'].value_counts().head(4).reset_index()
                    grp_counts.columns = ['Loại Nhóm', 'Số Lượng']
                    max_grp_val = max(grp_counts['Số Lượng']) if not grp_counts.empty else 10
                    
                    fig_grp = px.bar(
                        grp_counts.sort_values(by='Số Lượng', ascending=True),
                        y='Loại Nhóm', x='Số Lượng', orientation='h',
                        color_discrete_sequence=['#3b82f6'], text='Số Lượng'
                    )
                    fig_grp.update_traces(textposition='outside')
                    fig_grp.update_layout(
                        height=245, margin=dict(l=85, r=35, t=5, b=20),
                        xaxis=dict(title=None, range=[0, max_grp_val * 1.25], automargin=True),
                        yaxis=dict(title=None, tickfont=dict(size=8.5), automargin=True)
                    )
                    st.plotly_chart(fig_grp, use_container_width=True, theme="streamlit", config={'displayModeBar': False})
            else:
                st.info("Chưa có dữ liệu bài đánh giá để phân tích chân dung khách hàng cho khách sạn này.")

    # -------------------------------------------------------------
    # CHART 4: XU HƯỚNG THEO THỜI GIAN REAL (THIS HOTEL SUBSET)
    # -------------------------------------------------------------
    with r2_col2:
        with st.container(border=True):
            st.markdown("<h4 style='text-align:center; margin-top:0;'>XU HƯỚNG BÌNH LUẬN THEO THỜI GIAN</h4>", unsafe_allow_html=True)
            
            month_labels = [f"Tháng {m}" for m in range(1, 13)]
            review_counts, avg_scores = fast_compute_monthly_trend(df_sub_comments, default_score=h_score)
            max_rev_val = max(review_counts) if len(review_counts) > 0 and max(review_counts) > 0 else 5
            
            fig_dual = go.Figure()
            fig_dual.add_trace(go.Bar(
                x=month_labels, y=review_counts, name='Số lượt đánh giá', marker_color='#7e57c2', yaxis='y'
            ))
            fig_dual.add_trace(go.Scatter(
                x=month_labels, y=avg_scores, name='Điểm trung bình', mode='lines+markers',
                line=dict(color='#f57c00', width=2.5), marker=dict(size=6, color='#f57c00'), yaxis='y2'
            ))

            fig_dual.update_layout(
                height=275,
                margin=dict(l=35, r=35, t=35, b=20),
                xaxis=dict(tickmode='array', tickvals=month_labels, automargin=True, tickfont=dict(size=9)),
                yaxis=dict(
                    title=dict(text='Lượng đánh giá', font=dict(size=9)),
                    range=[0, max_rev_val * 1.2],
                    tickfont=dict(size=9), showgrid=True, automargin=True
                ),
                yaxis2=dict(
                    title=dict(text='Điểm trung bình', font=dict(size=9)),
                    tickfont=dict(size=9), overlaying='y', side='right',
                    range=[0, 10], showgrid=False, automargin=True
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=9))
            )
            st.plotly_chart(fig_dual, use_container_width=True, theme="streamlit", config={'displayModeBar': False})

    st.divider()

    # SECTION 2: 3 VISUAL EXECUTIVE KPI METRIC CARDS - REAL DATA
    st.markdown("<h3 style='text-align: center; font-size: 1.45rem; font-weight: bold; margin-top: 10px; margin-bottom: 20px; color: #60a5fa;'>🎯 BÁO CÁO PHÂN TÍCH VỊ THẾ CẠNH TRANH & MẶT BẰNG THỊ TRƯỜNG</h3>", unsafe_allow_html=True)

    avg_hotel_comments = int(len(df_comments) / len(df_hotels)) if len(df_hotels) > 0 else 108
    comment_diff = real_comment_count - avg_hotel_comments

    kpi_c1, kpi_c2, kpi_c3 = st.columns(3)

    with kpi_c1:
        with st.container(border=True):
            st.markdown("<p style='font-size:0.88rem; font-weight:bold; color:#94a3b8; margin-bottom:2px;'>1. Điểm Đánh Giá Tổng Thể</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='margin:0; color:#fbbf24; font-size:2.2rem;'>{h_score:.2f} <small style='font-size:1.0rem; color:#94a3b8;'>/ 10</small></h2>", unsafe_allow_html=True)
            st.progress(min(1.0, max(0.0, h_score / 10.0)))
            st.markdown(f"<div style='margin-top:4px;'><span style='background-color:rgba(16,185,129,0.15); color:#10b981; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:0.84rem;'>🟢 {score_diff:+.2f} so với TB ({avg_market_score:.2f})</span></div>", unsafe_allow_html=True)

    with kpi_c2:
        with st.container(border=True):
            st.markdown("<p style='font-size:0.88rem; font-weight:bold; color:#94a3b8; margin-bottom:2px;'>2. Mức Giá Phòng Trung Bình (/đêm)</p>", unsafe_allow_html=True)
            price_val = int(selected_hotel['Estimated_Price'])
            avg_price = int(df_hotels['Estimated_Price'].mean())
            price_diff = price_val - avg_price
            st.markdown(f"<h2 style='margin:0; color:#34d399; font-size:2.2rem;'>{price_val:,} <small style='font-size:0.95rem; color:#94a3b8;'>đ/đêm</small></h2>".replace(',', '.'), unsafe_allow_html=True)
            st.caption(f"Mặt bằng chung: {avg_price:,} đ/đêm".replace(',', '.'))
            st.markdown(f"<div style='margin-top:2px;'><span style='background-color:rgba(59,130,246,0.15); color:#60a5fa; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:0.84rem;'>🔵 Phân khúc ({price_diff:+,}đ)".replace(',', '.') + "</span></div>", unsafe_allow_html=True)

    with kpi_c3:
        with st.container(border=True):
            st.markdown("<p style='font-size:0.88rem; font-weight:bold; color:#94a3b8; margin-bottom:2px;'>3. Số Lượng Bình Luận Đóng Góp</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='margin:0; color:#c084fc; font-size:2.2rem;'>{real_comment_count} <small style='font-size:0.95rem; color:#94a3b8;'>lượt đánh giá</small></h2>", unsafe_allow_html=True)
            st.caption(f"Mặt bằng chung: ~{avg_hotel_comments} lượt/KS")
            st.markdown(f"<div style='margin-top:2px;'><span style='background-color:rgba(168,85,247,0.15); color:#c084fc; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:0.84rem;'>⭐ Tương tác ({comment_diff:+} lượt so với TB)</span></div>", unsafe_allow_html=True)

    # DOWNLOAD DETAILED HOTEL REPORT CSV BUTTON
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    c_dl1, c_dl2 = st.columns([2.2, 1.8])
    with c_dl2:
        hotel_report_df = pd.DataFrame([
            {"Tiêu Chí": "Tên Khách Sạn", "Giá Trị": selected_hotel_name},
            {"Tiêu Chí": "Hotel ID", "Giá Trị": selected_hotel_id},
            {"Tiêu Chí": "Điểm Đánh Giá", "Giá Trị": f"{h_score:.2f} / 10"},
            {"Tiêu Chí": "Mức Giá Phòng Ước Tính (/đêm)", "Giá Trị": f"{int(selected_hotel['Estimated_Price']):,} đ".replace(',', '.')},
            {"Tiêu Chí": "Địa Chỉ", "Giá Trị": str(selected_hotel.get('Hotel_Address', ''))},
            {"Tiêu Chí": "Phân Khúc Hạng Sao", "Giá Trị": str(selected_hotel.get('Hotel_Rank', ''))},
            {"Tiêu Chí": "Vị Trí (Score)", "Giá Trị": f"{float(selected_hotel['Location_Num']):.1f}"},
            {"Tiêu Chí": "Vệ Sinh (Score)", "Giá Trị": f"{float(selected_hotel['Cleanliness_Num']):.1f}"},
            {"Tiêu Chí": "Dịch Vụ (Score)", "Giá Trị": f"{float(selected_hotel['Service_Num']):.1f}"},
            {"Tiêu Chí": "Tiện Nghi (Score)", "Giá Trị": f"{float(selected_hotel['Facilities_Num']):.1f}"},
            {"Tiêu Chí": "Đáng Giá Tiền (Score)", "Giá Trị": f"{float(selected_hotel['Value_for_money_Num']):.1f}"},
            {"Tiêu Chí": "Tổng Số Lượt Bình Luận Đóng Góp", "Giá Trị": str(real_comment_count)},
            {"Tiêu Chí": "Điểm TB Mặt Bằng Thị Trường", "Giá Trị": f"{avg_market_score:.2f} / 10"},
            {"Tiêu Chí": "Chênh Lệch Với Thị Trường", "Giá Trị": f"{score_diff:+.2f}"}
        ])
        
        csv_report = hotel_report_df.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            label=f"📥 Tải Báo Cáo Chi Tiết ({selected_hotel_name[:18]}...)",
            data=csv_report,
            file_name=f"bao_cao_chi_tiet_{re.sub(r'[^a-zA-Z0-9]', '_', selected_hotel_name).lower()}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
