# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import zipfile

from src.utils.nlp_cleaner import clean_text, extract_star_constraint, compute_aspect_matrix
from src.models.user_profiler import NATIONALITY_PROFILES, load_real_reviewer_personas
from src.pipeline.hybrid_engine import parse_star_numeric, get_star_badge, calculate_hybrid_scores, parse_nlp_query_constraints

# ═══ 24+ High Resolution Hotel Photos ═══
HOTEL_IMAGES = [
    "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=600&q=80",
    "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?w=600&q=80",
    "https://images.unsplash.com/photo-1571896349842-33c89424de2d?w=600&q=80",
    "https://images.unsplash.com/photo-1540555700478-4be289fbecef?w=600&q=80",
    "https://images.unsplash.com/photo-1563911302283-d2bc129e7570?w=600&q=80",
    "https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=600&q=80",
    "https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?w=600&q=80",
    "https://images.unsplash.com/photo-1455587734955-081b22074882?w=600&q=80",
    "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=600&q=80",
    "https://images.unsplash.com/photo-1520483691742-e1c7e441820d?w=600&q=80",
    "https://images.unsplash.com/photo-1584132967334-10e028bd69f7?w=600&q=80",
    "https://images.unsplash.com/photo-1445019980597-93fa8acb246c?w=600&q=80",
    "https://images.unsplash.com/photo-1611892440504-42a792e24d32?w=600&q=80",
    "https://images.unsplash.com/photo-1590490360182-c33d955e3476?w=600&q=80",
    "https://images.unsplash.com/photo-1578683010236-d716f9a3f461?w=600&q=80",
    "https://images.unsplash.com/photo-1618773928121-c32242e63f39?w=600&q=80",
    "https://images.unsplash.com/photo-1564501049412-61c2a3083791?w=600&q=80",
    "https://images.unsplash.com/photo-1596394516093-501ba68a0ba6?w=600&q=80",
    "https://images.unsplash.com/photo-1549294413-26f195200c16?w=600&q=80",
    "https://images.unsplash.com/photo-1512918728675-ed5a9ecdebfd?w=600&q=80",
    "https://images.unsplash.com/photo-1559599238-308793637427?w=600&q=80",
    "https://images.unsplash.com/photo-1587213811864-46e59f6873b1?w=600&q=80",
    "https://images.unsplash.com/photo-1625244724120-1fd1d34d00f6?w=600&q=80",
    "https://images.unsplash.com/photo-1615460549969-36fa19521a4f?w=600&q=80",
]

def parse_clean_name(name_str):
    s = str(name_str).strip().replace('_', ' ').replace('*', '').strip()
    if '(' in s:
        s = s[:s.index('(')].strip()
    words = s.split()
    cleaned = " ".join([w.capitalize() if not w.isupper() else w for w in words])
    return cleaned if cleaned else str(name_str).strip()

def parse_clean_address(addr_str):
    s = str(addr_str).strip().replace('_', ' ').replace('*', '').strip()
    s = re.sub(r',?\s*Việt Nam\s*,?', '', s, flags=re.IGNORECASE)
    s = re.sub(r',?\s*\d{5,6}\s*', '', s)
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return ', '.join(parts[:3]) if len(parts) > 3 else (s if s else 'Nha Trang, Khánh Hòa')

def _clean_review_text(text):
    if not text or str(text) == 'nan':
        return ""
    t = str(text)
    junk = [
        r"Dịch văn bản sang tiếng Việt.*",
        r"Xem bản dịch.*",
        r"Xem văn bản gốc.*",
        r"Translated from.*",
        r"Biên dịch bởi Agoda.*",
        r"Originally written in.*"
    ]
    for pattern in junk:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE)
    t = t.strip()
    if t:
        t = t[0].upper() + t[1:]
    return t

def safe_float_score(val, default=8.0):
    if val is None or str(val).strip().lower() in ['no information', 'nan', 'none', '']:
        return None
    try:
        val_str = str(val).replace(',', '.').strip()
        return float(val_str)
    except (ValueError, TypeError):
        return default


def get_numeric_price(rank_str, desc_str='', score=8.0):
    try: sc = float(str(score).replace(',', '.'))
    except Exception: sc = 8.0
    star_val = parse_star_numeric(rank_str, desc_str)
    if star_val >= 4.8: base = 1800000
    elif star_val >= 3.8: base = 1100000
    elif star_val >= 2.8: base = 650000
    elif star_val >= 1.8: base = 400000
    else: base = 250000
    return int(round(base * (sc / 8.0), -4))

@st.cache_data
def load_booking_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    info_path = os.path.join(data_dir, 'hotel_info.csv')
    df_hotels = pd.read_csv(info_path) if os.path.exists(info_path) else pd.DataFrame()
        
    comments_gz = os.path.join(data_dir, 'hotel_comments.csv.gz')
    comments_path = os.path.join(data_dir, 'hotel_comments.csv')
    if os.path.exists(comments_gz):
        df_comments = pd.read_csv(comments_gz)
    elif os.path.exists(comments_path):
        df_comments = pd.read_csv(comments_path)
    else:
        df_comments = pd.DataFrame()
        
    cos_path = os.path.join(data_dir, 'cosine_sim.pkl')
    cosine_sim = None
    if os.path.exists(cos_path):
        with open(cos_path, 'rb') as f:
            cosine_sim = pickle.load(f)
            
    svd_zip = os.path.join(data_dir, 'svd_model.zip')
    svd_path = os.path.join(data_dir, 'svd_model.pkl')
    svd_model = None
    if os.path.exists(svd_zip):
        with zipfile.ZipFile(svd_zip, 'r') as zf:
            with zf.open('svd_model.pkl') as f:
                svd_model = pickle.load(f)
    elif os.path.exists(svd_path):
        with open(svd_path, 'rb') as f:
            svd_model = pickle.load(f)
            
    df_aspects = compute_aspect_matrix(df_comments) if not df_comments.empty else pd.DataFrame()
    real_personas = load_real_reviewer_personas(df_comments)

    if not df_hotels.empty:
        df_hotels['Score_Num'] = pd.to_numeric(df_hotels['Total_Score'].astype(str).str.replace(',', '.'), errors='coerce').fillna(8.0)
        df_hotels['Estimated_Price'] = [
            get_numeric_price(row.get('Hotel_Rank', ''), row.get('Hotel_Description', ''), row.get('Total_Score', '8.5'))
            for _, row in df_hotels.iterrows()
        ]
        df_hotels['Clean_Name'] = [parse_clean_name(r) for r in df_hotels['Hotel_Name']]
        df_hotels['Clean_Addr'] = [parse_clean_address(r) for r in df_hotels['Hotel_Address']]
        
        if 'Hotel_Description' in df_hotels.columns:
            clean_descs = []
            for d in df_hotels['Hotel_Description']:
                if pd.notna(d):
                    s = clean_text(str(d))[:220].strip()
                    if s:
                        s = s[0].upper() + s[1:]
                    clean_descs.append(s)
                else:
                    clean_descs.append('')
            df_hotels['Clean_Desc'] = clean_descs
        else:
            df_hotels['Clean_Desc'] = ''
            
        df_hotels['Star_Num'] = [parse_star_numeric(r['Hotel_Rank'], r['Hotel_Description'], r['Estimated_Price']) for _, r in df_hotels.iterrows()]
        df_hotels['Star_Badge'] = [get_star_badge(r['Hotel_Rank'], r['Hotel_Description'], r['Estimated_Price']) for _, r in df_hotels.iterrows()]
        df_hotels['Price_Display'] = [f"{p:,}".replace(',', '.') + " đ" for p in df_hotels['Estimated_Price']]

    return df_hotels, df_comments, cosine_sim, svd_model, df_aspects, real_personas

def get_dynamic_badges(row, star_val):
    desc = str(row.get('Hotel_Description', '')).lower()
    badges = []
    if star_val >= 3.0 and any(w in desc for w in ['hồ bơi', 'bể bơi', 'pool']):
        badges.append("<span style='background:rgba(251,146,60,0.15); color:#fb923c; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600;'>🏊 Hồ bơi</span>")
    if any(w in desc for w in ['biển', 'bãi biển', 'beach']):
        badges.append("<span style='background:rgba(168,85,247,0.15); color:#a855f7; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600;'>🏖️ Gần biển</span>")
    if star_val >= 4.0 or any(w in desc for w in ['sáng', 'buffet', 'breakfast']):
        badges.append("<span style='background:rgba(56,189,248,0.15); color:#38bdf8; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600;'>🍳 Bữa sáng</span>")
    if star_val >= 4.0 and any(w in desc for w in ['spa', 'massage']):
        badges.append("<span style='background:rgba(244,114,182,0.15); color:#f472b6; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600;'>💆 Spa</span>")
    if star_val < 3.0:
        badges.append("<span style='background:rgba(74,222,128,0.15); color:#4ade80; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600;'>⚡ Giá tốt</span>")
    return " ".join(badges)

def _get_active_profile(real_personas):
    is_logged_in = st.session_state.get('is_logged_in', st.session_state.get('logged_in', False))
    if is_logged_in:
        u_nat = st.session_state.get('user_nationality', 'Việt Nam')
        uid = st.session_state.get('user_id')
        base = NATIONALITY_PROFILES.get(u_nat, NATIONALITY_PROFILES["Việt Nam"]).copy()
        base['reviewer_id'] = uid
        return base
    return None

def _render_login_header(real_personas):
    """Render Login & AI Insights Popover/Expander in top right header."""
    is_logged_in = st.session_state.get('is_logged_in', False) or st.session_state.get('logged_in', False)
    
    if is_logged_in:
        uid = st.session_state.get('user_id', '')
        u_name = st.session_state.get('user_name', 'User')
        u_nat = st.session_state.get('user_nationality', '')
        flag = NATIONALITY_PROFILES.get(u_nat, {}).get('flag', '👤')
        
        prof = NATIONALITY_PROFILES.get(u_nat, NATIONALITY_PROFILES["Việt Nam"])
        w_loc = int(prof.get('w_loc', prof.get('aspect_weights', {}).get('Loc', 0.0)) * 100)
        w_clean = int(prof.get('w_clean', prof.get('aspect_weights', {}).get('Clean', 0.0)) * 100)
        w_staff = int(prof.get('w_staff', prof.get('aspect_weights', {}).get('Staff', 0.0)) * 100)
        w_pool = int(prof.get('w_pool', prof.get('aspect_weights', {}).get('Pool', 0.0)) * 100)
        w_food = int(prof.get('w_food', prof.get('aspect_weights', {}).get('Food', 0.0)) * 100)
        pref_summary = prof.get('pref_summary', 'N/A')

        with st.popover(f"{flag} {u_name}", use_container_width=True):
            st.markdown(f"""
            <div style="font-size:0.85rem; line-height:1.45;">
                👤 <b>Khách hàng:</b> {u_name}<br>
                🆔 <b>Mã định danh (ID):</b> <code>{uid}</code><br>
                🌐 <b>Quốc tịch:</b> {flag} {u_nat}<br>
                🎯 <b>Gu du lịch:</b> <i>{pref_summary}</i>
            </div>
            <div style="font-size:0.8rem; line-height:1.4; margin-top:8px; background:rgba(59,89,152,0.05); padding:8px; border-radius:6px; border:1px solid rgba(59,89,152,0.1); color:var(--text-color);">
                <b>💡 Trọng số Aspect Cá nhân hóa:</b><br>
                📍 Vị trí: <b>{w_loc}%</b> | 🧼 Vệ sinh: <b>{w_clean}%</b> | 👥 Phục vụ: <b>{w_staff}%</b><br>
                🏊 Hồ bơi: <b>{w_pool}%</b> | 🍳 Ăn sáng: <b>{w_food}%</b>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            if st.button("🚪 Đăng xuất", key="btn_logout", type="secondary", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.is_logged_in = False
                st.session_state.user_id = None
                st.session_state.user_name = None
                st.session_state.user_nationality = None
                st.rerun()
    else:
        if real_personas:
            persona_list = list(real_personas.values())
            display_opts = ["-- Chọn tài khoản --"] + [
                f"{p['flag']} {p['reviewer_name']} ({p['nationality']})"
                for p in persona_list
            ]
            with st.popover("🔑 Đăng nhập", use_container_width=True):
                st.markdown("### 🔑 Đăng Nhập Hệ Thống")
                st.caption("Chọn tài khoản để cá nhân hóa đề xuất:")
                sel = st.selectbox("Tài khoản:", range(len(display_opts)),
                                   format_func=lambda i: display_opts[i], index=0,
                                   label_visibility="collapsed", key="login_select")
                if sel > 0:
                    chosen = persona_list[sel - 1]
                    if st.button("Đăng nhập", type="primary", use_container_width=True, key="btn_login"):
                        st.session_state.logged_in = True
                        st.session_state.is_logged_in = True
                        st.session_state.user_id = chosen['reviewer_id']
                        st.session_state.user_name = chosen['reviewer_name']
                        st.session_state.user_nationality = chosen['nationality']
                        st.rerun()
        else:
            st.caption("👤 Khách")

def render():
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "mode1"

    df_hotels, df_comments, cosine_sim, svd_model, df_aspects, real_personas = load_booking_data()
    is_logged_in = st.session_state.get('is_logged_in', False) or st.session_state.get('logged_in', False)

    # Tự động nhận diện ngôn ngữ của truy vấn khi ở chế độ Guest
    if not is_logged_in:
        search_val = st.session_state.get('search_query_input', '').strip()
        detected_nat = None
        if search_val:
            q_lower = search_val.lower()
            english_kws = {'hotel', 'resort', 'beach', 'pool', 'swimming', 'service', 'clean', 'breakfast', 'room', 'friendly', 'staff', 'location', 'cheap', 'luxury', 'five star', 'four star', 'stay'}
            korean_kws = {'호텔', '리조트', '수영장', '조식', '친절', '깨끗', '바다', '비치'}
            
            eng_matches = sum(1 for kw in english_kws if kw in q_lower)
            kor_matches = sum(1 for kw in korean_kws if kw in q_lower)
            
            if eng_matches > kor_matches and eng_matches > 0:
                detected_nat = "Hoa Kỳ"
            elif kor_matches > eng_matches and kor_matches > 0:
                detected_nat = "Hàn Quốc"
                
        st.session_state.detected_nat_lang = detected_nat
        if detected_nat and detected_nat != st.session_state.get('selected_persona_key', 'Việt Nam'):
            st.session_state.selected_persona_key = detected_nat
            st.session_state.guest_nat_selectbox = detected_nat

    # ═══ 1. HEADER CONTAINER ═══
    with st.container(border=True):
        col_h1, col_h2 = st.columns([2.8, 1.2], vertical_alignment="center")
        with col_h1:
            st.markdown("""
            <div style="padding: 2px 0;">
                <h2 style="margin:0; font-size:1.65rem; color:var(--text-color); font-weight:700;">🏨 agoda | Khách Sạn Nha Trang</h2>
                <p style="margin:2px 0 0; font-size:0.85rem; color:var(--text-color); opacity:0.75;">Tìm khách sạn phù hợp nhất cho bạn tại Nha Trang, Khánh Hòa</p>
            </div>
            """, unsafe_allow_html=True)
        with col_h2:
            _render_login_header(real_personas)

    # ═══ 2. ULTRA-COMPACT SEARCH & FILTER CONTAINER ═══
    with st.container(border=True):
        if not is_logged_in:
            c_m1, c_m2, c_m3 = st.columns([1.5, 1.5, 1.0], vertical_alignment="center")
            with c_m1:
                b1t = "primary" if st.session_state.search_mode == "mode1" else "secondary"
                if st.button("💬 Tìm theo mô tả", type=b1t, use_container_width=True):
                    st.session_state.search_mode = "mode1"
                    st.rerun()
            with c_m2:
                b2t = "primary" if st.session_state.search_mode == "mode2" else "secondary"
                if st.button("🎛️ Lọc nâng cao", type=b2t, use_container_width=True):
                    st.session_state.search_mode = "mode2"
                    st.rerun()
            with c_m3:
                if "selected_persona_key" not in st.session_state:
                    st.session_state.selected_persona_key = "Việt Nam"
                nat_opts = ["Việt Nam", "Hàn Quốc", "Hoa Kỳ", "Úc"]
                nat_flags = {"Việt Nam": "🇻🇳", "Hàn Quốc": "🇰🇷", "Hoa Kỳ": "🇺🇸", "Úc": "🇦🇺"}
                selected_nat = st.selectbox(
                    "Quốc tịch khách:",
                    options=nat_opts,
                    format_func=lambda x: f"{nat_flags.get(x, '')} {x}",
                    index=nat_opts.index(st.session_state.selected_persona_key) if st.session_state.selected_persona_key in nat_opts else 0,
                    key="guest_nat_selectbox",
                    label_visibility="collapsed"
                )
                if selected_nat != st.session_state.selected_persona_key:
                    st.session_state.selected_persona_key = selected_nat
                    st.rerun()
        else:
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                b1t = "primary" if st.session_state.search_mode == "mode1" else "secondary"
                if st.button("💬 Tìm theo mô tả", type=b1t, use_container_width=True):
                    st.session_state.search_mode = "mode1"
                    st.rerun()
            with c_m2:
                b2t = "primary" if st.session_state.search_mode == "mode2" else "secondary"
                if st.button("🎛️ Lọc nâng cao", type=b2t, use_container_width=True):
                    st.session_state.search_mode = "mode2"
                    st.rerun()

        # Model Selector Pills
        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
        col_m_label, col_m_pills = st.columns([1.2, 4.0], vertical_alignment="center")
        with col_m_label:
            st.markdown("<span style='font-size:0.83rem; font-weight:700; color:var(--text-color);'>🤖 Động cơ AI:</span>", unsafe_allow_html=True)
        with col_m_pills:
            model_mode_opts = {
                "🏆 2-Stage Hybrid (Chính thức)": "hybrid",
                "📝 Content-Based (Chỉ NLP/Aspect)": "content",
                "👥 Collaborative Filtering (Chỉ SVD)": "collaborative"
            }
            if "selected_model_mode_key" not in st.session_state:
                st.session_state.selected_model_mode_key = "🏆 2-Stage Hybrid (Chính thức)"
            
            sel_model_str = st.segmented_control(
                "Động cơ AI:",
                options=list(model_mode_opts.keys()),
                default=st.session_state.selected_model_mode_key,
                key="model_engine_segmented",
                label_visibility="collapsed"
            )
            new_mode_key = sel_model_str or "🏆 2-Stage Hybrid (Chính thức)"
            if new_mode_key != st.session_state.selected_model_mode_key:
                st.session_state.selected_model_mode_key = new_mode_key
                st.session_state.has_searched = False
                st.rerun()

            active_model_mode = model_mode_opts.get(st.session_state.selected_model_mode_key, "hybrid")

        filtered_df = df_hotels.copy()
        search_query = ""
        top_k = 6

        if st.session_state.search_mode == "mode1":
            c_in, c_k, c_btn = st.columns([5, 1.5, 1.5], vertical_alignment="bottom")
            with c_in:
                st.markdown("<label style='font-size:0.85rem; font-weight:600; color:var(--text-color);'>Nhập nhu cầu du lịch bằng văn bản tự nhiên:</label>", unsafe_allow_html=True)
                search_query = st.text_input(
                    "Nhu cầu:",
                    value=st.session_state.get('preset_query', ''),
                    placeholder="Ví dụ: Khách sạn 3 sao gần trung tâm có buffet sáng, bể bơi...",
                    label_visibility="collapsed",
                    key="search_query_input"
                )
            with c_k:
                st.markdown("<label style='font-size:0.85rem; font-weight:600; color:var(--text-color);'>Số lượng:</label>", unsafe_allow_html=True)
                top_k = st.selectbox("Số lượng hiển thị:", options=[3, 6, 9, 12], index=1, label_visibility="collapsed")
            with c_btn:
                btn_m1_clicked = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True, key="btn_m1_search")

            # Quick search sample chips
            st.markdown("<div style='margin-top:6px; font-size:0.8rem; color:var(--text-color); opacity:0.8;'>💡 <b>Mẫu tìm kiếm nhanh:</b></div>", unsafe_allow_html=True)
            chip1, chip2, chip3 = st.columns(3)
            with chip1:
                if st.button("📍 KS 3 sao gần trung tâm buffet", use_container_width=True, key="chip1"):
                    st.session_state.preset_query = "khách sạn 3 sao gần trung tâm có buffet sáng"
                    st.session_state.has_searched = True
                    st.rerun()
            with chip2:
                if st.button("🏊 Resort 5 sao bể bơi vô cực", use_container_width=True, key="chip2"):
                    st.session_state.preset_query = "resort 5 sao có bể bơi vô cực view biển"
                    st.session_state.has_searched = True
                    st.rerun()
            with chip3:
                if st.button("🏖️ Căn hộ homestay gần biển", use_container_width=True, key="chip3"):
                    st.session_state.preset_query = "căn hộ homestay gần biển giá rẻ"
                    st.session_state.has_searched = True
                    st.rerun()

            if btn_m1_clicked:
                st.session_state.has_searched = True
                st.session_state.preset_query = search_query

        else:
            fc1, fc2, fc3, fc4 = st.columns([1.1, 1.0, 1.1, 1.0], gap="small")
            with fc1:
                locations = ["Tất cả khu vực", "Cam Ranh", "Lộc Thọ", "Vĩnh Phước", "Vĩnh Hải", "Tân Lập", "Vĩnh Nguyên", "Phước Hải"]
                selected_loc = st.selectbox("📍 Khu vực", options=locations)
            with fc2:
                star_opts = ["Tất cả", "5⭐", "4⭐", "3⭐", "1-2⭐"]
                selected_star = st.selectbox("⭐ Số sao", options=star_opts)
            with fc3:
                room_opts = ["Tất cả loại phòng", "Standard / Superior", "Deluxe / Executive", "Suite / Penthouse", "Villa / Biệt thự", "Family / Gia đình"]
                selected_room = st.selectbox("🛏️ Loại phòng", options=room_opts)
            with fc4:
                top_k = st.selectbox("📋 Số lượng hiển thị", options=[3, 6, 9, 12], index=1)

            row2_c1, row2_c2 = st.columns([1.3, 1.1], gap="medium")
            with row2_c1:
                min_p, max_p = st.slider("💰 Khoảng giá (/đêm)", min_value=100000, max_value=2500000, value=(100000, 2500000), step=100000, format="%d đ")
                min_score = st.slider("📊 Điểm tối thiểu", min_value=0.0, max_value=10.0, value=0.0, step=0.5)

            with row2_c2:
                pill_options = ["🏊 Hồ bơi", "🏖️ Gần biển", "🍳 Buffet sáng", "💆 Spa", "🏡 Villa"]
                selected_pills = st.pills("🏷️ Tiện ích", options=pill_options, selection_mode="multi", key="amenities_pills")
                
                pill_pattern_map = {
                    "🏊 Hồ bơi": r'hồ bơi|bể bơi|pool',
                    "🏖️ Gần biển": r'biển|bãi biển|beach|ocean',
                    "🍳 Buffet sáng": r'buffet|ăn sáng|breakfast',
                    "💆 Spa": r'spa|massage',
                    "🏡 Villa": r'villa|biệt thự',
                }
                active_amenity_patterns = [pill_pattern_map[p] for p in (selected_pills or []) if p in pill_pattern_map]

                btn_m2_clicked = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True, key="btn_m2_search")
                if btn_m2_clicked:
                    st.session_state.has_searched = True

            if selected_loc != "Tất cả khu vực":
                filtered_df = filtered_df[filtered_df['Hotel_Address'].str.contains(selected_loc, case=False, na=False)]
            
            if selected_star == "5⭐":
                filtered_df = filtered_df[filtered_df['Star_Num'] >= 4.8]
            elif selected_star == "4⭐":
                filtered_df = filtered_df[(filtered_df['Star_Num'] >= 3.8) & (filtered_df['Star_Num'] < 4.8)]
            elif selected_star == "3⭐":
                filtered_df = filtered_df[(filtered_df['Star_Num'] >= 2.8) & (filtered_df['Star_Num'] < 3.8)]
            elif selected_star == "1-2⭐":
                filtered_df = filtered_df[filtered_df['Star_Num'] < 2.8]
            
            room_kw_map = {
                "Standard / Superior": r'standard|superior',
                "Deluxe / Executive": r'deluxe|executive',
                "Suite / Penthouse": r'suite|penthouse',
                "Villa / Biệt thự": r'villa|biệt thự',
                "Family / Gia đình": r'family|gia đình',
            }
            if selected_room != "Tất cả loại phòng" and selected_room in room_kw_map:
                filtered_df = filtered_df[filtered_df['Hotel_Description'].str.contains(room_kw_map[selected_room], case=False, na=False, regex=True)]

            for pat in active_amenity_patterns:
                filtered_df = filtered_df[filtered_df['Hotel_Description'].str.contains(pat, case=False, na=False, regex=True)]

            if min_score > 0.0:
                filtered_df = filtered_df[filtered_df['Score_Num'] >= min_score]
            filtered_df = filtered_df[(filtered_df['Estimated_Price'] >= min_p) & (filtered_df['Estimated_Price'] <= max_p)]

    # Check if user has triggered search
    has_searched = st.session_state.get('has_searched', False)
    if not has_searched and not search_query.strip():
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("""
            <div style='text-align:center; padding: 25px 15px;'>
                <h3 style='margin:0 0 10px 0; font-size:1.4rem; color:var(--agoda-primary); font-weight:800;'>🔍 VUI LÒNG NHẬP NHU CẦU VÀ BẤM "TÌM KIẾM"</h3>
                <p style='font-size:0.92rem; color:var(--text-color); opacity:0.85; max-width:650px; margin:0 auto 15px auto;'>
                    Hệ thống Đề xuất Agoda Nha Trang với động cơ <b>2-Stage Hybrid Engine</b> sẵn sàng phục vụ. Bạn hãy nhập tiêu chuẩn mong muốn hoặc nhấp chọn các mẫu gợi ý nhanh phía trên.
                </p>
                <div style='display:flex; justify-content:center; gap:20px; font-size:0.85rem; font-weight:600; opacity:0.9;'>
                    <span>🛡️ Lọc Cứng AND-Logic NLP</span>
                    <span>⚡ Trọng Số Năng Động (Dynamic Weighting)</span>
                    <span>👤 Cá Nhân Hóa Theo Quốc Tịch</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    if is_logged_in:
        active_profile = _get_active_profile(real_personas)
    else:
        active_profile = NATIONALITY_PROFILES.get(st.session_state.get('selected_persona_key', 'Việt Nam'), NATIONALITY_PROFILES["Việt Nam"])
        detected_nat = st.session_state.get('detected_nat_lang')
        if detected_nat and detected_nat == st.session_state.selected_persona_key:
            st.info(f"ℹ️ **Phát hiện truy vấn Tiếng Anh/Hàn**: Đã tự động đổi sang hồ sơ du khách **{active_profile.get('flag', '🇺🇸')} {detected_nat}** để tối ưu hóa trọng số xếp hạng.")

    # Cold-Start Warning banner when selecting Collaborative Filtering with descriptive text query
    # Cold-Start Warning banner when selecting Collaborative Filtering with descriptive text query
    if active_model_mode == "collaborative":
        if search_query.strip():
            st.warning("⚠️ **CẢNH BÁO COLD-START WARNING**: Thuật toán Lọc cộng tác (Surprise SVD) không có khả năng hiểu ngữ nghĩa văn bản hay lọc từ khóa mô tả. Điểm Cosine Text Similarity đã bị tắt ($0\%$) và Giai đoạn 1 Lọc cứng NLP đã bị bỏ qua. Hệ thống đang xếp hạng thuần theo lịch sử tương tác chung. Bạn nên chuyển sang **2-Stage Hybrid** hoặc **Content-Based** để có kết quả chính xác hơn.")
        else:
            st.warning("⚠️ **CHẾ ĐỘ COLLABORATIVE FILTERING**: Đã tắt hoàn toàn các bộ lọc NLP/Aspect. Kết quả dựa trên 100% điểm dự đoán SVD.")

    # [STAGE 1] Hard Constraint Filter for Mode 1 (Bypassed for pure Collaborative SVD mode)
    if st.session_state.search_mode == "mode1" and search_query.strip() and active_model_mode != "collaborative":
        filtered_df = parse_nlp_query_constraints(filtered_df, search_query)

    # [STAGE 2] Hybrid Soft Ranking
    hybrid_results = calculate_hybrid_scores(
        filtered_df, df_aspects, svd_model, active_profile,
        search_query=search_query, cosine_sim=cosine_sim, model_mode=active_model_mode
    )

    if hybrid_results.empty:
        st.warning("Không tìm thấy kết quả trùng khớp hoàn toàn. Hãy thử nới lỏng tiêu chí lọc.")
    else:
        if 'Hybrid_Score' in hybrid_results.columns:
            hybrid_results['Hybrid_Score'] = hybrid_results['Hybrid_Score'].fillna(0.5)

        c_title, c_sort = st.columns([2.2, 1.2])
        with c_title:
            st.markdown(f"### Khách sạn phù hợp với bạn ({len(hybrid_results)} kết quả)")
        with c_sort:
            sort_option = st.selectbox("Sắp xếp theo:", [
                "🎯 Độ phù hợp cao nhất",
                "💵 Giá: Thấp đến Cao",
                "💎 Giá: Cao đến Thấp",
                "⭐ Đánh giá cao nhất"
            ], index=0, label_visibility="collapsed")

        if sort_option == "💵 Giá: Thấp đến Cao":
            hybrid_results = hybrid_results.sort_values(by='Estimated_Price', ascending=True)
        elif sort_option == "💎 Giá: Cao đến Thấp":
            hybrid_results = hybrid_results.sort_values(by='Estimated_Price', ascending=False)
        elif sort_option == "⭐ Đánh giá cao nhất":
            hybrid_results = hybrid_results.sort_values(by='Score_Num', ascending=False)
        else:
            hybrid_results = hybrid_results.sort_values(by='Hybrid_Score', ascending=False)

        for idx, (_, row) in enumerate(hybrid_results.head(top_k).iterrows()):
            hid = row.get('Hotel_ID', idx)
            h_comments = df_comments[df_comments['Hotel ID'] == hid] if not df_comments.empty else pd.DataFrame()
            n_reviews = len(h_comments)
            
            raw_score = row.get('Hybrid_Score', 0.5)
            if pd.isna(raw_score) or np.isnan(raw_score):
                raw_score = 0.5
            match_pct = int(round(float(raw_score) * 100))
            
            score_display = row.get('Total_Score', row.get('Score_Num', 8.0))
            desc_full = str(row.get('Hotel_Description', '')).strip()
            desc_short = str(row.get('Clean_Desc', '')).strip()

            with st.container(border=True):
                col_img, col_info, col_act = st.columns([1.2, 2.5, 1.3], gap="medium")
                
                with col_img:
                    img_url = HOTEL_IMAGES[idx % len(HOTEL_IMAGES)]
                    st.markdown(f'''
                    <div style="position:relative;">
                        <img src="{img_url}" style="width:100%; height:165px; object-fit:cover; border-radius:8px;" alt="Hotel Photo">
                        <span style="position:absolute; top:8px; left:8px; background:rgba(34,197,94,0.95); color:white; padding:3px 9px; border-radius:12px; font-weight:700; font-size:0.76rem; box-shadow:0 2px 4px rgba(0,0,0,0.2);">🎯 Phù hợp {match_pct}%</span>
                    </div>
                    ''', unsafe_allow_html=True)

                with col_info:
                    st.markdown(f"<h4 style='margin:0 0 2px 0; font-size:1.1rem; color:var(--text-color); font-weight:700;'>{row['Clean_Name']} <span style='color:var(--agoda-star); font-size:0.95rem;'>{row['Star_Badge']}</span></h4>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:0.82rem; color:var(--text-color); opacity:0.75; margin-bottom:4px;'>📍 {row['Clean_Addr']}</div>", unsafe_allow_html=True)

                    if desc_short and len(desc_short) > 5:
                        st.markdown(f"<div style='font-size:0.8rem; color:var(--text-color); opacity:0.85; margin-bottom:6px; line-height:1.4;'>Mô tả: {desc_short[:95]}...</div>", unsafe_allow_html=True)

                    c_btn1, c_btn2, c_btn3 = st.columns(3)
                    with c_btn1:
                        with st.popover(f"💬 {n_reviews} review ▾", use_container_width=True):
                            st.markdown(f"#### 💬 Nhận xét thực tế ({n_reviews})")
                            if h_comments.empty:
                                st.caption("Chưa có bình luận chi tiết.")
                            else:
                                for _, c in h_comments.head(4).iterrows():
                                    nat = str(c.get('Nationality', ''))
                                    sc_val = c.get('Score', '')
                                    t_title = _clean_review_text(c.get('Title', ''))[:80]
                                    t_body = _clean_review_text(c.get('Positive', c.get('Comment', '')))[:220]
                                    t_neg = _clean_review_text(c.get('Negative', ''))[:120]
                                    sc_num = float(str(sc_val).replace(',', '.') or 0)
                                    sc_col = "#4ade80" if sc_num >= 7 else "#fbbf24"
                                    
                                    st.markdown(f"""
                                    <div style="border-left: 3px solid var(--border-hairline); padding: 6px 10px; margin-bottom: 8px; background: var(--secondary-background-color); border-radius: 4px;">
                                        <span style="font-weight:600; font-size:0.83rem; color:var(--text-color);">{nat}</span>
                                        <span style="color:{sc_col}; font-weight:700; float:right;">⭐ {sc_val}</span>
                                        {"<div style='font-weight:600; font-size:0.82rem; margin-top:2px; color:var(--text-color);'>" + t_title + "</div>" if t_title else ""}
                                        {"<div style='font-size:0.8rem; color:var(--text-color); opacity:0.85; margin-top:2px;'>👍 " + t_body + "</div>" if t_body else ""}
                                        {"<div style='font-size:0.8rem; color:var(--agoda-danger); margin-top:2px;'>👎 " + t_neg + "</div>" if t_neg else ""}
                                    </div>
                                    """, unsafe_allow_html=True)
                    with c_btn2:
                        with st.popover("📖 Chi tiết ▾", use_container_width=True):
                            st.markdown(f"#### {row['Clean_Name']}")
                            st.markdown(f"📍 **Địa chỉ:** {row['Clean_Addr']}")
                            st.markdown(f"⭐ **Hạng sao:** {row['Star_Badge']} ({row['Star_Num']} sao)")
                            st.markdown(f"📝 **Mô tả chi tiết:**\n\n{desc_full}")
                    with c_btn3:
                        with st.popover("🧮 Điểm AI ▾", use_container_width=True):
                            svd_r = row.get('SVD_Rating', 8.2)
                            svd_n = int(round(float(row.get('SVD_Norm', 0.5)) * 100))
                            asp_s = int(round(float(row.get('Aspect_Match_Score', 0.8)) * 100))
                            cos_s = int(round(float(row.get('Cosine_Score', 0.0)) * 100))
                            str_s = int(round(float(row.get('Star_Match_Score', 1.0)) * 100))
                            
                            st.markdown(f"#### 🧮 Phân Tích Điểm Đề Xuất Hybrid")
                            st.markdown(f"**Tổng điểm Phù Hợp: {match_pct}%**")
                            
                            st.markdown(f"👥 **Collaborative SVD (Đồng sở thích):** **{svd_r:.1f} / 10**")
                            st.progress(svd_n / 100.0, text=f"SVD Norm: {svd_n}%")
                            
                            st.markdown(f"🎯 **Aspect Match (Gu quốc tịch & khía cạnh):**")
                            st.progress(asp_s / 100.0, text=f"Aspect Score: {asp_s}%")
                            
                            st.markdown(f"📝 **Cosine Text Similarity (Mô tả NLP):**")
                            st.progress(cos_s / 100.0, text=f"Cosine Score: {cos_s}%")
                            
                            st.markdown(f"⭐ **Star & Location Match (Hạng sao & vị trí):**")
                            st.progress(str_s / 100.0, text=f"Star Match: {str_s}%")

                    # Row 4: Dynamic Badges
                    st.markdown(get_dynamic_badges(row, row['Star_Num']), unsafe_allow_html=True)

                # ── Column 3: Price & Action CTA (Right Column Rating Badge) ──
                with col_act:
                    score_val = safe_float_score(score_display, default=8.0)
                    if score_val is None:
                        rating_tag = "⚪ Chưa có đánh giá"
                        badge_bg = "rgba(128,128,128,0.12)"
                        badge_col = "var(--text-color)"
                        score_text_display = "Chưa đánh giá"
                    else:
                        score_text_display = f"{score_val:.1f} / 10"
                        if score_val >= 9.0:
                            rating_tag = "🟢 Tuyệt vời"
                            badge_bg = "rgba(34,197,94,0.15)"
                            badge_col = "#22c55e"
                        elif score_val >= 8.0:
                            rating_tag = "🔵 Rất tốt"
                            badge_bg = "rgba(59,130,246,0.15)"
                            badge_col = "#3b82f6"
                        else:
                            rating_tag = "🟡 Tốt"
                            badge_bg = "rgba(245,158,11,0.15)"
                            badge_col = "#f59e0b"

                    st.markdown(f"""
                    <div style='text-align:right; margin-bottom:6px;'>
                        <span style='background:{badge_bg}; color:{badge_col}; border:1px solid {badge_col}; padding:3px 8px; border-radius:8px; font-weight:700; font-size:0.82rem;'>
                            {rating_tag} ({score_text_display})
                        </span>
                    </div>
                    <div style='text-align:right; margin-bottom:8px;'>
                        <span style='font-size:1.3rem; font-weight:800; color:var(--agoda-primary);'>{row['Price_Display']}</span><br>
                        <span style='font-size:0.75rem; color:var(--text-color); opacity:0.7;'>/ đêm (đã gồm thuế)</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("🛎️ ĐẶT PHÒNG NGAY", key=f"book_{hid}_{idx}", type="primary", use_container_width=True):
                        st.balloons()
                        st.success(f"🎉 Bạn đã chọn **{row['Clean_Name']}** thành công!")
