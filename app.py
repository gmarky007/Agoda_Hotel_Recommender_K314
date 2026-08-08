# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(
    page_title="Agoda Hotel Recommender | Nha Trang",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══ DESIGN TOKENS: CSS Variables (Dark/Light dual mode) ═══
st.markdown("""
<style>
    /* ── Light Mode ── */
    :root {
        --agoda-primary: #3b5998;
        --agoda-primary-light: #5b7ec2;
        --agoda-accent: #ff385c;
        --agoda-accent-hover: #e00b41;
        --agoda-success: #22c55e;
        --agoda-success-bg: rgba(34,197,94,0.12);
        --agoda-warning: #f59e0b;
        --agoda-danger: #ef4444;
        --agoda-danger-bg: rgba(239,68,68,0.10);
        --agoda-star: #eab308;
        --bg-canvas: #ffffff;
        --bg-surface: #f7f7f7;
        --bg-card: #ffffff;
        --bg-card-hover: #f0f5ff;
        --text-ink: #222222;
        --text-body: #3f3f3f;
        --text-muted: #6a6a6a;
        --text-on-primary: #ffffff;
        --border-hairline: #dddddd;
        --border-strong: #c1c1c1;
        --shadow-card: 0 1px 3px rgba(0,0,0,0.08);
        --radius-card: 12px;
        --radius-pill: 24px;
        --radius-btn: 8px;
    }
    /* ── Dark Mode ── */
    @media (prefers-color-scheme: dark) {
      :root {
        --agoda-primary: #5b8def;
        --agoda-primary-light: #7ba6f7;
        --agoda-accent: #ff5a7d;
        --agoda-accent-hover: #ff7a96;
        --agoda-success: #4ade80;
        --agoda-success-bg: rgba(74,222,128,0.12);
        --agoda-danger: #f87171;
        --agoda-danger-bg: rgba(248,113,113,0.10);
        --agoda-star: #facc15;
        --bg-canvas: #0e1117;
        --bg-surface: #1a1f2e;
        --bg-card: #1e2538;
        --bg-card-hover: #252d42;
        --text-ink: #fafafa;
        --text-body: #c9d1d9;
        --text-muted: #8b949e;
        --border-hairline: #30363d;
        --border-strong: #484f58;
        --shadow-card: 0 1px 4px rgba(0,0,0,0.3);
      }
    }

    /* ── Page Layout: 70% centered chiều ngang màn hình đồng nhất ── */
    .main .block-container,
    div[data-testid="stMainBlockContainer"],
    section.main > div.block-container {
        max-width: 70% !important;
        width: 70% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }

    /* ── Global Component Styles & Non-wrapping Buttons ── */
    .stButton>button, div[data-testid="stPopover"] > button {
        border-radius: var(--radius-btn) !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }
    div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }
    .stSelectbox label, .stTextInput label, .stSlider label {
        font-weight: 600 !important;
    }

    /* ── Primary Action Buttons (CTA) ── */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, var(--agoda-primary), #2b4379) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(59,89,152,0.3) !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2b4379, #1e3059) !important;
        box-shadow: 0 4px 10px rgba(59,89,152,0.4) !important;
        transform: translateY(-1px);
    }

    /* ── Secondary / Tab Toggle Buttons (Outline chìm nhẹ nhàng) ── */
    .stButton>button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid var(--border-hairline) !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
    }
    .stButton>button[kind="secondary"]:hover {
        border-color: var(--agoda-primary) !important;
        color: var(--agoda-primary) !important;
        background: rgba(59,89,152,0.05) !important;
    }

    /* ── Amenity Toggle Pill ── */
    .amenity-pill {
        display: inline-flex; align-items: center; justify-content: center;
        padding: 6px 14px; border-radius: var(--radius-pill);
        font-size: 0.82rem; font-weight: 600; cursor: pointer;
        transition: all 0.2s ease; margin: 2px 3px; border: 1.5px solid var(--border-hairline);
        color: var(--text-muted); background: transparent;
    }
    .amenity-pill.active {
        border-color: var(--agoda-success); color: var(--agoda-success);
        background: var(--agoda-success-bg);
    }

    /* ── Compact Filter Container & Widget Margins ── */
    div[data-testid="stVerticalBlock"] > div.element-container {
        margin-bottom: 0.25rem !important;
    }
    div[data-testid="stSlider"] {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
    div[data-testid="stSelectbox"] {
        margin-bottom: 0.1rem !important;
    }

    /* ── Compact card spacing ── */
    div[data-testid="stExpander"] {
        margin-top: -8px !important;
    }

    /* ── Author card ── */
    .author-card {
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        padding: 12px 14px; border-radius: 10px; color: #94a3b8;
        font-size: 0.8rem; line-height: 1.5;
    }
    .author-card b { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

import importlib
import src.utils.nlp_cleaner
import src.models.user_profiler
import src.pipeline.hybrid_engine
importlib.reload(src.utils.nlp_cleaner)
importlib.reload(src.models.user_profiler)
importlib.reload(src.pipeline.hybrid_engine)

from modules import agoda_booking, partner_insights
importlib.reload(agoda_booking)
importlib.reload(partner_insights)

m_bu = importlib.import_module("modules.1_business_understanding")
m_du = importlib.import_module("modules.2_data_understanding")
m_dp = importlib.import_module("modules.3_data_preparation")
m_md = importlib.import_module("modules.4_modeling")
m_ev = importlib.import_module("modules.5_evaluation")
m_ai = importlib.import_module("modules.6_author_info")

importlib.reload(m_bu)
importlib.reload(m_du)
importlib.reload(m_dp)
importlib.reload(m_md)
importlib.reload(m_ev)
importlib.reload(m_ai)

# ═══ SIDEBAR ═══

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "user_nationality" not in st.session_state:
    st.session_state.user_nationality = None

st.sidebar.image("assets/logo.png", width=150)
st.sidebar.caption("Hệ thống Đề xuất Khách sạn thông minh")

st.sidebar.divider()

nav_choice = st.sidebar.radio(
    "📌 Điều hướng:",
    ["🏨 Tìm & Đặt Phòng", "📊 Báo Cáo Dành Cho Đối Tác", "📑 Quy Trình CRISP-DM"],
    index=0
)

st.sidebar.divider()

# Author info
st.sidebar.markdown("""
<div class="author-card">
    <b>👨‍💻 Nhóm tác giả thực hiện:</b><br>
    • Nguyễn Văn Nam<br>
    • Lê Văn Lưu<br>
    <b>Lớp:</b> K314 — Đồ Án Tốt Nghiệp Data Science & Machine Learning
</div>
""", unsafe_allow_html=True)

# Route
if nav_choice == "🏨 Tìm & Đặt Phòng":
    agoda_booking.render()
elif nav_choice == "📊 Báo Cáo Dành Cho Đối Tác":
    partner_insights.render()
else:
    st.markdown("""
    <div style="background:var(--bg-card); padding:15px; border-radius:12px; border:1px solid var(--agoda-primary-light); margin-bottom:20px; box-shadow:0 4px 6px rgba(0,0,0,0.04);">
        <h4 style="margin-top:0; color:var(--agoda-primary); text-align:center; font-weight:700; font-size:1.05rem;">🔄 QUY TRÌNH CHUẨN CRISP-DM TRONG DỰ ÁN ĐỀ XUẤT LAI AGODA</h4>
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-top:10px;">
            <div style="text-align:center; flex:1; min-width:90px; padding:6px; background:rgba(59,130,246,0.06); border-radius:8px; border:1px dashed var(--agoda-primary-light);">
                <span style="font-size:1.2rem;">🎯</span><br>
                <span style="font-size:0.75rem; font-weight:700; color:var(--text-color);">1. Business Under.</span>
            </div>
            <div style="color:var(--agoda-primary-light); font-weight:700;">➡️</div>
            <div style="text-align:center; flex:1; min-width:90px; padding:6px; background:rgba(59,130,246,0.06); border-radius:8px; border:1px dashed var(--agoda-primary-light);">
                <span style="font-size:1.2rem;">🔍</span><br>
                <span style="font-size:0.75rem; font-weight:700; color:var(--text-color);">2. Data Under.</span>
            </div>
            <div style="color:var(--agoda-primary-light); font-weight:700;">➡️</div>
            <div style="text-align:center; flex:1; min-width:90px; padding:6px; background:rgba(59,130,246,0.06); border-radius:8px; border:1px dashed var(--agoda-primary-light);">
                <span style="font-size:1.2rem;">🛠️</span><br>
                <span style="font-size:0.75rem; font-weight:700; color:var(--text-color);">3. Data Prep.</span>
            </div>
            <div style="color:var(--agoda-primary-light); font-weight:700;">➡️</div>
            <div style="text-align:center; flex:1; min-width:90px; padding:6px; background:rgba(59,130,246,0.06); border-radius:8px; border:1px dashed var(--agoda-primary-light);">
                <span style="font-size:1.2rem;">🤖</span><br>
                <span style="font-size:0.75rem; font-weight:700; color:var(--text-color);">4. Modeling</span>
            </div>
            <div style="color:var(--agoda-primary-light); font-weight:700;">➡️</div>
            <div style="text-align:center; flex:1; min-width:90px; padding:6px; background:rgba(59,130,246,0.06); border-radius:8px; border:1px dashed var(--agoda-primary-light);">
                <span style="font-size:1.2rem;">📊</span><br>
                <span style="font-size:0.75rem; font-weight:700; color:var(--text-color);">5. Evaluation</span>
            </div>
            <div style="color:var(--agoda-primary-light); font-weight:700;">➡️</div>
            <div style="text-align:center; flex:1; min-width:90px; padding:6px; background:rgba(59,130,246,0.06); border-radius:8px; border:1px dashed var(--agoda-primary-light);">
                <span style="font-size:1.2rem;">👨‍💻</span><br>
                <span style="font-size:0.75rem; font-weight:700; color:var(--text-color);">6. Author Info</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "1️⃣ Business Understanding",
        "2️⃣ Data Understanding",
        "3️⃣ Data Preparation",
        "4️⃣ Modeling",
        "5️⃣ Evaluation",
        "6️⃣ Info Tác giả"
    ])
    with t1:
        m_bu.render()
    with t2:
        m_du.render()
    with t3:
        m_dp.render()
    with t4:
        m_md.render()
    with t5:
        m_ev.render()
    with t6:
        m_ai.render()

