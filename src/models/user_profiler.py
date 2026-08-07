# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

NATIONALITY_PROFILES = {
    "Hàn Quốc": {
        "flag": "🇰🇷",
        "desc": "Khách Hàn Quốc (Đi Cặp đôi / Resort)",
        "pref_summary": "Ưu tiên Resort mặt biển, Hồ bơi vô cực, View đẹp chụp ảnh, Gần quán ăn ngon",
        "aspect_weights": {"Pool": 0.35, "Loc": 0.25, "Clean": 0.20, "Staff": 0.10, "Food": 0.10},
        "target_price_min": 1200000,
        "target_price_max": 3500000,
        "target_star": 5.0,
        "reviewer_id": None  # Will be injected from real data
    },
    "Việt Nam": {
        "flag": "🇻🇳",
        "desc": "Khách Việt Nam (Gia đình / Tiết kiệm)",
        "pref_summary": "Ưu tiên Giá tốt hợp lý, Buffet sáng phong phú, Gần trung tâm & Chợ đêm",
        "aspect_weights": {"Food": 0.30, "Loc": 0.25, "Clean": 0.20, "Staff": 0.15, "Pool": 0.10},
        "target_price_min": 450000,
        "target_price_max": 1200000,
        "target_star": 3.0,
        "reviewer_id": None
    },
    "Hoa Kỳ": {
        "flag": "🇺🇸",
        "desc": "Khách Mỹ / Châu Âu (Solo / Trải nghiệm)",
        "pref_summary": "Ưu tiên Không gian yên tĩnh, Nhân viên tiếng Anh tốt, Vệ sinh sạch sẽ & Tour trải nghiệm",
        "aspect_weights": {"Staff": 0.30, "Clean": 0.30, "Loc": 0.20, "Food": 0.10, "Pool": 0.10},
        "target_price_min": 600000,
        "target_price_max": 2000000,
        "target_star": 4.0,
        "reviewer_id": None
    },
    "Úc": {
        "flag": "🇦🇺",
        "desc": "Khách Úc (Nghỉ dưỡng gia đình)",
        "pref_summary": "Ưu tiên Căn hộ sát biển, Hồ bơi rộng, Phục vụ thân thiện & Dịch vụ Spa",
        "aspect_weights": {"Pool": 0.30, "Clean": 0.25, "Staff": 0.25, "Loc": 0.10, "Food": 0.10},
        "target_price_min": 1000000,
        "target_price_max": 2800000,
        "target_star": 4.0,
        "reviewer_id": None
    }
}

# Nationality mapping: dataset value → NATIONALITY_PROFILES key
_NAT_MAP = {
    "Hàn Quốc": "Hàn Quốc",
    "Việt Nam": "Việt Nam",
    "Hoa Kỳ": "Hoa Kỳ",
    "Úc": "Úc",
}

def inject_reviewer_ids(df_comments):
    """Inject real reviewer_ids from dataset into NATIONALITY_PROFILES.
    Must be called once after loading df_comments."""
    if df_comments.empty:
        return
    for nat_data, profile_key in _NAT_MAP.items():
        sub = df_comments[df_comments['Nationality'] == nat_data]
        if not sub.empty and profile_key in NATIONALITY_PROFILES:
            NATIONALITY_PROFILES[profile_key]['reviewer_id'] = str(sub.iloc[0]['Reviewer ID'])


def load_real_reviewer_personas(df_comments):
    """Load real reviewer personas with actual reviewer_ids from dataset.
    Returns dict keyed by reviewer_id string."""
    if df_comments.empty:
        return {}

    # Also inject into NATIONALITY_PROFILES as side-effect
    inject_reviewer_ids(df_comments)

    personas = {}
    for nat_data, profile_key in _NAT_MAP.items():
        sub = df_comments[df_comments['Nationality'] == nat_data]
        if not sub.empty and profile_key in NATIONALITY_PROFILES:
            nat_info = NATIONALITY_PROFILES[profile_key]
            # Pick top 3 most active reviewers for this nationality
            top_reviewers = sub.groupby('Reviewer ID').size().sort_values(ascending=False).head(3)
            for rid in top_reviewers.index:
                rid_str = str(rid)
                row = sub[sub['Reviewer ID'] == rid].iloc[0]
                personas[rid_str] = {
                    "reviewer_id": rid_str,
                    "reviewer_name": str(row.get('Reviewer Name', 'Khách du lịch')),
                    "nationality": profile_key,
                    "flag": nat_info["flag"],
                    "group_name": str(row.get('Group Name', 'Khách du lịch')),
                    "desc": f"{nat_info['flag']} {row.get('Reviewer Name', 'Khách')} ({profile_key})",
                    "pref_summary": nat_info["pref_summary"],
                    "aspect_weights": nat_info["aspect_weights"],
                    "target_price_min": nat_info["target_price_min"],
                    "target_price_max": nat_info["target_price_max"],
                    "target_star": nat_info["target_star"],
                    "review_count": int(top_reviewers[rid])
                }

    return personas
