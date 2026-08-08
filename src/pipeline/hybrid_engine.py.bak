# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from src.utils.nlp_cleaner import extract_star_constraint, clean_text

def parse_star_numeric(rank_str, desc_str="", price=0):
    s = str(rank_str).strip().lower()
    
    # Trích xuất số đứng trước chữ sao/star (ví dụ: '4 sao trên 5' -> 4.0)
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:sao|star)', s)
    if match:
        try:
            val = float(match.group(1))
            if 1.0 <= val <= 5.0:
                return val
        except Exception:
            pass
            
    if '5 sao' in s or '5 star' in s: return 5.0
    if '4 sao' in s or '4 star' in s or '4.5' in s: return 4.0
    if '3 sao' in s or '3 star' in s or '3.5' in s: return 3.0
    if '2 sao' in s or '2 star' in s or '2.5' in s: return 2.0
    if '1 sao' in s or '1 star' in s: return 1.0

    d = str(desc_str).lower()
    match_d = re.search(r'(\d+(?:\.\d+)?)\s*(?:sao|star)', d)
    if match_d:
        try:
            val = float(match_d.group(1))
            if 1.0 <= val <= 5.0:
                return val
        except Exception:
            pass

    # Infer equivalent tier from price range if unrated
    if price >= 1800000: return 5.0
    if price >= 1100000: return 4.0
    if price >= 450000: return 3.0
    return 2.0

def get_star_badge(rank_str, desc_str="", price=0):
    s = str(rank_str).strip().lower()
    if '5' in s or '4' in s or '3' in s or '2' in s or '1' in s:
        val = int(round(parse_star_numeric(rank_str, desc_str, price)))
        return f"{val}⭐"
    
    # Unrated accommodation mapping
    val = int(round(parse_star_numeric(rank_str, desc_str, price)))
    return f"{val}⭐ (Tương đương)"


PROPERTY_TYPE_MAP = {
    'villa': r'villa|biệt thự',
    'biệt thự': r'villa|biệt thự',
    'homestay': r'homestay|nhà riêng|nhà dân|house|home|studio',
    'nhà riêng': r'homestay|nhà riêng|nhà dân|house|home',
    'resort': r'resort|khu nghỉ dưỡng',
    'khu nghỉ dưỡng': r'resort|khu nghỉ dưỡng',
    'căn hộ': r'căn hộ|apartment|condotel|chung cư|apt',
    'apartment': r'căn hộ|apartment|condotel|chung cư',
    'khách sạn': r'khách sạn|hotel',
    'hotel': r'khách sạn|hotel'
}

AMENITY_MAP = {
    'hồ bơi': r'hồ bơi|bể bơi|pool|swimming',
    'bể bơi': r'hồ bơi|bể bơi|pool|swimming',
    'pool': r'hồ bơi|bể bơi|pool|swimming',
    'biển': r'biển|bãi biển|beach|ocean|sea|view biển',
    'gần biển': r'biển|bãi biển|beach|ocean|sea|view biển',
    'bữa sáng': r'bữa sáng|buffet|breakfast|ăn sáng',
    'buffet': r'bữa sáng|buffet|breakfast|ăn sáng',
    'spa': r'spa|massage|sauna|xông hơi',
    'gym': r'gym|fitness|thể thao',
    'sân bay': r'sân bay|airport|đưa đón',
    'bếp': r'bếp|kitchen|nấu ăn',
    'máy giặt': r'máy giặt|laundry|washing',
    'ban công': r'ban công|balcony',
    'thú cưng': r'thú cưng|pet|chó mèo',
    'bãi đậu xe': r'đỗ xe|bãi đậu xe|bãi xe|parking',
    'đỗ xe': r'đỗ xe|bãi đậu xe|bãi xe|parking',
    'trung tâm': r'trung tâm|center|central'
}

def parse_nlp_query_constraints(df, query):
    """
    Stage 1: Hard Constraint Filter (AND logic)
    Extracts star ratings, property types, and amenities from natural language query
    and strictly filters the dataframe.
    """
    if not query or not query.strip():
        return df.copy()
        
    q_clean = clean_text(query)
    res = df.copy()
    
    # 1. Star Constraint
    q_star = extract_star_constraint(query)
    if q_star is not None:
        # Tightened star range: e.g. for 3-star query, matches 2.8 to 3.4; for 5-star, 4.8 to 5.0
        res = res[(res['Star_Num'] >= q_star - 0.2) & (res['Star_Num'] <= q_star + 0.4)]
        
    # 2. Property Types (OR if multiple specified, e.g. villa hoặc homestay)
    matched_types = [pat for k, pat in PROPERTY_TYPE_MAP.items() if k in q_clean]
    if matched_types:
        combined_type_pat = '|'.join(set(matched_types))
        res = res[res['Hotel_Description'].str.contains(combined_type_pat, case=False, na=False, regex=True) | res['Hotel_Name'].str.contains(combined_type_pat, case=False, na=False, regex=True)]
        
    # 3. Mandatory Amenities (STRICT AND LOGIC)
    matched_amenities = []
    for k, pat in AMENITY_MAP.items():
        if k in q_clean and pat not in matched_amenities:
            matched_amenities.append(pat)
            
    for pat in matched_amenities:
        res = res[res['Hotel_Description'].str.contains(pat, case=False, na=False, regex=True)]
        
    return res


def compute_text_relevance(search_query, hotel_name, hotel_desc):
    """Compute keyword-based relevance score between search query and hotel info.
    Returns float in [0, 1]."""
    if not search_query or not search_query.strip():
        return 0.0
    
    query_clean = clean_text(search_query)
    hotel_text = clean_text(str(hotel_name) + " " + str(hotel_desc))
    
    # Remove star-related words from query for relevance (already handled by star filter)
    query_clean = re.sub(r'\d+\s*(?:sao|star)', '', query_clean).strip()
    if not query_clean:
        return 0.0
    
    stop_tokens = {'phòng', 'ngủ', 'm²', 'm2', 'có', 'ở', 'tại', 'và', 'cho', 'với', 'cần', 'tìm', 'khách', 'sạn'}
    query_tokens = set(query_clean.split()) - stop_tokens
    hotel_tokens = set(hotel_text.split())
    
    if not query_tokens:
        query_tokens = set(query_clean.split())
    
    # Count matching tokens
    matches = query_tokens & hotel_tokens
    
    # Check specific numbers in query (e.g., 157)
    num_penalty = False
    for word in query_clean.split():
        if word.isdigit() and len(word) >= 3: # e.g. 157
            if word not in hotel_tokens:
                num_penalty = True
                break

    if num_penalty:
        return 0.0

    # Jaccard-like relevance = matches / query_tokens
    relevance = len(matches) / len(query_tokens)
    
    # Boost for Vietnamese keyword synonyms
    boost_pairs = {
        'biển': ['beach', 'bãi', 'ocean', 'sea'],
        'bơi': ['pool', 'hồ'],
        'sạch': ['clean', 'vệ', 'sinh'],
        'sáng': ['buffet', 'breakfast', 'ăn'],
        'trung': ['tâm', 'center', 'central'],
        'gần': ['tiện', 'thuận'],
        'đẹp': ['view', 'cảnh', 'panorama'],
        'sang': ['trọng', 'luxury', 'cao', 'cấp'],
        'spa': ['massage', 'thư', 'giãn'],
        'yên': ['tĩnh', 'quiet', 'peaceful'],
    }
    
    bonus = 0.0
    for q_tok in query_tokens:
        for key, synonyms in boost_pairs.items():
            if q_tok == key or q_tok in synonyms:
                for syn in [key] + synonyms:
                    if syn in hotel_tokens and syn not in matches:
                        bonus += 0.1
                        break
    
    return min(1.0, relevance + bonus)


def calculate_hybrid_scores(df_hotels, df_aspects, svd_model, reviewer_profile, 
                            search_query="", cosine_sim=None, alpha=0.35):
    """Calculate 4-factor hybrid recommendation scores.
    
    Formula: S = 0.35*Cosine + 0.35*SVD + 0.15*Aspect + 0.15*Star
    
    Args:
        df_hotels: DataFrame of hotels to score
        df_aspects: DataFrame of NLP aspect ratings per hotel
        svd_model: Surprise SVD model for collaborative filtering
        reviewer_profile: Dict with reviewer_id, aspect_weights, target_star
        search_query: User's search text for keyword relevance
        cosine_sim: Pre-computed cosine similarity matrix (740x740) — reserved for future use
    """
    if df_hotels.empty:
        return df_hotels
        
    res_df = df_hotels.copy()
    
    # 1. Text Relevance / Cosine Score
    q_clean = clean_text(search_query) if search_query else ""
    q_no_star = re.sub(r'\d+\s*(?:sao|star)', '', q_clean).strip()
    is_pure_star_query = bool(q_clean and not q_no_star)

    if search_query and search_query.strip() and not is_pure_star_query:
        res_df['Cosine_Score'] = [
            compute_text_relevance(search_query, row.get('Hotel_Name', ''), row.get('Hotel_Description', ''))
            for _, row in res_df.iterrows()
        ]
        max_cos = res_df['Cosine_Score'].max()
        if max_cos > 0:
            res_df['Cosine_Score'] = res_df['Cosine_Score'] / max_cos
        else:
            res_df['Cosine_Score'] = 0.0
    else:
        res_df['Cosine_Score'] = 0.0

    # 2. Collaborative Surprise SVD Score
    reviewer_id = str(reviewer_profile.get('reviewer_id', '101')) if reviewer_profile else '101'
    if reviewer_id == 'None' or not reviewer_id:
        reviewer_id = '101'
        
    svd_scores = []
    for _, row in res_df.iterrows():
        h_id = str(row.get('Hotel_ID', ''))
        if svd_model is not None and h_id:
            try:
                est = svd_model.predict(reviewer_id, h_id).est
            except Exception:
                est = 8.2
        else:
            est = 8.4
        svd_scores.append(round(est, 2))
        
    res_df['SVD_Rating'] = svd_scores
    svd_arr = np.array(svd_scores)
    s_min, s_max = svd_arr.min(), svd_arr.max()
    if s_max > s_min:
        res_df['SVD_Norm'] = (svd_arr - s_min) / (s_max - s_min)
    else:
        res_df['SVD_Norm'] = 0.5
    
    # 3. Demographic & Aspect Match Score
    aspect_weights = reviewer_profile.get('aspect_weights', {"Pool": 0.2, "Loc": 0.2, "Clean": 0.2, "Staff": 0.2, "Food": 0.2}) if reviewer_profile else {}
    
    if not df_aspects.empty:
        merge_cols = ['Hotel ID', 'Rating_Clean', 'Rating_Loc', 'Rating_Pool', 'Rating_Staff', 'Rating_Food']
        available_cols = [c for c in merge_cols if c in df_aspects.columns]
        res_df = res_df.merge(df_aspects[available_cols], left_on='Hotel_ID', right_on='Hotel ID', how='left')
        if 'Hotel ID' in res_df.columns:
            res_df.drop(columns=['Hotel ID'], inplace=True)
    
    aspect_match_scores = []
    for _, row in res_df.iterrows():
        r_clean = 8.0 if pd.isna(row.get('Rating_Clean')) else float(row.get('Rating_Clean', 8.0))
        r_loc = 8.0 if pd.isna(row.get('Rating_Loc')) else float(row.get('Rating_Loc', 8.0))
        r_pool = 8.0 if pd.isna(row.get('Rating_Pool')) else float(row.get('Rating_Pool', 8.0))
        r_staff = 8.0 if pd.isna(row.get('Rating_Staff')) else float(row.get('Rating_Staff', 8.0))
        r_food = 8.0 if pd.isna(row.get('Rating_Food')) else float(row.get('Rating_Food', 8.0))

        score_sum = (
            aspect_weights.get('Clean', 0.2) * (r_clean / 10.0) +
            aspect_weights.get('Loc', 0.2) * (r_loc / 10.0) +
            aspect_weights.get('Pool', 0.2) * (r_pool / 10.0) +
            aspect_weights.get('Staff', 0.2) * (r_staff / 10.0) +
            aspect_weights.get('Food', 0.2) * (r_food / 10.0)
        )
        aspect_match_scores.append(round(score_sum, 4))
        
    res_df['Aspect_Match_Score'] = aspect_match_scores
    
    # 4. Star Match Score
    target_star = reviewer_profile.get('target_star', 3.0) if reviewer_profile else 3.0
    star_match_scores = []
    for _, row in res_df.iterrows():
        star_val = parse_star_numeric(row.get('Hotel_Rank', ''), row.get('Hotel_Description', ''), row.get('Estimated_Price', 0))
        diff = abs(star_val - target_star)
        match_s = max(0.0, 1.0 - (diff * 0.4))
        star_match_scores.append(round(match_s, 4))
        
    res_df['Star_Match_Score'] = star_match_scores

    # 5. Dynamic Weight Assignment
    if is_pure_star_query:
        # Pure star query -> Weight shifted heavily to Aspect (50%) and SVD (30%)
        w_cos, w_svd, w_asp, w_star = 0.00, 0.30, 0.50, 0.20
    else:
        # Descriptive query or empty -> Standard hybrid weights
        w_cos, w_svd, w_asp, w_star = 0.25, 0.15, 0.30, 0.30

    res_df['Hybrid_Score'] = (
        w_cos * res_df['Cosine_Score'] +
        w_svd * res_df['SVD_Norm'] +
        w_asp * res_df['Aspect_Match_Score'] +
        w_star * res_df['Star_Match_Score']
    ).round(4)
    
    return res_df.sort_values(by='Hybrid_Score', ascending=False)
