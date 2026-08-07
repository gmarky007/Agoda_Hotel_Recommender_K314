# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_star_constraint(query_text):
    if not isinstance(query_text, str):
        return None
    match = re.search(r'(\d+)\s*(?:sao|star)', query_text, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            if 1 <= val <= 5:
                return val
        except Exception:
            pass
    return None

def compute_aspect_matrix(df_comments):
    if df_comments.empty:
        return pd.DataFrame()

    title_col = 'Title' if 'Title' in df_comments.columns else None
    titles = df_comments[title_col].fillna('').astype(str) if title_col else pd.Series([''] * len(df_comments))
    
    body_col = 'Body' if 'Body' in df_comments.columns else ('Comment' if 'Comment' in df_comments.columns else ('Positive' if 'Positive' in df_comments.columns else None))
    bodies = df_comments[body_col].fillna('').astype(str) if body_col else pd.Series([''] * len(df_comments))
    text_combined = (titles + " " + bodies).str.lower()
    
    scores = pd.to_numeric(df_comments['Score'].astype(str).str.replace(',', '.'), errors='coerce').fillna(8.0)

    p_clean = text_combined.str.contains(r'sạch|thơm|clean|dơ|bụi|vệ sinh', regex=True)
    p_loc = text_combined.str.contains(r'vị trí|trung tâm|gần biển|tiện lợi|location|beach', regex=True)
    p_pool = text_combined.str.contains(r'hồ bơi|bể bơi|pool|vô cực', regex=True)
    p_staff = text_combined.str.contains(r'nhân viên|phục vụ|nhiệt tình|lịch sự|staff|friendly', regex=True)
    p_food = text_combined.str.contains(r'ăn sáng|buffet|breakfast|ngon|đồ ăn', regex=True)

    temp_df = pd.DataFrame({
        'Hotel ID': df_comments['Hotel ID'].astype(str),
        'Score': scores,
        'Aspect_Clean': p_clean.astype(int) * scores,
        'Aspect_Loc': p_loc.astype(int) * scores,
        'Aspect_Pool': p_pool.astype(int) * scores,
        'Aspect_Staff': p_staff.astype(int) * scores,
        'Aspect_Food': p_food.astype(int) * scores,
        'Count_Clean': p_clean.astype(int),
        'Count_Loc': p_loc.astype(int),
        'Count_Pool': p_pool.astype(int),
        'Count_Staff': p_staff.astype(int),
        'Count_Food': p_food.astype(int),
    })

    grouped = temp_df.groupby('Hotel ID').agg({
        'Score': 'mean',
        'Aspect_Clean': 'sum',
        'Count_Clean': 'sum',
        'Aspect_Loc': 'sum',
        'Count_Loc': 'sum',
        'Aspect_Pool': 'sum',
        'Count_Pool': 'sum',
        'Aspect_Staff': 'sum',
        'Count_Staff': 'sum',
        'Aspect_Food': 'sum',
        'Count_Food': 'sum',
    }).reset_index()

    # Calculate average rating per aspect if mentioned, else default to overall score
    for asp in ['Clean', 'Loc', 'Pool', 'Staff', 'Food']:
        grouped[f'Rating_{asp}'] = np.where(
            grouped[f'Count_{asp}'] > 0,
            grouped[f'Aspect_{asp}'] / grouped[f'Count_{asp}'],
            grouped['Score']
        )
        grouped[f'Rating_{asp}'] = grouped[f'Rating_{asp}'].round(2)

    return grouped
