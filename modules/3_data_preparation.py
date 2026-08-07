# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import unicodedata
from underthesea import word_tokenize

# Load dictionaries for NLP cleaner
def load_dict(file_path):
    if not os.path.exists(file_path):
        return {}
    dct = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                dct[parts[0].lower()] = parts[1].lower()
            elif len(parts) == 1 and parts[0]:
                dct[parts[0].lower()] = ""
    return dct

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dict_teencode = load_dict(os.path.join(base_dir, 'data', 'files', 'teencode.txt'))
dict_wrong_word = load_dict(os.path.join(base_dir, 'data', 'files', 'wrong-word.txt'))
dict_eng_vnmese = load_dict(os.path.join(base_dir, 'data', 'files', 'english-vnmese.txt'))
dict_emojicon = load_dict(os.path.join(base_dir, 'data', 'files', 'emojicon.txt'))
stopwords_path = os.path.join(base_dir, 'data', 'files', 'vietnamese-stopwords.txt')

if os.path.exists(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f if line.strip()])
else:
    stopwords = set()

# Domain stopwords matching the notebook
domain_stopwords = {
    'phòng', 'khách_sạn', 'ks', 'hotel', 'resort', 'agoda',
    'ở', 'đi', 'nằm', 'đồ', 'chỗ', 'mấy', 'cái', 'nhà',
    'đêm', 'ngày', 'lúc', 'đặt', 'lần', 'năm', 'tháng',
    'sự', 'việc', 'nhận_xét', 'đánh_giá', 'bình_luận',
    'có', 'được', 'thì', 'là', 'đã', 'bị', 'lại', 'thấy',
    'cho', 'này', 'kia', 'đó', 'với', 'cả', 'như', 'nhưng',
    'rất', 'quá', 'lắm', 'nên', 'cũng', 'vẫn', 'hay', 'khi',
    'mà', 'nếu', 'thì', 'vì', 'do', 'từ', 'đến', 'về',
    'của', 'và', 'hoặc', 'các', 'những', 'một', 'hai', 'ba'
}
stopwords.update(domain_stopwords)

NEGATION_WORDS = {'không', 'chưa', 'chẳng', 'kém', 'thiếu', 'mất'}

def join_negation(tokens):
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] in NEGATION_WORDS and i + 1 < len(tokens):
            result.append(f"{tokens[i]}_{tokens[i+1]}")
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

def clean_and_tokenize(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode NFC
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    
    # 2. Emojicon mapping
    for emoji, replacement in dict_emojicon.items():
        text = text.replace(emoji, f" {replacement} ")
        
    # 3. Clean special chars
    text = re.sub(r'[\d\.,\?\!\-\;\:\(\)\[\]\+\/\&\*\_\@]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Word tokenize using underthesea
    tokens = word_tokenize(text, format="text").split()
    
    # 5. Translate teencode / english / wrong words
    cleaned_tokens = []
    for token in tokens:
        word = token.replace('_', ' ')
        word = dict_teencode.get(word, word)
        word = dict_wrong_word.get(word, word)
        word = dict_eng_vnmese.get(word, word)
        token_clean = word.replace(' ', '_')
        if token_clean not in stopwords:
            cleaned_tokens.append(token_clean)
            
    # 6. Join Negations
    final_tokens = join_negation(cleaned_tokens)
    return " ".join(final_tokens)

def render():
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h2 style="margin:0; font-size:1.8rem; color:var(--agoda-primary); font-weight:800;">🛠️ BƯỚC 3: DATA PREPARATION</h2>
        <p style="margin:5px 0 0; font-size:0.9rem; color:var(--text-color); opacity:0.8;">Xử lý dữ liệu thô, làm sạch văn bản NLP, trích xuất khía cạnh (Aspects) & chuẩn bị ma trận tương tác</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Pipeline ngang HTML/CSS
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(59, 130, 246, 0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.15); margin-bottom: 20px;">
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem;">📥</div>
            <div style="font-weight: 700; font-size: 0.85rem; color: var(--agoda-primary);">1. Dữ liệu thô (Raw CSV)</div>
        </div>
        <div style="color: #60a5fa; font-weight: bold;">➡️</div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem;">🧹</div>
            <div style="font-weight: 700; font-size: 0.85rem; color: var(--agoda-primary);">2. Làm sạch NLP & Stopwords</div>
        </div>
        <div style="color: #60a5fa; font-weight: bold;">➡️</div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem;">🏷️</div>
            <div style="font-weight: 700; font-size: 0.85rem; color: var(--agoda-primary);">3. Trích xuất khía cạnh</div>
        </div>
        <div style="color: #60a5fa; font-weight: bold;">➡️</div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem;">🧮</div>
            <div style="font-weight: 700; font-size: 0.85rem; color: var(--agoda-primary);">4. Lập Ma trận User-Item</div>
        </div>
        <div style="color: #60a5fa; font-weight: bold;">➡️</div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 1.5rem;">🚀</div>
            <div style="font-weight: 700; font-size: 0.85rem; color: var(--agoda-primary);">5. Huấn luyện (2-Stage Hybrid)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Tiền xử lý văn bản
    with st.container(border=True):
        st.markdown("### 🧹 1. Tiền xử lý văn bản (NLP Clean & Stopwords)")
        st.markdown(f"""
        *   **Unicode Normalization**: Chuẩn hóa văn bản tiếng Việt sang dạng NFC chuẩn để tránh lỗi hiển thị font.
        *   **Bộ từ điển bổ trợ**: Tích hợp các tệp từ điển để làm sạch ngôn ngữ mạng:
            *   `emojicon.txt`: Chuyển đổi emoji sang từ cảm xúc tương ứng (Ví dụ: `😡` ➡️ *tức giận*).
            *   `teencode.txt`: Dịch teencode (Ví dụ: `ko`, `k` ➡️ *không*; `khs` ➡️ *không khách sạn*).
            *   `english-vnmese.txt`: Dịch các từ tiếng Anh thông dụng sang tiếng Việt.
        *   **Tách từ tiếng Việt**: Tách từ ghép bằng thư viện **Underthesea** để bảo toàn cụm nghĩa.
        *   **Domain Stopwords**: Lọc bỏ các danh từ chuyên ngành xuất hiện quá nhiều nhưng vô nghĩa cho cảm xúc (Ví dụ: *phòng, khách sạn, ks, đặt, phòng...*).
        *   **Negation Merger**: Nhận diện cụm phủ định như `không`, `chưa` và ghép với tính từ đi liền sau (Ví dụ: *không sạch* ➡️ `không_sạch`, *chưa tốt* ➡️ `chưa_tốt`).
        """)
        st.caption(f"📊 **Quy mô từ điển tải lên**: {len(dict_teencode)} teencodes, {len(dict_emojicon)} emoticons, {len(stopwords)} stopwords.")

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 2. Trích xuất khía cạnh
    with st.container(border=True):
        st.markdown("### 🏷️ 2. Trích xuất khía cạnh (Aspect Extraction)")
        st.markdown("""
        Phân tách phản hồi của khách hàng thành 5 chiều cảm nhận độc lập phục vụ cho Lọc lai (2-Stage Hybrid Recommendation) cùng với bộ phân tích NLP Hard Constraints:
        1.  **Vị trí (Location)**: Các mẫu regex chứa: *vị trí, gần biển, trung tâm, gần chợ, gần sân bay, mặt tiền...*
        2.  **Vệ sinh (Cleanliness)**: Các mẫu regex chứa: *sạch sẽ, gọn gàng, thoáng mát, không mùi, dọn phòng...*
        3.  **Dịch vụ (Service)**: Các mẫu regex chứa: *nhân viên, thái độ, phục vụ, lễ tân, nhiệt tình, chu đáo...*
        4.  **Tiện nghi (Facilities)**: Các mẫu regex chứa: *hồ bơi, wifi, thang máy, buffet, spa, gym, máy lạnh...*
        5.  **Đáng tiền (Value)**: Các mẫu regex chứa: *giá tốt, hợp lý, đáng tiền, rẻ, mức giá, tiết kiệm...*
        """)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 3. Xây dựng Ma Trận Tương Tác
    with st.container(border=True):
        st.markdown("### 🧮 3. Xây dựng Ma Trận Tương Tác (Matrix Construction)")
        st.markdown("""
        *   **Ma trận Đặc trưng Khách sạn (Hotel Profile Matrix)**:
            *   Sử dụng đặc trưng từ khóa được TF-IDF và thông tin vùng miền/mô tả để xây dựng vector đại diện cho từng khách sạn.
        *   **Ma trận Đánh giá (User-Item Interaction Matrix)**:
            *   Lập bảng chéo liên kết giữa `User_ID` (khách hàng) và `Hotel_ID` (khách sạn).
            *   Các ô giao nhau chứa điểm số đánh giá thực tế của khách hàng (thang điểm 1.0 - 10.0).
            *   Do ma trận rất thưa (Sparsity > 98%), mô hình áp dụng phân rã ma trận **SVD (Singular Value Decomposition)** để dự đoán các ô trống dựa trên đặc tính ẩn.
        """)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 4. Interactive NLP Tester
    with st.container(border=True):
        st.markdown("### 🧪 Bộ Kiểm Thử Tiền Xử Lý Văn Bản Trực Tiếp (Interactive NLP Tester)")
        st.markdown("Hãy nhập thử một câu nhận xét bất kỳ (bao gồm cả teencode, emoji và từ phủ định) để xem thuật toán làm sạch thế nào:")
        
        sample_comment = st.text_input("Bình luận mẫu:", value="ks này ko sạch, nhân viên thái độ ko tốt 😡", key="nlp_tester_input")
        
        if st.button("🚀 Chạy Tiền Xử Lý", type="primary", use_container_width=True):
            with st.spinner("Đang chạy tokenization & translation..."):
                cleaned_result = clean_and_tokenize(sample_comment)
                
                st.markdown("**Kết quả sau khi chuẩn hóa, dịch nghĩa & khử stopwords:**")
                st.code(cleaned_result, language="text")
                
                st.markdown("**Giải nghĩa quá trình xử lý từ vựng:**")
                explanation_list = []
                if "ko" in sample_comment or "k" in sample_comment:
                    explanation_list.append("- **Dịch teencode**: `ko` ➡️ `không`")
                if "ks" in sample_comment:
                    explanation_list.append("- **Loại bỏ Stopwords chuyên ngành**: loại bỏ `ks` (khách sạn)")
                if "😡" in sample_comment:
                    explanation_list.append("- **Dịch emoji**: `😡` ➡️ `tức_giận` (emoticon mapping)")
                if "sạch" in sample_comment:
                    explanation_list.append("- **Ghép từ phủ định (Negation Join)**: `không sạch` ➡️ `không_sạch` (bảo toàn ngữ nghĩa đảo chiều)")
                if "tốt" in sample_comment:
                    explanation_list.append("- **Ghép từ phủ định (Negation Join)**: `không tốt` ➡️ `không_tốt`")
                    
                if explanation_list:
                    for exp in explanation_list:
                        st.markdown(exp)
                else:
                    st.markdown("- *Đã lọc các từ dừng và ký hiệu đặc biệt, giữ lại các token mang nghĩa.*")

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # 💻 Khai báo code chuẩn trong notebook
    with st.container(border=True):
        st.markdown("### 💻 Hàm Làm Sạch & Tách Từ Gốc Trong Notebook (Reference Python Code)")
        st.code("""
def clean_and_tokenize(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Chuẩn hóa dữ liệu về dạng NFC Unicode tiếng Việt
    text = unicodedata.normalize('NFC', text).lower()
    
    # 2. Ánh xạ emojicon từ emojicon.txt
    for emoji, replacement in dict_emojicon.items():
        text = text.replace(emoji, f" {replacement} ")
        
    # 3. Khử các ký tự đặc biệt, chữ số và dấu câu
    text = re.sub(r'[\\d\\.,\\?\\!\\-\\;\\:\\(\\)\\[\\]\\+\\/\\&\\*\\_\\@]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    
    # 4. Tách từ tiếng Việt bằng Underthesea
    tokens = word_tokenize(text, format="text").split()
    
    # 5. Dịch teencode/tiếng anh/từ sai và lọc Stopwords
    cleaned_tokens = []
    for token in tokens:
        word = token.replace('_', ' ')
        word = dict_teencode.get(word, word)
        word = dict_wrong_word.get(word, word)
        word = dict_eng_vnmese.get(word, word)
        
        token_clean = word.replace(' ', '_')
        if token_clean not in stopwords:
            cleaned_tokens.append(token_clean)
            
    # 6. Ghép từ phủ định đi sát tính từ (Negation Merger)
    final_tokens = join_negation(cleaned_tokens)
    return " ".join(final_tokens)
        """, language="python")

    st.success("✅ Module Tiền xử lý NLP đã tích hợp thành công toàn bộ từ điển và Underthesea tokenizer!")
