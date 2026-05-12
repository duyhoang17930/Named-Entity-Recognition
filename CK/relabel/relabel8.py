import pandas as pd
import ast
import json

def fix_ner_dataset(input_path, output_path):
    # 1. Đọc dữ liệu
    df = pd.read_csv(input_path)
    
    # Chuyển đổi string representation của list thành list thực tế
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    def apply_fixes(tokens, labels):
        new_labels = list(labels)
        
        for i in range(len(tokens)):
            token = tokens[i]
            token_lower = token.lower()
            prev_token = tokens[i-1].lower() if i > 0 else ""
            next_token = tokens[i+1].lower() if i < len(tokens)-1 else ""

            # --- LỖI 1: SILVER / GOLD (Hàng hóa vs Tên người) ---
            # Nếu Silver/Gold đi kèm với các từ khóa tài chính thì phải là 'O'
            finance_keywords = ['price', 'prices', 'ounce', 'fell', 'rise', 'drop', 'slump', 'trading', 'spot', 'liquidated', 'ounces', '$']
            if token_lower in ['silver', 'gold']:
                is_commodity = False
                if prev_token in finance_keywords or next_token in finance_keywords:
                    is_commodity = True
                if i > 1 and tokens[i-2].lower() in finance_keywords:
                    is_commodity = True
                
                # Ngoại lệ: "Nate Silver" là người
                if prev_token == 'nate':
                    new_labels[i] = 'I-PERSON' if labels[i].startswith('I-') else 'B-PERSON'
                elif is_commodity:
                    new_labels[i] = 'O'

            # --- LỖI 2: DRONES (Akinci / Akincis) ---
            # Thường bị nhầm thành PERSON hoặc ORG
            if 'akinci' in token_lower:
                new_labels[i] = 'B-PRODUCT'

            # --- LỖI 3: CÁC TỔ CHỨC BỊ GẮN SAI ---
            # Janes (tổ chức tình báo quốc phòng) hay bị nhầm thành PERSON
            if token_lower == 'janes':
                new_labels[i] = 'B-ORG'
            # Krystal (phóng viên) bị gắn nhãn ORG
            if token_lower == 'krystal' and labels[i] == 'B-ORG':
                new_labels[i] = 'B-PERSON'
            # Hearts (CLB bóng đá)
            if token_lower == 'hearts' and labels[i] != 'B-ORG':
                new_labels[i] = 'B-ORG'

            # --- LỖI 4: TÊN PHIM / TÁC PHẨM ---
            # "It Was Just an Accident" bị gắn nhãn NORP
            movie_tokens = ["it", "was", "just", "an", "accident"]
            if token_lower == "it" and i + 4 < len(tokens):
                phrase = [t.lower() for t in tokens[i:i+5]]
                if phrase == movie_tokens:
                    new_labels[i] = 'B-WORK_OF_ART'
                    for j in range(1, 5):
                        new_labels[i+j] = 'I-WORK_OF_ART'

            # --- LỖI 5: SỐ THỨ TỰ & NGÀY THÁNG ---
            # "250th" thường bị nhãn DATE, đúng phải là ORDINAL
            if '250th' in token_lower:
                new_labels[i] = 'B-ORDINAL'
            
            # --- LỖI 6: THỰC THỂ QUỐC GIA/TỔ CHỨC (GPE vs NORP) ---
            if token == 'U.S.' and next_token in ['government', 'administration', 'military']:
                new_labels[i] = 'B-ORG' # Trong ngữ cảnh bộ máy chính quyền

            # --- LỖI 7: FIX LỖI DOANH NGHIỆP TRONG TIN CHUYỂN NHƯỢNG ---
            football_clubs = ['chelsea', 'arsenal', 'liverpool', 'rangers', 'celtic', 'everton', 'wolves', 'westham']
            if token_lower in football_clubs:
                new_labels[i] = 'B-ORG'

        return new_labels

    # 2. Chạy hàm sửa lỗi
    fixed_labels = []
    for idx, row in df.iterrows():
        fixed_labels.append(apply_fixes(row['tokens'], row['labels']))
    
    df['labels'] = fixed_labels

    # 3. Kiểm tra tính nhất quán BIO (B- sau O, I- sau B-)
    # Đảm bảo không có I-Tag nào mồ côi
    for labels in df['labels']:
        for i in range(len(labels)):
            if labels[i].startswith('I-'):
                prev = labels[i-1] if i > 0 else 'O'
                if prev == 'O':
                    labels[i] = 'B-' + labels[i][2:]

    # 4. Xuất file
    # Chuyển list lại thành dạng string giống format gốc
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))
    
    df.to_csv(output_path, index=False, quoting=1) # quoting=1 để giữ dấu ngoặc kép quanh list
    print(f"✅ Đã sửa lỗi và lưu tại: {output_path}")

# Thực thi
fix_ner_dataset('dataset_fixed_v8.csv', 'dataset_fixed_v9.csv')