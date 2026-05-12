import pandas as pd
import ast
import re

def fix_ner_dataset(input_path, output_path):
    # 1. Load dữ liệu
    df = pd.read_csv(input_path)
    
    # Chuyển đổi chuỗi string thành list (vì CSV lưu list dưới dạng string)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    # DANH SÁCH TỪ VỰNG ĐỂ FIX
    DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
              "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
    ORDINALS = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"}
    
    fixed_rows = 0

    def apply_rules(tokens, labels):
        new_labels = list(labels)
        
        for i in range(len(tokens)):
            token = tokens[i]
            token_clean = token.strip('",.\'()')
            
            # --- QUY TẮC 1: Sửa lỗi bỏ sót DATE (Thứ, Tháng, Năm) ---
            if token_clean in DAYS or token_clean in MONTHS:
                if new_labels[i] == 'O':
                    new_labels[i] = 'B-DATE'
            
            if re.match(r'^(19|20)\d{2}$', token_clean): # Nhận diện năm 19xx hoặc 20xx
                if new_labels[i] == 'O':
                    new_labels[i] = 'B-DATE'

            # --- QUY TẮC 2: Sửa lỗi bỏ sót ORDINAL ---
            if token_clean.lower() in ORDINALS:
                if new_labels[i] == 'O':
                    new_labels[i] = 'B-ORDINAL'

            # --- QUY TẮC 3: Sửa lỗi gán nhãn sai NORP ---
            # "Constitution" không phải là NORP (nhóm người), nó là LAW hoặc WORK_OF_ART
            if "Constitution" in token:
                new_labels[i] = 'B-LAW'
            
            # "COVID-19" hoặc "pandemic" bị gán NORP là sai
            if "COVID" in token.upper() or "pandemic" in token.lower():
                if "NORP" in new_labels[i]:
                    new_labels[i] = 'O'

            # --- QUY TẮC 4: Sửa lỗi bóc tách sai cụm từ logic ---
            # Ví dụ: "de", "facto" không nên là B-ORG
            if token.lower() in ["de", "facto"]:
                if "ORG" in new_labels[i]:
                    new_labels[i] = 'O'
            
            # Sửa lỗi sở hữu cách: "Haiti 's" -> 's nên là O
            if token == "'s" or token == "’s":
                if i > 0 and new_labels[i] != 'O':
                    new_labels[i] = 'O'

            # --- QUY TẮC 5: Nhất quán hóa thực thể chính ---
            # Fix lỗi "Biden" đứng trước "administration" bị bỏ sót
            if token == "Biden" or token == "Trump":
                if new_labels[i] == 'O':
                    new_labels[i] = 'B-PERSON'
            
            # Fix "US" hoặc "U.S." phải luôn là GPE
            if token.upper() in ["US", "U.S.", "USA"]:
                if new_labels[i] == 'O' or "NORP" in new_labels[i]:
                    new_labels[i] = 'B-GPE'

        # --- QUY TẮC 6: Đảm bảo tính logic IOB (Quan trọng nhất) ---
        # Nhãn I-XYZ chỉ được xuất hiện sau B-XYZ hoặc I-XYZ
        for j in range(1, len(new_labels)):
            current_tag = new_labels[j]
            if current_tag.startswith('I-'):
                entity_type = current_tag[2:]
                prev_tag = new_labels[j-1]
                if prev_tag == 'O' or prev_tag[2:] != entity_type:
                    # Nếu sai logic, biến I- thành B-
                    new_labels[j] = 'B-' + entity_type

        return new_labels

    # Áp dụng hàm sửa lỗi
    for idx, row in df.iterrows():
        old_l = row['labels']
        new_l = apply_rules(row['tokens'], row['labels'])
        if old_l != new_l:
            df.at[idx, 'labels'] = new_l
            fixed_rows += 1

    # Lưu file
    # Chuyển về định dạng chuỗi giống như file gốc của bạn
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))
    
    df.to_csv(output_path, index=False, quoting=1) # quoting=1 để bao ngoặc kép các list
    print(f"✅ Hoàn thành! Đã sửa {fixed_rows} dòng có lỗi.")
    print(f"📂 File đã lưu tại: {output_path}")

# Chạy script
fix_ner_dataset('dataset_fixed_v5.csv', 'dataset_fixed_v6_final.csv')