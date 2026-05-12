import pandas as pd
import ast
import re

def fix_ner_dataset(input_file, output_file):
    # 1. Load dataset
    print(f"--- Đang đọc file {input_file} ---")
    df = pd.read_csv(input_file)
    
    # Chuyển đổi string đại diện cho list thành list thực tế
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    # 2. Định nghĩa từ điển sửa lỗi nhanh (Token-based Fix)
    # Định dạng: { "Token": "Label_Chuan" }
    token_fixes = {
        # Địa lý & Quốc gia (Thống nhất về GPE)
        "U.S.": "GPE", "US": "GPE", "United States": "GPE", "USA": "GPE",
        "Haiti": "GPE", "Cuba": "GPE", "India": "GPE", "Brazil": "GPE", 
        "Mexico": "GPE", "Turkey": "GPE", "Russia": "GPE", "Ukraine": "GPE",
        "Washington": "GPE", "D.C.": "GPE", "New York": "GPE",
        
        # Sản phẩm & Công nghệ (Thống nhất về PRODUCT)
        "ChatGPT": "PRODUCT", "Grok": "PRODUCT", "Notepad++": "PRODUCT",
        "Akincis": "PRODUCT", "Akinci": "PRODUCT", "Bayraktar": "PRODUCT",
        "777X": "PRODUCT", "GE9X": "PRODUCT", "iPhone": "PRODUCT",
        "Model Y": "PRODUCT", "Model 3": "PRODUCT", "Dreamliner": "PRODUCT",
        "SRAM": "PRODUCT", "GPU": "PRODUCT", "GPUs": "PRODUCT",
        
        # Tổ chức (ORG)
        "SpaceX": "ORG", "xAI": "ORG", "OpenAI": "ORG", "Nvidia": "ORG",
        "Alphabet": "ORG", "Google": "ORG", "Meta": "ORG", "Tesla": "ORG",
        "Reuters": "ORG", "UNICEF": "ORG", "CME": "ORG", "J.P. Morgan": "ORG",
        "JP Morgan": "ORG", "Vanguard": "ORG", "BlueCo": "ORG",
        
        # Luật pháp (LAW)
        "Constitution": "LAW", "Amendment": "LAW", "Act": "LAW",
    }

    # 3. Hàm xử lý logic chính cho từng dòng
    def refine_labels(row):
        tokens = row['tokens']
        labels = row['labels']
        new_labels = labels[:]

        for i in range(len(tokens)):
            token = tokens[i]
            prev_token = tokens[i-1] if i > 0 else ""
            
            # A. Sửa dựa trên từ điển token_fixes
            if token in token_fixes:
                target_label = token_fixes[token]
                # Quyết định là B- hay I-
                if i > 0 and new_labels[i-1].endswith(target_label):
                    new_labels[i] = f"I-{target_label}"
                else:
                    new_labels[i] = f"B-{target_label}"

            # B. Sửa lỗi logic cụ thể (Contextual Fixes)
            
            # Ví dụ: "U.S. Constitution" -> B-LAW, I-LAW
            if token == "Constitution" and "U.S." in prev_token:
                new_labels[i-1] = "B-LAW"
                new_labels[i] = "I-LAW"

            # Ví dụ: Tên người cụ thể hay bị sai (Donald Trump, Elon Musk)
            if token in ["Donald", "Elon", "Joe", "Sam", "Kristi", "Marius"]:
                new_labels[i] = "B-PERSON"
            if token in ["Trump", "Musk", "Biden", "Altman", "Noem", "Hoiby"]:
                if i > 0 and new_labels[i-1] == "B-PERSON":
                    new_labels[i] = "I-PERSON"
                else:
                    new_labels[i] = "B-PERSON"

            # Ví dụ: Đơn vị tiền tệ (MONEY)
            if token == "$" or token == "£" or token == "yuan":
                new_labels[i] = "B-MONEY"
                if i+1 < len(tokens) and re.match(r'^\d', tokens[i+1]):
                    new_labels[i+1] = "I-MONEY"

        # C. Fix BIO Consistency (Lỗi quan trọng nhất)
        # Đảm bảo không có nhãn I- đứng sau O hoặc đứng sau I- của loại khác
        final_labels = []
        for j in range(len(new_labels)):
            curr = new_labels[j]
            if curr.startswith("I-"):
                entity_type = curr.split("-")[1]
                if j == 0 or final_labels[j-1] == "O":
                    final_labels.append(f"B-{entity_type}") # Chuyển I thành B nếu đứng đầu
                elif final_labels[j-1].split("-")[-1] != entity_type:
                    final_labels.append(f"B-{entity_type}") # Chuyển I thành B nếu khác loại trước đó
                else:
                    final_labels.append(curr)
            else:
                final_labels.append(curr)

        return final_labels

    # 4. Thực thi sửa lỗi
    print("--- Đang thực thi fix lỗi logic và BIO ---")
    df['labels'] = df.apply(refine_labels, axis=1)

    # 5. Kiểm tra sơ bộ
    print(f"Tổng số dòng xử lý: {len(df)}")
    
    # 6. Lưu file (Đảm bảo giữ đúng định dạng list trong CSV)
    # Lưu ý: Chuyển lại về dạng chuỗi giống file gốc để các công cụ đọc được
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))
    
    df.to_csv(output_file, index=False, quoting=1) # quoting=1 để bao quanh bằng dấu ngoặc kép
    print(f"--- Đã lưu file sạch tại: {output_file} ---")

# Chạy script
if __name__ == "__main__":
    # Thay tên file đầu vào của bạn ở đây
    input_fn = "dataset_fixed_v12.csv" 
    output_fn = "dataset_fixed_v13.csv"
    fix_ner_dataset(input_fn, output_fn)