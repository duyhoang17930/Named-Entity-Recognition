import pandas as pd
import ast
import csv
import re

def super_ner_cleaner(input_path, output_path):
    print(f"--- Đang khởi động bộ lọc dữ liệu NER chuyên sâu (v3 - Ultimate) ---")
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    corrected_rows = []
    
    # 1. Từ điển thực thể cố định (Mở rộng từ file của bạn)
    ENTITY_MAP = {
        # GPE
        "Haiti": "GPE", "U.S.": "GPE", "US": "GPE", "USA": "GPE", "Venezuela": "GPE", 
        "Russia": "GPE", "Ukraine": "GPE", "China": "GPE", "India": "GPE", "Mexico": "GPE", 
        "Washington": "GPE", "Gaza": "GPE", "Kyiv": "GPE", "Israel": "GPE", "Taiwan": "GPE",
        "Cuba": "GPE", "Niger": "GPE", "Poland": "GPE", "Germany": "GPE", "Argentina": "GPE",
        # PERSON
        "Trump": "PERSON", "Donald": "PERSON", "Joe": "PERSON", "Biden": "PERSON",
        "Noem": "PERSON", "Epstein": "PERSON", "Mandelson": "PERSON", "Jeffrey": "PERSON",
        "Elon": "PERSON", "Musk": "PERSON", "Zelenskiy": "PERSON", "Zelenskyy": "PERSON",
        "Putin": "PERSON", "Vladimir": "PERSON", "Modi": "PERSON", "Narendra": "PERSON",
        "Kushner": "PERSON", "Jared": "PERSON", "Bannon": "PERSON", "Steve": "PERSON",
        "Mette-Marit": "PERSON", "Hoiby": "PERSON", "Høiby": "PERSON", "Haakon": "PERSON",
        # ORG
        "OpenAI": "ORG", "Nvidia": "ORG", "NVIDIA": "ORG", "Tesla": "ORG", "SpaceX": "ORG", 
        "xAI": "ORG", "Reuters": "ORG", "Google": "ORG", "Alphabet": "ORG", "Disney": "ORG", 
        "Microsoft": "ORG", "Vanguard": "ORG", "Chevron": "ORG", "PDVSA": "ORG", "CME": "ORG",
        "LDP": "ORG", "UNICEF": "ORG", "NATO": "ORG", "FBI": "ORG", "DOJ": "ORG", "State": "ORG",
        "Department": "ORG", "House": "ORG", "Senate": "ORG", "Congress": "ORG",
        # PRODUCT
        "Bitcoin": "PRODUCT", "Grok": "PRODUCT", "Notepad++": "PRODUCT", "ChatGPT": "PRODUCT",
        "Wegovy": "PRODUCT", "Zepbound": "PRODUCT", "Azure": "PRODUCT",
    }

    # Danh sách chức danh cần đưa về O
    TITLES = {"Judge", "Secretary", "President", "CEO", "CFO", "Minister", "Chancellor", 
              "Ambassador", "Spokesperson", "Representative", "Officer", "Staffer", "Prince", "Princess", "Duke"}

    for idx, row in df.iterrows():
        tokens = ast.literal_eval(row['tokens'])
        labels = ast.literal_eval(row['labels'])
        
        # --- BƯỚC 1: XỬ LÝ LỖI DẤU CÂU VÀ RÁC WEB ---
        for i in range(len(tokens)):
            # Xoá nhãn của dấu câu
            if re.match(r'^[\"\'’“”.,\-\(\);:!/?]+$', tokens[i]) or tokens[i] in ["'s", "’s"]:
                labels[i] = "O"
            
            # Xoá rác Web
            if tokens[i].lower() in ["opens", "new", "tab"]:
                labels[i] = "O"

            # Đưa chức danh về O
            if tokens[i] in TITLES:
                labels[i] = "O"

        # --- BƯỚC 2: ÁP DỤNG TỪ ĐIỂN VÀ SỬA LOẠI THỰC THỂ ---
        for i in range(len(tokens)):
            t = tokens[i]
            # Sửa lỗi AI bị gán nhầm
            if t.upper() == "AI":
                labels[i] = "O" # Hoặc "B-TECHNOLOGY" tùy schema, nhưng file này hay gán nhầm NORP nên đưa về O
            
            # Tra từ điển
            if t in ENTITY_MAP:
                clean_tag = ENTITY_MAP[t]
                # Nếu từ trước đó cùng loại, gán I-, nếu không gán B-
                if i > 0 and labels[i-1].endswith("-" + clean_tag):
                    labels[i] = f"I-{clean_tag}"
                else:
                    labels[i] = f"B-{clean_tag}"

        # --- BƯỚC 3: SỬA LỖI CỤM TỪ GHÉP (GPE/LOC/ORG) ---
        i = 0
        while i < len(tokens) - 1:
            combined_2 = f"{tokens[i]} {tokens[i+1]}"
            if combined_2 == "United States":
                labels[i], labels[i+1] = "B-GPE", "I-GPE"
            elif combined_2 == "New York":
                labels[i], labels[i+1] = "B-GPE", "I-GPE"
            elif combined_2 == "Wall Street":
                labels[i], labels[i+1] = "B-LOC", "I-LOC"
            elif combined_2 == "White House":
                labels[i], labels[i+1] = "B-FAC", "I-FAC"
            i += 1

        # --- BƯỚC 4: BÁM DÍNH SỐ VÀ ĐƠN VỊ (MONEY/CARDINAL) ---
        for i in range(1, len(tokens)):
            unit_tokens = {"million", "billion", "trillion", "yuan", "dollars", "euros", "pounds", "bpd", "kg"}
            if tokens[i].lower() in unit_tokens:
                if labels[i-1] != "O":
                    tag_type = labels[i-1].split("-")[-1]
                    labels[i] = f"I-{tag_type}"

        # --- BƯỚC 5: CHUẨN HOÁ LOGIC BIO (QUAN TRỌNG NHẤT) ---
        # Quy tắc: B-A B-A -> B-A I-A | O I-A -> O B-A
        final_labels = []
        for i in range(len(labels)):
            curr = labels[i]
            if curr == "O":
                final_labels.append("O")
                continue
            
            prefix, tag = curr.split("-")
            prev = final_labels[i-1] if i > 0 else "O"
            
            if prev == "O":
                # Không bao giờ được bắt đầu bằng I-
                final_labels.append(f"B-{tag}")
            else:
                prev_prefix, prev_tag = prev.split("-")
                if prev_tag == tag:
                    # Nếu từ trước cùng loại, từ sau phải là I-
                    final_labels.append(f"I-{tag}")
                else:
                    # Nếu khác loại, từ sau phải là B-
                    final_labels.append(f"B-{tag}")

        corrected_rows.append({
            "tokens": str(tokens),
            "labels": str(final_labels)
        })

    # Xuất kết quả
    output_df = pd.DataFrame(corrected_rows)
    output_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"--- Đã hoàn thành! File sạch lưu tại: {output_path} ---")

# Chạy script
if __name__ == "__main__":
    super_ner_cleaner('dataset_fixed_v2.csv', 'dataset_fixed_v3.csv')