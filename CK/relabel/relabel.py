import pandas as pd
import ast
import csv
import re

def comprehensive_ner_cleaner(input_path, output_path):
    print(f"--- Đang khởi động bộ lọc dữ liệu NER chuyên sâu ---")
    
    # Đọc dữ liệu
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    corrected_rows = []
    
    # Bộ đếm thống kê chi tiết các loại lỗi
    logs = {
        "GPE_MISLABELED": 0,    # Quốc gia bị gán thành ORG/PERSON
        "WEB_ARTIFACTS": 0,     # Lỗi 'opens new tab'
        "AI_NORP_FIX": 0,       # AI bị gán nhầm thành dân tộc/chính trị
        "TITLE_CLEANING": 0,    # Chức danh (President, CEO) đưa về O
        "MONEY_UNIT_FIX": 0,    # Nối số với đơn vị (billion, million)
        "PUNCTUATION_FIX": 0,   # Dấu câu bị gán nhãn thực thể
        "BIO_INCONSISTENCY": 0, # Lỗi I- đứng sau O hoặc B- sai loại
        "ENTITY_UNIFORM": 0,    # Đồng nhất các tên riêng (Trump, OpenAI...)
        "LOWERCASE_CLEAN": 0    # Từ viết thường (the, of, in) bị gán nhãn ở đầu câu
    }

    # 1. Từ điển thực thể cố định (Luôn đúng trong mọi ngữ cảnh của file này)
    ENTITY_MAP = {
        "Haiti": "B-GPE", "U.S.": "B-GPE", "US": "B-GPE", "United States": "GPE",
        "Venezuela": "B-GPE", "Russia": "B-GPE", "Ukraine": "B-GPE", "China": "B-GPE",
        "India": "B-GPE", "Mexico": "B-GPE", "Washington": "B-GPE", "Gaza": "B-GPE",
        "Trump": "B-PERSON", "Donald": "B-PERSON", "Joe": "B-PERSON", "Biden": "B-PERSON",
        "Noem": "B-PERSON", "Epstein": "B-PERSON", "Mandelson": "B-PERSON",
        "Elon": "B-PERSON", "Musk": "B-PERSON", "Zelenskiy": "B-PERSON",
        "OpenAI": "B-ORG", "Nvidia": "B-ORG", "NVIDIA": "B-ORG", "Tesla": "B-ORG",
        "SpaceX": "B-ORG", "xAI": "B-ORG", "Reuters": "B-ORG", "Google": "B-ORG",
        "Alphabet": "B-ORG", "Disney": "B-ORG", "Microsoft": "B-ORG", "Fidesz": "B-ORG",
        "Bitcoin": "B-PRODUCT", "Grok": "B-PRODUCT", "Notepad++": "B-PRODUCT",
        "Monday": "B-DATE", "Tuesday": "B-DATE", "Wednesday": "B-DATE", "Thursday": "B-DATE",
        "Friday": "B-DATE", "Saturday": "B-DATE", "Sunday": "B-DATE",
        "January": "B-DATE", "February": "B-DATE", "March": "B-DATE", "October": "B-DATE"
    }

    # 2. Danh sách chức danh (Nên là O theo quy tắc đa số các tập dữ liệu chuẩn)
    TITLES = {"Judge", "Secretary", "President", "CEO", "CFO", "Minister", "Chancellor", 
              "Ambassador", "Spokesperson", "Representative", "Officer", "Staffer"}

    # 3. Từ dừng/Từ nối không được mang nhãn B- (trừ khi là một phần của Org)
    STOPS = {"the", "a", "an", "and", "of", "to", "in", "on", "with", "it", "that", "this", "is", "was"}

    for idx, row in df.iterrows():
        tokens = ast.literal_eval(row['tokens'])
        labels = ast.literal_eval(row['labels'])
        new_labels = labels.copy()

        # --- BƯỚC 1: XỬ LÝ QUY TẮC CỨNG VÀ NGỮ CẢNH ---
        for i in range(len(tokens)):
            token = tokens[i]
            token_low = token.lower()

            # Sửa rác Web: 'opens new tab' -> luôn là O
            if token_low in {"opens", "new", "tab"}:
                if new_labels[i] != "O":
                    new_labels[i] = "O"
                    logs["WEB_ARTIFACTS"] += 1
                continue

            # Sửa dấu câu: Luôn là O
            if re.match(r'^[\"\'’“”.,\-\(\);:!]+$', token):
                if new_labels[i] != "O":
                    new_labels[i] = "O"
                    logs["PUNCTUATION_FIX"] += 1
                continue

            # Đồng nhất thực thể từ Entity Map
            if token in ENTITY_MAP:
                target = ENTITY_MAP[token]
                if new_labels[i] != target:
                    new_labels[i] = target
                    logs["ENTITY_UNIFORM"] += 1

            # Sửa chức danh đứng trước tên riêng (ví dụ: President Trump)
            if token in TITLES:
                if new_labels[i] != "O":
                    new_labels[i] = "O"
                    logs["TITLE_CLEANING"] += 1

            # Sửa lỗi AI
            if token.upper() == "AI" and "NORP" in new_labels[i]:
                new_labels[i] = "O"
                logs["AI_NORP_FIX"] += 1

            # Xử lý các từ đơn vị (billion, yuan...) phải là I-
            if token_low in {"billion", "million", "trillion", "yuan", "dollars", "euros", "pounds", "bpd"}:
                if i > 0 and ("CARDINAL" in new_labels[i-1] or "MONEY" in new_labels[i-1] or "QUANTITY" in new_labels[i-1]):
                    tag_type = new_labels[i-1].split("-")[-1]
                    new_labels[i] = f"I-{tag_type}"
                    logs["MONEY_UNIT_FIX"] += 1

            # Xử lý từ nối viết thường bị gán nhãn sai
            if token_low in STOPS and new_labels[i].startswith("B-"):
                # Chỉ đưa về O nếu nó không đứng trước một thực thể cùng loại
                if i+1 < len(tokens) and new_labels[i+1].split("-")[-1] != new_labels[i].split("-")[-1]:
                    new_labels[i] = "O"
                    logs["LOWERCASE_CLEAN"] += 1

        # --- BƯỚC 2: XỬ LÝ CỤM TỪ ĐẶC BIỆT ---
        for i in range(len(tokens) - 1):
            # Ví dụ: 'United States' -> B-GPE I-GPE
            if tokens[i] == "United" and tokens[i+1] == "States":
                new_labels[i], new_labels[i+1] = "B-GPE", "I-GPE"
            # Ví dụ: 'Wall Street'
            if tokens[i] == "Wall" and tokens[i+1] == "Street":
                new_labels[i], new_labels[i+1] = "B-LOC", "I-LOC"

        # --- BƯỚC 3: KIỂM TRA LOGIC BIO (BẮT BUỘC) ---
        # Đảm bảo không có I- nào mồ côi hoặc sai loại
        for i in range(len(new_labels)):
            if new_labels[i] == "O": continue
            
            curr_prefix, curr_type = new_labels[i].split("-")
            prev_label = new_labels[i-1] if i > 0 else "O"
            
            if curr_prefix == "I":
                if prev_label == "O":
                    new_labels[i] = f"B-{curr_type}"
                    logs["BIO_INCONSISTENCY"] += 1
                else:
                    prev_type = prev_label.split("-")[-1]
                    if prev_type != curr_type:
                        new_labels[i] = f"B-{curr_type}"
                        logs["BIO_INCONSISTENCY"] += 1
            
            elif curr_prefix == "B":
                if prev_label != "O":
                    prev_prefix, prev_type = prev_label.split("-")
                    if prev_type == curr_type:
                        # Nếu 2 nhãn B cùng loại cạnh nhau, cái sau phải là I
                        # Trừ trường hợp danh sách: "Haiti, China" -> B-GPE, O, B-GPE
                        pass # Logic này phụ thuộc vào token ngăn cách

        corrected_rows.append({
            "tokens": str(tokens),
            "labels": str(new_labels)
        })

    # Xuất kết quả ra file mới
    output_df = pd.DataFrame(corrected_rows)
    output_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    # In báo cáo chi tiết
    print("\n" + "="*60)
    print(f"{'BÁO CÁO SỬA LỖI CHI TIẾT':^60}")
    print("="*60)
    print(f"1. Chuyển địa danh về GPE (Haiti, US...):....... {logs['ENTITY_UNIFORM']} lỗi")
    print(f"2. Xoá rác Web (opens new tab):................. {logs['WEB_ARTIFACTS']} lỗi")
    print(f"3. Sửa nhãn AI (NORP -> O):..................... {logs['AI_NORP_FIX']} lỗi")
    print(f"4. Đưa chức danh (President, CEO) về O:......... {logs['TITLE_CLEANING']} lỗi")
    print(f"5. Sửa lỗi dấu câu mang nhãn thực thể:........... {logs['PUNCTUATION_FIX']} lỗi")
    print(f"6. Sửa lỗi đơn vị tiền tệ/số lượng (I-):........ {logs['MONEY_UNIT_FIX']} lỗi")
    print(f"7. Sửa lỗi từ nối (the, of, and) gán nhãn sai:.. {logs['LOWERCASE_CLEAN']} lỗi")
    print(f"8. Sửa vi phạm logic BIO (I- sau O):............ {logs['BIO_INCONSISTENCY']} lỗi")
    print("-" * 60)
    print(f"TỔNG SỐ LỖI ĐÃ ĐƯỢC XỬ LÝ: {sum(logs.values())}")
    print("="*60)
    print(f"File sạch đã được lưu tại: {output_path}")

# --- CHẠY SCRIPT ---
if __name__ == "__main__":
    # Tên file đầu vào của bạn (đảm bảo file này nằm cùng thư mục script)
    input_file = 'output_fixed.csv' 
    output_file = 'dataset_fixed_v2.csv'
    
    # Tạo file mẫu để chạy test nếu bạn chưa có file dataset.csv
    import os
    if not os.path.exists(input_file):
        print(f"Vui lòng đặt tên file dữ liệu là {input_file}")
    else:
        comprehensive_ner_cleaner(input_file, output_file)