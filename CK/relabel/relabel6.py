import pandas as pd
import ast

def fix_ner_dataset(input_file, output_file):
    # 1. Load dữ liệu
    df = pd.read_csv(input_file)
    
    # Chuyển đổi string đại diện cho list thành list thật trong Python
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    def apply_fixes(tokens, labels):
        new_labels = list(labels)
        n = len(tokens)

        # --- QUY TẮC 1: FIX CÁC THỰC THỂ CỐ ĐỊNH (PHRASE MATCHING) ---
        # Định nghĩa các cụm từ quan trọng và nhãn đúng của chúng
        phrase_fixes = [
            (["U.S.", "Department", "of", "Homeland", "Security"], "ORG"),
            (["Department", "of", "Homeland", "Security"], "ORG"),
            (["U.S.", "Department", "of", "Justice"], "ORG"),
            (["Department", "of", "Justice"], "ORG"),
            (["U.S.", "Justice", "Department"], "ORG"),
            (["Justice", "Department"], "ORG"),
            (["U.S.", "District", "Judge"], "O"), # Chức danh thường để O hoặc theo schema riêng
            (["Wall", "Street", "Journal"], "ORG"),
            (["New", "York", "Times"], "ORG"),
            (["United", "States"], "GPE"),
            (["White", "House"], "FAC"),
            (["Fifth", "Amendment"], "LAW"),
            (["U.S.", "Constitution"], "LAW"),
            (["Supreme", "Court"], "ORG"),
            (["Premier", "League"], "ORG"),
            (["European", "Union"], "ORG"),
            (["United", "Arab", "Emirates"], "GPE"),
            (["House", "of", "Representatives"], "ORG"),
            (["House", "Oversight", "Committee"], "ORG"),
            (["Federal", "Reserve"], "ORG"),
            (["Temporary", "Protected", "Status"], "O"), # Thường là một trạng thái pháp lý, không phải ORG
        ]

        for phrase, label_type in phrase_fixes:
            p_len = len(phrase)
            for i in range(n - p_len + 1):
                if [t.lower() for t in tokens[i:i+p_len]] == [p.lower() for p in phrase]:
                    new_labels[i] = f"B-{label_type}"
                    for j in range(1, p_len):
                        new_labels[i+j] = f"I-{label_type}"

        # --- QUY TẮC 2: FIX TOKEN ĐƠN LẺ & NHẤT QUÁN HÓA ---
        for i in range(n):
            token = tokens[i]
            token_lower = token.lower()

            # Fix Reuters luôn là ORG
            if token_lower == "reuters":
                new_labels[i] = "B-ORG"
            
            # Fix British/French/Chinese luôn là NORP (không phải PERSON)
            if token_lower in ["british", "french", "chinese", "brazilian", "japanese", "haitian", "russian", "ukrainian", "indian", "american"]:
                new_labels[i] = "B-NORP"

            # Fix các thành phố/quốc gia cụ thể
            if token_lower in ["havana", "kyiv", "moscow", "washington", "beijing"]:
                if i + 2 < n and tokens[i+1] == "," and tokens[i+2] == "D.C.":
                    new_labels[i] = "B-GPE"
                    new_labels[i+1] = "O"
                    new_labels[i+2] = "I-GPE"
                else:
                    new_labels[i] = "B-GPE"

            # Fix sở hữu cách 's không bao giờ là B- hay I-
            if token in ["'s", "’s", "’"]:
                new_labels[i] = "O"

            # Fix Cardinal nhầm lẫn (ví dụ "60" trong "60 Minutes")
            if token == "60" and i+1 < n and tokens[i+1].lower() == "minutes":
                new_labels[i] = "B-WORK_OF_ART"
                new_labels[i+1] = "I-WORK_OF_ART"

        # --- QUY TẮC 3: HẬU XỬ LÝ (POST-PROCESSING) ---
        # Đảm bảo tính hợp lệ của chuỗi IOB (I- không được đứng sau O hoặc sau một nhãn khác loại)
        for i in range(1, len(new_labels)):
            if new_labels[i].startswith("I-"):
                current_type = new_labels[i].split("-")[1]
                prev_label = new_labels[i-1]
                
                if prev_label == "O":
                    new_labels[i] = f"B-{current_type}"
                else:
                    prev_type = prev_label.split("-")[1]
                    if current_type != prev_type:
                        new_labels[i] = f"B-{current_type}"

        return new_labels

    # Áp dụng hàm sửa lỗi cho từng hàng
    new_labels_column = []
    for index, row in df.iterrows():
        fixed = apply_fixes(row['tokens'], row['labels'])
        new_labels_column.append(fixed)
    
    df['labels'] = new_labels_column

    # Chuyển về định dạng string-list để lưu CSV giống file gốc
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))

    # Lưu file
    df.to_csv(output_file, index=False, quoting=1) # quoting=1 để đảm bảo dấu ngoặc kép đúng định dạng
    print(f"Đã hoàn thành! File sạch được lưu tại: {output_file}")

# Thực thi script
if __name__ == "__main__":
    input_csv = "dataset_fixed_v6_final.csv"
    output_csv = "dataset_fixed_v7.csv"
    fix_ner_dataset(input_csv, output_csv)