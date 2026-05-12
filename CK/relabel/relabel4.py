import pandas as pd
import ast
import re

def fix_ner_dataset(input_file, output_file):
    # 1. Đọc dữ liệu
    df = pd.read_csv(input_file)
    
    # Chuyển đổi string đại diện cho list thành list thật trong Python
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    fixed_labels_list = []

    for idx, row in df.iterrows():
        tokens = row['tokens']
        labels = row['labels']
        
        # Tạo bản sao để sửa
        new_labels = list(labels)

        # --- BƯỚC 1: XÓA NHÃN SAI CHO CÁC TỪ KHÔNG PHẢI THỰC THỂ (HEADERS) ---
        trash_words = ["Summary", "Companies", "Lawsuit", "claims", "says", "opens", "new", "tab"]
        for i, token in enumerate(tokens):
            if token in trash_words or token.startswith("http"):
                new_labels[i] = 'O'

        # --- BƯỚC 2: SỬA LỖI ĐỨT ĐOẠN (BRIDGE LOGIC) ---
        # Ví dụ: B-ORG, O, I-ORG (Department of State)
        for i in range(1, len(tokens) - 1):
            if new_labels[i] == 'O':
                prev_label = new_labels[i-1]
                next_label = new_labels[i+1]
                
                # Nếu từ ở giữa là từ nối và hai bên cùng loại thực thể
                if tokens[i].lower() in ["of", "and", "the", "&", "for", "in"]:
                    if prev_label != 'O' and next_label != 'O':
                        type_prev = prev_label.split('-')[-1]
                        type_next = next_label.split('-')[-1]
                        if type_prev == type_next:
                            new_labels[i] = f"I-{type_prev}"
                            # Đảm bảo token tiếp theo là I- thay vì B- để nối liền
                            new_labels[i+1] = f"I-{type_next}"

        # --- BƯỚC 3: XỬ LÝ TIỀN TỆ (MONEY) ---
        # Gộp các cụm như: $ (B-MONEY) 10 (B-CARDINAL) billion (I-CARDINAL)
        for i in range(len(tokens)):
            if tokens[i] == '$' or tokens[i].lower() in ['usd', 'gbp', 'eur']:
                new_labels[i] = 'B-MONEY'
                # Loang nhãn MONEY ra các token số/đơn vị phía sau
                j = i + 1
                while j < len(tokens) and (re.match(r'^[\d.,]+$', tokens[j]) or tokens[j].lower() in ['million', 'billion', 'trillion', 'bn', 'm']):
                    new_labels[j] = 'I-MONEY'
                    j += 1

        # --- BƯỚC 4: THỐNG NHẤT THỰC THỂ ĐẶC BIỆT ---
        for i, token in enumerate(tokens):
            # "JE" hoặc "Jeffrey Epstein" luôn là PERSON
            if token in ["JE", "Epstein"] or (i < len(tokens)-1 and token == "Jeffrey" and tokens[i+1] == "Epstein"):
                if new_labels[i] != 'O':
                    prefix = "B" # Tạm thời để B, bước cuối sẽ fix IOB
                    new_labels[i] = f"{prefix}-PERSON"
            
            # West Bank thống nhất là LOC
            if token == "West" and i < len(tokens)-1 and tokens[i+1] == "Bank":
                new_labels[i] = "B-LOC"
                new_labels[i+1] = "I-LOC"

        # --- BƯỚC 5: FIX LỖI LOGIC IOB (BẮT BUỘC) ---
        # Đây là bước quan trọng nhất để model không bị crash
        final_labels = []
        for i in range(len(new_labels)):
            current = new_labels[i]
            
            if current == 'O':
                final_labels.append('O')
                continue
            
            tag_parts = current.split('-')
            if len(tag_parts) != 2: # Phòng trường hợp nhãn sai định dạng
                final_labels.append('O')
                continue
                
            prefix, tag_type = tag_parts
            
            if i == 0:
                # Token đầu tiên luôn phải là B- nếu không phải O
                final_labels.append(f"B-{tag_type}")
            else:
                prev = final_labels[i-1]
                if prev == 'O':
                    # Sau O phải là B-
                    final_labels.append(f"B-{tag_type}")
                else:
                    prev_prefix, prev_type = prev.split('-')
                    if prev_type == tag_type:
                        # Nếu cùng loại với từ trước, giữ nguyên nhãn hiện tại (B hoặc I)
                        # nhưng ưu tiên I để nối chuỗi
                        final_labels.append(f"I-{tag_type}")
                    else:
                        # Nếu khác loại, bắt buộc phải là B-
                        final_labels.append(f"B-{tag_type}")

        fixed_labels_list.append(final_labels)

    # 2. Lưu kết quả
    df['labels'] = fixed_labels_list
    
    # Chuyển list về lại định dạng string giống file gốc để dễ đọc
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))
    
    df.to_csv(output_file, index=False, quoting=1) # quoting=1 để bao ngoặc kép các trường
    print(f"✅ Đã fix xong! File mới: {output_file}")

# Chạy script
if __name__ == "__main__":
    fix_ner_dataset('dataset_final_v4.csv', 'dataset_fixed_v5.csv')