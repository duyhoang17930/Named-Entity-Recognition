import pandas as pd
import ast
import re
import string

def fix_ner_labels(tokens, labels):
    """
    Hàm xử lý chi tiết để sửa lỗi gán nhãn NER.
    """
    new_labels = list(labels)
    punctuation = set(string.punctuation) | {"''", "``", "’", "“", "”", "–"}

    for i in range(len(tokens)):
        token = tokens[i]
        label = labels[i]
        
        # 1. Ép tất cả dấu câu về nhãn 'O'
        if token in punctuation:
            new_labels[i] = 'O'
            continue

        # 2. Xử lý dấu sở hữu cách (Possessive pronouns/markers)
        if token.lower() in ["'s", "’s", "'"]:
            new_labels[i] = 'O'
            continue

        # 3. Đồng nhất thực thể địa lý/tổ chức phổ biến bị gán nhầm là 'O'
        if token == "U.S." and label == 'O':
            new_labels[i] = 'B-GPE'
        if token == "Reuters" and label == 'O':
            new_labels[i] = 'B-ORG'
        if token == "Washington" and label in ['O', 'B-LOC']:
            new_labels[i] = 'B-GPE'
        if token == "China" and label == 'O':
            new_labels[i] = 'B-GPE'

        # 4. Xử lý nhãn MONEY (Ký hiệu $ thường bị tách lẻ)
        if token == "$":
            new_labels[i] = 'B-MONEY'
            # Nếu token sau là số, đảm bảo nó là I-MONEY
            if i + 1 < len(tokens) and re.match(r'^[\d.,]+[mb]?$', tokens[i+1].lower()):
                new_labels[i+1] = 'I-MONEY'

        # 5. Xử lý các nhãn số lượng viết tắt (ví dụ: 10bn, 3.5m)
        if re.search(r'\d+(bn|m|tn)$', token.lower()):
            if label == 'O':
                new_labels[i] = 'B-CARDINAL'

    # 6. Sửa lỗi logic IOB (Nhãn I- không thể đứng sau O hoặc đứng sau một nhãn khác loại)
    # Ví dụ: O, I-PERSON -> O, B-PERSON
    final_labels = []
    for i in range(len(new_labels)):
        curr_label = new_labels[i]
        if curr_label.startswith('I-'):
            entity_type = curr_label.split('-')[1]
            if i == 0: # I- ở đầu câu
                final_labels.append(f'B-{entity_type}')
            else:
                prev_label = final_labels[i-1]
                if prev_label == 'O':
                    final_labels.append(f'B-{entity_type}')
                else:
                    prev_entity_type = prev_label.split('-')[1]
                    if prev_entity_type != entity_type:
                        final_labels.append(f'B-{entity_type}')
                    else:
                        final_labels.append(curr_label)
        else:
            final_labels.append(curr_label)

    return final_labels

def main():
    input_file = 'dataset_fixed_v9.csv'
    output_file = 'dataset_fixed_v10.csv'

    print(f"--- Đang đọc file {input_file} ---")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    # Chuyển đổi chuỗi string đại diện cho list thành list thực tế trong Python
    print("--- Đang giải mã cấu trúc dữ liệu ---")
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    # Áp dụng logic sửa lỗi
    print("--- Đang tiến hành fix lỗi nhãn (Punctuation, IOB Logic, Consistency) ---")
    df['labels'] = df.apply(lambda row: fix_ner_labels(row['tokens'], row['labels']), axis=1)

    # Kiểm tra lại độ dài (Đảm bảo tokens và labels vẫn khớp nhau)
    len_check = df.apply(lambda row: len(row['tokens']) == len(row['labels']), axis=1)
    if not len_check.all():
        print("CẢNH BÁO: Có sự lệch độ dài giữa tokens và labels tại một số dòng!")
    else:
        print("Xác nhận: Độ dài tokens và labels khớp 100%.")

    # Lưu kết quả
    # Chuyển về định dạng string list để giống với file gốc
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))

    df.to_csv(output_file, index=False, quoting=1) # quoting=1 để giữ dấu ngoặc kép cho list
    print(f"--- ĐÃ HOÀN THÀNH ---")
    print(f"File sạch đã được lưu tại: {output_file}")

if __name__ == "__main__":
    main()