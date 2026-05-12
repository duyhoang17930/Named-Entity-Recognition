import pandas as pd
import ast
import re

def fix_ner_dataset(input_path, output_path):
    # 1. Tải dữ liệu
    print(f"--- Đang tải dữ liệu từ {input_path} ---")
    df = pd.read_csv(input_path)

    # Chuyển đổi string list thành list thực tế
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    def clean_labels(tokens, labels):
        new_labels = list(labels)
        
        # DANH SÁCH TỪ KHÓA CẦN FIX NGỮ CẢNH
        product_keywords = ['ChatGPT', 'Grok', 'Azure', 'GE9X', '777X', 'Dreamliner']
        law_keywords = ['Constitution', 'Amendment']
        org_keywords = ['OpenAI', 'xAI', 'SpaceX', 'Tesla', 'Nvidia', 'Waymo']

        for i in range(len(tokens)):
            token = tokens[i]
            label = new_labels[i]

            # --- NHÓM 1: FIX LỖI THỰC THỂ SAI LOẠI (CONTEXT FIX) ---
            
            # Fix ChatGPT bị gán nhãn NORP -> PRODUCT
            if 'ChatGPT' in token and 'NORP' in label:
                new_labels[i] = label.replace('NORP', 'PRODUCT')
            
            # Fix Silver/Gold bị gán nhãn PERSON trong bài báo tài chính -> O hoặc PRODUCT
            # Ở đây chuyển về O vì chúng là hàng hóa thông thường
            if token.lower() in ['silver', 'gold', 'platinum', 'copper'] and 'PERSON' in label:
                new_labels[i] = 'O'

            # Fix Constitution/Amendment -> LAW
            if token in law_keywords:
                new_labels[i] = 'B-LAW' if (i == 0 or new_labels[i-1] == 'O') else 'I-LAW'

            # Fix Q1, Q2, Q3, Q4 -> DATE
            if re.match(r'Q[1-4]', token):
                new_labels[i] = 'B-DATE'

            # Fix các từ nối trong tên tổ chức (của, và, cho...)
            # Nếu "of", "and", "the" nằm giữa 2 nhãn ORG thì phải là I-ORG
            if token.lower() in ['of', 'and', 'for', 'the'] and i > 0 and i < len(tokens)-1:
                if 'ORG' in new_labels[i-1] and 'ORG' in new_labels[i+1]:
                    new_labels[i] = 'I-ORG'

            # --- NHÓM 2: FIX DẤU SỞ HỮU CÁCH VÀ KÝ TỰ ĐẶC BIỆT ---
            
            # Dấu sở hữu cách 's hoặc ’s phải luôn là O (hoặc theo tiêu chuẩn là O)
            if token in ["'s", "’s", "'", "’"]:
                new_labels[i] = 'O'
            
            # Dấu phẩy, dấu chấm ở cuối câu không được mang nhãn thực thể
            if token in [',', '.', '(', ')', ':', ';'] and label != 'O':
                # Ngoại lệ: Dấu chấm trong tên viết tắt như U.S.
                if not (token == '.' and i > 0 and len(tokens[i-1]) == 1):
                    new_labels[i] = 'O'

        # --- NHÓM 3: FIX LỖI ĐỊNH DẠNG BIO (QUAN TRỌNG NHẤT) ---
        # Quy tắc: I-TYPE chỉ được xuất hiện sau B-TYPE hoặc I-TYPE cùng loại.
        # Quy tắc: Không được có B-TYPE ngay sau B-TYPE nếu cùng một thực thể.
        
        final_labels = []
        for i in range(len(new_labels)):
            curr_lab = new_labels[i]
            if curr_lab == 'O':
                final_labels.append('O')
                continue
            
            parts = curr_lab.split('-')
            prefix = parts[0] # B hoặc I
            etype = parts[1]  # PERSON, ORG, vv...

            if i > 0:
                prev_lab = final_labels[i-1]
                if prev_lab != 'O':
                    prev_etype = prev_lab.split('-')[1]
                    
                    # Nếu nhãn hiện tại là B-TYPE nhưng nhãn trước đó cũng là TYPE
                    # và chúng thuộc cùng một thực thể (dựa trên logic token)
                    if etype == prev_etype:
                        # Nếu từ trước đó không phải là kết thúc câu, biến B thành I
                        final_labels.append(f"I-{etype}")
                    else:
                        final_labels.append(curr_lab)
                else:
                    # Nếu trước đó là O, thì nhãn này bắt buộc phải là B-
                    final_labels.append(f"B-{etype}")
            else:
                # Token đầu tiên luôn là B- nếu không phải O
                final_labels.append(f"B-{etype}")

        return final_labels

    # Áp dụng hàm sửa lỗi
    print("--- Đang thực hiện sửa lỗi logic và BIO ---")
    df['labels_fixed'] = df.apply(lambda x: clean_labels(x['tokens'], x['labels']), axis=1)

    # 4. Kiểm tra lại lần cuối (Double Check)
    # Đảm bảo không có nhãn I- đứng sau O
    def final_check(labels):
        checked = list(labels)
        for i in range(len(checked)):
            if checked[i].startswith('I-'):
                if i == 0 or checked[i-1] == 'O':
                    checked[i] = 'B-' + checked[i].split('-')[1]
        return checked

    df['labels_fixed'] = df['labels_fixed'].apply(final_check)

    # 5. Lưu file
    # Chuyển về định dạng string giống file gốc để dễ sử dụng
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels_fixed'].apply(lambda x: str(x))

    df[['tokens', 'labels']].to_csv(output_path, index=False, quoting=1)
    print(f"--- Đã lưu file sạch tại: {output_path} ---")

# Chạy Script
if __name__ == "__main__":
    INPUT_FILE = "dataset_fixed_v11.csv"
    OUTPUT_FILE = "dataset_fixed_v12.csv"
    fix_ner_dataset(INPUT_FILE, OUTPUT_FILE)