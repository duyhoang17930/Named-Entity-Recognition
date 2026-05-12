import pandas as pd
import ast
import re

def fix_ner_labels(input_file, output_file):
    # Đọc dữ liệu
    df = pd.read_csv(input_file)
    
    # Chuyển đổi string list thành list thật
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    # 1. Từ điển sửa lỗi nhanh (Hard-coded replacements)
    org_whitelist = {'Tesla', 'Meta', 'Amazon', 'SpaceX', 'OpenAI', 'Google', 'Alphabet', 'Hostinger', 'Neuralink', 'Microsoft', 'Nvidia', 'Cerebras', 'Groq', 'Disney'}
    product_whitelist = {'ChatGPT', 'Grok', 'Wegovy', 'Zepbound', 'Model Y', 'Model 3', 'Azure'}
    gpe_fixes = {'U.S.': 'B-GPE', 'US': 'B-GPE', 'UK': 'B-GPE', 'Washington': 'B-GPE'}

    def process_row(tokens, labels):
        new_labels = list(labels)
        
        for i in range(len(tokens)):
            token = tokens[i]
            prev_token = tokens[i-1] if i > 0 else ""
            next_token = tokens[i+1] if i < len(tokens)-1 else ""

            # --- LUẬT 1: Sửa các tổ chức/sản phẩm bị bỏ sót ---
            if token in org_whitelist and new_labels[i] == 'O':
                new_labels[i] = 'B-ORG'
            if token in product_whitelist and new_labels[i] == 'O':
                new_labels[i] = 'B-PRODUCT'
            if token == 'Hostinger': # Lỗi cụ thể trong file của bạn
                new_labels[i] = 'B-ORG'

            # --- LUẬT 2: Sửa thực thể GPE (Địa chính trị) ---
            if token in gpe_fixes and new_labels[i] == 'O':
                new_labels[i] = gpe_fixes[token]

            # --- LUẬT 3: Sửa lỗi phân mảnh DATE (Vd: February 3 , 2026) ---
            # Nếu token hiện tại là số hoặc dấu phẩy và nằm giữa 2 nhãn DATE
            if token in [',', '.'] or token.isdigit():
                if i > 0 and i < len(tokens)-1:
                    if 'DATE' in new_labels[i-1] and ('DATE' in labels[i+1] or 'CARDINAL' in labels[i+1]):
                        new_labels[i] = 'I-DATE'
            
            # --- LUẬT 4: Sửa nhãn CARDINAL/QUANTITY bên trong DATE ---
            if 'DATE' in new_labels[i-1] and (new_labels[i] == 'B-CARDINAL' or new_labels[i] == 'I-CARDINAL'):
                new_labels[i] = 'I-DATE'

            # --- LUẬT 5: Xử lý dấu sở hữu cách ('s) ---
            if token == "'s" or token == "’s":
                if i > 0 and new_labels[i-1] != 'O':
                    # 's thường nên là O hoặc theo thực thể trước đó (tùy convention, ở đây chọn O để sạch)
                    new_labels[i] = 'O' 

            # --- LUẬT 6: Củng cố nhãn MONEY (Vd: $ 100 billion) ---
            if token == '$':
                new_labels[i] = 'B-MONEY'
                # Gán I-MONEY cho các token số tiếp theo
                curr = i + 1
                while curr < len(tokens) and (tokens[curr].replace('.', '').isdigit() or tokens[curr] in ['billion', 'million', 'trillion']):
                    new_labels[curr] = 'I-MONEY'
                    curr += 1

        # --- LUẬT 7: Kiểm tra logic IOB (B- luôn đi trước I-) ---
        final_labels = []
        for i in range(len(new_labels)):
            curr_lab = new_labels[i]
            if curr_lab.startswith('I-'):
                entity_type = curr_lab.split('-')[1]
                if i == 0 or (new_labels[i-1] == 'O') or (entity_type not in new_labels[i-1]):
                    final_labels.append('B-' + entity_type)
                else:
                    final_labels.append(curr_lab)
            else:
                final_labels.append(curr_lab)
                
        return final_labels

    # Áp dụng hàm sửa lỗi
    fixed_data = []
    for t, l in zip(df['tokens'], df['labels']):
        fixed_data.append(process_row(t, l))
    
    df['labels'] = fixed_data

    # Chuyển list ngược lại thành string để lưu CSV
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))

    # Lưu file
    df.to_csv(output_file, index=False, quoting=1) # quoting=1 để đảm bảo bọc trong dấu ngoặc kép
    print(f"Đã xử lý xong! File sạch đã được lưu tại: {output_file}")

# Thực thi script
if __name__ == "__main__":
    # Thay tên file đầu vào của bạn ở đây
    input_fn = 'dataset_fixed_v13.csv' 
    output_fn = 'dataset_fixed_v14.csv'
    
    try:
        fix_ner_labels(input_fn, output_fn)
    except FileNotFoundError:
        print("Không tìm thấy file đầu vào. Vui lòng kiểm tra lại tên file.")