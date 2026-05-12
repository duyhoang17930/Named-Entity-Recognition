import pandas as pd
import ast
import re

def fix_labels(tokens, labels):
    new_labels = list(labels)
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_lower = token.lower()

        # 1. FIX: M & A (Thường bị gán nhãn ORG nhầm)
        # Kiểm tra cụm "M", "&", "A" hoặc "M&A"
        if token_lower == 'm' and i+2 < len(tokens) and tokens[i+1] == '&' and tokens[i+2].lower() == 'a':
            new_labels[i] = 'O'
            new_labels[i+1] = 'O'
            new_labels[i+2] = 'O'
            i += 3
            continue
        if 'm&a' in token_lower:
            new_labels[i] = 'O'

        # 2. FIX: Dấu sở hữu cách ('s, ' hoặc ’s) luôn là O
        if token in ["'s", "'", "’s", "’"]:
            new_labels[i] = 'O'

        # 3. FIX: "nonwhite" hoặc "non-white" thường không nên là NORP (tùy guideline)
        if 'nonwhite' in token_lower or 'non-white' in token_lower:
            new_labels[i] = 'O'

        # 4. FIX: Cụm "U.S. Department of..." 
        # Nếu "U.S." là B-GPE và ngay sau là B-ORG, hợp nhất thành 1 ORG
        if token == 'U.S.' and i+1 < len(tokens):
            if labels[i] == 'B-GPE' and labels[i+1].endswith('ORG'):
                new_labels[i] = 'B-ORG'
                new_labels[i+1] = 'I-ORG'

        # 5. FIX: "the stars" (trong ngữ cảnh vũ trụ) không phải là LOC
        if token_lower == 'stars' and labels[i] == 'B-LOC':
            # Kiểm tra nếu trước đó là "the" hoặc "to"
            new_labels[i] = 'O'

        # 6. FIX: Các thực thể bị dính gạch nối (U.S.-backed, Chinese-linked)
        # Thường chỉ phần tên quốc gia là thực thể
        if '-' in token and labels[i] != 'O':
            parts = token.split('-')
            # Nếu là "U.S.-backed", giữ "U.S." là thực thể, phần sau là O 
            # Nhưng vì trong CoNLL token đã bị tách hoặc chưa, ta xử lý chuỗi:
            if any(p.lower() in ['backed', 'linked', 'led', 'era'] for p in parts):
                # Giữ nhãn nếu là phần đầu, nhưng thường các model hay gán nhãn cả cụm
                pass 

        # 7. FIX: Tên các công ty có hậu tố (Inc, LLC, Corp)
        if token in ['Inc', 'LLC', 'Corp', 'Corporation', 'Ltd'] and i > 0:
            if labels[i-1].endswith('ORG'):
                new_labels[i] = 'I-ORG'

        # 8. FIX: Các thuật ngữ tự chế như "Muskonomy"
        if 'muskonomy' in token_lower:
            new_labels[i] = 'O'

        # 9. FIX: "Wednesday", "February" đôi khi bị gán B-ORG hoặc B-NORP nhầm
        days_months = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday',
                       'january','february','march','april','may','june','july','august','september','october','november','december']
        if token_lower in days_months and not labels[i].endswith('DATE'):
            new_labels[i] = 'B-DATE'

        i += 1
    
    # Rà soát cuối cùng: Đảm bảo tính nhất quán I- ngay sau B-
    for j in range(1, len(new_labels)):
        if new_labels[j].startswith('I-'):
            prev_tag = new_labels[j-2:] # Lấy loại thực thể (ví dụ: PERS)
            curr_tag = new_labels[j-2:]
            if new_labels[j-1] == 'O': # Không thể có I- ngay sau O
                new_labels[j] = 'B-' + new_labels[j][2:]
                
    return new_labels

def main():
    input_file = 'dataset_fixed_v14.csv'
    output_file = 'dataset_fixed_v15.csv'

    print(f"Đang đọc file {input_file}...")
    df = pd.read_csv(input_file)

    # Chuyển đổi string đại diện cho list thành list thực tế
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    print("Đang tiến hành fix lỗi nhãn...")
    
    fixed_labels_list = []
    for index, row in df.iterrows():
        fixed_labels = fix_labels(row['tokens'], row['labels'])
        fixed_labels_list.append(fixed_labels)

    df['labels'] = fixed_labels_list

    # Chuyển ngược lại thành định dạng string để lưu CSV giống file gốc
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))

    df.to_csv(output_file, index=False, quoting=1) # quoting=1 để bao ngoặc kép các trường
    print(f"Hoàn thành! File sạch đã được lưu tại: {output_file}")

if __name__ == "__main__":
    main()