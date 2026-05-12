import pandas as pd
import ast
import re

def fix_ner_labels(input_file, output_file):
    # Load dữ liệu
    print(f"--- Đang đọc file: {input_file} ---")
    df = pd.read_csv(input_file)

    # Chuyển đổi string list thành list thật
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 
            'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
              'September', 'October', 'November', 'December', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fixed_rows = 0

    for idx, row in df.iterrows():
        tokens = row['tokens']
        labels = row['labels']
        new_labels = labels[:]
        is_changed = False

        for i in range(len(tokens)):
            token = tokens[i]
            prev_label = new_labels[i-1] if i > 0 else "O"
            
            # 1. FIX TỪ NỐI: and, or, with, & không được là I-PERSON hoặc B-PERSON (trừ ORG)
            if token.lower() in ['and', 'or', 'with', '&']:
                if 'PERSON' in new_labels[i] or 'GPE' in new_labels[i]:
                    new_labels[i] = "O"
                    is_changed = True

            # 2. FIX NGÀY THÁNG: Monday, January... phải là B-DATE nếu đang là O
            if token in days or token in months:
                if new_labels[i] == "O":
                    new_labels[i] = "B-DATE"
                    is_changed = True

            # 3. FIX MONEY: Dấu $ và các con số đi kèm tiền tệ
            if token == '$' or (token.isdigit() and i > 0 and tokens[i-1] == '$'):
                if new_labels[i] == "O":
                    new_labels[i] = "B-MONEY" if token == '$' else "I-MONEY"
                    is_changed = True

            # 4. FIX THỰC THỂ CỐ ĐỊNH (Common Entities)
            mapping = {
                'Trump': 'B-PERSON',
                'Biden': 'B-PERSON',
                'Reuters': 'B-ORG',
                'FBI': 'B-ORG',
                'U.S.': 'B-GPE',
                'US': 'B-GPE',
                'Haiti': 'B-GPE',
                'Gaza': 'B-GPE',
                'OpenAI': 'B-ORG',
                'Nvidia': 'B-ORG',
                'Tesla': 'B-ORG'
            }
            if token in mapping and new_labels[i] == "O":
                new_labels[i] = mapping[token]
                is_changed = True

            # 5. FIX DẤU CÂU: Dấu phẩy, chấm, ngoặc đơn dính vào thực thể
            if re.match(r'[.,()\-"]', token):
                if new_labels[i] != "O":
                    new_labels[i] = "O"
                    is_changed = True

        # 6. FIX LOGIC BIO (Quan trọng nhất)
        # Quy tắc: I-XXX phải sau B-XXX hoặc I-XXX. Nếu O -> I-XXX thì chuyển I thành B.
        for i in range(len(new_labels)):
            if new_labels[i].startswith("I-"):
                entity_type = new_labels[i].split("-")[1]
                if i == 0:
                    new_labels[i] = "B-" + entity_type
                    is_changed = True
                else:
                    prev_tag = new_labels[i-1]
                    if prev_tag == "O":
                        new_labels[i] = "B-" + entity_type
                        is_changed = True
                    elif prev_tag != "O":
                        prev_entity_type = prev_tag.split("-")[1]
                        if prev_entity_type != entity_type:
                            new_labels[i] = "B-" + entity_type
                            is_changed = True

        if is_changed:
            df.at[idx, 'labels'] = new_labels
            fixed_rows += 1

    # Chuyển ngược lại thành format string để lưu CSV giống file cũ
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))

    # Lưu file
    df.to_csv(output_file, index=False, quoting=1) # quoting=1 để đảm bảo bọc trong dấu ngoặc kép
    print(f"--- Hoàn thành! ---")
    print(f"Số dòng đã được chỉnh sửa: {fixed_rows}/{len(df)}")
    print(f"File mới đã lưu tại: {output_file}")

# Chạy script
if __name__ == "__main__":
    # Thay đổi tên file đầu vào của bạn ở đây
    input_name = 'dataset_fixed_v10.csv'
    output_name = 'dataset_fixed_v11.csv'
    fix_ner_labels(input_name, output_name)