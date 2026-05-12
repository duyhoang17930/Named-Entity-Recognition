import pandas as pd
import ast
import re

def fix_dataset(input_file, output_file):
    # Đọc dữ liệu
    df = pd.read_csv(input_file)
    
    # Chuyển đổi chuỗi string thành list thực tế
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    # Danh sách các quy tắc fix
    SYSTEM_WORDS = {'Summary', 'Companies', 'Note', 'Page', 'Reuters', 'Photo', 'File'}
    
    ORG_LIST = {'OpenAI', 'Nvidia', 'SpaceX', 'xAI', 'Alphabet', 'Google', 'Meta', 'Amazon', 
                'Tesla', 'NASA', 'FBI', 'DOJ', 'DHS', 'ICE', 'LSEG', 'CME', 'UNICEF', 'BBC', 
                'Microsoft', 'Oracle', 'Vanguard', 'Chevron', 'Hyundai'}
    
    GPE_LIST = {'U.S.', 'US', 'USA', 'Haiti', 'Ukraine', 'Russia', 'China', 'India', 
                'Mexico', 'Venezuela', 'Canada', 'Spain', 'Germany', 'Poland', 'Norway'}
    
    PER_LIST = {'Trump', 'Donald', 'Biden', 'Joe', 'Musk', 'Elon', 'Zelenskiy', 'Putin', 
                'Epstein', 'Jeffrey', 'Noem', 'Kristi', 'Vance', 'Mandelson'}

    NORP_LIST = {'Haitian', 'Haitians', 'Democratic', 'Republican', 'Russian', 'Chinese', 
                 'Indian', 'Brazilian', 'American', 'European', 'British'}

    def apply_fixes(tokens, labels):
        new_labels = labels[:]
        
        for i in range(len(tokens)):
            token = tokens[i]
            clean_token = re.sub(r'[^\w\s]', '', token) # Bỏ dấu câu để check từ điển

            # 1. Fix từ khóa hệ thống
            if token in SYSTEM_WORDS:
                new_labels[i] = 'O'
                continue

            # 2. Fix ký hiệu tiền tệ (Luôn bắt đầu thực thể MONEY)
            if token in {'$', '£', '€'}:
                new_labels[i] = 'B-MONEY'
                continue

            # 3. Chuẩn hóa thực thể dựa trên từ điển (Ưu tiên các thực thể quan trọng)
            if token in PER_LIST:
                new_labels[i] = 'B-PERSON' if (i == 0 or new_labels[i-1] == 'O') else 'I-PERSON'
            elif token in ORG_LIST:
                new_labels[i] = 'B-ORG' if (i == 0 or new_labels[i-1] == 'O') else 'I-ORG'
            elif token in GPE_LIST:
                new_labels[i] = 'B-GPE' if (i == 0 or new_labels[i-1] == 'O') else 'I-GPE'
            elif token in NORP_LIST:
                new_labels[i] = 'B-NORP' if (i == 0 or new_labels[i-1] == 'O') else 'I-NORP'

            # 4. Loại bỏ nhãn sai cho các từ loại thông thường (Adverbs/Verbs)
            if token.lower() in {'substantially', 'seems', 'likely', 'claimed', 'blocking', 'blocked'}:
                new_labels[i] = 'O'

        # 5. Sửa lỗi logic IOB (Second Pass)
        final_labels = []
        for i in range(len(new_labels)):
            current = new_labels[i]
            if current == 'O':
                final_labels.append('O')
                continue
            
            prefix, tag = current.split('-')
            
            if prefix == 'I':
                # Nếu I-tag đứng đầu hoặc đứng sau O hoặc đứng sau một tag khác loại
                if i == 0 or final_labels[i-1] == 'O' or final_labels[i-1].split('-')[1] != tag:
                    final_labels.append(f'B-{tag}')
                else:
                    final_labels.append(current)
            else: # prefix == 'B'
                # Nếu B-tag đứng sau một tag cùng loại, có thể chuyển thành I- để tạo thực thể dài
                # Tuy nhiên để an toàn, ta giữ nguyên B- nếu đó là bắt đầu từ mới
                final_labels.append(current)

        return final_labels

    # Áp dụng hàm sửa lỗi
    fixed_data = []
    for t, l in zip(df['tokens'], df['labels']):
        fixed_data.append(apply_fixes(t, l))
    
    df['labels'] = fixed_data

    # Chuyển ngược lại thành định dạng chuỗi giống file gốc
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))

    # Lưu file
    df.to_csv(output_file, index=False, quoting=1) # quoting=1 để đảm bảo bọc trong dấu ngoặc kép
    print(f"Đã sửa lỗi và lưu vào {output_file}")

# Thực thi
if __name__ == "__main__":
    input_path = 'dataset_fixed_v3.csv'
    output_path = 'dataset_final_v4.csv'
    fix_dataset(input_path, output_path)