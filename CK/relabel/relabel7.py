import pandas as pd
import ast
import re

def fix_ner_dataset(input_file, output_file):
    # 1. Load dữ liệu
    df = pd.read_csv(input_file)
    
    # Chuyển đổi chuỗi dạng list "['a', 'b']" thành list thật trong Python
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)

    # DANH SÁCH CÁC QUY TẮC FIX NGỮ NGHĨA (SEMANTIC MAPPING)
    # Các từ này thường xuyên bị gán sai trong file của bạn
    org_list = [
        'Chelsea', 'Spurs', 'Arsenal', 'Wolves', 'Celtic', 'Rangers', 'Leicester', 
        'Liverpool', 'Everton', 'Sunderland', 'Wrexham', 'Al-Nassr', 'Al-Hilal', 
        'Al-Ittihad', 'Al-Ahli', 'JPMorgan', 'Nvidia', 'OpenAI', 'Google', 'Meta', 
        'Amazon', 'Tesla', 'SpaceX', 'xAI', 'Microsoft', 'Bloomberg', 'Reuters',
        'Department', 'Justice', 'Homeland', 'Security', 'FBI', 'CIA', 'UNICEF'
    ]
    
    gpe_list = ['U.S.', 'US', 'UK', 'Britain', 'China', 'India', 'Venezuela', 'Norway', 'Haiti']

    def clean_labels(tokens, labels):
        new_labels = []
        
        # TẦNG 1: KHỬ NHÃN LỖI VÀ FIX NGỮ NGHĨA CƠ BẢN
        for i, (token, label) in enumerate(zip(tokens, labels)):
            clean_label = label
            
            # Sửa nhãn rác B-O, I-O thành O
            if label in ['B-O', 'I-O']:
                clean_label = 'O'
            
            # Fix các CLB bóng đá và công ty công nghệ (thường bị gán NORP/GPE sai)
            if any(org_word in token for org_word in org_list):
                if clean_label != 'O':
                    prefix = 'B-' if 'B-' in clean_label or (i > 0 and new_labels[i-1] == 'O') else 'I-'
                    clean_label = f"{prefix}ORG"
                else:
                    # Nếu token quan trọng bị gán O, cân nhắc khôi phục (ví dụ Nvidia)
                    if token in ['Nvidia', 'OpenAI', 'SpaceX']:
                        clean_label = 'B-ORG'

            # Fix quốc gia
            if token in gpe_list and clean_label != 'O' and 'ORG' not in clean_label:
                prefix = 'B-' if 'B-' in clean_label else 'I-'
                clean_label = f"{prefix}GPE"

            # Khử lỗi Matcha ice cream bị gán ORG
            if "Matcha" in token or "ice cream" in token:
                clean_label = 'O'

            new_labels.append(clean_label)

        # TẦNG 2: SỬA LỖI LIÊN KẾT (VÍ DỤ: "Department of Justice")
        # Nếu "Department" (B-ORG) và "Justice" (I-ORG) bị ngăn cách bởi "of" (O)
        for i in range(1, len(new_labels) - 1):
            if tokens[i].lower() in ['of', 'and', '&']:
                prev_label = new_labels[i-1]
                next_label = new_labels[i+1]
                if prev_label != 'O' and next_label != 'O':
                    prev_type = prev_label.split('-')[-1]
                    next_type = next_label.split('-')[-1]
                    if prev_type == next_type:
                        new_labels[i] = f"I-{prev_type}"

        # TẦNG 3: SỬA LOGIC BIO (QUAN TRỌNG NHẤT)
        final_labels = []
        for i, label in enumerate(new_labels):
            if label == 'O':
                final_labels.append('O')
                continue
            
            parts = label.split('-')
            if len(parts) != 2: # Phòng trường hợp nhãn sai định dạng
                final_labels.append('O')
                continue
                
            prefix, tag = parts
            
            if i == 0:
                # Token đầu tiên luôn phải là B-
                final_labels.append(f"B-{tag}")
            else:
                prev_label = final_labels[i-1]
                if prev_label == 'O':
                    # Sau O phải là B-
                    final_labels.append(f"B-{tag}")
                else:
                    prev_prefix, prev_tag = prev_label.split('-')
                    if tag != prev_tag:
                        # Nếu đổi loại thực thể, phải là B-
                        final_labels.append(f"B-{tag}")
                    else:
                        # Nếu cùng loại thực thể: 
                        # B-ORG tiếp sau B-ORG thì cái thứ 2 phải là I-ORG
                        final_labels.append(f"I-{tag}")
        
        return final_labels

    # Áp dụng hàm sửa lỗi
    fixed_data = []
    for _, row in df.iterrows():
        fixed_labels = clean_labels(row['tokens'], row['labels'])
        fixed_data.append(fixed_labels)
    
    df['labels'] = fixed_data

    # Chuyển về định dạng string để lưu CSV giống file gốc
    df['tokens'] = df['tokens'].apply(lambda x: str(x))
    df['labels'] = df['labels'].apply(lambda x: str(x))
    
    # Lưu file
    df.to_csv(output_file, index=False, quoting=1) # quoting=1 để bao đóng ngoặc kép toàn bộ list
    print(f"Đã hoàn thành! File sạch lưu tại: {output_file}")

# Chạy script
fix_ner_dataset('dataset_fixed_v7.csv', 'dataset_fixed_v8.csv')