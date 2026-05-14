# Optimized NER Training Script
# Uses final_data.csv and applies various optimizations to boost F1 score

import ast
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data from final_data.csv...")
df = pd.read_csv('D:/Python/NLP/CK/final_data.csv')

def parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df['tokens'] = df['tokens'].apply(parse_list)
df['labels'] = df['labels'].apply(parse_list)

# Filter valid lengths
df = df[df['tokens'].apply(len) == df['labels'].apply(len)].reset_index(drop=True)
print(f"Loaded {len(df)} sentences")

# Get labels
all_labels = set()
for labels in df['labels']:
    all_labels.update(labels)

label_list = sorted(list(all_labels))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
print(f"Labels ({len(label_list)}): {label_list[:10]}...")

# Split data
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

print(f"\nData split:")
print(f"  Train: {len(train_df)}")
print(f"  Val: {len(val_df)}")
print(f"  Test: {len(test_df)}")

# Save label mappings
import json
with open('D:/Python/NLP/CK/label_mappings.json', 'w') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f)