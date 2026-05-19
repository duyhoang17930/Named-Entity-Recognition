import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
from collections import Counter

relabel_dir = r"D:\Python\NLP\CK\relabel"
version_files = [
    "dataset_fixed_v2.csv",
    "dataset_fixed_v3.csv",
    "dataset_fixed_v5.csv",
    "dataset_fixed_v6_final.csv",
    "dataset_fixed_v7.csv",
    "dataset_fixed_v8.csv",
    "dataset_fixed_v9.csv",
    "dataset_fixed_v10.csv",
    "dataset_fixed_v11.csv",
    "dataset_fixed_v12.csv",
    "dataset_fixed_v13.csv",
    "dataset_fixed_v14.csv",
    "dataset_fixed_v15.csv",
]

# Collect total entity counts (excluding O)
total_entities = []
labels = []

for i, filename in enumerate(version_files):
    filepath = os.path.join(relabel_dir, filename)
    if not os.path.exists(filepath):
        continue

    try:
        df = pd.read_csv(filepath)
        all_labels = []
        for labels_str in df['labels']:
            try:
                labels_list = ast.literal_eval(labels_str)
                all_labels.extend(labels_list)
            except:
                continue

        label_counts = Counter(all_labels)
        if 'O' in label_counts:
            del label_counts['O']
        if 'B-O' in label_counts:
            del label_counts['B-O']
        if 'I-O' in label_counts:
            del label_counts['I-O']

        total = sum(label_counts.values())
        total_entities.append(total)
        version_num = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15][i]
        labels.append(f"v{version_num}")
        print(f"{filename}: {total} entities")
    except Exception as e:
        print(f"Error: {e}")

# Plot total entities
plt.figure(figsize=(12, 6))
plt.plot(labels, total_entities, marker='o', linewidth=2.5, markersize=10, color='#2E86AB')
plt.fill_between(range(len(total_entities)), total_entities, alpha=0.2, color='#2E86AB')

for i, v in enumerate(total_entities):
    plt.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

plt.xlabel('Dataset Version', fontsize=12)
plt.ylabel('Total Number of Entities', fontsize=12)
plt.title('Total Entity Count Changes After Each Relabel Iteration', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

output_path = os.path.join(relabel_dir, "relabel_total_entities.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nChart saved to: {output_path}")