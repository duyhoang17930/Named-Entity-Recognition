import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
from collections import Counter
import re

# Define files to process (v2 to v15)
relabel_dir = r"D:\Python\NLP\CK\relabel"
version_files = [
    "dataset_fixed_v2.csv",
    "dataset_fixed_v3.csv",
    "dataset_fixed_v4.csv",
    "dataset_fixed_v5.csv",
    "dataset_fixed_v5 khá.csv",
    "dataset_fixed_v6_final.csv",
    "dataset_fixed_v7.csv",
    "dataset_fixed_v8.csv",
    "dataset_fixed_v8 90.csv",
    "dataset_fixed_v9.csv",
    "dataset_fixed_v10.csv",
    "dataset_fixed_v11.csv",
    "dataset_fixed_v12.csv",
    "dataset_fixed_v13.csv",
    "dataset_fixed_v14.csv",
    "dataset_fixed_v15.csv",
]

# Collect entity counts for each version
version_counts = []
version_labels = []

for i, filename in enumerate(version_files):
    filepath = os.path.join(relabel_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
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

        # Count labels excluding "O"
        label_counts = Counter(all_labels)
        if 'O' in label_counts:
            del label_counts['O']

        version_counts.append(label_counts)
        version_labels.append(f"v{i+2}")
        print(f"{filename}: {sum(label_counts.values())} entities")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Get all unique entity types
all_entity_types = set()
for counts in version_counts:
    all_entity_types.update(counts.keys())

all_entity_types = sorted(list(all_entity_types))
print(f"\nEntity types found: {all_entity_types}")

# Create line chart data
plt.figure(figsize=(14, 8))

colors = plt.cm.tab20(range(len(all_entity_types)))

for idx, entity_type in enumerate(all_entity_types):
    entity_values = [version_counts[i].get(entity_type, 0) for i in range(len(version_counts))]
    plt.plot(version_labels, entity_values, marker='o', label=entity_type, color=colors[idx], linewidth=2, markersize=6)

plt.xlabel('Dataset Version', fontsize=12)
plt.ylabel('Number of Entities', fontsize=12)
plt.title('Entity Count Changes After Each Relabel Iteration', fontsize=14)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the chart
output_path = os.path.join(relabel_dir, "relabel_entity_counts.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nChart saved to: {output_path}")