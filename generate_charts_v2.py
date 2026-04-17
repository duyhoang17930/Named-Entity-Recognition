"""
Generate Comprehensive Charts for NER Pipeline Analysis
Comparing: clean_news.csv -> manual_labeled.csv -> relabeled_output.csv -> relabeled_output_encoded.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory
CHARTS_DIR = "D:/Python/NLP/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

def parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

print("=" * 70)
print("GENERATING COMPREHENSIVE CHARTS FOR NER PIPELINE")
print("=" * 70)

# ============================================================================
# LOAD ALL DATASETS
# ============================================================================
print("\n[1] Loading Datasets...")

# Dataset 1: Clean News (After Preprocessing)
clean_df = pd.read_csv("clean_news.csv")
print(f"   clean_news.csv: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")

# Dataset 2: Manual Labeled
manual_df = pd.read_csv("manual_labeled.csv")
print(f"   manual_labeled.csv: {manual_df.shape[0]} rows")

# Dataset 3: Relabeled Output (Custom Mapping)
relabeled_df = pd.read_csv("relabeled_output.csv")
print(f"   relabeled_output.csv: {relabeled_df.shape[0]} rows")

# Dataset 4: Relabeled Encoded
encoded_df = pd.read_csv("relabeled_output_encoded.csv")
print(f"   relabeled_output_encoded.csv: {encoded_df.shape[0]} rows")

# Parse labels
print("\n[2] Parsing Labels...")
manual_df['labels_decoded'] = manual_df['labels'].apply(parse_list)
relabeled_df['labels_decoded'] = relabeled_df['labels'].apply(parse_list)
encoded_df['labels_decoded'] = encoded_df['labels'].apply(parse_list)

# Calculate statistics
def get_label_stats(df):
    all_labels = []
    for labels in df['labels_decoded']:
        all_labels.extend(labels)
    return Counter(all_labels)

manual_labels = get_label_stats(manual_df)
relabeled_labels = get_label_stats(relabeled_df)

# Get entity types
def get_entity_types(label_counts):
    entity_types = {}
    for label, count in label_counts.items():
        if label == 'O':
            entity_types['O'] = count
        elif label.startswith('B-') or label.startswith('I-'):
            entity_type = label[2:]
            entity_types[entity_type] = entity_types.get(entity_type, 0) + count
    return entity_types

manual_entities = get_entity_types(manual_labels)
relabeled_entities = get_entity_types(relabeled_labels)

# ============================================================================
# CHARTS - PART 1: PREPROCESSING ANALYSIS
# ============================================================================
print("\n[3] Generating Preprocessing Charts...")

# Chart 1: Dataset sizes at each stage
fig, ax = plt.subplots(figsize=(12, 6))
stages = ['Clean News\n(Preprocessed)', 'Manual\nLabeled', 'Relabeled\n(Custom)', 'Encoded\n(Final)']
sizes = [clean_df.shape[0], manual_df.shape[0], relabeled_df.shape[0], encoded_df.shape[0]]
colors = ['#2ECC71', '#3498DB', '#E74C3C', '#9B59B6']

bars = ax.bar(stages, sizes, color=colors, edgecolor='black', linewidth=2)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Dataset Size at Each Pipeline Stage', fontsize=14, fontweight='bold')
for bar, size in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{size:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/01_dataset_sizes.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 01_dataset_sizes.png")

# Chart 2: Sentence length distribution - Before vs After cleaning
clean_df['original_len'] = clean_df['Sentence'].astype(str).apply(len)
clean_df['clean_len'] = clean_df['clean_sentence'].astype(str).apply(len)
clean_df['word_count'] = clean_df['clean_sentence'].astype(str).apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(clean_df['original_len'], bins=30, color='#E74C3C', alpha=0.7, edgecolor='black')
axes[0].axvline(clean_df['original_len'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {clean_df['original_len'].mean():.0f}")
axes[0].set_xlabel('Character Length', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Original Sentence Length', fontsize=12, fontweight='bold')
axes[0].legend()

axes[1].hist(clean_df['clean_len'], bins=30, color='#2ECC71', alpha=0.7, edgecolor='black')
axes[1].axvline(clean_df['clean_len'].mean(), color='green', linestyle='--', linewidth=2, label=f"Mean: {clean_df['clean_len'].mean():.0f}")
axes[1].set_xlabel('Character Length', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Cleaned Sentence Length', fontsize=12, fontweight='bold')
axes[1].legend()

axes[2].hist(clean_df['word_count'], bins=30, color='#3498DB', alpha=0.7, edgecolor='black')
axes[2].axvline(clean_df['word_count'].mean(), color='blue', linestyle='--', linewidth=2, label=f"Mean: {clean_df['word_count'].mean():.1f}")
axes[2].set_xlabel('Word Count', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title('Word Count (After Cleaning)', fontsize=12, fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/02_preprocessing_lengths.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 02_preprocessing_lengths.png")

# Chart 3: Cleaning reduction percentage
clean_df['reduction_pct'] = ((clean_df['original_len'] - clean_df['clean_len']) / clean_df['original_len'] * 100)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(clean_df['reduction_pct'].dropna(), bins=30, color='#9B59B6', alpha=0.7, edgecolor='black')
ax.axvline(clean_df['reduction_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {clean_df['reduction_pct'].mean():.1f}%")
ax.axvline(clean_df['reduction_pct'].median(), color='blue', linestyle='--', linewidth=2, label=f"Median: {clean_df['reduction_pct'].median():.1f}%")
ax.set_xlabel('Reduction Percentage (%)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Character Reduction After Preprocessing', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/03_cleaning_reduction.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 03_cleaning_reduction.png")

# Chart 4: Scatter - Original vs Clean length
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(clean_df['original_len'], clean_df['clean_len'], alpha=0.5, c='#3498DB', edgecolors='black', linewidth=0.5, s=50)
max_val = max(clean_df['original_len'].max(), clean_df['clean_len'].max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='No Change Line')
ax.set_xlabel('Original Sentence Length', fontsize=12)
ax.set_ylabel('Clean Sentence Length', fontsize=12)
ax.set_title('Original vs Cleaned Sentence Length', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/04_original_vs_clean.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 04_original_vs_clean.png")

# ============================================================================
# CHARTS - PART 2: MANUAL LABELING ANALYSIS
# ============================================================================
print("\n[4] Generating Manual Labeling Charts...")

# Chart 5: Label distribution (Manual)
fig, ax = plt.subplots(figsize=(14, 7))
labels_sorted = sorted(manual_labels.items(), key=lambda x: x[1], reverse=True)
label_names = [x[0] for x in labels_sorted]
label_values = [x[1] for x in labels_sorted]

colors = plt.cm.viridis(np.linspace(0, 1, len(label_names)))
bars = ax.bar(label_names, label_values, color=colors, edgecolor='black')
ax.set_xlabel('NER Labels', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('NER Label Distribution (Manual Labeling)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, label_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            str(val), ha='center', va='bottom', fontsize=8, rotation=45)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/05_manual_label_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 05_manual_label_distribution.png")

# Chart 6: Entity type pie chart (Manual)
fig, ax = plt.subplots(figsize=(12, 8))
entity_names = sorted(manual_entities.keys(), key=lambda x: manual_entities[x], reverse=True)
entity_values = [manual_entities[e] for e in entity_names]

colors = plt.cm.Set3(np.linspace(0, 1, len(entity_names)))
wedges, texts, autotexts = ax.pie(entity_values, labels=entity_names, autopct='%1.1f%%',
                                    colors=colors, startangle=90, pctdistance=0.75)
ax.set_title('Entity Type Distribution (Manual Labeling)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/06_manual_entity_pie.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 06_manual_entity_pie.png")

# Chart 7: Token count distribution (Manual)
manual_df['token_count'] = manual_df['labels_decoded'].apply(len)
manual_df['entity_count'] = manual_df['labels_decoded'].apply(lambda x: sum(1 for l in x if l.startswith('B-')))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(manual_df['token_count'], bins=30, color='#E74C3C', alpha=0.7, edgecolor='black')
axes[0].axvline(manual_df['token_count'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {manual_df['token_count'].mean():.1f}")
axes[0].set_xlabel('Token Count per Sentence', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Token Count Distribution (Manual)', fontsize=12, fontweight='bold')
axes[0].legend()

axes[1].hist(manual_df['entity_count'], bins=20, color='#3498DB', alpha=0.7, edgecolor='black')
axes[1].axvline(manual_df['entity_count'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {manual_df['entity_count'].mean():.1f}")
axes[1].set_xlabel('Entity Count per Sentence', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Entity Count Distribution (Manual)', fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/07_manual_token_entity_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 07_manual_token_entity_dist.png")

# Chart 8: BIO distribution (Manual)
bio_manual = {'Begin (B-)': 0, 'Inside (I-)': 0, 'Outside (O)': 0}
for label in manual_labels.keys():
    if label.startswith('B-'): bio_manual['Begin (B-)'] += manual_labels[label]
    elif label.startswith('I-'): bio_manual['Inside (I-)'] += manual_labels[label]
    elif label == 'O': bio_manual['Outside (O)'] += manual_labels[label]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#E74C3C', '#3498DB', '#2ECC71']
bars = ax.bar(bio_manual.keys(), bio_manual.values(), color=colors, edgecolor='black', linewidth=2)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('BIO Tag Distribution (Manual Labeling)', fontsize=14, fontweight='bold')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f'{bar.get_height():,}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/08_manual_bio_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 08_manual_bio_distribution.png")

# ============================================================================
# CHARTS - PART 3: RELABELED OUTPUT ANALYSIS
# ============================================================================
print("\n[5] Generating Relabeled Charts...")

# Chart 9: Label distribution (Relabeled)
fig, ax = plt.subplots(figsize=(14, 7))
labels_sorted = sorted(relabeled_labels.items(), key=lambda x: x[1], reverse=True)
label_names = [x[0] for x in labels_sorted]
label_values = [x[1] for x in labels_sorted]

colors = plt.cm.plasma(np.linspace(0, 1, len(label_names)))
bars = ax.bar(label_names, label_values, color=colors, edgecolor='black')
ax.set_xlabel('NER Labels', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('NER Label Distribution (Relabeled - Custom Mapping)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/09_relabeled_label_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 09_relabeled_label_distribution.png")

# Chart 10: Entity type pie chart (Relabeled)
fig, ax = plt.subplots(figsize=(12, 8))
entity_names = sorted(relabeled_entities.keys(), key=lambda x: relabeled_entities[x], reverse=True)
entity_values = [relabeled_entities[e] for e in entity_names]

colors = plt.cm.Paired(np.linspace(0, 1, len(entity_names)))
wedges, texts, autotexts = ax.pie(entity_values, labels=entity_names, autopct='%1.1f%%',
                                   colors=colors, startangle=90, pctdistance=0.75)
ax.set_title('Entity Type Distribution (Relabeled - Custom Mapping)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/10_relabeled_entity_pie.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 10_relabeled_entity_pie.png")

# Chart 11: Token count distribution (Relabeled)
relabeled_df['token_count'] = relabeled_df['labels_decoded'].apply(len)
relabeled_df['entity_count'] = relabeled_df['labels_decoded'].apply(lambda x: sum(1 for l in x if l.startswith('B-')))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(relabeled_df['token_count'], bins=30, color='#9B59B6', alpha=0.7, edgecolor='black')
axes[0].axvline(relabeled_df['token_count'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {relabeled_df['token_count'].mean():.1f}")
axes[0].set_xlabel('Token Count per Sentence', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Token Count Distribution (Relabeled)', fontsize=12, fontweight='bold')
axes[0].legend()

axes[1].hist(relabeled_df['entity_count'], bins=20, color='#1ABC9C', alpha=0.7, edgecolor='black')
axes[1].axvline(relabeled_df['entity_count'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {relabeled_df['entity_count'].mean():.1f}")
axes[1].set_xlabel('Entity Count per Sentence', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Entity Count Distribution (Relabeled)', fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/11_relabeled_token_entity_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 11_relabeled_token_entity_dist.png")

# Chart 12: BIO distribution (Relabeled)
bio_relabeled = {'Begin (B-)': 0, 'Inside (I-)': 0, 'Outside (O)': 0}
for label in relabeled_labels.keys():
    if label.startswith('B-'): bio_relabeled['Begin (B-)'] += relabeled_labels[label]
    elif label.startswith('I-'): bio_relabeled['Inside (I-)'] += relabeled_labels[label]
    elif label == 'O': bio_relabeled['Outside (O)'] += relabeled_labels[label]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#9B59B6', '#1ABC9C', '#E67E22']
bars = ax.bar(bio_relabeled.keys(), bio_relabeled.values(), color=colors, edgecolor='black', linewidth=2)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('BIO Tag Distribution (Relabeled - Custom Mapping)', fontsize=14, fontweight='bold')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f'{bar.get_height():,}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/12_relabeled_bio_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 12_relabeled_bio_distribution.png")

# ============================================================================
# CHARTS - PART 4: COMPARISON CHARTS (Manual vs Relabeled)
# ============================================================================
print("\n[6] Generating Comparison Charts...")

# Chart 13: Side-by-side Label Comparison (Top 15)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

manual_sorted = sorted(manual_labels.items(), key=lambda x: x[1], reverse=True)[:15]
manual_labels_list = [x[0] for x in manual_sorted]
manual_vals = [x[1] for x in manual_sorted]

axes[0].barh(manual_labels_list[::-1], manual_vals[::-1], color='#3498DB', edgecolor='black')
axes[0].set_xlabel('Count', fontsize=11)
axes[0].set_title('Top 15 Labels (Manual)', fontsize=12, fontweight='bold')

relabeled_sorted = sorted(relabeled_labels.items(), key=lambda x: x[1], reverse=True)[:15]
relabeled_labels_list = [x[0] for x in relabeled_sorted]
relabeled_vals = [x[1] for x in relabeled_sorted]

axes[1].barh(relabeled_labels_list[::-1], relabeled_vals[::-1], color='#E74C3C', edgecolor='black')
axes[1].set_xlabel('Count', fontsize=11)
axes[1].set_title('Top 15 Labels (Relabeled)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/13_label_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 13_label_comparison.png")

# Chart 14: Entity Type Comparison Bar Chart
fig, ax = plt.subplots(figsize=(14, 7))
all_entities = sorted(set(manual_entities.keys()) | set(relabeled_entities.keys()))

x = np.arange(len(all_entities))
width = 0.35

manual_vals = [manual_entities.get(e, 0) for e in all_entities]
relabeled_vals = [relabeled_entities.get(e, 0) for e in all_entities]

bars1 = ax.bar(x - width/2, manual_vals, width, label='Manual', color='#3498DB', edgecolor='black')
bars2 = ax.bar(x + width/2, relabeled_vals, width, label='Relabeled', color='#E74C3C', edgecolor='black')

ax.set_xlabel('Entity Type', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Entity Type Comparison: Manual vs Relabeled', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(all_entities, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/14_entity_type_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 14_entity_type_comparison.png")

# Chart 15: BIO Comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.35

bars1 = ax.bar(x - width/2, [bio_manual['Begin (B-)'], bio_manual['Inside (I-)'], bio_manual['Outside (O)']],
               width, label='Manual', color='#3498DB', edgecolor='black')
bars2 = ax.bar(x + width/2, [bio_relabeled['Begin (B-)'], bio_relabeled['Inside (I-)'], bio_relabeled['Outside (O)']],
               width, label='Relabeled', color='#E74C3C', edgecolor='black')

ax.set_xlabel('BIO Tag', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('BIO Distribution: Manual vs Relabeled', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Begin (B-)', 'Inside (I-)', 'Outside (O)'])
ax.legend()

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/15_bio_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 15_bio_comparison.png")

# Chart 16: Label Changes (Difference)
common_labels = set(manual_labels.keys()) & set(relabeled_labels.keys())
label_changes = {}
for label in common_labels:
    diff = relabeled_labels[label] - manual_labels.get(label, 0)
    if diff != 0:
        label_changes[label] = diff

fig, ax = plt.subplots(figsize=(12, 6))
if label_changes:
    labels = list(label_changes.keys())
    values = list(label_changes.values())
    colors = ['#2ECC71' if v > 0 else '#E74C3C' for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Label', fontsize=12)
    ax.set_ylabel('Change (Relabeled - Manual)', fontsize=12)
    ax.set_title('Label Count Changes: Manual vs Relabeled', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/16_label_changes.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 16_label_changes.png")

# ============================================================================
# CHARTS - PART 5: ENCODED DATA ANALYSIS
# ============================================================================
print("\n[7] Generating Encoded Data Charts...")

# Chart 17: Encoded label distribution
encoded_df['labels_encoded_decoded'] = encoded_df['labels_encoded'].apply(parse_list)

all_encoded = []
for labels in encoded_df['labels_encoded_decoded']:
    all_encoded.extend(labels)
encoded_counts = Counter(all_encoded)

fig, ax = plt.subplots(figsize=(14, 6))
sorted_encoded = sorted(encoded_counts.items(), key=lambda x: x[0])
labels_enc = [str(x[0]) for x in sorted_encoded]
vals_enc = [x[1] for x in sorted_encoded]

ax.bar(labels_enc, vals_enc, color='#9B59B6', edgecolor='black')
ax.set_xlabel('Encoded Label ID', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Encoded Label Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/17_encoded_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 17_encoded_distribution.png")

# Chart 18: Label mapping table visualization
label_ids = sorted(encoded_counts.keys())
id_to_label = {i: 'O' for i in label_ids}

# Create a simple visualization of the label encoding
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Create label mapping data
unique_labels = sorted(set(manual_labels.keys()) | set(relabeled_labels.keys()))
label_mapping_data = []
for i, label in enumerate(unique_labels):
    manual_count = manual_labels.get(label, 0)
    relabeled_count = relabeled_labels.get(label, 0)
    label_mapping_data.append([label, i, manual_count, relabeled_count])

# Add header
table_data = [['Label', 'ID', 'Manual Count', 'Relabeled Count']] + label_mapping_data[:20]

table = ax.table(cellText=table_data, loc='center', cellLoc='center', colColours=['#3498DB']*4)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
ax.set_title('Label Encoding Mapping (First 20)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/18_label_encoding.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 18_label_encoding.png")

# ============================================================================
# CHARTS - PART 6: ADVANCED ANALYSIS
# ============================================================================
print("\n[8] Generating Advanced Analysis Charts...")

# Chart 19: Top Entities Bar Chart (Relabeled)
entity_mentions = []
for labels in relabeled_df['labels_decoded']:
    for label in labels:
        if label.startswith('B-'):
            entity_mentions.append(label[2:])
entity_mention_counts = Counter(entity_mentions)

fig, ax = plt.subplots(figsize=(12, 6))
top_entities = entity_mention_counts.most_common(15)
entities = [x[0] for x in top_entities]
counts = [x[1] for x in top_entities]

bars = ax.barh(entities[::-1], counts[::-1], color='#1ABC9C', edgecolor='black')
ax.set_xlabel('Frequency', fontsize=12)
ax.set_title('Top 15 Most Common Named Entities (Relabeled)', fontsize=14, fontweight='bold')
for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, str(count), va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/19_top_entities.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 19_top_entities.png")

# Chart 20: Entity Co-occurrence Heatmap
entity_types_list = [e for e in relabeled_entities.keys() if e != 'O'][:8]
cooccurrence = {e1: {e2: 0 for e2 in entity_types_list} for e1 in entity_types_list}

for labels in relabeled_df['labels_decoded']:
    entities_in_sentence = [label[2:] for label in labels if label.startswith('B-')]
    for e1 in entities_in_sentence:
        for e2 in entities_in_sentence:
            if e1 != e2 and e1 in cooccurrence and e2 in cooccurrence[e1]:
                cooccurrence[e1][e2] += 1

cooccurrence_matrix = np.array([[cooccurrence[e1][e2] for e2 in entity_types_list]
                                  for e1 in entity_types_list])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cooccurrence_matrix, xticklabels=entity_types_list, yticklabels=entity_types_list,
            cmap='YlOrRd', annot=True, fmt='d', ax=ax, cbar_kws={'label': 'Co-occurrence'})
ax.set_title('Entity Co-occurrence Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/20_entity_cooccurrence.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 20_entity_cooccurrence.png")

# Chart 21: Sentence complexity comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(manual_df['token_count'], manual_df['entity_count'], alpha=0.5, c='#3498DB',
           edgecolors='black', linewidth=0.5, s=50, label='Manual')
ax.scatter(relabeled_df['token_count'], relabeled_df['entity_count'], alpha=0.5, c='#E74C3C',
           edgecolors='black', linewidth=0.5, s=50, label='Relabeled')
ax.set_xlabel('Token Count', fontsize=12)
ax.set_ylabel('Entity Count', fontsize=12)
ax.set_title('Token Count vs Entity Count', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/21_complexity_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 21_complexity_comparison.png")

# Chart 22: Label frequency ratio (Manual vs Relabeled)
fig, ax = plt.subplots(figsize=(14, 7))
common = set(manual_labels.keys()) & set(relabeled_labels.keys())
ratios = []
labels_for_plot = []
for label in common:
    if manual_labels[label] > 0:
        ratio = relabeled_labels[label] / manual_labels[label]
        ratios.append(ratio)
        labels_for_plot.append(label)

colors = ['#2ECC71' if r >= 1 else '#E74C3C' for r in ratios]
bars = ax.bar(labels_for_plot, ratios, color=colors, edgecolor='black')
ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Equal')
ax.set_xlabel('Label', fontsize=12)
ax.set_ylabel('Ratio (Relabeled / Manual)', fontsize=12)
ax.set_title('Label Frequency Ratio: Relabeled / Manual', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/22_label_ratio.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 22_label_ratio.png")

# Chart 23: Box plot - Token count by dataset
fig, ax = plt.subplots(figsize=(10, 6))
data = [manual_df['token_count'].values, relabeled_df['token_count'].values]
bp = ax.boxplot(data, labels=['Manual', 'Relabeled'], patch_artist=True)
colors_box = ['#3498DB', '#E74C3C']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax.set_ylabel('Token Count', fontsize=12)
ax.set_title('Token Count Distribution (Box Plot)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/23_token_boxplot.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 23_token_boxplot.png")

# Chart 24: Word frequency in entities
all_entity_words = []
for labels, tokens in zip(relabeled_df['labels_decoded'], relabeled_df['tokens'].apply(parse_list)):
    for i, label in enumerate(labels):
        if label.startswith('B-') and i < len(tokens):
            all_entity_words.append(tokens[i].lower())

entity_word_counts = Counter(all_entity_words).most_common(20)

fig, ax = plt.subplots(figsize=(12, 6))
words = [x[0] for x in entity_word_counts]
counts = [x[1] for x in entity_word_counts]
bars = ax.barh(words[::-1], counts[::-1], color='#F39C12', edgecolor='black')
ax.set_xlabel('Frequency', fontsize=12)
ax.set_title('Top 20 Entity Words (First Token)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/24_entity_words.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 24_entity_words.png")

# Chart 25: Summary Statistics Table
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

summary_data = [
    ['Metric', 'Manual Labeled', 'Relabeled (Custom)', 'Difference', '% Change'],
    ['Total Samples', f'{manual_df.shape[0]:,}', f'{relabeled_df.shape[0]:,}',
     f'{relabeled_df.shape[0]-manual_df.shape[0]:,}', f'{((relabeled_df.shape[0]-manual_df.shape[0])/manual_df.shape[0]*100):.1f}%'],
    ['Unique Labels', f'{len(manual_labels)}', f'{len(relabeled_labels)}',
     f'{len(relabeled_labels)-len(manual_labels)}', ''],
    ['Total Tokens', f'{sum(manual_labels.values()):,}', f'{sum(relabeled_labels.values()):,}',
     f'{sum(relabeled_labels.values())-sum(manual_labels.values()):,}',
     f'{(sum(relabeled_labels.values())-sum(manual_labels.values()))/sum(manual_labels.values())*100:.1f}%'],
    ['Entity Tokens', f'{sum(v for k,v in manual_labels.items() if k!="O"):,}',
     f'{sum(v for k,v in relabeled_labels.items() if k!="O"):,}',
     f'{sum(v for k,v in relabeled_labels.items() if k!="O") - sum(v for k,v in manual_labels.items() if k!="O"):,}', ''],
    ['Avg Tokens/Sentence', f'{manual_df["token_count"].mean():.1f}', f'{relabeled_df["token_count"].mean():.1f}',
     f'{relabeled_df["token_count"].mean()-manual_df["token_count"].mean():.1f}', ''],
    ['Avg Entities/Sentence', f'{manual_df["entity_count"].mean():.1f}', f'{relabeled_df["entity_count"].mean():.1f}',
     f'{relabeled_df["entity_count"].mean()-manual_df["entity_count"].mean():.1f}', ''],
]

colors_header = ['#3498DB', '#3498DB', '#E74C3C', '#9B59B6', '#9B59B6']
table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                 loc='center', cellLoc='center',
                 colColours=colors_header)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)
ax.set_title('Pipeline Summary Statistics', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/25_summary_table.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 25_summary_table.png")

# Chart 26: Stacked Bar - Entity types percentage
fig, ax = plt.subplots(figsize=(12, 6))
entity_pcts = {}
for e in set(manual_entities.keys()) | set(relabeled_entities.keys()):
    m_pct = manual_entities.get(e, 0) / sum(manual_entities.values()) * 100 if sum(manual_entities.values()) > 0 else 0
    r_pct = relabeled_entities.get(e, 0) / sum(relabeled_entities.values()) * 100 if sum(relabeled_entities.values()) > 0 else 0
    entity_pcts[e] = (m_pct, r_pct)

sorted_entities = sorted(entity_pcts.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)[:10]
entities = [x[0] for x in sorted_entities]
manual_pcts = [x[1][0] for x in sorted_entities]
relabeled_pcts = [x[1][1] for x in sorted_entities]

x = np.arange(len(entities))
width = 0.35

bars1 = ax.bar(x - width/2, manual_pcts, width, label='Manual', color='#3498DB', edgecolor='black')
bars2 = ax.bar(x + width/2, relabeled_pcts, width, label='Relabeled', color='#E74C3C', edgecolor='black')

ax.set_xlabel('Entity Type', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Entity Type Percentage Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(entities, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/26_entity_percentage.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 26_entity_percentage.png")

# Chart 27: Radar Chart for entity comparison
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

common_entities = sorted(set(manual_entities.keys()) & set(relabeled_entities.keys()))
common_entities = [e for e in common_entities if e != 'O'][:8]

if len(common_entities) > 2:
    angles = np.linspace(0, 2 * np.pi, len(common_entities), endpoint=False).tolist()
    angles += angles[:1]

    manual_vals = [manual_entities.get(e, 0) for e in common_entities]
    manual_vals += manual_vals[:1]
    relabeled_vals = [relabeled_entities.get(e, 0) for e in common_entities]
    relabeled_vals += relabeled_vals[:1]

    # Normalize
    max_val = max(max(manual_vals), max(relabeled_vals))
    manual_vals = [v/max_val for v in manual_vals]
    relabeled_vals = [v/max_val for v in relabeled_vals]

    ax.plot(angles, manual_vals, 'o-', linewidth=2, label='Manual', color='#3498DB')
    ax.fill(angles, manual_vals, alpha=0.25, color='#3498DB')
    ax.plot(angles, relabeled_vals, 'o-', linewidth=2, label='Relabeled', color='#E74C3C')
    ax.fill(angles, relabeled_vals, alpha=0.25, color='#E74C3C')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(common_entities)
    ax.set_title('Entity Type Radar Chart', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/27_radar_chart.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 27_radar_chart.png")

# Chart 28: Multi-level pie - Entity distribution by dataset
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Manual
manual_top = dict(sorted(manual_entities.items(), key=lambda x: x[1], reverse=True)[:7])
axes[0].pie(manual_top.values(), labels=manual_top.keys(), autopct='%1.1f%%',
            colors=plt.cm.Set2(np.linspace(0, 1, len(manual_top))), startangle=90)
axes[0].set_title('Manual - Top Entities', fontsize=12, fontweight='bold')

# Relabeled
relabeled_top = dict(sorted(relabeled_entities.items(), key=lambda x: x[1], reverse=True)[:7])
axes[1].pie(relabeled_top.values(), labels=relabeled_top.keys(), autopct='%1.1f%%',
            colors=plt.cm.Set3(np.linspace(0, 1, len(relabeled_top))), startangle=90)
axes[1].set_title('Relabeled - Top Entities', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/28_entity_comparison_pies.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 28_entity_comparison_pies.png")

# Chart 29: Histogram - Label length distribution
manual_label_lens = [len(l) for l in manual_labels.keys()]
relabeled_label_lens = [len(l) for l in relabeled_labels.keys()]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(manual_label_lens, bins=10, alpha=0.5, label='Manual', color='#3498DB', edgecolor='black')
ax.hist(relabeled_label_lens, bins=10, alpha=0.5, label='Relabeled', color='#E74C3C', edgecolor='black')
ax.set_xlabel('Label Length (characters)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Label Length Distribution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/29_label_length.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 29_label_length.png")

# Chart 30: Violin plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create data for violin
violin_data = []
labels_violin = []
for i, label in enumerate(sorted(common_labels, key=lambda x: manual_labels.get(x,0), reverse=True)[:10]):
    manual_count = manual_labels.get(label, 0)
    relabeled_count = relabeled_labels.get(label, 0)
    violin_data.append([manual_count, relabeled_count])
    labels_violin.append(label)

# Plot as grouped bar for simplicity
x = np.arange(len(labels_violin))
width = 0.35

bars1 = ax.bar(x - width/2, [v[0] for v in violin_data], width, label='Manual', color='#3498DB', edgecolor='black')
bars2 = ax.bar(x + width/2, [v[1] for v in violin_data], width, label='Relabeled', color='#E74C3C', edgecolor='black')

ax.set_xlabel('Label', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Top 10 Labels Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels_violin, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/30_top_labels_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 30_top_labels_comparison.png")

print("\n" + "=" * 70)
print(f"COMPLETED! Generated 30 charts in {CHARTS_DIR}")
print("=" * 70)

# List all files
import glob
chart_files = sorted(glob.glob(f"{CHARTS_DIR}/*.png"))
print(f"\nTotal Charts Generated: {len(chart_files)}")
for f in chart_files:
    print(f"  - {os.path.basename(f)}")
