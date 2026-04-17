"""
Generate comprehensive charts for NER Pipeline Analysis
Analyzes: Crawl -> Preprocessing -> Encoding -> Custom Mapping vs SpaCy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast
import re
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory
CHARTS_DIR = "D:/Python/NLP/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

print("=" * 60)
print("GENERATING CHARTS FOR NER PIPELINE ANALYSIS")
print("=" * 60)

# ============================================================================
# 1. CRAWL DATA ANALYSIS
# ============================================================================
print("\n[1/6] Loading Crawl Data...")
crawl_df = pd.read_csv("Day1_Crawl/ket_qua_sentence.csv")
print(f"   Crawl Dataset: {crawl_df.shape[0]} rows, {crawl_df.shape[1]} columns")

# Chart 1: Dataset size comparison (Crawl vs Preprocessed)
fig, ax = plt.subplots(figsize=(10, 6))
stages = ['Crawl\n(Day1)', 'After Preprocessing\n(Day2)', 'After Encoding\n(Day3)', 'After Custom Mapping\n(Day4)']
sizes = [crawl_df.shape[0], 0, 0, 0]  # Will update with actual values
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# Load other datasets to get sizes
try:
    clean_df = pd.read_csv("clean_news.csv")
    sizes[1] = clean_df.shape[0]
except:
    sizes[1] = crawl_df.shape[0]

try:
    encoded_df = pd.read_csv("Day3_Encoding/ner_dataset_view.csv")
    sizes[2] = encoded_df.shape[0]
except:
    sizes[2] = 0

try:
    relabeled_df = pd.read_csv("relabeled_output.csv")
    sizes[3] = relabeled_df.shape[0]
except:
    sizes[3] = 0

bars = ax.bar(stages, sizes, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Dataset Size at Each Pipeline Stage', fontsize=14, fontweight='bold')

for bar, size in zip(bars, sizes):
    if size > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/01_dataset_size_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: 01_dataset_size_comparison.png")

# Chart 2: Sentence length distribution (Crawl)
crawl_df['sentence_len'] = crawl_df['Sentence'].astype(str).apply(len)
crawl_df['word_count'] = crawl_df['Sentence'].astype(str).apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Character length distribution
axes[0].hist(crawl_df['sentence_len'], bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
axes[0].axvline(crawl_df['sentence_len'].mean(), color='red', linestyle='--', label=f"Mean: {crawl_df['sentence_len'].mean():.0f}")
axes[0].set_xlabel('Character Length', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Sentence Character Length Distribution (Crawl)', fontsize=12, fontweight='bold')
axes[0].legend()

# Word count distribution
axes[1].hist(crawl_df['word_count'], bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
axes[1].axvline(crawl_df['word_count'].mean(), color='red', linestyle='--', label=f"Mean: {crawl_df['word_count'].mean():.1f}")
axes[1].set_xlabel('Word Count', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Word Count Distribution (Crawl)', fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/02_crawl_sentence_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: 02_crawl_sentence_distributions.png")

# Chart 3: Title word cloud / Top words in titles
title_words = ' '.join(crawl_df['Title'].astype(str).tolist()).lower()
title_word_counts = Counter(re.findall(r'\b\w+\b', title_words))
top_title_words = dict(title_word_counts.most_common(20))

fig, ax = plt.subplots(figsize=(12, 6))
words = list(top_title_words.keys())
counts = list(top_title_words.values())
bars = ax.barh(words[::-1], counts[::-1], color='#45B7D1', edgecolor='black')
ax.set_xlabel('Frequency', fontsize=11)
ax.set_title('Top 20 Most Common Words in Titles (Crawl Data)', fontsize=14, fontweight='bold')

for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/03_crawl_title_word_frequency.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: 03_crawl_title_word_frequency.png")

# Chart 4: Missing values heatmap
fig, ax = plt.subplots(figsize=(10, 5))
missing = crawl_df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:
    bars = ax.bar(missing.index, missing.values, color='#FF6B6B', edgecolor='black')
    ax.set_ylabel('Missing Count', fontsize=11)
    ax.set_title('Missing Values in Crawl Data', fontsize=14, fontweight='bold')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(bar.get_height()), ha='center', fontsize=10)
else:
    ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
    ax.set_title('Missing Values in Crawl Data', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/04_crawl_missing_values.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: 04_crawl_missing_values.png")

# ============================================================================
# 2. PREPROCESSING DATA ANALYSIS
# ============================================================================
print("\n[2/6] Loading Preprocessing Data...")

try:
    clean_df = pd.read_csv("clean_news.csv")
    print(f"   Preprocessed Dataset: {clean_df.shape[0]} rows")

    # Chart 5: Before vs After preprocessing comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sample some data to show transformation
    sample_idx = min(5, len(crawl_df))
    x = np.arange(sample_idx)
    width = 0.35

    crawl_lens = crawl_df['Sentence'].astype(str).apply(len).values[:sample_idx]
    clean_lens = clean_df['clean_sentence'].astype(str).apply(len).values[:sample_idx]

    bars1 = ax.bar(x - width/2, crawl_lens, width, label='Before (Sentence)', color='#FF6B6B', edgecolor='black')
    bars2 = ax.bar(x + width/2, clean_lens, width, label='After (clean_sentence)', color='#4ECDC4', edgecolor='black')

    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Character Length', fontsize=11)
    ax.set_title('Before vs After Preprocessing (Character Length)', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/05_preprocessing_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 05_preprocessing_comparison.png")

    # Chart 6: Cleaning impact statistics
    clean_df['original_len'] = crawl_df['Sentence'].astype(str).apply(len).values[:len(clean_df)]
    clean_df['clean_len'] = clean_df['clean_sentence'].astype(str).apply(len)
    clean_df['reduction_pct'] = ((clean_df['original_len'] - clean_df['clean_len']) / clean_df['original_len'] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reduction percentage distribution
    axes[0].hist(clean_df['reduction_pct'].dropna(), bins=30, color='#96CEB4', edgecolor='black', alpha=0.7)
    axes[0].axvline(clean_df['reduction_pct'].mean(), color='red', linestyle='--',
                    label=f"Mean: {clean_df['reduction_pct'].mean():.1f}%")
    axes[0].set_xlabel('Reduction Percentage (%)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Character Reduction After Cleaning', fontsize=12, fontweight='bold')
    axes[0].legend()

    # Before vs After scatter
    axes[1].scatter(clean_df['original_len'], clean_df['clean_len'], alpha=0.5, c='#45B7D1', edgecolors='black', linewidth=0.5)
    max_val = max(clean_df['original_len'].max(), clean_df['clean_len'].max())
    axes[1].plot([0, max_val], [0, max_val], 'r--', label='No Change Line')
    axes[1].set_xlabel('Original Length', fontsize=11)
    axes[1].set_ylabel('Clean Length', fontsize=11)
    axes[1].set_title('Original vs Clean Length', fontsize=12, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/06_preprocessing_impact.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 06_preprocessing_impact.png")

except Exception as e:
    print(f"   Warning: Could not load preprocessing data: {e}")

# ============================================================================
# 3. ENCODING DATA ANALYSIS (SpaCy)
# ============================================================================
print("\n[3/6] Loading SpaCy Encoding Data...")

try:
    encoded_df = pd.read_csv("Day3_Encoding/ner_dataset_view.csv")
    print(f"   Encoded Dataset: {encoded_df.shape[0]} rows")

    # Parse encoded labels
    def parse_list(x):
        try:
            return ast.literal_eval(x)
        except:
            return []

    encoded_df['labels_decoded'] = encoded_df['labels'].apply(parse_list)

    # Get label distribution
    all_labels = []
    for labels in encoded_df['labels_decoded']:
        all_labels.extend(labels)

    label_counts = Counter(all_labels)

    # Chart 7: Label distribution (SpaCy)
    fig, ax = plt.subplots(figsize=(14, 7))
    labels_sorted = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    label_names = [x[0] for x in labels_sorted]
    label_values = [x[1] for x in labels_sorted]

    colors = plt.cm.viridis(np.linspace(0, 1, len(label_names)))
    bars = ax.bar(label_names, label_values, color=colors, edgecolor='black')
    ax.set_xlabel('NER Labels', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('NER Label Distribution (SpaCy Encoding)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/07_spacy_label_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 07_spacy_label_distribution.png")

    # Chart 8: Entity type pie chart
    entity_types = {}
    for label, count in label_counts.items():
        if label == 'O':
            entity_types['O'] = count
        elif label.startswith('B-') or label.startswith('I-'):
            entity_type = label[2:]
            entity_types[entity_type] = entity_types.get(entity_type, 0) + count

    fig, ax = plt.subplots(figsize=(12, 8))
    entity_names = list(entity_types.keys())
    entity_values = list(entity_types.values())

    # Sort by value
    sorted_entities = sorted(zip(entity_names, entity_values), key=lambda x: x[1], reverse=True)
    entity_names = [x[0] for x in sorted_entities]
    entity_values = [x[1] for x in sorted_entities]

    colors = plt.cm.Set3(np.linspace(0, 1, len(entity_names)))
    wedges, texts, autotexts = ax.pie(entity_values, labels=entity_names, autopct='%1.1f%%',
                                        colors=colors, startangle=90, pctdistance=0.75)
    ax.set_title('Entity Type Distribution (SpaCy)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/08_spacy_entity_pie.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 08_spacy_entity_pie.png")

    # Chart 9: Token count per sentence
    encoded_df['token_count'] = encoded_df['labels_decoded'].apply(len)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(encoded_df['token_count'], bins=30, color='#9B59B6', edgecolor='black', alpha=0.7)
    ax.axvline(encoded_df['token_count'].mean(), color='red', linestyle='--',
               label=f"Mean: {encoded_df['token_count'].mean():.1f}")
    ax.axvline(encoded_df['token_count'].median(), color='blue', linestyle='--',
               label=f"Median: {encoded_df['token_count'].median():.0f}")
    ax.set_xlabel('Token Count', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Token Count Distribution per Sentence (SpaCy)', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/09_spacy_token_count.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 09_spacy_token_count.png")

    # Chart 10: B-I-O distribution
    bio_counts = {'B- (Begin)': 0, 'I- (Inside)': 0, 'O (Outside)': 0}
    for label in all_labels:
        if label.startswith('B-'):
            bio_counts['B- (Begin)'] += 1
        elif label.startswith('I-'):
            bio_counts['I- (Inside)'] += 1
        elif label == 'O':
            bio_counts['O (Outside)'] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    bars = ax.bar(bio_counts.keys(), bio_counts.values(), color=colors, edgecolor='black')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('BIO Tag Distribution (SpaCy)', fontsize=14, fontweight='bold')

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'{bar.get_height():,}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/10_spacy_bio_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 10_spacy_bio_distribution.png")

except Exception as e:
    print(f"   Warning: Could not load SpaCy encoding data: {e}")

# ============================================================================
# 4. CUSTOM MAPPING DATA ANALYSIS
# ============================================================================
print("\n[4/6] Loading Custom Mapping Data...")

try:
    relabeled_df = pd.read_csv("relabeled_output.csv")
    print(f"   Custom Mapping Dataset: {relabeled_df.shape[0]} rows")

    # Parse labels
    relabeled_df['labels_decoded'] = relabeled_df['labels'].apply(parse_list)

    # Get label distribution
    all_custom_labels = []
    for labels in relabeled_df['labels_decoded']:
        all_custom_labels.extend(labels)

    custom_label_counts = Counter(all_custom_labels)

    # Chart 11: Label distribution (Custom Mapping)
    fig, ax = plt.subplots(figsize=(14, 7))
    labels_sorted = sorted(custom_label_counts.items(), key=lambda x: x[1], reverse=True)
    label_names = [x[0] for x in labels_sorted]
    label_values = [x[1] for x in labels_sorted]

    colors = plt.cm.plasma(np.linspace(0, 1, len(label_names)))
    bars = ax.bar(label_names, label_values, color=colors, edgecolor='black')
    ax.set_xlabel('NER Labels', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('NER Label Distribution (Custom Mapping)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/11_custom_label_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 11_custom_label_distribution.png")

    # Chart 12: Entity type pie chart (Custom)
    custom_entity_types = {}
    for label, count in custom_label_counts.items():
        if label == 'O':
            custom_entity_types['O'] = count
        elif label.startswith('B-') or label.startswith('I-'):
            entity_type = label[2:]
            custom_entity_types[entity_type] = custom_entity_types.get(entity_type, 0) + count

    fig, ax = plt.subplots(figsize=(12, 8))
    entity_names = list(custom_entity_types.keys())
    entity_values = list(custom_entity_types.values())

    sorted_entities = sorted(zip(entity_names, entity_values), key=lambda x: x[1], reverse=True)
    entity_names = [x[0] for x in sorted_entities]
    entity_values = [x[1] for x in sorted_entities]

    colors = plt.cm.Paired(np.linspace(0, 1, len(entity_names)))
    wedges, texts, autotexts = ax.pie(entity_values, labels=entity_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90, pctdistance=0.75)
    ax.set_title('Entity Type Distribution (Custom Mapping)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/12_custom_entity_pie.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 12_custom_entity_pie.png")

except Exception as e:
    print(f"   Warning: Could not load custom mapping data: {e}")

# ============================================================================
# 5. COMPARISON: SPACY vs CUSTOM MAPPING
# ============================================================================
print("\n[5/6] Comparing SpaCy vs Custom Mapping...")

try:
    # Chart 13: Side-by-side label comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # SpaCy labels
    spacy_sorted = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    spacy_labels = [x[0] for x in spacy_sorted]
    spacy_values = [x[1] for x in spacy_sorted]

    axes[0].barh(spacy_labels[::-1], spacy_values[::-1], color='#3498DB', edgecolor='black')
    axes[0].set_xlabel('Count', fontsize=11)
    axes[0].set_title('Top 15 Labels (SpaCy)', fontsize=12, fontweight='bold')

    # Custom labels
    custom_sorted = sorted(custom_label_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    custom_labels = [x[0] for x in custom_sorted]
    custom_values = [x[1] for x in custom_sorted]

    axes[1].barh(custom_labels[::-1], custom_values[::-1], color='#E74C3C', edgecolor='black')
    axes[1].set_xlabel('Count', fontsize=11)
    axes[1].set_title('Top 15 Labels (Custom Mapping)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/13_spacy_vs_custom_label_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 13_spacy_vs_custom_label_comparison.png")

    # Chart 14: Entity type comparison
    fig, ax = plt.subplots(figsize=(14, 7))

    all_entity_types = sorted(set(list(entity_types.keys()) + list(custom_entity_types.keys())))

    x = np.arange(len(all_entity_types))
    width = 0.35

    spacy_vals = [entity_types.get(e, 0) for e in all_entity_types]
    custom_vals = [custom_entity_types.get(e, 0) for e in all_entity_types]

    bars1 = ax.bar(x - width/2, spacy_vals, width, label='SpaCy', color='#3498DB', edgecolor='black')
    bars2 = ax.bar(x + width/2, custom_vals, width, label='Custom Mapping', color='#E74C3C', edgecolor='black')

    ax.set_xlabel('Entity Type', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Entity Type Comparison: SpaCy vs Custom Mapping', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_entity_types, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/14_entity_type_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 14_entity_type_comparison.png")

    # Chart 15: Label changes analysis (what changed)
    # Compare label by label for same sentences
    common_labels = set(label_counts.keys()) & set(custom_label_counts.keys())

    changed_labels = {}
    for label in common_labels:
        diff = custom_label_counts[label] - label_counts.get(label, 0)
        if diff != 0:
            changed_labels[label] = diff

    fig, ax = plt.subplots(figsize=(12, 6))
    if changed_labels:
        labels = list(changed_labels.keys())
        values = list(changed_labels.values())
        colors = ['#2ECC71' if v > 0 else '#E74C3C' for v in values]
        bars = ax.bar(labels, values, color=colors, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Label', fontsize=11)
        ax.set_ylabel('Change (Custom - SpaCy)', fontsize=11)
        ax.set_title('Label Count Changes: Custom Mapping vs SpaCy', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No Significant Changes', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/15_label_changes.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 15_label_changes.png")

    # Chart 16: BIO distribution comparison
    bio_spacy = {'B-': 0, 'I-': 0, 'O': 0}
    bio_custom = {'B-': 0, 'I-': 0, 'O': 0}

    for label in all_labels:
        if label.startswith('B-'):
            bio_spacy['B-'] += 1
        elif label.startswith('I-'):
            bio_spacy['I-'] += 1
        elif label == 'O':
            bio_spacy['O'] += 1

    for label in all_custom_labels:
        if label.startswith('B-'):
            bio_custom['B-'] += 1
        elif label.startswith('I-'):
            bio_custom['I-'] += 1
        elif label == 'O':
            bio_custom['O'] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(x - width/2, [bio_spacy['B-'], bio_spacy['I-'], bio_spacy['O']],
                   width, label='SpaCy', color='#3498DB', edgecolor='black')
    bars2 = ax.bar(x + width/2, [bio_custom['B-'], bio_custom['I-'], bio_custom['O']],
                   width, label='Custom Mapping', color='#E74C3C', edgecolor='black')

    ax.set_xlabel('BIO Tag', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('BIO Distribution Comparison: SpaCy vs Custom Mapping', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Begin (B-)', 'Inside (I-)', 'Outside (O)'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/16_bio_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 16_bio_comparison.png")

except Exception as e:
    print(f"   Warning: Could not generate comparison charts: {e}")

# ============================================================================
# 6. ADDITIONAL ANALYSIS
# ============================================================================
print("\n[6/6] Generating additional analysis charts...")

# Chart 17: Word cloud style - most tagged entities
try:
    # Find most common entities (non-O labels)
    entity_mentions = []
    for labels in relabeled_df['labels_decoded']:
        for i, label in enumerate(labels):
            if label != 'O' and label.startswith('B-'):
                entity_mentions.append(label[2:])

    entity_mention_counts = Counter(entity_mentions)

    fig, ax = plt.subplots(figsize=(12, 6))
    top_entities = entity_mention_counts.most_common(15)
    entities = [x[0] for x in top_entities]
    counts = [x[1] for x in top_entities]

    bars = ax.barh(entities[::-1], counts[::-1], color='#9B59B6', edgecolor='black')
    ax.set_xlabel('Frequency', fontsize=11)
    ax.set_title('Top 15 Most Common Named Entities (Custom Mapping)', fontsize=14, fontweight='bold')

    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/17_top_entities.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 17_top_entities.png")
except Exception as e:
    print(f"   Warning: {e}")

# Chart 18: Sentence complexity analysis
try:
    relabeled_df['entity_count'] = relabeled_df['labels_decoded'].apply(
        lambda x: sum(1 for l in x if l.startswith('B-'))
    )
    relabeled_df['token_count'] = relabeled_df['labels_decoded'].apply(len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Entity count distribution
    axes[0].hist(relabeled_df['entity_count'], bins=20, color='#1ABC9C', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Entities per Sentence', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Entity Count Distribution', fontsize=12, fontweight='bold')

    # Token count vs Entity count scatter
    axes[1].scatter(relabeled_df['token_count'], relabeled_df['entity_count'],
                    alpha=0.5, c='#E67E22', edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('Token Count', fontsize=11)
    axes[1].set_ylabel('Entity Count', fontsize=11)
    axes[1].set_title('Token Count vs Entity Count', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/18_entity_complexity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 18_entity_complexity.png")
except Exception as e:
    print(f"   Warning: {e}")

# Chart 19: Heatmap of entity co-occurrence
try:
    # Create entity co-occurrence matrix
    entity_types_list = list(custom_entity_types.keys())
    entity_types_list = [e for e in entity_types_list if e != 'O']

    cooccurrence = {e1: {e2: 0 for e2 in entity_types_list} for e1 in entity_types_list}

    for labels in relabeled_df['labels_decoded']:
        entities_in_sentence = []
        for label in labels:
            if label.startswith('B-'):
                entities_in_sentence.append(label[2:])

        for e1 in entities_in_sentence:
            for e2 in entities_in_sentence:
                if e1 != e2 and e1 in cooccurrence and e2 in cooccurrence[e1]:
                    cooccurrence[e1][e2] += 1

    # Convert to matrix
    cooccurrence_matrix = np.array([[cooccurrence[e1][e2] for e2 in entity_types_list]
                                      for e1 in entity_types_list])

    # Only keep top entities for readability
    top_n = min(10, len(entity_types_list))
    top_entity_names = [x[0] for x in sorted(custom_entity_types.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    top_entity_names = [e for e in top_entity_names if e != 'O']

    matrix_filtered = np.array([[cooccurrence[e1][e2] for e2 in top_entity_names]
                                 for e1 in top_entity_names])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix_filtered, xticklabels=top_entity_names, yticklabels=top_entity_names,
                cmap='YlOrRd', annot=True, fmt='d', ax=ax, cbar_kws={'label': 'Co-occurrence Count'})
    ax.set_title('Entity Co-occurrence Heatmap (Top Entities)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/19_entity_cooccurrence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 19_entity_cooccurrence.png")
except Exception as e:
    print(f"   Warning: {e}")

# Chart 20: Summary statistics table
try:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Create summary data
    summary_data = [
        ['Stage', 'Samples', 'Unique Labels', 'Total Entities', 'Avg Tokens/Sentence'],
        ['Crawl (Day1)', f'{crawl_df.shape[0]:,}', '-', '-', f'{crawl_df["Sentence"].astype(str).apply(lambda x: len(x.split())).mean():.1f}'],
    ]

    if 'clean_df' in dir():
        summary_data.append(['Preprocessing (Day2)', f'{clean_df.shape[0]:,}', '-', '-',
                            f'{clean_df["clean_sentence"].astype(str).apply(lambda x: len(x.split())).mean():.1f}'])

    if 'encoded_df' in dir():
        summary_data.append(['Encoding (Day3 - SpaCy)', f'{encoded_df.shape[0]:,}', f'{len(label_counts)}',
                            f'{sum(v for k,v in label_counts.items() if k!="O"):,}',
                            f'{encoded_df["token_count"].mean():.1f}'])

    if 'relabeled_df' in dir():
        summary_data.append(['Custom Mapping (Day4)', f'{relabeled_df.shape[0]:,}', f'{len(custom_label_counts)}',
                            f'{sum(v for k,v in custom_label_counts.items() if k!="O"):,}',
                            f'{relabeled_df["token_count"].mean():.1f}'])

    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     loc='center', cellLoc='center',
                     colColours=['#3498DB']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    ax.set_title('Pipeline Summary Statistics', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/20_summary_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 20_summary_statistics.png")
except Exception as e:
    print(f"   Warning: {e}")

print("\n" + "=" * 60)
print(f"COMPLETED! Generated 20 charts in {CHARTS_DIR}")
print("=" * 60)

# List all generated files
import glob
chart_files = sorted(glob.glob(f"{CHARTS_DIR}/*.png"))
print(f"\nGenerated Charts:")
for f in chart_files:
    print(f"  - {os.path.basename(f)}")
