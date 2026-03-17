import pandas as pd
import numpy as np
import re

# Read the source data - use the original Sentence column for better NER
df = pd.read_csv("clean_news.csv")
sentences = df["Sentence"].dropna().tolist()

# New label set provided by user
LABEL_SET = {
    'B-CARDINAL': 0, 'B-DATE': 1, 'B-FAC': 2, 'B-GPE': 3, 'B-LANGUAGE': 4,
    'B-LOC': 5, 'B-MONEY': 6, 'B-NORP': 7, 'B-ORDINAL': 8, 'B-ORG': 9,
    'B-PERSON': 10, 'B-PRODUCT': 11, 'B-QUANTITY': 12, 'B-TIME': 13,
    'B-WORK_OF_ART': 14, 'I-CARDINAL': 15, 'I-DATE': 16, 'I-FAC': 17,
    'I-GPE': 18, 'I-LOC': 19, 'I-MONEY': 20, 'I-NORP': 21, 'I-ORG': 22,
    'I-PERSON': 23, 'I-PRODUCT': 24, 'I-QUANTITY': 25, 'I-TIME': 26, 'O': 27
}

print("Loading NER model...")
from transformers import pipeline
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print("Using transformers NER model")

def tokenize_simple(text):
    """Simple tokenizer"""
    tokens = re.findall(r"[\w']+|[.,!?;:()\"']", text)
    return [t for t in tokens if t.strip()]

def label_with_transformers(sentence):
    """Label using transformers NER with proper mapping"""
    # Get NER results
    ner_results = ner_pipeline(sentence)

    tokens = tokenize_simple(sentence)
    labels = ["O"] * len(tokens)

    # Map NER results to tokens
    # dslim/bert-base-NER: PER, ORG, LOC, MISC
    category_map = {
        'PER': 'PERSON',
        'ORG': 'ORG',
        'LOC': 'LOC',
        'MISC': 'NORP'
    }

    # Track entity positions for proper B-/I- labeling
    entity_positions = {}
    for result in ner_results:
        word = result['word'].lower()
        entity_group = result['entity_group']
        mapped = category_map.get(entity_group, entity_group)

        # Try to find matching token
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if token_lower == word or token_lower.startswith(word) or word.startswith(token_lower):
                if len(word) > 1 or len(token) > 1:
                    entity_positions[i] = mapped

    # Apply B-/I- labeling based on position
    sorted_positions = sorted(entity_positions.keys())
    prev_pos = -2
    prev_entity = None

    for pos in sorted_positions:
        entity_type = entity_positions[pos]
        if pos == prev_pos + 1 and entity_type == prev_entity:
            labels[pos] = 'I-' + entity_type
        else:
            labels[pos] = 'B-' + entity_type
        prev_pos = pos
        prev_entity = entity_type

    # Apply pattern-based rules for DATE, MONEY, CARDINAL, ORDINAL
    for i, token in enumerate(tokens):
        t = token.strip()

        if t in [',', '.', '!', '?', ';', ':', '(', ')', '"', "'"]:
            continue

        # Skip if already labeled as named entity (except if it's a weaker category)
        if labels[i] not in ['O', 'B-NORP', 'I-NORP']:
            continue

        # Date patterns
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        if t in months or t in days:
            labels[i] = 'B-DATE'
        elif re.match(r'^(19|20)\d{2}$', t):
            labels[i] = 'B-DATE'
        elif re.match(r'^\$[\d,]+(\.\d{2})?$', t):
            labels[i] = 'B-MONEY'
        elif re.match(r'^\d+(,\d{3})*(\.\d+)?$', t):
            labels[i] = 'B-CARDINAL'
        elif re.match(r'^\d+(st|nd|rd|th)$', t):
            labels[i] = 'B-ORDINAL'

    return tokens, labels

# Build vocabulary
token2id = {}
id2token = {}
idx = 1

dataset = []

print("Processing sentences...")
for i, sentence in enumerate(sentences):
    if i % 500 == 0:
        print(f"Processing {i}/{len(sentences)}...")

    tokens, labels = label_with_transformers(str(sentence))

    # Build vocabulary
    for token in tokens:
        if token not in token2id:
            token2id[token] = idx
            id2token[idx] = token
            idx += 1

    # Convert labels to numeric IDs
    label_ids = []
    for label in labels:
        if label in LABEL_SET:
            label_ids.append(LABEL_SET[label])
        else:
            label_ids.append(LABEL_SET['O'])

    dataset.append({
        'tokens': tokens,
        'labels': labels,
        'token_ids': [token2id[t] for t in tokens],
        'label_ids': label_ids
    })

print(f"Total unique tokens: {len(token2id)}")
print(f"Total sentences: {len(dataset)}")

# Save to CSV
MAX_LEN = 50

def pad(seq, max_len, pad_value=0):
    return seq[:max_len] + [pad_value] * (max_len - len(seq))

rows = []
for item in dataset:
    token_ids = pad(item['token_ids'], MAX_LEN)
    label_ids = pad(item['label_ids'], MAX_LEN)
    rows.append({
        'tokens': str(token_ids),
        'labels': str(label_ids)
    })

result_df = pd.DataFrame(rows)
result_df.to_csv('ner_dataset_view.csv', index=False)

print("\nSaved to ner_dataset_view.csv")

# Print some examples
print("\nExample labeling (first 10 sentences with entities):")
entity_count = 0
for i in range(len(dataset)):
    has_entity = any(l != 'O' for l in dataset[i]['labels'])
    if has_entity and entity_count < 10:
        print(f"\n--- Sentence {i+1} ---")
        print(f"  Text: {' '.join(dataset[i]['tokens'][:25])}...")
        print(f"  Entities:")
        for token, label in zip(dataset[i]['tokens'], dataset[i]['labels']):
            if label != 'O':
                print(f"    {token}: {label}")
        entity_count += 1

# Print label distribution
label_counts = {}
for item in dataset:
    for label in item['labels']:
        label_counts[label] = label_counts.get(label, 0) + 1

print("\n\nLabel distribution:")
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count}")
