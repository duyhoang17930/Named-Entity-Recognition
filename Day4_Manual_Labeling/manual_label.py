import pandas as pd
import re
import os
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

INPUT_FILE = "clean_news.csv"
OUTPUT_FILE = "manual_labeled.csv"
COMPLETE_FILE = "complete.csv"

# Label set
LABEL_SET = [
    'O',
    'B-CARDINAL','I-CARDINAL',
    'B-DATE','I-DATE',
    'B-FAC','I-FAC',
    'B-GPE','I-GPE',
    'B-LANGUAGE',
    'B-LOC','I-LOC',
    'B-MONEY','I-MONEY',
    'B-NORP','I-NORP',
    'B-ORDINAL',
    'B-ORG','I-ORG',
    'B-PERSON','I-PERSON',
    'B-PRODUCT','I-PRODUCT',
    'B-QUANTITY','I-QUANTITY',
    'B-TIME','I-TIME',
    'B-WORK_OF_ART'
]

print("Loading NER model...")
from transformers import pipeline
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print("Model loaded!")

import nltk

def tokenize(text):
    return nltk.word_tokenize(text)

def auto_label(sentence):
    """Auto-label using transformers NER with pattern rules"""
    ner_results = ner_pipeline(sentence)
    tokens = tokenize(sentence)
    labels = ["O"] * len(tokens)

    # Map NER results
    category_map = {
        'PER': 'PERSON',
        'ORG': 'ORG',
        'LOC': 'LOC',
        'MISC': 'NORP'
    }

    entity_positions = {}
    for result in ner_results:
        word = result['word'].lower()
        entity_group = result['entity_group']
        mapped = category_map.get(entity_group, entity_group)

        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if token_lower == word or token_lower.startswith(word) or word.startswith(token_lower):
                if len(word) > 1 or len(token) > 1:
                    entity_positions[i] = mapped

    # Apply B-/I- labeling
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

    # Pattern-based rules
    for i, token in enumerate(tokens):
        t = token.strip()
        if t in [',', '.', '!', '?', ';', ':', '(', ')', '"', "'"]:
            continue
        if labels[i] not in ['O', 'B-NORP', 'I-NORP']:
            continue

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

# Load data
df = pd.read_csv(INPUT_FILE)
sentences = df["Sentence"].dropna().tolist()

# Load existing labeled data (keep the first 200 lines)
if os.path.exists(OUTPUT_FILE):
    labeled_df = pd.read_csv(OUTPUT_FILE)
    existing_count = len(labeled_df)
    print(f"Existing labeled: {existing_count}")
else:
    labeled_df = pd.DataFrame(columns=["tokens", "labels"])
    existing_count = 0

if os.path.exists(COMPLETE_FILE):
    complete_df = pd.read_csv(COMPLETE_FILE)
else:
    complete_df = pd.DataFrame(columns=["sentence", "tokens", "labels"])

print(f"Auto-labeling {len(sentences)} remaining sentences...")

# Auto-label all remaining sentences
new_labeled = []
new_complete = []

for i, sentence in enumerate(sentences):
    if i % 500 == 0:
        print(f"Processing {i}/{len(sentences)}...")

    tokens, labels = auto_label(str(sentence))

    new_labeled.append({
        "tokens": tokens,
        "labels": labels
    })

    new_complete.append({
        "sentence": str(sentence),
        "tokens": tokens,
        "labels": labels
    })

# Append new labeled data
labeled_df = pd.concat([labeled_df, pd.DataFrame(new_labeled)], ignore_index=True)
labeled_df.to_csv(OUTPUT_FILE, index=False)

complete_df = pd.concat([complete_df, pd.DataFrame(new_complete)], ignore_index=True)
complete_df.to_csv(COMPLETE_FILE, index=False)

# Clear clean_news.csv (all processed)
empty_df = pd.DataFrame(columns=["Sentence"])
empty_df.to_csv(INPUT_FILE, index=False)

print(f"\n✅ Done! Auto-labeled {len(new_labeled)} sentences")
print(f"Total in {OUTPUT_FILE}: {len(labeled_df)}")
print(f"Total in {COMPLETE_FILE}: {len(complete_df)}")
