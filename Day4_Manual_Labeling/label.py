import pandas as pd
import re
import os
from transformers import pipeline

INPUT_FILE = "Day4_Manual_Labeling/clean_news.csv"
OUTPUT_FILE = "Day4_Manual_Labeling/manual_labeled.csv"   # file đã có data tay
COMPLETE_FILE = "complete.csv"
MAX_SENTENCES = 50   # xử lý mỗi lần

# Load model
print("Loading Transformer NER...")
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print("Model loaded!")

# Label mapping (giữ giống bạn)
category_map = {
    'PER': 'PERSON',
    'ORG': 'ORG',
    'LOC': 'LOC',
    'MISC': 'NORP'
}

def tokenize(text):
    return re.findall(r"[\w']+|[.,!?;:()\"']", text)

# LABEL SET của bạn
LABEL_SET = set([
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
])

# mapping từ model → label của bạn
category_map = {
    'PER': 'PERSON',
    'ORG': 'ORG',
    'LOC': 'LOC',
    'MISC': 'NORP'
}

def normalize_label(prefix, entity):
    label = f"{prefix}-{entity}"
    return label if label in LABEL_SET else "O"


def label_with_transformers(sentence):
    tokens = tokenize(sentence)
    labels = ["O"] * len(tokens)

    ner_results = ner_pipeline(sentence)

    entity_positions = {}

    for result in ner_results:
        word = result['word'].lower()
        entity_group = result['entity_group']

        mapped = category_map.get(entity_group, entity_group)

        for i, token in enumerate(tokens):
            token_lower = token.lower()

            if token_lower == word or token_lower in word or word in token_lower:
                entity_positions[i] = mapped

    # B-I logic + FILTER LABEL_SET
    sorted_positions = sorted(entity_positions.keys())

    prev_pos = -2
    prev_entity = None

    for pos in sorted_positions:
        entity_type = entity_positions[pos]

        if pos == prev_pos + 1 and entity_type == prev_entity:
            labels[pos] = normalize_label("I", entity_type)
        else:
            labels[pos] = normalize_label("B", entity_type)

        prev_pos = pos
        prev_entity = entity_type

    # RULE BOOST (cũng phải normalize)
    for i, token in enumerate(tokens):

        if labels[i] != "O":
            continue

        t = token.strip()

        if re.match(r'^(19|20)\d{2}$', t):
            labels[i] = normalize_label("B", "DATE")

        elif re.match(r'^\d+(,\d{3})*$', t):
            labels[i] = normalize_label("B", "CARDINAL")

        elif re.match(r'^\$[\d,]+', t):
            labels[i] = normalize_label("B", "MONEY")

    return tokens, labels


# Load data
df = pd.read_csv(INPUT_FILE)

if os.path.exists(OUTPUT_FILE):
    labeled_df = pd.read_csv(OUTPUT_FILE)
else:
    labeled_df = pd.DataFrame(columns=["tokens", "labels"])

if os.path.exists(COMPLETE_FILE):
    complete_df = pd.read_csv(COMPLETE_FILE)
else:
    complete_df = pd.DataFrame(columns=["sentence", "tokens", "labels"])


print(f"Processing {min(MAX_SENTENCES, len(df))} sentences...")

i = 0
count = 0

while i < len(df) and count < MAX_SENTENCES:

    sentence = str(df.loc[i, "Sentence"])

    print(f"\n[{count}] {sentence[:80]}...")

    tokens, labels = label_with_transformers(sentence)

    # Save
    new_row1 = pd.DataFrame([{
        "tokens": tokens,
        "labels": labels
    }])
    labeled_df = pd.concat([labeled_df, new_row1], ignore_index=True)

    new_row2 = pd.DataFrame([{
        "sentence": sentence,
        "tokens": tokens,
        "labels": labels
    }])
    complete_df = pd.concat([complete_df, new_row2], ignore_index=True)

    # remove processed
    df = df.drop(index=i).reset_index(drop=True)

    count += 1


# Save files
labeled_df.to_csv(OUTPUT_FILE, index=False)
complete_df.to_csv(COMPLETE_FILE, index=False)
df.to_csv(INPUT_FILE, index=False)

print("\n✅ DONE - Added auto labeled data!")