import pandas as pd
import spacy
import numpy as np
import pickle
import re

# load model spaCy
nlp = spacy.load("en_core_web_sm")

# đọc dataset
df = pd.read_csv("clean_news.csv")

sentences = df["clean_sentence"].dropna().tolist()

dataset = []

# ===== entity correction dictionary =====
job_words = {
    "judge","president","minister","director",
    "official","spokesman","spokeswoman"
}

country_dict = {
    "US":"GPE",
    "USA":"GPE",
    "China":"GPE",
    "France":"GPE",
    "Germany":"GPE",
    "Russia":"GPE",
    "Vietnam":"GPE",
    "UK":"GPE"
}

months = {
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
}


# ===== entity correction function =====
def correct_entities(tokens, labels):

    for i, token in enumerate(tokens):

        t = token.strip()

        # job title không phải PERSON
        if t.lower() in job_words:
            labels[i] = "O"

        # số không phải PERSON
        if t.isdigit():
            labels[i] = "O"

        # sửa country
        if t in country_dict:
            labels[i] = "B-" + country_dict[t]

        # sửa month → DATE
        if t in months:
            labels[i] = "B-DATE"

        # year pattern
        if re.match(r"\d{4}", t):
            labels[i] = "B-DATE"

    return labels


# ====== NER labeling ======
for sentence in sentences:

    doc = nlp(sentence)

    tokens = [token.text for token in doc]
    labels = ["O"] * len(tokens)

    for ent in doc.ents:
        start = ent.start
        end = ent.end
        label = ent.label_

        labels[start] = "B-" + label
        for i in range(start + 1, end):
            labels[i] = "I-" + label

    # ===== correction =====
    labels = correct_entities(tokens, labels)

    dataset.append((tokens, labels))


# debug sample
print("\nExample tokens + labels:")
for t,l in zip(dataset[0][0], dataset[0][1]):
    print(t,l)


# ====== tạo vocab token ======
token2id = {}
id2token = {}

idx = 1

for tokens, labels in dataset:
    for token in tokens:
        if token not in token2id:
            token2id[token] = idx
            id2token[idx] = token
            idx += 1


# ====== encode label ======
label_set = set()

for tokens, labels in dataset:
    label_set.update(labels)

label2id = {label:i for i,label in enumerate(sorted(label_set))}
id2label = {i:label for label,i in label2id.items()}

print("\nLabel set:", label2id)


# ====== encode dataset ======
encoded_dataset = []

for tokens, labels in dataset:

    token_ids = [token2id[token] for token in tokens]
    label_ids = [label2id[label] for label in labels]

    encoded_dataset.append({
        "tokens": token_ids,
        "labels": label_ids
    })


# ====== padding ======
MAX_LEN = 50

def pad(seq, max_len, pad_value=0):
    return seq[:max_len] + [pad_value]*(max_len-len(seq))


X = []
y = []

for item in encoded_dataset:
    X.append(pad(item["tokens"], MAX_LEN))
    y.append(pad(item["labels"], MAX_LEN))


X = np.array(X)
y = np.array(y)

print("\nDataset shape:", X.shape)


# ====== save dataset ======
with open("ner_dataset.pkl","wb") as f:
    pickle.dump({
        "X":X,
        "y":y,
        "token2id":token2id,
        "label2id":label2id
    },f)

print("\nSaved dataset to ner_dataset.pkl")