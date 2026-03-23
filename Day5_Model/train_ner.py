"""
NER Training Script
Trains a Named Entity Recognition model using the labeled dataset.
"""

import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load and parse the data
print("Loading data...")
df = pd.read_csv('manual_labeled.csv')

# Parse the string representations of lists
def parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df['tokens'] = df['tokens'].apply(parse_list)
df['labels'] = df['labels'].apply(parse_list)

# Filter out rows with mismatched lengths
df = df[df['tokens'].apply(len) == df['labels'].apply(len)].reset_index(drop=True)
print(f"Loaded {len(df)} sentences")

# Get all unique labels
all_labels = set()
for labels in df['labels']:
    all_labels.update(labels)

label_list = sorted(list(all_labels))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

print(f"Labels: {label_list}")
print(f"Number of labels: {len(label_list)}")

# Save label mappings
import json
with open('label_mappings.json', 'w') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)

# Install required packages if needed
import subprocess
import sys

def install_packages():
    packages = ['transformers', 'datasets', 'seqeval', 'accelerate']
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

install_packages()

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report
import torch

# Convert data to HuggingFace format
def prepare_data(examples, tokenizer, max_length=128):
    # Tokenize the text
    tokenized_inputs = tokenizer(
        examples['tokens'],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )

    # Align labels with tokens
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(label[word_idx], 0))
            else:
                # For subword tokens, use the same label or -100
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Prepare datasets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

# Convert to datasets
train_dataset = Dataset.from_pandas(train_df[['tokens', 'labels']].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[['tokens', 'labels']].reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df[['tokens', 'labels']].reset_index(drop=True))

# Load tokenizer and model
model_name = "distilbert-base-uncased"
print(f"\nLoading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Tokenize datasets
print("Tokenizing datasets...")
train_tokenized = train_dataset.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=train_dataset.column_names
)
val_tokenized = val_dataset.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=val_dataset.column_names
)
test_tokenized = test_dataset.map(
    lambda x: prepare_data(x, tokenizer),
    batched=True,
    remove_columns=test_dataset.column_names
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove -100 labels
    true_predictions = []
    true_labels = []
    for pred, label in zip(predictions, labels):
        temp_pred = []
        temp_label = []
        for p, l in zip(pred, label):
            if l != -100:
                temp_pred.append(id2label[p])
                temp_label.append(id2label[l])
        if temp_pred:
            true_predictions.append(temp_pred)
            true_labels.append(temp_label)

    # Use seqeval for evaluation
    from seqeval.metrics import classification_report
    report = classification_report(true_labels, true_predictions, digits=4)
    print("\n" + report)
    return {}

# Training arguments
training_args = TrainingArguments(
    output_dir="./ner_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    seed=42,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
print("\nStarting training...")
trainer.train()

# Evaluate on test set
print("\nEvaluating on test set...")
results = trainer.evaluate(test_tokenized)
print(f"Test results: {results}")

# Save the model
model_path = "./ner_model_final"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print(f"\nModel saved to {model_path}")

# Run prediction example
print("\n" + "="*50)
print("Example prediction:")
print("="*50)

def predict_ner(text, model, tokenizer, id2label):
    # Tokenize
    tokens = text.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)

    # Get predictions
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

    # Map predictions back to words
    word_ids = inputs.word_ids()
    predictions_per_word = []
    previous_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        elif word_idx != previous_word_idx:
            predictions_per_word.append(id2label[predictions[idx]])
        previous_word_idx = word_idx

    # Print results
    for token, pred in zip(tokens, predictions_per_word):
        print(f"{token:15} -> {pred}")

# Test with a sample sentence
sample_text = "Bill Clinton met with Donald Trump in Washington D.C. on January 15, 2024."
predict_ner(sample_text, model, tokenizer, id2label)

print("\nTraining complete!")
