"""
NER Inference Script
Load and use a trained NER model for predictions.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

# Load label mappings
with open('label_mappings.json', 'r') as f:
    mappings = json.load(f)
    id2label = mappings['id2label']
    label2id = mappings['label2id']

# Load model
model_path = "./ner_model_final"
print(f"Loading model from {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

def predict_ner(text, show_details=True):
    """
    Predict NER tags for input text.

    Args:
        text: Input sentence (string)
        show_details: If True, prints detailed results

    Returns:
        List of (word, entity) tuples
    """
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
            predictions_per_word.append(id2label[str(predictions[idx])])
        previous_word_idx = word_idx

    results = list(zip(tokens, predictions_per_word))

    if show_details:
        print(f"\nInput: {text}")
        print("-" * 40)
        print(f"{'TOKEN':<20} {'ENTITY':<15}")
        print("-" * 40)
        for token, pred in results:
            print(f"{token:<20} {pred:<15}")

        # Group entities
        print("\n" + "-" * 40)
        print("Grouped Entities:")
        print("-" * 40)
        current_entity = None
        entity_text = []
        for token, pred in results:
            if pred.startswith('B-'):
                if entity_text:
                    print(f"  {current_entity}: {' '.join(entity_text)}")
                current_entity = pred[2:]
                entity_text = [token]
            elif pred.startswith('I-') and current_entity == pred[2:]:
                entity_text.append(token)
            else:
                if entity_text:
                    print(f"  {current_entity}: {' '.join(entity_text)}")
                    entity_text = []
                current_entity = None
                if not pred.startswith('O'):
                    current_entity = pred[2:]
                    entity_text = [token]
        if entity_text:
            print(f"  {current_entity}: {' '.join(entity_text)}")

    return results


def batch_predict(texts, show_details=False):
    """Predict NER for multiple sentences."""
    results = []
    for text in texts:
        tokens = text.split()
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
            predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

        word_ids = inputs.word_ids()
        predictions_per_word = []
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                predictions_per_word.append(id2label[str(predictions[idx])])
            previous_word_idx = word_idx

        results.append(list(zip(tokens, predictions_per_word)))
    return results


# Example usage
if __name__ == "__main__":
    # Test sentences
    test_sentences = [
        "Bill Clinton met with Donald Trump in Washington D.C. on January 15, 2024.",
        "The United States and China signed a trade agreement in New York.",
        "Apple Inc. CEO Tim Cook announced new products at the event.",
    ]

    for sentence in test_sentences:
        predict_ner(sentence)
        print()
