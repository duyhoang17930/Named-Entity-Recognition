import json
import csv
import ast
import re

def normalize_token(token):
    """Normalize token by removing unicode chars and punctuation"""
    token = re.sub(r'[\u200b\u2060\u200c\u200d\u00a0]', '', token)
    token = token.strip('.,;:!?()[]{}\'\"-')
    return token.lower().strip()

# Load entity mappings from data.json (uses Python dict syntax with single quotes)
def load_entity_mappings(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read()
        data = ast.literal_eval(content)

    # Create single token map
    entity_map = {}
    # Create multi-word phrase map
    multi_word_map = {}

    for entity_type, entities in data.items():
        for entity in entities:
            entity_normalized = entity.lower().strip()
            entity_map[entity_normalized] = entity_type

            # Build multi-word mappings for 2+ word entities
            words = entity_normalized.split()
            if len(words) >= 2:
                if words[0] not in multi_word_map:
                    multi_word_map[words[0]] = []
                multi_word_map[words[0]].append({
                    'phrase': entity_normalized,
                    'type': entity_type,
                    'words': words
                })

    return entity_map, multi_word_map

# Relabel tokens - ONLY add labels where original is O
def relabel_data(csv_path, entity_map, multi_word_map, output_path):
    results = []

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)

        for row in reader:
            tokens = ast.literal_eval(row['tokens'])
            labels = ast.literal_eval(row['labels'])

            new_labels = []
            i = 0
            while i < len(tokens):
                token = tokens[i]
                token_lower = normalize_token(token)
                original_label = labels[i]

                # Check for multi-word phrase matches first
                if token_lower in multi_word_map:
                    matched = False
                    for phrase_info in multi_word_map[token_lower]:
                        phrase_words = phrase_info['words']
                        phrase_len = len(phrase_words)

                        if i + phrase_len <= len(tokens):
                            remaining_tokens = [normalize_token(t) for t in tokens[i:i+phrase_len]]
                            if remaining_tokens == phrase_words:
                                # Always apply multi-word matching - overwrite any existing labels
                                for j in range(phrase_len):
                                    if j == 0:
                                        new_labels.append(f'B-{phrase_info["type"]}')
                                    else:
                                        new_labels.append(f'I-{phrase_info["type"]}')
                                i += phrase_len
                                matched = True
                                break
                    if matched:
                        continue

                # Check single token match - always use data.json type (overwrite existing)
                if token_lower in entity_map:
                    entity_type = entity_map[token_lower]

                    # Check for chain continuation: if previous was B- of same type
                    if i > 0 and new_labels[i-1].endswith(f'-{entity_type}'):
                        new_labels.append(f'I-{entity_type}')
                    else:
                        new_labels.append(f'B-{entity_type}')
                else:
                    # Chain continuation: only if previous was B- AND original was I-
                    if i > 0 and new_labels[i-1].startswith('B-') and original_label.startswith('I-'):
                        prev_type = new_labels[i-1][2:]
                        new_labels.append(f'I-{prev_type}')
                    else:
                        new_labels.append(original_label)

                i += 1

            results.append({
                'tokens': tokens,
                'labels': new_labels
            })

    # Write results to output CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tokens', 'labels'])
        writer.writeheader()

        for result in results:
            writer.writerow({'tokens': str(result['tokens']), 'labels': str(result['labels'])})

    return len(results)

if __name__ == '__main__':
    json_path = 'D:\\Python\\NLP\\Day4.5_Relabeling\\data.json'
    csv_path = 'D:\\Python\\NLP\\Day4.5_Relabeling\\manual_labeled.csv'
    output_path = 'D:\\Python\\NLP\\Day4.5_Relabeling\\relabeled_output.csv'

    print("Loading entity mappings from data.json...")
    entity_map, multi_word_map = load_entity_mappings(json_path)
    print(f"Loaded {len(entity_map)} single token mappings")
    print(f"Loaded {len(multi_word_map)} multi-word phrase starting words")

    print("\nRelabeling data...")
    count = relabel_data(csv_path, entity_map, multi_word_map, output_path)
    print(f"Relabeled {count} rows")
    print(f"\nOutput saved to: {output_path}")