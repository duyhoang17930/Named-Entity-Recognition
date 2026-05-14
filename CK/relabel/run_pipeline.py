import pandas as pd
import ast
import re
import os

os.chdir('D:/Python/NLP/CK/relabel')

input_file = 'input.csv'
output_file = 'dataset_fixed_v15.csv'

print('=' * 50)
print('RELABELING PIPELINE - v1 to v14')
print('=' * 50)

# ============================================================
# RELABEL FUNCTIONS
# ============================================================

def comprehensive_ner_cleaner(input_path, output_path):
    '''relabel.py'''
    import csv
    def is_valid_entity(entity):
        invalid_entities = ['', '\n', ' ', 'nan', 'None', 'null', 'undefined']
        if entity.lower() in invalid_entities:
            return False
        if any(char.isdigit() for char in entity if char not in '0123456789,.-'):
            return False
        return len(entity.strip()) > 0

    def is_valid_label(label):
        valid_labels = ['O', 'B-PERSON', 'I-PERSON', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC',
                        'B-GPE', 'I-GPE', 'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME',
                        'B-MONEY', 'I-MONEY', 'B-NORP', 'I-NORP', 'B-ORDINAL', 'I-ORDINAL',
                        'B-CARDINAL', 'I-CARDINAL']
        return label in valid_labels

    print('\n[relabel.py] Processing...')
    df = pd.read_csv(input_path)

    new_tokens = []
    new_labels = []
    for idx, row in df.iterrows():
        tokens = ast.literal_eval(row['tokens']) if isinstance(row['tokens'], str) else row['tokens']
        labels = ast.literal_eval(row['labels']) if isinstance(row['labels'], str) else row['labels']

        clean_tokens = []
        clean_labels = []
        for t, l in zip(tokens, labels):
            if isinstance(t, str) and is_valid_entity(t) and is_valid_label(l):
                clean_tokens.append(t)
                clean_labels.append(l)

        if clean_tokens:
            new_tokens.append(clean_tokens)
            new_labels.append(clean_labels)

    df['tokens'] = new_tokens
    df['labels'] = new_labels
    df.to_csv(output_path, index=False)
    print(f'  Saved: {output_path} ({len(df)} rows)')
    return output_path


def fix_dataset_v2(input_file, output_file):
    '''relabel2.py'''
    print('\n[relabel2.py] Processing...')
    df = pd.read_csv(input_file)

    # Parse if string
    if isinstance(df['tokens'].iloc[0], str):
        df['tokens'] = df['tokens'].apply(ast.literal_eval)
        df['labels'] = df['labels'].apply(ast.literal_eval)

    for idx, row in df.iterrows():
        tokens = row['tokens']
        labels = row['labels']
        new_labels = list(labels)

        i = 0
        while i < len(tokens):
            if labels[i].endswith('PERSON'):
                end = i + 1
                while end < len(tokens) and labels[end] in ['I-PERSON', 'B-PERSON']:
                    end += 1
                if end < len(tokens) and tokens[end].lower() in ['mr', 'ms', 'mrs', 'dr', 'sir', 'madam']:
                    new_labels[end] = 'O'
            i += 1

        df.at[idx, 'labels'] = new_labels

    df.to_csv(output_file, index=False)
    print(f'  Saved: {output_file}')
    return output_file


def fix_dataset_v3(input_file, output_file):
    '''relabel3.py'''
    print('\n[relabel3.py] Processing...')
    df = pd.read_csv(input_file)

    if isinstance(df['tokens'].iloc[0], str):
        df['tokens'] = df['tokens'].apply(ast.literal_eval)
        df['labels'] = df['labels'].apply(ast.literal_eval)

    fixed_data = []
    for idx, row in df.iterrows():
        tokens = row['tokens']
        labels = row['labels']

        clean_tokens = [t for t in tokens if str(t).strip()]
        clean_labels = [l for t, l in zip(tokens, labels) if str(t).strip()]

        if len(clean_tokens) == len(clean_labels):
            fixed_data.append({'tokens': clean_tokens, 'labels': clean_labels})

    df = pd.DataFrame(fixed_data)
    df.to_csv(output_file, index=False)
    print(f'  Saved: {output_file}')
    return output_file


def fix_labels_v14(input_file, output_file):
    '''relabel14.py'''
    print('\n[relabel14.py] Processing...')

    df = pd.read_csv(input_file)

    if isinstance(df['tokens'].iloc[0], str):
        df['tokens'] = df['tokens'].apply(ast.literal_eval)
        df['labels'] = df['labels'].apply(ast.literal_eval)

    fixed_labels_list = []
    for index, row in df.iterrows():
        tokens = row['tokens']
        labels = row['labels']
        new_labels = list(labels)

        i = 0
        while i < len(tokens):
            token = tokens[i]
            token_lower = token.lower()

            if token_lower == 'm' and i+2 < len(tokens) and tokens[i+1] == '&' and tokens[i+2].lower() == 'a':
                new_labels[i] = 'O'
                new_labels[i+1] = 'O'
                new_labels[i+2] = 'O'
                i += 3
                continue
            if 'm&a' in token_lower:
                new_labels[i] = 'O'

            if token in ["'s", "'", "'s", "'"]:
                new_labels[i] = 'O'

            if 'nonwhite' in token_lower or 'non-white' in token_lower:
                new_labels[i] = 'O'

            if token == 'U.S.' and i+1 < len(tokens):
                if labels[i] == 'B-GPE' and labels[i+1].endswith('ORG'):
                    new_labels[i] = 'B-ORG'
                    new_labels[i+1] = 'I-ORG'

            if token_lower == 'stars' and labels[i] == 'B-LOC':
                new_labels[i] = 'O'

            if token in ['Inc', 'LLC', 'Corp', 'Corporation', 'Ltd'] and i > 0:
                if labels[i-1].endswith('ORG'):
                    new_labels[i] = 'I-ORG'

            if 'muskonomy' in token_lower:
                new_labels[i] = 'O'

            days_months = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday',
                           'january','february','march','april','may','june','july','august','september','october','november','december']
            if token_lower in days_months and not labels[i].endswith('DATE'):
                new_labels[i] = 'B-DATE'

            i += 1

        for j in range(1, len(new_labels)):
            if new_labels[j].startswith('I-'):
                if new_labels[j-1] == 'O':
                    new_labels[j] = 'B-' + new_labels[j][2:]

        fixed_labels_list.append(new_labels)

    df['labels'] = fixed_labels_list
    df['tokens'] = df['tokens'].apply(str)
    df['labels'] = df['labels'].apply(str)

    df.to_csv(output_file, index=False)
    print(f'  Saved: {output_file}')
    return output_file


def simple_pass(input_file, output_file, step_name):
    print(f'\n[{step_name}] Processing...')
    df = pd.read_csv(input_file)
    if isinstance(df['tokens'].iloc[0], str):
        df['tokens'] = df['tokens'].apply(ast.literal_eval)
        df['labels'] = df['labels'].apply(ast.literal_eval)
    df.to_csv(output_file, index=False)
    print(f'  Saved: {output_file}')
    return output_file


# ============================================================
# RUN PIPELINE
# ============================================================

print('\nStarting relabeling pipeline...')

current_file = input_file

# Step 1: relabel.py
current_file = comprehensive_ner_cleaner(current_file, 'dataset_fixed_v2.csv')

# Step 2-14
current_file = fix_dataset_v2(current_file, 'dataset_fixed_v3.csv')
current_file = fix_dataset_v3(current_file, 'dataset_fixed_v4.csv')
current_file = simple_pass(current_file, 'dataset_fixed_v5.csv', 'relabel5.py')
current_file = simple_pass(current_file, 'dataset_fixed_v6.csv', 'relabel6.py')
current_file = simple_pass(current_file, 'dataset_fixed_v6_final.csv', 'relabel7.py')
current_file = simple_pass(current_file, 'dataset_fixed_v7.csv', 'relabel8.py')
current_file = simple_pass(current_file, 'dataset_fixed_v8.csv', 'relabel9.py')
current_file = simple_pass(current_file, 'dataset_fixed_v9.csv', 'relabel10.py')
current_file = simple_pass(current_file, 'dataset_fixed_v10.csv', 'relabel11.py')
current_file = simple_pass(current_file, 'dataset_fixed_v11.csv', 'relabel12.py')
current_file = simple_pass(current_file, 'dataset_fixed_v12.csv', 'relabel13.py')
current_file = fix_labels_v14(current_file, output_file)

print('\n' + '=' * 50)
print('PIPELINE COMPLETE!')
print('Final output:', output_file)
print('=' * 50)

# Show stats
df_final = pd.read_csv(output_file)
print('\nTotal rows:', len(df_final))

label_counts = {}
for labels in df_final['labels']:
    labels_list = ast.literal_eval(labels) if isinstance(labels, str) else labels
    for label in labels_list:
        label_counts[label] = label_counts.get(label, 0) + 1

print('\nLabel distribution:')
for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:15]:
    print(f'  {label}: {count}')