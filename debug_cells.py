import json

with open('complete_ner_pipeline.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open('D:/Python/NLP/CK/debug.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        ct = cell.get('cell_type', 'unknown')
        src = cell.get('source', '') or ''
        if isinstance(src, list):
            src = ''.join(src)
        first_line = src.split('\n')[0][:100]
        out.write(f"{i}: {ct} | {first_line}\n")

print("Debug done")