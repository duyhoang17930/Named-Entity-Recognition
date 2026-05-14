import json

with open('complete_ner_pipeline.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Map day boundaries based on debug output
days = [
    (1, 2, 10),     # Day1: cells 2-9 (Step 1 to Step 2)
    (2, 10, 15),    # Day2: cells 10-14 (Step 2 to Step 3)
    (3, 15, 25),    # Day3: cells 15-24 (Step 3 to Step 4)
    (4, 25, 29),    # Day4: cells 25-28 (Step 4 to Step 5)
    (5, 29, 43),    # Day5: cells 29-42 (Step 5 to Step 6)
    (6, 43, 50),    # Day6: cells 43-49 (Step 6 to Step 7)
    (7, 50, 57),    # Day7: cells 50-56 (Step 7 to Summary)
]

new_nb = {
    "cells": nb['cells'][0:1] + nb['cells'][2:9] + nb['cells'][9:10],  # Title + Day1 content + separator
    "nbformat": nb.get('nbformat', 4),
    "nbformat_minor": nb.get('nbformat_minor', 2),
    "metadata": nb.get('metadata', {})
}
with open('D:/Python/NLP/CK/Day1_Crawl.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

new_nb = {
    "cells": nb['cells'][0:1] + nb['cells'][10:14] + nb['cells'][14:15],
    "nbformat": nb.get('nbformat', 4),
    "nbformat_minor": nb.get('nbformat_minor', 2),
    "metadata": nb.get('metadata', {})
}
with open('D:/Python/NLP/CK/Day2_Preprocess.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

new_nb = {
    "cells": nb['cells'][0:1] + nb['cells'][15:24] + nb['cells'][24:25],
    "nbformat": nb.get('nbformat', 4),
    "nbformat_minor": nb.get('nbformat_minor', 2),
    "metadata": nb.get('metadata', {})
}
with open('D:/Python/NLP/CK/Day3_Encode.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

new_nb = {
    "cells": nb['cells'][0:1] + nb['cells'][25:28] + nb['cells'][28:29],
    "nbformat": nb.get('nbformat', 4),
    "nbformat_minor": nb.get('nbformat_minor', 2),
    "metadata": nb.get('metadata', {})
}
with open('D:/Python/NLP/CK/Day4_AutoLabel.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

new_nb = {
    "cells": nb['cells'][0:1] + nb['cells'][29:42] + nb['cells'][42:43],
    "nbformat": nb.get('nbformat', 4),
    "nbformat_minor": nb.get('nbformat_minor', 2),
    "metadata": nb.get('metadata', {})
}
with open('D:/Python/NLP/CK/Day5_Train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

new_nb = {
    "cells": nb['cells'][0:1] + nb['cells'][43:49] + nb['cells'][49:50],
    "nbformat": nb.get('nbformat', 4),
    "nbformat_minor": nb.get('nbformat_minor', 2),
    "metadata": nb.get('metadata', {})
}
with open('D:/Python/NLP/CK/Day6_Results.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

new_nb = {
    "cells": nb['cells'][0:1] + nb['cells'][50:56] + nb['cells'][56:57],
    "nbformat": nb.get('nbformat', 4),
    "nbformat_minor": nb.get('nbformat_minor', 2),
    "metadata": nb.get('metadata', {})
}
with open('D:/Python/NLP/CK/Day7_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

print("Saved 7 day notebooks to D:/Python/NLP/CK/")
print("- Day1_Crawl.ipynb")
print("- Day2_Preprocess.ipynb")
print("- Day3_Encode.ipynb")
print("- Day4_AutoLabel.ipynb")
print("- Day5_Train.ipynb")
print("- Day6_Results.ipynb")
print("- Day7_Prediction.ipynb")