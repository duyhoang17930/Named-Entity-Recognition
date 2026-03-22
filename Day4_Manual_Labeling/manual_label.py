import pandas as pd
import re
import ast
import os

INPUT_FILE = "clean_news.csv"
OUTPUT_FILE = "manual_labeled.csv"
COMPLETE_FILE = "complete.csv"
MAX_SENTENCES = 20

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

def tokenize(text):
    return re.findall(r"[\w']+|[.,!?;:()\"']", text)

 
df = pd.read_csv(INPUT_FILE)
 
if os.path.exists(OUTPUT_FILE):
    labeled_df = pd.read_csv(OUTPUT_FILE)
else:
    labeled_df = pd.DataFrame(columns=["tokens", "labels"])

if os.path.exists(COMPLETE_FILE):
    complete_df = pd.read_csv(COMPLETE_FILE)
else:
    complete_df = pd.DataFrame(columns=["sentence", "tokens", "labels"])

 
i = 0
while i < min(MAX_SENTENCES, len(df)):

    sentence = df.loc[i, "Sentence"]

    print("\n==============================")
    print(f"Sentence: {sentence}")

    tokens = tokenize(sentence)
    print(tokens)
    print("Length:", len(tokens))

    user_input = input("Labels: ").strip()

    if user_input.lower() == "skip":
        i += 1
        continue

    try:
        labels = ast.literal_eval(user_input)
    except:
        print("❌ Format sai → nhập lại")
        continue

    if len(labels) != len(tokens):
        print(len(labels), "❌ Sai số lượng label → nhập lại")
        continue
 
    new_row1 = pd.DataFrame([{
        "tokens": tokens,
        "labels": labels
    }])
    labeled_df = pd.concat([labeled_df, new_row1], ignore_index=True)
    labeled_df.to_csv(OUTPUT_FILE, index=False)
 
    new_row2 = pd.DataFrame([{
        "sentence": sentence,
        "tokens": tokens,
        "labels": labels
    }])
    complete_df = pd.concat([complete_df, new_row2], ignore_index=True)
    complete_df.to_csv(COMPLETE_FILE, index=False)
 
    df = df.drop(index=i).reset_index(drop=True)
    df.to_csv(INPUT_FILE, index=False)

    print("✅ Saved + Removed sentence!")
 

print("\n🎉 DONE!")