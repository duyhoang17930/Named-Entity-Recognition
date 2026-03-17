import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

INPUT_FILE = "ket_qua_sentence.csv"
OUTPUT_FILE = "clean_news.csv"
STATS_FILE = "stats.txt"
 
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


class TextPreprocessor:

    def __init__(self):

        noise_patterns = [
            r'summary.*?-',
            r'opens new tab',
            r'reuters inside track.*',
            r'reporting by.*',
            r'editing by.*',
            r'our standards.*',
            r'can be reached at.*'
        ]
 
        self.noise_regex = [re.compile(p) for p in noise_patterns]

    def clean_text(self, text):

        if pd.isna(text):
            return ""

        text = str(text).lower()
 
        for pattern in self.noise_regex:
            text = pattern.sub('', text)
 
        text = re.sub(r'\S+@\S+', '', text)
 
        text = re.sub(r'[^a-z0- 9\s]', ' ', text)
 
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = nltk.word_tokenize(text)

        tokens = [
            lemmatizer.lemmatize(w)
            for w in tokens
            if w not in stop_words and len(w) > 2
        ]

        return " ".join(tokens)


class DataStats:

    @staticmethod
    def raw_stats(df):

        print("\n===== RAW DATA STATS =====")
        print("Total sentences:", len(df))
        print("Unique titles:", df["Title"].nunique())
        print("Missing values:", df.isna().sum().sum())

    @staticmethod
    def clean_stats(df, column):

        lengths = df[column].str.split().apply(len)

        print("\n===== CLEAN DATA STATS =====")
        print("Clean samples:", len(df))
        print("Avg sentence length:", round(lengths.mean(), 2))
        print("Max length:", lengths.max())
        print("Min length:", lengths.min())

        all_words = " ".join(df[column]).split()
        common = Counter(all_words).most_common(10)

        print("\nTop 10 frequent words:")
        for word, count in common:
            print(f"{word}: {count}")

        return lengths.mean(), common

    @staticmethod
    def save_stats(raw_count, clean_count, avg_len, common_words):

        with open(STATS_FILE, "w", encoding="utf-8") as f:
            f.write("===== DATA STATISTICS =====\n\n")

            f.write(f"Raw sentences: {raw_count}\n")
            f.write(f"Clean sentences: {clean_count}\n")
            f.write(f"Removed sentences: {raw_count - clean_count}\n")
            f.write(f"Average length: {round(avg_len,2)} words\n\n")

            f.write("Top 10 frequent words:\n")
            for word, count in common_words:
                f.write(f"{word}: {count}\n")
 
print("Loading data...")
df = pd.read_csv(INPUT_FILE)

raw_count = len(df)

DataStats.raw_stats(df)

processor = TextPreprocessor()

print("\nCleaning text...")
df["clean_sentence"] = df["Sentence"].apply(processor.clean_text)

df = df[df["clean_sentence"].str.split().str.len() >= 6]

df.drop_duplicates(subset=["clean_sentence"], inplace=True)

clean_count = len(df)

print("\nRemoved sentences:", raw_count - clean_count)

avg_len, common_words = DataStats.clean_stats(df, "clean_sentence")

DataStats.save_stats(raw_count, clean_count, avg_len, common_words)

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ DONE PREPROCESS")
print("Saved clean data to:", OUTPUT_FILE)
print("Saved stats to:", STATS_FILE)
