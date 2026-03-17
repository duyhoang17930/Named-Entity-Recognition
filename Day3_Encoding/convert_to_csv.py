import pickle
import pandas as pd

with open("ner_dataset.pkl","rb") as f:
    data = pickle.load(f)

df = pd.DataFrame({
    "tokens": data["X"].tolist(),
    "labels": data["y"].tolist()
})

df.to_csv("ner_dataset_view.csv", index=False)