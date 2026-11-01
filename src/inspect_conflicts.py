import pandas as pd

news = pd.read_csv("data/combined_news.csv")

# Find texts that have multiple labels
label_counts = news.groupby("text")["label"].nunique()
conflicting_texts = label_counts[label_counts > 1].index

print("Number of conflicting texts:", len(conflicting_texts))

for text in conflicting_texts:
    print("\nConflicting text:", repr(text))

    records = news[news["text"] == text]

    print("\nLabel distribution:")
    print(records["label"].value_counts())

    print("\nNumber of records:", len(records))