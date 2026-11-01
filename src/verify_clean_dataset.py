import pandas as pd

news = pd.read_csv("data/final_news_dataset.csv")

print("FINAL DATASET VERIFICATION")

print("\nDataset shape:")
print(news.shape)

print("\nMissing values:")
print(news.isnull().sum())

print("\nDuplicate article texts:")
print(news["text"].duplicated().sum())

print("\nEmpty article texts:")
print(news["text"].str.strip().eq("").sum())

print("\nLabel distribution:")
print(news["label"].value_counts())