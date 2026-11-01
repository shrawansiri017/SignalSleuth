import pandas as pd

# Load combined dataset
news = pd.read_csv("data/combined_news.csv")

print("BEFORE CLEANING")
print("Total records:", len(news))

# Remove empty and whitespace-only article texts
news = news[news["text"].str.strip().ne("")].copy()

print("\nAfter removing empty texts:", len(news))

# Remove duplicate article texts
news = news.drop_duplicates(subset=["text"]).copy()

print("After removing duplicate texts:", len(news))

# Save cleaned dataset
news.to_csv("data/final_news_dataset.csv", index=False)

print("\nFINAL DATASET")
print("Dataset shape:", news.shape)

print("\nLabel distribution:")
print(news["label"].value_counts())

print("\nFinal dataset saved successfully!")