import pandas as pd

# Load the combined dataset
news = pd.read_csv("data/combined_news.csv")

# Check duplicate article text
print("Duplicate article texts:")
print(news["text"].duplicated().sum())

# Check whether identical text has conflicting labels
label_counts = news.groupby("text")["label"].nunique()

conflicts = label_counts[label_counts > 1]

print("\nArticles with conflicting labels:")
print(len(conflicts))

# Check empty or whitespace-only articles
empty_text = news["text"].str.strip().eq("").sum()

print("\nEmpty article texts:")
print(empty_text)