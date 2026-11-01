
import pandas as pd

# Load cleaned datasets
fake_news = pd.read_csv("data/Fake_cleaned.csv")
real_news = pd.read_csv("data/True_cleaned.csv")

# Assign labels
fake_news["label"] = 1
real_news["label"] = 0

# Combine both datasets
combined_news = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)

# Shuffle the combined dataset
combined_news = combined_news.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save the combined dataset
combined_news.to_csv(
    "data/combined_news.csv",
    index=False
)

print("Combined dataset created successfully!")
print("\nDataset shape:", combined_news.shape)
print("\nLabel distribution:")
print(combined_news["label"].value_counts())

print("\nFirst five records:")
print(combined_news.head())
