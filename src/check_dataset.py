
import pandas as pd

# Load original datasets
fake_news = pd.read_csv("data/Fake.csv")
real_news = pd.read_csv("data/True.csv")

# Display duplicate counts before cleaning
print("Before cleaning:")
print("Fake duplicates:", fake_news.duplicated().sum())
print("Real duplicates:", real_news.duplicated().sum())

# Remove duplicate rows
fake_news = fake_news.drop_duplicates().copy()
real_news = real_news.drop_duplicates().copy()

# Display counts after cleaning
print("\nAfter cleaning:")
print("Fake dataset:", fake_news.shape)
print("Real dataset:", real_news.shape)

# Save cleaned datasets
fake_news.to_csv("data/Fake_cleaned.csv", index=False)
real_news.to_csv("data/True_cleaned.csv", index=False)

print("\nCleaned datasets saved successfully!")
