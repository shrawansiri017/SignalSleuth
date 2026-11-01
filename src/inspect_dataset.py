
import pandas as pd

# Read the fake and real news datasets
fake_news = pd.read_csv("data/Fake.csv")
real_news = pd.read_csv("data/True.csv")

# Display dataset information
print("FAKE NEWS DATASET")
print(fake_news.shape)
print(fake_news.head())

print("\nREAL NEWS DATASET")
print(real_news.shape)
print(real_news.head())
