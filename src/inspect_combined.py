
import pandas as pd

news = pd.read_csv("data/combined_news.csv")

print("DATASET INFORMATION")
news.info()

print("\nCOLUMN NAMES")
print(news.columns.tolist())

print("\nMISSING VALUES")
print(news.isnull().sum())

print("\nLABEL COUNTS")
print(news["label"].value_counts())
