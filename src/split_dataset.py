import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
news = pd.read_csv("data/final_news_dataset.csv")

# Input data (article text)
X = news["text"]

# Output labels (fake or real)
y = news["label"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display dataset sizes
print("TRAINING DATA")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTESTING DATA")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nDataset splitting completed successfully!")