import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load cleaned dataset
news = pd.read_csv("data/final_news_dataset.csv")

# Input and output
X = news["text"]
y = news["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),

    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train model
print("Training Fake News Detection Model...")

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-------------------------")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Real News", "Fake News"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fake_news_model.pkl")

print("\nModel saved successfully!")
print("Location: models/fake_news_model.pkl")