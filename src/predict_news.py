
import joblib

MODEL_PATH = "models/fake_news_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)


def predict_news(text):

    if not text or not text.strip():
        return {
            "error": "Please enter a news article."
        }

    # Predict class
    prediction = int(model.predict([text])[0])

    # Prediction probabilities
    probabilities = model.predict_proba([text])[0]
    confidence = float(probabilities[prediction]) * 100

    # Calculate word contributions
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    features = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    word_scores = features.toarray()[0] * coefficients

    # Select words supporting the predicted class
    if prediction == 1:
        selected_indices = word_scores.argsort()[-5:][::-1]
        explanation_type = "Words contributing towards FAKE prediction"
    else:
        selected_indices = word_scores.argsort()[:5]
        explanation_type = "Words contributing towards REAL prediction"

    important_words = [
        {
            "word": feature_names[i],
            "score": round(float(abs(word_scores[i])), 4)
        }
        for i in selected_indices
        if abs(word_scores[i]) > 0
    ]

    return {
        "prediction": "FAKE" if prediction == 1 else "REAL",
        "label": prediction,
        "confidence": round(confidence, 2),
        "explanation_type": explanation_type,
        "explanation": important_words
    }
