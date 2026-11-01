# 🔍 SignalSleuth
### Explainable AI-Based Fake News and Deepfake Detection System for Social Media

SignalSleuth is an AI-powered web application designed to identify potentially misleading textual content and manipulated visual media. It combines machine learning and pretrained deep learning models to analyze news articles, images, and videos.

The platform provides prediction labels, confidence scores, and interpretable information to help users understand the model's results.

---

## 🚀 Key Features

### 1. Fake News Detection
- Analyzes news articles and textual content.
- Uses TF-IDF for text feature extraction.
- Uses Logistic Regression for classification.
- Classifies submitted text as REAL or FAKE.
- Displays prediction confidence.
- Highlights influential words contributing to the prediction.

### 2. Image Deepfake Detection
- Accepts image uploads.
- Uses a pretrained deepfake detection model.
- Identifies potential real or fake visual content.
- Displays the prediction and confidence score.

### 3. Video Deepfake Detection
- Accepts video uploads.
- Samples 8 frames for model analysis.
- Uses a pretrained deepfake detection model.
- Displays the prediction and confidence score.

### 4. Explainable Results
- Provides confidence scores.
- Shows influential textual features.
- Helps users understand the model's predictions.

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Machine Learning | Scikit-learn |
| Text Processing | TF-IDF |
| Classification | Logistic Regression |
| Deep Learning | PyTorch |
| Pretrained Models | Hugging Face Transformers |
| Image Processing | Pillow |
| Video Processing | OpenCV |

---

## 🧠 Models and Dataset

### Fake News Detection
- Dataset: ISOT Fake and Real News Dataset.
- Original files: Fake.csv and True.csv.
- Final cleaned dataset: 38,644 records.
- Real news records: 21,191.
- Fake news records: 17,453.
- Training/testing split: 80% / 20%.
- Text vectorization: TF-IDF.
- Classification algorithm: Logistic Regression.

**Held-out test accuracy: 98.94%**

This result applies to the held-out split of the ISOT dataset and should not be interpreted as guaranteed real-world accuracy.

### Image and Video Deepfake Detection

Pretrained model:

`KoreaPeter/ms-eff-gcvit-deepfake-b0-celeb-df-v2`

- Based on a pretrained deepfake detection model.
- The original model was trained using Celeb-DF-v2.
- Integrated using Hugging Face Transformers.
- Video analysis uses 8 sampled frames.

The pretrained model was integrated into SignalSleuth; it was not trained from scratch by our team.

---

## 🏗️ System Architecture

```text
             USER
               |
               v
       WEB APPLICATION
      HTML / CSS / JS
               |
               v
        FLASK BACKEND
               |
       +-------+-------+
       |       |       |
       v       v       v
      TEXT   IMAGE   VIDEO
       |       |       |
       v       v       v
     TF-IDF  PRETRAINED DEEPFAKE MODEL
       |
       v
   LOGISTIC REGRESSION
       |
       +-------+-------+
               |
               v
       PREDICTION RESULTS
               |
               v
     LABEL + CONFIDENCE
       + EXPLANATION
```

---

## 📂 Project Structure

```text
SignalSleuth/
│
├── data/                   # Dataset files (not included)
│
├── models/
│   └── fake_news_model.pkl
│
├── src/
│   ├── deepfake_detector.py
│   ├── predict_news.py
│   ├── train_text_model.py
│   ├── clean_text_dataset.py
│   ├── combine_dataset.py
│   └── other utility scripts
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── uploads/                # Temporary uploaded files
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.13
- Git
- VS Code (recommended)
- Internet connection for downloading the pretrained deepfake model on first use.

### Step 1: Clone the Repository

```bash
git clone https://github.com/shrawansiri017/SignalSleuth.git
```

### Step 2: Navigate to the Project

```bash
cd SignalSleuth
```

### Step 3: Create a Virtual Environment

```bash
py -3.13 -m venv .venv
```

### Step 4: Activate the Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python app.py
```

### Step 7: Open in Browser

Visit:

```text
http://127.0.0.1:5000/
```

---

## 📊 Results

The implemented system has been tested across all three detection modules.

| Module | Demonstration Result |
|---|---|
| Fake News Detection | REAL prediction |
| Image Deepfake Detection | REAL — 98.31% |
| Video Deepfake Detection | REAL — 85.15% |

The text classification model achieved 98.94% accuracy on its held-out ISOT test split.

These individual image and video outputs are demonstration predictions, not measured accuracy figures.

---

## ⚠️ Limitations

- The text classifier learns patterns from the ISOT dataset and does not independently verify facts.
- Dataset-specific patterns may affect real-world performance.
- The pretrained deepfake model may perform differently on media from datasets or domains it has not encountered.
- Video analysis uses only 8 sampled frames.
- Confidence scores represent model outputs and do not guarantee authenticity.
- The system should be treated as an AI-assisted detection tool, not a definitive fact-checking service.

---

## 🔮 Future Scope

- Improve performance using more diverse datasets.
- Integrate audio deepfake detection.
- Add trusted-source and fact-verification mechanisms.
- Improve video analysis with more advanced temporal methods.
- Enhance explainability using additional techniques.
- Deploy the platform for wider accessibility.

---

## 👥 Project Team

**Mini Project — Batch D18**

- S. Siri
- Chintala Pavani
- Mv Hari Kashyap

**Mentor:** M. Praveen Kumar, Assistant Professor.

---

## 📌 Acknowledgements

- ISOT Fake and Real News Dataset.
- Hugging Face and the pretrained deepfake model developers.
- Open-source Python and machine learning communities.

---

**SignalSleuth — Helping users look beyond the digital illusion.** 🔍