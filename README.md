# 🔍 SignalSleuth

### Explainable AI-Based Fake News and Deepfake Detection System for Social Media

**SignalSleuth** is an AI-powered web application that helps users analyze potentially misleading news articles and manipulated visual media. It combines machine learning, pretrained deep learning models, and explainable AI techniques to analyze text, images, and videos.

The platform provides prediction labels, confidence scores, and interpretable information to help users understand the model's output.

🌐 **Live Website:** https://signalsleuth.onrender.com

💻 **GitHub Repository:** https://github.com/shrawansiri017/SignalSleuth

---

## 🚀 Key Features

### 1. Fake News Detection
- Analyze news articles by entering textual content.
- Uses TF-IDF for text feature extraction.
- Uses Logistic Regression for classification.
- Classifies submitted content as REAL or FAKE.
- Displays prediction confidence.
- Shows influential words contributing to the model prediction.
- Identifies selected language patterns and phrases for additional context.

### 2. Image Deepfake Detection
- Upload images for AI-based analysis.
- Uses a pretrained deepfake detection model.
- Displays REAL or FAKE predictions and confidence scores.
- Supports common image formats.

### 3. Video Deepfake Detection
- Upload videos for AI-based analysis.
- Samples video frames for model analysis.
- Uses a pretrained deepfake detection model.
- Displays prediction results and confidence scores.

### 4. Explainable AI
- Displays influential words from the text classification model.
- Identifies selected language patterns in submitted articles.
- Shows matched phrases and explanatory observations.
- Helps users better understand the analysis output.

*These observations are not independent fact verification or proof that an article is false.*

### 5. User Authentication and Dashboard
- User registration and login.
- Password hashing for account security.
- Personal dashboard.
- Detection activity history.
- Summary of news, image, and video analyses.
- Guest users can try up to four detections before registration is required.

### 6. Modern Web Interface
- Responsive multi-page website.
- Bright Electric Teal and Vivid Violet colour theme.
- Navigation between detection modules.
- Dedicated About and Help pages.

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
| Image Processing | Pillow, OpenCV |
| Database | SQLite |
| Authentication | Werkzeug Security |
| Deployment | Render |
| Version Control | Git, GitHub |

---

## 🧠 Models and Dataset

### Fake News Detection

- Dataset: ISOT Fake and Real News Dataset.
- Original datasets: Fake.csv and True.csv.
- Final cleaned dataset: 38,644 records.
- Real news records: 21,191.
- Fake news records: 17,453.
- Training/testing split: 80% / 20%.
- Feature extraction: TF-IDF.
- Classification algorithm: Logistic Regression.

**Held-out test accuracy: 98.94%**

This accuracy applies to the held-out ISOT dataset split. It does not guarantee the same performance on all real-world news articles.

### Image and Video Deepfake Detection

Pretrained model:

`KoreaPeter/ms-eff-gcvit-deepfake-b0-celeb-df-v2`

- Integrated using Hugging Face Transformers.
- Uses PyTorch for deep learning inference.
- Supports image and video analysis.
- Video analysis uses sampled frames.

The pretrained deepfake model was integrated into SignalSleuth; it was not trained from scratch by our team.

---

## 🏗️ System Architecture

```text
                  USER
                    |
                    v
              SIGNALSLEUTH
              WEB INTERFACE
                    |
                    v
              FLASK BACKEND
                    |
          +---------+---------+
          |         |         |
          v         v         v
         TEXT      IMAGE     VIDEO
          |         |         |
          v         +----+----+
       TF-IDF             |
          |               v
          v        PRETRAINED DEEPFAKE
      LOGISTIC          MODEL
     REGRESSION           |
          |               |
          +-------+-------+
                  |
                  v
           PREDICTION RESULTS
                  |
                  v
          LABEL + CONFIDENCE
             + EXPLANATION
                  |
                  v
          DASHBOARD / HISTORY
```

---

## 📂 Project Structure

```text
SignalSleuth/
│
├── data/                       # Dataset files (not included)
│
├── models/
│   └── fake_news_model.pkl
│
├── src/
│   ├── deepfake_detector.py
│   ├── predict_news.py
│   ├── train_text_model.py
│   └── dataset utility scripts
│
├── static/
│   ├── css/
│   │   ├── redesign.css
│   │   └── style.css
│   │
│   └── js/
│       ├── detection.js
│       ├── main.js
│       ├── redesign-detection.js
│       └── script.js
│
├── templates/
│   ├── auth/
│   │   ├── login.html
│   │   └── register.html
│   │
│   ├── pages/
│   │   ├── home.html
│   │   ├── fake_news.html
│   │   ├── image_detection.html
│   │   ├── video_detection.html
│   │   ├── dashboard.html
│   │   ├── about.html
│   │   └── help.html
│   │
│   └── base.html
│
├── uploads/                    # Temporary uploaded media
├── app.py
├── database.py
├── requirements.txt
├── .gitignore
└── README.md
```

The SQLite database file is generated locally when the application initializes. It is not included in the GitHub repository.

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.13
- Git
- Visual Studio Code (recommended)
- Internet connection for downloading dependencies and the pretrained deepfake model.

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

## 🌐 Live Deployment

SignalSleuth is deployed using Render.

**Live Website:** https://signalsleuth.onrender.com

The GitHub repository is connected to the deployment workflow so that new code changes can be deployed from the repository.

Note: Free hosting services may experience cold starts. Local SQLite storage on an ephemeral hosting filesystem should not be treated as permanent storage; persistent storage or a managed database is recommended for production use.

---

## 📊 Model Performance and Results

The text classification model achieved **98.94% accuracy** on its held-out ISOT dataset test split.

The image and video modules have also been tested with sample media and return model-based predictions and confidence scores.

These results demonstrate the functionality of the prototype, not guaranteed real-world detection accuracy.

---

## ⚠️ Limitations

- The text classifier learns patterns from its training dataset and does not independently verify facts.
- Dataset-specific patterns may affect real-world performance.
- The pretrained deepfake model may perform differently on media outside its training domain.
- Video analysis examines sampled frames rather than every frame.
- Confidence scores represent model outputs and do not guarantee authenticity.
- Language-pattern observations are heuristic indicators, not proof of misinformation.
- The source-credibility indicators displayed in the interface are not a live external fact-checking service.
- The application should be treated as an AI-assisted analysis tool, not a definitive fact-checking system.

---

## 🔮 Future Scope

- Improve detection using larger and more diverse datasets.
- Integrate audio deepfake detection.
- Add trusted-source and fact-verification mechanisms.
- Improve video analysis using advanced temporal methods.
- Enhance explainability using SHAP or LIME.
- Improve database persistence and deployment scalability.
- Extend the platform with additional media analysis capabilities.

---

## 📌 Acknowledgements

- ISOT Fake and Real News Dataset.
- Hugging Face and the pretrained deepfake model developers.
- Open-source Python and machine learning communities.
- Flask, Scikit-learn, PyTorch and OpenCV communities.

---

**SignalSleuth — Helping users look beyond the digital illusion.** 🔍