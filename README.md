# 🛍️ SentimentAI meow – Amazon Review Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Instant sentiment insights powered by Machine Learning**  
> Classify any text as Positive, Neutral, or Negative – trained on 500k+ real Amazon reviews.



---

## ✨ Features

- 🧠 **Machine Learning backend** – Logistic Regression + TF‑IDF  
- 🎨 **Beautiful dark UI** – Blue & Burgundy glassmorphic design  
- ⚡ **Real‑time predictions** – Instant sentiment + confidence scores  
- 📊 **Confidence breakdown** – Probability bars for all three classes  
- 🖱️ **Example buttons** – One‑click testing  
- 🚀 **Easy to run** – Streamlit app, no frontend knowledge required  

---

## 📊 Dataset

- **Source:** [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) (Kaggle)  
- **Size:** 568,423 real customer reviews  
- **Label mapping:**  
  - Rating 4–5 → **Positive**  
  - Rating 3 → **Neutral**  
  - Rating 1–2 → **Negative**  

| Sentiment | Count   |
|-----------|---------|
| Positive  | 443,756 |
| Negative  | 82,030  |
| Neutral   | 42,637  |

---

## 🧠 Model & Performance

- **Algorithm:** Logistic Regression (classical ML, fast & interpretable)  
- **Features:** TF‑IDF (unigrams + bigrams, max 10,000 features)  
- **Accuracy:** ~80% on test set  
- **Classification report:**

precision recall f1-score support
Positive 0.84 0.92 0.88 88752
Neutral 0.55 0.38 0.45 8527
Negative 0.70 0.56 0.62 16406

Accuracy: 0.80
Macro avg: 0.70 0.62 0.65


- **Confusion Matrix:** Generated during training as `<img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/0e6c8ab4-3980-4e6c-a871-63af979fc549" />
`

---

## 🛠️ Tech Stack

| Category          | Tools |
|-------------------|-------|
| **Language**      | Python 3.11 |
| **Data handling** | pandas, numpy, kagglehub |
| **Text processing**| NLTK (tokenization, stopwords, lemmatization) |
| **ML & features** | scikit-learn (TF-IDF, Logistic Regression) |
| **Visualization** | matplotlib, seaborn |
| **Web app**       | Streamlit + custom CSS |
| **Model storage** | joblib |

---

## 📁 Project Structure
sentiment-scope-ai/
├── train_sentiment.py # Training script (downloads data, preprocesses, trains, saves model)
├── app.py # Streamlit UI (dark theme, prediction logic)
├── models/ # Created after training
│ ├── model.pkl # Trained Logistic Regression model
│ └── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Python dependencies
├── README.md # This file
└── confusion_matrix.png # Generated during training

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/karnikaxspace/SentimaticAI-meow.git
cd SentimaticAI-meow

```
---
### 📈 Future Improvements
Batch CSV upload – analyze hundreds of reviews at once

BERT fine‑tuning – higher accuracy (code ready, commented out)

Deploy online – Streamlit Cloud, Hugging Face Spaces, or Render

Explainability – SHAP/LIME to show which words influenced prediction

Dark/Light theme toggle – user preference

---
### Acknowledgments
Dataset: Amazon Product Reviews by Arham Rumi (Kaggle)

Built as an internship project – extending NLP skills from chatbot to text classification

Icons: Unicode emojis

---
### 👤 Author
  Karnika Kumari
