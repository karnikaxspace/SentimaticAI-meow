"""
Sentiment Analysis Training – Amazon Product Reviews
Fixed NLTK: download 'punkt_tab' and other resources.
"""

import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub

# ========== DOWNLOAD NLTK DATA (FIXED) ==========
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)      # <-- NEW: required for word_tokenize
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ========== 1. Download dataset using kagglehub ==========
print("Downloading Amazon product reviews from Kaggle...")
path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
csv_file = os.path.join(path, "Reviews.csv")
print(f"File downloaded to: {csv_file}")

# ========== 2. Load CSV with proper parsing ==========
print("Loading CSV with correct parsing (this may take a minute)...")
df = pd.read_csv(
    csv_file,
    encoding='latin1',
    quotechar='"',
    escapechar='\\',
    on_bad_lines='skip',      # skip any malformed lines
    low_memory=False
)
print(f"✅ Loaded {len(df)} reviews")

# Optional: sample for faster training (uncomment to use 10k samples)
# df = df.sample(n=10000, random_state=42)

# ========== 3. Map rating to sentiment ==========
def rating_to_sentiment(rating):
    if rating > 3:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

# Check column name – dataset may have 'Score' or 'star_rating'
if 'Score' in df.columns:
    rating_col = 'Score'
elif 'star_rating' in df.columns:
    rating_col = 'star_rating'
else:
    # Fallback: guess first numeric column
    rating_col = df.select_dtypes(include=[np.number]).columns[0]
    print(f"Using '{rating_col}' as rating column")

df["sentiment"] = df[rating_col].apply(rating_to_sentiment)
print(df["sentiment"].value_counts())

# Keep only text and label – find text column
if 'Text' in df.columns:
    text_col = 'Text'
elif 'review_body' in df.columns:
    text_col = 'review_body'
elif 'review' in df.columns:
    text_col = 'review'
else:
    # Fallback: first string column
    text_col = df.select_dtypes(include=['object']).columns[0]
    print(f"Using '{text_col}' as text column")

df = df[[text_col, "sentiment"]].rename(columns={text_col: "text"})

# ========== 4. Preprocessing ==========
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 2]
    return " ".join(tokens)

print("Cleaning text (this may take a few minutes)...")
df["cleaned_text"] = df["text"].apply(clean_text)

# ========== 5. Train/Test Split ==========
X = df["cleaned_text"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 6. TF-IDF Vectorisation ==========
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ========== 7. Train Logistic Regression ==========
print("Training Logistic Regression...")
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# ========== 8. Evaluate ==========
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["Negative","Neutral","Positive"],
            yticklabels=["Negative","Neutral","Positive"], cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("confusion_matrix.png")
plt.show()

# ========== 9. Save Model & Vectorizer ==========
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("\n✅ Model and vectorizer saved to 'models/'")