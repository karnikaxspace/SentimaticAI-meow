# Premium SentimentScope AI - Streamlit App
# Theme: Dark Burgundy & Blue Accents

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import traceback

st.set_page_config(page_title="SentimentMeow😼 AI", page_icon="✨", layout="wide")

# Session state
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# ---------- CHECK MODEL FILES ----------
st.sidebar.markdown("### 🔍 Debug Info")
model_path = "models/model.pkl"
vec_path = "models/vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vec_path):
    st.sidebar.success("✅ Model files found")
else:
    st.sidebar.error("❌ Model files missing! Run train_sentiment.py")

# ---------- LOAD NLTK ----------
@st.cache_resource
def load_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

load_nltk()
model, vectorizer = load_model()

# Stop if model not loaded
if model is None or vectorizer is None:
    st.error("Failed to load model. Check the debug info in the sidebar.")
    st.stop()

# ---------- PREPROCESSING ----------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    confidence = {cls: round(prob, 3) for cls, prob in zip(model.classes_, probs)}
    return pred, confidence

# ---------- UI STYLES (dark burgundy) ----------
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none; }
[data-testid="stSidebarCollapsedControl"] { display: none; }
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #f0f0f0 !important; }
.stApp { background: linear-gradient(135deg, #1a0f12 0%, #2a1a20 100%); }
.main-card { max-width: 950px; margin: auto; padding: 2rem; }
.hero-card {
    background: rgba(30, 15, 20, 0.85);
    backdrop-filter: blur(14px);
    border-radius: 32px;
    padding: 2.5rem;
    box-shadow: 0 12px 35px rgba(0,0,0,0.3);
    border: 1px solid rgba(139, 44, 61, 0.5);
}
.info-box {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 20px;
    background: rgba(139, 44, 61, 0.2);
    border: 1px solid rgba(139, 44, 61, 0.4);
    color: #f5e6ea;
}
.stTextArea textarea {
    border-radius: 24px !important;
    border: 2px solid #8b2c3d !important;
    background-color: #2a1a20 !important;
    color: #f0f0f0 !important;
    padding: 1rem !important;
}
.stButton button {
    border-radius: 30px;
    background: linear-gradient(90deg, #4a6fa5, #8b2c3d);
    color: white;
}
.result-card { padding: 2rem; border-radius: 28px; margin-top: 1rem; animation: fadeIn 0.5s; }
.positive-card { background: rgba(44,62,102,0.85); border: 2px solid #4a6fa5; }
.neutral-card { background: rgba(114,128,155,0.85); border: 2px solid #9aa9c1; }
.negative-card { background: rgba(139,44,61,0.85); border: 2px solid #c95a6f; }
.conf-card { background: rgba(30,15,20,0.9); padding: 1rem; border-radius: 18px; margin-bottom: 1rem; border-left: 4px solid; }
.footer { text-align: center; color: #b39ba5; padding: 2rem; font-size: 0.9rem; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ---------- UI LAYOUT ----------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("""
<div class="hero-card">
<div style="text-align:center;">
<h1 style="font-size:3rem; background: linear-gradient(90deg, #b8c7e7, #e3a5b5); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
🛍️ SentimentMeow 😼 
</h1>
<p style="font-size:1.2rem; color:#f0e6e9;">Instant sentiment insights</p>
</div>
<div class="info-box">🧠 Paste an Amazon review below – AI predicts Positive, Neutral, or Negative.</div>
</div>
""", unsafe_allow_html=True)

# ---------- EXAMPLE BUTTONS (placed above text_area to update session state) ----------
st.markdown("#### 💡 Try Examples")
col1, col2, col3 = st.columns(3)
if col1.button("😊 Loved it", use_container_width=True):
    st.session_state.review_text = "Fantastic quality and fast shipping. Highly recommended!"
if col2.button("😐 It's okay", use_container_width=True):
    st.session_state.review_text = "The product works as expected but nothing special."
if col3.button("😞 Disappointed", use_container_width=True):
    st.session_state.review_text = "Terrible quality. Broke within two days."

# ---------- TEXT INPUT ----------
st.subheader("✍️ Enter Review")
review = st.text_area(
    label="Review text",
    value=st.session_state.review_text,
    key="review_text",
    placeholder="Example: This product exceeded my expectations. Excellent quality!",
    height=160,
    label_visibility="collapsed"
)

if review:
    st.caption(f"Characters: {len(review)} | Words: {len(review.split())}")

# ---------- ANALYZE BUTTON ----------
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    analyze = st.button("✨ Analyze Review", use_container_width=True)

if analyze:
    if review.strip():
        try:
            sentiment, conf = predict_sentiment(review)
            sent_lower = sentiment.lower()
            emoji = {"positive":"😍","neutral":"😐","negative":"😞"}
            card = {"positive":"positive-card","neutral":"neutral-card","negative":"negative-card"}
            highest = max(conf.values())
            if highest >= 0.90:
                interp = "🎯 High confidence"
            elif highest >= 0.70:
                interp = "📌 Moderate confidence"
            else:
                interp = "⚠️ Mixed sentiment"
            st.markdown(f"""
            <div class="result-card {card[sent_lower]}">
            <h2>{emoji[sent_lower]} {sentiment.title()}</h2>
            <p><b>Your Review:</b></p>
            <i>"{review}"</i><br><br>
            <b>{interp}</b>
            </div>
            """, unsafe_allow_html=True)
            st.subheader("Confidence Breakdown")
            colors = {"positive":"#4a6fa5","neutral":"#b0bedb","negative":"#c95a6f"}
            icons = {"positive":"😍","neutral":"😐","negative":"😞"}
            for cls, prob in conf.items():
                cl = cls.lower()
                st.markdown(f'<div class="conf-card" style="border-left-color:{colors[cl]}"><b>{icons[cl]} {cls.title()}</b> — {prob*100:.1f}%</div>', unsafe_allow_html=True)
                st.progress(prob)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.code(traceback.format_exc())
    else:
        st.warning("Please enter a review first.")

st.markdown('<div class="footer">Trained on 500k+ Amazon Reviews • Logistic Regression • TF‑IDF • Streamlit</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)