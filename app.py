import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------------
# Load Dataset & Train Model
# -------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("train.csv")

    # Create binary label
    df['label'] = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].max(axis=1)
    df = df[['comment_text', 'label']]
    df.columns = ['text', 'label']

    df['text'] = df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = train_model()

# -------------------------------
# Prediction Function
# -------------------------------
def predict(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return prediction

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Cyberbullying Detection", layout="centered")

st.title("💬 Cyberbullying Detection System")
st.write("Detect whether a comment is toxic or normal using NLP")

user_input = st.text_area("Enter a comment:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict(user_input)

        if result == 1:
            st.error("🚨 Cyberbullying / Toxic Comment Detected")
        else:
            st.success("✅ Normal Comment")

# Footer
st.markdown("---")
st.caption("Built using NLP & Machine Learning")