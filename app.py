
import re
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="centered"
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "best_model.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("🔍 Fake Online Product Review Detection")
st.write("Machine Learning-Based Product Review Classification")

st.info(
    "This model identifies computer-generated-like and "
    "original-like review patterns, not verified fraud."
)

review = st.text_area(
    "Enter your product review:",
    placeholder="Type or paste a product review...",
    height=150
)

if st.button("Predict Review", type="primary"):
    cleaned = clean_text(review)

    if not cleaned:
        st.warning("Please enter a valid product review.")

    elif not MODEL_PATH.exists():
        st.error("Model file not found. Run train.py first.")

    else:
        model = load_model()
        prediction = model.predict([cleaned])[0]

        if prediction == "fake":
            st.warning("⚠️ Computer-Generated-Like Review")
        elif prediction == "genuine":
            st.success("✅ Original-Like Review")
        else:
            st.error("Unexpected prediction label.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([cleaned])[0]
            index = list(model.classes_).index(prediction)
            confidence = probabilities[index]

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.caption(
                "Confidence is the model's estimated "
                "class probability, not proof of fraud."
            )

st.divider()
st.caption("BTech CSE - Data Science | ML Semester Project")
