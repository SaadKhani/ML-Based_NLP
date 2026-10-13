
import pickle
import streamlit as st

# Load model and vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf_vectorizer = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="NLP Text Predictor",
    page_icon="🧠",
    layout="centered"
)

# Frontend
st.title("🧠 NLP Text Predictor")
st.write("Enter your text below to get a prediction.")

user_text = st.text_area(
    "Enter text",
    placeholder="Type your text here..."
)

if st.button("Predict", type="primary"):
    if user_text.strip():

        # Convert text into TF-IDF features
        text_tfidf = tfidf_vectorizer.transform([user_text])

        # Predict using your trained model
        prediction = model.predict(text_tfidf)

        # Display prediction
        st.success(f"Prediction: {prediction[0]}")

    else:
        st.warning("Please enter some text.")
