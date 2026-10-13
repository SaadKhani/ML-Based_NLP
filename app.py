
import pickle
import streamlit as st

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the fitted TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf_vectorizer = pickle.load(file)


# Page configuration
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🧠",
    layout="centered"
)

# Emotion labels in the specified order
emotions = [
    "sadness",
    "anger",
    "love",
    "surprise",
    "fear",
    "joy"
]


# Frontend
st.title("🧠 Emotion Detection")
st.write(
    "Enter a sentence below and let the NLP model "
    "predict the emotion expressed in the text."
)

st.divider()

# Text input
user_text = st.text_area(
    "Enter your text",
    placeholder="Example: I am so happy today!",
    height=150
)

# Prediction
if st.button("Predict Emotion", type="primary", use_container_width=True):

    if user_text.strip():

        try:
            # Convert input text into TF-IDF features
            text_tfidf = tfidf_vectorizer.transform([user_text])

            # Get prediction from the trained model
            prediction = model.predict(text_tfidf)[0]

            # Convert numeric prediction into emotion label
            if isinstance(prediction, (int, float)) or str(prediction).isdigit():
                prediction = int(prediction)

                if 0 <= prediction < len(emotions):
                    predicted_emotion = emotions[prediction]
                else:
                    st.error(f"Unknown emotion class: {prediction}")
                    st.stop()
            else:
                # Prediction is already a text label
                predicted_emotion = str(prediction).lower()

            # Display result
            st.subheader("Prediction Result")

            emotion_icons = {
                "sadness": "😢",
                "anger": "😠",
                "love": "❤️",
                "surprise": "😲",
                "fear": "😨",
                "joy": "😊"
            }

            icon = emotion_icons.get(predicted_emotion, "🧠")

            st.success(
                f"{icon} Predicted Emotion: "
                f"{predicted_emotion.title()}"
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    else:
        st.warning("Please enter some text before predicting.")

st.divider()
st.caption("Powered by Python, Scikit-learn, TF-IDF and Streamlit.")
