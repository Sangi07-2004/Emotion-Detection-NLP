import streamlit as st
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EmotionAI",
    page_icon="🎭",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("random_forest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()


# =========================================================
# EMOTION MAPPING
# =========================================================

# If LabelEncoder was used during training,
# sklearn assigns classes alphabetically:

emotion_names = {
    0: "Sadness 😢",
    1: "Anger 😡",
    2: "Love ❤️",
    3: "Surprise 😮",
    4: "Fear 😨",
    5: "Joy 😊"
}


# Description of emotions

emotion_description = {
    0: "Frustration, irritation or strong disagreement",
    1: "Fear, anxiety, nervousness or feeling threatened",
    2: "Happiness, excitement or positive feelings",
    3: "Affection, care, romance or emotional connection",
    4: "Sadness, loneliness, disappointment or feeling low",
    5: "Unexpectedness, shock, curiosity or astonishment"
}


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎭 EmotionAI")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Model")

st.sidebar.write(
    "**Algorithm:** Random Forest Classifier"
)

st.sidebar.write(
    "**Features:** TF-IDF"
)

st.sidebar.write(
    "**Classes:** 6"
)

st.sidebar.write(
    "**Test Accuracy:** 88.44%"
)

st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Emotions")

st.sidebar.write("0 → 😢 Sadness")
st.sidebar.write("1 → 😡 Anger")
st.sidebar.write("2 → ❤️ Love")
st.sidebar.write("3 → 😮 Surprise")
st.sidebar.write("4 → 😨 Fear")
st.sidebar.write("5 → 😊 Joy")


# =========================================================
# MAIN TITLE
# =========================================================

st.title("🎭 EmotionAI")

st.markdown(
    "Enter a sentence and let the machine learning model "
    "predict the emotion."
)


# =========================================================
# TEXT INPUT
# =========================================================

text = st.text_area(
    "Enter your text:",
    placeholder="Example: i feel deeply loved and cared for by the people around me",
    height=200
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🚀 Analyze Emotion"):

    # Check empty input
    if not text.strip():

        st.warning("⚠️ Please enter some text first.")

    else:

        # -------------------------------------------------
        # STEP 1: Convert text into TF-IDF
        # -------------------------------------------------

        text_tfidf = vectorizer.transform([text])


        # -------------------------------------------------
        # STEP 2: Prediction
        # -------------------------------------------------

        prediction = model.predict(text_tfidf)[0]


        # -------------------------------------------------
        # STEP 3: Probability
        # -------------------------------------------------

        probabilities = model.predict_proba(text_tfidf)[0]

        confidence = max(probabilities)


        # -------------------------------------------------
        # DEBUG INFORMATION
        # -------------------------------------------------

        # Uncomment these if you want to check the actual
        # prediction and classes.

        # st.write("Raw Prediction:", prediction)
        # st.write("Model Classes:", model.classes_)


        # -------------------------------------------------
        # HANDLE DIFFERENT LABEL TYPES
        # -------------------------------------------------

        # If model returns numpy integer
        try:
            prediction_int = int(prediction)

            predicted_emotion = emotion_names.get(
                prediction_int,
                f"Unknown (Class {prediction_int})"
            )

            description = emotion_description.get(
                prediction_int,
                "Emotion could not be identified."
            )

        except (ValueError, TypeError):

            # If model directly returns string labels
            string_mapping = {
                "sadness": "Sadness 😢",
                "anger": "Anger 😡",
                "love": "Love ❤️",
                "surprise": "Surprise 😮",
                "fear": "Fear 😨",
                "joy": "Joy 😊"
            }

            predicted_emotion = string_mapping.get(
                str(prediction).lower(),
                str(prediction)
            )

            description = ""


        # =================================================
        # RESULT
        # =================================================

        st.markdown("---")

        st.subheader("🔮 Predicted Emotion")

        st.success(
            f"### {predicted_emotion}"
        )

        st.write(
            f"**Confidence:** {confidence:.2%}"
        )

        if description:

            st.info(description)


        # =================================================
        # EMOTION PROBABILITY
        # =================================================

        st.subheader("📊 Emotion Probability")


        # Mapping for display
        display_names = {
            0: "Sadness 😢",
            1: "Anger 😡",
            2: "Love ❤️",
            3: "Surprise 😮",
            4: "Fear 😨",
            5: "Joy 😊"
        }


        # Display probabilities
        for cls, probability in zip(
            model.classes_,
            probabilities
        ):

            try:

                cls_int = int(cls)

                emotion = display_names.get(
                    cls_int,
                    f"Unknown (Class {cls_int})"
                )

            except (ValueError, TypeError):

                string_mapping = {
                    "anger": "Anger 😡",
                    "fear": "Fear 😨",
                    "joy": "Joy 😊",
                    "love": "Love ❤️",
                    "sadness": "Sadness 😢",
                    "surprise": "Surprise 😮"
                }

                emotion = string_mapping.get(
                    str(cls).lower(),
                    str(cls)
                )


            st.write(
                f"**{emotion}** — {probability:.2%}"
            )

            st.progress(float(probability))


        # =================================================
        # MODEL INFORMATION
        # =================================================

        st.markdown("---")

        st.caption(
            "Model: Random Forest Classifier | "
            "Features: TF-IDF | "
            "Test Accuracy: 88.44%"
        )