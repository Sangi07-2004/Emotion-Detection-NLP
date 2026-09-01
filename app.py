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
# LOAD MODEL, VECTORIZER AND EMOTION MAPPING
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("linear_svc_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    emotion_numbers = joblib.load("emotion_mapping.pkl")

    reverse_emotion = {
        value: key
        for key, value in emotion_numbers.items()
    }

    return model, vectorizer, reverse_emotion


try:

    model, vectorizer, reverse_emotion = load_model()

except Exception as e:

    st.error("❌ Model files could not be loaded.")

    st.write("Make sure these files are in the same folder as app.py:")

    st.code("""
linear_svc_model.pkl
tfidf_vectorizer.pkl
emotion_mapping.pkl
""")

    st.stop()


# =========================================================
# EMOTION INFORMATION
# =========================================================

emotion_emojis = {

    "sadness": "😢",
    "anger": "😡",
    "love": "❤️",
    "surprise": "😮",
    "fear": "😨",
    "joy": "😊"

}


emotion_description = {

    "sadness":
        "Sadness, loneliness, disappointment or feeling low",

    "anger":
        "Frustration, irritation or strong disagreement",

    "love":
        "Affection, care, romance or emotional connection",

    "surprise":
        "Unexpectedness, shock, curiosity or astonishment",

    "fear":
        "Fear, anxiety, nervousness or feeling threatened",

    "joy":
        "Happiness, excitement or positive feelings"

}


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎭 EmotionAI")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Model")

st.sidebar.write(
    "**Algorithm:** Linear Support Vector Classifier"
)

st.sidebar.write(
    "**Features:** TF-IDF"
)

st.sidebar.write(
    "**Classes:** 6"
)

st.sidebar.write(
    "**Test Accuracy:** 90.63%"
)

st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Emotions")

st.sidebar.write("😡 Anger")
st.sidebar.write("😨 Fear")
st.sidebar.write("😊 Joy")
st.sidebar.write("❤️ Love")
st.sidebar.write("😢 Sadness")
st.sidebar.write("😮 Surprise")


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
    placeholder=(
        "Example: I feel deeply loved and cared for "
        "by the people around me"
    ),
    height=200
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🚀 Analyze Emotion"):

    # -----------------------------------------------------
    # CHECK EMPTY INPUT
    # -----------------------------------------------------

    if not text.strip():

        st.warning("⚠️ Please enter some text first.")

    else:

        # -------------------------------------------------
        # STEP 1: TF-IDF
        # -------------------------------------------------

        text_tfidf = vectorizer.transform([text])


        # -------------------------------------------------
        # STEP 2: PREDICTION
        # -------------------------------------------------

        prediction = model.predict(text_tfidf)[0]


        # -------------------------------------------------
        # STEP 3: CONVERT NUMBER TO EMOTION
        # -------------------------------------------------

        try:

            prediction_int = int(prediction)

            predicted_emotion = reverse_emotion[
                prediction_int
            ].lower()

        except (ValueError, TypeError):

            predicted_emotion = str(
                prediction
            ).lower()


        # -------------------------------------------------
        # EMOTION DETAILS
        # -------------------------------------------------

        emoji = emotion_emojis.get(
            predicted_emotion,
            "🙂"
        )

        description = emotion_description.get(
            predicted_emotion,
            "Emotion detected from your text."
        )


        # =================================================
        # PROBABILITY
        # =================================================

        probabilities = None
        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                text_tfidf
            )[0]

            confidence = max(probabilities)


        # =================================================
        # RESULT
        # =================================================

        st.markdown("---")

        st.subheader("🔮 Predicted Emotion")


        st.success(
            f"### {predicted_emotion.capitalize()} {emoji}"
        )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        if confidence is not None:

            st.write(
                f"**Confidence:** {confidence:.2%}"
            )

        else:

            st.write(
                "**Confidence:** Not available"
            )


        # -------------------------------------------------
        # DESCRIPTION
        # -------------------------------------------------

        st.info(description)


        # =================================================
        # EMOTION PROBABILITY
        # =================================================

        if probabilities is not None:

            st.subheader("📊 Emotion Probability")


            # Keep display order consistent
            display_order = [
                "sadness",
                "anger",
                "love",
                "surprise",
                "fear",
                "joy"
            ]


            probability_dict = {}


            # Create emotion → probability mapping
            for cls, probability in zip(
                model.classes_,
                probabilities
            ):

                try:

                    cls_int = int(cls)

                    emotion = reverse_emotion[
                        cls_int
                    ].lower()

                except (ValueError, TypeError):

                    emotion = str(
                        cls
                    ).lower()

                probability_dict[
                    emotion
                ] = probability


            # -------------------------------------------------
            # DISPLAY ALL EMOTIONS
            # -------------------------------------------------

            for emotion in display_order:

                probability = probability_dict.get(
                    emotion,
                    0
                )

                emoji = emotion_emojis.get(
                    emotion,
                    "🙂"
                )


                st.write(
                    f"**{emotion.capitalize()} "
                    f"{emoji}** — "
                    f"{probability:.2%}"
                )


                st.progress(
                    float(probability)
                )


        else:

            st.info(
                "Probability information is not available "
                "for this model."
            )


        # =================================================
        # MODEL INFORMATION
        # =================================================

        st.markdown("---")

        st.caption(
            "Model: Linear Support Vector Classifier | "
            "Features: TF-IDF | "
            "Test Accuracy: 90.63%"
        )