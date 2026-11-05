
import os
import tempfile

import streamlit as st
from PIL import Image

from src.predict_news import predict_news
from src.deepfake_detector import analyze_image, analyze_video


st.set_page_config(
    page_title="SignalSleuth",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 SignalSleuth")
st.subheader("AI-Based Fake News and Deepfake Detection")
st.write(
    "Analyze news articles, images and videos using "
    "machine learning and pretrained deepfake models."
)

st.info(
    "AI predictions are not guarantees of authenticity. "
    "Always verify important information independently."
)

module = st.sidebar.radio(
    "Choose Detection Module",
    [
        "Fake News Detection",
        "Image Deepfake Detection",
        "Video Deepfake Detection"
    ]
)


def show_deepfake_result(results):
    if isinstance(results, dict):
        results = [results]

    fake = next(
        (x for x in results if "fake" in x["label"].lower()),
        None
    )
    real = next(
        (x for x in results if "real" in x["label"].lower()),
        None
    )

    if fake and real:
        result = (
            fake if float(fake["score"]) > float(real["score"])
            else real
        )
    else:
        result = fake or real

    if not result:
        st.error("Could not identify the model prediction.")
        return

    confidence = float(result["score"]) * 100
    prediction = result["label"].upper()

    st.subheader("Detection Result")
    st.metric("Prediction", prediction)
    st.metric("Confidence", f"{confidence:.2f}%")

    st.write("Model output:")
    st.json(results)


if module == "Fake News Detection":

    st.header("📰 Fake News Detection")
    news_text = st.text_area(
        "Paste your news article here:",
        height=220
    )

    if st.button("Analyze News", type="primary"):
        if not news_text.strip():
            st.warning("Please enter a news article.")
        else:
            try:
                with st.spinner("Analyzing news..."):
                    result = predict_news(news_text)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.subheader("Detection Result")
                    st.metric(
                        "Prediction",
                        result["prediction"]
                    )
                    st.metric(
                        "Confidence",
                        f'{result["confidence"]:.2f}%'
                    )

                    st.subheader("Explainable AI")
                    st.write(result["explanation_type"])

                    for item in result["explanation"]:
                        st.write(
                            f'• {item["word"]} — '
                            f'Score: {item["score"]}'
                        )

            except Exception as error:
                st.error(f"Analysis failed: {error}")


elif module == "Image Deepfake Detection":

    st.header("🖼️ Image Deepfake Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", width=450)

        if st.button("Analyze Image", type="primary"):
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp:
                    temp.write(uploaded_image.getvalue())
                    temp_path = temp.name

                with st.spinner("Analyzing image..."):
                    results = analyze_image(temp_path)

                show_deepfake_result(results)

            except Exception as error:
                st.error(f"Image analysis failed: {error}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


elif module == "Video Deepfake Detection":

    st.header("🎥 Video Deepfake Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv", "webm"]
    )

    if uploaded_video:
        st.video(uploaded_video)

        if st.button("Analyze Video", type="primary"):
            temp_path = None

            try:
                suffix = os.path.splitext(
                    uploaded_video.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as temp:
                    temp.write(uploaded_video.getvalue())
                    temp_path = temp.name

                with st.spinner(
                    "Analyzing video frames. Please wait..."
                ):
                    results = analyze_video(temp_path)

                show_deepfake_result(results)

            except Exception as error:
                st.error(f"Video analysis failed: {error}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


st.divider()
st.caption(
    "SignalSleuth | Helping users look beyond the digital illusion 🔍"
)
