
import os
import threading

from PIL import Image

MODEL_NAME = "KoreaPeter/ms-eff-gcvit-deepfake-b0-celeb-df-v2"

image_classifier = None
video_classifier = None

image_load_error = None
video_load_error = None

image_lock = threading.Lock()
video_lock = threading.Lock()


def load_image_model():
    global image_classifier, image_load_error

    with image_lock:
        if image_classifier is not None:
            return image_classifier

        if image_load_error is not None:
            raise RuntimeError(
                f"Image model previously failed to load: {image_load_error}"
            )

        try:
            print("Loading deepfake image model...")

            from transformers import pipeline

            image_classifier = pipeline(
                "image-classification",
                model=MODEL_NAME,
                trust_remote_code=True,
                device=-1,
                token=os.getenv("HF_TOKEN")
            )

            print("Image model loaded successfully!")
            return image_classifier

        except Exception as error:
            image_load_error = str(error)
            print("Image model loading failed:", image_load_error)
            raise RuntimeError(
                f"Could not load image model: {image_load_error}"
            ) from error


def load_video_model():
    global video_classifier, video_load_error

    with video_lock:
        if video_classifier is not None:
            return video_classifier

        if video_load_error is not None:
            raise RuntimeError(
                f"Video model previously failed to load: {video_load_error}"
            )

        try:
            print("Loading deepfake video model...")

            from transformers import pipeline

            video_classifier = pipeline(
                "video-classification",
                model=MODEL_NAME,
                trust_remote_code=True,
                device=-1,
                token=os.getenv("HF_TOKEN")
            )

            print("Video model loaded successfully!")
            return video_classifier

        except Exception as error:
            video_load_error = str(error)
            print("Video model loading failed:", video_load_error)
            raise RuntimeError(
                f"Could not load video model: {video_load_error}"
            ) from error


def analyze_image(image_path):
    classifier = load_image_model()

    with Image.open(image_path) as uploaded_image:
        image = uploaded_image.convert("RGB")

    results = classifier(image)

    return results


def analyze_video(video_path):
    classifier = load_video_model()

    results = classifier(
        video_path,
        num_frames=8,
        agg_mode="conf"
    )

    return results
