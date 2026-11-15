
import os
import threading

import torch
from PIL import Image

# Existing model name retained for the legacy video loader
MODEL_NAME = "KoreaPeter/ms-eff-gcvit-deepfake-b0-celeb-df-v2"

# New AI-generated image detection model
IMAGE_MODEL_NAME = "bendavidsteel/openfake-swinv2-base"

# Image model instances
deepfake_classifier = None
image_processor = None
model_load_error = None

# Prevent simultaneous model loading
model_lock = threading.Lock()


def load_deepfake_model():
    global deepfake_classifier
    global image_processor
    global model_load_error

    if deepfake_classifier is not None:
        return deepfake_classifier

    with model_lock:
        if deepfake_classifier is not None:
            return deepfake_classifier

        try:
            print("Loading OpenFake image detection model...")

            from transformers import (
                AutoImageProcessor,
                AutoModelForImageClassification
            )

            image_processor = AutoImageProcessor.from_pretrained(
                IMAGE_MODEL_NAME
            )

            deepfake_classifier = (
                AutoModelForImageClassification.from_pretrained(
                    IMAGE_MODEL_NAME
                )
            )

            deepfake_classifier.to("cpu")
            deepfake_classifier.eval()

            model_load_error = None

            print("OpenFake image model loaded successfully!")

            return deepfake_classifier

        except Exception as error:
            model_load_error = str(error)

            print("Image model loading failed:", model_load_error)

            raise RuntimeError(
                f"Could not load OpenFake image model: {model_load_error}"
            ) from error


# Existing native video model loader retained
video_classifier = None


def load_video_model():
    global video_classifier

    if video_classifier is None:
        from transformers import pipeline

        print("Loading native video deepfake model...")

        video_classifier = pipeline(
            "video-classification",
            model=MODEL_NAME,
            trust_remote_code=True,
            device=-1,
            token=os.getenv("HF_TOKEN")
        )

        print("Native video model loaded successfully!")

    return video_classifier


# New OpenFake image analysis
def analyze_image(image_path):
    classifier = load_deepfake_model()

    with Image.open(image_path) as uploaded_image:
        image = uploaded_image.convert("RGB")

    inputs = image_processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        logits = classifier(**inputs).logits

        probabilities = (logits / 2.0).softmax(dim=-1)[0]

    real_score = probabilities[0].item()
    fake_score = probabilities[1].item()

    results = [
        {
            "label": "REAL",
            "score": real_score
        },
        {
            "label": "FAKE",
            "score": fake_score
        }
    ]

    print("OPENFAKE IMAGE MODEL OUTPUT:", results)

    return results


# Existing DeMamba video detector preserved
def analyze_video(video_path):
    from src.demamba_video import analyze_demamba_video

    return analyze_demamba_video(video_path)
