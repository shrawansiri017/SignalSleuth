
from transformers import pipeline
from PIL import Image
import torch

MODEL_NAME = "KoreaPeter/ms-eff-gcvit-deepfake-b0-celeb-df-v2"

device = 0 if torch.cuda.is_available() else -1

image_classifier = None
video_classifier = None


def load_image_model():
    global image_classifier

    if image_classifier is None:
        print("Loading deepfake image model...")
        image_classifier = pipeline(
            "image-classification",
            model=MODEL_NAME,
            trust_remote_code=True,
            device=device
        )

    return image_classifier


def load_video_model():
    global video_classifier

    if video_classifier is None:
        print("Loading deepfake video model...")
        video_classifier = pipeline(
            "video-classification",
            model=MODEL_NAME,
            trust_remote_code=True,
            device=device
        )

    return video_classifier


def analyze_image(image_path):
    classifier = load_image_model()
    image = Image.open(image_path).convert("RGB")
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
