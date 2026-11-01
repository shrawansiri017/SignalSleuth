
import os
import uuid

import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from src.predict_news import predict_news
from src.deepfake_detector import analyze_image, analyze_video

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


def save_upload(file):
    filename = secure_filename(file.filename)

    if not filename or "." not in filename:
        raise ValueError("Invalid filename.")

    extension = filename.rsplit(".", 1)[-1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"
    path = os.path.join(UPLOAD_FOLDER, unique_name)

    file.save(path)
    return path, filename, extension


def format_deepfake_result(results):
    if isinstance(results, dict):
        results = [results]

    if not results:
        raise ValueError("The model returned no predictions.")

    predictions = []

    for item in results:
        label = str(item.get("label", "Unknown"))
        score = float(item.get("score", 0))

        predictions.append({
            "label": label,
            "confidence": round(score * 100, 2)
        })

    fake_result = next(
        (item for item in predictions
         if "fake" in item["label"].lower()),
        None
    )

    real_result = next(
        (item for item in predictions
         if "real" in item["label"].lower()),
        None
    )

    if fake_result and real_result:
        final_result = (
            fake_result if fake_result["confidence"] >
            real_result["confidence"] else real_result
        )
    elif fake_result:
        final_result = fake_result
    elif real_result:
        final_result = real_result
    else:
        raise ValueError(
            "The model returned labels that could not be identified."
        )

    return {
        "prediction": final_result["label"].upper(),
        "confidence": final_result["confidence"],
        "predictions": predictions,
        "message": (
            "Prediction from a pretrained deepfake detection model. "
            "This result is not a guarantee of authenticity."
        )
    }


@app.route("/api/predict", methods=["POST"])
def analyze_news():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Please enter a news article."}), 400

        result = predict_news(data["text"])

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/api/inspect-image", methods=["POST"])
def inspect_image():
    path = None

    try:
        if "image" not in request.files:
            return jsonify({"error": "Please upload an image."}), 400

        file = request.files["image"]

        if not file.filename:
            return jsonify({"error": "No image selected."}), 400

        path, original_name, extension = save_upload(file)

        if extension not in IMAGE_EXTENSIONS:
            return jsonify({"error": "Unsupported image format."}), 400

        with Image.open(path) as image:
            image.verify()

        results = analyze_image(path)
        response = format_deepfake_result(results)

        response["filename"] = original_name
        response["media_type"] = "image"

        return jsonify(response)

    except Exception as error:
        return jsonify({"error": str(error)}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)


@app.route("/api/inspect-video", methods=["POST"])
def inspect_video():
    path = None

    try:
        if "video" not in request.files:
            return jsonify({"error": "Please upload a video."}), 400

        file = request.files["video"]

        if not file.filename:
            return jsonify({"error": "No video selected."}), 400

        path, original_name, extension = save_upload(file)

        if extension not in VIDEO_EXTENSIONS:
            return jsonify({"error": "Unsupported video format."}), 400

        capture = cv2.VideoCapture(path)

        if not capture.isOpened():
            return jsonify({"error": "Invalid or corrupted video."}), 400

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        if frame_count <= 0:
            return jsonify({"error": "The video contains no readable frames."}), 400

        results = analyze_video(path)
        response = format_deepfake_result(results)

        response["filename"] = original_name
        response["media_type"] = "video"

        return jsonify(response)

    except Exception as error:
        return jsonify({"error": str(error)}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "File is too large. Maximum upload size is 50 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=True)
