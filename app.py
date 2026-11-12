
import os
import uuid
import secrets

import cv2
from PIL import Image
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session, flash
)
from werkzeug.utils import secure_filename

from src.predict_news import predict_news
from src.deepfake_detector import analyze_image, analyze_video

from database import (
    init_db,
    register_user,
    login_user as authenticate_user,
    get_guest_count,
    save_detection,
    get_user_history
)


app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY", "signalsleuth-development-secret-change-later"
)

UPLOAD_FOLDER = "uploads"
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

GUEST_LIMIT = 4

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

init_db()


# -------------------- COMMON HELPERS --------------------

def get_guest_id():
    if "guest_id" not in session:
        session["guest_id"] = secrets.token_urlsafe(24)

    return session["guest_id"]


def check_detection_access():
    if session.get("user_id"):
        return None

    guest_id = get_guest_id()

    if get_guest_count(guest_id) >= GUEST_LIMIT:
        return jsonify({
            "error": (
                "You have used all 4 free guest detections. "
                "Please register to continue."
            ),
            "limit_reached": True,
            "register_url": url_for("register")
        }), 403

    return None


def record_detection(detection_type, filename, prediction, confidence):
    user_id = session.get("user_id")
    guest_id = None if user_id else get_guest_id()

    save_detection(
        user_id,
        guest_id,
        detection_type,
        filename,
        prediction,
        confidence
    )


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


# -------------------- WEBSITE PAGES --------------------

@app.route("/")
def home():
    return render_template("pages/home.html")


@app.route("/fake-news")
def fake_news():
    return render_template("pages/fake_news.html")


@app.route("/image-detection")
def image_detection():
    return render_template("pages/image_detection.html")


@app.route("/video-detection")
def video_detection():
    return render_template("pages/video_detection.html")


@app.route("/about")
def about():
    return render_template("pages/about.html")


@app.route("/help")
def help_page():
    return render_template("pages/help.html")


@app.route("/dashboard")
def dashboard():
    if not session.get("user_id"):
        flash("Please log in to view your dashboard.", "error")
        return redirect(url_for("login"))

    history = get_user_history(session["user_id"])

    total = len(history)
    news_count = sum(
        1 for item in history if item["detection_type"] == "news"
    )
    image_count = sum(
        1 for item in history if item["detection_type"] == "image"
    )
    video_count = sum(
        1 for item in history if item["detection_type"] == "video"
    )

    return render_template(
        "pages/dashboard.html",
        history=history,
        total=total,
        news_count=news_count,
        image_count=image_count,
        video_count=video_count
    )


# -------------------- AUTHENTICATION --------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Please complete all required fields.", "error")

        elif password != confirm_password:
            flash("Passwords do not match.", "error")

        elif len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")

        else:
            created = register_user(name, email, password)

            if created:
                user = authenticate_user(email, password)

                session.clear()
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]

                flash("Your account was created successfully!", "success")
                return redirect(url_for("dashboard"))

            flash("An account with this email already exists.", "error")

    return render_template("auth/register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        user = authenticate_user(email, password)

        if user:
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]

            flash("Welcome back!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("auth/login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


# -------------------- FAKE NEWS DETECTION API --------------------

@app.route("/api/predict", methods=["POST"])
def analyze_news():
    access_error = check_detection_access()

    if access_error:
        return access_error

    try:
        data = request.get_json(silent=True)

        if not data or "text" not in data:
            return jsonify({"error": "Please enter a news article."}), 400

        result = predict_news(data["text"])

        if "error" in result:
            return jsonify(result), 400

        record_detection(
            "news",
            None,
            result["prediction"],
            result["confidence"]
        )

        return jsonify(result)

    except Exception as error:
        return jsonify({"error": str(error)}), 500


# -------------------- IMAGE DEEPFAKE DETECTION API --------------------

@app.route("/api/inspect-image", methods=["POST"])
def inspect_image():
    access_error = check_detection_access()

    if access_error:
        return access_error

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

        record_detection(
            "image",
            original_name,
            response["prediction"],
            response["confidence"]
        )

        return jsonify(response)

    except Exception as error:
        return jsonify({"error": str(error)}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)


# -------------------- VIDEO DEEPFAKE DETECTION API --------------------

@app.route("/api/inspect-video", methods=["POST"])
def inspect_video():
    access_error = check_detection_access()

    if access_error:
        return access_error

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
            return jsonify({
                "error": "The video contains no readable frames."
            }), 400

        results = analyze_video(path)
        response = format_deepfake_result(results)

        response["filename"] = original_name
        response["media_type"] = "video"

        record_detection(
            "video",
            original_name,
            response["prediction"],
            response["confidence"]
        )

        return jsonify(response)

    except Exception as error:
        return jsonify({"error": str(error)}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)


# -------------------- ERROR HANDLING --------------------

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "File is too large. Maximum upload size is 50 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=True)
