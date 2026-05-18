import uuid
from pathlib import Path

import cv2
import torch
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from explainability import create_gradcam_explanations
from train_deepfake_detector import build_model, read_video_frames


UPLOAD_FOLDER = Path("static/uploads")
PREFERRED_MODEL_PATH = Path("models/resnet18_lstm_deepfake_detector_best.pt")
FALLBACK_MODEL_PATH = Path("models/deepfake_detector_best.pt")
MODEL_PATH = PREFERRED_MODEL_PATH if PREFERRED_MODEL_PATH.exists() else FALLBACK_MODEL_PATH
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv"}

app = Flask(__name__)
app.secret_key = "change_this_secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_detector():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_name = checkpoint.get("model_name", config.get("model", "small_cnn"))
    class_names = checkpoint.get("class_names", ["real", "fake"])

    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, class_names, device


model, model_config, class_names, device = load_detector()


def predict_video(video_path: Path) -> dict[str, float | str]:
    frames = read_video_frames(
        video_path,
        model_config.get("frames_per_video", 8),
        model_config.get("image_size", 128),
        train=False,
        face_crop=model_config.get("face_crop", True),
    )

    with torch.no_grad():
        logits = model(frames.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_index = int(probabilities.argmax().item())
    prediction = class_names[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_probability": float(probabilities[class_names.index("real")].item()),
        "fake_probability": float(probabilities[class_names.index("fake")].item()),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if not file or not allowed_file(file.filename):
            flash("Allowed file types: mp4, mov, avi, webm, mkv")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)

        try:
            prediction = predict_video(filepath)
        except Exception as error:
            flash(f"Unable to process the uploaded video: {error}")
            return redirect(request.url)

        is_fake = prediction["prediction"] == "fake"
        result = "Fake" if is_fake else "Real"
        confidence = prediction["confidence"] * 100

        return render_template(
            "index.html",
            result=result,
            confidence=confidence,
            filename=filename,
            is_video=True,
            model_confidence=confidence,
            real_probability=prediction["real_probability"] * 100,
            fake_probability=prediction["fake_probability"] * 100,
        )

    return render_template("index.html", result=None, confidence=None)


@app.route("/extract_faces/<filename>", methods=["GET"])
def extract_faces(filename):
    filepath = UPLOAD_FOLDER / secure_filename(filename)
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    capture = cv2.VideoCapture(str(filepath))
    if not capture.isOpened():
        return jsonify({"error": "Unable to open video"}), 400

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = 8
    frame_indices = [
        int(index)
        for index in torch.linspace(0, max(0, total_frames - 1), sample_count).tolist()
    ]

    faces = []
    face_dir = UPLOAD_FOLDER / "faces"
    face_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        for x, y, width, height in detections[:1]:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            image_name = f"{uuid.uuid4().hex}.jpg"
            image_path = face_dir / image_name
            cv2.imwrite(str(image_path), frame)
            faces.append(
                {
                    "frame_idx": frame_index,
                    "image": url_for(
                        "static", filename=f"uploads/faces/{image_name}", _external=False
                    ),
                }
            )
            break

    capture.release()

    if not faces:
        return jsonify({"error": "No faces detected in video"}), 400

    return jsonify({"success": True, "faces": faces})


@app.route("/api/predict/<filename>", methods=["GET"])
def api_predict(filename):
    filepath = UPLOAD_FOLDER / secure_filename(filename)
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        prediction = predict_video(filepath)
    except Exception as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(
        {
            "prediction": prediction["prediction"],
            "confidence": prediction["confidence"],
            "real_probability": prediction["real_probability"],
            "fake_probability": prediction["fake_probability"],
        }
    )


@app.route("/explain/<filename>", methods=["GET"])
def explain(filename):
    filepath = UPLOAD_FOLDER / secure_filename(filename)
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        explanations = create_gradcam_explanations(
            model=model,
            video_path=filepath,
            output_dir=UPLOAD_FOLDER / "explanations",
            image_url_prefix="/static/uploads/explanations",
            class_names=class_names,
            device=device,
            frames_per_video=model_config.get("frames_per_video", 8),
            image_size=model_config.get("image_size", 128),
            face_crop=model_config.get("face_crop", True),
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(explanations)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=False)
