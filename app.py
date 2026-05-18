from pathlib import Path
import tempfile

import streamlit as st
import torch

from train_deepfake_detector import build_model, read_video_frames


PREFERRED_MODEL_PATH = Path("models/resnet18_lstm_deepfake_detector_best.pt")
FALLBACK_MODEL_PATH = Path("models/deepfake_detector_best.pt")
MODEL_PATH = PREFERRED_MODEL_PATH if PREFERRED_MODEL_PATH.exists() else FALLBACK_MODEL_PATH
ALLOWED_TYPES = ["mp4", "mov", "avi", "webm", "mkv"]


@st.cache_resource
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


def predict_video(video_path: Path) -> dict[str, float | str]:
    model, config, class_names, device = load_detector()
    frames = read_video_frames(
        video_path,
        config.get("frames_per_video", 8),
        config.get("image_size", 128),
        train=False,
        face_crop=config.get("face_crop", True),
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


st.title("Deepfake Detector")
st.write("Upload a video to classify it as real or fake.")

uploaded_file = st.file_uploader("Choose a video...", type=ALLOWED_TYPES)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Classify"):
        prediction = None
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = Path(temp_file.name)

        try:
            with st.spinner("Processing video frames..."):
                prediction = predict_video(temp_path)
        except Exception as error:
            st.error(f"Unable to process the uploaded video: {error}")
        finally:
            temp_path.unlink(missing_ok=True)

        if prediction is not None:
            label = "Fake" if prediction["prediction"] == "fake" else "Real"
            confidence = prediction["confidence"] * 100
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2f}%")
            st.write(f"Real probability: {prediction['real_probability'] * 100:.2f}%")
            st.write(f"Fake probability: {prediction['fake_probability'] * 100:.2f}%")
