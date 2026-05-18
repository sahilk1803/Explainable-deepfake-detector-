import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from train_deepfake_detector import build_model, crop_largest_face, read_video_frames


PREFERRED_MODEL_PATH = Path("models/resnet18_lstm_deepfake_detector_best.pt")
FALLBACK_MODEL_PATH = Path("models/deepfake_detector_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether a video is real or fake.")
    parser.add_argument("video", help="Path to a video file.")
    parser.add_argument(
        "--model",
        default=str(PREFERRED_MODEL_PATH if PREFERRED_MODEL_PATH.exists() else FALLBACK_MODEL_PATH),
        help="Path to a trained checkpoint.",
    )
    parser.add_argument("--frames-per-video", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--segments",
        type=int,
        default=1,
        help="Split the video into this many segments and print each segment score.",
    )
    return parser.parse_args()


def read_video_segment(
    path: Path,
    segment_index: int,
    segment_count: int,
    frames_per_video: int,
    image_size: int,
    face_crop: bool,
) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine frame count for: {path}")

    start = int(total_frames * segment_index / segment_count)
    end = int(total_frames * (segment_index + 1) / segment_count) - 1
    end = max(start, min(end, total_frames - 1))
    indices = np.linspace(start, end, frames_per_video, dtype=int).tolist()
    frames = []

    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if face_crop:
            frame = crop_largest_face(frame)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    capture.release()

    if not frames:
        raise RuntimeError(f"Could not read segment {segment_index + 1} from video: {path}")

    while len(frames) < frames_per_video:
        frames.append(frames[-1])

    video = np.stack(frames[:frames_per_video]).astype(np.float32) / 255.0
    video = (video - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(video).permute(0, 3, 1, 2)


def predict_tensor(model: torch.nn.Module, video: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(video.unsqueeze(0))
        return torch.softmax(logits, dim=1).squeeze(0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", ["real", "fake"])

    frames_per_video = args.frames_per_video or config.get("frames_per_video", 8)
    image_size = args.image_size or config.get("image_size", 128)

    model_name = checkpoint.get("model_name", config.get("model", "small_cnn"))
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Device: {device}")

    video_path = Path(args.video)
    face_crop = config.get("face_crop", True)

    if args.segments <= 1:
        video = read_video_frames(
            video_path,
            frames_per_video,
            image_size,
            train=False,
            face_crop=face_crop,
        )
        probabilities = predict_tensor(model, video.to(device))
    else:
        segment_probabilities = []
        for segment_index in range(args.segments):
            video = read_video_segment(
                video_path,
                segment_index,
                args.segments,
                frames_per_video,
                image_size,
                face_crop,
            )
            segment_probabilities.append(predict_tensor(model, video.to(device)))

        probabilities = torch.stack(segment_probabilities).mean(dim=0)
        fake_index = class_names.index("fake")
        print("Segment scores:")
        for index, segment_probs in enumerate(segment_probabilities, start=1):
            print(
                f"  segment {index:02d}: "
                f"real={segment_probs[class_names.index('real')].item():.3f} "
                f"fake={segment_probs[fake_index].item():.3f}"
            )
        print(f"Max fake segment: {max(p[fake_index].item() for p in segment_probabilities):.3f}")

    predicted_index = int(probabilities.argmax().item())
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.3f}")
    for class_name, probability in zip(class_names, probabilities.tolist()):
        print(f"{class_name}: {probability:.3f}")


if __name__ == "__main__":
    main()
