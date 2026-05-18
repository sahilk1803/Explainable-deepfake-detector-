import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class TrainConfig:
    fake_dir: str = "Fake"
    real_dir: str = "Real"
    output_dir: str = "models"
    epochs: int = 8
    batch_size: int = 4
    frames_per_video: int = 8
    image_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 0
    model: str = "resnet18_lstm"
    face_crop: bool = True
    max_videos_per_class: int | None = None
    pretrained: bool = True
    freeze_backbone: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_videos(
    fake_dir: Path, real_dir: Path, max_videos_per_class: int | None = None
) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label_dir, label in ((real_dir, 0), (fake_dir, 1)):
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing dataset folder: {label_dir}")
        videos = sorted(
            path for path in label_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise FileNotFoundError(f"No video files found in: {label_dir}")
        if max_videos_per_class is not None:
            videos = videos[:max_videos_per_class]
        samples.extend((path, label) for path in videos)
    return samples


def stratified_split(
    samples: list[tuple[Path, int]], val_ratio: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    by_source: dict[str, list[tuple[Path, int]]] = {}
    for path, label in samples:
        source_id = path.stem if label == 0 else path.stem.split("_", 1)[0]
        by_source.setdefault(source_id, []).append((path, label))

    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    source_groups = list(by_source.values())
    rng.shuffle(source_groups)
    val_count = max(1, round(len(source_groups) * val_ratio))

    for group in source_groups[:val_count]:
        val.extend(group)
    for group in source_groups[val_count:]:
        train.extend(group)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def sample_frame_indices(total_frames: int, count: int, train: bool) -> list[int]:
    if total_frames <= 0:
        return [0] * count

    if train:
        return sorted(random.randrange(total_frames) for _ in range(count))

    if count == 1:
        return [total_frames // 2]
    return np.linspace(0, max(0, total_frames - 1), count, dtype=int).tolist()


FACE_DETECTOR = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)


def crop_largest_face(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return frame

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    pad = int(max(width, height) * 0.25)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + width + pad)
    y2 = min(frame.shape[0], y + height + pad)
    return frame[y1:y2, x1:x2]


def read_video_frames(
    path: Path, count: int, image_size: int, train: bool, face_crop: bool = True
) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total_frames, count, train)
    frames: list[np.ndarray] = []

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
        raise RuntimeError(f"Could not read frames from video: {path}")

    while len(frames) < count:
        frames.append(frames[-1])

    video = np.stack(frames[:count]).astype(np.float32) / 255.0
    video = (video - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(video).permute(0, 3, 1, 2)


class DeepfakeVideoDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[tuple[Path, int]],
        frames_per_video: int,
        image_size: int,
        train: bool,
        face_crop: bool,
    ) -> None:
        self.samples = list(samples)
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        self.train = train
        self.face_crop = face_crop

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        frames = read_video_frames(
            path, self.frames_per_video, self.image_size, self.train, self.face_crop
        )
        return frames, torch.tensor(label, dtype=torch.long)


class SmallDeepfakeCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 192),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(192, 2),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count, channels, height, width = videos.shape
        frames = videos.reshape(batch_size * frame_count, channels, height, width)
        frame_logits = self.classifier(self.features(frames))
        return frame_logits.reshape(batch_size, frame_count, 2).mean(dim=1)


class VideoFrameClassifier(nn.Module):
    def __init__(self, frame_model: nn.Module) -> None:
        super().__init__()
        self.frame_model = frame_model

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count, channels, height, width = videos.shape
        frames = videos.reshape(batch_size * frame_count, channels, height, width)
        frame_logits = self.frame_model(frames)
        return frame_logits.reshape(batch_size, frame_count, 2).mean(dim=1)


class MobileNetLSTMClassifier(nn.Module):
    def __init__(
        self, frame_model: nn.Module, hidden_size: int = 128, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.features = frame_model.features
        self.avgpool = frame_model.avgpool
        feature_size = frame_model.classifier[0].in_features
        self.temporal = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count, channels, height, width = videos.shape
        frames = videos.reshape(batch_size * frame_count, channels, height, width)
        features = self.features(frames)
        features = self.avgpool(features).flatten(1)
        sequence = features.reshape(batch_size, frame_count, -1)
        _, (hidden, _) = self.temporal(sequence)
        return self.classifier(hidden[-1])


class ResNet18LSTMClassifier(nn.Module):
    def __init__(
        self, frame_model: nn.Module, hidden_size: int = 128, dropout: float = 0.3
    ) -> None:
        super().__init__()
        feature_size = frame_model.fc.in_features
        frame_model.fc = nn.Identity()
        self.features = frame_model
        self.temporal = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count, channels, height, width = videos.shape
        frames = videos.reshape(batch_size * frame_count, channels, height, width)
        features = self.features(frames)
        sequence = features.reshape(batch_size, frame_count, -1)
        _, (hidden, _) = self.temporal(sequence)
        return self.classifier(hidden[-1])


def build_model(
    name: str, pretrained: bool = False, freeze_backbone: bool = False
) -> nn.Module:
    if name in {"small_cnn", "SmallDeepfakeCNN"}:
        return SmallDeepfakeCNN()
    if name in {"mobilenet", "VideoFrameClassifier"}:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        frame_model = models.mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            for parameter in frame_model.features.parameters():
                parameter.requires_grad = False
        in_features = frame_model.classifier[-1].in_features
        frame_model.classifier[-1] = nn.Linear(in_features, 2)
        return VideoFrameClassifier(frame_model)
    if name in {"mobilenet_lstm", "MobileNetLSTMClassifier"}:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        frame_model = models.mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            for parameter in frame_model.features.parameters():
                parameter.requires_grad = False
        return MobileNetLSTMClassifier(frame_model)
    if name in {"resnet18_lstm", "ResNet18LSTMClassifier"}:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        frame_model = models.resnet18(weights=weights)
        if freeze_backbone:
            for name, parameter in frame_model.named_parameters():
                if not name.startswith("fc."):
                    parameter.requires_grad = False
        return ResNet18LSTMClassifier(frame_model)
    raise ValueError(f"Unknown model: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 1,
    total_epochs: int = 1,
    phase: str = "train",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    total_batches = len(loader)
    progress_interval = max(1, total_batches // 10)
    phase_started_at = time.perf_counter()

    for batch_index, (videos, labels) in enumerate(loader, start=1):
        videos = videos.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(videos)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += labels.size(0)

        if (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % progress_interval == 0
        ):
            elapsed_seconds = time.perf_counter() - phase_started_at
            average_batch_seconds = elapsed_seconds / batch_index
            remaining_seconds = average_batch_seconds * (total_batches - batch_index)
            current_accuracy = total_correct / max(1, total_items)
            print(
                f"Epoch {epoch:02d}/{total_epochs} {phase} "
                f"{batch_index}/{total_batches} "
                f"({batch_index / total_batches * 100:.0f}%) "
                f"loss={total_loss / max(1, total_items):.4f} "
                f"acc={current_accuracy:.3f} "
                f"eta={remaining_seconds / 60:.1f}m",
                flush=True,
            )

    return {
        "loss": total_loss / max(1, total_items),
        "accuracy": total_correct / max(1, total_items),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    metrics: dict[str, float],
    class_names: list[str],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": config.model,
            "config": asdict(config),
            "metrics": metrics,
            "class_names": class_names,
        },
        path,
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a video deepfake detector.")
    parser.add_argument("--fake-dir", default="Fake")
    parser.add_argument("--real-dir", default="Real")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=["resnet18_lstm", "mobilenet", "mobilenet_lstm", "small_cnn"],
        default="resnet18_lstm",
    )
    parser.add_argument("--face-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-videos-per-class", type=int, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=True
    )
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    samples = collect_videos(
        Path(config.fake_dir), Path(config.real_dir), config.max_videos_per_class
    )
    train_samples, val_samples = stratified_split(samples, config.val_ratio, config.seed)

    train_dataset = DeepfakeVideoDataset(
        train_samples,
        config.frames_per_video,
        config.image_size,
        train=True,
        face_crop=config.face_crop,
    )
    val_dataset = DeepfakeVideoDataset(
        val_samples,
        config.frames_per_video,
        config.image_size,
        train=False,
        face_crop=config.face_crop,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model, config.pretrained, config.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = (
        "resnet18_lstm_deepfake_detector_best.pt"
        if config.model == "resnet18_lstm"
        else "deepfake_detector_best.pt"
    )
    best_path = output_dir / checkpoint_name
    metrics_path = output_dir / "training_metrics.json"

    history = []
    best_val_accuracy = -1.0
    training_started_at = time.perf_counter()
    print(f"Device: {device}")
    print(f"Training videos: {len(train_samples)} | Validation videos: {len(val_samples)}")
    print(
        f"Settings: epochs={config.epochs}, batch_size={config.batch_size}, "
        f"frames_per_video={config.frames_per_video}, image_size={config.image_size}, "
        f"model={config.model}",
        flush=True,
    )

    for epoch in range(1, config.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            epoch=epoch,
            total_epochs=config.epochs,
            phase="train",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch,
            total_epochs=config.epochs,
            phase="val",
        )
        epoch_seconds = time.perf_counter() - epoch_started_at
        elapsed_seconds = time.perf_counter() - training_started_at
        average_epoch_seconds = elapsed_seconds / epoch
        remaining_seconds = average_epoch_seconds * (config.epochs - epoch)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:02d}/{config.epochs} "
            f"train_loss={epoch_metrics['train_loss']:.4f} "
            f"train_acc={epoch_metrics['train_accuracy']:.3f} "
            f"val_loss={epoch_metrics['val_loss']:.4f} "
            f"val_acc={epoch_metrics['val_accuracy']:.3f} "
            f"epoch_time={epoch_seconds / 60:.1f}m "
            f"eta={remaining_seconds / 60:.1f}m",
            flush=True,
        )

        if val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                best_path,
                model,
                config,
                epoch_metrics,
                class_names=["real", "fake"],
            )

    metrics = {
        "config": asdict(config),
        "class_names": ["real", "fake"],
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "best_val_accuracy": best_val_accuracy,
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved best model to: {best_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
