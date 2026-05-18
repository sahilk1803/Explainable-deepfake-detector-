import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from train_deepfake_detector import crop_largest_face, sample_frame_indices


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(
        self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, video: torch.Tensor, target_index: int) -> tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(video)
        score = logits[:, target_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture model activations.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cams = torch.relu((weights * self.activations).sum(dim=1))
        cams = cams.detach().cpu().numpy()

        normalized = []
        for cam in cams:
            cam = cam - cam.min()
            denominator = cam.max()
            if denominator > 0:
                cam = cam / denominator
            normalized.append(cam)

        probabilities = torch.softmax(logits.detach(), dim=1).squeeze(0).cpu()
        return np.stack(normalized), probabilities


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    model_name = type(model).__name__

    if model_name == "ResNet18LSTMClassifier":
        return model.features.layer4[-1].conv2
    if model_name == "VideoFrameClassifier":
        return model.frame_model.features[-1]
    if model_name == "MobileNetLSTMClassifier":
        return model.features[-1]
    if model_name == "SmallDeepfakeCNN":
        return model.features[-2]

    raise RuntimeError(f"Grad-CAM is not configured for model type: {model_name}")


def _prepare_frame(frame: np.ndarray, image_size: int, face_crop: bool) -> tuple[np.ndarray, np.ndarray]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if face_crop:
        rgb_frame = crop_largest_face(rgb_frame)

    display_frame = cv2.resize(rgb_frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    model_frame = display_frame.astype(np.float32) / 255.0
    model_frame = (model_frame - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return display_frame, model_frame


def create_gradcam_explanations(
    model: nn.Module,
    video_path: Path,
    output_dir: Path,
    image_url_prefix: str,
    class_names: list[str],
    device: torch.device,
    frames_per_video: int,
    image_size: int,
    face_crop: bool,
    max_outputs: int = 6,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(total_frames, frames_per_video, train=False)
    display_frames: list[np.ndarray] = []
    model_frames: list[np.ndarray] = []
    used_indices: list[int] = []

    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue

        display_frame, model_frame = _prepare_frame(frame, image_size, face_crop)
        display_frames.append(display_frame)
        model_frames.append(model_frame)
        used_indices.append(int(frame_index))

    capture.release()

    if not model_frames:
        raise RuntimeError(f"Could not read frames from video: {video_path}")

    while len(model_frames) < frames_per_video:
        model_frames.append(model_frames[-1])
        display_frames.append(display_frames[-1])
        used_indices.append(used_indices[-1])

    video = torch.from_numpy(np.stack(model_frames[:frames_per_video])).permute(0, 3, 1, 2)
    video = video.unsqueeze(0).to(device)

    target_layer = get_gradcam_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    try:
        with torch.enable_grad():
            cams, probabilities = gradcam.generate(video, target_index=class_names.index("fake"))
    finally:
        gradcam.close()

    frame_probabilities = []
    with torch.no_grad():
        for frame_position in range(frames_per_video):
            frame_clip = video[:, frame_position : frame_position + 1].repeat(
                1, frames_per_video, 1, 1, 1
            )
            frame_logits = model(frame_clip)
            frame_probs = torch.softmax(frame_logits, dim=1).squeeze(0).cpu()
            frame_probabilities.append(
                {
                    "real_probability": float(frame_probs[class_names.index("real")].item()),
                    "fake_probability": float(frame_probs[class_names.index("fake")].item()),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    explanations = []
    step = max(1, len(display_frames) // max_outputs)
    selected_positions = list(range(0, len(display_frames), step))[:max_outputs]

    for position in selected_positions:
        display_frame = display_frames[position]
        cam = cv2.resize(cams[position], (display_frame.shape[1], display_frame.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR), 0.55, heatmap, 0.45, 0)

        image_name = f"{uuid.uuid4().hex}.jpg"
        image_path = output_dir / image_name
        cv2.imwrite(str(image_path), overlay)
        explanations.append(
            {
                "frame_idx": used_indices[position],
                "image": f"{image_url_prefix}/{image_name}",
                "real_probability": frame_probabilities[position]["real_probability"],
                "fake_probability": frame_probabilities[position]["fake_probability"],
            }
        )

    return {
        "success": True,
        "target_class": "fake",
        "fake_probability": float(probabilities[class_names.index("fake")].item()),
        "real_probability": float(probabilities[class_names.index("real")].item()),
        "frames": explanations,
    }
