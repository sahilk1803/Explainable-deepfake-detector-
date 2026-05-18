# Deepfake Video Detection System

This project implements a deepfake video detection pipeline for academic demonstration and research-oriented evaluation. It is designed around a CNN + LSTM architecture using PyTorch, with preprocessing, training, command-line prediction, and two web interfaces.

## Objectives

- Design and develop a deepfake video detection system using PyTorch.
- Extract spatial facial-frame features using a pretrained ResNet18 CNN.
- Capture temporal relationships between consecutive video frames using an LSTM network.
- Preprocess video data through frame extraction, face detection, face cropping, normalization, and tensor conversion.
- Train and evaluate the detector using FaceForensics++ style real and manipulated video folders, with support for smaller dataset experiments.
- Provide command-line, Flask, and Streamlit interaction modes.
- Keep the system easy to demonstrate, understand, and extend for academic purposes.

## Architecture

The report-aligned model is `resnet18_lstm`.

1. A fixed number of frames is sampled from each video.
2. The largest detected face is cropped from each frame when face cropping is enabled.
3. Frames are resized and normalized into PyTorch tensors.
4. A pretrained ResNet18 extracts spatial features from each frame.
5. An LSTM receives the sequence of frame features and learns temporal consistency patterns.
6. A final classifier predicts one of two classes: `real` or `fake`.

The older `mobilenet`, `mobilenet_lstm`, and `small_cnn` options are still available for comparison and fallback demos.

## Important Files

- `train_deepfake_detector.py`: training pipeline, dataset loading, preprocessing, model definitions, checkpoint saving.
- `predict_deepfake.py`: command-line prediction utility for testing a video directly.
- `frontend.py`: Flask-based web interface with upload, prediction, confidence display, and face extraction.
- `app.py`: Streamlit-based alternative interface.
- `templates/index.html`: Flask UI template.
- `models/`: trained checkpoints and training metrics.

## Training the Report-Aligned Model

Use this command to train the ResNet18 + LSTM model:

```powershell
python train_deepfake_detector.py --fake-dir Deepfakes --real-dir original --output-dir models --epochs 8 --batch-size 8 --frames-per-video 8 --image-size 160 --model resnet18_lstm --pretrained --freeze-backbone --max-videos-per-class 500
```

This saves the report-aligned checkpoint as:

```text
models/resnet18_lstm_deepfake_detector_best.pt
```

The Flask app, Streamlit app, and CLI automatically use this checkpoint when it exists. Until it is trained, they fall back to the existing `models/deepfake_detector_best.pt` checkpoint so demonstrations can still run.

## Running the Interfaces

Flask:

```powershell
python frontend.py
```

Open:

```text
http://127.0.0.1:8501/
```

Streamlit:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Command-line prediction:

```powershell
python predict_deepfake.py path\to\video.mp4
```

