# Final Mini Project

Slim runnable copy of the deepfake detection project. Dataset folders are intentionally excluded.

## Run Flask app

```powershell
python frontend.py
```

Open http://127.0.0.1:8501/

## Run Streamlit app

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

## CLI prediction

```powershell
python predict_deepfake.py path\to\video.mp4
```
