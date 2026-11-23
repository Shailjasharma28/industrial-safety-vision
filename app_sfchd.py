from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv("PPE_MODEL_PATH", "runs/detect/sfchd_1/weights/best.pt")
)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (PROJECT_ROOT / MODEL_PATH).resolve()


FALLBACK_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "sfchd_1" / "weights" / "best.pt"
if not MODEL_PATH.is_file() and FALLBACK_MODEL_PATH.is_file():
    MODEL_PATH = FALLBACK_MODEL_PATH
_model: YOLO | None = None


def detect(image):
    global _model
    if image is None:
        return None
    if not MODEL_PATH.is_file():
        raise gr.Error(f"Trained weights not found at {MODEL_PATH}. Run src/train_sfchd.py first or set PPE_MODEL_PATH.")
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    result = _model.predict(image, imgsz=640, conf=0.10, verbose=False)[0]
    return result.plot()[:, :, ::-1]


demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy", label="Upload a workplace image"),
    outputs=gr.Image(type="numpy", label="Detected worker and PPE classes"),
    title="Industrial Safety Vision",
    description=(
        "YOLO11n trained on SFCHD for person, helmet, clothing, and head-related classes. "
        "It does not explicitly detect no-helmet or no-vest. Screening demo only; not safety-rated."
    ),
)

if __name__ == "__main__":
    demo.launch()
