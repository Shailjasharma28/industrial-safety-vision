from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

MODEL_PATH = Path(os.getenv("PPE_MODEL_PATH", "runs/detect/ppe-baseline/weights/best.pt"))
_model: YOLO | None = None


def detect(image):
    global _model
    if image is None:
        return None
    if not MODEL_PATH.is_file():
        raise gr.Error(f"Trained weights not found at {MODEL_PATH}. Run src/train.py first or set PPE_MODEL_PATH.")
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    result = _model.predict(image, imgsz=640, conf=0.25, verbose=False)[0]
    return result.plot()[:, :, ::-1]


demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy", label="Upload a workplace image"),
    outputs=gr.Image(type="numpy", label="Detected worker and PPE classes"),
    title="Industrial Safety Vision",
    description="YOLO baseline for visible helmet, vest, worker, and missing-PPE labels. Screening demo only; not safety-rated.",
)

if __name__ == "__main__":
    demo.launch()
