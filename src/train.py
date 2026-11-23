from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate a small PPE detection baseline.")
    parser.add_argument("--data", default="data/data.yaml", help="Path to the exported dataset YAML")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise SystemExit(
            f"Dataset YAML not found: {data_path}\n"
            "Download the Construction Safety v2 YOLOv11 export and extract train/, valid/, and test/ under data/."
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project="runs/detect",
        name="ppe-baseline",
        exist_ok=True,
        patience=8,
        amp=True,
        plots=True,
        seed=42,
    )

    best = Path("runs/detect/ppe-baseline/weights/best.pt")
    if best.is_file():
        print("\nValidation with best checkpoint:")
        metrics = YOLO(str(best)).val(data=str(data_path), imgsz=args.imgsz, device=device)
        print(metrics.results_dict)
        print(f"Best checkpoint: {best.resolve()}")


if __name__ == "__main__":
    main()
