from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


project_root = Path(__file__).resolve().parents[1]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SFCHD PPE detection baseline.")
    parser.add_argument("--data", default="data/sfchd/QY_final_dataset/sfchd.yaml", 
                        help="Path to the SFCHD dataset YAML")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()

    if not data_path.is_file():
        raise SystemExit(
            f"Dataset YAML not found: {data_path}"
             
        )
    run_dir = project_root / "runs" / "detect" / "sfchd-baseline"


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
        project=str(run_dir.parent),
        name= run_dir.name,
        exist_ok=False,
        patience=8,
        amp=True,
        plots=True,
        seed=42,
    )

    best = run_dir / "weights" / "best.pt"
    if best.is_file():
        print("\nValidation with best checkpoint:")
        metrics = YOLO(str(best)).val(data=str(data_path), imgsz=args.imgsz, device=device)
        print(metrics.results_dict)
        print(f"Best checkpoint: {best.resolve()}")


if __name__ == "__main__":
    main()
