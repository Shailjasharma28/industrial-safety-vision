from ultralytics import YOLO

MODEL_PATH = "runs/detect/ppe-baseline/weights/best.pt"

def main():
    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data = "data/data.yaml",
        split = "test",
        imgsz=640,
        device=0,
    )

    print(metrics.results_dict)


if __name__ == "__main__":
    main()