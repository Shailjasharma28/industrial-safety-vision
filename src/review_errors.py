from pathlib import Path
import csv
import random

import cv2
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT 
    / "runs"
    / "detect"
    / "ppe-baseline"
    / "weights"
    / "best.pt"
)

DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "validation_review"

NUMBER_IMAGES = 10
RANDOM_SEED= 42
CONFIDENCE = 0.25

def get_class_names(data_yaml_path):
    with data_yaml_path.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)


    names = data_config["names"]

    # Names are in a dict
    if isinstance(names, dict):
        return{int(class_id): name for class_id, name in names.items()}

    # names are in a list
    return {class_id: name for class_id, name in enumerate(names)}


def draw_ground_truth(image, label_path, class_names):
    image_with_boxes = image.copy()

    if not label_path.exists():
        return image_with_boxes

    image_height, image_width = image.shape[:2]

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.split()

            # YOLO labels are in format (class, center_x, center_y, width, height)

            if len(values) != 5:
                continue

            class_id = int(values[0])
            center_x, center_y, box_width, box_height = map(float, values[1:])

            x1 = int((center_x - box_width / 2) * image_width)
            y1 = int((center_y - box_height / 2) * image_height)
            x2 = int((center_x + box_width / 2) * image_width)
            y2 = int((center_y + box_height / 2) * image_height)

            label = class_names.get(class_id, str(class_id))

            cv2.rectangle(image_with_boxes, (x1,y1), (x2,y2), (0,180,0), 2)
            cv2.putText(
                image_with_boxes,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 180,0),
                2,

            )

    return image_with_boxes


def draw_predictions(image, prediction, class_names):
    # Draw the model's predicted boxes 
    image_with_boxes = image.copy()
    boxes = prediction.boxes

    if boxes is None:
        return image_with_boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        class_name = class_names.get(class_id, str(class_id))
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image_with_boxes,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return image_with_boxes


def main():
    images_folder = PROJECT_ROOT / "data"/ "valid"/ "images"
    labels_folder = PROJECT_ROOT / "data"/ "valid"/ "labels"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}"
        )

    if not images_folder.exists():
        raise FileNotFoundError(
            f"Validation images folder not found: {images_folder}"
        )

    class_names = get_class_names(DATA_YAML)
    image_paths = sorted(
        path for path in images_folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if len(image_paths) < NUMBER_IMAGES:
        raise ValueError(
            f"Found only: {len(image_paths)} validation images."
            f"Nee at least {NUMBER_IMAGES}"
        )


    # RANDOM SEED
    rng = random.Random(RANDOM_SEED)
    selected_images = sorted(rng.sample(image_paths, NUMBER_IMAGES))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)           

    model = YOLO(str(MODEL_PATH))
    notes_path = OUTPUT_DIR / "review_notes.csv"

    with notes_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "confidence_threshold", "your_notes"])

        for image_path in selected_images:
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Could not open image: {image_path}")
                continue

            label_path = labels_folder / f"{image_path.stem}.txt"

            result = model.predict(
                source=image,
                imgsz=640,
                conf=CONFIDENCE,
                verbose=False,
            )[0]

            ground_truth_image = draw_ground_truth(
                image, label_path, class_names
            )

            prediction_image = draw_predictions(
                image, result, class_names
            )

            cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_original.jpg"), image)
            cv2.imwrite(
                str(OUTPUT_DIR / f"{image_path.stem}_ground_truth.jpg"),
                ground_truth_image
            )

            cv2.imwrite(
                str(OUTPUT_DIR / f"{image_path.stem}_predictions.jpg"),
                prediction_image
            )

            writer.writerow([image_path.name, CONFIDENCE, ""])

            print(f"Saved review images for {image_path.name}")

    print(f"\nDone. Open this folder to review the images:\n{OUTPUT_DIR}")
    print(f"Write your observations in:\n{notes_path}")

if __name__=="__main__":
    main()
