from pathlib import Path

import cv2

project_root = Path(__file__).resolve().parents[1]
dataset = project_root / "data" / "sfchd" / "QY_final_dataset"

image_name = "192.168.11.13_neg_1c_0.jpg"
image_path = dataset / "images" / image_name
label_path = dataset / "labels" / f"{Path(image_name).stem}.txt"

class_names = (dataset/ "classes.txt").read_text(encoding="utf-8").splitlines()

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"Could not open image: {image_path}"
    )

height, width = image.shape[:2]

for row in label_path.read_text(encoding="utf-8").splitlines():
    class_id, x_center, y_center, box_width, box_height = map(float, row.split())
    class_id = int(class_id)

    x1 = int((x_center - box_width / 2) * width)
    y1 = int((y_center - box_height / 2) *height)
    x2 = int((x_center + box_width / 2)* width)
    y2 = int((y_center + box_height / 2) * height)


    cv2.rectangle(image, (x1,y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        class_names[class_id],
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (0, 255,0),
        2,
    )

    cv2.imshow("SFCHD labels", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()