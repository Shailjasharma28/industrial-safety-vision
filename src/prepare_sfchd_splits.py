from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
dataset = project_root / "data" / "sfchd" / "QY_final_dataset"

for split in ("train", "val"):
    source_list = dataset / "new_split_yolo" / f"{split}.txt"
    local_paths = []

    for line in source_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        image_name = line.strip().replace("\\", "/").split("/")[-1]
        image_path = dataset / "images" / image_name
        label_path = dataset / "labels"/ f"{Path(image_name).stem}.txt"

        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(
                f"Missing image or label for {image_name}"
            )

        local_paths.append(image_path.as_posix())

    output_list = dataset / f"{split}_local.txt"
    output_list.write_text("\n".join(local_paths), encoding="utf-8")
    print(f"{split}: saved {len(local_paths)} image paths to {output_list}")