# Industrial Safety Vision

A learning project for detecting people and visible personal protective equipment (PPE) in images. It contains two YOLO11n experiments and local Gradio demos: a construction-image baseline and a second model trained with the SFCHD industrial-camera dataset.

This is a portfolio and screening demo. It is not safety-rated and must not be used to make workplace compliance decisions.

## Experiments

### 1. Construction Safety baseline

- Dataset: Roboflow 100 Construction Safety, version 2 (`release`), with `helmet`, `no-helmet`, `no-vest`, `person`, and `vest` classes.
- Training: YOLO11n, 30 epochs, 640-pixel images, batch size 4.
- Test split: 90 images and 623 labeled instances.

| Precision | Recall | mAP50 | mAP50–95 |
| ---: | ---: | ---: | ---: |
| 0.818 | 0.743 | 0.798 | 0.426 |

Test results are from the dataset's supplied test split. Per-class results and training artifacts are available locally in the Ultralytics run output; model weights and run files are not included in this repository.

### 2. SFCHD industrial-camera experiment

- Dataset: SFCHD, collected from cameras at two chemical plants. The source describes seven classes: `person`, `helmet`, `self_clothes`, `safety_clothes`, `head`, `blur_head`, and `blur_clothes`.
- Training: YOLO11n, 30 epochs, 640-pixel images, batch size 4.
- Usable split lists in the downloaded archive: 9,897 training images and 2,475 validation images.
- The downloaded archive contained 12,372 image files, while the paper reports 12,373. Its `test.json` was empty; the six entries in `test.txt` all overlap the validation list. There is no independent test result for this experiment.

| Validation set | Precision | Recall | mAP50 | mAP50–95 |
| ---: | ---: | ---: | ---: | ---: |
| 2,475 images, 10,173 instances | 0.791 | 0.706 | 0.744 | 0.476 |

Per-class validation metrics:

| Class | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| person | 0.935 | 0.936 | 0.961 | 0.703 |
| helmet | 0.899 | 0.884 | 0.917 | 0.594 |
| self_clothes | 0.700 | 0.753 | 0.794 | 0.552 |
| safety_clothes | 0.896 | 0.921 | 0.946 | 0.623 |
| head | 0.783 | 0.698 | 0.709 | 0.382 |
| blur_head* | 0.657 | 0.359 | 0.405 | 0.201 |
| blur_clothes | 0.668 | 0.393 | 0.475 | 0.278 |

\* Class ID 5 is `blur_head` in the dataset's `classes.txt`. The training YAML used the misspelled name `blue_head`, so the original metrics output used that name. The numeric metrics refer to the same class ID; the class-name typo does not change the learned class ID.

Ten validation images were also reviewed visually at confidence 0.25. Most inspected detections matched the labels; the review found a possible extra-detection case and an apparent error involving the blurred classes. This small sample is qualitative and does not replace an independent test set.

The demo currently runs with a fixed confidence threshold of 0.10. In a visual comparison, lowering the threshold from 0.25 to 0.10 revealed additional low-confidence boxes, including overlapping detections. This illustrates the precision/recall trade-off; it does not improve the trained model by itself. The displayed scores are model confidence scores, not calibrated probabilities.

The two experiments use different datasets, class definitions, and evaluation splits. Their metrics should not be treated as a direct head-to-head model comparison.

## Data sources and licensing

- [Roboflow 100 Construction Safety v2](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/2): the project page lists a CC BY 4.0 license. Attribute the dataset source when sharing results. Do not commit the downloaded images or labels.
- [SFCHD paper](https://arxiv.org/abs/2306.02098) and [author repository](https://github.com/lijfrank/SFCHD-SCALE): the paper says the dataset is publicly available, but we could not identify a dataset reuse license in the repository. This repository excludes SFCHD images, labels, split files, and trained weights. Confirm the dataset and model terms with the authors before redistributing those materials or hosting a public demo based on them.
- Ultralytics YOLO is distributed under AGPL-3.0 or an Enterprise license. Review the [Ultralytics license terms](https://www.ultralytics.com/license) before publishing or deploying the complete application. Dataset and software/model terms are separate.

Downloaded datasets (including local YAML files and absolute-path split manifests), model weights, and run outputs are intentionally excluded from this repository. The code in this repository does not yet have its own license; until one is added, GitHub visibility does not grant permission to reuse it.

## Setup

Windows PowerShell, from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements pin PyTorch 2.11.0 with CUDA 12.8 wheels for the development machine used in this project. An NVIDIA driver is required for CUDA. See the [official PyTorch installation guide](https://pytorch.org/get-started/previous-versions/) for other configurations.

## Train

### Construction Safety baseline

Download the YOLOv11 export and extract `train/`, `valid/`, and `test/` under `data/`, keeping the exported `data.yaml` and attribution notes.

```powershell
python src/train.py --data data/data.yaml --epochs 30 --imgsz 640 --batch 4
```

### SFCHD experiment

Download the dataset from the author repository and extract it under `data/sfchd/QY_final_dataset/`. Run the split-preparation script to create local train/validation path lists:

```powershell
python src/prepare_sfchd_splits.py
```

Create `data/sfchd/QY_final_dataset/sfchd.yaml` with the local dataset path, `train_local.txt`, `val_local.txt`, and the seven class names listed above. Then train:

```powershell
python src/train_sfchd.py
```

Training on a 4 GB laptop GPU can take several hours. If CUDA runs out of memory, lower the batch size to 2.

## Run the local demos

The demos require the corresponding trained weights, which are not included here.

```powershell
python app.py
```

`app.py` runs the Construction Safety baseline demo. The SFCHD demo is:

```powershell
python app_sfchd.py
```

Upload one image to see object detections and confidence scores. The SFCHD model detects its seven listed classes; it does not have explicit `no-helmet` or `no-vest` classes. The SFCHD demo checks `runs/detect/sfchd-baseline/weights/best.pt` first, then the existing local checkpoint at `runs/detect/sfchd_1/weights/best.pt`. The construction demo expects `runs/detect/runs/detect/ppe-baseline/weights/best.pt`. Set `PPE_MODEL_PATH` to override the model path.

## Project files

- `src/train.py` — Construction Safety baseline training and validation.
- `src/train_sfchd.py` — SFCHD training and validation.
- `src/prepare_sfchd_splits.py` — creates local image path lists from the archive's split files.
- `src/review_errors.py` and `src/review_sfchd.py` — save sampled images with labels and predictions for visual review.
- `src/evaluate.py` — baseline test-split evaluation.
- `app.py` and `app_sfchd.py` — local Gradio demos.

## Limitations

- SFCHD has no usable independent test set in the downloaded archive; its reported score is validation performance.
- The SFCHD model showed weaker recall on `head`, `blur_head`, and `blur_clothes` than on `person`, `helmet`, and `safety_clothes`.
- A confidence threshold changes which detections are displayed. Lowering it can reveal more candidates but also adds false or duplicate detections.
- These public images and labels do not establish performance on a particular worksite or camera system. A real safety deployment would require authorized site-specific data, careful annotation, independent evaluation, and qualified human oversight.
