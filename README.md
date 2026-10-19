# Wafer_Anomaly_Detector

[![Open Fixed Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarvajithsankar/Wafer_Anomaly_Detector/blob/main/wafer_guard_fixed.ipynb)
[![Open Original Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarvajithsankar/Wafer_Anomaly_Detector/blob/main/bnypi.ipynb)

This repository contains the audit and repaired notebook for the wafer-defect detection task.

## Overview

The goal of this project was to review an original PyTorch notebook that reported a release-gate result of F1 >= 0.90, identify the defects that invalidate the reported metric, repair the pipeline, and document honest evaluation at the natural defect prevalence.

## Included files

- `wafer_guard_fixed.ipynb` — repaired, end-to-end notebook used for the corrected evaluation
- `bnypi.ipynb` — untouched original notebook (reference copy)
- `AUDIT.md` — ranked findings, fix descriptions, final numbers, and recommendation
- `README.md`
- `.gitignore`

## Running in Colab

1. Click the badge above.
2. Upload `secom_telemetry_modified.csv` (from the task brief's Drive folder) to the Colab session's main folder, using the folder icon on the left.
3. Runtime > Run all.
4. Note that the CSV is not in this repo.

## Dataset note

The dataset `secom_telemetry_modified.csv` is not in this repository; it comes from the task brief's Drive folder.

## Requirements

- Python 3.10+
- Optional: GPU for faster training; CPU is supported
- Packages:

```bash
pip install "torch>=2.2" "scikit-learn>=1.2" "imbalanced-learn>=0.11" pandas numpy joblib matplotlib jupyter
```

## Quick start

```bash
git clone https://github.com/sarvajithsankar/Wafer_Anomaly_Detector.git
cd Wafer_Anomaly_Detector
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install "torch>=2.2" "scikit-learn>=1.2" "imbalanced-learn>=0.11" pandas numpy joblib matplotlib jupyter
```

Then place `secom_telemetry_modified.csv` next to the notebooks and run:

```bash
jupyter notebook wafer_guard_fixed.ipynb
```

## Sanity checks included in the repaired pipeline

The repaired notebook includes three key checks:

1. Chronological split validation
   - training samples must be earlier than validation/test samples
   - scaler/imputer are fit only on training data

2. Shape assertions before loss computation
   - verifies `reconstructed.shape == batch_x.shape`
   - verifies `logits.shape == batch_y.shape`

3. Legacy PS-L mask handling
   - identifies legacy probe-station rows where `sensor_0 == 0.0` and firmware codes appear in `sensor_1`
   - masks these values before imputation and preserves signal metadata with indicator columns

## Evaluation approach

The repaired notebook uses a time-ordered split and evaluates at the natural defect prevalence instead of on a synthetic balanced test distribution. The final numbers and recommendation are recorded in `AUDIT.md`.

## Contact

sarvajith sankar — sarvajith2knot8@gmail.com
