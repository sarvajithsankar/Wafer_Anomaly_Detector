# wafer-guard-audit

My submission for the GDG Tech Wildcard Entry (AI/ML track). The task: take a PyTorch notebook that reports F1 >= 0.90 for detecting defective wafers, find out what's actually wrong with it, fix it, and report numbers I can stand behind.

The original notebook runs fine and clears its own release gate, but a notebook running without errors doesn't mean the result is valid. This repo has my audit of it, the repaired version, and the metrics measured at the real defect rate (about 6.6%).

STATUS: <update this once the repair is done>

## What's in here

- `wafer_guard_fixed.ipynb` - the repaired notebook
- `AUDIT.md` - what I found, how I fixed it, and the final numbers
- `wafer_guard_original.ipynb` - the untouched original, so the before/after comparison can be reproduced

I haven't included the dataset (`secom_telemetry_modified.csv`). It comes with the task brief, in the Drive folder linked there.

## The data

Each row is a wafer lot with 590 sensor readings, a timestamp and a pass/fail label (-1 pass, +1 fail). Only about 6.6% of lots fail.

One thing to know: some wafers went through a legacy probe station (PS-L) that logs things differently. On those, `sensor_0` is exactly 0.0 as a placeholder rather than a real reading, and `sensor_1` holds firmware error codes (100, 200, 300) instead of a measurement. The brief says to treat these properly and not scale or fill them in like normal values.

## Setup

You need Python 3.10+ and the CSV. A GPU helps but isn't required, the notebook falls back to CPU.

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install "torch>=2.2" "scikit-learn>=1.2" "imbalanced-learn>=0.11" pandas numpy joblib matplotlib jupyter
```

Put `secom_telemetry_modified.csv` next to the notebook, since it's read with a relative path.

## Running it

In Colab, upload the notebook and the CSV and hit Runtime > Run all. Locally:

```bash
jupyter notebook wafer_guard_fixed.ipynb
```

Run it top to bottom. The metrics print at the end. Hyperparameters are in the `HP` dict near the top (batch size, epochs, learning rate, layer sizes, and so on). The seed is fixed at 1337, though GPU runs can still vary a little.

## Tests

<write what you actually added here, and delete the rest>

I'm planning three simple checks: that no training row comes after a test row in time (and the scaler stats come from training data only), that the model output and target shapes match at the loss, and that the PS-L sentinel and error codes never end up in the scaled features.

To run them: <command or cell name>

## How the model works

It's a supervised autoencoder. An encoder squeezes the 590 sensors down to a small latent code, a decoder tries to rebuild the input from it, and a small head on the same code predicts defect or not. The loss adds reconstruction error (MSE) and classification error (BCE with logits). Training uses AdamW with a cosine schedule.

<if the repair changes the architecture, say how here>

## What I changed and why

<fill this in from the final notebook: how the data is split, what's fit on training data only, how class imbalance is handled, how PS-L is handled, how the threshold was picked and on which split>

The longer version, with evidence, is in [AUDIT.md](AUDIT.md).

## Results

<copy these straight from the notebook output, don't round or estimate>

- Test set size:
- Defect rate in test set:
- Threshold:
- Precision:
- Recall:
- F1:
- PR-AUC:
- ROC-AUC:
- Confusion matrix: TN= FP= FN= TP=

## Limitations

<edit to match your final work>

With only a few percent of lots failing, there just aren't many positive examples, so these numbers carry a lot of uncertainty. I also only used one train/test cut, so a different cut could give different results, and I haven't modelled drift after the training period.

## Author

<your name> - <github or email>
