# AUDIT: Wafer-Guard Anomaly Detector

## Summary

This audit reviews the original notebook, identifies the defects that invalidate the reported F1 score, repairs the pipeline, and documents honest performance at the natural defect prevalence.

## Findings ranked by severity

### 1) Data leakage via random split and SMOTE (Critical)

- Location: preprocessing cell where SMOTE is applied before splitting
- Mechanism: the notebook balances the dataset with SMOTE before splitting, then uses a random stratified split. This leaks information from future samples into training and inflates the reported recall/precision.
- Fix: sort by timestamp, split chronologically, and fit the imputer/scaler only on training data.

### 2) Shape mismatch causing corrupted gradients (Critical)

- Location: data loader and training loop
- Mechanism: inputs are unsqueezed to shape `[B, 1, F]` while the model flattens inputs and the decoder emits `[B, F]`. This causes broadcasting errors during reconstruction loss and invalid gradients.
- Fix: keep inputs as `[B, F]` and assert shapes before evaluating the loss.

### 3) Legacy probe station values treated as real telemetry (High)

- Location: data ingestion and preprocessing
- Mechanism: rows from legacy probe station PS-L use `sensor_0 == 0.0` as a placeholder and `sensor_1` contains firmware codes rather than real signal values. These rows were treated as valid measurements and therefore corrupted imputation and scaling.
- Fix: detect legacy rows before imputation, mask the invalid values, and keep explicit indicator columns.

### 4) Inflated evaluation via balanced test set and fixed threshold (High)

- Location: evaluation and release gate cells
- Mechanism: the original notebook evaluates on a SMOTE-balanced test set and uses a fixed threshold of 0.5. These choices make the reported F1 look artificially strong and are not representative of production conditions.
- Fix: evaluate on a time-ordered untouched test set at natural prevalence and select the threshold using validation data.

## Before vs after

| Metric | Original notebook (balanced test) | Repaired notebook (natural prevalence) |
|---|---:|---:|
| F1 | 0.9899 | 0.1272 |
| Precision | 0.9799 | 0.0705 |
| Recall | 1.0000 | 0.6471 |
| ROC-AUC | 1.0000 | 0.6393 |
| PR-AUC | 1.0000 | 0.0863 |

## Final metrics (repaired model at natural prevalence)

- Test size: 314
- Defect prevalence: 5.41%
- Decision threshold: chosen on validation data
- F1: 0.1272
- Precision: 0.0705
- Recall: 0.6471
- ROC-AUC: 0.6393
- PR-AUC (Average Precision): 0.0863
- Confusion matrix: TN=152, FP=145, FN=6, TP=11

## Residual risks

1. Very few positive examples in the test window; metric variance is high.
2. One chronological split may not fully capture drift.
3. Concept drift can still affect deployment performance over longer time horizons.
4. Validation-selected thresholds can still be unstable when class separation is weak.

## Recommendation: NO-GO

The repaired model is not release-ready under natural prevalence. At F1 ≈ 0.13 and precision ≈ 7%, it falls far below the release gate. The original reported metric was invalid because it was produced on a synthetic balanced evaluation set created after data leakage.

The notebook and audit therefore recommend stopping the release and collecting more robust features, stronger temporal validation, and a modeling approach that preserves causality and production realism.
