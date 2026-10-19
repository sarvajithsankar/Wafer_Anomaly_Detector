# AUDIT: Wafer-Guard Anomaly Detector

## Findings (ranked by severity)

### 1. Data Leakage via Random Split and SMOTE (Critical)
**Location:** Cell 8 (Preprocessing & class rebalancing)
**Mechanism:** The notebook uses `train_test_split` with `stratify=y_bal` on SMOTE-balanced data, ignoring the temporal `timestamp` column. This causes future data to leak into training. Additionally, `SimpleImputer` and `StandardScaler` are fit on the entire dataset before splitting, not just training data. SMOTE artificially balances the test set to 50% defect rate, far from the natural 6.6%.
**Evidence:**
- Original defect rate after SMOTE: 50.00% (from run_original.py)
- Timestamp column exists but is not used for splitting
- Test set after SMOTE: 586 samples with 50% defect rate (balanced)
- Test prevalence (fixed, chronological): 5.41% (natural)
**Fix:** Parse timestamp, sort chronologically, split 60%/20%/20% in time order. Fit imputer/scaler on train only. Remove SMOTE; use `BCEWithLogitsLoss(pos_weight=neg/pos)` computed on train only.

### 2. Corrupted Gradients due to Shape Mismatch (Critical)
**Location:** Cell 14 (Training loop, line 24)
**Mechanism:** The model receives inputs with shape `[B, 1, F]` (unsqueeze added in Cell 10) but the decoder outputs `[B, F]`. The reconstruction loss computes `(reconstructed - batch_x)` which broadcasts to `[B, B, F]` instead of element-wise `[B, F]`, corrupting gradients.
**Evidence:**
- batch_x.shape: torch.Size([64, 1, 590])
- reconstructed.shape: torch.Size([64, 590])
- (reconstructed - batch_x).shape: torch.Size([64, 64, 590]) ← WRONG DIMENSION
**Fix:** Remove `unsqueeze(1)` from data loader so inputs stay `[B, F]`. Add assertions before loss to verify `reconstructed.shape == batch_x.shape` and `logits.shape == batch_y.shape`.

### 3. Invariant Blindness to Legacy Probe Station PS-L (High)
**Location:** Cell 6 (Data loading)
**Mechanism:** 146 rows (9.32%) from legacy probe station PS-L have `sensor_0 == 0.0` (placeholder) and `sensor_1 in {100, 200, 300}` (firmware error codes, not measurements). These are treated as normal sensor values, polluted into imputation statistics, and scaled, destroying the invariant that should distinguish PS-L from normal stations.
**Evidence:**
- sensor_0 == 0.0 rows: 146
- legacy rows (sensor_0==0 & sensor_1 in {100,200,300}): 146
- sensor_1 value counts: 100.00 (61), 300.00 (50), 200.00 (35)
**Fix:** Set sensor_0 and sensor_1 to NaN for legacy rows BEFORE imputing. After scaling, set them to 0.0 (masked, not fabricated). Append 4 one-hot columns: `is_legacy`, `fw_code_100`, `fw_code_200`, `fw_code_300` (not scaled).

### 4. Inflated Evaluation via Balanced Test Set and Fixed Threshold (High)
**Location:** Cell 16 (Evaluation) and Cell 18 (Release gate)
**Mechanism:** Evaluation uses SMOTE-balanced test set (50% defects) with a fixed 0.5 threshold. This inflates metrics. The release gate passes if F1 ≥ 0.90, which the original notebook achieves (0.9899) on artificial data.
**Evidence:**
- Original F1: 0.9899 (on 50% balanced test)
- Original precision: 0.9799
- Original recall: 1.0000
- Original confusion: TN=287, FP=6, FN=0, TP=293
- Fixed F1: 0.1272 (on 5.41% natural test)
**Fix:** Choose threshold on validation set (maximize F1 on PR curve). Evaluate once on untouched test set at natural prevalence. Remove PASSED logic; print measured F1 only.

## Before vs After

| Metric | Original (Balanced) | Repaired (Natural) |
|--------|-------------------|---------------------|
| F1     | 0.9899            | 0.1272              |
| Precision | 0.9799          | 0.0705              |
| Recall  | 1.0000            | 0.6471              |
| Test Prevalence | 50.00%      | 5.41%               |
| ROC-AUC | 1.0000            | 0.6393              |
| PR-AUC  | 1.0000            | 0.0863              |

## Final Metrics (Repaired Model at Natural Prevalence)

- Test size: 314
- Test defect prevalence: 5.41%
- Decision threshold: 0.0000 (chosen on validation set)
- F1: 0.1272
- Precision: 0.0705
- Recall: 0.6471
- ROC-AUC: 0.6393
- PR-AUC (Average Precision): 0.0863
- Confusion matrix: TN=152, FP=145, FN=6, TP=11

## Residual Risks

1. **Few positives:** Only 17 defects in test set (5.41% of 314). Metrics have high variance.
2. **Single split:** No cross-validation or multiple time splits. Performance may vary across different time windows.
3. **Concept drift:** Prevalence shifts over time (train 8.09%, val 3.51%, test 5.41%). Model may not generalize to future periods.
4. **Threshold at boundary:** Best validation threshold is 0.0000, suggesting the model struggles to distinguish classes.

## Recommendation: NO-GO

The repaired model achieves F1 = 0.1272 at natural prevalence, far below the release gate of 0.90. Precision is 0.0705 (7% of predictions are correct when predicting defect). While recall is reasonable (64.7%), the high false positive rate (145 FP vs 11 TP) would cause excessive line stops in production. The ROC-AUC of 0.6393 indicates only marginally better than random guessing. Do not deploy. The original metrics were inflated by data leakage, corrupted gradients, invariant blindness, and artificial test balancing.
