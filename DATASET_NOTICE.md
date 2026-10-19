# Dataset removal notice

The file `secom_telemetry_modified.csv` has been removed from this repository to keep the project lean and avoid committing a large, protected dataset directly to GitHub.

## How to run the notebooks

Download the dataset from the task brief / provided Drive link and place it in the repository root before running:

```bash
jupyter notebook wafer_guard_fixed.ipynb
```

## Files intentionally kept

- `secom_legacy_mask_EVALUATOR_ONLY.csv` — small evaluator-only file used for legacy probe-station checks

## Note

If your environment has a shared data store (internal artifact bucket, S3, or similar), you may prefer to point the notebook to that storage location instead of keeping the raw data in Git.
