# Artifacts

Trained models, the feature schema and the decoder configuration live here at
runtime. The folder is gitignored by default because the forest is large.

## What must exist for the API to start

| File | Produced by | Consumed by |
|---|---|---|
| `model.joblib` | `scripts/run_p3.py final` (kept random forest) | `api/inference.py` |
| `feature_schema.json` | Person 1 / `scripts/run_p2.py` | `api/inference.py` |
| `decoder_config.json` | `scripts/run_p4.py decoder-tune` | `api/inference.py` |
| `eval_trips.txt` | Person 4 (trip ids served by `/trips`) | `api/main.py` |

`GET /health` reports the model name and the SHA-256 of each of these four files,
so a deployment can be checked against the record without opening the model.

There is no `preprocessor.joblib`. Every model that can be the final model is
self-contained: logistic regression, KNN and SVM keep their scaler inside a
scikit-learn `Pipeline`, and the random forest needs no scaling. This is recorded
as amendment A2 in `docs/CONTRACTS.md`.

## Optional artifacts kept for reproducibility

| File | Produced by |
|---|---|
| `p2_knn.joblib`, `p2_svm_rbf.joblib` | `scripts/run_p2.py` (experiment artifacts; the API does not load them) |
| `manifest.json` | `scripts/run_p4.py manifest` (git commit, library versions, checksums) |

## Sizes and submission

The kept random forest is about 665 MB. That is too large for a normal GitHub
commit and is why `artifacts/**` is ignored. The repository therefore carries the
code, the result records and the instructions to rebuild it; the weights travel
separately (zipped drive link or in-class handover), together with
`manifest.json`, which pins the git commit and the checksums.

The smaller P2 models remain useful for demos on a laptop with limited RAM. To
point the API at one instead, copy it to `artifacts/model.joblib` and restart; the
feature schema and decoder config are unchanged.

