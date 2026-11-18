# Artifacts

Trained models, the fitted preprocessor, the feature schema and the decoder
configuration live here at runtime. The folder is gitignored by default.

## What must exist for the API to start

| File | Produced by | Consumed by |
|---|---|---|
| `model.joblib` | Person 3 (final tuned model) | `api/inference.py` |
| `preprocessor.joblib` | Person 1 | `api/inference.py` |
| `feature_schema.json` | Person 1 | `api/inference.py` |
| `decoder_config.json` | Person 4 | `api/inference.py` |
| `eval_trips.txt` | Person 4 | `api/main.py` (`/trips`) |

## Submission note

Deliverables asks for "the saved model weights and required configuration files,
subject to repository size constraints". Before you submit, force-add the small
metadata files and the model if it fits comfortably:

```powershell
git add -f artifacts/model.joblib artifacts/preprocessor.joblib `
            artifacts/feature_schema.json artifacts/decoder_config.json
```

Check the size first. If `model.joblib` is large (an RBF SVM with Platt scaling
can be tens of MB), prefer the smaller of your candidate models for deployment
and say so in the report — it is a legitimate engineering decision, not a
shortcut.
