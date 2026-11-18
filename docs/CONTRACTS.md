# Interface contracts

Four people building in parallel only works if the hand-off points are frozen.
This document is those hand-offs. If you need to change one, say so in the group
chat **before** editing, because three other people are coding against it.

---

## C1 — Raw data on disk

```
data/raw/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

| File | Columns | Index meaning |
|---|---|---|
| `.track` | `lon lat timestamp` | row = observation index |
| `.nodes` | `lon lat` | **row index = node_id** |
| `.arcs` | `from_node_id to_node_id` | **row index = arc_id** |
| `.route` | `arc_id` | chronological ground truth |

All whitespace-separated, no headers. Implemented in `io/loaders.py`. **Frozen.**

---

## C2 — Candidate set ↔ classifier probabilities  *(the one that breaks silently)*

`candidates.Generator` produces a `CandidateSet` with three parallel arrays of
equal length `n_pairs`, **sorted so that observation 0's candidates come first**:

```python
obs_idx : int64[n_pairs]   # index into the trip's Track rows
arc_id  : int64[n_pairs]   # index into the trip's RoadNetwork.arcs
dist_m  : float64[n_pairs] # observation-to-arc distance
```

The classifier returns:

```python
proba : float64[n_pairs, n_classes]   # row i scores candidates[i]
```

**Row `i` of `proba` must describe pair `i` of the CandidateSet.** There is no
lookup table, no reordering, no groupby. If a model sorts its input, filters
rows, or returns probabilities for a deduplicated candidate list, the Viterbi
stage will attach probabilities to the wrong arcs and produce confident,
completely wrong routes with no error message.

Rule: whoever builds `X` from the `CandidateSet` keeps the row order identical to
the `CandidateSet`, and `meta` carries `obs_idx`/`arc_id` through unchanged so
predictions can be scattered back.

---

## C3 — Feature matrix

```python
X, y, meta = build_feature_matrix(trip, candidates)
```

| Returns | Type | Shape / keys |
|---|---|---|
| `X` | `float64` | `(n_pairs, n_features)`, no NaN |
| `y` | `int64` or `None` | `(n_pairs,)` arc ids; `None` when the trip has no `.route` |
| `meta` | `dict` | `{"trip_id", "obs_idx", "arc_id", "columns", ...}` |

- `meta["columns"]` is committed to `artifacts/feature_schema.json`.
- Column order is part of the contract: the API loads a model trained on that
  exact order. Renaming or reordering a column silently invalidates every saved
  model.
- Additive changes only (append new columns at the end), and bump a `schema_version`.

---

## C4 — Model interface

Every model in `models/registry.ALL_MODELS` implements:

```python
def fit(self, X, y, groups=None) -> self
def predict_proba(self, X) -> float64[n_samples, n_classes]   # rows sum to 1
def predict(self, X) -> int64[n_samples]                      # == classes_[argmax]
def describe(self) -> dict                                    # JSON-serialisable config
classes_ : np.ndarray                                         # set during fit
```

Validate yours with `pathsnap.models.base.check_classifier(model, X)` inside a
test. `describe()` feeds the report's configuration column — it is not optional
decoration.

---

## C5 — Result records

One file per model: `experiments/results/model_<name>.json`, a serialised
`models.base.ModelRecord`. Written by `evaluation/compare.save_record`,
read by `build_comparison_table`. Never type metrics into the report by hand.

Required populated fields: `name`, `family`, `config`, `seed`, `cv_macro_f1`,
`test_top1_accuracy`, `test_macro_f1`, `test_log_loss` (or `None` → "N/A"),
`route_continuity`, `inference_ms_per_obs`, `artifact_path`.

---

## C6 — Decoder input

```python
viterbi_decode(trip_id, candidates, proba, network, n_obs, cfg) -> DecodeResult
```

- `n_obs` is the number of GPS observations in the trip, so the output length
  matches the GPS timeline even when some observations have no candidates.
- Output `arc_per_obs` has shape `(n_obs,)`; use `-1` where no arc could be chosen.
- `DecodeResult.log_likelihood` is the Viterbi path score in log space.

---

## C7 — API responses

`GET /health` → `{"status": "ok", "model": str, "artifacts_loaded": bool}`

`GET /trips` → `{"trips": [trip_id, ...]}`

`POST /predict/{trip_id}` →
```json
{
  "trip_id": "00000001",
  "arc_per_obs": [12, 12, 15],
  "route_geometry": [[8.59, 49.84], [8.60, 49.85]],
  "n_observations": 3,
  "decode_mode": "hmm",
  "metrics": {"top1_accuracy": 0.91, "macro_f1": 0.44, "route_continuity": 0.98}
}
```

`metrics` contains **aggregate numbers only**. Never return reference arc ids —
the UI must not be able to reconstruct the answer.

---

## C8 — Artifacts on disk

| File | Owner | Format |
|---|---|---|
| `artifacts/model.joblib` | P3 (final tuned) | joblib dump of the model object |
| `artifacts/preprocessor.joblib` | P1 | joblib dump (scaler / encoders) |
| `artifacts/feature_schema.json` | P1 | `{"version": 1, "columns": [...]}` |
| `artifacts/decoder_config.json` | P4 | serialised `DecoderConfig` |
| `artifacts/eval_trips.txt` | P4 | one trip id per line |

The API refuses to start if any of these is missing. Fail fast, not mid-demo.
