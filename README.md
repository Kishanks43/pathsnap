# PathSnap

Distributed map-matching and route reconstruction — UE24CS352A Machine Learning hackathon.

Given noisy GPS observations and the road network they were recorded in, predict
which road segment the vehicle was on at each observation, then reconstruct a
continuous, drivable route through the network.

```
raw GPS + road network  ->  candidate arcs  ->  features  ->  classifier  ->  HMM/Viterbi  ->  route
                                                                                |
                                                                    FastAPI  <--+-->  UI (other laptop)
```

## Team

Four members, four workstreams. **Read [`docs/TEAM_PLAN.md`](docs/TEAM_PLAN.md) first** — it
assigns every file in this repository to exactly one person.

| Person | Workstream | Model |
|---|---|---|
| P1 | Data, candidates, features, labels, EDA | **Model 1** — Logistic Regression |
| P2 | Eval harness, comparison table | **Model 2** — KNN, **Model 5** — RBF SVM |
| P3 | Bayesian optimisation | **Model 3** — Random Forest |
| P4 | HMM/Viterbi, FastAPI, UI *(later)* | **Model 4** — HistGradientBoosting |

## Repository layout

```
pathsnap/
├── src/pathsnap/
│   ├── config.py             paths and constants
│   ├── io/                   raw file readers + typed schema (DONE - shared contract)
│   ├── geo/                  geodesy, projection, point-to-segment (DONE - shared contract)
│   ├── candidates/           candidate generation            [P1 - DONE]
│   ├── features/             features + cached matrices      [P1 - DONE]
│   ├── labels.py             reference-arc labels + trust    [P1 - DONE]
│   ├── models/               five classifiers + registry     [P1, P2, P3, P4]
│   ├── evaluation/           metrics (DONE), CV, comparison  [P2]
│   ├── tuning/               Bayesian hyperparameter search  [P3]
│   ├── hmm/                  Viterbi route reconstruction    [P4]
│   ├── viz/                  Folium maps                     [P1, P4]
│   └── api/                  FastAPI backend + inference      [P4]
├── ui/                       evaluator-facing UI             [P4]
├── scripts/                  check_data.py and run entry points
├── experiments/              configs, logs, results (RESULTS ARE COMMITTED)
├── artifacts/                model, preprocessor, schema (gitignored until submission)
├── reports/maps/             Folium HTML + screenshots       [P1, P4]
├── docs/                     TEAM_PLAN, CONTRACTS, AGENT_BRIEFS, SETUP, EXPERIMENT_LOG
└── tests/                    unit tests (run these before every push)
```

Files marked `DONE` are shared infrastructure. Do not rewrite them; if a change is
genuinely needed, announce it in the group chat first because three other people
depend on the exact behaviour.

## Quick start

```powershell
# 1. clone, then install
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. put the data in place (see data/README.md)
python scripts/check_data.py

# 3. verify the scaffold works
pytest -q

# 4. run the API
uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000

# 5. reproduce P2's models and the comparison table
python scripts/run_p2.py                 # KNN + RBF SVM (~20 min on a 16-core laptop)
python scripts/run_p2.py --skip-svm      # KNN only
python scripts/run_p2.py --skip-knn      # SVM only
python scripts/run_p2.py --table-only    # rebuild comparison.csv/.md from records

# 5b. Person 4's HistGradientBoosting (the fifth model, ~2 min)
python scripts/run_model4.py

# 6. reproduce P3 (random forest + Bayesian tuning) - about 1.5 h on a 16-core laptop
python scripts/run_p3.py untuned         # default forest: group CV, student test, importances
python scripts/run_p3.py tune --model random_forest --n-trials 25
python scripts/run_p3.py final --model random_forest   # full-data tuned vs untuned, writes model.joblib

# 7. reproduce P4 (decoder, student-test HMM vs classifier, manifest)
python scripts/run_p4.py decoder-tune    # decoder settings chosen on 10 held-out TRAIN trips
python scripts/run_p4.py student-test    # 10 student trips, HMM vs classifier-only, maps
python scripts/run_p4.py manifest        # artifacts/manifest.json with checksums

# 8. serve it (Laptop 1) and open the UI (Laptop 2)
.\scripts\run_api.ps1                    # http://<laptop-1>:8000/docs
.\scripts\run_ui.ps1 -Api http://<laptop-1>:8000

# 9. checks
python -m pytest -q
```

The two slow models are deliberately subsampled (KNN 100k, SVM 30k training
rows per fold; validation folds are always scored in full). The caps are printed,
written into `experiments/results/model_*.json` and explained in the module
docstring and `docs/EXPERIMENT_LOG.md`; nothing is silently truncated.

Detailed per-person setup, git identity and branch commands:
[`docs/SETUP.md`](docs/SETUP.md).

Measured dataset facts (including two that change the design): [`docs/DATA_FACTS.md`](docs/DATA_FACTS.md).

## Rules we must not break

These come straight from the brief and are the ones teams actually lose marks on:

1. **No leakage.** The 10 evaluator-testing trips are not in this repository. Ground
   truth never enters the prediction path, never reaches the UI, and never informs a
   feature, threshold or hyperparameter.
2. **Splits are by trip.** No observation from one trip may be in two partitions or
   two CV folds. Use `groups=trip_id`.
3. **Five models, real numbers.** Every number in the report must be reproducible
   from this repo. No typed-in metrics, no fabricated curves. All five records
   (`logreg`, `knn`, `random_forest`, `hist_gradient_boosting`, `svm_rbf`) are
   present and `assert_five_models` passes.
4. **Keep the experiment record.** Configs, seeds, Optuna trials and result CSVs are
   committed, not just the final model.
5. **Everyone can defend their own section** in the viva. That is how individual
   contribution is actually assessed.

## Status

All four workstreams are implemented for this build: the P1 feature pipeline and
logistic regression, the P2 group-aware CV harness with KNN and RBF SVM, the P3
random forest plus the Optuna search, and P4's HistGradientBoosting, Viterbi
decoder, FastAPI service, Streamlit UI, prediction maps and artifact manifest.
All five model records are present and `assert_five_models` passes. The random
forest won on accuracy and is deployed; HistGradientBoosting is the speed option
(5.5 s to train against 235 s).

`pytest -q` passes. The long-running stages write their records to
`experiments/results/`; the API and UI run from `artifacts/`, which is gitignored
because the forest is 665 MB (see `artifacts/README.md`). The CI definition lives in
`ci/github-actions.yml`: copy it to `.github/workflows/ci.yml` with a token that has
the `workflow` scope to enable Actions.
