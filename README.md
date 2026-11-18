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

# 4. run the API (once P4 has implemented it)
uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
```

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
   from this repo. No typed-in metrics, no fabricated curves.
4. **Keep the experiment record.** Configs, seeds, Optuna trials and result CSVs are
   committed, not just the final model.
5. **Everyone can defend their own section** in the viva. That is how individual
   contribution is actually assessed.

## Status

`pytest` passes on the scaffold. Implementation is split across the four
workstreams; each stub raises `NotImplementedError` with a docstring explaining
exactly what is expected and how it will be judged.
