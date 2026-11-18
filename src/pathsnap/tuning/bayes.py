"""Bayesian hyperparameter optimisation of the selected model.

OWNER: Person 3
STATUS: implemented

REQUIRED BY: Models & Training Guidelines section 7, and highlighted twice in
the integrity policy ("Demonstrate Bayesian hyperparameter optimisation",
"Retain the optimisation search record").

WHAT THE MARKING SCHEME LOOKS FOR
---------------------------------
1. A justified search space (not "I searched everything").
2. A stated objective - and it must be cross-validated macro F1, not training
   accuracy.
3. Group-aware CV over the 80 training trips (pass `groups=trip_id`).
4. The full trial record retained: every trial's params and value.
5. The best configuration identified.
6. The model refit with it.
7. Tuned vs untuned comparison on the 10 student-test trips.
8. Computational cost reported (wall-clock, number of trials).

LEAKAGE WARNING
---------------
The evaluator-testing partition must not inform the search space, the objective,
the number of trials, or the final choice. If you peek at it to decide
`n_trials`, that is a violation, and it is visible in the git history and in
your own report wording. Everything here uses train trips for the search and
student-test trips for the final comparison only.

HARD CONSTRAINTS
----------------
- Use Optuna's TPE sampler (`optuna.samplers.TPESampler`) - that is the Bayesian
  component. Random search would miss the requirement.
- Persist the study so the record survives: `storage="sqlite:///experiments/results/trials.db"`,
  `load_if_exists=True`. trials.db is gitignored; export a CSV of trials at the
  end so the *record* is committed, as required.
"""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from pathsnap.config import RESULTS_DIR
from pathsnap.evaluation import cv
from pathsnap.models.registry import ALL_MODELS, build_model


def suggest_params(trial, model_name: str) -> dict[str, Any]:
    """Map an Optuna trial to the bounded search space for a registered model."""
    if model_name == "random_forest":
        return {
            # 100-600 trees trades ensemble stability against training cost.
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            # A bounded depth search plus None tests whether the baseline cap is useful.
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 25, 30]),
            # These controls limit overly specific splits and leaf memorisation.
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            # sqrt/log2 are standard RF choices; 0.5 tests a broader feature subset.
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            # Compare no reweighting with sklearn's global and per-bootstrap balancing.
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
        }
    if model_name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 25),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    if model_name == "svm_rbf":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 1e1, log=True),
        }
    if model_name not in ALL_MODELS:
        raise KeyError(f"unknown model {model_name!r}; available: {sorted(ALL_MODELS)}")
    raise ValueError(f"no Bayesian search space is defined for model {model_name!r}")


def tune(
    model_name: str,
    X,
    y,
    groups,
    n_trials: int = 40,
    seed: int = 42,
    results_dir: Path | None = None,
    n_splits: int = 5,
    *,
    obs_idx: np.ndarray,
    arc_id: np.ndarray,
) -> dict[str, Any]:
    """Tune a model using P2's group-aware CV and persist the full experiment.

    Pass training rows only (and apply the documented label-trust mask before
    calling). `groups` must identify the source trip for each row. Student-test
    data is deliberately not an input to this function. `obs_idx` and `arc_id`
    are required so P2's CV harness computes macro F1 at the observation level.
    """
    if model_name not in ALL_MODELS:
        raise KeyError(f"unknown model {model_name!r}; available: {sorted(ALL_MODELS)}")
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    obs_idx = np.asarray(obs_idx)
    arc_id = np.asarray(arc_id)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D feature matrix")
    if y.ndim != 1 or groups.ndim != 1 or len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y, and groups must have matching row counts")
    if obs_idx.ndim != 1 or arc_id.ndim != 1 or len(X) != len(obs_idx) or len(X) != len(arc_id):
        raise ValueError("obs_idx and arc_id must be 1-D arrays matching the rows of X")
    if len(np.unique(groups)) < 2:
        raise ValueError("group-aware CV requires at least two distinct trip groups")

    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    config_dir = results_dir.parent / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"pathsnap-{model_name}-seed-{seed}-folds-{n_splits}"
    database_path = (results_dir / "trials.db").resolve()
    storage_url = f"sqlite:///{database_path.as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage_url,
        load_if_exists=True,
    )

    experiment_config = {
        "model_name": model_name,
        "study_name": study_name,
        "objective": "mean group-aware cross-validation macro F1",
        "cv": {
            "implementation": "pathsnap.evaluation.cv.cross_validate_model",
            "n_splits": n_splits,
            "groups": "trip_id",
            "metrics": "observation-level macro F1 using obs_idx and arc_id",
        },
        "sampler": "TPESampler",
        "seed": seed,
        "requested_trials_this_run": n_trials,
        "training_rows": len(X),
        "training_groups": len(np.unique(groups)),
        "label_filter": "caller must apply StackedFeatures.trusted() before tuning",
        "database": str(database_path),
    }
    config_path = config_dir / f"bayes_{model_name}.json"
    config_path.write_text(json.dumps(experiment_config, indent=2), encoding="utf-8")

    def objective(trial) -> float:
        params = suggest_params(trial, model_name)
        model = build_model(model_name, **_seeded_model_params(model_name, params, seed))
        result = cv.cross_validate_model(
            model,
            X,
            y,
            groups,
            n_splits=n_splits,
            seed=seed,
            obs_idx=obs_idx,
            arc_id=arc_id,
        )
        value = float(result.mean_macro_f1)
        if not np.isfinite(value):
            raise ValueError(f"CV returned a non-finite macro F1: {value}")
        return value

    try:
        study.optimize(objective, n_trials=n_trials)
    finally:
        export_trials(study, results_dir / "trials.csv")

    completed_trials = [trial for trial in study.trials if trial.value is not None]
    if not completed_trials:
        raise RuntimeError(f"Optuna study {study_name!r} has no completed trials")

    best_config = {
        "model_name": model_name,
        "study_name": study_name,
        "seed": seed,
        "objective": experiment_config["objective"],
        "cv": experiment_config["cv"],
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials_total": len(study.trials),
        "n_trials_this_run": n_trials,
    }
    best_config_path = results_dir / f"best_config_{model_name}.json"
    best_config_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    summary = {
        "study_name": study_name,
        "database": str(database_path),
        "complete_trials": len(completed_trials),
        "failed_trials": sum(trial.value is None for trial in study.trials),
        "best_trial_number": study.best_trial.number,
    }
    return {
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "study_summary": summary,
        "trials_path": str(results_dir / "trials.csv"),
        "config_path": str(config_path),
        "best_config_path": str(best_config_path),
    }


def export_trials(study, path: Path) -> Path:
    """Write every Optuna trial, including failed trials, to a machine-readable CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parameter_names = sorted({name for trial in study.trials for name in trial.params})
    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration_seconds",
        "params",
        *(f"param_{name}" for name in parameter_names),
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "value": "" if trial.value is None else trial.value,
                "datetime_start": (
                    "" if trial.datetime_start is None else trial.datetime_start.isoformat()
                ),
                "datetime_complete": (
                    "" if trial.datetime_complete is None else trial.datetime_complete.isoformat()
                ),
                "duration_seconds": (
                    "" if trial.duration is None else trial.duration.total_seconds()
                ),
                "params": json.dumps(trial.params, sort_keys=True),
            }
            row.update({f"param_{name}": trial.params.get(name, "") for name in parameter_names})
            writer.writerow(row)
    return path


def _seeded_model_params(model_name: str, params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Set a model's random seed where its constructor exposes that option."""
    model_class = ALL_MODELS[model_name]
    if "random_state" in inspect.signature(model_class).parameters:
        return {**params, "random_state": seed}
    return params
