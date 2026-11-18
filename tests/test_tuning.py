from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import optuna
import pytest

from pathsnap.evaluation import cv
from pathsnap.tuning.bayes import suggest_params, tune


@pytest.mark.parametrize(
    ("model_name", "fixed_params"),
    [
        (
            "random_forest",
            {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 4,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
        ),
        ("logreg", {"C": 1.0}),
        ("knn", {"n_neighbors": 5, "weights": "distance"}),
        ("svm_rbf", {"C": 1.0, "gamma": 0.1}),
        (
            "hist_gradient_boosting",
            {
                "learning_rate": 0.1,
                "max_leaf_nodes": 31,
                "l2_regularization": 0.01,
            },
        ),
    ],
)
def test_suggest_params_supports_each_registered_model(model_name, fixed_params):
    params = suggest_params(optuna.trial.FixedTrial(fixed_params), model_name)
    assert params == fixed_params


def test_suggest_params_rejects_unknown_model():
    with pytest.raises(KeyError, match="unknown model"):
        suggest_params(optuna.trial.FixedTrial({}), "not_a_model")


def test_tune_uses_authoritative_cv_and_persists_trials_and_config(monkeypatch, tmp_path):
    X = np.zeros((24, 19), dtype=np.float64)
    y = np.array([0, 1] * 12)
    groups = np.repeat(np.arange(6), 4)
    obs_idx = np.tile(np.arange(4), 6)
    arc_id = np.arange(len(X)) % 5
    calls = []

    def fake_cross_validate_model(
        model,
        passed_X,
        passed_y,
        passed_groups,
        n_splits,
        seed,
        *,
        obs_idx,
        arc_id,
    ):
        calls.append((model, passed_X, passed_y, passed_groups, n_splits, seed, obs_idx, arc_id))
        return SimpleNamespace(mean_macro_f1=0.6 + model.min_samples_leaf / 100.0)

    monkeypatch.setattr(cv, "cross_validate_model", fake_cross_validate_model)
    results_dir = tmp_path / "results"
    first = tune(
        "random_forest",
        X,
        y,
        groups,
        n_trials=2,
        seed=11,
        results_dir=results_dir,
        n_splits=3,
        obs_idx=obs_idx,
        arc_id=arc_id,
    )
    second = tune(
        "random_forest",
        X,
        y,
        groups,
        n_trials=1,
        seed=11,
        results_dir=results_dir,
        n_splits=3,
        obs_idx=obs_idx,
        arc_id=arc_id,
    )

    assert len(calls) == 3
    assert all(call[3] is groups for call in calls)
    assert all(call[4:6] == (3, 11) for call in calls)
    assert all(call[6] is obs_idx and call[7] is arc_id for call in calls)
    assert first["n_trials"] == 2
    assert second["n_trials"] == 3

    with (results_dir / "trials.csv").open(newline="", encoding="utf-8") as file:
        trials = list(csv.DictReader(file))
    assert len(trials) == 3
    assert all(row["state"] == "COMPLETE" for row in trials)

    config = json.loads((tmp_path / "configs" / "bayes_random_forest.json").read_text())
    best = json.loads((results_dir / "best_config_random_forest.json").read_text())
    assert config["seed"] == 11
    assert config["cv"]["implementation"] == "pathsnap.evaluation.cv.cross_validate_model"
    assert best["best_params"] == second["best_params"]
    assert best["best_value"] == pytest.approx(second["best_value"])


def test_tune_requires_aligned_training_groups(tmp_path):
    with pytest.raises(ValueError, match="matching row counts"):
        tune(
            "random_forest",
            np.zeros((4, 19)),
            np.array([0, 1, 0, 1]),
            np.array([0, 0, 1]),
            n_trials=1,
            results_dir=tmp_path / "results",
            obs_idx=np.arange(4),
            arc_id=np.arange(4),
        )


def test_tune_requires_multiple_trip_groups(tmp_path):
    with pytest.raises(ValueError, match="at least two distinct trip groups"):
        tune(
            "random_forest",
            np.zeros((4, 19)),
            np.array([0, 1, 0, 1]),
            np.zeros(4, dtype=int),
            n_trials=1,
            results_dir=tmp_path / "results",
            obs_idx=np.arange(4),
            arc_id=np.arange(4),
        )
