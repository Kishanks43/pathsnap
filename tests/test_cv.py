"""Tests for the group-aware cross-validation harness.

Covers group_kfold (group isolation, index coverage, fallback) and
cross_validate_model (return type, fold count, observation-level metrics).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pathsnap.evaluation.cv import (
    CVResult,
    FoldResult,
    cross_validate_model,
    group_kfold,
    observation_level_predictions,
    zero_true_probability_fraction,
)
from pathsnap.models.base import check_classifier
from pathsnap.models.instance_based import KNNModel, measure_knn_inference_cost
from pathsnap.models.kernel import SVMModel, measure_platt_scaling_cost
from pathsnap.models.linear import LogisticRegressionModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n_obs: int = 50,
    candidates_per_obs: int = 10,
    n_features: int = 19,
    n_groups: int = 10,
    seed: int = 42,
):
    """Generate a fake observation-candidate dataset suitable for CV tests."""
    rng = np.random.RandomState(seed)
    n_rows = n_obs * candidates_per_obs
    X = rng.randn(n_rows, n_features)

    # One positive per observation (first candidate is correct).
    y = np.zeros(n_rows, dtype=np.int64)
    obs_idx = np.repeat(np.arange(n_obs), candidates_per_obs)
    arc_id = rng.randint(0, 100, size=n_rows)

    for i in range(n_obs):
        start = i * candidates_per_obs
        y[start] = 1

    groups = np.repeat(
        np.arange(n_groups), n_obs * candidates_per_obs // n_groups
    )
    # Pad if needed
    if len(groups) < n_rows:
        groups = np.concatenate(
            [groups, np.full(n_rows - len(groups), n_groups - 1)]
        )
    return X, y, groups[:n_rows], obs_idx, arc_id


# ---------------------------------------------------------------------------
# group_kfold
# ---------------------------------------------------------------------------

class TestGroupKfold:

    def test_group_kfold_produces_correct_number_of_splits(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 5)
        y = rng.randint(0, 2, size=100)
        groups = np.repeat(np.arange(10), 10)

        splits = list(group_kfold(X, y, groups, n_splits=5))
        assert len(splits) == 5
        for train_idx, valid_idx in splits:
            assert len(train_idx) > 0
            assert len(valid_idx) > 0

    def test_group_kfold_no_group_leaks(self):
        rng = np.random.RandomState(1)
        X = rng.randn(100, 5)
        y = rng.randint(0, 2, size=100)
        groups = np.repeat(np.arange(10), 10)

        for train_idx, valid_idx in group_kfold(X, y, groups, n_splits=5):
            train_groups = set(groups[train_idx].tolist())
            valid_groups = set(groups[valid_idx].tolist())
            assert train_groups & valid_groups == set(), (
                f"Group leak: {train_groups & valid_groups}"
            )

    def test_group_kfold_covers_all_indices(self):
        rng = np.random.RandomState(2)
        X = rng.randn(100, 5)
        y = rng.randint(0, 2, size=100)
        groups = np.repeat(np.arange(10), 10)

        all_valid: set[int] = set()
        for _, valid_idx in group_kfold(X, y, groups, n_splits=5):
            all_valid.update(valid_idx.tolist())

        assert all_valid == set(range(len(X)))

    def test_group_kfold_fallback_few_groups(self):
        """When there are fewer unique groups than n_splits the function
        should not crash — it falls back to GroupShuffleSplit."""
        rng = np.random.RandomState(3)
        X = rng.randn(40, 5)
        y = rng.randint(0, 2, size=40)
        groups = np.repeat([0, 1], 20)  # only 2 groups

        splits = list(group_kfold(X, y, groups, n_splits=5))
        assert len(splits) == 5
        for train_idx, valid_idx in splits:
            assert len(train_idx) > 0
            assert len(valid_idx) > 0


# ---------------------------------------------------------------------------
# cross_validate_model
# ---------------------------------------------------------------------------

class TestCrossValidateModel:

    def test_cross_validate_model_returns_cvresult(self):
        X, y, groups, _, _ = make_synthetic_data(n_obs=30, candidates_per_obs=5, n_groups=5)
        model = LogisticRegressionModel(C=0.1, max_iter=500)

        result = cross_validate_model(model, X, y, groups, n_splits=3)

        assert isinstance(result, CVResult)
        assert result.model_name == "logreg"
        assert isinstance(result.mean_macro_f1, float)
        assert isinstance(result.std_macro_f1, float)
        assert isinstance(result.mean_top1_accuracy, float)
        # mean_log_loss is float or None
        assert result.mean_log_loss is None or isinstance(result.mean_log_loss, float)

    def test_cross_validate_model_fold_count(self):
        X, y, groups, _, _ = make_synthetic_data(n_obs=30, candidates_per_obs=5, n_groups=5)
        model = LogisticRegressionModel(C=0.1, max_iter=500)
        n_splits = 3

        result = cross_validate_model(model, X, y, groups, n_splits=n_splits)

        assert len(result.folds) == n_splits
        for i, fold in enumerate(result.folds):
            assert isinstance(fold, FoldResult)
            assert fold.fold == i
            assert fold.n_train > 0
            assert fold.n_valid > 0

    def test_cross_validate_model_with_obs_level_metrics(self):
        X, y, groups, obs_idx, arc_id = make_synthetic_data(
            n_obs=30, candidates_per_obs=5, n_groups=5
        )
        model = LogisticRegressionModel(C=0.1, max_iter=500)

        result = cross_validate_model(
            model, X, y, groups,
            obs_idx=obs_idx, arc_id=arc_id,
            n_splits=3,
        )

        assert isinstance(result, CVResult)
        assert len(result.folds) == 3
        # Observation-level metrics should still produce valid floats.
        for fold in result.folds:
            assert isinstance(fold.macro_f1, float)
            assert isinstance(fold.top1_accuracy, float)
            assert 0.0 <= fold.top1_accuracy <= 1.0

    def test_cross_validate_model_positional_args_backwards_compat(self):
        X, y, groups, _, _ = make_synthetic_data(n_obs=30, candidates_per_obs=5, n_groups=5)
        model = LogisticRegressionModel(C=0.1, max_iter=500)
        result = cross_validate_model(model, X, y, groups, 3, 42)
        assert isinstance(result, CVResult)
        assert len(result.folds) == 3

    def test_observation_level_separates_overlapping_trip_obs_idx(self):
        """Two trips both having obs_idx=[0, 1] must produce 4 observations, not 2."""
        # Trip 0: obs 0 (arcs 10, 11 -> true 10), obs 1 (arcs 10, 12 -> true 12)
        # Trip 1: obs 0 (arcs 10, 11 -> true 11), obs 1 (arcs 10, 12 -> true 10)
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        obs_idx = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        arc_id = np.array([10, 11, 10, 12, 10, 11, 10, 12])
        y = np.array([1, 0, 0, 1, 0, 1, 1, 0])
        # Perfect probabilities for all 4 observations
        proba = np.array([
            [0.1, 0.9], [0.8, 0.2],
            [0.7, 0.3], [0.2, 0.8],
            [0.9, 0.1], [0.1, 0.9],
            [0.1, 0.9], [0.9, 0.1],
        ])
        classes = np.array([0, 1])

        y_true_obs, y_pred_obs = observation_level_predictions(
            y, proba, classes, groups, obs_idx, arc_id
        )
        assert len(y_true_obs) == 4
        assert np.array_equal(y_true_obs, y_pred_obs)
        # Trip 0's arc 10 and Trip 1's arc 10 must have distinct namespaced IDs
        assert y_true_obs[0] != y_true_obs[3]


# ---------------------------------------------------------------------------
# P2 models: KNN (Model 2) & RBF SVM (Model 5) contract + CV checks
# ---------------------------------------------------------------------------

class TestP2Models:

    def test_zero_true_probability_fraction(self):
        classes = np.array([0, 1])
        proba = np.array([[0.9, 0.1], [1.0, 0.0], [0.4, 0.6], [0.0, 1.0]])
        y = np.array([0, 1, 0, 0])
        # row 1 true class is 1 but p=0; row 3 true class is 0 but p=0
        assert zero_true_probability_fraction(y, proba, classes) == pytest.approx(0.5)

    def test_knn_satisfies_check_classifier_and_cv(self):
        X, y, groups, obs_idx, arc_id = make_synthetic_data(
            n_obs=40, candidates_per_obs=5, n_groups=5, seed=7
        )
        model = KNNModel(n_neighbors=5, weights="distance", n_jobs=1)
        model.fit(X, y, groups=groups)

        check_classifier(model, X[:25])
        desc = model.describe()
        json.dumps(desc)  # must be JSON-serialisable
        assert desc["n_neighbors"] == 5
        assert desc["n_train_samples"] == len(X)

        cv = cross_validate_model(
            KNNModel(n_neighbors=3, n_jobs=1),
            X, y, groups, n_splits=3, obs_idx=obs_idx, arc_id=arc_id,
        )
        assert cv.model_name == "knn"
        assert len(cv.folds) == 3

    def test_knn_subsample_is_recorded_not_silent(self):
        """A capped KNN run must disclose the cap and the rows actually used."""
        X, y, groups, _, _ = make_synthetic_data(
            n_obs=40, candidates_per_obs=5, n_groups=5, seed=13
        )
        model = KNNModel(n_neighbors=3, max_train_samples=120, n_jobs=1)
        model.fit(X, y, groups=groups)

        check_classifier(model, X[:20])
        desc = model.describe()
        assert desc["max_train_samples"] == 120
        assert desc["n_train_total"] == len(X)
        assert desc["n_train_samples"] == 120
        assert desc["subsample_strategy"] == "stratified_by_label"

    def test_knn_inference_cost_is_json_serialisable(self):
        X, y, _, _, _ = make_synthetic_data(
            n_obs=40, candidates_per_obs=5, n_groups=5, seed=17
        )
        cost = measure_knn_inference_cost(
            X, y, train_sizes=(50, 100), n_query=30, n_neighbors=3, n_jobs=1
        )
        json.dumps(cost)
        assert len(cost["sizes"]) == 2
        assert [row["n_train"] for row in cost["sizes"]] == [50, 100]
        for row in cost["sizes"]:
            assert row["n_query"] == 30
            assert row["predict_seconds"] >= 0.0
            assert row["ms_per_row"] >= 0.0

    def test_svm_satisfies_check_classifier_with_subsampling_and_platt_cost(self):
        X, y, groups, obs_idx, arc_id = make_synthetic_data(
            n_obs=40, candidates_per_obs=5, n_groups=5, seed=11
        )
        # Deliberately include borderline samples and subsampling
        model = SVMModel(C=1.0, gamma="scale", probability=True, max_train_samples=120, random_state=42)
        model.fit(X, y, groups=groups)

        check_classifier(model, X)
        desc = model.describe()
        json.dumps(desc)  # must be JSON-serialisable
        assert desc["n_train_total"] == len(X)
        assert desc["n_train_used"] == 120
        assert desc["subsample_strategy"] == "stratified_by_label"
        assert desc["n_support_vectors"] > 0

        cost = measure_platt_scaling_cost(X, y, max_train_samples=100, random_state=42)
        assert cost["n_train_used"] == 100
        assert cost["fit_no_platt_s"] >= 0.0
        assert cost["fit_with_platt_s"] >= 0.0
        assert cost["n_support_vectors"] > 0

        cv = cross_validate_model(
            SVMModel(max_train_samples=80, random_state=42),
            X, y, groups, n_splits=3, obs_idx=obs_idx, arc_id=arc_id,
        )
        assert cv.model_name == "svm_rbf"
        assert len(cv.folds) == 3
