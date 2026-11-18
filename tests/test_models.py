from __future__ import annotations

import csv

import numpy as np
import pytest

from pathsnap.features.build import FEATURE_COLUMNS
from pathsnap.models.base import check_classifier
from pathsnap.models.tree import RandomForestModel


def _imbalanced_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(240, len(FEATURE_COLUMNS)))
    y = np.zeros(len(X), dtype=np.int64)
    y[:36] = 1
    rng.shuffle(y)
    return X, y


def test_random_forest_satisfies_classifier_contract_with_imbalance():
    X, y = _imbalanced_data()
    model = RandomForestModel(
        n_estimators=12,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=17,
        n_jobs=1,
    ).fit(X, y)

    check_classifier(model, X[:32])
    assert model.classes_.tolist() == [0, 1]
    assert model.describe()["depth_limited"] is True
    assert model.describe()["class_weight"] == "balanced_subsample"
    assert model.feature_importances().shape == (len(FEATURE_COLUMNS),)


def test_random_forest_reproducibility_and_importance_export(tmp_path):
    X, y = _imbalanced_data(seed=3)
    kwargs = {
        "n_estimators": 10,
        "max_depth": 5,
        "random_state": 9,
        "n_jobs": 1,
    }
    first = RandomForestModel(**kwargs).fit(X, y)
    second = RandomForestModel(**kwargs).fit(X, y)
    np.testing.assert_array_equal(first.predict_proba(X[:20]), second.predict_proba(X[:20]))

    path = first.export_feature_importances(tmp_path / "importances_rf.csv")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(FEATURE_COLUMNS)
    assert rows[0]["feature"] in FEATURE_COLUMNS
    assert float(rows[0]["importance"]) >= float(rows[-1]["importance"])


def test_random_forest_rejects_non_schema_feature_count():
    model = RandomForestModel(n_estimators=2, n_jobs=1)
    with pytest.raises(ValueError, match="19 feature columns"):
        model.fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))


def test_random_forest_requires_fit_before_prediction():
    with pytest.raises(RuntimeError, match="call fit"):
        RandomForestModel().predict_proba(np.zeros((1, len(FEATURE_COLUMNS))))
