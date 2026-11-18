"""The model contract every candidate classifier must satisfy.

Four workstreams produce models in parallel, so the interface is fixed here and
nothing else may change it:

    fit(X, y, groups=None)      -> self
    predict_proba(X)            -> (n_samples, n_classes) float64, rows sum to 1
    predict(X)                  -> (n_samples,) int labels
    classes_                    -> sorted unique labels seen during fit

`X` is always a 2-D float64 numpy array produced by
`pathsnap.features.build.build_feature_matrix`. A model MUST NOT touch raw
files, do its own file I/O, or re-derive features — that keeps all five models
comparable on identical inputs, which the brief requires.

Every model must also expose `describe()` returning a JSON-serialisable dict of
its configuration, because the report needs a configuration column per model
and the results table is generated from this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class CandidateClassifier(Protocol):
    classes_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> "CandidateClassifier": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass
class ModelRecord:
    """Everything the report and the comparison table need about one model.

    One instance per trained model, written to
    ``experiments/results/model_<name>.json`` by ``pathsnap.evaluation.compare``.
    """

    name: str
    family: str
    config: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    cv_macro_f1: float | None = None
    test_top1_accuracy: float | None = None
    test_macro_f1: float | None = None
    test_log_loss: float | None = None
    candidate_miss_rate: float | None = None
    route_continuity: float | None = None
    train_seconds: float | None = None
    inference_ms_per_obs: float | None = None
    artifact_path: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def check_classifier(model: CandidateClassifier, X: np.ndarray) -> None:
    """Fail fast if a model breaks the contract. Call this in your unit tests."""
    proba = model.predict_proba(X)
    if proba.ndim != 2:
        raise ValueError(f"predict_proba must return 2-D, got shape {proba.shape}")
    if proba.shape[0] != X.shape[0]:
        raise ValueError("predict_proba row count != X row count")
    if proba.shape[1] != len(model.classes_):
        raise ValueError("predict_proba column count != len(classes_)")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("predict_proba rows do not sum to 1")
    if (proba < -1e-9).any():
        raise ValueError("predict_proba contains negative values")
    if not np.array_equal(model.predict(X), model.classes_[proba.argmax(axis=1)]):
        raise ValueError("predict() disagrees with argmax of predict_proba()")
