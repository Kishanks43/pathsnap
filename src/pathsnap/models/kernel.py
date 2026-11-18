"""Model 5 - RBF-kernel Support Vector Machine.

OWNER: Person 2
FAMILY: Kernel-based
CLASS: 5 of 5 required models (the fifth model, assigned to Person 2 because the
       measurement burden - calibration and runtime - sits with the eval harness)

WHY THIS MODEL IS IN THE LINEUP
-------------------------------
An RBF SVM draws a smooth non-linear boundary around each class, which is a
different inductive bias from trees (axis-aligned splits) and from LogReg
(linear). It gives the comparison table a genuinely different family, which the
brief asks for ("experiment with models from multiple algorithm families").

THE HONEST TRADEOFF - THIS IS YOUR STRONGEST REPORT POINT
--------------------------------------------------------
RBF SVM gives excellent accuracy-per-parameter but probability calibration is
expensive: `probability=True` runs internal 5-fold Platt scaling, so training
time and model size blow up. On ~10^5 candidate rows this can dominate the whole
experiment run. Measure it, report it, and use it as evidence for why the
deployed model is (or is not) the SVM. If it is too slow, scale the feature
matrix and/or reduce the training sample *for this model only*, and say so
explicitly - do not silently subsample and present the result as if it used all
data.

DEFINITION OF DONE
------------------
- `C` and `gamma` stated, and the feature pipeline standardises (mandatory: SVM
  is scale-sensitive).
- Training time and inference time measured and recorded; expect these to be
  the worst of the five models.
- Platt-scaled probabilities recorded, plus a note on calibration quality
  (compare log loss against the Random Forest).

REPORT REQUIREMENTS (Deliverables 3.2)
--------------------------------------
A. suitability / strengths / limitations  B. hyperparameters + objective
C. student-test performance + over/underfitting evidence
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMModel:
    name = "svm_rbf"
    family = "kernel-based"

    def __init__(
        self,
        C: float = 1.0,
        gamma: str | float = "scale",
        kernel: str = "rbf",
        probability: bool = True,
        max_train_samples: int | None = None,
        random_state: int = 42,
    ):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.probability = probability
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self._pipe: Pipeline | None = None
        self.classes_: np.ndarray | None = None
        self.fit_seconds: float | None = None
        self.n_train_total_: int | None = None
        self.n_train_used_: int | None = None
        self.subsample_strategy_: str = "none"
        self.n_support_vectors_: int | None = None

    def _maybe_subsample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministically subsample training rows when max_train_samples is set.

        Stratifies on binary label ``y`` to preserve the positive/negative ratio
        (approx 12.7% positive on the candidate matrix). Never applied silently:
        ``n_train_total_``, ``n_train_used_``, and ``subsample_strategy_`` are
        recorded in ``describe()``.
        """
        n_total = len(X)
        self.n_train_total_ = int(n_total)
        if self.max_train_samples is None or n_total <= self.max_train_samples:
            self.n_train_used_ = int(n_total)
            self.subsample_strategy_ = "none"
            return X, y

        _, counts = np.unique(y, return_counts=True)
        if len(counts) >= 2 and int(counts.min()) >= 2:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=self.max_train_samples,
                random_state=self.random_state,
            )
            idx, _ = next(splitter.split(X, y))
            self.subsample_strategy_ = "stratified_by_label"
        else:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n_total, size=self.max_train_samples, replace=False)
            self.subsample_strategy_ = "uniform_random"

        self.n_train_used_ = int(len(idx))
        return X[idx], y[idx]

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        X_fit, y_fit = self._maybe_subsample(X, y)
        self._pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=self.C,
                        gamma=self.gamma,
                        kernel=self.kernel,
                        probability=self.probability,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        t0 = time.perf_counter()
        self._pipe.fit(X_fit, y_fit)
        self.fit_seconds = time.perf_counter() - t0
        clf: SVC = self._pipe.named_steps["clf"]
        self.classes_ = clf.classes_
        self.n_support_vectors_ = int(np.sum(clf.n_support_))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._pipe is not None, "call fit() first"
        return np.asarray(self._pipe.predict_proba(X), dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._pipe is not None and self.classes_ is not None, "call fit() first"
        if self.probability:
            # Use argmax of Platt-calibrated probabilities so predict() is always
            # consistent with predict_proba() (sklearn SVC.predict uses the raw
            # decision_function, which can disagree near the boundary).
            proba = self.predict_proba(X)
            return self.classes_[proba.argmax(axis=1)]
        return self._pipe.predict(X)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "probability": self.probability,
            "scaling": "StandardScaler fitted on training folds only",
            "max_train_samples": self.max_train_samples,
            "n_train_total": self.n_train_total_,
            "n_train_used": self.n_train_used_,
            "subsample_strategy": self.subsample_strategy_,
            "n_support_vectors": self.n_support_vectors_,
            "fit_seconds": round(self.fit_seconds, 4) if self.fit_seconds is not None else None,
            "note": "probability=True triggers internal 5-fold Platt scaling",
        }


def measure_platt_scaling_cost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    gamma: str | float = "scale",
    max_train_samples: int | None = None,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Measure the wall-clock training overhead of internal 5-fold Platt scaling.

    Fits two identical RBF SVMs on the same training slice — one with
    ``probability=False`` (raw hinge-loss SVM) and one with ``probability=True``
    (adds internal 5-fold CV sigmoid calibration) — and reports the cost ratio.
    """
    raw = SVMModel(
        C=C,
        gamma=gamma,
        probability=False,
        max_train_samples=max_train_samples,
        random_state=random_state,
    )
    raw.fit(X, y)

    calibrated = SVMModel(
        C=C,
        gamma=gamma,
        probability=True,
        max_train_samples=max_train_samples,
        random_state=random_state,
    )
    calibrated.fit(X, y)

    t_raw = raw.fit_seconds or 0.0
    t_cal = calibrated.fit_seconds or 0.0
    return {
        "n_train_used": calibrated.n_train_used_ or len(X),
        "n_support_vectors": calibrated.n_support_vectors_ or 0,
        "fit_no_platt_s": t_raw,
        "fit_with_platt_s": t_cal,
        "extra_seconds": t_cal - t_raw,
        "overhead_ratio": (t_cal / t_raw) if t_raw > 0 else float("inf"),
    }
