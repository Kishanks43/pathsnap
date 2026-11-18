"""Model 2 - Distance-Weighted K-Nearest-Neighbours.

OWNER: Person 2
FAMILY: Instance-based
CLASS: 2 of 5 required models (Person 2 also owns Model 5, the RBF SVM)

WHY THIS MODEL IS IN THE LINEUP
-------------------------------
Map-matching is a "find the arc that looks most like this observation" problem,
which is literally a nearest-neighbour question. Distance weighting lets nearby
examples dominate, and the model makes no assumption about the shape of the
decision boundary - a useful contrast with LogReg.

ITS KNOWN WEAKNESS - SAY THIS IN THE REPORT
-------------------------------------------
KNN has no idea about class imbalance or rare arcs, and `predict_proba` is just
a weighted vote among k neighbours, so probabilities are coarse (jumps of
1/k). That is a real problem for the Viterbi emission term. Reporting this
limitation and showing the log-loss penalty is a strong report point.

REPORT REQUIREMENTS (same generic set, Deliverables 3.2)
--------------------------------------------------------
A. suitability / strengths / limitations  B. hyperparameters + objective
C. student-test performance + over/underfitting evidence

DEFINITION OF DONE
------------------
- k tuned coarsely by CV within this workstream (e.g. k in {3,5,7,11}) and the
  chosen value justified - this is a *within-model* choice, not the Bayesian
  optimisation stage.
- Standardise features: KNN is dominated by whatever column has the largest
  range, so unscaled metres-vs-counts features make it useless.
- Report inference time; KNN scoring cost grows with training set size and that
  matters on evaluation day.

SUBSAMPLING
-----------
Brute-force KNN at 1.45M candidate rows is memory- and time-infeasible: scoring
one held-out fold means a ~270k x 1.16M distance matrix. ``max_train_samples``
therefore caps the *training* rows with a deterministic, label-stratified
subsample (``StratifiedShuffleSplit`` on the binary label, seed recorded).
Validation folds are never subsampled. The cap, the actual rows used and the
strategy are all written into ``describe()`` so the comparison table cannot
silently present a small-sample run as a full-data run.
``measure_knn_inference_cost`` records how inference cost grows with the
training-set size - that is the evidence for the report's cost paragraph.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class KNNModel:
    name = "knn"
    family = "instance-based"

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        metric: str = "euclidean",
        algorithm: str = "brute",
        max_train_samples: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.algorithm = algorithm
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._pipe: Pipeline | None = None
        self.classes_: np.ndarray | None = None
        self.fit_seconds: float | None = None
        self.n_train_total_: int | None = None
        self.n_train_samples_: int | None = None
        self.subsample_strategy_: str = "none"

    def _maybe_subsample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministically subsample training rows when max_train_samples is set."""
        n_total = len(X)
        self.n_train_total_ = int(n_total)
        if self.max_train_samples is None or n_total <= self.max_train_samples:
            self.n_train_samples_ = int(n_total)
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

        self.n_train_samples_ = int(len(idx))
        return X[idx], y[idx]

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        X_fit, y_fit = self._maybe_subsample(np.asarray(X, dtype=np.float32), y)
        self._pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=self.n_neighbors,
                        weights=self.weights,
                        metric=self.metric,
                        algorithm=self.algorithm,
                        n_jobs=self.n_jobs,
                    ),
                ),
            ]
        )
        t0 = time.perf_counter()
        self._pipe.fit(X_fit, y_fit)
        self.fit_seconds = time.perf_counter() - t0
        self.classes_ = self._pipe.named_steps["clf"].classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._pipe is not None, "call fit() first"
        X32 = np.asarray(X, dtype=np.float32)
        return np.asarray(self._pipe.predict_proba(X32), dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._pipe is not None and self.classes_ is not None, "call fit() first"
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "metric": self.metric,
            "algorithm": self.algorithm,
            "scaling": "StandardScaler fitted on training folds only",
            "max_train_samples": self.max_train_samples,
            "n_train_total": self.n_train_total_,
            "n_train_samples": self.n_train_samples_,
            "subsample_strategy": self.subsample_strategy_,
            "fit_seconds": round(self.fit_seconds, 4) if self.fit_seconds is not None else None,
            "note": "predict_proba is a k-neighbour vote -> coarse probability estimates",
        }


def measure_knn_inference_cost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_sizes: tuple[int, ...] = (25_000, 50_000, 100_000),
    n_query: int = 10_000,
    n_neighbors: int = 5,
    weights: str = "distance",
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Measure how KNN inference cost grows with the size of the training set.

    Brute-force KNN has no trained parameters: ``fit`` only stores the reference
    rows, and every prediction is a distance computation against *all* of them.
    Inference cost is therefore O(n_train x n_query), which is the reason KNN
    cannot be the deployment model even though its accuracy may be competitive.

    One model is fitted per size on a deterministic row sample, then scored on a
    **fixed** query sample drawn from rows outside every training slice, so the
    only thing that changes between measurements is the training-set size.

    Returns a JSON-serialisable dict with per-size wall-clock timings and the
    derived milliseconds per scored row. The caller divides by candidates per
    observation to express the result per GPS observation.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if n_query <= 0:
        raise ValueError("n_query must be positive")

    sizes = sorted({int(s) for s in train_sizes if int(s) > 0})
    if not sizes:
        raise ValueError("train_sizes must contain at least one positive size")

    rng = np.random.RandomState(random_state)
    order = rng.permutation(len(X))
    max_train = min(sizes[-1], len(X))
    query_start = max_train
    query_stop = min(query_start + n_query, len(X))
    if query_stop <= query_start:
        raise ValueError(
            f"need at least {max_train + 1} rows to measure; got {len(X)}"
        )
    # Rows used for training and rows used for the query are disjoint.
    pool = order[:max_train]
    query = X[order[query_start:query_stop]]

    results: list[dict[str, Any]] = []
    for size in sizes:
        if size > len(pool):
            continue
        model = KNNModel(
            n_neighbors=n_neighbors,
            weights=weights,
            max_train_samples=None,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(X[pool[:size]], y[pool[:size]])

        t0 = time.perf_counter()
        model.predict_proba(query)
        predict_seconds = time.perf_counter() - t0

        results.append(
            {
                "n_train": int(size),
                "n_query": int(len(query)),
                "fit_seconds": float(model.fit_seconds or 0.0),
                "predict_seconds": float(predict_seconds),
                "ms_per_row": float(predict_seconds * 1000.0 / len(query)),
            }
        )

    return {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "query_rows_shared_across_sizes": int(len(query)),
        "random_state": random_state,
        "sizes": results,
    }
