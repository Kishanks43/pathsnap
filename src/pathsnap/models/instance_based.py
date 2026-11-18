"""Model 2 - Distance-Weighted K-Nearest-Neighbours.

OWNER: Person 2
FAMILY: Instance-based
CLASS: 2 of 5 required models (Models 1-2 belong to Person 2)

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
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class KNNModel:
    name = "knn"
    family = "instance-based"

    def __init__(self, n_neighbors: int = 5, weights: str = "distance", metric: str = "minkowski"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self._pipe: Pipeline | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        self._pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric
                    ),
                ),
            ]
        )
        self._pipe.fit(X, y)
        self.classes_ = self._pipe.named_steps["clf"].classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._pipe is not None, "call fit() first"
        return self._pipe.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._pipe is not None, "call fit() first"
        return self._pipe.predict(X)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "metric": self.metric,
            "scaling": "StandardScaler fitted on training folds only",
            "note": "predict_proba is a k-neighbour vote -> coarse probability estimates",
        }
