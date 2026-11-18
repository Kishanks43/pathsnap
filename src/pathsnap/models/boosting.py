"""Model 4 - HistGradientBoosting.

OWNER: Person 4
FAMILY: Boosting
CLASS: 4 of 5 required models (Person 4 also owns the decoder and the API)

WHY THIS MODEL IS IN THE LINEUP
-------------------------------
Boosting corrects its own errors sequentially, which suits a task where a small
number of hard cases (parallel roads, junction approaches) dominate the loss.
HistGradientBoosting is also the most computationally efficient strong learner
available here, which matters because evaluation day has a hard time budget.

WHAT IT MUST DEMONSTRATE
-----------------------
Sequential error correction and computational efficiency, compared directly
against the Random Forest: report training time and inference time side by side
and say which you would deploy and why.

DEFINITION OF DONE
------------------
- Early stopping enabled with an internal validation split (`early_stopping=True`,
  `validation_fraction`), and the number of iterations actually used recorded.
- `learning_rate` and `max_iter` stated.
- Note in the report that HistGB bins features internally, so it is less
  sensitive to feature scaling than LogReg/KNN/SVM - a concrete cross-model
  insight rather than a textbook sentence.

REPORT REQUIREMENTS (Deliverables 3.2)
--------------------------------------
A. suitability / strengths / limitations  B. hyperparameters + objective
C. student-test performance + over/underfitting evidence (use the internal
   validation curve that early stopping already produces)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class HistGradientBoostingModel:
    name = "hist_gradient_boosting"
    family = "boosting"

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 300,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 0.0,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self._clf: HistGradientBoostingClassifier | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        self._clf = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            random_state=self.random_state,
        )
        self._clf.fit(X, y)
        self.classes_ = self._clf.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._clf is not None, "call fit() first"
        return self._clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._clf is not None, "call fit() first"
        return self._clf.predict(X)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "l2_regularization": self.l2_regularization,
            "early_stopping": True,
            "random_state": self.random_state,
            "objective": "log loss (multiclass)",
        }
