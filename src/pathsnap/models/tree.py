"""Model 3 - Random Forest.

OWNER: Person 3
FAMILY: Ensemble (bagging of trees)
CLASS: 3 of 5 required models (Person 3 also owns the Bayesian optimisation stage)

WHY THIS MODEL IS IN THE LINEUP
-------------------------------
The features mix metres, degrees, counts and flags, with non-linear
interactions (an arc is plausible if it is close AND roughly aligned AND
connected to the previous choice). Trees capture that without manual scaling,
and averaging many trees gives reasonably smooth probabilities for the HMM.

WHAT IT MUST DEMONSTRATE
-----------------------
Stability and generalisation: report CV variance across folds, not just the
mean. Show `max_depth` / `min_samples_leaf` effects, and be explicit whether
depth was limited to control the overfitting that an unrestricted forest
typically shows on 80 trips.

DEFINITION OF DONE
------------------
- `n_jobs=-1` for speed, but record wall-clock training time honestly.
- `random_state` set and recorded (seeds are required to be retained).
- Feature importances exported to experiments/results/importances_rf.csv - the
  report needs evidence, and importances are cheap evidence.
- OOB score recorded if `oob_score=True` is used; explain what it is.

REPORT REQUIREMENTS (Deliverables 3.2)
--------------------------------------
A. suitability / strengths / limitations  B. hyperparameters + objective
C. student-test performance + over/underfitting evidence
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    name = "random_forest"
    family = "tree/ensemble"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._clf: RandomForestClassifier | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        self._clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
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

    def feature_importances(self) -> np.ndarray:
        assert self._clf is not None, "call fit() first"
        return self._clf.feature_importances_

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "objective": "gini impurity",
        }
