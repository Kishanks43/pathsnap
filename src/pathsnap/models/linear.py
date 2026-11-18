"""Model 1 - Multinomial Logistic Regression.

OWNER: Person 2
FAMILY: Linear / probabilistic
CLASS: 1 of 5 required models (Models 1-2 belong to Person 2)

WHY THIS MODEL IS IN THE LINEUP
-------------------------------
It is the honest baseline. It produces calibrated class probabilities, which is
exactly what the HMM emission model needs, and it is fast enough to train
inside a cross-validation loop many times. If a fancier model cannot clearly
beat it, that is a finding worth reporting - not a failure.

WHAT THE REPORT MUST ANSWER FOR THIS MODEL (Deliverables 3.2, per-model section)
--------------------------------------------------------------------------------
A. Why suitable for map-matching? Strengths? Limitations?
B. Important hyperparameters and their initial values; loss/objective used.
C. Student-test performance; over/underfitting evidence.

DEFINITION OF DONE
------------------
- `describe()` returns the exact config used (C value, solver, max_iter,
  whether features were standardised, class_weight).
- Because features have wildly different scales (metres vs degrees vs counts),
  standardise inside a sklearn Pipeline and fit the scaler on train folds only.
- Set max_iter high enough to converge and record the convergence warning if
  it still appears - do not silently ignore it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:
    name = "logreg"
    family = "linear/probabilistic"

    def __init__(self, C: float = 1.0, max_iter: int = 2000, class_weight: str | None = None):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self._pipe: Pipeline | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        self._pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=self.C, max_iter=self.max_iter, class_weight=self.class_weight)),
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
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "scaling": "StandardScaler fitted on training folds only",
        }


# TODO(Person 2): tune nothing here yet - hyperparameter search is Person 3's
# job on the *winning* model. Just make sure the untuned baseline is recorded.
