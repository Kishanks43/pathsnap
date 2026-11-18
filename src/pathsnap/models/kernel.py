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
        random_state: int = 42,
    ):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.probability = probability
        self.random_state = random_state
        self._pipe: Pipeline | None = None
        self.classes_: np.ndarray | None = None
        self.fit_seconds: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
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
        self._pipe.fit(X, y)
        self.fit_seconds = time.perf_counter() - t0
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
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "probability": self.probability,
            "fit_seconds": self.fit_seconds,
            "note": "probability=True triggers internal 5-fold Platt scaling",
        }
