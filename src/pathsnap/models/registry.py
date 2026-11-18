"""Registry of the five models that satisfy the "at least five models" requirement.

Keep this list at five or more. The comparison table and the report both iterate
over `ALL_MODELS`, so adding a sixth model here automatically gets it evaluated
and tabulated - that is the intended extension point.

Ownership (one model each, fifth to P2):
    Model 1 -> Person 1   linear.py
    Model 2 -> Person 2   instance_based.py
    Model 3 -> Person 3   tree.py
    Model 4 -> Person 4   boosting.py
    Model 5 -> Person 2   kernel.py
"""

from __future__ import annotations

from pathsnap.models.base import CandidateClassifier
from pathsnap.models.boosting import HistGradientBoostingModel
from pathsnap.models.instance_based import KNNModel
from pathsnap.models.kernel import SVMModel
from pathsnap.models.linear import LogisticRegressionModel
from pathsnap.models.tree import RandomForestModel

#: Constructor callables, in the order they should appear in the report table.
ALL_MODELS: dict[str, type] = {
    "logreg": LogisticRegressionModel,
    "knn": KNNModel,
    "random_forest": RandomForestModel,
    "hist_gradient_boosting": HistGradientBoostingModel,
    "svm_rbf": SVMModel,
}


def build_model(name: str, **overrides) -> CandidateClassifier:
    """Instantiate a registered model by name, with optional config overrides."""
    if name not in ALL_MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(ALL_MODELS)}")
    return ALL_MODELS[name](**overrides)


__all__ = ["ALL_MODELS", "build_model", "CandidateClassifier"]
