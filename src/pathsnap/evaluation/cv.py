"""Group-aware cross-validation.

OWNER: Person 2 (shared harness - Person 3 uses it for the Bayesian search)
STATUS: STUB - implement alongside Model 1; everyone else will import it.

WHY GROUP-AWARE
---------------
The brief is explicit: no observations from the same trip may be spread across
different partitions/folds. If a fold contains rows from a trip that also
appears in the training fold, the model memorises that trip's geometry and the
CV score is meaningless. So splits are always by `trip_id`, never by row.

GroupKFold / GroupShuffleSplit from sklearn are fine and honest; you do not have
to write the splitter by hand. What you must do is state the number of folds and
show the fold-to-fold variance, not just the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_valid: int
    macro_f1: float
    top1_accuracy: float
    log_loss: float | None


@dataclass
class CVResult:
    model_name: str
    folds: list[FoldResult]
    mean_macro_f1: float
    std_macro_f1: float
    mean_top1_accuracy: float
    mean_log_loss: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_folds": len(self.folds),
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "mean_top1_accuracy": self.mean_top1_accuracy,
            "mean_log_loss": self.mean_log_loss,
            "folds": [f.__dict__ for f in self.folds],
        }


def group_kfold(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 42
):
    """Yield (train_idx, valid_idx) with whole trips kept on one side.

    Use sklearn.model_selection.GroupKFold. If any group is smaller than
    n_splits / has too few rows, fall back to GroupShuffleSplit and say so in
    the report - changing the splitter silently is not acceptable.
    """
    raise NotImplementedError("Person 2: implement group-aware splitting.")


def cross_validate_model(model, X, y, groups, n_splits: int = 5, seed: int = 42) -> CVResult:
    """Fit `model` on each training fold and score it on the held-out fold.

    Record the fold-to-fold std of macro F1: 'stability across validation folds'
    is an explicit selection criterion in the brief.
    """
    raise NotImplementedError("Person 2: implement the CV loop.")
