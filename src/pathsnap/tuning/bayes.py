"""Bayesian hyperparameter optimisation of the selected model.

OWNER: Person 3
STATUS: STUB

REQUIRED BY: Models & Training Guidelines section 7, and highlighted twice in
the integrity policy ("Demonstrate Bayesian hyperparameter optimisation",
"Retain the optimisation search record").

WHAT THE MARKING SCHEME LOOKS FOR
---------------------------------
1. A justified search space (not "I searched everything").
2. A stated objective - and it must be cross-validated macro F1, not training
   accuracy.
3. Group-aware CV over the 80 training trips (pass `groups=trip_id`).
4. The full trial record retained: every trial's params and value.
5. The best configuration identified.
6. The model refit with it.
7. Tuned vs untuned comparison on the 10 student-test trips.
8. Computational cost reported (wall-clock, number of trials).

LEAKAGE WARNING
---------------
The evaluator-testing partition must not inform the search space, the objective,
the number of trials, or the final choice. If you peek at it to decide
`n_trials`, that is a violation, and it is visible in the git history and in
your own report wording. Everything here uses train trips for the search and
student-test trips for the final comparison only.

HARD CONSTRAINTS
----------------
- Use Optuna's TPE sampler (`optuna.samplers.TPESampler`) - that is the Bayesian
  component. Random search would miss the requirement.
- Persist the study so the record survives: `storage="sqlite:///experiments/results/trials.db"`,
  `load_if_exists=True`. trials.db is gitignored; export a CSV of trials at the
  end so the *record* is committed, as required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def suggest_params(trial, model_name: str) -> dict[str, Any]:
    """Map an Optuna trial to a hyperparameter dict for the chosen model.

    Suggested spaces (justify whatever you actually use):
        logreg        C: log-uniform 1e-3 .. 1e2
        random_forest n_estimators: 100..600, max_depth: 3..30 or None,
                      min_samples_leaf: 1..20
        hist_gb       learning_rate: 1e-3..3e-1 (log), max_leaf_nodes: 15..127,
                      l2_regularization: 1e-8..1e1 (log)
        svm_rbf       C: 1e-2..1e2 (log), gamma: 1e-4..1e0 (log)
        knn           n_neighbors: 1..25, weights: categorical
    """
    raise NotImplementedError("Person 3: implement the search spaces above.")


def tune(
    model_name: str,
    X,
    y,
    groups,
    n_trials: int = 40,
    seed: int = 42,
    results_dir: Path | None = None,
    objective_fn: Callable | None = None,
) -> dict[str, Any]:
    """Run the Bayesian search and return {"best_params", "best_value", "n_trials", "study_summary"}.

    The objective must be the group-aware CV macro F1 computed with
    pathsnap.evaluation.cv.cross_validate_model, averaged over folds.
    """
    raise NotImplementedError("Person 3: implement the Optuna study.")


def export_trials(study, path: Path) -> Path:
    """Write every trial to CSV so the search record is committed to the repo."""
    raise NotImplementedError("Person 3: implement trial export.")
