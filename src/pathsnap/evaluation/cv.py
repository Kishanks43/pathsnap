"""Group-aware cross-validation.

OWNER: Person 2 (shared harness - Person 3 uses it for the Bayesian search)
STATUS: implemented

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

import copy
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from pathsnap.evaluation.metrics import macro_f1, safe_log_loss, top1_accuracy


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_valid: int
    macro_f1: float
    top1_accuracy: float
    log_loss: float | None
    zero_prob_fraction: float = 0.0


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
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        print(
            f"WARNING: only {len(unique_groups)} unique groups but n_splits={n_splits}; "
            f"falling back to GroupShuffleSplit.",
            file=sys.stderr,
        )
        splitter = GroupShuffleSplit(
            n_splits=n_splits, test_size=1 / n_splits, random_state=seed
        )
    else:
        splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(X, y, groups)


def zero_true_probability_fraction(
    y: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Fraction of rows whose true label was given exactly zero probability.

    ``safe_log_loss`` returns ``None`` when this is non-zero, because the true
    log loss is infinite. That is the honest result, but "N/A" on its own is not
    evidence. This diagnostic quantifies *how often* a model assigns zero
    probability to the truth, which is the mechanism behind KNN/SVM calibration
    failures and therefore the number to quote next to the log-loss cell.
    """
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=np.float64)
    classes = np.asarray(classes)
    if len(y) == 0:
        return 0.0
    index = {label: i for i, label in enumerate(classes.tolist())}
    cols = np.fromiter((index[v] for v in y.tolist()), dtype=np.int64, count=len(y))
    p_true = proba[np.arange(len(y)), cols]
    return float((p_true <= 0.0).mean())


def observation_level_predictions(
    y: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    groups: np.ndarray,
    obs_idx: np.ndarray,
    arc_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-candidate binary probabilities into per-observation arc picks.

    Groups rows by ``(group, obs_idx)`` in O(N log N) so that observations from
    different trips (which each start at ``obs_idx == 0``) never collide, and
    namespaces ``arc_id`` by ``group`` so per-trip road network IDs never
    collide when computing observation-level macro F1 across trips.

    For each observation, selects the candidate with the highest predicted
    probability of class ``1``. Observations whose true arc was not offered
    (candidate miss: all ``y == 0``) receive true label ``-1`` so Top-1 accuracy
    honestly reflects candidate misses.

    Returns ``(y_true_obs, y_pred_obs)`` of shape ``(n_unique_obs,)``.
    """
    n = len(y)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    classes_arr = np.asarray(classes)
    pos_matches = np.flatnonzero(classes_arr == 1)
    pos_col = int(pos_matches[0]) if len(pos_matches) > 0 else (1 if proba.shape[1] > 1 else 0)

    g_arr = np.asarray(groups, dtype=np.int64)
    o_arr = np.asarray(obs_idx, dtype=np.int64)
    a_arr = np.asarray(arc_id, dtype=np.int64)
    y_arr = np.asarray(y, dtype=np.int64)

    # Namespace arc_id per trip so arc 5 in trip A != arc 5 in trip B.
    arc_stride = int(a_arr.max()) + 1 if len(a_arr) > 0 else 1
    arc_stride = max(arc_stride, 1)
    unique_arc = g_arr * arc_stride + a_arr

    order = np.lexsort((o_arr, g_arr))
    g_s = g_arr[order]
    o_s = o_arr[order]
    a_s = unique_arc[order]
    y_s = y_arr[order]
    p_s = proba[order, pos_col]

    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (g_s[1:] != g_s[:-1]) | (o_s[1:] != o_s[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)

    n_obs = len(starts)
    y_true_obs = np.full(n_obs, -1, dtype=np.int64)
    y_pred_obs = np.empty(n_obs, dtype=np.int64)

    for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        best = int(np.argmax(p_s[s:e]))
        y_pred_obs[i] = a_s[s + best]
        pos = np.flatnonzero(y_s[s:e] == 1)
        if len(pos) > 0:
            y_true_obs[i] = a_s[s + int(pos[0])]

    return y_true_obs, y_pred_obs


def cross_validate_model(
    model,
    X,
    y,
    groups,
    n_splits: int = 5,
    seed: int = 42,
    *,
    obs_idx: np.ndarray | None = None,
    arc_id: np.ndarray | None = None,
    verbose: bool = False,
) -> CVResult:
    """Fit `model` on each training fold and score it on the held-out fold.

    Record the fold-to-fold std of macro F1: 'stability across validation folds'
    is an explicit selection criterion in the brief.

    When ``obs_idx`` and ``arc_id`` are supplied (the standard PathSnap contract),
    Top-1 accuracy and macro F1 are computed at the **observation** level after
    taking argmax over each observation's candidate scores (grouping by
    ``(group, obs_idx)``). Log loss is always computed at the candidate row level
    via ``safe_log_loss``.
    """
    fold_results: list[FoldResult] = []
    model_name = getattr(model, "name", type(model).__name__)

    for fold_i, (train_idx, valid_idx) in enumerate(
        group_kfold(X, y, groups, n_splits=n_splits, seed=seed)
    ):
        t0 = time.perf_counter()
        model_clone = copy.deepcopy(model)
        model_clone.fit(X[train_idx], y[train_idx], groups=groups[train_idx])

        proba = model_clone.predict_proba(X[valid_idx])
        y_valid = y[valid_idx]

        # --- metrics ---
        if obs_idx is not None and arc_id is not None:
            y_true_obs, y_pred_obs = observation_level_predictions(
                y_valid,
                proba,
                model_clone.classes_,
                groups[valid_idx],
                obs_idx[valid_idx],
                arc_id[valid_idx],
            )
            fold_top1 = top1_accuracy(y_true_obs, y_pred_obs)
            fold_macro_f1 = macro_f1(y_true_obs, y_pred_obs)
        else:
            preds = model_clone.predict(X[valid_idx])
            fold_top1 = top1_accuracy(y_valid, preds)
            fold_macro_f1 = macro_f1(y_valid, preds)

        # Log loss is always at the candidate (row) level
        fold_ll = safe_log_loss(y_valid, proba, model_clone.classes_)
        fold_zero_p = zero_true_probability_fraction(
            y_valid, proba, model_clone.classes_
        )

        fold_results.append(
            FoldResult(
                fold=fold_i,
                n_train=len(train_idx),
                n_valid=len(valid_idx),
                macro_f1=fold_macro_f1,
                top1_accuracy=fold_top1,
                log_loss=fold_ll,
                zero_prob_fraction=fold_zero_p,
            )
        )
        if verbose:
            elapsed = time.perf_counter() - t0
            ll_str = f"{fold_ll:.4f}" if fold_ll is not None else "N/A"
            print(
                f"    [{model_name}] fold {fold_i + 1}/{n_splits} "
                f"(train={len(train_idx):,}, valid={len(valid_idx):,}): "
                f"macro_f1={fold_macro_f1:.4f}, top1={fold_top1:.4f}, "
                f"ll={ll_str} ({elapsed:.1f}s)",
                flush=True,
            )

    f1_scores = np.array([f.macro_f1 for f in fold_results])
    top1_scores = np.array([f.top1_accuracy for f in fold_results])
    ll_scores = [f.log_loss for f in fold_results]

    mean_ll: float | None = None
    if all(v is not None for v in ll_scores):
        mean_ll = float(np.mean(ll_scores))

    return CVResult(
        model_name=model_name,
        folds=fold_results,
        mean_macro_f1=float(f1_scores.mean()),
        std_macro_f1=float(f1_scores.std()),
        mean_top1_accuracy=float(top1_scores.mean()),
        mean_log_loss=mean_ll,
    )
