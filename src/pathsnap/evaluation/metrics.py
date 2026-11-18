"""Shared metrics.

WHY THIS FILE EXISTS
--------------------
The brief requires all five models to be compared on identical metrics. If each
workstream computes F1 its own way the comparison table is worthless, so the
metric implementations live here and nobody re-implements them.

READ THIS BEFORE TRUSTING THE NUMBERS
-------------------------------------
The brief does not define `route continuity` or `sequence-level performance`.
The definitions below are *one* defensible choice. The team must confirm they
match what you intend to claim in the report, and the report must state the
definition explicitly. Do not report a number without being able to say exactly
how it was computed.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Observation-level Top-1 Accuracy: best candidate matches the reference."""
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over all classes present in y_true. This is the primary metric."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def safe_log_loss(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float | None:
    """Mean negative log-likelihood of the true class, or None when undefined.

    Why not sklearn.metrics.log_loss: sklearn clips probabilities to ~1e-15, so a
    model that assigns exactly zero probability to the truth reports a finite
    score instead of infinity. That silently launders a broken model into a merely
    bad-looking one. Here, if any observation's true class was given zero
    probability, the log loss is genuinely infinite and we return None so the
    comparison table prints N/A with an explanation.

    Also returns None if y_true contains a label that is not in `classes`
    (i.e. the model never saw that class at all).
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    classes = np.asarray(classes)

    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError("proba shape does not match classes")

    index = {label: i for i, label in enumerate(classes.tolist())}
    try:
        cols = np.fromiter((index[v] for v in y_true.tolist()), dtype=np.int64, count=len(y_true))
    except KeyError:
        return None
    if len(y_true) == 0:
        return None

    p_true = proba[np.arange(len(y_true)), cols]
    if (p_true <= 0.0).any():
        return None
    return float(-np.mean(np.log(p_true)))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=classes)


def candidate_miss_rate(
    cand_arc_per_obs: list[np.ndarray] | np.ndarray, true_arc_per_obs: np.ndarray
) -> float:
    """Fraction of observations whose reference arc is NOT in the candidate set.

    This is a property of candidate generation, not of the classifier: if the
    true arc was never offered, no model can possibly get that observation
    right. Report it once, next to every model's accuracy, so a low score is
    not blamed on the classifier when candidate generation is at fault.
    """
    true_arc_per_obs = np.asarray(true_arc_per_obs)
    hits = [
        int(true in set(np.asarray(c).tolist()))
        for c, true in zip(cand_arc_per_obs, true_arc_per_obs, strict=True)
    ]
    return float(1.0 - (np.mean(hits) if hits else 0.0))


def route_continuity(pred_arc_ids: np.ndarray, arcs: np.ndarray) -> float:
    """Fraction of consecutive predicted arcs that are connected in the network.

    Two predicted arcs count as connected when they are the same arc, or when
    the first arc's head node equals the second arc's tail node. A higher value
    means a more physically drivable predicted route.

    `arcs` is the (n_arcs, 2) endpoint table from RoadNetwork.
    """
    pred = np.asarray(pred_arc_ids, dtype=np.int64)
    if len(pred) < 2:
        return 1.0
    a, b = pred[:-1], pred[1:]
    same = a == b
    connected = arcs[a, 1] == arcs[b, 0]
    # Treat a reversed traversal (b -> a) as connected too; document if you change this.
    reversed_ok = arcs[a, 0] == arcs[b, 1]
    ok = same | connected | reversed_ok
    return float(ok.mean())


def sequence_exact_match(pred_arc_ids: np.ndarray, ref_arc_ids: np.ndarray) -> float:
    """1.0 if the predicted and reference arc sequences are identical, else 0.0.

    Sequence-level performance is a single number here because each trip is one
    sequence. Report it per trip, then the mean across trips.
    """
    pred, ref = np.asarray(pred_arc_ids), np.asarray(ref_arc_ids)
    return float(len(pred) == len(ref) and np.array_equal(pred, ref))


def sequence_f1(pred_arc_ids: np.ndarray, ref_arc_ids: np.ndarray) -> float:
    """Token-level F1 of the predicted arc sequence against the reference.

    Useful when exact match is always 0 (which is common early on) but you still
    want a sequence-level signal that is not identical to observation accuracy.
    """
    pred, ref = np.asarray(pred_arc_ids), np.asarray(ref_arc_ids)
    if len(pred) == 0 and len(ref) == 0:
        return 1.0
    if len(pred) == 0 or len(ref) == 0:
        return 0.0
    # Longest common subsequence -> order-sensitive overlap.
    dp = np.zeros((len(pred) + 1, len(ref) + 1), dtype=np.int32)
    for i in range(1, len(pred) + 1):
        for j in range(1, len(ref) + 1):
            dp[i, j] = dp[i - 1, j - 1] + 1 if pred[i - 1] == ref[j - 1] else max(dp[i - 1, j], dp[i, j - 1])
    lcs = int(dp[-1, -1])
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 0.0 if lcs == 0 else float(2 * precision * recall / (precision + recall))
