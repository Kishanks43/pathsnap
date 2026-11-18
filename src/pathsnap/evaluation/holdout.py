"""Held-out (student-test) evaluation shared by every model runner.

OWNER: Person 2 (moved here from scripts/run_p2.py so Persons 3 and 4 score models
with exactly the same code).

Two functions:

- ``evaluate_on_test``: observation-level Top-1, macro F1, log loss and inference
  cost for one model on one partition.
- ``mean_route_continuity``: per-trip route continuity from the predicted arc per
  observation. Only the road network is read, never the reference route.
"""

from __future__ import annotations

import time

import numpy as np

from pathsnap.evaluation.cv import observation_level_predictions, zero_true_probability_fraction
from pathsnap.evaluation.metrics import macro_f1, safe_log_loss, top1_accuracy


def evaluate_on_test(model, test, trust_mask: np.ndarray) -> tuple[dict, np.ndarray]:
    """Observation-level evaluation on a held-out partition.

    Top-1 accuracy and macro F1 are computed after taking argmax over each
    observation's candidate scores, never over candidate rows. Log loss is
    candidate-row level via ``safe_log_loss``.
    """
    X = test.X[trust_mask]
    y = test.y[trust_mask]
    groups = test.groups[trust_mask]
    obs_idx = test.obs_idx[trust_mask]
    arc_id = test.arc_id[trust_mask]

    t0 = time.perf_counter()
    proba = model.predict_proba(X)
    elapsed = time.perf_counter() - t0

    y_true_obs, y_pred_obs = observation_level_predictions(
        y, proba, model.classes_, groups, obs_idx, arc_id
    )
    n_obs = len(y_true_obs)
    ll = safe_log_loss(y, proba, model.classes_)
    zero_p = zero_true_probability_fraction(y, proba, model.classes_)

    metrics = {
        "top1_accuracy": top1_accuracy(y_true_obs, y_pred_obs),
        "macro_f1": macro_f1(y_true_obs, y_pred_obs),
        "log_loss": ll,
        "zero_true_probability_fraction": zero_p,
        "n_observations": int(n_obs),
        "n_candidate_rows": len(y),
        "inference_seconds": float(elapsed),
        "inference_ms_per_obs": (elapsed * 1000.0) / n_obs if n_obs else None,
        "inference_ms_per_row": (elapsed * 1000.0) / len(y) if len(y) else None,
    }
    return metrics, y_pred_obs


def mean_route_continuity(test, trust_mask: np.ndarray, y_pred_obs: np.ndarray) -> dict:
    """Mean route continuity per held-out trip.

    ``y_pred_obs`` comes from ``observation_level_predictions``, which groups by
    ``(trip, obs_idx)`` in ascending order, so it can be split back into trips by
    the number of distinct trusted observations per trip.
    """
    from pathsnap.config import STUDENT_TEST_DIR
    from pathsnap.evaluation.metrics import route_continuity
    from pathsnap.io.loaders import load_network

    trusted_obs = test.obs_idx[trust_mask]
    groups = test.groups[trust_mask]
    unique_groups = np.unique(groups)
    # observation_level_predictions namespaces arc ids as group*stride + arc so that
    # identical arc ids in different trips never merge when computing macro F1.
    # Route continuity is per trip, so undo that namespace here.
    arc_stride = int(test.arc_id[trust_mask].max()) + 1

    per_trip: dict[str, float] = {}
    start = 0
    for group_idx in unique_groups.tolist():
        n_obs_trip = int(np.unique(trusted_obs[groups == group_idx]).size)
        end = start + n_obs_trip
        pred = y_pred_obs[start:end] % arc_stride
        start = end

        trip_id = test.trip_ids[group_idx]
        arc_path = STUDENT_TEST_DIR / trip_id / f"{trip_id}.arcs"
        if not arc_path.exists():
            continue
        network = load_network(arc_path, trip_id)
        per_trip[trip_id] = route_continuity(pred, network.arcs)

    scores = list(per_trip.values())
    return {
        "mean_route_continuity": float(np.mean(scores)) if scores else None,
        "per_trip": per_trip,
    }
