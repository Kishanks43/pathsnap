"""Feature engineering: turn a CandidateSet into the matrix every model consumes.

OWNER: Person 1
STATUS: implemented - SCHEMA v1 IS FROZEN. Add columns at the end only.

THE ONE RULE
------------
`build_feature_matrix` returns a plain float64 (n_pairs, n_features) array with
a fixed, ordered column list. Every model in the project sees exactly this
matrix, so all five models are compared on identical inputs - which is what makes
the comparison table in the report legitimate.

LEAKAGE RULES ENFORCED HERE
---------------------------
- No feature reads the reference route. `y` and `label_residual_m` are produced
  from it, but neither is ever inside X.
- Sequential features use only the *previous observation* and the road network.
  The "previous arc" is the previous observation's nearest candidate, not its
  reference arc - using the reference would leak the answer one step back.
- Nothing is normalised across trips. Trips are in different cities
  (docs/DATA_FACTS.md), so every quantity is in metres / seconds / degrees and
  carries its own units. Scaling belongs inside each model's pipeline, fitted on
  training folds only.

Y IS BINARY, ONE ROW PER CANDIDATE
----------------------------------
Each candidate pair is labelled 1 if this arc is the observation's reference arc,
else 0. So `y.shape == (n_pairs,)` and a model learns "is this candidate the
right one?". Per-observation Top-1 accuracy is then argmax over each
observation's candidate scores; macro F1 must be computed at the *observation*
level after that argmax, not over candidate rows (candidate rows are ~90%
negative, which would inflate macro F1 misleadingly).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pathsnap.config import ARTIFACTS_DIR
from pathsnap.geo.distance import LocalProjection, haversine_m, point_to_segment_distance
from pathsnap.io.schema import CandidateSet, Trip
from pathsnap.labels import ReferenceArcs

SCHEMA_VERSION = 1

#: Frozen feature columns. Order is part of the contract (docs/CONTRACTS.md C3).
FEATURE_COLUMNS: list[str] = [
    "dist_m",             # perpendicular distance observation -> candidate arc
    "log1p_dist",
    "dist_rank",          # 0 = nearest candidate for this observation
    "dist_ratio",         # dist_m / (min dist for this observation + 1)
    "arc_len_m",
    "t_along",            # fraction along the arc [0, 1] of the closest point
    "tail_out_degree",    # out-degree of the arc's from-node
    "head_in_degree",     # in-degree of the arc's to-node
    "has_reverse",        # 1 if the reverse arc also exists
    "arc_bearing_deg",    # 0..360, clockwise from local +y (north-ish)
    "has_prev",           # 0 on the first observation of a trip
    "prev_dist_m",        # previous observation's distance to THIS arc
    "delta_dist",         # dist_m - prev_dist_m
    "connected_prev",     # 1 if previous best arc's head == this arc's tail
    "step_m",             # distance between previous and current observation
    "dt_s",               # time between observations, clipped to [0, 60]
    "speed_mps",          # step_m / dt_s, clipped to [0, 40]
    "heading_change_deg", # |step bearing - arc bearing|, 0..180
    "seq_pos",            # observation index / (n_obs - 1)
]


def _bearing_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Clockwise bearing in degrees. North (increasing y) is 0."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def _cumulative_counts(sorted_obs: np.ndarray, n: int) -> np.ndarray:
    """Per-pair index of the first row of each observation's candidate block."""
    counts = np.bincount(sorted_obs, minlength=n)
    return np.repeat(np.cumsum(counts) - counts, counts)


def build_feature_matrix(
    trip: Trip,
    candidates: CandidateSet,
    reference: ReferenceArcs | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Return (X, y, meta).

    X : float64, shape (n_pairs, len(FEATURE_COLUMNS))
    y : int64 {0,1}, shape (n_pairs,), or None when the trip has no reference route
    meta : dict with {"trip_id", "obs_idx", "arc_id", "columns", "n_obs",
                      "label_residual_m", "schema_version"}
    """
    n_obs = len(trip.track)
    obs_idx = candidates.obs_idx
    arc_id = candidates.arc_id
    n_pairs = obs_idx.size

    if n_pairs == 0:
        empty = np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float64)
        return empty, (np.empty(0, dtype=np.int64) if reference is not None else None), {
            "trip_id": trip.trip_id,
            "obs_idx": obs_idx,
            "arc_id": arc_id,
            "columns": list(FEATURE_COLUMNS),
            "n_obs": n_obs,
            "label_residual_m": np.empty(0),
            "schema_version": SCHEMA_VERSION,
        }

    proj: LocalProjection = candidates.meta.get("projection") or LocalProjection.from_points(
        trip.track.lon, trip.track.lat
    )
    net = trip.network
    px, py = proj.forward(trip.track.lon, trip.track.lat)
    node_x, node_y = proj.forward(net.nodes[:, 0], net.nodes[:, 1])
    ax, ay = node_x[net.arcs[:, 0]], node_y[net.arcs[:, 0]]
    bx, by = node_x[net.arcs[:, 1]], node_y[net.arcs[:, 1]]

    n_nodes = net.n_nodes
    out_degree = np.bincount(net.arcs[:, 0], minlength=n_nodes)
    in_degree = np.bincount(net.arcs[:, 1], minlength=n_nodes)

    keys = net.arcs[:, 0].astype(np.int64) * max(n_nodes, 1) + net.arcs[:, 1]
    rev_keys = net.arcs[:, 1].astype(np.int64) * max(n_nodes, 1) + net.arcs[:, 0]
    keys_sorted = np.sort(keys)
    pos = np.searchsorted(keys_sorted, rev_keys)
    has_reverse = (pos < len(keys_sorted)) & (keys_sorted[np.minimum(pos, len(keys_sorted) - 1)] == rev_keys)

    arc_len = net.arc_lengths_m()
    dist_m = candidates.dist_m.astype(np.float64)
    t_along = np.asarray(candidates.meta.get("t_along", np.zeros(n_pairs)), dtype=np.float64)

    # --- per-observation context -------------------------------------------------
    # Candidates are sorted by obs then distance (CONTRACTS C2): the first row of
    # each observation's block is its nearest candidate. That nearest arc is used
    # as the sequential "previous arc" proxy - never the reference arc, which
    # would leak the answer one step backwards.
    counts = np.bincount(obs_idx, minlength=n_obs)
    obs_starts = np.cumsum(counts) - counts          # one entry per observation
    pair_starts = np.repeat(obs_starts, counts)      # one entry per candidate row
    dist_rank = np.arange(n_pairs) - pair_starts

    present = counts > 0
    best_arc = np.full(n_obs, -1, dtype=np.int64)
    best_arc[present] = arc_id[obs_starts[present]]
    nearest_dist = np.full(n_obs, np.inf, dtype=np.float64)
    nearest_dist[present] = dist_m[obs_starts[present]]

    dist_ratio = dist_m / (nearest_dist[obs_idx] + 1.0)

    has_prev = (obs_idx > 0).astype(np.float64)
    prev_obs = np.maximum(obs_idx - 1, 0)
    prev_arc = best_arc[prev_obs]
    prev_valid = (obs_idx > 0) & (prev_arc >= 0)

    prev_dist, _ = point_to_segment_distance(
        px[prev_obs], py[prev_obs], ax[arc_id], ay[arc_id], bx[arc_id], by[arc_id]
    )
    prev_dist = np.where(prev_valid, prev_dist, dist_m)

    connected = np.zeros(n_pairs, dtype=np.float64)
    if prev_valid.any():
        connected[prev_valid] = (
            net.arcs[np.maximum(prev_arc[prev_valid], 0), 1] == net.arcs[arc_id[prev_valid], 0]
        ).astype(np.float64)

    step_m = np.zeros(n_pairs, dtype=np.float64)
    dt_s = np.zeros(n_pairs, dtype=np.float64)
    heading_change = np.zeros(n_pairs, dtype=np.float64)

    if prev_valid.any():
        step_m_all = haversine_m(
            trip.track.lon[prev_obs], trip.track.lat[prev_obs],
            trip.track.lon[obs_idx], trip.track.lat[obs_idx],
        )
        step_m[prev_valid] = step_m_all[prev_valid]

        dt_all = np.clip(
            np.abs(trip.track.timestamp[obs_idx] - trip.track.timestamp[prev_obs]), 0.0, 60.0
        )
        dt_s[prev_valid] = dt_all[prev_valid]

        step_bearing = _bearing_deg(px[obs_idx] - px[prev_obs], py[obs_idx] - py[prev_obs])
        arc_bearing = _bearing_deg(bx[arc_id] - ax[arc_id], by[arc_id] - ay[arc_id])
        diff = np.abs(((step_bearing - arc_bearing + 180.0) % 360.0) - 180.0)
        heading_change[prev_valid] = diff[prev_valid]

    speed = np.zeros(n_pairs, dtype=np.float64)
    nonzero = dt_s > 0
    speed[nonzero] = np.clip(step_m[nonzero] / dt_s[nonzero], 0.0, 40.0)

    seq_pos = (obs_idx / max(n_obs - 1, 1)).astype(np.float64)

    arc_bearing_all = _bearing_deg(bx[arc_id] - ax[arc_id], by[arc_id] - ay[arc_id])

    X = np.column_stack(
        [
            dist_m,
            np.log1p(dist_m),
            dist_rank.astype(np.float64),
            dist_ratio,
            arc_len[arc_id],
            t_along,
            out_degree[net.arcs[arc_id, 0]].astype(np.float64),
            in_degree[net.arcs[arc_id, 1]].astype(np.float64),
            has_reverse[arc_id].astype(np.float64),
            arc_bearing_all,
            has_prev,
            prev_dist,
            dist_m - prev_dist,
            connected,
            step_m,
            dt_s,
            speed,
            heading_change,
            seq_pos,
        ]
    ).astype(np.float64, copy=False)

    if not np.isfinite(X).all():
        bad = ~np.isfinite(X).all(axis=0)
        raise ValueError(
            f"non-finite features in columns {[c for c, b in zip(FEATURE_COLUMNS, bad) if b]}"
        )

    y = None
    residual = np.zeros(n_pairs, dtype=np.float64)
    if reference is not None:
        ref_arc = reference.arc_per_obs[obs_idx]
        y = (ref_arc == arc_id).astype(np.int64)
        residual = reference.dist_per_obs[obs_idx]

    meta = {
        "trip_id": trip.trip_id,
        "obs_idx": obs_idx,
        "arc_id": arc_id,
        "columns": list(FEATURE_COLUMNS),
        "n_obs": n_obs,
        "label_residual_m": residual,
        "schema_version": SCHEMA_VERSION,
    }
    return X, y, meta


def feature_schema() -> dict:
    return {"version": SCHEMA_VERSION, "columns": list(FEATURE_COLUMNS)}


def write_feature_schema(path: Path | None = None) -> Path:
    """Persist the schema so the API can validate column order at load time."""
    path = Path(path) if path else ARTIFACTS_DIR / "feature_schema.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feature_schema(), indent=2), encoding="utf-8")
    return path
