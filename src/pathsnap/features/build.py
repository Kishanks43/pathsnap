"""Feature engineering: turn a CandidateSet into the matrix every model consumes.

OWNER: Person 1 (Data & Features)
STATUS: STUB - Person 1 builds this; Persons 2 and 3 are blocked until the
        column contract is frozen.

THE ONE RULE
------------
`build_feature_matrix` must return a plain float64 (n_pairs, n_features) array
plus a list of column names. No pandas index tricks, no NaN unless a model can
handle it. Every model in the project sees exactly this matrix, so all five
models are compared on identical inputs - which is what makes the comparison
table in the report legitimate.

Leakage rule: a feature may use the *past* (previous observations) and the road
network, but never the reference route. Also never fit a scaler on student-test
or evaluator-test data - fit on train, transform on everything else.

FEATURE IDEAS (pick what you can justify, then measure it)
----------------------------------------------------------
Geometry of the candidate itself
  - perpendicular distance from observation to candidate arc, in metres
  - log1p(distance)  (compresses the long tail, helps linear models)
  - relative distance rank among this observation's candidates
  - distance to arc endpoint / fraction along the arc

Road-network context of the candidate
  - arc length (metres)
  - number of arcs sharing the tail node (junction degree)
  - whether the arc is one-way-ish (no reverse arc present)

Sequential context (this is what makes the classifier competitive with HMM)
  - distance from the *previous* observation's chosen candidate to this arc
  - connectivity flag: is this arc's tail reachable from the previous arc's head
  - heading change in degrees between previous observation step and arc bearing
  - observed speed between observations vs arc direction
  - time gap since previous observation
  - index in the trip (first observation has no history - decide how to encode
    that, e.g. a `has_prev` flag rather than NaN)

ACCEPTANCE CRITERIA
-------------------
- Returns (X, y, meta) where y is None when the trip has no reference route.
- `meta` carries obs_idx, arc_id and trip_id so predictions can be mapped back.
- Column names returned alongside, and dumped to
  experiments/results/feature_schema.json so the API loads a compatible model.
- Runs over all 80 train trips without error; time it and record it.
"""

from __future__ import annotations

import numpy as np

from pathsnap.io.schema import CandidateSet, Trip


def build_feature_matrix(
    trip: Trip, candidates: CandidateSet
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Return (X, y, meta).

    X : float64, shape (n_pairs, n_features)
    y : int64 arc ids, shape (n_pairs,), or None when no reference route exists
    meta : dict with at least keys {"trip_id", "obs_idx", "arc_id", "columns"}
    """
    raise NotImplementedError(
        "Person 1: implement feature engineering. Freeze column names early so "
        "Persons 2 and 3 can train against a stable schema."
    )
