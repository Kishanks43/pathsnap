"""Candidate generation: for each GPS observation, propose the road arcs it may be on.

OWNER: Person 1 (Data & Features)
STATUS: STUB - implement this first, the whole team is blocked on it.

WHY IT MATTERS
--------------
Map-matching is a classification problem over *candidates*, not over all arcs.
A GPS fix 8 m off a road could be on that road or on a parallel one, so you
offer a small candidate set per observation and let a model score them.
`candidate_miss_rate` (see pathsnap.evaluation.metrics) measures how often the
correct arc is missing from the set. If that number is bad, every model's
accuracy will look bad for reasons that have nothing to do with the models.

SUGGESTED APPROACH (decide, then justify in the report)
-------------------------------------------------------
1. Project the trip's observations and node coordinates into a local metric
   frame with pathsnap.geo.distance.LocalProjection.
2. For every arc, compute the perpendicular distance from the observation to
   the arc segment (point_to_segment_distance) - distance, not just endpoint
   distance, otherwise long arcs are unfairly penalised.
3. Keep arcs within `radius_m`, take the `max_candidates` nearest, and always
   keep at least one candidate so `obs_idx` covers every observation.
4. Record dist_m, and optionally the projected position along the arc, for the
   feature stage.

ACCEPTANCE CRITERIA
-------------------
- Every observation has >= 1 candidate (assert it).
- Candidate miss rate on train trips is reported and reasonably low.
- No dependency on reference routes: candidates must be generated from GPS +
  network geometry only, or you have leaked the answer into the pipeline.
"""

from __future__ import annotations

from pathsnap.io.schema import CandidateSet, Trip


def generate_candidates(
    trip: Trip,
    radius_m: float = 50.0,
    max_candidates: int = 10,
) -> CandidateSet:
    """Return a CandidateSet for one trip.

    Must produce parallel arrays obs_idx / arc_id / dist_m of equal length,
    sorted so that candidates for observation 0 come first (the HMM stage
    relies on this layout).
    """
    raise NotImplementedError(
        "Person 1: implement candidate generation. See the module docstring."
    )
