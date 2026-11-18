"""Reference-arc assignment: turn a `.route` sequence into a per-observation label.

OWNER: Person 1
STATUS: implemented, but READ THE ASSUMPTION SECTION - it is a real limitation
        and the report must state it.

THE PROBLEM
-----------
`.route` is a chronological list of arc ids the vehicle traversed. It is NOT one
label per GPS observation. To train a candidate classifier we need, for each GPS
observation, the single arc the vehicle was on at that moment.

ASSUMPTION USED HERE
--------------------
For each observation we take the route arc whose geometry is closest to the
observation (perpendicular distance, in metres). Rationale: the reference route
is the travelled path, so the true position is on it, and the nearest point on
that path is the most likely position. This is the standard way to derive
point-level labels from a path-level reference.

Where it can be wrong, and you must say so in the report:
- A route that revisits the same arc, or passes a parallel segment twice, can
  produce a label from the wrong pass. `n_backwards` counts observations whose
  chosen route position moved *earlier* in the route than the previous
  observation's, which is the observable symptom of this.
- Exact-match accuracy against these labels understates a geographically
  sensible prediction on an adjacent arc.

If `n_backwards` is a large fraction of a trip, that trip's labels are suspect.
We report the number per trip rather than silently trusting the alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathsnap.geo.distance import LocalProjection, point_to_segment_distance
from pathsnap.io.schema import CandidateSet, Trip


@dataclass(frozen=True)
class ReferenceArcs:
    """Per-observation reference arc, derived from the trip's `.route`."""

    trip_id: str
    arc_per_obs: np.ndarray     # int64 (n_obs,), -1 where the route is missing/empty
    dist_per_obs: np.ndarray    # float64 (n_obs,), distance from obs to that arc
    route_pos: np.ndarray       # int64 (n_obs,), index into trip.route
    n_backwards: int

    @property
    def resolved(self) -> np.ndarray:
        return self.arc_per_obs >= 0

    @property
    def coverage(self) -> float:
        return float(self.resolved.mean()) if len(self.arc_per_obs) else 0.0


def _arc_endpoints_m(network, proj: LocalProjection):
    """Arc endpoints in the trip's local metric frame: (ax, ay, bx, by)."""
    nx, ny = proj.forward(network.nodes[:, 0], network.nodes[:, 1])
    ax, ay = nx[network.arcs[:, 0]], ny[network.arcs[:, 0]]
    bx, by = nx[network.arcs[:, 1]], ny[network.arcs[:, 1]]
    return ax, ay, bx, by


def assign_reference_arcs(trip: Trip, chunk: int = 512, proj: LocalProjection | None = None) -> ReferenceArcs:
    """Assign each observation the closest arc from the trip's reference route.

    Chunked over observations so a 19k-observation trip with a 2k-arc route does
    not materialise a 19k x 2k distance matrix (~340 MB).
    """
    n_obs = len(trip.track)
    route = trip.route

    if route is None or len(route) == 0:
        return ReferenceArcs(
            trip_id=trip.trip_id,
            arc_per_obs=np.full(n_obs, -1, dtype=np.int64),
            dist_per_obs=np.full(n_obs, np.inf),
            route_pos=np.full(n_obs, -1, dtype=np.int64),
            n_backwards=0,
        )

    proj = proj or LocalProjection.from_points(trip.track.lon, trip.track.lat)
    ax_all, ay_all, bx_all, by_all = _arc_endpoints_m(trip.network, proj)
    ax, ay, bx, by = ax_all[route], ay_all[route], bx_all[route], by_all[route]

    px, py = proj.forward(trip.track.lon, trip.track.lat)

    arc_per_obs = np.empty(n_obs, dtype=np.int64)
    dist_per_obs = np.empty(n_obs, dtype=np.float64)
    route_pos = np.empty(n_obs, dtype=np.int64)

    for start in range(0, n_obs, chunk):
        stop = min(start + chunk, n_obs)
        d, _ = point_to_segment_distance(
            px[start:stop, None],
            py[start:stop, None],
            ax[None, :],
            ay[None, :],
            bx[None, :],
            by[None, :],
        )
        best = d.argmin(axis=1)
        route_pos[start:stop] = best
        dist_per_obs[start:stop] = d[np.arange(stop - start), best]
        arc_per_obs[start:stop] = route[best]

    n_backwards = int((np.diff(route_pos) < 0).sum())

    return ReferenceArcs(
        trip_id=trip.trip_id,
        arc_per_obs=arc_per_obs,
        dist_per_obs=dist_per_obs,
        route_pos=route_pos,
        n_backwards=n_backwards,
    )


def split_candidates_by_obs(candidates: CandidateSet, n_obs: int) -> list[np.ndarray]:
    """Return a list of length n_obs of candidate arc ids per observation.

    Uses the fact that a CandidateSet is sorted by obs_idx (CONTRACTS C2), so this
    is a set of contiguous slices rather than a groupby.
    """
    order = np.argsort(candidates.obs_idx, kind="stable")
    obs_sorted = candidates.obs_idx[order]
    arc_sorted = candidates.arc_id[order]

    counts = np.bincount(obs_sorted, minlength=n_obs)
    out: list[np.ndarray] = []
    pos = 0
    for count in counts:
        out.append(arc_sorted[pos : pos + count])
        pos += count
    return out


def candidate_miss_rate(reference: ReferenceArcs, candidates: CandidateSet, n_obs: int) -> float:
    """Fraction of observations whose reference arc is absent from the candidates.

    Reuses the shared definition in `pathsnap.evaluation.metrics` so P1 and P2
    cannot drift apart on what "miss rate" means.
    """
    from pathsnap.evaluation.metrics import candidate_miss_rate as _miss

    per_obs = split_candidates_by_obs(candidates, n_obs)
    valid = reference.resolved
    if not valid.any():
        return float("nan")
    return _miss([per_obs[i] for i in np.flatnonzero(valid)], reference.arc_per_obs[valid])


@dataclass
class LabelSummary:
    """Aggregate label quality across trips - goes straight into the report."""

    n_trips: int = 0
    n_observations: int = 0
    coverage: float = 0.0
    reference_miss_rate: float = 0.0
    backwards_fraction: float = 0.0
    median_obs_to_arc_m: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_trips": self.n_trips,
            "n_observations": self.n_observations,
            "reference_arc_coverage": round(self.coverage, 4),
            "candidate_miss_rate": round(self.reference_miss_rate, 4),
            "backwards_alignment_fraction": round(self.backwards_fraction, 4),
            "median_obs_to_reference_arc_m": round(self.median_obs_to_arc_m, 2),
        }

    def __str__(self) -> str:
        return (
            f"trips={self.n_trips} obs={self.n_observations:,} "
            f"coverage={self.coverage:.4f} miss_rate={self.reference_miss_rate:.4f} "
            f"backwards={self.backwards_fraction:.4f} "
            f"median_residual={self.median_obs_to_arc_m:.1f}m"
        )
