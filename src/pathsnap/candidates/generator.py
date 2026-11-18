"""Candidate generation: for each GPS observation, propose the road arcs it may be on.

OWNER: Person 1
STATUS: implemented (spatial index + per-trip projection).

WHY A SPATIAL INDEX
-------------------
Measured facts (see docs/DATA_FACTS.md): the largest training network has 307,392
arcs and the largest trip has 19,313 observations. Comparing every observation
against every arc is ~6e9 point-to-segment evaluations for that one trip. Instead
we bucket arcs into a uniform grid of ~100 m cells by bounding box and only
evaluate the arcs in the observation's own cell.

Correctness note: an arc inserted by *bounding box* can be slightly outside the
radius but is never missing from the cell of a point within the radius of it. So
the grid never loses a true candidate - the radius filter afterwards may drop
false positives, which is exactly what we want.

A handful of arcs are much longer than a cell (max observed: 9.3 km, p99: 289 m).
Those are not binned (their bounding box would touch tens of thousands of cells);
they are evaluated against every observation in the trip. There are very few of
them, so this stays cheap and, unlike approximate rasterisation, cannot miss a
candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from pathsnap.geo.distance import LocalProjection, point_to_segment_distance
from pathsnap.io.schema import CandidateSet, Trip

#: An arc whose bounding box covers more cells than this is not binned.
MAX_CELLS_PER_ARC = 4096


@dataclass
class CandidateConfig:
    """Candidate-generation settings. Changing these changes the learning problem."""

    radius_m: float = 50.0
    max_candidates: int = 10
    cell_m: float | None = None        # defaults to 2 * radius_m
    kd_fallback_k: int = 8             # neighbours used when a cell is empty

    def resolved_cell_m(self) -> float:
        return float(self.cell_m if self.cell_m is not None else max(2.0 * self.radius_m, 25.0))


@dataclass
class _Grid:
    """Binned arcs plus the long-arc list that is always evaluated."""

    keys: np.ndarray          # sorted unique cell keys that contain arcs
    starts: np.ndarray        # start offset of each key's arc block in `arcs`
    counts: np.ndarray        # number of arcs in each key's block
    arcs: np.ndarray          # arc ids, grouped by key
    long_arcs: np.ndarray
    x0: float
    y0: float
    cell: float
    width: int
    _cache: dict = field(default_factory=dict)

    def cell_key(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx = np.floor((x - self.x0) / self.cell).astype(np.int64)
        gy = np.floor((y - self.y0) / self.cell).astype(np.int64)
        return gy * self.width + gx

    def arcs_in_key(self, key: int) -> np.ndarray:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        i = int(np.searchsorted(self.keys, key))
        if i < len(self.keys) and self.keys[i] == key:
            block = self.arcs[self.starts[i] : self.starts[i] + self.counts[i]]
        else:
            block = np.empty(0, dtype=np.int64)
        self._cache[key] = block
        return block


def _build_grid(ax, ay, bx, by, x0: float, y0: float, cell: float) -> _Grid:
    n_arcs = len(ax)
    cx0 = np.clip(np.floor((np.minimum(ax, bx) - x0) / cell).astype(np.int64), 0, None)
    cx1 = np.clip(np.floor((np.maximum(ax, bx) - x0) / cell).astype(np.int64), 0, None)
    cy0 = np.clip(np.floor((np.minimum(ay, by) - y0) / cell).astype(np.int64), 0, None)
    cy1 = np.clip(np.floor((np.maximum(ay, by) - y0) / cell).astype(np.int64), 0, None)

    width = int(cx1.max()) + 1 if n_arcs else 1
    counts = (cx1 - cx0 + 1) * (cy1 - cy0 + 1)
    binned = counts <= MAX_CELLS_PER_ARC
    long_arcs = np.flatnonzero(~binned)

    arc_ids = np.flatnonzero(binned)
    c = counts[arc_ids]
    repeat = np.repeat(arc_ids, c)
    offsets = np.repeat(np.cumsum(c) - c, c)
    local = np.arange(len(repeat)) - offsets
    row_width = (cx1 - cx0 + 1)[repeat]
    cx = cx0[repeat] + (local % row_width)
    cy = cy0[repeat] + (local // row_width)
    keys = cy * width + cx

    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    arcs_sorted = repeat[order]
    unique_keys, starts, block_counts = np.unique(
        keys_sorted, return_index=True, return_counts=True
    )

    return _Grid(
        keys=unique_keys,
        starts=starts,
        counts=block_counts,
        arcs=arcs_sorted,
        long_arcs=long_arcs,
        x0=x0,
        y0=y0,
        cell=cell,
        width=width,
    )


def generate_candidates(
    trip: Trip,
    radius_m: float = 50.0,
    max_candidates: int = 10,
    config: CandidateConfig | None = None,
    proj: LocalProjection | None = None,
) -> CandidateSet:
    """Return a CandidateSet for one trip, sorted by observation then distance.

    Every observation is guaranteed at least one candidate (the nearest arc is
    always kept, even if it is further than `radius_m`). The count of observations
    that needed that fallback is exposed in ``meta``.
    """
    cfg = config or CandidateConfig(radius_m=radius_m, max_candidates=max_candidates)
    n_obs = len(trip.track)
    net = trip.network

    proj = proj or LocalProjection.from_points(trip.track.lon, trip.track.lat)
    px, py = proj.forward(trip.track.lon, trip.track.lat)

    node_x, node_y = proj.forward(net.nodes[:, 0], net.nodes[:, 1])
    ax, ay = node_x[net.arcs[:, 0]], node_y[net.arcs[:, 0]]
    bx, by = node_x[net.arcs[:, 1]], node_y[net.arcs[:, 1]]

    if net.n_arcs == 0 or n_obs == 0:
        empty = np.empty(0, dtype=np.int64)
        return CandidateSet(
            trip_id=trip.trip_id,
            obs_idx=empty,
            arc_id=empty,
            dist_m=np.empty(0),
            meta={"n_obs": n_obs, "n_obs_without_candidates": n_obs, "config": cfg},
        )

    cell = cfg.resolved_cell_m()
    x0 = float(min(px.min(), ax.min(), bx.min()))
    y0 = float(min(py.min(), ay.min(), by.min()))
    grid = _build_grid(ax, ay, bx, by, x0, y0, cell)

    obs_keys = grid.cell_key(px, py)
    unique_obs_keys, inverse = np.unique(obs_keys, return_inverse=True)

    # Collect candidate arc ids per distinct observation cell.
    per_key: list[np.ndarray] = [grid.arcs_in_key(int(k)) for k in unique_obs_keys]
    if grid.long_arcs.size:
        per_key = [np.union1d(block, grid.long_arcs) for block in per_key]

    # `per_key` is indexed by position in `unique_obs_keys`; `inverse` maps each
    # observation to its position, so per-observation blocks are per_key[inverse[i]].
    key_sizes = np.array([b.size for b in per_key], dtype=np.int64)
    obs_parts: list[np.ndarray] = []
    arc_parts: list[np.ndarray] = []

    nonempty_obs = np.flatnonzero(key_sizes[inverse] > 0)
    if nonempty_obs.size:
        obs_parts.append(np.repeat(nonempty_obs, key_sizes[inverse[nonempty_obs]]))
        arc_parts.append(np.concatenate([per_key[inverse[i]] for i in nonempty_obs]))

    empty_obs = np.flatnonzero(key_sizes[inverse] == 0)
    if empty_obs.size:
        # Cell empty (observation outside the network footprint) -> nearest arcs by
        # midpoint, so the "at least one candidate" guarantee still holds.
        mids = np.column_stack([(ax + bx) / 2.0, (ay + by) / 2.0])
        tree = cKDTree(mids)
        k = int(min(cfg.kd_fallback_k, net.n_arcs))
        _, nn = tree.query(np.column_stack([px[empty_obs], py[empty_obs]]), k=k)
        nn = np.atleast_2d(nn)
        obs_parts.append(np.repeat(empty_obs, k))
        arc_parts.append(nn.reshape(-1))

    obs_rep = np.concatenate(obs_parts) if obs_parts else np.empty(0, dtype=np.int64)
    arc_rep = np.concatenate(arc_parts) if arc_parts else np.empty(0, dtype=np.int64)

    dist, t_along = point_to_segment_distance(
        px[obs_rep], py[obs_rep], ax[arc_rep], ay[arc_rep], bx[arc_rep], by[arc_rep]
    )

    # Order by observation, then by distance, then rank within each observation.
    order = np.lexsort((dist, obs_rep))
    obs_s, arc_s, dist_s, t_s = obs_rep[order], arc_rep[order], dist[order], t_along[order]

    per_obs_counts = np.bincount(obs_s, minlength=n_obs)
    rank = np.arange(len(obs_s)) - np.repeat(np.cumsum(per_obs_counts) - per_obs_counts, per_obs_counts)

    keep = ((dist_s <= cfg.radius_m) & (rank < cfg.max_candidates)) | (rank == 0)
    obs_s, arc_s, dist_s, t_s = obs_s[keep], arc_s[keep], dist_s[keep], t_s[keep]
    rank = rank[keep]

    n_obs_without = int(n_obs - np.unique(obs_s).size) if obs_s.size else n_obs
    n_beyond_radius = int(((dist_s > cfg.radius_m) & (rank == 0)).sum())

    return CandidateSet(
        trip_id=trip.trip_id,
        obs_idx=obs_s.astype(np.int64),
        arc_id=arc_s.astype(np.int64),
        dist_m=dist_s.astype(np.float64),
        meta={
            "n_obs": n_obs,
            "n_obs_without_candidates": n_obs_without,
            "n_obs_nearest_beyond_radius": n_beyond_radius,
            "radius_m": cfg.radius_m,
            "max_candidates": cfg.max_candidates,
            "cell_m": cell,
            "n_grid_cells_with_arcs": int(grid.keys.size),
            "n_long_arcs_binned_separately": int(grid.long_arcs.size),
            "t_along": t_s.astype(np.float64),
            "projection": proj,
            "config": cfg,
        },
    )
