"""Typed containers for the four raw file types.

These are the shared vocabulary of the whole project. Do not change the field
names or dtypes without updating docs/CONTRACTS.md, because four separate
workstreams depend on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Track:
    """GPS observations for one trip.

    Columns mirror the raw .track file:
        lon, lat  : float64, angular coordinates (degrees)
        timestamp : float64, seconds (nominally ~1 Hz, NOT guaranteed)
    """

    trip_id: str
    df: pd.DataFrame

    def __len__(self) -> int:
        return len(self.df)

    @property
    def lon(self) -> np.ndarray:
        return self.df["lon"].to_numpy()

    @property
    def lat(self) -> np.ndarray:
        return self.df["lat"].to_numpy()

    @property
    def timestamp(self) -> np.ndarray:
        return self.df["timestamp"].to_numpy()

    def time_deltas(self) -> np.ndarray:
        """Seconds between consecutive observations (length n-1)."""
        return np.diff(self.timestamp)


@dataclass(frozen=True)
class RoadNetwork:
    """Road network for one trip.

    nodes : (n_nodes, 2) float64 [lon, lat]. Row index IS the node_id.
    arcs  : (n_arcs, 2) int64 [from_node_id, to_node_id]. Row index IS the arc_id.

    Connectivity is directed: arc (u -> v) does not imply (v -> u).
    """

    trip_id: str
    nodes: np.ndarray
    arcs: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])

    def arc_endpoints(self, arc_ids: np.ndarray) -> np.ndarray:
        """Return (k, 2, 2) lon/lat array for the given arc ids."""
        arc_ids = np.asarray(arc_ids, dtype=np.int64)
        ends = self.arcs[arc_ids]                      # (k, 2) node ids
        return self.nodes[ends]                        # (k, 2, 2)

    def arc_lengths_m(self) -> np.ndarray:
        """Great-circle length of every arc, in metres. Shape (n_arcs,)."""
        from pathsnap.geo.distance import haversine_m  # local import avoids a cycle

        ends = self.nodes[self.arcs]                   # (n_arcs, 2, 2)
        return haversine_m(ends[:, 0, 0], ends[:, 0, 1], ends[:, 1, 0], ends[:, 1, 1])


@dataclass(frozen=True)
class Trip:
    """One complete trip: track + network + optional reference route."""

    trip_id: str
    track: Track
    network: RoadNetwork
    route: np.ndarray | None = None   # chronological ground-truth arc_ids, or None

    path: Path | None = None

    @property
    def has_reference_route(self) -> bool:
        return self.route is not None and len(self.route) > 0

    def summary(self) -> dict:
        return {
            "trip_id": self.trip_id,
            "n_observations": len(self.track),
            "n_nodes": self.network.n_nodes,
            "n_arcs": self.network.n_arcs,
            "has_reference_route": self.has_reference_route,
            "n_route_arcs": 0 if self.route is None else int(len(self.route)),
        }


@dataclass
class CandidateSet:
    """Output of candidate generation for one trip.

    One row per (observation, candidate arc) pair. `obs_idx` indexes into the
    trip's Track rows; `arc_id` indexes into the trip's RoadNetwork.arcs.

    Produced by  : pathsnap.candidates.generator.generate_candidates
    Consumed by  : pathsnap.features.build, pathsnap.hmm.viterbi
    """

    trip_id: str
    obs_idx: np.ndarray          # int64, shape (n_pairs,)
    arc_id: np.ndarray           # int64, shape (n_pairs,)
    dist_m: np.ndarray           # float64, distance from observation to candidate arc
    meta: dict = field(default_factory=dict)

    def n_candidates_per_obs(self) -> np.ndarray:
        """Shape (n_observations,) count of candidates offered per observation."""
        return np.bincount(self.obs_idx, minlength=int(self.obs_idx.max()) + 1)
