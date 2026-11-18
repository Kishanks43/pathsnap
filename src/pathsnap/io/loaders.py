"""Readers for the four raw file types, plus trip discovery.

All raw files are whitespace-separated with NO header row.

    .track : lon  lat  timestamp
    .nodes : lon  lat                     (row index = node_id)
    .arcs  : from_node_id  to_node_id     (row index = arc_id)
    .route : arc_id                       (chronological ground truth)

Loaders never mutate files and never drop rows silently: anything anomalous is
reported through `validate_trip` so the team can decide what to do about it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.io.schema import RoadNetwork, Track, Trip

_TRACK_COLS = ["lon", "lat", "timestamp"]
_NODE_COLS = ["lon", "lat"]
_ARC_COLS = ["from_node_id", "to_node_id"]


def _read_whitespace(path: Path, names: list[str], dtypes: dict) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing raw file: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtypes)
    return df


def load_track(path: str | Path, trip_id: str | None = None) -> Track:
    path = Path(path)
    df = _read_whitespace(
        path, _TRACK_COLS, {"lon": np.float64, "lat": np.float64, "timestamp": np.float64}
    )
    return Track(trip_id=trip_id or path.stem, df=df)


def load_network(path: str | Path, trip_id: str | None = None) -> RoadNetwork:
    """Load `.nodes` and `.arcs` given the path to either one of them."""
    path = Path(path)
    stem = path.with_suffix("")
    nodes_df = _read_whitespace(
        stem.with_suffix(".nodes"), _NODE_COLS, {"lon": np.float64, "lat": np.float64}
    )
    arcs_df = _read_whitespace(
        stem.with_suffix(".arcs"),
        _ARC_COLS,
        {"from_node_id": np.int64, "to_node_id": np.int64},
    )
    return RoadNetwork(
        trip_id=trip_id or path.stem,
        nodes=nodes_df.to_numpy(dtype=np.float64),
        arcs=arcs_df.to_numpy(dtype=np.int64),
    )


def load_route(path: str | Path) -> np.ndarray | None:
    """Load a `.route` file. Returns None when the file is absent or empty."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        series = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.int64)
    except pd.errors.EmptyDataError:
        return None
    if series.empty:
        return None
    return series.iloc[:, 0].to_numpy(dtype=np.int64)


def load_trip(trip_dir: str | Path) -> Trip:
    """Load a trip directory containing ``<id>.track/.nodes/.arcs/.route``."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    return Trip(
        trip_id=trip_id,
        track=load_track(trip_dir / f"{trip_id}.track", trip_id),
        network=load_network(trip_dir / f"{trip_id}.arcs", trip_id),
        route=load_route(trip_dir / f"{trip_id}.route"),
        path=trip_dir,
    )


def discover_trips(partition_dir: str | Path) -> list[Path]:
    """Return sorted trip directories inside a partition root."""
    partition_dir = Path(partition_dir)
    if not partition_dir.is_dir():
        raise FileNotFoundError(f"partition directory not found: {partition_dir}")
    return sorted(p for p in partition_dir.iterdir() if p.is_dir())


def iter_trips(partition_dir: str | Path, limit: int | None = None):
    """Yield loaded trips from a partition directory."""
    for i, trip_dir in enumerate(discover_trips(partition_dir)):
        if limit is not None and i >= limit:
            return
        yield load_trip(trip_dir)


def validate_trip(trip: Trip) -> list[str]:
    """Return a list of human-readable problems. Empty list means clean.

    This is deliberately conservative: it reports, it does not repair.
    """
    issues: list[str] = []
    t, net = trip.track, trip.network

    if len(t) < 2:
        issues.append("track has fewer than 2 observations")
    if not np.isfinite(t.lon).all() or not np.isfinite(t.lat).all():
        issues.append("track contains non-finite coordinates")
    else:
        if t.lon.min() < -180 or t.lon.max() > 180:
            issues.append("longitude outside [-180, 180] (column order swapped?)")
        if t.lat.min() < -90 or t.lat.max() > 90:
            issues.append("latitude outside [-90, 90] (column order swapped?)")

    dts = t.time_deltas()
    if len(dts) and (dts <= 0).any():
        issues.append(f"{(dts <= 0).sum()} non-increasing timestamps")

    if net.n_nodes == 0:
        issues.append("network has zero nodes")
    if net.n_arcs == 0:
        issues.append("network has zero arcs")
    else:
        if net.arcs.min() < 0 or net.arcs.max() >= net.n_nodes:
            issues.append("arc references a node id outside [0, n_nodes)")

    if trip.route is not None:
        if trip.route.min() < 0 or trip.route.max() >= net.n_arcs:
            issues.append("route references an arc id outside [0, n_arcs)")

    return issues
