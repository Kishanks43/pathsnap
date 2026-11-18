"""Tests for candidate generation, labelling and the feature matrix.

These use a small synthetic trip laid out along a straight road, so they run
without the real dataset and assert exact, checkable answers.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathsnap.candidates.generator import CandidateConfig, generate_candidates
from pathsnap.features.build import FEATURE_COLUMNS, build_feature_matrix
from pathsnap.io.schema import RoadNetwork, Track, Trip
from pathsnap.labels import (
    assign_reference_arcs,
    candidate_miss_rate,
    split_candidates_by_obs,
)

# 49.85 N, 8.58 E. 0.00045 deg latitude is ~50 m.
LAT0, LON0 = 49.85, 8.58
D_LAT = 0.00045


def _make_trip(n_obs: int = 30, noise_m: float = 0.0, seed: int = 0) -> Trip:
    """A 4-arc straight road running north, with GPS points along it."""
    nodes = np.array([[LON0, LAT0 + i * D_LAT] for i in range(5)], dtype=np.float64)
    arcs = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)

    rng = np.random.default_rng(seed)
    ticks = np.linspace(0.0, 3.6, n_obs)          # across 4 arcs of ~50 m each
    lat = LAT0 + ticks * D_LAT
    lon = np.full(n_obs, LON0)
    if noise_m:
        # 1 degree lat ~ 111 km -> metres to degrees
        lon = lon + rng.normal(0.0, noise_m / 111_000.0 / np.cos(np.radians(LAT0)), n_obs)
        lat = lat + rng.normal(0.0, noise_m / 111_000.0, n_obs)

    track = Track(
        trip_id="synthetic",
        df=__import__("pandas").DataFrame(
            {"lon": lon, "lat": lat, "timestamp": np.arange(n_obs, dtype=np.float64) + 1.0}
        ),
    )
    route = np.array([0, 1, 2, 3], dtype=np.int64)
    return Trip(
        trip_id="synthetic",
        track=track,
        network=RoadNetwork(trip_id="synthetic", nodes=nodes, arcs=arcs),
        route=route,
    )


def test_every_observation_gets_a_candidate() -> None:
    trip = _make_trip()
    c = generate_candidates(trip, radius_m=100.0, max_candidates=10)
    counts = np.bincount(c.obs_idx, minlength=len(trip.track))
    assert (counts >= 1).all()
    assert c.meta["n_obs_without_candidates"] == 0


def test_candidates_are_sorted_and_parallel_arrays() -> None:
    trip = _make_trip()
    c = generate_candidates(trip, radius_m=100.0)
    assert len(c.obs_idx) == len(c.arc_id) == len(c.dist_m) == len(c.meta["t_along"])
    assert np.all(np.diff(c.obs_idx) >= 0), "candidates must be grouped by observation"
    assert (c.dist_m >= 0).all()


def test_distances_within_a_block_are_ascending() -> None:
    trip = _make_trip(noise_m=5.0)
    c = generate_candidates(trip, radius_m=100.0)
    counts = np.bincount(c.obs_idx, minlength=len(trip.track))
    start = 0
    for count in counts:
        block = c.dist_m[start : start + count]
        assert np.all(np.diff(block) >= -1e-9), "rank 0 must be the nearest candidate"
        start += count


def test_clean_track_matches_the_expected_arcs() -> None:
    trip = _make_trip(noise_m=0.0)
    ref = assign_reference_arcs(trip)
    assert ref.coverage == 1.0
    assert ref.n_backwards == 0
    # Points run north along arcs 0,1,2,3 in order.
    assert list(ref.arc_per_obs) == sorted(ref.arc_per_obs)
    assert set(np.unique(ref.arc_per_obs)) == {0, 1, 2, 3}
    assert np.all(ref.dist_per_obs < 1.0)


def test_candidate_miss_rate_is_zero_on_a_clean_track() -> None:
    trip = _make_trip()
    c = generate_candidates(trip, radius_m=100.0)
    ref = assign_reference_arcs(trip)
    assert candidate_miss_rate(ref, c, len(trip.track)) == 0.0


def test_radius_controls_candidate_count() -> None:
    trip = _make_trip(noise_m=2.0)
    tight = generate_candidates(trip, radius_m=20.0, max_candidates=10)
    wide = generate_candidates(trip, radius_m=200.0, max_candidates=10)
    assert len(wide.obs_idx) >= len(tight.obs_idx)


def test_small_radius_still_keeps_one_candidate() -> None:
    trip = _make_trip(noise_m=10.0)
    c = generate_candidates(trip, radius_m=0.5, max_candidates=5)
    counts = np.bincount(c.obs_idx, minlength=len(trip.track))
    assert (counts >= 1).all()


def test_feature_matrix_contract() -> None:
    trip = _make_trip(noise_m=3.0)
    c = generate_candidates(trip, radius_m=100.0)
    ref = assign_reference_arcs(trip)
    X, y, meta = build_feature_matrix(trip, c, ref)

    assert X.shape == (len(c.obs_idx), len(FEATURE_COLUMNS))
    assert X.dtype == np.float64
    assert np.isfinite(X).all(), "features must not contain NaN/inf"
    assert y is not None and y.dtype == np.int64
    assert set(np.unique(y)) <= {0, 1}
    assert meta["columns"] == FEATURE_COLUMNS
    assert meta["schema_version"] == 1
    assert len(meta["label_residual_m"]) == len(X)


def test_exactly_one_positive_candidate_per_observation() -> None:
    trip = _make_trip(noise_m=3.0)
    c = generate_candidates(trip, radius_m=100.0)
    ref = assign_reference_arcs(trip)
    X, y, meta = build_feature_matrix(trip, c, ref)

    per_obs = np.bincount(meta["obs_idx"], weights=y, minlength=meta["n_obs"])
    assert (per_obs <= 1).all(), "at most one candidate can be the reference arc"
    assert y.sum() > 0


def test_no_reference_route_returns_none_labels() -> None:
    trip = _make_trip()
    trip_no_route = Trip(
        trip_id=trip.trip_id, track=trip.track, network=trip.network, route=None
    )
    c = generate_candidates(trip_no_route, radius_m=100.0)
    X, y, meta = build_feature_matrix(trip_no_route, c, None)
    assert y is None
    assert X.shape[1] == len(FEATURE_COLUMNS)


def test_sequential_features_flag_the_first_observation() -> None:
    trip = _make_trip(noise_m=3.0)
    c = generate_candidates(trip, radius_m=100.0)
    X, _, meta = build_feature_matrix(trip, c, None)
    col = FEATURE_COLUMNS.index("has_prev")
    first_obs_rows = meta["obs_idx"] == 0
    assert np.all(X[first_obs_rows, col] == 0.0)
    assert X[~first_obs_rows, col].max() == 1.0


def test_split_candidates_by_obs_covers_every_observation() -> None:
    trip = _make_trip()
    c = generate_candidates(trip, radius_m=100.0)
    per_obs = split_candidates_by_obs(c, len(trip.track))
    assert len(per_obs) == len(trip.track)
    assert sum(len(p) for p in per_obs) == len(c.obs_idx)


def test_candidate_config_defaults() -> None:
    cfg = CandidateConfig()
    assert cfg.radius_m == 50.0
    assert cfg.resolved_cell_m() == pytest.approx(100.0)   # 2 x radius
    assert CandidateConfig(radius_m=30.0).resolved_cell_m() == pytest.approx(60.0)
    assert CandidateConfig(radius_m=5.0).resolved_cell_m() == pytest.approx(25.0)  # floor
