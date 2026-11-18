"""Loader and schema tests. These run without the real dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathsnap.geo.distance import LocalProjection, haversine_m, point_to_segment_distance
from pathsnap.io.loaders import load_route, load_track, load_trip, validate_trip


def _write_trip(root: Path, trip_id: str = "00000001", with_route: bool = True) -> Path:
    d = root / trip_id
    d.mkdir(parents=True)
    (d / f"{trip_id}.track").write_text(
        "8.595203\t49.844053\t1.0\n8.595294\t49.844441\t2.0\n8.595352\t49.844883\t3.0\n"
    )
    (d / f"{trip_id}.nodes").write_text(
        "8.581679\t49.857880\n8.584637\t49.858295\n8.580173\t49.858567\n"
    )
    (d / f"{trip_id}.arcs").write_text("0\t1\n0\t2\n1\t2\n")
    if with_route:
        (d / f"{trip_id}.route").write_text("0\n2\n")
    return d


def test_load_trip_shapes(tmp_path: Path) -> None:
    trip = load_trip(_write_trip(tmp_path))
    assert len(trip.track) == 3
    assert trip.network.n_nodes == 3
    assert trip.network.n_arcs == 3
    assert trip.route is not None and list(trip.route) == [0, 2]
    assert trip.has_reference_route


def test_clean_trip_validates(tmp_path: Path) -> None:
    assert validate_trip(load_trip(_write_trip(tmp_path))) == []


def test_missing_route_is_none(tmp_path: Path) -> None:
    trip = load_trip(_write_trip(tmp_path, with_route=False))
    assert trip.route is None
    assert not trip.has_reference_route


def test_empty_route_is_none(tmp_path: Path) -> None:
    d = _write_trip(tmp_path, with_route=False)
    (d / f"{d.name}.route").write_text("")
    assert load_route(d / f"{d.name}.route") is None


def test_validate_flags_out_of_range_arc(tmp_path: Path) -> None:
    d = _write_trip(tmp_path)
    (d / f"{d.name}.route").write_text("0\n99\n")
    issues = validate_trip(load_trip(d))
    assert any("arc id" in i for i in issues)


def test_validate_flags_non_increasing_time(tmp_path: Path) -> None:
    d = _write_trip(tmp_path)
    (d / f"{d.name}.track").write_text(
        "8.595203\t49.844053\t5.0\n8.595294\t49.844441\t5.0\n8.595352\t49.844883\t9.0\n"
    )
    issues = validate_trip(load_trip(d))
    assert any("timestamps" in i for i in issues)


def test_arc_lengths_are_metres(tmp_path: Path) -> None:
    net = load_trip(_write_trip(tmp_path)).network
    lengths = net.arc_lengths_m()
    assert lengths.shape == (3,)
    assert (lengths > 0).all()
    assert (lengths < 2000).all()  # city-scale arcs, sanity bound


def test_haversine_known_distance() -> None:
    # ~111.2 km per degree of latitude at the equator.
    d = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert 111_000 < float(d) < 111_400


def test_local_projection_round_trip() -> None:
    proj = LocalProjection(lon0=8.58, lat0=49.85)
    x, y = proj.forward(8.60, 49.86)
    lon, lat = proj.inverse(x, y)
    assert np.isclose(lon, 8.60, atol=1e-9)
    assert np.isclose(lat, 49.86, atol=1e-9)


def test_point_to_segment_behaviour() -> None:
    # Point at (0, 1) against a horizontal segment from (0,0) to (10,0).
    dist, t = point_to_segment_distance(
        np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]),
        np.array([10.0]), np.array([0.0]),
    )
    assert np.isclose(dist[0], 1.0)
    assert np.isclose(t[0], 0.0)

    # Beyond the end of the segment: distance clamps to the endpoint.
    dist2, t2 = point_to_segment_distance(
        np.array([20.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]),
        np.array([10.0]), np.array([0.0]),
    )
    assert np.isclose(dist2[0], 10.0)
    assert np.isclose(t2[0], 1.0)


def test_load_track_requires_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_track(tmp_path / "nope.track")
