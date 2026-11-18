"""Geodesy helpers.

Two facts drive every function here:

1. lon/lat are angular degrees. Euclidean distance in degrees is meaningless
   for comparing GPS noise against road geometry, so distances are always
   computed in metres.
2. The dataset is small geographically (a city), so a local tangent-plane
   projection around the data centroid is accurate to well under a metre and
   is far cheaper than full geodesic maths inside tight loops.

Use `haversine_m` for one-off distances and `LocalProjection` when you need to
do a lot of work in a flat, metre-based frame (e.g. candidate generation).
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(
    lon1: float | np.ndarray,
    lat1: float | np.ndarray,
    lon2: float | np.ndarray,
    lat2: float | np.ndarray,
) -> np.ndarray:
    """Great-circle distance in metres between two lon/lat points."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(a, dtype=np.float64)) for a in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class LocalProjection:
    """Equirectangular projection to a local metric frame (x, y) in metres.

    Anchored at a reference lon/lat; accurate to <1 m over city scales, which is
    far below GPS noise (~5-15 m) in this dataset.
    """

    def __init__(self, lon0: float, lat0: float) -> None:
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._cos_lat0 = float(np.cos(np.radians(self.lat0)))

    @classmethod
    def from_points(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    def forward(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        x = np.radians(lon - self.lon0) * EARTH_RADIUS_M * self._cos_lat0
        y = np.radians(lat - self.lat0) * EARTH_RADIUS_M
        return x, y

    def inverse(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon = self.lon0 + np.degrees(np.asarray(x) / (EARTH_RADIUS_M * self._cos_lat0))
        lat = self.lat0 + np.degrees(np.asarray(y) / EARTH_RADIUS_M)
        return lon, lat


def point_to_segment_distance(
    px: np.ndarray, py: np.ndarray, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to each segment, in the same units as the inputs.

    All five arguments must broadcast to a common shape (typically the
    (n_points, n_segments) outer product is built by the caller).

    Returns
    -------
    dist : numpy array, perpendicular distance from point to the segment
           (clamped to the nearer endpoint when the projection falls outside).
    t    : fraction along the segment [0, 1] of the closest point.
    """
    px, py, ax, ay, bx, by = (
        np.asarray(v, dtype=np.float64) for v in (px, py, ax, ay, bx, by)
    )
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    # Guard against zero-length segments.
    safe_len2 = np.where(seg_len2 > 0, seg_len2, np.nan)
    t = ((px - ax) * dx + (py - ay) * dy) / safe_len2
    t = np.where(np.isnan(t), 0.0, np.clip(t, 0.0, 1.0))
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t
