"""Sanity-check the local dataset before anyone starts modelling.

Run this first, on every machine, after copying the data in:

    python scripts/check_data.py

It answers the questions the Dataset Exploration brief asks for (trip counts,
observation counts, file availability, coordinate plausibility, timestamps,
network size, reference-route availability) and prints a compact inventory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from pathsnap.config import STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402
from pathsnap.io.loaders import discover_trips, load_trip, validate_trip  # noqa: E402


def inventory(partition_dir: Path, label: str) -> list[dict]:
    trips = discover_trips(partition_dir)
    rows: list[dict] = []
    problems: list[str] = []

    for trip_dir in trips:
        trip = load_trip(trip_dir)
        issues = validate_trip(trip)
        for issue in issues:
            problems.append(f"{trip.trip_id}: {issue}")
        dt = trip.track.time_deltas()
        rows.append(
            {
                "trip": trip.trip_id,
                "obs": len(trip.track),
                "nodes": trip.network.n_nodes,
                "arcs": trip.network.n_arcs,
                "route_arcs": 0 if trip.route is None else len(trip.route),
                "dt_median": float(np.median(dt)) if len(dt) else float("nan"),
                "dt_max": float(dt.max()) if len(dt) else float("nan"),
                "lon0": float(trip.track.lon.min()),
                "lon1": float(trip.track.lon.max()),
                "lat0": float(trip.track.lat.min()),
                "lat1": float(trip.track.lat.max()),
            }
        )

    obs = np.array([r["obs"] for r in rows])
    print(f"\n=== {label} ===")
    print(f"trips                : {len(rows)}")
    print(f"observations total   : {obs.sum():,}")
    print(f"observations/trip    : min {obs.min()}  median {int(np.median(obs))}  max {obs.max()}")
    print(f"nodes total          : {sum(r['nodes'] for r in rows):,}")
    print(f"arcs total           : {sum(r['arcs'] for r in rows):,}")
    print(f"trips with a route   : {sum(1 for r in rows if r['route_arcs'] > 0)}/{len(rows)}")
    print(f"median dt (s)        : {np.nanmedian([r['dt_median'] for r in rows]):.2f}")
    print(f"worst dt gap (s)     : {np.nanmax([r['dt_max'] for r in rows]):.1f}")
    print(f"lon range            : {min(r['lon0'] for r in rows):.4f} .. {max(r['lon1'] for r in rows):.4f}")
    print(f"lat range            : {min(r['lat0'] for r in rows):.4f} .. {max(r['lat1'] for r in rows):.4f}")

    if problems:
        print(f"\n!! {len(problems)} validation issue(s):")
        for p in problems[:20]:
            print("   -", p)
        if len(problems) > 20:
            print(f"   ... and {len(problems) - 20} more")
    else:
        print("\nNo validation issues found.")

    return rows


def main() -> int:
    if not TRAIN_DIR.is_dir():
        print(f"ERROR: training data not found at {TRAIN_DIR}")
        print("See data/README.md for the expected layout.")
        return 1

    inventory(TRAIN_DIR, "train (80 trips, for fitting + CV)")
    if STUDENT_TEST_DIR.is_dir():
        inventory(STUDENT_TEST_DIR, "student_test (10 trips, development evaluation)")
    else:
        print(f"\nWARNING: student_test not found at {STUDENT_TEST_DIR}")

    print("\nReminder: the 10 evaluator-testing trips are not in this repository.")
    print("They must never be used for training, feature selection or tuning.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
