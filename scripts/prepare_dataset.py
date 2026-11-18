"""Prepare cached candidate features for every trip in a partition.

OWNER: Person 1

Run once per machine (and again with --force after changing features):

    python scripts/prepare_dataset.py                     # both partitions
    python scripts/prepare_dataset.py --partition train
    python scripts/prepare_dataset.py --limit 5           # smoke test

Writes data/processed/<partition>/<trip_id>.npz plus a quality summary at
experiments/results/candidate_quality.json, which the report cites directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from pathsnap.candidates.generator import CandidateConfig  # noqa: E402
from pathsnap.config import RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR, ensure_dirs  # noqa: E402
from pathsnap.features.build import write_feature_schema  # noqa: E402
from pathsnap.features.dataset import (  # noqa: E402
    TRUST_RESIDUAL_M,
    load_trip_features,
    prepare_trip,
    save_trip_features,
    trip_path,
)
from pathsnap.io.loaders import discover_trips, load_trip  # noqa: E402

PARTITIONS = {"train": TRAIN_DIR, "student_test": STUDENT_TEST_DIR}


def prepare_partition(
    partition: str,
    source_dir: Path,
    cfg: CandidateConfig,
    limit: int | None,
    force: bool,
) -> list[dict]:
    trips = discover_trips(source_dir)
    if limit:
        trips = trips[:limit]

    rows: list[dict] = []
    t_start = time.perf_counter()

    for i, trip_dir in enumerate(trips, 1):
        trip_id = trip_dir.name
        cached = trip_path(partition, trip_id)
        if cached.exists() and not force:
            tf = load_trip_features(trip_id, partition)
        else:
            tf = prepare_trip(load_trip(trip_dir), config=cfg)
            save_trip_features(tf, partition)

        s = tf.summary()
        s["partition"] = partition
        rows.append(s)
        print(
            f"  [{i:>2}/{len(trips)}] {trip_id}  obs={s['n_obs']:>6}  pairs={s['n_pairs']:>7}  "
            f"cand/obs={s['candidates_per_obs']:>4.1f}  miss={s['candidate_miss_rate']:.4f}  "
            f"trusted={s['trusted_fraction']:.2%}  y+={s['positives']}",
            flush=True,
        )

    elapsed = time.perf_counter() - t_start
    print(f"  {partition}: {len(rows)} trips in {elapsed:.1f}s")
    return rows


def summarise(rows: list[dict], threshold: float = TRUST_RESIDUAL_M) -> dict:
    if not rows:
        return {}
    miss = np.array([r["candidate_miss_rate"] for r in rows], dtype=float)
    pairs = int(sum(r["n_pairs"] for r in rows))
    obs = int(sum(r["n_obs"] for r in rows))
    trusted = float(np.average([r["trusted_fraction"] for r in rows], weights=[r["n_pairs"] for r in rows]))
    return {
        "n_trips": len(rows),
        "n_observations": obs,
        "n_candidate_rows": pairs,
        "candidates_per_observation": round(pairs / obs, 3) if obs else 0.0,
        "candidate_miss_rate_mean": round(float(np.nanmean(miss)), 4),
        "candidate_miss_rate_median": round(float(np.nanmedian(miss)), 4),
        "candidate_miss_rate_p90": round(float(np.nanpercentile(miss, 90)), 4),
        "candidate_miss_rate_worst": round(float(np.nanmax(miss)), 4),
        "trusted_fraction": round(trusted, 4),
        "trust_residual_m": threshold,
        "trips_worst_10": sorted(
            ({"trip_id": r["trip_id"], "miss_rate": round(r["candidate_miss_rate"], 4),
              "trusted_fraction": round(r["trusted_fraction"], 4), "n_obs": r["n_obs"]} for r in rows),
            key=lambda r: -r["miss_rate"],
        )[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", choices=[*PARTITIONS, "all"], default="all")
    ap.add_argument("--radius", type=float, default=100.0, help="candidate radius in metres (default 100)")
    ap.add_argument("--max-candidates", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="only the first N trips (smoke test)")
    ap.add_argument("--force", action="store_true", help="recompute even if a cache exists")
    args = ap.parse_args()

    ensure_dirs()
    cfg = CandidateConfig(radius_m=args.radius, max_candidates=args.max_candidates)

    schema_path = write_feature_schema()
    print(f"feature schema v1 written to {schema_path}")

    selected = list(PARTITIONS) if args.partition == "all" else [args.partition]
    summary: dict[str, dict] = {}

    for partition in selected:
        print(f"\n=== {partition} ===")
        rows = prepare_partition(partition, PARTITIONS[partition], cfg, args.limit, args.force)
        summary[partition] = summarise(rows)

    out = RESULTS_DIR / "candidate_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    for partition, s in summary.items():
        print(
            f"{partition:14s} trips={s['n_trips']:>3}  obs={s['n_observations']:>7,}  "
            f"rows={s['n_candidate_rows']:>9,}  cand/obs={s['candidates_per_observation']:.2f}  "
            f"miss_mean={s['candidate_miss_rate_mean']:.4f}  worst={s['candidate_miss_rate_worst']:.4f}  "
            f"trusted={s['trusted_fraction']:.2%}"
        )
    print(f"\nquality report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
