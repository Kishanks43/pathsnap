#!/usr/bin/env python
"""P4 runner: decoder tuning, HMM vs classifier-only on student_test, artifact manifest.

Stages (run in this order):

    python scripts/run_p4.py decoder-tune   # choose the Viterbi settings on TRAIN trips only
    python scripts/run_p4.py student-test   # decode the 10 student trips; HMM vs classifier-only
    python scripts/run_p4.py manifest       # checksums + versions for artifacts/manifest.json
    python scripts/run_p4.py all

Leakage rules
-------------
- decoder-tune uses 10 held-out TRAINING trips, scored by a model that was trained on
  the other 70 training trips. The student-test partition is not read in that stage.
- student-test runs the final pipeline exactly as the API does (Predictor), once,
  with the settings chosen above. Nothing is tuned on it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

SEED = 42
N_HOLDOUT_TRIPS = 10
BETA_GRID = [10.0, 25.0, 50.0, 100.0, 200.0]
HOPS_GRID = [1, 2]


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n", flush=True)


def _trip_views(trip_dir: Path):
    """Track, network and candidate features for one trip, WITHOUT the reference route."""
    from pathsnap.api.inference import CANDIDATE_CONFIG
    from pathsnap.candidates.generator import generate_candidates
    from pathsnap.features.build import build_feature_matrix
    from pathsnap.io.loaders import load_network, load_track
    from pathsnap.io.schema import Trip

    tid = trip_dir.name
    track = load_track(trip_dir / f"{tid}.track", tid)
    network = load_network(trip_dir / f"{tid}.arcs", tid)
    trip = Trip(trip_id=tid, track=track, network=network, route=None, path=trip_dir)
    candidates = generate_candidates(trip, config=CANDIDATE_CONFIG)
    X, _, _ = build_feature_matrix(trip, candidates, None)
    return trip, candidates, X


def _truth(trip_dir: Path, trip):
    """Reference arc per observation and the trusted mask. Used for scoring only."""
    from pathsnap.io.loaders import load_route
    from pathsnap.io.schema import Trip
    from pathsnap.labels import assign_reference_arcs

    route = load_route(trip_dir / f"{trip.trip_id}.route")
    scored = Trip(trip_id=trip.trip_id, track=trip.track, network=trip.network, route=route, path=trip_dir)
    ref = assign_reference_arcs(scored)
    trusted = ref.resolved & (ref.dist_per_obs <= 25.0)
    return ref.arc_per_obs, trusted


def _chosen_model_params(results_dir: Path) -> dict:
    """Hyperparameters of the random forest that the P3 final stage kept."""
    comparison = results_dir / "tuned_vs_untuned_random_forest.json"
    if not comparison.exists():
        return {}
    info = json.loads(comparison.read_text(encoding="utf-8"))
    if info.get("kept") == "tuned":
        return dict(info.get("tuned_params", {}))
    return {}


def stage_decoder_tune() -> None:
    from pathsnap.config import RESULTS_DIR, ARTIFACTS_DIR, TRAIN_DIR, ensure_dirs
    from pathsnap.features.dataset import load_partition
    from pathsnap.hmm.viterbi import DecoderConfig, classifier_only, viterbi_decode
    from pathsnap.models.registry import build_model

    ensure_dirs()
    _banner("P4 decoder tuning on 10 held-out TRAINING trips")
    train = load_partition("train")
    trust = train.trusted()
    n_trips = len(train.trip_ids)
    rng = np.random.RandomState(SEED)
    holdout = sorted(rng.permutation(n_trips)[:N_HOLDOUT_TRIPS].tolist())
    held_ids = [train.trip_ids[g] for g in holdout]
    print(f"  held-out trips (not used to train the scorer): {held_ids}", flush=True)

    keep = trust & ~np.isin(train.groups, holdout)
    params = _chosen_model_params(RESULTS_DIR)
    print(f"  scorer = random_forest {params or '(default config)'} on {keep.sum():,} rows", flush=True)
    model = build_model("random_forest", **{**params, "random_state": SEED})
    t0 = time.perf_counter()
    model.fit(train.X[keep], train.y[keep], groups=train.groups[keep])
    print(f"  scorer fitted in {time.perf_counter() - t0:.1f}s", flush=True)

    views = []
    for g in holdout:
        trip_dir = TRAIN_DIR / train.trip_ids[g]
        trip, cands, X = _trip_views(trip_dir)
        proba = model.predict_proba(X)
        truth, trusted = _truth(trip_dir, trip)
        views.append((trip, cands, proba, truth, trusted))

    rows = []
    for beta in BETA_GRID:
        for hops in HOPS_GRID:
            cfg = DecoderConfig(beta_m=beta, max_hops=hops)
            correct = total = clf_correct = forced = 0
            cont_sum = cont_n = 0.0
            for trip, cands, proba, truth, trusted in views:
                res = viterbi_decode(trip.trip_id, cands, proba, trip.network, len(trip.track), cfg, track=trip.track)
                pred = res.arc_per_obs
                clf = classifier_only(cands, proba, len(trip.track))
                correct += int(((pred == truth) & trusted).sum())
                clf_correct += int(((clf == truth) & trusted).sum())
                total += int(trusted.sum())
                forced += res.n_forced_emission_only
            row = {
                "beta_m": beta,
                "max_hops": hops,
                "hmm_top1": correct / total,
                "classifier_top1": clf_correct / total,
                "forced_steps": forced,
            }
            rows.append(row)
            print(
                f"  beta={beta:>6.1f} hops={hops}: HMM top-1 {row['hmm_top1']:.4f}  "
                f"classifier {row['classifier_top1']:.4f}  forced steps {forced}",
                flush=True,
            )

    best = max(rows, key=lambda r: (r["hmm_top1"], -r["forced_steps"]))
    chosen = DecoderConfig(beta_m=best["beta_m"], max_hops=best["max_hops"])
    (ARTIFACTS_DIR / "decoder_config.json").write_text(json.dumps(chosen.to_dict(), indent=2), encoding="utf-8")
    with (RESULTS_DIR / "decoder_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / "decoder_tuning.json").write_text(
        json.dumps(
            {
                "held_out_train_trips": held_ids,
                "scorer": "random_forest",
                "scorer_params": params,
                "grid": {"beta_m": BETA_GRID, "max_hops": HOPS_GRID},
                "chosen": chosen.to_dict(),
                "chosen_hmm_top1": best["hmm_top1"],
                "chosen_classifier_top1": best["classifier_top1"],
                "note": "student_test not used; decoder settings chosen on training trips only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  chosen: beta_m={chosen.beta_m}, max_hops={chosen.max_hops} -> artifacts/decoder_config.json", flush=True)


def stage_student_test() -> None:
    from pathsnap.api.inference import Predictor
    from pathsnap.config import ARTIFACTS_DIR, PROJECT_ROOT, RESULTS_DIR, STUDENT_TEST_DIR
    from pathsnap.io.loaders import load_trip
    from pathsnap.viz.folium_maps import build_prediction_map

    _banner("P4 student_test: Viterbi (HMM) vs classifier-only, same trips and model")
    predictor = Predictor(ARTIFACTS_DIR).load()
    trip_dirs = sorted(p for p in STUDENT_TEST_DIR.iterdir() if p.is_dir())
    rows, preds = [], {}
    for trip_dir in trip_dirs:
        t0 = time.perf_counter()
        pred = predictor.predict_trip(trip_dir)
        score = predictor.score_against_reference(pred, trip_dir)
        elapsed = time.perf_counter() - t0
        preds[trip_dir.name] = (trip_dir, pred)
        top_h, top_c = score.get("top1_accuracy"), score.get("classifier_only_top1_accuracy")
        n = score.get("trusted_observations_scored") or 0
        rows.append({
            "trip_id": trip_dir.name,
            "n_obs": pred.n_observations,
            "trusted_obs_scored": n,
            "top1_hmm": top_h,
            "top1_classifier": top_c,
            "delta_hmm_minus_clf": None if top_h is None else top_h - top_c,
            "route_continuity_hmm": score.get("route_continuity"),
            "road_set_overlap": score.get("road_set_overlap"),
            "gap_resets": pred.diagnostics["gap_resets"],
            "forced_steps": pred.diagnostics["n_forced_emission_only_steps"],
            "obs_without_candidates": pred.diagnostics["n_obs_without_candidates"],
            "seconds": round(elapsed, 2),
        })
        print(
            f"  {trip_dir.name}: obs={pred.n_observations:>6}  HMM top-1 {top_h if top_h is None else round(top_h, 4)}  "
            f"classifier {top_c if top_c is None else round(top_c, 4)}  ({elapsed:.1f}s)",
            flush=True,
        )

    scored = [r for r in rows if r["top1_hmm"] is not None]
    pooled_h = sum(r["top1_hmm"] * r["trusted_obs_scored"] for r in scored) / sum(r["trusted_obs_scored"] for r in scored)
    pooled_c = sum(r["top1_classifier"] * r["trusted_obs_scored"] for r in scored) / sum(r["trusted_obs_scored"] for r in scored)
    worse = sorted((r for r in scored if r["delta_hmm_minus_clf"] < 0), key=lambda r: r["delta_hmm_minus_clf"])
    better = sorted((r for r in scored if r["delta_hmm_minus_clf"] > 0), key=lambda r: -r["delta_hmm_minus_clf"])

    fields = list(rows[0].keys())
    with (RESULTS_DIR / "hmm_vs_classifier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "trips": len(rows),
        "pooled_top1_hmm": pooled_h,
        "pooled_top1_classifier": pooled_c,
        "trips_hmm_better": len(better),
        "trips_hmm_worse": len(worse),
        "trips_equal": len(scored) - len(better) - len(worse),
        "worst_case": worse[0] if worse else None,
        "best_case": better[0] if better else None,
        "decoder_config": json.loads((ARTIFACTS_DIR / "decoder_config.json").read_text(encoding="utf-8")),
    }
    (RESULTS_DIR / "hmm_vs_classifier.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n  pooled top-1: HMM {pooled_h:.4f} vs classifier {pooled_c:.4f}; "
        f"HMM better on {len(better)} trips, worse on {len(worse)}",
        flush=True,
    )

    maps_dir = PROJECT_ROOT / "reports" / "maps"
    picks = [r["trip_id"] for r in (worse[:1] + better[:1])]
    for tid in picks:
        trip_dir, pred = preds[tid]
        trip = load_trip(trip_dir)  # route used for drawing the reference line only
        out = build_prediction_map(
            trip, pred.arc_per_obs, maps_dir / f"pred_{tid}.html",
            classifier_only_arc_per_obs=pred.classifier_arc_per_obs,
        )
        print(f"  map -> {out}", flush=True)


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parents[1])
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_manifest() -> None:
    import sklearn

    from pathsnap.api.inference import REQUIRED_ARTIFACTS
    from pathsnap.config import ARTIFACTS_DIR, RESULTS_DIR

    _banner("P4 artifact manifest")
    present = {name: _sha256(ARTIFACTS_DIR / name) for name in REQUIRED_ARTIFACTS if (ARTIFACTS_DIR / name).exists()}
    model_path = ARTIFACTS_DIR / "model.joblib"
    record = {}
    kept = RESULTS_DIR / "tuned_vs_untuned_random_forest.json"
    if kept.exists():
        record = json.loads(kept.read_text(encoding="utf-8"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "seed": SEED,
        "model": {
            "file": "model.joblib",
            "sha256": _sha256(model_path) if model_path.exists() else None,
            "kept_config": record.get("kept"),
            "tuned_params": record.get("tuned_params"),
        },
        "artifacts_sha256": present,
    }
    (ARTIFACTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  -> artifacts/manifest.json ({len(present)} required artifacts hashed)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 runner: decoder + student-test + manifest")
    parser.add_argument("stage", choices=["decoder-tune", "student-test", "manifest", "all"])
    args = parser.parse_args()
    if args.stage in ("decoder-tune", "all"):
        stage_decoder_tune()
    if args.stage in ("student-test", "all"):
        stage_student_test()
    if args.stage in ("manifest", "all"):
        stage_manifest()


if __name__ == "__main__":
    main()
