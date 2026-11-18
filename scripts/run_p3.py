#!/usr/bin/env python
"""P3 runner: untuned Random Forest, Bayesian tuning of the CV winner, tuned vs untuned.

Stages (each writes its results to experiments/results/):

    python scripts/run_p3.py untuned     # RF default config: group CV, student test, importances
    python scripts/run_p3.py tune        # Optuna TPE search on the CV winner (training trips only)
    python scripts/run_p3.py final       # full-data CV of tuned vs untuned, refit, student test
    python scripts/run_p3.py all         # the three stages above, in order

Leakage rules (see pathsnap/tuning/bayes.py):
- Only training trips feed the search and the choice of final configuration.
- The choice between tuned and untuned is made on full-data CV. The student test
  partition is scored afterwards, for reporting only.

Why the search uses a subsample
-------------------------------
One 5-fold CV of a 300-tree forest on the full 1.08M-row training folds takes about
15 minutes here, so 25 trials on full data would take most of a day. The search
objective therefore uses a random subsample of whole observations (every candidate
row of a chosen observation is kept, so per-observation metrics stay exact). The
final tuned-vs-untuned decision is then made on FULL training data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

SEED = 42
N_SPLITS = 5


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n", flush=True)


def _load():
    from pathsnap.config import RESULTS_DIR, ensure_dirs
    from pathsnap.features.dataset import load_partition

    ensure_dirs()
    train = load_partition("train")
    test = load_partition("student_test")
    trust_train = train.trusted()
    trust_test = test.trusted()
    data = {
        "X": train.X[trust_train],
        "y": train.y[trust_train],
        "groups": train.groups[trust_train],
        "obs_idx": train.obs_idx[trust_train],
        "arc_id": train.arc_id[trust_train],
    }
    return RESULTS_DIR, data, test, trust_test


def _subsample_by_observation(data: dict, target_rows: int, seed: int) -> dict:
    """Keep all candidate rows of a random subset of observations (exact per-obs metrics)."""
    n = len(data["y"])
    if target_rows >= n:
        return data
    keys = data["groups"].astype(np.int64) * (int(data["obs_idx"].max()) + 1) + data["obs_idx"]
    _, inverse = np.unique(keys, return_inverse=True)
    rng = np.random.RandomState(seed)
    chosen = rng.rand(int(inverse.max()) + 1) < target_rows / n
    mask = chosen[inverse]
    return {k: v[mask] for k, v in data.items()}


def _score_on_student_test(model, test, trust_test) -> tuple[dict, dict]:
    from pathsnap.evaluation.holdout import evaluate_on_test, mean_route_continuity

    metrics, pred_obs = evaluate_on_test(model, test, trust_test)
    cont = mean_route_continuity(test, trust_test, pred_obs)
    metrics["route_continuity"] = cont["mean_route_continuity"]
    return metrics, cont


def _cv_summary(cv) -> dict:
    return {
        "mean_macro_f1": cv.mean_macro_f1,
        "std_macro_f1": cv.std_macro_f1,
        "mean_top1_accuracy": cv.mean_top1_accuracy,
        "mean_log_loss": cv.mean_log_loss,
        "folds": [f.__dict__ for f in cv.folds],
    }


def stage_untuned() -> None:
    from pathsnap.evaluation.cv import cross_validate_model
    from pathsnap.models.tree import RandomForestModel

    results_dir, data, test, trust_test = _load()

    _banner("P3 untuned Random Forest: group-aware CV")
    cv = cross_validate_model(
        RandomForestModel(random_state=SEED),
        data["X"], data["y"], data["groups"],
        n_splits=N_SPLITS, seed=SEED,
        obs_idx=data["obs_idx"], arc_id=data["arc_id"], verbose=True,
    )
    print(f"  CV macro F1 = {cv.mean_macro_f1:.4f} +/- {cv.std_macro_f1:.4f}", flush=True)

    _banner("P3 untuned Random Forest: full-train refit + student test")
    model = RandomForestModel(random_state=SEED)
    t0 = time.perf_counter()
    model.fit(data["X"], data["y"], groups=data["groups"])
    fit_seconds = time.perf_counter() - t0
    print(f"  fit {fit_seconds:.1f}s on {len(data['y']):,} rows", flush=True)

    metrics, cont = _score_on_student_test(model, test, trust_test)
    print(
        f"  student test: top-1 = {metrics['top1_accuracy']:.4f}  macro F1 = {metrics['macro_f1']:.4f}  "
        f"log loss = {metrics['log_loss']}  continuity = {metrics['route_continuity']}",
        flush=True,
    )
    importances_path = model.export_feature_importances(results_dir / "importances_rf.csv")
    print(f"  importances -> {importances_path}", flush=True)

    out = {
        "stage": "untuned",
        "config": model.describe(),
        "cv": _cv_summary(cv),
        "student_test": metrics,
        "train_seconds": fit_seconds,
        "route_continuity_per_trip": cont["per_trip"],
    }
    (results_dir / "rf_untuned.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  -> {results_dir / 'rf_untuned.json'}", flush=True)


def _cv_winner(results_dir: Path) -> str:
    """The model with the highest full-data CV macro F1 among saved results."""
    scores: dict[str, float] = {}
    for path in sorted(results_dir.glob("model_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("cv_macro_f1") is not None:
            scores[rec["name"]] = float(rec["cv_macro_f1"])
    untuned = results_dir / "rf_untuned.json"
    if untuned.exists():
        rf = json.loads(untuned.read_text(encoding="utf-8"))
        scores.setdefault("random_forest", float(rf["cv"]["mean_macro_f1"]))
    if not scores:
        raise RuntimeError("no CV results found; run the models first")
    print("  CV macro F1 by model:", {k: round(v, 4) for k, v in scores.items()}, flush=True)
    return max(scores, key=scores.get)


def stage_tune(model_name: str | None, n_trials: int, tune_rows: int) -> None:
    from pathsnap.tuning.bayes import tune

    results_dir, data, _, _ = _load()
    winner = model_name or _cv_winner(results_dir)
    _banner(f"P3 Bayesian search (Optuna TPE) on '{winner}': {n_trials} trials")
    sub = _subsample_by_observation(data, tune_rows, SEED)
    print(
        f"  objective rows: {len(sub['y']):,} of {len(data['y']):,} "
        f"(whole observations, seed {SEED}); {N_SPLITS}-fold group CV per trial",
        flush=True,
    )
    t0 = time.perf_counter()
    result = tune(
        winner,
        sub["X"], sub["y"], sub["groups"],
        n_trials=n_trials, seed=SEED, results_dir=results_dir, n_splits=N_SPLITS,
        obs_idx=sub["obs_idx"], arc_id=sub["arc_id"],
    )
    elapsed = time.perf_counter() - t0
    summary = {
        "model_name": winner,
        "best_params": result["best_params"],
        "best_objective_value": result["best_value"],
        "n_trials": result["n_trials"],
        "objective_rows": int(len(sub["y"])),
        "search_seconds": elapsed,
        "trials_csv": result["trials_path"],
        "study": result["study_summary"],
    }
    (results_dir / f"tune_summary_{winner}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  best objective (subsample CV macro F1) = {result['best_value']:.4f}", flush=True)
    print(f"  best params = {result['best_params']}", flush=True)
    print(f"  search took {elapsed / 60:.1f} min -> {result['trials_path']}", flush=True)


def stage_final(model_name: str | None) -> None:
    import joblib

    from pathsnap.config import ARTIFACTS_DIR, PROJECT_ROOT
    from pathsnap.evaluation.compare import build_comparison_table, save_record
    from pathsnap.evaluation.cv import cross_validate_model
    from pathsnap.models.base import ModelRecord
    from pathsnap.models.registry import build_model

    results_dir, data, test, trust_test = _load()
    winner = model_name or _cv_winner(results_dir)
    best = json.loads((results_dir / f"tune_summary_{winner}.json").read_text(encoding="utf-8"))
    tuned_params = best["best_params"]
    seed_kw = {"random_state": SEED} if winner == "random_forest" else {}

    _banner(f"P3 final: tuned vs untuned '{winner}' on full training data")
    untuned_model = build_model(winner, **seed_kw)
    tuned_model = build_model(winner, **{**tuned_params, **seed_kw})

    untuned_file = results_dir / "rf_untuned.json"
    if winner == "random_forest" and untuned_file.exists():
        untuned_cv = json.loads(untuned_file.read_text(encoding="utf-8"))["cv"]
        print("  reusing full-data untuned CV from rf_untuned.json", flush=True)
    else:
        untuned_cv = _cv_summary(
            cross_validate_model(
                untuned_model, data["X"], data["y"], data["groups"], n_splits=N_SPLITS, seed=SEED,
                obs_idx=data["obs_idx"], arc_id=data["arc_id"], verbose=True,
            )
        )
    tuned_cv = _cv_summary(
        cross_validate_model(
            tuned_model, data["X"], data["y"], data["groups"], n_splits=N_SPLITS, seed=SEED,
            obs_idx=data["obs_idx"], arc_id=data["arc_id"], verbose=True,
        )
    )

    # Decision rule, fixed before any student-test number is looked at.
    keep_tuned = tuned_cv["mean_macro_f1"] > untuned_cv["mean_macro_f1"]
    label = "tuned" if keep_tuned else "untuned"
    print(
        f"  full-data CV macro F1: untuned {untuned_cv['mean_macro_f1']:.4f}  "
        f"tuned {tuned_cv['mean_macro_f1']:.4f}  -> keep {label}",
        flush=True,
    )

    fits: dict[str, dict] = {}
    for name, model in [("untuned", untuned_model), ("tuned", tuned_model)]:
        t0 = time.perf_counter()
        model.fit(data["X"], data["y"], groups=data["groups"])
        fit_s = time.perf_counter() - t0
        metrics, cont = _score_on_student_test(model, test, trust_test)
        fits[name] = {"model": model, "fit_s": fit_s, "metrics": metrics, "cont": cont}
        print(
            f"  {name:>8}: fit {fit_s:.1f}s  student top-1 = {metrics['top1_accuracy']:.4f}  "
            f"macro F1 = {metrics['macro_f1']:.4f}  log loss = {metrics['log_loss']}",
            flush=True,
        )

    final = fits[label]
    cv_final = tuned_cv if keep_tuned else untuned_cv

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(final["model"], artifact)

    record = ModelRecord(
        name=winner,
        family=final["model"].family,
        config=final["model"].describe(),
        seed=SEED,
        cv_macro_f1=cv_final["mean_macro_f1"],
        test_top1_accuracy=final["metrics"]["top1_accuracy"],
        test_macro_f1=final["metrics"]["macro_f1"],
        test_log_loss=final["metrics"]["log_loss"],
        route_continuity=final["cont"]["mean_route_continuity"],
        train_seconds=final["fit_s"],
        inference_ms_per_obs=final["metrics"]["inference_ms_per_obs"],
        artifact_path=artifact.relative_to(PROJECT_ROOT).as_posix(),
        notes=(
            f"{label} config chosen on full-data CV; {best['n_trials']} Optuna TPE trials "
            f"on a {best['objective_rows']:,}-row observation subsample; full-data CV macro F1 "
            f"untuned {untuned_cv['mean_macro_f1']:.4f} vs tuned {tuned_cv['mean_macro_f1']:.4f}"
        ),
    )
    save_record(record, results_dir)

    comparison = {
        "model_name": winner,
        "kept": label,
        "tuned_params": tuned_params,
        "full_data_cv": {"untuned": untuned_cv, "tuned": tuned_cv},
        "student_test": {k: v["metrics"] for k, v in fits.items()},
        "train_seconds": {k: v["fit_s"] for k, v in fits.items()},
        "final_config": final["model"].describe(),
    }
    (results_dir / f"tuned_vs_untuned_{winner}.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(f"  -> artifacts/model.joblib ({label}); model_{winner}.json and tuned_vs_untuned written", flush=True)
    print(build_comparison_table(results_dir), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 runner: Random Forest + Bayesian tuning")
    parser.add_argument("stage", choices=["untuned", "tune", "final", "all"])
    parser.add_argument("--model", default=None, help="model to tune (default: CV winner)")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--tune-rows", type=int, default=150_000)
    args = parser.parse_args()

    if args.stage in ("untuned", "all"):
        stage_untuned()
    if args.stage in ("tune", "all"):
        stage_tune(args.model, args.n_trials, args.tune_rows)
    if args.stage in ("final", "all"):
        stage_final(args.model)


if __name__ == "__main__":
    main()
