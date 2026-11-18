#!/usr/bin/env python
"""P2 runner: train KNN & RBF SVM, cross-validate, evaluate on student_test,
measure Platt-scaling and KNN inference cost, persist ModelRecords and build
the comparison table.

Usage
-----
    python scripts/run_p2.py                        # full run (KNN + SVM)
    python scripts/run_p2.py --skip-svm             # KNN only (faster)
    python scripts/run_p2.py --skip-knn             # SVM only
    python scripts/run_p2.py --table-only           # rebuild the table, no training
    python scripts/run_p2.py --knn-subsample 75000  # smaller KNN training cap

Why the two subsample caps exist
--------------------------------
* KNN is brute force: one held-out fold is a ~270k x 1.16M distance matrix. The
  cap is the largest training size whose full 5-fold sweep finished in ~10 min
  on a 16-core laptop (measured inference ~0.08 ms/candidate row at 100k
  reference rows, so ~23 s per 270k-row fold).
* RBF SVM is O(n^2..n^3) and `probability=True` adds internal 5-fold Platt
  scaling. 30k rows is the largest sample whose 5-fold CV fits this run budget:
  in the recorded run a 30k fit took ~23-27 s and each fold (fit + scoring
  270k rows) 78-104 s. Validation folds are NEVER subsampled, and the cap plus
  the actual row count is written into the record so nobody can mistake this
  for a full-data run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


def _banner(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")
    sys.stdout.flush()


def _candidate_miss_rate(partition: str) -> float | None:
    """Read the prepared-dataset miss rate from candidate_quality.json.

    C10 says candidate quality is quoted from this file, never recomputed by
    hand in a model script.
    """
    from pathsnap.config import RESULTS_DIR

    path = RESULTS_DIR / "candidate_quality.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(partition, {}).get("candidate_miss_rate_mean")


from pathsnap.evaluation.holdout import evaluate_on_test, mean_route_continuity  # noqa: E402


def save_artifact(model, filename: str) -> str:
    """Persist a fitted model under artifacts/ and return a repo-relative path."""
    import joblib

    from pathsnap.config import ARTIFACTS_DIR, PROJECT_ROOT

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    joblib.dump(model, path)
    return path.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 runner: KNN + RBF SVM pipeline")
    parser.add_argument("--skip-svm", action="store_true", help="Skip the SVM model.")
    parser.add_argument("--skip-knn", action="store_true", help="Skip the KNN model.")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Only rebuild comparison.csv/.md from existing records (no training).",
    )
    parser.add_argument(
        "--knn-subsample",
        type=int,
        default=100_000,
        help="Max KNN training rows per fold (default: 100000, see module docstring).",
    )
    parser.add_argument(
        "--svm-subsample",
        type=int,
        default=30_000,
        help="Max SVM training rows per fold (default: 30000, see module docstring).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    if args.table_only:
        from pathsnap.config import RESULTS_DIR
        from pathsnap.evaluation.compare import (
            assert_five_models,
            build_comparison_table,
        )

        _banner("Comparison table (no training)")
        print(build_comparison_table(RESULTS_DIR))
        try:
            assert_five_models(RESULTS_DIR)
            print("  Five-model guard: PASS")
        except AssertionError as exc:
            print(
                f"  Five-model guard: INCOMPLETE -- {exc}\n"
                "  This is expected until the remaining owners save their records "
                "with save_record(); do not fabricate rows."
            )
        return

    if args.skip_knn and args.skip_svm:
        parser.error("--skip-knn and --skip-svm cannot both be set")

    run_started = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    _banner("1. Loading data")
    try:
        from pathsnap.config import CONFIGS_DIR, RESULTS_DIR, ensure_dirs
        from pathsnap.features.dataset import load_partition

        ensure_dirs()
        train = load_partition("train")
        test = load_partition("student_test")
    except FileNotFoundError as exc:
        print(
            f"ERROR: data not available -- {exc}\n"
            "Run `python scripts/prepare_dataset.py` first to build the cached "
            "feature matrices.",
            file=sys.stderr,
        )
        sys.exit(1)

    trust_train = train.trusted()
    trust_test = test.trusted()

    X_train = train.X[trust_train]
    y_train = train.y[trust_train]
    groups_train = train.groups[trust_train]
    obs_idx_train = train.obs_idx[trust_train]
    arc_id_train = train.arc_id[trust_train]

    print(f"Train: {len(X_train):,} trusted rows ({trust_train.sum():,}/{len(train.X):,})")
    X_test = test.X[trust_test]
    print(f"Test:  {len(X_test):,} trusted rows ({trust_test.sum():,}/{len(test.X):,})")
    print(f"Trips: {len(train.trip_ids)} train / {len(test.trip_ids)} student_test")
    print(f"Features: {X_train.shape[1]}")

    miss_rate = _candidate_miss_rate("student_test")
    config_snapshot = {
        "run": "p2_knn_svm",
        "argv": sys.argv[1:],
        "utc_started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "n_splits": args.n_splits,
        "knn_subsample": args.knn_subsample,
        "svm_subsample": args.svm_subsample,
        "n_train_trusted_rows": int(len(X_train)),
        "n_test_trusted_rows": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "student_test_candidate_miss_rate": miss_rate,
        "python": platform.python_version(),
    }
    try:
        import sklearn

        config_snapshot["scikit_learn"] = sklearn.__version__
    except Exception:  # pragma: no cover - informational only
        pass
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIGS_DIR / "p2_run.json").write_text(
        json.dumps(config_snapshot, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 2. KNN (Model 2): k-selection via group-aware CV, then final fit
    # ------------------------------------------------------------------
    knn = None
    knn_metrics = None
    knn_cont = None
    best_cv_knn = None
    best_k = 5

    if not args.skip_knn:
        from pathsnap.evaluation.cv import cross_validate_model
        from pathsnap.models.instance_based import (
            KNNModel,
            measure_knn_inference_cost,
        )

        _banner("2. KNN k-selection (group-aware CV)")
        knn_sweep: list[dict] = []
        for k in [3, 5, 7, 11]:
            model = KNNModel(
                n_neighbors=k,
                max_train_samples=args.knn_subsample,
                random_state=args.seed,
            )
            cv = cross_validate_model(
                model,
                X_train,
                y_train,
                groups_train,
                n_splits=args.n_splits,
                seed=args.seed,
                obs_idx=obs_idx_train,
                arc_id=arc_id_train,
                verbose=True,
            )
            print(
                f"  k={k:>2d}: CV macro F1 = {cv.mean_macro_f1:.4f} "
                f"+/- {cv.std_macro_f1:.4f}  |  top-1 = {cv.mean_top1_accuracy:.4f}  |  "
                f"log loss = {cv.mean_log_loss}",
                flush=True,
            )
            knn_sweep.append(
                {
                    "k": k,
                    "mean_macro_f1": cv.mean_macro_f1,
                    "std_macro_f1": cv.std_macro_f1,
                    "mean_top1_accuracy": cv.mean_top1_accuracy,
                    "mean_log_loss": cv.mean_log_loss,
                    "folds": [f.__dict__ for f in cv.folds],
                }
            )
            if best_cv_knn is None or cv.mean_macro_f1 > best_cv_knn.mean_macro_f1:
                best_k = k
                best_cv_knn = cv

        assert best_cv_knn is not None
        print(
            f"\n  -> Best k = {best_k} "
            f"(CV macro F1 = {best_cv_knn.mean_macro_f1:.4f} "
            f"+/- {best_cv_knn.std_macro_f1:.4f})"
        )

        _banner("3. KNN inference-cost scaling")
        knn_cost = measure_knn_inference_cost(
            X_train,
            y_train,
            train_sizes=(25_000, 50_000, 100_000),
            n_query=10_000,
            n_neighbors=best_k,
            random_state=args.seed,
        )
        for row in knn_cost["sizes"]:
            print(
                f"  n_train={row['n_train']:>7,}: {row['ms_per_row']:.4f} ms per "
                f"candidate row (query n={row['n_query']:,})"
            )
        (RESULTS_DIR / "knn_inference_cost.json").write_text(
            json.dumps(knn_cost, indent=2), encoding="utf-8"
        )

        _banner("4. Training final KNN")
        knn = KNNModel(
            n_neighbors=best_k,
            max_train_samples=args.knn_subsample,
            random_state=args.seed,
        )
        knn.fit(X_train, y_train, groups=groups_train)
        print(
            f"  KNN fitted (k={best_k}, {knn.n_train_samples_:,}/{knn.n_train_total_:,} "
            f"rows, strategy={knn.subsample_strategy_}, {knn.fit_seconds:.2f}s)"
        )

        (RESULTS_DIR / "cv_knn.json").write_text(
            json.dumps({"k_sweep": knn_sweep, "best_cv": best_cv_knn.to_dict()}, indent=2),
            encoding="utf-8",
        )

        _banner("5. KNN student-test evaluation")
        knn_metrics, knn_pred_obs = evaluate_on_test(knn, test, trust_test)
        knn_cont = mean_route_continuity(test, trust_test, knn_pred_obs)
        print(
            f"  KNN -> top-1 = {knn_metrics['top1_accuracy']:.4f}  |  "
            f"macro F1 = {knn_metrics['macro_f1']:.4f}  |  "
            f"log loss = {knn_metrics['log_loss']}  |  "
            f"zero-p(true) = {knn_metrics['zero_true_probability_fraction']:.4%}  |  "
            f"continuity = {knn_cont['mean_route_continuity']}  |  "
            f"infer = {knn_metrics['inference_ms_per_obs']:.3f} ms/obs"
        )
    else:
        print("\n  [--skip-knn] Skipping KNN model.\n")

    # ------------------------------------------------------------------
    # SVM (Model 5): explicit stratified subsampling + Platt cost
    # ------------------------------------------------------------------
    svm = None
    svm_metrics = None
    svm_cont = None
    cv_svm = None
    platt_cost = None

    if not args.skip_svm:
        from pathsnap.evaluation.cv import cross_validate_model
        from pathsnap.models.kernel import SVMModel, measure_platt_scaling_cost

        _banner("6. SVM (RBF): Platt-scaling cost")
        print(
            f"  Subsample cap: {args.svm_subsample:,} rows (from {len(X_train):,}), "
            "stratified by binary label y to preserve the ~12.7% positive rate."
        )
        platt_cost = measure_platt_scaling_cost(
            X_train,
            y_train,
            C=1.0,
            gamma="scale",
            max_train_samples=args.svm_subsample,
            random_state=args.seed,
        )
        print(
            f"  Platt scaling overhead: raw={platt_cost['fit_no_platt_s']:.2f}s vs "
            f"Platt={platt_cost['fit_with_platt_s']:.2f}s "
            f"({platt_cost['overhead_ratio']:.2f}x, +{platt_cost['extra_seconds']:.2f}s, "
            f"{platt_cost['n_support_vectors']:,} support vectors)"
        )

        _banner("7. SVM (RBF) group-aware CV")
        # The cap is applied inside each training fold; the held-out fold is
        # always scored in full, so there is no leakage from validation rows.
        cv_svm = cross_validate_model(
            SVMModel(
                C=1.0,
                gamma="scale",
                probability=True,
                max_train_samples=args.svm_subsample,
                random_state=args.seed,
            ),
            X_train,
            y_train,
            groups_train,
            n_splits=args.n_splits,
            seed=args.seed,
            obs_idx=obs_idx_train,
            arc_id=arc_id_train,
            verbose=True,
        )
        print(
            f"  SVM CV: macro F1 = {cv_svm.mean_macro_f1:.4f} "
            f"+/- {cv_svm.std_macro_f1:.4f}  |  top-1 = {cv_svm.mean_top1_accuracy:.4f}  |  "
            f"log loss = {cv_svm.mean_log_loss}"
        )

        _banner("8. Training final SVM")
        svm = SVMModel(
            C=1.0,
            gamma="scale",
            probability=True,
            max_train_samples=args.svm_subsample,
            random_state=args.seed,
        )
        svm.fit(X_train, y_train, groups=groups_train)
        print(
            f"  SVM fitted ({svm.n_train_used_:,}/{svm.n_train_total_:,} rows, "
            f"strategy={svm.subsample_strategy_}, {svm.n_support_vectors_:,} SVs, "
            f"{svm.fit_seconds:.2f}s)"
        )

        (RESULTS_DIR / "cv_svm_rbf.json").write_text(
            json.dumps({"platt_cost": platt_cost, "cv": cv_svm.to_dict()}, indent=2),
            encoding="utf-8",
        )

        _banner("9. SVM student-test evaluation")
        svm_metrics, svm_pred_obs = evaluate_on_test(svm, test, trust_test)
        svm_cont = mean_route_continuity(test, trust_test, svm_pred_obs)
        print(
            f"  SVM -> top-1 = {svm_metrics['top1_accuracy']:.4f}  |  "
            f"macro F1 = {svm_metrics['macro_f1']:.4f}  |  "
            f"log loss = {svm_metrics['log_loss']}  |  "
            f"zero-p(true) = {svm_metrics['zero_true_probability_fraction']:.4%}  |  "
            f"continuity = {svm_cont['mean_route_continuity']}  |  "
            f"infer = {svm_metrics['inference_ms_per_obs']:.3f} ms/obs"
        )
    else:
        print("\n  [--skip-svm] Skipping SVM model.\n")

    # ------------------------------------------------------------------
    # Save model records
    # ------------------------------------------------------------------
    _banner("10. Saving model records")
    from pathsnap.evaluation.compare import (
        assert_five_models,
        build_comparison_table,
        save_record,
    )
    from pathsnap.models.base import ModelRecord

    if knn is not None and knn_metrics is not None and best_cv_knn is not None:
        knn_artifact = save_artifact(knn, "p2_knn.joblib")
        knn_record = ModelRecord(
            name="knn",
            family="instance-based",
            config=knn.describe(),
            seed=args.seed,
            cv_macro_f1=best_cv_knn.mean_macro_f1,
            test_top1_accuracy=knn_metrics["top1_accuracy"],
            test_macro_f1=knn_metrics["macro_f1"],
            test_log_loss=knn_metrics["log_loss"],
            candidate_miss_rate=miss_rate,
            route_continuity=knn_cont["mean_route_continuity"],
            train_seconds=knn.fit_seconds,
            inference_ms_per_obs=knn_metrics["inference_ms_per_obs"],
            artifact_path=knn_artifact,
            notes=(
                f"CV macro F1 std={best_cv_knn.std_macro_f1:.4f}; "
                f"k selected from {{3,5,7,11}} by the same 5-fold group-aware CV, "
                "so the recorded CV score is optimistically biased by model selection; "
                f"trained on {knn.n_train_samples_:,}/{knn.n_train_total_:,} rows "
                f"(label-stratified subsample, cap={args.knn_subsample:,}); "
                f"zero-probability true labels on {knn_metrics['zero_true_probability_fraction']:.4%} "
                f"of test candidate rows (log loss is infinite when >0); "
                "probabilities are k-neighbour votes, not calibrated"
            ),
        )
        save_record(knn_record, RESULTS_DIR)
        print(f"  -> {RESULTS_DIR / 'model_knn.json'} (artifact: {knn_artifact})")

    if svm is not None and cv_svm is not None and svm_metrics is not None:
        svm_artifact = save_artifact(svm, "p2_svm_rbf.joblib")
        platt_note = (
            f"Platt overhead={platt_cost['overhead_ratio']:.2f}x "
            f"({platt_cost['fit_no_platt_s']:.1f}s -> {platt_cost['fit_with_platt_s']:.1f}s)"
            if platt_cost
            else ""
        )
        svm_record = ModelRecord(
            name="svm_rbf",
            family="kernel-based",
            config=svm.describe(),
            seed=args.seed,
            cv_macro_f1=cv_svm.mean_macro_f1,
            test_top1_accuracy=svm_metrics["top1_accuracy"],
            test_macro_f1=svm_metrics["macro_f1"],
            test_log_loss=svm_metrics["log_loss"],
            candidate_miss_rate=miss_rate,
            route_continuity=svm_cont["mean_route_continuity"],
            train_seconds=svm.fit_seconds,
            inference_ms_per_obs=svm_metrics["inference_ms_per_obs"],
            artifact_path=svm_artifact,
            notes=(
                f"CV macro F1 std={cv_svm.std_macro_f1:.4f}; "
                f"stratified row subsample {svm.n_train_used_:,}/{svm.n_train_total_:,} "
                f"(cap={args.svm_subsample:,}); validation folds scored in full; "
                f"zero-probability true labels on {svm_metrics['zero_true_probability_fraction']:.4%} "
                f"of test candidate rows; "
                f"{platt_note}"
            ),
        )
        save_record(svm_record, RESULTS_DIR)
        print(f"  -> {RESULTS_DIR / 'model_svm_rbf.json'} (artifact: {svm_artifact})")

    # ------------------------------------------------------------------
    # Comparison table & five-model guard
    # ------------------------------------------------------------------
    _banner("11. Comparison table")
    table = build_comparison_table(RESULTS_DIR)
    print(table)

    try:
        assert_five_models(RESULTS_DIR)
        print("  Five-model guard: PASS")
    except AssertionError as exc:
        print(
            f"  Five-model guard: INCOMPLETE -- {exc}\n"
            "  This is expected until the remaining owners save their records "
            "with save_record(); do not fabricate rows."
        )

    config_snapshot["wall_clock_seconds_last_invocation"] = round(
        time.perf_counter() - run_started, 1
    )
    (CONFIGS_DIR / "p2_run.json").write_text(
        json.dumps(config_snapshot, indent=2), encoding="utf-8"
    )
    _banner("Done")


if __name__ == "__main__":
    main()
