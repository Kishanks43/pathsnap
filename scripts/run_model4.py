"""Model 4 (HistGradientBoosting): fit on `train`, score `student_test`.

OWNER: Person 4

This is the Model 4 counterpart of scripts/run_model1.py. It uses the same
cached feature matrices (schema v1, 19 columns), the same trusted-label filter
and the same observation-level metric code (imported from run_model1), so its
numbers sit in the same table as Model 1's.

Run (needs data/raw/train and data/raw/student_test on this machine):

    python scripts/prepare_dataset.py          # once; builds data/processed/*
    python scripts/run_model4.py

Outputs:
    experiments/results/model_hist_gradient_boosting.json     ModelRecord for the report table
    experiments/results/model4_student_test_predictions.csv   one row per GPS observation
    experiments/results/model4_detail.json                    CV folds + test metrics
    artifacts/hist_gradient_boosting.joblib                   fitted model (gitignored)
    experiments/logs/model4_<timestamp>.log                   console output

STAND-IN MODE (--student-test-only)
-----------------------------------
If data/raw/train is not on this machine, the official run is impossible.
`--student-test-only` runs leave-one-trip-out over the 10 student_test trips
instead: each trip is scored by a model that never saw it. Every number is
out-of-sample, but the training set is 9 trips instead of 80, so it is NOT the
project's Model 4 result. Its files are named `standin_*` so they can never be
mistaken for (or picked up as) a `model_*.json` record, and they must not be
quoted in the report.

There is deliberately no automatic fallback between the two modes: the older
train_model.py fell back to random data without being asked, and that is the
failure this script is built to avoid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

# Same metric definitions as Model 1 (docs/CONTRACTS.md C9). Importing them, not
# copying them, is what keeps the two models comparable.
from run_model1 import evaluate, nearest_candidate_baseline, observation_table
from sklearn.model_selection import GroupKFold

from pathsnap.config import ARTIFACTS_DIR, LOGS_DIR, RESULTS_DIR, ensure_dirs
from pathsnap.evaluation.cv import cross_validate_model, group_kfold
from pathsnap.evaluation.holdout import evaluate_on_test
from pathsnap.features.build import FEATURE_COLUMNS
from pathsnap.features.dataset import StackedFeatures, load_partition
from pathsnap.models.base import ModelRecord, check_classifier
from pathsnap.models.boosting import HistGradientBoostingModel

SEED = 42
N_SPLITS = 5

LOG: list[str] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(msg)


# --------------------------------------------------------------------------- model
def fit_model(X: np.ndarray, y: np.ndarray) -> tuple[HistGradientBoostingModel, float]:
    """Class defaults (lr 0.1, 300 iterations, 31 leaves, early stopping, seed 42)."""
    model = HistGradientBoostingModel(random_state=SEED)
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def score_rows(model: HistGradientBoostingModel, X: np.ndarray) -> np.ndarray:
    """P(candidate is the reference arc) for every row, in row order (CONTRACTS C2)."""
    pos = np.flatnonzero(model.classes_ == 1)
    if pos.size != 1:
        raise RuntimeError(f"expected a binary {{0,1}} target, got classes_={model.classes_}")
    return model.predict_proba(X)[:, int(pos[0])]


def n_iter_used(model: HistGradientBoostingModel) -> int:
    return int(model._clf.n_iter_)


# ----------------------------------------------------------------------- predictions
def obs_metrics(d: StackedFeatures, score: np.ndarray, mask: np.ndarray) -> dict:
    return evaluate(
        observation_table(d.group_index[mask], d.obs_idx[mask], d.arc_id[mask], d.y[mask], score[mask])
    )


def predictions_table(d: StackedFeatures, score: np.ndarray) -> pd.DataFrame:
    """One row per GPS observation of every trip (CONTRACTS C6: -1 = no arc chosen).

    Covers ALL observations, trusted or not: the trust filter only applies to
    fitting and scoring, never to the inference path.
    """
    df = pd.DataFrame(
        {
            "g": d.group_index,
            "obs": d.obs_idx,
            "arc": d.arc_id,
            "y": d.y,
            "s": score,
            "trusted": d.trusted(),
        }
    )
    key = ["g", "obs"]
    top = df.loc[df.groupby(key)["s"].idxmax()].set_index(key)[["arc", "s", "trusted"]]
    top.columns = ["pred_arc_id", "pred_prob", "trusted"]
    ref = df[df["y"] == 1].groupby(key)["arc"].first().rename("ref_arc_id")
    n_cand = df.groupby(key).size().rename("n_candidates")
    out = top.join(ref).join(n_cand)

    full = pd.MultiIndex.from_arrays(
        [
            np.repeat(np.arange(len(d.trip_ids)), d.n_obs_per_trip),
            np.concatenate([np.arange(n) for n in d.n_obs_per_trip]),
        ],
        names=key,
    )
    out = out.reindex(full)
    out["pred_arc_id"] = out["pred_arc_id"].fillna(-1).astype(np.int64)
    out["n_candidates"] = out["n_candidates"].fillna(0).astype(np.int64)
    out["trusted"] = out["trusted"].fillna(False).astype(bool)
    # ref_arc_id is blank when the reference arc is not among the candidates (a miss).
    out["correct"] = out["pred_arc_id"].eq(out["ref_arc_id"])
    out = out.reset_index()
    out.insert(0, "trip_id", [d.trip_ids[g] for g in out.pop("g")])
    out = out.rename(columns={"obs": "obs_idx"})
    return out[["trip_id", "obs_idx", "pred_arc_id", "pred_prob", "n_candidates", "trusted", "ref_arc_id", "correct"]]


def per_trip_table(d: StackedFeatures, score: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    rows = []
    for g, trip_id in enumerate(d.trip_ids):
        m = mask & (d.group_index == g)
        if not m.any():
            continue
        r = obs_metrics(d, score, m)
        rows.append({"trip_id": trip_id, "n_obs": r["n_obs"], "top1_accuracy": round(r["top1_accuracy"], 4),
                     "candidate_miss_rate": round(r["candidate_miss_rate"], 4)})
    return pd.DataFrame(rows)


def summarise_predictions(pred: pd.DataFrame) -> None:
    n = len(pred)
    none = int((pred["pred_arc_id"] < 0).sum())
    say(f"  predictions: {n:,} observations, {n - none:,} with a predicted arc, "
        f"{none:,} with no candidates (pred_arc_id = -1)")


# ------------------------------------------------------------------------- official
def grouped_cv(d: StackedFeatures, mask: np.ndarray) -> dict:
    """Score with the shared harness so Model 4 is comparable with the other four."""
    X, y, groups = d.X[mask], d.y[mask], d.group_index[mask]
    obs, arc = d.obs_idx[mask], d.arc_id[mask]

    cv = cross_validate_model(
        HistGradientBoostingModel(random_state=SEED),
        X, y, groups, n_splits=N_SPLITS, seed=SEED,
        obs_idx=obs, arc_id=arc,
    )
    valid_trips = [len(np.unique(groups[va])) for _, va in
                   group_kfold(X, y, groups, n_splits=N_SPLITS, seed=SEED)]
    folds = []
    for k, (f, n_trips) in enumerate(zip(cv.folds, valid_trips), 1):
        m = {
            "fold": k,
            "macro_f1": f.macro_f1,
            "top1_accuracy": f.top1_accuracy,
            "log_loss": f.log_loss,
            "n_trips_valid": n_trips,
        }
        folds.append(m)
        ll = "N/A" if f.log_loss is None else f"{f.log_loss:.4f}"
        say(f"    fold {k}: macro_f1={m['macro_f1']:.4f} top1={m['top1_accuracy']:.4f} "
            f"log_loss={ll} trips_valid={n_trips}")

    return {
        "folds": folds,
        "cv_macro_f1_mean": cv.mean_macro_f1,
        "cv_macro_f1_std": cv.std_macro_f1,
        "cv_top1_mean": cv.mean_top1_accuracy,
        "cv_log_loss_mean": cv.mean_log_loss,
    }


def run_official(started: datetime) -> int:
    try:
        train = load_partition("train")
    except FileNotFoundError as e:
        say(f"ERROR: {e}")
        say("The train partition is not prepared on this machine. Either copy data/raw/train in and run")
        say("`python scripts/prepare_dataset.py`, or use `--student-test-only` for a clearly-labelled stand-in.")
        return 2
    test = load_partition("student_test")
    tr_mask, te_mask = train.trusted(), test.trusted()
    say(f"train : {len(train.trip_ids)} trips, {int(tr_mask.sum()):,} trusted rows of {len(train.y):,}")
    say(f"test  : {len(test.trip_ids)} trips, {int(te_mask.sum()):,} trusted rows of {len(test.y):,}")
    say(f"features: {len(FEATURE_COLUMNS)} columns (schema v1)")

    say(f"\n[cv] {N_SPLITS}-fold GroupKFold over {len(train.trip_ids)} training trips")
    cv = grouped_cv(train, tr_mask)
    say(f"  CV macro F1 = {cv['cv_macro_f1_mean']:.4f} +/- {cv['cv_macro_f1_std']:.4f}")

    say("\n[final] refit on all trusted training rows, score student_test")
    model, t_fit = fit_model(train.X[tr_mask], train.y[tr_mask])
    check_classifier(model, test.X[te_mask][:5000])

    test_m, _ = evaluate_on_test(model, test, te_mask)
    obs_key = test.group_index[te_mask].astype(np.int64) * 10_000_000 + test.obs_idx[te_mask]
    covered = pd.DataFrame({"obs": obs_key, "y": test.y[te_mask]}).groupby("obs")["y"].max()
    test_m["candidate_miss_rate"] = float(1.0 - covered.mean())
    inf_ms = test_m["inference_ms_per_obs"]

    # One extra inference pass for the per-trip table and the predictions CSV.
    score = score_rows(model, test.X)
    base_m = nearest_candidate_baseline(test, te_mask)
    say(f"  test (HistGB)          : {test_m}")
    say(f"  test (nearest, no ML)  : {base_m}")
    say(f"  train_seconds={t_fit:.1f} iterations_used={n_iter_used(model)} inference_ms_per_obs={inf_ms:.4f}")

    per_trip = per_trip_table(test, score, te_mask)
    say("\n  per-trip top-1 on student_test:")
    say(per_trip.to_string(index=False))

    pred = predictions_table(test, score)
    summarise_predictions(pred)
    pred_path = RESULTS_DIR / "model4_student_test_predictions.csv"
    pred.to_csv(pred_path, index=False)

    artifact_path = ARTIFACTS_DIR / "hist_gradient_boosting.joblib"
    joblib.dump(model, artifact_path)

    record = ModelRecord(
        name="hist_gradient_boosting",
        family="boosting",
        config={**model.describe(), "n_iter_used": n_iter_used(model), "features": "all_19",
                "label_trust_m": 25.0, "cv": f"GroupKFold({N_SPLITS}) by trip, shared evaluation/cv.py"},
        seed=SEED,
        cv_macro_f1=cv["cv_macro_f1_mean"],
        test_top1_accuracy=test_m["top1_accuracy"],
        test_macro_f1=test_m["macro_f1"],
        test_log_loss=test_m["log_loss"],
        candidate_miss_rate=test_m["candidate_miss_rate"],
        route_continuity=None,
        train_seconds=t_fit,
        inference_ms_per_obs=inf_ms,
        artifact_path=str(artifact_path.relative_to(ARTIFACTS_DIR.parent)),
        notes=(
            "route_continuity N/A: requires the HMM-decoded route. Metrics are observation-level on "
            f"trusted labels. CV std macro F1 = {cv['cv_macro_f1_std']:.4f} over {N_SPLITS} folds. "
            "sklearn's early stopping holds out a random 10% of ROWS (not whole trips) as its internal "
            "validation set, so the stopping point can be later than a trip-aware split would choose."
        ),
    )
    record_path = RESULTS_DIR / "model_hist_gradient_boosting.json"
    record_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    detail_path = RESULTS_DIR / "model4_detail.json"
    detail_path.write_text(
        json.dumps({"cv": cv, "test": test_m, "nearest_candidate_test": base_m,
                    "per_trip": per_trip.to_dict("records")}, indent=2, default=float),
        encoding="utf-8",
    )
    say(f"\nwrote {record_path}\nwrote {pred_path}\nwrote {detail_path}\nwrote {artifact_path}")
    return 0


# ------------------------------------------------------------------------- stand-in
def run_standin(started: datetime) -> int:
    say("*** STAND-IN MODE: leave-one-trip-out over student_test. NOT the official Model 4 number. ***")
    try:
        d = load_partition("student_test")
    except FileNotFoundError as e:
        say(f"ERROR: {e}")
        say("Run `python scripts/prepare_dataset.py --partition student_test` first.")
        return 2
    mask = d.trusted()
    n_trips = len(d.trip_ids)
    say(f"student_test: {n_trips} trips, {int(mask.sum()):,} trusted rows of {len(d.y):,}")
    say(f"each fold trains on {n_trips - 1} trips and scores the one it never saw\n")

    score = np.full(len(d.y), np.nan)
    iters = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_trips).split(d.X, d.y, d.group_index), 1):
        tr = tr[mask[tr]]  # fit on trusted rows only
        model, t_fit = fit_model(d.X[tr], d.y[tr])
        score[va] = score_rows(model, d.X[va])
        iters.append(n_iter_used(model))
        say(f"  held out {d.trip_ids[int(d.group_index[va[0]])]}: fit {t_fit:.1f}s, iterations used {iters[-1]}")
    assert not np.isnan(score).any()

    m = obs_metrics(d, score, mask)
    base = nearest_candidate_baseline(d, mask)
    say(f"\n  out-of-sample (HistGB, leave-one-trip-out): {m}")
    say(f"  nearest candidate, no ML                  : {base}")
    per_trip = per_trip_table(d, score, mask)
    say("\n  per-trip top-1 (each trip scored by a model trained on the other 9):")
    say(per_trip.to_string(index=False))

    pred = predictions_table(d, score)
    summarise_predictions(pred)
    pred_path = RESULTS_DIR / "standin_model4_loto_predictions.csv"
    pred.to_csv(pred_path, index=False)
    summary_path = RESULTS_DIR / "standin_model4_loto_summary.json"
    summary_path.write_text(
        json.dumps({"mode": "leave-one-trip-out over student_test (stand-in, not the official Model 4)",
                    "seed": SEED, "metrics": m, "nearest_candidate": base,
                    "per_trip": per_trip.to_dict("records"), "iterations_used_per_fold": iters},
                   indent=2, default=float),
        encoding="utf-8",
    )
    say(f"\nwrote {pred_path}\nwrote {summary_path}")
    say("Reminder: stand-in numbers are for a sanity check only; do not put them in the report.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-test-only", action="store_true",
                    help="stand-in: leave-one-trip-out over student_test (use only when data/raw/train is missing)")
    args = ap.parse_args()

    ensure_dirs()
    started = datetime.now()
    say(f"Model 4 (HistGradientBoosting) run started {started.isoformat(timespec='seconds')}")
    rc = run_standin(started) if args.student_test_only else run_official(started)

    tag = "model4_standin" if args.student_test_only else "model4"
    log_path = LOGS_DIR / f"{tag}_{started.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say(f"log: {log_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
