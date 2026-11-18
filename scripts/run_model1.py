"""Model 1 (Logistic Regression): group-aware CV, student-test evaluation, feature ablation.

OWNER: Person 1

Run:
    python scripts/run_model1.py

Outputs (all committed, all reproducible from this script):
    experiments/results/model_logreg.json      ModelRecord for the report table
    experiments/results/model1_ablation.csv    geometry-only vs all features, + nearest-candidate baseline
    experiments/logs/model1_<timestamp>.log    full console output

NOTE ON THE SPLITTER
--------------------
CV and student-test scoring use P2's shared harness
(`evaluation/cv.cross_validate_model` and `evaluation/holdout.evaluate_on_test`),
the same code KNN, SVM and Random Forest use. Fold assignment is still
GroupKFold(5) by trip; arc ids are namespaced per trip and log loss is the
candidate-row value, as in every other model record.

METRIC DEFINITIONS (observation level, see docs/CONTRACTS.md C9)
-----------------------------------------------------------------
- Scores are evaluated on trusted observations only (label residual <= 25 m).
- Per observation: pick the candidate with the highest score (argmax).
- top1_accuracy : fraction of trusted observations whose argmax is the reference
                  arc. Observations whose reference arc is missing from the
                  candidate set count as wrong.
- macro_f1      : macro F1 over reference arc ids, computed on observations whose
                  reference arc IS in the candidate set (labels are undefined otherwise).
- log_loss      : model records use the shared harness's candidate-row value
                  (`safe_log_loss`), so they are comparable with KNN/SVM/RF. Only
                  the nearest-candidate baseline uses the observation-level
                  renormalised value described below; it reports N/A instead.
- route_continuity : N/A here. It needs the decoded route, which is produced by
                  the HMM stage (P4). Recorded as null, not estimated.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from pathsnap.config import LOGS_DIR, RESULTS_DIR, ensure_dirs
from pathsnap.evaluation.cv import cross_validate_model, group_kfold
from pathsnap.evaluation.holdout import evaluate_on_test
from pathsnap.evaluation.metrics import macro_f1
from pathsnap.features.build import FEATURE_COLUMNS
from pathsnap.features.dataset import StackedFeatures, load_partition
from pathsnap.models.base import ModelRecord
from pathsnap.models.linear import LogisticRegressionModel

SEED = 42
N_SPLITS = 5
EPS = 1e-15
OBS_KEY_STRIDE = 10_000_000

GEOMETRY_COLS = list(range(10))            # first 10 columns: candidate geometry + network context
ALL_COLS = list(range(len(FEATURE_COLUMNS)))  # 19 columns, schema v1

LOG: list[str] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(msg)


def observation_table(group, obs_idx, arc_id, y, score) -> pd.DataFrame:
    """One row per observation: argmax arc, reference arc (if covered), renormalised p(ref)."""
    df = pd.DataFrame(
        {
            "obs": group.astype(np.int64) * OBS_KEY_STRIDE + obs_idx.astype(np.int64),
            "arc": arc_id,
            "y": y,
            "s": score,
        }
    )
    top = df.loc[df.groupby("obs")["s"].idxmax(), ["obs", "arc", "y"]].set_index("obs")
    top = top.rename(columns={"arc": "pred_arc", "y": "top_is_ref"})

    pos = df[df["y"] == 1].groupby("obs").agg(ref_arc=("arc", "first"), s_ref=("s", "first"))
    tot = df.groupby("obs")["s"].sum().rename("s_total")

    out = top.join(pos, how="left").join(tot, how="left")
    out["has_ref"] = out["ref_arc"].notna()
    out["top1"] = out["top_is_ref"] == 1
    denom = out["s_total"].where(out["s_total"] > 0)
    out["p_ref"] = (out["s_ref"] / denom).fillna(0.0)
    return out


def evaluate(table: pd.DataFrame) -> dict:
    covered = table[table["has_ref"]]
    f1 = macro_f1(covered["ref_arc"].astype(np.int64).to_numpy(), covered["pred_arc"].to_numpy())
    p = np.clip(covered["p_ref"].to_numpy(), EPS, 1.0)
    return {
        "n_obs": len(table),
        "n_obs_covered": len(covered),
        "top1_accuracy": float(table["top1"].mean()),
        "top1_accuracy_covered": float(covered["top1"].mean()) if len(covered) else float("nan"),
        "macro_f1": float(f1),
        "log_loss": float(-np.mean(np.log(p))) if len(p) else float("nan"),
        "candidate_miss_rate": float(1.0 - table["has_ref"].mean()),
    }


def fit_model(X: np.ndarray, y: np.ndarray, cols: list[int]):
    model = LogisticRegressionModel(C=1.0, max_iter=2000)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X[:, cols], y)
    elapsed = time.perf_counter() - t0
    n_conv = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return model, elapsed, n_conv


def score_rows(model, X: np.ndarray, cols: list[int]) -> np.ndarray:
    return model.predict_proba(X[:, cols])[:, 1]


def grouped_cv(d: StackedFeatures, mask: np.ndarray, cols: list[int]) -> dict:
    """Score with P2's shared harness so every model uses one implementation."""
    X, y, groups = d.X[:, cols][mask], d.y[mask], d.group_index[mask]

    cv = cross_validate_model(
        LogisticRegressionModel(C=1.0, max_iter=2000),
        X, y, groups, n_splits=N_SPLITS, seed=SEED,
        obs_idx=d.obs_idx[mask], arc_id=d.arc_id[mask],
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


def nearest_candidate_baseline(d: StackedFeatures, mask: np.ndarray, dist_col: int = 0) -> dict:
    """No model: pick the candidate with the smallest GPS-to-arc distance.

    Its scores are raw negative distances, not probabilities, so log loss is
    undefined and reported as None (N/A) rather than a meaningless number.
    """
    s = -d.X[mask, dist_col]
    m = evaluate(observation_table(d.group_index[mask], d.obs_idx[mask], d.arc_id[mask], d.y[mask], s))
    m["log_loss"] = None
    return m


def student_test(model, d: StackedFeatures, mask: np.ndarray, cols: list[int]) -> tuple[dict, float]:
    """Student-test scoring through P2's shared held-out helper."""
    from types import SimpleNamespace

    view = SimpleNamespace(
        X=d.X[:, cols], y=d.y, groups=d.group_index,
        obs_idx=d.obs_idx, arc_id=d.arc_id,
    )
    shared, _ = evaluate_on_test(model, view, mask)

    obs_key = d.group_index[mask].astype(np.int64) * OBS_KEY_STRIDE + d.obs_idx[mask]
    covered = pd.DataFrame({"obs": obs_key, "y": d.y[mask]}).groupby("obs")["y"].max()
    m = {
        "top1_accuracy": shared["top1_accuracy"],
        "macro_f1": shared["macro_f1"],
        "log_loss": shared["log_loss"],
        "candidate_miss_rate": float(1.0 - covered.mean()),
        "n_obs": shared["n_observations"],
    }
    return m, shared["inference_ms_per_obs"]


def main() -> int:
    ensure_dirs()
    started = datetime.now()
    say(f"Model 1 run started {started.isoformat(timespec='seconds')}")

    train = load_partition("train")
    test = load_partition("student_test")
    tr_mask = train.trusted()
    te_mask = test.trusted()
    say(f"train : {len(train.trip_ids)} trips, {int(tr_mask.sum()):,} trusted rows of {len(train.y):,}")
    say(f"test  : {len(test.trip_ids)} trips, {int(te_mask.sum()):,} trusted rows of {len(test.y):,}")
    say(f"features: {len(FEATURE_COLUMNS)} columns (schema v1)")

    results: dict[str, dict] = {}

    say("\n[baseline] nearest candidate, no model")
    results["nearest_candidate"] = {
        "cv": None,
        "test": nearest_candidate_baseline(test, te_mask),
    }
    results["nearest_candidate"]["train"] = nearest_candidate_baseline(train, tr_mask)
    say(f"  train: {results['nearest_candidate']['train']}")
    say(f"  test : {results['nearest_candidate']['test']}")

    configs = {"geometry_only": GEOMETRY_COLS, "all_19": ALL_COLS}
    for name, cols in configs.items():
        say(f"\n[{name}] {len(cols)} columns: {[FEATURE_COLUMNS[c] for c in cols][:4]}...")
        say(f"  {N_SPLITS}-fold GroupKFold over {len(train.trip_ids)} training trips")
        cv = grouped_cv(train, tr_mask, cols)
        say(f"  CV macro F1 = {cv['cv_macro_f1_mean']:.4f} +/- {cv['cv_macro_f1_std']:.4f}")

        say("  refit on all trusted training rows, score student_test")
        model, t_fit, n_conv = fit_model(train.X[tr_mask], train.y[tr_mask], cols)
        test_m, inf_ms = student_test(model, test, te_mask, cols)
        say(f"  test: {test_m}")
        say(f"  train_seconds={t_fit:.1f} inference_ms_per_obs={inf_ms:.4f} convergence_warnings={n_conv}")

        results[name] = {
            "cv": cv,
            "test": test_m,
            "train_seconds": t_fit,
            "inference_ms_per_obs": inf_ms,
            "convergence_warnings": n_conv,
            "columns": [FEATURE_COLUMNS[c] for c in cols],
            "model": model,
        }

    # --- ablation table -----------------------------------------------------------
    rows = []
    for name in ("nearest_candidate", "geometry_only", "all_19"):
        r = results[name]
        cv = r.get("cv")
        rows.append({
            "config": name,
            "n_features": 1 if name == "nearest_candidate" else len(r["columns"]),
            "cv_macro_f1_mean": round(cv["cv_macro_f1_mean"], 4) if cv else None,
            "cv_macro_f1_std": round(cv["cv_macro_f1_std"], 4) if cv else None,
            "cv_top1_mean": round(cv["cv_top1_mean"], 4) if cv else None,
            "test_top1_accuracy": round(r["test"]["top1_accuracy"], 4),
            "test_macro_f1": round(r["test"]["macro_f1"], 4),
            "test_log_loss": round(r["test"]["log_loss"], 4) if r["test"]["log_loss"] is not None else None,
            "test_candidate_miss_rate": round(r["test"]["candidate_miss_rate"], 4),
        })
    ablation = pd.DataFrame(rows)
    ablation_path = RESULTS_DIR / "model1_ablation.csv"
    ablation.to_csv(ablation_path, index=False)
    say("\n=== ablation ===")
    say(ablation.to_string(index=False))

    # --- ModelRecord for the report table -----------------------------------------
    final = results["all_19"]
    final.pop("model", None)
    results["geometry_only"].pop("model", None)

    model_obj = LogisticRegressionModel(C=1.0, max_iter=2000)
    record = ModelRecord(
        name="logreg",
        family="linear/probabilistic",
        config={**model_obj.describe(), "features": "all_19", "label_trust_m": 25.0,
                "cv": f"GroupKFold({N_SPLITS}) by trip, shared evaluation/cv.py"},
        seed=SEED,
        cv_macro_f1=final["cv"]["cv_macro_f1_mean"],
        test_top1_accuracy=final["test"]["top1_accuracy"],
        test_macro_f1=final["test"]["macro_f1"],
        test_log_loss=final["test"]["log_loss"],
        candidate_miss_rate=final["test"]["candidate_miss_rate"],
        route_continuity=None,
        train_seconds=final["train_seconds"],
        inference_ms_per_obs=final["inference_ms_per_obs"],
        artifact_path=None,
        notes=(
            "route_continuity N/A: requires the HMM-decoded route (P4). "
            "Metrics are observation-level on trusted labels. "
            f"CV std macro F1 = {final['cv']['cv_macro_f1_std']:.4f} over {N_SPLITS} folds. "
            f"convergence_warnings (final fit) = {final['convergence_warnings']}."
        ),
    )
    record_path = RESULTS_DIR / "model_logreg.json"
    record_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    say(f"\nwrote {record_path}")
    say(f"wrote {ablation_path}")

    # --- full run detail for the record --------------------------------------------
    detail_path = RESULTS_DIR / "model1_detail.json"
    detail_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    log_path = LOGS_DIR / f"model1_{started.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
