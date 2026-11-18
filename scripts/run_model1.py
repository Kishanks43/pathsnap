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
The CV loop here uses sklearn's GroupKFold directly. P2 owns
pathsnap/evaluation/cv.py; once that lands, replace `grouped_cv` below with its
cross_validate_model so every model is scored by one implementation.

METRIC DEFINITIONS (observation level, see docs/CONTRACTS.md C9)
-----------------------------------------------------------------
- Scores are evaluated on trusted observations only (label residual <= 25 m).
- Per observation: pick the candidate with the highest score (argmax).
- top1_accuracy : fraction of trusted observations whose argmax is the reference
                  arc. Observations whose reference arc is missing from the
                  candidate set count as wrong.
- macro_f1      : macro F1 over reference arc ids, computed on observations whose
                  reference arc IS in the candidate set (labels are undefined otherwise).
- log_loss      : observation-level negative log-likelihood of the reference
                  candidate, with scores renormalised to sum to 1 within each
                  observation. Computed on covered observations only.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.config import LOGS_DIR, RESULTS_DIR, ensure_dirs  # noqa: E402
from pathsnap.evaluation.metrics import macro_f1  # noqa: E402
from pathsnap.features.build import FEATURE_COLUMNS  # noqa: E402
from pathsnap.features.dataset import StackedFeatures, load_partition  # noqa: E402
from pathsnap.models.base import ModelRecord  # noqa: E402
from pathsnap.models.linear import LogisticRegressionModel  # noqa: E402

SEED = 42
N_SPLITS = 5
EPS = 1e-15
OBS_KEY_STRIDE = 10_000_000

GEOMETRY_COLS = list(range(0, 10))            # first 10 columns: candidate geometry + network context
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
        "n_obs": int(len(table)),
        "n_obs_covered": int(len(covered)),
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
    X, y, groups = d.X[mask], d.y[mask], d.group_index[mask]
    obs, arc = d.obs_idx[mask], d.arc_id[mask]

    folds = []
    splitter = GroupKFold(n_splits=N_SPLITS)
    for k, (tr, va) in enumerate(splitter.split(X, y, groups), 1):
        model, t_fit, n_conv = fit_model(X[tr], y[tr], cols)
        s = score_rows(model, X[va], cols)
        m = evaluate(observation_table(groups[va], obs[va], arc[va], y[va], s))
        m.update({"fold": k, "fit_seconds": t_fit, "convergence_warnings": n_conv,
                  "n_trips_valid": int(len(np.unique(groups[va])))})
        folds.append(m)
        say(f"    fold {k}: macro_f1={m['macro_f1']:.4f} top1={m['top1_accuracy']:.4f} "
            f"log_loss={m['log_loss']:.4f} trips_valid={m['n_trips_valid']} fit={t_fit:.1f}s")

    f1s = np.array([f["macro_f1"] for f in folds])
    return {
        "folds": folds,
        "cv_macro_f1_mean": float(f1s.mean()),
        "cv_macro_f1_std": float(f1s.std(ddof=1)),
        "cv_top1_mean": float(np.mean([f["top1_accuracy"] for f in folds])),
        "cv_log_loss_mean": float(np.mean([f["log_loss"] for f in folds])),
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
    X = d.X[mask]
    t0 = time.perf_counter()
    s = score_rows(model, X, cols)
    t_inf = time.perf_counter() - t0
    m = evaluate(observation_table(d.group_index[mask], d.obs_idx[mask], d.arc_id[mask], d.y[mask], s))
    n_obs = len(np.unique(d.group_index[mask].astype(np.int64) * OBS_KEY_STRIDE + d.obs_idx[mask]))
    return m, 1000.0 * t_inf / max(n_obs, 1)


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
                "cv": f"GroupKFold({N_SPLITS}) by trip, sklearn (temporary until P2 cv.py lands)"},
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
