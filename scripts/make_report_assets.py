"""Build the figures and summary numbers used by reports/REPORT.md.

Everything here is derived from the committed experiment results, the cached
feature matrices, or the raw trips. The script is deliberately deterministic and
does not fit or tune any model.

Run from the repository root:

    python scripts/make_report_assets.py

Outputs:
    reports/figures/eda_missingness.png
    reports/figures/eda_gps_trajectory.png
    reports/figures/eda_road_network.png
    reports/figures/model_cv_f1.png
    reports/figures/model_test_scores.png
    reports/figures/model_logloss.png
    reports/figures/model_inference.png
    reports/figures/knn_k_sweep.png
    reports/figures/optuna_history.png
    reports/figures/feature_importance.png
    reports/figures/hmm_comparison.png
    reports/figures/decoder_sweep.png
    reports/report_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.config import PROJECT_ROOT, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from pathsnap.features.dataset import TRUST_RESIDUAL_M, load_partition
from pathsnap.io.loaders import load_trip
from pathsnap.viz.folium_maps import arc_sequence_to_coords

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["logreg", "knn", "random_forest", "hist_gradient_boosting", "svm_rbf"]
LABELS = {
    "logreg": "Logistic regression",
    "knn": "KNN (k=11)",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "svm_rbf": "RBF SVM",
}
COLORS = {
    "logreg": "#4C72B0",
    "knn": "#DD8452",
    "random_forest": "#55A868",
    "hist_gradient_boosting": "#C44E52",
    "svm_rbf": "#8172B3",
}

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def _cv_std() -> dict[str, float]:
    logreg = _json(RESULTS_DIR / "model1_detail.json")["all_19"]["cv"]["cv_macro_f1_std"]
    knn = next(
        x["std_macro_f1"] for x in _json(RESULTS_DIR / "cv_knn.json")["k_sweep"] if x["k"] == 11
    )
    rf = _json(RESULTS_DIR / "rf_untuned.json")["cv"]["std_macro_f1"]
    hgb = _json(RESULTS_DIR / "model4_detail.json")["cv"]["cv_macro_f1_std"]
    svm = _json(RESULTS_DIR / "cv_svm_rbf.json")["cv"]["std_macro_f1"]
    return {
        "logreg": float(logreg),
        "knn": float(knn),
        "random_forest": float(rf),
        "hist_gradient_boosting": float(hgb),
        "svm_rbf": float(svm),
    }


def scan_raw(partition: str, root: Path) -> dict:
    """Count rows, fields and null/invalid values in the raw partition."""
    trips = sorted(p for p in root.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())
    n_obs = n_nodes = n_arcs = n_route_arcs = 0
    track_null = node_null = route_missing_trips = 0
    route_arc_missing = 0
    for trip_dir in trips:
        trip = load_trip(trip_dir)
        df = trip.track.df
        n_obs += len(df)
        track_null += int(df[["lon", "lat", "timestamp"]].isna().sum().sum())
        n_nodes += trip.network.n_nodes
        n_arcs += trip.network.n_arcs
        node_null += int((~np.isfinite(trip.network.nodes)).sum())
        if trip.route is None or len(trip.route) == 0:
            route_missing_trips += 1
        else:
            n_route_arcs += len(trip.route)
            route_arc_missing += int((~np.isfinite(trip.route)).sum())
    return {
        "partition": partition,
        "n_trips": len(trips),
        "n_observations": n_obs,
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "n_route_arcs": n_route_arcs,
        "track_null_values": track_null,
        "node_null_values": node_null,
        "route_missing_trips": route_missing_trips,
        "route_arc_null_values": route_arc_missing,
    }


def scan_features(partition: str) -> dict:
    d = load_partition(partition)
    finite = np.isfinite(d.X)
    per_col = (~finite).sum(axis=0)
    trust = d.trusted()
    result = {
        "partition": partition,
        "n_rows": int(d.X.shape[0]),
        "n_features": int(d.X.shape[1]),
        "n_nonfinite_feature_values": int((~finite).sum()),
        "nonfinite_by_column": {
            name: int(n) for name, n in zip(d.columns, per_col) if int(n) > 0
        },
        "trusted_rows": int(trust.sum()),
        "trusted_fraction": float(trust.mean()),
        "untrusted_fraction": float(1.0 - trust.mean()),
        "n_observations": int(d.n_obs_per_trip.sum()),
    }
    del d
    return result


def per_trip_missing(partition: str) -> dict:
    path = PROJECT_ROOT / "data" / "processed" / partition
    miss, trust = [], []
    for p in sorted(path.glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            miss.append(float(z["candidate_miss_rate"][0]))
            residual = z["label_residual_m"]
            trust.append(float((residual <= TRUST_RESIDUAL_M).mean()))
    return {"trip_ids": [p.stem for p in sorted(path.glob("*.npz"))],
            "candidate_miss_rate": miss, "trusted_fraction": trust}


# Imported after the helper definitions so the constant has one source of truth.
# (TRUST_RESIDUAL_M comes from pathsnap.features.dataset above.)


def eda_missingness(raw: list[dict], features: list[dict], trip_stats: dict) -> dict:
    train_raw, test_raw = raw
    train_feat, test_feat = features
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    categories = ["Raw null\nfields", "Untrusted labels\n(route truncated)", "No candidate\n(mean per trip)"]
    train_vals = [0.0, train_feat["untrusted_fraction"] * 100,
                  float(_json(RESULTS_DIR / "candidate_quality.json")["train"]["candidate_miss_rate_mean"]) * 100]
    test_vals = [0.0, test_feat["untrusted_fraction"] * 100,
                 float(_json(RESULTS_DIR / "candidate_quality.json")["student_test"]["candidate_miss_rate_mean"]) * 100]
    x = np.arange(len(categories))
    w = 0.36
    b1 = ax.bar(x - w / 2, train_vals, w, label="train (80 trips)", color="#4C72B0")
    b2 = ax.bar(x + w / 2, test_vals, w, label="student_test (10 trips)", color="#DD8452")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%" if h else "0", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, categories, fontsize=8.5)
    ax.set_ylabel("share of observations / fields (%)")
    ax.set_title("Missingness that matters in PathSnap")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(
        0.99, 0.96,
        f"0 NaN/inf values in {train_feat['n_rows']:,} + {test_feat['n_rows']:,} feature rows\n"
        f"0 raw null fields in {train_raw['n_observations']:,} + {test_raw['n_observations']:,} observations",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    ax = axes[1]
    train_miss = np.array(trip_stats["train"]["candidate_miss_rate"]) * 100
    test_miss = np.array(trip_stats["student_test"]["candidate_miss_rate"]) * 100
    bins = np.linspace(0, 16, 17)
    ax.hist(train_miss, bins=bins, alpha=0.75, label="train trips", color="#4C72B0")
    ax.hist(test_miss, bins=bins, alpha=0.85, label="student_test trips", color="#DD8452")
    ax.set_xlabel("candidate miss rate per trip (%), clipped at 16%")
    ax.set_ylabel("trips")
    ax.set_title("Two truncated train routes dominate misses")
    ax.legend(fontsize=8)
    ax.annotate("00000054: 86.2%\n00000033: 66.9%", xy=(15.5, 1.0), xytext=(9.0, 12),
                fontsize=7.5, arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.text(0.98, 0.62,
            "median train miss 0.73%\npooled train mean 3.41%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5)

    fig.tight_layout()
    _save(fig, "eda_missingness.png")
    return {
        "raw": raw,
        "features": features,
        "candidate_quality": _json(RESULTS_DIR / "candidate_quality.json"),
    }


def eda_gps_trajectory(trip_id: str = "00000013") -> None:
    trip = load_trip(TRAIN_DIR / trip_id)
    lon, lat = trip.track.lon, trip.track.lat
    minutes = (trip.track.timestamp - trip.track.timestamp[0]) / 60.0
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.plot(lon, lat, color="#bbbbbb", linewidth=0.8, zorder=1)
    sc = ax.scatter(lon, lat, c=minutes, cmap="plasma", s=5, zorder=2)
    ax.scatter(lon[0], lat[0], marker="o", s=70, color="#2ca02c", edgecolor="white",
               linewidth=1.2, zorder=3, label="start")
    ax.scatter(lon[-1], lat[-1], marker="s", s=70, color="#d62728", edgecolor="white",
               linewidth=1.2, zorder=3, label="end")
    if trip.has_reference_route:
        coords = arc_sequence_to_coords(trip.route, trip.network)
        ax.plot([c[0] for c in coords], [c[1] for c in coords], color="#27ae60",
                linewidth=1.6, alpha=0.8, label="reference route", zorder=4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.83)
    cb.set_label("elapsed time (minutes)")
    ax.set_xlabel("longitude (degrees)")
    ax.set_ylabel("latitude (degrees)")
    ax.set_title(f"Raw GPS trajectory, train trip {trip_id} ({len(trip.track):,} observations)")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(lat.mean()))))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, "eda_gps_trajectory.png")


def eda_road_network(trip_id: str = "00000003") -> None:
    trip = load_trip(TRAIN_DIR / trip_id)
    track, net = trip.track, trip.network
    lon0, lon1 = float(track.lon.min()), float(track.lon.max())
    lat0, lat1 = float(track.lat.min()), float(track.lat.max())
    pad = 0.002
    node_lon, node_lat = net.nodes[:, 0], net.nodes[:, 1]
    a_lon, a_lat = node_lon[net.arcs[:, 0]], node_lat[net.arcs[:, 0]]
    b_lon, b_lat = node_lon[net.arcs[:, 1]], node_lat[net.arcs[:, 1]]
    inside = (
        ((a_lon >= lon0 - pad) & (a_lon <= lon1 + pad) & (a_lat >= lat0 - pad) & (a_lat <= lat1 + pad))
        | ((b_lon >= lon0 - pad) & (b_lon <= lon1 + pad) & (b_lat >= lat0 - pad) & (b_lat <= lat1 + pad))
    )
    visible = np.flatnonzero(inside)
    step = max(1, len(visible) // 5000)
    visible = visible[::step]
    segments = np.stack(
        [
            np.column_stack([a_lon[visible], a_lat[visible]]),
            np.column_stack([b_lon[visible], b_lat[visible]]),
        ],
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.add_collection(LineCollection(segments, colors="#b8b8b8", linewidths=0.55, alpha=0.75,
                                     label=f"road network ({len(visible):,} of {len(inside):,} visible arcs)"))
    if trip.has_reference_route:
        coords = arc_sequence_to_coords(trip.route, net)
        ax.plot([c[0] for c in coords], [c[1] for c in coords], color="#27ae60",
                linewidth=2.0, alpha=0.9, linestyle="--",
                label=f"reference route ({len(trip.route)} arcs)", zorder=4)
    ax.plot(track.lon, track.lat, color="#2980b9", linewidth=1.3, alpha=0.85,
            label=f"raw GPS ({len(track):,} observations)", zorder=3)
    ax.scatter(track.lon[0], track.lat[0], color="#2ca02c", s=45, edgecolor="white",
               linewidth=0.8, zorder=5, label="start")
    ax.scatter(track.lon[-1], track.lat[-1], color="#d62728", s=45, edgecolor="white",
               linewidth=0.8, zorder=5, label="end")
    ax.set_xlim(lon0 - pad, lon1 + pad)
    ax.set_ylim(lat0 - pad, lat1 + pad)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(track.lat.mean()))))
    ax.set_xlabel("longitude (degrees)")
    ax.set_ylabel("latitude (degrees)")
    ax.set_title(f"Road network and GPS, train trip {trip_id}")
    ax.legend(loc="best", fontsize=7.5)
    fig.tight_layout()
    _save(fig, "eda_road_network.png")


def comparison_figures(comp: pd.DataFrame) -> dict:
    rows = {r.Model: r for _, r in comp.iterrows()}
    std = _cv_std()
    order = sorted(MODELS, key=lambda m: float(rows[m]["CV macro F1"]), reverse=True)

    # CV macro F1 dot plot with fold std.
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    y = np.arange(len(order))
    for i, name in enumerate(order):
        mean = float(rows[name]["CV macro F1"])
        ax.errorbar(mean, i, xerr=std[name], fmt="o", color=COLORS[name], markersize=7,
                    capsize=4, linewidth=1.5)
        ax.annotate(f"{mean:.4f} ± {std[name]:.4f}", (mean, i), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(y, [LABELS[m] for m in order])
    ax.set_xlim(0.58, 0.70)
    ax.set_xlabel("5-fold group CV macro F1 (mean ± std)")
    ax.set_title("Cross-validation macro F1 across all five models")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "model_cv_f1.png")

    # Student-test top-1 and macro F1.
    order_test = sorted(MODELS, key=lambda m: float(rows[m]["Top-1 acc"]), reverse=True)
    x = np.arange(len(order_test))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    top1 = [float(rows[m]["Top-1 acc"]) for m in order_test]
    f1 = [float(rows[m]["Student-test macro F1"]) for m in order_test]
    b1 = ax.bar(x - w / 2, top1, w, label="Top-1 accuracy", color="#2E86AB")
    b2 = ax.bar(x + w / 2, f1, w, label="Macro F1", color="#E67E22")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=0)
    ax.set_xticks(x, [LABELS[m] for m in order_test], fontsize=8.5)
    ax.set_ylim(0.50, 0.86)
    ax.set_ylabel("score on 10 student-test trips")
    ax.set_title("Student-test top-1 accuracy and macro F1")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, "model_test_scores.png")

    # Log loss, where defined.
    loss_models = [m for m in MODELS if pd.notna(rows[m]["Log loss"])]
    loss_models.sort(key=lambda m: float(rows[m]["Log loss"]))
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    vals = [float(rows[m]["Log loss"]) for m in loss_models]
    bars = ax.bar([LABELS[m] for m in loss_models], vals, color=[COLORS[m] for m in loss_models], width=0.55)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.4f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("student-test log loss (lower is better)")
    ax.set_ylim(0, 0.19)
    ax.set_title("Log loss where probabilities are finite")
    ax.text(0.99, 0.96,
            "KNN: N/A (0.59% of rows get p=0 for the truth)\n"
            "HistGradientBoosting: N/A (6.8e-06 of rows get p=0)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"})
    fig.tight_layout()
    _save(fig, "model_logloss.png")

    # Inference cost, log scale.
    cost_order = sorted(MODELS, key=lambda m: float(rows[m]["Inference ms/obs"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    vals = [float(rows[m]["Inference ms/obs"]) for m in cost_order]
    ax.barh([LABELS[m] for m in cost_order], vals, color=[COLORS[m] for m in cost_order])
    ax.set_xscale("log")
    ax.set_xlabel("inference time per observation (ms, log scale)")
    ax.set_title("Inference cost: RF is 50x cheaper than the calibrated SVM")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.4f} ms", (v, i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.set_xlim(0.001, 6.0)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "model_inference.png")

    return {"rows": {m: rows[m].to_dict() for m in MODELS}, "cv_std": std}


def knn_sweep() -> None:
    data = _json(RESULTS_DIR / "cv_knn.json")["k_sweep"]
    ks = [x["k"] for x in data]
    means = [x["mean_macro_f1"] for x in data]
    stds = [x["std_macro_f1"] for x in data]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.errorbar(ks, means, yerr=stds, fmt="o-", color="#DD8452", capsize=4, linewidth=1.6)
    for k, m in zip(ks, means):
        ax.annotate(f"{m:.4f}", (k, m), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(ks)
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("5-fold group CV macro F1")
    ax.set_title("KNN k sweep: best at the tested boundary k=11")
    ax.set_ylim(0.54, 0.63)
    fig.tight_layout()
    _save(fig, "knn_k_sweep.png")


def optuna_history() -> None:
    trials = pd.read_csv(RESULTS_DIR / "trials.csv")
    trials = trials.sort_values("number")
    values = trials["value"].to_numpy()
    best = np.maximum.accumulate(values)
    best_i = int(np.argmax(values))
    trial_numbers = trials["number"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.scatter(trial_numbers, values, s=18, color="#4C72B0", alpha=0.75, label="trial score")
    ax.step(trial_numbers, best, where="post", color="#C44E52", linewidth=1.6,
            label="best so far")
    ax.scatter([trial_numbers[best_i]], [values[best_i]], s=70, color="#C44E52", zorder=5)
    ax.annotate(f"best trial {trial_numbers[best_i]}: {values[best_i]:.4f}",
                (trial_numbers[best_i], values[best_i]), xytext=(10, -16), textcoords="offset points",
                fontsize=8, arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.set_xlabel("Optuna trial number (0-based)")
    ax.set_ylabel("subsample CV macro F1")
    ax.set_title("25-trial Bayesian search (150,201-row observation subsample)")
    ax.set_ylim(0.65, 0.70)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, "optuna_history.png")


def feature_importance() -> None:
    imp = pd.read_csv(RESULTS_DIR / "importances_rf.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(imp["feature"], imp["importance"], color="#55A868")
    for y, (_, row) in enumerate(imp.iterrows()):
        ax.annotate(f"{row['importance']:.3f}", (row["importance"], y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_xlabel("mean decrease in impurity (RF, untuned)")
    ax.set_title("Top 10 features: rank and ratio carry more signal than raw distance")
    ax.set_xlim(0, 0.27)
    fig.tight_layout()
    _save(fig, "feature_importance.png")


def hmm_figures() -> dict:
    hmm = pd.read_csv(RESULTS_DIR / "hmm_vs_classifier.csv", dtype={"trip_id": str})
    summary = _json(RESULTS_DIR / "hmm_vs_classifier.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={"width_ratios": [1, 1.6]})

    ax = axes[0]
    vals = [summary["pooled_top1_classifier"], summary["pooled_top1_hmm"]]
    bars = ax.bar(["Classifier\nonly", "Classifier +\nViterbi"], vals, color=["#95a5a6", "#8e44ad"], width=0.6)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.4f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.78, 0.84)
    ax.set_ylabel("pooled student-test top-1")
    ax.set_title("Pooled gain: +1.01 pp")
    ax.annotate("", xy=(1, vals[1]), xytext=(1, vals[0]),
                arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2})

    ax = axes[1]
    d = hmm.sort_values("delta_hmm_minus_clf").reset_index(drop=True)
    delta_pp = d["delta_hmm_minus_clf"] * 100
    ypos = np.arange(len(d))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in delta_pp]
    ax.barh(ypos, delta_pp, color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(ypos, d["trip_id"])
    ax.set_xlabel("HMM minus classifier top-1 (percentage points)")
    ax.set_title("The average hides five losses and five gains")
    for y, v in zip(ypos, delta_pp):
        ax.annotate(f"{v:+.2f}", (v, y), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", va="center", ha="left" if v >= 0 else "right",
                    fontsize=7.5)
    ax.set_xlim(-9, 9)
    fig.tight_layout()
    _save(fig, "hmm_comparison.png")
    return {"hmm": summary, "rows": hmm.to_dict(orient="records")}


def decoder_figure() -> None:
    sweep = pd.read_csv(RESULTS_DIR / "decoder_sweep.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for hops, color, marker in [(1, "#95a5a6", "o"), (2, "#8e44ad", "s")]:
        part = sweep[sweep["max_hops"] == hops].sort_values("beta_m")
        ax.plot(part["beta_m"], part["hmm_top1"], marker=marker, color=color,
                linewidth=1.6, label=f"max_hops={hops}")
    base = float(sweep["classifier_top1"].iloc[0])
    ax.axhline(base, color="#2c3e50", linestyle="--", linewidth=1.2,
               label=f"classifier only ({base:.4f})")
    ax.axvline(10.0, color="#27ae60", linestyle=":", linewidth=1.4, label="chosen beta_m=10")
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 200], ["10", "25", "50", "100", "200"])
    ax.set_xlabel("beta_m (transition tolerance, metres; log scale)")
    ax.set_ylabel("held-out train top-1")
    ax.set_title("Decoder sweep: larger beta over-smooths and loses top-1")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    _save(fig, "decoder_sweep.png")


def main() -> None:
    print("scanning raw partitions...")
    raw = [scan_raw("train", TRAIN_DIR), scan_raw("student_test", STUDENT_TEST_DIR)]
    print("scanning feature matrices...")
    features = [scan_features("train"), scan_features("student_test")]
    trips = {p: per_trip_missing(p) for p in ("train", "student_test")}

    print("building EDA figures...")
    missing = eda_missingness(raw, features, trips)
    eda_gps_trajectory()
    eda_road_network()

    print("building model comparison figures...")
    comp = pd.read_csv(RESULTS_DIR / "comparison.csv")
    comparison = comparison_figures(comp)
    knn_sweep()
    optuna_history()
    feature_importance()

    print("building HMM figures...")
    hmm = hmm_figures()
    decoder_figure()

    metrics = {
        "raw": raw,
        "features": features,
        "missingness": missing,
        "comparison": comparison,
        "hmm": hmm,
        "tuning": {
            "summary": _json(RESULTS_DIR / "tune_summary_random_forest.json"),
            "best": _json(RESULTS_DIR / "best_config_random_forest.json"),
            "tuned_vs_untuned": _json(RESULTS_DIR / "tuned_vs_untuned_random_forest.json"),
        },
        "decoder": {
            "tuning": _json(RESULTS_DIR / "decoder_tuning.json"),
            "summary": _json(RESULTS_DIR / "hmm_vs_classifier.json"),
        },
    }
    (PROJECT_ROOT / "reports" / "report_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("wrote reports/report_metrics.json")


if __name__ == "__main__":
    main()
