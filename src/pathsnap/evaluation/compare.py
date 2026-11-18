"""Model comparison: the single source of truth for the report's comparison table.

OWNER: Person 2 (must cover all five models, including Person 3's)
STATUS: implemented

WHY A SEPARATE MODULE
---------------------
Deliverables 3.1 requires one consolidated table with, per model: algorithm
family, key configuration, CV macro F1, Top-1 accuracy, route continuity,
student-test macro F1, log loss, inference time. If each workstream writes its
own numbers in its own format, assembling that table at 3 a.m. is where
hackathon reports fall apart.

Every model run should end by appending one ModelRecord here, and the report
table should be generated from these files - never typed by hand. That also
satisfies "all results reported in the PDF must be verifiable using the
repository".

OUTPUT FILES
------------
experiments/results/model_<name>.json   one ModelRecord per model
experiments/results/comparison.csv      the finished 5-row table
experiments/results/comparison.md       markdown version to paste into the report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pathsnap.models.base import ModelRecord
from pathsnap.models.registry import ALL_MODELS

_REPORT_ORDER = {name: idx for idx, name in enumerate(ALL_MODELS)}
# Runtime measurements belong in their own column (inference ms/obs) or in the
# experiment log; they are not "key configuration" and vary between runs.
_CONFIG_SKIP_KEYS = {"name", "family", "fit_seconds"}


def _fmt(value: float | None) -> str:
    """Format a float to 4 decimal places, or 'N/A' for None."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _format_config(config: dict) -> str:
    if not config:
        return ""
    parts = [
        f"{k}={str(v).replace('|', '/').replace(chr(10), ' ')}"
        for k, v in config.items()
        if k not in _CONFIG_SKIP_KEYS and v is not None
    ]
    return ", ".join(parts)


def save_record(record: ModelRecord, results_dir: Path) -> Path:
    """Write one model's record to experiments/results/model_<name>.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"model_{record.name}.json"
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def load_records(results_dir: Path) -> list[ModelRecord]:
    """Load all model records from *results_dir*, sorted by name."""
    records: list[ModelRecord] = []
    for p in sorted(results_dir.glob("model_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        records.append(ModelRecord(**data))
    records.sort(key=lambda r: r.name)
    return records


def build_comparison_table(results_dir: Path) -> str:
    """Read every record and return a markdown table in report order.

    Columns, in this order (matches Deliverables 3.1):
        Model | Family | Key configuration | CV macro F1 | Top-1 acc |
        Route continuity | Student-test macro F1 | Log loss | Inference ms/obs
    """
    records = sorted(
        load_records(results_dir),
        key=lambda r: (_REPORT_ORDER.get(r.name, len(_REPORT_ORDER)), r.name),
    )

    columns = [
        "Model",
        "Family",
        "Key configuration",
        "CV macro F1",
        "Top-1 acc",
        "Route continuity",
        "Student-test macro F1",
        "Log loss",
        "Inference ms/obs",
    ]

    rows: list[dict[str, str]] = []
    for r in records:
        cfg = _format_config(r.config)
        rows.append(
            {
                "Model": r.name,
                "Family": r.family,
                "Key configuration": cfg,
                "CV macro F1": _fmt(r.cv_macro_f1),
                "Top-1 acc": _fmt(r.test_top1_accuracy),
                "Route continuity": _fmt(r.route_continuity),
                "Student-test macro F1": _fmt(r.test_macro_f1),
                "Log loss": _fmt(r.test_log_loss),
                "Inference ms/obs": _fmt(r.inference_ms_per_obs),
            }
        )

    # --- markdown table ---
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for row in rows:
        body_lines.append("| " + " | ".join(row[c] for c in columns) + " |")
    md = "\n".join([header, sep, *body_lines]) + "\n"

    # --- persist csv and md ---
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(results_dir / "comparison.csv", index=False)
    (results_dir / "comparison.md").write_text(md, encoding="utf-8")

    return md


def assert_five_models(results_dir: Path) -> None:
    """Guard rail for submission: fail loudly if fewer than five models exist.

    Run this in CI or as the last step before you zip the repo. The brief says a
    minimum of five; a silently incomplete repo is an avoidable deduction.
    """
    records = load_records(results_dir)
    if len(records) < 5:
        names = [r.name for r in records]
        raise AssertionError(
            f"Need at least 5 models but found {len(records)}: {names}"
        )
