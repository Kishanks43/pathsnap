"""Model comparison: the single source of truth for the report's comparison table.

OWNER: Person 2 (must cover all five models, including Person 3's)
STATUS: STUB

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

from pathlib import Path

from pathsnap.models.base import ModelRecord


def save_record(record: ModelRecord, results_dir: Path) -> Path:
    """Write one model's record to experiments/results/model_<name>.json."""
    raise NotImplementedError("Person 2: implement record persistence.")


def load_records(results_dir: Path) -> list[ModelRecord]:
    raise NotImplementedError("Person 2: implement record loading.")


def build_comparison_table(results_dir: Path) -> str:
    """Read every record and return a markdown table in report order.

    Columns, in this order (matches Deliverables 3.1):
        Model | Family | Key configuration | CV macro F1 | Top-1 acc |
        Route continuity | Student-test macro F1 | Log loss | Inference ms/obs
    """
    raise NotImplementedError("Person 2: implement table generation.")


def assert_five_models(results_dir: Path) -> None:
    """Guard rail for submission: fail loudly if fewer than five models exist.

    Run this in CI or as the last step before you zip the repo. The brief says a
    minimum of five; a silently incomplete repo is an avoidable deduction.
    """
    raise NotImplementedError("Person 2: implement the five-model guard.")
