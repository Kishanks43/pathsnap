"""Project-wide paths and constants.

Everything that points at a file on disk lives here so that no module
hardcodes a path. Dataset roots are configurable via environment variables
so the repo works on every teammate's machine.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Data -------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = Path(os.environ.get("PATHSNAP_TRAIN", RAW_DIR / "train"))
STUDENT_TEST_DIR = Path(os.environ.get("PATHSNAP_STUDENT_TEST", RAW_DIR / "student_test"))

# --- Outputs ----------------------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"

# --- Dataset facts (verify these, do not trust blindly) ----------------------
N_TRAIN_TRIPS = 80
N_STUDENT_TEST_TRIPS = 10
N_EVALUATOR_TEST_TRIPS = 10

# Row indices in the raw files are 0-based; arc_id == row index in .arcs
# and node_id == row index in .nodes.
FILE_SUFFIXES = ("track", "nodes", "arcs", "route")

# Reproducibility: every experiment must record its seed.
DEFAULT_SEED = 42


def ensure_dirs() -> None:
    """Create output directories (safe to call repeatedly)."""
    for d in (ARTIFACTS_DIR, RESULTS_DIR, LOGS_DIR, CONFIGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
