"""Entry point for Model 4 (HistGradientBoosting).

This used to fall back to random synthetic data when the real features could not
be loaded, which produced meaningless accuracy numbers. It now just runs
scripts/run_model4.py, which uses the real cached features and stops with a clear
error if they are missing.

    python train_model.py                      # train on `train`, score `student_test`
    python train_model.py --student-test-only  # stand-in when data/raw/train is missing
"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "scripts" / "run_model4.py"

if __name__ == "__main__":
    if not SCRIPT.exists():
        sys.exit(f"missing {SCRIPT} - copy run_model4.py into the scripts folder first")
    sys.exit(subprocess.call([sys.executable, str(SCRIPT), *sys.argv[1:]]))