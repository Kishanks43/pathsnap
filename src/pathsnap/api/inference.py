"""Inference pipeline: raw trip -> predicted route, with no ground-truth leakage.

OWNER: Person 4
STATUS: STUB

This module is the only place the API is allowed to get predictions from. It
loads artifacts once at startup, then answers per-trip requests. Keep it free of
FastAPI imports so it can be unit-tested and reused by the UI smoke test.

LEAKAGE CONTRACT (non-negotiable)
---------------------------------
`predict_trip` receives a trip directory path. It may read .track, .nodes and
.arcs. It must NOT read the .route file: that is the answer. Scoring happens in
a separate function (`score_against_reference`) called only by the API layer for
display, and it takes the reference separately so the model path can never touch
it. Evaluators inspect repos for hardcoded or leaked predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Prediction:
    trip_id: str
    arc_per_obs: np.ndarray      # int64, shape (n_obs,)
    route_geometry: list[list[float]]  # [[lon, lat], ...] for drawing the line
    n_observations: int
    decode_mode: str             # "classifier_only" or "hmm"


class Predictor:
    """Holds the loaded model + decoder config for the lifetime of the server."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self._model = None
        self._scaler = None
        self._decoder_cfg = None
        self._feature_columns: list[str] = []

    def load(self) -> "Predictor":
        """Load model, scaler, feature schema and decoder config from disk.

        Raise a clear error at startup if any artifact is missing - a server that
        starts fine and then 500s during evaluation is worse than one that fails
        fast.
        """
        raise NotImplementedError("Person 4: implement artifact loading.")

    def predict_trip(self, trip_dir: Path) -> Prediction:
        """Run candidates -> features -> model -> Viterbi for one trip.

        Steps: load track+network (NOT route), generate candidates, build the
        feature matrix with exactly the training feature columns in the same
        order, predict_proba, decode, convert the arc sequence to lon/lat
        geometry for the UI.
        """
        raise NotImplementedError("Person 4: implement the inference pipeline.")

    def score_against_reference(self, pred: Prediction, trip_dir: Path) -> dict:
        """Compute display metrics after the prediction is already made.

        Kept separate on purpose so the prediction path has no access to truth.
        """
        raise NotImplementedError("Person 4: implement reference scoring.")
