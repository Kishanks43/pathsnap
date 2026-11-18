"""Inference pipeline: raw trip -> predicted route, with no ground-truth leakage.

OWNER: Person 4
STATUS: implemented

This module is the only place the API gets predictions from. It loads artifacts once
at startup, then answers per-trip requests. It has no FastAPI imports, so it can be
unit-tested and reused by the UI.

LEAKAGE CONTRACT (non-negotiable)
---------------------------------
`predict_trip` reads .track, .nodes and .arcs only. It never reads the .route file.
Scoring against the reference happens in `score_against_reference`, which is called
by the API layer after the prediction already exists.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pathsnap.candidates.generator import CandidateConfig, generate_candidates
from pathsnap.features.build import FEATURE_COLUMNS, build_feature_matrix
from pathsnap.hmm.viterbi import DecoderConfig, classifier_only, viterbi_decode
from pathsnap.io.loaders import load_network, load_track
from pathsnap.io.schema import Trip
from pathsnap.viz.folium_maps import arc_sequence_to_coords

REQUIRED_ARTIFACTS = (
    "model.joblib",
    "feature_schema.json",
    "decoder_config.json",
    "eval_trips.txt",
)

#: Must match the settings the training matrices were built with
#: (scripts/prepare_dataset.py defaults: radius 100 m, at most 10 candidates).
CANDIDATE_CONFIG = CandidateConfig(radius_m=100.0, max_candidates=10)


class ArtifactError(RuntimeError):
    """Raised at startup when a required artifact is missing or inconsistent."""


@dataclass
class Prediction:
    trip_id: str
    arc_per_obs: np.ndarray            # int64, shape (n_obs,), -1 when no candidate
    classifier_arc_per_obs: np.ndarray
    route_geometry: list[list[float]]  # [[lon, lat], ...] for drawing
    n_observations: int
    decode_mode: str                   # "hmm"
    model_name: str
    diagnostics: dict = field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_trip_dir(trip_id: str, roots: list[Path]) -> Path | None:
    """Locate a trip folder under the configured roots. Trip ids are digits only."""
    if not trip_id.isdigit():
        return None
    for root in roots:
        candidate = root / trip_id
        if (candidate / f"{trip_id}.track").exists():
            return candidate
    return None


class Predictor:
    """Holds the loaded model + decoder config for the lifetime of the server."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.model_name = "unknown"
        self.decoder_cfg: DecoderConfig | None = None
        self.eval_trips: list[str] = []
        self.manifest: dict = {}
        self.artifact_hashes: dict[str, str] = {}

    def load(self) -> Predictor:
        """Load every required artifact, or fail with a message naming what is missing."""
        import joblib

        missing = [name for name in REQUIRED_ARTIFACTS if not (self.artifacts_dir / name).exists()]
        if missing:
            raise ArtifactError(
                f"missing artifacts in {self.artifacts_dir}: {missing}. "
                "Run scripts/run_p3.py and scripts/run_p4.py first."
            )

        schema = json.loads((self.artifacts_dir / "feature_schema.json").read_text(encoding="utf-8"))
        if list(schema["columns"]) != list(FEATURE_COLUMNS):
            raise ArtifactError("feature_schema.json does not match the code's FEATURE_COLUMNS")

        self.model = joblib.load(self.artifacts_dir / "model.joblib")
        if not hasattr(self.model, "predict_proba"):
            raise ArtifactError("model.joblib has no predict_proba")
        self.model_name = getattr(self.model, "name", type(self.model).__name__)

        cfg_raw = json.loads((self.artifacts_dir / "decoder_config.json").read_text(encoding="utf-8"))
        self.decoder_cfg = DecoderConfig(**cfg_raw)

        self.eval_trips = [
            line.strip()
            for line in (self.artifacts_dir / "eval_trips.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        manifest_path = self.artifacts_dir / "manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.artifact_hashes = {name: _sha256(self.artifacts_dir / name) for name in REQUIRED_ARTIFACTS}
        return self

    def predict_trip(self, trip_dir: Path) -> Prediction:
        """Candidates -> features -> model -> Viterbi, for one trip folder.

        Reads .track, .nodes and .arcs only. The .route file is never opened here.
        """
        if self.model is None or self.decoder_cfg is None:
            raise ArtifactError("Predictor.load() must be called before predict_trip()")
        trip_dir = Path(trip_dir)
        trip_id = trip_dir.name
        arc_path = trip_dir / f"{trip_id}.arcs"
        track = load_track(trip_dir / f"{trip_id}.track", trip_id)
        network = load_network(arc_path, trip_id)
        trip = Trip(trip_id=trip_id, track=track, network=network, route=None, path=trip_dir)

        t0 = time.perf_counter()
        candidates = generate_candidates(trip, config=CANDIDATE_CONFIG)
        X, _, meta = build_feature_matrix(trip, candidates, None)
        if list(meta["columns"]) != list(FEATURE_COLUMNS):
            raise ArtifactError("feature columns differ from the schema used for training")
        proba = self.model.predict_proba(X)
        t_model = time.perf_counter() - t0

        n_obs = len(track)
        decoded = viterbi_decode(trip_id, candidates, proba, network, n_obs, self.decoder_cfg, track=track)
        classifier_arc = classifier_only(candidates, proba, n_obs)
        geometry = arc_sequence_to_coords(decoded.arc_per_obs[decoded.arc_per_obs >= 0], network)

        return Prediction(
            trip_id=trip_id,
            arc_per_obs=decoded.arc_per_obs,
            classifier_arc_per_obs=classifier_arc,
            route_geometry=geometry,
            n_observations=n_obs,
            decode_mode="hmm",
            model_name=self.model_name,
            diagnostics={
                "n_candidate_rows": len(candidates.obs_idx),
                "n_obs_without_candidates": int(decoded.n_skipped_obs),
                "n_forced_emission_only_steps": int(decoded.n_forced_emission_only),
                "gap_resets": int(decoded.meta.get("gap_resets", 0)),
                "model_seconds": round(t_model, 4),
            },
        )

    def score_against_reference(self, pred: Prediction, trip_dir: Path) -> dict:
        """Aggregate metrics against the reference route, computed AFTER prediction.

        Kept separate on purpose: the prediction path has no access to the truth.
        Only aggregate numbers are returned, never reference arc ids.
        """
        from pathsnap.evaluation.metrics import route_continuity
        from pathsnap.io.loaders import load_route
        from pathsnap.labels import assign_reference_arcs

        trip_dir = Path(trip_dir)
        trip_id = trip_dir.name
        route = load_route(trip_dir / f"{trip_id}.route")
        if route is None or len(route) == 0:
            return {"top1_accuracy": None, "route_continuity": None, "reference_available": False}

        network = load_network(trip_dir / f"{trip_id}.arcs", trip_id)
        track = load_track(trip_dir / f"{trip_id}.track", trip_id)
        trip = Trip(trip_id=trip_id, track=track, network=network, route=route, path=trip_dir)
        ref = assign_reference_arcs(trip)
        resolved = ref.resolved & (ref.dist_per_obs <= 25.0)
        decoded = pred.arc_per_obs
        top1 = float((decoded[resolved] == ref.arc_per_obs[resolved]).mean()) if resolved.any() else None
        clf_top1 = (
            float((pred.classifier_arc_per_obs[resolved] == ref.arc_per_obs[resolved]).mean())
            if resolved.any()
            else None
        )
        used = decoded[decoded >= 0]
        route_set = set(route.tolist())
        pred_set = set(used.tolist())
        jaccard = len(route_set & pred_set) / len(route_set | pred_set) if route_set | pred_set else None
        return {
            "reference_available": True,
            "top1_accuracy": top1,
            "classifier_only_top1_accuracy": clf_top1,
            "route_continuity": route_continuity(used, network.arcs) if used.size >= 2 else None,
            "road_set_overlap": jaccard,
            "trusted_observations_scored": int(resolved.sum()),
        }


def file_checksums(artifacts_dir: Path) -> dict[str, str]:
    return {name: _sha256(Path(artifacts_dir) / name) for name in REQUIRED_ARTIFACTS if (Path(artifacts_dir) / name).exists()}
