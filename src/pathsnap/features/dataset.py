"""Trip-level feature preparation, caching and stacking.

OWNER: Person 1
STATUS: implemented.

WHY THIS EXISTS
---------------
Candidate generation and feature extraction take time and produce ~1.45M rows
across the 80 training trips. Recomputing that for every model run and every
Optuna trial is wasted work, so it is done once and cached per trip:

    data/processed/<partition>/<trip_id>.npz

Persons 2-4 then load the cached arrays. This module is the only sanctioned way
to get from raw files to a training matrix, so all five models are guaranteed to
see identical inputs - a requirement for a legitimate comparison table.

THE LABEL-TRUST FILTER
----------------------
`label_residual_m` is the distance from the observation to the reference arc it
was assigned. Some trips have reference routes that cover only part of the
trajectory (see docs/DATA_FACTS.md, Fact 6), which pushes residuals to >100 m and
makes those labels simply wrong.

`trusted()` drops observations whose residual exceeds `TRUST_RESIDUAL_M`. The
threshold was chosen from the data before any model was trained: 87% of training
observations fall under 25 m, and candidate miss rate drops from 3.4% to 1.2%
when the untrusted ones are removed. It is a label-quality filter, not a way to
delete hard examples; the criterion is documented and applies unchanged to
student_test.

Observations are never deleted from the *inference* path - only from supervised
fitting and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pathsnap.candidates.generator import CandidateConfig, generate_candidates
from pathsnap.config import PROCESSED_DIR
from pathsnap.features.build import FEATURE_COLUMNS, SCHEMA_VERSION, build_feature_matrix
from pathsnap.io.schema import Trip
from pathsnap.labels import ReferenceArcs, assign_reference_arcs

#: Observations further than this from their reference arc are not trusted labels.
TRUST_RESIDUAL_M = 25.0


@dataclass
class TripFeatures:
    trip_id: str
    X: np.ndarray
    y: np.ndarray | None
    obs_idx: np.ndarray
    arc_id: np.ndarray
    label_residual_m: np.ndarray
    n_obs: int
    columns: list[str]
    candidate_miss_rate: float = float("nan")

    def trusted(self, threshold: float = TRUST_RESIDUAL_M) -> np.ndarray:
        """Boolean mask over candidate rows whose observation label is trustworthy."""
        return self.label_residual_m <= threshold

    def summary(self) -> dict:
        n = len(self.obs_idx)
        trust = self.trusted()
        return {
            "trip_id": self.trip_id,
            "n_obs": int(self.n_obs),
            "n_pairs": int(n),
            "candidates_per_obs": float(n / self.n_obs) if self.n_obs else 0.0,
            "positives": int(self.y.sum()) if self.y is not None else None,
            "trusted_fraction": float(trust.mean()) if n else 0.0,
            "candidate_miss_rate": self.candidate_miss_rate,
        }


def prepare_trip(
    trip: Trip,
    config: CandidateConfig | None = None,
    reference: ReferenceArcs | None = None,
) -> TripFeatures:
    """Candidates -> features -> (optional) labels, for one trip."""
    from pathsnap.labels import candidate_miss_rate as _miss

    cfg = config or CandidateConfig()
    reference = reference if reference is not None else assign_reference_arcs(trip)
    candidates = generate_candidates(trip, config=cfg)

    X, y, meta = build_feature_matrix(trip, candidates, reference if trip.has_reference_route else None)

    miss = float("nan")
    if trip.has_reference_route:
        miss = _miss(reference, candidates, len(trip.track))

    return TripFeatures(
        trip_id=trip.trip_id,
        X=X,
        y=y,
        obs_idx=meta["obs_idx"],
        arc_id=meta["arc_id"],
        label_residual_m=meta["label_residual_m"],
        n_obs=meta["n_obs"],
        columns=meta["columns"],
        candidate_miss_rate=miss,
    )


def trip_path(partition: str, trip_id: str, processed_dir: Path | None = None) -> Path:
    root = Path(processed_dir) if processed_dir else PROCESSED_DIR
    return root / partition / f"{trip_id}.npz"


def save_trip_features(tf: TripFeatures, partition: str, processed_dir: Path | None = None) -> Path:
    path = trip_path(partition, tf.trip_id, processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=tf.X.astype(np.float32),
        y=np.full(0, -1, dtype=np.int64) if tf.y is None else tf.y.astype(np.int64),
        has_y=np.array([tf.y is not None]),
        obs_idx=tf.obs_idx.astype(np.int64),
        arc_id=tf.arc_id.astype(np.int64),
        label_residual_m=tf.label_residual_m.astype(np.float32),
        n_obs=np.array([tf.n_obs], dtype=np.int64),
        columns=np.array(tf.columns),
        schema_version=np.array([SCHEMA_VERSION], dtype=np.int64),
        candidate_miss_rate=np.array([tf.candidate_miss_rate], dtype=np.float64),
    )
    return path


def load_trip_features(trip_id: str, partition: str, processed_dir: Path | None = None) -> TripFeatures:
    path = trip_path(partition, trip_id, processed_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `python scripts/prepare_dataset.py --partition {partition}`"
        )
    with np.load(path, allow_pickle=False) as z:
        version = int(z["schema_version"][0])
        if version != SCHEMA_VERSION:
            raise RuntimeError(
                f"cached features for {trip_id} are schema v{version}, code expects "
                f"v{SCHEMA_VERSION} - re-run scripts/prepare_dataset.py --force"
            )
        has_y = bool(z["has_y"][0])
        return TripFeatures(
            trip_id=trip_id,
            X=z["X"].astype(np.float64),
            y=z["y"].astype(np.int64) if has_y else None,
            obs_idx=z["obs_idx"],
            arc_id=z["arc_id"],
            label_residual_m=z["label_residual_m"].astype(np.float64),
            n_obs=int(z["n_obs"][0]),
            columns=[str(c) for c in z["columns"]],
            candidate_miss_rate=float(z["candidate_miss_rate"][0]),
        )


@dataclass
class StackedFeatures:
    """All trips for a partition stacked into one matrix, plus grouping keys.

    `group_index` maps every row to a trip; pass it as `groups` to sklearn's
    GroupKFold so a trip never straddles a fold (CONTRACTS C1 / the brief's rule).
    """

    X: np.ndarray
    y: np.ndarray
    obs_idx: np.ndarray
    arc_id: np.ndarray
    label_residual_m: np.ndarray
    group_index: np.ndarray
    trip_ids: list[str]
    n_obs_per_trip: np.ndarray
    columns: list[str]

    def trusted(self, threshold: float = TRUST_RESIDUAL_M) -> np.ndarray:
        return self.label_residual_m <= threshold

    @property
    def groups(self) -> np.ndarray:
        """Group labels for GroupKFold: one label per trip."""
        return self.group_index

    def summary(self) -> dict:
        trust = self.trusted()
        return {
            "n_trips": len(self.trip_ids),
            "n_rows": int(len(self.X)),
            "n_features": int(self.X.shape[1]),
            "positive_rate": float(self.y.mean()) if len(self.y) else None,
            "trusted_fraction": float(trust.mean()) if len(trust) else None,
            "trusted_rows": int(trust.sum()),
        }


def load_partition(
    partition: str,
    processed_dir: Path | None = None,
    trips: list[str] | None = None,
) -> StackedFeatures:
    """Stack every cached trip in a partition into one matrix."""
    root = (Path(processed_dir) if processed_dir else PROCESSED_DIR) / partition
    trip_ids = trips if trips is not None else sorted(p.stem for p in root.glob("*.npz"))
    if not trip_ids:
        raise FileNotFoundError(f"no cached trips in {root} - run scripts/prepare_dataset.py")

    parts = [load_trip_features(t, partition, processed_dir) for t in trip_ids]
    columns = parts[0].columns
    for p in parts[1:]:
        if p.columns != columns:
            raise RuntimeError(f"column mismatch between cached trips: {p.trip_id}")

    return StackedFeatures(
        X=np.vstack([p.X for p in parts]),
        y=np.concatenate([p.y if p.y is not None else np.zeros(len(p.obs_idx), dtype=np.int64) for p in parts]),
        obs_idx=np.concatenate([p.obs_idx for p in parts]),
        arc_id=np.concatenate([p.arc_id for p in parts]),
        label_residual_m=np.concatenate([p.label_residual_m for p in parts]),
        group_index=np.repeat(np.arange(len(parts), dtype=np.int32), [len(p.obs_idx) for p in parts]),
        trip_ids=[p.trip_id for p in parts],
        n_obs_per_trip=np.array([p.n_obs for p in parts], dtype=np.int64),
        columns=columns,
    )


__all__ = [
    "FEATURE_COLUMNS",
    "TRUST_RESIDUAL_M",
    "TripFeatures",
    "StackedFeatures",
    "prepare_trip",
    "save_trip_features",
    "load_trip_features",
    "load_partition",
]
