"""Tests for the inference pipeline and the FastAPI service (Person 4).

Uses a stand-in model trained on random numbers so the API can be exercised without
the real artifacts. Tests that need the student-test trip files skip when they are
not present.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from pathsnap.api import main as api_main
from pathsnap.api.inference import ArtifactError, Predictor, find_trip_dir
from pathsnap.config import STUDENT_TEST_DIR
from pathsnap.features.build import FEATURE_COLUMNS
from pathsnap.hmm.viterbi import DecoderConfig


def _make_artifacts(root: Path, trip_ids: list[str]) -> Path:
    rng = np.random.RandomState(0)
    X = rng.randn(400, len(FEATURE_COLUMNS))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    clf.name = "stand_in"
    joblib.dump(clf, root / "model.joblib")
    (root / "feature_schema.json").write_text(
        json.dumps({"version": 1, "columns": list(FEATURE_COLUMNS)}), encoding="utf-8"
    )
    (root / "decoder_config.json").write_text(json.dumps(DecoderConfig().to_dict()), encoding="utf-8")
    (root / "eval_trips.txt").write_text("\n".join(trip_ids) + "\n", encoding="utf-8")
    return root


@pytest.fixture()
def artifacts(tmp_path: Path, monkeypatch) -> Path:
    real = sorted(p.name for p in STUDENT_TEST_DIR.iterdir() if p.is_dir()) if STUDENT_TEST_DIR.exists() else []
    root = tmp_path / "artifacts"
    root.mkdir()
    _make_artifacts(root, real or ["00000001"])
    monkeypatch.setattr(api_main, "ARTIFACTS_DIR", root)
    monkeypatch.setattr(api_main, "STUDENT_TEST_DIR", STUDENT_TEST_DIR)
    monkeypatch.setattr(api_main, "_request_log_path", tmp_path / "requests.jsonl")
    return root


class TestPredictor:

    def test_missing_artifact_fails_at_load(self, tmp_path: Path):
        with pytest.raises(ArtifactError, match="missing artifacts"):
            Predictor(tmp_path).load()

    def test_schema_mismatch_is_rejected(self, tmp_path: Path):
        root = _make_artifacts(tmp_path, ["00000001"])
        (root / "feature_schema.json").write_text(
            json.dumps({"version": 1, "columns": ["wrong"]}), encoding="utf-8"
        )
        with pytest.raises(ArtifactError, match="does not match"):
            Predictor(root).load()

    def test_trip_id_must_be_digits(self):
        assert find_trip_dir("../etc", [STUDENT_TEST_DIR]) is None


class TestApi:

    def test_health_reports_loaded_model(self, artifacts):
        with TestClient(api_main.app) as client:
            body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["artifacts_loaded"] is True
        assert body["model"] == "stand_in"
        assert set(body["artifact_sha256"]) == {"model.joblib", "feature_schema.json", "decoder_config.json", "eval_trips.txt"}

    def test_trips_come_from_manifest_file(self, artifacts):
        with TestClient(api_main.app) as client:
            trips = client.get("/trips").json()["trips"]
        assert len(trips) >= 1

    def test_unknown_trip_is_404(self, artifacts):
        with TestClient(api_main.app) as client:
            response = client.post("/predict/99999999")
        assert response.status_code == 404

    @pytest.mark.skipif(not STUDENT_TEST_DIR.exists(), reason="student_test data not present")
    def test_predict_returns_one_road_per_observation_and_no_reference(self, artifacts):
        trip_id = sorted(p.name for p in STUDENT_TEST_DIR.iterdir() if p.is_dir())[0]
        trip_dir = STUDENT_TEST_DIR / trip_id
        n_obs = sum(1 for _ in open(trip_dir / f"{trip_id}.track", encoding="utf-8"))
        with TestClient(api_main.app) as client:
            response = client.post(f"/predict/{trip_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["n_observations"] == n_obs
        assert len(body["arc_per_obs"]) == n_obs
        assert body["decode_mode"] == "hmm"
        assert len(body["route_geometry"]) > 0
        # Ground truth must never leave the server: metrics are aggregate numbers only.
        allowed = {
            "reference_available", "top1_accuracy", "classifier_only_top1_accuracy",
            "route_continuity", "road_set_overlap", "trusted_observations_scored",
        }
        assert set(body["metrics"]) <= allowed
        assert "route" not in body
