"""FastAPI backend - runs on Laptop 1 and is called by the UI on Laptop 2.

OWNER: Person 4
STATUS: implemented

RUN (development)
-----------------
    uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000

RUN (evaluation day)
--------------------
    uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000 --log-level info

`--host 0.0.0.0` is mandatory: the evaluator's UI runs on a different laptop and
must reach this server over the LAN.

ENDPOINTS (docs/CONTRACTS.md C7)
--------------------------------
GET  /health              -> status, model name, artifacts_loaded, manifest hash
GET  /trips               -> trip ids listed in artifacts/eval_trips.txt
POST /predict/{trip_id}   -> predicted route plus aggregate metrics

HARD RULES FROM THE BRIEF
-------------------------
- Ground truth stays server-side: responses carry aggregate metrics only, never
  reference arc ids.
- The prediction path always runs the model for the requested trip.
- Pydantic response models keep the UI contract from drifting.
- Every request is logged (trip id, status, elapsed ms) to the console and to
  experiments/logs/api_requests.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pathsnap.api.inference import ArtifactError, Predictor, find_trip_dir
from pathsnap.config import ARTIFACTS_DIR, LOGS_DIR, STUDENT_TEST_DIR

logger = logging.getLogger("pathsnap.api")
_predictor: Predictor | None = None
_request_log_path = LOGS_DIR / "api_requests.jsonl"


def _eval_roots() -> list[Path]:
    """Folders searched for trip directories, in order."""
    roots = []
    eval_dir = os.environ.get("PATHSNAP_EVAL_DIR")
    if eval_dir:
        roots.append(Path(eval_dir))
    roots.append(STUDENT_TEST_DIR)
    return roots


def _log_request(record: dict) -> None:
    logger.info(
        "trip=%s status=%s elapsed_ms=%.1f endpoint=%s",
        record.get("trip_id", "-"),
        record.get("status"),
        record.get("elapsed_ms", 0.0),
        record.get("endpoint"),
    )
    _request_log_path.parent.mkdir(parents=True, exist_ok=True)
    with _request_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every artifact once. A missing artifact stops the server at startup."""
    global _predictor
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _predictor = Predictor(ARTIFACTS_DIR).load()
    logger.info("loaded model=%s artifacts=%s", _predictor.model_name, sorted(_predictor.artifact_hashes))
    yield
    _predictor = None


app = FastAPI(title="PathSnap API", version="1.0.0", lifespan=lifespan)


class PredictMetrics(BaseModel):
    reference_available: bool
    top1_accuracy: float | None = None
    classifier_only_top1_accuracy: float | None = None
    route_continuity: float | None = None
    road_set_overlap: float | None = None
    trusted_observations_scored: int | None = None


class PredictResponse(BaseModel):
    trip_id: str
    arc_per_obs: list[int]
    route_geometry: list[list[float]]
    n_observations: int
    decode_mode: str
    model: str
    diagnostics: dict[str, float | int | None]
    metrics: PredictMetrics


class HealthResponse(BaseModel):
    status: str
    model: str
    artifacts_loaded: bool
    artifact_sha256: dict[str, str]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _predictor is not None and _predictor.model is not None
    return HealthResponse(
        status="ok" if loaded else "unavailable",
        model=getattr(_predictor, "model_name", "unknown"),
        artifacts_loaded=loaded,
        artifact_sha256=getattr(_predictor, "artifact_hashes", {}),
    )


@app.get("/trips")
def list_trips() -> dict:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="predictor not initialised")
    return {"trips": _predictor.eval_trips}


@app.post("/predict/{trip_id}", response_model=PredictResponse)
def predict(trip_id: str) -> PredictResponse:
    """Run the full pipeline for one trip and return the route and aggregate metrics."""
    t0 = time.perf_counter()
    if _predictor is None:
        raise HTTPException(status_code=503, detail="predictor not initialised")

    trip_dir = find_trip_dir(trip_id, _eval_roots())
    if trip_dir is None:
        _log_request({"trip_id": trip_id, "status": 404, "endpoint": "predict",
                      "elapsed_ms": (time.perf_counter() - t0) * 1000})
        raise HTTPException(status_code=404, detail=f"trip {trip_id!r} not found")

    try:
        pred = _predictor.predict_trip(trip_dir)
        scores = _predictor.score_against_reference(pred, trip_dir)
    except ArtifactError as exc:
        _log_request({"trip_id": trip_id, "status": 500, "endpoint": "predict",
                      "elapsed_ms": (time.perf_counter() - t0) * 1000, "error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _log_request({
        "trip_id": trip_id,
        "status": 200,
        "endpoint": "predict",
        "elapsed_ms": round(elapsed_ms, 1),
        "n_observations": pred.n_observations,
        "reference_available": scores.get("reference_available", False),
    })
    return PredictResponse(
        trip_id=trip_id,
        arc_per_obs=pred.arc_per_obs.tolist(),
        route_geometry=pred.route_geometry,
        n_observations=pred.n_observations,
        decode_mode=pred.decode_mode,
        model=pred.model_name,
        diagnostics={**pred.diagnostics, "elapsed_ms": round(elapsed_ms, 1)},
        metrics=PredictMetrics(**scores),
    )


@app.exception_handler(Exception)
async def _unhandled(request, exc):  # pragma: no cover - last-resort safety net
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})
