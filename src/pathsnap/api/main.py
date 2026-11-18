"""FastAPI backend - runs on Laptop 1 and is called by the UI on Laptop 2.

OWNER: Person 4
STATUS: STUB

RUN (development)
-----------------
    uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000

RUN (evaluation day)
--------------------
    uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000 --log-level info

`--host 0.0.0.0` is mandatory: the evaluator's UI runs on a different laptop and
must reach this server over the LAN. Bind to 127.0.0.1 and the demo fails.

REQUIRED ENDPOINTS
------------------
GET  /health              -> {"status": "ok", "model": "<name>", "artifacts_loaded": true}
GET  /trips               -> the ten evaluator-trip IDs the UI can offer
POST /predict/{trip_id}   -> {"trip_id", "arc_per_obs", "route_geometry",
                              "n_observations", "decode_mode",
                              "metrics": {...}}   # metrics computed server-side

HARD RULES FROM THE BRIEF
--------------------------
- Ground truth must stay server-side: never return raw reference labels, only
  aggregate metrics.
- The API must actually do the work for whatever trip is requested; hardcoding
  outputs or bypassing the model is explicitly prohibited and evaluators read
  the API logs on this laptop.
- Pydantic response models so the UI contract cannot silently drift.
- Log every request with the trip id and elapsed milliseconds - you will be
  asked to show these logs during evaluation.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pathsnap.api.inference import Predictor
from pathsnap.config import ARTIFACTS_DIR

app = FastAPI(title="PathSnap API", version="0.1.0")

# Loaded once at startup; a missing artifact should crash the server loudly
# rather than surface as a 500 during the demo.
_predictor: Predictor | None = None


class PredictResponse(BaseModel):
    trip_id: str
    arc_per_obs: list[int]
    route_geometry: list[list[float]]
    n_observations: int
    decode_mode: str
    metrics: dict[str, float | str | None]


@app.on_event("startup")
def _startup() -> None:
    global _predictor
    _predictor = Predictor(ARTIFACTS_DIR)
    _predictor.load()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": getattr(_predictor, "model_name", "unknown"),
        "artifacts_loaded": _predictor is not None,
    }


@app.get("/trips")
def list_trips() -> dict:
    """Return the ten evaluator-trip ids.

    TODO(Person 4): read these from a manifest file (e.g.
    artifacts/eval_trips.txt) rather than globbing, so the list is auditable.
    """
    raise NotImplementedError("Person 4: implement /trips")


@app.post("/predict/{trip_id}", response_model=PredictResponse)
def predict(trip_id: str) -> PredictResponse:
    """Run the full pipeline for one trip and return route + server-side metrics."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="predictor not initialised")
    raise NotImplementedError("Person 4: implement /predict")
