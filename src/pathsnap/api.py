"""FastAPI service for PathSnap prediction and server-side development scoring."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .adapters import make_inference_components
from .decoder.inference import IntegrationError, predict_trip
from .paths import ARTIFACT_DIR, DATA_ROOT
from .scoring import score_after_prediction


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("pathsnap.api")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        log_file = Path(os.getenv("PATHSNAP_LOG_FILE", "logs/api.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


logger = _configure_logging()
app = FastAPI(title="PathSnap API", version="0.1.0")
origins = [v.strip() for v in os.getenv("PATHSNAP_CORS_ORIGINS", "*").split(",") if v.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_TRIP_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_EVALUATOR_ROOT = Path(os.getenv("PATHSNAP_EVALUATOR_DIR", "")) if os.getenv("PATHSNAP_EVALUATOR_DIR") else None
_DEV_PARTITION = os.getenv("PATHSNAP_SERVE_PARTITION", "student_test")
_DATA_ROOT = Path(os.getenv("PATHSNAP_DATA", str(DATA_ROOT)))


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=64)


def _trip_root() -> tuple[Path, str]:
    if _EVALUATOR_ROOT is not None and _EVALUATOR_ROOT.is_dir():
        return _EVALUATOR_ROOT, "evaluator"
    if _DEV_PARTITION not in ("train", "student_test"):
        raise HTTPException(status_code=500, detail="invalid PATHSNAP_SERVE_PARTITION")
    return _DATA_ROOT / _DEV_PARTITION, _DEV_PARTITION


def _trip_dir(trip_id: str) -> Path:
    if not _TRIP_ID.fullmatch(trip_id):
        raise HTTPException(status_code=400, detail="invalid trip_id")
    root, _ = _trip_root()
    directory = root / trip_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="trip not found")
    for suffix in ("track", "nodes", "arcs"):
        if not (directory / f"{trip_id}.{suffix}").is_file():
            raise HTTPException(status_code=500, detail=f"trip is missing .{suffix} data")
    return directory


@app.get("/health")
def health():
    model_path = ARTIFACT_DIR / "model.joblib"
    try:
        from .clf import _load
        _load()
        loaded = True
        error = None
    except Exception as exc:  # report startup issue without leaking local paths
        loaded = False
        error = type(exc).__name__
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded,
            "model_artifact_present": model_path.is_file(), "version": app.version,
            "error_type": error}


@app.get("/trips")
def trips():
    root, partition = _trip_root()
    if not root.is_dir():
        return {"partition": partition, "trip_ids": []}
    trip_ids = sorted(
        folder.name for folder in root.iterdir()
        if folder.is_dir() and _TRIP_ID.fullmatch(folder.name)
    )
    return {"partition": partition, "trip_ids": trip_ids}


@app.post("/predict")
def predict(request: PredictRequest):
    directory = _trip_dir(request.trip_id)
    root, partition = _trip_root()
    logger.info(json.dumps({"event": "request_start", "trip_id": request.trip_id,
                            "partition": partition}, separators=(",", ":")))
    try:
        # Inference sees only track, nodes and arcs. Reference loading is below,
        # after predict_trip has completed successfully.
        components, score_state = make_inference_components(directory)
        prediction = predict_trip(directory, components)
        metrics = None
        route_file = directory / f"{request.trip_id}.route"
        if route_file.is_file():
            metrics = score_after_prediction(directory, prediction, score_state)
            # Avoid returning the potentially huge per-class arc-ID matrices to the UI.
            for key in ("classifier_observation_metrics", "hmm_observation_metrics"):
                if key in metrics:
                    metrics[key].pop("confusion_matrix", None)
        response = {
            "trip_id": request.trip_id,
            "partition": partition,
            "observations": prediction["observations"],
            "classifier_arc_ids": prediction["classifier_arc_ids"],
            "hmm_arc_ids": prediction["hmm_arc_ids"],
            "route_arc_ids": prediction["route_arc_ids"],
            "route_polyline": prediction["route_polyline"],
            "route_breaks": prediction["route_breaks"],
            "metrics": metrics,
            "stage_timings_s": prediction["stage_timings_s"],
            "inference_seconds": prediction["inference_seconds"],
            "warnings": prediction["warnings"],
        }
        summary = {
            "event": "request_complete", "trip_id": request.trip_id,
            "partition": partition, "inference_seconds": prediction["inference_seconds"],
            "observations": len(prediction["observations"]),
            "hmm_top1": metrics.get("hmm_top1_accuracy") if metrics else None,
            "route_breaks": len(prediction["route_breaks"]),
        }
        logger.info(json.dumps(summary, separators=(",", ":")))
        return response
    except (FileNotFoundError, ValueError, IntegrationError) as exc:
        logger.exception("prediction failed for trip_id=%s", request.trip_id)
        raise HTTPException(status_code=422, detail=f"prediction failed: {type(exc).__name__}") from exc
    except Exception as exc:
        logger.exception("internal prediction error for trip_id=%s", request.trip_id)
        raise HTTPException(status_code=500, detail=f"prediction failed: {type(exc).__name__}") from exc
