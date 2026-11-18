"""Evaluator-facing API for PathSnap; reference labels remain server-side."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .clf import _load as load_model
from .decoder.config import DecoderConfig
from .decoder.inference import InferenceComponents, predict_trip
from .io import load_trip
from .paths import ARTIFACT_DIR, DATA_ROOT
from .pipeline import make_components
from .scoring import score_after_prediction

_TRIP_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_EVALUATOR_ENV = os.getenv("PATHSNAP_EVALUATOR_DIR")
_EVALUATOR_ROOT = Path(_EVALUATOR_ENV).expanduser() if _EVALUATOR_ENV else None
_DATA_ROOT = Path(os.getenv("PATHSNAP_DATA", str(DATA_ROOT))).expanduser()
_DEV_PARTITION = "student_test"
_MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.json"


def _model_name() -> str:
    try:
        card = json.loads(_MODEL_CARD_PATH.read_text(encoding="utf-8"))
        raw = str(card.get("model", "lightgbm")).lower()
        model = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}.get(raw, raw.replace("_", " ").title())
        return f"{model} (exported)"
    except (OSError, ValueError, TypeError):
        return "LightGBM (exported)"


def _trip_root() -> tuple[Path, str]:
    if _EVALUATOR_ENV:
        if _EVALUATOR_ROOT is None or not _EVALUATOR_ROOT.is_dir():
            raise RuntimeError("PATHSNAP_EVALUATOR_DIR is set but does not name a readable directory")
        return _EVALUATOR_ROOT, "evaluator"
    return _DATA_ROOT / _DEV_PARTITION, _DEV_PARTITION


def _trip_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and _TRIP_ID.fullmatch(p.name))


def _validate_evaluator_partition() -> None:
    if not _EVALUATOR_ENV:
        return
    root, _ = _trip_root()
    folders = _trip_directories(root)
    if len(folders) != 10:
        raise RuntimeError(f"Evaluator partition must contain exactly 10 trip folders; found {len(folders)}")
    for folder in folders:
        for suffix in ("track", "nodes", "arcs"):
            if not (folder / f"{folder.name}.{suffix}").is_file():
                raise RuntimeError(f"Evaluator trip {folder.name} is missing required .{suffix} input")


def _warm_up() -> None:
    load_model()
    root, _ = _trip_root()
    folders = _trip_directories(root)
    if folders:
        # Run one complete ground-truth-free prediction to warm both model and graph code.
        components, _ = _components_with_context(folders[0])
        predict_trip(folders[0], components, DecoderConfig())


logger = logging.getLogger("pathsnap.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_path = Path(os.getenv("PATHSNAP_LOG_FILE", "logs/api.log"))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

app = FastAPI(title="PathSnap API", version="1.0.0")
origins = [v.strip() for v in os.getenv("PATHSNAP_CORS_ORIGINS", "*").split(",") if v.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.on_event("startup")
def startup():
    _validate_evaluator_partition()
    _warm_up()


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=64)


def _resolve_trip(trip_id: str) -> tuple[Path, str]:
    if not _TRIP_ID.fullmatch(trip_id):
        raise HTTPException(status_code=400, detail="Invalid trip_id")
    root, partition = _trip_root()
    directory = root / trip_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")
    for suffix in ("track", "nodes", "arcs"):
        if not (directory / f"{trip_id}.{suffix}").is_file():
            raise HTTPException(status_code=500, detail=f"Trip '{trip_id}' is missing .{suffix} data")
    return directory, partition


def _components_with_context(trip_dir: Path):
    """Wrap the official A/B pipeline callbacks to retain private post-score inputs."""
    base = make_components(trip_dir)
    state: dict = {"candidate_sets": None, "probabilities": None, "graph": None}

    def generate(points, graph):
        sets = base.generate_candidates(points, graph)
        state["candidate_sets"] = [list(row) for row in sets]
        state["graph"] = graph
        return sets

    def build_features(observations, candidate_sets, graph):
        features = base.build_features(observations, candidate_sets, graph)
        state["features"] = features
        return features

    def score(features):
        probabilities = base.score_candidates(features)
        state["probabilities"] = np.asarray(probabilities, dtype=float)
        return probabilities

    components = InferenceComponents(
        generate_candidates=generate,
        build_features=build_features,
        score_candidates=score,
        projection_factory=base.projection_factory,
    )
    return components, state


def _to_latlon_segments(segments):
    return [[[float(lat), float(lon)] for lon, lat in segment] for segment in segments]


def _classifier_segments(arc_ids, graph):
    segments, current = [], []
    for arc_id in arc_ids:
        if arc_id is None:
            if current:
                segments.extend(graph.route_polyline(current))
                current = []
        else:
            current.append(int(arc_id))
    if current:
        segments.extend(graph.route_polyline(current))
    return _to_latlon_segments(segments)


def _route_metrics(metrics: dict, kind: str) -> dict:
    obs_key = f"{kind}_observation_metrics"
    continuity_key = f"{kind}_route_continuity"
    sequence = metrics.get("stitched_route_metrics" if kind == "hmm" else "classifier_route_metrics", {})
    observation = metrics.get(obs_key, {})
    continuity = metrics.get(continuity_key, {})
    return {
        "top1_accuracy": metrics.get(f"{kind}_top1_accuracy"),
        "macro_f1": observation.get("macro_f1"),
        "precision": observation.get("macro_precision"),
        "recall": observation.get("macro_recall"),
        "log_loss": metrics.get("candidate_log_loss") if kind == "classifier" else None,
        "route_continuity": continuity.get("route_continuity"),
        "continuity_breaks": continuity.get("route_break_count"),
        "sequence_f1": sequence.get("route_arc_set_f1"),
        "candidate_miss_rate": metrics.get("candidate_miss_rate"),
        "n_labelled": observation.get("labeled_observation_count"),
    }


def _candidate_confidences(arc_ids, candidate_sets, probabilities):
    confidence = []
    offset = 0
    for arc_id, candidates in zip(arc_ids, candidate_sets):
        row = np.maximum(0.0, probabilities[offset:offset + len(candidates)])
        offset += len(candidates)
        total = float(row.sum())
        normalized = row / total if total > 0 else (np.full(len(row), 1 / len(row)) if len(row) else row)
        selected = next((float(normalized[i]) for i, candidate in enumerate(candidates)
                         if candidate.arc_id == arc_id), None)
        confidence.append(selected)
    return confidence


@app.get("/health")
def health():
    try:
        _validate_evaluator_partition()
        load_model()
        root, partition = _trip_root()
        available = len(_trip_directories(root))
        if _EVALUATOR_ENV and available != 10:
            raise RuntimeError("evaluator folder count changed after startup")
        return {"status": "ok", "model_loaded": True, "model_name": _model_name(),
                "version": app.version, "partition": partition, "trips_available": available}
    except Exception as exc:
        logger.exception("health check failed")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {type(exc).__name__}") from exc


@app.get("/trips")
def trips():
    try:
        _validate_evaluator_partition()
        root, partition = _trip_root()
        items = []
        for directory in _trip_directories(root):
            trip = load_trip(directory, with_labels=False)
            items.append({"trip_id": trip.trip_id, "num_observations": trip.n_obs,
                          "num_nodes": len(trip.node_lonlat), "num_arcs": len(trip.arcs)})
        return {"partition": partition, "trips": items}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("trip listing failed")
        raise HTTPException(status_code=503, detail=f"Unable to list trips: {type(exc).__name__}") from exc


@app.post("/predict")
def predict(request: PredictRequest):
    directory, partition = _resolve_trip(request.trip_id)
    start = time.perf_counter()
    try:
        # All prediction callbacks use pipeline.py; route labels are read only below.
        components, state = _components_with_context(directory)
        prediction = predict_trip(directory, components, DecoderConfig())
        score_start = time.perf_counter()
        raw_metrics = score_after_prediction(directory, prediction, state) if (directory / f"{request.trip_id}.route").is_file() else None
        score_seconds = time.perf_counter() - score_start
        observations = prediction["observations"]
        raw_gps = [([float(o["coordinates"][1]), float(o["coordinates"][0])] if o["coordinates"] else None)
                   for o in observations]
        classifier_segments = _classifier_segments(prediction["classifier_arc_ids"], state["graph"])
        hmm_segments = _to_latlon_segments(prediction["route_polyline"])
        probabilities = state["probabilities"]
        confidence = _candidate_confidences(prediction["classifier_arc_ids"], state["candidate_sets"], probabilities)
        observation_rows = []
        for obs, conf in zip(observations, confidence):
            coords = obs["coordinates"]
            observation_rows.append({
                "index": obs["index"], "timestamp": obs["timestamp"],
                "lat": float(coords[1]) if coords else None,
                "lon": float(coords[0]) if coords else None,
                "classifier_arc_id": obs["classifier_arc_id"],
                "hmm_arc_id": obs["hmm_arc_id"], "classifier_confidence": conf,
            })
        if raw_metrics is None:
            metrics_out = None
        else:
            metrics_out = {
                "classifier_only": _route_metrics(raw_metrics, "classifier"),
                "hmm": _route_metrics(raw_metrics, "hmm"),
            }
        stage = prediction["stage_timings_s"]
        timings = {
            "load_and_index": stage.get("graph_and_spatial_index", 0.0) + stage.get("projection", 0.0),
            "candidates_and_features": stage.get("candidate_generation", 0.0) + stage.get("feature_generation", 0.0),
            "classifier_inference": stage.get("classifier_scoring", 0.0),
            "hmm_viterbi": stage.get("hmm_viterbi", 0.0) + stage.get("route_stitching", 0.0),
            "server_scoring": score_seconds,
            "total": time.perf_counter() - start,
        }
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        response = {
            "trip_id": request.trip_id, "model_name": _model_name(),
            "num_observations": len(observations), "raw_gps": raw_gps,
            "hmm_route_segments": hmm_segments,
            "hmm_route_polyline": [point for segment in hmm_segments for point in segment],
            "classifier_route_segments": classifier_segments,
            "observations": observation_rows, "metrics": metrics_out,
            "timings_sec": timings, "server_timestamp": now,
        }
        if raw_metrics:
            clf = metrics_out["classifier_only"]["top1_accuracy"]
            hmm = metrics_out["hmm"]["top1_accuracy"]
            continuity = metrics_out["hmm"]["route_continuity"]
            breaks = metrics_out["hmm"]["continuity_breaks"]
            logger.info("[%s] POST /predict trip_id=%s | obs=%d | clf_top1=%s | hmm_top1=%s | continuity=%s | breaks=%s | total=%.2fs",
                        now, request.trip_id, len(observations),
                        f"{clf:.1%}" if clf is not None else "n/a",
                        f"{hmm:.1%}" if hmm is not None else "n/a",
                        f"{continuity:.1%}" if continuity is not None else "n/a",
                        breaks if breaks is not None else "n/a", timings["total"])
        else:
            logger.info("[%s] POST /predict trip_id=%s | obs=%d | metrics=n/a | total=%.2fs",
                        now, request.trip_id, len(observations), timings["total"])
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("prediction failed for trip_id=%s", request.trip_id)
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}") from exc
