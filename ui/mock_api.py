"""DEV-ONLY Mock FastAPI server for testing Person D's UI locally.

IMPORTANT:
- This file is ONLY a local development stub to test `ui/app.py` before
  Person C's real FastAPI backend (with Person A's trained model and Person B's
  HMM/Viterbi decoder) is deployed on Laptop 1.
- Do NOT use this mock server for final evaluation.
- Even in this dev stub, `_mock_predict_without_route` reads ONLY `.track`,
  `.nodes`, and `.arcs`; `.route` is read solely inside `_score_against_route`
  after prediction finishes, matching the production anti-leakage boundary.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel


def _resolve_data_dir() -> Path:
    env_dir = os.getenv("PATHSNAP_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    candidates = [
        Path(__file__).resolve().parent.parent / "Student folder" / "student_test",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DATA_DIR = _resolve_data_dir()

app = FastAPI(
    title="PathSnap DEV Mock API",
    version="0.1.0-dev-mock",
    description="Development stub for UI integration testing only.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trip_id: str


def _list_trip_dirs() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _load_trip_inputs(trip_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ONLY .track, .nodes, .arcs (never .route)."""
    tid = trip_dir.name
    track = np.loadtxt(trip_dir / f"{tid}.track", dtype=np.float64)
    nodes = np.loadtxt(trip_dir / f"{tid}.nodes", dtype=np.float64)
    arcs = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=np.int64)
    if track.ndim == 1:
        track = track.reshape(1, -1)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)
    return track, nodes, arcs


def _project_to_meters(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    scale_x = 111_320.0 * math.cos(math.radians(lat0))
    scale_y = 110_540.0
    return (lon - lon0) * scale_x, (lat - lat0) * scale_y


def _mock_predict_without_route(trip_dir: Path) -> dict[str, Any]:
    """Simple nearest-arc geometry stub + window smoothing (no ML, no .route)."""
    t0 = time.perf_counter()
    track, nodes, arcs = _load_trip_inputs(trip_dir)
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    lon0 = float(np.mean(track[:, 0]))
    lat0 = float(np.mean(track[:, 1]))

    px, py = _project_to_meters(track[:, 0], track[:, 1], lon0, lat0)
    nx, ny = _project_to_meters(nodes[:, 0], nodes[:, 1], lon0, lat0)

    ax1, ay1 = nx[arcs[:, 0]], ny[arcs[:, 0]]
    ax2, ay2 = nx[arcs[:, 1]], ny[arcs[:, 1]]
    mid_x = 0.5 * (ax1 + ax2)
    mid_y = 0.5 * (ay1 + ay2)

    # Bounding-box prefilter around trajectory (+300 m) for fast mock lookup
    pad = 300.0
    mask = (
        (mid_x >= px.min() - pad)
        & (mid_x <= px.max() + pad)
        & (mid_y >= py.min() - pad)
        & (mid_y <= py.max() + pad)
    )
    cand_arc_ids = np.nonzero(mask)[0]
    if len(cand_arc_ids) == 0:
        cand_arc_ids = np.arange(len(arcs))

    sx1, sy1 = ax1[cand_arc_ids], ay1[cand_arc_ids]
    sx2, sy2 = ax2[cand_arc_ids], ay2[cand_arc_ids]
    dx = sx2 - sx1
    dy = sy2 - sy1
    seg_len2 = dx * dx + dy * dy + 1e-9
    t_feat = time.perf_counter() - t1

    # Mock classifier-only: perpendicular distance to nearest arc (chunked)
    t2 = time.perf_counter()
    n_obs = len(track)
    clf_arcs = np.empty(n_obs, dtype=np.int64)
    clf_conf = np.empty(n_obs, dtype=np.float64)
    chunk = 200
    for start in range(0, n_obs, chunk):
        end = min(start + chunk, n_obs)
        cx = px[start:end, None]
        cy = py[start:end, None]
        proj = ((cx - sx1[None, :]) * dx[None, :] + (cy - sy1[None, :]) * dy[None, :]) / seg_len2[None, :]
        proj = np.clip(proj, 0.0, 1.0)
        qx = sx1[None, :] + proj * dx[None, :]
        qy = sy1[None, :] + proj * dy[None, :]
        dist2 = (cx - qx) ** 2 + (cy - qy) ** 2
        best_local = np.argmin(dist2, axis=1)
        best_dist = np.sqrt(dist2[np.arange(end - start), best_local])
        clf_arcs[start:end] = cand_arc_ids[best_local]
        clf_conf[start:end] = np.exp(-best_dist / 25.0)
    t_clf = time.perf_counter() - t2

    # Mock HMM smoothing: 7-point temporal mode filter to preview UI comparison
    t3 = time.perf_counter()
    hmm_arcs = clf_arcs.copy()
    half_win = 3
    for i in range(n_obs):
        lo = max(0, i - half_win)
        hi = min(n_obs, i + half_win + 1)
        hmm_arcs[i] = Counter(clf_arcs[lo:hi].tolist()).most_common(1)[0][0]
    t_hmm = time.perf_counter() - t3

    # Build map geometry ([lat, lon] order for Folium)
    raw_gps = [[round(float(lat), 6), round(float(lon), 6)] for lon, lat in track[:, :2]]

    # Deduplicate consecutive identical arcs for clean route polyline
    hmm_polyline: list[list[float]] = []
    last_arc = -1
    for arc_id in hmm_arcs.tolist():
        if arc_id == last_arc:
            continue
        last_arc = arc_id
        u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
        p_u = [round(float(nodes[u, 1]), 6), round(float(nodes[u, 0]), 6)]
        p_v = [round(float(nodes[v, 1]), 6), round(float(nodes[v, 0]), 6)]
        if not hmm_polyline or hmm_polyline[-1] != p_u:
            hmm_polyline.append(p_u)
        hmm_polyline.append(p_v)

    clf_segments: list[list[list[float]]] = []
    seen_clf: set[int] = set()
    for arc_id in clf_arcs.tolist():
        if arc_id in seen_clf:
            continue
        seen_clf.add(arc_id)
        u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
        clf_segments.append(
            [
                [round(float(nodes[u, 1]), 6), round(float(nodes[u, 0]), 6)],
                [round(float(nodes[v, 1]), 6), round(float(nodes[v, 0]), 6)],
            ]
        )

    observations = [
        {
            "index": i,
            "timestamp": float(track[i, 2]),
            "lat": round(float(track[i, 1]), 6),
            "lon": round(float(track[i, 0]), 6),
            "classifier_arc_id": int(clf_arcs[i]),
            "hmm_arc_id": int(hmm_arcs[i]),
            "classifier_confidence": round(float(clf_conf[i]), 4),
        }
        for i in range(n_obs)
    ]

    return {
        "arcs_array": arcs,
        "clf_arcs": clf_arcs,
        "hmm_arcs": hmm_arcs,
        "raw_gps": raw_gps,
        "hmm_route_polyline": hmm_polyline,
        "classifier_route_segments": clf_segments,
        "observations": observations,
        "timings": {
            "load_and_index": round(t_load, 3),
            "candidates_and_features": round(t_feat, 3),
            "classifier_inference": round(t_clf, 3),
            "hmm_viterbi": round(t_hmm, 3),
        },
    }


def _continuity_stats(pred_arcs: np.ndarray, arcs: np.ndarray) -> tuple[float, int]:
    """Compute head-to-tail continuity over consecutive distinct predicted arcs."""
    seq: list[int] = []
    for a in pred_arcs.tolist():
        if not seq or seq[-1] != a:
            seq.append(a)
    if len(seq) <= 1:
        return 1.0, 0
    breaks = 0
    for i in range(1, len(seq)):
        if int(arcs[seq[i - 1], 1]) != int(arcs[seq[i], 0]):
            breaks += 1
    return (len(seq) - 1 - breaks) / (len(seq) - 1), breaks


def _score_against_route(trip_dir: Path, pred: dict[str, Any]) -> dict[str, Any]:
    """Separate server-side scoring step: reads `.route` ONLY after prediction."""
    route_path = trip_dir / f"{trip_dir.name}.route"
    if not route_path.exists():
        return {"classifier_only": {}, "hmm": {}}

    ref_arcs = np.loadtxt(route_path, dtype=np.int64)
    if ref_arcs.ndim == 0:
        ref_arcs = ref_arcs.reshape(1)
    ref_set = set(ref_arcs.tolist())
    arcs = pred["arcs_array"]

    def _eval_seq(pred_arcs: np.ndarray, include_logloss: bool) -> dict[str, Any]:
        in_ref = np.isin(pred_arcs, list(ref_set))
        approx_top1 = float(np.mean(in_ref))
        pred_set = set(pred_arcs.tolist())
        inter = len(pred_set & ref_set)
        prec = inter / len(pred_set) if pred_set else 0.0
        rec = inter / len(ref_set) if ref_set else 0.0
        seq_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        cont, breaks = _continuity_stats(pred_arcs, arcs)
        return {
            "top1_accuracy": round(approx_top1, 4),
            "macro_f1": round(seq_f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "log_loss": round(max(0.05, 1.0 - approx_top1), 4) if include_logloss else None,
            "route_continuity": round(cont, 4),
            "continuity_breaks": int(breaks),
            "sequence_f1": round(seq_f1, 4),
            "candidate_miss_rate": 0.0,
        }

    return {
        "classifier_only": _eval_seq(pred["clf_arcs"], include_logloss=True),
        "hmm": _eval_seq(pred["hmm_arcs"], include_logloss=False),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    trips = _list_trip_dirs()
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": "DEV MOCK (Nearest-Arc Stub — Replace with Laptop 1 Real API)",
        "version": "0.1.0-dev-mock",
        "partition": DATA_DIR.name if DATA_DIR.exists() else "missing",
        "trips_available": len(trips),
    }


@app.get("/trips")
def list_trips() -> dict[str, Any]:
    trips = []
    for trip_dir in _list_trip_dirs():
        tid = trip_dir.name
        trips.append(
            {
                "trip_id": tid,
                "num_observations": _count_lines(trip_dir / f"{tid}.track"),
                "num_nodes": _count_lines(trip_dir / f"{tid}.nodes"),
                "num_arcs": _count_lines(trip_dir / f"{tid}.arcs"),
            }
        )
    return {"partition": DATA_DIR.name if DATA_DIR.exists() else "missing", "trips": trips}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    trip_dir = DATA_DIR / req.trip_id
    if not trip_dir.exists() or not trip_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Trip '{req.trip_id}' not found in {DATA_DIR.name}.")

    t_total_start = time.perf_counter()
    pred = _mock_predict_without_route(trip_dir)

    t_score_start = time.perf_counter()
    metrics = _score_against_route(trip_dir, pred)
    t_score = time.perf_counter() - t_score_start
    t_total = time.perf_counter() - t_total_start

    timings = pred["timings"]
    timings["server_scoring"] = round(t_score, 3)
    timings["total"] = round(t_total, 3)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    clf_acc = metrics.get("classifier_only", {}).get("top1_accuracy", 0.0) * 100
    hmm_acc = metrics.get("hmm", {}).get("top1_accuracy", 0.0) * 100
    hmm_cont = metrics.get("hmm", {}).get("route_continuity", 0.0) * 100
    print(
        f"[{ts}] POST /predict trip_id={req.trip_id} | obs={len(pred['observations'])} | "
        f"clf_top1={clf_acc:.1f}% | hmm_top1={hmm_acc:.1f}% | continuity={hmm_cont:.1f}% | "
        f"total={t_total:.2f}s",
        flush=True,
    )

    return {
        "trip_id": req.trip_id,
        "model_name": "DEV MOCK (Nearest-Arc Stub)",
        "num_observations": len(pred["observations"]),
        "raw_gps": pred["raw_gps"],
        "hmm_route_polyline": pred["hmm_route_polyline"],
        "classifier_route_segments": pred["classifier_route_segments"],
        "observations": pred["observations"],
        "metrics": metrics,
        "timings_sec": timings,
        "server_timestamp": ts,
    }
