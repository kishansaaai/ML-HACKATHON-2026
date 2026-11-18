# PathSnap API Contract (Person C ↔ Person D)

This contract defines the exact HTTP endpoints and JSON payloads between **Laptop 1 (FastAPI Backend — Person C)** and **Laptop 2 (Evaluator UI — Person D)**.

> **Leakage Rule:** The UI sends **only** `trip_id`. Ground-truth `.route` files remain strictly server-side on Laptop 1 and are **never** returned in any API response.

---

## Base Configuration

- **Laptop 1 bind address:** `0.0.0.0:8000`
- **CORS:** Allow Laptop 2 origin (or `*` for LAN demo)
- **Laptop 2 config:** `PATHSNAP_API_URL=http://<LAPTOP_1_IP>:8000` (configured in `ui/config.py` or Streamlit sidebar)

---

## 1. `GET /health`

Checks that the backend, trained model, scaler, and HMM configuration are loaded.

### Response `200 OK`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "LightGBM (Bayesian-tuned)",
  "version": "1.0.0",
  "partition": "evaluator_test",
  "trips_available": 10
}
```

---

## 2. `GET /trips`

Lists the 10 reserved evaluation trips discovered from the server-side evaluation folder.

### Response `200 OK`
```json
{
  "partition": "evaluator_test",
  "trips": [
    {
      "trip_id": "00000000",
      "num_observations": 1520,
      "num_nodes": 8542,
      "num_arcs": 18196
    }
  ]
}
```
*(Note: The UI also accepts `{"trips": ["00000000", "00000007", ...]}` as a plain list of trip ID strings.)*

---

## 3. `POST /predict`

Runs the full server-side prediction pipeline (`predict_trip` using `.track`, `.nodes`, `.arcs` only), then computes evaluation metrics in the separate server-side scoring module (`scoring.py`).

### Request Body
```json
{
  "trip_id": "00000000"
}
```

### Response `200 OK`
All map coordinates use **`[lat, lon]`** order (WGS-84 degrees) for direct Folium rendering.

```json
{
  "trip_id": "00000000",
  "model_name": "LightGBM (Bayesian-tuned)",
  "num_observations": 1520,
  "raw_gps": [
    [56.251559, 54.963519],
    [56.251523, 54.963143]
  ],
  "hmm_route_polyline": [
    [56.251560, 54.963520],
    [56.251525, 54.963140]
  ],
  "classifier_route_segments": [
    [[56.251560, 54.963520], [56.251525, 54.963140]]
  ],
  "observations": [
    {
      "index": 0,
      "timestamp": 1.0,
      "lat": 56.251559,
      "lon": 54.963519,
      "classifier_arc_id": 17630,
      "hmm_arc_id": 17630,
      "classifier_confidence": 0.91
    }
  ],
  "metrics": {
    "classifier_only": {
      "top1_accuracy": 0.842,
      "macro_f1": 0.815,
      "precision": 0.831,
      "recall": 0.804,
      "log_loss": 0.214,
      "route_continuity": 0.786,
      "continuity_breaks": 62,
      "sequence_f1": 0.801,
      "candidate_miss_rate": 0.018
    },
    "hmm": {
      "top1_accuracy": 0.934,
      "macro_f1": 0.912,
      "precision": 0.920,
      "recall": 0.906,
      "log_loss": null,
      "route_continuity": 0.992,
      "continuity_breaks": 2,
      "sequence_f1": 0.928,
      "candidate_miss_rate": 0.018
    }
  },
  "timings_sec": {
    "load_and_index": 0.42,
    "candidates_and_features": 0.68,
    "classifier_inference": 0.11,
    "hmm_viterbi": 1.35,
    "server_scoring": 0.04,
    "total": 2.60
  },
  "server_timestamp": "2026-11-18T11:30:00Z"
}
```

### Error Responses
- `404 Not Found`: `{"detail": "Trip '99999999' not found."}`
- `500 Internal Server Error`: `{"detail": "Inference failed: <reason>"}`

---

## Logging Requirement (Laptop 1 Console)

Every `POST /predict` request must print a clear console log line on Laptop 1 for the evaluator, e.g.:
```text
[2026-11-18 11:30:00] POST /predict trip_id=00000000 | obs=1520 | clf_top1=84.2% | hmm_top1=93.4% | continuity=99.2% | total=2.60s
```
