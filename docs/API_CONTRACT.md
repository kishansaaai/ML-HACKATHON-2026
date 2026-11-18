# PathSnap API Contract (Person C ↔ Person D)

This contract defines the HTTP endpoints and JSON payloads between **Laptop 1 (FastAPI Backend — Person C)** and **Laptop 2 (Evaluator UI — Person D)**.

> **Example Values Note:** Example values are illustrative only; real results come from the API.
>
> **Leakage Rule:** The UI sends **only** `trip_id`. Ground-truth `.route` files remain strictly server-side on Laptop 1 and are **never** returned in any API response.
>
> **Coordinate Order:** This contract uses **`[lat, lon]`** (WGS-84 degrees, with `lat` in `[-90, 90]` and `lon` in `[-180, 180]`) for **every** coordinate in every field. The backend is responsible for converting the decoder's internal `[lon, lat]` output to `[lat, lon]` before returning JSON.

---

## Base Configuration

- **Laptop 1 bind address:** `0.0.0.0:8000`
- **CORS:** Allow Laptop 2 origin (or `*` for LAN demo)
- **Laptop 2 config:** `PATHSNAP_API_URL=http://<LAPTOP_1_IP>:8000` (configured in `ui/config.py` or Streamlit sidebar)

---

## 1. `GET /health`

Checks that the backend, trained model (plain LightGBM; no feature preprocessing transform), and HMM configuration are loaded.

- `model_name` must be read from `artifacts/model_card.json` (key `"model"`), e.g. `"lightgbm"`.
- `partition` identifies the active server-side trip directory (e.g. `"evaluator_test"` during final evaluation). The UI shows a red warning banner if `model_name` contains `"MOCK"` (case-insensitive) or if `partition` is not the evaluator partition.

### Response `200 OK`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "<string from artifacts/model_card.json key 'model', e.g. lightgbm>",
  "version": "<semver string>",
  "partition": "<evaluator_test | student_test>",
  "trips_available": "<int>"
}
```

---

## 2. `GET /trips`

Lists the evaluation trips discovered from the server-side evaluation folder.

### Response `200 OK`
```json
{
  "partition": "<evaluator_test | student_test>",
  "trips": [
    {
      "trip_id": "<trip_id string>",
      "num_observations": "<int>",
      "num_nodes": "<int>",
      "num_arcs": "<int>"
    }
  ]
}
```
*(Note: The UI also accepts `{"trips": ["<trip_id>", ...]}` as a plain list of trip ID strings.)*

---

## 3. `POST /predict`

Runs the full server-side prediction pipeline (`predict_trip` using `.track`, `.nodes`, `.arcs` only), converts decoder `[lon, lat]` geometry to `[lat, lon]`, and then computes evaluation metrics in the separate server-side scoring module (`scoring.py`).

### Request Body
```json
{
  "trip_id": "<trip_id string>"
}
```

### Route Geometry & Disconnections
- **`hmm_route_segments` (primary):** A list of polylines, where each polyline is a list of `[lat, lon]` points representing one contiguous connected route segment. Wherever the decoder encounters a route break / disconnection, it starts a **new** segment in `hmm_route_segments`. The UI draws each segment as an independent `PolyLine` and never draws a connecting line across breaks.
- **`hmm_route_polyline` (optional / legacy):** A single flat list of `[lat, lon]` points. Used by the UI **only** as a fallback when `hmm_route_segments` is absent.

### Response `200 OK`
```json
{
  "trip_id": "<trip_id string>",
  "model_name": "<string from artifacts/model_card.json key 'model', e.g. lightgbm>",
  "num_observations": "<int>",
  "raw_gps": [
    ["<float lat>", "<float lon>"],
    ["<float lat>", "<float lon>"]
  ],
  "hmm_route_segments": [
    [
      ["<float lat>", "<float lon>"],
      ["<float lat>", "<float lon>"]
    ],
    [
      ["<float lat>", "<float lon>"],
      ["<float lat>", "<float lon>"]
    ]
  ],
  "hmm_route_polyline": [
    ["<float lat>", "<float lon>"],
    ["<float lat>", "<float lon>"]
  ],
  "classifier_route_segments": [
    [
      ["<float lat>", "<float lon>"],
      ["<float lat>", "<float lon>"]
    ]
  ],
  "observations": [
    {
      "index": "<int>",
      "timestamp": "<float seconds>",
      "lat": "<float lat>",
      "lon": "<float lon>",
      "classifier_arc_id": "<int>",
      "hmm_arc_id": "<int>",
      "classifier_confidence": "<float 0-1>"
    }
  ],
  "metrics": {
    "classifier_only": {
      "top1_accuracy": "<float 0-1>",
      "macro_f1": "<float 0-1>",
      "precision": "<float 0-1>",
      "recall": "<float 0-1>",
      "log_loss": "<float >=0 or null>",
      "route_continuity": "<float 0-1>",
      "continuity_breaks": "<int >=0>",
      "sequence_f1": "<float 0-1>",
      "candidate_miss_rate": "<float 0-1>"
    },
    "hmm": {
      "top1_accuracy": "<float 0-1>",
      "macro_f1": "<float 0-1>",
      "precision": "<float 0-1>",
      "recall": "<float 0-1>",
      "log_loss": "<float >=0 or null>",
      "route_continuity": "<float 0-1>",
      "continuity_breaks": "<int >=0>",
      "sequence_f1": "<float 0-1>",
      "candidate_miss_rate": "<float 0-1>"
    }
  },
  "timings_sec": {
    "load_and_index": "<float seconds>",
    "candidates_and_features": "<float seconds>",
    "classifier_inference": "<float seconds>",
    "hmm_viterbi": "<float seconds>",
    "server_scoring": "<float seconds>",
    "total": "<float seconds>"
  },
  "server_timestamp": "<ISO-8601 UTC string>"
}
```

### Error Responses
- `404 Not Found`: `{"detail": "Trip '<trip_id>' not found."}`
- `500 Internal Server Error`: `{"detail": "Inference failed: <reason>"}`

---

## Logging Requirement (Laptop 1 Console)

Every `POST /predict` request must print a live console log line on Laptop 1 for the evaluator, e.g.:
```text
[<ISO-8601 UTC>] POST /predict trip_id=<trip_id> | obs=<int> | clf_top1=<float>% | hmm_top1=<float>% | continuity=<float>% | breaks=<int> | total=<float>s
```
