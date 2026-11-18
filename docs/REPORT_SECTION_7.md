# Section 7: API, UI and Deployment

*(Draft prepared by Person D for inclusion in the team's consolidated Technical Report PDF. Replace screenshot placeholders below with actual captures from the live two-laptop run against the real backend.)*

---

## 7.1 Two-Laptop Distributed Deployment Architecture

PathSnap is deployed across two laptops connected over a shared local network (LAN / hotspot) to enforce a strict boundary between model inference / evaluation and user presentation:

- **Laptop 1 — Model & FastAPI Backend (`0.0.0.0:8000`):**
  - Hosts the exported LightGBM classifier (described by `artifacts/model_card.json`, key `"model"`), feature configuration, road-network loader, and HMM/Viterbi decoder. No separate feature transform is used at inference time.
  - Exposes three REST endpoints (`GET /health`, `GET /trips`, `POST /predict`) with CORS enabled for cross-laptop UI access.
  - Runs the prediction pipeline (`predict_trip`) strictly on `.track`, `.nodes`, and `.arcs`, and converts the decoder's `[lon, lat]` coordinates to `[lat, lon]` before serializing JSON. Only after prediction completes does the isolated server-side scoring module (`scoring.py`) read the trip's `.route` file to compute evaluation metrics.
  - Logs every incoming request (timestamp, `trip_id`, stage timings, and resulting metrics) to the Laptop 1 console for live verification during evaluation.

- **Laptop 2 — Evaluator User Interface (`Streamlit + Folium`, default port `8501`):**
  - Connects to Laptop 1 using a single configurable API base URL (`PATHSNAP_API_URL`, configurable via `ui/config.py`, environment variable, or the sidebar input).
  - Performs **zero** local machine learning inference, route decoding, or file reads. It fetches the ten evaluation trip IDs via `GET /trips`, sends only `{"trip_id": "<id>"}` to `POST /predict`, and renders the returned JSON response.
  - Displays:
    - The model name (from `GET /health`, e.g. `LightGBM (exported)`) as a small chip at the top of the page, and the active `partition` in the sidebar connection status (`student_test` while rehearsing, `evaluator` at the real run). A red "Backend unreachable" box appears only if `GET /health` fails.
    - A trip dropdown (one entry per trip returned by `GET /trips`, labelled with its GPS-point count) and a **Match route** button with a loading spinner.
    - A KPI card for the selected trip (HMM Top-1 accuracy, route continuity with break count, sequence F1, server time), each compared against the classifier alone where applicable.
    - An interactive Folium map (`OpenStreetMap` tiles, `prefer_canvas=True`, auto-fitted bounds, legend, and `LayerControl`) that validates `[lat, lon]` coordinates (warning if any out-of-range point is skipped) and draws each connected segment in `hmm_route_segments` as its own `PolyLine` without joining disconnected breaks, alongside the route-break count (`metrics.hmm.continuity_breaks`).
    - Below the map, tabs **Classifier vs HMM** (metric table), **Timings** (per-stage server latency), **Observations** (first 100 per-point records) and **API response** (the JSON with long lists collapsed), plus a **Download map (HTML)** button.
    - Clear, readable error states if Laptop 1 is unreachable or returns an HTTP error.

---

## 7.2 API Endpoint Summary

| Endpoint | Method | Request | Response Highlights |
| --- | --- | --- | --- |
| `/health` | `GET` | — | Backend readiness, `model_loaded`, `model_name` (from `artifacts/model_card.json`), active `partition`, available trip count |
| `/trips` | `GET` | — | List of the server-side evaluation trip IDs (and optional observation/arc counts) |
| `/predict` | `POST` | `{"trip_id": "..."}` | Raw GPS `[lat, lon]`, disconnected-safe `hmm_route_segments` (with optional legacy `hmm_route_polyline`), classifier-only segments, per-observation predictions, Classifier vs. HMM metrics, per-stage server timings (no `.route` labels returned) |

---

## 7.3 Working UI Screenshots (Laptop 2)

> **[Insert Figure 7.1: `docs/screenshots/ui_trip_selector.png`]**
> *Figure 7.1: PathSnap Evaluator Console running on Laptop 2, showing the model-name chip at the top, the partition in the sidebar connection status, and the trip dropdown with the Match route button.*

> **[Insert Figure 7.2: `docs/screenshots/ui_route_map_and_metrics.png`]**
> *Figure 7.2: Interactive Folium route visualization (with route-break count and separate segment polylines) and KPI card and the Classifier vs HMM table returned by Laptop 1 for a selected evaluation trip.*

---

## 7.4 Successful API Response Logs on Laptop 1 CMD

> **[Insert Figure 7.3: `docs/screenshots/laptop1_api_cmd_logs.png`]**
> *Figure 7.3: Command Prompt / terminal output on Laptop 1 showing live `GET /health`, `GET /trips`, and `POST /predict` HTTP 200 responses with per-trip metrics and inference timings.*

---

## 7.5 Deployment & Reproduction Steps

1. **Laptop 1 (Backend):**
   ```powershell
   pip install -r requirements.txt
   pip install -e .
   uvicorn pathsnap.api:app --app-dir src --host 0.0.0.0 --port 8000
   ```
2. **Laptop 2 (Evaluator UI):**
   ```powershell
   cd ui
   pip install -r requirements.txt
   run_ui.bat http://<LAPTOP_1_IP>:8000
   ```
3. Open `http://localhost:8501` on Laptop 2, confirm the chip at the top shows the real model name and the sidebar shows `Partition: evaluator`, choose a trip in the dropdown, and click **Match route**.
