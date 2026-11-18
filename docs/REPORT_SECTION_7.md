# Section 7: API, UI and Deployment

*(Draft prepared by Person D for inclusion in the team's consolidated Technical Report PDF. Replace screenshot placeholders with actual captures from the live two-laptop run.)*

---

## 7.1 Two-Laptop Distributed Deployment Architecture

PathSnap is deployed across two laptops connected over a shared local network (LAN / hotspot) to enforce a strict boundary between model inference / evaluation and user presentation:

- **Laptop 1 — Model & FastAPI Backend (`0.0.0.0:8000`):**
  - Hosts the exported classifier, preprocessing scaler, feature configuration, road-network loader, and HMM/Viterbi decoder.
  - Exposes three REST endpoints (`GET /health`, `GET /trips`, `POST /predict`) with CORS enabled for cross-laptop browser/UI access.
  - Runs the prediction pipeline (`predict_trip`) strictly on `.track`, `.nodes`, and `.arcs`. Only after prediction completes does the isolated server-side scoring module (`scoring.py`) read the trip's `.route` file to compute evaluation metrics.
  - Logs every incoming request (timestamp, `trip_id`, stage timings, and resulting metrics) to the Laptop 1 console for live verification during evaluation.

- **Laptop 2 — Evaluator User Interface (`Streamlit + Folium`, port `8501`):**
  - Connects to Laptop 1 using a single configurable API base URL (`PATHSNAP_API_URL`, configurable via `ui/config.py`, environment variable, or the sidebar input).
  - Performs **zero** local machine learning inference, route decoding, or file reads. It fetches the ten evaluation trip IDs via `GET /trips`, sends only `{"trip_id": "<id>"}` to `POST /predict`, and renders the returned JSON response.
  - Displays:
    - Live backend health status, model version, and network latency.
    - Ten selectable trip cards with a dedicated **Run Map-Matching** trigger and loading spinner.
    - An interactive Folium map (`prefer_canvas=True`, auto-fitted bounds, legend, and `LayerControl`) plotting raw GPS observations, the HMM/Viterbi predicted route, and an optional toggleable classifier-only comparison layer.
    - A Classifier-Only vs. Classifier + HMM metrics comparison table and per-stage server latency breakdown.
    - Clear, readable error states if Laptop 1 is unreachable or returns an HTTP error.

---

## 7.2 API Endpoint Summary

| Endpoint | Method | Request | Response Highlights |
| --- | --- | --- | --- |
| `/health` | `GET` | — | Backend readiness, `model_loaded`, model name/version, available trip count |
| `/trips` | `GET` | — | List of the 10 server-side evaluation trip IDs (and optional observation/arc counts) |
| `/predict` | `POST` | `{"trip_id": "..."}` | Raw GPS coordinates, HMM route polyline, classifier-only segments, per-observation predictions, Classifier vs. HMM metrics, per-stage server timings (no `.route` labels returned) |

---

## 7.3 Working UI Screenshots (Laptop 2)

> **[Insert Figure 7.1: `docs/screenshots/ui_trip_selector.png`]**
> *Figure 7.1: PathSnap Evaluator Console running on Laptop 2, showing the live connection badge to Laptop 1 and the ten dynamically loaded evaluation trip cards.*

> **[Insert Figure 7.2: `docs/screenshots/ui_route_map_and_metrics.png`]**
> *Figure 7.2: Interactive Folium route visualization and Classifier-Only vs. Classifier + HMM evaluation metrics returned by Laptop 1 for a selected evaluation trip.*

---

## 7.4 Successful API Response Logs on Laptop 1 CMD

> **[Insert Figure 7.3: `docs/screenshots/laptop1_api_cmd_logs.png`]**
> *Figure 7.3: Command Prompt / terminal output on Laptop 1 showing live `GET /health`, `GET /trips`, and `POST /predict` HTTP 200 responses with per-trip accuracy, continuity, and inference timings.*

---

## 7.5 Deployment & Reproduction Steps

1. **Laptop 1 (Backend):**
   ```powershell
   pip install -r requirements.txt
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Laptop 2 (Evaluator UI):**
   ```powershell
   cd ui
   pip install -r requirements.txt
   run_ui.bat http://<LAPTOP_1_IP>:8000
   ```
3. Open `http://localhost:8501` on Laptop 2, verify the green **Connected** badge, select any of the ten evaluation trips, and click **Run Map-Matching**.
