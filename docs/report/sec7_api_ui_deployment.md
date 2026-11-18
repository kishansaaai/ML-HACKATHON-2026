# 7. API, UI and Deployment

## 7.1 Architecture

**Laptop 1** runs the FastAPI server (`src/pathsnap/api.py`): road data, the LightGBM model, the HMM decoder and server-side scoring. **Laptop 2** runs the Streamlit UI (`ui/`), which sends only a `trip_id` and draws what comes back. The reference route never leaves Laptop 1.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | status, `model_loaded`, `model_name`, version, `partition`, number of trips available |
| `GET /trips` | the partition name and its trips with counts of observations, nodes and arcs |
| `POST /predict` `{"trip_id": ...}` | runs candidates, classifier and HMM; returns raw GPS, HMM route segments, classifier route segments, per-observation arc IDs and confidence, classifier-only and HMM metrics, per-stage timings |

The full field list is in `docs/API_CONTRACT.md`.

## 7.2 No ground truth leakage

- Prediction (`pathsnap.pipeline.predict`) loads trips with `with_labels=False` and cannot read `.route`.
- `src/pathsnap/scoring.py` opens `.route` only after the prediction is finished, to compute the metrics; the response contains no reference-route field.
- A test runs `/predict` on a trip copy without a `.route` file; it still returns a prediction and `metrics: null` (`tests/test_api_integration.py`).
- The evaluator folder is set with `PATHSNAP_EVALUATOR_DIR`; startup fails unless it holds exactly 10 valid trip folders, so the server never silently falls back to development data. The evaluator trips are not in the repository (`.gitignore`).
- Nothing is hard-coded: every trip is processed by the same code path.

## 7.3 Logging

Every `POST /predict` prints one readable line on the API console and in `logs/api.log`, for example:

`POST /predict trip_id=00000052 | obs=262 | clf_top1=97.3% | hmm_top1=98.1% | continuity=95.8% | total=0.73s`

## 7.4 UI

The UI shows the model name (from `/health`) at the top, the connection status, API address and partition in the sidebar, a dropdown with the trips returned by `/trips` and a **Match route** button. After a run it shows four headline numbers (Top-1, route continuity, sequence F1, server time with the difference from the classifier alone), the Folium map (raw GPS, classifier-only, HMM route, start and end markers, layer control), and tabs for the classifier-versus-HMM table, per-stage timings, the per-observation records and the raw API summary. Errors (unreachable server, bad trip) show readable messages.

## 7.5 Timings

`python scripts/evaluate_api.py` runs all 10 student-test trips through the API and checks that its pooled Top-1 equals the offline pipeline's (difference 0.0000 % in the last run). Server time per trip, measured on the development laptop:

| Trip | GPS points | Server time (s) |
| --- | ---: | ---: |
| 00000000 | 2,503 | 2.44 |
| 00000007 | 1,354 | 0.85 |
| 00000009 | 1,049 | 3.05 |
| 00000040 | 2,995 | 3.57 |
| 00000052 | 262 | 0.73 |
| 00000056 | 3,958 | 15.36 |
| 00000065 | 768 | 3.19 |
| 00000076 | 1,486 | 6.88 |
| 00000077 | 3,911 | 6.35 |
| 00000091 | 874 | 2.58 |

The slowest trip is 00000056 at 15.4 s, under the 30 s target.

## 7.6 Running the two-laptop system

1. **Laptop 1** (both laptops on the same Wi-Fi or hotspot; allow inbound TCP on the API port in the firewall): `pip install -r requirements.txt`, `pip install -e .`, then `set PATHSNAP_EVALUATOR_DIR=<folder with the 10 evaluator trips>` and `uvicorn pathsnap.api:app --app-dir src --host 0.0.0.0 --port 8000`. Keep this console visible: it prints the log line of each request.
2. **Laptop 2:** `pip install -r ui/requirements.txt`, then `cd ui` and `run_ui.bat http://<laptop-1-ip>:8000`; the UI opens at `http://localhost:8501`.
3. Check `http://<laptop-1-ip>:8000/health` first: it must show the model name and the right partition.

**Status.** The API and UI run together from this repository on one development laptop; the API was reached through the laptop's LAN address with the firewall rule in place. The complete two-laptop run on the evaluator trips is performed at the evaluation. The step-by-step checklist is `docs/DEMO_CHECKLIST.md`.
