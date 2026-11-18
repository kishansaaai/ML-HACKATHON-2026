> DRAFT prepared with AI assistance from repository files. The team must rewrite this in its own words and be able to explain every statement (assessment policy).

# 7. API, UI and Deployment

The guideline (`Deliverables.pdf`, section 7) asks for screenshots of the working UI and of successful API responses on the CMD of the receiving laptop, and states that the submitted source code must match what is demonstrated at evaluation. This draft describes the implementation in words and tables; the screenshots are listed in 7.8 and still have to be taken on the real two-laptop setup.

## 7.1 Architecture

| Laptop | Runs | Source |
| --- | --- | --- |
| Laptop 1 | FastAPI service: exported LightGBM classifier, HMM/Viterbi decoder, road data for each trip, server-side scoring | `src/pathsnap/api.py`, `pipeline.py`, `scoring.py`, `artifacts/` |
| Laptop 2 | Streamlit + Folium UI; sends only a trip id, draws the answer | `ui/app.py`, `ui/api_client.py`, `ui/map_renderer.py`, `ui/config.py` |

The UI does no inference and has no path to dataset files (`ui/app.py` docstring; it talks only to `/health`, `/trips`, `/predict`).

## 7.2 Endpoints (read from `src/pathsnap/api.py`; contract in `docs/API_CONTRACT.md`)

| Endpoint | Request | Response fields |
| --- | --- | --- |
| `GET /health` | none | `status`, `model_loaded`, `model_name` (from `artifacts/model_card.json` key `model`, formatted e.g. "LightGBM (exported)"), `version`, `partition`, `trips_available`. Returns 503 if the model or partition check fails. |
| `GET /trips` | none | `partition`, `trips[]` each with `trip_id`, `num_observations`, `num_nodes`, `num_arcs`. Trips are loaded with `with_labels=False`. |
| `POST /predict` | `{"trip_id": "<id>"}` (1-64 characters, pattern `[A-Za-z0-9_-]`) | see below |

`POST /predict` response fields: `trip_id`, `model_name`, `num_observations`, `raw_gps` (list of `[lat, lon]`), `hmm_route_segments` (list of polylines of `[lat, lon]`; one polyline per connected segment), `hmm_route_polyline` (flat list, legacy fallback), `classifier_route_segments`, `observations[]` (`index`, `timestamp`, `lat`, `lon`, `classifier_arc_id`, `hmm_arc_id`, `classifier_confidence`), `metrics` (`classifier_only` and `hmm`, each with `top1_accuracy`, `macro_f1`, `precision`, `recall`, `log_loss` (classifier only; null for HMM), `route_continuity`, `continuity_breaks`, `sequence_f1`, `candidate_miss_rate`, `n_labelled`; `metrics` is null when the trip has no `.route` file), `timings_sec` (`load_and_index`, `candidates_and_features`, `classifier_inference`, `hmm_viterbi`, `server_scoring`, `total`), `server_timestamp` (UTC ISO-8601).

Errors: 400 invalid `trip_id`; 404 trip folder not found; 500 missing `.track`/`.nodes`/`.arcs` or inference failure (`"Inference failed: <exception type>"`, the exception type only, not a stack trace); 503 for `/health` or `/trips` failures. CORS origins default to `*` and can be restricted with `PATHSNAP_CORS_ORIGINS`; only GET and POST are allowed.

Real `GET /health` output captured from the running service (one development machine, `student_test` partition, not the evaluator run):

```json
{"status":"ok","model_loaded":true,"model_name":"LightGBM (exported)","version":"1.0.0","partition":"student_test","trips_available":10}
```

## 7.3 How ground truth stays on the server

1. The UI sends only `trip_id` and the API never returns `.route` contents or labels. The response carries predicted geometry, per-observation predictions and metrics only.
2. Prediction uses `pathsnap.pipeline.make_components`, which loads each trip with `with_labels=False` (`src/pathsnap/pipeline.py`); `predict` reads only `.track`, `.nodes` and `.arcs` (README, "Predict a route").
3. `.route` is opened in one place, `score_after_prediction` in `src/pathsnap/scoring.py` (`load_trip(trip_dir, with_labels=True)`), which `/predict` calls only after `predict_trip` has returned. `api.py` itself checks only that the `.route` file exists (`is_file()`) to decide whether metrics can be computed; it does not read it.
4. `/trips` loads trips with `with_labels=False`.
5. The reserved evaluator trips are not stored in the repository (README, "Install").

[MISSING: a team-written statement of how this was verified beyond reading the code, e.g. a test or a run where the `.route` file is absent; `/predict` then returns `metrics: null` by construction in `api.py`.]

## 7.4 Partition handling (fails loudly)

* Default: with no environment variable the service serves the 10 `student_test` trips from `PATHSNAP_DATA` (default dataset root) and reports `partition: "student_test"`.
* `PATHSNAP_EVALUATOR_DIR` set: the service serves that folder and reports `partition: "evaluator"`. At startup `_validate_evaluator_partition()` raises a `RuntimeError` unless the directory exists, contains exactly 10 trip folders, and each folder has `<id>.track`, `<id>.nodes` and `<id>.arcs`. It never silently falls back to development data. `/health` re-checks the partition on every call and returns 503 if the folder count changed.
* Startup runs a warm-up: loads the model and runs one full ground-truth-free prediction on the first trip so the first evaluator request is not slow.
* On the UI side the partition is displayed in the sidebar connection status and the model name as a chip at the top of the page; checking that it says `evaluator` before the demo is a manual step (`docs/DEMO_CHECKLIST.md`).

## 7.5 Logging

`src/pathsnap/api.py` logs to the console and to a file (default `logs/api.log`, override with `PATHSNAP_LOG_FILE`). Every successful `POST /predict` writes one line of this form (format string in the source):

```
<time> INFO [<UTC timestamp>] POST /predict trip_id=<id> | obs=<n> | clf_top1=<pct> | hmm_top1=<pct> | continuity=<pct> | breaks=<int> | total=<s>s
```

uvicorn additionally prints `GET /health`, `GET /trips` and `POST /predict` access lines with HTTP status. Failures are logged with a stack trace via `logger.exception`. For trips without reference routes, the line reports `metrics=n/a` instead of label-derived scores.

## 7.6 Timings

`results/api_student_test.csv` (produced by `python scripts/evaluate_api.py`, `student_test`, frozen decoder; the `seconds` column is the server-side total per request, measured on the development laptop):

| Trip | Observations | seconds |
|---|---:|---:|
| 00000000 | 2503 | 2.44 |
| 00000007 | 1354 | 0.85 |
| 00000009 | 1049 | 3.05 |
| 00000040 | 2995 | 3.57 |
| 00000052 | 262 | 0.73 |
| 00000056 | 3958 | 15.36 |
| 00000065 | 768 | 3.19 |
| 00000076 | 1486 | 6.88 |
| 00000077 | 3911 | 6.35 |
| 00000091 | 874 | 2.58 |

The slowest trip is 00000056 at 15.36 s, under the 30 s target in the guidelines. Timings depend on the machine and on cache warmth; re-measure on the demo laptop. `scripts/evaluate_api.py` also checks that the API's pooled Top-1 equals the offline pipeline's (results/hmm_tuning/final_student_test_check.csv); the difference in this run was 0.0000 %.

## 7.7 Running the two-laptop system (README, "Deployment: two laptops")

Both laptops on the same Wi-Fi or hotspot. The evaluator trips stay on Laptop 1.

Laptop 1, from the repository root (keep the window visible; it prints one line per request):

```
pip install -r requirements.txt
pip install -e .
set PATHSNAP_EVALUATOR_DIR=D:\path\to\evaluator_trips      (omit to serve the 10 student_test trips)
uvicorn pathsnap.api:app --app-dir src --host 0.0.0.0 --port 8000
```

Allow inbound TCP 8000 in the firewall, then open `http://<laptop1-ip>:8000/health` from Laptop 2.

Laptop 2:

```
pip install -r ui/requirements.txt
cd ui
run_ui.bat http://<laptop1-ip>:8000          (macOS/Linux: ./run_ui.sh http://<laptop1-ip>:8000)
```

The browser opens `http://localhost:8501`. The UI shows, top to bottom: a header with the model-name chip; a sidebar with the API address field, a **Check connection** button and the connection status (latency, version, partition); a trip dropdown (trip id and GPS-point count) with a **Match route** button; a KPI card (HMM Top-1 accuracy, route continuity with break count, sequence F1, server time, each against the classifier alone where applicable); the Folium map (section 6); then the tabs **Classifier vs HMM**, **Timings**, **Observations**, **API response**; and a **Download map (HTML)** button. If Laptop 1 is unreachable, the UI shows "Backend unreachable" or an error with three checks to perform.

Rehearsal and demo steps: `docs/DEMO_CHECKLIST.md`.

## 7.8 Screenshots to take (all [MISSING], to be captured on the real two-laptop setup)

| Needed | What it must show |
| --- | --- |
| UI home | Laptop 2 browser: model chip at the top, sidebar showing `Partition: evaluator`, trip dropdown |
| UI result | After **Match route** on an evaluator trip: KPI card, map with legend and layer control, Classifier vs HMM table |
| API console | CMD of Laptop 1 showing the `POST /predict trip_id=...` log lines with HTTP 200 access lines, and the `GET /health` output |
| Startup | Laptop 1 CMD at startup, showing the evaluator partition check passing (no error) |

Be explicit in every caption about which partition and which machine setup the screenshot came from. Screenshots taken on a single machine against `student_test` are rehearsal evidence only and must not be presented as the two-laptop evaluator run.

## 7.9 Checks before the report is final

* [MISSING: confirmation that the code demonstrated at evaluation is the committed code (commit hash to quote).]
* [MISSING: observed latency on the real two-laptop network (the UI shows "Server time" and "round trip" for every run).]
