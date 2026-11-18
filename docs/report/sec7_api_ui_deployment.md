# 7. API, UI and Deployment

## 7.1 Architecture

**Laptop 1** runs the FastAPI server (`src/pathsnap/api.py`): road data, the LightGBM model, the HMM decoder and server-side scoring. **Laptop 2** runs the Streamlit UI (`ui/`), which sends only a `trip_id` and draws what comes back. The reference route never leaves Laptop 1.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | status, `model_loaded`, `model_name`, version, `partition`, number of trips available |
| `GET /trips` | the partition name and its trips with counts of observations, nodes and arcs |
| `POST /predict` `{"trip_id": ...}` | runs candidates, classifier and HMM; returns raw GPS, HMM and classifier route segments, per-observation arc IDs and confidence, classifier-only and HMM metrics, per-stage timings |

The full field list is in `docs/API_CONTRACT.md`.

## 7.2 Screenshots of the working UI

![Figure 9. The evaluator UI after Match route on student_test trip 00000052: connection status and partition in the sidebar, model name, trip selector, headline metrics (HMM Top-1, route continuity, sequence F1, server time with the difference from the classifier alone) and the Folium route map with legend and layer control.](docs/report/fig/ui_match_route.png)

The screenshot was taken from the running UI of this repository (`ui/app.py`) connected to the running API on the development partition student_test; for evaluation the API reads the evaluator folder instead and the UI shows that partition name.

## 7.3 API responses and console log (receiving laptop)

Real responses of the running API (student_test partition, `curl`/`requests`, abbreviated where noted):

```
GET /health
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "LightGBM (exported)",
  "version": "1.0.0",
  "partition": "student_test",
  "trips_available": 10
}

GET /trips  (partition=student_test, 10 trips; first two shown)
{
  "partition": "student_test",
  "trips": [
    {
      "trip_id": "00000000",
      "num_observations": 2503,
      "num_nodes": 8542,
      "num_arcs": 18196
    },
    {
      "trip_id": "00000007",
      "num_observations": 1354,
      "num_nodes": 2061,
      "num_arcs": 4369
    }
  ]
}

POST /predict {"trip_id": "00000052"}  -> HTTP 200 (summary of the response)
{
  "trip_id": "00000052",
  "num_observations": 262,
  "metrics": {
    "classifier_only": {
      "top1_accuracy": 0.9733,
      "macro_f1": 0.902,
      "route_continuity": 0.9349,
      "continuity_breaks": 17
    },
    "hmm": {
      "top1_accuracy": 0.9809,
      "macro_f1": 0.9174,
      "route_continuity": 0.9579,
      "continuity_breaks": 11
    }
  },
  "timings_sec": {
    "load_and_index": 0.028,
    "candidates_and_features": 0.031,
    "classifier_inference": 0.076,
    "hmm_viterbi": 0.165,
    "server_scoring": 0.094,
    "total": 0.595
  },
  "fields": [
    "classifier_route_segments",
    "hmm_route_polyline",
    "hmm_route_segments",
    "metrics",
    "model_name",
    "num_observations",
    "observations",
    "raw_gps",
    "server_timestamp",
    "timings_sec",
    "trip_id"
  ]
}
```

![Figure 10. Screenshot of the CMD console of the API process on the receiving laptop (Laptop 1) while it serves student_test requests: server start-up, `GET /health`, `GET /trips` and one log line per `POST /predict` with trip, observation count, classifier and HMM Top-1, continuity and total time. Lines from 127.0.0.1 are development requests sent from the same laptop through the UI; the health check and the last `POST /predict` (trip 00000052) come from the second laptop, 10.222.216.186.](docs/report/fig/api_cmd_console.png)

The same log lines as text (also written to `logs/api.log`):

```
[2026-11-18T16:18:09Z] POST /predict trip_id=00000052 | obs=262 | clf_top1=97.3% | hmm_top1=98.1% | continuity=95.8% | total=1.30s
[2026-11-18T16:18:18Z] POST /predict trip_id=00000052 | obs=262 | clf_top1=97.3% | hmm_top1=98.1% | continuity=95.8% | total=0.21s
[2026-11-18T16:18:24Z] POST /predict trip_id=00000009 | obs=1049 | clf_top1=88.5% | hmm_top1=89.3% | continuity=93.9% | total=2.49s
[2026-11-18T16:18:36Z] POST /predict trip_id=00000077 | obs=3911 | clf_top1=78.3% | hmm_top1=77.8% | continuity=98.2% | total=5.29s
[2026-11-18T16:18:46Z] POST /predict trip_id=00000000 | obs=2503 | clf_top1=95.3% | hmm_top1=95.6% | continuity=99.2% | total=2.43s
[2026-11-18T16:19:05Z] POST /predict trip_id=00000091 | obs=874 | clf_top1=91.2% | hmm_top1=91.0% | continuity=95.3% | total=2.04s
[2026-11-18T16:20:36Z] POST /predict trip_id=00000091 | obs=874 | clf_top1=91.2% | hmm_top1=91.0% | continuity=95.3% | total=4.33s
[2026-11-18T16:21:06Z] POST /predict trip_id=00000052 | obs=262 | clf_top1=97.3% | hmm_top1=98.1% | continuity=95.8% | total=0.94s
```

## 7.4 No ground-truth leakage

- Prediction (`pathsnap.pipeline.predict`) loads trips with `with_labels=False` and cannot read `.route`.
- `src/pathsnap/scoring.py` opens `.route` only after the prediction is finished, to compute the metrics; the response contains no reference-route field.
- A test runs `/predict` on a trip copy without a `.route` file; it still returns a prediction and `metrics: null` (`tests/test_api_integration.py`).
- The evaluator folder is set with `PATHSNAP_EVALUATOR_DIR`; startup fails unless it holds exactly 10 valid trip folders, so the server never silently falls back to development data. The evaluator trips are not in the repository (`.gitignore`).
- Nothing is hard-coded: every trip goes through the same code path.

## 7.5 Timings

`python scripts/evaluate_api.py` runs all 10 student-test trips through the API and checks that the pooled Top-1 equals the offline pipeline's (difference 0.0000 % in the last run). Server time per trip on the development laptop:

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

The step-by-step checklist is `docs/DEMO_CHECKLIST.md`.
