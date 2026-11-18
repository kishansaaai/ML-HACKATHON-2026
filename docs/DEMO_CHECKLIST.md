# PathSnap Two-Laptop Demo & Rehearsal Checklist (Person D + Person C)

Use this checklist before and during the live final evaluation.

---

## 0. First Step Before Starting the Evaluation

- [ ] **Confirm the UI is talking to the real backend on the evaluator partition.**
  - The top of the page on Laptop 2 shows `Model LightGBM (exported)` (the name comes from `artifacts/model_card.json`).
  - The sidebar shows `Partition: evaluator` (it shows `student_test` while rehearsing).
  - If the page shows `Backend unreachable`, or the partition is not the evaluator one at the real run, stop and fix Laptop 1 first.

---

## 1. Pre-Demo Network Setup (15 Minutes Before Evaluation)

- [ ] Connect **Laptop 1** (API + Model) and **Laptop 2** (UI) to the **same Wi-Fi network or mobile hotspot** (a dedicated mobile hotspot avoids campus client-isolation issues).
- [ ] On **Laptop 1**, find the local IPv4 address:
  - Windows: `ipconfig` → note `IPv4 Address` (e.g. `192.168.43.10`)
  - macOS/Linux: `ifconfig` or `ip addr`
- [ ] On **Laptop 1**, allow inbound TCP port `8000` in Windows Defender Firewall (or verify Laptop 2 can reach `http://<LAPTOP_1_IP>:8000/health`).
- [ ] Verify the **10 reserved evaluator trips** are placed ONLY on **Laptop 1** in the server-side evaluation folder (never copied to Laptop 2, never committed to GitHub).

---

## 2. Exact Startup Order

### Step 1 — Start FastAPI Backend on Laptop 1 (Person C)
1. Open a visible Command Prompt / PowerShell window on **Laptop 1** (keep it maximized or side-by-side so the evaluator can watch live logs).
2. Run the production FastAPI server bound to `0.0.0.0`:
   ```powershell
   set PATHSNAP_EVALUATOR_DIR=<folder with the 10 evaluator trips>   (omit while rehearsing on student_test)
   uvicorn pathsnap.api:app --app-dir src --host 0.0.0.0 --port 8000
   ```
3. Confirm startup logs show:
   - Trained model and frozen HMM configuration loaded cleanly; startup prints the display model name and loaded decoder settings
   - All 10 reserved evaluator trips detected

### Step 2 — Verify Reachability from Laptop 2 (Person D)
1. On **Laptop 2**, open a browser or terminal and check:
   ```text
   http://<LAPTOP_1_IP>:8000/health
   ```
2. Confirm HTTP `200 OK` with `"model_loaded": true`, real `model_name`, the right `partition`, and `"trips_available": 10`.

### Step 3 — Start Evaluator UI on Laptop 2 (Person D)
1. Run the launcher pointing at Laptop 1:
   ```powershell
   cd ui
   run_ui.bat http://<LAPTOP_1_IP>:8000
   ```
2. Confirm the browser opens `http://localhost:8501` and displays:
   - The real model name at the top and the partition in the sidebar
   - All **10 evaluation trips** in the trip dropdown, loaded dynamically from `GET /trips`

---

## 3. Rehearsal Verification (Using `student_test` Before Final Eval)

- [ ] Run **all 10 `student_test` trips** end-to-end from Laptop 2 → Laptop 1. (On `student_test` the sidebar partition reads `student_test` until Laptop 1 is switched to the evaluator folder.)
- [ ] Verify every trip finishes cleanly (no timeout, no 500 error) and renders:
  - Raw GPS observations (blue dots)
  - HMM / Viterbi predicted route drawn as separate polylines across breaks (`hmm_route_segments`) with route-break count shown next to the map
  - Classifier-only predictions toggle (orange dashed layer in LayerControl)
  - Classifier-only vs. HMM comparison metrics & stage timings returned by the API
  - No skipped-coordinate warnings (confirming `[lon, lat]` → `[lat, lon]` conversion is working)
- [ ] Verify Laptop 1 console prints a fresh log entry for every single `POST /predict` request.
- [ ] Verify **zero `.route` data** appears in the UI or in the **Observations** and **API response** tabs.
- [ ] Test error handling once: temporarily stop Laptop 1 or enter a wrong IP and confirm Laptop 2 shows a clean, readable error box (not a raw Python traceback).

---

## 4. Live Evaluator Walkthrough Script

1. **Show the header on Laptop 2:** point out the real model name, the evaluator partition and Laptop 1's address in the sidebar, and the 10 trips in the dropdown loaded via `GET /trips`.
2. **Let the evaluator pick ANY trip:** Select the evaluator's chosen trip in the dropdown and press **Match route**.
3. **Show Laptop 1 (CMD/Console):** Immediately point to the Laptop 1 terminal receiving `POST /predict` with `trip_id`, running the pipeline live, and printing stage timings and server-scored metrics.
4. **Inspect the returned result on Laptop 2:**
   - Review the **Classifier-Only vs. Classifier + HMM** metrics and route-break count displayed beside the map.
   - Zoom into the Folium map and toggle **Classifier-Only Predictions** vs. **HMM / Viterbi Predicted Route** to compare both layers visually.
   - (Optional) Open the **Observations** and **API response** tabs to show that ground-truth `.route` labels are never sent to Laptop 2.
5. **Offer to run additional trips** chosen freely by the evaluator to confirm all 10 trips run live.

---

## 5. Screenshots to Capture for Report Section 7

Capture and save in `docs/screenshots/` for the consolidated PDF report (using the real backend, not `mock_api.py`):
1. `ui_trip_selector.png` — Laptop 2 UI showing the model name, the sidebar connection info and partition, and the trip dropdown with the 10 evaluation trips.
2. `ui_route_map_and_metrics.png` — Completed prediction showing KPI metrics, the Classifier vs HMM tab, route-break count, and interactive Folium map.
3. `ui_layer_comparison.png` — Zoomed-in Folium view comparing Raw GPS, Classifier-Only (dashed orange), and HMM Predicted Route (green).
4. `laptop1_api_cmd_logs.png` — Laptop 1 CMD/PowerShell terminal showing live `GET /health`, `GET /trips`, and `POST /predict` logs with HTTP `200 OK` responses and timings (supplied with Person C).
