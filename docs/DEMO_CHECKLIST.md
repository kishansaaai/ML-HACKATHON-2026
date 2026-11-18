# PathSnap Two-Laptop Demo & Rehearsal Checklist (Person D + Person C)

Use this checklist before and during the live final evaluation.

---

## 0. First Step Before Starting the Evaluation

- [ ] **Confirm the banner shows the real model name (not MOCK) before the evaluator starts.**
  - The top banner on Laptop 2 must be **green**, showing the real `model_name` from `artifacts/model_card.json` (e.g. `lightgbm`) and the evaluator partition.
  - If the red **`DEV MOCK BACKEND - NOT FOR FINAL EVALUATION`** banner is visible, stop immediately and point Laptop 2 at Laptop 1's real FastAPI server running on the evaluator partition.

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
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
3. Confirm startup logs show:
   - Trained LightGBM model (matching `artifacts/model_card.json`), feature order, and HMM configuration loaded cleanly
   - All 10 reserved evaluator trips detected

### Step 2 — Verify Reachability from Laptop 2 (Person D)
1. On **Laptop 2**, open a browser or terminal and check:
   ```text
   http://<LAPTOP_1_IP>:8000/health
   ```
2. Confirm HTTP `200 OK` with `"model_loaded": true`, real `model_name` (not MOCK), evaluator `partition`, and `"trips_available": 10`.

### Step 3 — Start Evaluator UI on Laptop 2 (Person D)
1. Run the launcher pointing at Laptop 1:
   ```powershell
   cd ui
   run_ui.bat http://<LAPTOP_1_IP>:8000
   ```
2. Confirm the browser opens `http://localhost:8501` and displays:
   - **Green backend banner** in the page header with the real `model_name` and evaluator `partition` (not the red MOCK banner)
   - All **10 evaluation trip cards** loaded dynamically from `GET /trips`

---

## 3. Rehearsal Verification (Using `student_test` Before Final Eval)

- [ ] Run **all 10 `student_test` trips** end-to-end from Laptop 2 → Laptop 1. (Note: on `student_test`, the red non-evaluator safety banner is expected until Laptop 1 is switched to the evaluator partition.)
- [ ] Verify every trip finishes cleanly (no timeout, no 500 error) and renders:
  - Raw GPS observations (blue dots)
  - HMM / Viterbi predicted route drawn as separate polylines across breaks (`hmm_route_segments`) with route-break count shown next to the map
  - Classifier-only predictions toggle (orange dashed layer in LayerControl)
  - Classifier-only vs. HMM comparison metrics & stage timings returned by the API
  - No skipped-coordinate warnings (confirming `[lon, lat]` → `[lat, lon]` conversion is working)
- [ ] Verify Laptop 1 console prints a fresh log entry for every single `POST /predict` request.
- [ ] Verify **zero `.route` data** appears in the UI or in the `Inspect Returned Observations & API JSON` expander.
- [ ] Test error handling once: temporarily stop Laptop 1 or enter a wrong IP and confirm Laptop 2 shows a clean, readable error box (not a raw Python traceback).

---

## 4. Live Evaluator Walkthrough Script

1. **Confirm & show the green header banner on Laptop 2:** Point out the real `model_name` and evaluator `partition`, Laptop 1's IP in the sidebar, and the 10 evaluation trip cards loaded via `GET /trips`.
2. **Let the evaluator pick ANY trip:** Click the evaluator's chosen trip card and press **Run Map-Matching**.
3. **Show Laptop 1 (CMD/Console):** Immediately point to the Laptop 1 terminal receiving `POST /predict` with `trip_id`, running the pipeline live, and printing stage timings and server-scored metrics.
4. **Inspect the returned result on Laptop 2:**
   - Review the **Classifier-Only vs. Classifier + HMM** metrics and route-break count displayed beside the map.
   - Zoom into the Folium map and toggle **Classifier-Only Predictions** vs. **HMM / Viterbi Predicted Route** to compare both layers visually.
   - (Optional) Open the **Inspect Returned Observations & API JSON** expander to show that ground-truth `.route` labels are never sent to Laptop 2.
5. **Offer to run additional trips** chosen freely by the evaluator to confirm all 10 trips run live.

---

## 5. Screenshots to Capture for Report Section 7

Capture and save in `docs/screenshots/` for the consolidated PDF report (using the real backend, not `mock_api.py`):
1. `ui_trip_selector.png` — Laptop 2 UI showing the green backend status banner, sidebar connection info, and the 10 dynamically loaded evaluation trip cards.
2. `ui_route_map_and_metrics.png` — Completed prediction showing KPI metrics, Classifier vs. HMM comparison table, route-break count, and interactive Folium map.
3. `ui_layer_comparison.png` — Zoomed-in Folium view comparing Raw GPS, Classifier-Only (dashed orange), and HMM Predicted Route (green).
4. `laptop1_api_cmd_logs.png` — Laptop 1 CMD/PowerShell terminal showing live `GET /health`, `GET /trips`, and `POST /predict` logs with HTTP `200 OK` responses and timings (supplied with Person C).
