# PathSnap Two-Laptop Demo & Rehearsal Checklist (Person D + Person C)

Use this checklist before and during the live final evaluation.

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
   - Trained model, scaler, feature order, and HMM config loaded cleanly
   - All 10 reserved evaluator trips detected

### Step 2 — Verify Reachability from Laptop 2 (Person D)
1. On **Laptop 2**, open a browser or terminal and check:
   ```text
   http://<LAPTOP_1_IP>:8000/health
   ```
2. Confirm HTTP `200 OK` with `"model_loaded": true` and `"trips_available": 10`.

### Step 3 — Start Evaluator UI on Laptop 2 (Person D)
1. Run the launcher pointing at Laptop 1:
   ```powershell
   cd ui
   run_ui.bat http://<LAPTOP_1_IP>:8000
   ```
2. Confirm the browser opens `http://localhost:8501` and displays:
   - Green **Connected** status badge in the sidebar with model name & version
   - All **10 evaluation trip cards** loaded dynamically from `GET /trips`

---

## 3. Rehearsal Verification (Using `student_test` Before Final Eval)

- [ ] Run **all 10 `student_test` trips** end-to-end from Laptop 2 → Laptop 1.
- [ ] Verify every trip finishes cleanly (no timeout, no 500 error) and renders:
  - Raw GPS observations (blue dots)
  - HMM / Viterbi predicted route (green line)
  - Classifier-only predictions toggle (orange dashed layer in LayerControl)
  - Classifier-only vs. HMM comparison metrics & stage timings
- [ ] Verify Laptop 1 console prints a fresh log entry for every single `POST /predict` request.
- [ ] Verify **zero `.route` data** appears in the UI or in the `Expand -> API Response Summary` inspector.
- [ ] Test error handling once: temporarily stop Laptop 1 or enter a wrong IP and confirm Laptop 2 shows a clean, readable error box (not a raw Python traceback).

---

## 4. Live Evaluator Walkthrough Script

1. **Show Laptop 2 (UI):** Point out the sidebar showing Laptop 1's IP, green `Connected` badge, and the 10 evaluation trip cards loaded via `GET /trips`.
2. **Let the evaluator pick ANY trip:** Click the evaluator's chosen trip card and press **Run Map-Matching**.
3. **Show Laptop 1 (CMD/Console):** Immediately point to the Laptop 1 terminal receiving `POST /predict` with `trip_id`, running the pipeline live, and printing stage timings + accuracy.
4. **Inspect the returned result on Laptop 2:**
   - Show the **HMM Top-1 Accuracy** and **Route Continuity** gains over Classifier-Only.
   - Zoom into the Folium map and toggle **Classifier-Only Predictions** vs. **HMM / Viterbi Predicted Route** to visually show how HMM eliminates parallel-street jumps and discontinuities.
   - (Optional) Open the **Inspect Returned Observations & API JSON** expander to prove ground-truth `.route` labels are never sent to Laptop 2.
5. **Offer to run another trip** (or several) chosen freely by the evaluator to confirm all 10 trips work live.

---

## 5. Screenshots to Capture for Report Section 7

Capture and save in `docs/screenshots/` for the consolidated PDF report:
1. `ui_trip_selector.png` — Laptop 2 UI showing the 10 evaluation trip cards and green `Connected` sidebar.
2. `ui_route_map_and_metrics.png` — Completed prediction showing KPI metrics, Classifier vs. HMM comparison table, and interactive Folium map.
3. `ui_layer_comparison.png` — Zoomed-in Folium view comparing Raw GPS, Classifier-Only (dashed orange), and HMM Predicted Route (green).
4. `laptop1_api_cmd_logs.png` — Laptop 1 CMD/PowerShell terminal showing live `GET /health`, `GET /trips`, and `POST /predict` logs with HTTP `200 OK` responses and timings (supplied with Person C).
