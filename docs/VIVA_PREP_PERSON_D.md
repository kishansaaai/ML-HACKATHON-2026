# Viva Prep Notes — Person D (UI, Demo & High-Level Pipeline)

Every team member must be able to explain their own work **and** the end-to-end PathSnap pipeline during viva. Use these concise talking points.

---

## 1. Explaining Your Own Component (UI & Distributed Architecture)

- **Why two laptops?**
  - Enforces clean client-server separation: **Laptop 1** hosts the trained model, road networks, HMM/Viterbi decoder, and ground-truth `.route` files; **Laptop 2** runs only a thin presentation UI.
- **How do you prevent data leakage / hardcoding in the UI?**
  - The UI (`ui/app.py`, `ui/api_client.py`) has **no access** to dataset folders or `.route` files and performs zero ML inference.
  - It sends only `{"trip_id": "<id>"}` to `POST /predict`. Clicking **Run** always triggers a fresh live API request (never cached by trip ID), visible immediately in Laptop 1's CMD logs.
  - The API never returns `.route` ground truth—only predicted routes and server-computed metrics.
- **How does the UI guard against accidentally demoing a mock or dev backend?**
  - On load (and refresh), the UI calls `GET /health` and displays `model_name` (read server-side from `artifacts/model_card.json`, key `"model"`) and `partition` in the page header.
  - If `model_name` contains `"MOCK"` (case-insensitive) or `partition` is not the evaluator partition, the header shows a prominent red **`DEV MOCK BACKEND - NOT FOR FINAL EVALUATION`** banner; only a real model on the evaluator partition shows the green ready banner.
- **How are route breaks and coordinate order handled on the map?**
  - The API contract uses `[lat, lon]` for every coordinate (the backend converts the decoder's internal `[lon, lat]` output to `[lat, lon]`).
  - `ui/map_renderer.py` validates every coordinate (`lat` in `[-90, 90]`, `lon` in `[-180, 180]`, finite numbers), skips invalid points, and triggers a UI warning if any point is skipped so a swapped `[lon, lat]` bug cannot silently misplace a route.
  - When the decoder splits a disconnected route into `hmm_route_segments`, `map_renderer.py` draws each segment as its own `folium.PolyLine` (never connecting separate segments with false straight lines) and falls back to `hmm_route_polyline` only if `hmm_route_segments` is absent. The route-break count (`metrics.hmm.continuity_breaks`) is displayed right beside the map.
- **Why Streamlit + Folium?**
  - Matches the project's Python + Folium stack, renders interactive multi-layer maps (`Raw GPS`, `HMM Route`, `Classifier-Only`) on `OpenStreetMap` tiles with `LayerControl`, legends, and `prefer_canvas=True`.
  - `st_folium(..., returned_objects=[])` avoids redundant Streamlit reruns while panning or zooming.
- **How is cross-laptop networking configured?**
  - Laptop 1 binds FastAPI to `0.0.0.0:8000` with CORS enabled.
  - Laptop 2 configures `API_BASE_URL` in one place (`ui/config.py`, `PATHSNAP_API_URL` env var, or the sidebar field).

---

## 2. High-Level Pipeline Overview (Shadowing Person B & Person A)

Be ready to explain what happens on Laptop 1 after Laptop 2 sends `trip_id`:

1. **Local Coordinate Projection (Person B):**
   - Raw `.track` and `.nodes` are in `(longitude, latitude)` degrees. Degree-Euclidean distance is distorted because 1° of longitude shrinks by $\cos(\text{lat})$.
   - Each trip carries its own road graph in its own geographic region, so each trip is projected to local metres.
2. **Spatial Indexing & Candidate Generation (Person B + Person A):**
   - Each arc is indexed in a spatial index (`STRtree` / KD-tree). For each GPS point, candidate arcs within radius $R$ metres (with nearest-$K$ fallback) are retrieved. We track **Candidate Miss Rate** because missing the true arc caps downstream accuracy.
3. **Training Labels & Feature Engineering (Person A):**
   - `.route` lists traversed road arcs (fewer entries than 1 Hz GPS points), so per-point labels are derived via monotone dynamic-programming alignment along `.route`.
   - Features per `(observation, candidate)` include geometric and motion features (such as perpendicular distance, projection fraction, heading alignment, speed, arc length, candidate distance rank/gap, candidate count, and node degrees).
4. **Model Selection & Deployed Classifier (Person A + Person C):**
   - Multiple models across distinct algorithm families are trained and compared using **trip-grouped CV** (never splitting one trip across folds) and `student_test`.
   - The deployed model is plain **LightGBM** (tree-based, requiring no feature standardization), with its metadata recorded in `artifacts/model_card.json`.
5. **HMM & Viterbi Route Reconstruction (Person B):**
   - Independent classifier predictions can jump between disconnected parallel streets.
   - **Hidden states:** candidate arcs at each GPS step.
   - **Emission probability:** classifier probability for each candidate arc.
   - **Transition probability:** penalizes mismatches between network shortest-path distance (computed via distance-bounded cached Dijkstra on the trip's sparse graph) and straight-line GPS displacement.
   - **Viterbi decoding** runs in log-space, assigns a finite penalty to unreachable pairs, and splits/restarts segments across large GPS gaps or disconnections—producing the separate segments in `hmm_route_segments`.
6. **Server-Side Evaluation (Person C):**
   - Only after `predict_trip` finishes does `scoring.py` read `.route` to compute Observation Top-1 Accuracy, Macro F1, Route Continuity, Continuity Breaks, and Sequence F1.
