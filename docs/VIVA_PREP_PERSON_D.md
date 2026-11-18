# Viva Prep Notes — Person D (UI, Demo & High-Level Pipeline)

Every team member must be able to explain their own work **and** the end-to-end PathSnap pipeline during viva. Use these concise talking points.

---

## 1. Explaining Your Own Component (UI & Distributed Architecture)

- **Why two laptops?**
  - Enforces clean client-server separation: **Laptop 1** hosts the trained model, road networks, HMM/Viterbi decoder, and ground-truth `.route` files; **Laptop 2** runs only a thin presentation UI.
- **How do you prevent data leakage / hardcoding in the UI?**
  - The UI (`ui/app.py`, `ui/api_client.py`) has **no access** to dataset folders or `.route` files.
  - It sends only `{"trip_id": "<id>"}` to `POST /predict`. Clicking **Run** always triggers a fresh live API request (never cached by trip ID), visible immediately in Laptop 1's CMD logs.
  - The API never returns `.route` ground truth—only predicted routes and server-computed metrics.
- **Why Streamlit + Folium?**
  - Matches the hackathon's Python + Folium stack, renders interactive multi-layer maps (`Raw GPS`, `HMM Route`, `Classifier-Only`) with `LayerControl`, legends, and `prefer_canvas=True` for smooth performance on trips with thousands of GPS points.
  - `st_folium(..., returned_objects=[])` avoids redundant Streamlit reruns while panning/zooming.
- **How is cross-laptop networking configured?**
  - Laptop 1 binds FastAPI to `0.0.0.0:8000` with CORS enabled.
  - Laptop 2 configures `API_BASE_URL` in one place (`ui/config.py`, `PATHSNAP_API_URL` env var, or the sidebar field).

---

## 2. High-Level Pipeline Overview (Shadowing Person B & Person A)

Be ready to explain what happens on Laptop 1 after Laptop 2 sends `trip_id`:

1. **Local Coordinate Projection (Person B):**
   - Raw `.track` and `.nodes` are in `(longitude, latitude)` degrees. Degree-Euclidean distance is distorted because 1° of longitude shrinks by $\cos(\text{lat})$.
   - Each trip sits in its own region and has its own road graph (up to 155k nodes / 307k arcs), so each trip is projected to local metres (UTM or per-trip equirectangular projection).
2. **Spatial Indexing & Candidate Generation (Person B + Person A):**
   - Each arc is indexed in a spatial index (`STRtree` / KD-tree). For each GPS point, candidate arcs within radius $R$ metres (with nearest-$K$ fallback) are retrieved. We track **Candidate Miss Rate** because missing the true arc caps downstream accuracy.
3. **Training Labels & Feature Engineering (Person A):**
   - `.route` lists traversed road arcs (far fewer than 1 Hz GPS points), so per-point labels are derived via monotone DP alignment along `.route`.
   - Features per `(observation, candidate)` include perpendicular distance, fractional projection along the arc, heading alignment vs. arc bearing, speed, arc length, rank/gap to nearest candidate, candidate count, and node degrees. Preprocessing is fitted on the 80 training trips only.
4. **Model Selection & Bayesian Tuning (Person A + Person C):**
   - At least 5 models across distinct families are compared using **trip-grouped CV** (never splitting one trip across folds) and `student_test`.
   - The best model is tuned with Bayesian optimisation (e.g., Optuna TPE) on the 80 training trips.
5. **HMM & Viterbi Route Reconstruction (Person B):**
   - Independent classifier predictions can jump between disconnected parallel streets.
   - **Hidden states:** candidate arcs at each GPS step.
   - **Emission probability:** calibrated classifier probability $P(\text{arc} \mid \text{features})$.
   - **Transition probability:** Newson–Krumm style exponential decay on $|\text{network shortest-path distance} - \text{great-circle/straight-line GPS distance}|$, computed via distance-bounded cached Dijkstra on the trip's sparse graph.
   - **Viterbi decoding** runs in log-space to avoid underflow, assigns a finite penalty to unreachable transitions, and restarts segments across large GPS time gaps (some trips have gaps > 5 s, up to 42+ hours in `train`).
6. **Server-Side Evaluation (Person C):**
   - Only after `predict_trip` returns the predicted arcs and stitched route does `scoring.py` read `.route` to compute Observation Top-1 Accuracy, Macro F1, Route Continuity, Continuity Breaks, and Sequence F1.
