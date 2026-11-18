# ML Hackathon 2026 – Team Plan & PRD

PathSnap must turn noisy 1 Hz GPS tracks into the sequence of road arcs the vehicle really drove, using a trained classifier plus HMM/Viterbi decoding, served by a FastAPI backend on Laptop 1 and shown in a UI on Laptop 2. This plan splits the work across four people: two own the hard modelling and decoding, one (C) owns data, maps, simple models, API, scoring, deployment and report, and one (D) owns a deliberately small evaluator UI and demo support.

## 1. Problem and goals

Raw GPS drifts onto parallel streets or jumps across buildings. We snap each observation to a road arc, then use road connectivity to make the whole route coherent.

**What the data looks like** (checked from the files in `Student folder`):

- 80 trips in `train`, 10 in `student_test`. The 10 evaluator trips are withheld and arrive only at final evaluation.
- Each trip folder has `.track` (lon, lat, time), `.nodes` (lon, lat), `.arcs` (from_node, to_node; row index = arc_id) and `.route` (ground-truth arc_id sequence). No headers.
- Every trip carries its **own road network**, and trips sit in different regions (e.g. lon≈8.6/lat≈49.8 vs lon≈55/lat≈56). Train trip 1 has 88,809 nodes and 165,790 arcs; student trip 0 has 8,542 nodes and 18,196 arcs. So: one graph per trip, never one global graph.
- A route lists far fewer arcs than there are GPS points (about 525 arcs for 1,315 points in train trip 1). Per-observation labels do not exist and must be derived from the route.
- Coordinates are in degrees, so distances need a local projected system (UTM or a per-trip equirectangular projection), not degree-Euclidean.

**Goals**

| # | Goal | How we know it is met |
| --- | --- | --- |
| G1 | Per-observation arc prediction that beats nearest-road snapping | Top-1 accuracy on `student_test` above the nearest-arc baseline |
| G2 | HMM/Viterbi improves on classifier-only output | Higher accuracy and route continuity on `student_test`, with cases where it fails explained |
| G3 | Works live on all 10 evaluator trips through the API | Each trip returns a route and server-side score; API logs visible on Laptop 1 |
| G4 | Fully reproducible, honest submission | Public GitHub repo with README; PDF whose numbers match repo outputs |
| G5 | Every member can defend every part | Each person walks a teammate through their code before evaluation |

**Hard rules from the guidelines** (violating any costs marks or disqualifies):

- Train and compare at least 5 models, documented with configs, seeds, training and inference time.
- Bayesian hyperparameter optimisation with group-aware CV (group = trip) on the 80 training trips; keep every trial.
- Evaluator data and labels never touch training, feature selection, thresholds or tuning. Preprocessing is fitted on training data only.
- Ground truth stays server-side. No hardcoded outputs. The UI sends only a trip ID.
- No sharing of code, weights or configs with other teams. Disclose AI assistance. No fabricated metrics.
- Deliverables: one public GitHub repo and one consolidated PDF report.

## 2. Product requirements (PRD)

PathSnap is one pipeline, `predict_trip(trip)`, that takes raw GPS plus a trip's road network and returns a predicted arc for every observation and a connected route. Everything else (training, API, UI, report) exists to build, expose, or prove that function.

### 2.1 Users

| User | What they need |
| --- | --- |
| Evaluator (final demo) | Open the UI on Laptop 2, pick any of 10 trips, see raw GPS vs predicted route and accuracy; see live request logs on Laptop 1 |
| Evaluator (viva) | Hear any member explain data, features, model choice, Bayesian search, HMM and Viterbi, API and UI |
| Team | One reproducible command per stage; a repo that regenerates every number in the PDF |

### 2.2 Functional requirements

Owners are Person A, B (hard track) and C, D (supporting track); see section 3. P0 = must ship, P1 = if time allows.

| ID | Requirement | Pri | Owner |
| --- | --- | --- | --- |
| FR1 | Loader and dataset inventory for all partitions: trip counts, rows per file, missing or invalid points, timestamp gaps, extent, node and arc counts, connectivity | P0 | C |
| FR2 | Per-trip projection to metres, road graph, arc geometry, spatial index | P0 | B |
| FR3 | Candidate generation: arcs within radius R of each GPS point, with a nearest-K fallback | P0 | A |
| FR4 | Feature engineering per (observation, candidate) pair | P0 | A |
| FR5 | Derive per-observation reference arc from `.route` by monotone alignment | P0 | A |
| FR6 | Train and compare at least 5 models with group-aware CV, seeds and logs | P0 | A (heavy models), C (simple models) |
| FR7 | Bayesian hyperparameter optimisation of the chosen model, full trial record | P0 | A |
| FR8 | HMM with Viterbi decoding, gap and disconnection handling | P0 | B |
| FR9 | Metrics module: Top-1, candidate miss rate, macro F1, precision, recall, log loss, route continuity, sequence-level score | P0 | B |
| FR10 | Exported artifacts and a single inference entry point used by the API | P0 | A + B |
| FR11 | FastAPI backend with server-side scoring, request logging | P0 | C |
| FR12 | Simple evaluator UI on Laptop 2: trip list, run button, map, metric panel; only renders the API's JSON | P0 | D |
| FR13 | Folium maps: GPS, road network, reference route, predicted route, layer control, legend | P0 | C |
| FR14 | EDA plots and data-quality record | P0 | C |
| FR15 | Technical report (PDF) in the mandated structure | P0 | C writes, all supply analysis |
| FR16 | Repo layout, README, requirements, run scripts | P0 | C |
| FR17 | Two-laptop setup (IP, firewall, CORS, startup scripts) and integration tests over all 10 `student_test` trips | P0 | C |
| FR18 | UI screenshots, demo rehearsal checklist, report section 7 text | P0 | D |

### 2.3 Modelling approach

The classifier scores candidate arcs; the HMM picks the best connected sequence of them. This is the design we start from; Day-1 experiments may change parameters but not the shape.

```
Laptop 2 · evaluator UI
  Streamlit UI (Person D) · ten trip cards · map · classifier vs HMM metrics
        │ POST /predict (trip_id)            ▲ route + metrics, no labels
        ▼                                    │
Laptop 1 · model and API
  FastAPI (Person C) · /health /trips /predict · logs every request
        │
        ▼
  ┌─ predict_trip sees only .track .nodes .arcs ───────────────────────┐
  │ Candidates ─▶ Features ─▶ Classifier ─▶ HMM / Viterbi              │
  │ (A)           (A)         P(arc correct)  (B)                      │
  └────────────────────────────────────────────────────────────────────┘
        ▲ artifacts (model, scaler, feature order, HMM config)
        │ prediction
        ▼
  Scoring (Person C): reads .route only AFTER prediction, returns accuracy
```

The UI sends only a trip ID. The prediction path never touches `.route`; the scoring module, which does, runs afterwards on Laptop 1 and returns only metrics.

1. **Project and index.** Convert each trip to local metres. Treat each arc as a straight segment between its two nodes. Build a spatial index (STRtree or KD-tree) per trip.
2. **Generate candidates.** For each GPS point take all arcs within R metres (start at 50–100 m, choose R on training trips). If fewer than K arcs fall inside, add nearest ones up to K. Log the candidate miss rate, because a missed reference arc caps every later stage.
3. **Build features** per (point, arc): perpendicular distance, fractional position along the arc, heading difference between the vehicle's motion direction and the arc bearing, speed, arc length, distance rank among candidates, gap to the nearest candidate, candidate count, node degrees at both arc ends, and distance to the previous and next point's projection. Fit any scaler on training trips only.
4. **Label.** A candidate is positive when it is the reference arc for that point. The route gives arcs, not per-point labels, so derive them with a monotone alignment (dynamic programming over route order, minimising point-to-arc distance). Count how often the alignment is ambiguous and report it as a data limitation.
5. **Classify.** The model outputs P(correct | features) per candidate; normalise within each observation. Models that give no probabilities (SVM, KNN) are calibrated or adapted, and the report says how.
6. **Decode.** Hidden states are the candidate arcs at each step, so each step has its own state set. Emission = calibrated classifier probability, optionally multiplied by a Gaussian on perpendicular distance. Transition = Newson–Krumm style: exp(−|network distance between the two candidates − straight-line distance between the two points| / β). Viterbi runs in log space.
7. **Edge cases.** Unreachable pairs get a large finite penalty instead of zero so the decoder never dies. No candidates, or a time gap above a threshold: close the segment, restart Viterbi, and stitch segments afterwards. Cap network search with a distance cutoff; at 165k arcs per trip, uncached shortest paths are the main runtime risk.
8. **Output.** One arc per observation plus a stitched route (shortest path between consecutive distinct arcs), returned next to the classifier-only arcs for comparison.

**Metrics** (the primary selection metric is Top-1 on `student_test`; ties broken by route continuity, then log loss):

| Metric | Definition |
| --- | --- |
| Observation Top-1 accuracy | Highest-scored candidate equals the reference arc, per GPS point |
| Candidate miss rate | Share of points whose reference arc is not in the candidate set |
| Macro F1, precision, recall | Candidate-level, averaged over correct and incorrect classes; confusion matrix kept |
| Log loss | Candidate-level probability quality |
| Route continuity | Share of consecutive predicted arc pairs that are identical or connected head-to-tail; also count of breaks |
| Sequence-level score | Overlap (F1 or Jaccard) of predicted vs reference arc sets, and edit distance of arc sequences |
| Time | Training and per-trip inference seconds |

*Assumption to confirm:* the guidelines do not define route continuity or sequence-level score exactly, so the definitions above are ours. State them in the report.

**Model line-up** (at least five, one per family; every one needs the report's A/B/C questions answered):

| Family | Candidates | Owner |
| --- | --- | --- |
| Linear / probabilistic | Logistic Regression, Gaussian Naive Bayes | C |
| Instance-based | KNN (distance-weighted, scaled) | C |
| Tree ensemble | Random Forest or Extra Trees | A |
| Boosting | HistGradientBoosting, LightGBM or XGBoost | A |
| Neural | Shallow or deep MLP | A |

Bayesian search (Optuna TPE or scikit-optimize) tunes the winner, objective = mean group-CV Top-1 plus log loss tie-break, folds grouped by trip, search space justified in the report.

### 2.4 API and UI

| Endpoint | Behaviour |
| --- | --- |
| `GET /health` | Model and artifacts loaded, version string |
| `GET /trips` | The ten evaluator trip IDs, from a server-side folder |
| `POST /predict` with `{trip_id}` | Runs the full pipeline, then scores server-side. Returns observations with classifier arc and HMM arc, route polyline, metrics, timings. Never returns the reference route |

- The prediction code reads `.track`, `.nodes`, `.arcs` only. A separate scoring module reads `.route` after prediction finishes. This separation is the leakage evidence a reviewer will look for.
- Every request is logged with trip ID, timestamp, stage timings and result summary, printed to the Laptop 1 console for the evaluator.
- Do not cache results by trip ID. Cache only graphs and indexes; a result cache can look like hardcoding.
- UI (recommended: Streamlit with folium): ten trip cards, Run button, map with raw GPS, predicted route and optional classifier-only layer, metric panel (classifier vs HMM), spinner, readable error state. API base URL is a config value, since Laptop 2 must reach Laptop 1 over the network; enable CORS and bind the API to `0.0.0.0`.
- Target under 30 seconds per trip on Laptop 1 (to be measured on `student_test`; assumption, no limit was stated).

### 2.5 Acceptance criteria

- [ ] 5 or more models trained, logged with seeds, configs, CV and `student_test` scores
- [ ] Bayesian trial log saved; tuned vs untuned compared on identical metrics and partitions
- [ ] HMM vs classifier-only table, with at least one case where HMM does not help explained
- [ ] All 10 `student_test` trips run end to end through the live API from Laptop 2
- [ ] No `.route` read anywhere in the prediction path; no trip ID appears in model code
- [ ] Folium map in the PDF and the HTML or generating code in the repo
- [ ] Every number in the PDF reproducible from the repo; README gives install, run and deploy steps
- [ ] No secrets in the repo; AI use disclosed; each member can explain every section

## 3. Team split

A and B carry the hard track: the model and the decoder, where the marks and the risk sit. C and D carry the supporting track. C is the broad supporting engineer: data, maps, simple models, API, scoring, deployment, tests and report. D has a deliberately small, low-complexity scope: the evaluator UI and demo support. Names are not known yet, so people are Person A to D.

| Person | Track | Owns | Why this difficulty |
| --- | --- | --- | --- |
| A | Hard | Candidates, features, label alignment, heavy models, Bayesian search | Label derivation from route-only ground truth and feature design decide whether anything downstream works |
| B | Hard | Projection, graph, shortest paths, HMM/Viterbi, metrics, `predict_trip` | Graph search on 165k-arc networks must be fast; decoder edge cases (gaps, disconnection) are subtle |
| C | Supporting (broad) | EDA, inventory, Folium, simple models, plots, FastAPI, scoring, logging, two-laptop setup, tests, report, README | Mostly well-trodden libraries, but a wide scope; needs care, writing and steady integration work |
| D | Supporting (minimal) | Simple UI, UI screenshots, demo checklist, report section 7 text | A small page that renders the API's JSON; no algorithms |

### Person A – ML lead (hard)

- Candidate generation with radius and nearest-K fallback; tune R by candidate miss rate on training trips. **Hard**
- Monotone alignment of GPS points to `.route` arcs to create labels; report ambiguity. **Hard**
- Feature set (section 2.3 step 3) and feature ablations. **Hard**
- Train Random Forest or Extra Trees, boosting, MLP with trip-grouped CV. **Medium**
- Bayesian search: define space, objective, run trials, save the log, refit, tuned vs untuned. **Hard**
- Export model, scaler, config, feature order. Deliver a `score_candidates` function to B. **Medium**
- Report: sections 3 (heavy models) and 4.

### Person B – Decoder lead (hard)

- Per-trip local projection, arc geometry, spatial index. Ship this first, since A needs it. **Medium**
- Sparse graph plus bounded shortest-path engine with caching; benchmark on the 88k-node trip. **Hard**
- Emission and transition models, log-space Viterbi, segment restart on gaps, penalties for unreachable pairs. **Hard**
- Stitch final route; compare classifier-only vs HMM; tune β, R and gap threshold on training and `student_test` only. **Hard**
- Metrics module (section 2.3 table), used by everyone. **Medium**
- `predict_trip(trip_dir)` that chains A's model and B's decoder. **Medium**
- Report: section 5.

### Person C – Data and insight (supporting)

- Loader and dataset inventory (counts, gaps, missing values, extent, network connectivity, isolated nodes, odd trips). **Easy**
- EDA plots and the data-quality log; Folium maps with legends and layer control. **Easy to medium**
- Logistic Regression, Gaussian NB and KNN on A's feature table, same CV folds and logging harness. **Easy**
- Comparison plots: CV macro F1, `student_test` Top-1 and F1, log loss, confusion matrices. **Easy**
- Own the report PDF, assemble sections, check every number against repo outputs. Own README and repo layout. **Medium (volume)**
- FastAPI app: `/health`, `/trips`, `/predict`, startup model loading, request logging to console and file. Start with a stub that returns canned-shape JSON from the nearest-arc baseline so D can build the UI early; swap in `predict_trip` when B ships it. **Easy to medium**
- Server-side scoring module (`scoring.py`) that reads `.route` only after prediction; keep it out of the prediction import path. **Medium**
- Two-laptop setup: IP, firewall, CORS, config file, startup scripts; rehearse with `student_test` as stand-in evaluator trips. **Medium**
- Integration tests across all 10 `student_test` trips, including a test that prediction runs with `.route` removed. **Medium**

### Person D – UI and demo support (supporting, minimal)

- Simple Streamlit page (or plain HTML with Leaflet): list the trip IDs from `GET /trips`, a Run button that sends the selected trip ID to `POST /predict`, then show the returned route on a map next to the raw GPS, plus the returned metrics. The page does no computation. **Easy**
- Spinner while waiting and a readable message if the API is unreachable or returns an error. **Easy**
- API base URL in one config value so Laptop 2 can point at Laptop 1. **Easy**
- Take UI screenshots for the report and write report section 7 (UI and API-console screenshots, with C supplying the console shots). **Easy**
- Demo checklist: start order, URLs, what to show the evaluator (UI, then API logs on Laptop 1). Run it once with C before the demo. **Easy**
- Shadow B once (viva pairing) so you can explain the pipeline at a high level.

### Contracts to freeze in the first hour

These let all four work in parallel against stubs.

| Contract | From → to | Shape |
| --- | --- | --- |
| `load_trip(dir, with_labels=False)` | C → all | Dict of projected points, timestamps, nodes, arcs; labels only when asked |
| Trip graph and index | B → A | Projected arc geometry, nearest-arcs query, adjacency |
| Candidate table | A → B, C | One row per (trip, obs, arc): features, `label`, trip group id; Parquet or CSV |
| `score_candidates(df) → probs` | A → B | Reads saved artifacts; same feature order as training |
| `predict_trip(dir) → dict` | B → C | Per-observation classifier arc and HMM arc, route polyline, timings; no labels |
| `evaluate(trip_id, prediction) → dict` | B + C | Server-side only; loads `.route` itself |
| `POST /predict` response JSON | C → D | Shape in section 2.4; the UI only renders it |

### Working agreements

- **Pairs for the viva:** A with C, B with D. Each pair walks through each other's code once before the demo; everyone must be able to explain every section.
- **Fixed seeds, one CV split file** shared by all models so comparisons are fair.
- **No evaluator data** on any laptop until the final session, and never in the repo.
- **Spare capacity:** when C or D finish early they take error analysis, failure-case maps, and ablations, not new unreviewed code in the hard track.
- **Git:** one branch per person, small PRs, commit history kept honest because organisers compare repository history.

## 4. Timeline, risks and deliverables

The key schedule move is a thin end-to-end path (nearest-arc baseline through API and UI) by the end of phase 2, so integration risk is paid early and the models drop in later. The hackathon length is not stated in the documents, so phases are relative; stretch or squeeze them to the real clock. The guidelines say a first-phase evaluation happens before lunch, so confirm what must exist by then.

| Lane | 1 Contracts + 2 Baseline | 3 Models + HMM | 4 Integrate | 5 Report |
| --- | --- | --- | --- | --- |
| A (hard) | Candidates, labels, features | Models + Bayes | Tune, export | Report 3, 4 |
| B (hard) | Projection, graph, paths | HMM / Viterbi | Tune decoder | Report 5 |
| C (support) | Loader, EDA, Folium, API + scoring stub | Simple models, real API | Wire real model, plots, tests | PDF + README, laptop setup |
| D (support) | UI against stub API | UI polish | UI on Laptop 2 | Screenshots, demo checklist |

Gates: **Contracts frozen** (end of phase 1) → **Thin path live** (end of phase 2) → **Model + decoder frozen** (end of phase 4) → **Submit** (end of phase 5).

### Risks

| Risk | Impact | Mitigation | Owner |
| --- | --- | --- | --- |
| Label alignment from route is noisy | Bad labels cap accuracy | Monotone DP, measure ambiguity, report it as a limitation, check on Folium maps | A |
| Candidate misses | Reference arc never offered, so no model can win | Tune R and nearest-K on training trips; track miss rate per trip | A |
| Shortest paths too slow on 165k-arc trips | API takes minutes per trip | Bounded Dijkstra with cutoff, sparse matrices, per-trip graph cache, benchmark early | B |
| HMM underperforms classifier-only | Weak story for section 5 | Tune β and penalties on train and `student_test`; report honestly where it fails | B |
| Leakage via `.route` in prediction path | Disqualification risk | Separate scoring module, code review, test that prediction runs with `.route` removed | B, C |
| Two-laptop networking fails on demo day | No live demo | Rehearse early on the real network, static IP or hotspot, fallback config, bind `0.0.0.0` | C |
| Report numbers do not match repo | Integrity flag | Every table generated by a script that writes CSV; C checks each number | C |
| Team cannot explain AI-assisted code | Viva marks lost | Paired walkthroughs, short design notes per module, AI use disclosed | All |
| Model files too large for GitHub | Missing artifacts | Pick a compact model, compress, or use Git LFS | A |

### Deliverables checklist

- [ ] Public GitHub repo: data processing, features, training scripts for every model, evaluation, Bayesian search, HMM, API, UI, Folium code, results CSVs, logs, model artifacts, README, requirements
- [ ] One consolidated PDF: EDA, model comparison table and per-model answers, Bayesian section, HMM section, Folium section, API and UI section with screenshots
- [ ] Screenshots of the UI and of successful API responses in the Laptop 1 console
- [ ] Interactive Folium HTML or its generating code committed
- [ ] Live demo: API on Laptop 1, UI on Laptop 2, all 10 evaluator trips, API logs shown
