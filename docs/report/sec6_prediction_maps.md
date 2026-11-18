> DRAFT prepared with AI assistance from repository files. The team must rewrite this in its own words and be able to explain every statement (assessment policy).

# 6. Geographical Visualisation Using Folium

All numbers below come from `results/hmm_tuning/final_student_test_check.csv` (exported LightGBM scores, tuned decoder from `configs/decoder_best.json`, `student_test` partition, 10 trips, 19,143 labelled observations; source: `results/hmm_tuning/final_student_test_check.json`). `student_test` is a development partition, not the evaluator partition.

## 6.1 Where the maps come from

| Map | Code | What it shows | Partition |
| --- | --- | --- | --- |
| Prediction map (the one the evaluator sees) | `ui/map_renderer.py`, function `build_trip_map(prediction)`, rendered inside `ui/app.py` | Raw GPS, classifier-only route, HMM/Viterbi route, start/end markers | Whatever the API serves (`student_test` in rehearsal, evaluator trips at the demo) |
| EDA map (not a prediction) | `scripts/make_eda_map.py`, output `reports/eda_map.html` | Raw GPS track, sampled road network, supplied reference route (`.route`), start/end | Chosen with `--trip` / `--partition` (README example: trip 00000001, train) |

The prediction map renders only what `POST /predict` returned; it never reads dataset files. The UI offers a **Download map (HTML)** button that saves the interactive Folium HTML (`pathsnap_trip_<trip_id>.html`).

## 6.2 Layers of the prediction map

Defined in `ui/map_renderer.py`. Base tiles are OpenStreetMap, `prefer_canvas=True`, bounds auto-fitted to the trip (`fit_bounds` with 24 px padding), scale bar on.

| Layer (layer-control name) | Style | Default | Source field in the API response |
| --- | --- | --- | --- |
| Raw GPS Observations | blue circle markers (radius 3), tooltip with observation index and lat/lon; subsampled above `MAX_GPS_MARKERS_DISPLAY` (2000, `ui/config.py`) | on | `raw_gps` |
| Classifier-Only Predictions | orange dashed polylines | off (toggle) | `classifier_route_segments` |
| HMM / Viterbi Predicted Route | green polylines, weight 5, one `PolyLine` per segment with tooltip "Segment i/n" | on | `hmm_route_segments` (fallback: `hmm_route_polyline` if segments are absent) |
| Start & End Markers | green (start) and red (end) circle markers | on | first/last point of the HMM route (fallback: first/last GPS point) |

A fixed HTML legend (bottom left, "PathSnap Map Legend") lists raw GPS, the HMM route, the classifier-only route (only when that layer exists) and start/end. A `LayerControl` (top right, not collapsed) toggles the layers. Every coordinate is `[lat, lon]`; points that are not finite or fall outside WGS-84 ranges are skipped and counted, and the UI shows a warning if any were skipped.

## 6.3 How route breaks are drawn

When the decoder meets a disconnection or a large gap it starts a new segment instead of inventing a connecting road. The API sends `hmm_route_segments` as a list of polylines, and the map draws each one as its own `folium.PolyLine`; segments are never concatenated, so no false straight line crosses a break. The classifier-only route is built the same way in `src/pathsnap/api.py` (`_classifier_segments`): a `None` arc id closes the current run, and each run is converted to geometry separately. The caption under the map in `ui/app.py` states the number of segments drawn and the number of continuity breaks (`metrics.hmm.continuity_breaks`).

Two different "break" counts exist and should not be confused:

* `hmm_breaks` in the table below is the continuity-metric count of consecutive observations whose arcs are not connected (official route-continuity definition in `src/pathsnap/decoder/metrics.py`), counted per observation pair.
* `stitched_breaks` is the number of breaks in the stitched route (`route_breaks` in `src/pathsnap/pipeline.py`; `scripts/evaluate_api.py` computes it as number of `hmm_route_segments` minus one), i.e. where a new polyline segment starts. Only this affects how many separate green segments appear on the map: a trip with 0 stitched breaks is drawn as one segment even if it has many continuity-metric breaks.

## 6.4 Classifier-only vs classifier + HMM per trip (student_test)

Top-1 is observation-level Top-1 accuracy against the label derived from `.route`; continuity is the official route-continuity score. Delta is HMM minus classifier in percentage points.

| Trip | Obs | Clf Top-1 | HMM Top-1 | Delta (pp) | Clf continuity | HMM continuity | Clf breaks | HMM breaks | Stitched breaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00000000 | 2503 | 95.33% | 95.57% | +0.24 | 98.28% | 99.20% | 43 | 20 | 2 |
| 00000007 | 1354 | 95.94% | 94.98% | -0.96 | 99.93% | 99.93% | 1 | 1 | 0 |
| 00000009 | 1049 | 88.47% | 89.32% | +0.86 | 91.22% | 93.89% | 92 | 64 | 0 |
| 00000040 | 2995 | 96.13% | 96.39% | +0.27 | 98.13% | 98.76% | 56 | 37 | 0 |
| 00000052 | 262 | 97.33% | 98.09% | +0.76 | 93.49% | 95.79% | 17 | 11 | 0 |
| 00000056 | 3958 | 91.83% | 94.37% | +2.54 | 92.82% | 95.80% | 284 | 166 | 1 |
| 00000065 | 768 | 86.72% | 87.11% | +0.39 | 96.74% | 97.78% | 25 | 17 | 1 |
| 00000076 | 1486 | 92.60% | 92.80% | +0.20 | 94.01% | 95.08% | 89 | 73 | 1 |
| 00000077 | 3911 | 78.34% | 77.81% | -0.54 | 97.03% | 98.18% | 116 | 71 | 0 |
| 00000091 | 874 | 91.19% | 90.96% | -0.23 | 94.16% | 95.30% | 51 | 41 | 0 |

Aggregate (from `final_student_test_check.json`): weighted Top-1 90.21% (classifier) vs 90.71% (HMM), delta +0.50 pp; mean per-trip continuity 95.58% vs 96.97%; total continuity breaks 774 vs 501; HMM better on 7 trips, worse on 3 (00000007, 00000077, 00000091), tied on 0. Decoder settings used: `beta_m` 200, `emission_sigma_m` 25, `gap_threshold_s` 30, `transition_cutoff_m` 10000, `unreachable_log_penalty` -5 (`configs/decoder_best.json`).

Reading the table honestly: the HMM improves continuity on every trip except 00000007 where it is unchanged, but the Top-1 gain is small overall and negative on three trips. Trip 00000052 (262 points, the smallest) and 00000056 (the largest improvement) are natural candidates for the representative map in the report; 00000077 (lowest Top-1, 78%) is a natural "HMM did not help" example. Why 00000077 is hard is analysed in `results/hmm_tuning/failure_analysis_top10.csv` [MISSING: the team has not yet written its own interpretation of that file; do not copy this draft's wording].

## 6.5 Representative map and what it shows

Representative map: `python scripts/make_prediction_map.py --trip 00000052 --part student_test` writes the interactive Folium map `reports/prediction_map_00000052.html` and the static PDF view `reports/figures/prediction_map_00000052.png`. Both are generated from the same development trip and show four distinct layers: raw GPS points, the trip road network, the reference route loaded only after prediction, and the predicted classifier + HMM route. The map is development evidence and must never be generated from evaluator data.

![Trip 00000052: raw GPS, road network, reference route and predicted route](reports/figures/prediction_map_00000052.png)

Draft caption (verify against the real map before use): "Trip 00000052 (student_test): blue dots are the 262 raw GPS observations; the green line is the HMM/Viterbi route. In a live UI run of this trip on one development machine, the KPI card showed HMM Top-1 98.09% and 11 continuity breaks (matches the table above), with the route following one main road." The classifier-only route can be shown by ticking its layer.

## 6.6 Requirement coverage and data boundary

The four-layer development map satisfies the requested exported map view. It is produced by `scripts/make_prediction_map.py`; that script first predicts using track and road files only, then separately loads the reference route for visualization. It is intentionally a separate report artifact from the evaluator UI, which does not receive the reference route or road geometry.

| Required layer | Present in the development map? | Notes |
| --- | --- | --- |
| Original GPS observations | Yes | Raw GPS Observations layer |
| Predicted route | Yes | HMM / Viterbi layer (plus optional classifier-only layer) |
| Road-network geometry | Yes | Rendered from the selected trip's `.nodes` and `.arcs` files, cropped to the GPS extent plus a small margin. |
| Reference route | Yes, for development trips with labels | Loaded only after `predict()` returns; never sent to the UI/API prediction response. |

The map caption and title identify `student_test`; this is a development visualization, not evaluator evidence. Keep the UI's prediction-only behavior unchanged so the no-leakage boundary remains intact.

## 6.7 Items the team must still do

* [TEAM ACTION: replace the AI-assisted draft caption with the team's own reading of the map and explain at least one case where HMM decoding reduced Top-1 (see trips 00000007, 00000077 and 00000091 in the table).]
