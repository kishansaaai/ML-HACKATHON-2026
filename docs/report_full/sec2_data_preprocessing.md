> DRAFT prepared with AI assistance from repository files. The team must rewrite this in its own words and be able to explain every statement (assessment policy).

# 2. Dataset Understanding and Exploratory Data Analysis

## 2.0 Which partition each analysis uses

| Analysis | Partition | File |
| --- | --- | --- |
| Dataset inventory, missing and invalid values, timestamp gaps, extent, network size and connectivity | Training (80 trips) and student test (10 trips), reported separately | results/data_inventory.csv, results/data_inventory.summary.json |
| Label alignment quality | Training and student test, reported separately | results/label_stats.csv, results/label_quality.csv |
| Candidate radius sweep | Labelled training points | results/candidate_sweep.csv |
| Feature ablation | CV on training; also student_test | results/ablation.csv |
| EDA histograms | Training and student test, coloured by partition | reports/figures/*.svg |
| Folium EDA map and static map view | Training trip 00000001 | reports/eda_map.html, reports/figures/report_map_view.png |

The evaluator partition is not in the repository and is not used in any analysis (README.md).

## 2.1 How did you preprocess the data?

**Reading and checking raw files.** The inventory script (`scripts/inventory.py`) checks every trip folder for `.track`, `.nodes`, `.arcs` and `.route`. All 90 trip folders contain valid files in the supplied layout (docs/DATA_QUALITY.md). The files have no headers: `.track` is longitude, latitude, time in seconds; `.nodes` is longitude, latitude; `.arcs` is from-node, to-node, and the 0-based row number is the arc id; `.route` is one column of arc ids (Dataset Exploration.pdf, section 3).

**Coordinates.** Coordinates are angles, so degree-Euclidean distance is not used. Each trip is projected to local metres with an equirectangular projection around the median of its track. The distortion over one trip is far below the GPS noise, and no UTM zone has to be chosen. Longitude is read before latitude. (docs/notes/candidates_labels_features.md, item 1.)

**One graph per trip.** Trips come from many regions and each has its own network, so networks are never merged and each trip has its own projection origin (docs/notes/candidates_labels_features.md; docs/DATA_QUALITY.md).

**Candidates.** Candidate arcs for each GPS point are all arcs within R = 50 m. If fewer than 6 are found, the radius is doubled (up to 400 m) until 6 nearest arcs are found, with at most 24 per point. A KD-tree over points sampled every 10 m along each arc, plus an exact point-to-segment distance, makes the search exact. The choice of R is justified in 2.4 below (docs/notes/candidates_labels_features.md).

**Labels (training and evaluation only).** The route has far fewer arcs than there are GPS points, so a per-point label does not exist in the raw data. A monotone dynamic-programming alignment of points to the route's arcs minimises the capped point-to-arc distance. A point further than 100 m from its aligned arc is left unlabelled (-1) and is excluded from training and from Top-1 (docs/notes/candidates_labels_features.md, item 3). The label is attached after features are built; `.route` is not read by the candidate or feature code (docs/notes/README_A_pipeline.md, viva question 5).

**Training sample.** Models are trained on every 3rd GPS point because consecutive 1 Hz points are nearly duplicates. Scoring and metrics always use every point. Extra Trees and Random Forest use every 6th point (results/experiments.csv: `train_stride` 3 and 6; docs/notes/candidates_labels_features.md).

**Learned preprocessing.** Tree and boosting models use the raw features. Logistic Regression and KNN use a `StandardScaler` inside the model pipeline (`standardized: true` in results/experiments.csv; scripts/train_simple.py). [MISSING: a statement from the code owner that the scaler is fitted on the training folds only; the pipeline structure suggests it but the team should confirm it.]

**Cross-validation split.** Five folds grouped by trip, seed 42, balanced by number of GPS points, stored in configs/cv_folds.json. Points of one trip are strongly correlated, so a random split would leak (docs/notes/models_and_tuning.md).

### Missing-value and invalid-value analysis

The tables below are column totals of results/data_inventory.csv. Row counts match results/data_inventory.summary.json (training 208,417 track rows, 1,933,485 nodes, 3,876,904 arcs; student test 19,160, 90,689, 185,041).

| Check | Training (80 trips) | Student test (10 trips) |
| --- | ---: | ---: |
| Invalid GPS points | 0 | 0 |
| Invalid node coordinates | 0 | 0 |
| Invalid arc endpoints | 0 | 0 |
| Duplicate track rows | 0 | 0 |
| Non-increasing timestamps | 0 | 0 |
| Isolated nodes | 0 | 0 |
| Duplicate node coordinates | 89 | 3 |
| Self-loop arcs | 22 | 2 |
| Median timestamp gap (s) | 1.0 | 1.0 |
| Intervals longer than 2 s | 2,937 | 138 |
| Intervals longer than 5 s | 1,216 | 44 |
| Intervals longer than 10 s | 590 | 27 |

Source for all rows: results/data_inventory.csv. The 2,937 and 44 figures also appear in docs/DATA_QUALITY.md. For intervals longer than 5 s in training, docs/DATA_QUALITY.md says 1,260 but the CSV column sums to 1,216 (see "Open items" at the end).

So the raw GPS files have no missing or invalid coordinates. The missing information in this dataset is of other kinds:

1. **Unlabelled points.** Training trips have 98.0 % of points labelled on average; student test has 99.96 % (results/label_quality.csv: `frac_labelled_mean` 0.9803 and 0.9996). The minimum is 14.7 % (trip 00000054) (results/label_quality.csv).
2. **Time gaps.** Most intervals are exactly 1 s, but some trips have long gaps (see 2.2). Neighbour-based features are marked unavailable when the neighbouring point is more than 10 s away, using a sentinel value of 100 m (`MAX_NBR_GAP_S`, `PERP_SENTINEL` in src/pathsnap/features.py). The decoder restarts at gaps (docs/section5_decoder.md).
3. **Candidate misses.** A point whose reference arc is not among its candidates. Measured miss rate with the nearest-K fallback: about 0.61 % on training and 0.010 % on student test (results/experiments.csv, test_candidate_miss_rate 0.000104) (docs/DATA_QUALITY.md; docs/notes/candidates_labels_features.md). Mean of per-trip miss rate in results/label_quality.csv: 0.0061 (train), 5.07e-05 (student test).

## 2.2 What patterns did you observe during data exploration?

**Sampling.** Tracks are nominally 1 Hz. Roughly 96.5 % of training intervals are exactly 1 s (docs/DATA_QUALITY.md); Dataset Exploration.pdf says about 97 %. Gaps are concentrated in a few trips. For example, training trip 00000004 has 215 intervals longer than 10 s in 1,064 points, and trip 00000081 has 86 (results/data_inventory.csv). In the student test partition, trips 00000076 and 00000065 hold the intervals longer than 10 s (24 and 3) (results/data_inventory.csv).

**Trip size.** Training tracks range from 361 to 19,313 GPS rows (median 1,529); student test tracks from 262 to 3,958 (median 1,420) (results/data_inventory.csv). Histogram: reports/figures/track_rows_by_trip.svg.

**Network size.** Training networks range from 299 to 155,003 nodes and 649 to 307,392 arcs per trip (results/data_inventory.csv). A single global graph would be wasteful and wrong, since trips are in different parts of the world (training longitudes from -100.4 to 137.2, latitudes from -34.6 to 61.3 in results/data_inventory.csv).

**Connectivity.** Networks are not single connected components. In the training partition 79 of 80 trips have more than one weakly connected non-isolated component; the median is 20 and the maximum is 421. All 10 student test trips have more than one (median 11, maximum 100) (results/data_inventory.csv; docs/DATA_QUALITY.md). The decoder therefore needs explicit handling for unreachable transitions. Histogram: reports/figures/network_components_by_trip.svg. Components were counted on an undirected reading of the directed arcs.

**Two-way roads.** About 84 % of arcs have a reverse arc, so both directions of a road are always candidates, and the heading and neighbour features are what separate them (docs/notes/candidates_labels_features.md).

**The reference route is a real driven path.** Route arcs connect head-to-tail in 99.9 % of consecutive pairs (docs/notes/candidates_labels_features.md); the minimum per-trip share is 0.9813 in training and 0.9973 in student test (results/label_quality.csv: `route_consecutive_min`).

**GPS noise relative to the route.** Median distance from a point to its aligned route arc is 4.63 m and the 90th percentile 11.8 m on training; 4.74 m and 12.5 m on student test (results/label_quality.csv). 3.3 % of training points on average lie more than 30 m from their reference arc, mostly in a few trips (results/label_quality.csv: `frac_gt_30m_mean` 0.0331).

**Unreliable reference trips.** Training trip 00000054: 2,847 of 19,313 points labelled (14.7 %), 90.2 % of points more than 30 m from the route, median distance 140.1 m. Training trip 00000033: 2,866 of 5,844 labelled (49.0 %), median distance 122.4 m (results/label_stats.csv). These are limitations of the reference routes, not of the model. Their labels are unreliable, so training and CV results include noisy labels from these two trips. [MISSING: the team has not recorded whether results were re-run without trips 54 and 33; no such file exists in results/.]

**Ambiguous labels.** About 6.0 % of training points and 7.3 % of student test points are ambiguous: a non-adjacent route arc, far in route order, lies within 3 m of the assigned one, typically on out-and-back or overlapping stretches. The monotone alignment resolves them using route order (results/label_quality.csv: `frac_ambiguous_mean` 0.0603 and 0.0734; docs/notes/candidates_labels_features.md).

**Straight-line arcs.** Arcs are straight segments between nodes with no shape points, so on a curved road a GPS point can be far from the arc's straight segment (docs/notes/candidates_labels_features.md, Limitations).

### Plots and maps in this section

| Item | File | What it shows |
| --- | --- | --- |
| GPS observations per trip | reports/figures/track_rows_by_trip.svg | Histogram, training and student test |
| Timestamp gaps longer than 2 s per trip | reports/figures/time_gaps_by_trip.svg | Histogram |
| Weakly connected road components per trip | reports/figures/network_components_by_trip.svg | Histogram (road-network connectivity) |
| Train trip 00000001: GPS track, reference route, sampled road network | reports/figures/report_map_view.png | Static view; the GPS track (blue) and reference route (red) overlap closely |
| Interactive Folium map | reports/eda_map.html | Layers: GPS trajectory, reference route (.route), road network (sampled, hidden by default), start and end markers, layer control. Generated by `python scripts/make_eda_map.py --trip 00000001 --partition train` |
| Candidate radius sweep | results/plots/candidate_sweep.png | Miss rate vs radius |

The Folium map draws at most 2,500 road arcs (sampled by stride) so that the file stays small (`max_road_arcs=2500` in scripts/make_eda_map.py); it is a sample, not the full network. It is an EDA map, not a prediction map; prediction maps belong to Section 6. [MISSING: GPS trajectory plots for more than one trip; the repository has one map trip (00000001) and the per-trip histograms.] [Team to confirm: the saved eda_map.html was generated with the script default trip 00000001; the HTML was not opened to verify the trip id.]

## 2.3 What features did you extract and why?

There are 27 features per (GPS point, candidate arc) pair in five groups (src/pathsnap/features.py). The same code path builds features for training, evaluation and live inference; no label, trip id or evaluator data enters the features.

| Group | Count | Features | Why |
| --- | ---: | --- | --- |
| Geometry | 5 | perp, frac, along_m, rem_m, arc_len | How close the point is to the arc and where along it |
| Relative | 4 | dist_min, dist_gap, rank_dist, n_cand | How this candidate compares with the other candidates of the same point |
| Topology | 3 | two_way, deg_out_end, deg_in_start | Whether the road is two-way and how connected its ends are |
| Motion | 6 | speed, moving, head_absdiff, head_cos, dt_prev, dt_next | Heading mismatch between vehicle motion and arc direction separates the two directions of a road; time gaps tell the model when other features are unreliable |
| Neighbours | 9 | perp_prev1, perp_next1, perp_prev3, perp_next3, perp_win, win_gap, rank_win, along_prev1, along_next1 | Whether the points just before and after also fit the same arc |

Feature groups and counts: src/pathsnap/features.py. The reasons follow docs/notes/candidates_labels_features.md and the ablation below. [Team to confirm the precise definition of each individual feature from the code before the viva.]

### Evidence that the features matter (LightGBM, untuned, same grouped CV)

| Setting | Features | CV Top-1 | student_test Top-1 |
| --- | ---: | ---: | ---: |
| All features | 27 | 0.8748 | 0.9021 |
| Without geometry | 22 | 0.8738 | 0.8971 |
| Without relative | 23 | 0.8735 | 0.8983 |
| Without topology | 24 | 0.8733 | 0.8872 |
| Without motion | 21 | 0.8463 | 0.8854 |
| Without neighbours | 18 | 0.8571 | 0.8854 |
| Only geometry | 5 | 0.5380 | 0.5477 |
| Only relative | 4 | 0.5510 | 0.5440 |
| Only topology | 3 | 0.3483 | 0.2988 |
| Only motion | 6 | 0.3768 | 0.3975 |
| Only neighbours | 9 | 0.8429 | 0.8682 |

Source: results/ablation.csv. Reading: removing motion features costs the most CV Top-1, then removing neighbour features. The neighbour group alone reaches 0.8429 CV Top-1, so it is the strongest single group. Geometry, relative or topology alone give 0.35 to 0.55, similar to or below the nearest-road baseline (0.5536 CV Top-1, results/experiments.csv). The drops from removing geometry, relative or topology (0.1 to 0.2 points of CV Top-1) are far smaller than the fold-to-fold spread (about 3.7 points, results/experiments.csv `cv_top1_std`), and each setting is one seed, so those small differences should not be over-interpreted (docs/notes/models_and_tuning.md).

## 2.4 Why a 50 m candidate radius

Share of labelled training points whose reference arc lies outside the radius (results/candidate_sweep.csv; plot results/plots/candidate_sweep.png):

| Radius (m) | 10 | 20 | 30 | 50 | 80 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Miss rate | 0.1994 | 0.0509 | 0.0290 | 0.0138 | 0.0031 | 0.0 |

With the nearest-6 fallback, the miss rate at R = 50 m is 0.61 % on training and 0.010 % on student test (results/experiments.csv, test_candidate_miss_rate 0.000104), at 11.4 candidates per point on average. A larger radius buys little and increases table size and training time (docs/notes/candidates_labels_features.md).

A missed reference arc caps every later stage, since no classifier or decoder can choose an arc that is not offered. The miss rate is small but not zero.

## 2.5 Design choices made because of what the data showed

- Candidate classifier (probability per candidate) rather than multi-class prediction over arcs, because the set of arcs differs per point and per trip and the decoder needs per-candidate probabilities (docs/notes/candidates_labels_features.md).
- No class weights. Positives are about 9 % of candidates, and re-weighting would distort the probabilities the decoder uses. Calibration was checked instead (ECE 0.0027 on student_test; results/plots/calibration_lightgbm.png) (docs/notes/candidates_labels_features.md).
- Per-trip graphs and projections, because of the network size spread and the many regions.
- Time gaps and disconnected components are handled in the decoder (Section 5) because 79 of 80 training networks have more than one component.

## Open items found while checking the evidence (resolved)

1. Intervals longer than 5 s in training: results/data_inventory.csv (`timestamp_gap_gt_5s`) sums to 1,216. docs/DATA_QUALITY.md said 1,260; it has been corrected to 1,216. This section uses 1,216.
2. Share of 1 s intervals: docs/DATA_QUALITY.md says about 96.5 %; Dataset Exploration.pdf says about 97 %. These are close; the team should quote the number measured from results/data_inventory.csv.
3. docs/DATA_QUALITY.md used to quote an earlier (untuned-decoder) API run. results/api_student_test.csv was regenerated with `python scripts/evaluate_api.py` using the frozen decoder: classifier Top-1 0.9021, HMM 0.9071, mean trip continuity 0.9558 and 0.9697, which now match results/hmm_tuning/final_student_test_check.csv.
