# 2. Dataset Understanding and Exploratory Data Analysis

Partitions used: **training** (80 trips) for all fitting, cross-validation and decoder tuning; **student_test** (10 trips) only for development comparisons; **evaluator_test** is not in the repository and was never used for any choice. Every table and figure below names its partition.

## 2.1 How did you preprocess the data?

- **Parse and validate.** Each trip folder holds `.track` (longitude, latitude, time), `.nodes`, `.arcs` and, for development trips, `.route`; none has a header, and the arc ID is the row number in `.arcs`. Rows with invalid coordinates, duplicated rows and non-increasing timestamps are checked per trip (Table 2); none had to be removed.
- **One road graph per trip.** Every trip has its own network, so graphs are never merged and no map is shared.
- **Project to metres.** Each trip is projected with an equirectangular projection around the median of its own track; features and decoder share the same origin, so all distances are in metres.
- **Candidates.** All arcs within R = 50 m of a GPS point; if fewer than 6 are found, the radius doubles (up to 400 m) until the 6 nearest are included, at most 24 per point. R = 50 m was chosen from the miss-rate curve (`results/candidate_sweep.csv`): the true arc lies outside the radius for 5.1 % of labelled points at 20 m, 2.9 % at 30 m, 1.4 % at 50 m and 0.3 % at 80 m before the fallback; with the fallback the real miss rate is 0.61 % on training and 0.005 % on student_test.
- **Labels.** The reference route has far fewer arcs than GPS points, so a per-point label comes from a monotone dynamic-programming alignment of the points to the route arcs. Points more than 100 m from their aligned arc stay unlabelled; 98.0 % of training points are labelled (student_test 100.0 %). Training trips 00000054 and 00000033 have reference routes that barely follow their GPS (for 00000054 only 14.7 % of points are labelled); they were kept and this is listed as a limitation.
- **Sampling and scaling.** Models are fitted on every 3rd point (consecutive 1 Hz points are almost duplicates; Extra Trees and Random Forest use every 6th); scoring always uses every point. Tree and boosting models use raw features; logistic regression and KNN use a `StandardScaler` inside a scikit-learn pipeline, so it is fitted only on the data passed to `fit`.
- **Splitting.** 5-fold cross-validation grouped by trip (seed 42, `configs/cv_folds.json`): points inside a trip are strongly correlated, so a random split would leak.

## 2.2 What patterns did you observe, and which features did you extract and why?

**Patterns observed**

- GPS points lie close to their road: the median distance to the aligned arc is 4.6 m on training and 4.7 m on student_test, with a 90th percentile of 11.8 m and 12.5 m. Yet picking the nearest arc is right only 0.5360 of the time on student_test (Table 3), because parallel roads, junctions and reverse-direction twins of the same street lie within a few metres.
- Sampling is nominally 1 Hz (median step 1.0 s in both partitions) with 1,216 steps longer than 5 s in training and 44 in student_test (Figure 2), so the decoder must restart after long gaps.
- Trips differ a lot in size: 361 to 19,313 GPS points and 19,526 arcs in the median training network (Figure 2); a few trips are very long, so long decoding time is dominated by the biggest trips.
- The reference route is a reference, not perfect truth: two training trips do not match their GPS.

**Features (27 per candidate, five groups) and why**

| Group | # | Features | Why chosen |
| --- | ---: | --- | --- |
| Geometry | 5 | perpendicular distance, fractional / metric position along the arc, remaining length, arc length | how well the point fits this arc |
| Relative | 4 | distance to the closest candidate, gap to it, distance rank, number of candidates | the nearest arc is often wrong, so compare with the competition |
| Topology | 3 | two-way flag, out-degree at the arc end, in-degree at its start | one-way streets and junction structure limit plausible moves |
| Motion | 6 | speed, moving flag, heading difference (absolute and cosine), time to previous / next point | direction of travel separates the two directions of a street |
| Neighbours | 9 | the same arc's perpendicular distance and position at the previous and next 1 and 3 points, window minimum, gap and rank | consistency over time suppresses single-point noise |

*Table 1. The 27 features by group (partition: training; built in `src/pathsnap/features.py`).*

Feature-group ablation (LightGBM, grouped CV on training; student_test shown for reference only):

| Setting | Features | CV Top-1 | Student-test Top-1 |
| --- | ---: | ---: | ---: |
| all features | 27 | 0.8748 | 0.9021 |
| without geometry | 22 | 0.8738 | 0.8971 |
| without relative | 23 | 0.8735 | 0.8983 |
| without topology | 24 | 0.8733 | 0.8872 |
| without motion | 21 | 0.8463 | 0.8854 |
| without neighbours | 18 | 0.8571 | 0.8854 |
| only geometry | 5 | 0.5380 | 0.5477 |
| only relative | 4 | 0.5510 | 0.5440 |
| only topology | 3 | 0.3483 | 0.2988 |
| only motion | 6 | 0.3768 | 0.3975 |
| only neighbours | 9 | 0.8429 | 0.8682 |

*Table 1b. Feature ablation (`results/ablation.csv`).*

Motion and neighbour features matter most: removing motion lowers CV Top-1 by about 2.9 points and removing neighbours by about 1.8 points, while the 9 neighbour features alone already reach 0.843. Geometry, relative and topology features overlap: removing any one of them changes CV Top-1 by only 0.1 to 0.15 points.

## 2.3 Missing values, trajectories and the road network

| Check | Training (80 trips) | Student test (10 trips) |
| --- | ---: | ---: |
| GPS points | 208,417 | 19,160 |
| Road nodes | 1,933,485 | 90,689 |
| Directed arcs | 3,876,904 | 185,041 |
| Missing or invalid GPS coordinates | 0 | 0 |
| Invalid node coordinates | 0 | 0 |
| Invalid arc endpoints | 0 | 0 |
| Duplicate track rows | 0 | 0 |
| Non-increasing timestamps | 0 | 0 |
| Isolated nodes | 0 | 0 |
| Duplicate node coordinates / self-loop arcs | 89 / 22 | 3 / 2 |
| Time steps longer than 5 s | 1,216 | 44 |

*Table 2. Data inventory and missing-value / validity analysis (partitions: training and student_test; `results/data_inventory.csv`). The evaluator partition is not in the repository.*

There are no missing values: every GPS, node and arc entry is valid. The issues that matter are the long time gaps, the few duplicate node coordinates and self-loops (harmless for matching), and the imperfect reference routes of two training trips.

![Figure 1. GPS trajectory, road network and reference route of training trip 00000002 (partition: training).](docs/report/fig/eda_gps_network.png)

![Figure 2. Distribution over trips of GPS points, long time gaps and road arcs (partitions: training and student_test).](docs/report/fig/eda_data_quality.png)

![Figure 3. Interactive Folium EDA map of training trip 00000001 (partition: training): GPS trajectory and reference route with layer control; the sampled road network can be switched on. Generated by `scripts/make_eda_map.py`, exported as `reports/eda_map.html`.](docs/report/fig/eda_folium_map.png)
