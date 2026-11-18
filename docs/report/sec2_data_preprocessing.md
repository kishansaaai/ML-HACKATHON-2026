# 2. Data Understanding and Preprocessing

## 2.1 Dataset

Each trip folder has `.track` (longitude, latitude, time in seconds, about 1 Hz), `.nodes` (longitude, latitude), `.arcs` (from-node, to-node; the row number is the arc ID) and, for development trips only, `.route` (the true arc sequence). Files have no headers and every trip has its own road network. Source: `results/data_inventory.csv`, `results/data_inventory.summary.json`.

| | Training (80 trips) | Student test (10 trips) |
| --- | ---: | ---: |
| GPS points | 208,417 | 19,160 |
| Road nodes | 1,933,485 | 90,689 |
| Directed arcs | 3,876,904 | 185,041 |
| Invalid GPS / node / arc entries | 0 | 0 |
| Duplicate track rows, non-increasing timestamps, isolated nodes | 0 | 0 |
| Duplicate node coordinates / self-loop arcs | 89 / 22 | 3 / 2 |
| Median time step | 1.0 s | 1.0 s |
| Time steps longer than 5 s | 1,216 | 44 |

There are no missing values. The data issues that matter are the longer time gaps (the decoder restarts after a gap above 30 s) and noisy or partly unreliable reference routes (below).

## 2.2 Preprocessing

- **Projection.** Each trip is projected to local metres (equirectangular projection around the median of its track); distances are never computed in degrees. The classifier features and the decoder use the same projection origin.
- **One graph per trip.** Networks are never merged.
- **Candidates.** All arcs within R = 50 m of a GPS point; if fewer than 6 are found, the radius doubles (up to 400 m) until 6 nearest arcs are found, at most 24 per point. A KD-tree over points along each arc plus an exact point-to-segment distance makes this exact. R = 50 m was chosen from the miss-rate curve (`results/candidate_sweep.csv`): the share of labelled points whose true arc is outside R is 5.1 % at 20 m, 2.9 % at 30 m, 1.4 % at 50 m and 0.3 % at 80 m before the fallback; with the fallback the real miss rate is 0.61 % on training and 0.010 % on student test.
- **Labels (training and evaluation only).** The reference route has far fewer arcs than there are GPS points, so a per-point label is derived by a monotone dynamic-programming alignment of points to the route arcs. Points more than 100 m from their aligned arc stay unlabelled. About 98 % of training points are labelled. Training trips 00000054 and 00000033 have reference routes that barely follow the GPS (for 00000054 only 14.7 % of points are labelled), so their labels are unreliable; they were kept and this is a limitation.
- **Training sample.** Models are fitted on every 3rd point (consecutive 1 Hz points are nearly duplicates); extra-trees and random forest use every 6th. Scoring and metrics always use every point.
- **Scaling.** Tree and boosting models use raw features; logistic regression and KNN use a `StandardScaler` inside a scikit-learn pipeline, so it is fitted only on the data passed to `fit`.
- **Cross-validation.** 5 folds grouped by trip (seed 42, balanced by point count, `configs/cv_folds.json`). Points of one trip are strongly correlated, so a random split would leak.

## 2.3 Features (27)

| Group | Features | What it captures |
| --- | ---: | --- |
| Geometry | 5 | perpendicular distance to the arc, position along it (fraction, metres from start, metres remaining), arc length |
| Relative | 4 | how the arc compares with the other candidates of the same point: distance to the closest, gap to it, distance rank, number of candidates |
| Topology | 3 | two-way flag, out-degree at the arc's end node, in-degree at its start node |
| Motion | 6 | speed, moving flag, heading difference between track and arc (absolute difference and cosine), time to the previous and next point |
| Neighbours | 9 | the same arc's perpendicular distance and position at the previous and next points (1 and 3 steps), window minimum, gap and rank |

Feature-group ablation (LightGBM, same grouped CV; `results/ablation.csv`):

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

Motion (heading, speed) and neighbour features matter most: dropping motion costs about 2.9 points of CV Top-1 and dropping the neighbours about 1.8 points, while the 9 neighbour features alone already reach 0.843. Geometry, relative and topology features overlap: removing any one of them changes CV Top-1 by only 0.1 to 0.15 points (its effect on the student test is larger for topology, 0.9021 to 0.8872).

## 2.4 Exploratory plots

EDA histograms (time gaps, track length, network size and connectivity) and a Folium EDA map of a training trip are in `reports/figures/` and `reports/eda_map.html` (`scripts/plot_eda.py`, `scripts/make_eda_map.py`). A Folium prediction map with GPS, network, reference and predicted route is in section 6.
