# Candidates, labels and features (Person A)

Code: `src/pathsnap/geo.py` (interim, B owns), `labels.py`, `candidates.py`, `features.py`; run by `scripts/make_candidates.py`.

## What it does
1. **Per-trip projection** to local metres (equirectangular around the track median). Distortion over a trip is far below GPS noise, and no UTM zone choice is needed. Every trip has its own network in its own region, so one graph per trip.
2. **Candidates**: all arcs within R = 50 m of a GPS point; if fewer than 6 (`k_min`), the radius doubles (up to 400 m) until 6 nearest are found; at most 24 per point. A KD-tree over points sampled every 10 m along each arc plus an exact point-to-segment distance makes the search exact.
3. **Labels** (training/evaluation only): a monotone dynamic-programming alignment of points to the route's arcs (`labels.py`), minimising the capped point-to-arc distance. A point further than 100 m from its aligned arc is left unlabelled (-1) and excluded from training and from Top-1.
4. **Features** (27, five groups): geometry, relative-to-other-candidates, topology, motion/heading, neighbour support (how well the points just before/after fit the same arc). No label, trip id or evaluator data enters the features; labels are attached after features are built.

## Findings that shaped the design (measured on the 90 development trips)
- Trips are from many regions worldwide; networks range from about 900 to 155,000 nodes. So: per-trip graphs, per-trip projection.
- About 84 % of arcs have a reverse arc (two-way roads), so both directions of a road are always candidates; the heading features and the neighbour-support features separate them.
- Route arcs connect head-to-tail in 99.9 % of consecutive pairs, so the route is a real driven path.
- Median distance from a point to its aligned route arc is about 4.6 m (90th percentile about 12 m).
- **Candidate radius**: the share of labelled training points whose reference arc lies outside R is 5.1 % at 20 m, 2.9 % at 30 m, 1.4 % at 50 m, 0.3 % at 80 m (`results/candidate_sweep.csv`, `results/plots/candidate_sweep.png`). With the `k_min` fallback, the real miss rate at R = 50 is 0.61 % on train and 0.010 % on `student_test` (`results/experiments.csv` test_candidate_miss_rate = 0.000104), at 11.4 candidates per point on average. Larger R buys little and costs table size and training time.
- **Label quality limitation (report this)**: 98.0 % of training points are labelled on average, but a few trips have reference routes that barely follow the GPS: train trip 54 has only 14.7 % labelled points (about 90 % of its points are more than 30 m from the route), trip 33 has 49 %. These are limitations of the reference route, not of the model. `student_test` is 99.96 % labelled. About 6-7 % of points are "ambiguous" (a non-adjacent route arc, far in route order, lies within 3 m of the assigned one), typically out-and-back or overlapping stretches; monotonic alignment resolves these using route order.
- About 3.3 % of training points lie more than 30 m from their reference arc (mostly those odd trips).

## Decisions and rejected alternatives
- Chose a **candidate classifier** (probability per candidate) over direct multi-class arc prediction, because the set of arcs differs per point and per trip, and the HMM needs per-candidate emission probabilities.
- **No class weights**: positives are 9 % of candidates, and re-weighting would distort the probabilities the HMM consumes; calibration was checked instead (ECE 0.0027 on `student_test`).
- **Training on every 3rd GPS point** (consecutive 1 Hz points are nearly duplicates) to cut training time; scoring and metrics always use every point.

## Limitations
- Straight-line arcs (no road shape points) mean a point on a curved road can be far from its arc's straight segment.
- Long gaps (up to days in a few trips) make neighbour-based features unavailable; the sentinel value plus the `dt_*` features let the model cope, and the HMM must restart segments there (Person B).
- Interim `geo.py`/`io.py`: Person B and C own the final versions; names and signatures are the agreed contract.

## Likely viva questions
1. Why not Euclidean distance in degrees? (Angular units; 1 degree of longitude shrinks with latitude.)
2. How do you get per-point labels from a route with far fewer arcs? (Monotone DP alignment; explain the prefix-minimum trick.)
3. Why are both directions of a road candidates, and what tells them apart? (Heading difference and neighbour projection order.)
4. How did you choose R = 50 m? (Miss-rate vs radius curve plus candidate count and cost.)
5. Where could label leakage occur and how is it prevented? (Labels attached after features; `.route` never read in `candidates`/`features`.)
