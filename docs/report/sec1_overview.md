# 1. Overview

**Problem.** Given a noisy GPS track and the road network of that trip (nodes and directed arcs), recover the sequence of road arcs the vehicle actually drove. Every trip has its own road network, so the solution cannot rely on a fixed map.

**Approach.** The pipeline has two learned/decoded stages:
1. **Candidate classifier (LightGBM).** For each GPS point, all arcs within 50 m (at least 6 nearest) become candidates. A LightGBM model scores each (point, arc) pair with the probability that the arc is the correct one, using 27 geometry, motion, topology and neighbourhood features.
2. **HMM / Viterbi decoder.** The normalised classifier probabilities are the emission scores; a transition score based on the difference between road distance and GPS distance links consecutive points. Viterbi gives one connected route; where no legal connection exists, the route is cut into separate segments instead of inventing roads.

A FastAPI server (Laptop 1) runs both stages and scores the result against the hidden reference route on the server only; a Streamlit UI (Laptop 2) shows the map and metrics.

**Data partitions**

| Partition | Trips | Use |
| --- | ---: | --- |
| Training | 80 | Model fitting, 5-fold trip-grouped cross-validation, Bayesian search, decoder tuning (out-of-fold scores) |
| Student test | 10 | Development comparison of models and decoder only |
| Evaluator test | 10 | Not in the repository; never used for any choice (features, model, hyper-parameters, thresholds, decoder settings) |

**Headline results (development data; `student_test` is only 10 trips).**

| Item | Value |
| --- | ---: |
| Final model | single LightGBM (`artifacts/model.joblib`), trained on the 80 training trips |
| Grouped 5-fold CV Top-1 (training trips) | 0.8748 (std 0.0374) |
| Student-test Top-1 / macro-F1 / log loss (classifier) | 0.9021 / 0.9406 / 0.0480 |
| Student-test Top-1, classifier only vs classifier + HMM | 0.9021 vs 0.9071 |
| Route continuity, classifier only vs classifier + HMM | 0.9558 vs 0.9697 |
| Route breaks, classifier only vs classifier + HMM | 774 vs 501 |
| Slowest trip through the API (target 30 s) | 15.4 s (trip 00000056) |

Route continuity is the share of adjacent predicted pairs that are the same arc or connected head-to-tail. Sources: `results/metrics_table.csv`, `results/experiments.csv`, `results/hmm_tuning/final_student_test_check.csv`, `results/api_student_test.csv`.
