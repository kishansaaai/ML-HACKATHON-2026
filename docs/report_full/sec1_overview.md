> DRAFT prepared with AI assistance from repository files. The team must rewrite this in its own words and be able to explain every statement (assessment policy).

# 1. Overview

## 1.1 The problem

GPS receivers in vehicles are noisy. A car driving along one road can appear on a parallel street or off the road network altogether (Main Guidelines.pdf). PathSnap takes the noisy 1 Hz GPS points of one trip and the road network of that trip, and returns the road arc (a directed road segment) each GPS point belongs to, joined into one connected route.

Input files per trip are `.track` (longitude, latitude, time), `.nodes`, `.arcs` and, for development trips only, `.route` (the reference sequence of arcs). Every trip has its own road network, so one graph is built per trip (docs/PLAN_AND_PRD.md, section 1; docs/notes/candidates_labels_features.md).

## 1.2 Approach in one paragraph

For each GPS point the system finds every road arc within 50 m (at least 6 nearest arcs are always offered), computes 27 features for each (point, arc) pair, and a classifier gives the probability that the arc is the correct one. A hidden Markov model (HMM) with Viterbi decoding then chooses the most plausible connected sequence of arcs, using those probabilities as emissions and road-network distance for transitions (docs/notes/candidates_labels_features.md; docs/section5_decoder.md). The classifier used everywhere in the final system is a single LightGBM model (artifacts/model_card.json). The classifier models are compared in Section 3 of this report; the decoder is described in Section 5.

```
raw .track/.nodes/.arcs -> candidates (<=50 m, >=6 nearest) -> 27 features -> LightGBM P(arc correct)
                        -> HMM/Viterbi (beta 200 m, gap 30 s, sigma 25 m, penalty -5) -> arc per point + stitched route
```

Decoder settings are in configs/decoder_best.json. Prediction reads only `.track`, `.nodes` and `.arcs`; the `.route` file is opened only by the scoring code after prediction (README.md, section "Predict a route").

## 1.3 Data partitions and how each is used

| Partition | Trips | Use in this project | Source |
| --- | ---: | --- | --- |
| Training | 80 | Model fitting, 5-fold trip-grouped cross-validation, Bayesian search, decoder tuning on out-of-fold scores | results/data_inventory.summary.json; docs/notes/models_and_tuning.md |
| Student test | 10 | Development comparison of models and decoder only | docs/notes/models_and_tuning.md |
| Evaluator test | 10 | Not in the repository. Never used for any choice (features, model, hyperparameters, thresholds, decoder settings) | README.md, "Rules the code follows" |

`student_test` is a development comparison set, not an unbiased final estimate. Ten trips is a small sample (docs/DATA_QUALITY.md).

## 1.4 Headline results (all on development data)

| Item | Value | Source |
| --- | --- | --- |
| Final classifier, observation Top-1 on student_test | 0.9021 | results/experiments.csv (lightgbm, latest row) |
| Final classifier, 5-fold grouped CV Top-1 (mean, std) | 0.8748 (0.0374) | results/experiments.csv |
| Non-learned baseline, nearest road, Top-1 on student_test | 0.5360 | results/experiments.csv (nearest_arc) |
| Non-learned baseline, nearest road + heading rule, Top-1 on student_test | 0.8301 | results/experiments.csv (nearest_heading_aware) |
| Classifier-only vs classifier + HMM, Top-1 on student_test (19,143 labelled points) | 0.9021 vs 0.9071 | docs/section5_decoder.md; results/hmm_tuning/final_student_test_check.csv |
| Classifier-only vs classifier + HMM, mean route continuity on student_test | 0.9558 vs 0.9697 | docs/section5_decoder.md |
| Classifier-only vs classifier + HMM, route breaks on student_test | 774 vs 501 | docs/section5_decoder.md |
| Same comparison out of fold on 80 training trips, Top-1 | 0.8790 vs 0.8792 | docs/section5_decoder.md |

Route continuity here is the official definition: the share of adjacent predicted pairs that are the same arc or head-to-tail connected (docs/REPORT_EVIDENCE_MAP.md). Columns named `*_distinct*` in results/experiments.csv are a stricter variant that is not the official metric and are not quoted in this report.

Plain reading of these numbers: the learned classifier is about 7 points above the heading-rule baseline (docs/notes/models_and_tuning.md). The HMM changes Top-1 very little on the 80 training trips (0.8790 to 0.8792, out of fold) but reduces route breaks (8,245 to 5,505) and was worse than classifier-only on 28 of 80 training trips (docs/REPORT_EVIDENCE_MAP.md; docs/section5_decoder.md). Details belong to Section 5.

## 1.5 Limitations stated up front

1. The three boosting models (LightGBM, HistGradientBoosting, XGBoost) are statistically tied. Their gaps are about 0.1 points of Top-1, and fold-to-fold standard deviation is about 3.7 points (docs/notes/models_and_tuning.md; results/experiments.csv).
2. Bayesian tuning of LightGBM (Optuna, 34 trials) gave no gain over the default configuration (results/tuning_summary_lightgbm.json; docs/notes/models_and_tuning.md).
3. Extra Trees and Random Forest were run in a reduced setting (50 trees, every 6th training point), so part of their gap to boosting may come from that setting (results/experiments.csv; docs/notes/models_and_tuning.md).
4. KNN was fitted on a 3,000-row subsample (results/experiments.csv, config `fit_rows_per_fold`: 3000).
5. Training trips 54 and 33 have unreliable reference alignment: 14.7 % and 49 % of their points could be labelled (results/label_stats.csv; docs/DATA_QUALITY.md).
6. Every model was run with one seed (42); there are no repeated-CV confidence intervals (docs/notes/models_and_tuning.md).
7. The supplied `.route` is reference information, not independently verified ground truth (Dataset Exploration.pdf, section 6). Per-point labels are derived from it by an alignment step and are themselves uncertain.

## 1.6 Team and roles

The plan splits work into Person A (candidates, labels, features, heavy models, Bayesian search), Person B (projection, graph, HMM/Viterbi, metrics), Person C (EDA, simple models, API, scoring, deployment, report) and Person D (evaluator UI, demo support) (docs/PLAN_AND_PRD.md, section 3).

[MISSING: team name, member names and roll numbers; the repository files only use "Person A to D".]

## 1.7 AI assistance disclosure

The repository README states that AI tools were used to help write and review code and notes, and that team members are responsible for understanding and explaining all submitted work (README.md, section "AI assistance"). [MISSING: the team must confirm the exact wording and extent of AI use for the final disclosure, per the event policy.]

## 1.8 Where each required item is found

| Required report part | This report | Evidence in repository |
| --- | --- | --- |
| Dataset understanding and EDA | Section 2 | results/data_inventory.csv, reports/, results/label_stats.csv |
| Model comparison and per-model answers | Section 3 | results/experiments.csv, results/metrics_table.csv, results/plots/ |
| Bayesian optimisation | Section 4 (other author) | results/tuning_trials_lightgbm.csv, results/tuning_summary_lightgbm.json |
| HMM and Viterbi | Section 5 (other author) | docs/section5_decoder.md, results/hmm_tuning/ |
| Folium maps, API, UI | Sections 6 and 7 (other authors) | reports/eda_map.html, ui/, src/pathsnap/api.py |
