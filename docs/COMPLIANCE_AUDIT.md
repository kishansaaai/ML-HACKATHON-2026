# PathSnap compliance audit

Independent read-only audit of branch `integration/final` (== `main`) against the five organiser PDFs: `Deliverables.pdf` (7 pp), `Main Guidelines.pdf` (2), `Models and Training Guidelines.pdf` (6), `Dataset Exploration.pdf` (4), `Assessment and Plagiarism Policy.pdf` (3). Date of audit: 2026-11-18. PDF text was extracted with pypdf (the Read tool had no PDF renderer). Page numbers are PDF pages. Rows 3.x/4/5 etc. in the Deliverables PDF were mapped to the requirement IDs below.

## Summary

| Status | Count |
| --- | ---: |
| PASS | 51 |
| PARTIAL | 19 |
| FAIL | 1 |
| CANNOT-VERIFY | 10 |
| **Total requirements checked** | **81** |

Headline: the code, experiments, tuning, decoder, API and UI are in good shape and rule-compliant (no leakage, no hardcoding, evaluator partition never touched by code). The weak side is the deliverable that is not code: the committed PDF report is stale and incomplete, there is no Folium map with all four required layers, and nothing has been run across two laptops.

## Requirement table

| ID | Requirement | Source PDF + page | Status | Evidence |
| --- | --- | --- | --- | --- |
| D1 | GitHub repo is public and accessible to evaluators | Deliverables.pdf p1-2 | PASS | `gh repo view` returned visibility PUBLIC for kishansaaai/ML-HACKATHON-2026; branch integration/final == main. |
| D2 | Data processing code (read, inspect, clean, preprocess raw data) | Deliverables.pdf p1 | PASS | `src/pathsnap/io.py`, `scripts/inventory.py` (invalid/duplicate/ordering checks), `scripts/make_candidates.py`, `src/pathsnap/labels.py`. |
| D3 | Feature engineering source code | Deliverables.pdf p1 | PASS | `src/pathsnap/features.py`, `candidates.py`; 27 features listed in `artifacts/feature_order.json`; none use trip id or `.route`. |
| D4 | Training scripts for every model experimented with | Deliverables.pdf p1 | PASS | `scripts/train_heavy.py` (LightGBM, HistGB, XGBoost, ExtraTrees, RF), `scripts/train_simple.py` (LogReg, GaussianNB, KNN), `scripts/baseline_nearest.py`; `src/pathsnap/models/registry.py`. HistGB tuning attempt was stopped and its record not saved (`results/tune_log_hist_gb_stopped.txt`, disclosed in `docs/notes/models_and_tuning.md:39`). |
| D5 | Evaluation code: comparison, metrics, plots, confusion matrices | Deliverables.pdf p1 | PASS | `scripts/analysis_plots.py`, `src/pathsnap/cand_metrics.py`, `decoder/metrics.py`; `results/plots/{model_comparison,confusion_all_models,confusion_lightgbm,calibration_lightgbm}.png`, `results/metrics_table.csv`. |
| D6 | Bayesian optimisation code, search config, experiment results | Deliverables.pdf p1; Models and Training Guidelines.pdf p4 | PASS | `scripts/tune.py` (Optuna TPESampler seed 42, tune.py:42), `src/pathsnap/tuning.py` (search space, objective), `results/tuning_trials_lightgbm.csv` (34 trials), `results/tuning_summary_lightgbm.json`. |
| D7 | HMM and Viterbi source for route reconstruction/decoding | Deliverables.pdf p1 | PASS | `src/pathsnap/decoder/{hmm,graph,stitch,inference,config}.py`; `configs/decoder_best.json`; `results/hmm_tuning/`. |
| D8 | API backend code incl. endpoints and model loading | Deliverables.pdf p1 | PASS | `src/pathsnap/api.py` (`/health`, `/trips`, `/predict`; `load_model()` at startup warm-up). |
| D9 | Evaluator UI source code | Deliverables.pdf p1 | PASS | `ui/app.py`, `api_client.py`, `map_renderer.py`, `config.py`, launchers. |
| D10 | Deployment: config files, dependencies, run instructions | Deliverables.pdf p1 | PASS | `requirements.txt`, `ui/requirements.txt`, `pyproject.toml`, README sections 'Install' and 'Deployment: two laptops', `scripts/run_api.{cmd,ps1}`, `ui/run_ui.{bat,sh}`. |
| D11 | Folium map-generation code and other plotting scripts | Deliverables.pdf p1 | PASS | `scripts/make_eda_map.py`, `ui/map_renderer.py`, `scripts/plot_eda.py`, `scripts/analysis_plots.py`, `scripts/plot_simple_models.py`. |
| D12 | Supporting results (CSV, logs, configs, summaries) | Deliverables.pdf p1 | PASS | `results/` (experiments.csv, metrics_table.csv, hmm_tuning/*, logs), `configs/cv_folds.json`, `configs/decoder_best.json`. |
| D13 | All training/eval/UI/API code committed; not only the final model | Deliverables.pdf p2 | PASS | `git ls-files` lists scripts/, src/, ui/, tests/. Working tree clean apart from files other agents are adding (docs/report, docs/screenshots, scripts/build_report_pdf.py, tests/test_report_pdf.py) and a modified docs/notes file; these are not yet committed. |
| D14 | Saved model weights and required config files included | Deliverables.pdf p2 | PASS | `artifacts/model.joblib` (2.0 MB), `feature_order.json`, `model_card.json`; `.git` is 34 MB. |
| D15 | README with installation, execution and deployment instructions | Deliverables.pdf p2 | PASS | `README.md` (Install, Reproduce, Predict, API, Deployment: two laptops). Minor: `baseline_nearest.py` is documented as 'run inside scripts/'. |
| D16 | Repository organised into clearly named folders | Deliverables.pdf p2 | PASS | src/, scripts/, ui/, configs/, artifacts/, results/, reports/, docs/, tests/. An empty untracked `pathsnap/` folder sits at the repo root (harmless). |
| D17 | No passwords, API keys or secrets committed | Deliverables.pdf p2 | PASS | `git grep -iE 'api[_-]?key/password/secret/token/BEGIN PRIVATE'` returned only a checklist line in docs/PLAN_AND_PRD.md:156; no .env/key files in `git ls-files`. |
| D18 | One consolidated technical report PDF following the prescribed structure | Deliverables.pdf p2 | PASS | `Report.pdf` at the repo root (editable `Report.docx`); sources in `docs/report/`, built by `scripts/build_report_pdf.py`. |
| D19 | Report includes actual results, tables, graphs, geographical visualisations | Deliverables.pdf p2 | PASS | `Report.pdf` embeds EDA plots, Folium map views, model comparison and tuning plots, the prediction map, UI and API screenshots; every number comes from `results/`. |
| D20 | Every number in the report is verifiable from the repo | Deliverables.pdf p2, p7 | PARTIAL | Most trace to results/. Inconsistencies: `docs/DATA_QUALITY.md:31` and `results/api_student_test.csv` give HMM Top-1 0.8991 (older decoder) vs frozen decoder 0.9071 in `results/hmm_tuning/final_student_test_check.json`; `scripts/build_report.py:106` hand-types numbers/trip ids in prose. (At audit time another agent was modifying results/api_student_test.csv and scripts/evaluate_api.py; re-check after it commits.) |
| D21 | EDA: preprocessing explained; missing-value analysis | Deliverables.pdf p2 | PARTIAL | Missing/invalid analysis exists: `results/data_inventory.csv` columns invalid_track_points, duplicate_*, nonincreasing_timestamps, isolated_nodes; `docs/DATA_QUALITY.md`. The explanatory text must be in the PDF (team-written). |
| D22 | EDA: GPS trajectory plots and road-network visualisations | Deliverables.pdf p2 | PARTIAL | Only interactive map `reports/eda_map.html` (GPS, sampled road arcs, reference route) and 3 SVG summary charts (rows, time gaps, components per trip). No static GPS-trajectory or road-network figure besides the map screenshot. |
| D23 | EDA: at least one Folium map; state partition used per analysis | Deliverables.pdf p2 | PARTIAL | `reports/eda_map.html` generated by `scripts/make_eda_map.py` (default trip 00000001, partition arg). Partition labelling in the PDF is a text task. |
| D24 | Train and compare >= 5 models; every model documented | Deliverables.pdf p3 | PASS | `results/experiments.csv`: 8 learned models (LightGBM, HistGB, XGBoost, ExtraTrees, RandomForest, LogReg, GaussianNB, KNN) + LightGBM-tuned + 2 non-learned baselines. |
| D25 | Consolidated comparison table (CV F1, Top-1, continuity, test F1, log loss, inference time) | Deliverables.pdf p3 | PASS | `results/metrics_table.csv` has all columns (cv_macro_f1, cv_top1, cv_route_continuity, test_macro_f1, test_top1, test_logloss, infer_time_s_per_trip). Note: not-applicable cells (e.g. GaussianNB continuity) must be explained in the PDF. |
| D26 | Per-model subsections answering A1-A2, B5-B6, C7-C9 | Deliverables.pdf p3-4 | CANNOT-VERIFY | Team-written text; only drafts under docs/report/ (untracked). Hyperparameters and test metrics per model are available in `results/experiments.csv` (config_json column). |
| D27 | Overfitting/underfitting evidence per model (Q9) | Deliverables.pdf p4 | PARTIAL | CV mean/std and CV-vs-student_test gap are logged (`cv_top1_std` ~0.037), but training-set scores are never computed (`scripts/train_heavy.py` evaluates only validation folds and student_test), so a train-vs-validation gap cannot be shown. |
| D28 | Model visualisations: CV F1, student-test Top-1/F1, log loss, confusion matrices from real results | Deliverables.pdf p4 | PASS | `results/plots/model_comparison.png`, `simple_model_comparison.png`, `confusion_all_models.png`, `confusion_lightgbm.csv`. |
| D29 | No artificial training curves; loss plots only for epoch-wise models | Deliverables.pdf p4 | PASS | No training-curve plot exists in `scripts/` or `results/plots/`; all models are non-epoch (GBDT/RF/NB/KNN/LR) so none is required. |
| D30 | Bayesian HPO answers (params, ranges, objective, trials, best, tuned vs untuned incl. CV F1, test Top-1/F1, log loss, inference time) | Deliverables.pdf p5 | PASS | `src/pathsnap/tuning.py:26-45` ranges; objective `mean CV top1 - 0.05*logloss`; 34 trials, 850 s; best params in `tuning_summary_lightgbm.json`; `lightgbm` vs `lightgbm_tuned` rows in `experiments.csv` (same folds, same student_test). Result is honest: tuned did not beat untuned (CV Top-1 0.8746 vs 0.8748; test 0.9003 vs 0.9021) so the untuned model is exported. |
| D31 | HMM section: states, emissions, transitions, gaps, disconnected segments, improvement evidence | Deliverables.pdf p5-6; Models and Training Guidelines.pdf p4-5 | PASS | `docs/section5_decoder.md`; `results/hmm_tuning/frozen_candidate_full80.json` (OOF, 80 trips: Top-1 0.8790 -> 0.8792, continuity 0.9506 -> 0.9623), `final_student_test_check.json` (0.9021 -> 0.9071; 7 better, 3 worse trips). Gains are small and mixed; report must say so. |
| D32 | Classifier-only vs classifier+HMM comparison incl. sequence-level performance | Deliverables.pdf p5-6; Models and Training Guidelines.pdf p5 | PARTIAL | Pointwise Top-1 and continuity: saved. Sequence-level (route arc-set F1 / edit distance) for HMM-stitched route is NOT saved in any results file; `/predict` computes it live per trip. Live check on trip 00000052: classifier sequence_f1 0.881 vs HMM 0.689 (HMM worse), which should be investigated and reported. |
| D33 | Folium maps with all four layers: raw GPS, road network, reference route, predicted route; colours, legend, extent, layer control | Deliverables.pdf p6 | PARTIAL | No single map has all four. `reports/eda_map.html`: GPS + sampled roads + reference, no prediction. `ui/map_renderer.py`: GPS + classifier route + HMM route + legend + LayerControl, no road network and (by design) no reference. Needs a dev-partition report-map script combining all four. |
| D34 | PDF includes at least one representative Folium screenshot of the real implementation | Deliverables.pdf p6 | PARTIAL | `reports/figures/report_map_view.png` is the EDA map (no predicted route). docs/screenshots/ now has UI PNGs from another agent (not audited). |
| D35 | Interactive Folium HTML or generating code committed | Deliverables.pdf p6 | PASS | `reports/eda_map.html` and `scripts/make_eda_map.py` tracked. |
| D36 | No unrelated or manually fabricated maps | Deliverables.pdf p6 | PASS | Both maps are produced by committed code from real trip files. |
| D37 | Screenshots of the working UI | Deliverables.pdf p6 | CANNOT-VERIFY | Needs the UI running against the API; docs/screenshots/ has 2 PNGs added by another agent (not verified or in tracked history). |
| D38 | Screenshots of successful API responses on the CMD of the receiving laptop | Deliverables.pdf p6 | CANNOT-VERIFY | Needs Laptop 1 console. Locally verified that the console prints one line per request (uvicorn output: `POST /predict trip_id=00000052 / obs=262 / clf_top1=97.3% / hmm_top1=98.1% ...`). |
| D39 | Submitted code must match implementation demonstrated at evaluation | Deliverables.pdf p7 | CANNOT-VERIFY | Freeze a commit/tag before the demo. Verified `DecoderConfig` defaults (`decoder/config.py`) equal `configs/decoder_best.json`; the API uses `DecoderConfig()` (api.py:251), and the JSON file itself is not read by code. |
| D40 | AI assistance disclosed; every member can explain code and results | Deliverables.pdf p7; Assessment and Plagiarism Policy.pdf p2 | CANNOT-VERIFY | README 'AI assistance' paragraph exists. Whether members understand the work is a viva matter; only `docs/VIVA_PREP_PERSON_D.md` exists (one person's prep). |
| G1 | End-to-end system: explore, features, >=5 models, select, Bayesian HPO, HMM, export, API, UI, integrate, report | Main Guidelines.pdf p2 | PARTIAL | All stages implemented and tested (see D2-D14, T-rows); the final report stage is incomplete (D18). |
| G2 | Two-laptop architecture: Laptop 1 hosts classifier, HMM, road data and server-side scoring; Laptop 2 shows ten trips, sends selected trip, shows route and results | Main Guidelines.pdf p2; Models and Training Guidelines.pdf p5 | PARTIAL | Design matches: API holds model, decoder, `.route` scoring (api.py, scoring.py); UI only calls `/health`, `/trips`, `/predict` (ui/api_client.py). Never run across two machines; README deployment steps untested. |
| T1 | >= 5 distinct models from different families trained before selection | Models and Training Guidelines.pdf p2 | PASS | Boosting x3, tree ensembles x2, linear (LogReg), probabilistic (GaussianNB), instance-based (KNN): `results/experiments.csv`. |
| T2 | Record the configuration of each model | Models and Training Guidelines.pdf p2 | PASS | `config_json` column in `results/experiments.csv`; `src/pathsnap/models/registry.py`. |
| T3 | 80 training trips used for fitting and CV | Models and Training Guidelines.pdf p2 | PASS | `scripts/train_heavy.py:run` fits only on `load_part('train')`; `configs/cv_folds.json` verified: 5 folds, 80 unique trips == `Student folder/train` listing, 0 overlap with student_test, folds of 15/16/16/16/17 trips (grouped by trip). |
| T4 | student_test used only for development comparison | Models and Training Guidelines.pdf p2 | PASS | `train_heavy.py` uses student_test only in the final `evaluate(..., 'student_test')`; `tune.py` exports by CV Top-1 only (tune.py:72-74); `model_card.json` selected_by states this. |
| T5 | Preprocessing applied consistently | Models and Training Guidelines.pdf p2 | PASS | Same `FEATURE_ORDER` for train/inference (`clf.py` reorders by `feature_order.json`); StandardScaler inside sklearn pipelines for LogReg/KNN (`train_simple.py:36-49`), refit per fold. |
| T6 | Do not select by training accuracy alone | Models and Training Guidelines.pdf p2, p4 | PASS | Selection by grouped-CV Top-1 (+0.05 logloss in search); LightGBM chosen among statistically tied boosters with explicit reasoning in `docs/notes/models_and_tuning.md:30`. |
| T7 | Record training and inference time | Models and Training Guidelines.pdf p2 | PASS | `train_time_s`, `infer_time_s_per_trip` in `results/metrics_table.csv`. |
| T8 | Retain experiment logs, seeds, configs, results | Models and Training Guidelines.pdf p2 | PASS | Seed 42 everywhere (registry.py:10, make_folds.py, tune.py:42, train_simple.py:32); logs `results/*_log.txt`, `rerun_all_log.txt`. |
| T9 | Report metrics: Accuracy, Precision, Recall, F1, Confusion matrix, Log loss, Obs Top-1, Candidate Miss Rate, Route Continuity, Sequence-level, train/inference time | Models and Training Guidelines.pdf p2-3 | PASS | All present in `results/metrics_table.csv` (test_precision, test_recall, test_candidate_miss_rate, test_route_continuity, test_seq_arc_f1, ...) and `confusion_all_models.png`. |
| T10 | Explain the primary selection metric and why | Models and Training Guidelines.pdf p3 | PARTIAL | Explained in `docs/notes/models_and_tuning.md:7` (observation Top-1); must also appear in the PDF. |
| T11 | Comparison covers fold stability, probability quality, speed, complexity, decoder compatibility | Models and Training Guidelines.pdf p3-4 | PARTIAL | Fold std (`cv_top1_std`), log loss, `calibration_lightgbm.png`, timings are in the repo; model complexity / decoder-compatibility argument is in notes only, not yet in the PDF. |
| T12 | Bayesian search strategy; group-aware CV on 80 training trips; refit; compare tuned vs untuned; report cost | Models and Training Guidelines.pdf p4 | PASS | Optuna TPE; `CVProblem` uses `cv_folds.json` trip-grouped folds (tuning.py:47-56); refit via `run()` in tune.py; search_seconds=850.2 recorded. Caveat: search used every 6th training point and a time-box (34 of 40 trials). |
| T13 | Evaluator partition not used to choose search space, objective, hyperparameters, model or thresholds | Models and Training Guidelines.pdf p4 | PASS | No path in src/scripts/tests reads an evaluator folder: `paths.partition_dir` accepts only train/student_test; `grep -i evaluator` hits only api.py guards, docstrings and tests. |
| T14 | HMM: hidden-state representation, candidate-to-arc correspondence, emission/transition, Viterbi recurrence, missing candidates, invalid transitions, GPS gaps, one arc per observation | Models and Training Guidelines.pdf p4-5 | PASS | `docs/section5_decoder.md`; `decoder/hmm.py`, `stitch.py`; gap restart at 30 s, unreachable penalty -5, null segments for missing candidates; `/predict` returns `hmm_arc_id` for every observation. |
| T15 | Compare classifier-only vs classifier+HMM: pointwise, sequence-level, continuity analysis | Models and Training Guidelines.pdf p5 | PARTIAL | See D32: pointwise and continuity saved; sequence-level HMM vs classifier not aggregated in a results file. |
| T16 | Decoder parameters optimised on training and student-testing data only | Models and Training Guidelines.pdf p5 | PASS | `scripts/tune_hmm.py --oof` used out-of-fold LightGBM scores on training trips (`results/pipeline_train_oof_*.csv`, `results/hmm_tuning*/`); one post-freeze check on student_test (`final_student_test_check`). |
| T17 | Tuning scope: decoder grid is small | Models and Training Guidelines.pdf p5 | PARTIAL | Pilot sweeps used only 8 training trips (`pilot_sweep_summary.csv`) over beta/gap/sigma/penalty; only the final config was run on all 80. The penalty value -5 (the key finding) sits at the edge of the explored range (-5, -10, -20), so the optimum is not bracketed. |
| T18 | Required components all present (data, experimentation, BO, HMM, API, UI, artifacts, report, repo) | Models and Training Guidelines.pdf p5 | PARTIAL | All present except the final report (D18). |
| T19 | Final eval: API on Laptop 1, UI on Laptop 2, evaluator opens UI, UI lists ten reserved trips | Models and Training Guidelines.pdf p5 | CANNOT-VERIFY | `/trips` feeds the UI list (api.py:226, ui/app.py); cannot be run on the organisers' data or two laptops here. |
| T20 | UI sends only the selected trip ID; API runs the full pipeline and scores with internal reference data; UI shows route and results | Models and Training Guidelines.pdf p5-6 | PASS | `PredictRequest` has only `trip_id` (api.py:111); live POST /predict for 00000052 on port 8773 returned keys `trip_id, model_name, num_observations, raw_gps, hmm_route_segments, hmm_route_polyline, classifier_route_segments, observations, metrics, timings_sec, server_timestamp` with metrics Top-1 0.973 -> 0.981. |
| T21 | System works for all ten reserved trips, not just one | Models and Training Guidelines.pdf p6 | CANNOT-VERIFY | All 10 student_test trips pass in-process (`tests/test_api_integration.py`). Risks for unseen data: API aborts at startup unless the evaluator folder holds exactly 10 trip folders with `.track/.nodes/.arcs` directly under it (api.py:58-68); metrics are null if `.route` is absent. Check the organisers' layout in advance. |
| T22 | Ground truth server-side; never exposed to model or UI | Models and Training Guidelines.pdf p6 | PASS | See Leakage section: no reference-route field in `/predict`; `.route` opened only in `scoring.score_after_prediction` after `predict_trip`; `io.load_trip(with_labels=False)` default; route-free leakage test passes. |
| T23 | No hardcoded outputs or bypass of the API; show API logs on the receiving laptop | Models and Training Guidelines.pdf p6 | PASS | No hardcoded trip ids/answers in src/ or ui/ (only a docstring example at pipeline.py:4). Console log per request confirmed in uvicorn output. Showing it live is a demo task. |
| T24 | Independent implementation; no sharing of models/code/results between teams | Models and Training Guidelines.pdf p6; Assessment and Plagiarism Policy.pdf p2 | CANNOT-VERIFY | Cannot be verified from the repo; git history has 4 authors on one day (no copy-paste markers checked). |
| S1 | Structured exploration outputs: counts per partition, obs per trip, file availability/sizes, dtypes, missing/invalid, timestamp gaps, extent, nodes/arcs, connectivity, isolated nodes, route structure | Dataset Exploration.pdf p2-4 | PARTIAL | `results/data_inventory.csv` (90 rows) covers all except file sizes in bytes; summary in `data_inventory.summary.json`, `docs/DATA_QUALITY.md`. |
| S2 | Geodesic/projected distance, justified; coordinate order checked | Dataset Exploration.pdf p2-3 | PASS | `decoder/projection.py` `LocalProjection` (per-trip equirectangular metres), `geo.py`; lon,lat order handled in `io.py`; rationale in `docs/section5_decoder.md`. |
| S3 | Reference route limitations identified, not treated as error-free truth | Dataset Exploration.pdf p3 | PASS | `results/label_quality.csv`, `label_stats.csv`; trip 00000054 only ~14.7% aligned (`docs/DATA_QUALITY.md`). |
| S4 | 80/10/10 partitions; no trip across partitions | Dataset Exploration.pdf p3 | PASS | Verified: train 80, student_test 10, intersection 0; folds are by trip; evaluator trips are absent from the repo. |
| S5 | Evaluator data not used for training, feature selection, algorithm or hyperparameter choice | Dataset Exploration.pdf p3 | PASS | Code-level: see T13. Repo history cannot prove a human never opened the data, but no code path does. |
| S6 | No manual alteration of predictions; no hardcoded expected answers | Dataset Exploration.pdf p3 | PASS | Hardcoding grep (below) clean. |
| S7 | Learned preprocessing fitted on training data only | Dataset Exploration.pdf p3 | PASS | GBDT models take raw features; StandardScaler is inside per-fold sklearn pipelines; exported model fitted on train only (`tune.py:80-86`, model_card `trained_on`). |
| S8 | Keep evaluator ground truth separate from the prediction pipeline | Dataset Exploration.pdf p3 | PASS | `pipeline.make_components` calls `load_trip(with_labels=False)`; `scoring.py` is the only `.route` reader at serve time. |
| S9 | Recommended exploration outputs (distribution plots, trajectory patterns, Folium with network, network tables, quality record) | Dataset Exploration.pdf p4 | PARTIAL | Quality record and Folium map exist; no GPS coordinate-distribution or trajectory-pattern plot. |
| A1 | Evidence of meaningful experimentation; recorded results; Bayesian HPO and its search record retained | Assessment and Plagiarism Policy.pdf p1 | PASS | `results/experiments.csv`, `tuning_trials_lightgbm.csv`. |
| A2 | Explain strengths and limitations of the chosen approach | Assessment and Plagiarism Policy.pdf p1 | CANNOT-VERIFY | Team-written text (docs/report drafts untracked); notes in `docs/notes/models_and_tuning.md`. |
| A3 | Evaluate models and decoder using permitted development data only | Assessment and Plagiarism Policy.pdf p1 | PASS | See T13/T16. |
| A4 | No fabricated metrics/graphs/code results; no AI-written analysis presented as experiments | Assessment and Plagiarism Policy.pdf p2 | PARTIAL | All checked metrics trace to CSV/JSON files (e.g. metrics_table.csv equals experiments.csv rows). Weaknesses: stale PDF (D18), contradictory numbers in `docs/DATA_QUALITY.md` (D20), hand-typed prose numbers in `scripts/build_report.py`. |
| A5 | Every member can explain dataset, features, models, BO, HMM, Viterbi, API, UI, results (viva) | Assessment and Plagiarism Policy.pdf p2-3 | CANNOT-VERIFY | Only people can show this; viva notes exist only for Person D. |
| A6 | Evaluator ground truth never enters student-side pipeline or UI; no hardcoded predictions; repo examined for hard-coding | Assessment and Plagiarism Policy.pdf p3 | PASS | Leakage and hardcoding checks below. |

## Hands-on checks

### 1. Leakage (PASS)

- `grep` for `.route`, `with_labels`, `load_trip` over `src/pathsnap/pipeline.py`, `src/pathsnap/decoder/`, `src/pathsnap/clf.py`, `src/pathsnap/api.py`, `ui/`:
  - `pipeline.py:40` calls `load_trip(trip_dir, with_labels=False)`; `decoder/` and `clf.py` contain no `.route` access (`clf.py` loads only `artifacts/`).
  - `api.py` touches the route only in two places: line 253 (`(directory / f"{trip_id}.route").is_file()`, an existence check made after `predict_trip` returned) and the call to `scoring.score_after_prediction`, the single serve-time `.route` reader (`scoring.py:17`).
  - `ui/app.py`, `api_client.py`, `map_renderer.py` never open any trip file. `ui/mock_api.py` (dev stub, labelled "do not use for final evaluation") reads `.route` only inside `_score_against_route`.
  - `features.py` / `candidates.py` contain no `route` reference; the 27 features are GPS/graph geometry only (no trip id).
- Live run: started the API on port 8773 (`PYTHONPATH=<repo>\src`, log file redirected to the scratchpad), `POST /predict {"trip_id":"00000052"}` returned top-level keys `trip_id, model_name, num_observations, raw_gps, hmm_route_segments, hmm_route_polyline, classifier_route_segments, observations, metrics, timings_sec, server_timestamp`; per-observation keys `index, timestamp, lat, lon, classifier_arc_id, hmm_arc_id, classifier_confidence`. No reference route or per-point label field. `metrics` are aggregate scores only (by design: "UI displays evaluation results"). `trip_id` `../train/00000001` gave HTTP 400; unknown id gave 404. Process killed afterwards (port 8773 free).
- Residual note: aggregate metrics computed from `.route` are returned to the UI. This is required by the brief ("API calculates prediction accuracy using its internal reference data") and is not label leakage.

### 2. Hardcoding (PASS)

- `grep -E '\b0000[0-9]{4}\b'` over `src/` and `ui/` finds one hit: the docstring example `predict("Student folder/student_test/00000052")` at `pipeline.py:4`. No lookup tables, expected-metric literals or per-trip branches in `src/` or `ui/`.
- Outside the prediction path, `scripts/build_report.py:106` embeds hand-typed prose (trip `00000054`, "79/80") and should be regenerated from files or deleted in favour of the new report build.

### 3. Evaluator-partition safety (PASS)

- `git ls-files | grep -iE 'evaluator|eval_data'` returns nothing; `.gitignore` also blocks `eval_data/` and `evaluator*/`. Only `Student folder/train` (320 files = 80 trips) and `student_test` (40 files = 10 trips) are tracked.
- `PATHSNAP_EVALUATOR_DIR` set to a missing directory or a folder without exactly 10 valid trips raises `RuntimeError` at startup (`api.py:44-68`, called from the startup hook at line 107) and in `/health` and `/trips`; there is no fallback to student_test in that case.
- Edge case (low risk): `PATHSNAP_EVALUATOR_DIR=""` (set but empty) is falsy, so the API silently serves student_test; the UI sidebar shows the partition, which exposes this.
- No script reads an evaluator folder: `paths.partition_dir` raises for anything but `train`/`student_test`; `eval_pipeline.py`, `tune_hmm.py`, `train_heavy.py`, `tune.py` take only those partitions; `evaluate_api.py:82` refuses to run when the variable is set.

### 4. Train/test hygiene (PASS)

- `configs/cv_folds.json`: 5 folds, 80 distinct trips equal to the `train` listing, none in `student_test`; sizes 15/16/16/16/17; trip-grouped by construction (`scripts/make_folds.py`, seed 42).
- `scripts/train_heavy.py` fits on `train` rows only (per-fold subsets and the final refit); student_test only scored. `scripts/tune.py` chooses tuned vs untuned by CV Top-1 (line 72). `model_card.json` records `trained_on: 80 training trips only`.
- HMM: tuning used OOF LightGBM scores on training trips (`scripts/make_oof_scores.py`, `tune_hmm.py --oof`, `results/pipeline_train_oof_*.csv`, `results/hmm_tuning/`); the frozen config was then checked once on student_test (`final_student_test_check.json`). Student_test is allowed for decoder work by the Models PDF p5, so this is compliant either way.
- Results are modest and honestly recorded: tuned LightGBM did not beat untuned; HMM adds +0.0002 Top-1 on OOF training and +0.005 on student_test while raising continuity; HMM was worse on 28/80 training trips.

### 5. Reproducibility (PARTIAL)

- Seeds: 42 in model registry, folds, Optuna sampler, KNN subsample, LogReg.
- Third-party imports found by AST scan of `src/`, `scripts/`, `ui/`, `tests/` (fastapi, folium, httpx, joblib, lightgbm, matplotlib, numpy, optuna, pandas, pydantic, pytest, reportlab, requests, scipy, sklearn, streamlit, streamlit_folium, xgboost) are all covered by `requirements.txt` / `ui/requirements.txt`; `pyarrow` is listed for parquet. Exception: the new untracked `tests/test_report_pdf.py` imports `pypdf`, which is in neither file.
- Gaps: requirements are lower bounds only (`>=`), with no lock file or version record of the environment that produced `results/`; `pyproject.toml` declares no dependencies; `data_cache/` is gitignored and must be rebuilt (README documents the order, ~30 s plus training); `docs/section5_decoder.md` uses `.venv/bin/python` Unix paths; `build_report.py` text cites a non-existent `scripts/export_model.py`.
- README run commands: every script named in the README exists in `scripts/`.

### 6. Deliverable inventory

Present: data processing, features, training for 8 models + baselines, evaluation and plots, Bayesian search code and 34-trial record, HMM code and tuning logs, API, UI, deployment files and README, EDA Folium HTML + generator, model artifacts, CSV/log results, 31 tests.

Missing or incomplete:
- Final consolidated PDF report: done (`Report.pdf`, sources in `docs/report/`).
- A Folium map showing GPS + road network + reference route + predicted route.
- Per-model written subsections, tuned-vs-untuned write-up, HMM discussion in PDF form.
- Training-set scores (for an over/under-fitting argument).
- Sequence-level classifier-vs-HMM comparison saved as a results file.
- UI and API-console screenshots from the real two-laptop run (docs/screenshots/ is untracked and not verified).
- Reconciled student_test numbers (`docs/DATA_QUALITY.md` / `results/api_student_test.csv` are from an older decoder).

### 7. Tests

`python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_report_pdf.py` (PYTHONPATH=src): **31 passed, 10 subtests passed, 2 warnings in 69.8 s** (FastAPI `on_event` deprecation only).

## Blocking gaps (ordered by importance, then effort)

1. **Technical report PDF (Deliverable 2) does not exist in submission-ready form.** Rewrite and rebuild it with all sections: per-model subsections (A1-A2, B5-B6, C7-C9), 3.1 table, 3.3 plots, Bayesian answers, HMM comparison, Folium, UI/API screenshots, AI disclosure. Effort: high, but sources are in place; regenerate from files (retire `scripts/build_report.py`, remove hand-typed numbers).
2. **No Folium map with GPS + road network + reference route + predicted route (Deliverables p6).** Add one script (reuse `ui/map_renderer.py` and `make_eda_map.py`) that renders a student_test trip with all four layers, a legend and layer control; commit the HTML and embed a screenshot. Effort: low-medium.
3. **Nothing has run across two laptops.** Do the README deployment on real hardware and capture UI screenshots plus the Laptop-1 console showing `POST /predict` lines. Effort: medium; also verify the firewall and the evaluator folder layout. The API refuses to start unless the evaluator folder has exactly 10 trip folders with `.track/.nodes/.arcs`; confirm that against the organisers' package, or relax the check to a warning on the day.
4. **Contradictory numbers.** `docs/DATA_QUALITY.md:31` and `results/api_student_test.csv` (HMM Top-1 0.8991) conflict with the frozen decoder (0.9071). Rerun `scripts/evaluate_api.py` with the final config or delete the stale figures. Effort: low.
5. **HMM story needs honest treatment.** Gains are tiny (OOF +0.0002 Top-1; worse on 28/80 trips); sequence-level F1 on trip 00000052 is lower with HMM (0.689 vs 0.881). Save a sequence-level classifier-vs-HMM table for the 10 student_test trips and explain it. Effort: low-medium.
6. **Over/underfitting evidence.** Log training-fold scores (or training-set Top-1/log loss) per model so Q9 can be answered with data. Effort: low-medium (rerun `train_heavy.py` with a train-score line).
7. **Freeze the demo commit.** Tag the commit used on the day so the code matches the demonstration (Deliverables p7). Effort: trivial. Commit the new untracked files only if they are intended.
8. **Minor:** pin dependency versions (`pip freeze` of the working env); add `pypdf` if the new test stays; guard against an empty `PATHSNAP_EVALUATOR_DIR`; add static GPS/trajectory plots to EDA; delete or clearly isolate `ui/mock_api.py`, which contains a stub predictor and reads `.route` (the UI already warns when connected to a mock).

## Things only the team can do

- Write the analysis in their own words and be able to defend it at the viva (Deliverables p7, Policy p2-3); every member must be able to explain the data, features, models, Bayesian search, HMM/Viterbi, API and UI. Only Person D has viva notes so far.
- Run the real two-laptop demonstration, take genuine UI and API-console screenshots, and show API logs live on Laptop 1.
- Test against the organisers' evaluator package (folder layout, `.route` availability, exactly 10 trips) without using it for any tuning; set `PATHSNAP_EVALUATOR_DIR` only on the day.
- Confirm no sharing of models, code or results with other teams (cannot be proven from the repository).
- Disclose AI assistance in the report as the event requires (README has a paragraph; the report needs its own) and make sure no AI-generated explanation is submitted that the team cannot explain.
- Decide whether to keep the committed dataset copy (`Student folder/`, 360 files) in the public repo.
