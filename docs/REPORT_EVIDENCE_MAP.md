# Report evidence map (Deliverables.pdf sections vs files in this repository)

Purpose: every number in the PDF report must be verifiable from the repo. This maps each required section to the
script that produces the evidence and the file that holds it. It is an index, not report text: the analysis
and explanations must be written by the team in their own words and every member must be able to explain them.

Status as of this commit (integration/a-b): DONE = evidence exists and was reproduced; PARTIAL; MISSING.

| Report section | Evidence (script -> file) | Status |
| --- | --- | --- |
| 2.2 preprocessing, patterns, features | `src/pathsnap/{io,geo,labels,candidates,features}.py`; `docs/notes/candidates_labels_features.md`; `results/label_stats.csv`, `results/label_quality.csv`, `results/candidate_sweep.csv` | DONE |
| 2.2 missing-value analysis, GPS and road-network plots, Folium map | Person C branch: `scripts/inventory.py`, `plot_eda.py`, `make_eda_map.py` -> `results/data_inventory.*`, `reports/` | PARTIAL (on C's branch, not merged) |
| 3.1 comparison table (>= 5 models) | `scripts/train_heavy.py`, `baseline_nearest.py`, `rerun_all_models.sh` -> `results/experiments.csv`, `results/metrics_table.csv` | DONE (5 boosting/ensemble models + 2 baselines); C adds 3 simple models |
| 3.2 per-model questions A-C | numbers in `results/metrics_table.csv`, `docs/notes/models_and_tuning.md`; overfitting evidence: train Top-1 94.5 % vs CV 87.5 % | MISSING: team writes the answers |
| 3.3 plots (CV macro-F1, test Top-1/F1, log loss, confusion matrices) | `scripts/analysis_plots.py` -> `results/plots/model_comparison.png`, `confusion_all_models.png`, `calibration_lightgbm.png` | DONE |
| 4 Bayesian optimisation | `scripts/tune.py`, `src/pathsnap/tuning.py` -> `results/tuning_trials_lightgbm.csv`, `tuning_summary_lightgbm.json`, tuned row in `experiments.csv` | DONE (34 trials, no gain over untuned) |
| 5 HMM / Viterbi | `src/pathsnap/decoder/`, `docs/section5_decoder.md`; classifier vs HMM on `student_test`: `results/hmm_tuning/final_student_test_check.csv` (one check after the settings were frozen) | DONE |
| 5 decoder tuning on training trips | `scripts/tune_hmm.py`, `configs/decoder_best.json`, `results/hmm_tuning/` (8-trip pilot sweep, then all 80 training trips with out-of-fold scores; failure cases in `failure_analysis_top10.csv`) | DONE (verified by re-running `student_test`) |
| 6 Folium prediction map (GPS, network, reference, predicted) | UI: `ui/map_renderer.py` (Person D branch); needs real API output | PARTIAL |
| 7 API, UI, deployment screenshots | `src/pathsnap/api.py` (Person C branch), `ui/` (Person D branch), `scripts/run_api.*` | MISSING (needs two-laptop run and screenshots) |
| 8 AI-use disclosure | root `README.md` states AI assistance; team to confirm wording | PARTIAL |

## Rules the evidence must respect
- Official route continuity = share of ALL adjacent predicted pairs that are the same arc or head-to-tail connected
  (`pathsnap.decoder.metrics.route_continuity`). Columns named `*_distinct*` in `experiments.csv` are a stricter, non-official variant: do not quote them as route continuity.
- `student_test` is a development comparison set. The evaluator partition is never used for any choice.
- Report the strong non-learned baseline (nearest arc + heading rule, 83.0 % `student_test` Top-1) alongside the 53.6 % nearest-arc baseline.
- State the limitations: three boosting models are statistically tied (fold spread about 3.7 points); tuning gave no gain; the tuned HMM gives a large continuity gain (breaks 8,245 to 5,505 on 80 training trips) but only a small Top-1 gain (0.8790 to 0.8792 out of fold; 0.9021 to 0.9071 on `student_test`), and it is worse on 28 of 80 training trips; training trips 54 and 33 have unreliable reference alignment; Extra Trees and Random Forest were run with 50 trees and every 6th point; KNN (Person C) was fit on a 3,000-row subsample.

## Reproducing from a fresh clone (verified)
A fresh clone of this branch passes its tests (27 passed, 1 skipped for a file that needs generated data), builds a wheel containing all
subpackages, and `scripts/make_candidates.py` regenerates all 180 candidate/label tables identical to the ones used for the reported results.
