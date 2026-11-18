# Models, selection and tuning (Person A)

Code: `scripts/train_heavy.py`, `scripts/baseline_nearest.py`, `scripts/tune.py`, `scripts/ablation.py`, `scripts/analysis_plots.py`; `src/pathsnap/models/registry.py`, `tuning.py`, `clf.py`. Every number below is in `results/experiments.csv` (seeds, configs, timings included).

## Protocol
- 5-fold **trip-grouped** CV over the 80 training trips (`configs/cv_folds.json`, seed 42, folds balanced by number of GPS points); `student_test` (10 trips) is the development hold-out. Evaluator data is never used.
- Primary selection metric: **observation Top-1 accuracy** (is the best-scored candidate the reference arc?). It is the route-matching objective and is directly what the HMM refines. Secondary: log loss (the HMM uses the probabilities), macro F1, route continuity, speed.
- The three boosting models and the nearest-road baseline use every 3rd GPS point for training; Extra Trees and Random Forest were time-boxed (50 trees, every 6th point). Scoring always uses every point.

## Results (student_test = 10 trips, CV = 5 grouped folds)

| Model | Family | CV Top-1 | test Top-1 | test macro-F1 | test log loss | test route continuity | fit time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistGradientBoosting | boosting | 87.55 % | 90.07 % | 0.940 | 0.0488 | 82.9 % | 17 s |
| **LightGBM (selected)** | boosting | 87.48 % | 90.21 % | 0.941 | 0.0480 | 82.7 % | 12 s |
| XGBoost | boosting | 87.47 % | 90.16 % | 0.941 | 0.0483 | 83.4 % | 18 s |
| Random Forest (50 trees) | bagging | 87.33 % | 88.81 % | 0.935 | 0.0561 | 83.0 % | 22 s |
| Extra Trees (50 trees) | randomised trees | 87.03 % | 88.41 % | 0.933 | 0.0561 | 81.8 % | 24 s |
| Nearest road (no learning) | baseline | 55.36 % | 53.60 % | n/a | n/a | 45.7 % | n/a |

Fold-to-fold standard deviation of Top-1 is about 3.7 points for every model (folds range from about 82 % to 91 %): some trips are much harder than others.

## Selection and honesty notes
- The three boosting models are **statistically tied**: their gaps (about 0.1 points) are far inside the fold spread. HistGradientBoosting has a marginally higher CV Top-1; LightGBM has the best `student_test` Top-1 and log loss and the fastest fit. We selected LightGBM; the team should say so and not claim it is significantly better.
- Boosting beats the bagging/randomised-tree models by about 1.5 points on `student_test` and has clearly lower log loss (0.048 vs 0.056); the tree ensembles were run in a reduced configuration, so part of that gap may come from the smaller setup.
- Calibration of the selected model on `student_test`: ECE 0.0027 (`results/plots/calibration_lightgbm.png`), so its probabilities are usable as HMM emissions. Candidate-level precision 0.903, recall 0.881 at threshold 0.5.
- Overfitting evidence: LightGBM scores 94.5 % Top-1 on its own training points vs 87.5 % in grouped CV, so it fits seen trips better than unseen ones (a gap of about 7 points). Held-out trips are the right estimate of deployment accuracy.

## Bayesian hyper-parameter optimisation (LightGBM)
- Method: Optuna TPE (seed 42, 8 random start trials), objective `mean CV Top-1 - 0.05 * log loss` over the 5 grouped folds, search on every 6th training point, 34 trials in 850 s (`results/tuning_trials_lightgbm.csv`, `results/tuning_summary_lightgbm.json`).
- Space: num_leaves 15-127 (log), learning_rate 0.03-0.2 (log), n_estimators 100-500, min_child_samples 10-200 (log), subsample 0.5-1, colsample_bytree 0.5-1, reg_lambda and reg_alpha (log). The ranges bracket the defaults and cover the usual under/over-fitting directions for GBDTs.
- Best: num_leaves 61, learning_rate 0.095, n_estimators 250, min_child_samples 63, subsample 0.85, colsample_bytree 0.65, reg_lambda 0.79, reg_alpha 0.06.
- **Result: no gain.** Refit with the same protocol as the untuned model: CV Top-1 87.46 % (untuned 87.48 %), `student_test` Top-1 90.03 % (90.21 %), macro-F1 0.9416 (0.9406), log loss 0.0481 (0.0480). All 34 trials sit between 87.0 % and 87.5 % CV Top-1; the defaults were already near the plateau. By our CV-only rule the **untuned model is exported** (`artifacts/`). A tuning attempt for HistGradientBoosting was started and stopped unfinished; its search record was not saved and is not used anywhere.

## Limitations
- A single seed for each model; no repeated-CV confidence intervals.
- Models see only features computed from GPS plus the road geometry (no learned road-type or speed-limit information, which the dataset does not provide).
- The MLP (neural family) and the simple models (Logistic Regression, Gaussian NB, KNN, Person C) are not in these tables yet.

## Likely viva questions
1. Why Top-1 and not accuracy or F1 for model selection? (Class imbalance; Top-1 matches the matching task.)
2. Why group the CV by trip? (Points of one trip are strongly correlated; random splits would leak.)
3. Why did tuning not help, and is that a failure? (Plateau; honest negative result.)
4. Why no class weights? (Probabilities for the HMM.)
5. What does the 94.5 % vs 87.5 % gap tell you? (Generalisation across trips; some trips harder or with noisy reference routes.)
