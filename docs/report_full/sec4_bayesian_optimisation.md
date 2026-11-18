> DRAFT prepared with AI assistance from repository files. The team must rewrite this in its own words and be able to explain every statement (assessment policy).

# 4. Bayesian Hyperparameter Optimisation (chosen model: LightGBM)

This section follows the six questions in Deliverables.pdf, section 4, and the nine-step process in Models and Training Guidelines.pdf, section 7. Short version: we ran a Bayesian search of 34 trials, found a configuration that is statistically indistinguishable from the default, and therefore exported the untuned model. All numbers come from the files cited after them.

**Data partitions used.** The search used only the 80 training trips (5 trip-grouped folds, `configs/cv_folds.json`). The 10 `student_test` trips were used only to compare the refitted tuned and untuned models. The evaluator partition was never used for the search space, objective, parameters, model choice or any threshold (src/pathsnap/tuning.py docstring; docs/notes/models_and_tuning.md, "Protocol").

## 4.1 Which hyperparameters were included in the search?

Eight LightGBM hyperparameters (src/pathsnap/tuning.py, function `suggest`): `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`. They control tree size (`num_leaves`, `min_child_samples`), boosting strength (`learning_rate`, `n_estimators`) and randomisation/regularisation (`subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`). `subsample_freq` stayed fixed at 1, as in the untuned configuration (results/experiments.csv, `config_json`).

## 4.2 What ranges or distributions were considered?

| Hyperparameter | Distribution (src/pathsnap/tuning.py) | Untuned initial value (results/experiments.csv, artifacts/model_card.json) | Best found (results/tuning_summary_lightgbm.json) |
| --- | --- | --- | --- |
| num_leaves | integer, log scale, 15 to 127 | 63 | 61 |
| learning_rate | float, log scale, 0.03 to 0.2 | 0.1 | 0.0952 |
| n_estimators | integer, step 50, 100 to 500 | 300 | 250 |
| min_child_samples | integer, log scale, 10 to 200 | 40 | 63 |
| subsample | float, uniform, 0.5 to 1.0 | 0.8 | 0.848 |
| colsample_bytree | float, uniform, 0.5 to 1.0 | 0.8 | 0.653 |
| reg_lambda | float, log scale, 0.001 to 30 | 1.0 | 0.786 |
| reg_alpha | float, log scale, 0.001 to 10 | not set | 0.0623 |

The team's stated reason: the ranges bracket the defaults and cover the usual under- and over-fitting directions for gradient-boosted trees (docs/notes/models_and_tuning.md, "Bayesian hyper-parameter optimisation").

## 4.3 What objective function was optimised?

Maximise, over the 5 grouped folds, `mean observation Top-1 - 0.05 * mean candidate log loss` (results/tuning_summary_lightgbm.json, `objective`; src/pathsnap/tuning.py, `LOGLOSS_WEIGHT = 0.05`).

Why Top-1: it is the route-matching metric, "does the best-scored candidate equal the reference arc", and it is what the HMM refines. The small log-loss term only breaks near-ties in favour of better-calibrated probabilities, which the HMM uses as emissions. Its contribution is about 0.003 against a Top-1 near 0.87, so Top-1 dominates (src/pathsnap/tuning.py docstring).

Group-aware validation: every fold holds out whole trips, so no points of a trip appear in both training and validation (as required by Models and Training Guidelines.pdf, section 7, step 3). For speed, each trial trains on every 6th GPS point of the training folds (`SEARCH_STRIDE = 6`); validation folds are always scored on all their points.

## 4.4 How many optimisation trials were performed?

34 trials, all completed, using Optuna's TPE sampler (a Bayesian, model-based method) with seed 42 and 8 random start trials (scripts/tune.py; results/tuning_trials_lightgbm.csv; results/tuning_summary_lightgbm.json).

Computational cost: the search took 850.2 s in total; individual trials took between 11.3 s and 64.4 s (results/tuning_summary_lightgbm.json; `user_attrs_seconds` in results/tuning_trials_lightgbm.csv). The script defaults are `--trials 40` and `--timeout 840` (scripts/tune.py), which is consistent with the search having stopped at the time budget before 40 trials, but the exact command line used is [MISSING: not recorded in the repository]. Refitting the tuned model with the same protocol as the untuned one took 11.5 s (results/experiments.csv, `train_time_s`).

All 34 trials lie in a narrow band: CV Top-1 from 0.8703 to 0.8747 and objective from 0.8669 to 0.8715 (computed from results/tuning_trials_lightgbm.csv). The fold-to-fold standard deviation of Top-1 is about 0.037 (results/experiments.csv, `cv_top1_std`), roughly eight times the 0.0044 spread between the best and worst trial.

## 4.5 What were the best hyperparameter values obtained?

Trial 18 (zero-based) had the best objective, 0.87152, with CV Top-1 0.87472 on the stride-6 search protocol (results/tuning_summary_lightgbm.json). Its parameters are in the last column of the table in 4.2: `num_leaves` 61, `learning_rate` 0.0952, `n_estimators` 250, `min_child_samples` 63, `subsample` 0.848, `colsample_bytree` 0.653, `reg_lambda` 0.786, `reg_alpha` 0.0623. The next best trials (27, 17, 23, 19) reached objectives 0.8713, 0.8711, 0.8710, 0.8710, i.e. within 0.0005 of the best (results/tuning_trials_lightgbm.csv), so many different settings are almost equally good.

## 4.6 Did the tuned model outperform the untuned model?

No. Both models were refitted under one protocol (training on every 3rd point, same folds, same seed 42) and evaluated on the same partitions with the same code (results/experiments.csv, latest rows `lightgbm` and `lightgbm_tuned`):

| Metric | Untuned | Tuned | Difference (tuned - untuned) |
| --- | --- | --- | --- |
| CV macro F1 (5 grouped folds) | 0.9270 | 0.9262 | -0.0007 |
| CV Top-1 | 0.8748 | 0.8746 | -0.0002 |
| CV log loss | 0.0650 | 0.0646 | -0.0004 |
| Student-test Top-1 | 0.9021 | 0.9003 | -0.0018 |
| Student-test macro F1 | 0.9406 | 0.9416 | +0.0010 |
| Student-test log loss | 0.04804 | 0.04809 | +0.00005 |
| Student-test route continuity (official) | 0.9558 | 0.9588 | +0.0029 |
| Student-test route breaks | 774 | 741 | -33 |
| Fit time, seconds | 12.3 | 11.5 | -0.9 |
| Inference time per trip, seconds | 0.091 | 0.099 | +0.008 |

Reading the table honestly:

- On CV Top-1, the metric we select on, the tuned model is 0.0002 lower. On student-test Top-1 it is 0.0018 lower. Macro F1, log loss and continuity move by tiny amounts in both directions. None of these gaps is large compared with the fold spread of about 0.037, and each model was trained with a single seed, so we do not claim that either is better.
- Inference time is not a reliable differentiator. In the earlier run of the same models the order was reversed (0.088 s untuned, 0.082 s tuned per trip; results/experiments.csv, rows with timestamps 10:34 and 11:25), so the 0.008 s difference is timing noise on this machine.
- The route-continuity improvement (+0.0029, 33 fewer breaks) is a student-test number. Continuity was not the selection criterion and was not used to make the decision, so we treat it as a side observation only.
- The tuned model was chosen only if its CV Top-1 beat the untuned one; it did not (0.8746 vs 0.8748). The untuned model is therefore the exported model (scripts/tune.py, selection rule; artifacts/model_card.json, `"model": "lightgbm"`; results/tune_log_lightgbm.txt, last lines). The decision used CV only, never `student_test`.

**Possible reasons (not tested).** The default configuration may already sit on a flat plateau for this feature set, which is what the narrow band of 34 trials suggests. The search also ran on every 6th point while the refit used every 3rd, so the search objective is a cheaper proxy of the final protocol. We did not run experiments that would separate these explanations.

**Other notes.**
- A tuning run for HistGradientBoosting was started and stopped unfinished; its search record was not saved and is not used anywhere (docs/notes/models_and_tuning.md).
- Why the CV and student-test numbers differ in level: the exported model scores about 94.5% Top-1 on its own training points versus about 87.5% in grouped CV (docs/notes/models_and_tuning.md, "Overfitting evidence"); held-out trips are the correct estimate of deployment accuracy.
- A negative tuning result is still a result: it tells us that the default model is not leaving obvious accuracy on the table within this search space.

## 4.7 Evidence checklist for the report figures

- Trial history plot (objective per trial and best so far): `results/plots/tuning_trials_lightgbm.png`, produced by `python scripts/plot_tuning.py` from `results/tuning_trials_lightgbm.csv`. [MISSING: a parameter-importance plot.]
- Tuned vs untuned comparison chart: [MISSING: not in results/plots/; the table in 4.6 can be turned into a bar chart from results/experiments.csv].

## 4.8 How to reproduce

From the repository root, Python 3.12 (README.md, "Install" and "Reproduce the experiments"):

```
pip install -r requirements.txt
pip install -e .
python scripts/make_candidates.py --jobs 6      # candidate and feature tables
python scripts/make_folds.py                    # configs/cv_folds.json
python scripts/train_heavy.py --models lightgbm hist_gb xgboost    # untuned LightGBM row in results/experiments.csv
python scripts/tune.py --model lightgbm         # Optuna TPE search, refit, tuned vs untuned, export
```

`scripts/tune.py` writes `results/tuning_trials_lightgbm.csv`, `results/tuning_summary_lightgbm.json` and the `lightgbm_tuned` row of `results/experiments.csv`, and exports the winner to `artifacts/`. Add `--no-export` to leave `artifacts/` untouched. Re-running will append new rows and may produce a slightly different search because the time budget limits the number of trials.
