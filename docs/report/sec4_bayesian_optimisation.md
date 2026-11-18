# 4. Bayesian Hyper-parameter Optimisation (LightGBM)

**1. Which hyper-parameters were searched, and 2. in which ranges?**

| Hyper-parameter | Range | Scale | Untuned default | Best found |
| --- | --- | --- | ---: | ---: |
| `num_leaves` | 15 to 127 | log | 63 | 61 |
| `learning_rate` | 0.03 to 0.2 | log | 0.1 | 0.0952 |
| `n_estimators` | 100 to 500 | step 50 | 300 | 250 |
| `min_child_samples` | 10 to 200 | log | 40 | 63 |
| `subsample` | 0.5 to 1.0 | linear | 0.8 | 0.848 |
| `colsample_bytree` | 0.5 to 1.0 | linear | 0.8 | 0.653 |
| `reg_lambda` (L2) | 0.001 to 30 | log | 1.0 | 0.786 |
| `reg_alpha` (L1) | 0.001 to 10 | log | 0.0 | 0.062 |

*Table 4. Search space of the Optuna study (`src/pathsnap/tuning.py`) with the untuned and best values.*

**3. Objective.** Maximise grouped-CV Top-1 minus 0.05 times candidate log loss (mean over the 5 trip-grouped folds of the 80 training trips, every 6th training point to keep a trial fast). The student test and evaluator trips played no part in the search.

**4. Number of trials.** 34 trials of the TPE sampler (Optuna) in 850 s (`results/tuning_trials_lightgbm.csv`, `results/tuning_summary_lightgbm.json`, log `results/tune_log_lightgbm.txt`).

**5. Best values.** The best trial reached an objective of 0.8715 (CV Top-1 0.8747) with the values in the last column of Table 4.

![Figure 7. Optuna search history: objective of each trial and best value so far (partition: training, grouped CV).](docs/report/fig/tuning_trials_lightgbm.png)

**6. Did the tuned model outperform the untuned model?** No. Both were refitted with the same protocol (every 3rd training point, same folds, same partitions):

| Metric | Untuned | Tuned |
| --- | ---: | ---: |
| CV macro-F1 | 0.9270 | 0.9262 |
| CV Top-1 | 0.8748 | 0.8746 |
| Student-test Top-1 | 0.9021 | 0.9003 |
| Student-test macro-F1 | 0.9406 | 0.9416 |
| Student-test log loss | 0.0480 | 0.0481 |
| Inference time (s per trip) | 0.0908 | 0.0992 |
| Training time (s) | 12.3 | 11.5 |

*Table 5. Untuned versus tuned LightGBM (`results/metrics_table.csv`).*

The CV difference (0.0002 Top-1) is far smaller than the fold-to-fold standard deviation (0.0374), and the tuned model is slightly worse on the student test, so tuning gave no gain. The rule fixed before the search was "export the configuration with the higher grouped-CV Top-1", so the untuned configuration was exported (`artifacts/model_card.json`). The most likely reason is that the features, not the hyper-parameters, limit accuracy (the ablation shows large feature effects and none from model settings); we did not test this further. Reproduce with `python scripts/tune.py --model lightgbm` and `python scripts/plot_tuning.py`.
