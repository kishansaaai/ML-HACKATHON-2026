# 4. Bayesian Optimisation

**Model tuned:** LightGBM, the chosen model. **Method:** Optuna with the TPE sampler (`scripts/tune.py`, `src/pathsnap/tuning.py`), 34 trials in 850 s. **Searched:** number of trees, learning rate, number of leaves, minimum child samples, subsample, column sample, L1 and L2 regularisation (bounds in `src/pathsnap/tuning.py`). **Objective:** mean grouped-CV Top-1 minus 0.05 times log loss, on the 80 training trips only, with every 6th training point to keep trials fast; the student test and the evaluator trips were not used in the search.

![Optuna search history](docs/report/fig/tuning_trials_lightgbm.png)

The best trial reached an objective of 0.8715 (CV Top-1 0.8747). Refitted with the same protocol as the untuned model (every 3rd point):

| | Untuned (default config) | Tuned |
| --- | ---: | ---: |
| CV Top-1 | 0.8748 | 0.8746 |
| CV macro-F1 | 0.9270 | 0.9262 |
| Student-test Top-1 | 0.9021 | 0.9003 |
| Student-test macro-F1 | 0.9406 | 0.9416 |
| Student-test log loss | 0.0480 | 0.0481 |

**Decision.** Tuning gave no gain: the CV difference (0.0002 Top-1) is far smaller than the fold-to-fold standard deviation (0.0374). The rule fixed before the search was "export the configuration with the higher grouped-CV Top-1", so the untuned configuration was exported (`artifacts/model_card.json`). The most likely reason is that the features, not the hyper-parameters, limit accuracy (the ablation shows large feature effects and none from model settings); we did not test this further. Reproduce with `python scripts/tune.py --model lightgbm` and `python scripts/plot_tuning.py`.
