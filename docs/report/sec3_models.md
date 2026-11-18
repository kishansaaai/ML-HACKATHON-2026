# 3. Model Experimentation and Comparison

## 3.1 Protocol and comparison table

Task: for every GPS point and candidate arc, output the probability that the arc is the correct one. Metrics: observation-level Top-1 accuracy (the highest-probability candidate of each point is the correct arc), candidate-level macro-F1, precision, recall and log loss, and route continuity (official definition above). All models use the same candidate and feature tables, seed 42 and the same trip-grouped folds, and every run is logged in `results/experiments.csv`. The student-test columns are a development comparison, never used to choose a model.

| Model | Family | CV macro-F1 | CV Top-1 | Test Top-1 | Test macro-F1 | Test log loss | Test route continuity | Inference s/trip |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lightgbm | Boosting (leaf-wise GBDT) | 0.9270 | 0.8748 | 0.9021 | 0.9406 | 0.0480 | 0.9558 | 0.0908 |
| hist_gb | Boosting (histogram GBDT, sklearn) | 0.9276 | 0.8755 | 0.9007 | 0.9399 | 0.0488 | 0.9578 | 0.1261 |
| xgboost | Boosting (XGBoost, depth-wise) | 0.9269 | 0.8747 | 0.9016 | 0.9413 | 0.0483 | 0.9587 | 0.0431 |
| extra_trees | Tree ensemble (randomised) | 0.9245 | 0.8703 | 0.8841 | 0.9328 | 0.0561 | 0.9531 | 0.0617 |
| random_forest | Tree ensemble (bagging) | 0.9265 | 0.8733 | 0.8881 | 0.9345 | 0.0561 | 0.9563 | 0.0802 |
| logistic_regression | Linear | 0.9021 | 0.8453 | 0.8548 | 0.9107 | 0.0736 | n/a | 0.0030 |
| gaussian_nb | Probabilistic | 0.7475 | 0.8191 | 0.8134 | 0.7497 | 1.0193 | n/a | 0.0499 |
| knn | Instance-based | 0.8614 | 0.8125 | 0.8217 | 0.8642 | 0.1367 | n/a | 0.4107 |

Route continuity was re-scored with the official definition for the five heavy models; for logistic regression, naive Bayes and KNN it was not re-scored, so it is left out. Extra Trees and Random Forest were run with 50 trees on every 6th point; KNN was fitted on a 3,000-row sample per fold; these are limitations of those rows.

Non-learned baselines (Top-1):

| Baseline | CV Top-1 | Test Top-1 |
| --- | ---: | ---: |
| Nearest road (not learned) | 0.5536 | 0.5360 |
| Nearest road + heading rule (not learned) | 0.8152 | 0.8301 |

**Reading the table.** LightGBM, HistGradientBoosting and XGBoost are statistically tied: their CV Top-1 values (0.8748, 0.8755, 0.8747) differ by 0.1 points, while the fold-to-fold standard deviation is 0.0374. LightGBM was chosen as the single final model because it is the strongest on student-test Top-1 and log loss among the tied three, trains fastest (about 12 s) and is simple to export, not because it is clearly better. The two bagging ensembles are 0.15 to 0.45 points lower in CV and 1.4 to 1.8 points lower on the student test. Logistic regression (0.8453 CV Top-1) is a strong simple model, KNN and naive Bayes are weaker, and a simple rule (nearest road plus heading) already reaches 0.8301 on the student test, so the learned model adds about 7 points over a good hand rule.

## 3.2 Per-model notes

**LightGBM (final).** *Why:* leaf-wise gradient boosting handles mixed-scale tabular features without scaling and captures interactions such as distance combined with heading. *Configuration:* 300 trees, learning rate 0.1, 63 leaves, min child samples 40, subsample 0.8, column sample 0.8, L2 1.0, seed 42, every 3rd training point. *Result:* CV Top-1 0.8748, student-test Top-1 0.9021, macro-F1 0.9406, log loss 0.0480. *Over/under-fitting:* the exported model reaches about 94.5 % Top-1 on its own training trips against 87.5 % in grouped CV (`docs/notes/models_and_tuning.md`), so it fits the training trips more tightly than unseen ones; student-test Top-1 is higher than CV because those 10 trips are easier on average, not because of a better model. *Confusion matrix (candidate level, student test):* TN 191,392, FP 1,814, FN 2,279, TP 16,862 (precision 0.9029, recall 0.8809). Calibration error is small (ECE 0.0027, `results/plots/calibration_lightgbm.png`), which matters because the HMM uses the probabilities as emission scores.

**HistGradientBoosting (scikit-learn).** Histogram-based boosting, a close relative of LightGBM; 200 iterations, learning rate 0.1, 63 leaf nodes, min leaf 40, L2 1.0. CV Top-1 0.8755, test 0.9007. Indistinguishable from LightGBM, slower to fit (20.7 s).

**XGBoost.** Depth-wise boosting; 300 trees, learning rate 0.1, depth 8, min child weight 5, subsample 0.8, column sample 0.8, L2 1.0. CV 0.8747, test 0.9016; fastest inference (0.043 s/trip) but slowest to fit (24.7 s). No accuracy advantage.

**Extra Trees and Random Forest.** Bagged trees (50 trees, every 6th point; min leaf 3; max features 0.5 for Extra Trees and square root for Random Forest). CV Top-1 0.8703 and 0.8733, test 0.8841 and 0.8881; clearly below boosting on the student test and worse in log loss (0.0561), so their probabilities are less useful as emissions.

**Logistic regression (Person C).** Scaled features, L2-regularised (C = 1.0, L-BFGS), linear decision boundary. CV Top-1 0.8453, test 0.8548, log loss 0.0736; the fastest model (0.003 s/trip). The gap to boosting (about 3 points) shows how much the non-linear interactions matter.

**Gaussian naive Bayes (Person C).** CV Top-1 0.8191, test 0.8134, but log loss 1.019: the independence assumption is badly violated (many features are correlated), so its probabilities are over-confident and unsuitable for an HMM.

**KNN (Person C).** Scaled features, 15 neighbours, distance-weighted, fitted on a 3,000-row sample per fold. CV Top-1 0.8125, test 0.8217; slow at prediction (0.41 s/trip) and limited by the small sample.

Training-set scores were recorded only for LightGBM, so over-fitting for the other models cannot be quantified from the repository. No loss curves exist because the models were fitted without epoch-wise logging.

## 3.3 Plots

![Model comparison: CV macro-F1, student-test Top-1 and log loss](docs/report/fig/model_comparison.png)

![Candidate-level confusion matrices of the heavy models](docs/report/fig/confusion_all_models.png)
