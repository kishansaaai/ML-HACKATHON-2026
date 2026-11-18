> DRAFT prepared with AI assistance from repository files. The team must rewrite this in its own words and be able to explain every statement (assessment policy).

# 3. Model Experimentation and Comparison

## 3.0 Protocol

- Task for the classifier: for each GPS point and each candidate arc, output the probability that the candidate is the reference arc. The best-scored candidate per point is the observation-level prediction.
- Cross-validation: 5 folds grouped by trip over the 80 training trips (configs/cv_folds.json, seed 42). Student test (10 trips) is the development hold-out for model comparison. The evaluator partition is never used (docs/notes/models_and_tuning.md).
- Primary selection metric: observation-level Top-1 accuracy, because it is the route-matching objective and the decoder refines it. Secondary: log loss (the decoder uses the probabilities), macro F1, route continuity, speed (docs/notes/models_and_tuning.md).
- Macro F1, precision, recall and log loss are candidate-level (each candidate row is correct or incorrect). Top-1 is per GPS point (docs/section5_decoder.md, "Evaluation definitions"). Positives are about 9 % of candidates (docs/notes/candidates_labels_features.md).
- Route continuity is the official definition: share of adjacent predicted pairs that are the same arc or head-to-tail connected, computed on classifier-only predictions here. The `*_distinct*` columns in results/experiments.csv are a stricter non-official variant and are not used.
- All models were trained on the same candidate and feature tables with the same folds, seed 42, and logged to results/experiments.csv. Where a model appears twice in that file (first run, then a re-score with the official continuity metric), this section uses the later row; the later rows reproduce the earlier metrics (docs/notes/models_and_tuning.md). Rounded values are in results/metrics_table.csv.
- Training-sample size: boosting models, Logistic Regression and Gaussian NB use every 3rd GPS point (678,859 rows). Extra Trees and Random Forest use every 6th point (339,848 rows) and 50 trees. KNN is fitted on 3,000 rows (results/experiments.csv).

## 3.1 Overall model comparison

Ten rows: eight learned models and two non-learned baselines. LightGBM (untuned) is the final model. The tuned LightGBM is listed for reference; its search is covered in Section 4. Source for every cell: results/experiments.csv (latest row for each model); the first six rows also match results/metrics_table.csv.

| Model | Family | Key configuration | CV macro F1 | CV Top-1 | Student-test Top-1 | Student-test macro F1 | Student-test log loss | Test route continuity | Inference (s per trip) | Fit time (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LightGBM (final)** | Boosting, leaf-wise GBDT | 300 trees, lr 0.1, 63 leaves, min child 40, subsample 0.8, colsample 0.8, lambda 1.0 | 0.9270 | 0.8748 | 0.9021 | 0.9406 | 0.0480 | 0.9558 | 0.0908 | 12.3 |
| HistGradientBoosting | Boosting, histogram GBDT (sklearn) | 200 iterations, lr 0.1, 63 leaf nodes, min leaf 40, L2 1.0 | 0.9276 | 0.8755 | 0.9007 | 0.9399 | 0.0488 | 0.9578 | 0.1261 | 20.7 |
| XGBoost | Boosting, depth-wise | 300 trees, lr 0.1, depth 8, min child weight 5, subsample 0.8, colsample 0.8, lambda 1.0 | 0.9269 | 0.8747 | 0.9016 | 0.9413 | 0.0483 | 0.9587 | 0.0431 | 24.7 |
| Extra Trees | Tree ensemble, randomised | 50 trees, min leaf 3, max features 0.5, stride 6 | 0.9245 | 0.8703 | 0.8841 | 0.9328 | 0.0561 | 0.9531 | 0.0617 | 25.4 |
| Random Forest | Tree ensemble, bagging | 50 trees, min leaf 3, max features sqrt, stride 6 | 0.9265 | 0.8733 | 0.8881 | 0.9345 | 0.0561 | 0.9563 | 0.0802 | 31.4 |
| Logistic Regression | Linear / probabilistic | C 1.0, lbfgs, 500 iterations, standardised features | 0.9021 | 0.8453 | 0.8548 | 0.9107 | 0.0736 | 0.9449 | 0.0030 | 12.0 |
| Gaussian Naive Bayes | Probabilistic / generative | var smoothing 1e-9 | 0.7475 | 0.8191 | 0.8134 | 0.7497 | 1.0193 | 0.9342 | 0.0499 | 3.3 |
| KNN | Instance-based, distance-weighted | k 15, Euclidean, standardised, **3,000-row fit subsample** | 0.8614 | 0.8125 | 0.8217 | 0.8642 | 0.1367 | 0.9175 | 0.4107 | 0.08 |
| Nearest road (baseline, no learning) | Non-learned baseline | nearest arc | N/A: rule outputs one arc, not candidate probabilities | 0.5536 | 0.5360 | N/A | N/A | 0.8025 | N/A | N/A |
| Nearest road + heading rule (baseline, no learning) | Non-learned baseline | nearest arc penalised by heading mismatch | N/A: as above | 0.8152 | 0.8301 | N/A | N/A | 0.9108 | N/A | N/A |
| LightGBM, Bayesian-tuned (see Section 4) | Boosting | 250 trees, lr 0.0952, 61 leaves, min child 63 | 0.9262 | 0.8746 | 0.9003 | 0.9416 | 0.0481 | 0.9588 | 0.0992 | 11.5 |

Notes on the table:

- "Inference (s per trip)" is the value in column `infer_time_s_per_trip`. For the same model it varies between runs (for example Extra Trees 0.0432 in the first run and 0.0617 in the re-score; HistGradientBoosting 0.1052 and 0.1261), so small differences in inference time are noise (results/experiments.csv).
- "Fit time" is for one full-training-set fit as logged, not the cross-validation total. Extra Trees and Random Forest fit times are for a smaller sample (stride 6) than the boosting models (stride 3), so fit times are not directly comparable. KNN's fit time is small because it only stores 3,000 rows; its cost is at inference (results/experiments.csv).
- Per-fold CV standard deviation of Top-1 is about 0.037 for every boosting and tree model (results/experiments.csv, `cv_top1_std`, 0.0374 for LightGBM). Folds range from about 82 % to 91 % (docs/notes/models_and_tuning.md).
- Baseline CV Top-1 for nearest road and for nearest road + heading rule are computed with the same folds, with no fitting.
- For the baselines, macro F1 and log loss are N/A because the rules produce a single chosen arc, not a probability for each candidate. Their Top-1 and route continuity are computed on the same per-point basis as for the learned models.
- Gaussian NB has a high candidate recall (0.9706) but very low precision (0.4089) and a log loss of 1.0193 on student test, so its probabilities are poor, even though its Top-1 (0.8134) is not far below the heading baseline (0.8301) (results/experiments.csv).

### Why LightGBM was selected, and how strong that claim is

LightGBM has the best student-test Top-1 (0.9021) and the lowest log loss (0.0480) among the five main models, and the fastest full fit among them (12.3 s). HistGradientBoosting has a marginally higher CV Top-1 (0.8755 vs 0.8748) (results/experiments.csv). The three boosting models are statistically tied: differences of about 0.1 points are far inside the fold spread of about 3.7 points. The team chose LightGBM and should not claim it is significantly better than HistGradientBoosting or XGBoost (docs/notes/models_and_tuning.md). Student test was used as a development comparison, which the guidelines allow; it has only ten trips.

The evidence does support three smaller conclusions:

1. Boosting beats the 50-tree bagging/randomised models by about 1.5 points of student-test Top-1 and has lower log loss (0.048 vs 0.056). Because the tree ensembles were run in a reduced setting, part of that gap may come from the setup, not the algorithm (docs/notes/models_and_tuning.md).
2. All the tree-based models are well above the heading-rule baseline; the learned classifier adds about 7 points on top of it (docs/notes/models_and_tuning.md). A model must therefore be judged against 0.8301, not only against the weak 0.5360 nearest-road baseline.
3. Logistic Regression, Gaussian NB and KNN are below the three boosting models on CV Top-1 and student-test Top-1. Gaussian NB and KNN are below the heading baseline on student test (0.8134 and 0.8217 vs 0.8301) (results/experiments.csv).

The final model's probabilities are usable by the decoder: expected calibration error on student test is 0.0027 (docs/notes/models_and_tuning.md; results/plots/calibration_lightgbm.png).

The exported model is the untuned LightGBM, because by the CV-only rule the tuned model did not beat it (artifacts/model_card.json; docs/notes/models_and_tuning.md).

## 3.2 Per-model subsections

For every model, A and B are partly textbook reasoning tied to the task, and C is measured. All C numbers come from results/experiments.csv. The team should replace the reasoning in A with its own explanation based on what it observed.

Common facts for all models:

- Model input: the 27 features of a (point, candidate) pair (src/pathsnap/features.py). Output: probability that the candidate is the reference arc.
- Student-test observation-level Top-1, macro F1 and log loss are in the table of 3.1 and repeated per model below. Candidate miss rate is 0.0001 (0.000104) on student test for every model, since it depends only on candidate generation (results/experiments.csv).
- Overfitting evidence in the repository exists only for LightGBM (see its subsection). For the other models, no training-set score was recorded: [MISSING: training-set Top-1 for HistGradientBoosting, XGBoost, Extra Trees, Random Forest, Logistic Regression, Gaussian NB and KNN]. For them, the comparison between CV and student-test scores is used, with the caveat in 3.4.
- No class weights are used for any model (docs/notes/candidates_labels_features.md).

### 3.2.1 LightGBM (final model)

**A. Model understanding**

*Why is it suitable?* Each (point, candidate) row is a tabular example with mixed-scale numeric features, and interactions matter: heading mismatch only helps when the neighbour-fit is also known. A boosted tree model learns such interactions without scaling. It gives probabilities that the decoder needs, and it is fast at inference (0.0908 s per trip, results/experiments.csv). The ablation shows that no single feature group is enough (best single group, neighbours, gets 0.8429 CV Top-1 against 0.8748 for all, results/ablation.csv), which fits a model that combines groups.

*Strengths and limitations (from our runs).* Strengths: best student-test Top-1 and log loss of the five main models, fastest full fit (12.3 s), well-calibrated output (ECE 0.0027, docs/notes/models_and_tuning.md). Limitations: it fits training trips better than unseen ones (train Top-1 94.5 % vs CV 87.5 %, docs/notes/models_and_tuning.md; docs/REPORT_EVIDENCE_MAP.md); it is tied with two other boosting models, so its selection is not statistically justified (3.1); Bayesian tuning gave no improvement, so hyperparameters were not the bottleneck (results/tuning_summary_lightgbm.json).

**B. Training and configuration**

*Important hyperparameters and initial values:* n_estimators 300, learning_rate 0.1, num_leaves 63, min_child_samples 40, subsample 0.8 (every iteration), colsample_bytree 0.8, reg_lambda 1.0, seed 42, every 3rd training point (678,859 rows) (results/experiments.csv; src/pathsnap/models/registry.py).

*Loss or objective:* LightGBM's binary classification objective, which minimises binary log loss, since `LGBMClassifier` is used with no objective set in src/pathsnap/models/registry.py. [Team to confirm from the LightGBM version in use.] Hyperparameter search (Section 4) used the objective mean CV Top-1 minus 0.05 times log loss (results/tuning_summary_lightgbm.json).

**C. Experimental results**

| Metric | Value |
| --- | ---: |
| Student-test Top-1 | 0.9021 |
| Student-test macro F1 | 0.9406 |
| Student-test log loss | 0.0480 |
| Student-test candidate precision / recall (threshold 0.5) | 0.9029 / 0.8809 |
| Student-test official route continuity (classifier only) | 0.9558, 774 breaks |
| CV Top-1 (std over folds) / CV macro F1 / CV log loss | 0.8748 (0.0374) / 0.9270 / 0.0650 |
| Student-test confusion counts (TN, FP, FN, TP) | 191,392 / 1,814 / 2,279 / 16,862 |

Overfitting or underfitting: there is overfitting to the training trips. LightGBM scores 94.5 % Top-1 on its own training points vs 87.5 % in grouped CV, a gap of about 7 points (docs/notes/models_and_tuning.md; docs/REPORT_EVIDENCE_MAP.md). Held-out trips are the right estimate of deployment accuracy. The model is not underfitting by this evidence: the CV score is well above the heading-rule baseline (0.8152 CV Top-1). [MISSING: the 94.5 % figure is quoted from the notes and is not in a results CSV; the team should recompute and save it.] Some of the training-CV gap comes from the two unreliable trips (54, 33), which add label noise, but their share has not been measured.

### 3.2.2 HistGradientBoosting (scikit-learn)

**A. Model understanding**

*Why suitable?* It is another histogram gradient-boosted tree model, handling the same tabular features, and serves as a cross-check that the LightGBM result is not specific to one library. It needs no feature scaling.

*Strengths and limitations.* Strength: marginally highest CV Top-1 (0.8755) and CV macro F1 (0.9276) of all models (results/experiments.csv). Limitations: slower inference than LightGBM and XGBoost (0.1261 s per trip vs 0.0908 and 0.0431) and a longer fit (20.7 s vs 12.3 s); its student-test Top-1 (0.9007) is not better than LightGBM's. The difference from LightGBM is within noise (3.1). A tuning attempt for this model was started and stopped unfinished; its record was not saved and is not used (docs/notes/models_and_tuning.md).

**B. Training and configuration**

*Hyperparameters and initial values:* max_iter 200, learning_rate 0.1, max_leaf_nodes 63, min_samples_leaf 40, l2_regularization 1.0, early stopping off, seed 42, stride 3 (results/experiments.csv; src/pathsnap/models/registry.py).

*Objective:* the library's binary classification loss (log loss) for `HistGradientBoostingClassifier` (no loss argument set in registry.py). [Team to confirm.]

**C. Experimental results**

| Metric | Value |
| --- | ---: |
| Student-test Top-1 | 0.9007 |
| Student-test macro F1 | 0.9399 |
| Student-test log loss | 0.0488 |
| Precision / recall | 0.8979 / 0.8833 |
| Official route continuity, breaks | 0.9578, 756 |
| CV Top-1 (std) / CV macro F1 | 0.8755 (0.0375) / 0.9276 |
| Confusion counts (TN, FP, FN, TP) | 191,283 / 1,923 / 2,234 / 16,907 |

Overfitting or underfitting: no training-set score was recorded [MISSING]. CV (0.8755) is below student test (0.9007), as for every model; see 3.4. Early stopping is off, so the 200 iterations are not chosen by a validation curve.

### 3.2.3 XGBoost

**A. Model understanding**

*Why suitable?* Depth-wise gradient boosting is the third boosting variant, included to see whether the tree-growing strategy (leaf-wise vs depth-wise) matters here. It works on the same unscaled features.

*Strengths and limitations.* Strengths: the fastest inference of the heavy models (0.0431 s per trip), student-test Top-1 (0.9016) and route continuity (0.9587) on par with LightGBM. Limitation: the slowest of the three boosting models to fit (24.7 s) in our run, and no accuracy advantage (results/experiments.csv).

**B. Training and configuration**

*Hyperparameters and initial values:* n_estimators 300, learning_rate 0.1, max_depth 8, min_child_weight 5, subsample 0.8, colsample_bytree 0.8, reg_lambda 1.0, `tree_method` hist, seed 42, stride 3 (results/experiments.csv; src/pathsnap/models/registry.py).

*Objective:* XGBClassifier's binary logistic objective (log loss), not set explicitly in registry.py. [Team to confirm.]

**C. Experimental results**

| Metric | Value |
| --- | ---: |
| Student-test Top-1 | 0.9016 |
| Student-test macro F1 | 0.9413 |
| Student-test log loss | 0.0483 |
| Precision / recall | 0.9041 / 0.8821 |
| Official route continuity, breaks | 0.9587, 755 |
| CV Top-1 (std) / CV macro F1 | 0.8747 (0.0376) / 0.9269 |
| Confusion counts (TN, FP, FN, TP) | 191,416 / 1,790 / 2,256 / 16,885 |

Overfitting or underfitting: no training-set score was recorded [MISSING]. Its CV and test scores track LightGBM's within 0.1 points, so it is not distinguishable from LightGBM.

### 3.2.4 Extra Trees (50 trees, stride 6)

**A. Model understanding**

*Why suitable?* A randomised tree ensemble gives nonlinear decision rules with fewer assumptions and is a standard contrast to boosting: averaging many randomised trees (variance reduction) versus sequential error correction.

*Strengths and limitations.* In our runs it is the weakest of the five main models: lowest student-test Top-1 among them (0.8841), lowest candidate recall among them (0.8573) and a higher log loss (0.0561) (results/experiments.csv). It has the most false negatives of the five (2,731 vs 2,279 for LightGBM; results/plots/confusion_all_models.png). Important caveat: it was run with only 50 trees and on every 6th point (339,848 rows) to fit the time budget, so the gap to boosting may partly be due to this reduced setup (docs/notes/models_and_tuning.md). A full-size run was not made. (The registry default is 100 trees; the logged run used 50, see results/experiments.csv and src/pathsnap/models/registry.py.)

**B. Training and configuration**

*Hyperparameters and initial values:* n_estimators 50, min_samples_leaf 3, max_features 0.5, seed 42, stride 6 (results/experiments.csv).

*Objective:* scikit-learn's default split criterion for classification (Gini impurity); no loss is minimised by gradient descent. Not set explicitly in registry.py. [Team to confirm.]

**C. Experimental results**

| Metric | Value |
| --- | ---: |
| Student-test Top-1 | 0.8841 |
| Student-test macro F1 | 0.9328 |
| Student-test log loss | 0.0561 |
| Precision / recall | 0.8985 / 0.8573 |
| Official route continuity, breaks | 0.9531, 862 |
| CV Top-1 (std) / CV macro F1 | 0.8703 (0.0379) / 0.9245 |
| Confusion counts (TN, FP, FN, TP) | 191,353 / 1,853 / 2,731 / 16,410 |

Overfitting or underfitting: no training-set score was recorded [MISSING]. Fully grown randomised trees with min leaf 3 can fit training points closely, but without a training score this is not shown by our experiments.

### 3.2.5 Random Forest (50 trees, stride 6)

**A. Model understanding**

*Why suitable?* Bagged decision trees are a robust, low-tuning baseline for tabular data with interactions, and a contrast to both boosting and the more random Extra Trees.

*Strengths and limitations.* It is better than Extra Trees on student-test Top-1 (0.8881 vs 0.8841), macro F1 and recall, with equal log loss (0.0561 each) and slightly lower precision, and its CV macro F1 (0.9265) is close to the boosting models', but its student-test Top-1 is below the boosting models' by about 1.5 points and its log loss is higher (0.0561 vs 0.048) (results/experiments.csv). It had the longest fit time in our table (31.4 s) even with the smaller sample. Same caveat as Extra Trees: 50 trees and stride 6.

**B. Training and configuration**

*Hyperparameters and initial values:* n_estimators 50, min_samples_leaf 3, max_features sqrt, seed 42, stride 6 (results/experiments.csv).

*Objective:* scikit-learn's default criterion (Gini impurity). [Team to confirm.]

**C. Experimental results**

| Metric | Value |
| --- | ---: |
| Student-test Top-1 | 0.8881 |
| Student-test macro F1 | 0.9345 |
| Student-test log loss | 0.0561 |
| Precision / recall | 0.8948 / 0.8668 |
| Official route continuity, breaks | 0.9563, 801 |
| CV Top-1 (std) / CV macro F1 | 0.8733 (0.0377) / 0.9265 |
| Confusion counts (TN, FP, FN, TP) | 191,256 / 1,950 / 2,550 / 16,591 |

Overfitting or underfitting: no training-set score was recorded [MISSING]. CV log loss (0.0733) is higher than its student-test log loss (0.0561) and than LightGBM's CV log loss (0.0650), results/experiments.csv.

### 3.2.6 Logistic Regression (Person C)

**A. Model understanding**

*Why suitable?* A linear model on standardised features is the simplest probabilistic baseline. It tests how much of the problem is solved by a weighted sum of the features, and gives the lowest-complexity way to get probabilities for the decoder.

*Strengths and limitations.* Strengths: the fastest inference by far (0.0030 s per trip), and it beats the heading baseline on student test (0.8548 vs 0.8301) with no interactions beyond what the features already encode (results/experiments.csv). Limitations: CV Top-1 below LightGBM (0.8453 vs 0.8748) and a log loss on student test (0.0736) well above the boosting models' (0.048 to 0.049); a single linear boundary cannot model interactions such as heading mismatch combined with neighbour fit. This is a statement from the measured gap, not a tested cause.

**B. Training and configuration**

*Hyperparameters and initial values:* C 1.0, solver lbfgs, max_iter 500, features standardised (StandardScaler), seed 42, stride 3 (results/experiments.csv; scripts/train_simple.py). Default L2 penalty of scikit-learn.

*Objective:* regularised log loss (cross-entropy) with the L2 penalty.

**C. Experimental results**

| Metric | Value |
| --- | ---: |
| Student-test Top-1 | 0.8548 |
| Student-test macro F1 | 0.9107 |
| Student-test log loss | 0.0736 |
| Precision / recall | 0.8509 / 0.8241 |
| Official route continuity, breaks | 0.9449, 1,042 |
| CV Top-1 (std) / CV macro F1 | 0.8453 (0.0339) / 0.9021 |

Confusion matrix: [MISSING: confusion matrix for Logistic Regression is not in results/plots/confusion_all_models.png, which covers only the heavy models]. Overfitting or underfitting: no training-set score recorded [MISSING]. Its CV and student-test scores are both below the boosting models', which is consistent with limited capacity (underfitting), but without a training score this is a hypothesis.

### 3.2.7 Gaussian Naive Bayes (Person C)

**A. Model understanding**

*Why suitable?* A generative baseline from the probabilistic family, very cheap to fit. It assumes features are independent given the class.

*Strengths and limitations.* The independence assumption is violated here: several features are strongly related by construction (for example the five neighbour-perpendicular distances and the relative-distance features). Measured consequence: very poor probabilities, with a student-test log loss of 1.0193 and a precision of 0.4089 at a recall of 0.9706, so it labels too many candidates as correct (results/experiments.csv). As a result its candidate macro F1 (0.7497) is the lowest of all models, even though its per-point Top-1 (0.8134) is only a little below the heading baseline. Its probabilities would not be suitable as decoder emissions without calibration. [Team to confirm that the independence explanation fits the viva answer; the correlation between features was not measured in a results file.]

**B. Training and configuration**

*Hyperparameters and initial values:* var_smoothing 1e-9, seed 42, stride 3 (results/experiments.csv).

*Objective:* maximum-likelihood estimation of a class prior and a Gaussian per feature per class; no iterative loss minimisation.

**C. Experimental results**

| Metric | Value |
| --- | ---: |
| Student-test Top-1 | 0.8134 |
| Student-test macro F1 | 0.7497 |
| Student-test log loss | 1.0193 |
| Precision / recall | 0.4089 / 0.9706 |
| Official route continuity, breaks | 0.9342, 1,215 |
| CV Top-1 (std) / CV macro F1 | 0.8191 (0.0336) / 0.7475 |

Confusion matrix: [MISSING, as for Logistic Regression]. Overfitting or underfitting: no training-set score [MISSING]. CV and student-test macro F1 (0.7475 and 0.7497) and log loss (1.0176 and 1.0193) are nearly identical, so the problem is systematic and not a train-versus-unseen-trip gap; this points to model misspecification (underfitting), not overfitting.

### 3.2.8 KNN (Person C, 3,000-row fit subsample)

**A. Model understanding**

*Why suitable?* Instance-based: it asks whether the candidate looks like previously seen correct or incorrect candidates in standardised feature space. It needs no training assumptions about functional form.

*Strengths and limitations.* Limitation that matters most for our run: it was fitted on only 3,000 randomly chosen rows per fold, because exact neighbour scoring over all candidate rows is impractical (scripts/train_simple.py). The comparison is therefore not a full KNN: its low scores (CV Top-1 0.8125, student-test Top-1 0.8217) reflect the subsample as well as the algorithm. It also has the slowest inference (0.4107 s per trip) and the highest log loss of the learned models other than Gaussian NB (0.1367) (results/experiments.csv). Strength: it is fitted instantly (0.0847 s), since it only stores the sample.

**B. Training and configuration**

*Hyperparameters and initial values:* k 15, distance-weighted votes, Euclidean (p = 2), standardised features, 3,000 fit rows per fold, seed 42 (results/experiments.csv).

*Objective:* none; the score is the distance-weighted share of positive neighbours.

**C. Experimental results**

| Metric | Value |
| --- | ---: |
| Student-test Top-1 | 0.8217 |
| Student-test macro F1 | 0.8642 |
| Student-test log loss | 0.1367 |
| Precision / recall | 0.8233 / 0.6902 |
| Official route continuity, breaks | 0.9175, 1,568 |
| CV Top-1 (std) / CV macro F1 | 0.8125 (0.0389) / 0.8614 |

Confusion matrix: [MISSING]. Overfitting or underfitting: no training-set score [MISSING]. A KNN fitted on 3,000 rows is more likely limited by data volume than by capacity; this has not been tested with a larger subsample.

### 3.2.9 Non-learned baselines (no questions A to C apply in the usual sense)

These are reference points, not trained models, so there are no hyperparameters to tune and no loss function, and macro F1 and log loss are N/A (3.1).

- Nearest road: choose the arc with the smallest perpendicular distance. CV Top-1 0.5536, student-test Top-1 0.5360, route continuity 0.8025, 3,419 breaks (results/experiments.csv).
- Nearest road + heading rule: nearest arc penalised by heading mismatch with the vehicle's motion. CV Top-1 0.8152, student-test Top-1 0.8301, route continuity 0.9108, 1,760 breaks (results/experiments.csv).

The second baseline shows how much of the task is solved by distance plus direction alone. The learned LightGBM adds about 7 points of student-test Top-1 on top of it (0.9021 vs 0.8301).

## 3.3 Model visualisations

| Required visualisation | File | Status |
| --- | --- | --- |
| CV macro-F1 comparison across models | results/plots/model_comparison.png (middle panel, five main models) | Covers LightGBM, HistGradientBoosting, XGBoost, Extra Trees, Random Forest. [MISSING: a single plot with CV macro F1 for Logistic Regression, Gaussian NB and KNN; the values are in 3.1 and results/experiments.csv] |
| Student-test Top-1 and macro F1 | results/plots/model_comparison.png (left and middle panels; left panel also has the two baselines) and results/plots/simple_model_comparison.png (Logistic Regression, Gaussian NB, KNN) | Two separate figures, not one combined plot |
| Log loss comparison | results/plots/model_comparison.png (right panel, five main models) | [MISSING: log loss bars for the simple models; values in 3.1] |
| Confusion matrices | results/plots/confusion_all_models.png (six heavy variants), results/plots/confusion_lightgbm.png; also results/confusion_lightgbm.csv | [MISSING: confusion matrices for Logistic Regression, Gaussian NB and KNN, though the counts for the heavy models are in results/experiments.csv] |
| Calibration | results/plots/calibration_lightgbm.png (student test, ECE 0.0027) | Final model only |
| Candidate radius | results/plots/candidate_sweep.png | See Section 2.4 |
| Training and validation loss curves | None | N/A: the models were fitted without recording epoch-wise histories (HistGradientBoosting has early stopping off, src/pathsnap/models/registry.py), and no artificial curves were created. [MISSING: no iteration-wise validation log exists for LightGBM or XGBoost in results/] |
| Folium maps | reports/eda_map.html and reports/figures/prediction_map_00000052.png (four-layer prediction map, student_test trip 00000052) | Interactive map: reports/prediction_map_00000052.html; generation in scripts/make_prediction_map.py |

What the plots show (read from results/plots/model_comparison.png): the CV and student-test Top-1 bars of the three boosting models are visually indistinguishable and the error bars (std over 5 folds) overlap those of Extra Trees and Random Forest; the two non-learned baselines are clearly lower, with the heading rule far above nearest road. In the confusion matrices (results/plots/confusion_all_models.png) the false negatives (correct arc not chosen at threshold 0.5) outnumber false positives for every model.

## 3.4 Limitations of this comparison

1. Three boosting models are statistically tied (CV fold spread about 3.7 points against gaps of about 0.1 points).
2. Optuna tuning gave no gain; the final model is the untuned LightGBM (results/tuning_summary_lightgbm.json; artifacts/model_card.json).
3. KNN used a 3,000-row subsample.
4. Extra Trees and Random Forest used 50 trees and every 6th point.
5. Training trips 54 and 33 have unreliable label alignment (results/label_stats.csv). They are included in training and CV.
6. Student-test scores are higher than CV scores for every learned model (for example LightGBM 0.9021 vs 0.8748). This is not evidence of no overfitting. Student test has 99.96 % of points labelled against 98.0 % on training and does not contain the two unreliable trips (results/label_quality.csv), and with ten trips it is a small sample. CV is the more conservative estimate.
7. One seed (42) per model, no repeated CV (docs/notes/models_and_tuning.md).
8. The neural-network family (MLP) listed in the plan is not in the results (docs/PLAN_AND_PRD.md model line-up; docs/notes/models_and_tuning.md, Limitations). The comparison has eight learned models from linear, probabilistic, instance-based, tree-ensemble and boosting families.
9. `student_test` is a development set; the evaluator partition was not used for any choice here.
10. Timing figures come from single runs on the team's machine and vary between runs.

## Open items and contradictions found in the evidence

1. docs/notes/models_and_tuning.md says the simple models "are not in this table yet" and results/metrics_table.csv has no rows for them, but results/experiments.csv does contain them. This draft takes them from results/experiments.csv. results/metrics_table.csv should be regenerated if it is meant to include them.
2. src/pathsnap/models/registry.py lists n_estimators 100 for Extra Trees and Random Forest as the initial values, while results/experiments.csv and the notes record 50. The 50-tree configuration is what was run.
3. The 94.5 % LightGBM training Top-1 appears only in docs (docs/notes/models_and_tuning.md, docs/REPORT_EVIDENCE_MAP.md), not in a results file.
4. docs/DATA_QUALITY.md used to quote an earlier API run (Top-1 0.9016). results/api_student_test.csv was regenerated and now gives classifier Top-1 0.9021, equal to results/experiments.csv (19,143 labelled points).
