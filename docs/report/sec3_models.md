# 3. Model Experimentation and Comparison

Eight learned models were trained and compared (the minimum is five), plus two non-learned baselines. All use the same candidate and feature tables, the same trip-grouped folds and seed 42, and every run is in `results/experiments.csv`. The student-test columns are a development comparison, never used to choose a model.

## 3.1 Overall model comparison

| Model | Algorithm family | Key configuration | CV macro-F1 | Top-1 (CV / test) | Route continuity (CV / test) | Student-test macro-F1 | Log loss | Inference time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | Boosting (leaf-wise GBDT) | 300 trees, lr 0.1, 63 leaves, min leaf 40, subsample 0.8, colsample 0.8, L2 1.0 | 0.9270 | 0.8748 / 0.9021 | 0.9501 / 0.9558 | 0.9406 | 0.0480 | 0.0908 s |
| HistGradientBoosting | Boosting (histogram GBDT) | 200 iterations, lr 0.1, 63 leaf nodes, min leaf 40, L2 1.0 | 0.9276 | 0.8755 / 0.9007 | 0.9498 / 0.9578 | 0.9399 | 0.0488 | 0.1261 s |
| XGBoost | Boosting (depth-wise GBDT) | 300 trees, lr 0.1, depth 8, min child weight 5, subsample 0.8, colsample 0.8, L2 1.0 | 0.9269 | 0.8747 / 0.9016 | 0.9506 / 0.9587 | 0.9413 | 0.0483 | 0.0431 s |
| Extra Trees | Tree ensemble (randomised) | 50 trees, min leaf 3, max features 0.5, every 6th point | 0.9245 | 0.8703 / 0.8841 | 0.9443 / 0.9531 | 0.9328 | 0.0561 | 0.0617 s |
| Random Forest | Tree ensemble (bagging) | 50 trees, min leaf 3, max features sqrt, every 6th point | 0.9265 | 0.8733 / 0.8881 | 0.9462 / 0.9563 | 0.9345 | 0.0561 | 0.0802 s |
| Logistic regression | Linear | C 1.0, L2, L-BFGS, max 500 iterations, standardised | 0.9021 | 0.8453 / 0.8548 | 0.9310 / 0.9449 | 0.9107 | 0.0736 | 0.0030 s |
| Gaussian naive Bayes | Probabilistic (generative) | variance smoothing 1e-9 | 0.7475 | 0.8191 / 0.8134 | 0.9139 / 0.9342 | 0.7497 | 1.0193 | 0.0499 s |
| KNN | Instance-based | k 15, distance-weighted, Euclidean, standardised, 3,000-row sample per fold | 0.8614 | 0.8125 / 0.8217 | 0.9019 / 0.9175 | 0.8642 | 0.1367 | 0.4107 s |

*Table 3. Overall comparison (partitions: CV on the 80 training trips, test on the 10 student_test trips; `results/experiments.csv`). Route continuity here is the share of adjacent predicted pairs that are the same arc or connected head to tail, for the classifier alone before the HMM. Inference time is seconds per trip for the classifier. No cell is N/A: every metric applies to every model.*

Non-learned baselines for reference:

| Baseline | CV Top-1 | Test Top-1 | CV route continuity | Test route continuity |
| --- | ---: | ---: | ---: | ---: |
| Nearest arc (not learned) | 0.5536 | 0.5360 | 0.7785 | 0.8025 |
| Nearest arc + heading rule (not learned) | 0.8152 | 0.8301 | 0.8878 | 0.9108 |

**Reading the table.** LightGBM, HistGradientBoosting and XGBoost are statistically tied: their CV Top-1 values (0.8748, 0.8755, 0.8747) differ by 0.1 points while the fold-to-fold standard deviation is 0.0374. LightGBM was chosen as the single final model because it is strongest on student-test Top-1 and log loss among the tied three, fits fastest (about 12 s) and is simple to export, not because it is clearly better. The two bagging ensembles are 0.15 to 0.45 points lower in CV and 1.4 to 1.8 points lower on the student test, logistic regression is a strong simple model, and a hand rule (nearest arc plus heading) already reaches 0.8301, so learning adds about 7 points over a good rule. Limitations of some rows: Extra Trees and Random Forest used 50 trees on every 6th point; KNN used a 3,000-row sample per fold.

## 3.2 Per-model analysis

### 3.2.1 LightGBM

**A. Model understanding.** *1. Why suitable?* The task is tabular: 27 mixed-scale numeric features per (point, arc) pair, interactions such as small distance with wrong heading, and a probability output that the HMM can use as emission. Boosted trees need no scaling and model such interactions. *2. Strengths and limitations here.* Strong accuracy, the fastest boosting fit (12.4 s), well-calibrated probabilities (ECE 0.0027, `calibration_lightgbm.png`) and a simple export. Limitations: many hyper-parameters; leaf-wise growth can over-fit small data; Bayesian tuning brought no gain (section 4).

**B. Training and configuration.** *3. Important hyper-parameters and initial values:* n_estimators = 300; learning_rate = 0.1; num_leaves = 63; min_child_samples = 40; subsample = 0.8; subsample_freq = 1; colsample_bytree = 0.8; reg_lambda = 1.0; seed 42, every 3th training point. *4. Loss / objective:* binary cross-entropy (log loss) of the arc-is-correct label, minimised by gradient boosting with L2 regularisation of the leaf weights.

**C. Experimental results.** *5. Student-test performance:* Top-1 accuracy 0.9021, macro-F1 0.9406, log loss 0.0480, route continuity 0.9558 (cross-validation: Top-1 0.8748, macro-F1 0.9270 with standard deviation 0.0174); training time 12.3 s, inference 0.091 s per trip. *6. Over- or under-fitting:* Some over-fitting: the exported model scores about 94.5 % Top-1 on its own training trips against 0.8748 in grouped CV (`docs/notes/models_and_tuning.md`), so it fits training trips more tightly than unseen ones. The student-test value (0.9021) is higher than CV because those 10 trips are easier on average; fold-to-fold Top-1 standard deviation is 0.0374.

### 3.2.2 HistGradientBoosting

**A. Model understanding.** *1. Why suitable?* It is the same family as LightGBM (histogram-binned gradient boosting) and ships with scikit-learn, so it tests whether the library choice matters. *2. Strengths and limitations here.* Equivalent accuracy to LightGBM (CV Top-1 0.8755). It fits slower here (20.7 s) and our configuration uses no row or column subsampling, which LightGBM uses as extra regularisation.

**B. Training and configuration.** *3. Important hyper-parameters and initial values:* max_iter = 200; learning_rate = 0.1; max_leaf_nodes = 63; min_samples_leaf = 40; l2_regularization = 1.0; seed 42, every 3th training point. *4. Loss / objective:* binary log loss with L2 regularisation of the leaf values; early stopping disabled so every model gets the same 200 iterations.

**C. Experimental results.** *5. Student-test performance:* Top-1 accuracy 0.9007, macro-F1 0.9399, log loss 0.0488, route continuity 0.9578 (cross-validation: Top-1 0.8755, macro-F1 0.9276 with standard deviation 0.0177); training time 20.7 s, inference 0.126 s per trip. *6. Over- or under-fitting:* The CV-to-test gap has the same size and direction as LightGBM's (CV 0.8755, test 0.9007); no training-set score was logged, so over-fitting cannot be quantified beyond that.

### 3.2.3 XGBoost

**A. Model understanding.** *1. Why suitable?* A depth-wise boosting implementation is a standard strong baseline for tabular data and gives a second view of boosting with different tree growth. *2. Strengths and limitations here.* Fastest inference of the heavy models (0.043 s per trip) and mature tooling; but the slowest fit (24.7 s) and an extra dependency, with no accuracy advantage over LightGBM (CV 0.8747, test 0.9016).

**B. Training and configuration.** *3. Important hyper-parameters and initial values:* n_estimators = 300; learning_rate = 0.1; max_depth = 8; min_child_weight = 5; subsample = 0.8; colsample_bytree = 0.8; reg_lambda = 1.0; seed 42, every 3th training point. *4. Loss / objective:* binary logistic loss (cross-entropy) with L2 regularisation (lambda 1.0), depth-8 trees.

**C. Experimental results.** *5. Student-test performance:* Top-1 accuracy 0.9016, macro-F1 0.9413, log loss 0.0483, route continuity 0.9587 (cross-validation: Top-1 0.8747, macro-F1 0.9269 with standard deviation 0.0173); training time 24.7 s, inference 0.043 s per trip. *6. Over- or under-fitting:* No training-set score was logged. CV and test accuracy are as close to LightGBM's as the fold spread allows, so we have no evidence of a different over-fitting behaviour.

### 3.2.4 Extra Trees

**A. Model understanding.** *1. Why suitable?* Randomised-threshold trees reduce variance and are robust to noisy features; a good bagging counterpart to boosting. *2. Strengths and limitations here.* Low variance and trivially parallel. Its probabilities are averaged leaf class fractions and are coarser: log loss 0.0561 against 0.0480 for LightGBM, which makes them poorer HMM emissions. We ran only 50 trees on every 6th point, which limits this row.

**B. Training and configuration.** *3. Important hyper-parameters and initial values:* n_estimators = 50; min_samples_leaf = 3; max_features = 0.5; seed 42, every 6th training point. *4. Loss / objective:* no global loss: each split minimises Gini impurity (scikit-learn default) and the ensemble averages the trees' class fractions.

**C. Experimental results.** *5. Student-test performance:* Top-1 accuracy 0.8841, macro-F1 0.9328, log loss 0.0561, route continuity 0.9531 (cross-validation: Top-1 0.8703, macro-F1 0.9245 with standard deviation 0.0169); training time 25.4 s, inference 0.062 s per trip. *6. Over- or under-fitting:* Lowest of the heavy models on student-test Top-1 (0.8841) and macro-F1. CV 0.8703 against test 0.8841 points to under-use of the data (fewer trees, sparser sample) rather than over-fitting; training scores were not logged.

### 3.2.5 Random Forest

**A. Model understanding.** *1. Why suitable?* Bootstrap-aggregated trees are a classic robust ensemble and a comparison point for the randomised variant. *2. Strengths and limitations here.* Stable and simple; slower to fit than boosting (31.4 s here) and, like Extra Trees, gives coarse probabilities (log loss 0.0561). Only 50 trees on every 6th point were used.

**B. Training and configuration.** *3. Important hyper-parameters and initial values:* n_estimators = 50; min_samples_leaf = 3; max_features = sqrt; seed 42, every 6th training point. *4. Loss / objective:* no global loss: Gini-impurity splits on bootstrap samples with square-root feature subsampling; probabilities are averaged leaf fractions.

**C. Experimental results.** *5. Student-test performance:* Top-1 accuracy 0.8881, macro-F1 0.9345, log loss 0.0561, route continuity 0.9563 (cross-validation: Top-1 0.8733, macro-F1 0.9265 with standard deviation 0.0173); training time 31.4 s, inference 0.080 s per trip. *6. Over- or under-fitting:* CV 0.8733 against test 0.8881; no training-set score logged, so no direct over-fitting evidence; its 1.4 point deficit to boosting on the test is consistent with the reduced settings.

### 3.2.6 Logistic regression

**A. Model understanding.** *1. Why suitable?* A linear, probabilistic baseline: if a linear combination of the 27 features were enough, a boosted model would not be needed. It also gives calibrated-by-construction probabilities. *2. Strengths and limitations here.* Extremely fast (0.003 s per trip) and strong for its simplicity (CV Top-1 0.8453), but it cannot represent interactions such as distance combined with heading, and it trails boosting by about 3 points in CV and 4.7 points on the student test.

**B. Training and configuration.** *3. Important hyper-parameters and initial values:* C = 1.0; max_iter = 500; solver = lbfgs; standardized = True; seed 42, every 3th training point. *4. Loss / objective:* L2-regularised log loss (C = 1.0), minimised with L-BFGS; features are standardised.

**C. Experimental results.** *5. Student-test performance:* Top-1 accuracy 0.8548, macro-F1 0.9107, log loss 0.0736, route continuity 0.9449 (cross-validation: Top-1 0.8453, macro-F1 0.9021 with standard deviation 0.0133); training time 12.0 s, inference 0.003 s per trip. *6. Over- or under-fitting:* Signs of under-fitting (high bias): it is about 3 points below the boosting models in CV and 4.7 points on the student test with the same features, which a linear boundary explains; its log loss (0.0736) is higher than the boosting models'. Training-set scores were not logged.

### 3.2.7 Gaussian naive Bayes

**A. Model understanding.** *1. Why suitable?* A generative probabilistic baseline that is cheap and needs no tuning. *2. Strengths and limitations here.* Fast to train (3.3 s), but its independence assumption is badly violated (many features are correlated), so its probabilities are over-confident: test log loss 1.0193, far above every other model, which makes it unsuitable as an HMM emission.

**B. Training and configuration.** *3. Important hyper-parameters and initial values:* var_smoothing = 1e-09; seed 42, every 3th training point. *4. Loss / objective:* no loss is minimised: per-class Gaussian likelihoods are fitted by maximum likelihood (variance smoothing 1e-9) and combined by Bayes' rule.

**C. Experimental results.** *5. Student-test performance:* Top-1 accuracy 0.8134, macro-F1 0.7497, log loss 1.0193, route continuity 0.9342 (cross-validation: Top-1 0.8191, macro-F1 0.7475 with standard deviation 0.0209); training time 3.3 s, inference 0.050 s per trip. *6. Over- or under-fitting:* This is model mis-specification rather than classical under-fitting: Top-1 (0.8134) is moderate, but log loss 1.0193 and macro-F1 0.7497 show poor calibration. Training scores were not logged.

### 3.2.8 KNN

**A. Model understanding.** *1. Why suitable?* Instance-based learning tests whether similar (point, arc) situations in the training set have the same label. *2. Strengths and limitations here.* Needs no training (0.08 s) but is the slowest at prediction (0.411 s per trip) and was fitted on a 3,000-row sample per fold, which limits it; its distance-weighted votes give coarse probabilities (log loss 0.1367).

**B. Training and configuration.** *3. Important hyper-parameters and initial values:* fit_rows_per_fold = 3000; n_neighbors = 15; p = 2; standardized = True; weights = distance; seed 42, every 3th training point. *4. Loss / objective:* no training objective: a distance-weighted majority vote of the 15 nearest standardised training rows.

**C. Experimental results.** *5. Student-test performance:* Top-1 accuracy 0.8217, macro-F1 0.8642, log loss 0.1367, route continuity 0.9175 (cross-validation: Top-1 0.8125, macro-F1 0.8614 with standard deviation 0.0142); training time 0.1 s, inference 0.411 s per trip. *6. Over- or under-fitting:* Under-fitting through the small fitting sample: CV Top-1 0.8125, test 0.8217, clearly the weakest learned model on Top-1 after naive Bayes. Training scores were not logged.

## 3.3 Model visualisations

![Figure 4. Cross-validation macro-F1, student-test Top-1 and macro-F1, and student-test log loss for all eight models (partitions: CV on training, test on student_test).](docs/report/fig/all_models_comparison.png)

![Figure 5. Candidate-level confusion matrices of the boosting and bagging models (including tuned LightGBM) on student_test. Confusion counts were not logged for the three simple models, so none is drawn for them rather than invented.](docs/report/fig/confusion_all_models.png)

![Figure 6. Calibration of the selected LightGBM model on student_test (ECE 0.0027); calibrated probabilities matter because the HMM consumes them.](docs/report/fig/calibration_lightgbm.png)

LightGBM at candidate level on student_test: TN 191,392, FP 1,814, FN 2,279, TP 16,862, precision 0.9029, recall 0.8809. Training-versus-validation loss curves are not shown: none of the models was fitted with epoch-wise training histories logged, and we do not draw artificial curves for them.
