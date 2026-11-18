> DRAFT prepared with AI assistance from repository files. The team must rewrite this in its own words and be able to explain every statement (assessment policy).

# 5. HMM and Viterbi Route Reconstruction

This section follows the six questions in Deliverables.pdf, section 5, and the route-reconstruction requirements in Models and Training Guidelines.pdf, sections 8 and 9. Code: `src/pathsnap/decoder/` (`hmm.py`, `graph.py`, `stitch.py`, `inference.py`, `metrics.py`, `config.py`, `tuning.py`). Notes: `docs/section5_decoder.md`.

**Short answer to "did it help?"** Partly. The HMM clearly improves route continuity and removes many route breaks. Its gain in per-point Top-1 accuracy is tiny on the 80 training trips (+0.0002) and it is worse than the classifier alone on 28 of those 80 trips. Section 5.6 gives the numbers and the failure cases.

**Data partitions used.** Decoder parameters were chosen only on the 80 training trips, using out-of-fold (OOF) classifier scores. The 10 `student_test` trips were used for one check after the parameters were frozen (`results/hmm_tuning/final_student_test_check.json`). The evaluator partition was never used for any choice (docs/section5_decoder.md).

## 5.0 Pipeline in one paragraph

For each trip the classifier gives a probability for every candidate arc of every GPS point. The decoder turns these into one arc per point with Viterbi, then joins the chosen arcs into a connected route by inserting shortest legal paths between them (`src/pathsnap/decoder/inference.py`, `predict_trip`). Prediction reads only `.track`, `.nodes` and `.arcs`; the reference route `.route` is opened only afterwards, by the scoring code (README.md, "Predict a route"; docs/section5_decoder.md, "Evaluation definitions"). Each trip has its own road network, so each trip gets its own graph and its own local coordinate frame in metres (docs/section5_decoder.md).

## 5.1 What are the hidden states in the implementation?

A hidden state at GPS observation *t* is one candidate: a directed road arc plus the position of the projected GPS point along that arc (`Candidate` in `src/pathsnap/decoder/graph.py`: `arc_id`, `fraction` from tail to head, `distance_m` perpendicular distance to the arc, and the projected point).

- The candidate-to-arc correspondence is preserved: every state keeps its original `arc_id` (row number in `.arcs`), and every graph edge keeps its source arc ID, so the decoder returns an arc ID per observation (docs/section5_decoder.md).
- Arcs are directed. A two-way road appears as two arcs, which are two different states. About 84% of arcs have a reverse arc (docs/notes/candidates_labels_features.md).
- The number of states differs by observation. Candidates are all arcs within 50 m of the point; if fewer than 6 are found the radius doubles (up to 400 m) until 6 are found; at most 24 per point; average 11.4 per point (docs/notes/candidates_labels_features.md). The Viterbi code stores one score and one backpointer per state, so sets of different sizes are fine (`_decode_segment` in `hmm.py`).

## 5.2 How are emission probabilities calculated?

The LightGBM probabilities of the candidates at one observation are clipped to be non-negative and normalised to sum to 1 across that observation's candidate set. The emission score is the log of the normalised probability (floored at 1e-12), minus a Gaussian distance term:

```
E_t(j) = log( p_j / sum_k p_k )  -  d_j^2 / (2 * sigma^2)        with sigma = 25 m
```

where `d_j` is the perpendicular distance from the GPS point to candidate `j` (`emission_log_scores` in `src/pathsnap/decoder/hmm.py`; sigma in `configs/decoder_best.json`). If all scores at a point are zero they are replaced by equal scores.

Note for the viva: this emission is a classifier posterior used in place of a generative likelihood p(GPS | arc), plus a distance penalty. It is a design choice, not a textbook HMM emission. The classifier was shown to be well calibrated on `student_test` (ECE 0.0027, docs/notes/models_and_tuning.md), which is the reason it is reasonable to use its probabilities this way.

## 5.3 How are transition probabilities calculated?

The transition score from candidate `i` at time *t-1* to candidate `j` at time *t* compares the driving distance on the road network with the straight-line distance between the two GPS points:

```
T(i, j) = - | network_distance(i, j)  -  gps_distance | / beta        beta = 200 m
T(i, j) = unreachable_log_penalty = -5                                if no path within the cutoff
```

(`transition_log_score` and `_decode_segment` in `hmm.py`.) These are log-scores, not normalised probabilities.

`network_distance` is directed: the rest of the source arc, plus a bounded Dijkstra shortest path between the arc head and the destination arc tail, plus the part of the destination arc up to the projected point. If both candidates are on the same arc and the destination lies ahead, it is the distance along the arc (`network_distance_matrix` in `graph.py`). Search is bounded by `min(10,000 m, gps_distance + beta * |penalty|)`, i.e. gps distance plus 1,000 m at the final settings. A path longer than that cannot score better than the unreachable penalty, so pruning does not change the best score (docs/section5_decoder.md; `hmm.py`).

**Viterbi recurrence** (`_decode_segment`, log space):

```
score_1(j) = E_1(j)
score_t(j) = max_i [ score_{t-1}(i) + T(i, j) ]  +  E_t(j)
backpointer_t(j) = argmax_i [ score_{t-1}(i) + T(i, j) ]
```

The best final state is chosen and the path is traced back through the backpointers.

**Route stitching.** Consecutive chosen arcs that differ are joined with the shortest directed path between them, duplicates are removed, and nothing is invented: if no legal path exists a break is recorded (`src/pathsnap/decoder/stitch.py`).

## 5.4 How are missing candidates and large GPS gaps handled?

Two separate cases.

**(a) An observation with no candidates** (for example an invalid coordinate). Its prediction is `None`. Decoding stops at that point and restarts as an independent segment at the next observation with candidates (`viterbi_decode` in `hmm.py`). The stitcher records a `missing_candidates` break instead of connecting across it (`stitch.py`). A warning is added to the result (`inference.py`).

**(b) The reference arc is not in the candidate set.** Then neither the classifier nor the HMM can predict it. The candidate miss rate on `student_test` is 0.0104% in `results/experiments.csv` (`test_candidate_miss_rate`, 0.000104). [CHECK: docs/notes/candidates_labels_features.md gives 0.005% for `student_test`; the team should reconcile the two before quoting one.] On the training trips the miss rate is 0.61% (docs/notes/candidates_labels_features.md); for example, training trip 00000081 has a 0.0060 miss rate (results/hmm_tuning/failure_analysis_top10.csv).

**Large GPS gaps.** If the time between consecutive points is more than 30 seconds, is negative, or is not a finite number, the decoder restarts a new segment instead of linking the two points (`_valid_pair` in `hmm.py`; `gap_threshold_s = 30` in `configs/decoder_best.json`). The stitcher records a `time_gap` break. Data context: about 96.5% of consecutive training intervals are exactly 1 s; 2,937 training intervals are longer than 2 s and 1,260 are longer than 5 s (docs/DATA_QUALITY.md).

Why 30 s: in the 8-trip pilot, gap values of 5, 15 and 30 s (beta 100) gave HMM Top-1 of 0.7432, 0.7457 and 0.7467, and 30 s and 120 s gave the same result at beta 100 (0.7467) and almost the same at beta 200 (0.7485 vs 0.7485; results/hmm_tuning/pilot_sweep_summary.csv). So 30 s is supported only weakly: it is not worse than 120 s and better than 5 s and 15 s on that pilot. [MISSING: no written reason in the repository for choosing 30 s over 120 s.]

## 5.5 What happens when consecutive GPS observations suggest disconnected road segments?

The transition score handles it inside the decoder, and the stitcher reports what remains.

1. If two candidates have no directed path within the cutoff, their transition gets the finite penalty of -5 (a constant, not minus infinity). Viterbi can still choose such a pair if the emission evidence outweighs it, so the decoder never fails on a trip, but such a choice costs the same as being about 1,000 m off in the distance term (5 x beta). Compare this with ordinary transitions, where a 50 m mismatch costs 0.25.
2. In the output route, if two chosen arcs are not head-to-tail connected, the stitcher inserts the shortest directed path when one exists. If none exists it appends the second arc and records a `disconnected` break. No edge is fabricated to hide it (`stitch.py`).
3. Route continuity (the official metric) counts the share of adjacent predicted pairs that are the same arc or head-to-tail connected; a pair with a `None` prediction counts as a break (`route_continuity` in `src/pathsnap/decoder/metrics.py`). Break counts below use this definition.
4. Separately, the stitched route has its own break count. After decoding with the final settings it was 365 breaks over the 80 training trips and 5 over the 10 `student_test` trips (sum of `stitched_breaks` in `results/hmm_tuning/frozen_candidate_full80.csv` and `final_student_test_check.csv`). Do not mix this count with the per-observation break counts in 5.6.

## 5.6 Did HMM decoding improve the final route?

### How the parameters were chosen

HMM parameters were chosen on training trips only, using OOF LightGBM scores: for each training trip, the score comes from a LightGBM model trained on the other folds (same recipe, same 5 trip-grouped folds; `src/pathsnap/oof.py`, `scripts/make_oof_scores.py`). Reason: the exported model is trained on all 80 training trips and scores them far better than unseen trips (about 94.5% Top-1 in-sample vs about 87.5% grouped CV; docs/notes/models_and_tuning.md), so tuning on in-sample scores would tune the decoder for over-confident probabilities. The OOF Top-1 pooled over folds matches the grouped CV result of 87.48% (docs/notes/integration_a_b.md).

The tuning was a manual one-factor-at-a-time sweep, not a Bayesian search: first on a pilot of 8 trips (00000001 to 00000006, 00000008, 00000010; 18,023 labelled observations), then one run of the chosen setting on all 80 trips. Pilot results (classifier-only Top-1 on these 8 trips is 0.7700, mean continuity 0.9448; `results/hmm_tuning/pilot_sweep_summary.csv`):

| Setting changed | HMM Top-1 | HMM mean continuity |
| --- | --- | --- |
| beta 30 (gap 120, no sigma, penalty -20) | 0.7249 | 0.9601 |
| beta 50 | 0.7345 | 0.9589 |
| beta 100 | 0.7467 | 0.9601 |
| beta 200 | 0.7485 | 0.9574 |
| beta 400 | 0.7443 | 0.9525 |
| beta 100, gap 5 s | 0.7432 | 0.9562 |
| beta 100, gap 15 s | 0.7457 | 0.9591 |
| beta 100, gap 30 s, sigma 25 | 0.7453 | 0.9615 |
| beta 150, gap 30 s, sigma 25 | 0.7478 | 0.9598 |
| beta 200, gap 30 s, sigma 25 (penalty -20) | 0.7497 | 0.9593 |
| beta 200, gap 30 s, sigma 50 | 0.7490 | 0.9576 |
| beta 200, gap 30 s, sigma 25, penalty -10 | 0.7583 | 0.9561 |
| beta 200, gap 30 s, sigma 25, penalty **-5** (selected) | 0.7687 | 0.9554 |

(`results/hmm_tuning/pilot_sweep_summary.csv`; the -20 row of the sigma 25 setting is also quoted in docs/section5_decoder.md.) A penalty of -40 was started and stopped after the first trip took 120.9 s; it has no aggregate score and was not used (docs/section5_decoder.md).

Observations that the team should state plainly:

- On the 8-trip pilot the HMM was below the classifier on Top-1 in every setting tried, including the selected one (0.7687 vs 0.7700).
- The unreachable penalty mattered far more than any other parameter (+0.019 Top-1 from -20 to -5), and it trades a little continuity (0.9593 to 0.9554) for Top-1.
- The penalty was only tested at -20, -10 and -5, and the best value is at the edge of the tested range. Whether less negative values would help further was not tested. [MISSING: no results for penalties above -5.]
- The pilot used the 8 lowest-numbered trips (not a random sample), and those 8 trips are also part of the 80-trip run, so the 80-trip result is not independent of the tuning. The only check that did not influence any choice is `student_test`.
- A helper for group-aware decoder search exists (`tune_decoder_group_cv` in `src/pathsnap/decoder/tuning.py`, selection rule Top-1, then continuity, then log loss), but it is used only in `tests/test_decoder.py`; the reported sweeps were run with `scripts/tune_hmm.py`. [MISSING: no written selection rule for the manual sweep; the final choice follows the highest pilot Top-1.]

Final settings: beta 200 m, gap 30 s, emission sigma 25 m, unreachable penalty -5 (`configs/decoder_best.json`; these are also the `DecoderConfig` defaults in `src/pathsnap/decoder/config.py`).

### Classifier-only vs classifier + HMM

Top-1 is weighted by labelled observations. Continuity is the mean over trips of the share of adjacent predicted pairs that are the same arc or head-to-tail connected.

| Data and scores | Labelled obs. | Top-1 classifier | Top-1 + HMM | Mean continuity classifier | Mean continuity + HMM | Breaks classifier | Breaks + HMM | HMM better / tied / worse trips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 training trips, OOF scores, tuned decoder | 187,866 | 0.8790 | 0.8792 | 0.9506 | 0.9623 | 8,245 | 5,505 | 47 / 5 / 28 |
| 10 `student_test` trips, exported model, tuned decoder (one check after freeze) | 19,143 | 0.9021 | 0.9071 | 0.9558 | 0.9697 | 774 | 501 | 7 / 0 / 3 |
| 10 `student_test` trips, exported model, earlier untuned decoder (beta 50 m, gap 120 s, no sigma, penalty -20) | 19,143 | 0.9021 | 0.8989 | 0.9558 | 0.9728 | 774 | 413 | 5 / 0 / 5 |

Sources: `results/hmm_tuning/frozen_candidate_full80.json`; `results/hmm_tuning/final_student_test_check.json`; third row from `results/pipeline_student_test_default.csv` (weighted Top-1 and means recomputed from the per-trip rows; the same figures, 89.89%, 97.28%, 413, are in docs/notes/integration_a_b.md, whose comparison table is the earlier untuned one and is superseded by the first two rows).

What the numbers support:

- **Route continuity and breaks improved.** On the 80 training trips continuity rose from 0.9506 to 0.9623 and breaks fell by 2,740 (8,245 to 5,505, about one third). The HMM had higher continuity on 74 trips, lower on 4 and equal on 2 (computed from `results/hmm_tuning/frozen_candidate_full80.csv`). On `student_test`, continuity rose from 0.9558 to 0.9697 and breaks fell from 774 to 501.
- **Top-1 barely moved.** +0.0002 on the training trips (0.8790 to 0.8792), too small to call an improvement. The mean per-trip Top-1 change is +0.0028 (median +0.0020), which is larger than the weighted change, so trips with many points gained less on average than short trips. On `student_test` the gain is +0.0050 (0.9021 to 0.9071), from 7 better and 3 worse trips; that is one development check on 10 trips and is not strong evidence on its own.
- **The tuned decoder trades continuity for Top-1 relative to the earlier untuned one.** On `student_test` the untuned decoder had fewer breaks (413 vs 501) and higher continuity (0.9728 vs 0.9697) but lower Top-1 (0.8989 vs 0.9071). We do not claim the tuned decoder is better on every metric.
- **Sequence-level performance is mixed, and worse for the HMM on average.** `python scripts/sequence_metrics.py` scores the whole route against the reference route on the 10 `student_test` trips (`results/sequence_metrics_student_test.csv`; the reference is read only after prediction). Mean arc-set F1 is 0.8983 for the classifier-only route and 0.8652 for the HMM route; the HMM is higher on 3 of 10 trips. The reason is visible in precision and recall: the HMM route has recall 0.9923 (it covers almost every reference arc) but precision 0.7756 (it also contains arcs that are not in the reference), against 0.9097 and 0.8915 for the classifier. The cause (for example detours inserted by route stitching, or the permissive unreachable penalty of -5) was not investigated, so we do not claim one. Edit distance is not comparable between the two columns of that file: the classifier route is one arc per GPS point without removing repeats, the HMM route is a stitched, de-duplicated arc sequence.

| Trip | Arc-set F1, classifier | Arc-set F1, HMM | HMM precision | HMM recall |
|---|---:|---:|---:|---:|
| 00000000 | 0.9223 | 0.8900 | 0.8054 | 0.9944 |
| 00000007 | 0.9837 | 0.9798 | 0.9758 | 0.9837 |
| 00000009 | 0.8617 | 0.9294 | 0.8699 | 0.9977 |
| 00000040 | 0.9453 | 0.9102 | 0.8351 | 1.0000 |
| 00000052 | 0.8814 | 0.6889 | 0.5314 | 0.9789 |
| 00000056 | 0.9104 | 0.9171 | 0.8623 | 0.9794 |
| 00000065 | 0.9034 | 0.8555 | 0.7474 | 1.0000 |
| 00000076 | 0.8583 | 0.8079 | 0.6827 | 0.9893 |
| 00000077 | 0.8859 | 0.8933 | 0.8072 | 1.0000 |
| 00000091 | 0.8310 | 0.7798 | 0.6391 | 1.0000 |

- **Runtime:** the decoding runs summed to 190.7 s for the 80 training trips (largest single trip 14.3 s) and 18.3 s for the 10 `student_test` trips (largest 4.9 s), including scoring (`runtime_seconds_sum` in the two JSON files; `seconds` in the two CSVs).

### Where the HMM did not help

On the 80 training trips the HMM lowered Top-1 on 28 trips. The ten largest regressions are in `results/hmm_tuning/failure_analysis_top10.csv`; the top three:

| Trip | Top-1 classifier | Top-1 HMM | Change | Continuity change | Breaks classifier to HMM | Candidate miss rate | Ambiguous-label fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00000081 | 0.9171 | 0.8456 | -0.0715 | +0.0099 | 165 to 133 | 0.0060 | 0.2906 |
| 00000083 | 0.8905 | 0.8261 | -0.0644 | -0.0398 | 63 to 123 | 0.0000 | 0.0000 |
| 00000010 | 0.6879 | 0.6448 | -0.0432 | -0.0055 | 210 to 225 | 0.0000 | 0.0000 |

What the files do and do not show:

- For trips 00000083 and 00000010 the candidate set contained the reference arc at every labelled point and no labels were ambiguous, so the loss is not explained by missing candidates or by ambiguous reference alignment. These are decoder error cases. The repository does not establish the cause, and we do not claim one.
- For trip 00000081, 29% of labels are flagged ambiguous and the stitched route has 83 breaks. This trip's reference alignment may be unreliable, but the files do not show that the ambiguity is what caused the drop.
- On trips 00000083 and 00000010 the HMM also had more route breaks than the classifier alone (63 to 123; 210 to 225), so on these trips it hurt both metrics.
- Trip 00000010 shows how sensitive the result is to the penalty: with the earlier -20 penalty its HMM Top-1 was 0.4858 (`results/hmm_tuning_extension/b200_g30.log`, the pilot run without sigma), against 0.6448 at -5 (`failure_analysis_top10.csv`), and the classifier alone gets 0.6879. These two runs differ in sigma as well as penalty, so the penalty is not the only difference.
- On `student_test` the HMM was worse on trips 00000007, 00000077 and 00000091 (per-trip rows in `results/hmm_tuning/final_student_test_check.csv`); the cause was not investigated. [MISSING: analysis of these three trips.]
- Trips 00000054 and 00000033 have reference routes that barely follow the GPS (14.7% and 49% labelled points; docs/notes/candidates_labels_features.md), so any Top-1 for them describes the reference route, not only the model.

**Conclusion for the report.** With the tuned decoder the HMM gives a clear continuity gain, a tiny Top-1 gain overall, and mixed per-trip behaviour. We should say that route continuity and fewer breaks are the real benefit, and that the Top-1 gain is not distinguishable from noise on the training trips.

## 5.7 Visualisation evidence needed

Folium map with the GPS points, the road network, the reference route and the predicted route: `python scripts/make_prediction_map.py --trip 00000052 --part student_test` writes `reports/prediction_map_00000052.html`: one Folium map with four toggleable layers (raw GPS points, road network of the trip, reference route drawn after prediction, predicted classifier + HMM route). It is a development-trip map and must never be run on evaluator data. [MISSING: a map for a failure trip such as 00000083; the exported model scored training trips in-sample, so a fair failure map needs the out-of-fold scores.]

## 5.8 How to reproduce

From the repository root, Python 3.12 (README.md; docs/section5_decoder.md, "Reproduction"; note that the docs write `.venv/bin/python`, here `python` is used as in README.md):

```
pip install -r requirements.txt
pip install -e .
python scripts/make_candidates.py --jobs 6
python scripts/make_folds.py
python scripts/train_heavy.py --models lightgbm hist_gb xgboost
python scripts/make_oof_scores.py                      # out-of-fold scores for tuning on training trips
```

All 80 training trips with the frozen settings (the `--unreachable-penalty -5` must be given explicitly; the script default is -20):

```
python scripts/tune_hmm.py --beta 200 --gap 30 --sigma 25 --unreachable-penalty -5 --tag frozen_candidate_full80_reproduction --oof
```

The single post-freeze check on `student_test`:

```
python scripts/tune_hmm.py --part student_test --beta 200 --gap 30 --sigma 25 --unreachable-penalty -5 --tag final_student_test_check_rerun
```

`scripts/tune_hmm.py` refuses to overwrite an existing tag, so use a fresh tag for reruns. The earlier untuned-decoder row (third row of the table in 5.6) is the existing file `results/pipeline_student_test_default.csv`. Running `python scripts/eval_pipeline.py` today uses the tuned defaults and, with the default tag, would write over that file, so give a new tag, and the earlier settings from docs/notes/integration_a_b.md, to regenerate it: `python scripts/eval_pipeline.py --beta 50 --gap 120 --sigma 0 --unreachable-penalty -20 --tag untuned_default_rerun`. Tests (synthetic decoding and a leakage test with no `.route` file present): `python -m pytest tests` (README.md).
