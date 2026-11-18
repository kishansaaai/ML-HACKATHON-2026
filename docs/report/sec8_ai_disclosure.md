# 8. Limitations

The points below are stated plainly so that the results are read with the right caution.

- **Small development set.** Student-test results are from 10 trips; the three boosting models are statistically tied (CV standard deviation 0.037 against differences of 0.1 points), and the choice of LightGBM rests on speed and simplicity as much as on accuracy.
- **Tuning gave no gain**, so the untuned configuration was exported.
- **Over-fitting** was measured only for LightGBM (94.5 % on training trips against 87.5 % in grouped CV); the other models have no training-set score.
- **Reduced settings** for some models: Extra Trees and Random Forest with 50 trees on every 6th point; KNN on a 3,000-row sample.
- **Labels are derived**, not given per point; training trips 00000054 and 00000033 have unreliable reference alignment.
- **HMM.** Its Top-1 gain is small (+0.0002 out of fold, +0.0050 on the student test) and it is worse on part of the trips; its route is more connected but has lower arc-level precision (section 5.3). Parameters were tuned on 8 pilot trips, and the best unreachable penalty lies at the edge of the tested range.
- **Deployment.** The full two-laptop run on the evaluator trips happens at the evaluation; timings were measured on one development laptop.
