# Person A pipeline: how to reproduce

```
pip install -r requirements.txt && pip install -e .

python scripts/make_candidates.py --jobs 6      # candidate+feature tables and labels, all 90 dev trips (~30 s)
python scripts/make_folds.py                    # configs/cv_folds.json (5 grouped folds, seed 42)
python scripts/baseline_nearest.py              # non-learned baseline (run from the scripts/ folder)
python scripts/train_heavy.py --models lightgbm hist_gb xgboost            # stride 3
python scripts/train_heavy.py --models extra_trees random_forest --stride 6 --n-estimators 50
python scripts/tune.py --model lightgbm         # Bayesian search, refit, tuned-vs-untuned, export to artifacts/
python scripts/ablation.py                      # feature-group ablation (about 25 min)
python scripts/analysis_plots.py                # sweep, comparison, calibration, confusion, label quality
python -m pytest tests/test_ml_pipeline.py
```

Outputs: `results/experiments.csv` (every model, seeds, configs, timings), `results/tuning_trials_lightgbm.csv`,
`results/ablation.csv`, `results/candidate_sweep.csv`, `results/label_quality.csv`, `results/plots/*.png`;
exported model in `artifacts/` (`model.joblib`, `feature_order.json`, `model_card.json`).
`data_cache/` is generated and git-ignored.

Scripts that import `train_heavy` (`baseline_nearest.py`, `tune.py`) must be run from the `scripts/` folder
or with `scripts/` on `PYTHONPATH`.

Hand-off to Person B: `from pathsnap.candidates import build_candidates`, `from pathsnap.clf import score_candidates, normalise_per_obs`.
`build_candidates(trip, graph)` uses no labels; `score_candidates` loads only `artifacts/`.
