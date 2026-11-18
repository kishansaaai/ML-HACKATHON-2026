#!/bin/sh
# Re-run every model with the official metric set (same configs, seeds and folds as the first comparison).
# Run from the repo root:  sh scripts/rerun_all_models.sh
set -e
cd "$(dirname "$0")"
python -u train_heavy.py --models lightgbm hist_gb xgboost --stride 3
python -u train_heavy.py --models lightgbm --stride 3 --params-from ../results/tuning_summary_lightgbm.json
python -u train_heavy.py --models extra_trees random_forest --stride 6 --n-estimators 50
python -u baseline_nearest.py
