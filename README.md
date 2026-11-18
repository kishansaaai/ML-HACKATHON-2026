# PathSnap: map matching and route reconstruction

PES University UE24CS352A ML Hackathon. Noisy GPS points are matched to a trip's own road network:
a LightGBM classifier scores every candidate road arc for every GPS point, and an HMM/Viterbi
decoder turns those scores into one connected route.

## Status (update as work lands)

| Part | Status |
| --- | --- |
| Candidate generation, labels, 27 features | done (`src/pathsnap/`) |
| Model comparison (5 models + baseline), Bayesian tuning, export | done (`scripts/`, `results/`, `artifacts/`) |
| HMM / Viterbi decoder | implemented, settings **not yet tuned** (`src/pathsnap/decoder/`) |
| End-to-end `predict(trip_dir)` | done (`src/pathsnap/pipeline.py`) |
| FastAPI backend, evaluator UI, Folium maps, EDA | **not yet in this repo** |
| Technical report (PDF) | **not yet written** |

## Repository layout

```
src/pathsnap/        library: io, geo (graph + index), labels, candidates, features, clf (inference),
                     models/registry, tuning, oof, pipeline, decoder/ (HMM, graph, stitching, metrics)
scripts/             runnable steps (data tables, folds, training, tuning, ablation, plots, evaluation)
configs/             cv_folds.json (5 trip-grouped folds, seed 42)
artifacts/           exported model: model.joblib, feature_order.json, model_card.json
results/             experiment logs and result tables (CSV/JSON), plots/
tests/               pytest suite
docs/                plan/PRD, notes per component (docs/notes/), decoder notes
Student folder/      provided dataset: train (80 trips) and student_test (10 trips)
```

## Install

Python 3.12. From the repository root:

```
pip install -r requirements.txt
pip install -e .
```

The dataset is expected in `Student folder/` (train/ and student_test/). Set `PATHSNAP_DATA` to use another location.
The reserved evaluation trips are never stored in this repository; the server reads them from a folder outside it.

## Reproduce the experiments

```
python scripts/make_candidates.py --jobs 6      # candidate + feature tables and labels for the 90 development trips (~30 s)
python scripts/make_folds.py                    # configs/cv_folds.json
python scripts/train_heavy.py --models lightgbm hist_gb xgboost                         # grouped CV + student_test
python scripts/train_heavy.py --models extra_trees random_forest --stride 6 --n-estimators 50
python scripts/baseline_nearest.py              # non-learned baseline (run inside scripts/)
python scripts/tune.py --model lightgbm         # Bayesian search (Optuna TPE), refit, tuned vs untuned, export
python scripts/ablation.py                      # feature-group ablation
python scripts/analysis_plots.py                # plots and tables for the report
sh scripts/rerun_all_models.sh                  # all of the model comparison in one go
```

Every model run appends a row to `results/experiments.csv` (configuration, seed, timings, metrics).
Details of each stage: `docs/notes/README_A_pipeline.md`, `docs/notes/candidates_labels_features.md`,
`docs/notes/models_and_tuning.md`.

## Predict a route

```python
from pathsnap.pipeline import predict
result = predict("Student folder/student_test/00000052")
result["hmm_arc_ids"], result["route_arc_ids"], result["route_polyline"], result["route_breaks"]
```

`predict` reads only `.track`, `.nodes` and `.arcs`. The reference route (`.route`) is opened only by the
scoring/evaluation code after prediction, never by the model or decoder.

Evaluate classifier-only vs classifier + HMM on development trips (labels used after prediction):

```
python scripts/eval_pipeline.py                         # student_test, default decoder settings
python scripts/make_oof_scores.py                       # out-of-fold scores for tuning the decoder on training trips
python scripts/eval_pipeline.py --part train --oof --beta 30 --tag b30
```

See `docs/notes/integration_a_b.md` for how the decoder is tuned and why out-of-fold scores are used for that.

## Tests

```
python -m pytest tests
```

## Rules the code follows

- Model fitting and cross-validation use the 80 training trips only, in trip-grouped folds; `student_test` is a development comparison set.
- The evaluator partition is never used to choose features, models, hyper-parameters, thresholds or decoder settings.
- Ground-truth labels stay server-side and are never exposed to the model or the UI.
- Every number in the report must come from a script and a file in this repository.

## Deployment (to be completed with the API and UI)

Laptop 1 runs the API (classifier, decoder, road data, server-side scoring); Laptop 2 runs the UI.
Run instructions and the API endpoint list will be added here when the backend and UI are committed.

## AI assistance

AI tools were used to help write and review code and notes. Reported results come from the committed scripts and
their output files. Team members are responsible for understanding and explaining all submitted work.
