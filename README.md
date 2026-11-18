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
| FastAPI backend, Folium maps and EDA | integrated locally on `feature/data-api-integration-v2` |
| Evaluator UI | see Person D branch `person-d/ui`; integration still required |
| Technical report (PDF) | generated draft; team-authored per-model discussion and demo screenshots remain |

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

## Person C: data, API and report

Create an environment from `requirements.txt` and install the package with `pip install -e .`. The data inventory and EDA can be reproduced with:

```
python scripts/inventory.py
python scripts/plot_eda.py
python scripts/make_eda_map.py --trip 00000001 --partition train
```

`python scripts/train_simple.py` appends Logistic Regression, Gaussian Naive Bayes, and KNN comparisons to `results/experiments.csv`. KNN uses a fixed 3,000-row training subsample per fold. Regenerate comparison plots and official `results/metrics_table.csv` with `python scripts/analysis_plots.py`.

Start the Laptop 1 API from the repository root:

```
uvicorn pathsnap.api:app --app-dir src --host 0.0.0.0 --port 8000
```

On Windows, `scripts\\run_api.cmd` starts the service using `.venv`. The service defaults to the 10-trip `student_test` partition. Set `PATHSNAP_EVALUATOR_DIR` to the evaluator trip folder before launch to serve evaluator data; startup fails unless that directory contains exactly 10 valid trip folders. It never silently falls back to development data. Optional `PATHSNAP_CORS_ORIGINS` configures UI origins and `PATHSNAP_LOG_FILE` changes the log path.

API endpoints match [Person D's current API contract](https://github.com/kishansaaai/ML-HACKATHON-2026/blob/person-d/ui/docs/API_CONTRACT.md): `/health`, `/trips`, and `/predict`. `/predict` accepts only a trip ID, returns predicted map layers/metrics, and never exposes ground truth. Scoring opens `.route` only after `pathsnap.pipeline` has finished prediction. `python scripts/evaluate_api.py` re-times the 10 development trips and checks the saved official pipeline Top-1 results. `python scripts/build_report.py` rebuilds `output/pdf/PathSnap_technical_report.pdf` from saved repository metrics.

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
