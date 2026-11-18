# PathSnap

PathSnap map matches noisy GPS observations to the road arcs driven. The integrated prediction path is: road candidates → engineered features → exported LightGBM probabilities → HMM/Viterbi sequence → stitched route. The FastAPI service exposes the result and scores development trips only after prediction has finished.

## Setup

Python 3.10+ is recommended. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

On Windows, if local PowerShell policy blocks activation, call the venv interpreter directly (`.venv\Scripts\python.exe`) without changing the policy.

## Run the API

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Use `GET /health`, `GET /trips`, and `POST /predict` with `{"trip_id":"00000000"}`. Set `PATHSNAP_DATA` to another data root, `PATHSNAP_SERVE_PARTITION` to `train` or `student_test`, and `PATHSNAP_EVALUATOR_DIR` to a folder of evaluator trip directories when supplied. `PATHSNAP_CORS_ORIGINS` is a comma-separated allowlist; its default `*` is intended for a local demo. Requests are logged to console and `logs/api.log` (override with `PATHSNAP_LOG_FILE`). The prediction response omits reference routes and all labels.

On Windows, `scripts\run_api.cmd` starts the same service without requiring PowerShell script activation.

## Data, analysis and models

- `python scripts/inventory.py` validates supplied trip files and writes `results/data_inventory.csv` and `results/data_inventory.summary.json`.
- `python scripts/plot_eda.py` writes inventory SVGs under `reports/figures/`.
- `python scripts/make_eda_map.py --trip 00000001 --partition train` writes an interactive Folium map at `reports/eda_map.html`.
- `python scripts/make_folds.py` regenerates trip-grouped folds; `python scripts/make_candidates.py` regenerates local candidate Parquet caches.
- `python scripts/train_simple.py` fits Person C's Logistic Regression, Gaussian NB, and sampled-fit KNN comparisons.
- `python scripts/analysis_plots.py` regenerates model, calibration, candidate-radius and confusion plots from saved results.
- `python scripts/export_model.py` refits the selected classifier on training trips and writes `artifacts/model.joblib`.
- `python scripts/evaluate_api.py --partition student_test` runs prediction plus server-side scoring for every development trip and writes per-trip classifier/HMM metrics to `results/api_student_test.csv`.
- `python scripts/build_report.py` creates `output/pdf/PathSnap_technical_report.pdf` from saved metrics and the API log.

Candidate caches and the Python environment are local generated files and are gitignored. The final evaluator set is not included. Never use evaluator labels for model selection.

## Files and ownership

`src/pathsnap` contains the shared loader, Person A's feature/model pipeline, Person B's decoder, and Person C's integration/API/scoring. See `docs/TEAM_ROLES_AND_DEPENDENCIES.md` for branch provenance and handoffs; `docs/DATA_QUALITY.md` for data findings; and `docs/PLAN_AND_PRD.md` for the project requirements. `results/experiments.csv` is the source table for comparison results. `artifacts/model_card.json` records the exported model's selection and training details.

## Current integration limits

All 10 `student_test` trips passed the in-process API-handler integration check, including a prediction with `.route` removed. The sandbox blocks local TCP binds, so HTTP-over-network and the two-laptop demonstration remain unverified. Trip `00000056` exceeded the 30-second target in the recorded run. Person D's UI, screenshots, and demo checklist are separate deliverables and are not included here. See the technical report for verified results and open handoffs.

## AI assistance

Codex assisted with data tooling, model comparison scaffolding, integration, analysis, documentation and report generation. Team members should review and understand the code and retain this disclosure in the final submission.
