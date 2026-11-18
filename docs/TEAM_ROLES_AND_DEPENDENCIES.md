# Team roles, completion status and handoffs

This records the role split and handoffs against the current repository snapshot. Presence in the repository does not imply another teammate has reviewed or accepted a component.

## Roles

| Person | Ownership | Current status / dependency |
| --- | --- | --- |
| A — ML lead | Candidates, features, route-to-track labels, heavy models, Bayesian tuning and artifact export | A's feature/model pipeline, model artifacts and experiment evidence are present in the integrated tree. A teammate review of the adapter and artifact provenance is still useful before final submission. |
| B — decoder lead | Projection, graph search, HMM/Viterbi, metrics, stitching and `predict_trip` | Decoder source and frozen tuning outputs are present. The measured frozen-setting `student_test` check is in `results/hmm_tuning/final_student_test_check.json`. |
| C — data and integration | Inventory, EDA/maps, simple models, plots, API/scoring/logs, network setup, integration checks, report and README | C's source and supporting result files are in this branch and `origin/main`. The latest API end-to-end run and real two-laptop rehearsal still need fresh verification. |
| D — UI and demo | UI, screenshots, UI report section and demo checklist | UI source, API contract and demo instructions are present under `ui/` and `docs/`. Real UI and Laptop 1 receiving-terminal screenshots are still required. |

## C work completed locally

- Shared trip loader follows A's `Trip` contract and keeps labels opt-in.
- Inventory, quality findings, generated plots and an interactive Folium map are included.
- Logistic Regression, Gaussian NB, and KNN use the same trip-grouped folds as A's model comparison. KNN fit is explicitly capped at a deterministic 3,000 training rows per fold for runtime.
- A's feature/model pipeline and B's decoder are integrated under `src/pathsnap`; `src/pathsnap/pipeline.py` bridges candidate/features/probabilities and the decoder graph contract.
- Exported LightGBM artifact and model card are present. Selection used trip-grouped CV, not `student_test`.
- FastAPI `/health`, `/trips`, `/predict` plus after-prediction scoring and logging are implemented. The response excludes the reference route.
- README and a technical report draft are supplied. AI assistance is disclosed.

## Cross-person dependencies and remaining work

1. **A integration review:** verify feature order, candidate configuration and model artifact provenance against A's training/export path. This is a review dependency, not a claim that the model is missing.
2. **B integration review:** review coordinate handling and the C-to-decoder adapter, and confirm the API loads the frozen decoder configuration from `configs/decoder_best.json`.
3. **C verification:** rerun `scripts/evaluate_api.py` and the tests after dependency setup. The committed `results/api_student_test.csv` predates the latest tuned decoder check and must not be presented as current API timing evidence.
4. **D/team evidence:** capture real screenshots of the working UI and successful HTTP responses on Laptop 1, then add them to the report. No screenshots should be recreated or faked.
5. **Two-laptop rehearsal:** connect the UI laptop to the backend laptop, confirm all ten development trips work, and verify evaluator partition startup fails loudly unless exactly ten valid trip folders are configured.
6. **Final submission:** ensure the public repository source matches the demonstrated implementation and the single consolidated PDF covers every required section in `Deliverables.pdf`.

## Ground-truth separation

`predict_trip` is given only `.track`, `.nodes`, and `.arcs`. `src/pathsnap/scoring.py` reads `.route` only after prediction, and the API only adds metrics when a route file is present. The separate evaluator directory can be supplied without route files; in that mode the response has `metrics: null`. The final evaluator set is not in this checkout.
