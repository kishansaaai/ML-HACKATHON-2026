# Team roles, completion status and handoffs

This records the project plan against the supplied A/B branches and the current local integration. It does not imply A/B/D have reviewed or accepted these changes.

## Roles

| Person | Ownership | Current status / dependency |
| --- | --- | --- |
| A — ML lead | Candidates, features, route-to-track labels, heavy models, Bayesian tuning and artifact export | Branch `a/ml-pipeline` supplied the candidate/model contract and research results. Its feature code and a compatible model artifact are integrated locally. Need A to review the adapter and provenance before merge. |
| B — decoder lead | Projection, graph search, HMM/Viterbi, metrics, stitching and `predict_trip` | Branch `feature/person-b-decoder` decoder source is under `src/pathsnap/decoder`. C implemented an adapter. Need B to review coordinate-origin treatment, performance, and evaluator behavior. |
| C — data and integration | Inventory, EDA/maps, simple models, plots, API/scoring/logs, network setup, integration checks, report and README | Core code and analysis are present. All-ten end-to-end API check and two-laptop network rehearsal are not verified in this restricted environment. |
| D — UI and demo | UI, screenshots, UI report section and demo checklist | Still required from D. The UI is not present in this checkout. |

## C work completed locally

- Shared trip loader follows A's `Trip` contract and keeps labels opt-in.
- Inventory, quality findings, generated plots and an interactive Folium map are included.
- Logistic Regression, Gaussian NB, and KNN use the same trip-grouped folds as A's model comparison. KNN fit is explicitly capped at a deterministic 3,000 training rows per fold for runtime.
- A's feature/model pipeline and B's decoder are integrated under `src/pathsnap`; `src/pathsnap/adapters.py` bridges candidate/features/probabilities and B's graph contract.
- Exported LightGBM artifact and model card are present. Selection used trip-grouped CV, not `student_test`.
- FastAPI `/health`, `/trips`, `/predict` plus after-prediction scoring and logging are implemented. The response excludes the reference route.
- README and a technical report draft are supplied. AI assistance is disclosed.

## Cross-person dependencies and remaining work

1. **A review:** confirm feature order and candidate configuration match training, and review the A `TripGraph` feature adapter. `artifacts/model.joblib` was reconstructed locally from A's training code because the branch's binary artifact was unavailable via the repository connector; it must be checked against A's own exported artifact before a final merge.
2. **B review:** confirm B's projection and candidate geometry contract and inspect the adapter's conversion. The adapter creates A's feature graph using B's origin; this is the explicit compatibility bridge. Confirm its numerical behavior on known trips.
3. **B performance check:** all 10 `student_test` predictions and scoring calls passed through the API handler in-process; the separate no-`.route` test also passed. This validates the handler path, not HTTP/TCP or remote-client behavior, because local sockets are restricted here. Trip `00000056` took 44.5 seconds inside inference, above the plan's 30-second target; optimize/profile this with B.
4. **D handoff:** implement the thin UI against `/trips` and `/predict`, add UI/API-console screenshots, report section 7 and demo checklist. C's API fields are in `src/pathsnap/api.py` and documented in README.
5. **Team/demo handoff:** run the server on Laptop 1 and UI on Laptop 2, set API URL/CORS/firewall as needed, and test all ten development trips end to end. The evaluator dataset must remain separate until the final session.
6. **Repository handoff:** reconcile this local work with the A/B branches and the team's target branch; nothing has been pushed or merged.

## Ground-truth separation

`predict_trip` is given only `.track`, `.nodes`, and `.arcs`. `src/pathsnap/scoring.py` reads `.route` only after prediction, and the API only adds metrics when a route file is present. The separate evaluator directory can be supplied without route files; in that mode the response has `metrics: null`. The final evaluator set is not in this checkout.
