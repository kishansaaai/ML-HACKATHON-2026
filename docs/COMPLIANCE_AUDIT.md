# Deliverables.pdf compliance review

Reviewed against the repository at commit `a2e918000a5d962a9a674927db5bf76600d090fe` and the requirements in `Deliverables.pdf`. The report is assembled from `docs/report/sec1_...md` through `sec8_...md` by `scripts/build_report_pdf.py` into the single canonical report `output/pdf/PathSnap_technical_report.pdf`.

## Current status

| Deliverable | Status | Evidence / remaining action |
| --- | --- | --- |
| One source repository with runnable code, artifacts, results, README setup and deployment steps | Present; verify clean install before submission | README, `requirements.txt`, `ui/requirements.txt`, `src/`, `scripts/`, `artifacts/`, `results/`. Rebuild ignored `data_cache/` as described in README. |
| One consolidated technical report PDF with required sections 2–7 | Built from all eight current section files; content is still a team-review draft | `scripts/build_report_pdf.py`; rebuilds the canonical PDF. The team must replace AI-assisted draft prose and resolve every `[MISSING]` / `[TEAM ACTION]` marker. |
| Section 2: dataset, prep, quality, feature patterns and geographic exploration | Partially covered | `docs/report/sec2_data_preprocessing.md`, inventory/quality results and EDA figures. Confirm prose and make sure the final PDF includes the actual generated figures. |
| Section 3: model comparison, per-model discussion, metrics, plots and fit analysis | Partially covered | `docs/report/sec3_models.md` contains comparison and per-model drafts. It explicitly identifies missing simple-model confusion matrices and missing training scores for several models; do not claim fit conclusions without evidence. |
| Section 4: Bayesian optimization method and tuned/untuned comparison | Draft coverage present | `docs/report/sec4_bayesian_optimisation.md`, `results/tuning_trials_lightgbm.csv`, `results/tuning_summary_lightgbm.json`. Team should verify every value and rewrite the interpretation. |
| Section 5: HMM/Viterbi and classifier comparison | Draft coverage present | `docs/report/sec5_hmm_viterbi.md`, `results/hmm_tuning/`, `configs/decoder_best.json`. Preserve the mixed per-trip results and distinguish observation continuity breaks from stitched geometry breaks. |
| Section 6: Folium map with GPS, road network, reference route and prediction, with PDF view | Implemented in generator; regenerate and inspect before submission | `scripts/make_prediction_map.py` produces interactive HTML and static PNG for `student_test` trip 00000052. Reference geometry is loaded only after prediction. |
| Section 7: working UI and successful API evidence from the receiving laptop | Not evidenced | `docs/report/sec7_api_ui_deployment.md` has required screenshot slots. The team must perform the real two-laptop run and add genuine, captioned UI and Laptop-1 CMD screenshots; single-machine rehearsal is not equivalent. |
| Source matches what is demonstrated | Not yet verifiable | After the real demo rehearsal, freeze and record the demonstrated commit hash. |
| AI disclosure and academic integrity | Draft only | `docs/report/sec8_ai_disclosure.md`; each team member must accurately describe AI use, rewrite analysis in their own words, and be able to defend the work. |

## Changes made in this review

- Corrected the PDF builder's expected section filenames. Previously it looked for obsolete names and silently skipped the real sections 2 and 4–7.
- Set the report builder to write the canonical `output/pdf/PathSnap_technical_report.pdf`, avoiding a second competing report filename.
- Updated the README to use the consolidated report builder and to state that team review and genuine demo evidence remain pending.
- Added a static four-layer map export for embedding in the PDF; the interactive map remains available as HTML.
- Added the required continuity-break count to successful `/predict` console log lines so implementation and `docs/API_CONTRACT.md` agree.

## Submission gate

Do not describe the report as final until the team has rebuilt and visually checked the PDF, rewritten the AI-assisted narrative, removed or resolved all draft markers, verified the metrics against their source files, and captured the real two-laptop screenshots. No evaluator-partition data should be used to tune or generate report metrics.
