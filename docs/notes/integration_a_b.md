# Integration of Person A (candidates + classifier) and Person B (HMM decoder)

Historical record of the initial `integration/a-b` snapshot (`a/ml-pipeline` + `feature/person-b-decoder` + the fixes below). This snapshot and the later Person C/D work are now in `main`; for current results use `docs/section5_decoder.md`, `results/hmm_tuning/`, and `docs/REPORT_EVIDENCE_MAP.md`.

## Fixes made while integrating
1. **Package clash.** Both branches had a top-level package named `pathsnap` (B at the repo root, A under `src/`). B's code now lives in `src/pathsnap/decoder/`; the root-level `pathsnap/` is gone. One `pip install -e .` covers everything. B's benchmark script was pointed at `src/`.
2. **One coordinate frame.** B's graph is centred on nodes + track, A's features on the track median. Same formula and Earth radius, different origin. `InferenceComponents` now has an optional `projection_factory`; `pathsnap.pipeline` passes A's origin so the classifier's features and the decoder candidates use identical metres. B's graph cache key now includes the projection origin (otherwise a graph built in another frame could be reused).
3. **Real adapter.** `src/pathsnap/pipeline.py`: `predict(trip_dir)` runs candidates, features, the exported LightGBM and the decoder. It reads only `.track`, `.nodes`, `.arcs`. Candidate rows and scores are kept aligned by row index, so an arc rejected by the decoder's validity checks cannot shift the scores.
4. **One definition of route continuity.** A's first comparison used a stricter *distinct-arc* variant. `cand_metrics.route_continuity` now implements the official (B's / PRD) definition and is tested for agreement with `pathsnap.decoder.metrics`; the old one is `distinct_arc_continuity`. Every model was then re-scored with the same code (`scripts/rerun_all_models.sh`): the first-run rows keep their values in `*_distinct*` columns, and the official columns are filled by the re-run rows. Do not quote `*_distinct*` numbers as "route continuity".
5. **Stitching speed.** `RoadGraph._dijkstra` now stops when the target node is settled (exact; verified against the full search on 180 random node pairs, 0 mismatches, plus a unit test). Slowest dev trip (3,958 points): 11.8 s -> 9.8 s, including the one-off ~2.5 s model load. The rest is the HMM's transition search; fine for the demo, worth re-timing on the largest evaluator trip.

> SUPERSEDED for the decoder numbers: Person B tuned the decoder afterwards (beta 200 m, gap 30 s, sigma 25 m, unreachable penalty -5);
> see `docs/section5_decoder.md` and `results/hmm_tuning/`. The table below is the earlier UNTUNED comparison, kept for the record.

## Pipeline results on the 10 `student_test` trips (development data, DEFAULT untuned decoder: beta 50 m, gap 120 s)
Reproduce: `python scripts/eval_pipeline.py` (writes `results/pipeline_student_test_default.csv`).

| | Top-1 (19,143 labelled points) | route continuity, official (trip mean) | route breaks |
| --- | --- | --- | --- |
| LightGBM alone | 90.21 % | 95.58 % | 774 |
| LightGBM + HMM | 89.89 % | 97.28 % | 413 |

In this historical untuned run the HMM was better on five trips and worse on five for Top-1; these numbers are superseded. Person B later selected and froze the decoder settings with out-of-fold scores on the 80 training trips. The current final development check and failure cases are reported in `docs/section5_decoder.md` and `results/hmm_tuning/final_student_test_check.json`. Do not use this historical table as the current result.

## Historical handoff to Person B: tuning the decoder (completed)
Out-of-fold (OOF) classifier scores exist so the decoder is tuned on realistic scores. The exported LightGBM was trained on all 80 training trips (Top-1 about 94.5 % on them vs 87.5 % on unseen trips), so tuning on its in-sample scores would tune for over-confident emissions.

```
python scripts/make_oof_scores.py                       # ~1 min; builds data_cache/oof/train_oof.parquet
python scripts/eval_pipeline.py --part train --oof --beta 30 --tag b30     # one decoder setting on the 80 training trips
python scripts/eval_pipeline.py --part student_test --tag final            # ONCE, at the end, with the chosen settings
```
- OOF scores come from the same LightGBM recipe (stride 3, default parameters) and the same 5 trip-grouped folds (`configs/cv_folds.json`). Check: pooled over folds their Top-1 is 87.48 % (std 3.74), identical to the recorded grouped-CV result.
- OOF is only for tuning. The final system, the demo and every `student_test`/evaluator number use the single exported model.
- In code: `predict(trip_dir, DecoderConfig(...), scorer=oof_scorer(trip_id))`.
- Tunable: `beta_m` (50), `gap_threshold_s` (120), `emission_sigma_m` (off), `unreachable_log_penalty` (-20). Measured cost: about 1 to 7 s per trip, so a full pass over the 80 training trips is roughly 5 to 8 minutes per setting; use a subset of trips, or a coarse search, for exploration.
- Choose settings with the training trips only. `student_test` is a one-time check; evaluator trips are never used.
- Record the search space, every trial and the final values; the report's HMM section needs them.
