# Section 5 — HMM and Viterbi route reconstruction (Person B)

## Implementation summary

Each trip gets its own locally centred equirectangular coordinate frame in metres. The same origin is
used for the trip's GPS fixes and road nodes. The source files store longitude then latitude; `.arcs`
rows store directed `(from_node, to_node)` indices, and the zero-based row number is retained as the
arc ID. A sparse adjacency list stores each directed edge once with its projected endpoint distance.
An in-repository uniform-grid index maps spatial cells back to original arc IDs and computes exact
point-to-segment projections after lookup.

The decoder state at observation *t* is a candidate arc plus a fractional position along its directed
geometry. Person A's candidate probabilities are normalized across that observation's candidate set.
The emission is the log of that normalized probability, optionally combined with a Gaussian penalty
on perpendicular distance. The transition score is
`-|network_distance - GPS_distance| / beta`. Network distance includes the remainder of the source
arc, a directed bounded Dijkstra path between graph nodes, and the prefix of the destination arc.
Unreachable candidates receive a finite configured penalty. Searches are pruned when a reachable path
would score worse than that penalty, which bounds work without changing the best score under the
configured penalty rule. The Viterbi recurrence keeps one score per candidate and a backpointer per
state, so candidate-set sizes may differ at each timestamp.

Missing-candidate observations remain null and split decoding into independent segments. Invalid or
non-increasing timestamps and gaps above the configured threshold also restart the decoder. Route
stitching inserts the shortest legal directed path between different selected arcs, removes redundant
consecutive duplicates, and reports breaks across unreachable pairs, missing candidates, or time gaps.
No graph edge is fabricated to hide a break.

## Evaluation definitions

Candidate-level precision, recall, F1, confusion matrix and log loss treat each candidate row as a
binary label: the aligned reference arc is positive and other candidates are negative. Candidate miss
rate counts a labeled observation whose aligned reference arc is absent from the generated candidates.
Observation Top-1 accuracy compares a single per-observation arc prediction with Person A's monotone
alignment label. It must never compare predictions with raw `.route` positions. Route continuity is
the fraction of adjacent predicted pairs that are either the same arc or directed head-to-tail
connected; the break count is reported alongside it. Route overlap uses arc sets (precision, recall,
F1 and Jaccard), while sequence edit distance compares ordered arc IDs.

Prediction and scoring are separate: `predict_trip` reads only `.track`, `.nodes`, and `.arcs`.
Ground-truth labels and the ordered reference route are accepted only by the post-inference metrics
module. The leakage test runs prediction in a temporary trip directory where no reference file exists.

## Results and limitations

### Tuning protocol and results

HMM parameters were compared with Person A's LightGBM probabilities generated out of fold (OOF) on
training trips. Pilot sweeps used the same eight trips (`00000001`, `02`, `03`, `04`, `05`, `06`,
`08`, `10`; 18,023 labeled observations). Beta values 30, 50, 100, 200 and 400 were evaluated at a
120 s gap. Additional pilot runs tested 5, 15 and 30 s gaps, emission sigmas of 25 and 50 m, and
beta values 100, 150 and 200. At beta 200, gap 30 s, sigma 25 m, the default unreachable penalty
of −20 gave weighted HMM Top-1 of 0.7497 versus classifier 0.7700, with mean trip continuity 0.9593
versus 0.9448. Changing the unreachable penalty to −10 raised pilot Top-1 to 0.7583; −5 raised it
to 0.7687, with continuity 0.9554. A −40 trial was stopped during the first trip after that trip
took 120.9 s; it has no aggregate score and was excluded from selection. The pilot summary table is
`results/hmm_tuning/pilot_sweep_summary.csv`.

The selected frozen configuration is beta 200 m, gap 30 s, emission sigma 25 m, unreachable penalty
−5, with the other decoder settings unchanged. The complete OOF comparison on all 80 training trips
included 187,866 labeled observations. Weighted Top-1 was 0.8790 for the classifier and 0.8792 for
the HMM (difference +0.0002). Mean trip continuity was 0.9506 and 0.9623 respectively; total breaks
fell from 8,245 to 5,505. HMM Top-1 was higher on 47 trips, tied on 5, and lower on 28. The small
aggregate Top-1 change should be read alongside those mixed per-trip outcomes.

The largest per-trip Top-1 regressions were trips `00000081` (−0.0715; continuity +0.0099),
`00000083` (−0.0644; continuity −0.0398), and `00000010` (−0.0432; continuity −0.0055). Trip
`00000081` also had 83 route-stitch breaks despite fewer adjacent-pair breaks. These examples are
reported as failure cases. Candidate diagnostics differ: trip `00000081` had a 0.0060 candidate miss
rate and 0.2906 ambiguous-label fraction, while trips `00000083` and `00000010` had no candidate
misses or ambiguous labels in the alignment statistics. The latter regressions therefore remain
decoder error cases even when candidate coverage is available; the current metrics do not establish
their cause. The full ranked table, including candidate diagnostics, is
`results/hmm_tuning/failure_analysis_top10.csv`.

After configuration freeze, one development check on the 10 `student_test` trips (19,143 labeled
observations, scored with the exported classifier) measured weighted Top-1 of 0.9021 for classifier
and 0.9071 for HMM; mean trip continuity was 0.9558 and 0.9697, and total breaks were 774 and 501.
This check was not used to select parameters. The separate evaluator-testing partition was not
accessed.

### Reproduction

Use the cached candidate tables and OOF scores where available. If OOF scores need rebuilding, run
`.venv/bin/python scripts/make_oof_scores.py`. The result runner writes per-trip CSV and summary JSON
under `results/hmm_tuning/` and refuses to overwrite an existing tag. The complete training run is
reproduced with:

```sh
.venv/bin/python scripts/tune_hmm.py --beta 200 --gap 30 --sigma 25 \
  --unreachable-penalty -5 --tag frozen_candidate_full80_reproduction --oof
```

The frozen config is recorded in `configs/decoder_best.json` and is the `DecoderConfig` default. The
single post-freeze development check was run with:

```sh
.venv/bin/python scripts/tune_hmm.py --part student_test --beta 200 --gap 30 \
  --sigma 25 --unreachable-penalty -5 --tag final_student_test_check
```

The final check result is in `results/hmm_tuning/final_student_test_check.csv` and its JSON summary.
Use a fresh tag for any rerun because the runner refuses to overwrite prior results.
Synthetic and leakage tests run with `.venv/bin/python -m unittest discover -s tests -v`.

## AI assistance disclosure draft

AI assistance was used to draft and review decoder code and synthetic tests. The team must verify,
understand, and be able to explain the submitted implementation. All reported measurements must come
from the committed benchmark or evaluation scripts; AI-generated text is not experimental evidence.
