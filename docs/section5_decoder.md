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

The supplied training trip `00000001` contains 88,809 node rows, 165,790 arc rows, 1,315 GPS rows,
and 525 reference-route rows. The GPS, node, and arc rows have 3, 2, and 2 whitespace-separated
columns respectively; the reference route has one integer per row. All 165,790 arc endpoints are in
range, the route IDs are in range, and there are no self-loop rows in this trip. Its timestamps have
1,314 intervals: 1,289 are exactly one second and the largest gap is 16 seconds. Measured engine timings and process
memory are stored in `outputs/decoder_benchmark_train_00000001.json`. On the current execution
environment, graph construction plus the spatial index took 0.895 s and peak process memory was
166.9 MB. For 100 sampled observations capped at eight candidates within 50 m, spatial lookup took
0.0027 s, 3,145 candidate transition pairs took 0.0192 s, Viterbi took 0.0192 s, and route stitching
took 0.0256 s. This is a primitive timing benchmark with uniform test scores, not a model evaluation
or a full-trip latency claim, and timings may differ on the demo laptop.

No classifier, candidate generator, trained artifact, or upstream feature module was present in the
repository at implementation time. Therefore classifier-only versus HMM accuracy, tuned beta/gap
values, decoder impact, and full-trip latency are not yet measured. Those claims must be generated
after Person A exports the agreed adapters and the team runs group-aware validation. Do not describe
the initial defaults in `DecoderConfig` as experimentally optimal. The pasted implementation brief
calls `student_test` a final holdout, while the official Models and Training Guidelines explicitly
permit student-testing trips for development comparisons and decoder optimisation. Per the brief's
instruction to resolve conflicts in favour of the official documents, use `student_test` only as
development data and reserve the separate evaluator-testing partition and labels for final evaluation.

## AI assistance disclosure draft

AI assistance was used to draft and review decoder code and synthetic tests. The team must verify,
understand, and be able to explain the submitted implementation. All reported measurements must come
from the committed benchmark or evaluation scripts; AI-generated text is not experimental evidence.
