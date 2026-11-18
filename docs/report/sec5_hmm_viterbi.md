# 5. HMM and Viterbi Route Reconstruction

## 5.1 Design (answers to the required questions)

- **Hidden states.** At each GPS point the state is one candidate arc together with the position of the point along it. Candidate sets differ in size from point to point.
- **Emission.** The log of the classifier's probability, normalised over that point's candidates, plus a Gaussian penalty on the perpendicular distance (sigma 25.0 m).
- **Transition.** The score between consecutive candidates is minus the absolute difference between the road distance along the network (remaining part of the source arc, a directed shortest path between nodes, and the part of the destination arc) and the straight GPS distance, divided by beta (200.0 m). Searches are bounded (10 km). Unreachable pairs get a finite penalty (-5.0).
- **Missing candidates.** A point without candidates is left empty and splits the decoding into independent segments; nothing is guessed for it.
- **Disconnected segments.** Time gaps above 30.0 s, invalid or non-increasing timestamps, missing candidates and unreachable transitions start a new segment. Route stitching inserts the shortest legal directed path between different selected arcs and never fabricates an edge; each break is reported and drawn as a separate polyline.
- **Why an HMM at all.** The classifier scores each point alone and can jump between roads; the transition term rewards routes whose road distance agrees with the GPS movement.

## 5.2 How the parameters were chosen

Tuned only on training trips with **out-of-fold** LightGBM scores (`scripts/make_oof_scores.py`): the exported model is trained on all 80 training trips, so its scores on them are over-confident (about 94.5 % Top-1 against 87.5 % unseen) and would mislead the tuning. A pilot sweep on 8 training trips tried beta 30, 50, 100, 150, 200 and 400, gap thresholds 5, 15, 30 and 120 s, emission sigma 25 and 50 m, and unreachable penalties -20, -10 and -5 (`results/hmm_tuning/pilot_sweep_summary.csv`); the frozen setting was then compared with the classifier on all 80 training trips, and checked once on the student test after freezing (`scripts/tune_hmm.py`, `configs/decoder_best.json`). The best penalty (-5) is at the edge of the range that was tried; less negative values were not tested.

| Setting | Value |
| --- | ---: |
| beta | 200.0 m |
| gap threshold | 30.0 s |
| emission sigma | 25.0 m |
| unreachable penalty | -5.0 |

## 5.3 Did the HMM improve the route?

**Training trips, out-of-fold (80 trips, 187,866 labelled points):** weighted Top-1 0.8790 for the classifier and 0.8792 for the HMM (+0.0002); mean continuity 0.9506 against 0.9623; route breaks 8,245 against 5,505. The HMM was better on Top-1 for 47 trips, tied on 5 and worse on 28. These 80 trips include the 8 pilot trips used for tuning.

**Student test (one check after freezing):**

| Trip | Labelled points | Top-1 classifier | Top-1 HMM | Continuity classifier | Continuity HMM | Breaks classifier | Breaks HMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00000000 | 2503 | 0.9533 | 0.9557 | 0.9828 | 0.9920 | 43 | 20 |
| 00000007 | 1354 | 0.9594 | 0.9498 | 0.9993 | 0.9993 | 1 | 1 |
| 00000009 | 1049 | 0.8847 | 0.8932 | 0.9122 | 0.9389 | 92 | 64 |
| 00000040 | 2995 | 0.9613 | 0.9639 | 0.9813 | 0.9876 | 56 | 37 |
| 00000052 | 262 | 0.9733 | 0.9809 | 0.9349 | 0.9579 | 17 | 11 |
| 00000056 | 3941 | 0.9183 | 0.9437 | 0.9282 | 0.9580 | 284 | 166 |
| 00000065 | 768 | 0.8672 | 0.8711 | 0.9674 | 0.9778 | 25 | 17 |
| 00000076 | 1486 | 0.9260 | 0.9280 | 0.9401 | 0.9508 | 89 | 73 |
| 00000077 | 3911 | 0.7834 | 0.7781 | 0.9703 | 0.9818 | 116 | 71 |
| 00000091 | 874 | 0.9119 | 0.9096 | 0.9416 | 0.9530 | 51 | 41 |
| **Pooled / mean** | 19,143 | 0.9021 | 0.9071 | 0.9558 | 0.9697 | 774 | 501 |

The HMM improves continuity on every trip except one that is unchanged and cuts breaks from 774 to 501; Top-1 improves on 7 trips and falls on 3 (00000007, 00000077, 00000091). The Top-1 gain is small (+0.0050) and comes from ten trips, so it is weak evidence on its own; the continuity gain is consistent.

**Sequence level (whole route against the reference route).** `python scripts/sequence_metrics.py` scores the full stitched route (`results/sequence_metrics_student_test.csv`):

| Trip | Arc-set F1 classifier | Arc-set F1 HMM | HMM precision | HMM recall |
| --- | ---: | ---: | ---: | ---: |
| 00000000 | 0.9223 | 0.8900 | 0.8054 | 0.9944 |
| 00000007 | 0.9837 | 0.9798 | 0.9758 | 0.9837 |
| 00000009 | 0.8617 | 0.9294 | 0.8699 | 0.9977 |
| 00000040 | 0.9453 | 0.9102 | 0.8351 | 1.0000 |
| 00000052 | 0.8814 | 0.6889 | 0.5314 | 0.9789 |
| 00000056 | 0.9104 | 0.9171 | 0.8623 | 0.9794 |
| 00000065 | 0.9034 | 0.8555 | 0.7474 | 1.0000 |
| 00000076 | 0.8583 | 0.8079 | 0.6827 | 0.9893 |
| 00000077 | 0.8859 | 0.8933 | 0.8072 | 1.0000 |
| 00000091 | 0.8310 | 0.7798 | 0.6391 | 1.0000 |
| **Mean** | 0.8983 | 0.8652 | 0.7756 | 0.9923 |

On this measure the HMM is **not** better: its route covers almost every reference arc (recall 0.9923) but also contains arcs that are not in the reference (precision 0.7756), so mean arc-set F1 is 0.8652 against 0.8983 for the classifier alone, and the HMM is higher on only 3 of 10 trips. The cause (for example detours inserted by the stitching, or the permissive unreachable penalty) was not investigated. So the honest summary is: the HMM gives a more connected route and slightly better point accuracy, but it adds spurious arcs.

## 5.4 Where it did not help

Largest per-trip Top-1 regressions on the training trips (`results/hmm_tuning/failure_analysis_top10.csv`): 00000081 (-0.0715), 00000083 (-0.0644) and 00000010 (-0.0432). Trip 00000081 also has many ambiguous labels, which may explain part of it; for 00000083 and 00000010 the candidate coverage and labels are fine, so these remain decoder error cases whose cause we did not establish.

**Runtime.** Slowest student-test trip through the whole API: 15.4 s (target 30 s).
