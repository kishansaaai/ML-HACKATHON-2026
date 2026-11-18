# Dataset inventory: initial findings

Generated from every supplied trip with `python scripts/inventory.py`; source tables are `results/data_inventory.csv` and `results/data_inventory.summary.json`. Figures can be regenerated with `python scripts/plot_eda.py`. Counts below describe this checked-in data snapshot, not the separate final evaluator set.

## Coverage

| Partition | Trips | GPS observations | Nodes | Arcs |
| --- | ---: | ---: | ---: | ---: |
| Train | 80 | 208,417 | 1,933,485 | 3,876,904 |
| Student test | 10 | 19,160 | 90,689 | 185,041 |

All 90 trip folders contain valid `.track`, `.nodes`, `.arcs` and `.route` files under the supplied layout. The shortest and longest tracks contain 262 and 19,313 observations, respectively.

## Quality and modeling implications

- No invalid GPS coordinates, invalid arc endpoints, or non-increasing timestamp transitions were found by the initial checks. No isolated nodes were found.
- In the training partition, approximately 96.5% of consecutive timestamp intervals are exactly 1 second. There are 2,937 intervals longer than 2 seconds across training trips; 1,260 intervals are longer than 5 seconds. The student-test partition has 44 intervals longer than 5 seconds. B's decoder should segment or otherwise handle these gaps explicitly.
- Every train trip has at least one weakly connected non-isolated graph component, but 79 of 80 have more than one. Per-trip component counts have median 20 and maximum 421. The topology is therefore not a single connected network; graph reachability and disconnected transitions need explicit handling. Components are counted on an undirected interpretation of the directed arc rows for this diagnostic.
- The median reference-route length is about 18.9% of the corresponding GPS row count. A GPS row is not an arc label: per-observation labels require A's monotone route alignment.
- Trip-local road networks must remain separate. Projection is performed per trip and coordinates from different trips must not share one origin.
- Route values are reference labels. In line with the partition policy, train and student-test routes are development/training references; final evaluator labels are unavailable to this inventory and must remain outside model selection and prediction.

## Label and candidate diagnostics

The training alignment diagnostics in `results/label_stats.csv` show about 98.0% labelled observations on average across trips. Trip `00000054` is the main outlier: only about 14.7% of its observations align to the reference route, and about 90.2% of all observations are more than 30 m from the aligned arc. Trip `00000033` is the next major low-coverage case (about 49% labelled). These records should be inspected on the interactive map before treating their derived labels as reliable.

At a raw radius of 50 m, the reference arc falls outside the radius for about 1.38% of labelled training observations. The production candidate policy adds a nearest-K fallback and caps candidate counts; the resulting measured miss rate in the saved feature tables is about 0.61% on training and 0.005% on `student_test`. See `results/candidate_sweep.csv` and the candidate plot for the radius trade-off. `student_test` figures are development diagnostics, not final evaluator estimates.

These are descriptive findings only. Monotone route alignment can still assign uncertain labels, and the student-test sample has only ten trips. They do not establish generalization to the withheld evaluator set.

The scored in-process API run is recorded in `results/api_student_test.csv`. Across its 19,160 observations, the classifier reached 0.9016 Top-1 and HMM/Viterbi reached 0.8991; transition continuity improved from 0.9602 to 0.9784. Thus, this development sample supports a continuity gain but not an overall observation-accuracy gain. One trip had a stitched-route break, and trip `00000056` exceeded the 30-second inference target at 44.5 seconds. See the per-trip table and decoder handoff in the technical report.

