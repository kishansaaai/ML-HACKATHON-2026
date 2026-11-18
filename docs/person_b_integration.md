# Person B integration contract

The repository currently contains the PathSnap planning document and data, but no implementation
from Person A or Person C. The decoder therefore accepts Person A's functions explicitly and does
not ship a replacement classifier or a production candidate policy.

```python
from pathsnap.decoder import Candidate, DecoderConfig
from pathsnap.decoder.inference import InferenceComponents, predict_trip

components = InferenceComponents(
    generate_candidates=generate_candidates,
    build_features=build_candidate_features,
    score_candidates=score_candidates,
)
prediction = predict_trip("Student folder/train/00000001", components, DecoderConfig())
```

`generate_candidates(projected_points, graph)` returns one list of `Candidate` objects per GPS
observation. Invalid GPS positions are passed as `None`; the orchestrator enforces an empty candidate
list at those positions. A `Candidate` contains the original `arc_id`, a directed tail-to-head
fraction, perpendicular distance in metres, and projected position. Person A can use
`graph.spatial_index.query(point_xy, radius_m)` plus `nearest(point_xy, k)` for the required fallback.

`build_features(observations, candidate_sets, graph)` returns features in flattened observation then
candidate order. `score_candidates(features)` returns either a flat probability/score vector in that
same order or nested values aligned to each observation's candidates. The decoder normalizes these
values within each observation before using log emissions. No labels or reference routes are passed
to any adapter.

Person C can import `predict_trip` and call `evaluate_predictions` only after prediction, supplying
monotone per-observation reference labels and route IDs from C's server-side scoring layer. The
prediction module itself never opens a reference file. The route polyline is grouped into connected
line segments, so the map does not draw a false connecting line across a reported route break.

## Local commands

The implementation has no third-party runtime dependencies.

```sh
python3 -m unittest discover -s tests -v
python3 scripts/benchmark_decoder.py "Student folder/train/00000001" \
  --output outputs/decoder_benchmark_train_00000001.json
```

The benchmark uses a bounded 50 m/eight-candidate sample and uniform scores solely to time graph,
transition, Viterbi and route-stitching primitives. It does not score predictions or read labels.
