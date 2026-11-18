# 6. Prediction Map (Folium)

`python scripts/make_prediction_map.py --trip 00000052 --part student_test` writes `reports/prediction_map_00000052.html`, an interactive Folium map with four toggleable layers: raw GPS points (red), the road network of the trip (grey), the reference route (orange, read only after the prediction, development trips only) and the predicted classifier + HMM route (blue). The layer control switches each one on and off.

![Trip 00000052 (student test): GPS points, predicted route over the reference route and the road network](docs/report/fig/prediction_map_00000052.png)

On this trip the predicted route follows the reference along the main road; the loop near the interchange is where the GPS wanders and where the HMM route is least accurate (trip 00000052: Top-1 classifier 0.9733, HMM 0.9809; but arc-set F1 0.8814 against 0.6889, see section 5.3). The evaluator UI draws the same GPS and predicted route layers from the API response, with disconnected route segments as separate polylines, never joined by a false line.
