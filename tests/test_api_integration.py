"""In-process API contract checks (HTTP/TCP needs a normal host runtime)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from pathsnap.api import PredictRequest, health, predict, trips
from pathsnap.paths import DATA_ROOT


class ApiIntegrationTests(unittest.TestCase):
    def test_health_and_trip_listing(self):
        status = health()
        self.assertTrue(status["model_loaded"])
        listing = trips()
        self.assertEqual(len(listing["trip_ids"]), 10)

    def test_all_student_test_trips(self):
        trip_ids = trips()["trip_ids"]
        for trip_id in trip_ids:
            with self.subTest(trip_id=trip_id):
                body = predict(PredictRequest(trip_id=trip_id))
                self.assertEqual(body["trip_id"], trip_id)
                self.assertTrue(body["observations"])
                self.assertEqual(len(body["classifier_arc_ids"]), len(body["observations"]))
                self.assertEqual(len(body["hmm_arc_ids"]), len(body["observations"]))
                self.assertIn("hmm_top1_accuracy", body["metrics"])
                self.assertNotIn("reference_route", body)

    def test_prediction_succeeds_without_route_file(self):
        source = DATA_ROOT / "student_test" / "00000000"
        with tempfile.TemporaryDirectory(prefix="pathsnap-no-route-") as temp:
            destination = Path(temp) / source.name
            destination.mkdir()
            for suffix in ("track", "nodes", "arcs"):
                shutil.copyfile(source / f"{source.name}.{suffix}", destination / f"{source.name}.{suffix}")
            from pathsnap.adapters import make_inference_components
            from pathsnap.decoder.inference import predict_trip

            components, _private_state = make_inference_components(destination)
            result = predict_trip(destination, components)
            self.assertEqual(result["trip_id"], source.name)
            self.assertEqual(len(result["observations"]), len(result["hmm_arc_ids"]))
            self.assertFalse((destination / f"{source.name}.route").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
