"""ASGI-level HTTP contract checks without opening a local TCP socket."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import httpx

from pathsnap import api
from pathsnap.paths import DATA_ROOT


class ApiIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # TestClient hangs in this environment with Starlette 1.10/httpx 0.28;
        # ASGITransport still exercises the real HTTP routes in-process.
        api._validate_evaluator_partition()
        api._warm_up()

    async def asyncSetUp(self):
        transport = httpx.ASGITransport(app=api.app, raise_app_exceptions=True)
        self.client = httpx.AsyncClient(transport=transport, base_url="http://testserver")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_and_trip_listing_contract(self):
        health = await self.client.get("/health")
        self.assertEqual(health.status_code, 200, health.text)
        payload = health.json()
        self.assertEqual(payload["partition"], "student_test")
        self.assertTrue(payload["model_loaded"])
        self.assertEqual(payload["trips_available"], 10)
        listing = await self.client.get("/trips")
        body = listing.json()
        self.assertEqual(body["partition"], "student_test")
        self.assertEqual(len(body["trips"]), 10)
        self.assertEqual(set(body["trips"][0]), {"trip_id", "num_observations", "num_nodes", "num_arcs"})

    async def test_all_ten_predictions_match_contract(self):
        trip_ids = [p.name for p in sorted((DATA_ROOT / "student_test").iterdir()) if p.is_dir()]
        self.assertEqual(len(trip_ids), 10)
        for trip_id in trip_ids:
            with self.subTest(trip_id=trip_id):
                response = await self.client.post("/predict", json={"trip_id": trip_id})
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["trip_id"], trip_id)
                self.assertEqual(len(body["raw_gps"]), body["num_observations"])
                self.assertEqual(len(body["observations"]), body["num_observations"])
                self.assertIn("hmm_route_segments", body)
                self.assertIn("hmm_route_polyline", body)
                self.assertIn("classifier_route_segments", body)
                self.assertEqual(set(body["metrics"]), {"classifier_only", "hmm"})
                for metric_group in body["metrics"].values():
                    self.assertIn("top1_accuracy", metric_group)
                    self.assertIn("route_continuity", metric_group)
                forbidden = {"route", "reference", "reference_route", "route_arc_ids"}
                self.assertFalse(forbidden.intersection(body))

    async def test_prediction_works_without_route_file(self):
        source = DATA_ROOT / "student_test" / "00000000"
        with tempfile.TemporaryDirectory(prefix="pathsnap-no-route-") as temp:
            destination = Path(temp) / source.name
            destination.mkdir()
            for suffix in ("track", "nodes", "arcs"):
                shutil.copyfile(source / f"{source.name}.{suffix}", destination / f"{source.name}.{suffix}")
            original_root = api._trip_root
            api._trip_root = lambda: (Path(temp), "evaluator")
            try:
                response = await self.client.post("/predict", json={"trip_id": source.name})
            finally:
                api._trip_root = original_root
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsNone(response.json()["metrics"])
            self.assertFalse((destination / f"{source.name}.route").exists())
            self.assertNotIn("reference_route", response.json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
