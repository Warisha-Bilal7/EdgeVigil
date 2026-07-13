"""EdgeVigil — Phase 1 tests (simulator + injector correctness)."""

import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "data"))

import numpy as np, pandas as pd
from simulate import TelemetrySimulator, METRICS, DEVICE_PROFILES
from inject_failures import FailureInjector, FailureInjection


class TestTelemetrySimulator(unittest.TestCase):
    def setUp(self):
        self.sim = TelemetrySimulator(seed=42)

    def test_simulate_device_shape(self):
        df = self.sim.simulate_device("s-0", "server", duration_hours=2, step_minutes=5)
        self.assertEqual(len(df), 24)
        for m in METRICS:
            self.assertIn(m, df.columns)
            self.assertFalse(df[m].isna().any())

    def test_simulate_fleet_device_counts(self):
        df = self.sim.simulate_fleet({"server": 2, "workstation": 3, "iot_sensor": 1}, duration_hours=1)
        counts = df["device_type"].value_counts()
        self.assertEqual(counts["server"],      2 * 12)
        self.assertEqual(counts["workstation"], 3 * 12)
        self.assertEqual(counts["iot_sensor"],  1 * 12)

    def test_device_type_separation(self):
        df         = self.sim.simulate_fleet({"server": 5, "iot_sensor": 5}, duration_hours=12)
        server_cpu = df.loc[df["device_type"] == "server",     "cpu"].mean()
        iot_cpu    = df.loc[df["device_type"] == "iot_sensor", "cpu"].mean()
        self.assertGreater(server_cpu - iot_cpu, 20)

    def test_values_within_bounds(self):
        df = self.sim.simulate_fleet({"workstation": 3}, duration_hours=24)
        for m in METRICS:
            p = DEVICE_PROFILES["workstation"][m]
            self.assertGreaterEqual(df[m].min(), p.min_value - 1e-6)
            self.assertLessEqual(df[m].max(), p.max_value + 1e-6)

    def test_held_out_device_type(self):
        self.sim.held_out_device_type("edge_gateway", base_on="iot_sensor",
                                       perturb={"cpu": {"mean": 1.5}})
        df = self.sim.simulate_device("gw-0", "edge_gateway", duration_hours=6)
        self.assertEqual(len(df), 72)
        self.assertGreater(df["cpu"].mean(), DEVICE_PROFILES["iot_sensor"]["cpu"].mean)

    def test_reproducible_with_seed(self):
        df1 = TelemetrySimulator(seed=7).simulate_device("s-0", "server", duration_hours=1)
        df2 = TelemetrySimulator(seed=7).simulate_device("s-0", "server", duration_hours=1)
        pd.testing.assert_frame_equal(df1, df2)


class TestFailureInjector(unittest.TestCase):
    def setUp(self):
        self.sim = TelemetrySimulator(seed=1)
        self.df  = self.sim.simulate_fleet({"server": 2, "workstation": 1}, duration_hours=24)
        self.inj = FailureInjector(seed=1)

    def test_does_not_mutate_input(self):
        original = self.df.copy(deep=True)
        spec = FailureInjection("server-000", "cpu", "sudden_spike",
                                 self.df["timestamp"].iloc[50], 15, 5.0)
        self.inj.inject(self.df, [spec])
        pd.testing.assert_frame_equal(self.df, original)

    def test_sudden_spike_window_count(self):
        onset = self.df["timestamp"].iloc[50]
        spec  = FailureInjection("server-000", "cpu", "sudden_spike", onset, 25, 5.0)
        labeled = self.inj.inject(self.df, [spec])
        device = labeled.loc[labeled["device_id"] == "server-000"].sort_values("timestamp")
        self.assertEqual(int(device["is_anomaly"].sum()), int(25 / 5))

    def test_gradual_drift_window_count(self):
        onset = self.df["timestamp"].iloc[50]
        spec  = FailureInjection("workstation-000", "cpu", "gradual_drift", onset, 60, 4.0)
        labeled = self.inj.inject(self.df, [spec])
        device = labeled.loc[labeled["device_id"] == "workstation-000"].sort_values("timestamp")
        self.assertEqual(int(device["is_anomaly"].sum()), 12)

    def test_slow_leak_never_reverts(self):
        onset = self.df["timestamp"].iloc[10]
        spec  = FailureInjection("server-001", "memory", "slow_leak", onset, 30, 4.0)
        labeled = self.inj.inject(self.df, [spec])
        device = labeled.loc[labeled["device_id"] == "server-001"].sort_values("timestamp").reset_index(drop=True)
        oi = device.index[device["timestamp"] >= onset][0]
        self.assertTrue(device.loc[oi:, "is_anomaly"].all())

    def test_failure_onset_recorded(self):
        onset = self.df["timestamp"].iloc[50]
        spec  = FailureInjection("server-000", "cpu", "sudden_spike", onset, 15, 5.0)
        labeled = self.inj.inject(self.df, [spec])
        recorded = labeled.loc[labeled["failure_onset"].notna(), "failure_onset"]
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded.iloc[0], onset)

    def test_multiple_injections_independent(self):
        specs = [
            FailureInjection("server-000", "cpu",    "sudden_spike",  self.df["timestamp"].iloc[30], 15, 5.0),
            FailureInjection("server-001", "memory", "gradual_drift", self.df["timestamp"].iloc[60], 30, 4.0),
        ]
        labeled = self.inj.inject(self.df, specs)
        self.assertEqual(labeled.loc[labeled["device_id"] == "server-000", "is_anomaly"].sum(), 3)
        self.assertEqual(labeled.loc[labeled["device_id"] == "server-001", "is_anomaly"].sum(), 6)

    def test_random_injection_batch_no_device_repeats(self):
        batch = self.inj.random_injection_batch(self.df, n_injections=3)
        ids   = [s.device_id for s in batch]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
