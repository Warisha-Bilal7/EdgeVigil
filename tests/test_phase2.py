"""
EdgeVigil — Phase 2 tests

windowing/Standardizer tests run anywhere (no torch needed).
Model shape tests skip automatically if torch isn't installed in this
environment — they're meant to run on a machine with the full requirements.txt
installed (e.g. your local venv).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "models"))

import numpy as np
import pandas as pd

from simulate import TelemetrySimulator
from inject_failures import FailureInjector, FailureInjection
from windowing import make_windows, Standardizer, METRICS

try:
    import torch
    from lstm_autoencoder import LSTMAutoencoder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestMakeWindows(unittest.TestCase):
    def setUp(self):
        self.sim = TelemetrySimulator(seed=3)
        self.df = self.sim.simulate_fleet({"server": 2, "iot_sensor": 1}, duration_hours=2, step_minutes=5)

    def test_shape(self):
        window_size = 10
        X, y, ts, dev, dtype = make_windows(self.df, window_size=window_size, step=1)
        n_steps_per_device = 24  # 2h * 60 / 5min
        expected_per_device = n_steps_per_device - window_size + 1
        self.assertEqual(X.shape, (expected_per_device * 3, window_size, len(METRICS)))
        self.assertEqual(len(y), len(X))
        self.assertEqual(len(ts), len(X))
        self.assertEqual(len(dev), len(X))
        self.assertEqual(len(dtype), len(X))

    def test_no_labels_all_false_without_injection(self):
        X, y, *_ = make_windows(self.df, window_size=10)
        self.assertFalse(y.any())

    def test_window_flagged_if_any_timestep_anomalous(self):
        injector = FailureInjector(seed=0)
        onset = self.df["timestamp"].iloc[5]
        spec = FailureInjection("server-000", "cpu", "sudden_spike", onset, duration_minutes=5, magnitude=5.0)
        labeled = injector.inject(self.df, [spec])

        X, y, ts, dev, dtype = make_windows(labeled, window_size=10, step=1)
        # The single anomalous row should make every window that contains it positive.
        device_mask = dev == "server-000"
        self.assertTrue(y[device_mask].any())
        self.assertGreaterEqual(y[device_mask].sum(), 1)

    def test_too_short_series_skipped(self):
        short_df = self.sim.simulate_device("s-short", "server", duration_hours=0.2, step_minutes=5)  # ~2-3 rows
        X, y, *_ = make_windows(short_df, window_size=10)
        self.assertEqual(len(X), 0)

    def test_windows_never_cross_device_boundary(self):
        X, y, ts, dev, dtype = make_windows(self.df, window_size=10, step=1)
        self.assertEqual(len(set(dev)), 3)  # each window belongs to exactly one device, 3 devices total

    def test_end_timestamp_is_last_row_of_window(self):
        window_size = 10
        device_df = self.df.loc[self.df["device_id"] == "server-000"].sort_values("timestamp").reset_index(drop=True)
        X, y, ts, dev, dtype = make_windows(self.df, window_size=window_size, step=1)
        first_window_end_ts = ts[dev == "server-000"][0]
        self.assertEqual(pd.Timestamp(first_window_end_ts), device_df["timestamp"].iloc[window_size - 1])


class TestStandardizer(unittest.TestCase):
    def test_fit_transform_roundtrip(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=[50, 10, 200], scale=[5, 2, 30], size=(100, 8, 3))
        scaler = Standardizer().fit(X)
        X_norm = scaler.transform(X)
        flat = X_norm.reshape(-1, 3)
        np.testing.assert_allclose(flat.mean(axis=0), [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(flat.std(axis=0), [1, 1, 1], atol=1e-6)

    def test_constant_feature_does_not_explode(self):
        X = np.ones((10, 5, 2))  # second feature also constant
        scaler = Standardizer().fit(X)
        X_norm = scaler.transform(X)
        self.assertTrue(np.isfinite(X_norm).all())

    def test_to_dict_from_dict_roundtrip(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 5, 4))
        scaler = Standardizer().fit(X)
        restored = Standardizer.from_dict(scaler.to_dict())
        np.testing.assert_allclose(restored.mean, scaler.mean)
        np.testing.assert_allclose(restored.std, scaler.std)

    def test_transform_before_fit_raises(self):
        scaler = Standardizer()
        with self.assertRaises(RuntimeError):
            scaler.transform(np.ones((5, 5, 3)))


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed in this environment")
class TestLSTMAutoencoder(unittest.TestCase):
    def test_forward_shape(self):
        batch, window, n_features = 4, 10, 5
        model = LSTMAutoencoder(n_features=n_features, window_size=window)
        x = torch.randn(batch, window, n_features)
        recon = model(x)
        self.assertEqual(tuple(recon.shape), (batch, window, n_features))

    def test_reconstruction_error_shape_and_nonnegative(self):
        batch, window, n_features = 4, 10, 5
        model = LSTMAutoencoder(n_features=n_features, window_size=window)
        x = torch.randn(batch, window, n_features)
        errors = model.reconstruction_error(x)
        self.assertEqual(tuple(errors.shape), (batch,))
        self.assertTrue((errors >= 0).all())

    def test_overfits_a_single_batch(self):
        # Sanity check the training loop actually reduces loss — catches
        # silently broken gradients (e.g. detached tensors) immediately.
        torch.manual_seed(0)
        batch, window, n_features = 16, 10, 5
        model = LSTMAutoencoder(n_features=n_features, window_size=window, hidden_size=16, latent_dim=8)
        x = torch.randn(batch, window, n_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            loss = ((model(x) - x) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0] * 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
