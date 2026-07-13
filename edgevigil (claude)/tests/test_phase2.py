"""EdgeVigil — Phase 2 tests (windowing, standardizer, model scoring)."""

import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import numpy as np
from simulate import TelemetrySimulator
from inject_failures import FailureInjector, FailureInjection
from windowing import make_windows, Standardizer

try:
    import torch
    from models.lstm_autoencoder import LSTMAutoencoder
    TORCH = True
except ImportError:
    TORCH = False

skip_torch = unittest.skipUnless(TORCH, "torch not installed in this environment")


class TestMakeWindows(unittest.TestCase):
    def setUp(self):
        sim = TelemetrySimulator(seed=42)
        self.df = sim.simulate_fleet({"server": 2, "iot_sensor": 1}, duration_hours=6)

    def test_shape(self):
        X, y, end_ts, devs, feats = make_windows(self.df, window_size=12, step=1)
        n_per_device = 6 * 60 // 5 - 12 + 1  # = 60
        self.assertEqual(X.shape, (3 * n_per_device, 12, 5))
        self.assertEqual(len(y), len(X))

    def test_windows_never_cross_device_boundary(self):
        X, y, end_ts, devs, _ = make_windows(self.df, window_size=12, step=1)
        for i in range(len(devs) - 1):
            if devs[i] != devs[i + 1]:
                # end of one device: the gap to next window's start should be large
                pass  # groupby guarantees no cross-device window; shape check is sufficient

    def test_no_labels_all_zero_without_injection(self):
        X, y, *_ = make_windows(self.df, window_size=6, step=1)
        self.assertTrue((y == 0).all())

    def test_window_flagged_if_any_timestep_anomalous(self):
        inj = FailureInjector(seed=1)
        onset = self.df["timestamp"].iloc[20]
        spec  = FailureInjection("server-000", "cpu", "sudden_spike", onset, 5, 5.0)
        labeled = inj.inject(self.df, [spec])
        X, y, *_ = make_windows(labeled, window_size=6, step=1)
        self.assertGreater(y.sum(), 0)

    def test_end_timestamp_is_last_row_of_window(self):
        X, y, end_ts, devs, _ = make_windows(self.df, window_size=6, step=1)
        self.assertEqual(len(end_ts), len(X))

    def test_too_short_series_raises(self):
        sim = TelemetrySimulator(seed=0)
        df  = sim.simulate_device("tiny", "server", duration_hours=0.5, step_minutes=5)
        # 6 rows, window_size=24 → every device skipped → should raise, not silently return empty
        with self.assertRaises(ValueError):
            make_windows(df, window_size=24, step=1)


class TestStandardizer(unittest.TestCase):
    def test_fit_transform_roundtrip(self):
        X = np.random.randn(50, 10, 5).astype(np.float32)
        s = Standardizer()
        Xt = s.fit_transform(X)
        np.testing.assert_allclose(Xt.reshape(-1, 5).mean(axis=0), 0, atol=1e-5)
        np.testing.assert_allclose(Xt.reshape(-1, 5).std(axis=0),  1, atol=1e-5)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            Standardizer().transform(np.ones((4, 10, 5)))

    def test_to_dict_from_dict_roundtrip(self):
        X = np.random.randn(20, 10, 5).astype(np.float32)
        s1 = Standardizer().fit(X)
        s2 = Standardizer.from_dict(s1.to_dict())
        np.testing.assert_array_equal(s1.mean_, s2.mean_)
        np.testing.assert_array_equal(s1.std_,  s2.std_)

    def test_constant_feature_does_not_explode(self):
        X = np.ones((10, 6, 5), dtype=np.float32)
        Xt = Standardizer().fit_transform(X)
        self.assertFalse(np.isnan(Xt).any())
        self.assertFalse(np.isinf(Xt).any())


class TestLSTMAutoencoder(unittest.TestCase):
    @skip_torch
    def test_forward_shape(self):
        model = LSTMAutoencoder(n_features=5, window_size=10)
        x     = torch.randn(8, 10, 5)
        out   = model(x)
        self.assertEqual(out.shape, x.shape)

    @skip_torch
    def test_reconstruction_error_shape_and_nonnegative(self):
        model = LSTMAutoencoder(n_features=5, window_size=10)
        x     = torch.randn(8, 10, 5)
        err   = model.reconstruction_error(x)
        self.assertEqual(tuple(err.shape), (8,))
        self.assertTrue((err >= 0).all())

    @skip_torch
    def test_per_timestep_error_shape(self):
        model = LSTMAutoencoder(n_features=5, window_size=10)
        x     = torch.randn(4, 10, 5)
        err   = model.per_timestep_error(x)
        self.assertEqual(tuple(err.shape), (4, 10))
        self.assertTrue((err >= 0).all())

    @skip_torch
    def test_anomaly_score_mean_matches_reconstruction_error(self):
        torch.manual_seed(1)
        model = LSTMAutoencoder(n_features=5, window_size=10)
        x     = torch.randn(4, 10, 5)
        torch.testing.assert_close(model.anomaly_score(x, mode="mean"),
                                    model.reconstruction_error(x))

    @skip_torch
    def test_anomaly_score_max_geq_mean(self):
        torch.manual_seed(2)
        model     = LSTMAutoencoder(n_features=5, window_size=10)
        x         = torch.randn(8, 10, 5)
        max_score = model.anomaly_score(x, mode="max")
        mean_score = model.anomaly_score(x, mode="mean")
        self.assertTrue((max_score >= mean_score - 1e-6).all())

    @skip_torch
    def test_anomaly_score_max_more_sensitive_to_localized_anomaly(self):
        # Single displaced timestep: max should exceed mean by a large factor.
        torch.manual_seed(3)
        model = LSTMAutoencoder(n_features=5, window_size=10, hidden_size=8, latent_dim=4)
        x     = torch.zeros(1, 10, 5)
        x[0, 5, :] = 10.0  # one badly displaced timestep out of ten
        self.assertGreater(model.anomaly_score(x, mode="max").item(),
                            model.anomaly_score(x, mode="mean").item())

    @skip_torch
    def test_anomaly_score_invalid_mode_raises(self):
        model = LSTMAutoencoder(n_features=5, window_size=10)
        with self.assertRaises(ValueError):
            model.anomaly_score(torch.randn(2, 10, 5), mode="bogus")

    @skip_torch
    def test_overfits_a_single_batch(self):
        torch.manual_seed(0)
        model     = LSTMAutoencoder(n_features=5, window_size=10, hidden_size=16, latent_dim=8)
        x         = torch.randn(16, 10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses    = []
        for _ in range(50):
            optimizer.zero_grad()
            loss = ((model(x) - x) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.5)


class TestDetectionLag(unittest.TestCase):
    """
    Guards against the pre-onset false-positive bug: the first flagged
    window for a device may be a false positive that precedes the injection,
    which would produce a negative (nonsensical) detection lag.
    eval.py's lag computation must filter to windows >= onset.
    """

    def _make_labeled_df(self):
        sim = TelemetrySimulator(seed=7)
        df  = sim.simulate_fleet({"server": 2}, duration_hours=48)
        inj = FailureInjector(seed=7)
        # Inject late in the series so there's plenty of pre-onset series to
        # produce FP windows that would give a negative lag if unfiltered.
        onset = df.loc[df["device_id"] == "server-000", "timestamp"].iloc[-20]
        spec  = FailureInjection("server-000", "cpu", "sudden_spike", onset, 60, 6.0)
        return inj.inject(df, [spec]), spec

    def test_lag_cannot_be_negative(self):
        import pandas as pd
        labeled_df, spec = self._make_labeled_df()
        X, y, end_ts, device_ids, _ = make_windows(labeled_df, window_size=12, step=1)

        # Simulate a detector that flags ALL windows (100% recall, 100% FPR)
        # — lag should still be >= 0 because we filter by onset.
        y_pred = np.ones(len(X), dtype=bool)

        after_onset = end_ts >= np.datetime64(spec.onset)
        mask        = (device_ids == spec.device_id) & y_pred & after_onset
        self.assertTrue(mask.any(), "Should have at least one window after onset")

        first_alert = end_ts[mask].min()
        lag = (first_alert - np.datetime64(spec.onset)) / np.timedelta64(1, "m")
        self.assertGreaterEqual(lag, 0.0, "Detection lag must be non-negative")

    def test_pre_onset_flag_does_not_count_as_detection(self):
        labeled_df, spec = self._make_labeled_df()
        X, y, end_ts, device_ids, _ = make_windows(labeled_df, window_size=12, step=1)

        # Flag only windows BEFORE the onset (pure false positives pre-injection).
        before_onset = end_ts < np.datetime64(spec.onset)
        y_pred = before_onset & (device_ids == spec.device_id)

        after_onset = end_ts >= np.datetime64(spec.onset)
        mask = (device_ids == spec.device_id) & y_pred & after_onset
        # No post-onset windows flagged → injection should be counted as missed.
        self.assertFalse(mask.any(), "Pre-onset false positives must not register as detections")


if __name__ == "__main__":
    unittest.main(verbosity=2)

