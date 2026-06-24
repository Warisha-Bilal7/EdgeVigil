"""
EdgeVigil — Sliding window utility
Phase 2/3: shared by the baseline LSTM-autoencoder and (later) the
domain-adversarial encoder — both consume the same windowed tensors.

Converts long-format telemetry (one row per device per timestep) into
fixed-length windows. A window is "anomalous" if any timestep inside it is
anomalous — mirrors how an operator would treat any window touching a
failure as something worth flagging, and keeps window-level F1 comparable
across failure kinds of different durations.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

METRICS = ["cpu", "memory", "disk_io", "network_latency", "temperature"]


def make_windows(df: pd.DataFrame, window_size: int, step: int = 1,
                  metrics: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window of length `window_size` across each device's telemetry
    independently — windows never cross device boundaries.

    Returns:
        X:             [n_windows, window_size, n_features] float64
        labels:        [n_windows] bool   — True if any timestep in the window is anomalous
        end_timestamps:[n_windows]         — timestamp of the window's last row
        device_ids:    [n_windows] object
        device_types:  [n_windows] object
    """
    metrics = metrics or METRICS
    X_list, labels_list, ts_list, dev_list, type_list = [], [], [], [], []

    for device_id, group in df.groupby("device_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < window_size:
            continue  # not enough history for even one window

        values = group[metrics].to_numpy(dtype=np.float64)
        has_labels = "is_anomaly" in group.columns
        anomaly = group["is_anomaly"].to_numpy() if has_labels else np.zeros(len(group), dtype=bool)
        timestamps = group["timestamp"].to_numpy()
        device_type = group["device_type"].iloc[0]

        n = len(group)
        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            X_list.append(values[start:end])
            labels_list.append(bool(anomaly[start:end].any()))
            ts_list.append(timestamps[end - 1])
            dev_list.append(device_id)
            type_list.append(device_type)

    if not X_list:
        return (np.empty((0, window_size, len(metrics))), np.empty(0, dtype=bool),
                np.empty(0, dtype="datetime64[ns]"), np.empty(0, dtype=object), np.empty(0, dtype=object))

    return (np.stack(X_list), np.array(labels_list, dtype=bool),
            np.array(ts_list), np.array(dev_list, dtype=object), np.array(type_list, dtype=object))


class Standardizer:
    """
    Per-feature mean/std. Fit ONLY on normal (failure-free) training windows,
    then applied unchanged at eval time — fitting on data containing failures
    would let the anomalies skew the "normal" baseline they're meant to stand out from.
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-6] = 1e-6  # guard against div-by-zero on a near-constant feature
        self.std = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Standardizer.fit() must be called before transform()")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        s = cls()
        s.mean = np.array(d["mean"])
        s.std = np.array(d["std"])
        return s


if __name__ == "__main__":
    try:
        from simulate import TelemetrySimulator
    except ImportError:
        from core.data.simulate import TelemetrySimulator

    sim = TelemetrySimulator(seed=0)
    df = sim.simulate_fleet({"server": 2}, duration_hours=2, step_minutes=5)
    X, y, ts, dev, dtype = make_windows(df, window_size=10, step=1)
    print(f"X shape: {X.shape}  (n_windows, window_size, n_features)")
    print(f"labels: {y.shape}, all False expected (no injections): {not y.any()}")

    scaler = Standardizer().fit(X)
    X_norm = scaler.transform(X)
    print(f"per-feature mean after standardizing (~0): {X_norm.reshape(-1, X_norm.shape[-1]).mean(axis=0).round(3)}")
