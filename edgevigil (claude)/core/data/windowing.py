"""
EdgeVigil — Windowing  (Phase 2)
Sliding-window construction over long-format telemetry.

A window's label = 1 if ANY timestep inside it is anomalous.
Windows never span two devices.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

METRICS = ["cpu", "memory", "disk_io", "network_latency", "temperature"]


def make_windows(df: pd.DataFrame, window_size: int, step: int = 1,
                  feature_cols: Optional[List[str]] = None) -> tuple:
    """
    Returns (X, y, end_timestamps, device_ids, feature_names).
      X   : [n_windows, window_size, n_features]  float32
      y   : [n_windows]  int64  (0/1 — any-anomaly-in-window)
    """
    feature_cols = feature_cols or METRICS
    has_labels   = "is_anomaly" in df.columns
    X_list, y_list, dev_list, end_list = [], [], [], []

    for device_id, group in df.groupby("device_id", sort=False):
        group  = group.sort_values("timestamp").reset_index(drop=True)
        values = group[feature_cols].to_numpy(dtype=np.float32)
        ts     = group["timestamp"].to_numpy()
        labels = group["is_anomaly"].to_numpy() if has_labels else None
        n      = len(group)
        if n < window_size:
            continue
        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            X_list.append(values[start:end])
            end_list.append(ts[end - 1])
            dev_list.append(device_id)
            y_list.append(int(labels[start:end].any()) if has_labels else 0)

    if not X_list:
        raise ValueError(f"No windows produced — every device has < window_size={window_size} rows.")

    return (
        np.stack(X_list),
        np.array(y_list, dtype=np.int64),
        np.array(end_list),
        np.array(dev_list),
        feature_cols,
    )


class Standardizer:
    """
    Per-feature z-score scaler. Always fit on TRAIN (normal) data only —
    fitting on test data leaks failure statistics into the normalization
    and quietly inflates detection scores.
    """
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_  = flat.std(axis=0)
        self.std_  = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Standardizer.fit() must be called before transform()")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        obj = cls()
        obj.mean_ = np.array(d["mean"], dtype=np.float64)
        obj.std_  = np.array(d["std"],  dtype=np.float64)
        return obj
