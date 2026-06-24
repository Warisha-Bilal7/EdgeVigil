"""
EdgeVigil — Failure Injection Framework
Phase 1: Data Foundation

Injects labeled failure patterns into simulator output so the anomaly core
(Phase 2/3) has ground truth to train and evaluate against. Three patterns,
matching the failure modes named in the roadmap:

    gradual_drift : metric ramps linearly away from baseline over a window,
                    then holds at the displaced level
    sudden_spike  : abrupt, brief deviation, full magnitude immediately
    slow_leak     : monotonic one-directional creep that does not revert —
                    once started, stays anomalous for the rest of the series
                    (models e.g. a memory leak, not a transient excursion)

Each injection writes:
    - modified telemetry values
    - a boolean `is_anomaly` column
    - a `failure_onset` timestamp on the first affected row, used downstream
      to compute detection lead time (onset vs. the model's first alert time)
"""

from dataclasses import dataclass
from typing import Literal, List, Optional

import numpy as np
import pandas as pd

FailureKind = Literal["gradual_drift", "sudden_spike", "slow_leak"]
ALL_METRICS = ["cpu", "memory", "disk_io", "network_latency", "temperature"]
ALL_KINDS = ["gradual_drift", "sudden_spike", "slow_leak"]


@dataclass
class FailureInjection:
    device_id: str
    metric: str
    kind: FailureKind
    onset: pd.Timestamp
    duration_minutes: float
    magnitude: float  # expressed in units of the metric's own baseline std


class FailureInjector:
    """
    Applies FailureInjection specs onto a telemetry DataFrame produced by
    TelemetrySimulator. Operates on a copy; never mutates the input in place.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def inject(self, df: pd.DataFrame, injections: List[FailureInjection]) -> pd.DataFrame:
        df = df.copy()
        if "is_anomaly" not in df.columns:
            df["is_anomaly"] = False
        if "failure_onset" not in df.columns:
            df["failure_onset"] = pd.NaT

        for spec in injections:
            df = self._apply_one(df, spec)
        return df

    def _apply_one(self, df: pd.DataFrame, spec: FailureInjection) -> pd.DataFrame:
        device_df = df.loc[df["device_id"] == spec.device_id].sort_values("timestamp")
        if device_df.empty:
            raise ValueError(f"No rows found for device_id '{spec.device_id}'")
        if len(device_df) < 2:
            raise ValueError(f"Device '{spec.device_id}' has fewer than 2 rows; can't infer step size")

        step = device_df["timestamp"].iloc[1] - device_df["timestamp"].iloc[0]
        baseline_std = device_df[spec.metric].std()
        if not baseline_std or np.isnan(baseline_std):
            baseline_std = 1.0

        onset_candidates = device_df.index[device_df["timestamp"] >= spec.onset]
        if len(onset_candidates) == 0:
            raise ValueError(f"Onset {spec.onset} is after the end of device '{spec.device_id}' telemetry")
        start = onset_candidates[0]

        n_window = max(1, int(spec.duration_minutes / (step.total_seconds() / 60)))
        device_tail = device_df.loc[start:]
        affected = device_tail.index[:n_window]
        delta = spec.magnitude * baseline_std

        if spec.kind == "sudden_spike":
            df.loc[affected, spec.metric] += delta
            anomalous_rows = affected

        elif spec.kind == "gradual_drift":
            ramp = np.linspace(0, delta, len(affected))
            df.loc[affected, spec.metric] += ramp
            anomalous_rows = affected

        elif spec.kind == "slow_leak":
            ramp = np.linspace(0, delta, len(affected))
            df.loc[affected, spec.metric] += ramp
            remainder = device_tail.index[len(affected):]
            if len(remainder) > 0:
                df.loc[remainder, spec.metric] += delta  # holds the leaked value, never reverts
            anomalous_rows = device_tail.index  # everything from onset to end of series

        else:
            raise ValueError(f"Unknown failure kind: {spec.kind}")

        df.loc[anomalous_rows, "is_anomaly"] = True
        if pd.isna(df.loc[start, "failure_onset"]):
            df.loc[start, "failure_onset"] = spec.onset

        return df

    def random_injection(self, df: pd.DataFrame, device_id: str,
                          metric: Optional[str] = None, kind: Optional[FailureKind] = None,
                          min_onset_frac: float = 0.3, max_onset_frac: float = 0.8,
                          magnitude_range: tuple = (2.5, 6.0)) -> FailureInjection:
        """Convenience generator for building synthetic eval sets at scale."""
        device_df = df.loc[df["device_id"] == device_id].sort_values("timestamp")
        n = len(device_df)
        metric = metric or self.rng.choice(ALL_METRICS)
        kind = kind or self.rng.choice(ALL_KINDS)
        onset_idx = int(self.rng.integers(int(n * min_onset_frac), int(n * max_onset_frac)))
        onset = device_df["timestamp"].iloc[onset_idx]
        magnitude = float(self.rng.uniform(*magnitude_range))
        duration = float(self.rng.choice([15, 30, 60, 120]))
        return FailureInjection(
            device_id=device_id, metric=str(metric), kind=str(kind),
            onset=onset, duration_minutes=duration, magnitude=magnitude,
        )

    def random_injection_batch(self, df: pd.DataFrame, n_injections: int,
                                device_ids: Optional[List[str]] = None) -> List[FailureInjection]:
        """One random injection per randomly chosen device, no device repeated."""
        pool = device_ids or df["device_id"].unique().tolist()
        chosen = self.rng.choice(pool, size=min(n_injections, len(pool)), replace=False)
        return [self.random_injection(df, device_id=str(d)) for d in chosen]


if __name__ == "__main__":
    try:
        from simulate import TelemetrySimulator  # running directly from core/data/
    except ImportError:
        from core.data.simulate import TelemetrySimulator  # running as `python -m core.data.inject_failures`

    sim = TelemetrySimulator(seed=1)
    df = sim.simulate_fleet({"server": 2, "workstation": 2, "iot_sensor": 2}, duration_hours=24)

    injector = FailureInjector(seed=1)
    spec = FailureInjection(
        device_id="server-000", metric="cpu", kind="gradual_drift",
        onset=df["timestamp"].iloc[100], duration_minutes=60, magnitude=4.0,
    )
    labeled = injector.inject(df, [spec])
    print(labeled.loc[labeled["is_anomaly"], ["timestamp", "device_id", "cpu", "is_anomaly"]].head(10))
    print(f"\nTotal anomalous rows: {int(labeled['is_anomaly'].sum())}")

    batch = injector.random_injection_batch(df, n_injections=3)
    for spec in batch:
        print(spec)
