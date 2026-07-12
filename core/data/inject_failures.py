"""
EdgeVigil — Failure Injection Framework  (Phase 1)
Injects labeled failure patterns into simulator output.

Patterns:
  gradual_drift : linear ramp to displaced level, then holds
  sudden_spike  : immediate full displacement for the window duration
  slow_leak     : ramps and never reverts — anomalous from onset to end of series
"""

from dataclasses import dataclass
from typing import List, Literal, Optional
import numpy as np
import pandas as pd

FailureKind = Literal["gradual_drift", "sudden_spike", "slow_leak"]
ALL_METRICS = ["cpu", "memory", "disk_io", "network_latency", "temperature"]
ALL_KINDS   = ["gradual_drift", "sudden_spike", "slow_leak"]


@dataclass
class FailureInjection:
    device_id: str
    metric: str
    kind: FailureKind
    onset: pd.Timestamp
    duration_minutes: float
    magnitude: float   # multiples of the metric's baseline std


class FailureInjector:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def inject(self, df: pd.DataFrame, injections: List[FailureInjection]) -> pd.DataFrame:
        df = df.copy()
        if "is_anomaly"    not in df.columns: df["is_anomaly"]    = False
        if "failure_onset" not in df.columns: df["failure_onset"] = pd.NaT
        for spec in injections:
            df = self._apply_one(df, spec)
        return df

    def _apply_one(self, df, spec):
        device_df = df.loc[df["device_id"] == spec.device_id].sort_values("timestamp")
        if device_df.empty:
            raise ValueError(f"No rows for device_id '{spec.device_id}'")
        step = device_df["timestamp"].iloc[1] - device_df["timestamp"].iloc[0]
        baseline_std = device_df[spec.metric].std() or 1.0

        onset_idx = device_df.index[device_df["timestamp"] >= spec.onset]
        if len(onset_idx) == 0:
            raise ValueError(f"Onset {spec.onset} is after end of device '{spec.device_id}' telemetry")
        start = onset_idx[0]

        n_window = max(1, int(spec.duration_minutes / (step.total_seconds() / 60)))
        device_tail = device_df.loc[start:]
        affected = device_tail.index[:n_window]
        delta = spec.magnitude * baseline_std

        if spec.kind == "sudden_spike":
            df.loc[affected, spec.metric] += delta
            anomalous_rows = affected
        elif spec.kind == "gradual_drift":
            df.loc[affected, spec.metric] += np.linspace(0, delta, len(affected))
            anomalous_rows = affected
        elif spec.kind == "slow_leak":
            df.loc[affected, spec.metric] += np.linspace(0, delta, len(affected))
            remainder = device_tail.index[len(affected):]
            if len(remainder): df.loc[remainder, spec.metric] += delta
            anomalous_rows = device_tail.index
        else:
            raise ValueError(f"Unknown failure kind: {spec.kind}")

        df.loc[anomalous_rows, "is_anomaly"] = True
        if pd.isna(df.loc[start, "failure_onset"]):
            df.loc[start, "failure_onset"] = spec.onset
        return df

    def random_injection(self, df, device_id, metric=None, kind=None,
                          min_onset_frac=0.3, max_onset_frac=0.8, magnitude_range=(2.5, 6.0)):
        device_df = df.loc[df["device_id"] == device_id].sort_values("timestamp")
        n = len(device_df)
        metric = metric or self.rng.choice(ALL_METRICS)
        kind   = kind   or self.rng.choice(ALL_KINDS)
        onset  = device_df["timestamp"].iloc[int(self.rng.integers(int(n*min_onset_frac), int(n*max_onset_frac)))]
        return FailureInjection(
            device_id=device_id, metric=str(metric), kind=str(kind), onset=onset,
            duration_minutes=float(self.rng.choice([15,30,60,120])),
            magnitude=float(self.rng.uniform(*magnitude_range)),
        )

    def random_injection_batch(self, df, n_injections, device_ids=None):
        pool   = device_ids or df["device_id"].unique().tolist()
        chosen = self.rng.choice(pool, size=min(n_injections, len(pool)), replace=False)
        return [self.random_injection(df, str(d)) for d in chosen]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from simulate import TelemetrySimulator

    sim = TelemetrySimulator(seed=1)
    df  = sim.simulate_fleet({"server": 2, "workstation": 2, "iot_sensor": 2}, duration_hours=24)
    injector = FailureInjector(seed=1)
    spec = FailureInjection("server-000", "cpu", "gradual_drift",
                             df["timestamp"].iloc[100], duration_minutes=60, magnitude=4.0)
    labeled = injector.inject(df, [spec])
    print(labeled.loc[labeled["is_anomaly"], ["timestamp","device_id","cpu","is_anomaly"]].head(10))
    print(f"\nTotal anomalous rows: {int(labeled['is_anomaly'].sum())}")
    for s in injector.random_injection_batch(df, n_injections=3):
        print(s)
