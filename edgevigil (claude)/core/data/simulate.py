"""
EdgeVigil — Telemetry Simulator  (Phase 1)
Generates synthetic multivariate telemetry for heterogeneous device fleets.
Each device type has a distinct "normal" distribution — that distinctness is
exactly the problem the domain-adversarial encoder (Phase 3) has to learn to ignore.

Output columns: timestamp, device_id, device_type, cpu, memory, disk_io,
                network_latency, temperature
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class MetricProfile:
    mean: float
    std: float
    ar_coef: float = 0.6
    diurnal_amplitude: float = 0.0
    diurnal_phase_hours: float = 14.0
    min_value: float = 0.0
    max_value: float = 100.0

DEVICE_PROFILES = {
    "server": {
        "cpu":             MetricProfile(55,  6,  0.70, 8,  14, 0,   100),
        "memory":          MetricProfile(65,  4,  0.85, 3,  14, 0,   100),
        "disk_io":         MetricProfile(30, 10,  0.50, 5,   2, 0,   100),
        "network_latency": MetricProfile( 4, .8,  0.60, 1,  14, 0.5,  50),
        "temperature":     MetricProfile(42, 1.5, 0.90, 1,  15, 20,   80),
    },
    "workstation": {
        "cpu":             MetricProfile(25, 15,  0.50, 20, 11, 0,   100),
        "memory":          MetricProfile(45, 12,  0.70, 15, 11, 0,   100),
        "disk_io":         MetricProfile(10,  8,  0.40,  8, 11, 0,   100),
        "network_latency": MetricProfile(12,  4,  0.40,  5, 11, 1,   200),
        "temperature":     MetricProfile(38,  3,  0.80,  4, 13, 15,   85),
    },
    "iot_sensor": {
        "cpu":             MetricProfile( 8,  3,  0.60,  2, 12, 0,   100),
        "memory":          MetricProfile(20,  3,  0.60,  1, 12, 0,   100),
        "disk_io":         MetricProfile( 2, 1.5, 0.30, .5, 12, 0,    50),
        "network_latency": MetricProfile(35, 15,  0.30, 10, 12, 2,   500),
        "temperature":     MetricProfile(30,  5,  0.50,  6, 15,-10,   70),
    },
}

METRICS = ["cpu", "memory", "disk_io", "network_latency", "temperature"]


def _generate_series(profile: MetricProfile, n_steps: int, step_minutes: float,
                      rng: np.random.Generator, start_hour: float = 0.0) -> np.ndarray:
    hours = (start_hour + np.arange(n_steps) * step_minutes / 60.0) % 24
    diurnal = profile.diurnal_amplitude * np.cos(2 * np.pi * (hours - profile.diurnal_phase_hours) / 24)
    series = np.empty(n_steps)
    series[0] = profile.mean + diurnal[0] + rng.normal(0, profile.std)
    innovation_std = profile.std * np.sqrt(max(1 - profile.ar_coef ** 2, 1e-6))
    for t in range(1, n_steps):
        target = profile.mean + diurnal[t]
        prev_target = profile.mean + diurnal[t - 1]
        series[t] = target + profile.ar_coef * (series[t-1] - prev_target) + rng.normal(0, innovation_std)
    return np.clip(series, profile.min_value, profile.max_value)


class TelemetrySimulator:
    def __init__(self, seed=None, profiles=None):
        self.rng = np.random.default_rng(seed)
        self.profiles = {k: dict(v) for k, v in (profiles or DEVICE_PROFILES).items()}

    def simulate_device(self, device_id, device_type, duration_hours,
                         step_minutes=5.0, start_timestamp=None):
        if device_type not in self.profiles:
            raise ValueError(f"Unknown device_type '{device_type}'")
        n_steps = int(duration_hours * 60 / step_minutes)
        start_timestamp = start_timestamp or pd.Timestamp("2026-01-01")
        start_hour = start_timestamp.hour + start_timestamp.minute / 60.0
        timestamps = pd.date_range(start=start_timestamp, periods=n_steps, freq=f"{step_minutes}min")
        data = {"timestamp": timestamps, "device_id": device_id, "device_type": device_type}
        for m in METRICS:
            data[m] = _generate_series(self.profiles[device_type][m], n_steps,
                                        step_minutes, self.rng, start_hour)
        return pd.DataFrame(data)

    def simulate_fleet(self, device_counts, duration_hours, step_minutes=5.0, start_timestamp=None):
        frames = []
        for dtype, count in device_counts.items():
            for i in range(count):
                frames.append(self.simulate_device(f"{dtype}-{i:03d}", dtype,
                                                    duration_hours, step_minutes, start_timestamp))
        return pd.concat(frames, ignore_index=True).sort_values(
            ["device_id", "timestamp"]).reset_index(drop=True)

    def held_out_device_type(self, name, base_on, perturb):
        if base_on not in self.profiles:
            raise ValueError(f"Unknown base device_type '{base_on}'")
        new_profile = {}
        for metric, profile in self.profiles[base_on].items():
            f = perturb.get(metric, {})
            new_profile[metric] = MetricProfile(
                mean=profile.mean * f.get("mean", 1.0),
                std=profile.std * f.get("std", 1.0),
                ar_coef=profile.ar_coef,
                diurnal_amplitude=profile.diurnal_amplitude * f.get("diurnal_amplitude", 1.0),
                diurnal_phase_hours=profile.diurnal_phase_hours,
                min_value=profile.min_value, max_value=profile.max_value,
            )
        self.profiles[name] = new_profile


if __name__ == "__main__":
    sim = TelemetrySimulator(seed=42)
    df = sim.simulate_fleet({"server": 3, "workstation": 3, "iot_sensor": 3}, duration_hours=24)
    print(df.groupby("device_type")[METRICS].mean().round(2))
    print(f"\nTotal rows: {len(df)}")
