"""
EdgeVigil — Telemetry Simulator
Phase 1: Data Foundation

Generates synthetic multivariate telemetry for heterogeneous device fleets
(servers, workstations, IoT sensors). Each device type has a distinct
"normal" baseline distribution — that distinctness is exactly the problem
the domain-adversarial encoder (Phase 3) has to learn to ignore.

Output: long-format pandas DataFrame with columns:
    timestamp, device_id, device_type, cpu, memory, disk_io, network_latency, temperature
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MetricProfile:
    """Generative process for one telemetry metric's normal behavior."""
    mean: float
    std: float
    ar_coef: float = 0.6                # AR(1) autocorrelation — telemetry isn't i.i.d. noise
    diurnal_amplitude: float = 0.0      # 0 = no daily cycle (e.g. server-room temp)
    diurnal_phase_hours: float = 14.0   # hour of day (0-24) where the cycle peaks
    min_value: float = 0.0
    max_value: float = 100.0


# Per-device-type baselines. Deliberately distinct distributions per type —
# pooling these without correction is the failure mode Phase 3's GRL exists to fix.
DEVICE_PROFILES = {
    "server": {
        "cpu": MetricProfile(mean=55, std=6, ar_coef=0.7, diurnal_amplitude=8, diurnal_phase_hours=14),
        "memory": MetricProfile(mean=65, std=4, ar_coef=0.85, diurnal_amplitude=3, diurnal_phase_hours=14),
        "disk_io": MetricProfile(mean=30, std=10, ar_coef=0.5, diurnal_amplitude=5, diurnal_phase_hours=2),
        "network_latency": MetricProfile(mean=4, std=0.8, ar_coef=0.6, diurnal_amplitude=1, diurnal_phase_hours=14, min_value=0.5, max_value=50),
        "temperature": MetricProfile(mean=42, std=1.5, ar_coef=0.9, diurnal_amplitude=1, diurnal_phase_hours=15, min_value=20, max_value=80),
    },
    "workstation": {
        "cpu": MetricProfile(mean=25, std=15, ar_coef=0.5, diurnal_amplitude=20, diurnal_phase_hours=11),
        "memory": MetricProfile(mean=45, std=12, ar_coef=0.7, diurnal_amplitude=15, diurnal_phase_hours=11),
        "disk_io": MetricProfile(mean=10, std=8, ar_coef=0.4, diurnal_amplitude=8, diurnal_phase_hours=11),
        "network_latency": MetricProfile(mean=12, std=4, ar_coef=0.4, diurnal_amplitude=5, diurnal_phase_hours=11, min_value=1, max_value=200),
        "temperature": MetricProfile(mean=38, std=3, ar_coef=0.8, diurnal_amplitude=4, diurnal_phase_hours=13, min_value=15, max_value=85),
    },
    "iot_sensor": {
        "cpu": MetricProfile(mean=8, std=3, ar_coef=0.6, diurnal_amplitude=2, diurnal_phase_hours=12),
        "memory": MetricProfile(mean=20, std=3, ar_coef=0.6, diurnal_amplitude=1, diurnal_phase_hours=12),
        "disk_io": MetricProfile(mean=2, std=1.5, ar_coef=0.3, diurnal_amplitude=0.5, diurnal_phase_hours=12, max_value=50),
        # Wireless link -> noisier, higher-variance latency than wired device types.
        "network_latency": MetricProfile(mean=35, std=15, ar_coef=0.3, diurnal_amplitude=10, diurnal_phase_hours=12, min_value=2, max_value=500),
        # Ambient-exposed -> wider swing and lower floor than rack-mounted hardware.
        "temperature": MetricProfile(mean=30, std=5, ar_coef=0.5, diurnal_amplitude=6, diurnal_phase_hours=15, min_value=-10, max_value=70),
    },
}

METRICS = ["cpu", "memory", "disk_io", "network_latency", "temperature"]


def _generate_series(profile: MetricProfile, n_steps: int, step_minutes: float,
                      rng: np.random.Generator, start_hour: float = 0.0) -> np.ndarray:
    """AR(1) process with a diurnal sinusoidal mean, clipped to physical bounds."""
    hours = (start_hour + np.arange(n_steps) * step_minutes / 60.0) % 24
    diurnal = profile.diurnal_amplitude * np.cos(2 * np.pi * (hours - profile.diurnal_phase_hours) / 24)

    series = np.empty(n_steps)
    series[0] = profile.mean + diurnal[0] + rng.normal(0, profile.std)
    # Scale innovation noise so the AR(1) process has stationary variance == profile.std**2.
    innovation_std = profile.std * np.sqrt(max(1 - profile.ar_coef ** 2, 1e-6))

    for t in range(1, n_steps):
        target = profile.mean + diurnal[t]
        prev_target = profile.mean + diurnal[t - 1]
        series[t] = target + profile.ar_coef * (series[t - 1] - prev_target) + rng.normal(0, innovation_std)

    return np.clip(series, profile.min_value, profile.max_value)


class TelemetrySimulator:
    """
    Generates multivariate telemetry for a fleet of simulated devices.

    Usage:
        sim = TelemetrySimulator(seed=42)
        df = sim.simulate_fleet(
            device_counts={"server": 5, "workstation": 10, "iot_sensor": 20},
            duration_hours=24 * 7,
            step_minutes=5,
        )
    """

    def __init__(self, seed: Optional[int] = None, profiles: Optional[dict] = None):
        self.rng = np.random.default_rng(seed)
        # Copy so held_out_device_type() mutations on one instance don't leak globally.
        self.profiles = {k: dict(v) for k, v in (profiles or DEVICE_PROFILES).items()}

    def simulate_device(self, device_id: str, device_type: str, duration_hours: float,
                         step_minutes: float = 5.0,
                         start_timestamp: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        if device_type not in self.profiles:
            raise ValueError(f"Unknown device_type '{device_type}'. Known types: {list(self.profiles.keys())}")

        n_steps = int(duration_hours * 60 / step_minutes)
        start_timestamp = start_timestamp or pd.Timestamp("2026-01-01 00:00:00")
        start_hour = start_timestamp.hour + start_timestamp.minute / 60.0

        timestamps = pd.date_range(start=start_timestamp, periods=n_steps, freq=f"{step_minutes}min")
        data = {"timestamp": timestamps, "device_id": device_id, "device_type": device_type}
        for metric in METRICS:
            profile = self.profiles[device_type][metric]
            data[metric] = _generate_series(profile, n_steps, step_minutes, self.rng, start_hour=start_hour)

        return pd.DataFrame(data)

    def simulate_fleet(self, device_counts: dict, duration_hours: float,
                        step_minutes: float = 5.0,
                        start_timestamp: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        frames = []
        for device_type, count in device_counts.items():
            for i in range(count):
                device_id = f"{device_type}-{i:03d}"
                frames.append(self.simulate_device(
                    device_id=device_id, device_type=device_type,
                    duration_hours=duration_hours, step_minutes=step_minutes,
                    start_timestamp=start_timestamp,
                ))
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    def held_out_device_type(self, name: str, base_on: str, perturb: dict) -> None:
        """
        Register a new device-type profile by perturbing an existing one.
        Use this to generate a device type the model never sees during training,
        for Phase 3's generalization-gap test (F1 with vs. without the GRL).

        Example:
            sim.held_out_device_type(
                "edge_gateway", base_on="iot_sensor",
                perturb={"cpu": {"mean": 1.4}, "network_latency": {"mean": 0.7}},
            )
        """
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
                min_value=profile.min_value,
                max_value=profile.max_value,
            )
        self.profiles[name] = new_profile


if __name__ == "__main__":
    sim = TelemetrySimulator(seed=42)
    df = sim.simulate_fleet(
        device_counts={"server": 3, "workstation": 3, "iot_sensor": 3},
        duration_hours=24,
        step_minutes=5,
    )
    print(df.groupby("device_type")[METRICS].mean().round(2))
    print(f"\nTotal rows: {len(df)}")
