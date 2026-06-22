"""
EdgeVigil — Public Benchmark Loaders
Phase 1: Data Foundation

Loaders for the two real-world benchmarks used to sanity-check the anomaly
core before trusting results on the synthetic fleet simulator:

    SMD  (Server Machine Dataset)   — multivariate, labeled, server-only
    NAB  (Numenta Anomaly Benchmark) — univariate/few-variate, labeled, mixed domains

Neither dataset ships with this repo — license terms require downloading
from source. Place the raw files as described below, then call these
loaders; they parse into the same long-format schema TelemetrySimulator
produces, so the same downstream windowing/encoder/eval code runs on both
real and synthetic data unchanged.

Download:
    SMD: https://github.com/NetManAIOps/OmniAnomaly
         -> place under data/raw/smd/{train,test,test_label}/<machine-id>.txt
    NAB: https://github.com/numenta/NAB
         -> place under data/raw/nab/{labels/combined_labels.json, data/**/*.csv}
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SMD — Server Machine Dataset
# ---------------------------------------------------------------------------
# Per the OmniAnomaly repo: train/test files are comma-separated floats, one
# row per timestep, 38 columns, no header. test_label files are single-column
# 0/1 labels aligned row-for-row with the corresponding test file.

def list_smd_machines(raw_dir: str) -> List[str]:
    """Discover available machine IDs from the train/ directory."""
    train_dir = Path(raw_dir) / "train"
    if not train_dir.exists():
        return []
    return sorted(p.stem for p in train_dir.glob("*.txt"))


def load_smd(raw_dir: str, machine_id: str) -> dict:
    """
    Load one SMD machine (e.g. 'machine-1-1') as train/test arrays + test labels.

    Returns:
        {"train": np.ndarray [T_train, 38],
         "test": np.ndarray [T_test, 38],
         "test_labels": np.ndarray [T_test] (0/1)}
    """
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "train" / f"{machine_id}.txt"
    test_path = raw_dir / "test" / f"{machine_id}.txt"
    label_path = raw_dir / "test_label" / f"{machine_id}.txt"

    for p in (train_path, test_path, label_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected SMD file not found: {p}\n"
                f"Download SMD from https://github.com/NetManAIOps/OmniAnomaly "
                f"and place files under {raw_dir}/{{train,test,test_label}}/"
            )

    return {
        "train": np.loadtxt(train_path, delimiter=","),
        "test": np.loadtxt(test_path, delimiter=","),
        "test_labels": np.loadtxt(label_path, delimiter=",").astype(int),
    }


def smd_to_long_format(arr: np.ndarray, machine_id: str,
                        labels: Optional[np.ndarray] = None,
                        step_minutes: float = 1.0) -> pd.DataFrame:
    """Convert an SMD array into the same long-format schema as simulator output."""
    n_steps, n_metrics = arr.shape
    timestamps = pd.date_range("2026-01-01", periods=n_steps, freq=f"{step_minutes}min")
    data = {"timestamp": timestamps, "device_id": machine_id, "device_type": "server"}
    for i in range(n_metrics):
        data[f"metric_{i}"] = arr[:, i]
    df = pd.DataFrame(data)
    if labels is not None:
        df["is_anomaly"] = labels.astype(bool)
    return df


# ---------------------------------------------------------------------------
# NAB — Numenta Anomaly Benchmark
# ---------------------------------------------------------------------------
# Format: data/<category>/<filename>.csv with columns [timestamp, value].
# Labels live in one combined_labels.json mapping
# "<category>/<filename>.csv" -> [list of anomaly window timestamps].

def load_nab_labels(raw_dir: str) -> dict:
    """Load the combined NAB label file: filename -> list of anomaly timestamps."""
    label_path = Path(raw_dir) / "labels" / "combined_labels.json"
    if not label_path.exists():
        raise FileNotFoundError(
            f"NAB labels not found at {label_path}\n"
            f"Download NAB from https://github.com/numenta/NAB and place "
            f"labels/combined_labels.json under {raw_dir}/"
        )
    with open(label_path) as f:
        return json.load(f)


def list_nab_series(raw_dir: str) -> List[str]:
    """Discover available NAB series as relative paths, keyed as in combined_labels.json."""
    return sorted(load_nab_labels(raw_dir).keys())


def load_nab_series(raw_dir: str, relative_csv_path: str) -> pd.DataFrame:
    """
    Load a single NAB series and attach point-anomaly labels.

    relative_csv_path example: 'realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv'
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "data" / relative_csv_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected NAB file not found: {csv_path}\n"
            f"Download NAB from https://github.com/numenta/NAB and place "
            f"the data/ directory under {raw_dir}/"
        )

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    anomaly_timestamps = set(pd.to_datetime(load_nab_labels(raw_dir).get(relative_csv_path, [])))
    df["is_anomaly"] = df["timestamp"].isin(anomaly_timestamps)
    df["device_id"] = relative_csv_path
    df["device_type"] = "nab_series"
    return df


if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/smd"
    machines = list_smd_machines(raw_dir)
    if machines:
        print(f"Found {len(machines)} SMD machines under {raw_dir}: {machines[:5]}")
    else:
        print(f"No SMD data found under {raw_dir}. See module docstring for download instructions.")
