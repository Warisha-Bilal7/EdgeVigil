"""
EdgeVigil — Public Benchmark Loaders  (Phase 1)

Loaders for SMD (Server Machine Dataset) and NAB (Numenta Anomaly Benchmark).
Neither ships with this repo — see module docstring for download links.
Both parse into the same long-format schema as TelemetrySimulator, so the
same windowing/encoder/eval code runs on real and synthetic data unchanged.

SMD: https://github.com/NetManAIOps/OmniAnomaly
     -> data/raw/smd/{train,test,test_label}/<machine-id>.txt
NAB: https://github.com/numenta/NAB
     -> data/raw/nab/{labels/combined_labels.json, data/**/*.csv}
"""

import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd


def list_smd_machines(raw_dir: str) -> List[str]:
    train_dir = Path(raw_dir) / "train"
    return sorted(p.stem for p in train_dir.glob("*.txt")) if train_dir.exists() else []


def load_smd(raw_dir: str, machine_id: str) -> dict:
    raw_dir = Path(raw_dir)
    paths = {k: raw_dir / k / f"{machine_id}.txt"
             for k in ("train", "test", "test_label")}
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(
                f"Expected SMD file not found: {p}\n"
                f"Download from https://github.com/NetManAIOps/OmniAnomaly"
            )
    return {
        "train":       np.loadtxt(paths["train"],      delimiter=","),
        "test":        np.loadtxt(paths["test"],       delimiter=","),
        "test_labels": np.loadtxt(paths["test_label"], delimiter=",").astype(int),
    }


def smd_to_long_format(arr, machine_id, labels=None, step_minutes=1.0):
    n_steps, n_metrics = arr.shape
    timestamps = pd.date_range("2026-01-01", periods=n_steps, freq=f"{step_minutes}min")
    data = {"timestamp": timestamps, "device_id": machine_id, "device_type": "server"}
    for i in range(n_metrics):
        data[f"metric_{i}"] = arr[:, i]
    df = pd.DataFrame(data)
    if labels is not None:
        df["is_anomaly"] = labels.astype(bool)
    return df


def load_nab_labels(raw_dir: str) -> dict:
    label_path = Path(raw_dir) / "labels" / "combined_labels.json"
    if not label_path.exists():
        raise FileNotFoundError(
            f"NAB labels not found at {label_path}\n"
            f"Download from https://github.com/numenta/NAB"
        )
    return json.loads(label_path.read_text())


def list_nab_series(raw_dir: str) -> List[str]:
    return sorted(load_nab_labels(raw_dir).keys())


def load_nab_series(raw_dir: str, relative_csv_path: str) -> pd.DataFrame:
    csv_path = Path(raw_dir) / "data" / relative_csv_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected NAB file not found: {csv_path}\n"
            f"Download from https://github.com/numenta/NAB"
        )
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    anomaly_ts = set(pd.to_datetime(load_nab_labels(raw_dir).get(relative_csv_path, [])))
    df["is_anomaly"]  = df["timestamp"].isin(anomaly_ts)
    df["device_id"]   = relative_csv_path
    df["device_type"] = "nab_series"
    return df


if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/smd"
    machines = list_smd_machines(raw_dir)
    if machines:
        print(f"Found {len(machines)} SMD machines: {machines[:5]}")
    else:
        print(f"No SMD data under {raw_dir}. See module docstring for download instructions.")
