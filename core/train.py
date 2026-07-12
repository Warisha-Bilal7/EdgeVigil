"""
EdgeVigil — Training entrypoint  (Phase 2)

Usage:
    python core/train.py --device-type server
    python core/train.py --device-type all
    python core/train.py --device-type server --score-mode max   (default)
    python core/train.py --device-type server --score-mode mean  (baseline comparison)

Trains a per-device-type LSTM-autoencoder on failure-free synthetic telemetry,
then saves model weights, scaler, and threshold derived from a held-out normal
validation split (never from test data — that has the injected failures eval.py
scores against; computing the threshold from the test set is leakage).

Phase 3 will add a shared encoder + GRL + domain classifier on top of this.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise SystemExit("torch is required for training. pip install torch")

from simulate import TelemetrySimulator, METRICS, DEVICE_PROFILES
from inject_failures import FailureInjector
from windowing import make_windows, Standardizer
from models.lstm_autoencoder import LSTMAutoencoder

DEVICE_TYPES       = list(DEVICE_PROFILES.keys())
DEFAULT_WINDOW_SIZE = 12   # 12 × 5-min steps = 1-hour window


def build_normal_dataset(device_type: str, seed: int = 0) -> "pd.DataFrame":
    """14 days of failure-free telemetry for training."""
    import pandas as pd
    sim = TelemetrySimulator(seed=seed)
    return sim.simulate_fleet({device_type: 8}, duration_hours=14 * 24)


def train_one(device_type: str, window_size: int = DEFAULT_WINDOW_SIZE, epochs: int = 30,
              batch_size: int = 64, lr: float = 1e-3, hidden_size: int = 32,
              latent_dim: int = 16, seed: int = 0, out_dir: str = "artifacts",
              threshold_percentile: float = 99.0, score_mode: str = "max"):
    torch.manual_seed(seed)

    df = build_normal_dataset(device_type, seed=seed)
    X, _, _, _, _ = make_windows(df, window_size=window_size, step=1)
    if len(X) < 20:
        raise RuntimeError(f"Only {len(X)} windows — increase duration_hours")

    n_val  = max(1, int(len(X) * 0.15))
    X_train, X_val = X[:-n_val], X[-n_val:]

    scaler  = Standardizer().fit(X_train)
    Xt      = torch.from_numpy(scaler.transform(X_train).astype(np.float32))
    val_tensor = torch.from_numpy(scaler.transform(X_val).astype(np.float32))

    model     = LSTMAutoencoder(len(METRICS), window_size, hidden_size, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader    = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=True)

    best_val, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = ((model(batch) - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mse = model.reconstruction_error(Xt).mean().item()
            val_mse   = model.reconstruction_error(val_tensor).mean().item()
        if val_mse < best_val:
            best_val   = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[{device_type}] epoch {epoch:3d}/{epochs}  "
                  f"train_mse={train_mse:.5f}  val_mse={val_mse:.5f}")

    model.load_state_dict(best_state)

    # Threshold from held-out NORMAL validation split — never from test data.
    # Uses anomaly_score (max-pooled by default) rather than reconstruction_error,
    # so the same scoring function is used at train time and eval time.
    model.eval()
    with torch.no_grad():
        val_errors = model.anomaly_score(val_tensor, mode=score_mode).numpy()
    threshold = float(np.percentile(val_errors, threshold_percentile))

    # Static per-feature thresholds for the Nagios-style baseline comparison.
    # Derived from the same normal training data so neither detector peeks at
    # failures when setting its threshold.
    static_feature_thresholds = {
        m: float(np.percentile(df[m].to_numpy(), threshold_percentile)) for m in METRICS
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": best_state,
        "n_features": len(METRICS),
        "window_size": window_size,
        "hidden_size": hidden_size,
        "latent_dim":  latent_dim,
    }, out_path / f"lstm_ae_{device_type}.pt")

    (out_path / f"scaler_{device_type}.json").write_text(json.dumps(scaler.to_dict()))

    (out_path / f"thresholds_{device_type}.json").write_text(json.dumps({
        "reconstruction_threshold": threshold,
        "threshold_percentile":     threshold_percentile,
        "score_mode":               score_mode,
        "val_errors":               val_errors.tolist(),  # saved so --percentile can re-derive without retraining
        "static_feature_thresholds": static_feature_thresholds,
    }))

    print(f"[{device_type}] saved to {out_dir}/  best val_mse={best_val:.5f}  "
          f"threshold={threshold:.5f} (p{threshold_percentile}, score_mode={score_mode})")
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-type", default="server",
                         help=f"one of {DEVICE_TYPES} or 'all'")
    parser.add_argument("--window-size",  type=int,   default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--out-dir",                  default="artifacts")
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--score-mode", choices=["max", "mean"], default="max",
                         help="max pools the worst timestep per window (default, fixes "
                              "mean-pooling dilution of brief anomalies); mean averages "
                              "the whole window (training-loss-equivalent, for comparison).")
    args = parser.parse_args()
    targets = DEVICE_TYPES if args.device_type == "all" else [args.device_type]
    for dt in targets:
        train_one(dt, window_size=args.window_size, epochs=args.epochs, out_dir=args.out_dir,
                  threshold_percentile=args.threshold_percentile, score_mode=args.score_mode)


if __name__ == "__main__":
    main()
