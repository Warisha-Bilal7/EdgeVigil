"""
EdgeVigil — Phase 2: Baseline anomaly model
Per-device-type LSTM-autoencoder, no domain adaptation. Trains on
failure-free ("normal") simulated telemetry only — this is an unsupervised
reconstruction model, so it never sees a labeled anomaly during training.
This run establishes the F1/FPR baseline that Phase 3's domain-adversarial
generalization-gap result needs to beat on a held-out device type.

Usage:
    python core/train.py --device-type server
    python core/train.py --device-type all              # trains server/workstation/iot_sensor
    python core/train.py --device-type server --epochs 50 --window-size 12
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from simulate import TelemetrySimulator          # noqa: E402
from windowing import make_windows, Standardizer, METRICS  # noqa: E402
from lstm_autoencoder import LSTMAutoencoder      # noqa: E402

DEFAULT_WINDOW_SIZE = 10  # timesteps per window, not minutes — at step_minutes=5 this is a 50-min window
DEVICE_TYPES = ["server", "workstation", "iot_sensor"]


def build_normal_dataset(device_type: str, n_devices: int = 10, duration_hours: float = 24 * 14,
                          step_minutes: float = 5.0, seed: int = 0):
    """Generate clean (failure-free) telemetry — Phase 2 trains on normal behavior only."""
    sim = TelemetrySimulator(seed=seed)
    return sim.simulate_fleet({device_type: n_devices}, duration_hours=duration_hours, step_minutes=step_minutes)


def train_one(device_type: str, window_size: int = DEFAULT_WINDOW_SIZE, epochs: int = 30,
              batch_size: int = 64, lr: float = 1e-3, hidden_size: int = 32, latent_dim: int = 16,
              seed: int = 0, out_dir: str = "artifacts", threshold_percentile: float = 99.0):
    torch.manual_seed(seed)

    df = build_normal_dataset(device_type, seed=seed)
    X, _, _, _, _ = make_windows(df, window_size=window_size, step=1)
    if len(X) < 20:
        raise RuntimeError(f"Only {len(X)} windows generated for '{device_type}' — increase duration_hours")

    scaler = Standardizer().fit(X)
    X_norm = scaler.transform(X).astype(np.float32)

    n_val = max(1, int(0.1 * len(X_norm)))
    X_train, X_val = X_norm[:-n_val], X_norm[-n_val:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=batch_size, shuffle=True)
    val_tensor = torch.from_numpy(X_val)

    model = LSTMAutoencoder(n_features=len(METRICS), window_size=window_size,
                             hidden_size=hidden_size, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = ((recon - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        train_loss = total_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = model.reconstruction_error(val_tensor).mean().item()
        best_val = min(best_val, val_loss)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[{device_type}] epoch {epoch:3d}/{epochs}  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")

    # Threshold from the held-out NORMAL validation split — never from test
    # data, since test data contains the injected failures eval.py scores
    # against. Anything else is leakage, even if it's just a percentile cut.
    model.eval()
    with torch.no_grad():
        val_errors = model.reconstruction_error(val_tensor).numpy()
    threshold = float(np.percentile(val_errors, threshold_percentile))

    # Per-feature thresholds for the Nagios-style static baseline, derived
    # from the same normal training distribution (raw, unwindowed) — fair
    # comparison point: both detectors only ever see normal behavior to set
    # their thresholds, neither peeks at the failures it's evaluated on.
    static_feature_thresholds = {
        m: float(np.percentile(df[m].to_numpy(), threshold_percentile)) for m in METRICS
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / f"lstm_ae_{device_type}.pt"
    scaler_path = out_path / f"scaler_{device_type}.json"
    thresholds_path = out_path / f"thresholds_{device_type}.json"

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": len(METRICS),
        "window_size": window_size,
        "hidden_size": hidden_size,
        "latent_dim": latent_dim,
    }, model_path)
    scaler_path.write_text(json.dumps(scaler.to_dict()))
    thresholds_path.write_text(json.dumps({
        "reconstruction_threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "val_errors": val_errors.tolist(),  # kept so eval.py can re-derive a different percentile without retraining
        "static_feature_thresholds": static_feature_thresholds,
    }))

    print(f"[{device_type}] saved model -> {model_path}, scaler -> {scaler_path}, "
          f"thresholds -> {thresholds_path}, best val_mse={best_val:.5f}")
    return model_path, scaler_path, thresholds_path, best_val


def main():
    parser = argparse.ArgumentParser(description="Phase 2: per-device-type LSTM-autoencoder baseline")
    parser.add_argument("--device-type", choices=DEVICE_TYPES + ["all"], default="all")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    args = parser.parse_args()

    targets = DEVICE_TYPES if args.device_type == "all" else [args.device_type]
    for dt in targets:
        train_one(dt, window_size=args.window_size, epochs=args.epochs, out_dir=args.out_dir,
                  threshold_percentile=args.threshold_percentile)


if __name__ == "__main__":
    main()
