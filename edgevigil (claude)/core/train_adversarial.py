"""
EdgeVigil — Domain-Adversarial Training  (Phase 3)

Usage:
    python core/train_adversarial.py
    python core/train_adversarial.py --epochs 50 --grl-lambda-max 0.5
    python core/train_adversarial.py --include-held-out   # trains on edge_gateway too
                                                           # (use only to confirm it helps;
                                                           #  kept OUT by default for the
                                                           #  generalization-gap eval)

What this does:
  1. Generates pooled telemetry across all 3 device types (server, workstation, iot_sensor)
  2. Trains a DomainAdversarialAutoencoder with:
       - reconstruction loss  : MSE between input and decoded output
       - domain loss          : cross-entropy on device-type label, with GRL reversal
       - total = recon_loss + alpha * domain_loss
  3. GRL lambda ramps from 0 -> grl_lambda_max over training, so the encoder
     first learns to reconstruct well before the adversarial pressure kicks in
     (the standard Ganin schedule — ramp too fast and reconstruction never converges)
  4. Saves the best checkpoint by val reconstruction MSE (not total loss, since
     domain loss improving means encoder is getting more confused about device type,
     which shouldn't be penalized at checkpoint time)

The generalization-gap eval (Phase 3's core result):
    Compare F1 on the held-out edge_gateway device type (never seen during training)
    between:
      - Phase 2 per-device-type LSTM-AE (no domain adaptation)
      - Phase 3 domain-adversarial model (trained on server/workstation/iot_sensor)
    A larger F1 gap = stronger evidence that GRL generalization is working.
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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from simulate import TelemetrySimulator, METRICS, DEVICE_PROFILES
from inject_failures import FailureInjector
from windowing import make_windows, Standardizer
if TORCH_AVAILABLE:
    from models.domain_adversarial import DomainAdversarialAutoencoder

TRAIN_DEVICE_TYPES = list(DEVICE_PROFILES.keys())   # server, workstation, iot_sensor
DOMAIN_TO_IDX      = {dt: i for i, dt in enumerate(TRAIN_DEVICE_TYPES)}
DEFAULT_WINDOW_SIZE = 12


def grl_lambda_schedule(epoch: int, total_epochs: int, lambda_max: float) -> float:
    """
    Ganin et al. schedule: ramp lambda from 0 to lambda_max over training.
    p = epoch / total_epochs in [0, 1].
    lambda = lambda_max * (2 / (1 + exp(-10 * p)) - 1)
    Starts near 0, ends near lambda_max, with the steepest rise in the middle.
    """
    p = epoch / total_epochs
    return lambda_max * (2 / (1 + np.exp(-10 * p)) - 1)


def build_pooled_dataset(device_types: list, seed: int = 0):
    """14 days of failure-free telemetry, pooled across all device types."""
    sim = TelemetrySimulator(seed=seed)
    counts = {dt: 8 for dt in device_types}
    df = sim.simulate_fleet(counts, duration_hours=14 * 24)
    return df


def build_held_out_device(seed: int = 0):
    """
    edge_gateway: a device type the model never sees during training.
    Derived from iot_sensor with a perturbed profile — higher CPU mean,
    more variable network latency, narrower temperature range.
    """
    sim = TelemetrySimulator(seed=seed)
    sim.held_out_device_type(
        "edge_gateway", base_on="iot_sensor",
        perturb={
            "cpu":             {"mean": 1.4, "std": 1.2},
            "network_latency": {"mean": 0.7, "std": 1.5},
            "temperature":     {"std": 0.7},
        },
    )
    return sim.simulate_fleet({"edge_gateway": 4}, duration_hours=3 * 24)


def train(epochs: int = 40, window_size: int = DEFAULT_WINDOW_SIZE,
           batch_size: int = 64, lr: float = 1e-3,
           hidden_size: int = 32, latent_dim: int = 16, domain_hidden: int = 16,
           alpha: float = 0.3, grl_lambda_max: float = 0.3,
           seed: int = 0, out_dir: str = "artifacts",
           threshold_percentile: float = 99.0, score_mode: str = "max",
           include_held_out: bool = False):
    torch.manual_seed(seed)

    device_types = TRAIN_DEVICE_TYPES + (["edge_gateway"] if include_held_out else [])
    df = build_pooled_dataset(device_types if not include_held_out else TRAIN_DEVICE_TYPES, seed=seed)
    if include_held_out:
        df_heldout = build_held_out_device(seed=seed)
        import pandas as pd
        df = pd.concat([df, df_heldout], ignore_index=True)

    # Build windows with device-type domain labels
    X_list, y_domain_list, dev_type_list = [], [], []
    for dt in (TRAIN_DEVICE_TYPES + (["edge_gateway"] if include_held_out else [])):
        sub = df[df["device_type"] == dt]
        if sub.empty:
            continue
        X_dt, _, _, _, _ = make_windows(sub, window_size=window_size, step=1)
        domain_idx = DOMAIN_TO_IDX.get(dt, len(TRAIN_DEVICE_TYPES))
        X_list.append(X_dt)
        y_domain_list.append(np.full(len(X_dt), domain_idx, dtype=np.int64))
        dev_type_list.append(dt)

    X_all      = np.concatenate(X_list)
    y_domain   = np.concatenate(y_domain_list)
    n_domains  = len(set(y_domain.tolist()))

    # Fit scaler on ALL training types so the same transform is used for all
    n_val = max(1, int(len(X_all) * 0.15))
    # Stratified-ish split: hold out the last 15% of each type's windows
    val_idxs = []
    offset = 0
    for X_dt in X_list:
        n = len(X_dt)
        nv = max(1, int(n * 0.15))
        val_idxs.extend(range(offset + n - nv, offset + n))
        offset += n
    val_idxs  = sorted(val_idxs)
    train_mask = np.ones(len(X_all), dtype=bool)
    train_mask[val_idxs] = False

    X_train, X_val           = X_all[train_mask], X_all[~train_mask]
    y_dom_train, y_dom_val   = y_domain[train_mask], y_domain[~train_mask]

    scaler = Standardizer().fit(X_train)
    Xt     = torch.from_numpy(scaler.transform(X_train).astype(np.float32))
    Xv     = torch.from_numpy(scaler.transform(X_val).astype(np.float32))
    yt     = torch.from_numpy(y_dom_train)
    yv     = torch.from_numpy(y_dom_val)

    model = DomainAdversarialAutoencoder(
        n_features=len(METRICS), window_size=window_size, n_domains=n_domains,
        hidden_size=hidden_size, latent_dim=latent_dim, domain_hidden=domain_hidden,
        grl_lambda=0.0,
    )
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    recon_loss  = nn.MSELoss()
    domain_loss = nn.CrossEntropyLoss()
    loader      = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    best_val_mse, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        lam = grl_lambda_schedule(epoch, epochs, grl_lambda_max)
        model.set_grl_lambda(lam)
        model.train()

        for (xb, yb) in loader:
            optimizer.zero_grad()
            recon, dom_logits = model.forward_with_domain(xb)
            loss = recon_loss(recon, xb) + alpha * domain_loss(dom_logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_recon_mse  = model.reconstruction_error(Xv).mean().item()
            train_recon    = model.reconstruction_error(Xt).mean().item()
            dom_acc_val    = (model.domain_classifier(
                                model.grl(model.encode(Xv))
                              ).argmax(dim=1) == yv).float().mean().item()

        if val_recon_mse < best_val_mse:
            best_val_mse = val_recon_mse
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}/{epochs}  "
                  f"train_mse={train_recon:.5f}  val_mse={val_recon_mse:.5f}  "
                  f"dom_acc={dom_acc_val:.3f}  grl_λ={lam:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_errors = model.anomaly_score(Xv, mode=score_mode).numpy()
    threshold = float(np.percentile(val_errors, threshold_percentile))

    # Per-device-type thresholds for static baseline (trained types only)
    df_train = build_pooled_dataset(TRAIN_DEVICE_TYPES, seed=seed)
    static_feature_thresholds = {
        m: float(np.percentile(df_train[m].to_numpy(), threshold_percentile))
        for m in METRICS
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "daa_model.pt"

    torch.save({
        "state_dict":  best_state,
        "n_features":  len(METRICS),
        "window_size": window_size,
        "n_domains":   n_domains,
        "hidden_size": hidden_size,
        "latent_dim":  latent_dim,
        "domain_hidden": domain_hidden,
        "domain_to_idx": DOMAIN_TO_IDX,
    }, model_path)

    (out_path / "scaler_adversarial.json").write_text(json.dumps(scaler.to_dict()))
    (out_path / "thresholds_adversarial.json").write_text(json.dumps({
        "reconstruction_threshold": threshold,
        "threshold_percentile":     threshold_percentile,
        "score_mode":               score_mode,
        "val_errors":               val_errors.tolist(),
        "static_feature_thresholds": static_feature_thresholds,
    }))

    print(f"\nSaved -> {model_path}  best val_mse={best_val_mse:.5f}  "
          f"threshold={threshold:.5f} (p{threshold_percentile}, score_mode={score_mode})")


def main():
    if not TORCH_AVAILABLE:
        raise SystemExit("torch is required. pip install torch")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",          type=int,   default=40)
    p.add_argument("--window-size",     type=int,   default=DEFAULT_WINDOW_SIZE)
    p.add_argument("--alpha",           type=float, default=0.3,
                    help="Weight of domain adversarial loss relative to reconstruction loss")
    p.add_argument("--grl-lambda-max",  type=float, default=0.3,
                    help="Maximum reversal strength (reached at final epoch)")
    p.add_argument("--out-dir",                     default="artifacts")
    p.add_argument("--threshold-percentile", type=float, default=99.0)
    p.add_argument("--score-mode", choices=["max","mean"], default="max")
    p.add_argument("--include-held-out", action="store_true",
                    help="Include edge_gateway in training (ablation only — "
                         "defeats the generalization-gap eval if used)")
    args = p.parse_args()
    train(
        epochs=args.epochs, window_size=args.window_size,
        alpha=args.alpha, grl_lambda_max=args.grl_lambda_max,
        out_dir=args.out_dir, threshold_percentile=args.threshold_percentile,
        score_mode=args.score_mode, include_held_out=args.include_held_out,
    )


if __name__ == "__main__":
    main()
