"""
EdgeVigil — Domain-Adversarial Evaluation  (Phase 3)

Usage:
    python core/eval_adversarial.py                           # all 3 trained types
    python core/eval_adversarial.py --device-type server      # single type
    python core/eval_adversarial.py --device-type edge_gateway  # generalization-gap test
    python core/eval_adversarial.py --percentile 97           # threshold sweep

The generalization-gap test (--device-type edge_gateway) is the core Phase 3
result: F1 on a device type never seen during training, comparing the
domain-adversarial model against the best Phase 2 per-device baseline.

dom_accuracy in the output is a diagnostic: during training the domain
classifier starts at random (~33% for 3 classes) and the encoder is
penalised for letting it do better. In a well-trained model you want
dom_accuracy back near chance (33%) — the encoder has successfully stopped
leaking device-type information into the latent space.
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
    from sklearn.metrics import f1_score, precision_score, recall_score
except ImportError:
    raise SystemExit("torch and scikit-learn required. pip install torch scikit-learn")

from simulate import TelemetrySimulator, METRICS, DEVICE_PROFILES
from inject_failures import FailureInjector
from windowing import make_windows, Standardizer
from models.domain_adversarial import DomainAdversarialAutoencoder

TRAIN_DEVICE_TYPES = list(DEVICE_PROFILES.keys())
DOMAIN_TO_IDX      = {dt: i for i, dt in enumerate(TRAIN_DEVICE_TYPES)}
DEFAULT_WINDOW_SIZE = 12


def load_model(model_path: Path):
    ckpt  = torch.load(model_path, map_location="cpu", weights_only=True)
    model = DomainAdversarialAutoencoder(
        n_features=ckpt["n_features"], window_size=ckpt["window_size"],
        n_domains=ckpt["n_domains"],   hidden_size=ckpt["hidden_size"],
        latent_dim=ckpt["latent_dim"], domain_hidden=ckpt["domain_hidden"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["window_size"], ckpt.get("domain_to_idx", DOMAIN_TO_IDX)


def build_test_fleet(device_type: str, seed: int = 99):
    sim = TelemetrySimulator(seed=seed)
    if device_type == "edge_gateway":
        sim.held_out_device_type(
            "edge_gateway", base_on="iot_sensor",
            perturb={"cpu": {"mean": 1.4, "std": 1.2},
                     "network_latency": {"mean": 0.7, "std": 1.5},
                     "temperature": {"std": 0.7}},
        )
    df = sim.simulate_fleet({device_type: 4}, duration_hours=3 * 24)
    inj = FailureInjector(seed=seed)
    injections = inj.random_injection_batch(df, n_injections=5)
    df = inj.inject(df, injections)
    return df, injections


def static_threshold_baseline_windows(df, feature_thresholds, window_size, step=1):
    row_flag = np.zeros(len(df), dtype=bool)
    for feat, thr in feature_thresholds.items():
        row_flag |= (df[feat].to_numpy() > thr)
    df = df.copy()
    df["_flag"] = row_flag
    flags = []
    for _, group in df.groupby("device_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)
        if n < window_size:
            continue
        f = group["_flag"].to_numpy()
        for s in range(0, n - window_size + 1, step):
            flags.append(bool(f[s:s+window_size].any()))
    return np.array(flags, dtype=bool)


def domain_accuracy(model, scaler, X: np.ndarray, true_domain_idx: int) -> float:
    """Fraction of windows correctly classified by the domain classifier.
    In a well-trained adversarial model, this should be near chance (1/n_domains)."""
    X_norm = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        z       = model.encode(torch.from_numpy(X_norm))
        logits  = model.domain_classifier(z)
        preds   = logits.argmax(dim=1).numpy()
    return float((preds == true_domain_idx).mean())


def evaluate(device_type: str = "server", model_dir: str = "artifacts",
              percentile: float = None, seed: int = 99) -> dict:
    model_path      = Path(model_dir) / "daa_model.pt"
    scaler_path     = Path(model_dir) / "scaler_adversarial.json"
    thresholds_path = Path(model_dir) / "thresholds_adversarial.json"

    for p in (model_path, scaler_path, thresholds_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run train_adversarial.py first")

    model, window_size, domain_to_idx = load_model(model_path)
    scaler      = Standardizer.from_dict(json.loads(scaler_path.read_text()))
    thresh_data = json.loads(thresholds_path.read_text())

    score_mode = thresh_data.get("score_mode", "max")
    threshold  = (float(np.percentile(thresh_data["val_errors"], percentile))
                  if percentile is not None
                  else thresh_data["reconstruction_threshold"])

    df, injections = build_test_fleet(device_type, seed=seed)
    X, y_true, end_ts, device_ids, _ = make_windows(df, window_size=window_size, step=1)
    if len(X) == 0:
        raise RuntimeError("No windows produced")

    X_norm = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        errors = model.anomaly_score(torch.from_numpy(X_norm), mode=score_mode).numpy()
    y_pred = errors > threshold

    lag_minutes, missed = [], []
    for spec in injections:
        after_onset = end_ts >= np.datetime64(spec.onset)
        mask = (device_ids == spec.device_id) & y_pred & after_onset
        if not mask.any():
            missed.append(spec.device_id)
            continue
        lag = (end_ts[mask].min() - np.datetime64(spec.onset)) / np.timedelta64(1, "m")
        lag_minutes.append(float(lag))

    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    fpr       = (y_pred & ~y_true.astype(bool)).sum() / max((~y_true.astype(bool)).sum(), 1)

    static_pred      = static_threshold_baseline_windows(
        df, thresh_data["static_feature_thresholds"], window_size=window_size, step=1)
    static_f1        = f1_score(y_true, static_pred, zero_division=0)
    static_precision = precision_score(y_true, static_pred, zero_division=0)
    static_recall    = recall_score(y_true, static_pred, zero_division=0)
    static_fpr       = (static_pred & ~y_true.astype(bool)).sum() / max((~y_true.astype(bool)).sum(), 1)

    # Domain accuracy: how much device-type info still leaks through the encoder?
    # Chance = 1/n_domains. Closer to chance = better domain adversarial training.
    dom_idx = domain_to_idx.get(device_type, -1)
    dom_acc = domain_accuracy(model, scaler, X, dom_idx) if dom_idx >= 0 else None
    chance  = 1.0 / model.n_domains

    return {
        "device_type":               device_type,
        "score_mode":                score_mode,
        "n_windows":                 int(len(X)),
        "n_injections":              len(injections),
        "n_injections_detected":     len(lag_minutes),
        "n_injections_missed":       len(missed),
        "threshold":                 round(float(threshold), 5),
        "f1":                        round(float(f1), 4),
        "precision":                 round(float(precision), 4),
        "recall":                    round(float(recall), 4),
        "model_fpr":                 round(float(fpr), 4),
        "static_baseline_f1":        round(float(static_f1), 4),
        "static_baseline_precision": round(float(static_precision), 4),
        "static_baseline_recall":    round(float(static_recall), 4),
        "static_baseline_fpr":       round(float(static_fpr), 4),
        "mean_detection_lag_minutes": round(float(np.mean(lag_minutes)), 2) if lag_minutes else None,
        "detection_lags_minutes":    [round(l, 2) for l in lag_minutes],
        # Domain leakage diagnostic
        "dom_accuracy":              round(dom_acc, 4) if dom_acc is not None else "n/a (held-out type)",
        "dom_chance":                round(chance, 4),
        "dom_leakage_ratio":         round(dom_acc / chance, 3) if dom_acc is not None else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-type", default="all",
                    help="Device type to evaluate, or 'all', or 'edge_gateway' "
                         "for the held-out generalization test")
    p.add_argument("--model-dir",   default="artifacts")
    p.add_argument("--percentile",  type=float, default=None)
    args = p.parse_args()

    if args.device_type == "all":
        targets = TRAIN_DEVICE_TYPES + ["edge_gateway"]
    else:
        targets = [args.device_type]

    for dt in targets:
        result = evaluate(dt, model_dir=args.model_dir, percentile=args.percentile)
        print(json.dumps(result, indent=2))
        print()


if __name__ == "__main__":
    main()
