"""
EdgeVigil — Evaluation Harness  (Phase 2)

Usage:
    python core/eval.py --device-type server
    python core/eval.py --device-type server --percentile 95   # re-derive threshold, no retrain

Reports (model vs. Nagios-style static baseline):
    F1, precision, recall, FPR, mean detection lag per injected failure.

Design notes:
  - Threshold is loaded from artifacts/thresholds_{device_type}.json (set from a
    held-out NORMAL validation split during training). --percentile re-derives
    from the saved val_errors without retraining.
  - Static baseline fires on ANY of the 5 metrics exceeding its own per-feature
    threshold at window-level granularity (not just cpu) so the comparison is
    apples-to-apples with the model's window-level F1/FPR.
  - score_mode is read from the thresholds file so it can't silently mismatch
    what the threshold was actually computed on. Older checkpoints (pre-score_mode
    persistence) fall back to 'mean' with a warning.
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
    raise SystemExit("torch and scikit-learn are required. pip install torch scikit-learn")

from simulate import TelemetrySimulator, METRICS, DEVICE_PROFILES
from inject_failures import FailureInjector
from windowing import make_windows, Standardizer
from models.lstm_autoencoder import LSTMAutoencoder

DEVICE_TYPES        = list(DEVICE_PROFILES.keys())
DEFAULT_WINDOW_SIZE = 12


def load_model(model_path: Path):
    ckpt        = torch.load(model_path, map_location="cpu", weights_only=True)
    window_size = ckpt["window_size"]
    model       = LSTMAutoencoder(ckpt["n_features"], window_size,
                                   ckpt["hidden_size"], ckpt["latent_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, window_size


def score_windows(model, scaler: Standardizer, X: np.ndarray, score_mode: str = "max") -> np.ndarray:
    X_norm = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        return model.anomaly_score(torch.from_numpy(X_norm), mode=score_mode).numpy()


def build_test_fleet(device_type: str, seed: int = 99):
    """3 days of telemetry with 5 injected failures, different seed from training."""
    sim = TelemetrySimulator(seed=seed)
    df  = sim.simulate_fleet({device_type: 4}, duration_hours=3 * 24)
    injector = FailureInjector(seed=seed)
    injections = injector.random_injection_batch(df, n_injections=5)
    df = injector.inject(df, injections)
    return df, injections


def static_threshold_baseline_windows(df, feature_thresholds: dict,
                                        window_size: int, step: int = 1) -> np.ndarray:
    """
    Nagios-style baseline at window granularity: flags a window if ANY of
    the 5 metrics exceeds its fixed threshold in ANY timestep of that window.
    Uses identical groupby/sort/skip logic as make_windows() so outputs are
    aligned 1:1 and the F1/FPR comparison is apples-to-apples.
    """
    row_flag = np.zeros(len(df), dtype=bool)
    for feature, threshold in feature_thresholds.items():
        row_flag |= (df[feature].to_numpy() > threshold)
    df = df.copy()
    df["_flag"] = row_flag
    window_flags = []
    for _, group in df.groupby("device_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n     = len(group)
        if n < window_size:
            continue
        flags = group["_flag"].to_numpy()
        for start in range(0, n - window_size + 1, step):
            window_flags.append(bool(flags[start:start + window_size].any()))
    return np.array(window_flags, dtype=bool)


def evaluate(device_type: str, model_dir: str = "artifacts",
              percentile: float = None, seed: int = 99) -> dict:
    model_path      = Path(model_dir) / f"lstm_ae_{device_type}.pt"
    scaler_path     = Path(model_dir) / f"scaler_{device_type}.json"
    thresholds_path = Path(model_dir) / f"thresholds_{device_type}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path} — run train.py first")
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"No thresholds file at {thresholds_path} — retrain with current train.py "
            f"(older checkpoints predate threshold persistence)")

    model, window_size = load_model(model_path)
    scaler     = Standardizer.from_dict(json.loads(scaler_path.read_text()))
    thresh_data = json.loads(thresholds_path.read_text())

    # score_mode must match what the threshold was fit on.
    score_mode = thresh_data.get("score_mode")
    if score_mode is None:
        score_mode = "mean"
        print(f"[{device_type}] WARNING: thresholds file predates score_mode persistence — "
              f"assuming 'mean'. Retrain with current train.py to use max-pooled scoring.")

    # Threshold from held-out normal validation split — never from the test set.
    if percentile is not None:
        threshold = float(np.percentile(thresh_data["val_errors"], percentile))
    else:
        threshold = thresh_data["reconstruction_threshold"]

    df, injections = build_test_fleet(device_type, seed=seed)
    X, y_true, end_ts, device_ids, _ = make_windows(df, window_size=window_size, step=1)
    if len(X) == 0:
        raise RuntimeError("No windows produced")

    errors = score_windows(model, scaler, X, score_mode=score_mode)
    y_pred = errors > threshold

    # Per-injection detection lag
    lag_minutes = []
    for spec in injections:
        mask = (device_ids == spec.device_id) & y_pred
        if not mask.any():
            continue
        first_alert = end_ts[mask].min()
        lag = (first_alert - np.datetime64(spec.onset)) / np.timedelta64(1, "m")
        lag_minutes.append(float(lag))

    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    fpr       = (y_pred & ~y_true.astype(bool)).sum() / max((~y_true.astype(bool)).sum(), 1)

    static_pred = static_threshold_baseline_windows(
        df, thresh_data["static_feature_thresholds"], window_size=window_size, step=1)
    static_f1        = f1_score(y_true, static_pred, zero_division=0)
    static_precision = precision_score(y_true, static_pred, zero_division=0)
    static_recall    = recall_score(y_true, static_pred, zero_division=0)
    static_fpr       = (static_pred & ~y_true.astype(bool)).sum() / max((~y_true.astype(bool)).sum(), 1)

    return {
        "device_type":                  device_type,
        "score_mode":                   score_mode,
        "n_windows":                    int(len(X)),
        "n_injections":                 len(injections),
        "n_injections_detected":        len(lag_minutes),
        "threshold":                    round(float(threshold), 5),
        "f1":                           round(float(f1), 4),
        "precision":                    round(float(precision), 4),
        "recall":                       round(float(recall), 4),
        "model_fpr":                    round(float(fpr), 4),
        "static_baseline_f1":           round(float(static_f1), 4),
        "static_baseline_precision":    round(float(static_precision), 4),
        "static_baseline_recall":       round(float(static_recall), 4),
        "static_baseline_fpr":          round(float(static_fpr), 4),
        "mean_detection_lag_minutes":   round(float(np.mean(lag_minutes)), 2) if lag_minutes else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-type", default="server")
    parser.add_argument("--model-dir",   default="artifacts")
    parser.add_argument("--percentile",  type=float, default=None,
                         help="Override persisted threshold using this percentile of the "
                              "saved validation errors (no retrain needed).")
    args = parser.parse_args()
    targets = DEVICE_TYPES if args.device_type == "all" else [args.device_type]
    for dt in targets:
        result = evaluate(dt, model_dir=args.model_dir, percentile=args.percentile)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
