"""
EdgeVigil — Phase 2: Evaluation harness
Loads a trained per-device-type LSTM-autoencoder, scores a held-out test
fleet with injected failures, and reports:

    - F1 / precision / recall on injected failure detection
    - False positive rate, model vs. a Nagios-style static threshold baseline
      (this comparison is the headline number per the README)
    - Mean detection lag: minutes between failure onset and the model's
      first alert on that device

A note on "detection lag" vs. the README's "lead time": lead time implies
catching a failure *before* it fully manifests. For gradual_drift and
slow_leak injections, the model can flag a window mid-ramp, before the
failure reaches full magnitude — that's genuine early warning. For
sudden_spike, there's no ramp to catch early, so "lag" is the honest framing
there. This script reports lag for all kinds and lets you slice by failure
kind if you want the lead-time framing only where it's earned.

Usage:
    python core/eval.py --device-type server
    python core/eval.py --device-type all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from simulate import TelemetrySimulator                     # noqa: E402
from inject_failures import FailureInjector                 # noqa: E402
from windowing import make_windows, Standardizer             # noqa: E402
from lstm_autoencoder import LSTMAutoencoder                 # noqa: E402


def build_test_fleet(device_type: str, n_devices: int = 5, duration_hours: float = 24 * 3,
                      step_minutes: float = 5.0, n_failures: int = 5, seed: int = 99):
    sim = TelemetrySimulator(seed=seed)
    df = sim.simulate_fleet({device_type: n_devices}, duration_hours=duration_hours, step_minutes=step_minutes)
    injector = FailureInjector(seed=seed)
    batch = injector.random_injection_batch(df, n_injections=min(n_failures, n_devices))
    return injector.inject(df, batch), batch


def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu")
    model = LSTMAutoencoder(n_features=ckpt["n_features"], window_size=ckpt["window_size"],
                             hidden_size=ckpt["hidden_size"], latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["window_size"]


def score_windows(model, scaler, X: np.ndarray) -> np.ndarray:
    X_norm = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        return model.reconstruction_error(torch.from_numpy(X_norm)).numpy()


def static_threshold_baseline_windows(df, feature_thresholds: dict, window_size: int, step: int = 1) -> np.ndarray:
    """
    Nagios-style baseline, properly comparable to the model: flags a WINDOW
    (not a single row) if ANY of the 5 metrics exceeds its own fixed
    threshold at ANY point in that window. Mirrors make_windows()'s exact
    groupby/sort/skip-short-device logic so the output aligns 1:1 with
    make_windows()'s y_true when called on the same df/window_size/step.
    """
    row_flag = np.zeros(len(df), dtype=bool)
    for feature, threshold in feature_thresholds.items():
        row_flag |= (df[feature].to_numpy() > threshold)

    df = df.copy()
    df["_flag"] = row_flag

    window_flags = []
    for _, group in df.groupby("device_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)
        if n < window_size:
            continue
        flags = group["_flag"].to_numpy()
        for start in range(0, n - window_size + 1, step):
            window_flags.append(bool(flags[start:start + window_size].any()))
    return np.array(window_flags, dtype=bool)


def evaluate(device_type: str, model_dir: str = "artifacts", percentile: float = None, seed: int = 99) -> dict:
    model_path = Path(model_dir) / f"lstm_ae_{device_type}.pt"
    scaler_path = Path(model_dir) / f"scaler_{device_type}.json"
    thresholds_path = Path(model_dir) / f"thresholds_{device_type}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path} — run train.py first")
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"No thresholds file at {thresholds_path} — retrain with the current train.py "
            f"(older checkpoints predate threshold persistence)"
        )

    model, window_size = load_model(model_path)
    scaler = Standardizer.from_dict(json.loads(scaler_path.read_text()))
    threshold_data = json.loads(thresholds_path.read_text())

    # Threshold comes from the held-out NORMAL validation split fit during
    # training — never recomputed from this test set's own error
    # distribution, since that set contains the failures being evaluated.
    # --percentile lets you explore a different operating point on the same
    # persisted validation errors without retraining.
    if percentile is not None:
        threshold = float(np.percentile(threshold_data["val_errors"], percentile))
    else:
        threshold = threshold_data["reconstruction_threshold"]

    df, injections = build_test_fleet(device_type, seed=seed)
    X, y_true, end_ts, device_ids, _ = make_windows(df, window_size=window_size, step=1)
    if len(X) == 0:
        raise RuntimeError("No windows produced — check window_size against series length")

    errors = score_windows(model, scaler, X)
    y_pred = errors > threshold

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    fpr = (y_pred & ~y_true).sum() / max((~y_true).sum(), 1)

    static_pred = static_threshold_baseline_windows(
        df, threshold_data["static_feature_thresholds"], window_size=window_size, step=1
    )
    static_f1 = f1_score(y_true, static_pred, zero_division=0)
    static_precision = precision_score(y_true, static_pred, zero_division=0)
    static_recall = recall_score(y_true, static_pred, zero_division=0)
    static_fpr = (static_pred & ~y_true).sum() / max((~y_true).sum(), 1)

    lag_minutes = []
    for spec in injections:
        dev_mask = device_ids == spec.device_id
        alerts = end_ts[dev_mask & y_pred]
        alerts_after_onset = alerts[alerts >= np.datetime64(spec.onset)]
        if len(alerts_after_onset) > 0:
            first_alert = alerts_after_onset.min()
            lag_minutes.append((first_alert - np.datetime64(spec.onset)) / np.timedelta64(1, "m"))

    return {
        "device_type": device_type,
        "n_windows": int(len(X)),
        "n_injections": len(injections),
        "n_injections_detected": len(lag_minutes),
        "threshold": round(float(threshold), 5),
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "model_fpr": round(float(fpr), 4),
        "static_baseline_f1": round(float(static_f1), 4),
        "static_baseline_precision": round(float(static_precision), 4),
        "static_baseline_recall": round(float(static_recall), 4),
        "static_baseline_fpr": round(float(static_fpr), 4),
        "mean_detection_lag_minutes": round(float(np.mean(lag_minutes)), 2) if lag_minutes else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: evaluate trained baseline against injected failures")
    parser.add_argument("--device-type", choices=["server", "workstation", "iot_sensor", "all"], default="all")
    parser.add_argument("--model-dir", default="artifacts")
    parser.add_argument("--percentile", type=float, default=None,
                         help="Override the persisted threshold by re-deriving it at this percentile "
                              "of the training run's held-out validation errors (no retrain needed).")
    args = parser.parse_args()

    targets = ["server", "workstation", "iot_sensor"] if args.device_type == "all" else [args.device_type]
    for dt in targets:
        print(json.dumps(evaluate(dt, model_dir=args.model_dir, percentile=args.percentile), indent=2))


if __name__ == "__main__":
    main()
