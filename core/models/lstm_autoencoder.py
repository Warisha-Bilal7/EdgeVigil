"""
EdgeVigil — LSTM Autoencoder  (Phase 2)

Architecture:
  Encoder: LSTM -> last hidden state -> linear projection to latent_dim
  Decoder: repeat latent -> LSTM -> linear projection back to n_features

Detection scoring (anomaly_score):
  mode='max'  — max per-timestep error in the window (DEFAULT for detection)
  mode='mean' — mean over the whole window (used for training loss only)

  The distinction matters: mean-pooling dilutes brief or ramping failures that
  only displace a few of the window's timesteps. At p99 threshold, server recall
  was 23% with mean-pooling; max-pooling is expected to close that gap without
  touching the training loss or retraining the weights.

  reconstruction_error() is kept as a thin alias to mean-pooled scoring so
  existing training-loop MSE logging doesn't break; anomaly_score() is the
  correct entry point for detection/eval.
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features: int, window_size: int,
                      hidden_size: int = 32, latent_dim: int = 16,
                      num_layers: int = 1, dropout: float = 0.0):
            super().__init__()
            self.n_features  = n_features
            self.window_size = window_size
            self.hidden_size = hidden_size
            self.latent_dim  = latent_dim

            self.encoder_lstm = nn.LSTM(n_features, hidden_size, num_layers,
                                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.encoder_fc   = nn.Linear(hidden_size, latent_dim)
            self.decoder_fc   = nn.Linear(latent_dim, hidden_size)
            self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.output_fc    = nn.Linear(hidden_size, n_features)

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            _, (h_n, _) = self.encoder_lstm(x)
            return self.encoder_fc(h_n[-1])

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            h = self.decoder_fc(z).unsqueeze(1).repeat(1, self.window_size, 1)
            out, _ = self.decoder_lstm(h)
            return self.output_fc(out)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.decode(self.encode(x))

        # ------------------------------------------------------------------
        # Scoring API
        # ------------------------------------------------------------------

        def reconstruction_error(self, x: "torch.Tensor") -> "torch.Tensor":
            """Mean-over-window MSE — used for training/val loss monitoring only."""
            return ((self.forward(x) - x) ** 2).mean(dim=(1, 2))

        def per_timestep_error(self, x: "torch.Tensor") -> "torch.Tensor":
            """MSE per timestep, averaged across features. Shape: [batch, window]."""
            return ((self.forward(x) - x) ** 2).mean(dim=2)

        def anomaly_score(self, x: "torch.Tensor", mode: str = "max") -> "torch.Tensor":
            """
            Per-window detection score. Separate from reconstruction_error()
            because mean-over-window dilutes brief/ramping anomalies.

            mode='max'  -> worst timestep in window  (use this for detection)
            mode='mean' -> equivalent to reconstruction_error()
            """
            per_step = self.per_timestep_error(x)
            if mode == "max":
                return per_step.max(dim=1).values
            elif mode == "mean":
                return per_step.mean(dim=1)
            else:
                raise ValueError(f"Unknown mode '{mode}', expected 'max' or 'mean'")
