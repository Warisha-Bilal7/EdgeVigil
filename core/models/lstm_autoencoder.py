"""
EdgeVigil — LSTM Autoencoder
Phase 2: baseline anomaly model, one instance trained per device type, no
domain adaptation. Phase 3 will swap LSTMEncoder for a shared encoder + GRL
domain classifier feeding this same decoder/reconstruction head — the
encoder/decoder split here is deliberate so that swap doesn't require
touching the decoder or the reconstruction-error scoring logic.
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Encodes a [batch, window, n_features] sequence into a latent vector."""

    def __init__(self, n_features: int, hidden_size: int = 32, latent_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]              # [batch, hidden_size] — final layer's last hidden state
        return self.to_latent(h_last)  # [batch, latent_dim]


class LSTMDecoder(nn.Module):
    """Decodes a latent vector back into a [batch, window, n_features] reconstruction."""

    def __init__(self, n_features: int, window_size: int, hidden_size: int = 32,
                 latent_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.window_size = window_size
        self.from_latent = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.to_output = nn.Linear(hidden_size, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seed = self.from_latent(z).unsqueeze(1).repeat(1, self.window_size, 1)  # [batch, window, hidden]
        out, _ = self.lstm(seed)
        return self.to_output(out)  # [batch, window, n_features]


class LSTMAutoencoder(nn.Module):
    """
    Reconstruction-based anomaly detector. High reconstruction error on a
    window means "this doesn't look like normal behavior for this device
    type" — that gap is the anomaly score.
    """

    def __init__(self, n_features: int, window_size: int, hidden_size: int = 32,
                 latent_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.encoder = LSTMEncoder(n_features, hidden_size, latent_dim, num_layers)
        self.decoder = LSTMDecoder(n_features, window_size, hidden_size, latent_dim, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window anomaly score: mean squared reconstruction error. Shape: [batch]."""
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=(1, 2))


if __name__ == "__main__":
    # Shape sanity check — no real data needed.
    batch, window, n_features = 8, 10, 5
    model = LSTMAutoencoder(n_features=n_features, window_size=window)
    x = torch.randn(batch, window, n_features)
    recon = model(x)
    errors = model.reconstruction_error(x)
    assert recon.shape == x.shape, f"expected {x.shape}, got {recon.shape}"
    assert errors.shape == (batch,), f"expected ({batch},), got {errors.shape}"
    print(f"recon shape OK: {recon.shape}")
    print(f"reconstruction_error shape OK: {errors.shape}, sample values: {errors[:3].tolist()}")
