"""
EdgeVigil — Domain-Adversarial Autoencoder  (Phase 3)

Architecture:
  Shared encoder  : LSTM -> last hidden state -> linear -> latent_dim
  Reconstruction  : latent -> LSTM decoder -> output_fc  (same as Phase 2)
  Domain branch   : latent -> GRL -> domain_classifier -> n_device_types

The GRL (Gradient Reversal Layer) reverses the sign of gradients flowing
from the domain classifier back into the shared encoder during backprop.
This forces the encoder to learn latent representations that are
  (a) informative enough to reconstruct the input well, AND
  (b) uninformative about which device type produced it.

Effect on Phase 2's IoT sensor problem: the high-variance network_latency
and temperature signals that made the IoT sensor hard to reconstruct will
no longer "contaminate" the server/workstation latent space, because the
encoder is penalized for encoding device-type-distinguishing information.
Expected outcome: lower reconstruction error on the pooled test fleet,
and better generalization to a held-out device type (edge_gateway) that
was never seen during training.

Same anomaly_score() API as LSTMAutoencoder so eval.py runs unchanged.

Reference: Ganin & Lempitsky, "Unsupervised Domain Adaptation by
Backpropagation", ICML 2015 — the exact technique used in HalluProbe
for cross-domain hallucination detection, applied here to device fleets.
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class GradientReversalFunction(torch.autograd.Function):
        """
        Forward pass: identity.
        Backward pass: multiply gradient by -lambda_ to reverse it.
        lambda_ is stored in ctx so it can vary per call (e.g. scheduled).
        """
        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.lambda_ = lambda_
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.lambda_, None

    class GradientReversalLayer(nn.Module):
        def __init__(self, lambda_: float = 1.0):
            super().__init__()
            self.lambda_ = lambda_

        def forward(self, x):
            return GradientReversalFunction.apply(x, self.lambda_)

        def set_lambda(self, lam: float):
            self.lambda_ = lam

    class DomainAdversarialAutoencoder(nn.Module):
        """
        Drop-in replacement for LSTMAutoencoder that adds domain-adversarial
        training. Shares the same anomaly_score() / reconstruction_error() API
        so eval.py requires no changes — it just loads this checkpoint instead.

        Parameters
        ----------
        n_features      : number of telemetry metrics (5)
        window_size     : number of timesteps per window
        n_domains       : number of device types in the training pool
        hidden_size     : LSTM hidden units
        latent_dim      : bottleneck dimension
        domain_hidden   : hidden units in the domain classifier MLP
        grl_lambda      : initial reversal strength (scheduled during training)
        """
        def __init__(self, n_features: int, window_size: int, n_domains: int,
                      hidden_size: int = 32, latent_dim: int = 16,
                      domain_hidden: int = 16, grl_lambda: float = 0.0):
            super().__init__()
            self.n_features  = n_features
            self.window_size = window_size
            self.n_domains   = n_domains
            self.latent_dim  = latent_dim

            # Shared encoder (identical to LSTMAutoencoder)
            self.encoder_lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
            self.encoder_fc   = nn.Linear(hidden_size, latent_dim)

            # Reconstruction decoder (identical to LSTMAutoencoder)
            self.decoder_fc   = nn.Linear(latent_dim, hidden_size)
            self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.output_fc    = nn.Linear(hidden_size, n_features)

            # Domain-adversarial branch
            self.grl = GradientReversalLayer(lambda_=grl_lambda)
            self.domain_classifier = nn.Sequential(
                nn.Linear(latent_dim, domain_hidden),
                nn.ReLU(),
                nn.Linear(domain_hidden, n_domains),
            )

        # ------------------------------------------------------------------
        # Forward passes
        # ------------------------------------------------------------------

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            _, (h_n, _) = self.encoder_lstm(x)
            return self.encoder_fc(h_n[-1])

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            h = self.decoder_fc(z).unsqueeze(1).repeat(1, self.window_size, 1)
            out, _ = self.decoder_lstm(h)
            return self.output_fc(out)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Reconstruction only — used for anomaly scoring."""
            return self.decode(self.encode(x))

        def forward_with_domain(self, x: "torch.Tensor"):
            """
            Returns (reconstruction, domain_logits).
            Used during training so both losses can be computed in one pass.
            """
            z     = self.encode(x)
            recon = self.decode(z)
            # GRL reverses gradients: domain classifier sees the right
            # direction; encoder sees reversed gradients and learns to
            # be domain-uninformative.
            domain_logits = self.domain_classifier(self.grl(z))
            return recon, domain_logits

        def set_grl_lambda(self, lam: float):
            self.grl.set_lambda(lam)

        # ------------------------------------------------------------------
        # Scoring API — identical to LSTMAutoencoder
        # ------------------------------------------------------------------

        def reconstruction_error(self, x: "torch.Tensor") -> "torch.Tensor":
            """Mean-over-window MSE. Training/val loss monitoring only."""
            return ((self.forward(x) - x) ** 2).mean(dim=(1, 2))

        def per_timestep_error(self, x: "torch.Tensor") -> "torch.Tensor":
            """MSE per timestep averaged across features. Shape: [batch, window]."""
            return ((self.forward(x) - x) ** 2).mean(dim=2)

        def anomaly_score(self, x: "torch.Tensor", mode: str = "max") -> "torch.Tensor":
            """
            Per-window detection score.
            mode='max'  -> worst timestep in window (default, fixes mean-pooling dilution)
            mode='mean' -> equivalent to reconstruction_error()
            """
            per_step = self.per_timestep_error(x)
            if mode == "max":
                return per_step.max(dim=1).values
            elif mode == "mean":
                return per_step.mean(dim=1)
            else:
                raise ValueError(f"Unknown mode '{mode}', expected 'max' or 'mean'")
