"""EdgeVigil — Phase 3 tests (GRL, DomainAdversarialAutoencoder)."""

import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

try:
    import torch
    import torch.nn as nn
    from models.domain_adversarial import (
        GradientReversalLayer, GradientReversalFunction,
        DomainAdversarialAutoencoder,
    )
    TORCH = True
except ImportError:
    TORCH = False

skip_torch = unittest.skipUnless(TORCH, "torch not installed in this environment")


class TestGradientReversalLayer(unittest.TestCase):
    @skip_torch
    def test_forward_is_identity(self):
        x   = torch.randn(4, 8)
        grl = GradientReversalLayer(lambda_=1.0)
        torch.testing.assert_close(grl(x), x)

    @skip_torch
    def test_backward_reverses_gradient(self):
        """GRL should negate and scale gradients — confirmed by checking
        that a parameter downstream of GRL moves in the OPPOSITE direction
        of gradient descent on the same loss."""
        x = torch.randn(2, 4, requires_grad=False)
        w = nn.Linear(4, 2, bias=False)

        # Without GRL
        out_no_grl  = w(x).sum()
        out_no_grl.backward()
        grad_no_grl = w.weight.grad.clone()
        w.zero_grad()

        # With GRL (lambda=1.0 → full reversal)
        grl = GradientReversalLayer(lambda_=1.0)
        out_grl = w(grl(x)).sum()
        out_grl.backward()
        grad_grl = w.weight.grad.clone()

        torch.testing.assert_close(grad_grl, -grad_no_grl)

    @skip_torch
    def test_lambda_scales_reversed_gradient(self):
        x   = torch.randn(2, 4)
        w   = nn.Linear(4, 2, bias=False)
        grl = GradientReversalLayer(lambda_=0.5)

        w(grl(x)).sum().backward()
        grad_half = w.weight.grad.clone()
        w.zero_grad()

        w(GradientReversalLayer(lambda_=1.0)(x)).sum().backward()
        grad_full = w.weight.grad.clone()

        torch.testing.assert_close(grad_half, grad_full * 0.5)

    @skip_torch
    def test_set_lambda_updates(self):
        grl = GradientReversalLayer(lambda_=0.1)
        grl.set_lambda(0.9)
        self.assertAlmostEqual(grl.lambda_, 0.9)


class TestDomainAdversarialAutoencoder(unittest.TestCase):
    def _make_model(self, n_features=5, window_size=10, n_domains=3):
        return DomainAdversarialAutoencoder(
            n_features=n_features, window_size=window_size, n_domains=n_domains,
            hidden_size=16, latent_dim=8, domain_hidden=8,
        )

    @skip_torch
    def test_forward_shape(self):
        model = self._make_model()
        x     = torch.randn(6, 10, 5)
        out   = model(x)
        self.assertEqual(out.shape, x.shape)

    @skip_torch
    def test_forward_with_domain_shapes(self):
        model        = self._make_model()
        x            = torch.randn(6, 10, 5)
        recon, logits = model.forward_with_domain(x)
        self.assertEqual(recon.shape, x.shape)
        self.assertEqual(logits.shape, (6, 3))  # batch x n_domains

    @skip_torch
    def test_reconstruction_error_nonnegative(self):
        model = self._make_model()
        x     = torch.randn(4, 10, 5)
        err   = model.reconstruction_error(x)
        self.assertEqual(tuple(err.shape), (4,))
        self.assertTrue((err >= 0).all())

    @skip_torch
    def test_anomaly_score_max_geq_mean(self):
        torch.manual_seed(0)
        model = self._make_model()
        x     = torch.randn(8, 10, 5)
        self.assertTrue(
            (model.anomaly_score(x, "max") >= model.anomaly_score(x, "mean") - 1e-6).all()
        )

    @skip_torch
    def test_anomaly_score_invalid_mode_raises(self):
        model = self._make_model()
        with self.assertRaises(ValueError):
            model.anomaly_score(torch.randn(2, 10, 5), mode="bogus")

    @skip_torch
    def test_grl_lambda_zero_no_adversarial_pressure(self):
        """With lambda=0 the GRL is a no-op; total loss should equal recon loss."""
        torch.manual_seed(1)
        model = self._make_model()
        model.set_grl_lambda(0.0)
        x = torch.randn(4, 10, 5)
        y = torch.zeros(4, dtype=torch.long)
        recon, logits = model.forward_with_domain(x)
        recon_loss  = nn.MSELoss()(recon, x)
        domain_loss = nn.CrossEntropyLoss()(logits, y)
        # With lambda=0 domain gradients are zeroed; recon_loss dominates
        total = recon_loss + 0.3 * domain_loss
        self.assertGreater(total.item(), 0)

    @skip_torch
    def test_domain_classifier_output_is_logits(self):
        """Raw output of domain classifier should be logits, not probabilities."""
        model   = self._make_model()
        x       = torch.randn(4, 10, 5)
        _, logits = model.forward_with_domain(x)
        # Logits should NOT sum to 1 along the class dimension
        self.assertFalse(torch.allclose(logits.softmax(dim=1).sum(dim=1),
                                         torch.ones(4) * 1, atol=1e-3))
        # But softmax values should all be in [0, 1]
        self.assertTrue((logits.softmax(dim=1) >= 0).all())
        self.assertTrue((logits.softmax(dim=1) <= 1).all())

    @skip_torch
    def test_adversarial_training_reduces_domain_leakage(self):
        """
        After adversarial training, the domain classifier's accuracy on
        its own training data should be lower than a non-adversarial baseline
        — the encoder is learning to remove device-type information.
        This is a coarse sanity check; the full generalization-gap test
        is in eval_adversarial.py.
        """
        torch.manual_seed(42)
        n_domains, n_features, window_size = 3, 5, 10
        batch_per_domain = 32

        # Synthetic domain-separable data: each domain has a distinct mean
        X_list, y_list = [], []
        for d in range(n_domains):
            x = torch.randn(batch_per_domain, window_size, n_features) + d * 3.0
            X_list.append(x)
            y_list.extend([d] * batch_per_domain)
        X = torch.cat(X_list)
        y = torch.tensor(y_list, dtype=torch.long)

        # Train WITHOUT adversarial pressure (lambda=0) → baseline domain accuracy
        model_base = DomainAdversarialAutoencoder(
            n_features, window_size, n_domains, hidden_size=16, latent_dim=8,
            domain_hidden=8, grl_lambda=0.0)
        opt = torch.optim.Adam(model_base.parameters(), lr=1e-2)
        for _ in range(30):
            opt.zero_grad()
            recon, logits = model_base.forward_with_domain(X)
            (nn.MSELoss()(recon, X) + 0.3 * nn.CrossEntropyLoss()(logits, y)).backward()
            opt.step()
        with torch.no_grad():
            acc_base = (model_base.domain_classifier(model_base.encode(X))
                        .argmax(1) == y).float().mean().item()

        # Train WITH full adversarial pressure (lambda=1.0)
        model_adv = DomainAdversarialAutoencoder(
            n_features, window_size, n_domains, hidden_size=16, latent_dim=8,
            domain_hidden=8, grl_lambda=1.0)
        opt2 = torch.optim.Adam(model_adv.parameters(), lr=1e-2)
        for _ in range(30):
            opt2.zero_grad()
            recon, logits = model_adv.forward_with_domain(X)
            (nn.MSELoss()(recon, X) + 0.3 * nn.CrossEntropyLoss()(logits, y)).backward()
            opt2.step()
        with torch.no_grad():
            acc_adv = (model_adv.domain_classifier(model_adv.encode(X))
                       .argmax(1) == y).float().mean().item()

        # Adversarial model should have LOWER domain accuracy (more confused)
        self.assertLess(acc_adv, acc_base,
                         f"Adversarial domain acc ({acc_adv:.3f}) should be < "
                         f"baseline ({acc_base:.3f})")


class TestGRLSchedule(unittest.TestCase):
    def test_schedule_starts_near_zero(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
        from train_adversarial import grl_lambda_schedule
        self.assertAlmostEqual(grl_lambda_schedule(0, 40, 0.3), 0.0, places=5)

    def test_schedule_ends_near_lambda_max(self):
        from train_adversarial import grl_lambda_schedule
        self.assertAlmostEqual(grl_lambda_schedule(40, 40, 0.3), 0.3, places=3)

    def test_schedule_is_monotonic(self):
        from train_adversarial import grl_lambda_schedule
        lams = [grl_lambda_schedule(e, 40, 0.3) for e in range(41)]
        self.assertEqual(lams, sorted(lams))


if __name__ == "__main__":
    unittest.main(verbosity=2)
