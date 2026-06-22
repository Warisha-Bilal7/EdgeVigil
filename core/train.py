"""
EdgeVigil — Training entrypoint
Phase 2/3 — not yet implemented.

Planned interface:
    python core/train.py --phase baseline        # Phase 2: per-device-type LSTM-AE, no domain adaptation
    python core/train.py --phase adversarial      # Phase 3: + GRL + domain classifier

Phase 2 establishes the F1/FPR baseline that Phase 3's domain-adversarial
generalization-gap result needs to beat on a held-out device type.
"""
raise NotImplementedError("Phase 2 (baseline anomaly model) hasn't been built yet.")
