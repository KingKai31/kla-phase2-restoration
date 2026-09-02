"""
Local-Lipschitz-based inference-time confidence signal, grounded in
Bhutto et al. (arXiv 2305.07618) - Local Lipschitz-based OOD detection for
deep-learning image reconstruction. Their 99.94% AUC number was measured
on a different domain (MRI reconstruction, AUTOMAP architecture) - NOT
claimed to apply here. Only the METHOD is borrowed: probe the model's
local output sensitivity to small input perturbations. A model that
changes its output a lot for a tiny input change is behaving less
smoothly around that input - a candidate signal for "this output is less
trustworthy," not a guaranteed one. See
scripts/validate_confidence_signal.py for whether it actually correlates
with real error on this task before it's trusted for anything.
"""
import torch


def estimate_local_lipschitz_confidence(model, x: torch.Tensor, epsilon: float = 1e-3,
                                         n_probes: int = 4) -> float:
    """x: a single-image batch, shape (1, C, H, W). Returns a scalar float -
    the max observed |f(x+noise) - f(x)| / epsilon across n_probes random
    directions, an empirical local Lipschitz estimate (not a certified
    bound - this is a cheap Monte-Carlo probe, not a proof)."""
    model.eval()
    with torch.no_grad():
        base_output = model(x)
        deltas = []
        for _ in range(n_probes):
            noise = torch.randn_like(x) * epsilon
            perturbed_output = model(x + noise)
            local_change = (perturbed_output - base_output).abs().mean()
            deltas.append(local_change.item())
        lipschitz_estimate = max(deltas) / epsilon
    return lipschitz_estimate
