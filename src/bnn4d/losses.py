"""Published objectives for the aleatoric and epistemic networks."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def gaussian_nll(
    target: Tensor,
    mean: Tensor,
    log_variance: Tensor,
    reduction: str = "mean",
    clamp_min: float = -6.0,
    clamp_max: float = 4.0,
) -> Tensor:
    """Heteroscedastic Gaussian NLL, Equation (2) in the paper with numerical stability clamping."""
    clamped_logv = torch.clamp(log_variance, min=clamp_min, max=clamp_max)
    loss = 0.5 * (math.log(2 * math.pi) + clamped_logv + (target - mean).square() * torch.exp(-clamped_logv))
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def variational_free_energy(
    prediction: Tensor,
    target: Tensor,
    kl_divergence: Tensor,
    training_size: int,
    observation_std: float = 1.0,
    kl_scale: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """One-sample Monte Carlo ELBO: Gaussian data fit + scaled KL complexity (Eq. 5).

    Returns total, data-fit and scaled complexity terms for transparent logging.
    """
    if training_size < 1 or observation_std <= 0:
        raise ValueError("training_size and observation_std must be positive")
    variance = observation_std**2
    data_fit = (0.5 * (math.log(2 * math.pi * variance) + (target - prediction).square() / variance)).mean()
    beta = kl_scale if kl_scale is not None else (1.0 / max(training_size, 10000))
    complexity = kl_divergence * beta
    return data_fit + complexity, data_fit, complexity

