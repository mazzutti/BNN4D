"""Published objectives for the aleatoric and epistemic networks."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def gaussian_nll(target: Tensor, mean: Tensor, log_variance: Tensor, reduction: str = "mean") -> Tensor:
    """Heteroscedastic Gaussian NLL, Equation (2) in the paper."""
    loss = 0.5 * (math.log(2 * math.pi) + log_variance + (target - mean).square() * torch.exp(-log_variance))
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
) -> tuple[Tensor, Tensor, Tensor]:
    """One-sample Monte Carlo ELBO: Gaussian data fit + KL/N (Eq. 5).

    Returns total, data-fit and scaled complexity terms for transparent logging.
    """
    if training_size < 1 or observation_std <= 0:
        raise ValueError("training_size and observation_std must be positive")
    variance = observation_std**2
    data_fit = (0.5 * (math.log(2 * math.pi * variance) + (target - prediction).square() / variance)).mean()
    complexity = kl_divergence / training_size
    return data_fit + complexity, data_fit, complexity

