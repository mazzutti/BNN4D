"""Small, explicit training loops for the two published objectives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .losses import gaussian_nll, variational_free_energy
from .models import AleatoricAutoencoder, EpistemicBNN


def train_aleatoric_epoch(
    model: AleatoricAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total, count = 0.0, 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        mean, log_variance = model(features)
        loss = gaussian_nll(target, mean, log_variance)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        total += loss.item() * features.shape[0]
        count += features.shape[0]
    return total / count


def train_epistemic_epoch(
    model: EpistemicBNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    training_size: int,
    observation_std: float = 1.0,
    device: torch.device | str = "cpu",
    max_grad_norm: float = 1.0,
    kl_scale: float | None = None,
) -> float:
    model.train()
    model.clear_sample()
    total, count = 0.0, 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features, sample=True)
        loss, _, _ = variational_free_energy(prediction, target, model.kl_divergence(), training_size, observation_std, kl_scale=kl_scale)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        total += loss.item() * features.shape[0]
        count += features.shape[0]
    return total / count


@torch.no_grad()
def evaluate_aleatoric(
    model: AleatoricAutoencoder,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> float:
    model.eval()
    total, count = 0.0, 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        mean, log_variance = model(features)
        loss = gaussian_nll(target, mean, log_variance)
        total += loss.item() * features.shape[0]
        count += features.shape[0]
    return total / count if count > 0 else 0.0


@torch.no_grad()
def evaluate_epistemic(
    model: EpistemicBNN,
    loader: DataLoader,
    training_size: int,
    observation_std: float = 1.0,
    device: torch.device | str = "cpu",
) -> float:
    """Evaluate predictive data-fit loss on the validation split.

    In Bayesian model evaluation, monitoring data likelihood on held-out samples
    provides a reliable, calibrated signal for early stopping and model selection
    without being skewed by asymptotic parameter complexity (KL) drift.
    """
    model.eval()
    model.clear_sample()
    total_fit, count = 0.0, 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        prediction = model(features, sample=False)
        _, data_fit, _ = variational_free_energy(prediction, target, model.kl_divergence(), training_size, observation_std)
        total_fit += data_fit.item() * features.shape[0]
        count += features.shape[0]
    return total_fit / count if count > 0 else 0.0

