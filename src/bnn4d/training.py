"""Small, explicit training loops for the two published objectives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .losses import gaussian_nll, variational_free_energy
from .models import AleatoricAutoencoder, EpistemicBNN


def train_aleatoric_epoch(model: AleatoricAutoencoder, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device | str = "cpu") -> float:
    model.train()
    total, count = 0.0, 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        mean, log_variance = model(features)
        loss = gaussian_nll(target, mean, log_variance)
        loss.backward()
        optimizer.step()
        total += loss.item() * features.shape[0]
        count += features.shape[0]
    return total / count


def train_epistemic_epoch(model: EpistemicBNN, loader: DataLoader, optimizer: torch.optim.Optimizer, training_size: int, observation_std: float = 1.0, device: torch.device | str = "cpu") -> float:
    model.train()
    total, count = 0.0, 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features, sample=True)
        loss, _, _ = variational_free_energy(prediction, target, model.kl_divergence(), training_size, observation_std)
        loss.backward()
        optimizer.step()
        total += loss.item() * features.shape[0]
        count += features.shape[0]
    return total / count

