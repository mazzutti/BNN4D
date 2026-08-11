"""Preparation of multi-vintage 4D maps described in Section 3.2."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def add_relative_gaussian_noise(data: Tensor, fraction: float, generator: torch.Generator | None = None) -> Tensor:
    """Add zero-mean noise whose std is ``fraction * data.std()``.

    The paper evaluates 0, 6, 12, 15, 17, 24 and 40 percent; 17 percent
    is selected for the reported field application.
    """
    if not 0 <= fraction:
        raise ValueError("fraction cannot be negative")
    if fraction == 0:
        return data.clone()
    noise = torch.randn(data.shape, dtype=data.dtype, device=data.device, generator=generator)
    return data + noise * (fraction * data.std(unbiased=False))


class FeatureStandardizer:
    """Training-set standardization with serializable statistics."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.mean_: Tensor | None = None
        self.scale_: Tensor | None = None

    def fit(self, values: Tensor) -> "FeatureStandardizer":
        self.mean_ = values.mean(dim=0)
        self.scale_ = values.std(dim=0, unbiased=False).clamp_min(self.epsilon)
        return self

    def transform(self, values: Tensor) -> Tensor:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before transform")
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values: Tensor) -> Tensor:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before inverse_transform")
        return values * self.scale_ + self.mean_


def build_sliding_features(
    seismic: Tensor | np.ndarray, pore_volume: Tensor | np.ndarray, window: int
) -> Tensor:
    """Build per-cell features without requiring targets (training or field use)."""
    seismic = torch.as_tensor(seismic, dtype=torch.float32)
    pore_volume = torch.as_tensor(pore_volume, dtype=torch.float32)
    if seismic.ndim != 4 or pore_volume.ndim != 2:
        raise ValueError("expected seismic [T,H,W,A] and pore_volume [H,W]")
    if seismic.shape[1:3] != pore_volume.shape:
        raise ValueError("spatial dimensions must agree")
    if not 1 <= window <= seismic.shape[0]:
        raise ValueError("window must be between 1 and the number of vintages")
    examples = []
    static = pore_volume.reshape(-1, 1)
    for end in range(window - 1, seismic.shape[0]):
        dynamic = seismic[end - window + 1 : end + 1].permute(1, 2, 0, 3)
        dynamic = dynamic.reshape(-1, window * seismic.shape[-1])
        examples.append(torch.cat((dynamic, static), dim=-1))
    return torch.cat(examples)


class SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Flatten temporal windows of map features into per-cell training samples.

    ``seismic`` has shape [vintage, rows, cols, attribute], conventionally
    near/mid/far/gradient. ``pore_volume`` is a static [rows, cols] map and is
    appended once. Targets have shape [vintage, rows, cols, 3]. A window ending
    at vintage t predicts the properties at t.
    """

    def __init__(self, seismic: Tensor | np.ndarray, pore_volume: Tensor | np.ndarray, targets: Tensor | np.ndarray, window: int) -> None:
        seismic = torch.as_tensor(seismic, dtype=torch.float32)
        pore_volume = torch.as_tensor(pore_volume, dtype=torch.float32)
        targets = torch.as_tensor(targets, dtype=torch.float32)
        if seismic.ndim != 4 or targets.ndim != 4 or pore_volume.ndim != 2:
            raise ValueError("expected seismic [T,H,W,A], pore_volume [H,W], targets [T,H,W,P]")
        if seismic.shape[:3] != targets.shape[:3] or seismic.shape[1:3] != pore_volume.shape:
            raise ValueError("vintage and spatial dimensions must agree")
        if not 1 <= window <= seismic.shape[0]:
            raise ValueError("window must be between 1 and the number of vintages")
        labels = []
        for end in range(window - 1, seismic.shape[0]):
            labels.append(targets[end].reshape(-1, targets.shape[-1]))
        self.features = build_sliding_features(seismic, pore_volume, window)
        self.targets = torch.cat(labels)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.features[index], self.targets[index]
