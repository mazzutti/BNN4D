from __future__ import annotations

import torch
from torch import Tensor


def normalized_rmse(prediction: Tensor, target: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Per-property RMSE normalized by the observed target range."""
    rmse = torch.sqrt(torch.mean((prediction - target).square(), dim=0))
    scale = (target.amax(dim=0) - target.amin(dim=0)).clamp_min(epsilon)
    return rmse / scale

