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


CANONICAL_UNISIM_WELLS: list[tuple[int, int]] = [
    # 21 Poços Exatos do Paper CAGEO (De Figueiredo et al. / UNISIM-I): (J, I)
    (181, 59),   # PROD09
    (112, 75),   # PROD08
    (118, 95),   # NJ21
    (185, 107),  # PROD21
    (185, 115),  # NJ19
    (84, 125),   # INJ05
    (82, 130),   # PROD05
    (142, 132),  # NJ17
    (117, 143),  # NJ15
    (188, 143),  # PROD10
    (57, 167),   # NJ23
    (94, 180),   # PROD12
    (101, 187),  # INJ03
    (25, 190),   # NJ22
    (142, 190),  # NJ06
    (76, 197),   # PROD14
    (127, 220),  # NJ10
    (98, 230),   # PROD25A
    (55, 235),   # NJ07
    (145, 245),  # PROD24A
    (98, 260),   # PROD23A
    # 5 Poços Adicionais do Modelo Completo UNISIM-I (Total 26 poços)
    (150, 45),   # INJ01
    (190, 60),   # INJ02
    (130, 115),  # INJ04
    (200, 130),  # INJ06
    (135, 205),  # INJ07
]


def select_subset_traces(
    total_samples: int,
    mask: np.ndarray | Tensor | None = None,
    n_traces: int = 26,
    method: str = "unisim_wells",
    seed: int = 42,
) -> np.ndarray:
    """Select a subset of spatial traces (pseudowells) for training.

    Methods:
    - ``unisim_wells``: Select from the 26 canonical UNISIM-I well positions (default).
      If n_traces < 26, selects the maximally dispersed subset via FPS.
    - ``spatial_optimal``: Farthest Point Sampling (FPS) for maximal spatial dispersion across entire field.
    - ``random``: Uniform random sampling over the active mask.
    """
    if n_traces > total_samples:
        raise ValueError(f"n_traces ({n_traces}) cannot exceed total_samples ({total_samples})")
    if n_traces <= 0:
        raise ValueError("n_traces must be positive")

    if mask is not None:
        mask_np = np.asarray(mask, dtype=bool)
        coords_i, coords_j = np.where(mask_np)
        coords = np.column_stack([coords_i, coords_j])
    else:
        coords = None

    if method == "unisim_wells":
        if coords is None:
            rng = np.random.RandomState(seed)
            return rng.choice(total_samples, size=n_traces, replace=False).astype(np.int64)
        
        # Map all 26 canonical wells to active cell indices
        canonical_indices = []
        for wi, wj in CANONICAL_UNISIM_WELLS:
            dist_sq = (coords[:, 0] - wi) ** 2 + (coords[:, 1] - wj) ** 2
            nearest = int(np.argmin(dist_sq))
            if nearest not in canonical_indices:
                canonical_indices.append(nearest)
        canonical_indices = np.array(canonical_indices, dtype=np.int64)

        if n_traces == len(canonical_indices):
            return canonical_indices
        elif n_traces < len(canonical_indices):
            # Apply FPS over the 26 canonical wells to guarantee optimal spatial dispersion
            well_coords = coords[canonical_indices]
            rng = np.random.RandomState(seed)
            selected_sub = [rng.randint(0, len(well_coords))]
            dists = np.sum((well_coords - well_coords[selected_sub[0]]) ** 2, axis=1)
            for _ in range(1, n_traces):
                next_w = int(np.argmax(dists))
                selected_sub.append(next_w)
                new_dists = np.sum((well_coords - well_coords[next_w]) ** 2, axis=1)
                dists = np.minimum(dists, new_dists)
            return canonical_indices[selected_sub]
        else:
            # n_traces > 26: take all 26 wells and fill remaining via FPS over active cells
            selected = list(canonical_indices)
            remaining = np.setdiff1d(np.arange(len(coords)), selected)
            dists = np.min([np.sum((coords[remaining] - coords[s]) ** 2, axis=1) for s in selected], axis=0)
            while len(selected) < n_traces:
                next_rem = int(np.argmax(dists))
                next_idx = int(remaining[next_rem])
                selected.append(next_idx)
                new_dists = np.sum((coords[remaining] - coords[next_idx]) ** 2, axis=1)
                dists = np.minimum(dists, new_dists)
            return np.array(selected, dtype=np.int64)

    elif method == "spatial_optimal":
        if coords is None:
            rng = np.random.RandomState(seed)
            return rng.choice(total_samples, size=n_traces, replace=False).astype(np.int64)
        rng = np.random.RandomState(seed)
        n = len(coords)
        selected = [rng.randint(0, n)]
        dists = np.sum((coords - coords[selected[0]]) ** 2, axis=1)
        for _ in range(1, n_traces):
            next_idx = int(np.argmax(dists))
            selected.append(next_idx)
            new_dists = np.sum((coords - coords[next_idx]) ** 2, axis=1)
            dists = np.minimum(dists, new_dists)
        return np.array(selected, dtype=np.int64)

    elif method == "random":
        rng = np.random.RandomState(seed)
        return rng.choice(total_samples, size=n_traces, replace=False).astype(np.int64)

    else:
        raise ValueError(f"Unknown trace selection method: {method}. Choose from 'unisim_wells', 'spatial_optimal', 'random'")


def build_sliding_features(
    seismic: Tensor | np.ndarray,
    temporal_window: Tensor | np.ndarray | None = None,
    window: int = 2,
    mask: Tensor | np.ndarray | None = None,
    include_deltas: bool = True,
    include_relative_deltas: bool = False,
    include_scalar: bool = True,
    time_shift: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Build per-cell features without requiring targets (training or field use)."""
    seismic = torch.as_tensor(seismic, dtype=torch.float32)
    if seismic.ndim != 4:
        raise ValueError("expected seismic [T,H,W,A]")

    if not 1 <= window <= seismic.shape[0]:
        raise ValueError("window must be between 1 and the number of vintages")

    tw_tensor: Tensor | None = None
    if temporal_window is not None:
        tw_tensor = torch.as_tensor(temporal_window, dtype=torch.float32)
        if tw_tensor.ndim != 4:
            raise ValueError("expected temporal_window [T,H,W,TW_CHANNELS]")
        if tw_tensor.shape[1:3] != seismic.shape[1:3]:
            raise ValueError("temporal_window spatial dimensions must agree with seismic")

    ts_tensor: Tensor | None = None
    if time_shift is not None:
        ts_tensor = torch.as_tensor(time_shift, dtype=torch.float32)
        if ts_tensor.ndim == 2:
            ts_tensor = ts_tensor.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
        elif ts_tensor.ndim == 3:
            if ts_tensor.shape[0] == seismic.shape[0]:
                ts_tensor = ts_tensor.unsqueeze(-1)  # [T, H, W, 1]
            else:
                ts_tensor = ts_tensor.unsqueeze(0)  # [1, H, W, S]
        if ts_tensor.shape[1:3] != seismic.shape[1:3]:
            raise ValueError("time_shift spatial dimensions must agree with seismic")

    mask_flat = torch.as_tensor(mask, dtype=torch.bool).reshape(-1) if mask is not None else None
    examples = []
    for end in range(window - 1, seismic.shape[0]):
        parts = []
        if include_scalar:
            # Dynamic window attributes (scalar summary)
            win_seis = seismic[end - window + 1 : end + 1]  # [W, H, W, A]
            dynamic = win_seis.permute(1, 2, 0, 3).reshape(-1, window * seismic.shape[-1])
            parts.append(dynamic)
            if include_deltas and window >= 2:
                # Explicit 4D differential attributes (monitor - baseline)
                delta = (win_seis[-1] - win_seis[0]).reshape(-1, seismic.shape[-1])
                parts.append(delta)
            if include_relative_deltas and window >= 2:
                # Explicit normalized relative 4D differential: (monitor - baseline) / (|baseline| + 1e-4)
                rel_delta = ((win_seis[-1] - win_seis[0]) / (torch.abs(win_seis[0]) + 1e-4)).reshape(-1, seismic.shape[-1])
                parts.append(rel_delta)

        # Dynamic 1D Stratal Temporal Waveform features
        if tw_tensor is not None:
            if tw_tensor.shape[0] >= window:
                win_tw = tw_tensor[end - window + 1 : end + 1]
                # Include base, monitor, explicit delta, and relative delta for stratal temporal window
                tw_base = win_tw[0].reshape(-1, tw_tensor.shape[-1])
                tw_mon = win_tw[-1].reshape(-1, tw_tensor.shape[-1])
                parts.extend([tw_base, tw_mon])
                if include_deltas and window >= 2:
                    delta_tw = (win_tw[-1] - win_tw[0]).reshape(-1, tw_tensor.shape[-1])
                    parts.append(delta_tw)
                if include_relative_deltas and window >= 2:
                    rel_delta_tw = ((win_tw[-1] - win_tw[0]) / (torch.abs(win_tw[0]) + 1e-4)).reshape(-1, tw_tensor.shape[-1])
                    parts.append(rel_delta_tw)
            else:
                tw_flat = tw_tensor[0].reshape(-1, tw_tensor.shape[-1])
                parts.append(tw_flat)

        if ts_tensor is not None:
            if ts_tensor.shape[0] >= window:
                win_ts = ts_tensor[end - window + 1 : end + 1]
                ts_flat = win_ts.permute(1, 2, 0, 3).reshape(-1, window * ts_tensor.shape[-1])
            else:
                ts_flat = ts_tensor[0].reshape(-1, ts_tensor.shape[-1])
            parts.append(ts_flat)
        if not parts:
            raise ValueError("no features selected: include_scalar is False and neither temporal_window nor time_shift was provided")
        cell_features = torch.cat(parts, dim=-1)
        if mask_flat is not None:
            cell_features = cell_features[mask_flat]
        examples.append(cell_features)
    return torch.cat(examples)


class SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Flatten temporal windows of map features into per-cell training samples.

    ``seismic`` has shape [vintage, rows, cols, attribute].
    ``temporal_window`` has shape [vintage, rows, cols, tw_channels] (optional).
    Targets have shape [vintage, rows, cols, 3]. A window ending at vintage t
    predicts the properties at t.
    """

    def __init__(
        self,
        seismic: Tensor | np.ndarray,
        temporal_window: Tensor | np.ndarray | None = None,
        targets: Tensor | np.ndarray | None = None,
        window: int = 2,
        mask: Tensor | np.ndarray | None = None,
        include_deltas: bool = True,
        include_relative_deltas: bool = False,
        include_scalar: bool = True,
        time_shift: Tensor | np.ndarray | None = None,
    ) -> None:
        seismic = torch.as_tensor(seismic, dtype=torch.float32)
        if targets is not None:
            targets = torch.as_tensor(targets, dtype=torch.float32)
            if seismic.ndim != 4 or targets.ndim != 4:
                raise ValueError("expected seismic [T,H,W,A], targets [T,H,W,P]")
            if seismic.shape[:3] != targets.shape[:3]:
                raise ValueError("vintage and spatial dimensions must agree")
        if not 1 <= window <= seismic.shape[0]:
            raise ValueError("window must be between 1 and the number of vintages")
        mask_flat = torch.as_tensor(mask, dtype=torch.bool).reshape(-1) if mask is not None else None
        labels = []
        if targets is not None:
            for end in range(window - 1, seismic.shape[0]):
                end_labels = targets[end].reshape(-1, targets.shape[-1])
                if mask_flat is not None:
                    end_labels = end_labels[mask_flat]
                labels.append(end_labels)
            self.targets = torch.cat(labels)
        else:
            self.targets = torch.empty((0,), dtype=torch.float32)
        self.features = build_sliding_features(
            seismic,
            temporal_window=temporal_window,
            window=window,
            mask=mask,
            include_deltas=include_deltas,
            include_relative_deltas=include_relative_deltas,
            include_scalar=include_scalar,
            time_shift=time_shift,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if len(self.targets) == 0:
            return self.features[index], torch.empty((0,), dtype=torch.float32)
        return self.features[index], self.targets[index]
