"""Publication-style visualizations for training and 4D inversion results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

PROPERTY_NAMES = ("ΔP", "ΔSw", "ΔSg")


def _save(fig: Figure, output: str | Path | None) -> Figure:
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_training_history(
    history: Sequence[float] | Mapping[str, Sequence[float]],
    *,
    log_y: bool = False,
    output: str | Path | None = None,
) -> Figure:
    """Plot one loss sequence or several named training/validation metrics."""
    series = {"loss": history} if not isinstance(history, Mapping) else history
    if not series or any(len(values) == 0 for values in series.values()):
        raise ValueError("history must contain at least one value per series")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for name, values in series.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=name, linewidth=1.8)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training history")
    if log_y:
        ax.set_yscale("log")
    if len(series) > 1:
        ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, output)


def plot_property_maps(
    mean: np.ndarray,
    uncertainty: np.ndarray,
    *,
    truth: np.ndarray | None = None,
    vintage: int = -1,
    property_names: Sequence[str] = PROPERTY_NAMES,
    output: str | Path | None = None,
) -> Figure:
    """Plot predicted property maps, uncertainty and optional ground truth.

    Arrays may be ``[T,H,W,P]`` or a single vintage ``[H,W,P]``.
    Prediction/truth colors use the same limits for direct comparison.
    """
    mean = np.asarray(mean)
    uncertainty = np.asarray(uncertainty)
    if mean.shape != uncertainty.shape or mean.ndim not in (3, 4):
        raise ValueError("mean and uncertainty must share shape [T,H,W,P] or [H,W,P]")
    if mean.ndim == 4:
        mean, uncertainty = mean[vintage], uncertainty[vintage]
    if truth is not None:
        truth = np.asarray(truth)
        if truth.ndim == 4:
            truth = truth[vintage]
        if truth.shape != mean.shape:
            raise ValueError("truth must match the selected prediction map")
    properties = mean.shape[-1]
    if len(property_names) != properties:
        raise ValueError("property_names length must match the property dimension")
    row_names = (["Ground truth"] if truth is not None else []) + ["Prediction", "Uncertainty"]
    fig, axes = plt.subplots(len(row_names), properties, figsize=(4 * properties, 3.3 * len(row_names)), squeeze=False, constrained_layout=True)
    for column, name in enumerate(property_names):
        reference = mean[..., column] if truth is None else np.concatenate((mean[..., column].ravel(), truth[..., column].ravel()))
        limit = max(float(np.nanmax(np.abs(reference))), np.finfo(float).eps)
        row = 0
        if truth is not None:
            image = axes[row, column].imshow(truth[..., column], cmap="RdBu_r", vmin=-limit, vmax=limit)
            fig.colorbar(image, ax=axes[row, column], shrink=0.8)
            row += 1
        image = axes[row, column].imshow(mean[..., column], cmap="RdBu_r", vmin=-limit, vmax=limit)
        fig.colorbar(image, ax=axes[row, column], shrink=0.8)
        image = axes[row + 1, column].imshow(uncertainty[..., column], cmap="magma", vmin=0)
        fig.colorbar(image, ax=axes[row + 1, column], shrink=0.8)
        for row_index, row_name in enumerate(row_names):
            axes[row_index, column].set_title(f"{row_name}: {name}")
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
    return _save(fig, output)


def plot_prediction_diagnostics(
    mean: np.ndarray,
    uncertainty: np.ndarray,
    truth: np.ndarray,
    *,
    property_names: Sequence[str] = PROPERTY_NAMES,
    output: str | Path | None = None,
) -> Figure:
    """Plot parity, standardized residuals and empirical interval coverage."""
    mean, uncertainty, truth = map(np.asarray, (mean, uncertainty, truth))
    if mean.shape != uncertainty.shape or mean.shape != truth.shape or mean.shape[-1] != len(property_names):
        raise ValueError("mean, uncertainty and truth must have equal shapes and matching properties")
    properties = mean.shape[-1]
    fig, axes = plt.subplots(3, properties, figsize=(4 * properties, 10), squeeze=False, constrained_layout=True)
    levels = np.linspace(0.25, 3.0, 12)
    normal_coverage = np.array([0.19741265, 0.38292492, 0.5467453, 0.68268949, 0.78870045, 0.8663856, 0.91988169, 0.95449974, 0.97555105, 0.98758067, 0.99379033, 0.9973002])
    for index, name in enumerate(property_names):
        predicted = mean[..., index].ravel()
        observed = truth[..., index].ravel()
        sigma = np.maximum(uncertainty[..., index].ravel(), np.finfo(np.float32).eps)
        valid = np.isfinite(predicted) & np.isfinite(observed) & np.isfinite(sigma)
        predicted, observed, sigma = predicted[valid], observed[valid], sigma[valid]
        low, high = min(predicted.min(), observed.min()), max(predicted.max(), observed.max())
        axes[0, index].scatter(observed, predicted, s=5, alpha=0.25)
        axes[0, index].plot([low, high], [low, high], "k--", linewidth=1)
        axes[0, index].set(title=f"Parity: {name}", xlabel="Observed", ylabel="Predicted")
        residual = (observed - predicted) / sigma
        residual_range = None
        if np.isclose(np.ptp(residual), 0.0, rtol=1e-7, atol=1e-12):
            residual_range = (float(residual[0]) - 0.5, float(residual[0]) + 0.5)
        axes[1, index].hist(residual, bins=40, range=residual_range, density=True, alpha=0.8)
        axes[1, index].set(title=f"Standardized residual: {name}", xlabel="(observed − predicted) / σ", ylabel="Density")
        empirical = [np.mean(np.abs(residual) <= level) for level in levels]
        axes[2, index].plot(normal_coverage, empirical, marker="o", label="Model")
        axes[2, index].plot([0, 1], [0, 1], "k--", label="Ideal")
        axes[2, index].set(title=f"Coverage: {name}", xlabel="Gaussian nominal coverage", ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1))
        axes[2, index].legend()
        for row in range(3):
            axes[row, index].grid(alpha=0.2)
    return _save(fig, output)
