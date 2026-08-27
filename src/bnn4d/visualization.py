"""Publication-style visualizations for training and 4D inversion results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

PROPERTY_NAMES = ("ΔP", "ΔSw", "ΔSg")

# Exact rainbow-spectrum colormap from EAGE paper (Magenta -> Blue -> Cyan -> Green -> Yellow -> Orange -> Red)
EAGE_CMAP = LinearSegmentedColormap.from_list(
    "eage_rainbow",
    [
        (0.00, "#ff00ff"),  # Magenta
        (0.10, "#8000ff"),  # Violet
        (0.20, "#0000ff"),  # Blue
        (0.32, "#0099ff"),  # Sky Blue
        (0.40, "#00ffff"),  # Cyan
        (0.50, "#00ff80"),  # Spring green
        (0.60, "#00ff00"),  # Green
        (0.72, "#b3ff00"),  # Yellow-Green
        (0.80, "#ffff00"),  # Yellow
        (0.90, "#ff8000"),  # Orange
        (1.00, "#ff0000"),  # Red
    ],
    N=256,
)


def _save(fig: Figure, output: str | Path | None) -> Figure:
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
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
        label = "Training Loss" if name == "train_loss" else ("Validation Loss" if name == "val_loss" else name)
        style = "--" if name == "train_loss" else "-"
        width = 1.0 if name == "train_loss" else 1.2
        ax.plot(np.arange(1, len(values) + 1), values, label=label, linestyle=style, linewidth=width)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training and Validation Loss History")
    if log_y:
        ax.set_yscale("log")
    if len(series) > 1:
        ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, output)


def plot_cv_history(
    fold_histories: Sequence[Mapping[str, Sequence[float]]],
    *,
    log_y: bool = False,
    output: str | Path | None = None,
) -> Figure:
    """Plot training and validation loss curves across all CV folds."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    n_folds = len(fold_histories)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_folds, 10)))

    for i, hist in enumerate(fold_histories):
        t_loss = hist.get("train_loss", [])
        v_loss = hist.get("val_loss", [])
        c = colors[i % len(colors)]
        if len(t_loss) > 0:
            ax.plot(np.arange(1, len(t_loss) + 1), t_loss, linestyle="--", color=c, alpha=0.55, linewidth=1.0, label=f"Fold {i+1} Train" if n_folds <= 5 else None)
        if len(v_loss) > 0:
            ax.plot(np.arange(1, len(v_loss) + 1), v_loss, linestyle="-", color=c, alpha=0.90, linewidth=1.2, label=f"Fold {i+1} Val" if n_folds <= 5 else None)

    ax.set(xlabel="Epoch", ylabel="Loss", title=f"{n_folds}-Fold Cross-Validation: Training & Validation Loss")
    if log_y:
        ax.set_yscale("log")
    else:
        # Robust y-limits to prevent single-epoch initialization spikes from compressing convergence
        losses_to_scale = []
        for hist in fold_histories:
            t = hist.get("train_loss", [])
            v = hist.get("val_loss", [])
            if len(t) > 1:
                losses_to_scale.extend(t[1:])
            elif len(t) > 0:
                losses_to_scale.extend(t)
            if len(v) > 0:
                losses_to_scale.extend(v)
        if len(losses_to_scale) > 0:
            ymin = float(np.nanmin(losses_to_scale))
            ymax = float(np.nanpercentile(losses_to_scale, 98))
            margin = max(0.1 * (ymax - ymin), 0.05)
            ax.set_ylim(ymin - margin, ymax + margin)

    ax.grid(alpha=0.25)
    if n_folds <= 5:
        ax.legend(ncol=2, fontsize=8)
    return _save(fig, output)


def plot_property_maps(
    mean: np.ndarray,
    uncertainty: np.ndarray,
    *,
    truth: np.ndarray | None = None,
    vintage: int = -1,
    property_names: Sequence[str] = PROPERTY_NAMES,
    mask: np.ndarray | None = None,
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
        property_names = tuple(str(p) for p in property_names)[:properties]
        if len(property_names) < properties:
            property_names = tuple(list(property_names) + [f"P{i}" for i in range(len(property_names), properties)])
    
    # Optional mask display
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        mean_disp = np.where(mask_arr[..., None], mean, np.nan)
        unc_disp = np.where(mask_arr[..., None], uncertainty, np.nan)
        truth_disp = np.where(mask_arr[..., None], truth, np.nan) if truth is not None else None
    else:
        mean_disp, unc_disp, truth_disp = mean, uncertainty, truth

    row_names = (["Ground truth"] if truth is not None else []) + ["Prediction", "Uncertainty"]
    fig, axes = plt.subplots(len(row_names), properties, figsize=(4.5 * properties, 3.8 * len(row_names)), squeeze=False, constrained_layout=True)
    for column, name in enumerate(property_names):
        if truth_disp is not None:
            valid_truth = truth_disp[..., column][np.isfinite(truth_disp[..., column])]
            limit = max(float(np.nanpercentile(np.abs(valid_truth), 99.5)) if len(valid_truth) > 0 else 1.0, 1e-6)
        else:
            valid_ref = mean_disp[..., column][np.isfinite(mean_disp[..., column])]
            limit = max(float(np.nanpercentile(np.abs(valid_ref), 99.5)) if len(valid_ref) > 0 else 1.0, 1e-6)
        row = 0
        if truth_disp is not None:
            image = axes[row, column].imshow(truth_disp[..., column], cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
            fig.colorbar(image, ax=axes[row, column], shrink=0.8)
            row += 1
        image = axes[row, column].imshow(mean_disp[..., column], cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
        fig.colorbar(image, ax=axes[row, column], shrink=0.8)
        
        valid_unc = unc_disp[..., column][np.isfinite(unc_disp[..., column])]
        unc_max = max(float(np.nanpercentile(valid_unc, 99.5)) if len(valid_unc) > 0 else 1.0, 1e-6)
        image = axes[row + 1, column].imshow(unc_disp[..., column], cmap="magma", vmin=0, vmax=unc_max, origin="lower")
        fig.colorbar(image, ax=axes[row + 1, column], shrink=0.8)
        for row_index, row_name in enumerate(row_names):
            axes[row_index, column].set_title(f"{row_name}: {name}", fontsize=11, fontweight="bold")
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
    return _save(fig, output)


def plot_prediction_diagnostics(
    mean: np.ndarray,
    uncertainty: np.ndarray,
    truth: np.ndarray,
    *,
    property_names: Sequence[str] = PROPERTY_NAMES,
    mask: np.ndarray | None = None,
    output: str | Path | None = None,
) -> Figure:
    """Plot parity, standardized residuals and empirical interval coverage."""
    mean, uncertainty, truth = map(np.asarray, (mean, uncertainty, truth))
    if mean.ndim == 4:
        mean, uncertainty, truth = mean[-1], uncertainty[-1], truth[-1]
    if mean.shape != uncertainty.shape or mean.shape != truth.shape:
        raise ValueError("mean, uncertainty and truth must have equal shapes")
    properties = mean.shape[-1]
    if len(property_names) != properties:
        property_names = tuple(str(p) for p in property_names)[:properties]
    
    fig, axes = plt.subplots(3, properties, figsize=(4.5 * properties, 10), squeeze=False, constrained_layout=True)
    levels = np.linspace(0.25, 3.0, 12)
    normal_coverage = np.array([0.19741265, 0.38292492, 0.5467453, 0.68268949, 0.78870045, 0.8663856, 0.91988169, 0.95449974, 0.97555105, 0.98758067, 0.99379033, 0.9973002])
    
    mask_flat = mask.ravel() if mask is not None else np.ones(mean[..., 0].size, dtype=bool)

    for index, name in enumerate(property_names):
        predicted = mean[..., index].ravel()[mask_flat]
        observed = truth[..., index].ravel()[mask_flat]
        sigma = np.maximum(uncertainty[..., index].ravel()[mask_flat], np.finfo(np.float32).eps)
        valid = np.isfinite(predicted) & np.isfinite(observed) & np.isfinite(sigma)
        predicted, observed, sigma = predicted[valid], observed[valid], sigma[valid]
        
        low, high = min(predicted.min(), observed.min()), max(predicted.max(), observed.max())
        axes[0, index].scatter(observed, predicted, s=4, alpha=0.3, edgecolors="none")
        axes[0, index].plot([low, high], [low, high], "r--", linewidth=1.5)
        corr = np.corrcoef(observed, predicted)[0, 1] if len(observed) > 1 else 0.0
        axes[0, index].set(title=f"Parity: {name} (R={corr:.3f})", xlabel="Observed Ground Truth", ylabel="Predicted")
        
        residual = (observed - predicted) / sigma
        residual_range = (-4, 4)
        axes[1, index].hist(residual, bins=40, range=residual_range, density=True, alpha=0.75, color="steelblue")
        # Standard normal curve
        x_norm = np.linspace(-4, 4, 100)
        axes[1, index].plot(x_norm, 1/np.sqrt(2*np.pi)*np.exp(-0.5*x_norm**2), "r--", label="Standard Normal")
        axes[1, index].set(title=f"Standardized residual: {name}", xlabel="(observed − predicted) / σ", ylabel="Density")
        axes[1, index].legend(fontsize=8)
        
        empirical = [np.mean(np.abs(residual) <= level) for level in levels]
        axes[2, index].plot(normal_coverage, empirical, marker="o", markersize=4, label="BNN Model", color="crimson")
        axes[2, index].plot([0, 1], [0, 1], "k--", label="Ideal")
        axes[2, index].set(title=f"Coverage: {name}", xlabel="Gaussian nominal coverage", ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1))
        axes[2, index].legend(fontsize=8)
        for row in range(3):
            axes[row, index].grid(alpha=0.25)
    return _save(fig, output)


def plot_eage_saturation_vp_comparison(
    predicted_dvp: np.ndarray,
    reference_dsw: np.ndarray,
    reference_dvp: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    output: str | Path | None = None,
) -> Figure:
    """Recreate the EAGE Figure 2 comparison: predicted ΔVP vs reference water saturation ΔSw."""
    if predicted_dvp.ndim == 3:
        predicted_dvp = predicted_dvp[-1]
    if reference_dsw.ndim == 3:
        reference_dsw = reference_dsw[-1]
    if reference_dvp is not None and reference_dvp.ndim == 3:
        reference_dvp = reference_dvp[-1]

    mask_arr = mask if mask is not None else np.ones(predicted_dvp.shape, dtype=bool)
    
    pred_disp = np.where(mask_arr, predicted_dvp, np.nan)
    dsw_disp = np.where(mask_arr, reference_dsw, np.nan)
    ref_vp_disp = np.where(mask_arr, reference_dvp, np.nan) if reference_dvp is not None else None
    
    fig, axes = plt.subplots(1, 3 if reference_dvp is not None else 2, figsize=(15 if reference_dvp is not None else 10.5, 4.5), constrained_layout=True)
    
    # EAGE Figure 2 saturation variation scale: -0.2 to 0.8
    sw_vmin = -0.2 if np.nanmin(dsw_disp) >= -0.25 else float(np.nanmin(dsw_disp))
    sw_vmax = 0.8 if np.nanmax(dsw_disp) <= 0.85 else float(np.nanmax(dsw_disp))
    im0 = axes[0].imshow(dsw_disp, cmap=EAGE_CMAP, vmin=sw_vmin, vmax=sw_vmax, origin="lower")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="ΔSw")
    axes[0].set_title("Water Saturation Variation (ΔSw)", fontsize=11, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # EAGE Figure 2 P-wave velocity variation scale: -20 to 180 m/s
    vp_max_val = max(float(np.nanmax(pred_disp)), float(np.nanmax(ref_vp_disp)) if ref_vp_disp is not None else 0.0, 100.0)
    vp_vmax = 180.0 if vp_max_val <= 185.0 else vp_max_val
    im1 = axes[1].imshow(pred_disp, cmap=EAGE_CMAP, vmin=-20, vmax=vp_vmax, origin="lower")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="m/s")
    axes[1].set_title("BNN Inverted ΔVP (P-Velocity Change)", fontsize=11, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    if ref_vp_disp is not None:
        im2 = axes[2].imshow(ref_vp_disp, cmap=EAGE_CMAP, vmin=-20, vmax=vp_vmax, origin="lower")
        fig.colorbar(im2, ax=axes[2], shrink=0.8, label="m/s")
        axes[2].set_title("Reference True ΔVP", fontsize=11, fontweight="bold")
        axes[2].set_xticks([])
        axes[2].set_yticks([])

    return _save(fig, output)


def plot_error_maps(
    mean: np.ndarray,
    truth: np.ndarray,
    *,
    uncertainty: np.ndarray | None = None,
    vintage: int = -1,
    property_names: Sequence[str] = PROPERTY_NAMES,
    mask: np.ndarray | None = None,
    output: str | Path | None = None,
) -> Figure:
    """Plot spatial residual error maps (Signed Error, Absolute Error, and Standardized Error)."""
    mean = np.asarray(mean)
    truth = np.asarray(truth)
    if mean.ndim == 4:
        mean, truth = mean[vintage], truth[vintage]
        if uncertainty is not None:
            uncertainty = np.asarray(uncertainty)[vintage]
    elif uncertainty is not None:
        uncertainty = np.asarray(uncertainty)

    if mean.shape != truth.shape:
        raise ValueError("mean and truth must have matching shapes")

    properties = mean.shape[-1]
    if len(property_names) != properties:
        property_names = tuple(str(p) for p in property_names)[:properties]

    residuals = truth - mean
    abs_errors = np.abs(residuals)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        res_disp = np.where(mask_arr[..., None], residuals, np.nan)
        abs_disp = np.where(mask_arr[..., None], abs_errors, np.nan)
        unc_disp = np.where(mask_arr[..., None], uncertainty, np.nan) if uncertainty is not None else None
    else:
        res_disp = residuals
        abs_disp = abs_errors
        unc_disp = uncertainty

    has_unc = unc_disp is not None
    row_names = ["Signed Residual (Truth - Pred)", "Absolute Error |Residual|"] + (["Standardized Error (|Residual| / σ)"] if has_unc else [])
    num_rows = len(row_names)

    fig, axes = plt.subplots(num_rows, properties, figsize=(4.5 * properties, 3.8 * num_rows), squeeze=False, constrained_layout=True)

    for col, name in enumerate(property_names):
        # Row 0: Signed Residual
        valid_res = res_disp[..., col][np.isfinite(res_disp[..., col])]
        limit = max(float(np.nanpercentile(np.abs(valid_res), 99)) if len(valid_res) > 0 else 1.0, 1e-6)
        im0 = axes[0, col].imshow(res_disp[..., col], cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower")
        fig.colorbar(im0, ax=axes[0, col], shrink=0.8)
        axes[0, col].set_title(f"Residual (Truth - Pred): {name}", fontsize=11, fontweight="bold")

        # Row 1: Absolute Error
        valid_abs = abs_disp[..., col][np.isfinite(abs_disp[..., col])]
        max_abs = max(float(np.nanpercentile(valid_abs, 99)) if len(valid_abs) > 0 else 1.0, 1e-6)
        im1 = axes[1, col].imshow(abs_disp[..., col], cmap="inferno", vmin=0, vmax=max_abs, origin="lower")
        fig.colorbar(im1, ax=axes[1, col], shrink=0.8)
        axes[1, col].set_title(f"Absolute Error: {name}", fontsize=11, fontweight="bold")

        # Row 2: Standardized Error (if uncertainty present)
        if has_unc:
            std_err = abs_disp[..., col] / np.maximum(unc_disp[..., col], 1e-6)
            valid_std = std_err[np.isfinite(std_err)]
            max_std = max(float(np.nanpercentile(valid_std, 99)) if len(valid_std) > 0 else 3.0, 1e-6)
            im2 = axes[2, col].imshow(std_err, cmap="viridis", vmin=0, vmax=max(max_std, 3.0), origin="lower")
            fig.colorbar(im2, ax=axes[2, col], shrink=0.8)
            axes[2, col].set_title(f"Standardized Error (|Err|/σ): {name}", fontsize=11, fontweight="bold")

        for r in range(num_rows):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    return _save(fig, output)


def plot_4configs_comparison(
    experiment_dirs: Mapping[str, Path | str],
    truth: np.ndarray,
    property_names: Sequence[str] = ("ΔVP", "ΔSw", "Δρ"),
    mask: np.ndarray | None = None,
    output_metrics: str | Path | None = None,
    output_maps: str | Path | None = None,
) -> tuple[Figure, Figure]:
    """Render publication-grade comparative metrics and spatial maps across the 4 ablation configurations, annotating exact inputs and outputs."""
    config_labels = list(experiment_dirs.keys())
    metrics_data = {"R": {name: [] for name in property_names}, "NRMSE": {name: [] for name in property_names}}
    loaded_preds = {}
    loaded_uncs = {}

    # Define input descriptions for each standard configuration
    input_specs = {
        "1. Scalar Slices / No TS": "INPUT: 32 Summary Features (Base, Mon, ΔA, Rel ΔA across 4 angles)",
        "2. Temporal Window 1D Only / No TS": "INPUT: 4 Features (4 Principal Orthogonal 1D Waveform Modes)",
        "3. Scalar + Temporal 1D / No TS": "INPUT: 36 Features (32 Summary + 4 Principal 1D Waveform Modes)",
        "4. Scalar Slices / With TS": "INPUT: 36 Features (32 Summary + 4 Seismic 4D Time-Shift Maps dt)",
        "5. Scalar + Temporal 1D / With TS": "INPUT: 40 Features (32 Summary + 4 Waveform Modes + 4 dt)",
        "1. No Static / No TS": "INPUT: 32 Summary Features (Base, Mon, ΔA, Rel ΔA across 4 angles)",
        "2. With Static / No TS": "INPUT: 36 Features (32 Summary + 4 Principal 1D Waveform Modes)",
        "3. No Static / With TS": "INPUT: 36 Features (32 Summary + 4 Time-Shift dt)",
        "4. With Static / With TS": "INPUT: 40 Features (32 Summary + 4 Waveform Modes + 4 dt)",
    }

    for label, exp_dir in experiment_dirs.items():
        exp_path = Path(exp_dir)
        pred_file = exp_path / "oof_predictions.npz"
        if not pred_file.exists():
            pred_file = exp_path / "predictions.npz"
        if not pred_file.exists():
            continue
        data = np.load(pred_file)
        mean = data["mean"][-1] if data["mean"].ndim == 4 else data["mean"]
        unc = data["uncertainty"][-1] if data["uncertainty"].ndim == 4 else data["uncertainty"]
        loaded_preds[label] = mean
        loaded_uncs[label] = unc

        # Compute metrics
        m_flat = mask.ravel() if mask is not None else np.ones(mean.shape[0] * mean.shape[1], dtype=bool)
        t_eval = truth[-1] if truth.ndim == 4 else truth
        p_eval = mean

        for i, name in enumerate(property_names):
            p_i = p_eval[..., i].ravel()[m_flat]
            t_i = t_eval[..., i].ravel()[m_flat]
            valid = np.isfinite(p_i) & np.isfinite(t_i)
            p_v, t_v = p_i[valid], t_i[valid]
            if len(p_v) > 0:
                r_val = float(np.corrcoef(p_v, t_v)[0, 1])
                nrmse_val = float(np.sqrt(np.mean((p_v - t_v) ** 2)) / max(float(t_v.max() - t_v.min()), 1e-6) * 100)
            else:
                r_val, nrmse_val = 0.0, 100.0
            metrics_data["R"][name].append((label, r_val))
            metrics_data["NRMSE"][name].append((label, nrmse_val))

    # Figure 1: Comparative Bar Charts for R and NRMSE with Input/Output Annotation
    fig_metrics = plt.figure(figsize=(15.5, 7.5), constrained_layout=True)
    gs = fig_metrics.add_gridspec(2, 2, height_ratios=[1, 0.24])
    ax_r = fig_metrics.add_subplot(gs[0, 0])
    ax_nrmse = fig_metrics.add_subplot(gs[0, 1])
    ax_info = fig_metrics.add_subplot(gs[1, :])

    num_configs = len(config_labels)
    x = np.arange(len(property_names))
    bar_width = 0.8 / max(num_configs, 1)

    colors = ["#4A90E2", "#9013FE", "#50E3C2", "#F5A623", "#E94E77", "#7ED321", "#B8E986", "#417505"]

    for idx, label in enumerate(config_labels):
        r_vals = [dict(metrics_data["R"][name]).get(label, 0.0) for name in property_names]
        nrmse_vals = [dict(metrics_data["NRMSE"][name]).get(label, 0.0) for name in property_names]
        offset = (idx - (num_configs - 1) / 2) * bar_width
        c = colors[idx % len(colors)]
        bars1 = ax_r.bar(x + offset, r_vals, bar_width, label=label, color=c, alpha=0.85, edgecolor="black", linewidth=0.8)
        bars2 = ax_nrmse.bar(x + offset, nrmse_vals, bar_width, label=label, color=c, alpha=0.85, edgecolor="black", linewidth=0.8)
        for b in bars1:
            ax_r.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
        for b in bars2:
            ax_nrmse.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax_r.set_xticks(x)
    ax_r.set_xticklabels([f"Output: {p}" for p in property_names], fontsize=11, fontweight="bold")
    ax_r.set_ylabel("Parity Correlation (R)", fontsize=11, fontweight="bold")
    ax_r.set_title("Parity Correlation (R) Across Input Configurations", fontsize=12, fontweight="bold")
    ax_r.set_ylim(0, 1.15)
    ax_r.grid(axis="y", linestyle="--", alpha=0.5)
    ax_r.legend(fontsize=8.5, loc="lower right")

    max_nrmse_val = max([max(dict(metrics_data["NRMSE"][name]).values(), default=10.0) for name in property_names])
    ax_nrmse.set_xticks(x)
    ax_nrmse.set_xticklabels([f"Output: {p}" for p in property_names], fontsize=11, fontweight="bold")
    ax_nrmse.set_ylabel("Normalized RMSE (%)", fontsize=11, fontweight="bold")
    ax_nrmse.set_title("Normalized RMSE (%) Across Input Configurations", fontsize=12, fontweight="bold")
    ax_nrmse.set_ylim(0, max(max_nrmse_val * 1.18, 15.0))
    ax_nrmse.grid(axis="y", linestyle="--", alpha=0.5)
    ax_nrmse.legend(fontsize=8.5, loc="upper right")

    # Bottom annotation panel: Detail Inputs and Outputs in English
    ax_info.axis("off")
    info_text = (
        "MODEL INPUT & OUTPUT SPECIFICATIONS:\n"
        "• TARGET OUTPUTS: ΔVP (P-Wave Velocity Change, m/s), ΔSw (Water Saturation Change), Δρ (Bulk Density Change) + Uncertainty (σ)\n"
        "• CONFIG 1 (Scalar Slices / No TS): 32 Summary Attributes (Base 8 + Monitor 8 + ΔA 8 + Rel ΔA/A 8 across 4 angles)\n"
        "• CONFIG 2 (Temporal Window 1D Only / No TS): 64 Stratal Attributes (Base 16 + Mon 16 + Δ 16 + Rel Δ/Base 16 in Top/Base sublayers)\n"
        "• CONFIG 3 (Scalar + Temporal 1D / No TS): 96 Attributes (32 Summary + 64 Stratal Temporal)\n"
        "• CONFIG 4 (Scalar Slices / With TS): 36 Attributes (32 Summary + 4 Seismic 4D Time-Shift Maps dt)\n"
        "• CONFIG 5 (Scalar + Temporal 1D / With TS): 100 Attributes (32 Summary + 64 Stratal Temporal + 4 Time-Shift dt)"
    )
    ax_info.text(0.01, 0.5, info_text, fontsize=8.8, va="center", ha="left", family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.2))

    if output_metrics is not None:
        _save(fig_metrics, output_metrics)

    t_eval = truth[-1] if truth.ndim == 4 else truth
    num_cols = len(config_labels) + 1  # Truth + each config
    generated_map_figs = {}

    # Generate individual map comparison figures for ALL predicted properties
    for prop_idx, prop_name in enumerate(property_names):
        fig_prop, axes_prop = plt.subplots(2, num_cols, figsize=(4.0 * num_cols, 8.2), squeeze=False, constrained_layout=True)
        t_prop = np.where(mask, t_eval[..., prop_idx], np.nan) if mask is not None else t_eval[..., prop_idx]

        # Choose appropriate colormap and bounds based on property physics
        if "sw" in prop_name.lower():
            cmap_name = EAGE_CMAP
            p_vmin, p_vmax = -0.2, 0.8
        else:
            cmap_name = "viridis" if "vp" in prop_name.lower() or "p" in prop_name.lower() else "plasma"
            finite_vals = t_prop[np.isfinite(t_prop)]
            if len(finite_vals) > 0:
                p_vmin = float(np.nanpercentile(finite_vals, 1))
                p_vmax = float(np.nanpercentile(finite_vals, 99))
            else:
                p_vmin, p_vmax = 0.0, 1.0

        # Column 0: Ground Truth
        im_gt = axes_prop[0, 0].imshow(t_prop, cmap=cmap_name, vmin=p_vmin, vmax=p_vmax, origin="lower")
        axes_prop[0, 0].set_title(f"GROUND TRUTH\nTrue Target: {prop_name}", fontsize=10, fontweight="bold", pad=8)
        axes_prop[0, 0].set_ylabel(f"OUTPUT 1:\nPrediction ({prop_name})", fontsize=10, fontweight="bold")
        fig_prop.colorbar(im_gt, ax=axes_prop[0, 0], shrink=0.75)
        axes_prop[1, 0].axis("off")
        axes_prop[1, 0].text(
            0.5, 0.5,
            f"Ground Truth: {prop_name}\nTrue Reference\n(26 Well Train\n37,935 Validation)",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            transform=axes_prop[1, 0].transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e9ecef", edgecolor="#adb5bd")
        )

        for c_idx, label in enumerate(config_labels, start=1):
            in_desc = input_specs.get(label, f"Input: {label}")
            if label in loaded_preds:
                p_map = loaded_preds[label][..., prop_idx]
                u_map = loaded_uncs[label][..., prop_idx]
                p_disp = np.where(mask, p_map, np.nan) if mask is not None else p_map
                u_disp = np.where(mask, u_map, np.nan) if mask is not None else u_map
                im_p = axes_prop[0, c_idx].imshow(p_disp, cmap=cmap_name, vmin=p_vmin, vmax=p_vmax, origin="lower")
                axes_prop[0, c_idx].set_title(f"{label}\n{in_desc}", fontsize=8.2, fontweight="bold", pad=8)
                fig_prop.colorbar(im_p, ax=axes_prop[0, c_idx], shrink=0.75)

                u_finite = u_disp[np.isfinite(u_disp)]
                u_max = float(np.nanpercentile(u_finite, 99)) if len(u_finite) > 0 else 0.1
                im_u = axes_prop[1, c_idx].imshow(u_disp, cmap="magma", vmin=0, vmax=max(u_max, 1e-4), origin="lower")
                axes_prop[1, c_idx].set_title(f"Predictive Uncertainty (σ):\n{label}", fontsize=8.2, fontweight="bold", pad=6)
                fig_prop.colorbar(im_u, ax=axes_prop[1, c_idx], shrink=0.75)

        axes_prop[1, 1].set_ylabel(f"OUTPUT 2:\nUncertainty σ({prop_name})", fontsize=10, fontweight="bold")

        for ax_row in axes_prop:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])

        clean_slug = prop_name.lower().replace("δ", "d").replace("Δ", "d").replace("ρ", "rho").replace("/", "_").replace(" ", "_")
        generated_map_figs[clean_slug] = fig_prop

        if output_maps is not None:
            out_p = Path(output_maps)
            stem = out_p.stem
            # Save dedicated file for this property
            prop_out_path = out_p.parent / f"{stem}_{clean_slug}{out_p.suffix}"
            _save(fig_prop, prop_out_path)
            # If this is dSw, also save to default output_maps path
            if clean_slug == "dsw" or prop_idx == 0:
                _save(fig_prop, out_p)

    # Figure 3: Combined Multi-Property Overview Grid (3 properties x 5 columns)
    fig_all, axes_all = plt.subplots(len(property_names), num_cols, figsize=(3.8 * num_cols, 3.4 * len(property_names)), squeeze=False, constrained_layout=True)
    for p_i, p_name in enumerate(property_names):
        t_p = np.where(mask, t_eval[..., p_i], np.nan) if mask is not None else t_eval[..., p_i]
        c_name = EAGE_CMAP if "sw" in p_name.lower() else ("viridis" if "vp" in p_name.lower() or "p" in p_name.lower() else "plasma")
        if "sw" in p_name.lower():
            v0, v1 = -0.2, 0.8
        else:
            fin = t_p[np.isfinite(t_p)]
            v0 = float(np.nanpercentile(fin, 1)) if len(fin) > 0 else 0.0
            v1 = float(np.nanpercentile(fin, 99)) if len(fin) > 0 else 1.0

        im0 = axes_all[p_i, 0].imshow(t_p, cmap=c_name, vmin=v0, vmax=v1, origin="lower")
        axes_all[p_i, 0].set_ylabel(f"Ground Truth & Pred:\n{p_name}", fontsize=9.5, fontweight="bold")
        if p_i == 0:
            axes_all[p_i, 0].set_title("GROUND TRUTH\nAlvo Real", fontsize=9.5, fontweight="bold", pad=6)
        fig_all.colorbar(im0, ax=axes_all[p_i, 0], shrink=0.75)

        for c_i, lbl in enumerate(config_labels, start=1):
            if lbl in loaded_preds:
                pm = loaded_preds[lbl][..., p_i]
                p_d = np.where(mask, pm, np.nan) if mask is not None else pm
                im_c = axes_all[p_i, c_i].imshow(p_d, cmap=c_name, vmin=v0, vmax=v1, origin="lower")
                if p_i == 0:
                    axes_all[p_i, c_i].set_title(f"{lbl}", fontsize=9, fontweight="bold", pad=6)
                fig_all.colorbar(im_c, ax=axes_all[p_i, c_i], shrink=0.75)

    for ax_r in axes_all:
        for ax in ax_r:
            ax.set_xticks([])
            ax.set_yticks([])

    if output_maps is not None:
        out_p = Path(output_maps)
        all_props_path = out_p.parent / f"{out_p.stem}_all_properties{out_p.suffix}"
        _save(fig_all, all_props_path)

    default_map_fig = generated_map_figs.get("dsw", next(iter(generated_map_figs.values()))) if generated_map_figs else fig_metrics
    return fig_metrics, default_map_fig
