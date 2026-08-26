"""Bayesian neural networks for dynamic reservoir-property estimation."""

from .models import AleatoricAutoencoder, EpistemicBNN
from .data import FeatureStandardizer, SlidingWindowDataset, add_relative_gaussian_noise, build_sliding_features
from .losses import gaussian_nll, variational_free_energy
from .metrics import normalized_rmse
from .sgy import extract_unisim_dataset, load_segy_cube, compute_sna, compute_rms
from .visualization import plot_property_maps, plot_training_history, plot_cv_history, plot_prediction_diagnostics, plot_eage_saturation_vp_comparison, plot_error_maps

__all__ = [
    "AleatoricAutoencoder",
    "EpistemicBNN",
    "FeatureStandardizer",
    "SlidingWindowDataset",
    "add_relative_gaussian_noise",
    "build_sliding_features",
    "gaussian_nll",
    "variational_free_energy",
    "normalized_rmse",
    "extract_unisim_dataset",
    "load_segy_cube",
    "compute_sna",
    "compute_rms",
    "plot_property_maps",
    "plot_training_history",
    "plot_cv_history",
    "plot_prediction_diagnostics",
    "plot_eage_saturation_vp_comparison",
    "plot_error_maps",
]
