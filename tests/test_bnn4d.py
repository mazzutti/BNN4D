import unittest

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from bnn4d.data import FeatureStandardizer, SlidingWindowDataset, add_relative_gaussian_noise
from bnn4d.losses import gaussian_nll, variational_free_energy
from bnn4d.models import AleatoricAutoencoder, EpistemicBNN, VariationalLinear
from bnn4d.sgy import compute_sna
from bnn4d.visualization import (
    plot_eage_saturation_vp_comparison,
    plot_error_maps,
    plot_prediction_diagnostics,
    plot_property_maps,
    plot_training_history,
)


class ModelTests(unittest.TestCase):
    def test_aleatoric_shapes_and_positive_std(self):
        model = AleatoricAutoencoder(9, widths=(12, 4, 12))
        mean, std = model.predict(torch.randn(7, 9))
        self.assertEqual(mean.shape, (7, 3))
        self.assertEqual(std.shape, (7, 3))
        self.assertTrue(torch.all(std > 0))

    def test_nll_matches_standard_normal(self):
        target = torch.zeros(2, 3)
        got = gaussian_nll(target, target, torch.zeros_like(target))
        self.assertAlmostEqual(got.item(), 0.5 * torch.log(torch.tensor(2 * torch.pi)).item(), places=6)

    def test_variational_layer_is_stochastic_and_kl_positive(self):
        layer = VariationalLinear(4, 2)
        x = torch.ones(3, 4)
        self.assertFalse(torch.equal(layer(x), layer(x)))
        self.assertGreater(layer.kl_divergence().item(), 0)

    def test_epistemic_statistics(self):
        model = EpistemicBNN(5, widths=(8, 3, 8))
        draws, mean, std = model.predict_distribution(torch.randn(4, 5), samples=5)
        self.assertEqual(draws.shape, (5, 4, 3))
        self.assertEqual(mean.shape, std.shape)
        self.assertTrue(torch.all(std > 0))
        total, fit, complexity = variational_free_energy(draws[0], torch.zeros(4, 3), model.kl_divergence(), 100)
        self.assertTrue(torch.allclose(total, fit + complexity))


class DataTests(unittest.TestCase):
    def test_sliding_window_layout(self):
        seismic = torch.arange(4 * 2 * 3 * 4).reshape(4, 2, 3, 4)
        pore_volume = torch.ones(2, 3)
        targets = torch.zeros(4, 2, 3, 3)
        data = SlidingWindowDataset(seismic, pore_volume, targets, window=2)
        self.assertEqual(len(data), 3 * 2 * 3)
        self.assertEqual(data.features.shape[1], 2 * 4 + 4 + 1)  # 2*channels + delta + pore_volume

    def test_sliding_window_with_time_shift_and_rel_deltas(self):
        seismic = torch.arange(4 * 2 * 3 * 4).reshape(4, 2, 3, 4)
        pore_volume = torch.ones(2, 3)
        time_shift = torch.ones(4, 2, 3, 2)
        targets = torch.zeros(4, 2, 3, 3)
        data = SlidingWindowDataset(
            seismic,
            pore_volume,
            targets,
            window=2,
            include_deltas=True,
            include_relative_deltas=True,
            time_shift=time_shift,
        )
        self.assertEqual(len(data), 3 * 2 * 3)
        # 2*4 (seismic) + 4 (delta) + 4 (rel_delta) + 2*2 (time_shift) + 1 (pore_vol) = 21
        self.assertEqual(data.features.shape[1], 8 + 4 + 4 + 4 + 1)

    def test_trace_selection_strategies(self):
        from bnn4d.data import select_subset_traces
        mask = np.ones((50, 50), dtype=bool)
        for method in ["unisim_wells", "spatial_optimal", "random"]:
            idx = select_subset_traces(total_samples=2500, mask=mask, n_traces=26, method=method, seed=42)
            self.assertEqual(len(idx), 26)
            self.assertEqual(len(np.unique(idx)), 26)
            self.assertTrue(np.all(idx >= 0) and np.all(idx < 2500))
        # Test subset selection from canonical wells
        idx_sub = select_subset_traces(total_samples=2500, mask=mask, n_traces=10, method="unisim_wells", seed=42)
        self.assertEqual(len(idx_sub), 10)
        self.assertEqual(len(np.unique(idx_sub)), 10)

    def test_standardization(self):
        values = torch.randn(100, 4) * 3 + 7
        scaled = FeatureStandardizer().fit(values).transform(values)
        self.assertTrue(torch.allclose(scaled.mean(0), torch.zeros(4), atol=1e-5))
        self.assertTrue(torch.allclose(scaled.std(0, unbiased=False), torch.ones(4), atol=1e-5))

    def test_noise_zero_is_copy(self):
        data = torch.randn(20)
        noisy = add_relative_gaussian_noise(data, 0)
        self.assertTrue(torch.equal(data, noisy))
        self.assertIsNot(data, noisy)

    def test_compute_sna(self):
        cube = np.array([[[-1.0, 2.0], [0.5, -3.0]]], dtype=np.float32)
        sna = compute_sna(cube)
        self.assertEqual(sna.shape, (1, 2))
        self.assertEqual(sna[0, 0], -1.0)
        self.assertEqual(sna[0, 1], -3.0)


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.random.default_rng(2).normal(size=(2, 4, 5, 3))
        self.uncertainty = np.full_like(self.mean, 0.5)
        self.truth = self.mean + 0.2

    def tearDown(self):
        plt.close("all")

    def test_training_history(self):
        figure = plot_training_history({"loss": [3.0, 2.0, 1.0]})
        self.assertEqual(len(figure.axes), 1)

    def test_property_maps(self):
        figure = plot_property_maps(self.mean, self.uncertainty, truth=self.truth)
        self.assertGreaterEqual(len(figure.axes), 9)

    def test_diagnostics(self):
        figure = plot_prediction_diagnostics(self.mean, self.uncertainty, self.truth)
        self.assertGreaterEqual(len(figure.axes), 9)

    def test_eage_comparison(self):
        figure = plot_eage_saturation_vp_comparison(self.mean[..., 0], self.truth[..., 1], self.truth[..., 0])
        self.assertGreaterEqual(len(figure.axes), 3)

    def test_error_maps(self):
        figure = plot_error_maps(self.mean, self.truth, uncertainty=self.uncertainty)
        self.assertGreaterEqual(len(figure.axes), 9)


if __name__ == "__main__":
    unittest.main()
