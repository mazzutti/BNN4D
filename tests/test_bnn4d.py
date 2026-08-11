import unittest

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from bnn4d.data import FeatureStandardizer, SlidingWindowDataset, add_relative_gaussian_noise
from bnn4d.losses import gaussian_nll, variational_free_energy
from bnn4d.models import AleatoricAutoencoder, EpistemicBNN, VariationalLinear
from bnn4d.visualization import plot_prediction_diagnostics, plot_property_maps, plot_training_history


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
        self.assertEqual(data.features.shape[1], 2 * 4 + 1)

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
        self.assertEqual(len(figure.axes), 9)


if __name__ == "__main__":
    unittest.main()
