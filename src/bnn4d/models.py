"""Networks described in Sukar, Côrte & MacBeth (2026).

The paper's Figures 1 and 2 specify a symmetric fully-connected
encoder/decoder with widths 1024, 768, 512, 256, a 256-dimensional
latent space, then 256, 512, 768, 1024.  The number of temporal input
features is intentionally configurable because the paper does not publish
the sliding-window length.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PAPER_WIDTHS = (1024, 768, 512, 256, 256, 256, 512, 768, 1024)


def _activation(name: str) -> Callable[[], nn.Module]:
    activations = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU, "tanh": nn.Tanh}
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name!r}; choose {tuple(activations)}") from exc


class AleatoricAutoencoder(nn.Module):
    """Deterministic encoder-decoder with heteroscedastic Gaussian output.

    It returns three property means (dP, dSw, dSg) and their log variances.
    Log variance is clamped only when requested; clamping prevents numerical
    overflow without changing the distributional formulation in the paper.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        widths: Sequence[int] = PAPER_WIDTHS,
        activation: str = "relu",
        log_variance_bounds: tuple[float, float] | None = (-20.0, 10.0),
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if not widths:
            raise ValueError("widths must contain at least the latent layer")
        act = _activation(activation)
        layers: list[nn.Module] = []
        previous = input_dim
        for width in widths:
            layers.extend((nn.Linear(previous, width), act()))
            previous = width
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(previous, output_dim)
        self.log_variance_head = nn.Linear(previous, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.widths = tuple(widths)
        self.log_variance_bounds = log_variance_bounds

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(x)
        mean = self.mean_head(hidden)
        log_variance = self.log_variance_head(hidden)
        if self.log_variance_bounds is not None:
            log_variance = log_variance.clamp(*self.log_variance_bounds)
        return mean, log_variance

    @torch.no_grad()
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return expected properties and aleatoric standard deviation."""
        mean, log_variance = self(x)
        return mean, torch.exp(0.5 * log_variance)


class VariationalLinear(nn.Module):
    """Mean-field Gaussian linear layer with a standard Gaussian prior."""

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0) -> None:
        super().__init__()
        if prior_std <= 0:
            raise ValueError("prior_std must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = float(prior_std)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.constant_(self.bias_rho, -5.0)

    @staticmethod
    def _sample(mu: Tensor, rho: Tensor) -> Tensor:
        sigma = F.softplus(rho)
        return mu + sigma * torch.randn_like(mu)

    def forward(self, x: Tensor, sample: bool = True) -> Tensor:
        if sample:
            weight = self._sample(self.weight_mu, self.weight_rho)
            bias = self._sample(self.bias_mu, self.bias_rho)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> Tensor:
        """Analytic KL(q || N(0, prior_std²)) for weights and biases."""
        prior_var = self.prior_std**2

        def kl(mu: Tensor, rho: Tensor) -> Tensor:
            sigma = F.softplus(rho)
            return (torch.log(self.prior_std / sigma) + (sigma.square() + mu.square()) / (2 * prior_var) - 0.5).sum()

        return kl(self.weight_mu, self.weight_rho) + kl(self.bias_mu, self.bias_rho)


class EpistemicBNN(nn.Module):
    """Fully variational form of every dense layer in the paper's network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        widths: Sequence[int] = PAPER_WIDTHS,
        activation: str = "relu",
        prior_std: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        act = _activation(activation)
        dimensions = (input_dim, *widths, output_dim)
        self.layers = nn.ModuleList(
            VariationalLinear(a, b, prior_std) for a, b in zip(dimensions[:-1], dimensions[1:])
        )
        self.activation = act()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.widths = tuple(widths)
        self.prior_std = prior_std

    def forward(self, x: Tensor, sample: bool = True) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x, sample=sample))
        return self.layers[-1](x, sample=sample)

    def kl_divergence(self) -> Tensor:
        return torch.stack([layer.kl_divergence() for layer in self.layers]).sum()

    @torch.no_grad()
    def predict_distribution(self, x: Tensor, samples: int = 500) -> tuple[Tensor, Tensor, Tensor]:
        """Monte Carlo posterior: samples, mean and epistemic std (paper: 500)."""
        if samples < 2:
            raise ValueError("samples must be at least 2 to estimate standard deviation")
        draws = torch.stack([self(x, sample=True) for _ in range(samples)])
        return draws, draws.mean(dim=0), draws.std(dim=0, unbiased=True)

