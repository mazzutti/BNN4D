"""Command-line interface for training and applying BNN4D models."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import FeatureStandardizer, SlidingWindowDataset, add_relative_gaussian_noise, build_sliding_features
from .models import PAPER_WIDTHS, AleatoricAutoencoder, EpistemicBNN
from .training import train_aleatoric_epoch, train_epistemic_epoch


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_npz(path: Path, require_targets: bool) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        required = {"seismic", "pore_volume"} | ({"targets"} if require_targets else set())
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(sorted(missing))}")
        return {name: archive[name] for name in archive.files}


def _standardizer_state(scaler: FeatureStandardizer) -> dict[str, torch.Tensor]:
    assert scaler.mean_ is not None and scaler.scale_ is not None
    return {"mean": scaler.mean_.cpu(), "scale": scaler.scale_.cpu()}


def _restore_standardizer(state: dict[str, torch.Tensor]) -> FeatureStandardizer:
    scaler = FeatureStandardizer()
    scaler.mean_, scaler.scale_ = state["mean"], state["scale"]
    return scaler


def train(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)
    device = _device(args.device)
    arrays = _load_npz(args.data, require_targets=True)
    dataset = SlidingWindowDataset(arrays["seismic"], arrays["pore_volume"], arrays["targets"], args.window)

    # Noise affects time-varying seismic attributes, never static pore volume.
    features = dataset.features.clone()
    features[:, :-1] = add_relative_gaussian_noise(features[:, :-1], args.noise, torch.Generator().manual_seed(args.seed))
    feature_scaler = FeatureStandardizer().fit(features)
    target_scaler = FeatureStandardizer().fit(dataset.targets)
    features = feature_scaler.transform(features)
    targets = target_scaler.transform(dataset.targets)
    loader = DataLoader(TensorDataset(features, targets), batch_size=args.batch_size, shuffle=True)

    widths = tuple(args.widths)
    if args.model == "aleatoric":
        model: AleatoricAutoencoder | EpistemicBNN = AleatoricAutoencoder(features.shape[1], widths=widths, activation=args.activation)
    else:
        model = EpistemicBNN(features.shape[1], widths=widths, activation=args.activation, prior_std=args.prior_std)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[float] = []
    for epoch in range(1, args.epochs + 1):
        if args.model == "aleatoric":
            loss = train_aleatoric_epoch(model, loader, optimizer, device)
        else:
            loss = train_epistemic_epoch(model, loader, optimizer, len(dataset), args.observation_std, device)
        history.append(loss)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d}/{args.epochs} loss={loss:.6f}")

    checkpoint = {
        "format_version": 1,
        "model_type": args.model,
        "model_config": {
            "input_dim": features.shape[1], "output_dim": targets.shape[1],
            "widths": widths, "activation": args.activation, "prior_std": args.prior_std,
        },
        "model_state": model.state_dict(),
        "feature_scaler": _standardizer_state(feature_scaler),
        "target_scaler": _standardizer_state(target_scaler),
        "window": args.window,
        "attribute_count": arrays["seismic"].shape[-1],
        "history": {"loss": history},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"checkpoint={args.output} device={device} samples={len(dataset)}")


def predict(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    if checkpoint["model_type"] == "aleatoric":
        model = AleatoricAutoencoder(config["input_dim"], config["output_dim"], config["widths"], config["activation"])
    else:
        model = EpistemicBNN(config["input_dim"], config["output_dim"], config["widths"], config["activation"], config["prior_std"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    arrays = _load_npz(args.data, require_targets=False)
    seismic, pore_volume = arrays["seismic"], arrays["pore_volume"]
    if seismic.shape[-1] != checkpoint["attribute_count"]:
        raise ValueError("input attribute count differs from the training data")
    features = build_sliding_features(seismic, pore_volume, checkpoint["window"])
    feature_scaler = _restore_standardizer(checkpoint["feature_scaler"])
    target_scaler = _restore_standardizer(checkpoint["target_scaler"])
    features = feature_scaler.transform(features).to(device)

    means, uncertainties = [], []
    for start in range(0, len(features), args.batch_size):
        batch = features[start : start + args.batch_size]
        if checkpoint["model_type"] == "aleatoric":
            mean, uncertainty = model.predict(batch)
        else:
            _, mean, uncertainty = model.predict_distribution(batch, args.samples)
        means.append(mean.cpu())
        uncertainties.append(uncertainty.cpu())
    mean = target_scaler.inverse_transform(torch.cat(means)).numpy()
    uncertainty = (torch.cat(uncertainties) * target_scaler.scale_.abs()).numpy()
    map_shape = (seismic.shape[0] - checkpoint["window"] + 1, *seismic.shape[1:3], config["output_dim"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, mean=mean.reshape(map_shape), uncertainty=uncertainty.reshape(map_shape))
    print(f"predictions={args.output} shape={map_shape} model={checkpoint['model_type']}")


def make_demo_data(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    seismic = rng.normal(size=(args.vintages, args.rows, args.cols, 4)).astype("float32")
    pore_volume = rng.uniform(0.2, 1.0, size=(args.rows, args.cols)).astype("float32")
    targets = np.empty((args.vintages, args.rows, args.cols, 3), dtype="float32")
    targets[..., 0] = 0.7 * seismic[..., 0] - 0.2 * seismic[..., 2] + pore_volume
    targets[..., 1] = 0.3 * seismic[..., 1] + 0.1 * seismic[..., 3]
    targets[..., 2] = -0.25 * seismic[..., 0] + 0.2 * seismic[..., 2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, seismic=seismic, pore_volume=pore_volume, targets=targets)
    print(f"demo_data={args.output} seismic={seismic.shape} targets={targets.shape}")


def plot_results(args: argparse.Namespace) -> None:
    # Import lazily so training/prediction do not require a graphical backend.
    import matplotlib

    matplotlib.use("Agg")
    from .visualization import plot_prediction_diagnostics, plot_property_maps, plot_training_history

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.predictions) as archive:
        if not {"mean", "uncertainty"}.issubset(archive.files):
            raise ValueError("prediction NPZ must contain mean and uncertainty")
        mean, uncertainty = archive["mean"], archive["uncertainty"]
    truth = None
    if args.data is not None:
        arrays = _load_npz(args.data, require_targets=True)
        if arrays["targets"].shape[0] < mean.shape[0]:
            raise ValueError("target data has fewer vintages than predictions")
        truth = arrays["targets"][-mean.shape[0] :]
    plot_property_maps(mean, uncertainty, truth=truth, vintage=args.vintage, output=args.output_dir / "property_maps.png")
    if truth is not None:
        plot_prediction_diagnostics(mean, uncertainty, truth, output=args.output_dir / "diagnostics.png")
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        history = checkpoint.get("history")
        if history:
            plot_training_history(history, log_y=args.log_y, output=args.output_dir / "training_history.png")
    created = sorted(path.name for path in args.output_dir.glob("*.png"))
    print(f"plots={args.output_dir} files={','.join(created)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bnn4d", description="4D reservoir-property estimation with uncertainty")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("make-demo-data", help="create a small synthetic NPZ for a smoke test")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo.npz"))
    demo.add_argument("--vintages", type=int, default=4)
    demo.add_argument("--rows", type=int, default=8)
    demo.add_argument("--cols", type=int, default=8)
    demo.add_argument("--seed", type=int, default=42)
    demo.set_defaults(func=make_demo_data)

    fit = sub.add_parser("train", help="train one of the paper's networks")
    fit.add_argument("--data", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--model", choices=("aleatoric", "epistemic"), default="aleatoric")
    fit.add_argument("--window", type=int, default=2)
    fit.add_argument("--noise", type=float, default=0.17, help="relative noise fraction (paper optimum: 0.17)")
    fit.add_argument("--epochs", type=int, default=400)
    fit.add_argument("--batch-size", type=int, default=256)
    fit.add_argument("--learning-rate", type=float, default=1e-3)
    fit.add_argument("--activation", choices=("relu", "elu", "gelu", "tanh"), default="relu")
    fit.add_argument("--widths", nargs="+", type=int, default=list(PAPER_WIDTHS))
    fit.add_argument("--prior-std", type=float, default=1.0)
    fit.add_argument("--observation-std", type=float, default=1.0)
    fit.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, ...")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--log-every", type=int, default=10)
    fit.set_defaults(func=train)

    infer = sub.add_parser("predict", help="predict maps and uncertainty from a checkpoint")
    infer.add_argument("--data", required=True, type=Path)
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--samples", type=int, default=500, help="epistemic Monte Carlo passes")
    infer.add_argument("--batch-size", type=int, default=4096)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--seed", type=int, default=42)
    infer.set_defaults(func=predict)

    plots = sub.add_parser("plot", help="render training curves, maps and diagnostics")
    plots.add_argument("--predictions", required=True, type=Path)
    plots.add_argument("--output-dir", required=True, type=Path)
    plots.add_argument("--data", type=Path, help="optional NPZ with targets for comparison")
    plots.add_argument("--checkpoint", type=Path, help="optional checkpoint with training history")
    plots.add_argument("--vintage", type=int, default=-1)
    plots.add_argument("--log-y", action="store_true", help="use logarithmic loss axis")
    plots.set_defaults(func=plot_results)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
