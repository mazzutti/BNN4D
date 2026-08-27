"""Command-line interface for training and applying BNN4D models."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    FeatureStandardizer,
    SlidingWindowDataset,
    add_relative_gaussian_noise,
    build_sliding_features,
    select_subset_traces,
)
from .models import PAPER_WIDTHS, AleatoricAutoencoder, EpistemicBNN
from .sgy import extract_unisim_dataset
from .training import evaluate_aleatoric, evaluate_epistemic, train_aleatoric_epoch, train_epistemic_epoch


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
        required = {"seismic"} | ({"targets"} if require_targets else set())
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


def prepare_unisim(args: argparse.Namespace) -> None:
    extract_unisim_dataset(
        sgy_dir=args.sgy_dir,
        output_path=args.output,
        grid_shape=tuple(args.grid_shape),
        angles=tuple(args.angles),
    )
    print(f"unisim_dataset={args.output} sgy_dir={args.sgy_dir}")


def train(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)
    device = _device(args.device)
    arrays = _load_npz(args.data, require_targets=True)
    mask = arrays.get("mask")
    use_time_shift = getattr(args, "use_time_shift", False)
    temporal_window = getattr(args, "temporal_window", False)
    relative_deltas = getattr(args, "relative_deltas", True)
    include_scalar = not getattr(args, "no_scalar", False)
    temporal_window_data = arrays.get("temporal_window") if temporal_window else None
    time_shift = arrays.get("time_shift") if use_time_shift else None
    dataset = SlidingWindowDataset(
        arrays["seismic"],
        temporal_window=temporal_window_data,
        targets=arrays["targets"],
        window=args.window,
        mask=mask,
        include_deltas=True,
        include_relative_deltas=relative_deltas,
        include_scalar=include_scalar,
        time_shift=time_shift,
    )

    n_samples = len(dataset)
    val_split = getattr(args, "val_split", 0.20)
    train_traces = getattr(args, "train_traces", None)
    trace_selection = getattr(args, "trace_selection", "unisim_wells")

    # Trace selection (e.g. 15 wells for training, rest for validation/testing)
    if train_traces is not None:
        train_sub = select_subset_traces(
            total_samples=n_samples,
            mask=mask,
            n_traces=train_traces,
            method=trace_selection,
            seed=args.seed,
        )
        val_sub = np.setdiff1d(np.arange(n_samples), train_sub)
        train_indices = torch.as_tensor(train_sub, dtype=torch.long)
        val_indices = torch.as_tensor(val_sub, dtype=torch.long)
        print(f"Using {len(train_indices)} traces for training via '{trace_selection}' strategy; {len(val_indices)} traces for blind validation.")
    elif val_split > 0.0:
        if not 0.0 <= val_split < 1.0:
            raise ValueError("val_split must be in [0.0, 1.0)")
        n_val = int(round(val_split * n_samples))
        n_train = n_samples - n_val
        indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(args.seed))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
    else:
        train_indices = torch.arange(n_samples)
        val_indices = torch.empty(0, dtype=torch.long)

    train_features = dataset.features[train_indices].clone()
    if args.noise > 0:
        train_features = add_relative_gaussian_noise(
            train_features, args.noise, torch.Generator().manual_seed(args.seed)
        )
    train_targets = dataset.targets[train_indices]

    # Fit scalers STRICTLY on training split (no data leakage!)
    feature_scaler = FeatureStandardizer().fit(train_features)
    target_scaler = FeatureStandardizer().fit(train_targets)

    train_x = feature_scaler.transform(train_features)
    train_y = target_scaler.transform(train_targets)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if len(val_indices) > 0:
        val_features = dataset.features[val_indices]
        val_targets = dataset.targets[val_indices]
        val_x = feature_scaler.transform(val_features)
        val_y = target_scaler.transform(val_targets)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False)

    widths = tuple(args.widths)
    if args.model == "aleatoric":
        model: AleatoricAutoencoder | EpistemicBNN = AleatoricAutoencoder(train_x.shape[1], output_dim=train_y.shape[1], widths=widths, activation=args.activation)
    else:
        model = EpistemicBNN(train_x.shape[1], output_dim=train_y.shape[1], widths=widths, activation=args.activation, prior_std=args.prior_std)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)

    patience = getattr(args, "patience", 10)
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    train_history: list[float] = []
    val_history: list[float] = []
    for epoch in range(1, args.epochs + 1):
        if args.model == "aleatoric":
            t_loss = train_aleatoric_epoch(model, train_loader, optimizer, device)
            v_loss = evaluate_aleatoric(model, val_loader, device) if val_loader else float("nan")
        else:
            kl_scale = min(1.0, epoch / 25.0) * (1.0 / max(len(train_indices), 10000))
            t_loss = train_epistemic_epoch(model, train_loader, optimizer, len(train_indices), args.observation_std, device, kl_scale=kl_scale)
            v_loss = evaluate_epistemic(model, val_loader, len(train_indices), args.observation_std, device) if val_loader else float("nan")
        scheduler.step()
        train_history.append(t_loss)
        if val_loader:
            val_history.append(v_loss)
            if v_loss < best_val_loss - 1e-4:
                best_val_loss = v_loss
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            val_str = f" val_loss={v_loss:.6f}" if val_loader else ""
            print(f"epoch={epoch:04d}/{args.epochs} train_loss={t_loss:.6f}{val_str}")
        if val_loader and patience > 0 and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch:04d} (best val_loss={best_val_loss:.6f} at epoch {best_epoch:04d})")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    prop_names = list(arrays["property_names"]) if "property_names" in arrays else ["ΔP", "ΔSw", "ΔSg"]
    history_dict: dict[str, list[float]] = {"train_loss": train_history}
    if val_history:
        history_dict["val_loss"] = val_history
    checkpoint = {
        "format_version": 1,
        "model_type": args.model,
        "model_config": {
            "input_dim": train_features.shape[1], "output_dim": train_targets.shape[1],
            "widths": widths, "activation": args.activation, "prior_std": args.prior_std,
            "use_time_shift": use_time_shift, "temporal_window": temporal_window,
        },
        "model_state": model.state_dict(),
        "feature_scaler": _standardizer_state(feature_scaler),
        "target_scaler": _standardizer_state(target_scaler),
        "window": args.window,
        "attribute_count": arrays["seismic"].shape[-1],
        "use_time_shift": use_time_shift,
        "temporal_window": temporal_window,
        "property_names": prop_names,
        "train_indices": train_indices.numpy(),
        "val_indices": val_indices.numpy(),
        "history": history_dict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"checkpoint={args.output} device={device} train_samples={len(train_indices)} val_samples={len(val_indices)}")


def cross_validate(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)
    device = _device(args.device)
    arrays = _load_npz(args.data, require_targets=True)
    mask = arrays.get("mask")
    use_time_shift = getattr(args, "use_time_shift", False)
    temporal_window = getattr(args, "temporal_window", False)
    relative_deltas = getattr(args, "relative_deltas", True)
    include_scalar = not getattr(args, "no_scalar", False)
    temporal_window_data = arrays.get("temporal_window") if temporal_window else None
    time_shift = arrays.get("time_shift") if use_time_shift else None
    dataset = SlidingWindowDataset(
        arrays["seismic"],
        temporal_window=temporal_window_data,
        targets=arrays["targets"],
        window=args.window,
        mask=mask,
        include_deltas=True,
        include_relative_deltas=relative_deltas,
        include_scalar=include_scalar,
        time_shift=time_shift,
    )

    n_samples = len(dataset)
    n_folds = args.folds
    if n_folds < 2:
        raise ValueError("folds must be at least 2")

    train_traces = getattr(args, "train_traces", None)
    trace_selection = getattr(args, "trace_selection", "unisim_wells")

    if train_traces is not None:
        fixed_train_idx = select_subset_traces(
            total_samples=n_samples,
            mask=mask,
            n_traces=train_traces,
            method=trace_selection,
            seed=args.seed,
        )
        field_indices = np.setdiff1d(np.arange(n_samples), fixed_train_idx)
        n_field = len(field_indices)

        if getattr(args, "spatial", False) and mask is not None:
            h, w = arrays["seismic"].shape[1:3]
            coords_i, coords_j = np.where(mask)
            f_coords_i, f_coords_j = coords_i[field_indices], coords_j[field_indices]
            block_h = int(np.ceil(np.sqrt(n_folds)))
            block_w = int(np.ceil(n_folds / block_h))
            grid_i = (f_coords_i * block_h // h).clip(0, block_h - 1)
            grid_j = (f_coords_j * block_w // w).clip(0, block_w - 1)
            field_fold_ids = (grid_i * block_w + grid_j) % n_folds
        else:
            perm = torch.randperm(n_field, generator=torch.Generator().manual_seed(args.seed)).numpy()
            field_fold_ids = np.zeros(n_field, dtype=int)
            for i, idx in enumerate(perm):
                field_fold_ids[idx] = i % n_folds

        print(f"Starting Fixed-Train {n_folds}-Fold CV (train_wells={len(fixed_train_idx)} fixed via '{trace_selection}', field_val_samples={n_field}, model={args.model})...")
    else:
        if getattr(args, "spatial", False) and mask is not None:
            h, w = arrays["seismic"].shape[1:3]
            coords_i, coords_j = np.where(mask)
            block_h = int(np.ceil(np.sqrt(n_folds)))
            block_w = int(np.ceil(n_folds / block_h))
            grid_i = (coords_i * block_h // h).clip(0, block_h - 1)
            grid_j = (coords_j * block_w // w).clip(0, block_w - 1)
            fold_ids = (grid_i * block_w + grid_j) % n_folds
        else:
            perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(args.seed)).numpy()
            fold_ids = np.zeros(n_samples, dtype=int)
            for i, idx in enumerate(perm):
                fold_ids[idx] = i % n_folds
        print(f"Starting {n_folds}-Fold Cross-Validation (model={args.model}, samples={n_samples}, spatial={getattr(args, 'spatial', False)})...")

    oof_predictions = np.full((n_samples, dataset.targets.shape[-1]), np.nan, dtype=np.float32)
    oof_uncertainties = np.full((n_samples, dataset.targets.shape[-1]), np.nan, dtype=np.float32)

    fold_histories: list[dict[str, list[float]]] = []

    for fold in range(n_folds):
        if train_traces is not None:
            train_idx = fixed_train_idx
            val_idx = field_indices[field_fold_ids == fold]
        else:
            val_mask = fold_ids == fold
            train_mask = ~val_mask
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

        train_feats = dataset.features[train_idx].clone()
        if args.noise > 0:
            train_feats = add_relative_gaussian_noise(
                train_feats, args.noise, torch.Generator().manual_seed(args.seed + fold)
            )
        train_targs = dataset.targets[train_idx]
        val_feats = dataset.features[val_idx]
        val_targs = dataset.targets[val_idx]

        f_scaler = FeatureStandardizer().fit(train_feats)
        t_scaler = FeatureStandardizer().fit(train_targs)

        train_x = f_scaler.transform(train_feats)
        train_y = t_scaler.transform(train_targs)
        val_x = f_scaler.transform(val_feats)
        val_y = t_scaler.transform(val_targs)

        train_batch_size = min(args.batch_size, len(train_idx))
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False)

        widths = tuple(args.widths)
        if args.model == "aleatoric":
            model = AleatoricAutoencoder(train_x.shape[1], output_dim=train_y.shape[1], widths=widths, activation=args.activation)
        else:
            model = EpistemicBNN(train_x.shape[1], output_dim=train_y.shape[1], widths=widths, activation=args.activation, prior_std=args.prior_std)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)

        best_val = float("inf")
        best_state = None
        best_epoch = 1
        stopped_epoch = args.epochs
        patience_counter = 0
        f_train_hist: list[float] = []
        f_val_hist: list[float] = []

        for epoch in range(1, args.epochs + 1):
            if args.model == "aleatoric":
                t_loss = train_aleatoric_epoch(model, train_loader, optimizer, device)
                v_loss = evaluate_aleatoric(model, val_loader, device)
            else:
                kl_scale = min(1.0, epoch / 25.0) * (1.0 / max(len(train_idx), 10000))
                t_loss = train_epistemic_epoch(model, train_loader, optimizer, len(train_idx), args.observation_std, device, kl_scale=kl_scale)
                v_loss = evaluate_epistemic(model, val_loader, len(train_idx), args.observation_std, device)
            scheduler.step()
            f_train_hist.append(t_loss)
            f_val_hist.append(v_loss)
            if v_loss < best_val - 1e-4:
                best_val = v_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                stopped_epoch = epoch
                break
        else:
            stopped_epoch = args.epochs

        fold_histories.append({"train_loss": f_train_hist, "val_loss": f_val_hist})

        if best_state is not None:
            model.load_state_dict(best_state)

        # Predict out-of-fold
        model.eval()
        with torch.no_grad():
            if args.model == "aleatoric":
                mu_s, std_s = model.predict(val_x.to(device))
                pred_f = t_scaler.inverse_transform(mu_s.cpu()).numpy()
                unc_f = (std_s.cpu() * t_scaler.scale_.abs()).numpy()
                if fold == 0 and train_traces is not None:
                    mu_tr, std_tr = model.predict(train_x.to(device))
                    oof_predictions[fixed_train_idx] = t_scaler.inverse_transform(mu_tr.cpu()).numpy()
                    oof_uncertainties[fixed_train_idx] = (std_tr.cpu() * t_scaler.scale_.abs()).numpy()
            else:
                draws = []
                x_val_gpu = val_x.to(device)
                for _ in range(args.samples):
                    model.draw_sample()
                    draws.append(model(x_val_gpu, sample=True).cpu())
                    model.clear_sample()
                draws_t = torch.stack(draws)
                pred_f = t_scaler.inverse_transform(draws_t.mean(dim=0)).numpy()
                unc_f = (draws_t.std(dim=0) * t_scaler.scale_.abs()).numpy()
                if fold == 0 and train_traces is not None:
                    draws_tr = []
                    x_tr_gpu = train_x.to(device)
                    for _ in range(args.samples):
                        model.draw_sample()
                        draws_tr.append(model(x_tr_gpu, sample=True).cpu())
                        model.clear_sample()
                    d_tr = torch.stack(draws_tr)
                    oof_predictions[fixed_train_idx] = t_scaler.inverse_transform(d_tr.mean(dim=0)).numpy()
                    oof_uncertainties[fixed_train_idx] = (d_tr.std(dim=0) * t_scaler.scale_.abs()).numpy()

        oof_predictions[val_idx] = pred_f
        oof_uncertainties[val_idx] = unc_f

        true_f = val_targs.numpy()
        corrs = [np.corrcoef(pred_f[:, i], true_f[:, i])[0, 1] for i in range(true_f.shape[-1])]
        nrmses = [np.sqrt(np.mean((pred_f[:, i] - true_f[:, i])**2)) / max(true_f[:, i].max() - true_f[:, i].min(), 1e-6) for i in range(true_f.shape[-1])]
        print(f"Fold {fold+1}/{n_folds} (epochs={stopped_epoch}/{args.epochs}, best_epoch={best_epoch}): best_val_loss={best_val:.4f} | R: {', '.join(f'{c:.3f}' for c in corrs)} | NRMSE: {', '.join(f'{n*100:.1f}%' for n in nrmses)}")

    true_all = dataset.targets.numpy()
    prop_names = list(arrays.get("property_names", ["ΔP", "ΔSw", "ΔSg"]))
    print(f"\n--- OVERALL OUT-OF-FOLD {n_folds}-FOLD CV RESULTS ---")
    for i, name in enumerate(prop_names):
        r = np.corrcoef(oof_predictions[:, i], true_all[:, i])[0, 1]
        nrmse = np.sqrt(np.mean((oof_predictions[:, i] - true_all[:, i])**2)) / max(true_all[:, i].max() - true_all[:, i].min(), 1e-6)
        print(f"  {name}: Out-of-Fold Parity R = {r:.4f} | NRMSE = {nrmse*100:.2f}%")

    num_vintages = arrays["seismic"].shape[0] - args.window + 1
    map_shape = (num_vintages, *arrays["seismic"].shape[1:3], dataset.targets.shape[-1])
    if mask is not None:
        mask_flat = mask.ravel()
        full_pred = np.full((num_vintages, mask.size, dataset.targets.shape[-1]), np.nan, dtype=np.float32)
        full_unc = np.full((num_vintages, mask.size, dataset.targets.shape[-1]), np.nan, dtype=np.float32)
        pts_per_v = mask_flat.sum()
        for v in range(num_vintages):
            full_pred[v, mask_flat] = oof_predictions[v * pts_per_v : (v + 1) * pts_per_v]
            full_unc[v, mask_flat] = oof_uncertainties[v * pts_per_v : (v + 1) * pts_per_v]
        pred_out = full_pred.reshape(map_shape)
        unc_out = full_unc.reshape(map_shape)
    else:
        pred_out = oof_predictions.reshape(map_shape)
        unc_out = oof_uncertainties.reshape(map_shape)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / "oof_predictions.npz"
    import json
    np.savez_compressed(
        out_file,
        mean=pred_out,
        uncertainty=unc_out,
        mask=mask,
        property_names=np.array(prop_names),
        fold_histories=json.dumps(fold_histories),
    )

    import matplotlib
    matplotlib.use("Agg")
    from .visualization import plot_cv_history, plot_eage_saturation_vp_comparison, plot_error_maps, plot_prediction_diagnostics, plot_property_maps
    truth = arrays["targets"][-num_vintages:]
    plot_property_maps(pred_out, unc_out, truth=truth, property_names=prop_names, mask=mask, output=args.output_dir / "oof_property_maps.png")
    plot_error_maps(pred_out, truth, uncertainty=unc_out, property_names=prop_names, mask=mask, output=args.output_dir / "oof_error_maps.png")
    plot_prediction_diagnostics(pred_out, unc_out, truth, property_names=prop_names, mask=mask, output=args.output_dir / "oof_diagnostics.png")
    plot_cv_history(fold_histories, output=args.output_dir / "oof_training_history.png")
    if "ΔVP" in prop_names and "ΔSw" in prop_names:
        vp_idx = list(prop_names).index("ΔVP")
        sw_idx = list(prop_names).index("ΔSw")
        plot_eage_saturation_vp_comparison(
            predicted_dvp=pred_out[..., vp_idx],
            reference_dsw=truth[..., sw_idx],
            reference_dvp=truth[..., vp_idx],
            mask=mask,
            output=args.output_dir / "oof_eage_comparison.png",
        )
    print(f"CV evaluation complete! Artifacts saved to {args.output_dir}")


@torch.no_grad()
def predict(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    if checkpoint["model_type"] == "aleatoric":
        model = AleatoricAutoencoder(config["input_dim"], config["output_dim"], config["widths"], config["activation"])
    else:
        model = EpistemicBNN(config["input_dim"], config["output_dim"], config["widths"], config["activation"], config["prior_std"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    arrays = _load_npz(args.data, require_targets=False)
    seismic = arrays["seismic"]
    use_time_shift = getattr(args, "use_time_shift", False) or config.get("use_time_shift", checkpoint.get("use_time_shift", False))
    temporal_window = getattr(args, "temporal_window", False) or config.get("temporal_window", checkpoint.get("temporal_window", False))
    temporal_window_data = arrays.get("temporal_window") if temporal_window else None
    time_shift = arrays.get("time_shift") if use_time_shift else None
    mask = arrays.get("mask")
    relative_deltas = getattr(args, "relative_deltas", True) or checkpoint.get("relative_deltas", True)
    if seismic.shape[-1] != checkpoint["attribute_count"]:
        raise ValueError("input attribute count differs from the training data")
    features = build_sliding_features(
        seismic,
        temporal_window=temporal_window_data,
        window=checkpoint["window"],
        mask=mask,
        include_deltas=True,
        include_relative_deltas=relative_deltas,
        time_shift=time_shift,
    )
    feature_scaler = _restore_standardizer(checkpoint["feature_scaler"])
    target_scaler = _restore_standardizer(checkpoint["target_scaler"])
    features = feature_scaler.transform(features).to(device)

    if checkpoint["model_type"] == "aleatoric":
        means, uncertainties = [], []
        for start in range(0, len(features), args.batch_size):
            batch = features[start : start + args.batch_size]
            mean, uncertainty = model.predict(batch)
            means.append(mean.cpu())
            uncertainties.append(uncertainty.cpu())
        mean_tensor = torch.cat(means)
        unc_tensor = torch.cat(uncertainties)
    else:
        # Epistemic Monte Carlo: sample global weights S times and evaluate all batches with each sample
        all_draws = []
        for _ in range(args.samples):
            model.draw_sample()
            sample_outs = []
            for start in range(0, len(features), args.batch_size):
                batch = features[start : start + args.batch_size]
                out = model(batch, sample=True)
                sample_outs.append(out.cpu())
            all_draws.append(torch.cat(sample_outs))
            model.clear_sample()
        draws = torch.stack(all_draws)  # [samples, N, output_dim]
        mean_tensor = draws.mean(dim=0)
        unc_tensor = draws.std(dim=0, unbiased=True)

    mean_res = target_scaler.inverse_transform(mean_tensor).numpy()
    unc_res = (unc_tensor * target_scaler.scale_.abs()).numpy()

    num_vintages = seismic.shape[0] - checkpoint["window"] + 1
    map_shape = (num_vintages, *seismic.shape[1:3], config["output_dim"])
    
    if mask is not None:
        mask_flat = mask.ravel()
        num_cells = mask.size
        full_mean = np.full((num_vintages, num_cells, config["output_dim"]), np.nan, dtype=np.float32)
        full_unc = np.full((num_vintages, num_cells, config["output_dim"]), np.nan, dtype=np.float32)
        pts_per_v = mask_flat.sum()
        for v in range(num_vintages):
            full_mean[v, mask_flat] = mean_res[v * pts_per_v : (v + 1) * pts_per_v]
            full_unc[v, mask_flat] = unc_res[v * pts_per_v : (v + 1) * pts_per_v]
        mean_out = full_mean.reshape(map_shape)
        unc_out = full_unc.reshape(map_shape)
    else:
        mean_out = mean_res.reshape(map_shape)
        unc_out = unc_res.reshape(map_shape)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_dict = {
        "mean": mean_out,
        "uncertainty": unc_out,
        "property_names": checkpoint.get("property_names", ["ΔP", "ΔSw", "ΔSg"]),
    }
    if mask is not None:
        out_dict["mask"] = mask
    np.savez_compressed(args.output, **out_dict)
    print(f"predictions={args.output} mean={mean_out.shape} uncertainty={unc_out.shape}")


def make_demo_data(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)
    seismic = np.random.randn(args.vintages, args.rows, args.cols, 4).astype(np.float32)
    temporal_window = np.random.randn(args.vintages, args.rows, args.cols, 12).astype(np.float32)
    targets = np.zeros((args.vintages, args.rows, args.cols, 3), dtype=np.float32)
    targets[..., 0] = 0.5 * seismic[..., 0] - 0.3 * seismic[..., 1]
    targets[..., 1] = 0.2 * seismic[..., 1] + 0.4 * seismic[..., 3]
    targets[..., 2] = -0.25 * seismic[..., 0] + 0.2 * seismic[..., 2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, seismic=seismic, temporal_window=temporal_window, targets=targets)
    print(f"demo_data={args.output} seismic={seismic.shape} targets={targets.shape}")


def plot_results(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from .visualization import (
        plot_eage_saturation_vp_comparison,
        plot_error_maps,
        plot_prediction_diagnostics,
        plot_property_maps,
        plot_training_history,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.predictions) as archive:
        if not {"mean", "uncertainty"}.issubset(archive.files):
            raise ValueError("prediction NPZ must contain mean and uncertainty")
        mean, uncertainty = archive["mean"], archive["uncertainty"]
        property_names = list(archive["property_names"]) if "property_names" in archive.files else None
        mask = archive["mask"] if "mask" in archive.files else None

    if args.property_names:
        property_names = args.property_names
    elif property_names is None:
        property_names = ["ΔP", "ΔSw", "ΔSg"][: mean.shape[-1]]

    truth = None
    if args.data is not None:
        arrays = _load_npz(args.data, require_targets=True)
        if arrays["targets"].shape[0] < mean.shape[0]:
            raise ValueError("target data has fewer vintages than predictions")
        truth = arrays["targets"][-mean.shape[0] :]
        if mask is None and "mask" in arrays:
            mask = arrays["mask"]

    plot_property_maps(mean, uncertainty, truth=truth, vintage=args.vintage, property_names=property_names, mask=mask, output=args.output_dir / "property_maps.png")
    if truth is not None:
        plot_error_maps(mean, truth, uncertainty=uncertainty, vintage=args.vintage, property_names=property_names, mask=mask, output=args.output_dir / "error_maps.png")
        plot_prediction_diagnostics(mean, uncertainty, truth, property_names=property_names, mask=mask, output=args.output_dir / "diagnostics.png")
        # If ΔVP and ΔSw are present, plot EAGE comparison
        if "ΔVP" in property_names and "ΔSw" in property_names:
            vp_idx = list(property_names).index("ΔVP")
            sw_idx = list(property_names).index("ΔSw")
            plot_eage_saturation_vp_comparison(
                predicted_dvp=mean[..., vp_idx],
                reference_dsw=truth[..., sw_idx],
                reference_dvp=truth[..., vp_idx],
                mask=mask,
                output=args.output_dir / "eage_comparison.png",
            )
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    prep = sub.add_parser("prepare-unisim", help="extract multi-angle 4D attributes from UNISIM SGYs")
    prep.add_argument("--sgy-dir", type=Path, default=Path("data/SGYs"))
    prep.add_argument("--output", type=Path, default=Path("artifacts/unisim_4d.npz"))
    prep.add_argument("--grid-shape", nargs=2, type=int, default=[234, 325])
    prep.add_argument("--angles", nargs="+", type=int, default=[10, 20, 30, 40])
    prep.set_defaults(func=prepare_unisim)

    fit = sub.add_parser("train", help="train one of the paper's networks")
    fit.add_argument("--data", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--model", choices=("aleatoric", "epistemic"), default="aleatoric")
    fit.add_argument("--window", type=int, default=2)
    fit.add_argument("--train-traces", type=int, default=None, help="number of traces/wells to train on (e.g. 15). Default: None (use all)")
    fit.add_argument("--trace-selection", choices=("unisim_wells", "spatial_optimal", "random"), default="unisim_wells", help="trace selection method (default: unisim_wells)")
    fit.add_argument("--use-time-shift", action="store_true", default=False, help="include 4D time-shift in features (default: False)")
    fit.add_argument("--temporal-window", action="store_true", default=False, help="include 1D temporal waveform window features (default: False)")
    fit.add_argument("--no-scalar", action="store_true", default=False, help="exclude scalar summary attributes (default: False)")
    fit.add_argument("--noise", type=float, default=0.17, help="relative noise fraction (paper optimum: 0.17)")
    fit.add_argument("--epochs", type=int, default=400)
    fit.add_argument("--batch-size", type=int, default=256)
    fit.add_argument("--learning-rate", type=float, default=1e-3)
    fit.add_argument("--activation", choices=("relu", "elu", "gelu", "tanh"), default="relu")
    fit.add_argument("--widths", nargs="+", type=int, default=list(PAPER_WIDTHS))
    fit.add_argument("--prior-std", type=float, default=1.0)
    fit.add_argument("--observation-std", type=float, default=0.10, help="likelihood noise scale in normalized space (default: 0.10)")
    fit.add_argument("--val-split", type=float, default=0.20, help="fraction of samples reserved for blind validation")
    fit.add_argument("--patience", type=int, default=10, help="epochs to wait for val_loss improvement before early stopping (0=disabled)")
    fit.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, ...")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--log-every", type=int, default=10)
    fit.set_defaults(func=train)

    cv_p = sub.add_parser("cv", help="run K-Fold cross validation and generate out-of-fold evaluations")
    cv_p.add_argument("--data", required=True, type=Path)
    cv_p.add_argument("--output-dir", required=True, type=Path)
    cv_p.add_argument("--model", choices=("aleatoric", "epistemic"), default="aleatoric")
    cv_p.add_argument("--folds", type=int, default=5)
    cv_p.add_argument("--spatial", action="store_true", help="use spatial block partitioning instead of random k-fold")
    cv_p.add_argument("--train-traces", type=int, default=None, help="number of traces/wells to evaluate on (e.g. 15). Default: None (use all)")
    cv_p.add_argument("--trace-selection", choices=("unisim_wells", "spatial_optimal", "random"), default="unisim_wells", help="trace selection method (default: unisim_wells)")
    cv_p.add_argument("--window", type=int, default=2)
    cv_p.add_argument("--use-time-shift", action="store_true", default=False, help="include 4D time-shift in features (default: False)")
    cv_p.add_argument("--temporal-window", action="store_true", default=False, help="include 1D temporal waveform window features (default: False)")
    cv_p.add_argument("--no-scalar", action="store_true", default=False, help="exclude scalar summary attributes (default: False)")
    cv_p.add_argument("--noise", type=float, default=0.0)
    cv_p.add_argument("--epochs", type=int, default=200)
    cv_p.add_argument("--patience", type=int, default=10)
    cv_p.add_argument("--batch-size", type=int, default=256)
    cv_p.add_argument("--learning-rate", type=float, default=1e-3)
    cv_p.add_argument("--activation", choices=("relu", "elu", "gelu", "tanh"), default="relu")
    cv_p.add_argument("--widths", nargs="+", type=int, default=list(PAPER_WIDTHS))
    cv_p.add_argument("--prior-std", type=float, default=1.0)
    cv_p.add_argument("--observation-std", type=float, default=0.10, help="likelihood noise scale in normalized space (default: 0.10)")
    cv_p.add_argument("--samples", type=int, default=100)
    cv_p.add_argument("--device", default="auto")
    cv_p.add_argument("--seed", type=int, default=42)
    cv_p.set_defaults(func=cross_validate)

    infer = sub.add_parser("predict", help="predict maps and uncertainty from a checkpoint")
    infer.add_argument("--data", required=True, type=Path)
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--use-time-shift", action="store_true", default=False, help="force include 4D time-shift in features")
    infer.add_argument("--temporal-window", action="store_true", default=False, help="force include 1D temporal waveform window features")
    infer.add_argument("--no-scalar", action="store_true", default=False, help="exclude scalar summary attributes")
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
    plots.add_argument("--property-names", nargs="+", type=str, help="override property names")
    plots.add_argument("--vintage", type=int, default=-1)
    plots.add_argument("--log-y", action="store_true", help="use logarithmic loss axis")
    plots.set_defaults(func=plot_results)

    comp = sub.add_parser("compare-ablations", help="compare ablation experiments (metrics and maps)")
    comp.add_argument("--data", required=True, type=Path, help="NPZ with truth targets and mask")
    comp.add_argument("--experiments", nargs="+", required=True, help="list of Label=Path or Paths of experiment directories")
    comp.add_argument("--output-dir", required=True, type=Path, help="output directory for comparison figures")
    comp.add_argument("--model-name", type=str, default=None, help="optional model suffix for filenames (e.g. epistemic, aleatoric)")
    comp.set_defaults(func=compare_ablations)

    run_all = sub.add_parser("run-all-ablations", help="execute all 5 ablation scenarios sequentially and generate comparison plots")
    run_all.add_argument("--data", required=True, type=Path, help="NPZ dataset path")
    run_all.add_argument("--output-dir", type=Path, default=Path("artifacts/ablation_study"), help="base output directory")
    run_all.add_argument("--model", choices=("all", "aleatoric", "epistemic"), default="all", help="which model to run across scenarios")
    run_all.add_argument("--train-traces", type=int, default=26)
    run_all.add_argument("--trace-selection", choices=("unisim_wells", "spatial_optimal", "random"), default="unisim_wells")
    run_all.add_argument("--activation", choices=("relu", "elu", "gelu", "tanh"), default="gelu")
    run_all.add_argument("--learning-rate", type=float, default=0.002)
    run_all.add_argument("--widths", nargs="+", type=int, default=[256, 128, 64, 128, 256])
    run_all.add_argument("--folds", type=int, default=5)
    run_all.add_argument("--epochs", type=int, default=150)
    run_all.add_argument("--patience", type=int, default=30)
    run_all.add_argument("--batch-size", type=int, default=256)
    run_all.add_argument("--prior-std", type=float, default=1.0)
    run_all.add_argument("--observation-std", type=float, default=0.10)
    run_all.add_argument("--samples", type=int, default=100)
    run_all.add_argument("--device", default="auto")
    run_all.add_argument("--seed", type=int, default=42)
    run_all.add_argument("--force", action="store_true", help="force re-run existing experiments")
    run_all.set_defaults(func=run_all_ablations)

    return parser


def run_all_ablations(args: argparse.Namespace) -> None:
    models_to_run = ["aleatoric", "epistemic"] if args.model == "all" else [args.model]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("exp1_scalar_no_ts", "1. Scalar Slices / No TS", False, True, False),
        ("exp2_temporal_only_no_ts", "2. Temporal Window 1D Only / No TS", True, False, False),
        ("exp3_scalar_temporal_no_ts", "3. Scalar + Temporal 1D / No TS", True, True, False),
        ("exp4_scalar_with_ts", "4. Scalar Slices / With TS", False, True, True),
        ("exp5_scalar_temporal_with_ts", "5. Scalar + Temporal 1D / With TS", True, True, True),
    ]

    for model_type in models_to_run:
        print(f"\n=======================================================")
        print(f"  RUNNING ALL 5 ABLATION SCENARIOS FOR MODEL: {model_type.upper()}")
        print(f"=======================================================\n")

        exp_mapping: dict[str, Path] = {}
        for folder_prefix, label, temporal_window, include_scalar, use_time_shift in scenarios:
            exp_out = args.output_dir / f"{folder_prefix}_{model_type}"
            exp_mapping[label] = exp_out

            if (exp_out / "oof_predictions.npz").exists() and not getattr(args, "force", False):
                print(f"[SKIP] {label} already exists in {exp_out}. Use --force to re-run.")
                continue

            print(f"\n>>> Running Scenario: {label} (model={model_type}) >>>")
            cv_args = argparse.Namespace(
                data=args.data,
                output_dir=exp_out,
                model=model_type,
                train_traces=args.train_traces,
                trace_selection=args.trace_selection,
                temporal_window=temporal_window,
                no_scalar=not include_scalar,
                use_time_shift=use_time_shift,
                relative_deltas=True,
                activation=args.activation,
                learning_rate=args.learning_rate,
                widths=args.widths,
                folds=args.folds,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                prior_std=args.prior_std,
                observation_std=args.observation_std,
                samples=args.samples,
                device=args.device,
                seed=args.seed,
                spatial=False,
                noise=0.0,
                window=2,
            )
            cross_validate(cv_args)

        # Generate comparative plots for this model type
        print(f"\n>>> Generating Comparative Plots for {model_type.upper()} across all 5 scenarios >>>")
        import matplotlib
        matplotlib.use("Agg")
        from .visualization import plot_4configs_comparison
        arrays = _load_npz(args.data, require_targets=True)
        truth = arrays["targets"]
        mask = arrays.get("mask")
        prop_names = list(arrays.get("property_names", ["ΔVP", "ΔSw", "Δρ"]))

        out_metrics = args.output_dir / f"comparison_5configs_metrics_{model_type}.png"
        out_maps = args.output_dir / f"comparison_5configs_maps_{model_type}.png"
        plot_4configs_comparison(
            experiment_dirs=exp_mapping,
            truth=truth,
            property_names=prop_names,
            mask=mask,
            output_metrics=out_metrics,
            output_maps=out_maps,
        )
        print(f"Plots saved to:\n  - {out_metrics}\n  - {out_maps}")


def compare_ablations(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from .visualization import plot_4configs_comparison
    arrays = _load_npz(args.data, require_targets=True)
    truth = arrays["targets"]
    mask = arrays.get("mask")
    prop_names = list(arrays.get("property_names", ["ΔP", "ΔSw", "ΔSg"]))

    exp_dirs = {}
    for entry in args.experiments:
        if "=" in entry:
            label, path_str = entry.split("=", 1)
        else:
            path_str = entry
            label = Path(path_str).name
        exp_dirs[label] = Path(path_str)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.model_name}" if getattr(args, "model_name", None) else ""
    out_metrics = args.output_dir / f"comparison_4configs_metrics{suffix}.png"
    out_maps = args.output_dir / f"comparison_4configs_maps{suffix}.png"
    plot_4configs_comparison(
        experiment_dirs=exp_dirs,
        truth=truth,
        property_names=prop_names,
        mask=mask,
        output_metrics=out_metrics,
        output_maps=out_maps,
    )
    print(f"Comparison plots generated in {args.output_dir} with suffix '{suffix}'")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
