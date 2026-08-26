"""Extraction of 4D seismic attributes and property targets from SEG-Y files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import segyio


def load_segy_cube(path: str | Path, grid_shape: tuple[int, int] = (234, 325)) -> np.ndarray:
    """Load a SEG-Y file and reshape traces to [inlines, crosslines, samples]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {path}")
    with segyio.open(str(path), "r", strict=False) as f:
        traces = f.trace.raw[:]
        n_traces, n_samples = traces.shape
        expected = grid_shape[0] * grid_shape[1]
        if n_traces != expected:
            raise ValueError(f"Trace count {n_traces} does not match grid shape {grid_shape} ({expected} traces)")
        # UNISIM SGY traces are stored with X (crossline, 325) as the outer/slow axis
        # and Y (inline, 234) as the inner/fast axis.
        return traces.reshape(grid_shape[1], grid_shape[0], n_samples).transpose(1, 0, 2).astype(np.float32)


def compute_sna(seismic_cube: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Compute Sum of Negative Amplitudes (SNA) along the depth/time axis."""
    neg = np.minimum(seismic_cube, 0.0)
    if mask is not None:
        neg = neg * mask
    return np.sum(neg, axis=-1).astype(np.float32)


def compute_rms(seismic_cube: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Compute Root Mean Square (RMS) amplitude along the depth/time axis."""
    sq = seismic_cube**2
    if mask is not None:
        sq = np.where(mask, sq, 0.0)
        count = np.maximum(np.sum(mask, axis=-1), 1)
        return np.sqrt(np.sum(sq, axis=-1) / count).astype(np.float32)
    return np.sqrt(np.mean(sq, axis=-1)).astype(np.float32)


def extract_unisim_dataset(
    sgy_dir: str | Path,
    output_path: str | Path | None = None,
    grid_shape: tuple[int, int] = (234, 325),
    angles: Sequence[int] = (10, 20, 30, 40),
) -> dict[str, np.ndarray]:
    """Extract multi-angle 4D seismic attributes (SNA + RMS) and target property changes from UNISIM SGYs.

    Parameters
    ----------
    sgy_dir : str or Path
        Root directory containing SGYs (with 2013/ and 2024/ subdirectories).
    output_path : str or Path, optional
        Path to save the resulting .npz archive.
    grid_shape : tuple[int, int]
        Spatial dimensions (inlines, crosslines). Default: (234, 325).
    angles : tuple of int
        Angle stacks to load. Default: (10, 20, 30, 40).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with arrays: 'seismic', 'pore_volume', 'targets', 'mask', 'property_names'.
    """
    sgy_dir = Path(sgy_dir)
    dir_2013 = sgy_dir / "2013"
    dir_2024 = sgy_dir / "2024"

    # 1. Load static Porosity and Vshale to define reservoir mask and petrophysical static maps
    porosity_path = sgy_dir / "Porosity.sgy"
    poro_cube = load_segy_cube(porosity_path, grid_shape)
    res_mask_3d = poro_cube > 0
    res_mask_2d = np.any(res_mask_3d, axis=-1)
    pore_volume = np.sum(poro_cube, axis=-1).astype(np.float32)

    vsh_path = sgy_dir / "Vshale.sgy"
    vsh_cube = load_segy_cube(vsh_path, grid_shape) if vsh_path.exists() else None

    h, w = grid_shape
    poro_map = np.zeros(grid_shape, dtype=np.float32)
    vsh_map = np.zeros(grid_shape, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            m = res_mask_3d[i, j]
            if np.any(m):
                poro_map[i, j] = np.mean(poro_cube[i, j, m])
                if vsh_cube is not None:
                    vsh_map[i, j] = np.mean(vsh_cube[i, j, m])

    yy, xx = np.indices(grid_shape)
    x_norm = ((xx - xx.mean()) / xx.std()).astype(np.float32)
    y_norm = ((yy - yy.mean()) / yy.std()).astype(np.float32)

    static_features = np.stack([pore_volume, poro_map, vsh_map, x_norm, y_norm], axis=-1)

    # 2. Load 2013 and 2024 seismic angle stacks and compute multi-attribute maps (SNA + RMS)
    num_angles = len(angles)
    # 2 attributes per angle: SNA and RMS -> 2 * num_angles total channels
    seismic_maps = np.zeros((2, h, w, 2 * num_angles), dtype=np.float32)

    for idx, deg in enumerate(angles):
        f2013 = dir_2013 / f"seismic2013_{deg}deg_NoiseFree.sgy"
        f2024 = dir_2024 / f"seismic2024_{deg}deg_TMonitor_NoiseFree.sgy"

        cube_13 = load_segy_cube(f2013, grid_shape)
        cube_24 = load_segy_cube(f2024, grid_shape)

        # SNA
        seismic_maps[0, ..., idx] = compute_sna(cube_13, res_mask_3d)
        seismic_maps[1, ..., idx] = compute_sna(cube_24, res_mask_3d)
        # RMS
        seismic_maps[0, ..., num_angles + idx] = compute_rms(cube_13, res_mask_3d)
        seismic_maps[1, ..., num_angles + idx] = compute_rms(cube_24, res_mask_3d)

    # 3. Load dynamic properties for baseline (2013) and monitor (2024)
    vp_13 = load_segy_cube(dir_2013 / "Pvelocity2013_NoiseFree.sgy", grid_shape)
    vp_24 = load_segy_cube(dir_2024 / "Pvelocity2024_TMonitor_NoiseFree.sgy", grid_shape)

    sw_13 = load_segy_cube(dir_2013 / "sw2013.sgy", grid_shape)
    sw_24 = load_segy_cube(dir_2024 / "sw2024_TMonitor.sgy", grid_shape)

    rho_13 = load_segy_cube(dir_2013 / "Density2013_NoiseFree.sgy", grid_shape)
    rho_24 = load_segy_cube(dir_2024 / "Density2024_TMonitor_NoiseFree.sgy", grid_shape)

    # Targets: [vintage, H, W, 3] -> (ΔVP, ΔSw, Δρ)
    # Vintage 0 (2013): baseline (change = 0)
    # Vintage 1 (2024): monitor change (2024 - 2013 vertically averaged in reservoir)
    targets = np.zeros((2, h, w, 3), dtype=np.float32)

    # 4. Compute 4D time-shift / time-strain maps: tau = ln(Vp2 / Vp1) and dt_ratio = (Vp2 - Vp1) / Vp2
    # Vintage 0 (2013): 0
    # Vintage 1 (2024): vertically averaged in reservoir
    time_shift_maps = np.zeros((2, h, w, 2), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            m = res_mask_3d[i, j]
            if np.any(m):
                v1 = vp_13[i, j, m]
                v2 = vp_24[i, j, m]
                targets[1, i, j, 0] = np.mean(v2 - v1)
                targets[1, i, j, 1] = np.mean(sw_24[i, j, m] - sw_13[i, j, m])
                targets[1, i, j, 2] = np.mean(rho_24[i, j, m] - rho_13[i, j, m])

                # tau = ln(dt / (dt + Delta_t)) = ln(Vp2 / Vp1)
                time_shift_maps[1, i, j, 0] = np.mean(np.log(v2 / np.maximum(v1, 1e-6)))
                # Delta_dt / (dt + Delta_dt) = (Vp2 - Vp1) / Vp2
                time_shift_maps[1, i, j, 1] = np.mean((v2 - v1) / np.maximum(v2, 1e-6))

    property_names = np.array(["ΔVP", "ΔSw", "Δρ"])

    data = {
        "seismic": seismic_maps,
        "pore_volume": pore_volume,
        "static_features": static_features,
        "time_shift": time_shift_maps,
        "targets": targets,
        "mask": res_mask_2d,
        "property_names": property_names,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_p, **data)

    return data
