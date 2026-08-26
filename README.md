# BNN4D: Bayesian Neural Networks for 4D Seismic Inversion

Python/PyTorch implementation of the 4D seismic reservoir property estimation and uncertainty quantification framework based on **Sukar, Côrte & MacBeth (2026)** (*“Dynamic Reservoir Property Estimation With Uncertainty Quantification From 4D Seismic Data Using Bayesian Neural Networks”*), adapted for the **UNISIM-I** benchmark reservoir.

---

## 1. Architecture & Modeling

The codebase provides two complementary Bayesian deep learning approaches to invert 4D seismic attributes directly into dynamic reservoir property changes with calibrated uncertainty estimates:

1. **Aleatoric Model (`AleatoricAutoencoder` with Residual Skip-Connections):**
   - Deep feedforward neural network with dual output heads: $\mu(x)$ (predictive mean) and $\log \sigma^2(x)$ (heteroscedastic log-variance of observation noise).
   - Optimized via Heteroscedastic Gaussian Negative Log-Likelihood:
     $$\mathcal{L}_{\text{aleatoric}} = \frac{1}{2N} \sum_{i=1}^N \left( \frac{\|y_i - \mu(x_i)\|^2}{\sigma^2(x_i)} + \log \sigma^2(x_i) \right)$$

2. **Epistemic Model (`EpistemicBNN` / Res-BNN):**
   - Fully variational Bayesian Neural Network where **all dense layers contain stochastic Gaussian weight distributions** $w \sim \mathcal{N}(\mu_w, \sigma_w^2)$.
   - Standard Gaussian prior $\mathcal{N}(0, \sigma_0^2 I)$.
   - Optimized via Variational Free Energy / ELBO with $\text{KL}(q(w) \| p(w))$ divergence regularizer and KL annealing warmup.
   - Inference via **Monte Carlo Variational Sampling**: $S$ stochastic draws estimate posterior expectation $\mathbb{E}[y]$ and epistemic model uncertainty $\sigma_{\text{epistemic}} = \text{std}(y^{(s)})$.

---

## 2. Model Inputs & Outputs

### Target Outputs:
The models simultaneously predict 3 dynamic petroelastic property changes between base and monitor surveys, alongside spatial uncertainty maps $\sigma$:
* **$\Delta V_P$:** P-wave compressional velocity change ($m/s$).
* **$\Delta S_w$:** Water saturation change in pore space ($0.0 - 1.0$).
* **$\Delta \rho$:** Bulk rock and fluid density change ($g/cm^3$).
* **$\sigma$:** Predictive uncertainty associated with each estimated property.

### The 4 Input Configurations (Ablation Study Matrix):

| Configuration | Total Features | Input Features Breakdown |
| :--- | :---: | :--- |
| **Config 1: No Static / No TS** | **32** | 16 Multi-Angle Amplitudes ($A_{\text{base}}, A_{\text{mon}}$ across 8 angles) + 8 Deltas $\Delta A$ + 8 Relative Deltas $\frac{\Delta A}{\|A\|}$ |
| **Config 2: With Static / No TS** | **37** | 32 4D Amplitudes + 5 Static Geology Maps (Porosity $\phi$, Shale Volume $V_{\text{sh}}$, Permeabilities $K_x, K_y, K_z$) |
| **Config 3: No Static / With TS** | **36** | 32 4D Amplitudes + 4 Seismic 4D Time-Shift Maps ($dt$) |
| **Config 4: With Static / With TS** | **41** | 32 4D Amplitudes + 4 Time-Shift ($dt$) + 5 Static Rock Property Maps |

---

## 3. Realistic Well Calibration Regime (26 UNISIM-I Wells + Blind Field CV)

In real exploration and production assets, models are trained **only on drilled well locations**, while predictions are deployed across the full 3D reservoir:

* **Training Set:** Strictly constrained to the **26 canonical UNISIM-I real wells** (`--train-traces 26 --trace-selection unisim_wells`) across all folds.
* **Blind Validation (Out-of-Fold):** The remaining **37,935 reservoir traces** are divided into 5 blind folds for strict field validation without data leakage.
* **Early Stopping & Annealing:** Validation loss monitoring with configurable patience (`--patience 30`), cosine annealing learning rate scheduler, and KL warmup.

---

## 4. Run & Debug via VS Code (`.vscode/launch.json`)

The [`.vscode/launch.json`](.vscode/launch.json) file includes preconfigured tasks for the **Run & Debug (F5)** panel:

### Batch Ablation Execution + Comparative Plots:
* **`0. [RUN-ALL] Run All 4 Ablations + Generate Comparison Plots (Aleatoric + Epistemic)`**
* **`0. [RUN-ALL] Run All 4 Ablations (Epistemic Only)`**
* **`0. [RUN-ALL] Run All 4 Ablations (Aleatoric Only)`**

### Individual Scenario Runs:
* `1. [EXP-1] 5-Fold CV: No Static / No TS (Aleatoric)`
* `2. [EXP-1] 5-Fold CV: No Static / No TS (Epistemic)`
* `3. [EXP-2] 5-Fold CV: With Static / No TS (Aleatoric)`
* `4. [EXP-2] 5-Fold CV: With Static / No TS (Epistemic)`
* `5. [EXP-3] 5-Fold CV: No Static / With TS (Aleatoric)`
* `6. [EXP-3] 5-Fold CV: No Static / With TS (Epistemic)`
* `7. [EXP-4] 5-Fold CV: With Static / With TS (Aleatoric)`
* `8. [EXP-4] 5-Fold CV: With Static / With TS (Epistemic)`
* `9. [PLOT-COMPARE] Generate 4-Scenario Comparative Plots (Aleatoric)`
* `10. [PLOT-COMPARE] Generate 4-Scenario Comparative Plots (Epistemic)`

---

## 5. Command-Line Interface (CLI)

Run via `uv run` or `python -m bnn4d.cli`:

```bash
# 1. Run all 4 ablation scenarios sequentially and generate comparison plots
uv run python -m bnn4d.cli run-all-ablations \
  --data artifacts/unisim_4d.npz \
  --output-dir artifacts/ablation_study \
  --model all \
  --epochs 150 \
  --patience 30

# 2. Run 5-Fold Cross-Validation for a specific configuration
uv run python -m bnn4d.cli cv \
  --data artifacts/unisim_4d.npz \
  --output-dir artifacts/cv_epistemic_pure \
  --model epistemic \
  --train-traces 26 \
  --trace-selection unisim_wells \
  --activation gelu \
  --learning-rate 0.002 \
  --widths 256 128 64 128 256 \
  --folds 5 \
  --epochs 150 \
  --patience 30

# 3. Generate comparative plots from existing experiment directories
uv run python -m bnn4d.cli compare-ablations \
  --data artifacts/unisim_4d.npz \
  --experiments \
    "1. No Static / No TS=artifacts/ablation_study/exp1_no_static_no_ts_epistemic" \
    "2. With Static / No TS=artifacts/ablation_study/exp2_with_static_no_ts_epistemic" \
    "3. No Static / With TS=artifacts/ablation_study/exp3_no_static_with_ts_epistemic" \
    "4. With Static / With TS=artifacts/ablation_study/exp4_with_static_with_ts_epistemic" \
  --output-dir artifacts/ablation_study \
  --model-name epistemic
```

---

## 6. Unit Tests

Run the full automated test suite:

```bash
uv run python -m unittest discover -s tests -v
```
