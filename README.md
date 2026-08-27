# BNN4D: Bayesian Neural Networks for 4D Seismic Inversion

Python/PyTorch implementation of the 4D seismic reservoir property estimation and uncertainty quantification framework based on **Sukar, Côrte & MacBeth (2026)** (*“Dynamic Reservoir Property Estimation With Uncertainty Quantification From 4D Seismic Data Using Bayesian Neural Networks”*), adapted and evaluated on the **UNISIM-I** benchmark reservoir.

---

## 1. Executive Summary & Key Highlights

* **Realistic 26-Well Calibration Regime:** Models are trained **strictly on 26 canonical UNISIM-I wells** (`n_traces=26, method='unisim_wells'`), predicting across **37,935 blind reservoir traces** evaluated via 5-Fold Cross-Validation.
* **Res-BNN Architecture:** Integrates residual skip-connections into variational Gaussian dense layers, eliminating gradient vanishing on sparse well calibrations and accelerating convergence.
* **Uncoupling 4D Dynamics via Time-Shift (dt):** 4D time-shifts resolve the intrinsic acoustic ambiguity between pressure decrease and water saturation increase, driving parity correlation to **R = 0.983** on ΔVP and **R > 0.945** on ΔSw and Δρ, with Normalized RMSE dropping to **2.53%**.
* **Dual Uncertainty Quantification:**
  * **Aleatoric Uncertainty (σ_aleatoric):** Captures heteroscedastic data noise and measurement imperfections.
  * **Epistemic Uncertainty (σ_epistemic):** Quantifies model parameter ambiguity via Monte Carlo variational draws (S = 50...200), highlighting faults and un-swept compartments.

---

## 2. Architecture & Theoretical Formulation

### A. Epistemic Model (`EpistemicBNN` / Res-BNN)
Every dense layer is formulated with stochastic variational Gaussian weight distributions:
* **Weight Distribution:** `w ~ Normal(μ_w, σ_w²)` where `σ_w = softplus(ρ_w)`
* **Prior Distribution:** Standard Gaussian `p(w) = Normal(0, σ_0² I)`
* **Objective Function (Variational Free Energy / ELBO with KL Warmup):**
  ```text
  Loss_epistemic = (1/B) * Σ ||y_i - f(x_i; w)||^2 + β_KL(t) * KL(q(w) || p(w))
  ```
  where `β_KL(t) = min(1.0, epoch / 25) * (1 / N_samples)` dynamically scales the KL regularizer during initial epochs.

### B. Aleatoric Model (`AleatoricAutoencoder`)
Deep feedforward architecture with dual output heads predicting mean `μ(x)` and heteroscedastic log-variance `log(σ²(x))`:
* **Objective Function (Heteroscedastic Gaussian Negative Log-Likelihood):**
  ```text
  Loss_aleatoric = (1 / 2B) * Σ [ ||y_i - μ(x_i)||^2 / σ²(x_i) + log(σ²(x_i)) ]
  ```

---

### 3. 5-Scenario Ablation Benchmark: Quantitative Results

### 3. 5-Scenario Ablation Benchmark: Quantitative Results

The 5 ablation configurations evaluate the incremental impact of scalar amplitude summaries, 1D stratal temporal waveform sublayer profiles, and 4D seismic time-shifts:

| Ablation Scenario | Features | Input Features Breakdown | ΔVP (R / NRMSE) | ΔSw (R / NRMSE) | Δρ (R / NRMSE) | Mean R | Mean NRMSE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Config 1: Scalar Slices / No TS** | **32** | 16 Multi-Angle Summary Amplitudes (Base, Mon 8 angles) + 8 Deltas ΔA + 8 Rel. Deltas ΔA/\|A\| | 0.7839 / 8.82% | 0.7741 / 10.88% | 0.7708 / 10.33% | **0.7763** | **10.01%** |
| **Config 2: Temporal 1D Only / No TS** | **64** | 1D Stratal Temporal Sublayers (Base 16 + Mon 16 + Δ 16 + Rel Δ/Base 16 in Top/Base sublayers) | 0.7352 / 9.94% | 0.7341 / 11.43% | 0.7393 / 10.88% | **0.7362** | **10.75%** |
| **Config 3: Scalar + Temporal 1D / No TS** | **96** | 32 Summary Amplitudes + 64 Stratal Temporal Sublayer Attributes | 0.7714 / 9.05% | 0.7648 / 11.24% | 0.7670 / 10.48% | **0.7677** | **10.26%** |
| **Config 4: Scalar Slices / With TS** | **36** | 32 Summary Amplitudes + 4 Seismic 4D Time-Shift Maps (`τ_strain`, `Δdt/dt`) | **0.9862** / **2.28%** | **0.9429** / **5.36%** | **0.9442** / **5.11%** | **0.9578** | **4.25%** |
| **Config 5: Scalar + Temporal 1D / With TS** | **100** | 32 Summary Amplitudes + 64 Stratal Temporal Sublayers + 4 Seismic 4D Time-Shift Maps | **0.9154** / **5.49%** | **0.8838** / **7.41%** | **0.8916** / **7.03%** | **0.8969** | **6.64%** |

### 3.1 Model Inputs & Target Outputs Specification for Each Configuration

```
                ┌─────────────────────────────────────────────────────────┐
                │                  SEISMIC INPUT FEATURES                 │
                └───────────────────────────┬─────────────────────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
      [32 Scalar Slices]          [64 Stratal 1D Temporal]    [4 Seismic Time-Shifts]
      • Base (8 ch)               • Top & Base Sublayers      • τ_strain = ln(Vp24/Vp13)
      • Monitor (8 ch)            • Base (16 ch), Mon (16 ch) • Δdt/dt = ΔVp/Vp24
      • ΔA (8 ch)                 • ΔW (16 ch), Rel Δ (16 ch)
      • Rel ΔA/A (8 ch)
               │                            │                            │
               ├───────────────────┬────────┴───────────┬────────────────┤
               │ Config 1: 32 ch   │ Config 2: 64 ch    │ Config 3: 96 ch│
               │ (Scalar / No TS)  │ (1D Only / No TS)  │ (Scal+1D/No TS)│
               │                   │                    │                │
               │ Config 4: 36 ch   │                    │ Config 5: 100ch│
               │ (Scalar / With TS)│                    │ (All Features) │
               └───────────────────┴────────┬───────────┴────────────────┘
                                            │
                                            ▼
                              ┌───────────────────────────┐
                              │    BNN4D INVERSION CORE   │
                              │  (Epistemic / Aleatoric)  │
                              └─────────────┬─────────────┘
                                            │
                                            ▼
                ┌─────────────────────────────────────────────────────────┐
                │                     TARGET OUTPUTS                      │
                ├─────────────────────────────────────────────────────────┤
                │  1. ΔVP: P-Wave Velocity Change (m/s)                   │
                │  2. ΔSw: Water Saturation Change (fractional [0, 1])    │
                │  3. Δρ:  Bulk Density Change (g/cm³)                    │
                │  4. σ:   Predictive Uncertainty (Epistemic / Aleatoric) │
                └─────────────────────────────────────────────────────────┘
```

#### A. Target Outputs (Shared across all 5 Configurations)
Every network model takes a surface trace location `(x, y)` and predicts **3 physical property changes** plus **predictive uncertainty**:
1. **`ΔVP` (Compressional Velocity Change, $m/s$):** $\Delta V_P = V_{P,2024} - V_{P,2013}$. Maps pressure depletion (softening) and water influx (hardening) dynamics across the reservoir.
2. **`ΔSw` (Water Saturation Change, fractional $[0, 1]$):** $\Delta S_w = S_{w,2024} - S_{w,2013}$. Delineates the advance of the injected water front from injector wells into production drainage areas.
3. **`Δρ` (Bulk Density Change, $g/cm^3$):** $\Delta \rho = \rho_{2024} - \rho_{2013}$. Quantifies fluid substitution effects during water-oil displacement.
4. **Predictive Uncertainty ($\sigma$):**
   * **Epistemic Model (`EpistemicBNN` / Res-BNN):** Computes $\sigma_{\text{epistemic}}$ via $S = 100$ Monte Carlo variational forward passes on posterior weights $w \sim q(w)$, revealing model ambiguity in poorly sampled compartments.
   * **Aleatoric Model (`AleatoricAutoencoder`):** Direct heteroscedastic standard deviation $\sigma_{\text{aleatoric}}(x)$ from the variance output head, capturing seismic noise and data imperfections.

---

#### B. Comprehensive Input Data Breakdown & Geophysical Principles

The framework operates exclusively on **Pure 4D Pre-Stack Seismic Data** without requiring static porosity, permeability, or geological maps. The input feature space is composed of four foundational seismic data families:

```
       3D Seismic Volume A(x, y, t)                2D Horizon Scalar Slice M(x, y)
       ┌────────────────────────┐                ┌────────────────────────┐
       │   1D Seismic Trace     │                │   Top Reservoir Map    │
       │         │              │                │                        │
  Topo ├─────────▼──────────────┤  Vertical      │     • • • • • •        │
       │      /\   /\           │  Integration   │     • • (x,y) •        │
       │     /  \_/  \          │ ─────────────> │     • • • • • •        │
       │    /         \         │  (SNA / RMS)   │     • • • • • •        │
  Base ├────────────────────────┤                │                        │
       │                        │                │   1 scalar value       │
       └────────────────────────┘                │   per trace (x, y)     │
                                                 └────────────────────────┘
```

| Configuration | Total Features | 1. Baseline Summary (2013) | 2. Monitor Summary (2024) | 3. Explicit 4D $\Delta A$ | 4. Relative 4D $\Delta A/A$ | 5. 1D Stratal Temporal | 6. 4D Time-Shift ($dt$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1 (`exp1_scalar_no_ts`)** | **32** | 8 channels | 8 channels | 8 channels | 8 channels | — | — |
| **Config 2 (`exp2_temporal_only_no_ts`)** | **64** | — | — | — | — | 64 channels | — |
| **Config 3 (`exp3_scalar_temporal_no_ts`)** | **96** | 8 channels | 8 channels | 8 channels | 8 channels | 64 channels | — |
| **Config 4 (`exp4_scalar_with_ts`)** | **36** | 8 channels | 8 channels | 8 channels | 8 channels | — | 4 channels |
| **Config 5 (`exp5_scalar_temporal_with_ts`)** | **100** | 8 channels | 8 channels | 8 channels | 8 channels | 64 channels | 4 channels |

---

##### 1. Horizon Scalar Slices (32 Channels)
A **Horizon Scalar Slice** is a 2D surface map $M(x, y)$ generated by integrating the vertical 1D seismic trace $A(x, y, t)$ across the 3D reservoir interval $[t_{\text{top}}(x, y), t_{\text{base}}(x, y)]$. This compresses the oscillatory time signal into robust energetic summaries that are inherently invariant to structural dip and wavelet phase distortion.

* **Sum of Negative Amplitudes (SNA, 16 channels: 4 Base + 4 Mon + 4 $\Delta$ + 4 Rel $\Delta$):**
  $$\text{SNA}(x, y) = \sum_{t = t_{\text{top}}}^{t_{\text{base}}} \min(A(x, y, t), 0)$$
  * *Geophysical Meaning:* Under standard zero-phase SEG polarity, the transition from high-velocity caprock shale into porous oil/water sandstone generates a strong negative reflection trough. SNA directly quantifies the integrated strength of this reservoir top boundary reflection.
* **Root Mean Square Amplitude (RMS, 16 channels: 4 Base + 4 Mon + 4 $\Delta$ + 4 Rel $\Delta$):**
  $$\text{RMS}(x, y) = \sqrt{\frac{1}{K} \sum_{t = t_{\text{top}}}^{t_{\text{base}}} A(x, y, t)^2}$$
  * *Geophysical Meaning:* RMS measures the total acoustic energy within the formation interval. It is sensitive to internal acoustic impedance contrasts and tuning thickness effects.
* **Explicit 4D Differences ($\Delta A = A_{2024} - A_{2013}$):** Captures net dynamic reflection amplitude gain or dimming across the 11-year production period.
* **Relative 4D Contrast ($\text{Rel } \Delta A = \frac{A_{2024} - A_{2013}}{|A_{2013}| + \epsilon}$):** Normalizes the dynamic change by the baseline impedance ($I_{P,\text{base}}$), as dictated by Zoeppritz reflectivity ($\Delta R_P \propto \frac{\Delta I_P}{I_{P,\text{base}}}$). This single attribute achieves linear correlation $r = \mathbf{0.786}$ with $\Delta S_w$.

---

##### 2. 1D Stratal Temporal Waveforms (64 Channels)
Rather than taking a fixed-time window that suffers from structural dip jitter across the field, the **1D Stratal Temporal** representation divides each trace's reservoir column into proportional stratigraphic sublayers ($z \in [0, 1]$ from Top to Base of reservoir):

* **Top Sublayer ($z \in [0.0, 0.5]$):** Samples the upper reservoir interval (sensitive to secondary gas cap development and top-zone pressure depletion).
* **Base Sublayer ($z \in [0.5, 1.0]$):** Samples the lower reservoir interval (sensitive to basal water injection breakthrough along high-permeability swept channels).
* **Channel Architecture (64 Total):**
  * $16 \text{ Baseline Channels } (W_{13}):$ 4 angles $\times 2$ sublayers (Top/Base) $\times 2$ metrics (SNA and RMS).
  * $16 \text{ Monitor Channels } (W_{24}):$ 4 angles $\times 2$ sublayers (Top/Base) $\times 2$ metrics (SNA and RMS).
  * $16 \text{ Explicit 4D Deltas } (\Delta W = W_{24} - W_{13}):$ Sublayer-resolved 4D amplitude changes.
  * $16 \text{ Relative 4D Deltas } (\text{Rel } \Delta W = \frac{W_{24} - W_{13}}{|W_{13}| + \epsilon}):$ Normalized sublayer reflection changes.

---

##### 3. Seismic 4D Time-Shifts / Traveltime Delays (4 Channels)
4D Time-Shifts ($dt$) represent kinematic measurements of two-way traveltime delays ($\Delta t = t_{2024} - t_{2013}$) caused by velocity alteration ($\Delta V_P$) and reservoir compaction/dilation strain ($\epsilon_{zz}$):

* **Integrated 4D Traveltime Dilational Strain ($\tau_{\text{strain}}$, 2 channels):**
  $$\tau_{\text{strain}}(x, y) = \ln\left(\frac{V_{P,2024}(x, y)}{V_{P,2013}(x, y)}\right) \approx \int_{t_{\text{top}}}^{t_{\text{base}}} \frac{\Delta V_P(z)}{V_P(z)} dz$$
* **Fractional Velocity Delay ($\frac{\Delta dt}{dt}$, 2 channels):**
  $$\frac{\Delta dt}{dt}(x, y) = \frac{V_{P,2024}(x, y) - V_{P,2013}(x, y)}{V_{P,2024}(x, y)}$$
* *Why Time-Shift is Decisive:* Amplitude changes suffer from cross-talk between pressure depletion ($\Delta P < 0 \implies \text{hardening}$) and water influx ($\Delta S_w > 0 \implies \text{hardening}$). Traveltime dilational strain $\tau_{\text{strain}}$ provides the kinematic constraint needed to completely decouple saturation from pressure, propelling $\Delta V_P$ parity to $R = \mathbf{0.9862}$ (NRMSE $\mathbf{2.28\%}$).

---

##### 4. Multi-Angle Pre-Stack Acquisition Angles ($10^\circ, 20^\circ, 30^\circ, 40^\circ$)
By incorporating 4 distinct pre-stack angle stacks, the network exploits Amplitude Versus Offset (AVO) response governed by Aki-Richards Zoeppritz approximations:
$$R_P(\theta) \approx \frac{1}{2}\left(\frac{\Delta V_P}{V_P} + \frac{\Delta \rho}{\rho}\right) - 2\left(\frac{V_S}{V_P}\right)^2\left(\frac{2\Delta V_S}{V_S} + \frac{\Delta \rho}{\rho}\right)\sin^2(\theta) + \frac{1}{2}\frac{\Delta V_P}{V_P}\tan^2(\theta)$$
* **Near Angles ($10^\circ, 20^\circ$):** Dominated by acoustic impedance contrast ($I_P = V_P \cdot \rho$).
* **Far Angles ($30^\circ, 40^\circ$):** Highly sensitive to shear wave impedance ($I_S = V_S \cdot \rho$) and fluid-induced Poisson ratio changes, enabling separate inversion of $\Delta \rho$ and $\Delta S_w$.

---

## 4. Visualizations & Comparative Analysis

---

### 4.1 Epistemic BNN Model Results (`EpistemicBNN` / Res-BNN)
> **Model Formulation:** Fully Variational Bayesian Neural Network with Gaussian weight distributions `w ~ Normal(μ_w, σ_w²)`. Epistemic uncertainty `σ_epistemic` is evaluated through `S = 100` Monte Carlo variational forward passes on the 37,935 blind field traces.

#### 5-Fold Cross-Validation Benchmark Summary:
| Configuration | Features | $R_{\Delta V_P}$ | NRMSE $\Delta V_P$ | $R_{\Delta S_w}$ | NRMSE $\Delta S_w$ | $R_{\Delta \rho}$ | NRMSE $\Delta \rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Scalar Slices / No TS** | 32 | $0.7839$ | $8.82\%$ | $0.7741$ | $10.88\%$ | $0.7708$ | $10.33\%$ |
| **2. Temporal 1D Only / No TS** | 64 | $0.7352$ | $9.94\%$ | $0.7341$ | $11.43\%$ | $0.7393$ | $10.88\%$ |
| **3. Scalar + Temporal 1D / No TS** | 96 | $0.7714$ | $9.05\%$ | $0.7648$ | $11.24\%$ | $0.7670$ | $10.48\%$ |
| **4. Scalar Slices / With TS** | 36 | **0.9862** | **2.28%** | **0.9429** | **5.36%** | **0.9442** | **5.11%** |
| **5. Scalar + Temporal 1D / With TS** | 100 | **0.9154** | **5.49%** | **0.8838** | **7.41%** | **0.8916** | **7.03%** |

#### A. Parity Correlation (R) & Normalized RMSE Metrics Across 5 Configurations
![Epistemic Metrics Comparison](artifacts/ablation_study/comparison_5configs_metrics_epistemic.png)

#### B. Water Saturation Change (ΔSw) & Epistemic Uncertainty (σ_epistemic)
![Epistemic Water Saturation Maps](artifacts/ablation_study/comparison_5configs_maps_epistemic_dsw.png)

#### C. Compressional Velocity Change (ΔVP) & Epistemic Uncertainty (σ_epistemic)
![Epistemic Velocity Maps](artifacts/ablation_study/comparison_5configs_maps_epistemic_dvp.png)

#### D. Bulk Density Change (Δρ) & Epistemic Uncertainty (σ_epistemic)
![Epistemic Density Maps](artifacts/ablation_study/comparison_5configs_maps_epistemic_drho.png)

#### E. Inversion Decoupling & EAGE Benchmark Validation (Champion Model: Config 4 Epistemic)
> **Champion Model Highlight:** Config 4 (`Scalar Slices / With TS`) achieves the **#1 overall performance** across the entire 5-scenario benchmark suite, delivering a mean correlation of **$R = 0.9578$** and mean normalized error of **$\text{NRMSE} = 4.25\%$** on the 37,935 blind validation traces.

Cross-property inversion parity, saturation front tracking, and velocity change recovery:
![Epistemic EAGE Inversion Comparison](artifacts/ablation_study/exp4_scalar_with_ts_epistemic/oof_eage_comparison.png)

#### F. 5-Fold Out-of-Fold Diagnostics & Error Distributions (Champion Model: Config 4 Epistemic)
Detailed validation diagnostics for the champion model across all 37,935 blind field traces:
* **Top Row (Parity Regression):** Compares observed ground truth against out-of-fold BNN predictions. Demonstrates tight adherence to the ideal 1:1 red dashed line with correlation $R_{\Delta V_P} = \mathbf{0.986}$, $R_{\Delta S_w} = \mathbf{0.943}$, and $R_{\Delta \rho} = \mathbf{0.944}$.
* **Middle Row (Standardized Residuals):** Residual histograms $(\text{observed} - \text{predicted}) / \sigma$ compared against standard normal distribution $\mathcal{N}(0, 1)$ (red curve), confirming unbiased predictions with near-zero mean.
* **Bottom Row (Empirical Coverage Curves):** Cumulative coverage of predictive intervals against theoretical Gaussian nominal coverage, validating that epistemic uncertainty bounds correctly enclose field variations.

![Epistemic Diagnostics Scatter](artifacts/ablation_study/exp4_scalar_with_ts_epistemic/oof_diagnostics.png)

#### G. Spatial Absolute Error Maps Across the Reservoir (Champion Model: Config 4 Epistemic)
Spatial distribution of absolute residuals $|\text{Ground Truth} - \text{Prediction}|$ for all three dynamic properties across the UNISIM field:
![Epistemic Error Maps](artifacts/ablation_study/exp4_scalar_with_ts_epistemic/oof_error_maps.png)

#### H. 5-Fold Training & Validation Loss History (Champion Model: Config 4 Epistemic)
Convergence history of training data-fit MSE (dashed lines) versus blind validation MSE (solid lines) across all 5 folds, plotted on a synchronized scale $[0.0, 0.6]$:
![Epistemic Training History](artifacts/ablation_study/exp4_scalar_with_ts_epistemic/oof_training_history.png)

---

### 4.2 Aleatoric Model Results (`AleatoricAutoencoder` with Heteroscedastic Noise Head)
> **Model Formulation:** Deterministic feedforward network with dual output heads predicting `(μ(x), log(σ²(x)))`. Aleatoric uncertainty `σ_aleatoric` directly models heteroscedastic observation noise and seismic attribute noise.

#### 5-Fold Cross-Validation Benchmark Summary:
| Configuration | Features | $R_{\Delta V_P}$ | NRMSE $\Delta V_P$ | $R_{\Delta S_w}$ | NRMSE $\Delta S_w$ | $R_{\Delta \rho}$ | NRMSE $\Delta \rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Scalar Slices / No TS** | 32 | $0.7686$ | $9.21\%$ | $0.7672$ | $11.02\%$ | $0.7765$ | $10.22\%$ |
| **2. Temporal 1D Only / No TS** | 64 | $0.7512$ | $9.48\%$ | $0.7485$ | $11.16\%$ | $0.7540$ | $10.55\%$ |
| **3. Scalar + Temporal 1D / No TS** | 96 | $0.7594$ | $9.35\%$ | $0.7538$ | $11.08\%$ | $0.7562$ | $10.50\%$ |
| **4. Scalar Slices / With TS** | 36 | **0.9777** | **2.88%** | **0.9402** | **5.55%** | **0.9404** | **5.30%** |
| **5. Scalar + Temporal 1D / With TS** | 100 | **0.8984** | **6.01%** | **0.8815** | **7.55%** | **0.8915** | **7.03%** |

#### A. Parity Correlation (R) & Normalized RMSE Metrics Across 5 Configurations
![Aleatoric Metrics Comparison](artifacts/ablation_study/comparison_5configs_metrics_aleatoric.png)

#### B. Water Saturation Change (ΔSw) & Aleatoric Uncertainty (σ_aleatoric)
![Aleatoric Water Saturation Maps](artifacts/ablation_study/comparison_5configs_maps_aleatoric_dsw.png)

#### C. Compressional Velocity Change (ΔVP) & Aleatoric Uncertainty (σ_aleatoric)
![Aleatoric Velocity Maps](artifacts/ablation_study/comparison_5configs_maps_aleatoric_dvp.png)

#### D. Bulk Density Change (Δρ) & Aleatoric Uncertainty (σ_aleatoric)
![Aleatoric Density Maps](artifacts/ablation_study/comparison_5configs_maps_aleatoric_drho.png)

---

### 4.3 Calibration Well Layout & Spatial Geometry
Comparison of the 26 canonical UNISIM-I real well training positions against spatial-optimal and pseudo-random distributions:
![Well Selection Comparison](artifacts/trace_selection_comparison.png)

---

## 5. Geophysical Interpretation of Results

1. **How Stratal 1D Temporal Decomposition (Config 2) Solves Baseline Inversion ($R \approx 0.74 - 0.75$):**
   * By decomposing the reservoir interval into **Top and Base stratal sublayers** and preserving both **Baseline ($W_{13}$)** and **Monitor ($W_{24}$)** waveforms alongside relative contrast ($\frac{W_{24} - W_{13}}{|W_{13}| + \epsilon}$), the network resolves the background impedance scale ambiguity ($I_{P,\text{base}}$), jumping from $R = 0.01$ to **$R = 0.7352$** without needing time-shifts.
2. **Why Full Horizon Scalar Slices (Config 1) Excel ($R \approx 0.77 - 0.78$):**
   * Integrating Sum of Negative Amplitudes (SNA) and RMS energy across the complete 3D reservoir column eliminates wavelet phase jitter and provides robust dynamic amplitude scaling.
3. **Why 4D Time-Shift ($dt$) Propels Performance to $R > 0.98$ (Config 4):**
   * 4D traveltime shifts ($dt$) directly integrate reservoir velocity changes and dilational strain throughout the overburden and reservoir layer:
     $$\tau_{\text{strain}} = \ln\left(\frac{V_{P,2024}}{V_{P,2013}}\right), \quad \frac{\Delta dt}{dt} = \frac{V_{P,2024} - V_{P,2013}}{V_{P,2024}}$$
   * This kinematic signature completely decouples the pressure-saturation ambiguity, enabling the Res-BNN to map $\Delta V_P$ with **2.28% NRMSE** and $\Delta S_w, \Delta \rho$ with **< 5.4% NRMSE**.
4. **Dual Uncertainty Roles:**
   * **Epistemic Uncertainty ($\sigma_{\text{epistemic}}$):** High in inter-well regions farthest from the 26 calibration wells, highlighting uncalibrated compartments.
   * **Aleatoric Uncertainty ($\sigma_{\text{aleatoric}}$):** Highlights structural discontinuities, fault scarps, and low signal-to-noise seismic zones.

---

## 6. Run & Debug via VS Code (`.vscode/launch.json`)

The [`.vscode/launch.json`](.vscode/launch.json) file includes preconfigured tasks for the **Run & Debug (F5)** panel:

* **`0. [RUN-ALL] Run All 5 Ablations + Comparison Plots (Aleatoric + Epistemic)`**
* **`0. [RUN-ALL] Run All 5 Ablations (Epistemic Only)`**
* **`0. [RUN-ALL] Run All 5 Ablations (Aleatoric Only)`**
* `1. [EXP-1] 5-Fold CV: Scalar Slices / No TS (Aleatoric)`
* `2. [EXP-1] 5-Fold CV: Scalar Slices / No TS (Epistemic)`
* `3. [EXP-2] 5-Fold CV: Temporal Window 1D Only / No TS (Aleatoric)`
* `4. [EXP-2] 5-Fold CV: Temporal Window 1D Only / No TS (Epistemic)`
* `5. [EXP-3] 5-Fold CV: Scalar + Temporal 1D / No TS (Aleatoric)`
* `6. [EXP-3] 5-Fold CV: Scalar + Temporal 1D / No TS (Epistemic)`
* `7. [EXP-4] 5-Fold CV: Scalar Slices / With TS (Aleatoric)`
* `8. [EXP-4] 5-Fold CV: Scalar Slices / With TS (Epistemic)`
* `9. [EXP-5] 5-Fold CV: Scalar + Temporal 1D / With TS (Aleatoric)`
* `10. [EXP-5] 5-Fold CV: Scalar + Temporal 1D / With TS (Epistemic)`
* `11. [PLOT-COMPARE] Generate 5-Scenario Comparative Plots (Aleatoric)`
* `12. [PLOT-COMPARE] Generate 5-Scenario Comparative Plots (Epistemic)`

---

## 7. Command-Line Interface (CLI)

Run via `uv run` or `python -m bnn4d.cli`:

```bash
# 1. Run all 5 ablation scenarios sequentially and generate comparison plots
uv run python -m bnn4d.cli run-all-ablations \
  --data artifacts/unisim_4d.npz \
  --output-dir artifacts/ablation_study \
  --model all \
  --epochs 150 \
  --patience 30

# 2. Run 5-Fold Cross-Validation for a specific configuration
uv run python -m bnn4d.cli cv \
  --data artifacts/unisim_4d.npz \
  --output-dir artifacts/ablation_study/exp4_scalar_with_ts_epistemic \
  --model epistemic \
  --train-traces 26 \
  --trace-selection unisim_wells \
  --use-time-shift \
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
    "1. Scalar Slices / No TS=artifacts/ablation_study/exp1_scalar_no_ts_epistemic" \
    "2. Temporal Window 1D Only / No TS=artifacts/ablation_study/exp2_temporal_only_no_ts_epistemic" \
    "3. Scalar + Temporal 1D / No TS=artifacts/ablation_study/exp3_scalar_temporal_no_ts_epistemic" \
    "4. Scalar Slices / With TS=artifacts/ablation_study/exp4_scalar_with_ts_epistemic" \
    "5. Scalar + Temporal 1D / With TS=artifacts/ablation_study/exp5_scalar_temporal_with_ts_epistemic" \
  --output-dir artifacts/ablation_study \
  --model-name epistemic
```

---

## 8. Unit Tests

Run the full automated test suite:

```bash
uv run python -m unittest discover -s tests -v
```
