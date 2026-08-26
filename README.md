# BNN4D: Bayesian Neural Networks for 4D Seismic Inversion

Implementação em Python/PyTorch do framework de quantificação de incertezas em inversão sísmica 4D baseado em **Sukar, Côrte e MacBeth (2026)** (*“Dynamic Reservoir Property Estimation With Uncertainty Quantification From 4D Seismic Data Using Bayesian Neural Networks”*), adaptado para o benchmark de reservatório **UNISIM-I**.

---

## 1. Arquitetura e Modelagem

O repositório implementa duas abordagens bayesianas complementares para mapear atributos sísmicos 4D diretamente em variações de propriedades de reservatório com quantificação de incertezas:

1. **Modelo Aleatórico (`AleatoricAutoencoder`):**
   - Rede feedforward profunda com cabeças duplas de saída: $\mu(x)$ (média preditiva) e $\log \sigma^2(x)$ (log-variância da incerteza dos dados/ruído de medição).
   - Otimizado via Negative Log-Likelihood Gaussiana Heteroscedástica:
     $$\mathcal{L}_{\text{aleatoric}} = \frac{1}{2N} \sum_{i=1}^N \left( \frac{\|y_i - \mu(x_i)\|^2}{\sigma^2(x_i)} + \log \sigma^2(x_i) \right)$$

2. **Modelo Epistêmico (`EpistemicBNN`):**
   - Rede neural bayesiana completa onde **todas as camadas lineares possuem pesos estocásticos** parametrizados por distribuições Gaussianas Variacionais $w \sim \mathcal{N}(\mu_w, \sigma_w^2)$.
   - Prior Gaussiano padrão $\mathcal{N}(0, \sigma_0^2 I)$.
   - Otimizado via Variational Free Energy / ELBO com regularização $\text{KL}(q(w) \| p(w))$.
   - Inferência por **Monte Carlo Dropout / Variational Draws**: $S$ passes estocásticos determinam a média $\mathbb{E}[y]$ e o desvio padrão epistêmico $\sigma_{\text{epistemic}} = \text{std}(y^{(s)})$.

---

## 2. Entradas e Saídas do Modelo

### Saídas Estimadas (Outputs / Targets):
O modelo infere simultaneamente 3 variações temporais de propriedades dinâmicas e petroelásticas entre levantamentos sísmicos, juntamente com o mapa espacial de incerteza $\sigma$:
* **$\Delta V_P$:** Variação de velocidade de onda compressional P ($m/s$).
* **$\Delta S_w$:** Variação de saturação de água no espaço poroso ($0.0 - 1.0$).
* **$\Delta \rho$:** Variação de densidade de rocha e fluidos combinados ($g/cm^3$).
* **$\sigma$:** Incerteza preditiva associada a cada propriedade estimada.

### As 4 Configurações de Entrada (Estudo de Ablation):

| Configuração | Total de Features | Detalhamento das Entradas (Inputs) |
| :--- | :---: | :--- |
| **Config 1: Sem Static / Sem TS** | **32** | 16 Amplitudes Multi-Ângulo ($A_{\text{base}}, A_{\text{mon}}$ em 8 ângulos) + 8 Deltas $\Delta A$ + 8 Deltas Relativos $\frac{\Delta A}{\|A\|}$ |
| **Config 2: Com Static / Sem TS** | **37** | 32 Amplitudes 4D + 5 Mapas Estáticos (Porosidade $\phi$, Argilosidade $V_{\text{sh}}$, Permeabilidades $K_x, K_y, K_z$) |
| **Config 3: Sem Static / Com TS** | **36** | 32 Amplitudes 4D + 4 Mapas de Time-Shift Sísmico 4D ($dt$) |
| **Config 4: Com Static / Com TS** | **41** | 32 Amplitudes 4D + 4 Time-Shift ($dt$) + 5 Mapas Estáticos de Rocha |

---

## 3. Esquema de Validação Realista (26 Poços UNISIM-I + Campo Cego)

Em cenários reais de exploração e produção, redes neurais são calibradas **apenas nas localizações dos poços perfurados**, enquanto a predição deve cobrir todo o campo:

* **Conjunto de Treino:** Fixado estritamente nos **26 poços canônicos do UNISIM-I** (`--train-traces 26 --trace-selection unisim_wells`) em todos os folds.
* **Validação Cega (Out-of-Fold):** Os **37.935 traços restantes de reservatório** são divididos em 5 folds cegos para validação espacial estrita sem vazamento de dados (*data leakage*).
* **Early Stopping:** Monitoramento da perda de validação com paciência configurável (`--patience 30`), salvando o melhor ponto de calibração.

---

## 4. Execução Rápida via VS Code (`.vscode/launch.json`)

O arquivo [`.vscode/launch.json`](.vscode/launch.json) contém configurações prontas para o painel **Run & Debug (F5)**:

### Execução em Lote dos 4 Cenários + Plots Comparativos:
* **`0. [RUN-ALL] Executar Todos os 4 Experimentos + Plots Comparativos (Aleatoric + Epistemic)`**
* **`0. [RUN-ALL] Executar Todos os 4 Experimentos (Apenas Epistemic)`**
* **`0. [RUN-ALL] Executar Todos os 4 Experimentos (Apenas Aleatoric)`**

### Execuções Individuais por Cenário:
* `1. [EXP-1] CV 5-Fold: Sem Static / Sem TS (Aleatoric)`
* `2. [EXP-1] CV 5-Fold: Sem Static / Sem TS (Epistemic)`
* `3. [EXP-2] CV 5-Fold: Com Static / Sem TS (Aleatoric)`
* `4. [EXP-2] CV 5-Fold: Com Static / Sem TS (Epistemic)`
* `5. [EXP-3] CV 5-Fold: Sem Static / Com TS (Aleatoric)`
* `6. [EXP-3] CV 5-Fold: Sem Static / Com TS (Epistemic)`
* `7. [EXP-4] CV 5-Fold: Com Static / Com TS (Aleatoric)`
* `8. [EXP-4] CV 5-Fold: Com Static / Com TS (Epistemic)`
* `9. [PLOT-COMPARE] Gerar Plots Comparativos dos 4 Cenários (Aleatoric)`
* `10. [PLOT-COMPARE] Gerar Plots Comparativos dos 4 Cenários (Epistemic)`

---

## 5. Linha de Comando (CLI)

O pacote expõe o comando `bnn4d` via `uv run` ou `python -m bnn4d.cli`:

```bash
# 1. Executar os 4 cenários de ablation sequencialmente e gerar plots comparativos
uv run python -m bnn4d.cli run-all-ablations \
  --data artifacts/unisim_4d.npz \
  --output-dir artifacts/ablation_study \
  --model all \
  --epochs 150 \
  --patience 30

# 2. Executar Cross-Validation de 5 folds para um modelo específico
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

# 3. Gerar gráficos comparativos a partir de diretórios de experimentos existentes
uv run python -m bnn4d.cli compare-ablations \
  --data artifacts/unisim_4d.npz \
  --experiments \
    "1. Sem Static / Sem TS=artifacts/ablation_study/exp1_no_static_no_ts_epistemic" \
    "2. Com Static / Sem TS=artifacts/ablation_study/exp2_with_static_no_ts_epistemic" \
    "3. Sem Static / Com TS=artifacts/ablation_study/exp3_no_static_with_ts_epistemic" \
    "4. Com Static / Com TS=artifacts/ablation_study/exp4_with_static_with_ts_epistemic" \
  --output-dir artifacts/comparison_plots
```

---

## 6. Testes Unitários

Para rodar os testes da suíte de dados, modelos e visualizações:

```bash
uv run python -m unittest discover -s tests -v
```
