# BNN4D

Implementação em Python/PyTorch das duas redes de Sukar, Côrte e MacBeth,
“Dynamic Reservoir Property Estimation With Uncertainty Quantification From
4D Seismic Data Using Bayesian Neural Networks” (2026).

## Correspondência com o artigo

- Encoder-decoder denso simétrico: `1024, 768, 512, 256, 256, 256, 512, 768, 1024`.
- Entradas: atributos 4D near/mid/far/gradient em múltiplos vintages e mapa
  estático de volume poroso.
- Saídas: `ΔP`, `ΔSw`, `ΔSg`.
- Modelo aleatório: cabeças separadas de média e log-variância, treinadas pela
  NLL gaussiana heteroscedástica (Eq. 2).
- Modelo epistêmico: **todas** as camadas densas têm posterior gaussiano
  mean-field e prior `N(0, 1)`, treinadas pelo ELBO (Eq. 5).
- Inferência epistêmica: 500 passes estocásticos; média e desvio padrão das
  amostras são a estimativa e a incerteza.
- Preparação: janelas deslizantes entre levantamentos, normalização e ruído
  gaussiano relativo (17% foi o nível ótimo no estudo).

O artigo não publica comprimento da janela, ativação, otimizador, learning rate
ou batch size. Esses valores não podem ser reproduzidos literalmente. A API os
mantém explícitos; `ReLU` é apenas o padrão convencional, não uma alegação sobre
o experimento original.

## Uso mínimo

```python
import torch
from bnn4d import AleatoricAutoencoder, EpistemicBNN
from bnn4d.losses import gaussian_nll

# Exemplo: janela de 2 vintages × 4 atributos + 1 volume poroso.
x = torch.randn(32, 9)
y = torch.randn(32, 3)

aleatoric = AleatoricAutoencoder(input_dim=9)
mean, log_variance = aleatoric(x)
loss = gaussian_nll(y, mean, log_variance)

epistemic = EpistemicBNN(input_dim=9, prior_std=1.0)
samples, prediction, uncertainty = epistemic.predict_distribution(x, samples=500)
```

Para dados em mapas, use `SlidingWindowDataset(seismic, pore_volume, targets,
window)`. As formas esperadas são `[vintage, linha, coluna, atributo]`, `[linha,
coluna]` e `[vintage, linha, coluna, 3]`.

## Validação

Sem instalar o pacote:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## CLI

Instale localmente com `python -m pip install -e .` para disponibilizar o
comando `bnn4d`, ou execute sem instalação usando `PYTHONPATH=src python -m
bnn4d.cli`.

```bash
# Gera um conjunto pequeno para testar o fluxo.
PYTHONPATH=src python -m bnn4d.cli make-demo-data --output artifacts/demo.npz

# Treina. Os padrões usam a arquitetura do artigo, 17% de ruído e 400 épocas.
PYTHONPATH=src python -m bnn4d.cli train \
  --data artifacts/demo.npz --output artifacts/aleatoric.pt --model aleatoric

# Produz mean e uncertainty, preservando dimensões [tempo, linha, coluna, 3].
PYTHONPATH=src python -m bnn4d.cli predict \
  --data artifacts/demo.npz --checkpoint artifacts/aleatoric.pt \
  --output artifacts/predictions.npz

# Gera mapas, histórico de treino e diagnósticos de calibração.
PYTHONPATH=src MPLCONFIGDIR=.matplotlib python -m bnn4d.cli plot \
  --predictions artifacts/predictions.npz --data artifacts/demo.npz \
  --checkpoint artifacts/aleatoric.pt --output-dir artifacts/plots
```

O arquivo de entrada `.npz` deve conter `seismic[T,H,W,A]`,
`pore_volume[H,W]` e, para treino, `targets[T,H,W,3]`. A saída contém os arrays
`mean` e `uncertainty`.

As configurações em `.vscode/launch.json` permitem gerar dados, treinar os dois
modelos, executar predição e visualizar os resultados pelo painel **Run and Debug**. As configurações demo
usam uma rede reduzida e três épocas para terminarem rapidamente; remova
`--widths` e use `--epochs 400` para a arquitetura publicada.

As funções Python `plot_training_history`, `plot_property_maps` e
`plot_prediction_diagnostics` estão em `bnn4d.visualization`. Todas retornam um
objeto `matplotlib.figure.Figure` e aceitam `output=...`, permitindo tanto uso
interativo quanto exportação automatizada.
# BNN4D
