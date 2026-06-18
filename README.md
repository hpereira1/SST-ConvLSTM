# Previsão de SST com ConvLSTM

Código e artefatos do TCC *"Avaliação de redes ConvLSTM para a previsão de temperatura da
superfície do mar na costa catarinense"*. Treina **14 variantes** de uma arquitetura ConvLSTM
para prever a SST de 1 a 7 dias à frente, sobre o produto OSTIA recortado na costa catarinense.

## Ambiente

Python 3.10, TensorFlow 2.16+.

```bash
conda env create -f envirement.yml && conda activate sstml
# ou: pip install -r requirements.txt
```

## Dados

`data/raw/ostia_sst_1996-2022_clean.nc` (SST diária em °C) — caminho padrão lido pelo
`config.py`. Divisão temporal: treino até 2016, validação 2017–2019, teste 2020–2022.

## Uso

```bash
python show_config.py                          # mostra a configuração
python train.py --variants v5                  # treina uma variante (nomes internos)
python evaluate.py --model <modelo.keras> --output-dir <dir>   # métricas no teste
python inference.py --model <modelo.keras> --data <dados.nc> --n-random 5 --output-dir <dir>
```

As variantes usam **nomes internos** no `train.py` (`vbase`, `v5`, …); o mapeamento para os
nomes descritivos (1C, …, 2C-BN-D-S-PH) e os resultados por modelo estão em `docs/resultados.md`.

## Estrutura

```
data/raw/      dados OSTIA
src/           pipeline (modelo, pré-processamento, perdas, config)
train.py / evaluate.py / inference.py / show_config.py
scripts/       geração de dados e figuras (ACC/RA, curvas, amostras)
runs/          14 modelos treinados (um dir por variante)
docs/          resultados consolidados e EDA
amostras_aleatorias/   3 amostras de predição por modelo
```
