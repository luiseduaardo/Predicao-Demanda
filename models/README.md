# Workflow com rastreamento dos experimentos

Essa etapa de modelagem tem como objetivo automatizar o processo de treinamento, validação e teste baseado no funcionamento framework [TrailKeeper](https://github.com/MauricioSight/TrailKeeper), que realiza isso.

Para isso, conseguimos realizar o acompanhamento dos resultados dos experimentos usando Weights & Biases e a realização da otimização dos hiperparâmetros dos modelos escolhidos usando Optuna.

A arquitetura dessa modelagem consiste em:

```markdown
.
├── modelos.py          # Escolha dos modelos, parâmetros e espaço de busca da otimização
├── experimento.py      # Treinamento, medição e registro no W&B.
├── otimizar.py         # Executa a otimização dos hiperparâmetros com Optuna
├── testar.py           # Mede o modelo treinado
├── carregar_dados.py   # Carregamento do dataset
└── metricas.py         # RMSE, MAE e R² em log e em unidades
```

## Como funciona

1. `rodar(modelo, {hiperparâmetros})` lê os dados, cria o modelo com esses hiperparâmetros e o treina.
2. Calcula as métricas no treino e na validação.
3. Envia hiperparâmetros e métricas ao W&B, e salva `modelo.joblib` e `config.json` em `models/runs/<run_id>/`.
4. `otimizar` repete isso várias vezes, deixando o Optuna escolher os hiperparâmetros de cada tentativa. As tentativas ficam agrupadas no W&B, cada uma como uma run.
5. Ao final, `otimizar` marca a melhor run com a tag `best` no W&B (modo online) e guarda o `run_id` dela para usar no `testar.py` (pelo terminal ele é impresso no fim; no notebook, veja `estudo.best_trial.user_attrs["run_id"]`). O progresso aparece como uma linha por trial no terminal, ou como um bloco que se atualiza no notebook; as mensagens do W&B ficam silenciadas.

## Uso

Crie e ative o ambiente virtual:

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Linux/Mac
source .venv/bin/activate
# On Windows (PowerShell)
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### Executando pelo terminal

#### Treinamento com os parâmetros padrão

```bash
python models/experimento.py baseline                          # dataset imputado (padrão)
python models/experimento.py baseline --dataset sem_faltantes
```

#### Tunagem dos hiperparâmetros

```bash
python models/otimizar.py elasticnet --n_trials 20
python models/otimizar.py elasticnet --n_trials 20 --dataset sem_faltantes
```

#### Teste final com base no ID

```bash
python models/testar.py <run_id>
```

### Executando em um notebook

```python
import sys
sys.path.append("../models")

from experimento import rodar
rodar("ridge", {"alpha": 10.0})
```

## Weights & Biases

Por padrão `MODO_WANDB = "online"`, então rode `wandb login` antes. Para trabalhar sem login, troque para `"offline"`: os registros ficam em `models/runs/wandb/` e podem ser enviados depois com `wandb sync`.

## Dados

Os modelos usam as versões de `data/processed/` geradas no notebook 4. O padrão é `imputado`. Use `sem_faltantes` com `--dataset sem_faltantes` no terminal ou `rodar(..., dataset="sem_faltantes")` no código. Sem essa opção, tudo roda no `imputado`.
