import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import wandb

from carregar_dados import DATASETS, carregar
from metricas import regression_metrics
from modelos import MODELOS

PROJETO_WANDB = "predicao-demanda"
MODO_WANDB = "online"
PASTA_RUNS = Path(__file__).resolve().parent / "runs"


def rodar(modelo: str, params: dict | None = None, dataset: str = "imputado", grupo: str | None = None) -> dict:
    if modelo not in MODELOS:
        raise ValueError(f"Modelo '{modelo}' não encontrado")

    params = {**MODELOS[modelo].get("params", {}), **(params or {})}
    run_id = f"{modelo}_{dataset}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    pasta = PASTA_RUNS / run_id
    pasta.mkdir(parents=True, exist_ok=True)

    config = {"modelo": modelo, "dataset": dataset, "params": params}
    run = wandb.init(project=PROJETO_WANDB, name=run_id, group=grupo, tags=[dataset, modelo],
                 mode=MODO_WANDB, dir=str(PASTA_RUNS), config=config)
    
    try:
        X, y, y_unidades = carregar(dataset, "treino")
        X_val, y_val, y_val_unidades = carregar(dataset, "validacao")

        estimador = MODELOS[modelo]["criar"](params)
        estimador.fit(X, y)

        treino = regression_metrics(y_unidades, estimador.predict(X), y)
        validacao = regression_metrics(y_val_unidades, estimador.predict(X_val), y_val)

        run.log({**{f"treino/{k}": v for k, v in treino.items()},
                 **{f"validacao/{k}": v for k, v in validacao.items()}})

        joblib.dump(estimador, pasta / "modelo.joblib")
        (pasta / "config.json").write_text(json.dumps(config, indent=2))
        print(f"[{run_id}] validação: " + "  ".join(f"{k}={v:.4f}" for k, v in validacao.items()))
    finally:
        run.finish()

    return validacao


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("modelo")
    p.add_argument("--dataset", default="imputado", choices=DATASETS)
    a = p.parse_args()

    rodar(a.modelo, dataset=a.dataset)
