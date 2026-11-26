"""Avalia no conjunto de teste um modelo já treinado. Use só com o modelo final escolhido.

    python models/testar.py <run_id>      # o run_id é o nome da pasta em models/runs/
"""
import json
import sys

import joblib

from carregar_dados import carregar
from experimento import PASTA_RUNS
from metricas import regression_metrics


def testar(run_id: str) -> dict:
    pasta = PASTA_RUNS / run_id
    config = json.loads((pasta / "config.json").read_text())
    estimador = joblib.load(pasta / "modelo.joblib")

    X, y, y_unidades = carregar(config["dataset"], "teste")
    metricas = regression_metrics(y_unidades, estimador.predict(X), y)

    (pasta / "metricas_teste.json").write_text(json.dumps(metricas, indent=2))
    print(f"[{run_id}] teste: " + "  ".join(f"{k}={v:.4f}" for k, v in metricas.items()))
    return metricas


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python models/testar.py <run_id>")
    testar(sys.argv[1])
