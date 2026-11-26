import argparse
from datetime import datetime

import optuna

from experimento import rodar
from modelos import MODELOS


def otimizar(modelo: str, n_trials: int = 20, metrica: str = "rmse", dataset: str = "imputado"):
    if modelo not in MODELOS:
        raise ValueError(f"Modelo '{modelo}' não encontrado. Adicione-o em models/modelos.py.")
    if not callable(MODELOS[modelo].get("espaco")):
        raise ValueError(f"Modelo '{modelo}' não tem espaço de busca de hiperparâmetros; use experimento.py.")

    grupo = f"otimizacao_{modelo}_{datetime.now():%Y%m%d_%H%M%S}"   # junta as tentativas no W&B
    direcao = "maximize" if metrica.startswith("r2") else "minimize"

    def objetivo(trial):
        params = MODELOS[modelo]["espaco"](trial)
        return rodar(modelo, params, dataset, grupo=grupo)[metrica]

    estudo = optuna.create_study(direction=direcao, sampler=optuna.samplers.TPESampler(seed=42))
    estudo.optimize(objetivo, n_trials=n_trials)

    print(f"\nMelhor {metrica} na validação: {estudo.best_value:.4f}")
    print(f"Melhores hiperparâmetros: {estudo.best_params}")
    return estudo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("modelo")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--metrica", default="rmse")
    p.add_argument("--dataset")
    a = p.parse_args()

    otimizar(a.modelo, a.n_trials, a.metrica, a.dataset)
