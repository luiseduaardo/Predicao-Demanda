import argparse
import time
from datetime import datetime

import optuna

from carregar_dados import DATASETS
from experimento import marcar_melhor, rodar
from modelos import MODELOS

optuna.logging.set_verbosity(optuna.logging.WARNING)   # tira o log de cada trial


def _painel():
    """No notebook, devolve uma função que reescreve sempre o mesmo bloco de texto. No terminal, None."""
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return None
    
    if get_ipython() is None or "IPKernelApp" not in get_ipython().config:
        return None
    
    bloco = display({"text/plain": ""}, raw=True, display_id=True)
    
    return lambda texto: bloco.update({"text/plain": texto}, raw=True)


def otimizar(modelo: str, n_trials: int = 20, metrica: str = "rmse", dataset: str = "imputado"):
    if modelo not in MODELOS:
        raise ValueError(f"Modelo '{modelo}' não encontrado. Adicione-o em models/modelos.py.")
    if not callable(MODELOS[modelo].get("espaco")):
        raise ValueError(f"Modelo '{modelo}' não tem espaço de busca de hiperparâmetros; use experimento.py.")

    grupo = f"otimizacao_{modelo}_{dataset}_{datetime.now():%Y%m%d_%H%M%S}"   # junta as tentativas no W&B
    menor_melhor = not metrica.startswith("r2")
    melhor = min if menor_melhor else max
    rotulo = f"{modelo} · {dataset}"
    atualizar = _painel()
    resultados = []
    inicio = time.time()

    def objetivo(trial):
        n = trial.number + 1
        if atualizar:
            texto = f"{rotulo} — trial {n}/{n_trials}"
            if resultados:
                ultimos = " · ".join(f"{v:.5g}" for v in resultados[-5:])
                texto += f"\n  últimos rmses: {ultimos} | melhor: {melhor(resultados):.5g}"
            atualizar(texto)

        params = MODELOS[modelo]["espaco"](trial)
        resultado = rodar(modelo, params, dataset, grupo=grupo, verbose=False)
        trial.set_user_attr("run_id", resultado["run_id"])
        trial.set_user_attr("wandb_id", resultado["wandb_id"])

        valor = resultado["validacao"][metrica]
        resultados.append(valor)
        if not atualizar:
            print(f"{rotulo} — trial {n}/{n_trials}: {metrica}={valor:.5g} (melhor: {melhor(resultados):.5g})", flush=True)
        return valor

    estudo = optuna.create_study(direction="minimize" if menor_melhor else "maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    estudo.optimize(objetivo, n_trials=n_trials)

    marcar_melhor(estudo.best_trial.user_attrs["wandb_id"])

    params = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in estudo.best_params.items())
    (atualizar or print)(
        f"✔ {rotulo} — {n_trials} trials em {time.time() - inicio:.0f}s | "
        f"melhor {metrica}: {estudo.best_value:.5g} (trial {estudo.best_trial.number + 1})\n  {params}"
    )
    return estudo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("modelo")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--metrica", default="rmse")
    p.add_argument("--dataset", default="imputado", choices=DATASETS)
    a = p.parse_args()

    estudo = otimizar(a.modelo, a.n_trials, a.metrica, a.dataset)
    print(f"  run_id: {estudo.best_trial.user_attrs['run_id']}  (teste final: python models/testar.py <run_id>)")
