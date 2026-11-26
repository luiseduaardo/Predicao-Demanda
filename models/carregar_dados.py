from pathlib import Path

import pandas as pd

PASTA_DADOS = Path(__file__).resolve().parents[1] / "data" / "processed"

ALVO = "qtd_total_log"          # o que o modelo prevê (log1p da demanda semanal)
ALVO_UNIDADES = "qtd_total"     # a mesma demanda, em unidades (usada nas métricas)
IDS = ["id_produto", "no_semana", "semana"]   # não entram como variável explicativa


def carregar(dataset: str, fase: str):
    """
    in: dataset: 'imputado' ou 'sem_faltantes'. fase: 'treino', 'validacao' ou 'teste'.
    out: Devolve X (variáveis), y (alvo em log) e y_unidades (alvo em unidades).
    """

    df = pd.read_csv(PASTA_DADOS / dataset / f"{fase}.csv")
    X = df.drop(columns=IDS + [ALVO, ALVO_UNIDADES])
    
    return X, df[ALVO], df[ALVO_UNIDADES]
